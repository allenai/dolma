from contextlib import ExitStack
import copy
from functools import partial
import multiprocessing
import os
import random
from typing import TYPE_CHECKING, Iterator, Sequence, Union

from necessary import necessary

with necessary("click") as CLICK_AVAILABLE:
    if TYPE_CHECKING or CLICK_AVAILABLE:
        import click

with necessary("dolma") as DOLMA_AVAILABLE:
    if TYPE_CHECKING or DOLMA_AVAILABLE:
        from dolma.core.paths import glob_path
        from dolma.core.data_types import InputSpecWithMetadata
        from msgspec.json import Decoder, Encoder
        from tqdm import tqdm
import smart_open

with necessary("openai", soft=True) as OPENAI_CLIENT_AVAILABLE:
    if TYPE_CHECKING or OPENAI_CLIENT_AVAILABLE:
        import openai

with necessary("together", soft=True) as TOGETHER_CLIENT_AVAILABLE:
    if TYPE_CHECKING or TOGETHER_CLIENT_AVAILABLE:
        import together

client = None


def init_client(model: str):
    global client
    if model.startswith("openai/"):
        # OpenAI model
        assert OPENAI_CLIENT_AVAILABLE, "OpenAI client is not available."
        api_key = os.environ.get("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)
    else:
        # Together model
        assert TOGETHER_CLIENT_AVAILABLE, "Together client is not available."
        api_key = os.environ.get("TOGETHER_API_KEY")
        client = together.Together(api_key=api_key)


def get_documents(
    paths: Union[Iterator[str], Sequence[str]],
    file_prob: float = 1.0,
    row_prob: float = 1.0,
    max_docs: int = -1
):
    random.shuffle(paths := list(paths))

    decoder = Decoder(InputSpecWithMetadata)

    with tqdm(desc="files") as pf, tqdm(desc="docs") as pd:
        for path in paths:
            pf.update()
            with smart_open.open(path, "rt") as f:
                for row in f:
                    if random.random() > row_prob:
                        continue

                    doc = decoder.decode(row)

                    yield doc

                    pd.update()

                    if random.random() > file_prob:
                        break

                    if max_docs > 0 and pd.n >= max_docs:
                        return


def label_document(
    doc: InputSpecWithMetadata,
    prompt: str,
    model_name: str,
    temperature: float,
) -> InputSpecWithMetadata:
    global client
    assert client is not None, "Client is not initialized."
    doc = copy.deepcopy(doc)

    metadata = doc.metadata
    assert metadata is not None, "Document metadata is missing."

    extracted_licenses = {
        license_name: "\n...\n".join(s.strip() for s in license_snippet)
        for license_name, license_snippet in metadata.get("attribute_spans", {}).items()
        if "copyright" not in license_name and isinstance(license_snippet, list)
    }

    labeled_licenses = {}

    # openai models do not need the "openai/" prefix when using the API
    model_name = model_name.lstrip("openai/")

    for license_name, license_snippet in extracted_licenses.items():
        license_prompt = prompt.format(source=metadata["warc_url"], snippet=license_snippet)
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[{"role": "user", "content": license_prompt}]
        )
        labeled_licenses[license_name] = {
            "label": response.choices[0].message.content,   # pyright: ignore
            "snippet": license_snippet,
            "prompt": license_prompt,
            "model": model_name,
            "temperature": temperature,
        }
    metadata["labeled_licenses"] = labeled_licenses
    return doc


def update_pbar(results: list, pbar: tqdm):
    for _ in results:
        pbar.update(1)
    pbar.refresh()


DEFAULT_PROMPT = """
Given the followings HTML snippets enclosed in ```quotes```, respond YES if the Creative Common license mentioned in the snippets refers to to the text content of the web page, otherwise respond NO.

Examples of "NO" include:
- The Creative Common license refers to the images on the page.
- The is another license mentioned on the snippet.
- License is not in an official Creative Common format.
- Mentions that "some", but not "all", of the content is licensed under Creative Common.

Examples of "YES" include:
- Copyright or ©️ is mentioned on the page AND text content is licensed under Creative Common; it is ok if the page is copyrighted if a Creative Common license is also mentioned.
- All "work" or "content" is mentioned as being Creative Common licensed.
- The Creative Commons tag appears on the footer on the page with no extra content.
- The content is in the public domain (i.e., a public domain license in mentioned).

You can use the source URL to help you make an assessment; for example, government and non-profit web pages are more likely to contain creative common licenses. However, DO NOT EXCLUSIVELY rely on the source URL to make a decision.

DO NOT return anything other than "YES" or "NO".

Source URL: {source}

Snippet:
```
{snippet}
```
""".strip()


@click.command()
@click.option(
    "--path",
    type=str,
    default="s3://ai2-llm/pretraining-data/sources/cccc/v2_nc_nd_fix/documents/*/*.gz",
    help="Either a glob pattern or a file path to annotate data from."
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed."
)
@click.option(
    "--model",
    type=str,
    default="openai/gpt-4o",
    help="Model to use for labeling; can be either an openai or together.ai hosted model."
)
@click.option(
    "--prompt",
    type=str,
    default=DEFAULT_PROMPT,
    help="Prompt to use for labeling."
)
@click.option(
    "--procs",
    type=int,
    default=1,
    help="Number of processes to use."
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    help="Temperature for sampling."
)
@click.option(
    "--output",
    type=str,
    default="labeled-documents.jsonl.gz",
)
@click.option(
    "--sample-files",
    type=float,
    default=1.0,
    help="Fraction of documents to sample."
)
@click.option(
    "--sample-rows",
    type=float,
    default=1.0,
    help="Fraction of rows to sample."
)
@click.option(
    "--max-docs",
    type=int,
    default=-1,
    help="Maximum number of documents to process."
)
def main(
    path: str,
    seed: int,
    model: str,
    procs: int,
    prompt: str,
    temperature: float,
    output: str,
    sample_files: float,
    sample_rows: float,
    max_docs: int
):
    random.seed(seed)
    label_fn = partial(label_document, prompt=prompt, model_name=model, temperature=temperature)

    with ExitStack() as stack:
        pool = stack.enter_context(multiprocessing.Pool(procs, initializer=init_client, initargs=(model,)))
        docs_it = get_documents(glob_path(path), file_prob=sample_files, row_prob=sample_rows, max_docs=max_docs)

        pbar = stack.enter_context(tqdm(desc="prompts"))
        update_fn = partial(update_pbar, pbar=pbar)
        result = pool.map_async(label_fn, docs_it, callback=update_fn)

        # Close the pool and wait for the work to finish
        pool.close()
        pool.join()

    # wait for the results to be ready
    output_data = result.get()

    with smart_open.open(output, "wb") as f:
        encoder = Encoder()
        for doc in output_data:
            f.write(encoder.encode(doc) + b"\n")


if __name__ == "__main__":
    # example:
    # python scripts/label-licenses.py --max-docs 100 --sample-files 0.9 --sample-rows 0.01 --procs 8
    main()
