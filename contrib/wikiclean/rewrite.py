#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "lm-deluge",
#     "click",
#     "smart_open",
#     "plyvel",
# ]
# ///

import json
from dataclasses import dataclass, field, MISSING
from typing import NamedTuple, Any
from pathlib import Path
from lm_deluge import LLMClient, SamplingParams
from lm_deluge.cache import LevelDBCache
import click
import regex
import smart_open
import tqdm

class DelugeRequest(NamedTuple):
    doc_id: str
    summary: str
    full_text: str
    request: str


@dataclass(frozen=True)
class BasePrompt:
    name: str = field(default=MISSING)          # pyright: ignore
    prompt: str = field(default=MISSING)        # pyright: ignore
    structured: bool = field(default=MISSING)   # pyright: ignore

    def format(self, **kwargs: Any) -> str:
        return self.prompt.format(**kwargs).strip()

    def postprocess(self, response: str) -> str:
        return response.strip()


@dataclass(frozen=True)
class SummaryPrompt(BasePrompt):
    name: str = "summary"
    prompt: str = """
You are a helpful assistant that rewrites wikipedia entries into more concise and clear texts.
The purpose of the rewritten article is to help prepare for a quiz competition, such as Jeopardy!
or Quiz Bowl.

The article is as follows:

```
{full_text}
```

The article has already been partially summarized as follows:

```
{summary}
```

Return a short summary (approximately 200 words) of the article. Do not include any other text.
"""
    structured: bool = False

@dataclass(frozen=True)
class QaPrompt(BasePrompt):
    name: str = "qa"
    prompt: str = """
You are a helpful assistant that rewrites Wikipedia articles into flashcards that can be used to practice for a quiz competition, such as Jeopardy! or Quiz Bowl.

## Input
The article is as follows:

```
{full_text}
```

## Output
- For each flashcard, return a compiled list of sentences that can be used as clues.
- Each sentence should be separated by a newline.
- Generate 10 to 50 challenging factoids (the more, the better; but keep it challenging).
- You must avoid pronouns and co-references when referring to the main topic of the article (e.g., do not use "it" or "the song" when referring to the main topic; use the full title of the song instead).
"""
    structured: bool = False

    def postprocess(self, response: str) -> str:
        response = regex.sub(r"\n+", "\n", response)
        sents = [regex.sub(r"^- ", "", sent).strip() for sent in response.split("\n")]
        return "\n".join(sents).strip()


PROMPTS: dict[str, type[BasePrompt]] = {
    "summary": SummaryPrompt,
    "qa": QaPrompt,
}


@click.command()
@click.option("--input-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--model", type=str, default="gpt-4.1-nano")
@click.option("--temperature", type=float, default=0.8)
@click.option("--max-new-tokens", type=int, default=512)
@click.option("--cache-path", type=click.Path(path_type=Path), default=None)
@click.option("--prompt", type=str, default="summary")
@click.option("--batch-size", type=int, default=10000)
def main(
    input_dir: Path,
    output_dir: Path,
    model: str,
    temperature: float,
    max_new_tokens: int,
    cache_path: Path | None,
    prompt: str,
    batch_size: int
):
    assert prompt in PROMPTS, f"Invalid prompt: {prompt}"
    prompt_obj = PROMPTS[prompt]()

    client = LLMClient(
        model,
        max_requests_per_minute=30_000,
        max_tokens_per_minute=150_000_000,
        max_concurrent_requests=10_000,
        sampling_params=[SamplingParams(temperature=temperature, max_new_tokens=max_new_tokens)],
        cache=LevelDBCache(str(cache_path)) if cache_path is not None else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with tqdm.tqdm(desc="Processing...", unit_scale=True) as pbar:
        for cnt, input_file in enumerate(input_dir.glob("*.gz")):
            output_file = output_dir / (input_file.relative_to(input_dir))

            pbar.set_description(f"Processing {input_file} ({cnt + 1})...")

            with smart_open.open(input_file, "r", encoding="utf-8") as input_file, \
                    smart_open.open(output_file, "w", encoding="utf-8") as output_file:

                requests: list[DelugeRequest] = []

                for line in input_file:
                    doc = json.loads(line)

                    if len(doc["attributes"]) == 0:
                        continue

                    _, _, summary = doc["attributes"]["wikiclean__wikiclean__summary"][0]
                    _, _, full_text = doc["attributes"]["wikiclean__wikiclean__full_text"][0]

                    formatted_prompt = prompt_obj.format(
                        full_text=full_text,
                        summary=summary
                    )

                    deluge_request = DelugeRequest(
                        doc_id=doc["id"],
                        summary=summary,
                        full_text=full_text,
                        request=formatted_prompt
                    )
                    requests.append(deluge_request)

                    if len(requests) >= batch_size:
                        resp = client.process_prompts_sync(
                            [request.request for request in requests],
                            show_progress=True,
                            return_completions_only=True
                        )
                        for request, response in zip(requests, resp):
                            if response is None:
                                continue

                            response_text = prompt_obj.postprocess(str(response))

                            if len(response_text) == 0:
                                continue

                            output_doc = {
                                "id": request.doc_id,
                                "text": response_text,
                                "source": "wikiclean",
                                "metadata": {
                                    "summary": request.summary,
                                    "full_text": request.full_text,
                                    "request": request.request,
                                    "model": model,
                                    "temperature": temperature,
                                    "max_new_tokens": max_new_tokens,
                                    "prompt_name": prompt_obj.name,
                                }
                            }
                            output_file.write(json.dumps(output_doc) + "\n")
                            pbar.update(1)
                        requests.clear()
                    pbar.refresh()


if __name__ == "__main__":
    main()
