import asyncio
from contextlib import ExitStack
from datetime import datetime
from email import message
import json
from pathlib import Path
import random
from time import sleep
from typing import Dict, List
from uuid import uuid4
import necessary
import smart_open
import tqdm

with necessary.necessary(("openai", "1.0.0")):
    import openai

with necessary.necessary("click"):
    import click


TOPICS = [
    "Academia",
    "Business",
    "Communication",
    "Culture",
    "Economy",
    "Education",
    "Energy",
    "Engineering",
    "Entertainment",
    "Ethics",
    "Food and drink",
    "Geography",
    "Government",
    "Health",
    "History",
    "Language",
    "Law",
    "Mass media",
    "Mathematics",
    "Military",
    "Nature",
    "People",
    "Philosophy",
    "Politics",
    "Religion",
    "Science",
    "Society",
    "Sports",
    "Technology",
    "Universe",
]

INSTRUCTION = """
I am creating a dataset to train a multi-label classification model.
This dataset will be used in an Intro to Natural Language Processing class at a university.

I have a total of {num_topics} topics, which I have collected from English Wikipedia.
You should return a SHORT PARAGRAPH that is relevant to a subset of topics I am going to provide.
You must answer in JSON format, with the following fields:
- "topics": repeat the list of topics I provide
- "style": the style of the text you write
- "text": the paragraph you write

Instruction on how to write each paragraph:
- The paragraph should be SHORT, at most 50 words long.
- The paragraph should be relevant to all the topics I provide.
- NEVER INCLUDE the name of the topics in the paragraph. THIS IS CRUCIAL.
- YOU MUST  write the paragraph in the style requested.
- The paragraph should discuss a concept at the INTERSECTION of ALL the topics I provide.

Input request: {{"topics": {formatted_topics}, "style": "{voice}"}}
""".strip()

TEXT_VARIATIONS = [
    "formal",
    "informal",
    "technical",
    "baby talk",
    "poetic",
    "dialogue between two people",
]


def make_message(topics: List[str], max_topics_per_example: int) -> Dict[str, str]:
    selected_topics = random.sample(topics, k=random.randint(1, max_topics_per_example))
    voice = random.choice(TEXT_VARIATIONS)
    formatted_topics = "[" + ", ".join(f'"{t.lower()}"' for t in selected_topics) + "]"
    return {
        "role": "user",
        "content": INSTRUCTION.format(
            num_topics=len(topics),
            formatted_topics=formatted_topics,
            voice=voice
        )
    }


def convert_timestamp(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


async def make_request(
    client: openai.AsyncOpenAI,
    message: Dict[str, str],
    num_concurrent: int
) -> Dict[str, str]:
    sleep(random.uniform(0.01, 0.01 * num_concurrent))
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[message],
        response_format={"type": "json_object"},
        temperature=1.0,
        seed=random.randint(0, 2**16 - 1),
    )
    return json.loads(response.choices[0].message.content)


async def runner(
    openai_api_key: str,
    train: int,
    dev: int,
    test: int,
    output: Path,
    seed: int,
    topics: List[str],
    model: str,
    max_topics_per_example: int,
    num_concurrent: int,
):
    output.mkdir(exist_ok=True, parents=True)
    random.seed(seed)
    max_topics_per_example = min(max_topics_per_example, len(topics))
    client = openai.AsyncOpenAI(api_key=openai_api_key)

    for split, num_examples in [("train", train), ("dev", dev), ("test", test)]:
        dest = output / f"{split}.jsonl"
        existing = 0
        if dest.exists():
            with smart_open.open(dest, "rt") as f:
                existing = sum(1 for _ in f)

        print(f"Generating {num_examples - existing} examples for {split} split.")

        with ExitStack() as stack:
            wf = stack.enter_context(smart_open.open(dest, "at"))
            pbar = stack.enter_context(tqdm.tqdm(desc=split, total=num_examples))

            # advance the random number generator to the correct position
            [make_message(topics, max_topics_per_example) for _ in range(existing)]
            pbar.update(existing)

            for _ in range(0, num_examples - existing, num_concurrent):
                responses = [
                    make_request(
                        client=client,
                        message=make_message(topics, max_topics_per_example),
                        num_concurrent=num_concurrent,
                    )
                    for _ in range(num_concurrent)
                ]

                for future in asyncio.as_completed(responses):
                    content = await future

                    if not (response_text := content.get("text")):
                        continue

                    if not (response_topics := content.get("topics")):
                        continue

                    if not (response_style := content.get("style")):
                        continue

                    ts = convert_timestamp(datetime.now())

                    output_dict = {
                        "source": f"openai-{model}",
                        "text": response_text,
                        "metadata": {"topics": ",".join(response_topics), "style": response_style},
                        "added": ts,
                        "created": ts,
                        "id": str(uuid4())
                    }
                    pbar.update(1)
                    wf.write(json.dumps(output_dict) + "\n")


@click.command()
@click.option("--openai-api-key", required=True, help="OpenAI API key", envvar="OPENAI_API_KEY")
@click.option("--train", default=1000, help="Number of training examples", type=int)
@click.option("--dev", default=100, help="Number of development examples", type=int)
@click.option("--test", default=100, help="Number of test examples", type=int)
@click.option("--output", required=True, help="Output path", type=click.Path(path_type=Path))
@click.option("--seed", default=3920, help="Random seed", type=int)
@click.option("--topics", default=TOPICS, help="Topics to use", type=str, multiple=True)
@click.option("--model", default="gpt-3.5-turbo-0125", help="OpenAI model to use")
@click.option("--max-topics-per-example", default=3, help="Maximum number of topics per example", type=int)
@click.option("--num-concurrent", default=10, help="Number of concurrent requests", type=int)
def main(
    openai_api_key: str,
    train: int,
    dev: int,
    test: int,
    output: Path,
    seed: int,
    topics: List[str],
    model: str,
    max_topics_per_example: int,
    num_concurrent: int,
):
    loop = asyncio.get_event_loop()
    tasks = asyncio.gather(
        runner(
            openai_api_key=openai_api_key,
            train=train,
            dev=dev,
            test=test,
            output=output,
            seed=seed,
            topics=topics,
            model=model,
            max_topics_per_example=max_topics_per_example,
            num_concurrent=num_concurrent,
        )
    )
    try:
        loop.run_until_complete(tasks)
    except Exception as e:
        raise Exception(f"An error occurred: {e}") from e
        # You can also perform any cleanup here if necessary
    finally:
        loop.close()


if __name__ == "__main__":
    main()
