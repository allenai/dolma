import pandas as pd
from pathlib import Path
import json
import smart_open
from ftfy import fix_text
import re
from contextlib import ExitStack
import datetime

import tqdm

DESTINATION_S3 = "s3://ai2-llm/pretraining-data/sources/max-hoffman_eli5/v0"
DCLM_SUBMISSION_SCORE = 3
DCLM_COMMENT_SCORE = 5
DCLM_MIN_ANSWERS = 3
ELI5_CREATED_AT = datetime.datetime(2019, 7, 22)


def format_to_dolma_timestamp(timestamp: datetime.datetime | None = None) -> str:
    """Format a timestamp as a string using near ISO-8601 format."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"



def safe_json_loads(s: str) -> dict | None:  # pyright: ignore
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def replace_urls(row: pd.Series) -> str:
    text, urls = tuple(row)
    if not text:
        return text
    for i, url in enumerate(urls['url']):
        text = text.replace(f"_URL_{i}_", url)
    return text


def read_eli5_data(split: str, data_dir: str = "/Users/lucas/code/eli5/data"):
    data_path = Path(data_dir)
    df = pd.read_parquet(data_path / f"eli5_{split}.parquet")

    df["title_urls"] = df["title_urls"].apply(json.loads)
    df["selftext_urls"] = df["selftext_urls"].apply(json.loads)
    df["answers_urls"] = df["answers_urls"].apply(json.loads)

    # # replace urls in title, selftext, and answers
    df["title_with_urls"] = df[["title", "title_urls"]].apply(replace_urls, axis=1)
    df["selftext_with_urls"] = df[["selftext", "selftext_urls"]].apply(replace_urls, axis=1)
    df["answers_with_urls"] = df[["answers", "answers_urls"]].apply(replace_urls, axis=1)

    # this is the one that might fail
    df["answers_with_urls"] = df["answers_with_urls"].apply(safe_json_loads)
    df["answers"] = df["answers"].apply(safe_json_loads)
    # Count and remove rows where JSON parsing failed

    initial_count = len(df)
    df = df.dropna(subset=['answers_with_urls', 'answers'])
    final_count = len(df)
    failures = initial_count - final_count

    print(f"Number of rows dropped in {split} due to JSON parsing failures: {failures}")

    # replace all NaNs with empty strings
    df = df.fillna("")

    return df


def main():

    for split in ["test", "validation", "train"]:
        df = read_eli5_data(split)
        eli5_created_at = format_to_dolma_timestamp(ELI5_CREATED_AT)

        with ExitStack() as stack:
            full_file = stack.enter_context(smart_open.open(f"{DESTINATION_S3}/conversation/{split}.jsonl.gz", "w"))
            dclm_file = stack.enter_context(smart_open.open(f"{DESTINATION_S3}/dclm/{split}.jsonl.gz", "w"))
            format_file = stack.enter_context(smart_open.open(f"{DESTINATION_S3}/individual/{split}.jsonl.gz", "w"))
            screen_file = stack.enter_context(smart_open.open(f"{DESTINATION_S3}/individual_filtered/{split}.jsonl.gz", "w"))

            for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
                all_text_and_answers = (
                    str(row["title_with_urls"]),
                    str(row["selftext_with_urls"]),
                    *[str(text) for text in row['answers_with_urls']['text']]
                )

                # use two newlines as separator or maximum number of newlines in the text, plus one
                newlines_as_separator = max(
                    2, max(text.count('\n') for text in all_text_and_answers) + 1
                )
                # separate the text with one newline
                full_text = ("\n" * newlines_as_separator).join(all_text_and_answers)

                answer_urls = {
                    f"_URL_{i}_": url for i, url in enumerate(row['answers_urls']['url'])
                }

                metadata = {
                    "q_id": str(row["q_id"]),
                    "title": {
                        "text": str(row["title"]),
                        "urls": [str(url) for url in row["title_urls"]["url"]]
                    },
                    "selftext": {
                        "text": str(row["selftext"]),
                        "urls": [str(url) for url in row["selftext_urls"]["url"]]
                    },
                    "answers": [
                        {
                            "a_id": str(a_id),
                            "text": str(text),
                            "score": int(score),
                            'urls': [str(url) for u_id, url in answer_urls.items() if u_id in text]
                        }
                        for a_id, text, score in
                        zip(row['answers']['a_id'], row['answers']['text'], row['answers']['score'])
                    ]
                }

                full_document = {
                    "text": full_text,
                    "id": str(row["q_id"]),
                    "source": "eli5",
                    "version": "v0_conversation",
                    "created": eli5_created_at,
                    "added": format_to_dolma_timestamp(),
                    "metadata": metadata
                }

                full_file.write(json.dumps(full_document) + "\n")

                dclm_answer = None

                title = fix_text(str(row["title"]))

                for score, a_id, answer in sorted(
                    zip(row['answers']['score'], row['answers']['a_id'], row['answers_with_urls']['text']),
                    key=lambda x: float(f"{x[0]}.{len(x[2])}")
                ):
                    newlines_as_separator = max(2, answer.count('\n') + 1, title.count('\n') + 1)
                    text = ("\n" * newlines_as_separator).join([title, fix_text(answer)])
                    answer_metadata = {
                        **{k: v for k, v in metadata.items() if k != "answers"},
                        **[answer for answer in metadata["answers"] if answer["a_id"] == a_id][0]  # pyright: ignore
                    }
                    answer_document = {
                        "text": text,
                        "id": f"{row['q_id']}_{a_id}",
                        "source": "eli5",
                        "version": "v0_individual",
                        "created": eli5_created_at,
                        "added": format_to_dolma_timestamp(),
                        "metadata": answer_metadata
                    }

                    format_file.write(json.dumps(answer_document) + "\n")

                    if score >= DCLM_COMMENT_SCORE and len(row['answers']['a_id']) >= DCLM_MIN_ANSWERS:
                        dclm_answer = {**answer_document, "version": "v0_dclm"}

                    if score >= DCLM_COMMENT_SCORE:
                        screen_document = {**answer_document, "version": "v0_screen"}
                        screen_file.write(json.dumps(screen_document) + "\n")

                if dclm_answer:
                    dclm_file.write(json.dumps(dclm_answer) + "\n")



if __name__ == "__main__":
    main()
