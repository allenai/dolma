import smart_open
import json
from dolma.core.paths import cached_path
import pandas as pd
from datasets import load_dataset
import hashlib
from tqdm import tqdm
import datetime
import re
DESTINATION_S3 = "s3://ai2-llm/pretraining-data/sources/teknium_OpenHermes-2.5/v1/documents/oh2_5.jsonl.gz"
dataset = load_dataset(
    "teknium/OpenHermes-2.5",
    split="train",
)

OPENHERMES_DATE = datetime.datetime(2023, 11, 12)


def format_to_dolma_timestamp(timestamp: datetime.datetime | None = None) -> str:
    """Format a timestamp as a string using near ISO-8601 format."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


with smart_open.open(DESTINATION_S3, 'w') as f:

    for row in tqdm(dataset):
        spacing = max(
            [len(span) for turn in row["conversations"] for span in re.findall(r'\n+', turn['value'])] + [1]
        )
        text = ("\n" * (spacing + 1)).join(turn['value'] for turn in row["conversations"])

        row_id = row['id'] or hashlib.md5(json.dumps(row).encode('utf-8')).hexdigest()

        source = f'openhermes-2.5'
        if row['source']:
            source += f'-{row["source"]}'

        version = 'v1'

        document = {
            'id': row_id,
            'source': source,
            'version': version,
            'text': text,
            'added': format_to_dolma_timestamp(),
            'created': format_to_dolma_timestamp(OPENHERMES_DATE),
            'metadata': row,
        }

        f.write(json.dumps(document) + '\n')
