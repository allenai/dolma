import datetime
import hashlib
import json
import smart_open
from contextlib import ExitStack

import tqdm
from datasets import load_dataset
from dolma.core.paths import mkdir_p


def convert_timestamp(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


def main():
    dataset = load_dataset("hails/mmlu_no_train", "all")

    dst_prefix = "/home/ubuntu/ai2-llm/pretraining-data/sources/mmlu/v0/documents/{format}/{split}/{subject}.jsonl.gz"

    eleuther_format = "{question}\n(A) {c0} (B) {c1} (C) {c2} (D) {c3}\nA: {a}"

    mmlu_date = datetime.datetime(year=2021, month=1, day=12)

    data_to_write = {}

    for split, data in dataset.items():
        for row in tqdm.tqdm(data, desc=split, total=len(data)):
            idx = hashlib.md5((transformed := json.dumps(row)).encode()).hexdigest()

            qp = dst_prefix.format(split=split, subject=row["subject"], format="question_only")
            ap = dst_prefix.format(split=split, subject=row["subject"], format="answer_only")
            ep = dst_prefix.format(split=split, subject=row["subject"], format="eleuther")

            base = {
                "id": idx,
                "source": "mmlu",
                "created": convert_timestamp(mmlu_date),
                "added": convert_timestamp(datetime.datetime.now()),
                "metadata": json.loads(transformed),
            }
            data_to_write.setdefault(qp, []).append({**base, "text": row["question"]})

            for choice in row["choices"]:
                data_to_write.setdefault(ap, []).append({**base, "text": choice})

            ef_text = eleuther_format.format(
                question=row["question"].strip(),
                c0=row["choices"][0],
                c1=row["choices"][1],
                c2=row["choices"][2],
                c3=row["choices"][3],
                a=['(A)', '(B)', '(C)', '(D)'][int(row["answer"])]
            )
            data_to_write.setdefault(ep, []).append({**base, "text": ef_text})

    for path, data in data_to_write.items():
        mkdir_p(path.rsplit("/", 1)[0])

        print(f"{len(data):,}\t{path.split('/documents/')[-1]}")

        with smart_open.open(path, "wt") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
