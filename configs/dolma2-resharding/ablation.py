#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "boto3",
#     "tqdm",
#     "pyyaml",
# ]
# ///

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
import tqdm
import yaml

BASE_URLS = {
    "all-dressed": "s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer/{topic}/{quality}",
    "s2pdf": "s3://ai2-llm/preprocessed/olmo3-final/s2pdfs/allenai/dolma2-tokenizer/{topic}",
    "stack-edu": "s3://ai2-llm/preprocessed/stack-edu/allenai/dolma2-tokenizer/{topic}",
    "finemath-3plus": "s3://ai2-llm/preprocessed/olmo3-final/math/allenai/dolma2-tokenizer/finemath_3plus_all",
    "wikipedia": "s3://ai2-llm/preprocessed/wikipedia-dolma-0823/allenai/dolma2-tokenizer",
    "arxiv": "s3://ai2-llm/preprocessed/proof-pile-2/v0_decontaminated-0625_tokenized/arxiv/train/allenai/dolma2-tokenizer",
}


def get_size_of_prefix(prefix: str, ext: str = ".npy") -> int:
    bucket, prefix = (p := urlparse(prefix)).netloc, p.path.lstrip("/")
    s3 = boto3.client("s3")

    total_size = 0
    continuation_token = None

    while True:
        if continuation_token:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=continuation_token)
        else:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        for obj in response.get("Contents", []):
            if "Key" not in obj:
                continue

            if not obj["Key"].endswith(ext):
                continue

            if "Size" not in obj:
                continue

            total_size += int(obj["Size"])

        if response.get("IsTruncated", False):
            continuation_token = response.get("NextContinuationToken")
        else:
            break

    return total_size


@dataclass(frozen=True)
class WeightConfig:
    domain: str
    topic: str | None
    quality: str | None
    weight: float

    @classmethod
    def from_dict(cls, d: dict) -> "WeightConfig":
        domain_topic_quality = d["domain"].split(":")
        if len(domain_topic_quality) == 3:
            return cls(
                domain=domain_topic_quality[0],
                topic=domain_topic_quality[1],
                quality=domain_topic_quality[2],
                weight=d["weight"],
            )
        elif len(domain_topic_quality) == 2:
            return cls(
                domain=domain_topic_quality[0], topic=domain_topic_quality[1], quality=None, weight=d["weight"]
            )
        elif len(domain_topic_quality) == 1:
            return cls(domain=domain_topic_quality[0], topic=None, quality=None, weight=d["weight"])
        else:
            raise ValueError(f"Invalid domain: {d['domain']}")

    @property
    def uri(self) -> str:
        fmt = {
            **({"topic": self.topic} if self.topic is not None else {}),
            **({"quality": self.quality} if self.quality is not None else {}),
        }
        try:
            return BASE_URLS[self.domain].format(**fmt)
        except KeyError:
            raise ValueError(f"Invalid domain: {self.domain}")

    @property
    def name(self) -> str:
        name = f"{self.domain}:{self.topic or ''}:{self.quality or ''}"
        return name.strip(":")


TOKEN_TARGET = 5_929_970_906_676
# TOKEN_TARGET = 6_000_000_000_000


cross_source_pstar = {
    "finemath-3plus": 0.025340376929054265,
    "arxiv": 0.008284928006565282,
    "wikipedia": 0.000416156026289699,
    "all-dressed": 0.752076181317783,
    "s2pdf": 0.134320140174659,
    "stack-edu": 0.06816936506111054,
}


def make_one_config(weight_config: WeightConfig, token_target: int) -> dict:
    base_uri = weight_config.uri
    actual_size = get_size_of_prefix(base_uri.rstrip("/") + "/") // 4  # 4 bytes per token
    desired_size = token_target * weight_config.weight
    sample_rate = math.ceil(desired_size / actual_size)

    msg = (
        f"Subset Name       : {weight_config.name}\n"
        f"Base URI          : {base_uri}\n"
        f"Natural tokens    : {actual_size / 1000 ** 3:6.2f} B\n"
        f"Desired tokens    : {desired_size / 1000 ** 3:6.2f} B\n"
        f"Target ratio      : {weight_config.weight:6.4f}\n"
        f"Repetition factor : {sample_rate}\n"
    )
    print(msg)
    return {
        "name": weight_config.name,
        "target_ratio": weight_config.weight,
        "repetition_factor": sample_rate,
        "paths": [base_uri.rstrip("/") + "/*.npy"],
    }


def main():

    weights = []

    # load everything from here except pes2o (not used) and dclm (we will load from snazzy2)
    raw_weights_path = Path(__file__).parent / "s2pdf/full_pstar_7rep_dclm_stackedu_conditional.json"
    with open(raw_weights_path, "r") as f:
        other_weights = [WeightConfig.from_dict(w) for w in json.load(f)]
        weights.extend([w for w in other_weights if w.domain in cross_source_pstar])

    snazzy2_weights_path = Path(__file__).parent / "all-dressed/vigintiles/snazzy2.json"
    with open(snazzy2_weights_path, "r") as f:
        snazzy2_weights = [
            WeightConfig(
                domain="all-dressed",
                topic=(topic_quality := k.split("/"))[0],
                quality=topic_quality[1],
                weight=v * cross_source_pstar["all-dressed"],
            )
            for k, v in json.load(f).items()
        ]
        weights.extend(snazzy2_weights)

    # gotta round up to 1 the weights
    if (sum_weights := sum(w.weight for w in weights)) < 1:
        print(f"Sum of weights is {sum_weights:.4f}, rounding up to 1")
        weights = [
            WeightConfig(domain=w.domain, topic=w.topic, quality=w.quality, weight=w.weight / sum_weights)
            for w in weights
        ]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for weight_config in weights:
            future = executor.submit(make_one_config, weight_config=weight_config, token_target=TOKEN_TARGET)
            futures.append(future)

        sources = []
        for future in as_completed(futures):
            try:
                source = future.result()
                sources.append(source)
            except Exception as e:
                print(f"Error making config for {weight_config.name}: {e}")
                for future in futures:
                    future.cancel()
                raise e

    # remove sources with target ratio = 0
    sources = [source for source in sources if source["target_ratio"] > 0]

    # sort sources by name
    sources = sorted(sources, key=lambda x: x["name"])

    # make and write config
    config = {"dataset": {"sources": sources}}
    with open(Path(__file__).parent / "ablation.yaml", "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    main()
