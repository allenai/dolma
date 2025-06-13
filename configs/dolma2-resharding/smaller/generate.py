import math
import boto3
import tqdm
import yaml
from urllib.parse import urlparse
from pathlib import Path
import json

cross_source_pstar = {
    "finemath-3plus": 0.025340376929054265,
    "arxiv": 0.008284928006565282,
    "s2orc": 0.011392852484537662,
    "wikipedia": 0.000416156026289699,
    "all-dressed": 0.752076181317783,
    "s2pdf": 0.134320140174659,
    "stack-edu": 0.06816936506111054,
}

paths = {
    "finemath-3plus": "s3://ai2-llm/preprocessed/olmo3-final/math/allenai/dolma2-tokenizer/finemath_3plus_all",
    "wikipedia": "s3://ai2-llm/preprocessed/wikipedia-dolma-0823/allenai/dolma2-tokenizer",
    "arxiv": "s3://ai2-llm/preprocessed/proof-pile-2/v0_decontaminated-0625_tokenized/arxiv/train/allenai/dolma2-tokenizer"
}

code_base_tokenized_path = "s3://ai2-llm/preprocessed/olmo3-final/s2pdfs/allenai/dolma2-tokenizer"
destination_path = "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer"

token_target = 6_000_000_000_000


def get_size_of_prefix(prefix: str, ext: str = ".npy") -> int:
    bucket, prefix = (p := urlparse(prefix)).netloc, p.path.lstrip("/")
    s3 = boto3.client("s3")

    total_size = 0
    continuation_token = None

    while True:
        if continuation_token:
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                ContinuationToken=continuation_token
            )
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


script_dir = Path(__file__).parent


def main():
    sizes = {}

    for subset, og_prefix in tqdm.tqdm(paths.items(), desc="Getting sizes"):
        sizes[subset] = get_size_of_prefix(f"{og_prefix}/") // 4 # 4 bytes per token

    for subset, natural_size in sizes.items():
        desired_size = token_target * cross_source_pstar[subset]
        print(f"Subset       : {subset}")
        print(f"Natural tokens : {natural_size / 1024 ** 3:.1f}B")
        print(f"Desired tokens : {desired_size / 1024 ** 3:.1f}B")
        print(f"Sampling rate  : {desired_size / natural_size:.2f}x")

        destination_prefix = f"{destination_path}/{subset}"

        # if destination exists, then get the final size nad print how much we are off
        dest_size = get_size_of_prefix(f"{destination_prefix}/") // 4
        if dest_size > 0:
            print(f"Final size     : {dest_size / 1024 ** 3:.1f}B")
            print(f"Off by         : {(dest_size - desired_size) / desired_size:.2%}")

        print('\n')

        lang_config = {
            "source_prefixes": [
                {
                    "prefix": f"{paths[subset]}/",
                    "sample_rate": desired_size / natural_size,
                }
            ],
            "destination_prefix": destination_prefix,
            "local_tempdir": f"/mnt/raid0/resharding/{subset}",
            "max_num_files": 8
        }
        dest = script_dir / f"config/{subset}.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "w") as f:
            yaml.dump(lang_config, f)


if __name__ == "__main__":
    main()
