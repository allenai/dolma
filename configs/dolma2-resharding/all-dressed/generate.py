from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import boto3
import yaml
from tqdm import tqdm
from urllib.parse import urlparse
from pathlib import Path
import sys

cross_source_pstar = {
    "finemath-3plus": 0.025340376929054265,
    "arxiv": 0.008284928006565282,
    "s2orc": 0.011392852484537662,
    "wikipedia": 0.000416156026289699,
    "all-dressed": 0.752076181317783,
    "s2pdf": 0.134320140174659,
    "stack-edu": 0.06816936506111054,
}



src_tokenized_path = "s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer"
dst_tokenized_path = "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed"

token_target = 6_000_000_000_000


def get_size_of_prefix(prefix: str, ext: str = ".npy") -> int:
    bucket, prefix = (p := urlparse(prefix)).netloc, p.path.lstrip("/")
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    total_size = 0
    for obj in response.get("Contents", []):
        if "Key" not in obj:
            continue

        if not obj["Key"].endswith(ext):
            continue

        if "Size" not in obj:
            continue

        total_size += int(obj["Size"])

    return total_size


def get_size_of_vig(topic: str, vig: str) -> tuple[str, str, int]:
    return topic, vig, get_size_of_prefix(f"{src_tokenized_path}/{topic}/{vig}/") // 4


script_dir = Path(__file__).parent

def single_config(path: str | Path):
    path = Path(path)

    print("=" * 40)
    print(f"Processing {path}")

    with open(path, "r") as f:
        flat_vig_config = json.load(f)

    vig_config_hier = {}
    for k, v in flat_vig_config.items():
        topic, vig = k.split("/")
        vig_config_hier.setdefault(topic, {})[vig] = v

    sizes = {}

    with ThreadPoolExecutor() as executor:
        futures = []
        for topic, vig_config in sorted(vig_config_hier.items()):
            for vig, _ in sorted(vig_config.items()):
                futures.append(
                    executor.submit(get_size_of_vig, topic=topic, vig=vig)
                )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Getting sizes"):
            topic, vig, size = future.result()
            sizes.setdefault(topic, {})[vig] = size


    desired_total_size = token_target * cross_source_pstar["all-dressed"]
    natural_total_size = sum(sum(v.values()) for v in sizes.values())

    print(f"Found {natural_total_size / 1024 ** 3:.1f}B tokens in {len(sizes)} topics")
    print(f"Desired size: {desired_total_size / 1024 ** 3:.1f}B tokens")
    print("\n---\n")


    for topic, vig_config in sizes.items():
        print(f"Topic          : {topic}")
        print(f"Natural tokens : {sum(vig_config.values()) / 1024 ** 3:.1f}B")
        print()

        topic_config = {
            "source_prefixes": [],
            "destination_prefix": f"{dst_tokenized_path}-{path.stem}/{topic}",
            "local_tempdir": f"/mnt/raid0/resharding/all-dressed/{topic}",
            "max_num_files": 32
        }

        total_topic_size = 0
        for vig, size in sorted(vig_config.items()):
            print(f"Vigintile     : {vig}")
            print(f"Natural tokens : {size / 1024 ** 3:.1f}B")
            desired_vig_size = desired_total_size * vig_config_hier[topic][vig]
            print(f"Desired tokens : {desired_vig_size / 1024 ** 3:.1f}B")
            print(f"Sampling rate  : {desired_vig_size / size:.2f}x")
            print()

            topic_config["source_prefixes"].append({
                "prefix": f"{src_tokenized_path}/{topic}/{vig}",
                "sample_rate": desired_vig_size / size,
            })
            total_topic_size += size

        print(f"Desired topic size : {total_topic_size / 1024 ** 3:.1f}B")
        print(f"Ratio of web       : {total_topic_size / desired_total_size:.1%}")
        print('\n---\n')

        dest = script_dir / f"config/{path.stem}/{topic}.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "w") as f:
            yaml.dump(topic_config, f)



if __name__ == "__main__":
    single_config(Path(sys.argv[1]))
