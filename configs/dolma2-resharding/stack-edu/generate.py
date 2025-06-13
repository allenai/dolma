import boto3
import yaml
from urllib.parse import urlparse
from pathlib import Path

cross_source_pstar = {
    "finemath-3plus": 0.025340376929054265,
    "arxiv": 0.008284928006565282,
    "pes2o": 0.011392852484537662,
    "wikipedia": 0.000416156026289699,
    "dclm": 0.752076181317783,
    "s2pdf": 0.134320140174659,
    "stack-edu": 0.06816936506111054,
}

stack_edu_pstar = {
    "C": 0.040545413474083136,
    "Cpp": 0.11992423990590854,
    "CSharp": 0.06145614902228962,
    "Go": 0.013141668585880971,
    "Java": 0.15971742894160593,
    "JavaScript": 0.08711993898613768,
    "Markdown": 0.16641522916681814,
    "PHP": 0.060681232466575974,
    "Python": 0.18292382056422074,
    "Ruby": 0.01313841950835558,
    "Rust": 0.014023586747942062,
    "Shell": 0.025543226210598954,
    "SQL": 0.018239409453020123,
    "Swift": 0.014179755937669661,
    "TypeScript": 0.02295048102888894,
}

code_base_tokenized_path = "s3://ai2-llm/preprocessed/stack-edu/allenai/dolma2-tokenizer"
destination_path = "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu"

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
    for lang in stack_edu_pstar:
        sizes[lang] = get_size_of_prefix(f"{code_base_tokenized_path}/{lang}/") // 4 # 4 bytes per token

    desired_code_size = token_target * cross_source_pstar["stack-edu"]
    natural_code_size = sum(sizes.values())

    print(f"Found {natural_code_size / 1024 ** 3:.1f}B tokens in {len(sizes)} languages")
    print(f"Desired code size: {desired_code_size / 1024 ** 3:.1f}B tokens")
    print("\n")

    for lang, size in sizes.items():
        print(f"Language       : {lang}")
        print(f"Natural tokens : {size / 1024 ** 3:.1f}B")
        desired_size = desired_code_size * stack_edu_pstar[lang]
        print(f"Desired tokens : {desired_size / 1024 ** 3:.1f}B")
        print(f"Sampling rate  : {desired_size / size:.2f}x")

        # if destination exists, then get the final size nad print how much we are off
        dest_size = get_size_of_prefix(f"{destination_path}/{lang}/") // 4
        if dest_size > 0:
            print(f"Final size     : {dest_size / 1024 ** 3:.1f}B")
            print(f"Off by         : {(dest_size - desired_size) / desired_size:.2%}")

        print('\n')

        lang_config = {
            "source_prefixes": [
                {
                    "prefix": f"{code_base_tokenized_path}/{lang}",
                    "sample_rate": desired_size / size,
                }
            ],
            "destination_prefix": f"{destination_path}/{lang}",
            "local_tempdir": f"/mnt/raid0/resharding/stack-edu/{lang}",
            "max_num_files": 8
        }
        dest = script_dir / f"config/{lang}.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "w") as f:
            yaml.dump(lang_config, f)


if __name__ == "__main__":
    main()
