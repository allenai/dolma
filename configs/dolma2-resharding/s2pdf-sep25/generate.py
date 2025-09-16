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


rename_categories = {
    "adult": "adult_content",
    "art_design": "art_and_design",
    "crime_law": "crime_and_law",
    "education_jobs": "education_and_jobs",
    "entertainment": "entertainment",
    "fashion_beauty": "fashion_and_beauty",
    "finance_business": "finance_and_business",
    "food_dining": "food_and_dining",
    "games": "games",
    "hardware": "electronics_and_hardware",
    "health": "health",
    "history": "history_and_geography",
    "home_hobbies": "home_and_hobbies",
    "industrial": "industrial",
    "literature": "literature",
    "politics": "politics",
    "religion": "religion",
    "science_tech": "science_math_and_technology",
    "social_life": "social_life",
    "software": "software",
    "software_dev": "software_development",
    "sports_fitness": "sports_and_fitness",
    "transportation": "transportation",
    "travel": "travel_and_tourism",
}

with open(Path(__file__).parent / "full_pstar_7rep_dclm_stackedu_conditional.json", "r") as f:
    full_pstar = json.load(f)

code_base_tokenized_path = "s3://ai2-llm/preprocessed/s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality_datadelve_norefs_mdtables_v2_denylisted_reshard_denyagain"

# save locally first
destination_path = "/mnt/raid0/ai2-llm/preprocessed/dolma2-0625/v0.2/allenai/dolma2-tokenizer/s2pdf"

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

    s2pdf_pstar = {
        d['domain'].replace("s2pdf:", ""): d['weight']
        for d in full_pstar if d['domain'].startswith("s2pdf:")
    }

    sizes = {}
    for lang in tqdm.tqdm(rename_categories, desc="Getting sizes"):
        sizes[lang] = get_size_of_prefix(f"{code_base_tokenized_path}/{lang}/") // 4 # 4 bytes per token


    assert math.isclose(sum(s2pdf_pstar.values()), cross_source_pstar["s2pdf"], rel_tol=1e-6)
    desired_code_size = token_target * sum(s2pdf_pstar.values())
    natural_code_size = sum(sizes.values())

    print(f"Found {natural_code_size / 1024 ** 3:.1f}B tokens in {len(sizes)} FoS")
    print(f"Desired s2pdf size: {desired_code_size / 1024 ** 3:.1f}B tokens")
    print("\n")

    total_size_computed = 0

    for i, (lang, size) in enumerate(sizes.items()):
        if size == 0:
            print(f"[WARNING] {lang} has no tokens; skipping\n")
            continue

        print(f"Subset         : {lang}")
        print(f"Natural tokens : {size / 1024 ** 3:.1f}B")
        desired_size = token_target * s2pdf_pstar[rename_categories[lang]]
        print(f"Desired tokens : {desired_size / 1024 ** 3:.1f}B")
        print(f"Sampling rate  : {desired_size / size:.2f}x")
        total_size_computed += desired_size

        # # if destination exists, then get the final size and print how much we are off
        # dest_size = get_size_of_prefix(f"{destination_path}/{lang}/") // 4
        # if dest_size > 0:
        #     print(f"Final size     : {dest_size / 1024 ** 3:.1f}B")
        #     print(f"Off by         : {(dest_size - desired_size) / desired_size:.2%}")

        print('\n')

        # max num files is:
        # - at least ceil(desired_size / size)
        # if asking < 10 billion tokens, then max num files is 2
        # if asking < 100 billion tokens, then max num files is 4
        # else it is 8

        max_num_files = math.ceil(desired_size / size)
        if desired_size < 10 * 10 ** 9:
            max_num_files = min(max_num_files, 2)
        elif desired_size < 100 * 10 ** 9:
            max_num_files = min(max_num_files, 4)
        else:
            max_num_files = min(max_num_files, 8)

        if desired_size == 0:
            print(f"Skipping {lang} because desired size is 0")
            continue

        lang_config = {
            "source_prefixes": [
                {
                    "prefix": f"{code_base_tokenized_path}/{lang}",
                    "sample_rate": desired_size / size,
                }
            ],
            "destination_prefix": f"{destination_path}/{rename_categories[lang]}",
            "local_tempdir": f"/mnt/raid0/resharding/s2pdf/{lang}",
            "max_workers": max_num_files,
            "max_num_files": max_num_files,
            "random_seed": 42 + i
        }
        dest = script_dir / f"config/{rename_categories[lang]}.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "w") as f:
            yaml.dump(lang_config, f)

    print(f"\n\nTotal size computed: {total_size_computed / 1024 ** 3:.1f}B")


if __name__ == "__main__":
    main()
