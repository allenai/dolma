import json
from typing import List
import smart_open


SRC_BASE = "s3://ai2-llm/pretraining-data/sources/cc-news"
SRC_PRFX = "v1-resiliparse"
LANG_THR = 100_000
DST_BASE = "${oc.env:HOME}/ai2-llm/pretraining-data/sources/cc-news"
DST_PRFX = f"v2-resiliparse-l{LANG_THR // 1000}k"


def base_stream_config(lang: str, year: int, months: List[int]):
    return {
        "name": f"cc-news_{year:04d}_{lang}",
        "documents": [
            f"{SRC_BASE}/{SRC_PRFX}/documents/{year:04d}-{month:02d}/*.zst"
            for month in months
        ],
        "compression": {"input": "zst", "output": "zst"},
        "output": {
            "path": f"{DST_BASE}/{DST_PRFX}/documents/{lang}/{year:04d}",
            "max_size_in_bytes": 10_000_000_000,
        },
        "attributes": ["ft_lang_id_1e2", "dolma_v2_tokenizer"],
        "filter": {
            "include": [
                # at least 50 tokens
                ".attributes.dolma_v2_tokenizer__dolma_v2_tokenizer__length[0][-1] >= 50",
                # make sure the language is present and the confidence is high enough and that it is the highest confidence
                (
                    f"(.attributes.ft_lang_id_1e2__ft_lang_id_1e2__{lang} != null) and "
                    + f"(.attributes.ft_lang_id_1e2__ft_lang_id_1e2__{lang}[0][-1] >= 0.5) and "
                    + f'((.attributes | to_entries | map(select(.key | startswith("ft_lang_id_1e2__ft_lang_id_1e2__"))) | max_by(.value) | .key ) == "ft_lang_id_1e2__ft_lang_id_1e2__{lang}")'
                ),
            ],
            "syntax": "jq",
        },
    }


def main():
    with smart_open.open("s3://ai2-llm/stats/cc-news/v1-resiliparse/attributes/ft_lang_id_1e2_summary.json") as f:
        lang_counts = json.load(f)

    languages = {k: v for k, v in lang_counts.items() if v >= LANG_THR}

    streams = []
    for year in range(2016, 2025):
        if year == 2016:
            months = list(range(8, 13))
        elif year == 2024:
            months = list(range(1, 8))
        else:
            months = list(range(1, 13))

        streams.extend([base_stream_config(lang, year, months) for lang in languages])

    with smart_open.open("configs/cc-news/mix_v2.json", "wt") as f:
        json.dump({"processes": 1, "streams": streams}, f, indent=2)


if __name__ == "__main__":
    main()
