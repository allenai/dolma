import json
import smart_open
import yaml


def base_stream_config(lang: str, year: int, month: int):
    return {
        "name": f"cc-news_{year:04d}-{month:02d}_{lang}",
        "documents": [
            f"s3://ai2-llm/pretraining-data/sources/cc-news/v1-resiliparse/documents/${year:04d}-${month:02d}/*.zst"
        ],
        "compression": {"input": "zst", "output": "zst"},
        "output": {
            "path": f"s3://ai2-llm/pretraining-data/sources/cc-news/v2-resiliparse/documents/{lang}/${year:04d}-${month:02d}",
            "max_size_in_bytes": 1_000_000_000,
        },
        "attributes": ["ft_lang_id_1e2", "dolma_v2_tokenizer"],
        "filter": {
            "include": [
                # at least 50 tokens
                "(.attributes.dolma_v2_tokenizer != null) and (.attributes.dolma_v2_tokenizer[0][-1] >= 50)",
                # make sure the language is present and the confidence is high enough and that it is the highest confidence
                (
                    f"(.attributes.ft_lang_id_1e2__ft_lang_id_1e2__{lang} != null) and "
                    + f"(.attributes.ft_lang_id_1e2__ft_lang_id_1e2__{lang}[0][-1] >= 0.5) and"
                    + f'((.attributes | to_entries | map(select(.key | startswith("ft_lang_id_1e2__ft_lang_id_1e2__"))) | max_by(.value) | .key ) == ft_lang_id_1e2__ft_lang_id_1e2__{lang})',
                ),
            ],
            "syntax": "jq",
        },
    }


def main():
    with smart_open.open("s3://ai2-llm/stats/cc-news/v1-resiliparse/attributes/ft_lang_id_1e2_summary.json") as f:
        lang_counts = json.load(f)

    languages = {k: v for k, v in lang_counts.items() if v >= 10000}

    streams = [
        base_stream_config(lang, year, month)
        for lang in languages
        for year in range(2016, 2024)
        for month in range(1, 13)
        if (year > 2016 or month > 8) and (year < 2024 or month < 8)
    ]

    with smart_open.open("configs/cc-news/mix_v2.sh", "wt") as f:
        yaml.dump({"streams": streams, "processes": 1}, f)
