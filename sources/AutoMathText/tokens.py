from copy import deepcopy
from dolma.cli.tokenizer import TokenizationConfig, TokenizerConfig, TokenizerCli
from multiprocessing import cpu_count
import numpy as np
import os


def main():
    base_config = TokenizationConfig(
        documents=[],
        destination=f"{os.environ['HOME'].rstrip('/')}/ai2-llm/preprocessed/math-ai_AutoMathText/v0",
        tokenizer=TokenizerConfig(
            name_or_path="allenai/dolma2-tokenizer",
            bos_token_id=None,
            eos_token_id=100257,
            pad_token_id=100277,
            segment_before_tokenization=False,
            encode_special_tokens=True,
        ),
        processes=cpu_count(),
        max_size=100_000_000,
        dtype='uint32',
        sample_ring_prop=True,
    )


    for subset in ["arxiv/*", "code/*", "web"]:
        config = deepcopy(base_config)
        config.documents = [
            f"/data/math-ai_AutoMathText/v0/documents/{subset}/*.jsonl.gz"
        ]
        config.destination = f"{config.destination}/{subset.rstrip('/*')}/{config.tokenizer.name_or_path}"
        TokenizerCli.run(config)

if __name__ == "__main__":
    main()
