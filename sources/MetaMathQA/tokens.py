from copy import deepcopy
from dolma.cli.tokenizer import TokenizationConfig, TokenizerConfig, TokenizerCli
from multiprocessing import cpu_count
import numpy as np
import os


def main():
    tokenizer = "allenai/dolma2-tokenizer"
    base_source = "s3://ai2-llm/pretraining-data/sources"
    base_destination = f"{os.environ['HOME'].rstrip('/')}/ai2-llm/preprocessed"

    config = TokenizationConfig(
        documents=[f"{base_source}/meta-math_MetaMathQA/v0/documents/train/*"],
        destination=f"{base_destination}/meta-math_MetaMathQA/v0/tokens/{tokenizer}",
        tokenizer=TokenizerConfig(
            name_or_path=tokenizer,
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
        seed=42,
    )
    TokenizerCli.run(config)

if __name__ == "__main__":
    main()
