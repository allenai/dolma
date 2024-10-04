from copy import deepcopy
from dolma.cli.tokenizer import TokenizationConfig, TokenizerConfig, TokenizerCli
from multiprocessing import cpu_count
import numpy as np
import os

# destination: ${oc.env:HOME}/ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/open-web-math/train/allenai/dolma2-tokenizer
# documents:
#   - s3://ai2-llm/pretraining-data/sources/proof-pile-2/v0_decontaminated/documents/open-web-math/train/*

# processes: 128
# seed: 3920
# max_size: 4_294_967_296
# dtype: uint32

# tokenizer:
#   name_or_path: allenai/dolma2-tokenizer
#   bos_token_id: null
#   eos_token_id: 100257
#   pad_token_id: 100277
#   segment_before_tokenization: false
#   encode_special_tokens: true

def main():
    base_config = TokenizationConfig(
        documents=[],
        destination=f"{os.environ['HOME'].rstrip('/')}/ai2-llm/preprocessed/mathpile",
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

    for name in ["MathPile", "MathPile_Commercial"]:
        for split in ["train", "validation"]:
            for subset in ["arXiv", "commoncrawl", "proofwiki", "stackexchange", "textbooks", "wikipedia"]:
                config = deepcopy(base_config)
                config.documents = [
                    f"/data/mathpile/v0/documents/{name}/{split}/{subset}/*"
                ]
                config.destination = f"{config.destination}/{name}/{split}/{subset}/{config.tokenizer.name_or_path}"
                TokenizerCli.run(config)

if __name__ == "__main__":
    main()
