import os
import click
from dolma.core.paths import cached_path
import numpy as np
from transformers import AutoTokenizer


@click.command()
@click.argument("tokenized_file")
@click.option("--tokenizer-name-or-path", default="allenai/dolma2-tokenizer")
@click.option("--dtype", default="uint32")
@click.option("--chunk-size", default=1024**2, type=int)
def inspect_tokenized(tokenized_file: str, tokenizer_name_or_path: str, dtype: str, chunk_size: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    print('Vocab size:', tokenizer.vocab_size)
    print('BOS token:', tokenizer.bos_token_id)
    print('EOS token:', tokenizer.eos_token_id)
    print('PAD token:', tokenizer.pad_token_id)
    print('UNK token:', tokenizer.unk_token_id)

    path = cached_path(tokenized_file)
    size = os.path.getsize(path)
    data = np.memmap(path, dtype=dtype, mode='r', shape=(size // 2,))

    collection = []
    i = 0
    while i < len(data):
        chunk = data[i : i + chunk_size]
        i += chunk_size

        while (chunk == tokenizer.eos_token_id).any():
            # split chunk into before and after eos
            eos_idx = np.where(chunk == tokenizer.eos_token_id)[0][0] + 1
            collection.extend(chunk[:eos_idx].tolist())
            output = tokenizer.decode(collection)
            print('#' * os.get_terminal_size().columns)
            print(output)
            input("#" * os.get_terminal_size().columns)
            # reset collection
            collection = []
            chunk = chunk[eos_idx:]

        collection.extend(chunk.tolist())


if __name__ == "__main__":
    inspect_tokenized()
