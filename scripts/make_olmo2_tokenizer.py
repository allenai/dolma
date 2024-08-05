import tqdm
from datasets import load_dataset
import tiktoken
from transformers import GPT2TokenizerFast

hf_tokenizer = GPT2TokenizerFast.from_pretrained("allenai/dolma2-tokenizer")
og_tokenizer = tiktoken.encoding_for_model("gpt-4")

# dataset = load_dataset("xnli", "all_languages")
dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", streaming=True)


cnt = 10_000
for item in tqdm.tqdm(dataset["train"]):
    encoded1 = og_tokenizer.encode(item["text"])
    encoded2 = hf_tokenizer.encode(item["text"])

    assert encoded1 == encoded2, f'encoding "{item["text"]}" is incorrect. "{encoded1}" != "{encoded2}"'

    decoded1 = og_tokenizer.decode(encoded1)
    decoded2 = hf_tokenizer.decode(encoded2, skip_special_tokens=True)

    assert decoded1 == decoded2, f'decoding "{item["text"]}" is incorrect. "{decoded1}" != "{decoded2}"'

    cnt -= 1
    if cnt == 0:
        break
