import argparse
import json
import multiprocessing
import os
import random
from dataclasses import dataclass
from urllib.parse import urlparse

import evaluate
import fsspec
import jq
import numpy as np
import smart_open
import torch
from msgspec.json import Decoder
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer

from smart_open.compression import (
    _handle_zstd,
    register_compressor,
)

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0


@dataclass(frozen=True)
class Document:
    text: str
    label: int


def read_file(path: str, label: int | None = None, selector: str | None = None, instances_limit: int = None,
              length_limit: int = None) -> list[Document]:
    if selector is not None:
        compiled_selector = jq.compile(selector)
        label_fn = lambda row: str(compiled_selector.input(row).first())
    elif label is not None:
        label_fn = lambda row: label
    else:
        raise ValueError("Either `label` or `selector` must be provided")

    decoder = Decoder()
    documents = []

    with smart_open.open(path) as f:
        for line in f:
            row = decoder.decode(line)
            label = label_fn(row)

            text = row["text"]
            if length_limit is not None:
                text = text[:length_limit]

            documents.append(Document(text=text, label=label))

            if instances_limit is not None and len(documents) >= instances_limit:
                break

    return documents


@dataclass(frozen=True)
class DataConfig:
    path: str
    label: str | None = None
    selector: str | None = None
    limit: int | None = None

    def expand(self, fs: fsspec.AbstractFileSystem | None = None) -> list["DataConfig"]:
        fs = fs or fsspec.get_filesystem_class(urlparse(self.path).scheme)()
        base_url_scheme = f"{urlparse(self.path).scheme}://"
        paths = [str(p) for p in fs.glob(self.path)] if "*" in self.path else [self.path]
        paths = [path if path.startswith(base_url_scheme) else f"{base_url_scheme}{path}" for path in paths]

        return [DataConfig(path=path, label=self.label, selector=self.selector, limit=self.limit) for path in paths]


def expand_config(config: DataConfig) -> list[DataConfig]:
    return config.expand()


def process_file(config: DataConfig) -> list[Document]:
    return read_file(path=config.path, label=config.label, selector=config.selector, instances_limit=config.limit)


class ClassifierDataset(Dataset):
    def __init__(
            self,
            configs: list[DataConfig],
            workers: int = 1,
    ):
        with multiprocessing.Pool(workers) as pool:
            expanded_configs = list(
                tqdm(
                    pool.imap_unordered(expand_config, configs),
                    total=len(configs),
                    desc="Expanding configs",
                )
            )

        expanded_configs = [item for sublist in expanded_configs for item in sublist][:10]

        with multiprocessing.Pool(workers) as pool:
            self.documents = list(
                tqdm(
                    pool.imap_unordered(process_file, expanded_configs),
                    total=len(expanded_configs),
                    desc="Reading files",
                )
            )

        self.documents = [item for sublist in self.documents for item in sublist]

        print(f"Read {len(self.documents):,} documents from {len(expanded_configs)} configs")

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx]


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def freeze_model_except_classifier(model):
    # Freeze all layers
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True


def collate_fn(batch, tokenizer):
    texts = [item.text for item in batch]
    labels = [item.label for item in batch]

    tokenized = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

    tokenized["labels"] = torch.tensor(labels)

    return tokenized


class Classifier:
    def __init__(
            self,
            base_model_name: str,
    ):
        self._base_model_name = base_model_name

        self._tokenizer = None
        self._model = None

    def fit(
            self,
            dataset: ClassifierDataset,
            validation_set_size: int = 1000,
            max_steps: int = 500,
    ) -> AutoModelForSequenceClassification:
        train_dataset, val_dataset = self._shuffle_split_test_val(dataset, validation_set_size)

        self._tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        self._model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
        freeze_model_except_classifier(self._model)

        training_args = TrainingArguments(
            output_dir="test_trainer",
            report_to="wandb" if args.use_wandb else "none",
            dataloader_num_workers=args.num_workers,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=1,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=50,
            max_steps=max_steps,
            load_best_model_at_end=True,
            save_total_limit=1,
        )

        if args.use_wandb:
            os.environ["WANDB_PROJECT"] = args.wandb_project
            os.environ["WANDB_ENTITY"] = args.wandb_entity

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=lambda batch: collate_fn(batch, self._tokenizer),
        )

        trainer.train()

    @staticmethod
    def _shuffle_split_test_val(dataset, validation_set_size):
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        train_indices = indices[:-validation_set_size]
        eval_indices = indices[-validation_set_size:]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, eval_indices)
        return train_dataset, val_dataset

    def score(self, test_dataset: ClassifierDataset):
        if self._model is None:
            raise ValueError("Model must be fit before testing")

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  collate_fn=lambda batch: collate_fn(batch, self._tokenizer))

        results = []

        self._model.eval()
        for batch in tqdm(test_loader, desc="Scoring test set"):
            with torch.no_grad():
                # move to cuda
                batch = {k: v.cuda() for k, v in batch.items() if k != "labels"}

                outputs = self._model(**batch)
                logits = outputs.logits
                score = torch.softmax(logits, dim=-1)[:, POSITIVE_LABEL]
                text = self._tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

                for i in range(len(logits)):
                    result = {"text": text[i], "score": score[i].item()}
                    results.append(result)

        return results



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a positive/negative classifier on a set of data sources"
    )
    parser.add_argument("-m", "--model-name", type=str, help="Hugging Face model name",
                        default="Snowflake/snowflake-arctic-embed-m")
    parser.add_argument("-p", "--positive-sources", type=str, nargs="+", required=True, help="Positive data sources")
    parser.add_argument("-n", "--negative-sources", type=str, nargs="+", required=True, help="Negative data sources")
    parser.add_argument("--test-source", type=str, help="Test data source to score (no labels)")
    parser.add_argument("--test-source-instance-limit", type=int, default=100000, help="Number of instances to load from the test source")
    parser.add_argument("--test-results-path", type=str, default="test_results.jsonl", help="Path to jsonl filename to write test scores")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers to use")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum number of training steps")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="qc", help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default="ai2-llm", help="Weights & Biases entity name")
    opts = parser.parse_args()

    return opts


def main(args: argparse.Namespace):
    positive_configs = [DataConfig(path=path, label=POSITIVE_LABEL, limit=10000) for path in args.positive_sources]
    negative_configs = [DataConfig(path=path, label=NEGATIVE_LABEL, limit=10000) for path in args.negative_sources]

    dataset = ClassifierDataset(positive_configs + negative_configs, workers=args.num_workers)

    classifier = Classifier(base_model_name=args.model_name)
    classifier.fit(dataset, max_steps=args.max_steps)

    if args.test_source:
        test_config = DataConfig(path=args.test_source, label=-1, limit=args.test_source_instance_limit)
        test_dataset = ClassifierDataset([test_config], workers=args.num_workers)
        test_results = classifier.score(test_dataset)

        with open(args.test_results_path, "w") as f:
            for result in test_results:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    args = parse_args()

    # add additional extension for smart_open
    register_compressor(".zstd", _handle_zstd)

    main(args)
