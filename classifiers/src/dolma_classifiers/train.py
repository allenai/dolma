import argparse
import json
import multiprocessing
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from urllib.parse import urlparse

import evaluate
import fsspec
import jq
import numpy as np
import smart_open
import torch
import boto3

from multiprocessing.dummy import Pool as ThreadPool
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

LOCAL_TMP_PATH = "/tmp/qc_model"


@dataclass(frozen=True)
class Document:
    text: str
    label: int


def read_file(path: str, label: int | None = None, selector: str | None = None, instances_read_limit: int = None,
              sample_per_file: int | None = None, filter_rows: Callable[[dict], bool] = None) -> list[Document]:
    if selector is not None:
        compiled_selector = jq.compile(selector)
        label_fn = lambda row: compiled_selector.input(row).first()
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

            if filter_rows is not None and not filter_rows(row):
                continue

            documents.append(Document(text=text, label=label))

            if 0 < instances_read_limit <= len(documents):
                break

    if len(documents) > sample_per_file:
        documents = random.sample(documents, sample_per_file)

    return documents


def download_model_from_s3(remote_path: str, bucket: str, local_path: str) -> None:
    s3 = boto3.client("s3")
    for file in ["model.safetensors", "config.json"]:
        print(f"Downloading {file}")
        remote_file = os.path.join(remote_path, file)
        local_file = os.path.join(local_path, file)
        s3.download_file(bucket, remote_file.lstrip("/"), local_file)

def upload_directory_to_s3(directory_path: str, bucket: str, folder: str) -> None:
    s3 = boto3.client("s3")
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            print(f"Uploading {file}")
            file_path: str = os.path.join(root, file)
            s3_path: str = os.path.join(folder, os.path.relpath(file_path, directory_path))
            s3.upload_file(file_path, bucket, s3_path)


@dataclass(frozen=True)
class DataConfig:
    path: str
    label: str | None = None
    selector: str | None = None
    sample: int | None = None
    filter: str | None = None

    def expand(self, fs: fsspec.AbstractFileSystem | None = None) -> list["DataConfig"]:
        fs = fs or fsspec.get_filesystem_class(urlparse(self.path).scheme)()
        base_url_scheme = f"{urlparse(self.path).scheme}://" if urlparse(self.path).scheme else ""
        paths = [str(p) for p in fs.glob(self.path)] if "*" in self.path else [self.path]
        paths = [path if path.startswith(base_url_scheme) else f"{base_url_scheme}{path}" for path in paths]

        sample_per_file = self.sample // len(paths) if self.sample is not None else 0
        sample_per_file_remainder = self.sample % len(paths) if self.sample is not None else 0

        data_configs = [DataConfig(path=path, label=self.label, selector=self.selector, sample=sample_per_file, filter=self.filter) for path in paths]
        data_configs[-1] = DataConfig(path=paths[-1], label=self.label, selector=self.selector, sample=sample_per_file + sample_per_file_remainder, filter=self.filter)

        return data_configs


def expand_config(config: DataConfig) -> list[DataConfig]:
    return config.expand()


def process_file(config: DataConfig) -> list[Document]:
    instances_read_limit = config.sample * 100  # read 100x the sample size to ensure we get a random sample yet do not read too much
    return read_file(path=config.path, label=config.label, selector=config.selector, sample_per_file=config.sample,
                     instances_read_limit=instances_read_limit, filter_rows=config.filter)


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

        expanded_configs = [item for sublist in expanded_configs for item in sublist]

        print(f"Expanded {len(configs)} configs to {len(expanded_configs)} configs")
        print(f"Loading {sum(c.sample for c in expanded_configs):,} samples")

        with ThreadPool(workers) as pool:
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
            base_model_name: str | None = None,
            load_model: str | None = None,
    ):
        if not base_model_name and not load_model:
            raise ValueError("Either `base_model_name` or `load_model` must be provided")

        self._base_model_name = base_model_name

        self._tokenizer = None
        self._model = None

        if load_model:
            parsed = urlparse(str(load_model))
            if parsed.scheme == "s3":
                os.makedirs(LOCAL_TMP_PATH, exist_ok=True)
                load_model = LOCAL_TMP_PATH
                download_model_from_s3(remote_path=parsed.path, bucket=parsed.netloc, local_path=load_model)
            self._model = AutoModelForSequenceClassification.from_pretrained(load_model).cuda()
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(load_model)
            except OSError:
                # not sure why this doesn't work out of the box, but in case there's an error we can load the original
                # base model from the config file
                config = json.load(open(os.path.join(load_model, "config.json")))
                self._tokenizer = AutoTokenizer.from_pretrained(config["_name_or_path"])

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
            output_dir=args.local_save_path,
            report_to="wandb" if args.use_wandb else "none",
            dataloader_num_workers=args.num_workers,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=1,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=50,
            save_steps=50,
            max_steps=max_steps,
            load_best_model_at_end=True,
            save_total_limit=1,
            run_name=args.run_name,
            log_level="debug",
            save_only_model=True,
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

    def score(self, test_dataset: ClassifierDataset, batch_size: int = 64):
        if self._model is None:
            raise ValueError("Model must be fit before testing")

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
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
    parser.add_argument("-r", "--run-name", type=str, default="qc_train", help="Run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-num-positive", type=int, default=10000, help="Maximum number of positive instances to load")
    parser.add_argument("--max-num-negative", type=int, default=10000, help="Maximum number of negative instances to load")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers to use")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum number of training steps")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="qc", help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default="ai2-llm", help="Weights & Biases entity name")
    parser.add_argument("--local-save-path", type=str, default="/tmp/qc_model", help="Local path to save model")
    parser.add_argument("--upload-to-s3", action="store_true", help="Upload model to S3")
    parser.add_argument("--s3-bucket", type=str, default="ai2-benb", help="S3 bucket name")
    parser.add_argument("--s3-path", type=str, default="qc", help="S3 path to upload model to")
    opts = parser.parse_args()

    return opts


def main(args: argparse.Namespace):
    random.seed(args.seed)

    num_positive_per_source = args.max_num_positive // len(args.positive_sources)
    num_negative_per_source = args.max_num_negative // len(args.negative_sources)
    positive_configs = [DataConfig(path=path, label=POSITIVE_LABEL, sample=num_positive_per_source) for path in args.positive_sources]
    negative_configs = [DataConfig(path=path, label=NEGATIVE_LABEL, sample=num_negative_per_source) for path in args.negative_sources]

    dataset = ClassifierDataset(positive_configs + negative_configs, workers=args.num_workers)

    if len(dataset.documents) != args.max_num_positive + args.max_num_negative:
        raise ValueError(f"Expected {args.max_num_positive + args.max_num_negative} documents, got {len(dataset.documents)}")

    classifier = Classifier(base_model_name=args.model_name)
    classifier.fit(dataset, max_steps=args.max_steps)

    if args.upload_to_s3:
        run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        upload_path = os.path.join(args.s3_path, args.run_name, run_date)
        print(f"Uploading model to S3. source={args.local_save_path}, bucket={args.s3_bucket}, path={upload_path}")
        upload_directory_to_s3(args.local_save_path, args.s3_bucket, upload_path)


if __name__ == "__main__":
    args = parse_args()

    # add additional extension for smart_open
    register_compressor(".zstd", _handle_zstd)

    main(args)
