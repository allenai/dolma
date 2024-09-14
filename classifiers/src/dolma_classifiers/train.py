import argparse
import glob
import json
from typing import Dict, List

import evaluate
import jmespath
import smart_open
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an encoder model for classification")
    parser.add_argument("--train-data", type=str, required=True, help="Glob expression for training data")
    parser.add_argument("--test-data", type=str, help="Glob expression for test data")
    parser.add_argument("--dev-data", type=str, help="Glob expression for dev data")
    parser.add_argument("--input-field", type=str, required=True, help="JMESPath expression for input field")
    parser.add_argument("--label-field", type=str, required=True, help="JMESPath expression for label field")
    parser.add_argument("--metrics", nargs="+", default=["accuracy"], help="HuggingFace Evaluate metrics")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Model name or path")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--wandb-entity", type=str, help="W&B entity")
    parser.add_argument("--wandb-project", type=str, help="W&B project")
    parser.add_argument("--wandb-run-name", type=str, help="W&B run name")
    return parser.parse_args()

def load_jsonl(file_pattern: str, input_field: str, label_field: str) -> Dataset:
    data = []
    for file in glob.glob(file_pattern):
        with smart_open.open(file, "rt") as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    "text": jmespath.search(input_field, item),
                    "label": jmespath.search(label_field, item)
                })
    return Dataset.from_list(data)

def preprocess_function(examples: Dict[str, List], tokenizer) -> Dict[str, List]:
    return tokenizer(examples["text"], truncation=True, padding="max_length")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = logits.argmax(axis=-1)
    return {metric: evaluate.load(metric).compute(predictions=predictions, references=labels)
            for metric in args.metrics}

def main(args):
    # Load datasets
    train_dataset = load_jsonl(args.train_data, args.input_field, args.label_field)
    test_dataset = load_jsonl(args.test_data, args.input_field, args.label_field) if args.test_data else None
    dev_dataset = load_jsonl(args.dev_data, args.input_field, args.label_field) if args.dev_data else None

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(set(train_dataset["label"]))
    )

    # Preprocess datasets
    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    if test_dataset:
        test_dataset = test_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    if dev_dataset:
        dev_dataset = dev_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Initialize W&B
    if args.wandb_entity and args.wandb_project:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.wandb_run_name)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch" if test_dataset else "no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to="wandb" if args.wandb_entity and args.wandb_project else None,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    if test_dataset:
        trainer.evaluate()

    # Save the model
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
