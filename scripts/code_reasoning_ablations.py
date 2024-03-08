"""
Evaluating reasoning abilities of model ablations with varying amount of code in pre-training
"""
import abc
import argparse
import os
import random
from abc import ABC

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_dataset
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, get_scheduler

models_checkpoints = {
    "c4-stack-5p": "../checkpoints/c4-stack-5p/step60000-unsharded/",
    "c4-stack-15p": "../checkpoints/c4-stack-15p/step60000-unsharded/",
    "GPT-Neox-20B-0p": "../checkpoints/GPT-Neox-20B-0p/step60000-unsharded/",
}


def get_args():
    args = argparse.ArgumentParser()

    args.add_argument("dataset", type=str, choices=["babi", "web_nlg", "gsm8k"])
    args.add_argument("--models", nargs="+", default=None)
    args.add_argument("--n-test-samples", type=int, default=None)
    args.add_argument("--n-train-samples", type=int, default=None)
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--max-few-shots", type=int, default=5)
    args.add_argument("--number-seeds", type=int, default=5)

    # only used for GSM8K train/eval
    args.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size")
    args.add_argument("--test_per_device_train_batch_size", type=int, default=16, help="Test Batch size")
    args.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    args.add_argument(
        "--n-solutions-sampled", default=20, type=int, help="Number of solutions to sample per example"
    )
    args.add_argument("--num-train-epochs", type=int, default=3, help="Number of training epochs")
    args.add_argument(
        "--warmup_ratio", type=float, default=0.03, help="Ratio of total training steps used for warmup."
    )

    return args.parse_args()


class Eval(ABC):
    @abc.abstractmethod
    def build_prompt(self, test_example, demonstrations):
        pass

    @abc.abstractmethod
    def get_num_new_tokens(self, test_example, tokenizer):
        pass

    @abc.abstractmethod
    def compute_metrics(self, prediction, test_example):
        pass


class BabiEval(Eval):
    def build_prompt(self, test_example, demonstrations):
        prompt = ""
        for demonstration in demonstrations:
            prompt += demonstration["passage"] + demonstration["question"] + " " + demonstration["answer"] + "\n\n"

        prompt += test_example["passage"] + test_example["question"]

        return prompt

    def get_num_new_tokens(self, test_example, tokenizer):
        # generate a bit more tokens than gold (otherwise we might think
        # prediction is correct even though the actual prediction is longer, thus
        # wrong) - this can be replaced with a stop token
        return len(tokenizer.encode(test_example["answer"], add_special_tokens=False)) + 5

    def compute_metrics(self, prediction, test_example):
        return {"accuracy": int(prediction == test_example["answer"])}


class WebNLGEval(Eval):
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)

    def build_prompt(self, test_example, demonstrations):
        prompt = """I will verbalize an abstract representation of a sentence in natural language. To do so, I will first show the representation and then the natural language. The text needs to include all of the information in the representation.\n\n"""
        for demonstration in demonstrations:
            prompt += ", ".join(demonstration["input"]) + "\n"
            prompt += f"{demonstration['target']}\n\n"

        prompt += ", ".join(test_example["input"]) + "\n"

        return prompt

    def get_num_new_tokens(self, test_example, tokenizer):
        return 200  # this is enough for the test set

    def compute_metrics(self, prediction, test_example):
        return {"rouge2_f1": self.scorer.score(test_example["target"], prediction)["rouge2"].fmeasure}


class GSM8KEval(Eval):
    def build_prompt(self, test_example, demonstrations):
        return ""

    def get_num_new_tokens(self, test_example, tokenizer):
        return 300  # this is enough for the test set

    def compute_metrics(self, prediction, test_example):
        def run_program(code):
            """Important: executing code outside a secure docker container is potentially dangerous"""
            if "import" in code:
                raise ValueError("Import is not allowed")
            if "open(" in code:
                raise ValueError("Open is not allowed")
            if "eval(" in code:
                raise ValueError("Eval is not allowed")
            if "exec(" in code:
                raise ValueError("Exec is not allowed")
            if "compile(" in code:
                raise ValueError("Compile is not allowed")

            try:
                exec(code)
                answer = locals()["solution"]()
                return {
                    "success": True,
                    "answer": answer,
                    "error": None,
                }
            except Exception as e:
                return {
                    "success": False,
                    "answer": None,
                    "error": str(e),
                }

        gold_answer = test_example["answer"].split("####")[1].strip()

        try:
            predicted_answer = run_program(prediction)["answer"]
            return {"accuracy": int(float(predicted_answer) == float(gold_answer))}
        except Exception:
            return {"accuracy": 0}


def complete_prompt(model, tokenizer, prompt: str, new_tokens=100):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_length = len(input_ids) + new_tokens

    input_ids = torch.Tensor([input_ids]).long().to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)
    entire_output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)[0]
    output_new_tokens = entire_output[len(input_ids[0]) :]
    decoded = tokenizer.decode(output_new_tokens, skip_special_tokens=True).strip()

    return decoded


def eval_only(evaluator, model, model_name, tokenizer, train_dataset, test_dataset, seed, args):
    """
    Evaluate the model on the test set by averaging over multiple number of few-shot examples and seeds, similar setup to https://arxiv.org/abs/2305.16264
    """
    train_dataset = list(train_dataset)
    test_dataset = list(test_dataset)

    results = []
    agg_results = []

    for few_shot in range(args.max_few_shots + 1):
        tqdm_loop = tqdm(test_dataset)

        all_metrics = []

        for ex in tqdm_loop:
            # always pick the same demonstrations for a given seed
            random.seed(seed)

            demonstrations = random.sample(train_dataset, few_shot)

            try:
                context = evaluator.build_prompt(ex, demonstrations)
                n_new_tokens = evaluator.get_num_new_tokens(ex, tokenizer)

                prediction = complete_prompt(
                    model=model, tokenizer=tokenizer, prompt=context, new_tokens=n_new_tokens
                )
                prediction = prediction.split("\n")[0].strip()

                ex_metrics = evaluator.compute_metrics(prediction, ex)

                all_metrics.append(ex_metrics)
                avg_metrics = {key: np.mean([ex[key] for ex in all_metrics]) for key in all_metrics[0].keys()}

                results.append(
                    {
                        "model": model_name,
                        "few_shot": few_shot,
                        "seed": seed,
                        "context": context,
                        "prediction": prediction,
                        "answer": ex.get("answer") or ex.get("target"),
                        **ex_metrics,
                    }
                )
                tqdm_loop.set_description(
                    f"model: {model_name}, few_shots: {few_shot}, seed: {seed}, "
                    + ", ".join([f"{key}: {value:.3f}" for key, value in avg_metrics.items()])
                )
            except Exception as e:
                print(e)
                print(f"Skipping...")
                continue

        avg_metrics = {key: np.mean([ex[key] for ex in all_metrics]) for key in all_metrics[0].keys()}
        agg_results.append({"model": model_name, "seed": seed, "few_shot": few_shot, **avg_metrics})

    return results, agg_results


def train_and_eval_gsm8k(evaluator, model, model_name, tokenizer, test_dataset, seed, args):
    def tokenize_function_train(example):
        output_ids = tokenizer.encode(example["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids = tokenizer.encode(example["question"], add_special_tokens=False) + output_ids
        labels = [-100] * len(tokenizer.encode(example["question"], add_special_tokens=False)) + output_ids
        return {"input_ids": input_ids, "labels": labels}

    def tokenize_function_eval(example):
        input_ids = tokenizer.encode(example["question"], add_special_tokens=False)
        labels = tokenizer.encode(example["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
        return {"input_ids": input_ids, "labels": labels}

    def evaluate(model, tokenizer, dataset, dataset_loader):
        eval_result = []
        eval_loop = tqdm(dataset_loader, desc="Evaluating")
        accuracies = []
        examples = list(dataset)
        i = 0
        for eval_batch in eval_loop:
            max_length = eval_batch["input_ids"].size(1) + evaluator.get_num_new_tokens(None, tokenizer)
            entire_output = model.generate(
                eval_batch["input_ids"],
                attention_mask=eval_batch["attention_mask"],
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.6,
                num_return_sequences=args.n_solutions_sampled,
            )
            output_new_tokens = entire_output[:, eval_batch["input_ids"].size(1) :]
            decoded = tokenizer.batch_decode(output_new_tokens, skip_special_tokens=True)

            for j, ex in enumerate(examples[i : i + args.test_per_device_train_batch_size]):
                predictions = []
                predictions_accuracies = []
                for prediction in decoded[j * args.n_solutions_sampled : (j + 1) * args.n_solutions_sampled]:
                    prediction = prediction.strip()
                    predictions.append(prediction)

                    accuracy = evaluator.compute_metrics(prediction, ex)["accuracy"]

                    predictions_accuracies.append(accuracy)

                eval_result.append(
                    {
                        "model": model_name,
                        "input": ex["question"],
                        "gold": ex["answer"].split("####")[1].strip(),
                        "prediction_0": predictions[0],
                        "pass@k": any(predictions_accuracies),
                        "pass_rate": np.mean(predictions_accuracies),
                    }
                )
                accuracies.append(any(predictions_accuracies))
            eval_loop.set_description(f"Evaluating, accuracy: {np.mean(accuracies):.3f}")
            i += args.test_per_device_train_batch_size
        return eval_result

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset = load_dataset("json", data_files={"train": "code_reasoning_ablations_gsm8k_code.jsonl"})[
        "train"
    ].map(lambda ex: {"answer": ex["python"]})
    train_dataset = train_dataset.shuffle(seed=seed).select(range(args.n_train_samples))

    accelerator = Accelerator()

    tokenized_datasets = {
        "train": train_dataset.map(
            tokenize_function_train,
            batched=False,
            remove_columns=[
                name
                for name in train_dataset.column_names
                if name not in ["input_ids", "labels", "attention_mask"]
            ],
        ),
        "test": test_dataset.map(
            tokenize_function_eval,
            batched=False,
            remove_columns=[
                name for name in test_dataset.column_names if name not in ["input_ids", "labels", "attention_mask"]
            ],
        ),
    }

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size,
    )

    test_dataloader = DataLoader(
        tokenized_datasets["test"],
        shuffle=False,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.test_per_device_train_batch_size,
    )

    optimizer = torch.optim.AdamW([p for n, p in model.named_parameters()], lr=args.learning_rate)
    num_training_steps_for_scheduler = len(train_dataloader) * args.num_train_epochs

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    acc_per_epoch = []
    results_per_epoch = []

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0

        loop = tqdm(train_dataloader)
        for step, batch in enumerate(loop):
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss

                total_loss += loss.detach().float()
                accelerator.backward(loss)

                loop.set_description(f"Epoch {epoch}, loss={total_loss / (step + 1):.3f}")

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

        model.eval()

        eval_results = evaluate(model, tokenizer, test_dataset, test_dataloader)
        acc = pd.DataFrame(eval_results)["pass@k"].mean()
        print(f"accuracy epoch {epoch}: {acc}")

        acc_per_epoch.append(acc)
        results_per_epoch.append(eval_results)

    best_acc = max(acc_per_epoch)
    best_epoch = acc_per_epoch.index(best_acc)

    agg_results = [{"model": model_name, "seed": seed, "pass@k": max(acc_per_epoch)}]
    return results_per_epoch[best_epoch], agg_results


def main():
    args = get_args()

    random.seed(args.seed)

    hf_dataset = {"babi": ("Muennighoff/babi",), "web_nlg": ("GEM/web_nlg", "en"), "gsm8k": ("gsm8k", "main")}[
        args.dataset
    ]

    print("Loading dataset...")
    train_dataset = load_dataset(*hf_dataset)["train"]
    test_dataset = load_dataset(*hf_dataset)["test"]
    print("Dataset loaded.")

    if args.n_test_samples is not None:
        sampled_indices = random.sample(range(len(test_dataset)), args.n_test_samples)
        test_dataset = test_dataset.select(sampled_indices)

    if args.n_train_samples is not None:
        sampled_indices = random.sample(range(len(train_dataset)), args.n_train_samples)
        train_dataset = train_dataset.select(sampled_indices)

    evaluator = {"babi": BabiEval, "web_nlg": WebNLGEval, "gsm8k": GSM8KEval}[args.dataset]()

    checkpoints_of_evaluated_models = (
        models_checkpoints
        if args.models is None
        else {
            model_name: checkpoint
            for model_name, checkpoint in models_checkpoints.items()
            if model_name in args.models
        }
    )

    results, agg_results = [], []
    for model_name, checkpoint in checkpoints_of_evaluated_models.items():
        print(f"*** Loading {model_name} ***")
        model = OLMoForCausalLM.from_pretrained(checkpoint)
        tokenizer = OLMoTokenizerFast.from_pretrained(checkpoint)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        print("Model loaded.")

        for seed in range(args.number_seeds):
            if args.dataset in ["babi", "web_nlg"]:
                seed_results, seed_agg_results = eval_only(
                    evaluator, model, model_name, tokenizer, train_dataset, test_dataset, seed, args
                )
            else:
                seed_results, seed_agg_results = train_and_eval_gsm8k(
                    evaluator, model, model_name, tokenizer, test_dataset, seed, args
                )
            results += seed_results
            agg_results += seed_agg_results

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    dir_path = os.path.join("results", args.dataset)
    os.makedirs(dir_path, exist_ok=True)

    df = pd.DataFrame(results)
    filepath = os.path.join(dir_path, f"results_{timestamp}.csv")
    print(f"Saving results to {filepath}")
    df.to_csv(filepath, index=False)

    df_agg = pd.DataFrame(agg_results)
    filepath = os.path.join(dir_path, f"results_agg_{timestamp}.csv")
    print(f"Saving aggregated results to {filepath}")
    df_agg.to_csv(filepath, index=False)

    # final results
    metric_keys = [key for key in df_agg.keys() if key not in ["model", "seed", "few_shot"]]
    print(df.groupby(["model"]).mean(numeric_only=True)[metric_keys])


if __name__ == "__main__":
    main()
