import argparse
import glob
import hashlib
import json

import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import boto3
import requests
import yaml
import jinja2
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import Tokenizer
import concurrent.futures

from llm_utils import cache, generate_response, LLMResponse
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed


client = OpenAI()


PROMPT_CONFIG = yaml.safe_load(open("prompts.yaml", "r"))

MODEL_ENGINE = "gpt-4o-2024-08-06"
CLASSIFIER_NAME = "v3"

OUTPUT_DIR = "output"
BATCHES_NAME = f"batches_{CLASSIFIER_NAME}"
OUTPUT_BUCKET = "ai2-benb"

LIMIT_KEYWORDS = None
LIMIT_TOPICS = None
LIMIT_DOCUMENTS = None
N_KEYWORDS_CALLS = 50  # 10
INFINIGRAM_DOCUMENTS = 40  # 20

tokenizer = Tokenizer.from_file("llama_tokenizer.json")

DATASETS = {
    "hellaswag": {
        "hf": "Rowan/hellaswag",
        "hf_contains_topics": False,
        "splits": ["validation"],
        "instance_to_question": lambda instance: f"Complete: '{instance['ctx']}...'. Answer: '{instance['endings'][int(instance['label'])]}'",
        "prompt_for_keyword_gen": "system_commonsense",
        "prompt_for_relevance": "commonsense",
        "topics_by_category": {
            "hellaswag": ["hellaswag"],
        },
        "max_questions_for_keywords": 200,
        "max_questions_for_relevance": {
            "hellaswag": 200,
        },
    },
    "mmlu": {
        "hf": "cais/mmlu",
        "hf_contains_topics": True,
        "splits": ["dev", "validation"],
        "instance_to_question": lambda instance: f"{instance['question']} ({instance['choices'][instance['answer']]})",
        "prompt_for_keyword_gen": "system_mmlu",
        "prompt_for_relevance": "mmlu",
        "topics_by_category": {
            'stem': [
                'astronomy',
                'college_physics',
                'conceptual_physics',
                'high_school_physics',
                'college_chemistry',
                'high_school_chemistry',
                'college_biology',
                'high_school_biology',
                'college_computer_science',
                'computer_security',
                'high_school_computer_science',
                'machine_learning',
                'abstract_algebra',
                'college_mathematics',
                'elementary_mathematics',
                'high_school_mathematics',
                'high_school_statistics',
                'electrical_engineering'
            ],
            'humanities': [
                'high_school_european_history',
                'high_school_us_history',
                'high_school_world_history',
                'prehistory',
                'formal_logic',
                'logical_fallacies',
                'moral_disputes',
                'moral_scenarios',
                'philosophy',
                'world_religions',
                'international_law',
                'jurisprudence',
                'professional_law'
            ],
            'social_sciences': [
                'high_school_government_and_politics',
                'public_relations',
                'security_studies',
                'us_foreign_policy',
                'human_sexuality',
                'sociology',
                'econometrics',
                'high_school_macroeconomics',
                'high_school_microeconomics',
                'high_school_geography',
                'high_school_psychology',
                'professional_psychology'
            ],
            'other': [
                'global_facts',
                'professional_accounting',
                'business_ethics',
                'management',
                'marketing',
                'anatomy',
                'clinical_knowledge',
                'college_medicine',
                'human_aging',
                'medical_genetics',
                'nutrition',
                'professional_medicine',
                'virology'
            ]
                # 'miscellaneous',
        },
        "max_questions_for_relevance": {
            "high_school_us_history": 7,
            "professional_law": 10,
            "high_school_european_history": 10,
            "high_school_world_history": 5,
            "moral_scenarios": 10
        },
    }
}

@cache()
def get_questions(dataset_name, topic, split="validation"):
    hf_name = DATASETS[dataset_name]["hf"]
    return list(load_dataset(hf_name, topic)[split])


def get_keywords_for_topic(dataset_name, topic, with_prompt=False, n_calls=6):
    if not DATASETS[dataset_name]["hf_contains_topics"]:
        topic = None

    questions = []
    for split in DATASETS[dataset_name]["splits"]:
        questions += [DATASETS[dataset_name]["instance_to_question"](q) for q in get_questions(dataset_name, topic, split=split)]

    if "max_questions_for_keywords" in DATASETS[dataset_name]:
        questions = random.Random(0).sample(questions, min(DATASETS[dataset_name]["max_questions_for_keywords"], len(questions)))
    system_prompt = PROMPT_CONFIG["keywords_prompt"][DATASETS[dataset_name]["prompt_for_keyword_gen"]]
    instance_prompt_template = PROMPT_CONFIG["keywords_prompt"]["instance"]
    instance_prompt = jinja2.Template(instance_prompt_template).render(topic=get_topic_name(topic),
                                                                       questions=questions)

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance_prompt},
    ]

    suggested_keywords = []
    def generate_keyword(seed):
        response = generate_response(MODEL_ENGINE, prompt, temperature=1.0, seed=seed)
        return [line.split(" ", 1)[1].strip('" ') for line in response.text.strip().split("\n") if line]

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_calls) as executor:
        futures = {executor.submit(generate_keyword, seed): seed for seed in range(n_calls)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=n_calls, desc="Generating keywords"):
            try:
                suggested_keywords += future.result()
            except Exception as e:
                print(f"Error generating keywords for seed {futures[future]}: {e}")

    suggested_keywords = sorted(set(suggested_keywords))
    if with_prompt:
        return suggested_keywords, prompt
    else:
        return suggested_keywords


@cache()
def call_infinigram_search(keyword, index, maxnum, max_disp_len, seed=None):
    payload = {
        'index': index,
        'query_type': 'search_docs',
        'query': keyword,
        'maxnum': maxnum,
        'max_disp_len': max_disp_len
    }
    return requests.post('https://api.infini-gram.io/', json=payload).json()


def query_documents(keyword, limit=INFINIGRAM_DOCUMENTS, api_max_return=10):
    documents = []

    for i in range(math.ceil(limit / api_max_return)):
        result = call_infinigram_search(keyword, 'v4_dolma-v1_7_llama', 10, 1024, seed=i)

        for document in result["documents"]:
            document_text = tokenizer.decode(document["token_ids"])
            documents.append({
                "document": document_text,
                "keyword": keyword,
            })
        if len(set(result["idxs"])) < api_max_return:
            break

    return documents


def get_relevant_documents_for_topic(dataset_name, topic):
    keywords, prompt = get_keywords_for_topic(dataset_name, topic, with_prompt=True, n_calls=N_KEYWORDS_CALLS)
    if LIMIT_KEYWORDS:
        keywords = keywords[:LIMIT_KEYWORDS]
    documents = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        future_to_keyword = {executor.submit(query_documents, keyword): keyword for keyword in keywords}
        for future in tqdm(concurrent.futures.as_completed(future_to_keyword), total=len(keywords), desc="Querying Infinigram"):
            try:
                documents += future.result()
            except Exception as e:
                print(f"Error querying documents for keyword {future_to_keyword[future]}: {e}")

    # remove duplicate documents dicts according only to document field
    unique_documents = {doc['document']: doc for doc in documents}.values()
    documents = list(unique_documents)

    if LIMIT_DOCUMENTS:
        documents = random.Random(0).sample(documents, min(LIMIT_DOCUMENTS, len(documents)))

    return {
        "keywords_prompt": prompt,
        "keywords": list(keywords),
        "documents": documents,
    }


def get_relevant_documents_per_topic(dataset_name):
    topics_by_category = DATASETS[dataset_name]["topics_by_category"]
    topics = [topic for category in topics_by_category for topic in topics_by_category[category]]

    if LIMIT_TOPICS:
        topics = topics[:LIMIT_TOPICS]

    output = {}

    for topic in tqdm(topics, desc="Topics"):
        output[topic] = get_relevant_documents_for_topic(dataset_name, topic)
        if topic is None:
            path = os.path.join(OUTPUT_DIR, "documents", dataset_name)
        else:
            path = os.path.join(OUTPUT_DIR, "documents", dataset_name, topic)

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "documents.json"), "w") as f:
            json.dump(output[topic], f, indent=2)

    return output


def get_topic_name(topic):
    if not topic:
        return topic
    words = topic.split("_")
    return " ".join([w.capitalize() for w in words])


def get_relevance_annotation_prompt(dataset_name, questions, document):
    questions_by_topic = defaultdict(list)
    for q in questions:
        questions_by_topic[get_topic_name(q['topic'])].append(DATASETS[dataset_name]["instance_to_question"](q))

    prompt_template = PROMPT_CONFIG["relevance_annotation_prompt"][DATASETS[dataset_name]["prompt_for_relevance"]]
    prompt = jinja2.Template(prompt_template).render(
        questions_by_topic=questions_by_topic,
        document=document["document"],
    )
    return [{"role": "user", "content": prompt}]


def save_annotations_batches(dataset_name, relevant_documents_per_topic):
    os.makedirs(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "requests"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "documents"), exist_ok=True)

    for category, topics in DATASETS[dataset_name]["topics_by_category"].items():
        category_questions = []
        category_documents = []
        for topic in topics:
            splits = DATASETS[dataset_name]["splits"]
            topic_questions = []
            for split in splits:
                topic_questions += get_questions(dataset_name, topic if DATASETS[dataset_name]["hf_contains_topics"] else None, split=split)

            limit_num_questions = DATASETS[dataset_name].get("max_questions_for_relevance", {}).get(topic)
            if limit_num_questions:
                topic_questions = random.Random(0).sample(topic_questions, min(limit_num_questions, len(topic_questions)))

            topic_documents = relevant_documents_per_topic.get(topic, {}).get("documents", [])
            if not topic_documents:
                print(f"No documents for topic {topic}!")
            for doc in topic_documents:
                category_documents.append({**doc, "topic": topic})
            for q in topic_questions:
                category_questions.append({**q, "topic": topic})

        questions_batch_size = 500
        random.Random(0).shuffle(category_questions)
        n_batches = max(len(category_questions) // questions_batch_size, 1)

        print(f"Creating {n_batches} batches for category {category}, with {len(category_questions)} questions and {len(category_documents)} documents")
        print(f"Total requests: {(n_batches * len(category_documents)):,}")

        for topic in topics:
            with open(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "documents", f"{topic}.jsonl"), "w") as f:
                for document in relevant_documents_per_topic.get(topic, {}).get("documents", []):
                    custom_id = hashlib.md5(document["document"].encode()).hexdigest()

                    f.write(json.dumps({
                        "custom_id": custom_id,
                        "text": document["document"],
                        "keyword": document["keyword"],
                        "topic": topic,
                    }) + "\n")

        for i in range(n_batches):
            sampled_questions = category_questions[i * questions_batch_size:(i + 1) * questions_batch_size]

            max_num_in_batch = 2000
            for j in range(math.ceil(len(category_documents) / max_num_in_batch)):
                category_documents_batch = category_documents[j * max_num_in_batch:(j + 1) * max_num_in_batch]
                batch_code = f"{category}_{i}_{j}"

                with open(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "requests", f"{batch_code}.jsonl"), "w") as f:
                    for document in category_documents_batch:
                        custom_id = hashlib.md5(document["document"].encode()).hexdigest()
                        prompt = get_relevance_annotation_prompt(dataset_name, sampled_questions, document)
                        f.write(json.dumps({
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": "gpt-4o-mini",
                                "messages": prompt,
                                "max_tokens": 400,
                                "temperature": 0.2,
                            }
                        }) + "\n")
                print(f"Saved batch {batch_code} with {len(category_documents_batch)} documents and {len(sampled_questions)} questions")


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def create_client_file(client, file_path, purpose):
    return client.files.create(file=Path(file_path), purpose=purpose)

def upload_batches(dataset_name):
    print(f"********** Uploading batches for {dataset_name} **********")
    for category in DATASETS[dataset_name]["topics_by_category"]:
        category_files = glob.glob(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "requests", f"{category}_*.jsonl"))
        category_files.sort()
        for file in tqdm(category_files):
            filename = os.path.basename(file)
            batch_input_file = create_client_file(client, os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "requests", filename), "batch")

            print(f"Uploaded batch input file {batch_input_file.id} ({filename})")

            batch_input_file_id = batch_input_file.id

            batch = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": f"classifier-batch-{filename}"
                }
            )

            # keep batch file id for later use
            os.makedirs(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "ids"), exist_ok=True)
            path = os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "ids", f"{batch.id}.json")
            with open(path, "w") as f:
                batch_info = json.loads(batch.json())
                f.write(json.dumps(batch_info, indent=4))
            print(f"Created batch {batch.id} with input file {batch_input_file_id} at {path}")


def print_batches_status():
    files = glob.glob(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "ids", "*.json"))
    for file in files:
        batch_info = json.load(open(file, "r"))
        batch_id = batch_info["id"]

        status = client.batches.retrieve(batch_id)
        response = json.loads(status.json())
        print(json.dumps(response, indent=4))


def download_completed_batches():
    os.makedirs(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "responses"), exist_ok=True)

    files = glob.glob(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "ids", "*.json"))
    for file in tqdm(files):
        batch_info = json.load(open(file, "r"))
        batch_id = batch_info["id"]

        status = client.batches.retrieve(batch_id)
        file_id = status.output_file_id

        if not file_id:
            print(f"Batch {batch_id} is not completed yet")
            continue

        file_response = client.files.content(file_id)
        save_path = os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "responses", f"{batch_id}.jsonl")
        with open(save_path, "w") as f:
            f.write(file_response.text)
            print(f"Saved response to {save_path}")


def collect_annotations():
    # download_completed_batches()

    os.makedirs(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "annotations"), exist_ok=True)

    custom_id_to_document = get_documents_by_custom_id()

    annotations_per_document_id = defaultdict(list)
    annotations_per_topic = defaultdict(list)
    files = glob.glob(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "responses", "*.jsonl"))
    for file in tqdm(files):
        with open(file, "r") as f:
            for line in f:
                response = json.loads(line)
                if not response["custom_id"]:
                    print(f"Response {response} has no custom_id")
                    continue
                document = custom_id_to_document[response["custom_id"]]

                annotation = annotation_object_from_response(document, response)
                annotations_per_topic[document["topic"]].append(annotation)
                annotations_per_document_id[document["custom_id"]].append(annotation)

    topics = {doc["topic"] for doc in custom_id_to_document.values()}
    for topic in topics:
        with open(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "annotations", f"{topic}.json"), "w") as f:
            json.dump(annotations_per_topic[topic], f, indent=2)

    s3 = boto3.client("s3")
    for dataset_name in DATASETS:
        for category, topics in DATASETS[dataset_name]["topics_by_category"].items():
            filename = f"{category}_annotated_all.jsonl"
            with open(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, filename), "w") as f:
                for topic in tqdm(topics):
                    for annotation in annotations_per_topic[topic]:
                        saved_annotation = {
                            "text": annotation["document"]["text"],
                            "score": annotation["annotation"],
                        }
                        f.write(json.dumps(saved_annotation) + "\n")

            # upload to s3

            upload_to_path = os.path.join("qc-data", CLASSIFIER_NAME, filename)
            s3.upload_file(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, filename), OUTPUT_BUCKET, upload_to_path)
            print(f"Uploaded {filename} to s3://{OUTPUT_BUCKET}/{upload_to_path}")


def annotation_object_from_response(document, response, scoring_phase="relevance score"):
    if type(response) is LLMResponse:
        response_txt = response.text
    else:
        response_txt = response["response"]["body"]["choices"][0]["message"]["content"]
    annotation = None
    if scoring_phase.lower() in response_txt.lower():
        try:
            annotation = int(re.findall(r"\d+", response_txt.lower().split(scoring_phase.lower())[-1].strip())[0])
        except:
            pass
    return {
        "document": document,
        "annotation": annotation,
        "explanation": response_txt,
    }


def get_documents_by_custom_id():
    custom_id_to_document = {}
    files = glob.glob(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "documents", "*.jsonl"))
    for file in files:
        with open(file, "r") as f:
            for line in f:
                document = json.loads(line)
                custom_id_to_document[document["custom_id"]] = document
    return custom_id_to_document


def cancel_batches():
    files = glob.glob(os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "ids", "*.json"))
    for file in files:
        batch_info = json.load(open(file, "r"))
        batch_id = batch_info["id"]

        status = client.batches.retrieve(batch_id)
        if status.status == "in_progress":
            client.batches.cancel(batch_id)
            print(f"Cancelled batch {batch_id}")


def run_on_sample(dataset_name, n_sample=25):
    def process_line(line):
        request_data = json.loads(line)
        response = generate_response(
            request_data["body"]["model"],
            request_data["body"]["messages"],
            max_output_tokens=request_data["body"]["max_tokens"],
            temperature=request_data["body"]["temperature"],
            top_p=request_data["body"].get("top_p")
        )
        annotation = annotation_object_from_response(custom_id_to_document[request_data["custom_id"]], response)
        return json.dumps(annotation, indent=2) + "\n"

    sample_annotations_dir = os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "sample_annotations")
    os.makedirs(sample_annotations_dir, exist_ok=True)

    custom_id_to_document = get_documents_by_custom_id()

    for category, topics in DATASETS[dataset_name]["topics_by_category"].items():
        requests_file = os.path.join(OUTPUT_DIR, "batches", BATCHES_NAME, "requests", f"{category}_*.jsonl")
        request_files = glob.glob(requests_file)
        requests = []
        for file in request_files:
            with open(file, "r") as f:
                lines = f.readlines()
                requests += lines

        sampled_lines = random.Random(0).sample(requests, min(n_sample, len(requests)))
        with open(os.path.join(sample_annotations_dir, f"{category}.jsonl"), "w") as f_out:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(tqdm(executor.map(process_line, sampled_lines), total=len(sampled_lines), desc=f"Processing {category}"))

            for result in results:
                f_out.write(result)

def main():
    parser = argparse.ArgumentParser(description="Manage batch operations")

    subparsers = parser.add_subparsers(dest="command")

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--dataset", nargs="+", choices=list(DATASETS.keys()), default=list(DATASETS.keys()))

    # Subparser for upload_batches
    upload_parser = subparsers.add_parser("upload_batches", parents=[parent_parser], help="Save and upload batches")
    upload_parser.add_argument("--dry-run", action="store_true", help="Only save batches without uploading")
    upload_parser.add_argument("--run-on-sample", action=argparse.BooleanOptionalAction, default=False)
    upload_parser.add_argument("--save-annotations", action=argparse.BooleanOptionalAction, default=True)

    subparsers.add_parser("batches_status", parents=[parent_parser], help="Print the status of batches")
    subparsers.add_parser("collect_annotations", parents=[parent_parser], help="Collect annotations")
    subparsers.add_parser("cancel_batches", parents=[parent_parser], help="Cancel all active batches")

    args = parser.parse_args()

    if args.command == "upload_batches":
        for dataset_name in args.dataset:
            if args.save_annotations:
                relevant_documents_per_topic = get_relevant_documents_per_topic(dataset_name)
                save_annotations_batches(dataset_name, relevant_documents_per_topic)
            if args.run_on_sample:
                run_on_sample(dataset_name)
            if not args.dry_run:
                upload_batches(dataset_name)
    elif args.command == "batches_status":
        print_batches_status()
    elif args.command == "collect_annotations":
        collect_annotations()
    elif args.command == "cancel_batches":
        cancel_batches()
    else:
        parser.print_help()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    main()
