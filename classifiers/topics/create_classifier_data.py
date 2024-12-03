import glob
import hashlib
import json
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

from llm_utils import cache, generate_response
from openai import OpenAI


client = OpenAI()


HF_MMLU = "cais/mmlu"
PROMPT_CONFIG = yaml.safe_load(open("prompts.yaml", "r"))

MODEL_ENGINE = "gpt-4o-2024-08-06"
BATCHES_VERSION = "batches_v1"

OUTPUT_DIR = "output"

LIMIT_KEYWORDS = None
LIMIT_TOPICS = None
LIMIT_DOCUMENTS = None

tokenizer = Tokenizer.from_file("llama_tokenizer.json")


@cache()
def get_mmlu_topics():
    response = requests.get(f"https://datasets-server.huggingface.co/splits?dataset={HF_MMLU}").json()
    return sorted(list(set(s["config"] for s in response["splits"]) - {"all", "auxiliary_train"}))


@cache()
def get_mmlu_questions(topic):
    return list(load_dataset(HF_MMLU, topic)["dev"])


def get_keywords_for_topic(topic, with_prompt=False, n_calls=4):
    questions = [f"{q['question']} ({q['choices'][q['answer']]})" for q in get_mmlu_questions(topic)]

    system_prompt = PROMPT_CONFIG["keywords_prompt"]["system"]
    instance_prompt_template = PROMPT_CONFIG["keywords_prompt"]["instance"]
    instance_prompt = jinja2.Template(instance_prompt_template).render(topic=get_topic_name(topic),
                                                                       questions=questions)

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance_prompt},
    ]

    suggested_keywords = []
    for i in range(n_calls):
        response = generate_response(MODEL_ENGINE, prompt, temperature=1.0, seed=i)
        suggested_keywords += [line.split(" ", 1)[1].strip('" ') for line in response.text.strip().split("\n") if line]

    suggested_keywords = sorted(set(suggested_keywords))
    if with_prompt:
        return suggested_keywords, prompt
    else:
        return suggested_keywords


@cache()
def call_infinigram_search(keyword, index, maxnum, max_disp_len):
    payload = {
        'index': index,
        'query_type': 'search_docs',
        'query': keyword,
        'maxnum': maxnum,
        'max_disp_len': max_disp_len
    }
    return requests.post('https://api.infini-gram.io/', json=payload).json()


def query_documents(keyword):
    result = call_infinigram_search(keyword, 'v4_dolma-v1_7_llama', 10, 1024)

    documents = []
    for document in result["documents"]:
        document_text = tokenizer.decode(document["token_ids"])
        documents.append({
            "document": document_text,
            "keyword": keyword,
        })

    return documents


def get_relevant_documents_for_topic(topic):
    keywords, prompt = get_keywords_for_topic(topic, with_prompt=True)
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
        documents = random.Random(0).sample(documents, LIMIT_DOCUMENTS)

    return {
        "keywords_prompt": prompt,
        "keywords": list(keywords),
        "documents": documents,
    }


def get_relevant_documents(topics):
    if LIMIT_TOPICS:
        topics = topics[:LIMIT_TOPICS]

    output = {}

    for topic in tqdm(topics, desc="Topics"):
        output[topic] = get_relevant_documents_for_topic(topic)

        os.makedirs(os.path.join(OUTPUT_DIR, "documents", topic), exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "documents", topic, "documents.json"), "w") as f:
            json.dump(output[topic], f, indent=2)

    return output


def get_topic_name(topic):
    words = topic.split("_")
    return " ".join([w.capitalize() for w in words])


def get_relevance_annotation_prompt(topic, document):
    questions = [f"{q['question']} ({q['choices'][q['answer']]})" for q in get_mmlu_questions(topic)]

    prompt_template = PROMPT_CONFIG["relevance_annotation_prompt"]
    prompt = jinja2.Template(prompt_template).render(
        topic=get_topic_name(topic),
        questions=questions,
        document=document["document"],
    )
    return [{"role": "user", "content": prompt}]


def annotate_document(topic, document):
    prompt = get_relevance_annotation_prompt(topic, document)
    response = generate_response(MODEL_ENGINE, prompt, temperature=0.2)
    annotation = int(re.findall(r"\d+", response.text.split("Educational score: ")[-1].strip())[0])

    return {
        "topic": topic,
        "document": document,
        "annotation": annotation,
        "explanation": response.text,
        "cost": response.cost,
    }


def get_relevance_annotations(relevant_documents_per_topic):
    output = []
    total_cost = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        future_to_document = {
            executor.submit(annotate_document, topic, document): (topic, document)
            for topic, topic_data in relevant_documents_per_topic.items()
            for document in topic_data["documents"]
        }

        tqdm_loop = tqdm(concurrent.futures.as_completed(future_to_document), total=len(future_to_document), desc="Annotating documents")
        for future in tqdm_loop:
            try:
                total_cost += future.result()["cost"]
                tqdm_loop.set_postfix(cost=total_cost)
                output.append(future.result())
            except Exception as e:
                topic, document = future_to_document[future]
                print(f"Error annotating document for topic {topic}: {e}")

    for topic in relevant_documents_per_topic:
        with open(os.path.join(OUTPUT_DIR, "documents", topic, "annotations.json"), "w") as f:
            topic_annotations = [a for a in output if a["topic"] == topic]
            json.dump(topic_annotations, f, indent=2)

    return output


def save_annotations_batches(relevant_documents_per_topic):
    os.makedirs(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "requests"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "documents"), exist_ok=True)

    for topic in relevant_documents_per_topic:
        with open(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "documents", f"{topic}.jsonl"), "w") as f:
            for document in relevant_documents_per_topic[topic]["documents"]:
                custom_id = hashlib.md5(json.dumps(document).encode()).hexdigest()

                f.write(json.dumps({
                    "custom_id": custom_id,
                    "text": document["document"],
                    "topic": topic,
                    "keyword": document["keyword"],
                }) + "\n")

        with open(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "requests", f"{topic}.jsonl"), "w") as f:
            for document in relevant_documents_per_topic[topic]["documents"]:
                custom_id = hashlib.md5(json.dumps(document).encode()).hexdigest()
                prompt = get_relevance_annotation_prompt(topic, document)
                f.write(json.dumps({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-2024-08-06",
                        "messages": prompt,
                        "max_tokens": 250,
                        "temperature": 0.2,
                    }
                }) + "\n")


def upload_batches(topics):
    for topic in topics:
        batch_input_file = client.files.create(
            file=Path(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "requests", f"{topic}.jsonl")),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"classifier-batch-{topic}"
            }
        )

        # keep batch file id for later use
        os.makedirs(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "ids"), exist_ok=True)
        path = os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "ids", f"{batch.id}.json")
        with open(path, "w") as f:
            batch_info = json.loads(batch.json())
            f.write(json.dumps(batch_info, indent=4))
        print(f"Created batch {batch.id} with input file {batch_input_file_id} at {path}")


def print_batches_status():
    files = glob.glob(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "ids", "*.json"))
    for file in files:
        batch_info = json.load(open(file, "r"))
        batch_id = batch_info["id"]

        status = client.batches.retrieve(batch_id)
        response = json.loads(status.json())
        print(json.dumps(response, indent=4))


def download_completed_batches():
    os.makedirs(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "responses"), exist_ok=True)

    files = glob.glob(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "ids", "*.json"))
    for file in files:
        batch_info = json.load(open(file, "r"))
        batch_id = batch_info["id"]

        status = client.batches.retrieve(batch_id)
        file_id = status.output_file_id

        file_response = client.files.content(file_id)
        save_path = os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "responses", f"{batch_id}.jsonl")
        with open(save_path, "w") as f:
            f.write(file_response.text)
            print(f"Saved response to {save_path}")


def collect_annotations(topics):
    # download_completed_batches()

    os.makedirs(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "annotations"), exist_ok=True)

    custom_id_to_document = {}
    files = glob.glob(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "documents", "*.jsonl"))
    for file in files:
        with open(file, "r") as f:
            for line in f:
                document = json.loads(line)
                custom_id_to_document[document["custom_id"]] = document

    annotations_per_topic = defaultdict(list)
    files = glob.glob(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "responses", "*.jsonl"))
    for file in files:
        with open(file, "r") as f:
            for line in f:
                response = json.loads(line)
                document = custom_id_to_document[response["custom_id"]]

                annotation = int(re.findall(r"\d+", response["response"]["body"]["choices"][0]["message"]["content"].split("Educational score: ")[-1].strip())[0])
                annotations_per_topic[document["topic"]].append({
                    "document": document,
                    "annotation": annotation,
                    "explanation": response["response"]["body"]["choices"][0]["message"]["content"],
                })

    for topic in topics:
        with open(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "annotations", f"{topic}.json"), "w") as f:
            json.dump(annotations_per_topic[topic], f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "annotated_all.jsonl"), "w") as f:
        for topic in topics:
            for annotation in annotations_per_topic[topic]:
                saved_annotation = {
                    "text": annotation["document"]["text"],
                    "score": annotation["annotation"],
                }
                f.write(json.dumps(saved_annotation) + "\n")

    # upload to s3
    s3 = boto3.client("s3")
    s3.upload_file(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "annotated_all.jsonl"), "ai2-benb", "qc-data/mmlu_annotations/annotated_all.jsonl")


def main():
    topics = get_mmlu_topics()

    # relevant_documents_per_topic = get_relevant_documents(topics)
    # save_annotations_batches(relevant_documents_per_topic)
    # upload_batches(topics)
    # print_batches_status()

    collect_annotations(topics)

    # # for generating annotations without batching:
    # annotations = get_relevance_annotations(relevant_documents_per_topic)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    main()
