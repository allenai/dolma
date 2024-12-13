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

from llm_utils import cache, generate_response
from openai import OpenAI

client = OpenAI()


HF_MMLU = "cais/mmlu"
PROMPT_CONFIG = yaml.safe_load(open("prompts.yaml", "r"))

MODEL_ENGINE = "gpt-4o-2024-08-06"
BATCHES_VERSION = "batches_v2"

OUTPUT_DIR = "output"

LIMIT_KEYWORDS = None
LIMIT_TOPICS = None
LIMIT_DOCUMENTS = None
N_KEYWORDS_CALLS = 10
INFINIGRAM_DOCUMENTS = 20

tokenizer = Tokenizer.from_file("llama_tokenizer.json")

mmlu_topics_by_category = {
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
        'miscellaneous',
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
}

@cache()
def get_mmlu_questions(topic, split="validation"):
    return list(load_dataset(HF_MMLU, topic)[split])


def get_keywords_for_topic(topic, with_prompt=False, n_calls=6):
    questions = []
    for split in ["dev", "validation"]:
        questions += [f"{q['question']} ({q['choices'][q['answer']]})" for q in get_mmlu_questions(topic, split=split)]

    system_prompt = PROMPT_CONFIG["keywords_prompt"]["system"]
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
        futures = {executor.submit(generate_keyword, i): i for i in range(n_calls)}
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


def get_relevant_documents_for_topic(topic):
    keywords, prompt = get_keywords_for_topic(topic, with_prompt=True, n_calls=N_KEYWORDS_CALLS)
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


def get_relevance_annotation_prompt(questions, document):
    questions_by_topic = defaultdict(list)
    for q in questions:
        questions_by_topic[get_topic_name(q['topic'])].append(f"{q['question']} (Answer: {q['choices'][q['answer']]})")

    prompt_template = PROMPT_CONFIG["relevance_annotation_prompt"]
    prompt = jinja2.Template(prompt_template).render(
        questions_by_topic=questions_by_topic,
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

    for category, topics in mmlu_topics_by_category.items():
        category_questions = []
        category_documents = []
        for topic in topics:
            topic_questions = get_mmlu_questions(topic, split="validation")
            topic_questions += get_mmlu_questions(topic, split="dev")
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
            with open(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "documents", f"{topic}.jsonl"), "w") as f:
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

                with open(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "requests", f"{batch_code}.jsonl"), "w") as f:
                    for document in category_documents_batch:
                        custom_id = hashlib.md5(document["document"].encode()).hexdigest()
                        prompt = get_relevance_annotation_prompt(sampled_questions, document)
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


def upload_batches():
    for category in mmlu_topics_by_category:
        category_files = glob.glob(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "requests", f"{category}_*.jsonl"))
        for file in tqdm(category_files):
            filename = os.path.basename(file)
            batch_input_file = client.files.create(
                file=Path(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "requests", filename)),
                purpose="batch"
            )
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

        if not file_id:
            print(f"Batch {batch_id} is not completed yet")
            continue

        file_response = client.files.content(file_id)
        save_path = os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "responses", f"{batch_id}.jsonl")
        with open(save_path, "w") as f:
            f.write(file_response.text)
            print(f"Saved response to {save_path}")


def collect_annotations(topics):
    download_completed_batches()

    os.makedirs(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "annotations"), exist_ok=True)

    custom_id_to_document = {}
    files = glob.glob(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "documents", "*.jsonl"))
    for file in files:
        with open(file, "r") as f:
            for line in f:
                document = json.loads(line)
                custom_id_to_document[document["custom_id"]] = document

    annotations_per_document_id = defaultdict(list)
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
                annotations_per_document_id[document["custom_id"]].append(annotation)

    for topic in topics:
        with open(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, "annotations", f"{topic}.json"), "w") as f:
            json.dump(annotations_per_topic[topic], f, indent=2)

    s3 = boto3.client("s3")
    for category, topics in mmlu_topics_by_category.items():
        filename = f"{category}_annotated_all.jsonl"
        with open(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, filename), "w") as f:
            for topic in topics:
                for annotation in annotations_per_topic[topic]:
                    saved_annotation = {
                        "text": annotation["document"]["text"],
                        "scores": annotations_per_document_id[annotation["document"]["custom_id"]],
                        "score": max(annotations_per_document_id[annotation["document"]["custom_id"]]),
                    }
                    f.write(json.dumps(saved_annotation) + "\n")

        # # upload to s3
        # s3.upload_file(os.path.join(OUTPUT_DIR, "batches", BATCHES_VERSION, filename), "ai2-benb", f"qc-data/mmlu_annotations/{filename}")


def main():
    topics = [t for category_topics in mmlu_topics_by_category.values() for t in category_topics]

    relevant_documents_per_topic = get_relevant_documents(topics)
    save_annotations_batches(relevant_documents_per_topic)
    upload_batches()
    # print_batches_status()
    #
    # collect_annotations(topics)

    # # for generating annotations without batching:
    # annotations = get_relevance_annotations(relevant_documents_per_topic)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    main()
