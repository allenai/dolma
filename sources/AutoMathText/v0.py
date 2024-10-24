
import os
import glob

from contextlib import ExitStack
from hashlib import md5
from tempfile import TemporaryDirectory
from typing import Any, Optional
import datetime
from queue import Queue
import json
from multiprocessing import cpu_count

import smart_open
from dolma.core.parallel import BaseParallelProcessor


def format_to_dolma_timestamp(timestamp: Optional[datetime.datetime] = None) -> str:
    """Format a timestamp as a string using near ISO-8601 format."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


def parse_date_web(date_str):
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")


def parse_date_arxiv(date_str):
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")


def parse_code_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        # If milliseconds are not present, try without them
        return datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")


class AutoWebMathProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(
        cls,
        queue: Queue,
        /,
        files: int = 0,
        docs: int = 0,
        words: int = 0,
    ):
        """
        This method is to update the progress bar. We keep
        track of three things:
        - files: the number of files processed
        - read_docs: the number of documents read in
        - written_docs: the number of documents written out
            (i.e., the number of documents that are not empty)
        """
        super().increment_progressbar(
            queue,
            files=files,
            docs=docs,
            words=words,
        )

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: Queue,
        **kwargs: Any,
    ):
        """
        This method is called for each file. It reads the file
        line by line, and writes to the destination file only
        if the document is not empty.
        """

        update_every_n_lines = 10_000
        docs = 0
        words = 0

        with ExitStack() as stack:
            # open source and destination files
            source_file = stack.enter_context(
                smart_open.open(source_path, "rt")
            )
            if destination_path.endswith(".jsonl"):
                destination_path += ".gz"

            destination_file = stack.enter_context(
                smart_open.open(destination_path, "wt")
            )

            # Set a fixed creation date
            created = datetime.datetime(2024, 1, 23)

            *_, source, subset, _ = source_path.split("/")
            for ln in source_file:
                # we first load the json document
                document = json.loads(ln)
                docs += 1
                docid = md5((ln + source + subset).encode('utf-8')).hexdigest()

                metadata = document.pop("meta")

                if "title" in document and "abstract" in document and "text" in document:
                    # arxiv subset
                    text = f"{document['title']}\n\n{document['abstract']}\n\n{document['text']}"
                    metadata["subset"] = subset
                    metadata["path"] = source_path
                    metadata["url"] = document.pop("url")
                    created = parse_date_arxiv(metadata["timestamp"])

                elif "url" in document and "date" in document:
                    created = parse_date_web(document["date"])
                    # this is web content
                    metadata["date"] = document["date"]
                    metadata["url"] = document["url"]
                    metadata["path"] = source_path
                    text = document["text"]
                elif "text" in document:
                    if metadata.get("max_stars_repo_stars_event_min_datetime", None) is not None:
                        created = min(
                            parse_code_date(metadata["max_stars_repo_stars_event_min_datetime"]),
                            created
                        )
                    if metadata.get("max_forks_repo_forks_event_min_datetime", None) is not None:
                        created = min(
                            parse_code_date(metadata["max_forks_repo_forks_event_min_datetime"]),
                            created
                        )
                    text = document["text"]
                    # this is a code document
                else:
                    raise ValueError(f"Unknown document type: {document}")

                output = {
                    "text": text.strip(),
                    "source": f"{source}_{subset}",
                    "added": format_to_dolma_timestamp(),
                    "created": format_to_dolma_timestamp(created),
                    "id": docid,
                    "metadata": metadata
                }

                words += len(text.split())

                # if the document is not empty,
                # we write it to output
                destination_file.write(json.dumps(output) + "\n")

                # we update the progress bar every
                # update_every_n_lines
                if docs > update_every_n_lines:
                    cls.increment_progressbar(queue, docs=docs, words=words)
                    docs = 0
                    words = 0

            # we update the progress bar one last time
            cls.increment_progressbar(
                queue,
                files=1,
                docs=docs,
                words=words,
            )


def main():

    base_source_prefix = '/data/math-ai_AutoMathText/raw/data'
    base_destination_prefix = '/data/math-ai_AutoMathText/v0/documents'


    jsonl_files = []
    for root, dirs, files in os.walk(base_source_prefix):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    print(f"Found {len(jsonl_files)} JSONL files.")


    with TemporaryDirectory() as tmpdir:

        # Create destination paths by combining end filepath after base_source_prefix with base_destination_prefix
        destinations = []
        temp_files = []
        for jsonl_file in jsonl_files:
            relative_path = os.path.relpath(jsonl_file, base_source_prefix)
            destination = os.path.join(base_destination_prefix, relative_path)
            destination_dir = os.path.dirname(destination)
            os.makedirs(destination_dir, exist_ok=True)
            destinations.append(os.path.dirname(destination))
            temp_file = os.path.join(tmpdir, os.path.dirname(destination))
            os.makedirs(temp_file, exist_ok=True)
            temp_files.append(temp_file)

        print(f"Created {len(destinations)} destination paths.")

        # create the processor
        processor = AutoWebMathProcessor(
            source_prefix=jsonl_files,
            destination_prefix=destinations,
            metadata_prefix=temp_files,
            num_processes=cpu_count() - 1,
            debug=False,
        )

        # run the processor
        processor()


if __name__ == "__main__":
    main()
