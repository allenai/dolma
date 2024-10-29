#!/usr/bin/env python3

import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import grequests
import jinja2
import urllib3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OPENAI_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"


class DocumentProcessor:
    def __init__(
        self,
        documents_path: str,
        destination: str,
        prompt_template: str,
        api_key: str,
        batch_size: int = 5,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        self.documents_path = documents_path
        self.destination = destination
        self.prompt_template = prompt_template
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.template = jinja2.Template(prompt_template)

    def _create_request(self, document: Dict[str, Any]) -> grequests.AsyncRequest:
        """Create a single grequest for a document."""
        try:
            # Render the prompt template with document fields
            prompt = self.template.render(**document)

            # Prepare the request payload
            payload = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that processes documents."},
                    {"role": "user", "content": prompt}
                ]
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            # Create the request object
            return grequests.post(
                OPENAI_API_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=30
            ), document

        except Exception as e:
            logger.error(f"Error creating request: {e}")
            return None

    def _process_response(self, response, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single response from the API."""
        try:
            if response.status_code == 200:
                result = response.json()
                document['gpt4_response'] = result['choices'][0]['message']['content']
            else:
                document['error'] = f"API Error: {response.status_code} - {response.text}"
        except Exception as e:
            document['error'] = f"Processing Error: {str(e)}"

        return document

    def _process_batch(self, batch: List[Dict[str, Any]], output_file: str):
        """Process a batch of documents and write results to output file."""
        # Create request objects for the batch
        request_pairs = [self._create_request(doc) for doc in batch]
        requests, documents = zip(*[pair for pair in request_pairs if pair is not None])

        # Make async requests
        responses = grequests.map(requests, size=len(requests))

        # Process responses and write to file
        with open(output_file, 'a') as f:
            for response, document in zip(responses, documents):
                result = self._process_response(response, document)
                f.write(json.dumps(result) + '\n')

    def _download_file(self, url: str, local_path: str) -> str:
        """Download a remote file to local storage."""
        with urllib3.PoolManager() as http:
            response = http.request('GET', url)
            if response.status == 200:
                with open(local_path, 'w') as f:
                    f.write(response.data.decode('utf-8'))
                return local_path
            else:
                raise Exception(f"Failed to download file: {response.status}")

    def _get_file_paths(self) -> List[str]:
        """Get list of files to process, handling both local and remote paths."""
        if urlparse(self.documents_path).scheme in ('http', 'https'):
            # Handle remote files
            temp_dir = Path('temp_downloads')
            temp_dir.mkdir(exist_ok=True)

            # Download remote files
            local_paths = []
            with urllib3.PoolManager() as http:
                response = http.request('GET', self.documents_path)
                if response.status == 200:
                    file_list = response.data.decode('utf-8').splitlines()
                    for url in file_list:
                        local_path = temp_dir / Path(urlparse(url).path).name
                        self._download_file(url, str(local_path))
                        local_paths.append(str(local_path))
            return local_paths
        else:
            # Handle local files
            return glob.glob(self.documents_path)

    def process_files(self):
        """Main method to process all files."""
        # Create destination directory if it doesn't exist
        os.makedirs(self.destination, exist_ok=True)

        # Get list of files to process
        file_paths = self._get_file_paths()
        logger.info(f"Found {len(file_paths)} files to process")

        for file_path in file_paths:
            try:
                # Read input file
                with open(file_path, 'r') as f:
                    documents = [json.loads(line) for line in f]

                # Create output file path
                output_file = os.path.join(
                    self.destination,
                    f"processed_{os.path.basename(file_path)}"
                )

                # Process documents in batches
                for i in range(0, len(documents), self.batch_size):
                    batch = documents[i:i + self.batch_size]
                    self._process_batch(batch, output_file)
                    logger.info(f"Processed batch {i//self.batch_size + 1} of file {file_path}")

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Process documents with GPT-4')
    parser.add_argument('--documents', required=True, help='Glob pattern for input documents')
    parser.add_argument('--destination', required=True, help='Output directory')
    parser.add_argument('--prompt', required=True, help='Prompt template')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for processing')

    args = parser.parse_args()

    # Read prompt template from file if it's a file path
    prompt_template = args.prompt
    if os.path.isfile(args.prompt):
        with open(args.prompt, 'r') as f:
            prompt_template = f.read()

    processor = DocumentProcessor(
        documents_path=args.documents,
        destination=args.destination,
        prompt_template=prompt_template,
        api_key=args.api_key,
        batch_size=args.batch_size
    )

    # Run the processor
    processor.process_files()

if __name__ == "__main__":
    main()
