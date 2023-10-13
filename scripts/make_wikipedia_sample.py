import argparse
import logging
from pathlib import Path
import re
import smart_open
import requests
import sys

try:
    import wikiextractor
except ImportError:
    print("Please install wikiextractor with `pip install wikiextractor`")
    sys.exit(1)

try





DUMP_URL = "https://dumps.wikimedia.org/simplewiki/{date}/{lang}wiki-{date}-pages-articles-multistream.xml.bz2"
LOGGER = logging.getLogger(__name__)



def download_file(url, filename):
    with open(filename, 'wb') as file:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=10485760):  # 10MB chunks
                file.write(chunk)



def download_wiki(date: str, lang: str, output: str):
    assert re.match(r"\d{8}", date), "Date must be in YYYYMMDD format"
    dump_url = DUMP_URL.format(date=date, lang=lang)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_path.suffix.endswith(".bz2"):
        LOGGER.warning("Output file does not end with .bz2. This is not recommended.")

    print(f"Downloading {dump_url} to {output_path}")
    with smart_open.open(dump_url, "rb") as f, smart_open.open(output_path, "wb") as g:
        g.write(f.read())
