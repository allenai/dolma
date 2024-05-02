import shutil
import traceback
from fastwarc.warc import ArchiveIterator
import argparse
from tqdm import tqdm
from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding, bytes_to_str
import fsspec
from time import sleep
import random
from io import BytesIO
from resiliparse.process_guard import time_guard, ExecutionTimeout
import json
import os
import re
import fasttext
import kenlm

# Temporary debugging imports
import uuid
from collections import defaultdict

from text_extract.extract import extract_text
from text_extract.utils import Config
from text_extract.latex_processing import get_math_config
from text_normalizer import normalize

# Load the fasttext model
if os.path.isdir('../models'):
    MODEL_PATH = '../models/math_score.bin'
else:
    MODEL_PATH = 'math_score.bin'

def score_text(text):
    normalized_text = normalize(text).replace('\n', ' ')
    # Remove any [EQUATION] tokens
    normalized_text = normalized_text.replace('[EQUATION]', '')
    pred = score_model.predict(normalized_text, k=2)
    if pred[0][0] == '__label__positive':
        prob = pred[1][0]
    else:
        prob = pred[1][1]

    return prob

# Load the kenlm model
LM_PATH = '../models/lm-v2.binary'

lm = kenlm.Model(LM_PATH)

def document_perplexity(text):
    text = normalize(text)
    score = lm.score(text)
    return 10 ** (-score / len(text.split()))

# Load the model
score_model = fasttext.load_model(MODEL_PATH)

def is_english(text):
    normalized_text = normalize(text).replace('\n', ' ')
    pred = lid_model.predict(normalized_text, k=1)
    if pred[0][0] == "__label__en" and pred[1][0] >= 0.5:
        return True
    return False

MODEL_BIN = '../models/lid.176.bin'
lid_model = fasttext.load_model(MODEL_BIN)

randomized_config = Config('configs/randomized_all.yaml')

MATH_KEYWORDS = [
    'MathJax',
    'mathjax',
    '<math',
    'math-container',
    'katex.min.css',
    'latex.php',
    'codecogs',
    'tex.cgi',
    'class="tex"',
    "class='tex'",
]
latex_math_commands = [
    "\\end", "\\begin", "\\ref", "\\frac", "\\label", "\\bf", "\\right", "\\left",
    "\\rm", "\\alpha", "\\mu", "\\def", "\\it", "\\pi", "\\sigma", "\\sum", "\\lambda",
    "\\beta", "\\nu", "\\partial", "\\int", "\\delta", "\\rho", "\\phi", "\\gamma",
    "\\omega", "\\over", "\\nonumber", "\\bar", "\\sqrt", "\\theta", "\\tau", "\\em",
    "\\rangle", "\\hat", "\\tilde", "\\cal", "\\hline", "\\item", "\\psi", "\\vec",
    "\\langle", "\\epsilon", "\\eta", "\\cdot", "\\in", "\\xi", "\\infty", "\\quad",
    "\\mathcal", "\\times", "\\emph", "\\mathbf", "\\prime", "\\be", "\\mathrm", "\\ee",
    "\\vspace", "\\pm", "\\chi", "\\ell", "\\text", "\\qquad", "\\noindent", "\\to",
    "\\varphi", "\\hspace", "\\leq", "\\cos", "\\eqref", "\\overline", "\\sin", "\\kappa",
    "\\hbox", "\\rightarrow", "\\varepsilon", "\\textit", "\\dagger", "\\big", "\\otimes",
    "\\equiv", "\\zeta", "\\dot", "\\ln"
]

latex_regex = re.compile('\\\\[a-z]{2,}')
original_regex = re.compile('|'.join(MATH_KEYWORDS))

def decode_html(html):
    """Decodes the html if possible.
    First try to decode with utf-8, then try to detect the encoding."""
    try:
        html = bytes_to_str(html, 'utf-8')
    except Exception as e:
        encoding = detect_encoding(html)
        if encoding is None or encoding == 'utf-8':
            return
        try:
            html = bytes_to_str(html, encoding)
        except Exception as e:
            return
    return html

def contains_math(data):
    # fast_match = fast_regex.search(data)
    # if fast_match:
    original_match = original_regex.search(data)
    if original_match:
        return True
    latex_match = latex_regex.search(data)
    text = ''
    if latex_match:
        data = data.replace('<template', '<div')
        data = data.replace('</template', '</div')
        tree = HTMLTree.parse(data)
        text = extract_plain_text(tree, 
                                  main_content=True, 
                                  alt_texts=False)
        for term in latex_math_commands:
            if term in text:
                return True
        score = score_text(text)
        if score > 0.8 and len(text) > 500:
            return True

    return False

def is_html(record):
    """Check that the record is an HTML record."""
    if record.headers is None:
        return False
    if record.http_headers is None:
        return False
    content_type = str(record.http_content_type)
    if content_type.startswith('text/html') or content_type.startswith('application/xhtml+xml'):
        return True
    return False

def extract(html, config):
    res = extract_text(html, config, fast=True)
    if res is None:
        return None
    text, info = res
    metadata = {
        'extraction_info': info,
        'config': config,
    }
    return text, metadata

def load_warc(warc_file):
    """Loads a WARC file with fsspec. Retries if it fails."""
    for i in range(10):
        try:
            with fsspec.open(warc_file, 'rb') as f:
                return f.read()
        except:
            if i == 9:
                raise Exception('Failed to read {}'.format(warc_file))
            print('Retrying to read {}'.format(warc_file))
            # Sleep a random amount of time
            sleep(random.random())

def process_warc(warc_file):
    """Yields extracted text from a WARC file."""

    doc_counter = defaultdict(int)

    # Error if it takes more than 20 minutes
    with time_guard(timeout=60*20):
        try:
            f = load_warc(warc_file)
            for i in range(10):
                try:
                    stream = BytesIO(f)
                    break
                except:
                    if i == 9:
                        print('Failed to read {}'.format(warc_file))
                        return
                    print('Retrying to read {}'.format(warc_file))
                    
            # We only want to extract text from the response records
            total_parsed = 0
            total_has_math = 0
            for record in tqdm(ArchiveIterator(stream)):
                try:
                    doc_counter['records'] += 1
                    if record.headers is None: continue
                    if record.http_headers is None: continue
                    if record.headers['WARC-Type'] != 'response': continue
                    if not is_html(record): continue
                    doc_counter['html'] += 1
                    # Extract text from the payload
                    html = record.reader.read()
                    html = decode_html(html)
                    url = record.headers.get('WARC-Target-URI')
                    record_date = record.record_date
                    if html is None: continue
                    if not contains_math(html): continue
                    doc_counter['passes prefilter'] += 1
                    randomized_config_sample = randomized_config.sample()
                    res = extract(html, randomized_config_sample)
                    total_parsed += 1
                    print(f'Running percentage: {total_has_math / total_parsed:.2f}, total parsed: {total_parsed}, total has math: {total_has_math}')
                    if res is None: continue
                    randomized_text, metadata = res

                    found_math = metadata['extraction_info']['found_math']
                    if not is_english(randomized_text): continue
                    doc_counter['is english'] += 1
                    score = score_text(randomized_text)
                    if found_math:
                        if score < 0.15: continue
                    else:
                        if score < 0.8: continue
                    doc_counter['passes score'] += 1

                    metadata['extraction_info']['math_score'] = score
                    perplexity = document_perplexity(randomized_text)
                    metadata['extraction_info']['perplexity'] = perplexity

                    if perplexity > 30_000: continue
                    doc_counter['passes perplexity'] += 1
                    total_has_math += 1

                    yield (url, (randomized_text, html, warc_file, str(metadata), str(record_date)))
                except Exception as e:
                    print(f'Execution timeout for {warc_file}')
                    print(e)
                    # Print a trace
                    traceback.print_exc()
        except:
            print(f'Execution (probably) timed out for {warc_file}')
            return
        
        print(f'Finished processing {warc_file}'
              f' with {total_has_math} math-containing pages out of {total_parsed} parsed pages')
        
        # Save the doc counter as a text file with a uuid name
        with open(f'{uuid.uuid4()}.txt', 'w') as f:
            f.write(json.dumps(doc_counter))
    

def main(warc_file, output_dir):
    data = []
    print('Extracting text from {}'.format(warc_file))
    for url, (text, html, _, metadata, _) in process_warc(warc_file):
        data.append({'url': url, 'value': {'text': text, 'no_latex': '', 'no_randomization': '', 'html': html, 'warc_path': warc_file, 'metadata': metadata}})
    # Remove directories if they exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    # Save the data as md files
    for i, d in enumerate(data):
        text = d['value']['text']
        html = d['value']['html']
        url = d['url']
        with open(os.path.join(output_dir, '{}.md'.format(i)), 'w') as f:
            f.write('# {}\n\n'.format(url))
            f.write(text)
        with open(os.path.join(output_dir, '{}.html'.format(i)), 'w') as f:
            f.write(html)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--warc_file', help='WARC file to extract text from')
    parser.add_argument('--output_dir', help='Output dir')
    args = parser.parse_args()
    main(args.warc_file, args.output_dir)