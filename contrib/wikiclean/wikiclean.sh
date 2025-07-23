#!/bin/bash

uv add --dev pip
uv run spacy download en_core_web_sm

PYTHONPATH=contrib/wikiclean:$PYTHONPATH uv run dolma tag \
    --taggers wikiclean \
    --tagger_modules wikiclean \
    --documents 's3://ai2-llm/pretraining-data/sources/wikipedia/v0/documents/lang=en/*.gz' \
    --destination tmp/wikiclean \
    --processes 10 \
    --debug
