release:
	maturin build

test: test-python test-rust

test-python: develop setup
	pytest -vs tests/python

test-rust: clean-test-data setup-test-data develop setup
	pytest -vs tests/rust


clean-test-data:
	rm -rf tests/work/*
	aws s3 rm --recursive s3://ai2-llm/pretraining-data/tests/mixer/

setup-test-data:
	aws s3 cp tests/data/documents.json.gz s3://ai2-llm/pretraining-data/tests/mixer/inputs/v0/documents/head/0000.json.gz
	aws s3 cp tests/data/pii-attributes.json.gz s3://ai2-llm/pretraining-data/tests/mixer/inputs/v0/attributes/pii/head/0000.json.gz
	aws s3 cp tests/data/toxicity-attributes.json.gz s3://ai2-llm/pretraining-data/tests/mixer/inputs/v0/attributes/toxicity/head/0000.json.gz
	aws s3 cp tests/data/sample-attributes.json.gz s3://ai2-llm/pretraining-data/tests/mixer/inputs/v0/attributes/sample/head/0000.json.gz
	aws s3 cp tests/data/duplicate-paragraphs.json.gz s3://ai2-llm/pretraining-data/tests/mixer/inputs/v0/attributes/duplicate_paragraphs/head/0000.json.gz

develop:
	maturin develop --extras=dev

setup:
	which cmake || sudo apt-get install --yes build-essential cmake
	which cargo || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
