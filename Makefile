UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
    OS_MESSAGE := "MacOS detected"
	CMAKE_SETUP := "which cmake || brew install cmake"
	PROTOBUF_SETUP := "which protoc || brew install protobuf"
	OPENSSL_SETUP := "which openssl || brew install openssl"
else ifeq ($(UNAME), Linux)
	OS_MESSAGE := "Linux detected"
	CMAKE_SETUP := "which cmake || sudo apt-get install --yes build-essential cmake"
	PROTOBUF_SETUP := "which protoc || sudo apt-get install --yes protobuf-compiler"
	OPENSSL_SETUP := "which openssl || sudo apt-get install --yes libssl-dev"
else
	OS_MESSAGE := "Unsupported OS; please install rust, cmake, protobuf, and openssl manually"
	CMAKE_SETUP := ""
	PROTOBUF_SETUP := ""
	OPENSSL_SETUP := ""
endif

setup:
	@echo "${OS_MESSAGE}: installing..."
	$(shell "${CMAKE_SETUP}")
	$(shell "${PROTOBUF_SETUP}")
	$(shell "${OPENSSL_SETUP}")
	which cargo || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	which maturin || pip install maturin

publish:
	maturin publish

test: setup develop setup-test test-python test-rust

test-python:
	pytest -vs tests/python

test-rust-clean:
	rm -rf tests/work/*
	aws s3 rm --recursive s3://ai2-llm/pretraining-data/tests/mixer/

test-rust-setup:
	aws s3 cp tests/data/documents.json.gz s3://ai2-llm/pretraining-data/tests/mixer/inputs/v0/documents/head/0000.json.gz
	aws s3 cp tests/data/pii-attributes.json.gz s3://ai2-llm/pretraining-data/tests/mixer/inputs/v0/attributes/pii/head/0000.json.gz
	aws s3 cp tests/data/toxicity-attributes.json.gz s3://ai2-llm/pretraining-data/tests/mixer/inputs/v0/attributes/toxicity/head/0000.json.gz
	aws s3 cp tests/data/sample-attributes.json.gz s3://ai2-llm/pretraining-data/tests/mixer/inputs/v0/attributes/sample/head/0000.json.gz
	aws s3 cp tests/data/duplicate-paragraphs.json.gz s3://ai2-llm/pretraining-data/tests/mixer/inputs/v0/attributes/duplicate_paragraphs/head/0000.json.gz

test-rust: test-rust-clean test-rust-setup
	cargo test -- --nocapture

develop:
	maturin develop --extras=dev

style:
	rustfmt --edition 2021 src/*.rs
	autopep8 --in-place --recursive python/
	isort python/
	black python/
	autopep8 --in-place --recursive tests/python/
	isort tests/python/
	black tests/python/
