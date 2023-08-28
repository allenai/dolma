UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
    OS_MESSAGE := "MacOS detected"
	CMAKE_SETUP := "which cmake || brew install cmake"
	PROTOBUF_SETUP := "which protoc || brew install protobuf"
	OPENSSL_SETUP := "which openssl || brew install openssl"
else ifeq ($(UNAME), Linux)
	OS_MESSAGE := "Linux detected"
	CMAKE_SETUP := "which cmake || conda install -c anaconda cmake"
	PROTOBUF_SETUP := "which protoc || conda install -c conda-forge protobuf"
	OPENSSL_SETUP := "which openssl || conda install -c anaconda openssl"
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
	which maturin || pip install maturin[patchelf]

publish:
	maturin publish

test: test-python test-rust

test-python:
	pytest -vs tests/python
	rm -rf tests/work/*

test-rust:
	cargo test -- --nocapture
	rm -rf tests/work/*

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
