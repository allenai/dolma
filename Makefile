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
	which maturin || pip install maturin[patchelf]

publish:
	maturin publish

test: test-python test-rust

test-python:
	maturin develop --extras="all"
	pytest -vsx tests/python
	rm -rf tests/work/*

test-rust:
	cargo test -- --nocapture
	rm -rf tests/work/*

develop:
	maturin develop --extras="all"

style:
	rustfmt --edition 2021 src/*.rs
	isort .
	black .
