# Check if uv is available
UV_AVAILABLE := $(shell command -v uv 2> /dev/null)

setup:
	@./setup.sh

publish:
	maturin publish

test: test-python test-rust

test-python:
ifdef UV_AVAILABLE
	uv run --reinstall-package dolma pytest -vsx tests/python
else
	maturin develop --extras="all"
	pytest -vsx tests/python
endif
	rm -rf tests/work/*

test-rust:
	cargo test -- --nocapture
	rm -rf tests/work/*

develop:
ifdef UV_AVAILABLE
	uv sync --extra all
else
	maturin develop --extras="all"
endif

run:
ifdef UV_AVAILABLE
	uv run --reinstall-package dolma dolma $(ARGS)
else
	maturin develop --release
	dolma $(ARGS)
endif

style:
	rustfmt --edition 2021 src/*.rs
ifdef UV_AVAILABLE
	uv run isort .
	uv run black .
else
	isort .
	black .
endif

check:
ifdef UV_AVAILABLE
	uv run isort --check .
	uv run black --check .
	uv run mypy tests/python/
	uv run mypy python/
	uv run flake8 tests/python/
	uv run flake8 python/
else
	isort --check .
	black --check .
	mypy tests/python/
	mypy python/
	flake8 tests/python/
	flake8 python/
endif
	rustfmt --edition 2021 src/*.rs --check
