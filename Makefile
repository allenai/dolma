setup:
	@./setup.sh

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

check:
	isort --check .
	black --check .
	mypy --disable-error-code="override" tests/python/
	mypy --disable-error-code="override" python/
	flake8 tests/python/
	flake8 python/
	rustfmt --edition 2021 src/*.rs --check
