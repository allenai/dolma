#!/bin/bash
set -e

UNAME="$(uname)"
PLATFORM="$(uname -m)"

if [[ $UNAME == "Darwin" ]]; then
  echo "MacOS detected..."
  which cmake || brew install cmake
  which protoc || brew install protobuf
  which openssl || brew install openssl
elif [[ $UNAME == "Linux" ]]; then
  echo "Linux detected..."
  which cmake || sudo apt-get install --yes build-essential cmake
  which protoc || sudo apt-get install --yes protobuf-compiler
  which openssl || sudo apt-get install --yes libssl-dev
else
  echo "Unsupported OS; please install rust, cmake, protobuf, maturin and openssl manually!"
  exit 1
fi

which cargo || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

if [[ $PLATFORM == "x86_64" ]]; then
  echo "x86_64 detected..."
  which maturin || pip install maturin[patchelf]
fi

if [[ $PLATFORM = "aarch64" ]]; then
  echo "aarch64 detected..."
  which maturin || pip install maturin
fi

if [[ $PLATFORM = arm* ]]; then
  echo "arm detected..."
  which maturin || pip install maturin
else
  echo "Unsupported platform; please install maturin manually"
  exit 0
fi
