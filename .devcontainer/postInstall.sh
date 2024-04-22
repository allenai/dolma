#!/bin/bash

PATH=/home/vscode/.cargo/bin:$PATH
cd dolma
source /home/vscode/miniforge3/bin/activate && pip install cmake "maturin[patchelf]>=1.1,<2.0"
