#!/bin/bash

PATH=/home/vscode/.cargo/bin:$PATH
cd dolma
source /home/vscode/miniforge3/bin/activate && pip install cmake "maturin>=1.5,<2.0"
