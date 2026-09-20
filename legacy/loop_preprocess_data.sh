#!/bin/bash

d="${1:?Usage: $0 <d>}"

python3 preprocess_data_ABI.py -d ${d}
