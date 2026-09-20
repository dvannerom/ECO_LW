#!/bin/bash

d="${1:?Usage: $0 <d>}"

python3 fit_ADM.py -d ${d} -r 2
