#!/bin/bash

d="${1:?Usage: $0 <d>}"

python3 narrowband_to_broadband.py -d ${d}
