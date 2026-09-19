#!/bin/bash

d="${1:?Usage: $0 <d>}"

python3 radiance_to_flux.py -d ${d}
