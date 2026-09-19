#!/bin/bash

for i in $(seq 246 274)
do
	echo $i
	python3 scene_id_pca.py -f data/preprocessed_files/abi_${i}_res2km_step1.npz
done
