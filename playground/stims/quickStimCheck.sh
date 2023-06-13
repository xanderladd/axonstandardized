#!/bin/bash

h5dump --header neg_stims.hdf5 | while read line; do
	if [[ "$line" == *"DATASET"* ]]; then
		echo "$line" | cut -d '"' -f 2
	fi
done
