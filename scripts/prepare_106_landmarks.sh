#!/bin/bash
python3 src/dataset/convert_98_to_106.py \
    --input data/raw/wflw/WFLW_annotations/list.json \
    --output data/processed/wflw_landmarks_106.json