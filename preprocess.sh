#!/bin/bash

python convert_fr3_preprocess.py \
    --input_dir /boot/common_data/fr3_block_stacking_0920_50ep \
    --output_dir /boot/common_data/fr3_block_stacking_seg_overlap_hdf5 \
    --downsample_rate 3 \
    --segment_mode overlap \
    --target_length 400 
