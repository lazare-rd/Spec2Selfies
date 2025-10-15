#!/bin/bash

FREEZE_STRATEGY="None"
OUTPUT_DIR="./data/saved_models/Inverter_PerfectModel"
DATASET_NAME="./data/scaffold_split_merged_spectra/arrow/tokenized_datasets/with_formula/perfect_model"

CMD="python run.py \
	   --overwrite_output_dir \
	   --use_lora=True \
	   --fp16=True \
	   --output_dir=${OUTPUT_DIR} \
	   --freeze_strategy=${FREEZE_STRATEGY} \
	   --debug_model \
	   --use_less_data 1000 \
	   --dataset_name=${DATASET_NAME} \
	   --use_perfect_model=True \
	   --exp_name=usingPerfectModel
	   --with_formula=False"

echo "Running command:"
echo $CMD
