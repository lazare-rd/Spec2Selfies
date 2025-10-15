#!/bin/bash

FREEZE_STRATEGY="None"
OUTPUT_DIR="./data/saved_models/Corrector_withFormulas"
DATASET_NAME="./data/scaffold_split_merged_spectra/arrow/hypothesis_dataset/with_formula"

CMD="python run.py \
	   --overwrite_output_dir \
	   --use_lora=True \
	   --fp16=True \
	   --output_dir=${OUTPUT_DIR} \
	   --freeze_strategy=${FREEZE_STRATEGY} \
	   --debug_model \
	   --experiment=corrector \
	   --with_formula=True \
	   --dataset_name=${DATASET_NAME} \
	   --use_less_data 1000"

echo "Running command:"
echo $CMD
