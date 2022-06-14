#!/bin/bash
export TASK_NAME=$1
# export CUDA_VISIBLE_DEVICES=$2
model_name_or_path=$2
# model_name_or_path=roberta-base
# model_name_or_path=princeton-nlp/unsup-simcse-roberta-base
hub_model_id="${model_name_or_path/\//"-"}-${TASK_NAME}"
output_dir="./fine-tune/$model_name_or_path/$TASK_NAME/"
find $output_dir -name *optimizer.pt -delete
find $output_dir -name *scheduler.pt -delete
find $output_dir -name *pytorch_model.bin -delete
rm -rf $output_dir/.git
