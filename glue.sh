#!/bin/bash
export TASK_NAME=$1
export CUDA_VISIBLE_DEVICES=$2
model_name_or_path=$3
output_dir="./fine-tune/$model_name_or_path/$TASK_NAME/"
# model_name_or_path=roberta-base
# model_name_or_path=princeton-nlp/unsup-simcse-bert-base-uncased
# python -m debugpy --listen 127.0.0.1:9999 --wait-for-client run_glue.py \
python run_glue.py \
  --model_name_or_path $model_name_or_path \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 8 \
  --warmup_ratio 0.06 \
  --learning_rate 2e-5 \
  --num_train_epochs $4 \
  --output_dir $output_dir
find $output_dir -name *pytorch_model.bin -delete
