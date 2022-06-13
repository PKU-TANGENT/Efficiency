#!/bin/bash
export TASK_NAME=$1
export CUDA_VISIBLE_DEVICES=$2
# model_name_or_path=$3
model_name_or_path=roberta-base
# model_name_or_path=princeton-nlp/unsup-simcse-bert-base-uncased
output_dir="./fine-tune/$model_name_or_path/$TASK_NAME/"
# python run_glue_hyper_search.py \
python -m debugpy --listen 127.0.0.1:9999 --wait-for-client run_glue_hyper_search.py \
  --model_name_or_path $model_name_or_path \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_eval_batch_size 8 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --evaluation_strategy "steps" \
  --save_total_limit 1 \
  --output_dir $output_dir
find $output_dir -name *optimizer.pt -delete
find $output_dir -name *scheduler.pt -delete
find $output_dir -path *checkpoint*pytorch_model.bin -delete