#!/bin/bash
export TASK_NAME=$1
export CUDA_VISIBLE_DEVICES=$2
# model_name_or_path=$3
model_name_or_path=roberta-base
# model_name_or_path=princeton-nlp/unsup-simcse-roberta-base
hub_model_id="${model_name_or_path/\//"-"}-${TASK_NAME}"
output_dir="./fine-tune/$model_name_or_path/$TASK_NAME/"
# python -m debugpy --listen 127.0.0.1:9999 --wait-for-client run_glue.py \
python run_glue.py \
  --model_name_or_path $model_name_or_path \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --hub_model_id $hub_model_id \
  --push_to_hub \
  --load_best_model_at_end \
  --greater_is_better True \
  # --private
# find $output_dir -name *optimizer.pt -delete
# find $output_dir -name *scheduler.pt -delete
# find $output_dir -name *pytorch_model.bin -delete
# rm -rf $output_dir/.git
