#!/bin/bash
export TOKENIZERS_PARALLELISM=false
# export TASK_NAME=$1
# export CUDA_VISIBLE_DEVICES=$2
# model_name_or_path=$3
export TASK_NAME=wnli
export CUDA_VISIBLE_DEVICES=7
model_name_or_path=roberta-base
per_device_train_batch_size=16
warmup_ratio=0.06
learning_rate=1e-5
model_head_lr=2e-5
# model_name_or_path="JeremiahZ/roberta-base-rte"
prefix="swam-"
suffix="-bs${per_device_train_batch_size}-warmup${warmup_ratio}-lr${learning_rate}-headlr${model_head_lr}"
hub_model_id="${prefix}${model_name_or_path/\//"-"}-${TASK_NAME}${suffix}"
output_dir="./fine-tune/${prefix}$model_name_or_path/${TASK_NAME}${suffix}/"
# python -m debugpy --listen 127.0.0.1:9999 --wait-for-client swam_glue.py \
python swam_glue.py \
  --task_name $TASK_NAME \
  --model_name_or_path $model_name_or_path \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $per_device_train_batch_size \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --learning_rate $learning_rate \
  --num_train_epochs 10 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --output_dir $output_dir \
  --load_best_model_at_end \
  --greater_is_better True \
  --early_stopping_patience 10 \
  --model_head_lr $model_head_lr \
  --private \
  # --push_to_hub \
  # --hub_model_id $hub_model_id \
  # --overwrite_output_dir \
# find $output_dir -name *optimizer.pt -delete
# find $output_dir -name *scheduler.pt -delete
# find $output_dir -name *pytorch_model.bin -delete
# rm -rf $output_dir/.git
