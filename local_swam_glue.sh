#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=roberta
# export TASK_NAME=$1
# export CUDA_VISIBLE_DEVICES=$2
# model_name_or_path=$3
export TASK_NAME=stsb
export CUDA_VISIBLE_DEVICES=7
model_name_or_path=roberta-base
# model_name_or_path="JeremiahZ/roberta-base-rte"
prefix="hyper-swam-"
hub_model_id="${prefix}${model_name_or_path/\//"-"}-${TASK_NAME}"
# output_dir="./fine-tune/${prefix}$model_name_or_path/$TASK_NAME/"
output_dir="ray_results/pbt_swam_stsb/pbt_swam_stsb/_objective_d87e4_00005_5_learning_rate=0.0000,model_head_lr=0.0001,num_train_epochs=13,warmup_ratio=0.0091,weight_decay=0.1855_2022-06-22_17-18-00/checkpoint_001170/checkpoint-1170"
local_eval_dir="local-${hub_model_id}"
# python -m debugpy --listen 127.0.0.1:9999 --wait-for-client swam_glue.py \
python swam_glue.py \
  --task_name $TASK_NAME \
  --model_name_or_path $output_dir \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --save_total_limit 1 \
  --output_dir $local_eval_dir \
  --load_best_model_at_end \
  --greater_is_better True \
  --private \
  --early_stopping_patience 10 \
  --hub_model_id $hub_model_id \
  --push_to_hub \
  # --do_train \
  # --overwrite_output_dir \
# find $output_dir -name *optimizer.pt -delete
# find $output_dir -name *scheduler.pt -delete
# find $output_dir -name *pytorch_model.bin -delete
# rm -rf $output_dir/.git
