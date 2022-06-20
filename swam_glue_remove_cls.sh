#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
# export TASK_NAME=$1
# export CUDA_VISIBLE_DEVICES=$2
# model_name_or_path=$3
export TASK_NAME=wnli
export CUDA_VISIBLE_DEVICES=7
model_name_or_path=roberta-base
# model_name_or_path="JeremiahZ/roberta-base-rte"
prefix="swam-"
hub_model_id="${prefix}${model_name_or_path/\//"-"}-${TASK_NAME}"
output_dir="./fine-tune/${prefix}$model_name_or_path/$TASK_NAME/"
# python -m debugpy --listen 127.0.0.1:9999 --wait-for-client swam_glue_remove_cls.py \
python swam_glue_remove_cls.py \
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
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --output_dir "${output_dir}remove-cls" \
  --load_best_model_at_end \
  --greater_is_better True \
  --private \
  --early_stopping_patience 10 \
  # --hub_model_id $hub_model_id \
  # --push_to_hub \
  # --overwrite_output_dir \
# find $output_dir -name *optimizer.pt -delete
# find $output_dir -name *scheduler.pt -delete
# find $output_dir -name *pytorch_model.bin -delete
# rm -rf $output_dir/.git
