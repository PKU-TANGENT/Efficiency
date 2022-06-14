#!/bin/bash
export TASK_NAME=$1
export CUDA_VISIBLE_DEVICES=$2
# model_name_or_path=$3
model_name_or_path=roberta-base
# model_name_or_path=princeton-nlp/unsup-simcse-roberta-base
hub_model_id="${model_name_or_path/\//"-"}-${TASK_NAME}"
output_dir="./fine-tune/$model_name_or_path/$TASK_NAME/"
eval_output_dir="./eval/$model_name_or_path/$TASK_NAME/"
# python -m debugpy --listen 127.0.0.1:9999 --wait-for-client run_glue.py \
python run_glue.py \
  --model_name_or_path $output_dir \
  --task_name $TASK_NAME \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 8 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --output_dir $eval_output_dir \
  --hub_model_id $hub_model_id \
  --push_to_hub \
  --load_best_model_at_end
find $output_dir -name *optimizer.pt -delete
find $output_dir -name *scheduler.pt -delete
find $output_dir -name *pytorch_model.bin -delete
# rm -rf $output_dir/.git
# rm -rf $eval_output_dir
