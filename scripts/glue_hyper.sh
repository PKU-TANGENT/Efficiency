#!/bin/bash
export TOKENIZERS_PARALLELISM=false
# export WANDB_DISABLED="true"
TASK_NAME=mrpc
# export CUDA_VISIBLE_DEVICES=0
model_name_or_path=roberta-base
# TASK_NAME=$2
if [[ "${TASK_NAME}" == "mrpc" ]]; then
  num_train_epochs=5
else
  num_train_epochs=10
fi
export CUDA_VISIBLE_DEVICES=$1
num_hidden_layers=$2
# model_name_or_path=$3
# ffn_layers=$2
# ffn_layers=10
freeze_backbone=False
IFS="-" read -r -a name_parser <<< "$model_name_or_path"
model_architecture="${name_parser[0]}"
if [[ "${model_architecture}" == "bert" ]]; then
  pooler_type=cls
else
  pooler_type=avg
fi
prefix=""
learning_rate=2e-5
# learning_rate=$2
suffix="-${pooler_type}-ffn_layer${ffn_layers}-lr${learning_rate}-num_hidden_layers${num_hidden_layers}-frozen${freeze_backbone}"
hub_model_id="${prefix}${model_name_or_path/\//"-"}${suffix}-${TASK_NAME}"
output_dir="./fine-tune/${prefix}${model_name_or_path}${suffix}/${TASK_NAME}/"
export WANDB_PROJECT=$model_name_or_path
# python -m debugpy --listen 127.0.0.1:9999 --wait-for-client run_glue.py \
python run_glue.py \
  --task_name $TASK_NAME \
  --model_name_or_path $model_name_or_path \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --learning_rate $learning_rate \
  --num_train_epochs $num_train_epochs \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --output_dir $output_dir \
  --load_best_model_at_end \
  --greater_is_better True \
  --private \
  --early_stopping_patience 5 \
  --model_class_name "Pooler${model_architecture^}ForSequenceClassification" \
  --model_package_name "modeling_${model_architecture}" \
  --pooler_type $pooler_type \
  --overwrite_output_dir \
  --num_hidden_layers $num_hidden_layers \
  --freeze_backbone $freeze_backbone \
  # --ffn_layers $ffn_layers \
  # --hub_model_id $hub_model_id \
  # --push_to_hub \
