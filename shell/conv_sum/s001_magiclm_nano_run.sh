# 先激活环境
export PROJECT_PATH=/home/jovyan/zhubin/code/LLaMA-Factory/

cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=${PROJECT_PATH}/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=${PROJECT_PATH}/config/deepspeed/zero_stage2_config.json

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

###运行训练时需要先进行配置
export DATASET="alpace_gpt4_zh_retain,firefly_summary_part,COIG_PC_core_summary_part,union_conversations_v3_norm"

export WANDB_PROJECT="MagicLM_Nano_Quant"
# wo_fb30 =without full baichuan_30
export WANDB_NAME="magiclm_nano_quant_1017_lora_rank32_lora_alpha1_lr1e4"
export OUTPUT_DIR=/home/jovyan/zhubin/saved_checkpoint/${WANDB_NAME}
export HOSTFILE=/home/jovyan/zhubin/code/LLaMA-Factory/config/hostfile
mkdir -p ${OUTPUT_DIR}

# FIRST_NODE=$(awk 'NR==1 {print $1}' ${HOSTFILE})
# MASTER_ADDR=$(hostname -I |awk '{print $1}')
# echo "Using IP address of ${MASTER_ADDR} for node ${FIRST_NODE}"
# --include="node12:0,1,2,3,4,5,6,7"

wandb offline
deepspeed --hostfile=${HOSTFILE} --include="node3" --master_port=${MASTER_PORT} --no_local_rank \
	src/train.py \
	--deepspeed ${DS_CONFIG_STAGE_2} \
	--stage sft \
	--template honor \
	--do_train \
	--model_name_or_path /home/jovyan/zhubin/DATA/models/MagicLM-3B-OmniQuant \
	--resize_vocab false \
	--use_fast_tokenizer false \
	--report_to wandb \
	--overwrite_output_dir \
	--overwrite_cache \
	--dataset ${DATASET} \
	--cutoff_len 2048 \
	--output_dir ${OUTPUT_DIR} \
	--num_train_epochs 2 \
	--overwrite_cache \
	--finetuning_type lora \
	--lora_rank 32 \
	--lora_alpha 1 \
	--lora_target wo,wqkv,w1,w2,w3 \
	--warmup_ratio 0.01 \
	--logging_steps 10 \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--preprocessing_num_workers 16 \
	--gradient_accumulation_steps 8 \
	--save_steps 5000 \
	--save_total_limit 2 \
	--learning_rate 2e-4 \
	--bf16 true \
	2>&1 | tee ${OUTPUT_DIR}/train.log

wait

deepspeed --hostfile=${HOSTFILE} --include="node3" --master_port=${MASTER_PORT} --no_local_rank \
	src/train.py \
	--deepspeed ${DS_CONFIG_STAGE_2} \
	--stage sft \
	--template honor \
	--do_train \
	--model_name_or_path /home/jovyan/zhubin/DATA/models/honor2_5b_patched_tokenizer/ \
	--resize_vocab false \
	--use_fast_tokenizer false \
	--report_to wandb \
	--overwrite_output_dir \
	--overwrite_cache \
	--dataset ${DATASET} \
	--cutoff_len 2048 \
	--output_dir /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_1016_callsumv3_ep3_lr1e4_bs4/ \
	--num_train_epochs 2 \
	--overwrite_cache \
	--finetuning_type lora \
	--lora_rank 32 \
	--lora_alpha 1 \
	--lora_target wo,wqkv,w1,w2,w3 \
	--warmup_ratio 0.01 \
	--logging_steps 10 \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--preprocessing_num_workers 16 \
	--gradient_accumulation_steps 8 \
	--save_steps 5000 \
	--save_total_limit 2 \
	--learning_rate 2e-4 \
	--bf16 true \
	2>&1 | tee $/home/jovyan/zhubin/saved_checkpoint/magiclm_nano_1016_callsumv3_ep3_lr1e4_bs4/train.log
