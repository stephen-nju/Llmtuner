#!/bin/bash

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# export PROJECT_PATH=/home/jovyan/zhubin/LLaMA-Factory/
# cd ${PROJECT_PATH}

# CUDA_VISIBLE_DEVICES=4 python src/web_demo.py \
# 	--model_name_or_path /home/jovyan/zhubin/saved_checkpoint/simpo_v1_ep3_lr5e6_bs2/checkpoint-1285/ \
# 	--template honor \
# 	--finetuning_type full

# CUDA_VISIBLE_DEVICES=0
# llamafactory-cli webchat \
# 	--model_name_or_path /home/jovyan/zhubin/saved_checkpoint/0925_Qwen2.5_3b_instruct_neft_alp_dig_callsumv3_ep3_lr2e5_bs4/checkpoint-3027/ \
# 	--template qwen \
# 	--finetuning_type full

CUDA_VISIBLE_DEVICES=3
llamafactory-cli webchat \
	--model_name_or_path /home/jovyan/zhubin/saved_checkpoint/0925_magiclm_nano_summary_3stage_lora_low/merge/ \
	--template honor \
	--finetuning_type full
