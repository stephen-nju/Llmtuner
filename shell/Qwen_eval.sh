#!/usr/bin/bash
Usage() {
	cat <<EOF
Usage:eval dataset
EOF
}

export PROJECT_PATH=/opt/nas/p/zhubin/code/LLaMA-Factory/
cd ${PROJECT_PATH}
export hoststr='node12 slots=8'
export model_name_or_path=/home/jovyan/cht/honor/pretrain_model/Qwen2.5-1.5B/
export adapter_name_or_path
export eval_dataset
export finetuning_type=lora
export template=qwen
export output_dir

options=$(getopt -l "help,model_name_or_path:,hoststr:,adapter_name_or_path:,eval_dataset:,finetuning_type:,template:,output_dir:" -o "h:d:t:n:m:g:" -a -- "$@")

eval set -- "$options"
# echo "$options"
while true; do
	case "$1" in
	-h | --help)
		Usage
		exit 0
		;;
	-m | --model_name_or_path)
		shift
		model_name_or_path="$1"
		;;
	--adapter_name_or_path)
		shift
		adapter_name_or_path="$1"
		;;
	-t | --template)
		shift
		template="$1"
		;;
	--finetuning_type)
		shift
		finetuning_type="$1"
		;;
	-d | --eval_dataset)
		shift
		eval_dataset="$1"
		;;
	--hoststr)
		shift
		hoststr="$1"
		;;
	--)
		shift
		break
		;;
	esac
	shift
done

optional_params=()
# 处理python脚本中default等于None的选项
if [[ -n $adapter_name_or_path ]]; then
	output_dir=${adapter_name_or_path}/output
	optional_params+=(--adapter_name_or_path ${adapter_name_or_path})
else
	output_dir=${model_name_or_path}/output
fi

attrun \
	--hoststr="${hoststr}" \
	torchrun \
	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
	src/train.py \
	--stage sft \
	--model_name_or_path ${model_name_or_path} \
	--resize_vocab true \
	--do_predict \
	--eval_dataset ${eval_dataset} \
	--template ${template} \
	--finetuning_type ${finetuning_type} \
	--output_dir ${output_dir} \
	--cutoff_len 4069 \
	--max_new_tokens 512 \
	--do_sample false \
	--per_device_eval_batch_size 2 \
	--predict_with_generate

attrun \
	--hoststr="${hoststr}" \
	torchrun \
	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
	src/train.py \
	--stage sft \
	--model_name_or_path ${model_name_or_path} \
	--resize_vocab true \
	--do_eval \
	--eval_dataset ${eval_dataset} \
	--template ${template} \
	--finetuning_type ${finetuning_type} \
	--output_dir ${output_dir} \
	--cutoff_len 4069 \
	--max_new_tokens 512 \
	--do_sample false \
	--per_device_eval_batch_size 2

# checkpoints=("checkpoint-1094" "checkpoint-2188" "checkpoint-3282")

# for checkpoint in "${checkpoints[@]}"; do
# 	echo "checkpoint: $checkpoint"
# 	attrun \
# 		--hoststr="${HOSTSTR}" \
# 		--include="${INCLUDE}" \
# 		torchrun \
# 		--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
# 		src/train.py \
# 		--stage sft \
# 		--model_name_or_path /home/jovyan/zhubin/DATA/models/honor2_5b_patched_tokenizer/ \
# 		--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_1128_neft_buls_lora_ep3_lr1e4_bs4/${checkpoint} \
# 		--resize_vocab true \
# 		--do_predict \
# 		--eval_dataset union_conversations_v4_dev_wo_magic_data \
# 		--template honor \
# 		--finetuning_type lora \
# 		--output_dir /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_1128_neft_buls_lora_ep3_lr1e4_bs4/$checkpoint/output \
# 		--cutoff_len 2048 \
# 		--max_new_tokens 512 \
# 		--do_sample false \
# 		--per_device_eval_batch_size 4 \
# 		--predict_with_generate
# done

# attrun \
# 	--hoststr="${HOSTSTR}" \
# 	--include="${INCLUDE}" \
# 	torchrun \
# 	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
# 	src/train.py \
# 	--stage sft \
# 	--model_name_or_path /home/jovyan/zhubin/DATA/models/honor2_5b_patched_tokenizer/ \
# 	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_1128_neft_uls_lora_ep3_lr5e5_bs4/ \
# 	--resize_vocab true \
# 	--do_predict \
# 	--eval_dataset union_conversations_v4_dev_wo_magic_data \
# 	--template honor \
# 	--finetuning_type lora \
# 	--output_dir /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_1128_neft_uls_lora_ep3_lr5e5_bs4/output \
# 	--cutoff_len 2048 \
# 	--max_new_tokens 512 \
# 	--do_sample false \
# 	--per_device_eval_batch_size 4 \
# 	--predict_with_generate

# attrun \
# 	--hoststr="${HOSTSTR}" \
# 	--include="${INCLUDE}" \
# 	torchrun \
# 	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
# 	src/train.py \
# 	--stage sft \
# 	--model_name_or_path /home/jovyan/zhubin/DATA/models/honor2_5b_patched_tokenizer/ \
# 	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_1121_neft_auls_lora_ep3_lr1e4_bs4/ \
# 	--resize_vocab true \
# 	--do_predict \
# 	--eval_dataset zhipu_dev \
# 	--template honor \
# 	--finetuning_type lora \
# 	--output_dir /home/jovyan/zhubin/code/LLaMA-Factory/saved_output/lora_honor_zhipu \
# 	--cutoff_len 4069 \
# 	--max_new_tokens 512 \
# 	--do_sample false \
# 	--per_device_eval_batch_size 2 \
# 	--predict_with_generate

# attrun \
# 	--hoststr="${HOSTSTR}" \
# 	--include="${INCLUDE}" \
# 	torchrun \
# 	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
# 	src/train.py \
# 	--stage sft \
# 	--model_name_or_path /home/jovyan/zhubin/DATA/models/honor2_5b_patched_tokenizer/ \
# 	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_1121_neft_auls_lora_ep3_lr1e4_bs4/ \
# 	--resize_vocab true \
# 	--do_predict \
# 	--eval_dataset test_department_data \
# 	--template honor \
# 	--finetuning_type lora \
# 	--output_dir /home/jovyan/zhubin/code/LLaMA-Factory/saved_output/lora_honor_test_department \
# 	--cutoff_len 4069 \
# 	--max_new_tokens 512 \
# 	--do_sample false \
# 	--per_device_eval_batch_size 2 \
# 	--predict_with_generate

# attrun \
# 	--hoststr="${HOSTSTR}" \
# 	--include="${INCLUDE}" \
# 	torchrun \
# 	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
# 	src/train.py \
# 	--stage sft \
# 	--model_name_or_path /home/jovyan/zhubin/DATA/models/honor2_5b_patched_tokenizer/ \
# 	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_1130_neft_budsls_lora_ep3_lr1e4_bs4/ \
# 	--resize_vocab true \
# 	--do_predict \
# 	--eval_dataset union_conversations_v4_dev_wo_magic_data \
# 	--template honor \
# 	--finetuning_type lora \
# 	--output_dir /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_1130_neft_budsls_lora_ep3_lr1e4_bs4/output \
# 	--cutoff_len 4069 \
# 	--max_new_tokens 512 \
# 	--do_sample false \
# 	--per_device_eval_batch_size 2 \
# 	--predict_with_generate

# attrun \
# 	--hoststr="${HOSTSTR}" \
# 	--include="${INCLUDE}" \
# 	torchrun \
# 	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
# 	src/train.py \
# 	--stage sft \
# 	--model_name_or_path /home/jovyan/zhubin/DATA/models/honor2_5b_patched_tokenizer/ \
# 	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_1126_neft_auls_loraplus_ep3_lr1e4_bs4/ \
# 	--resize_vocab true \
# 	--do_predict \
# 	--eval_dataset zhipu_dev \
# 	--template honor \
# 	--finetuning_type lora \
# 	--output_dir /home/jovyan/zhubin/code/LLaMA-Factory/saved_output/loraplus_honor \
# 	--cutoff_len 4069 \
# 	--max_new_tokens 512 \
# 	--do_sample false \
# 	--per_device_eval_batch_size 2 \
# 	--predict_with_generate

# attrun \
# 	--hoststr="${HOSTSTR}" \
# 	--include="${INCLUDE}" \
# 	torchrun \
# 	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
# 	src/train.py \
# 	--stage sft \
# 	--model_name_or_path /home/jovyan/cht/honor/pretrain_model/Qwen2.5-14B-Instruct/ \
# 	--resize_vocab true \
# 	--do_predict \
# 	--eval_dataset wuzhiqiang \
# 	--template qwen \
# 	--finetuning_type full \
# 	--output_dir /home/jovyan/zhubin/code/LLaMA-Factory/saved_output/wuzhiqiang_qwen14b \
# 	--cutoff_len 4069 \
# 	--max_new_tokens 512 \
# 	--do_sample false \
# 	--per_device_eval_batch_size 2 \
# 	--predict_with_generate

# attrun \
# 	--hoststr="${HOSTSTR}" \
# 	--include="${INCLUDE}" \
# 	torchrun \
# 	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
# 	src/train.py \
# 	--stage sft \
# 	--model_name_or_path /home/jovyan/cht/honor/pretrain_model/Qwen2.5-14B-Instruct/ \
# 	--resize_vocab true \
# 	--do_predict \
# 	--eval_dataset wuzhiqiang \
# 	--template qwen \
# 	--finetuning_type full \
# 	--output_dir /home/jovyan/zhubin/code/LLaMA-Factory/saved_output/wuzhiqiang \
# 	--cutoff_len 4069 \
# 	--max_new_tokens 512 \
# 	--do_sample false \
# 	--per_device_eval_batch_size 2 \
# 	--predict_with_generate

# attrun \
# 	--hoststr="${HOSTSTR}" \
# 	--include="${INCLUDE}" \
# 	torchrun \
# 	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
# 	src/train.py \
# 	--stage sft \
# 	--model_name_or_path /home/jovyan/zhubin/DATA/models/MagicLM-3B-Instruct-v0.1/ \
# 	--resize_vocab true \
# 	--do_predict \
# 	--eval_dataset wuzhiqiang \
# 	--template honor \
# 	--finetuning_type full \
# 	--output_dir /home/jovyan/zhubin/code/LLaMA-Factory/saved_output/magiclm-3b-instruct \
# 	--cutoff_len 4069 \
# 	--max_new_tokens 512 \
# 	--do_sample false \
# 	--per_device_eval_batch_size 2 \
# 	--predict_with_generate
