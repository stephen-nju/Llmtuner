#!/usr/bin/bash
Usage() {
	cat <<EOF
Usage: train magiclm nano
-m --model_name_or_path     base model name or path
-n  --name 				    runing experiment name 
-h  --help                  display help
-e  --epoch                 num train epochs
-l  --lr					learning rate
-b  --bs					train batch size
-d  --dataset               train dataset
EOF
}

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=160
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_ALGO=Ring
export PROJECT_PATH=/opt/nas/p/zhubin/code/Llmtuner/

cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=${PROJECT_PATH}/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=${PROJECT_PATH}/config/deepspeed/zero_stage2_config.json
if [[ ! $WANDB_PROJECT ]]; then
	export WANDB_PROJECT="LlmTuner"
fi

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export name
export dataset
export stage
export lr=2e-5
export epochs=3
export template=honor
export finetuning_type=lora
export batch_size=4
export include
export gradient_accumulation_steps=1
export model_name_or_path=/opt/nas/p/zhubin/DATA/models/honor2_5b_patched_tokenizer/
export resize_vocab=false
export save_strategy=steps
export save_steps=5000
export save_total_limit=10
export do_train=false
export do_eval=false
export logging_steps=5
export cutoff_len=2048
export warmup_ratio=0.03
#lora fintuning
export lora_rank=32
export lora_target=all
export lora_alpha=$((2 * ${lora_rank}))

# dpo parameter
export pref_loss=simpo
export pref_beta=0.1
export simpo_gamma=0.5
export ddp_timeout=180000000
# neftune
export neftune_noise_alpha

## eval
export eval_dataset
export eval_steps
export eval_strategy=no

options=$(getopt -l "help,do_train,do_eval,stage:,model_name_or_path:,name:,epochs:,lr:,batch_size:,template:,\
finetuning_type:,dataset:,cutoff_len:,include:,resize_vocab:,gradient_accumulation_steps:,eval_dataset:,eval_strategy:,eval_steps:,\
pref_loss:,pref_beta:,simpo_gamma:,ddp_timeout:,neftune_noise_alpha:,\
lora_rank:,lora_alpha:,lora_target:,\
save_steps:,save_total_limit:,logging_steps:,warmup_ratio:,save_strategy:" -o "e:l:d:b:n:m:g:" -a -- "$@")

eval set -- "$options"
# echo "$options"

while true; do
	case "$1" in
	-h | --help)
		Usage
		exit 0
		;;
	--do_train)
		do_train=true
		;;
	--do_eval)
		do_eval=true
		;;
	-m | --model_name_or_path)
		shift
		model_name_or_path="$1"
		;;
	--stage)
		shift
		stage="$1"
		;;
	-e | --epochs)
		shift
		epochs="$1"
		;;
	-l | --lr)
		shift
		lr="$1"
		;;
	-b | --batch_size)
		shift
		batch_size="$1"
		;;
	-t | --template)
		shift
		template="$1"
		;;
	--finetuning_type)
		shift
		finetuning_type="$1"
		;;
	-g | --gradient_accumulation_steps)
		shift
		gradient_accumulation_steps="$1"
		;;
	-d | --dataset)
		shift
		dataset="$1"
		;;
	-n | --name)
		shift
		name="$1"
		;;
	--include)
		shift
		include="$1"
		;;
	--save_total_limit)
		shift
		save_total_limit="$1"
		;;
	--save_strategy)
		shift
		save_strategy="$1"
		;;
	--save_steps)
		shift
		save_steps="$1"
		;;
	--logging_steps)
		shift
		logging_steps="$1"
		;;
	--cutoff_len)
		shift
		cutoff_len="$1"
		;;
	--warmup_ratio)
		shift
		warmup_ratio="$1"
		;;
	--pref_loss)
		shift
		pref_loss="$1"
		;;
	--pref_beta)
		shift
		pref_beta="$1"
		;;
	--simpo_gamma)
		shift
		simpo_gamma="$1"
		;;
	--neftune_noise_alpha)
		shift
		neftune_noise_alpha="$1"
		;;
	--resize_vocab)
		shift
		resize_vocab="$1"
		;;
	--lora_rank)
		shift
		lora_rank="$1"
		;;
	--lora_alpha)
		shift
		lora_alpha="$1"
		;;
	--lora_target)
		shift
		lora_target="$1"
		;;
	--eval_dataset)
		shift
		eval_dataset="$1"
		;;
	--eval_strategy)
		shift
		eval_strategy="$1"
		;;
	--eval_steps)
		shift
		eval_steps="$1"
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
if [[ ! $neftune_noise_alpha ]]; then
	echo "optinonal paramas neftune_noise_alpha is null"
else
	optional_params+=(--neftune_noise_alpha ${neftune_noise_alpha})
fi

if [[ $do_eval=true ]]; then
	optional_params+=(--eval_dataset ${eval_dataset})
	optional_params+=(--eval_steps ${eval_steps})
fi

deepspeed_params=()
if [[ -n $include ]]; then
	deepspeed_params+=(--include $include)
fi

export OUTPUT_DIR=/opt/nas/p/zhubin/saved_checkpoint/$name
export WANDB_DIR=$OUTPUT_DIR/logs
export HOSTFILE=/opt/nas/p/zhubin/code/Llmtuner/config/hostfile
mkdir -p ${OUTPUT_DIR}
mkdir -p ${WANDB_DIR}

export WANDB_MODE=offline
echo "wandb dir=$WANDB_DIR"
deepspeed --hostfile=${HOSTFILE} --master_port=${MASTER_PORT} "${deepspeed_params[@]}" --no_local_rank \
	src/train.py \
	--deepspeed ${DS_CONFIG_STAGE_2} \
	--stage ${stage} \
	--pref_beta ${pref_beta} \
	--pref_loss ${pref_loss} \
	--simpo_gamma ${simpo_gamma} \
	--template ${template} \
	--do_train ${do_train} \
	--do_eval ${do_eval} \
	--eval_strategy ${eval_strategy} \
	--model_name_or_path $model_name_or_path \
	--resize_vocab ${resize_vocab} \
	--use_fast_tokenizer false \
	--report_to wandb \
	--overwrite_output_dir \
	--overwrite_cache \
	--dataset ${dataset} \
	--cutoff_len ${cutoff_len} \
	--output_dir ${OUTPUT_DIR} \
	--num_train_epochs ${epochs} \
	--overwrite_cache \
	--finetuning_type ${finetuning_type} \
	--lora_rank ${lora_rank} \
	--lora_target ${lora_target} \
	--warmup_ratio ${warmup_ratio} \
	--logging_steps ${logging_steps} \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size ${batch_size} \
	--per_device_eval_batch_size ${batch_size} \
	--gradient_accumulation_steps ${gradient_accumulation_steps} \
	--preprocessing_num_workers 16 \
	--save_strategy ${save_strategy} \
	--save_steps ${save_steps} \
	--save_total_limit ${save_total_limit} \
	--learning_rate ${lr} \
	--ddp_timeout ${ddp_timeout} \
	--bf16 true \
	"${optional_params[@]}" \
	2>&1 | tee ${OUTPUT_DIR}/train.log
