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

export PROJECT_PATH=/home/jovyan/zhubin/code/LLaMA-Factory/
# export PYTHONPATH=$PYTHONPATH
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}:$PYTHONPATH
export DS_CONFIG_STAGE_3=${PROJECT_PATH}/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=${PROJECT_PATH}/config/deepspeed/zero_stage2_config.json
export WANDB_PROJECT="MLLM_LORA"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export name
export task_name
export dataset
export stage
export lr=2e-5
export epochs=3
export template=qwen2_vl
export finetuning_type=lora
export batch_size=4
export include="node4"
export gradient_accumulation_steps=1
export model_name_or_path
export resize_vocab=true
export save_strategy=steps
export save_steps=500
export save_total_limit=2
export do_train=false
export do_eval=false
export logging_steps=5
export cutoff_len=2048
export warmup_ratio=0.01
#lora finetuning
export lora_rank=32
export lora_target=all
export lora_dropout=0
export additional_target
export lora_alpha
#DORA finetuning
export use_dora=false
#LoRA+
export loraplus_lr_ratio
export loraplus_lr_embedding=1e-6

#pissa
export pissa_init=false
export pissa_iter=16
export pissa_convert=false

#longlora
export shift_attn

#galore
export use_galore=false
export galore_target=all
export galore_rank=16
export galore_update_interval=200
export galore_scale=0.25
export galore_proj_type=std
export galore_layerwise=false
export preprocessing_num_workers=16

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

export freeze_vision_tower=true

options=$(getopt -l "help,do_train,do_eval,stage:,model_name_or_path:,name:,task_name:,epochs:,lr:,batch_size:,template:,\
finetuning_type:,dataset:,cutoff_len:,include:,resize_vocab:,gradient_accumulation_steps:,eval_dataset:,eval_strategy:,eval_steps:,\
pref_loss:,pref_beta:,simpo_gamma:,ddp_timeout:,neftune_noise_alpha:,lora_alpha:,additional_target:,shift_attn:,\
lora_rank:,lora_target:,lora_dropout:,use_dora:,loraplus_lr_embedding:,pissa_init:,pissa_iter:,pissa_convert:,use_galore:,\
galore_target:,galore_rank:,galore_update_interval:,preprocessing_num_workers:,freeze_vision_tower:,\
galore_scale:,galore_proj_type:,galore_layerwise:,loraplus_lr_ratio:,\
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
	--task_name)
		shift
		task_name="$1"
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
	--lora_target)
		shift
		lora_target="$1"
		;;
	--additional_target)
		shift
		additional_target="$1"
		;;
	--lora_alpha)
		shift
		lora_alpha="$1"
		;;
	--lora_dropout)
		shift
		lora_dropout="$1"
		;;
	--use_dora)
		shift
		use_dora="$1"
		;;
	--shift_attn)
		shift
		shift_attn="$1"
		;;
	--loraplus_lr_ratio)
		shift
		loraplus_lr_ratio="$1"
		;;
	--loraplus_lr_embedding)
		shift
		loraplus_lr_embedding="$1"
		;;
	--pissa_init)
		shift
		pissa_init="$1"
		;;
	--pissa_iter)
		shift
		pissa_iter="$1"
		;;
	--pissa_convert)
		shift
		pissa_convert="$1"
		;;
	--use_galore)
		shift
		use_galore="$1"
		;;
	--galore_target)
		shift
		galore_target="$1"
		;;
	--galore_rank)
		shift
		galore_rank="$1"
		;;
	--galore_update_interval)
		shift
		galore_update_interval="$1"
		;;
	--galore_scale)
		shift
		galore_scale="$1"
		;;
	--galore_proj_type)
		shift
		galore_proj_type="$1"
		;;
	--galore_layerwise)
		shift
		galore_layerwise="$1"
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
	--preprocessing_num_workers)
		shift
		preprocessing_num_workers="$1"
		;;
	--freeze_vision_tower)
		shift
		freeze_vision_tower="$1"
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
# if [[ ! $neftune_noise_alpha ]]; then
# 	echo "optinonal paramas neftune_noise_alpha is null"
# else
# 	optional_params+=(--neftune_noise_alpha ${neftune_noise_alpha})
# fi

if [[ -n $neftune_noise_alpha ]]; then
	optional_params+=(--neftune_noise_alpha ${neftune_noise_alpha})
fi

if [[ -n $shift_attn ]]; then
	optional_params+=(--shift_attn ${shift_attn})
fi

if [[ -n $lora_alpha ]]; then
	optional_params+=(--lora_alpha ${lora_alpha})
fi

if [[ -n $additional_target ]]; then
	optional_params+=(--additional_target ${additional_target})
fi

## 注意判断语句的空格
if [[ $do_eval = true ]]; then
	optional_params+=(--eval_dataset ${eval_dataset})
	optional_params+=(--eval_steps ${eval_steps})
fi

echo "optional prams is ${optional_params}"

if [[ -n ${task_name} ]]; then
	export OUTPUT_DIR=/home/jovyan/zhubin/mllm_output/${task_name}/$name
else
	export OUTPUT_DIR=/home/jovyan/zhubin/mllm_output/${name}
fi

export WANDB_DIR=$OUTPUT_DIR/logs
export HOSTFILE=/home/jovyan/zhubin/code/LLaMA-Factory/config/hostfile
mkdir -p ${OUTPUT_DIR}
mkdir -p ${WANDB_DIR}

echo "wandb dir=$WANDB_DIR"
wandb offline
export WANDB_MODE=offline
deepspeed --hostfile=${HOSTFILE} --include=${include} --master_port=${MASTER_PORT} --no_local_rank \
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
	--resize_vocab true \
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
	--lora_dropout ${lora_dropout} \
	--use_dora ${use_dora} \
	--loraplus_lr_embedding ${loraplus_lr_embedding} \
	--pissa_init ${pissa_init} \
	--pissa_iter ${pissa_iter} \
	--pissa_convert ${pissa_convert} \
	--use_galore ${use_galore} \
	--galore_target ${galore_target} \
	--galore_rank ${galore_rank} \
	--galore_update_interval ${galore_update_interval} \
	--galore_scale ${galore_scale} \
	--galore_proj_type ${galore_proj_type} \
	--galore_layerwise ${galore_layerwise} \
	--warmup_ratio ${warmup_ratio} \
	--logging_steps ${logging_steps} \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size ${batch_size} \
	--per_device_eval_batch_size ${batch_size} \
	--gradient_accumulation_steps ${gradient_accumulation_steps} \
	--preprocessing_num_workers ${preprocessing_num_workers} \
	--save_strategy ${save_strategy} \
	--save_steps ${save_steps} \
	--save_total_limit ${save_total_limit} \
	--learning_rate ${lr} \
	--freeze_vision_tower ${freeze_vision_tower} \
	--ddp_timeout ${ddp_timeout} \
	--bf16 true \
	"${optional_params[@]}" \
	2>&1 | tee ${OUTPUT_DIR}/train.log
