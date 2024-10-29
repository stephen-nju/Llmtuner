Usage() {
	cat <<EOF
Usage: predict lora experiments
EOF
}

PROJECT_PATH=/home/jovyan/zhubin/code/LLaMA-Factory/
cd ${PROJECT_PATH}
wandb offline

export include="node4"
export stage=sft
export finetuning_type=lora
export eval_dataset=xxy_schedule_test
export model_name_or_path
export adapter_name_or_path
export output_dir
export template="qwen2_vl"

options=$(getopt -l "help,stage:,include:,model_name_or_path:,adapter_name_or_path:,eval_dataset:,finetuning_type:,template:,output_dir:," -o "m:" -a -- "$@")

eval set -- "$options"

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
	--include)
		shift
		include="$1"
		;;
	--adapter_name_or_path)
		shift
		adapter_name_or_path="$1"
		;;
	--eval_dataset)
		shift
		eval_dataset="$1"
		;;
	--finetuning_type)
		shift
		finetuning_type="$1"
		;;
	--output_dir)
		shift
		output_dir="$1"
		;;
	--template)
		shift
		template="$1"
		;;
	--)
		shift
		break
		;;
	esac
	shift
done

optional_params=()
if [[ -n ${adapter_name_or_path} ]]; then
	optional_params+=(--adapter_name_or_path ${adapter_name_or_path})
fi

attrun \
	--hoststr="${include} slots=8" \
	torchrun \
	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
	src/train.py \
	--stage ${stage} \
	--finetuning_type ${finetuning_type} \
	--resize_vocab \
	--eval_dataset ${eval_dataset} \
	--overwrite_cache \
	--do_predict \
	--predict_with_generate \
	--template ${template} \
	--do_sample false \
	--per_device_eval_batch_size 4 \
	--model_name_or_path ${model_name_or_path} \
	--output_dir ${output_dir} \
	"${optional_params[@]}"
