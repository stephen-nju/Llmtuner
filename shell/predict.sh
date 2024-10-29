PROJECT_PATH=/home/jovyan/zhubin/code/LLaMA-Factory/
cd ${PROJECT_PATH}

CUDA_VISIBLE_DEVICES=0 python src/train.py \
	--stage sft \
	--finetuning_type lora \
	--eval_dataset xxy_schedule_test \
	--overwrite_cache \
	--model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-2B-Instruct/ \
	--adapter_name_or_path /home/jovyan/zhubin/mllm_output/models/lora_xxy_schedule_Qwen2-VL-2B-Instruct_lk32_ltall_lr2e-05/ \
	--do_predict \
	--predict_with_generate \
	--template qwen2_vl \
	--do_sample false \
	--output_dir /home/jovyan/zhubin/code/LLaMA-Factory/saved_output/ \
	--per_device_eval_batch_size 4
