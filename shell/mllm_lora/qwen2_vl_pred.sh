PROJECT_PATH=/home/jovyan/zhubin/code/LLaMA-Factory/
cd ${PROJECT_PATH}
wandb offline

CUDA_VISIBLE_DEVICES=0 python src/train.py \
	--stage sft \
	--finetuning_type lora \
	--resize_vocab \
	--eval_dataset xxy_schedule_test \
	--overwrite_cache \
	--model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-2B-Instruct/ \
	--adapter_name_or_path /home/jovyan/zhubin/mllm_output/models/lora_xxy_schedule_Qwen2-VL-2B-Instruct_lk32_ltall_lr2e-05/ \
	--do_predict \
	--predict_with_generate \
	--template qwen2_vl \
	--do_sample false \
	--output_dir /home/jovyan/zhubin/mllm_output/models/lora_xxy_schedule_Qwen2-VL-2B-Instruct_lk32_ltall_lr2e-05/pred \
	--per_device_eval_batch_size 4
# CUDA_VISIBLE_DEVICES=0 python src/train.py \
# --eval_dataset xxy_schedule_test --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-2B-Instruct --adapter_name_or_path /home/jovyan/zhubin/mllm_output/models/loraplus_xxy_schedule_Qwen2-VL-2B-Instruct_lk32_ltall_llr32_lle1e-06_lr2e-05 --finetuning_type lora --template qwen2_vl --output_dir /home/jovyan/zhubin/mllm_output/models/loraplus_xxy_schedule_Qwen2-VL-2B-Instruct_lk32_ltall_llr32_lle1e-06_lr2e-05/pred --do_predict True --predict_with_generate True --overwrite_cache True --stage sft --resize_vocab True --do_sample False