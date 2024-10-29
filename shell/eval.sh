export PROJECT_PATH=/home/jovyan/zhubin/code/LLaMA-Factory/
cd ${PROJECT_PATH}

# attrun \
# 	--hoststr="node2 slots=8" \
# 	torchrun \
# 	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
# 	src/train.py \
# 	--stage sft \
# 	--model_name_or_path /home/jovyan/zhubin/DATA/models/honor2_5b_patched_tokenizer/ \
# 	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_1016_callsumv3_ep3_lr1e4_bs4/ \
# 	--resize_vocab true \
# 	--do_predict \
# 	--eval_dataset union_conversations_dev_v2 \
# 	--template honor \
# 	--finetuning_type lora \
# 	--output_dir /home/jovyan/zhubin/code/LLaMA-Factory/saved_output/lora/ \
# 	--cutoff_len 2048 \
# 	--max_new_tokens 512 \
# 	--do_sample false \
# 	--per_device_eval_batch_size 4 \
# 	--predict_with_generate

attrun \
	--hoststr="node2 slots=8" \
	torchrun \
	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
	src/train.py \
	--stage sft \
	--model_name_or_path /home/jovyan/zhubin/DATA/models/MagicLM-3B-OmniQuant/ \
	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_quant_1017_lora_rank32_lora_alpha1_lr1e4/ \
	--resize_vocab true \
	--do_predict \
	--eval_dataset union_conversations_dev_v2 \
	--template honor \
	--finetuning_type lora \
	--output_dir /home/jovyan/zhubin/code/LLaMA-Factory/saved_output/quant_lora/ \
	--cutoff_len 2048 \
	--max_new_tokens 512 \
	--do_sample false \
	--per_device_eval_batch_size 4 \
	--predict_with_generate
