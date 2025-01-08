./llmtrain.sh --do_train --stage sft --name=0108_glm4_edg_acfvcaulsd_markdown_ep2_lr2e4_bs4 --model_name_or_path /opt/nas/n/zhubin/DATA/models/THUDM/glm-edge-4b-chat --template glm4 \
	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,firefly_summary_part,vcsum_headlines,csds_dialogue,union_conversations_v4_norm_markdown,liantong_conversations_v1_markdown,samsum_chinese_markdown,dialogsum_chinese_markdown \
	--finetuning_type lora --lora_target all --batch_size 4 --gradient_accumulation_steps 8 --cutoff_len 4096 --epochs 2 --lr=2e-4 --save_strategy=steps --save_steps=500 --save_total_limit=100 \
	--eval_dataset=union_conversations_v5_dev_markdown --neftune_noise_alpha=5 --eval_strategy=steps --eval_steps=500 --warmup_ratio=0.02
