export data
export do_eval=false

if [[ "${do_eval}" = true ]]; then
	echo "iiiiiiiiiiiiiiiii"
fi

export r=2
export b=$((2 * ${r}))

echo ${b}
# options=$(getopt -l "help,do_train,do_eval,stage:,model_name_or_path:,name:,epochs:,lr:,batch_size:,template:,\
# finetuning_type:,dataset:,cutoff_len:,include:,resize_vocab:,gradient_accumulation_steps:,eval_dataset:,eval_strategy:,eval_steps:,\
# pref_loss:,pref_beta:,simpo_gamma:,ddp_timeout:,neftune_noise_alpha:,lora_alpha:,additional_target:,shift_attn:,\
# lora_rank:,lora_target:,lora_dropout:,use_dora:,loraplus_lr_embedding:,pissa_init:,pissa_iter:,pissa_convert:,use_galore:,\
# galore_target:,galore_rank:,galore_update_interval:,preprocessing_num_workers:,\
# galore_scale:,galore_proj_type:,galore_layerwise:,loraplus_lr_ratio:,\
# save_steps:,save_total_limit:,logging_steps:,warmup_ratio:,save_strategy:" -o "e:l:d:b:n:m:g:" -a -- "$@")

# eval set -- "$options"
# echo "$options"

export a
if [[ -n ${a} ]]; then
	echo "null"
fi
