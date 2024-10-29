import subprocess as sp
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence,List,Literal
from transformers import TrainingArguments, HfArgumentParser
import os
import typing
import json
from copy import deepcopy

dataset_map={}
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f_in:
        datas = json.load(f_in)
        return datas

def read_jsonl(file_path):
    datas=[]
    with open(file_path,"r",encoding="utf-8") as g:
        for line in g:
            datas.append(json.loads(line))
    return datas

def load_dataset_info():
    datainfo=read_json("/home/jovyan/zhubin/code/LLaMA-Factory/local_data/dataset_info.json")
    return datainfo

dataset_map=load_dataset_info()

@dataclass
class InputArguments:
    root_path: Optional[str] = field(default="")
    task_name: Optional[str] =field(default="")
    recursive: bool = field(default=False)
    skip_saved_checkpoint: bool=field(default=True,metadata={"help": "是否预测模型保存的中间checkpoint结果"},
)

@dataclass
class ExperimentsArguments:
    eval_dataset: Optional[str] = field(default="xxy_schedule_test")
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    )
    finetuning_type: Literal["lora", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    
    template: Optional[str] = field(
        default='qwen2_vl',
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )

    output_dir: Optional[str] = field(default=None)

    stage: Optional[str] = field(default="sft")


def parser_model_or_adapter(path):
    map = {}
    adapter_name_or_path = None
    adapter_config_name = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_config_name):
        adapter_name_or_path = path
        with open(adapter_config_name, "r", encoding="utf-8") as g:
            config = json.load(g)
        model_name_or_path = config["base_model_name_or_path"]
    else:
        # 成功保存的模型
        trainer_state_path=os.path.join(path,"trainer_state.json")
        if not os.path.exists(trainer_state_path):
            model_name_or_path=None
        else:
            model_name_or_path = path

    map["model_name_or_path"] = model_name_or_path
    if adapter_name_or_path is not None:
        map["adapter_name_or_path"] = adapter_name_or_path

    return map

def parse_experiments(root_path,task_name,recursive):
    # 读取模型路径
    experiments = []
    if recursive:
        output_dirs = [
        os.path.join(root_path,path)
        for path in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, path))
    ]
    else:
        output_dirs=[root_path]

    for output_dir in output_dirs:
        exp_config = {}
        model_or_adapter_config = parser_model_or_adapter(output_dir)
        if model_or_adapter_config["model_name_or_path"] is None:
            continue
        if not model_or_adapter_config.get("adapter_name_or_path"):
            # 如果adapter不存在，使用全量微调
            exp_config["finetuning_type"]="full"
        pred_dir = os.path.join(output_dir,task_name)
        exp_config["output_dir"] = pred_dir
        os.makedirs(pred_dir, exist_ok=True)
        exp_config.update(model_or_adapter_config)
        experiments.append(exp_config)

    return experiments


# 配置template参数


def run_preds(cmd):
    output = sp.run(cmd,env=dict(os.environ),shell=True, check=True,encoding="utf-8")
    return output


def update_experiment_args(experiment_config, default_exp_args):
    experiment_args = vars(deepcopy(default_exp_args))

    for k, v in experiment_config.items():
        experiment_args[k] = v

    return experiment_args


def main():
    # 模型预测
    parser = HfArgumentParser((InputArguments,ExperimentsArguments))
    input_args,exp_args= parser.parse_args_into_dataclasses()
    exps = parse_experiments(input_args.root_path,input_args.task_name,input_args.recursive)
    print(exps)
    for experiment in exps:
        exp_cmd_dict = update_experiment_args(experiment, exp_args)
        cmd = "./lorapred.sh "
        for k, v in exp_cmd_dict.items():
            if v is None:
                print(f"k=={k},v=={v}")
                continue
            cmd += f"--{k} {v} "
        print(cmd)
        output = run_preds(cmd)



if __name__ == "__main__":
    main()

