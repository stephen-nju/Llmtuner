import json
import os
import subprocess as sp
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence,List,Literal
from transformers import TrainingArguments, HfArgumentParser

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
@dataclass
class InputArguments:
    eval_dataset:Optional[str] = field(default="xxy_schedule_test")
    root_path: Optional[str] = field(default="")
    recursive: bool = field(default=False)


def load_dataset_info():
    datainfo=read_json("/home/jovyan/zhubin/code/LLaMA-Factory/local_data/dataset_info.json")
    return datainfo


def evaluate(input_args):
    dataset_map=load_dataset_info()
    eval_dataset_path=dataset_map[input_args.eval_dataset]["file_name"]
    ref_datas=read_json(eval_dataset_path)
    print(f"length evaluate dataset ={len(ref_datas)}")
    eval_result={}
    if input_args.recursive:
        experiment_list = [
            {   "name":path,
                "path":os.path.join(input_args.root_path,path)}
            for path in os.listdir(input_args.root_path)
            if os.path.isdir(os.path.join(input_args.root_path, path))
        ]
    else:
        experiment_list = [
            {   "name":os.path.split(input_args.root_path)[-1],
                "path":input_args.root_path
             }
        ]
    # print(experiment_list)
    success_experiments=[]
    for experiment in experiment_list:
        # 存在预测的结果数据
        path=experiment.get("path",None)
        pred_file_path=os.path.join(path,"pred/generated_predictions.jsonl")
        # print(pred_file_path)
        if os.path.exists(pred_file_path):
            pred=read_jsonl(pred_file_path)
            assert len(pred)==len(ref_datas)
            convert_pred=[]
            for p,r in zip(pred,ref_datas):
                r['result']=p["predict"]
                convert_pred.append(r)
            convert_pred_path=os.path.join(path,"pred/generated_predictions_convert.json")
            with open(convert_pred_path,"w",encoding="utf-8") as g:
                json.dump(convert_pred,g,ensure_ascii=False,indent=4)

            experiment["ref_file"]=eval_dataset_path
            experiment["pred_file"]=convert_pred_path
            success_experiments.append(experiment)
    #调用评测代码
    print(f"success_experiments={success_experiments}")
    for experiment in success_experiments:
        output=sp.run(
            f"python /home/jovyan/zhubin/code/mllm_lora_fintune/eval_script/main.py --ref_file {experiment['ref_file']} --pred_file {experiment['pred_file']}",
            shell=True,
            capture_output=True,
            encoding="utf-8",
        )
        # print(output)
        if output.returncode==0:
            eval_result[experiment["name"]]=output.stdout
        else:
            print(f"eval failure={experiment['name']}")
    with open("out.json","w",encoding="utf-8") as g:
        json.dump(eval_result,g,ensure_ascii=False,indent=4)

    for name,output in eval_result.items():
        print(f"name={name},output=={output}")

if __name__ == "__main__":

    parser = HfArgumentParser((InputArguments))
    input_args,= parser.parse_args_into_dataclasses()
    evaluate(input_args)
    # eval_dataset="/home/jovyan/zhubin/DATA/train_data/schedule/xxy_test_8垂域_test_v5_update_moretag_时间修改v2_医疗修改_v3.json"
    # root_path="/home/jovyan/zhubin/mllm_output/V1/"
    # evaluate(eval_dataset,root_path)
