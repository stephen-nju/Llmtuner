from sklearn.model_selection import ParameterGrid
import os
import copy


# 不同lora的超参数
def grid_search(param_map):
    return list(ParameterGrid(param_map))


def merge_exp_parameters(exp_parameters, train_parameters):
    # print(f"exp_parameters={exp_parameters},train_parameters={train_parameters}")
    merge_parameters = {}
    for k, v in train_parameters.items():
        # 将嵌套的参数展开
        if isinstance(v, dict):
            for subk, subv in v.items():
                merge_parameters[subk] = subv
        else:
            merge_parameters[k] = v

    for k, v in exp_parameters.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                merge_parameters[subk] = subv
        else:
            merge_parameters[k] = v
    return merge_parameters


def make_experiments(exp_map):
    search_exp_map = {}
    for k, v in exp_map.items():
        search_exp_map[k] = grid_search(v)
    return search_exp_map


def parser_model_name(s):
    model_name = os.path.split(s)[-1]
    return model_name


def update_experiment_name(experiment_config):
    new_experiment_config = copy.deepcopy(experiment_config)
    n = ""
    for k, v in experiment_config.items():
        if k == "name":
            continue
        n += f"_{k.lstrip('--')}{v}"

    new_experiment_config["name"] = f"{experiment_config['name']}={n}"

    return new_experiment_config


# def make_lora_experiments_name(name, parameters):
#     # LORA实验的命名规范
#     p_map = {}
#     for k, v in parameters.items():
#         k = k.lstrip("--")
#         p_map[k] = v
#     # 提取重要参数
#     dataset = p_map["dataset"]
#     model_name = parser_model_name(p_map["model_name_or_path"])
#     lr = p_map["lr"]
#     exp_name = None
#     if name == "lora" or name == "dora":
#         lora_rank = p_map["lora_rank"]
#         lora_target = p_map["lora_target"]
#         exp_name = f"{name}_{dataset}_{model_name}_lk{lora_rank}_lt{lora_target}_lr{lr}"
#     elif name == "loraplus":
#         lora_rank = p_map["lora_rank"]
#         lora_target = p_map["lora_target"]
#         loraplus_lr_ratio = p_map["loraplus_lr_ratio"]
#         loraplus_lr_embedding = p_map["loraplus_lr_embedding"]
#         exp_name = f"{name}_{dataset}_{model_name}_lk{lora_rank}_lt{lora_target}_llr{loraplus_lr_ratio}_lle{loraplus_lr_embedding}_lr{lr}"
#     elif name == "pissa":
#         lora_rank = p_map["lora_rank"]
#         lora_target = p_map["lora_target"]
#         pissa_init = p_map["pissa_init"]
#         pissa_iter = p_map["pissa_iter"]
#         pissa_convert = p_map["pissa_convet"]
#         exp_name = f"{name}_{dataset}_{model_name}_lk{lora_rank}_lt{lora_target}_pi{pissa_init}_pit{pissa_iter}_pc{pissa_convert}_lr{lr}"
#     elif name == "galore":
#         galore_target = p_map["galore_target"]
#         galore_update_interval = p_map["galore_update_interval"]
#         gs = p_map["galore_scale"]
#         gpt = p_map["galore_proj_type"]
#         exp_name = f"{name}_{dataset}_{model_name}_gt{galore_target}_gut{galore_update_interval}_gs{gs}_gpt{gpt}_lr{lr}"
#     return exp_name


def update_train_name(config):
    new_config = copy.deepcopy(config)
    model_name = parser_model_name(config["--model_name_or_path"])
    learing_rate = config["--lr"]
    bs = config["--batch_size"]
    max_length = config["--cutoff_len"]
    dataset = config["--dataset"]
    exp_name = config["name"]
    freeze_vision_tower = config['--freeze_vision_tower']
    new_config["name"] = (
        f"{model_name}_{dataset}_{exp_name}_freeze_vision{freeze_vision_tower}_maxseq{max_length}_lr{learing_rate}"
    )

    return new_config


def build_run_cmd(experiments):
    print(experiments)
    es = []
    for exp in experiments:
        name = exp.pop("name")
        cmd = f"./mllmtrain.sh --name '{name}' "
        if exp.get("--use_galore", False):
            cmd = f"./mllmtrain_torchrun.sh --name {name} "
        for k, v in exp.items():
            if k == "--lora_target":
                cmd += f"{k} '{v}' "
                print(cmd)
            else:
                cmd += f"{k} {v} "

        es.append(cmd)

    with open("run.sh", "w", encoding="utf-8") as g:
        for e in es:
            g.write(e + "\n")
            g.write("wait\n")


def main():
    experiments_list = [
        # {
        #     "name": ["lora"],
        #     "--finetuning_type": ["lora"],
        #     "--lora_rank": [32],
        #     "--lora_target": ["all"],
        # },
        # qwen2_vl训练vision_tower+connector+LLM的lora配置
        {
            "name": ["lora"],
            "--finetuning_type": ["lora"],
            "--lora_rank": [32],
            "--freeze_vision_tower":[False],
        "--lora_target": ["qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2"],
        },
        

        # {
        #     "name": ["dora"],
        #     "--use_dora": [True],
        #     "--lora_rank": [32],
        #     "--lora_target": ["all"],
        #     "--lora_alpha": [64],
        #     "--lora_dropout": [0.1],
        # },
        # {
        #     "name": ["loraplus"],
        #     "--finetuning_type": ["lora"],
        #     "--loraplus_lr_ratio": [32],
        #     "--loraplus_lr_embedding": [1e-6],
        #     "--lora_rank": [32],
        #     "--lora_target": ["all"],
        #     "--lora_alpha": [64],
        #     "--lora_dropout": [0.1],
        # },
        # {
        #     "name": ["pissa"],
        #     "--finetuning_type": ["lora"],
        #     "--pissa_init": [False],
        #     "--pissa_iter": [16],
        #     "--lora_rank": [32],
        #     "--lora_target": ["all"],
        #     "--pissa_convert": [False],
        # },
        # {
        #     "name": ["galore"],
        #     "--finetuning_type": ["full"],
        #     "--use_galore": [True],
        #     "--galore_target": ["all"],
        #     "--galore_update_interval": [200],
        #     "--galore_scale": [0.25],
        #     "--galore_proj_type": ["std"],
        #     "--galore_layerwise": [False],
        # },
    ]
    # 训练参数
    train_params = {
        "--do_train": [True],
        "--lr": [1e-4],
        "--stage": ["sft"],
        "--epochs": [3],
        "--batch_size": [4],
        "--include": ["node11"],
        "--cutoff_len": [2048],
        "--gradient_accumulation_steps": [1],
        "--preprocessing_num_workers": [16],
        "--models": [
            {
                "--model_name_or_path": "/home/jovyan/zhubin/DATA/models/Qwen2-VL-2B-Instruct",
                "--template": "qwen2_vl",
            },
            {
                "--model_name_or_path": "/home/jovyan/zhubin/DATA/models/Qwen2-VL-7B-Instruct",
                "--template": "qwen2_vl",
            },
        ],
        # "--dataset":["xxy_schedule_test"]
        "--task": [
            {
                "--task_name": "rym_ner",
                "--dataset": "rym_ner_train",
            },

            {
                "--task_name": "common_rym_ner",
                "--dataset": "common_rym_ner_train"
            },

            {
                "--task_name": "yjzl",
                "--dataset": "yjzl_train"
            },

            {
                "--task_name": "thzy",
                "--dataset": "thzy_train"
            },
            {
                "--task_name": "rccq",
                "--dataset": "rccq_train"
            },
            {"--task_name": "gjccq",
             "--dataset": "gjccq_train"},
            {

                "--task_name": "tzjx",
                "--dataset": "tzjx_train"
            }
        ]
    }

    # 将实验分为不同的部分的组合，通用的训练参数+不同的方案
    train_configs = grid_search(train_params)

    train_experiments = []
    for experiments in experiments_list:
        experiments_configs = grid_search(experiments)
        for experiment_config in experiments_configs:
            new_experiment_config = update_experiment_name(experiment_config)
            for train_config in train_configs:
                merge_config = merge_exp_parameters(
                    new_experiment_config, train_config)
                train_experiments.append(update_train_name(merge_config))

    build_run_cmd(train_experiments)


if __name__ == "__main__":
    main()
