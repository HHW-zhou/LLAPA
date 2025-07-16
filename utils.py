import pandas as pd
import logging
import time
import os
import torch
import numpy as np
import random
import argparse
from safetensors import safe_open
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from config.configs import ROOT_DIR, QUESTION_PREFIX, ANSWER_PREFIX, LLM_CONFIG_PATH

def evaluat_metrics2(preds, labels):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    N = len(preds)
    C = len(preds[0])
    
    
    # pre_y = (torch.sigmoid(preds) > 0.5).numpy()
    # truth_y = label.numpy()
    # N, C = pre_y.shape

    for i in range(N):
        pred = (torch.sigmoid( preds[i]) > 0.5).numpy()
        label = labels[i]
        
        for j in range(C):
            if pred[j] == label[j]:
                if label[j] == 1:
                    TP += 1
                else:
                    TN += 1
            elif label[j] == 1:
                FN += 1
            elif label[j] == 0:
                FP += 1

        # Accuracy = (TP + TN) / (N*C + 1e-10)
        Precision = TP / (TP + FP + 1e-10)
        Recall = TP / (TP + FN + 1e-10)
        F1_score = 2 * Precision * Recall / (Precision + Recall + 1e-10)

    return F1_score

# Calculation of evaluation metrics
def evaluat_metrics(preds, label):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # pre_y = preds.numpy()
    # truth_y = label.numpy()
    # N, C = pre_y.shape

    pre_y = preds
    truth_y = label
    N = len(pre_y)
    C = len(pre_y[0])

    for i in range(N):
        for j in range(C):
            if pre_y[i][j] == truth_y[i][j]:
                if truth_y[i][j] == 1:
                    TP += 1
                else:
                    TN += 1
            elif truth_y[i][j] == 1:
                FN += 1
            elif truth_y[i][j] == 0:
                FP += 1

        # Accuracy = (TP + TN) / (N*C + 1e-10)
        Precision = TP / (TP + FP + 1e-10)
        Recall = TP / (TP + FN + 1e-10)
        F1_score = 2 * Precision * Recall / (Precision + Recall + 1e-10)

    return F1_score

def isnull(var):
    if isinstance(var, float):
        return pd.isnull(var)
    elif isinstance(var, str):
        return var.strip() == 'nan'
    else:
        return False
    
def notnull(var):
    if isinstance(var, float):
        return not pd.isnull(var)
    elif isinstance(var, str):
        return not var.strip() == 'nan'
    else:
        return False
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setLogger():
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logfile = f'{log_dir}/{rq}.log'
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)

def load_model_weights(model, args):
    weight_path =  os.path.join(ROOT_DIR, args.model_dir)

    ###################################### load pp ###############################################
    pp_path = os.path.join(weight_path, 'pp/model.safetensors')
    if os.path.exists(pp_path) and args.load_pp == 1:
        print(f"Load PP weights from {pp_path}.")
        with safe_open(pp_path, framework="pt") as f:
            projector_weights = {key: f.get_tensor(key) for key in f.keys()}

        model.protein_projector.load_state_dict(projector_weights)

    ###################################### load peft ###############################################
    peft_path = os.path.join(weight_path, 'peft/model.safetensors')
    if os.path.exists(peft_path) and args.load_peft == 1:
        print(f"Load PEFT weights from {peft_path}.")
        with safe_open(peft_path, framework="pt") as f:
            peft_weights = {key: f.get_tensor(key) for key in f.keys()}

        set_peft_model_state_dict(model.llm, peft_weights)

    ###################################### load pe ###############################################
    pe_path1 = os.path.join(weight_path, 'pe/model-00001-of-00002.safetensors')
    pe_path2 = os.path.join(weight_path, 'pe/model-00002-of-00002.safetensors')

    if os.path.exists(pe_path1) and os.path.exists(pe_path2) and args.load_pe == 1:
        print(f"Load PE weights from {pe_path1}.")
        with safe_open(pe_path1, framework="pt") as f:
            pe_weights1 = {key: f.get_tensor(key) for key in f.keys()}

        with safe_open(pe_path2, framework="pt") as f:
            pe_weights2 = {key: f.get_tensor(key) for key in f.keys()}

        pe_weights1.update(pe_weights2)

        model.protein_encoder.load_state_dict(pe_weights1)
    ###################################### load ge ###############################################
    ge_path = os.path.join(weight_path, 'ge/model.safetensors')
    if os.path.exists(ge_path) and args.load_ge == 1:
        print(f"Load graph encoder weights from {ge_path}.")
        with safe_open(ge_path, framework="pt") as f:
            ge_weights = {key: f.get_tensor(key) for key in f.keys()}

        model.graph_encoder.load_state_dict(ge_weights)


def set_trainable_parameters(model, args):
    update_layer_list = []
    if args.update_projector == 1:
        update_layer_list.append("projector")
    if args.update_lora == 1:
        update_layer_list.append("lora")
    if args.update_ge == 1:
        update_layer_list.append("graph_encoder")
    if args.update_pe == 1:
        update_layer_list.append("protein_encoder")

    for name, param in model.named_parameters():
        param.requires_grad = False
        
        for layer_name in update_layer_list:
            if layer_name in name:
                param.requires_grad = True
                print(name)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_parse():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--backbone", type=str, default='llama2-7b')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--gpu_num", type=int, default=4)                  # gpu_num * batch_size * accumulate_step = 256
    parser.add_argument("--batch_size", type=int, default=1)                # 40G内存的batch设置为2/80的设置为4
    parser.add_argument("--accumulate_step", type=int, default=16)
    parser.add_argument("--step_to_save", type=int, default=1000)
    parser.add_argument("--samples_to_update_gradient", type=int, default=512)      # 每多少个样本更新一次参数
    parser.add_argument("--output_dir", type=str, default="./model_weights/stage1")
    parser.add_argument("--learning_rate", type=float, default=4e-5)

    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--report_to", type=str, default="none", choices=['wandb','none'])
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--lora_rank", type=int, default=256)

    # for evaluation
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--return_num", type=int, default=5)
    parser.add_argument('--local_rank', type=int, default=-1)

    parser.add_argument("--task", type=str, default="SHS27k")
    parser.add_argument('--ckpt', type=int, default=300)
    parser.add_argument("--model_dir", type=str, default="stage2_llama_SHS27k")
    
    parser.add_argument("--load_pp", type=int, default=1)
    parser.add_argument("--load_peft", type=int, default=1)
    parser.add_argument("--load_pe", type=int, default=1)
    parser.add_argument("--load_ge", type=int, default=1)
    
    
    parser.add_argument("--update_projector", type=int, default=0)
    parser.add_argument("--update_lora", type=int, default=0)
    parser.add_argument("--update_ge", type=int, default=0)
    parser.add_argument("--update_pe", type=int, default=0)
    
    parser.add_argument("--p_mode", type=str, default="p", choices=['r','p']) # residue, protein
    parser.add_argument("--stage", type=str, default="pretrain", choices=['pretrain','ft'])

    args = parser.parse_args()

    args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # args.gpu_num = torch.cuda.device_count()

    if torch.cuda.is_available():
        args.gradient_accumulation_steps = int(args.samples_to_update_gradient / args.gpu_num / args.batch_size)
    else:
        args.gradient_accumulation_steps = int(args.samples_to_update_gradient / args.batch_size)

    # print("==============================> ", args.samples_to_update_gradient, args.gpu_num, args.batch_size)
    # print("==============================> ", args.gradient_accumulation_steps)

    return args