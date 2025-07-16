import sys
import os
sys.path.append(os.path.abspath('../'))
print(sys.path)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第一个GPU
from models.configuration_llapa import LlapaConfig
from models.llapa import LlapaForConditionalGeneration

import torch
from transformers.models.auto import AutoModel, AutoTokenizer
from transformers import LlamaForCausalLM
from peft import get_peft_model, LoraConfig

from utils import load_model_weights

import pandas as pd
from utils import get_parse, setup_seed
from load_data import load_data, load_UPPIN
from tqdm import tqdm
import random
####################################################################################################################################

from config.configs import ROOT_DIR, QUESTION_PREFIX, ANSWER_PREFIX, LLM_CONFIG_PATH

def get_model(args):
    device = 'cuda:0'
    # ---------------------------------  加载模型  --------------------------------------
    config = LlapaConfig.from_pretrained(LLM_CONFIG_PATH)
    model = LlapaForConditionalGeneration(config)
    # ---------------------------------  加载llm  --------------------------------------
    llm_tokenizer = AutoTokenizer.from_pretrained(config.text_config._name_or_path, truncation_side="right")
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.add_special_tokens({"pad_token":"[PAD]"})
    llm_tokenizer.padding_side = 'right'
    
    llm_tokenizer.add_tokens(['<|proteinHere|>'])

    model.set_llm_tokenizer(llm_tokenizer)

    llm = LlamaForCausalLM.from_pretrained(config.text_config._name_or_path)
    llm.resize_token_embeddings(len(llm_tokenizer))

    peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.1,
        r=args.lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules= ["q_proj", "k_proj", "v_proj", "o_proj"],
        target_modules= ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","down_proj","up_proj","lm_head"],
    )

    peft_llm = get_peft_model(llm, peft_config)
    model.set_llm(peft_llm)

    # ---------------------------------  加载UPPIN features  --------------------------------------
    UPPIN_edges, UPPIN_seqs, UPPIN_features = load_UPPIN(exclude_tasks=[args.task])
    model.set_graph(UPPIN_edges)
    model.set_protein_features(UPPIN_features)

    # ---------------------------------  加载protein encoder  --------------------------------------
    model.init_protein_encoder()

    print("LOAD WEIGHTS ....................... ")
    load_model_weights(model, args)

    model.to(torch.bfloat16)
    model.to(device)

    return model, llm_tokenizer

    
def collate_fn(batch, llm_tokenizer):
    batch_size = len(batch)

    proteins = []
    ppi_idx_list = []
    cache_idx_list = []
    
    molecules = []
    cells = []
    texts = []

    for i in range(batch_size):
        data = batch[i]
        # print(data)
        texts.append((i, data['question'] + ANSWER_PREFIX))
        for protein in data['proteins']:
            proteins.append((i,protein))

        if 'ppi_idx' in data:
            for idx in data['ppi_idx']:
                ppi_idx_list.append((i,idx))
                
        if 'cache_idx' in data:
            cache_key = data['cache_key']
            cache_idx = data['cache_idx']
            if not isinstance(cache_idx, list):
                cache_idx = [cache_idx]
                
            for idx in cache_idx:
                cache_idx_list.append((i, cache_key, idx))

    ########################### tokenize texts #####################################
    texts_belong = [x[0] for x in texts]
    text_list = [x[1] for x in texts]
    text_input_dict = llm_tokenizer(text_list, padding='max_length', max_length=1024, add_special_tokens=True)
    ################################################################################

    cache_belong = [item[0] for item in cache_idx_list]
    cache_list = [(item[1], item[2]) for item in cache_idx_list]
    
    ppi_belong = [item[0] for item in ppi_idx_list]
    ppi_list = [item[1] for item in ppi_idx_list]

    return {
        'text_input_ids': text_input_dict['input_ids'],
        'text_attention_mask': text_input_dict['attention_mask'],
        'texts_belong': texts_belong,
        'cache_list':cache_list,
        'cache_belong':cache_belong,
        'ppi_list':ppi_list,
        'ppi_belong':ppi_belong,
        'proteins':proteins
    }


if __name__ == "__main__":
    args = get_parse()
    setup_seed(args.seed)

    model, llm_tokenizer = get_model(args)
    train_dataset, test_dataset = load_data(args)
    
    outputs = []
    for i in tqdm(range(len(test_dataset))):
        batch = [test_dataset[i]]
        inputs = collate_fn(batch, llm_tokenizer)

        question = test_dataset[i]['question']
        truth = test_dataset[i]['answer']

        # pred = ""

        with torch.no_grad():
            pred = model.generate_v1(**inputs)
            # pred = model.generate(**inputs)[0]

        if len(pred) < 5:
            print("Regenerate ......")
            print(test_dataset[i])
            
            new_seed = random.randint(1,10000)
            setup_seed(args.seed)
            with torch.no_grad():
                pred = model.generate_v1(**inputs)
                # pred = model.generate(**inputs)[0]

        # pred_no_topo = model.generate_v0(**inputs)

        print("=============================================================")
        print(test_dataset[i])
        print(question)
        print("Truthe: ", truth)
        print("Pred: ", pred)
        # print("Pred(no topo):", pred_no_topo)
        print('\n')
        
        tmp = test_dataset[i].copy()
        tmp['pred'] = pred
        
        outputs.append(tmp)

    # outputs = pd.DataFrame(outputs)
    # outputs.to_csv(f'./outputs/outputs_{args.task}_{args.ckpt}.csv', index=False)

    
