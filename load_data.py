# manual https://huggingface.co/docs/datasets/create_dataset
from datasets import Dataset,DatasetDict, load_from_disk, concatenate_datasets
from utils import isnull, notnull
import pandas as pd
import random
import json
import pickle
import os
from tqdm import tqdm
import torch
import itertools
from torch_geometric.data import Data
from torch_geometric.utils import degree
import torch.nn.functional as F

from config.configs import DATA_DIR, WORKERS, MAX_SEQ_LEN, MIN_SEQ_LEN, MAX_TEXT_LEN, QUESTION_PREFIX, ANSWER_PREFIX

def load_from_json(task):
    json_train = os.path.join(DATA_DIR, f'dataset/train/{task}_train.json')
    json_eval = os.path.join(DATA_DIR, f'dataset/eval/{task}_eval.json')

    if os.path.exists(json_train) and os.path.exists(json_eval):
        with open(json_train, 'r') as file:
            data_list_train = json.load(file)

        with open(json_eval, 'r') as file:
            data_list_eval = json.load(file)

        return data_list_train, data_list_eval
    else:
        return None, None
    
def save_to_json(task, data_list_train, data_list_eval):
    json_train = os.path.join(DATA_DIR, f'dataset/train/{task}_train.json')
    json_eval = os.path.join(DATA_DIR, f'dataset/eval/{task}_eval.json')

    with open(json_train, 'w') as f:
        json.dump(data_list_train, f)

    with open(json_eval, 'w') as f:
        json.dump(data_list_eval, f)

        
def find_sim_pidx(idx, x, x_masked):
    target_features = x[idx]
    
    cosine_sim = F.cosine_similarity(target_features, x_masked)
    max_value, max_index = torch.max(cosine_sim, dim=0)
    
    return max_index.item(), round(max_value.item(),2)
        
        
def get_RAG_promt(data_list, exclude_tasks):
    UPPIN_edges, UPPIN_seqs, UPPIN_features = load_UPPIN(exclude_tasks)
    
    #### construct graph
    ppi_list = UPPIN_edges
    x = UPPIN_features
    edge_index = torch.tensor(ppi_list, dtype=torch.long).t()
    data = Data(x=x, edge_index=edge_index)
    d = degree(edge_index[0])
    
    mask = d == 0
    x_masked = x.clone()
    x_masked[mask] = 0

    for data in data_list:
        data['sims'] = []
        
        sup_info = ""
        proteins = data['proteins']
        ppi_idx = [UPPIN_seqs.index(p) for p in proteins]
        
        assert len(proteins) == len(ppi_idx)
        
        data['ppi_idx'] = ppi_idx
        
        if len(ppi_idx) == 0:
            pass
        else:
            for i in range(len(ppi_idx)):
                idx = ppi_idx[i]
                if d[idx] == 0:
                    ppi_idx[i], sim = find_sim_pidx(idx, x, x_masked)
                    
                    print(f"Original idx: {idx}; New idx: {ppi_idx[i]}")
                    
                    sim = min(sim, 1.0)
                    
                    sup_info = sup_info + f" For protein {i+1}, the topological information in the PPI network of the protein with a cosine similarity of {sim} is <|reserved_special_token_0|>."
                    data['sims'].append(sim)
                else:
                    sup_info = sup_info + f" For protein {i+1}, its topological information in the PPI network is <|reserved_special_token_0|>."
                    data['sims'].append(1.0)
            
        data['question'] = data['question'] + sup_info
        
def load_UPPIN(exclude_tasks):
    UPPIN_edges = load_ppi_list("UPPIN", exclude_tasks)

    UPPIN_chains_path = os.path.join(DATA_DIR, 'dataset/processed_data/UPPIN_chains.csv')
    df = pd.read_csv(UPPIN_chains_path)
    UPPIN_seqs = df['seq'].to_list()

    UPPIN_features = load_protein_features('UPPIN')

    return UPPIN_edges, UPPIN_seqs, UPPIN_features



def load_pdb2020_complex(sample_ratio=0.8):
    def get_complex_text(proteins,protein_idx,pic50):
        pp = ['<|proteinHere|>' for i in range(len(proteins))]
        pp_str = ' '.join(pp)

        pt = ['<|reserved_special_token_0|>' for i in range(len(protein_idx)-1)]
        pt_str = ' '.join(pt)
        pt_str = pt_str + ' and <|reserved_special_token_0|>'

        # description_templates = [f"There is a complex containing the following proteins {pp_str}, whose topological information in a PPI network is {pt_str}, respectively. What is the binding affinity (log Kd) between these proteins? Carefully analyze the given protein features and their corresponding topological information, based on the definition of log Kd, answer this question in the form of 'Based on the given protein information, the binding affinity of this compound is log Kd = [predicted value].'"]
        description_templates = [f"There is a complex containing the following proteins {pp_str}. What is the binding affinity (log Kd) between these proteins? Carefully analyze the given protein features, based on the definition of log Kd, answer this question in the form of 'Based on the given protein information, the binding affinity of this compound is log Kd = [predicted value].'"]
        
        question_template = random.choice(description_templates)
        question = question_template

        answer = f"Based on the given protein information, the binding affinity of this compound is log Kd = {pic50}."

        the_data = {
            "question" : question,
            "answer" : answer,
            "proteins": proteins,
            "cache_key": 'UPPIN',
            "cache_idx":protein_idx
        }

        return the_data, len(QUESTION_PREFIX + question + ANSWER_PREFIX + answer)

    task = 'pdb2020_complex'
    data_list_train, data_list_eval = load_from_json(task)

    if data_list_train is None or data_list_eval is None:
        ##############################################################################
        df_path = os.path.join(DATA_DIR, 'dataset/processed_data/UPPIN_chains.csv')
        pt_df = pd.read_csv(df_path)
        all_chains = pt_df['seq'].to_list()
        ##############################################################################

        fname = os.path.join(DATA_DIR, 'dataset/pdb2020/PDB2020_PP_INDEX_WITH_CHAINS.csv')
        df = pd.read_csv(fname)

        data_list = []
        for i in range(len(df)):
            proteins = df['fasta_chains'][i]
            proteins = proteins.replace('[','')
            proteins = proteins.replace(']','')
            proteins = proteins.replace('\'','')
            proteins = proteins.replace(' ','')
            proteins = proteins.split(',')

            # proteins = list(set(proteins))

            # proteins = [seq for seq in proteins if len(seq) >= MIN_SEQ_LEN and len(seq) <= MAX_SEQ_LEN]
            proteins = [seq for seq in proteins if len(seq) > 0]

            if len(proteins) < 2:
                continue

            protein_idx = [all_chains.index(seq) for seq in proteins]

            pic50 = df['pic50'][i]
            pic50 = round(pic50, 2)

            the_data, text_len = get_complex_text(proteins,protein_idx,pic50)
            if text_len <= MAX_TEXT_LEN:
                data_list.append(the_data)

        random.shuffle(data_list)
        sample_num = int(len(data_list) * sample_ratio)

        data_list_train = data_list[:sample_num]
        data_list_eval = data_list[sample_num:]

        save_to_json(task, data_list_train, data_list_eval)

    return data_list_train, data_list_eval

def load_ppi_list(task, exclude_tasks=['SHS27k','SHS148k','pdb2020_complex']):
    ppi_path = os.path.join(DATA_DIR, f'dataset/processed_data/{task}_edges.json')

    if not os.path.exists(ppi_path):
        return None

    with open(ppi_path, 'r') as file:
        ppi_list = json.load(file)
        
        
    ppi_list_l = [(x[0],x[1]) for x in ppi_list]
    ppi_list_r = [(x[1],x[0]) for x in ppi_list]

    ppi_list = ppi_list_l + ppi_list_r
    ppi_set = set(ppi_list)
        
    #####################################################    
    data_list_train = []
    data_list_eval = []
    for et in exclude_tasks:
        tmp_train, tmp_eval = load_from_json(task)

        if tmp_train is not None and tmp_eval is not None:
            data_list_train.extend(tmp_train)
            data_list_eval.extend(tmp_eval)
    #####################################################

    # filter
    test_ppi_list = []
    for data in data_list_eval:
        pidx = data['ppi_idx']
        
        if len(pidx) < 2:
            continue
        
        combinations = list(itertools.combinations(pidx, 2))
        test_ppi_list.extend(combinations)
        
    test_ppi_l = [(x[0],x[1]) for x in test_ppi_list]
    test_ppi_r = [(x[1],x[0]) for x in test_ppi_list]
    test_ppi_list = test_ppi_l + test_ppi_r
    test_ppi_set = set(test_ppi_list)
    
    
    # ppi_list = list(ppi_set)
    ppi_list = list(ppi_set - test_ppi_set)

    return ppi_list

def load_protein_features(task='UPPIN'):
    pf_path = os.path.join(DATA_DIR, f'dataset/processed_data/{task}_tensor.pt')

    if not os.path.exists(pf_path):
        return None

    protein_features = torch.load(pf_path)
    return protein_features

def load_STRING(task='SHS27k', sample_ratio=0.8):
    def get_string(seq_a,seq_b,protein_idx,relations_str):
        # description_templates = [f"There are two proteins, namely <|proteinHere|> and <|proteinHere|>. Among the following seven types of relationships (reaction, binding, ptmod, activation, inhibition, catalysis, expression), which types of relationships exist between these two proteins?"]
        # description_templates = [f"There are two proteins, namely <|proteinHere|> and <|proteinHere|>. Among the following seven types of relationships (reaction, binding, ptmod, activation, inhibition, catalysis, expression), list all possible relationships between these two proteins."]
        description_templates = [f"There are two proteins, <|proteinHere|> and <|proteinHere|>. Among the following seven types of relationships (reaction, binding, ptmod, activation, inhibition, catalysis, expression), list all possible relationships between these two proteins. Carefully analyze the given protein features and their corresponding topological information, based on the definition of the seven protein relations, answer this question in the form of 'According to the given protein information, Their relationships include relation(s).' If multiple relationships may exist, separate them with comma."]

        question_template = random.choice(description_templates)
        question = question_template

        answer = f"According to the given protein information, their relationships include {relations_str}."

        the_data = {
            "question" : question,
            "answer" : answer,
            "proteins": [seq_a,seq_b],
            "cache_idx" : protein_idx,
            "cache_key" : 'UPPIN'
        }

        return the_data, len(QUESTION_PREFIX + question + ANSWER_PREFIX + answer)

    data_list_train, data_list_eval = load_from_json(task)
    ppi_path = os.path.join(DATA_DIR, f'dataset/STRING/processed_data/{task}_edges.json')
    ppi_list = None
    if os.path.exists(ppi_path):
        with open(ppi_path, 'r') as file:
            ppi_list = json.load(file)

    tmp_data_list_train = []
    tmp_data_list_eval = []
    if data_list_train is None or data_list_eval is None or ppi_list is None:
    # if data_list_train is None or data_list_eval is None:
        # class_map = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3, 'inhibition':4, 'catalysis':5, 'expression':6}
        class_map = {0:'reaction', 1:'binding', 2:'ptmod', 3:'activation', 4:'inhibition', 5:'catalysis', 6:'expression'}
        ppi_list, ppi_label_list, ppi_split_dict, protein_dict_by_idx = load_graph(task,'dfs',100)
        # ppi_list, ppi_label_list, ppi_split_dict, protein_dict_by_idx = load_graph(task,'random',100)
        # ppi_list, ppi_label_list, ppi_split_dict, protein_dict_by_idx = load_graph(task,'bfs',100)

        ##############################################################################
        df_path = os.path.join(DATA_DIR, 'dataset/processed_data/UPPIN_chains.csv')
        pt_df = pd.read_csv(df_path)
        all_chains = pt_df['seq'].to_list()
        ##############################################################################

        for ppi_idx in range(len(ppi_list)):
            ppi = ppi_list[ppi_idx]
            idx_a = ppi[0]
            idx_b = ppi[1]
            seq_a = protein_dict_by_idx[idx_a]['seq']
            seq_b = protein_dict_by_idx[idx_b]['seq']

            label = ppi_label_list[ppi_idx]
            label_to_str = [class_map[i] for i in range(len(label)) if label[i] == 1]
            relations_str = ', '.join(label_to_str)


            if len(seq_a) > MAX_SEQ_LEN or len(seq_a) < MIN_SEQ_LEN:
                continue

            if len(seq_b) > MAX_SEQ_LEN or len(seq_b) < MIN_SEQ_LEN:
                continue

            protein_idx = [all_chains.index(seq) for seq in [seq_a, seq_b]]
            the_data, text_len = get_string(seq_a, seq_b, protein_idx, relations_str)
            if text_len > MAX_TEXT_LEN:
                continue

            if ppi_idx in ppi_split_dict['train_index']:
                tmp_data_list_train.append(the_data)

            if ppi_idx in ppi_split_dict['test_index']:
                tmp_data_list_eval.append(the_data)

        data_list_train = tmp_data_list_train
        data_list_eval = tmp_data_list_eval
        save_to_json(task, data_list_train, data_list_eval)

    # return data_list_train, data_list_eval, ppi_list
    return data_list_train, data_list_eval

def load_CDRsq(sample_ratio=0.9):
    def CDRsq_text(seq,CDR1,CDR2,CDR3):
        question_templates = [
            "This antibody is <|proteinHere|>, what are its CDR regions?",
            "What are the CDR regions of this antibody, which is <|proteinHere|>?",
            "Could you tell me the CDR regions of this antibody with <|proteinHere|>?",
            "For this antibody that is <|proteinHere|>, what would its CDR regions be?",
            "Can you identify the CDR regions of this antibody, characterized by <|proteinHere|>?",
            "What would be the CDR regions for this antibody, which is <|proteinHere|>?",
            "I'm curious about the CDR regions of this antibody, which is <|proteinHere|>.",
            "Regarding this antibody with <|proteinHere|>, what are its CDR regions?",
            "This antibody has a <|proteinHere|>, could you identify its CDR regions?",
            "What are the CDR regions for a antibody with a <|proteinHere|>?",
            "In the case of this antibody, which is <|proteinHere|>, what are the CDR regions?"
        ]

        question_template = random.choice(question_templates)
        question = question_template

        answer_template = "The CDR regions of this antibody are: CDR1 (<CDR1>), CDR2 (<CDR2>), CDR3 (<CDR3>)."
        answer = answer_template.replace("<CDR1>",CDR1).replace("<CDR2>",CDR2).replace("<CDR3>",CDR3)

        proteins = [seq]

        the_data = {
            "question" : question,
            "answer" : answer,
            "proteins": proteins
        }

        return the_data, len(QUESTION_PREFIX + question + ANSWER_PREFIX + answer)

    task = "CDRs"
    data_list_train, data_list_eval = load_from_json(task)

    if data_list_train is None or data_list_eval is None:

        file_name_list = ['dataset/sabdab/PPI_plus.csv','dataset/sabdab/PPP_plus.csv','dataset/sabdab/PPPI_plus.csv']
        df_list = []
        for file_name in file_name_list:
            file_path = os.path.join(DATA_DIR, file_name)
            df_tmp = pd.read_csv(file_path)
            df_list.append(df_tmp)

        df = pd.concat(df_list)
        records = df.to_dict("records")

        data_list = []
        for record in records:
            if notnull(record['Hchain']) and notnull(record['Hchain_seq']):
                chain = record['Hchain']
                CDR1 = str(record['H1'])
                CDR2 = str(record['H2'])
                CDR3 = str(record['H3'])
                seq = str(record['Hchain_fasta_seq'])

                if len(seq) > MAX_SEQ_LEN or len(seq) < MIN_SEQ_LEN:
                    pass
                else:
                    the_data, text_len = CDRsq_text(seq,CDR1,CDR2,CDR3)

                    if text_len <= MAX_TEXT_LEN:
                        data_list.append(the_data)

            if notnull(record['Lchain']) and notnull(record['Lchain_seq']):
                chain = record['Lchain']
                CDR1 = str(record['L1'])
                CDR2 = str(record['L2'])
                CDR3 = str(record['L3'])
                seq = str(record['Lchain_fasta_seq'])

                if len(seq) > MAX_SEQ_LEN or len(seq) < MIN_SEQ_LEN:
                    pass
                else:
                    the_data, text_len = CDRsq_text(seq,CDR1,CDR2,CDR3)
                    if text_len <= MAX_TEXT_LEN:
                        data_list.append(the_data)

        random.shuffle(data_list)
        sample_num = int(len(data_list) * sample_ratio)
        data_list_train = data_list[:sample_num]
        data_list_eval = data_list[sample_num:]

        save_to_json(task, data_list_train, data_list_eval)

    return data_list_train, data_list_eval



#################################################################### uniprot #########################################################################
def get_des_qa(data):
    description_qa_templates = [
        "What is the function of the protein <|proteinHere|>?",
        "What role does the protein <|proteinHere|> play?",
        "Can you explain the purpose of the protein <|proteinHere|>?",
        "What is the biological activity of the protein <|proteinHere|>?",
        "What is the main task of the protein <|proteinHere|> in the cell?",
        "What is the primary function of the <|proteinHere|> protein?",
        "How does the protein <|proteinHere|> contribute to cellular processes?",
        "What is the significance of the protein <|proteinHere|> in terms of its function?",
        "What cellular function is the protein <|proteinHere|> responsible for?",
        "In what way does the protein <|proteinHere|> participate in the biological system?",
        "What specific role is carried out by the protein <|proteinHere|>?"
    ]

    question_template = random.choice(description_qa_templates)
    question = question_template

    data['question'] = question
    data['answer'] = data['des']
    data['proteins'] = [data['seq']]

    return data

def get_name_qa(data):
    name_qa_templates = [
        "What are the official names of the protein <|proteinHere|>?",
        "Can you tell me the official names of the protein <|proteinHere|>?",
        "What is the protein <|proteinHere|> officially called?",
        "Could you provide the official nomenclature for the protein <|proteinHere|>?",
        "Do you know the formal names of the protein <|proteinHere|>?",
        "What's the recognized naming for the protein <|proteinHere|>?",
        "I'm trying to find out the official names of <|proteinHere|>. Can you help?",
        "Can you identify the protein <|proteinHere|> by its official names?",
        "What are the formal designations of the protein <|proteinHere|>?",
        "In terms of official nomenclature, what is the protein <|proteinHere|> known as?",
        "What are the sanctioned names for the protein <|proteinHere|>?"
    ]

    question_template = random.choice(name_qa_templates)
    question = question_template

    data['question'] = question
    data['answer'] = data['off_name']
    data['proteins'] = [data['seq']]

    return data

def get_family_qa(data):
    family_qa_templates = [
        "What is the protein family that the protein <|proteinHere|> belongs to?",
        "Can you tell me the protein family of the protein <|proteinHere|>?",
        "To which protein family does the protein <|proteinHere|> belong?",
        "What family is the protein <|proteinHere|> a part of?",
        "Could you identify the protein family for the protein <|proteinHere|>?",
        "Do you know the family of the protein <|proteinHere|>?",
        "What's the family classification of the protein <|proteinHere|>?",
        "I'm trying to find out the protein family of <|proteinHere|>. Can you help?",
        "Can you classify the protein <|proteinHere|> by its family?",
        "What is the family grouping of the protein <|proteinHere|>?",
        "In terms of protein family, where does the protein <|proteinHere|> fit in?"
    ]

    question_template = random.choice(family_qa_templates)
    question = question_template

    data['question'] = question
    data['answer'] = data['sim']
    data['proteins'] = [data['seq']]

    return data

def get_location_qa(data):
    subcellular_location_qa_templates = [
        "What are the subcellular locations of the protein <|proteinHere|>?",
        "Can you tell me the subcellular locations of the protein <|proteinHere|>?",
        "Where within the cell can the protein <|proteinHere|> be found?",
        "Could you identify the intracellular locations of the protein <|proteinHere|>?",
        "Do you know where the protein <|proteinHere|> is located within the cell?",
        "What's the cellular localization of the protein <|proteinHere|>?",
        "I'm trying to find out the subcellular locations of <|proteinHere|>. Can you help?",
        "Can you specify the locations within the cell where the protein <|proteinHere|> is found?",
        "What are the intracellular sites of the protein <|proteinHere|>?",
        "In terms of subcellular localization, where is the protein <|proteinHere|> located?",
        "What are the cellular compartments where the protein <|proteinHere|> resides?"
    ]

    question_template = random.choice(subcellular_location_qa_templates)
    question = question_template

    data['question'] = question
    data['answer'] = data['sl']
    data['proteins'] = [data['seq']]

    return data


def get_align(data):
    description_align_templates = [
        "The embedding of this protein is <|proteinHere|>; what is its sequence?",
        "This protein has a embedding of <|proteinHere|>; can you tell me its sequence?",
        "With a embedding of <|proteinHere|>, what is the sequence of this protein?",
        "Given that this protein's embedding is <|proteinHere|>, what is its sequence?",
        "This protein's embedding is <|proteinHere|>; what sequence does it have?",
        "The protein's embedding is <|proteinHere|>; what is the corresponding sequence?",
        "For a protein with a embedding of <|proteinHere|>, what is its sequence?",
        "This protein exhibits a embedding of <|proteinHere|>; what is its sequence?",
        "Considering the embedding of this protein is <|proteinHere|>, what is its sequence?",
        "The embedding of this protein is <|proteinHere|>; could you provide its sequence?",
        "The protein has a embedding of <|proteinHere|>; what sequence does it possess?"
    ]

    question_template = random.choice(description_align_templates)
    question =  question_template 

    data['question'] = question
    data['answer'] = data['seq'] + '.'
    data['proteins'] = [data['seq']]

    return data

def get_align_length(data):
    description_align_templates = [
        "The embedding of this protein is <|proteinHere|>. What is its sequence length?",
        "This protein has a embedding of <|proteinHere|>. How long is its sequence?",
        "The protein's embedding is <|proteinHere|>. Can you tell me the length of its sequence?",
        "With a embedding of <|proteinHere|>, what is the sequence length of this protein?",
        "The sequence length of the protein with embedding <|proteinHere|> is what?",
        "For the protein embedded as <|proteinHere|>, what is the length of its sequence?",
        "What is the sequence length of the protein that has a embedding of <|proteinHere|>?",
        "Given that the embedding of this protein is <|proteinHere|>, what is its sequence length?",
        "The protein, which has a embedding of <|proteinHere|>, has what sequence length?",
        "What is the length of the sequence for the protein with the embedding <|proteinHere|>?",
        "If the embedding of this protein is <|proteinHere|>, what is the length of its sequence?"
    ]

    question_template = random.choice(description_align_templates)
    question = question_template 

    p_len = len(data['seq'])
    answer = f"The length of this protein sequence is {p_len}."

    data['question'] = question
    data['answer'] = answer
    data['proteins'] = [data['seq']]

    return data

def get_align_top5(data):
    description_align_templates = [
        "The embedding of this protein is <|proteinHere|>. What are the first 5 amino acids of this protein sequence?",
        "This protein has a embedding of <|proteinHere|>. What are the first 5 amino acids in its sequence?",
        "The protein's embedding is <|proteinHere|>. Can you tell me the first 5 amino acids of its sequence?",
        "With a embedding of <|proteinHere|>, what are the first 5 amino acids of this protein sequence?",
        "The first 5 amino acids of the protein with embedding <|proteinHere|> are what?",
        "For the protein embedded as <|proteinHere|>, what are the first 5 amino acids in its sequence?",
        "What are the first 5 amino acids in the sequence of the protein that has a embedding of <|proteinHere|>?",
        "Given that the embedding of this protein is <|proteinHere|>, what are the first 5 amino acids in its sequence?",
        "The protein, which has a embedding of <|proteinHere|>, has what first 5 amino acids in its sequence?",
        "What are the first 5 amino acids in the sequence for the protein with the embedding <|proteinHere|>?",
        "If the embedding of this protein is <|proteinHere|>, what are the first 5 amino acids in its sequence?"
    ]

    question_template = random.choice(description_align_templates)
    question = question_template 

    top5 = " ".join(data['seq'][:5])
    answer = f"The first 5 amino acids of this protein sequence is {top5}."

    data['question'] = question
    data['answer'] = answer
    data['proteins'] = [data['seq']]

    return data

def get_align_last5(data):
    description_align_templates = [
        "The embedding of this protein is <|proteinHere|>. What are the last 5 amino acids of this protein sequence?",
        "This protein has a embedding of <|proteinHere|>. What are the last 5 amino acids in its sequence?",
        "The protein's embedding is <|proteinHere|>. Can you tell me the last 5 amino acids of its sequence?",
        "With a embedding of <|proteinHere|>, what are the last 5 amino acids of this protein sequence?",
        "The last 5 amino acids of the protein with embedding <|proteinHere|> are what?",
        "For the protein embedded as <|proteinHere|>, what are the last 5 amino acids in its sequence?",
        "What are the last 5 amino acids in the sequence of the protein that has a embedding of <|proteinHere|>?",
        "Given that the embedding of this protein is <|proteinHere|>, what are the last 5 amino acids in its sequence?",
        "The protein, which has a embedding of <|proteinHere|>, has what last 5 amino acids in its sequence?",
        "What are the last 5 amino acids in the sequence for the protein with the embedding <|proteinHere|>?",
        "If the embedding of this protein is <|proteinHere|>, what are the last 5 amino acids in its sequence?"
    ]

    question_template = random.choice(description_align_templates)
    question = question_template 

    last5 = " ".join(data['seq'][-5:])
    answer = f"The last 5 amino acids of this protein sequence is {last5}."

    data['question'] = question
    data['answer'] = answer
    data['proteins'] = [data['seq']]

    return data

def load_uniprot():
    d_path = os.path.join(DATA_DIR,'dataset/pretrain/uniprot')

    if os.path.exists(d_path):
        uniprot_dataset_dict = load_from_disk(d_path)
    else:
        file_name = "dataset/uniprot_qa/uniprot.pickle"
        pickle_path = os.path.join(DATA_DIR,file_name)

        with open(pickle_path, "rb") as f:
            uniprot = pickle.load(f)

        all_seqs = []
        uniprot_dict_list = []
        for pro_name, pro_info in uniprot.items():
            if 'Sequence' not in pro_info or len(pro_info['Sequence']) < MIN_SEQ_LEN:
                continue
            else:
                seq = pro_info['Sequence']
                if seq not in all_seqs:
                    all_seqs.append(seq)
                    cache_idx = [len(all_seqs)-1]
                else:
                    cache_idx = [all_seqs.index(seq)]

            des = pro_info['Description'] if 'Description' in pro_info else ''
            off_name = pro_info['Name'] if 'Name' in pro_info else ''
            sim = pro_info['Similarity'] if 'Similarity' in pro_info else ''
            sl = pro_info['Subcellular_Location'] if 'Subcellular_Location' in pro_info else ''

            uniprot_dict_list.append({
                'pro_name':pro_name,
                'seq':seq,
                'des':des,
                'off_name':off_name,
                'sim':sim,
                'sl':sl,
                'cache_idx':cache_idx,
                'cache_key':'uniprot'
            })

        # save_all_seqs
        outputs = pd.DataFrame(data=all_seqs, columns=['seq'])
        outputs.to_csv(os.path.join(DATA_DIR, 'dataset/processed_data/uniprot_chains.csv'), index=False)

        # uniprot_dataset = Dataset.from_list(uniprot_dict_list[:1000])
        uniprot_dataset = Dataset.from_list(uniprot_dict_list)

        des_qa_dataset = uniprot_dataset.map(get_des_qa, batched=False, batch_size=16, with_rank=False, num_proc=WORKERS)
        des_qa_dataset = des_qa_dataset.remove_columns(['seq','des','off_name','sim','sl'])
        des_qa_dataset = des_qa_dataset.filter(lambda data: len(data["answer"])!=0, num_proc=WORKERS)
        split_des_qa_dataset = des_qa_dataset.train_test_split(test_size=0.05)
        split_des_qa_dataset.save_to_disk(os.path.join(DATA_DIR,'dataset/pretrain/des_qa'))

        fa_qa_dataset = uniprot_dataset.map(get_family_qa, batched=False, batch_size=16, with_rank=False, num_proc=WORKERS)
        fa_qa_dataset = fa_qa_dataset.remove_columns(['seq','des','off_name','sim','sl'])
        fa_qa_dataset = fa_qa_dataset.filter(lambda data: len(data["answer"])!=0, num_proc=WORKERS)
        split_fa_qa_dataset = fa_qa_dataset.train_test_split(test_size=0.05)
        split_fa_qa_dataset.save_to_disk(os.path.join(DATA_DIR,'dataset/pretrain/fa_qa'))

        sl_qa_dataset = uniprot_dataset.map(get_location_qa, batched=False, batch_size=16, with_rank=False, num_proc=WORKERS)
        sl_qa_dataset = sl_qa_dataset.remove_columns(['seq','des','off_name','sim','sl'])
        sl_qa_dataset = sl_qa_dataset.filter(lambda data: len(data["answer"])!=0, num_proc=WORKERS)
        split_sl_qa_dataset = sl_qa_dataset.train_test_split(test_size=0.05)
        split_sl_qa_dataset.save_to_disk(os.path.join(DATA_DIR,'dataset/pretrain/sl_qa'))

        align_dataset = uniprot_dataset.map(get_align, batched=False, batch_size=16, with_rank=False, num_proc=WORKERS)
        align_dataset = align_dataset.remove_columns(['seq','des','off_name','sim','sl'])
        align_dataset = align_dataset.filter(lambda data: len(data["answer"])!=0, num_proc=WORKERS)
        split_align_dataset = align_dataset.train_test_split(test_size=0.05)
        split_align_dataset.save_to_disk(os.path.join(DATA_DIR,'dataset/pretrain/align'))

        align_len_dataset = uniprot_dataset.map(get_align_length, batched=False, batch_size=16, with_rank=False, num_proc=WORKERS)
        align_len_dataset = align_len_dataset.remove_columns(['seq','des','off_name','sim','sl'])
        align_len_dataset = align_len_dataset.filter(lambda data: len(data["answer"])!=0, num_proc=WORKERS)
        split_align_len_dataset = align_len_dataset.train_test_split(test_size=0.05)
        split_align_len_dataset.save_to_disk(os.path.join(DATA_DIR,'dataset/pretrain/align_len'))

        align_top5_dataset = uniprot_dataset.map(get_align_top5, batched=False, batch_size=16, with_rank=False, num_proc=WORKERS)
        align_top5_dataset = align_top5_dataset.remove_columns(['seq','des','off_name','sim','sl'])
        align_top5_dataset = align_top5_dataset.filter(lambda data: len(data["answer"])!=0, num_proc=WORKERS)
        split_align_top5_dataset = align_top5_dataset.train_test_split(test_size=0.05)
        split_align_top5_dataset.save_to_disk(os.path.join(DATA_DIR,'dataset/pretrain/align_top5'))

        align_last5_dataset = uniprot_dataset.map(get_align_last5, batched=False, batch_size=16, with_rank=False, num_proc=WORKERS)
        align_last5_dataset = align_last5_dataset.remove_columns(['seq','des','off_name','sim','sl'])
        align_last5_dataset = align_last5_dataset.filter(lambda data: len(data["answer"])!=0, num_proc=WORKERS)
        split_align_last5_dataset = align_last5_dataset.train_test_split(test_size=0.05)
        split_align_last5_dataset.save_to_disk(os.path.join(DATA_DIR,'dataset/pretrain/align_last5'))

        # merge        
        updated_uniprot_train = concatenate_datasets([split_des_qa_dataset['train'], split_fa_qa_dataset['train'],
                                                      split_sl_qa_dataset['train'], split_align_dataset['train'],
                                                      split_align_len_dataset['train'], split_align_top5_dataset['train'],
                                                      split_align_last5_dataset['train']])

        updated_uniprot_test = concatenate_datasets([split_des_qa_dataset['test'], split_fa_qa_dataset['test'],
                                                      split_sl_qa_dataset['test'], split_align_dataset['test'],
                                                      split_align_len_dataset['test'], split_align_top5_dataset['test'],
                                                      split_align_last5_dataset['test']])

        uniprot_dataset_dict = DatasetDict({'train': updated_uniprot_train, 'test':updated_uniprot_test})
        uniprot_dataset_dict.save_to_disk(d_path)

    return uniprot_dataset_dict

def load_uniprot_sub_from_pretrain(task):
    d_path = os.path.join(DATA_DIR,f'dataset/pretrain/{task}')
    if os.path.exists(d_path):
        dataset = load_from_disk(d_path)
    else:
        _ = load_uniprot()
        dataset = load_from_disk(d_path)
    return dataset

def load_uniprot_sub(task, stage):
    if stage == 'pretrain':
        dataset = load_uniprot_sub_from_pretrain(task)
    elif stage == 'ft':
        d_path = os.path.join(DATA_DIR,f'dataset/ft/{task}')

        if os.path.exists(d_path):
            dataset = load_from_disk(d_path)
        else:
            dataset = load_uniprot_sub_from_pretrain(task)
            dataset = dataset['test']
            dataset = dataset.train_test_split(test_size=0.1)
            dataset.save_to_disk(d_path)

    return dataset

def load_data(args):
    task = args.task
    task = 'uniprot' if task == 'stage1' else task
    stage = args.stage

    if task == 'CDRs':
        data_list_train, data_list_eval = load_CDRsq()
    elif task == 'uniprot':
        dataset = load_uniprot()
        dataset['train'] = dataset['train'].shuffle(seed=42)
        return dataset['train'], dataset['test']
    elif task in ['align','align_len','align_top5','align_last5','des_qa','fa_qa','sl_qa']:
        dataset = load_uniprot_sub(task, stage)
        dataset['train'] = dataset['train'].shuffle(seed=42)
        return dataset['train'], dataset['test']
    elif task == 'pdb2020_complex':
        data_list_train, data_list_eval = load_pdb2020_complex()
        get_RAG_promt(data_list_train, exclude_tasks=[task])
        get_RAG_promt(data_list_eval, exclude_tasks=[task])
    elif task in ['SHS27k','SHS148k','STRING']:
        data_list_train, data_list_eval = load_STRING(task)
        get_RAG_promt(data_list_train,exclude_tasks=[task])
        get_RAG_promt(data_list_eval,exclude_tasks=[task])
    elif task == 'tsmmg':
        data_list_train, data_list_eval = load_tsmmg()
    elif task == 'multi':
        train_complex, eval_complex = load_pdb2020_complex()
        get_RAG_promt(train_complex,exclude_tasks=['pdb2020_complex'])
        get_RAG_promt(eval_complex,exclude_tasks=['pdb2020_complex'])
        
        train_shs27k, eval_shs27k = load_STRING('SHS27k')
        get_RAG_promt(train_shs27k,exclude_tasks=['SHS27k'])
        get_RAG_promt(eval_shs27k,exclude_tasks=['SHS27k'])
        
        train_shs148k, eval_shs148k = load_STRING('SHS148k')
        get_RAG_promt(train_shs148k,exclude_tasks=['SHS148k'])
        get_RAG_promt(eval_shs148k,exclude_tasks=['SHS148k'])
        
        data_list_train = train_complex + train_shs27k + train_shs148k
        data_list_eval = eval_complex + eval_shs27k + eval_shs148k
    else:
        pass

    random.shuffle(data_list_train)

    return Dataset.from_list(data_list_train), Dataset.from_list(data_list_eval)

def load_tsmmg():
    def get_tsmmg(prompt,smiles):
        question =  "### Human: " + prompt
        answer = smiles

        answer = ANSWER_PREFIX + answer

        proteins = []
        SMILESs = []

        the_data = {
            "question" : question,
            "answer" : answer,
            "proteins": proteins,
            "SMILESs" : SMILESs
        }

        return the_data, len(question + ANSWER_PREFIX + answer)

    fname = 'dataset/tsmmg/1.csv'
    df = pd.read_csv(fname)

    data_list = []
    for i in range(len(df)):
        desc = df['desc'][i]
        smiles = df['smiles'][i]
        the_data, text_len = get_tsmmg(desc, smiles)
        data_list.append(the_data)

    return data_list, data_list

################################################################################################################
def load_graph(task,split_mode,seed):
    protein_idx = 0
    ppi_idx = 0
    
    protein_dict_by_name = {}
    protein_dict_by_idx = {}
    ppi_dict = {}
    ppi_list = []
    ppi_label_list = []

    class_map = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3, 'inhibition':4, 'catalysis':5, 'expression':6}

    ppi_path = os.path.join(DATA_DIR, f'dataset/STRING/protein.actions.{task}.txt')
    seq_path = os.path.join(DATA_DIR, f'dataset/STRING/protein.{task}.sequences.dictionary.tsv')

    df_seq = pd.read_csv(seq_path, sep='\t')
    for i in range(len(df_seq)):
        protein_name = df_seq['id'][i]
        seq = df_seq['seq'][i]
        protein_dict_by_name[protein_name] = {
            'idx':protein_idx,
            'seq':seq
        }

        protein_dict_by_idx[protein_idx] = {
            'name':protein_name,
            'seq':seq
        }

        protein_idx = protein_idx + 1

    df_ppi = pd.read_csv(ppi_path, sep='\t')

    for i in range(len(df_ppi)):
        item_id_a = df_ppi['item_id_a'][i]
        item_id_b = df_ppi['item_id_b'][i]
        # seq_a = seq_dict[item_id_a]
        # seq_b = seq_dict[item_id_b]

        mode = df_ppi['mode'][i]

        key = f"{item_id_a}__{item_id_b}" if item_id_a < item_id_b else f"{item_id_b}__{item_id_a}"

        if key not in ppi_dict.keys():
            ppi_dict[key] = ppi_idx
            temp_label = [0, 0, 0, 0, 0, 0, 0]
            temp_label[class_map[mode]] = 1
            ppi_label_list.append(temp_label)
            ppi_idx = ppi_idx + 1
        else:
            index = ppi_dict[key]
            ppi_label_list[index][class_map[mode]] = 1

    # Python 3.7 以上字典类型是有序的
    for ppi in tqdm(ppi_dict.keys()):
        temp = ppi.strip().split('__')
        ppi_list.append(temp)

    ppi_num = len(ppi_list)
    for i in tqdm(range(ppi_num)):
        p1_name = ppi_list[i][0]
        p2_name = ppi_list[i][1]

        ppi_list[i][0] = protein_dict_by_name[p1_name]['idx']
        ppi_list[i][1] = protein_dict_by_name[p2_name]['idx']

    # ppi_g = dgl.to_bidirected(dgl.graph(ppi_list))
    # protein_data = ProteinDatasetDGL(prot_r_edge_path, prot_k_edge_path, prot_node_path, dataset)
    ppi_split_dict = split_dataset(ppi_list, task, split_mode, seed)


    with open(os.path.join(DATA_DIR, f'dataset/STRING/processed_data/{task}_edges.json'), 'w') as f:
        json.dump(ppi_list, f)

    # return protein_data, ppi_g, ppi_list, ppi_label_list, ppi_split_dict
    return ppi_list, ppi_label_list, ppi_split_dict, protein_dict_by_idx


def split_dataset(ppi_list, task, split_mode, seed):
    if not os.path.exists(os.path.join(DATA_DIR, "./dataset/STRING/processed_data/{}_{}.json".format(task, split_mode))):
        if split_mode == 'random':
            ppi_num = len(ppi_list)
            random_list = [i for i in range(ppi_num)]
            random.shuffle(random_list)

            ppi_split_dict = {}
            ppi_split_dict['train_index'] = random_list[: int(ppi_num*0.6)]
            ppi_split_dict['val_index'] = random_list[int(ppi_num*0.6) : int(ppi_num*0.8)]
            ppi_split_dict['test_index'] = random_list[int(ppi_num*0.8) :]

            jsobj = json.dumps(ppi_split_dict)
            with open(os.path.join(DATA_DIR, "./dataset/STRING/processed_data/{}_{}.json".format(task, split_mode)), 'w') as f:
                f.write(jsobj)
                f.close()

        elif split_mode == 'bfs' or split_mode == 'dfs':
            node_to_edge_index = {}
            ppi_num = len(ppi_list)

            for i in range(ppi_num):
                edge = ppi_list[i]
                if edge[0] not in node_to_edge_index.keys():
                    node_to_edge_index[edge[0]] = []
                node_to_edge_index[edge[0]].append(i)

                if edge[1] not in node_to_edge_index.keys():
                    node_to_edge_index[edge[1]] = []
                node_to_edge_index[edge[1]].append(i)
            
            node_num = len(node_to_edge_index)
            sub_graph_size = int(ppi_num * 0.4)

            if split_mode == 'bfs':
                selected_edge_index = get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size)
            elif split_mode == 'dfs':
                selected_edge_index = get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size)
            
            all_edge_index = [i for i in range(ppi_num)]
            unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

            random_list = [i for i in range(len(selected_edge_index))]
            random.shuffle(random_list)

            ppi_split_dict = {}
            ppi_split_dict['train_index'] = unselected_edge_index
            ppi_split_dict['val_index'] = [selected_edge_index[i] for i in random_list[:int(ppi_num*0.2)]]
            ppi_split_dict['test_index'] = [selected_edge_index[i] for i in random_list[int(ppi_num*0.2):]]

            jsobj = json.dumps(ppi_split_dict)
            with open(os.path.join(DATA_DIR, "dataset/STRING/processed_data/{}_{}.json".format(task, split_mode)), 'w') as f:
                f.write(jsobj)
                f.close()
        
        else:
            print("your mode is {}, you should use bfs, dfs or random".format(split_mode))
            return
    else:
        with open(os.path.join(DATA_DIR, "dataset/STRING/processed_data/{}_{}.json".format(task, split_mode)), 'r') as f:
            ppi_split_dict = json.load(f)
            f.close()

    print("Train_PPI: {} | Val_PPI: {} | Test_PPI: {}".format(len(ppi_split_dict['train_index']), len(ppi_split_dict['val_index']), len(ppi_split_dict['test_index'])))

    return ppi_split_dict

# Data splitting by BFS
def get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    candiate_node = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 20:
        random_node = random.randint(0, node_num - 1)
    candiate_node.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        selected_node.append(cur_node)

        for edge_index in node_to_edge_index[cur_node]:
            if edge_index not in selected_edge_index:
                selected_edge_index.append(edge_index)

                end_node = -1
                if ppi_list[edge_index][0] == cur_node:
                    end_node = ppi_list[edge_index][1]
                else:
                    end_node = ppi_list[edge_index][0]

                if end_node not in selected_node and end_node not in candiate_node:
                    candiate_node.append(end_node)
            else:
                continue

    # node_list = candiate_node + selected_node

    return selected_edge_index


# Data splitting by DFS
def get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    stack = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 20:
        random_node = random.randint(0, node_num - 1)
    stack.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = stack[-1]

        if cur_node in selected_node:
            flag = True

            for edge_index in node_to_edge_index[cur_node]:
                if flag:
                    end_node = -1

                    if ppi_list[edge_index][0] == cur_node:
                        end_node = ppi_list[edge_index][1]
                    else:
                        end_node = ppi_list[edge_index][0]

                    if end_node in selected_node:
                        continue
                    else:
                        stack.append(end_node)
                        flag = False
                else:
                    break

            if flag:
                stack.pop()
            continue

        else:
            selected_node.append(cur_node)

            for edge_index in node_to_edge_index[cur_node]:
                if edge_index not in selected_edge_index:
                    selected_edge_index.append(edge_index)

    return selected_edge_index

def load_proteins(dataset):
    if dataset in ['SHS27k','SHS148k','STRING']:
        seq_path = os.path.join(DATA_DIR, 'dataset/STRING/protein.{}.sequences.dictionary.tsv'.format(dataset))
        df_seq = pd.read_csv(seq_path, sep='\t')
    elif dataset in ['pdb2020_complex']:
        seq_path = os.path.join(DATA_DIR, 'dataset/pdb2020/processed_data/all_chains.csv'.format(dataset))
        df_seq = pd.read_csv(seq_path)
    elif dataset in ['all']:
        seq_path = os.path.join(DATA_DIR, 'dataset/processed_data/all_ppi_chains.csv'.format(dataset))
        df_seq = pd.read_csv(seq_path)
    else:
        pass

    proteins = []
    for i in range(len(df_seq)):
        # key = df_seq['id'][i]
        seq = df_seq['seq'][i]
        proteins.append(seq)

    return proteins

if __name__ == '__main__':
    from utils import get_parse
    args = get_parse()
    
    print(args)

    # the_dataset = load_CDRsq()
    # train_dataset, test_dataset = train_test_split(the_dataset, test_size=0.1)

    # align_data = load_align()
    # print("Len of align data: ", len(align_data))
    # print(align_data[0])

    # CDRs_data = load_data('CDRs')
    # print("Len of CDRs data: ", len(CDRs_data))
    # print(CDRs_data[0])

    # unipro_data = load_data('uniprot')
    # print("Len of uniprot data: ", len(unipro_data))
    # print(unipro_data[0])

    # all_data = load_data('stage1')
    # print("Len of all data: ", len(all_data))

    # args.task = 'uniprot'
    # args.stage = 'pretrain'
    # data_list_train, data_list_eval = load_data(args)
    # print(f"{args.task}-{args.stage}: Len of training data: {len(data_list_train)}; Len of test data: {len(data_list_eval)}")

    args.task = 'align_last5'
    args.stage = 'ft'
    data_list_train, data_list_eval = load_data(args)
    print(f"{args.task}-{args.stage}: Len of training data: {len(data_list_train)}; Len of test data: {len(data_list_eval)}")
    
    print(data_list_train[0])
    print(len(data_list_train), len(data_list_eval))

    # args.task = 'align'
    # args.stage = 'ft'
    # data_list_train, data_list_eval = load_data(args)
    # print(f"{args.task}-{args.stage}: Len of training data: {len(data_list_train)}; Len of test data: {len(data_list_eval)}")

    # args.task = 'uniprot'
    # args.stage = 'pretrain'
    # data_list_train, data_list_eval = load_data(args)
    # print(f"uniprot: Len of training data: {len(data_list_train)}; Len of test data: {len(data_list_eval)}")
    
    # data_list_train, data_list_eval = load_data("SHS148k")
    # print(f"SHS148k: Len of training data: {len(data_list_train)}; Len of test data: {len(data_list_eval)}")

    # args.task = 'SHS27k'
    # data_list_train, data_list_eval = load_data(args)
    # print(f"SHS27k: Len of training data: {len(data_list_train)}; Len of test data: {len(data_list_eval)}")

    # args.task = 'CDRs'
    # data_list_train, data_list_eval = load_data(args)
    # print(f"CDRs: Len of training data: {len(data_list_train)}; Len of test data: {len(data_list_eval)}")
    