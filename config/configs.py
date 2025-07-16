import os


ROOT_DIR = 'YOUR WORK PATH'
DATA_DIR = 'YOUR DATASET PATH'
LLM_CONFIG_PATH = os.path.join(ROOT_DIR, 'LLAPA/config/llama3-3b-backbone')

ROOT_DIR = '/mnt/ai4s_ceph_share/neogaryzhou'
DATA_DIR = '/mnt/ai4s_ceph_share/neogaryzhou/LLaPA_datasets'
LLM_CONFIG_PATH = os.path.join(ROOT_DIR, 'lla-pa3/config/llama3-3b-backbone')

WORKERS = 100
MAX_SEQ_LEN = 1024
MIN_SEQ_LEN = 10
MAX_TEXT_LEN = 2000
QUESTION_PREFIX = "### Human: "
ANSWER_PREFIX = " \n### Assistant: "