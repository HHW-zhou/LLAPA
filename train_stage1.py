import sys
import os
sys.path.append(os.path.abspath('../'))
print(sys.path)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第一个GPU

from transformers import Trainer, TrainingArguments
from transformers import LlamaForCausalLM

from utils import get_parse, setup_seed, setLogger, print_trainable_parameters, set_trainable_parameters
from load_data import load_data, load_UPPIN, load_protein_features
import logging  # 引入logging模块
from models.configuration_llapa import LlapaConfig
from models.llapa import LlapaForConditionalGeneration

from peft import LoraConfig
from accelerate import Accelerator
from transformers.models.auto import AutoTokenizer
from peft import get_peft_model, LoraConfig

from config.configs import ROOT_DIR, LLM_CONFIG_PATH

device_string = Accelerator().process_index

#logger
setLogger()
logger = logging.getLogger()

#args
args = get_parse()
setup_seed(args.seed)

# ---------------------------------  加载数据  --------------------------------------
train_dataset, test_dataset = load_data(args)

# ---------------------------------  加载模型  --------------------------------------
config = LlapaConfig.from_pretrained(LLM_CONFIG_PATH)
model = LlapaForConditionalGeneration(config)

# ---------------------------------  加载pllm  --------------------------------------
llm_tokenizer = AutoTokenizer.from_pretrained(config.text_config._name_or_path, truncation_side="right")
if llm_tokenizer.pad_token is None:
    llm_tokenizer.add_special_tokens({"pad_token":"[PAD]"})
llm_tokenizer.padding_side = 'right'           

llm_tokenizer.add_tokens(['<|proteinHere|>'])
# llm_tokenizer.add_tokens(['<|topoHere|>'])
model.set_llm_tokenizer(llm_tokenizer)

llm = LlamaForCausalLM.from_pretrained(config.text_config._name_or_path)
vocab_size = len(llm_tokenizer)
llm.resize_token_embeddings(vocab_size)

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

set_trainable_parameters(model, args)
# ---------------------------------  训练模型  --------------------------------------
def collate_fn(batch):
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
        texts.append((i, llm_tokenizer.bos_token + data['question'] + data['answer'] + llm_tokenizer.eos_token))
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
    text_input_dict = llm_tokenizer(text_list, padding='max_length', max_length=1024, add_special_tokens=False, truncation=True)
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

training_args = TrainingArguments(
    do_eval=False,
    output_dir=os.path.join(ROOT_DIR, "model_weights/llapa3/stage1_llama3_3b"),
    evaluation_strategy="no",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    log_level="debug",
    optim="paged_adamw_32bit",
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    save_steps=args.step_to_save,
    logging_steps=1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    fp16=args.fp16,               # 混合精度情况下模型内存消耗多0.5倍（相当于多加载了一个float16的模型）
    bf16=args.bf16,              # bf16精度小于fp16
    logging_dir='./log',     # 日志目录
    report_to= args.report_to,
    lr_scheduler_type=args.lr_scheduler_type,                   # constant/cosine
    gradient_checkpointing = False,
    ddp_find_unused_parameters = False,
    logging_strategy='steps',
    remove_unused_columns=False                                  # 很重要，不然会移除自定义的columns
)


class LlapaTrainer(Trainer):
    def save_model(self, output_dir = None, _internal_call=True):
            """
            Will save the model, so you can reload it using `from_pretrained()`.

            Will only save from the main process.
            """

            os.makedirs(output_dir, exist_ok=True)

            output_dir_pp = os.path.join(output_dir,'pp')
            os.makedirs(output_dir_pp, exist_ok=True)

            # output_dir_llm = os.path.join(output_dir,'peft')
            # os.makedirs(output_dir_llm, exist_ok=True)

            # output_dir_pe = os.path.join(output_dir,'pe')
            # os.makedirs(output_dir_pe, exist_ok=True)

            # output_dir_ge = os.path.join(output_dir,'ge')
            # os.makedirs(output_dir_ge, exist_ok=True)

            if self.is_deepspeed_enabled:
                try:
                    ################################# save protein projector ######################################
                    pp_state_dict = self.accelerator.get_state_dict(self.deepspeed.protein_projector)
                    if self.args.should_save:
                        self._save(output_dir_pp, state_dict=pp_state_dict)

                except ValueError:
                    logger.warning(
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    if self.args.should_save:
                        self._save(output_dir, state_dict={})

# 创建训练器
trainer = LlapaTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=test_dataset,
    # optimizers=optimizer,
    data_collator=collate_fn
)

# print(model)
print_trainable_parameters(trainer.model)

# 开始训练
trainer.train()