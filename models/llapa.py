# coding=utf-8

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import os
import random
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from transformers import PreTrainedModel, AutoTokenizer
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from .configuration_llapa import LlapaConfig
from torch_geometric.nn import GINConv, GraphNorm, SGConv, GCNConv
from transformers.generation import TopKLogitsWarper, TopPLogitsWarper


logger = logging.get_logger(__name__)


def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    return logits


@dataclass
class LlapaCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class LlapaMultiProteinProjector(nn.Module):
    def __init__(self, config: LlapaConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.protein_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, input_features):
        hidden_states = self.linear_1(input_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
    
class LlapaMultiResidueProjector(nn.Module):
    def __init__(self, config: LlapaConfig):
        super().__init__()

        self.linear = nn.Linear(config.protein_config.hidden_size, 1, bias=False)
        
        self.config = config

    def forward(self, input_features):
        hidden_states = self.linear(input_features)
        hidden_states = hidden_states.squeeze(-1)       # ==> batch, max_amino
        
        llm_like_embeddings = torch.zeros(hidden_states.shape[0], self.config.text_config.hidden_size)
        llm_like_embeddings[:,:hidden_states.shape[1]] = hidden_states
        # print(llm_like_embeddings)
        # print(llm_like_embeddings.shape)
        return llm_like_embeddings

class LlapaPreTrainedModel(PreTrainedModel):
    config_class = LlapaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.llm._supports_sdpa

class LlapaForConditionalGeneration(LlapaPreTrainedModel):
    def __init__(self, config: LlapaConfig):
        super().__init__(config)

        self.protein_tokenizer = None
        self.protein_encoder = None

        if config.p_mode == 'p':
            self.protein_projector = LlapaMultiProteinProjector(config)
        elif config.p_mode == 'r':
            self.protein_projector = LlapaMultiResidueProjector(config)

        self.graph_encoder = SGC(config)
        # self.graph_encoder = GIN(config)

        self.protein_features = None
        self.graph = None

        self.protein_token_id = None
        self.protein_idx_token_id = None

        # self.CTloss = ContrastiveLoss()
        self.CTloss = MaximalMutualInformationLoss()

        self.config = config
        
        self.cache_protein_features = {}
        
        self.post_init()

    def init_protein_encoder(self):
        if self.protein_encoder is None or self.protein_tokenizer is None:
            self.protein_tokenizer = AutoTokenizer.from_pretrained(self.config.protein_config._name_or_path)
            self.protein_encoder = AutoModel.from_pretrained(self.config.protein_config._name_or_path)
                
    def set_graph(self, ppi_list):
        if ppi_list is None:
            return False

        self.edge_index = torch.tensor(ppi_list, dtype=torch.long).t().to(self.device)

    def set_protein_features(self, protein_features):
        if protein_features is None:
            return False

        self.protein_features = protein_features.to(self.device)
        
    def set_cache_protein_features(self, protein_features, residue_features, cache_key):

        self.cache_protein_features[cache_key] = {
            'protein_features':protein_features,
            'residue_features':residue_features
        }

    def set_protein_tokenizer(self, tokenizer):
        self.protein_tokenizer = tokenizer

    def set_llm_tokenizer(self, tokenizer):
        self.llm_tokenizer = tokenizer
        self.pad_token_id = self.llm_tokenizer.pad_token_id

        protein_tokens = tokenizer.encode("<|proteinHere|>", add_special_tokens=False)
        protein_idx_tokens = tokenizer.encode("<|reserved_special_token_0|>", add_special_tokens=False)

        if len(protein_tokens) != 1:
            raise NotImplementedError
        
        if len(protein_idx_tokens) != 1:
            raise NotImplementedError

        self.protein_token_id = protein_tokens[0]
        self.protein_idx_token_id = protein_idx_tokens[0]

    def set_protein_encoder(self, protein_encoder):
        self.protein_encoder = protein_encoder

    def set_llm(self, llm):
        self.llm = llm

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.llm.set_decoder(decoder)

    def get_decoder(self):
        return self.llm.get_decoder()

    def tie_weights(self):
        pass
        # return self.llm.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.llm.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def get_list_according_to_belong(self, belong_list, item_list):
        groups = {}
        for key, value in zip(belong_list, item_list):
            if key not in groups:
                groups[key] = []
            groups[key].append(value)

        result = list(groups.values())
        return result

    def encode_graph(self, protein_idx_list):
        if self.protein_features.device != self.device:
            self.protein_features = self.protein_features.to(self.device)

        if self.edge_index.device != self.device:
            self.edge_index = self.edge_index.to(self.device)

        graph_embeddings = self.graph_encoder(self.protein_features, self.edge_index)

        idx = protein_idx_list

        return graph_embeddings[idx]
    
    def load_cache(self, cache_list, proteins_belong):
        protein_features = []

        for cache in cache_list:
            cache_key = cache[0]
            cache_idx = cache[1]

            cache_dir = os.path.join(self.config.DATA_DIR, f"dataset/processed_data/{cache_key}")
            cache_file_name = os.path.join(cache_dir, f"{cache_idx}_{self.config.p_mode}.pt")
            
            pf = torch.load(cache_file_name).to(dtype=self.dtype).to(self.device)

            protein_features.append(pf)

        protein_features = torch.stack(protein_features)
        projected_protein_features = self.protein_projector(protein_features)

        return self.get_list_according_to_belong(proteins_belong, projected_protein_features)

    def encode_proteins(self, cache_list, proteins_belong, proteins):
        assert len(cache_list) == len(proteins)
        
        # if self.protein_encoder is None or self.protein_tokenizer is None:
        #     self.init_protein_encoder()

        def __average_pool(residue_feature: torch.FloatTensor, residue_masks: torch.FloatTensor):
            lens = residue_masks.sum(dim=1)
            pooled = (residue_feature * residue_masks[:,:,None]).sum(dim=1) / lens[:, None]
            return pooled

        protein_list = [item[1] for item in proteins]

        protein_input_dict = self.protein_tokenizer(protein_list, padding='max_length', max_length=1024, truncation=True)
        protein_input_ids = protein_input_dict['input_ids']
        protein_attention_mask = protein_input_dict['attention_mask']

        protein_input_ids = torch.tensor(protein_input_ids).to(self.device)
        protein_attention_mask = torch.tensor(protein_attention_mask).to(torch.bool).to(self.device)

        with torch.no_grad():
            outputs = self.protein_encoder(input_ids=protein_input_ids, attention_mask=protein_attention_mask)

        residue_features = outputs['last_hidden_state']*protein_attention_mask[:,:,None]
        protein_features = __average_pool(residue_features, protein_attention_mask)


        # # save
        # for i in range(len(cache_list)):
        #     cache_key = cache_list[i][0]
        #     cache_idx = cache_list[i][1]
        #     p_features = protein_features[i]
        #     r_features = residue_features[i]

        #     p_dir = os.path.join(self.config.DATA_DIR,f'dataset/processed_data/{cache_key}')
        #     if not os.path.exists(p_dir):
        #         os.mkdir(p_dir)
        #     p_fname = os.path.join(p_dir, f'{cache_idx}_p.pt')
        #     torch.save(p_features, p_fname)

        #     r_fname = os.path.join(p_dir, f'{cache_idx}_r.pt')
        #     torch.save(r_features, r_fname)

        if self.config.p_mode == 'p':
            # print("*"*150)
            # print(protein_features)
            # print(protein_features.shape, protein_features.dtype, protein_features.device)
            # print("*"*150)
            
            # protein_features = torch.stack(protein_features)
            protein_features = protein_features.to(dtype=self.dtype)
            projected_protein_features = self.protein_projector(protein_features)
        elif self.config.p_mode == 'r':
            residue_features = residue_features.to(dtype=self.dtype).to(self.device)
            projected_protein_features = self.protein_projector(residue_features)
            
            projected_protein_features = projected_protein_features.to(dtype=self.dtype).to(self.device)

        return self.get_list_according_to_belong(proteins_belong, projected_protein_features)
    
    def fetch_proteins(self, cache_list, cache_belong, proteins):
        try:
            protein_embeddings = self.load_cache(cache_list, cache_belong)
            # print("--------------------------------------------------------> load cache protein succeed!")
        except Exception as e:
            # print(e)
            protein_embeddings = self.encode_proteins(cache_list, cache_belong, proteins)
            # print("==================> encode protein from scratch!")

        return protein_embeddings

    def encode_texts(self, input_ids, attention_mask, texts_belong):
        text_embedding = self.get_embedding(input_ids)
        # return self.get_list_according_to_belong(texts_belong, text_embedding)
        return text_embedding

    def encode_batch(self, text_input_ids, text_attention_mask, texts_belong, 
                            cache_list, cache_belong, ppi_list, ppi_belong, proteins):
        
        text_input_ids = torch.tensor(text_input_ids).to(self.device)
        text_attention_mask = torch.tensor(text_attention_mask).to(torch.bool).to(self.device)
        text_embeddings = self.encode_texts(text_input_ids, text_attention_mask, texts_belong)

        #####################################################################################################
        number_of_p_to_grad = 5
        
        if len(cache_belong) > 0:
            protein_embeddings = self.fetch_proteins(cache_list, cache_belong, proteins)
            
            for record in protein_embeddings:
                if len(record) > number_of_p_to_grad:
                    
                    valid_indexes = random.sample([i for i in range(len(record))], number_of_p_to_grad)
                    
                    for i in range(len(record)):
                        if i not in valid_indexes:
                            record[i] = record[i].detach()


        if len(ppi_belong) > 0:
            topo_embeddings = self.encode_graph(ppi_list)
            topo_embeddings = self.get_list_according_to_belong(ppi_belong, topo_embeddings)
            
            for record in topo_embeddings:
                if len(record) > number_of_p_to_grad:
                    
                    valid_indexes = random.sample([i for i in range(len(record))], number_of_p_to_grad)
                    
                    for i in range(len(record)):
                        if i not in valid_indexes:
                            record[i] = record[i].detach()

        contrast_loss = 0
        if len(cache_belong) > 0 and len(ppi_belong) > 0 and len(protein_embeddings)>1 and len(protein_embeddings) == len(topo_embeddings):
            contrast_loss = self.CTloss(protein_embeddings, topo_embeddings)
        else:
            pass


        if len(cache_belong) > 0:
            for i in range(text_input_ids.shape[0]):
                input_ids = text_input_ids[i]
                indices = [index for index, value in enumerate(input_ids) if value == self.protein_token_id]

                # print("========> ", indices)
                
                # print("*"*150)
                # print("number of protein_embeddings: ", len(protein_embeddings[i]))
                # print("number of protein_embeddings: ", len(indices))
                # print("indices: ", indices)
                # assert len(protein_embeddings[i]) == len(indices)
                # print("*"*150)

                for index in indices:
                    protein_embedding = protein_embeddings[i].pop(0).unsqueeze(0)
                    text_embeddings[i][index] = protein_embedding

        if len(ppi_belong) > 0:
            for i in range(text_input_ids.shape[0]):
                input_ids = text_input_ids[i]
                indices = [index for index, value in enumerate(input_ids) if value == self.protein_idx_token_id]

                # print("*"*150)
                # print("ppi_list: ", ppi_list)
                # print("number of topo embeddings: ", len(topo_embeddings[i]))
                # print("number of topo indices: ", len(indices))
                # print("indices: ", indices)
                # assert len(topo_embeddings[i]) == len(indices)
                # print("*"*150)

                for index in indices:
                    topo_embedding = topo_embeddings[i].pop(0)
                    text_embeddings[i][index] = topo_embedding

        labels = text_input_ids.clone()
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100
        if self.protein_token_id is not None:
            labels[labels == self.protein_token_id] = -100
        if self.protein_idx_token_id is not None:
            labels[labels == self.protein_idx_token_id] = -100

        #sft
        # for label in labels:
        #     pass

        return text_input_ids, text_embeddings, labels, text_attention_mask, contrast_loss

    def forward(self, text_input_ids, text_attention_mask, texts_belong, 
                cache_list, cache_belong, ppi_list, ppi_belong, proteins) -> Union[Tuple, LlapaCausalLMOutputWithPast]:

        inputs_ids, inputs_embeds, labels, attention_mask, contrast_loss = self.encode_batch(text_input_ids, 
                                                                              text_attention_mask, 
                                                                              texts_belong, 
                                                                              cache_list, 
                                                                              cache_belong, 
                                                                              ppi_list,
                                                                              ppi_belong,
                                                                              proteins)

        # print("inputs_embeds shape: ", inputs_embeds.shape)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )


        logits = outputs['logits']
        loss = outputs['loss']


        # print("===============================> loss: {}, contrast_loss: {}".format(loss, contrast_loss))

        loss = loss + contrast_loss

        return LlapaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

    def generate(self, text_input_ids, text_attention_mask, texts_belong, 
                cache_list, cache_belong, ppi_list, ppi_belong):
        # proteins = [(0,"DILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRG")]
        # texts = [(0,"### Human: The structure of this protein is "),(0,"; what is its sequence? \n### Assistant: ")]

        # inputs_ids, inputs_embeds, labels, attention_mask = self.encode_batch(batch)
        inputs_ids, inputs_embeds, labels, attention_mask, contrast_loss = self.encode_batch(text_input_ids, 
                                                                              text_attention_mask, 
                                                                              texts_belong, 
                                                                              cache_list, 
                                                                              cache_belong, 
                                                                              ppi_list,
                                                                              ppi_belong)

        next_token = -100
        results = []
        for i in range(inputs_embeds.shape[0]):
            inputs_embed = inputs_embeds[i]
            mask = attention_mask[i]
            inputs_embed = inputs_embed[mask]

            generated_tokens = []
            while True:
                if next_token == self.llm_tokenizer.pad_token_id or next_token == self.llm_tokenizer.eos_token_id or len(generated_tokens) > 512:
                    break

                # print("==========> ", inputs_embed.shape, inputs_embed.unsqueeze(0).shape)
                with torch.no_grad():               # 很重要，推理一定要加，不然爆显存
                    outputs = self.llm(inputs_embeds=inputs_embed.unsqueeze(0))

                # hidden_states = outputs[0]
                # last_hidden_states = hidden_states[:,-1,:]
                # logits = self.llm.lm_head(hidden_states)

                logits = outputs['logits']

                # print("=====> logits: ", logits)

                next_token_logits = logits[:,-1,:]                  # last_hidden_states
                filtered_next_token_logits = top_k_top_p_filtering(next_token_logits,top_k=50,top_p=0.95)
                # filtered_next_token_logits = top_k_top_p_filtering(next_token_logits,top_k=1,top_p=0.95)
                probs = F.softmax(filtered_next_token_logits,dim=-1)
                next_token = torch.multinomial(probs,num_samples=1).item()

                generated_tokens.append(next_token)

                # print(next_token, len(generated_tokens))

                # print("========================> ", next_token)

                # 拼接 next_token 的embedding到原始输入
                # inputs_embeds = torch.cat([inputs_embeds,last_hidden_states],dim=0)     # 这种方式不好,一是跟训练不一致, 二是hidden state的向量空间和embedding的不一定一样
                new_embedding = self.get_embedding(torch.tensor([next_token]))
                inputs_embed = torch.cat([inputs_embed,new_embedding],dim=0)

            results.append(self.llm_tokenizer.decode(generated_tokens, skip_special_tokens=True))

        return results
    
    def generate_v0(self, text_input_ids, text_attention_mask, texts_belong, 
                cache_list, cache_belong, ppi_list, ppi_belong):
        # proteins = [(0,"DILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRG")]
        # texts = [(0,"### Human: The structure of this protein is "),(0,"; what is its sequence? \n### Assistant: ")]

        # inputs_ids, inputs_embeds, labels, attention_mask = self.encode_batch(batch)
        inputs_ids, inputs_embeds, labels, attention_mask, contrast_loss = self.encode_batch(text_input_ids, 
                                                                              text_attention_mask, 
                                                                              texts_belong, 
                                                                              cache_list, 
                                                                              cache_belong, 
                                                                              ppi_list,
                                                                              ppi_belong)
        
        generation_output = self.llm.generate(
                inputs_embeds=inputs_embeds[0][attention_mask[0]].unsqueeze(dim=0),
                max_new_tokens=1024,
                do_sample=True,   
                temperature=1,
                top_k=20, 
                top_p=0.95, 
                num_return_sequences = 10,
                pad_token_id = self.llm_tokenizer.pad_token_id,
                return_dict_in_generate=True
        )

        outputs = []
        for seq in generation_output.sequences:
            output = self.llm_tokenizer.decode(seq, skip_special_tokens=True)
            # output = self.llm_tokenizer.decode(seq, skip_special_tokens=False)
            # print("\n", output)
            outputs.append(output)

        return outputs

    def generate_v1(self, text_input_ids, text_attention_mask, texts_belong, 
                cache_list, cache_belong, ppi_list, ppi_belong, proteins):
        # proteins = [(0,"DILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRG")]
        # texts = [(0,"### Human: The structure of this protein is "),(0,"; what is its sequence? \n### Assistant: ")]

        # inputs_ids, inputs_embeds, labels, attention_mask = self.encode_batch(batch)
        inputs_ids, inputs_embeds, labels, attention_mask, contrast_loss = self.encode_batch(text_input_ids, 
                                                                        text_attention_mask, 
                                                                        texts_belong, 
                                                                        cache_list, 
                                                                        cache_belong, 
                                                                        ppi_list,
                                                                        ppi_belong,
                                                                        proteins)
        
        generation_output = self.llm.generate(
                inputs_embeds=inputs_embeds[0][attention_mask[0]].unsqueeze(dim=0),
                max_new_tokens=1024,
                do_sample=False,   
                # temperature=1,
                # top_k=10, 
                # top_p=0.95, 
                num_return_sequences = 1,
                pad_token_id = self.llm_tokenizer.pad_token_id,
                return_dict_in_generate=True
        )

        for seq in generation_output.sequences:
            output = self.llm_tokenizer.decode(seq, skip_special_tokens=True)
            # output = self.llm_tokenizer.decode(seq, skip_special_tokens=False)
            # print("\n", output)

        return output
    
    def conversation(self, text_input_ids, text_attention_mask, texts_belong, 
                            protein_input_ids, protein_attention_mask, proteins_belong, protein_idx_list=None, protein_idx_belong=None):
        # proteins = [(0,"DILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRG")]
        # texts = [(0,"### Human: The structure of this protein is "),(0,"; what is its sequence? \n### Assistant: ")]

        # inputs_ids, inputs_embeds, labels, attention_mask = self.encode_batch(batch)
        inputs_ids, inputs_embeds, labels, attention_mask, contrast_loss = self.encode_batch(text_input_ids, text_attention_mask, texts_belong, 
                                                                                protein_input_ids, protein_attention_mask, proteins_belong,protein_idx_list,protein_idx_belong)

        # print("============================> inputs_embeds.shape ", inputs_embeds.shape)
        # print("============================> inputs_embeds[attention_mask].shape ", inputs_embeds[0][attention_mask[0]].unsqueeze(dim=0).shape)
        generation_output = self.llm.generate(
                inputs_embeds=inputs_embeds[0][attention_mask[0]].unsqueeze(dim=0),
                max_new_tokens=512,
                do_sample=False,   
                # temperature=1,
                # top_k=10, 
                # top_p=0.95, 
                num_return_sequences = 1,
                pad_token_id = self.llm_tokenizer.pad_token_id,
                return_dict_in_generate=True
        )

        for seq in generation_output.sequences:
            output = self.llm_tokenizer.decode(seq, skip_special_tokens=True)
            # print("\n", output)

        return output
    
    def get_embedding(self, input_ids):
        input_ids = torch.tensor(input_ids) if isinstance(input_ids, list) else input_ids

        if self.config.backbone == 'llama2':
            embeddings = self.llm.base_model.model.model.embed_tokens(input_ids.to(self.device))
            
        elif self.config.backbone == 'llama3':
            embeddings = self.llm.base_model.model.model.embed_tokens(input_ids.to(self.device))

        elif self.config.backbone == 'gpt2':
            embeddings = self.llm.transformer.wte(input_ids.to(self.device))
        
        return embeddings
    

class SGC(torch.nn.Module):
    def __init__(self, config):
        super(SGC, self).__init__()

        self.batchLayer = GraphNorm(config.protein_config.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.convs = nn.ModuleList()
        
        self.ac = nn.ReLU()
        self.convs.append(SGConv(config.protein_config.hidden_size, config.protein_config.hidden_size))
        self.convs.append(SGConv(config.protein_config.hidden_size, config.protein_config.hidden_size))
        
        self.projector = LlapaMultiProteinProjector(config)
        
    def forward(self, x, edge_index):
        # x = x.to(torch.float32)
        x = self.batchLayer(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.ac(x)
            x = self.dropout(x)

        x = self.projector(x)
        return x
    
    
class GIN(torch.nn.Module):
    def __init__(self, config):
        super(GIN, self).__init__()

        self.batchLayer = GraphNorm(config.protein_config.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.convs = nn.ModuleList()

        ############################################### GIN ###################################################
        self.convs.append(GINConv(nn.Sequential(nn.Linear(config.protein_config.hidden_size, config.protein_config.hidden_size), 
                                                 nn.ReLU(), 
                                                 nn.Linear(config.protein_config.hidden_size, config.protein_config.hidden_size), 
                                                 nn.ReLU(), 
                                                 nn.BatchNorm1d(config.protein_config.hidden_size))))
        
        self.convs.append(GINConv(nn.Sequential(nn.Linear(config.protein_config.hidden_size, config.protein_config.hidden_size), 
                                                 nn.ReLU(), 
                                                 nn.Linear(config.protein_config.hidden_size, config.protein_config.hidden_size), 
                                                 nn.ReLU(), 
                                                 nn.BatchNorm1d(config.protein_config.hidden_size))))
        ##########################################################################################################
        self.projector = LlapaMultiProteinProjector(config)
        
    def forward(self, x, edge_index):
        # x = x.to(torch.float32)
        x = self.batchLayer(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.dropout(x)

        x = self.projector(x)
        return x
    
    
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, a, b):
        batch_size = len(a)

        positive_distances = []
        negative_distances = []
        for i in range(batch_size):
            N = len(a[i])
            Nb = len(b[i])

            if N != Nb:
                continue
            
            for j in range(N):
                positive_distances.append(torch.norm(a[i][j].clone().detach() - b[i][j]))

            for j in range(N):
                for k in range(N):
                    if j != k:
                        negative_distances.append(torch.norm(a[i][j].clone().detach() - b[i][k]))

        positive_distances = torch.stack(negative_distances)
        negative_distances = torch.stack(negative_distances)
        
        # contrastive loss
        positive_loss = torch.mean(positive_distances ** 2)
        negative_loss = torch.mean(F.relu(self.margin - negative_distances) ** 2)
        
        loss = positive_loss + negative_loss
        return loss

# class MaximalMutualInformationLoss(nn.Module):
#     def __init__(self):
#         super(MaximalMutualInformationLoss, self).__init__()
 
#     def forward(self, protein_embeddings, topo_embeddings):
#         batch_number = len(protein_embeddings)    
#         total_loss = 0
#         for i in range(batch_number):
#             device = protein_embeddings[i][0].device
            
#             tmp_protein_embeddings = torch.stack(protein_embeddings[i]).clone().detach().to(device)
#             tmp_topo_embeddings = torch.stack(topo_embeddings[i]).to(device)

#             if tmp_protein_embeddings.size(0) != tmp_topo_embeddings.size(0):
#                 continue

#             batch_size = tmp_protein_embeddings.size(0)
            
#             # 计算相似度矩阵
#             scores = torch.matmul(tmp_protein_embeddings, tmp_topo_embeddings.T)
#             labels = torch.arange(batch_size).to(device)
#             loss = F.cross_entropy(scores, labels)
            
#             total_loss = total_loss + loss

#         total_loss = total_loss/batch_number

#         if torch.isnan(total_loss) or torch.isinf(total_loss):
#             print(protein_embeddings)
#             print(topo_embeddings)
#             return 0

#         return total_loss
    
class MaximalMutualInformationLoss(nn.Module):
    def __init__(self):
        super(MaximalMutualInformationLoss, self).__init__()
 
    def forward(self, protein_embeddings, topo_embeddings):
        batch_number = len(protein_embeddings)

        total_loss = 0
        for i in range(batch_number):
            
            tmp_protein_embeddings = torch.stack(protein_embeddings[i]).clone().detach()
            tmp_topo_embeddings = torch.stack(topo_embeddings[i])

            if tmp_protein_embeddings.size(0) != tmp_topo_embeddings.size(0):
                continue

            batch_size = tmp_protein_embeddings.size(0)
            
            # 计算相似度矩阵
            scores = torch.matmul(tmp_protein_embeddings, tmp_topo_embeddings.T)
            labels = torch.arange(batch_size).cuda()
            loss = F.cross_entropy(scores, labels)
            
            total_loss = total_loss + loss

        total_loss = total_loss/batch_number

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(protein_embeddings)
            print(topo_embeddings)
            return 0

        return total_loss
 
        
        