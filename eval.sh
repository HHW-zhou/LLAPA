#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python eval.py \
                    --model_dir=model_weights/llapa3/stage2_llama3_3b_SHS27k_p/checkpoint-1500 \
                    --task=SHS27k \
                    --p_mode=p \
                    --load_pp=1 \
                    --load_pe=0 \
                    --load_ge=1 \
                    --load_peft=1 \
                    --lora_rank=128
