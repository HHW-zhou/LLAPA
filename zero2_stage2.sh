

accelerate launch  --config_file=./scripts/accelerate_zero2.yaml train_stage2.py \
                    --backbone=llama3 \
                    --learning_rate=4e-5 \
                    --epochs=20 \
                    --batch_size=1 \
                    --report_to=none \
                    --bf16 \
                    --lora_rank=128 \
                    --step_to_save=100 \
                    --samples_to_update_gradient=32 \
                    --gpu_num=8 \
                    --update_projector=1 \
                    --update_lora=1 \
                    --update_ge=1 \
                    --update_pe=0 \
                    --model_dir=model_weights/llapa3/stage1_llama3-3b/checkpoint-3000 \
                    --load_pp=1 \
                    --load_peft=0 \
                    --load_pe=0 \
                    --load_ge=0 \
                    --task=SHS27k \
                    --p_mode=p \
                    --stage=ft