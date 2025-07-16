

accelerate launch  --config_file=./scripts/accelerate_zero2.yaml train_stage1.py \
                    --backbone=llama3 \
                    --learning_rate=4e-5 \
                    --epochs=3 \
                    --batch_size=2 \
                    --report_to=none \
                    --bf16 \
                    --lora_rank=256 \
                    --step_to_save=1000 \
                    --samples_to_update_gradient=512 \
                    --gpu_num=8 \
                    --task=stage1 \
                    --update_projector=1 \
                    --update_lora=0 \
                    --update_ge=0 \
                    --update_pe=0
