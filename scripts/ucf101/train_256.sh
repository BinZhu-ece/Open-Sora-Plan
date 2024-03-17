
accelerate launch \
    --config_file scripts/accelerate_configs/ddp_config.yaml \
    opensora/train/train.py \
    --model Latte-XL/122 \
    --dataset sky \
    --ae stabilityai/sd-vae-ft-mse \
    --data_path /remote-home/yeyang/UCF-101 \
    --extras 2 \
    --sample_rate 3 \
    --num_frames 16 \
    --max_image_size 256 \
    --gradient_checkpointing \
    --attention_mode flash \
    --train_batch_size=5 --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="tensorboard" \
    --checkpointing_steps=500 \
    --output_dir="ucf101-f16s3-256-imgvae188-bf16-ckpt-flash"