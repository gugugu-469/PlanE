export WANDB_DISABLED=True
now_time=$(command date +%m-%d-%H-%M-%S)
echo "now time ${now_time}"
cd ../../../../
export CUDA_VISIBLE_DEVICES=3
# 
# 
llamafactory-cli train \
    --use_unsloth true \
    --stage sft \
    --model_name_or_path xxx \
    --do_train \
    --lora_rank 8 \
    --lora_alpha 16 \
    --dataset cmeie_no_chaijie_sft \
    --template qwen3 \
    --finetuning_type lora \
    --output_dir xxx \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 200 \
    --learning_rate 1e-4 \
    --num_train_epochs 4.0 \
    --plot_loss \
    --preprocessing_num_workers  48 \
    --bf16 \
    --cutoff_len 8100 \
    --ddp_timeout 180000 \
    --lora_target all



