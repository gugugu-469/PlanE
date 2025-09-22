export CUDA_VISIBLE_DEVICES=2
cd ../LLaMA-Factory



src_path=(

)


export_path=(

)


for i in "${!src_path[@]}"; do
    echo "src_path:${src_path[$i]}"
    python src/export_model.py \
        --model_name_or_path xxx \
        --adapter_name_or_path "${src_path[$i]}" \
        --template llama3 \
        --finetuning_type lora \
        --export_dir "${export_path[$i]}" \
        --export_size 3 \
        --export_legacy_format False
done
