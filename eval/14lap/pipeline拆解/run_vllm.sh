#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export VLLM_USE_V1=0
export VLLM_USE_V0=1

models=(
'xx'
'xx'
)

temperature=0.3

for model_path in "${models[@]}"; do
    echo "model path:${model_path}"
    cmd="python pred_tup_vllm.py --model_path $model_path --temperature $temperature"

    echo "Running: $cmd"
    eval $cmd
    echo "run"
done