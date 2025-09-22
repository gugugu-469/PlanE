#!/bin/bash
export CUDA_VISIBLE_DEVICES="4"
export VLLM_USE_V1=0
export VLLM_USE_V0=1

models=(

)


# 定义温度列表
temperatures=(0.3)
# temperatures=(0.1 0.3 0.5 0.7)

# 三重循环
for temperature in "${temperatures[@]}"; do
    for model_path in "${models[@]}"; do
        # 构造命令
        echo "model path:${model_path}"
        cmd="python pred_tup_vllm.py --model_path $model_path --temperature $temperature"

        # 打印或执行命令
        echo "Running: $cmd"
        eval $cmd
        echo "run"
    done
done
