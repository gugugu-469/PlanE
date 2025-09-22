CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_14lap.py --config receipes/14lap_qwen3_nochaijie.yaml


CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_14lap.py --config receipes/14lap_qwen3_pipeline.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_14lap.py --config receipes/14lap_qwen3_bidirection.yaml



