CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_neg_dpo_baseline.py --config receipes/qwen/neg_baseline_dpo_ace05_不拆解.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_neg_dpo_baseline.py --config receipes/qwen/neg_baseline_dpo_ace05_双向拆解.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_neg_dpo_baseline.py --config receipes/qwen/neg_baseline_dpo_ace05_pipeline拆解.yaml

