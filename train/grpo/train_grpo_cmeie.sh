CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_cmeie.py --config receipes/cmeie_qwen3_nochaijie.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_cmeie.py --config receipes/cmeie_qwen3_pipeline.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_cmeie.py --config receipes/cmeie_qwen3_bidirection.yaml


CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_cmeie.py --config receipes/cmeie_llama3_nochaijie.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_cmeie.py --config receipes/cmeie_llama3_pipeline.yaml


CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_cmeie.py --config receipes/cmeie_llama3_bidirection.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_cmeie.py --config receipes/cmeie_glm_nochaijie.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_cmeie.py --config receipes/cmeie_glm_pipeline.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_cmeie.py --config receipes/cmeie_glm_bidirection.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_cmeie.py --config receipes/cmeie_intern3_nochaijie.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_cmeie.py --config receipes/cmeie_intern3_pipeline.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_cmeie.py --config receipes/cmeie_intern3_bidirection.yaml
