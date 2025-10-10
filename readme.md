# PlanE: Meta Planning of Data, Tuning, and Inference for Extractive-based LLMs

This repository contains the code and resources for the paper **"PlanE: Meta Planning of Data, Tuning, and Inference for Extractive-based LLMs"**. The project focuses on optimizing the combination of data, training, and inference strategies for large language models (LLMs) in information extraction tasks.

---

## Repository Structure

The repository is organized as follows:

```plaintext
.
├── analysis_exp/          # Results analysis scripts and outputs
├── datas/                 # Datasets used for training and evaluation
├── eval/                  # Code for model evaluation
├── llm_models/            # Pre-trained base LLMs
├── train/                 # Training code and scripts
├── trained_models/        # Checkpoints of trained models
└── README.md              # This file
```

### Directory Descriptions

1. **`analysis_exp/`**:

   - Contains scripts and notebooks for analyzing experimental results.
   - Includes visualizations, statistical analyses, and performance metrics.
2. **`datas/`**:

   - Stores datasets used for training and evaluation.
   - Each dataset is organized into subdirectories with clear naming conventions.
3. **`eval/`**:

   - Contains code for evaluating trained models.
   - Includes scripts for computing metrics such as F1 score, precision, and recall.
4. **`llm_models/`**:

   - Houses pre-trained base LLMs used in the experiments.
   - Models are stored in a format compatible with the training pipeline.
5. **`train/`**:

   - Includes scripts and configurations for training models.
   - Supports various training strategies and hyperparameter configurations.
6. **`trained_models/`**:

   - Stores checkpoints of trained models.
   - Each checkpoint is labeled with the corresponding experiment ID and timestamp.

---

## Environment Configuration for SFT

| Package           | Version     |
| ----------------- | ----------- |
| `torch`         | 2.5.1+cu121 |
| `transformers`  | 4.51.3      |
| `datasets`      | 3.2.0       |
| `peft`          | 0.15.1      |
| `accelerate`    | 1.2.1       |
| `sentencepiece` | 0.2.0       |
| `llamafactory`  | 0.9.3.dev0  |
| `unsloth`       | 2025.3.6    |

## Environment Configuration for GRPO, DPO and KTO

| Package           | Version     |
| ----------------- | ----------- |
| `torch`         | 2.5.1+cu121 |
| `transformers`  | 4.51.3      |
| `datasets`      | 3.1.0       |
| `peft`          | 0.15.2      |
| `trl`           | 0.17.0      |
| `accelerate`    | 1.3.0       |
| `sentencepiece` | 0.2.0       |

## Usage

### Training

To train a model, navigate to the `train/` directory and run the appropriate script:

SFT

```bash
cd ./train/sft/scripts/cmeie_glm4
bash no_chaijie_sft.sh
```

GRPO

```bash
cd ./train/grpo
bash train_grpo_cmeie.sh
```

DPO

```bash
cd ./train/dpo
bash train_dpo_cmeie.sh
```

KTO

```bash
cd ./train/kto
bash train_kto_cmeie.sh
```

### Evaluation

To evaluate a trained model, use the scripts in the `eval/` directory:

```bash
cd ./eval/ace05/bidirection
bash run_vllm.sh
```

### Analysis

For result analysis, explore the notebooks in the `analysis_exp/` directory.
