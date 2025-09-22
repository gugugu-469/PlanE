import logging
import os
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    set_seed,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
)
import os
from distutils.util import strtobool
import json
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from datetime import datetime
import logging
import random
import re 
import jsonlines
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from trl import DPOConfig, DPOTrainer, get_peft_config, ModelConfig, TrlParser
from peft import LoraConfig, get_peft_model

print('devices:{}'.format(torch.cuda.device_count()))
########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

########################
# Helper functions
########################
import re

def get_checkpoint(training_args: DPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def dpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: DPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print('dataset:{}'.format(script_args.dataset_id_or_path))
    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    with jsonlines.open(script_args.dataset_id_or_path, 'r') as f:
        datas = [data for data in f]
    train_dataset = Dataset.from_list(datas)

    #####################
    # Prepare and format dataset
    #####################

    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(ins, output):
        r1_prefix = [{
            "role": "user",
            "content": ins
          }]
        return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, add_generation_prompt=True, enable_thinking=False, use_system_prompt=False),'target': output}
    def format_dpo_sample(sample):
        prompt = tokenizer.apply_chat_template(
            [
                sample["chosen"][0]
            ],
            tokenize=False,
        )
        chosen = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False
        )
        rejected = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False
        )
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "score_chosen": sample['score_chosen'], 'score_rejected': sample['score_rejected']}
    train_dataset = train_dataset.map(
        format_dpo_sample, remove_columns=train_dataset.column_names
    )
    print('dataset sample:{}'.format(train_dataset[0]))
    # print('decode dataset sample:{}'.format(tokenizer.decode(dataset[0])))
    print(f"Columns: {train_dataset.features.keys()}")
    # train_dataset = train_dataset.select_columns(["prompt", "chosen", "rejected"])
    
    model_kwargs = dict(
        revision=model_args.model_revision,  # What revision from Huggingface to use, defaults to main
        trust_remote_code=model_args.trust_remote_code,  # Whether to trust the remote code, this also you to fine-tune custom architectures
        attn_implementation=model_args.attn_implementation,  # What attention implementation to use, defaults to flash_attention_2
        torch_dtype=(
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        ),  # What torch dtype to use, defaults to auto
        use_cache=False if training_args.gradient_checkpointing else True,  # Whether
        low_cpu_mem_usage=(
            True
            if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false"))
            else None
        ),  # Reduces memory usage on CPU for loading the model
    )

    # Check which training method to use and if 4-bit quantization is needed
    if model_args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
            bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
        )

    if model_args.use_peft:
        peft_config = get_peft_config(model_args)
    else:
        peft_config = None

    # Policy Model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    # Checks wether we use adapters for reference model or not
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(
              model_args.model_name_or_path, **model_kwargs
        )
    else:
        model_ref = None

    #########################
    # Instantiate DPO trainer
    #########################

    trainer = DPOTrainer(
      model=model,
      ref_model=model_ref,
      args=training_args,
      train_dataset=train_dataset,
      processing_class=tokenizer,
    #   eval_dataset=test_dataset,
      peft_config=peft_config,
    )


    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","dpo", "tutorial", "philschmid"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, DPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    model_args.trust_remote_code = True
    # Run the main training loop
    dpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()