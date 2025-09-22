import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

src_path=[

]
export_path=[

]
model_name = "xxx"



for lora_model_path, save_path in zip(src_path, export_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

    merged_model = lora_model.merge_and_unload()

    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"模型已合并并保存到: {save_path}")