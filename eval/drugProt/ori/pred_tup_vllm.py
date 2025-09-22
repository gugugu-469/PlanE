import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import jsonlines
from vllm import LLM, SamplingParams
import json
import re
import random
import gc
from vllm.distributed.parallel_state import destroy_model_parallel
import argparse

inp_prompt = '''Currently, you are a senior expert in information extraction. 
Your task is to extract relational triplets from the given text. First, extract the subject-object entity pairs from the given text, and based on these subject-object entity pairs, identify the subject entity types and the object entity types from the given entity type list. The given entity type  list is: {entity_list}.
Then, based on the subject-object entity pairs and their corresponding entity types, extract the possible relations from the given relation list. The given relation list is: {relation_list}.
The output format for the task is (subject||subject type||relation||object||object type).
The given text:"{text}"
'''

with jsonlines.open('xx/ori/DrugProt/DrugProt_dev.jsonl', 'r') as f:
    dev_datas = [data for data in f]
with jsonlines.open('xx/ori/DrugProt/DrugProt_test.jsonl', 'r') as f:
    test_datas = [data for data in f]
with open('xx/ori/DrugProt/schemas.json','r') as f:
    trip_types_list = json.load(f)

trip_types = sorted(set([data['predicate'] for data in trip_types_list]))
ent_type_list = []
for data in trip_types_list:
    ent_type_list.append(data['subject_type'])
    ent_type_list.append(data['object_type'])
ent_types = sorted(set(ent_type_list),reverse=True)
ent_types = str(ent_types).replace("'",'"')
trip_types = str(trip_types).replace("'",'"')

def get_apply_prompts(tokenizer, prompt_list):
    processed_prompt_list = []
    for prompt in prompt_list:
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking = False,
            use_system_prompt = False
        )
        processed_prompt_list.append(text)
    return processed_prompt_list

def vllm_chat(tokenizer, model, prompt_list, sampling_params):
    outputs = model.generate(prompt_list, sampling_params)

    response_list = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        response_list.append(generated_text)
    return response_list


# 基本的输出路径
base_out_dir = './pred_result'
tp_size = torch.cuda.device_count()

def main():
    parser = argparse.ArgumentParser(description='这是一个文件处理程序')
    
    # 添加命令行参数
    parser.add_argument('--model_path', type=str, help='模型路径')
    parser.add_argument('--temperature', type=float,help='温度')
    
    # 解析命令行参数
    args = parser.parse_args()
    model_path = args.model_path
    temperature = args.temperature

    if 'checkpoint' in model_path:
        model_name = os.path.basename(model_path)
        model_name_2 = os.path.basename(os.path.dirname(model_path))
        model_name_3 = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        now_out_dir = os.path.join(base_out_dir, '{}_{}_{}_temp_{}'.format(model_name_3, model_name_2, model_name, temperature))
    else:
        model_name = os.path.basename(model_path)
        model_name_2 = os.path.basename(os.path.dirname(model_path))
        now_out_dir = os.path.join(base_out_dir, '{}_{}_temp_{}'.format(model_name_2, model_name, temperature))
    print('model_path:{}'.format(model_path))
    print('temperature:{}'.format(temperature))
    print('model_name:{}'.format(model_name))
    print('now_out_dir:{}'.format(now_out_dir))
    if not os.path.exists(now_out_dir):
        os.makedirs(now_out_dir)


    sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=1000)
    model = LLM(
        model_path,
        tensor_parallel_size = tp_size,
        gpu_memory_utilization = 0.85,
        max_model_len = 2000,
        trust_remote_code = True
        )
    tokenizer = model.get_tokenizer()


    # data_type_list = ['dev', 'test']
    # use_datas_list = [dev_datas, test_datas]
    data_type_list = ['dev']
    use_datas_list = [dev_datas]
    overall_prompt_list = [
        ('c_to_hrt', inp_prompt),
    ]
    for data_type, use_datas in zip(data_type_list, use_datas_list):
        for prompt_name, prompt_format in overall_prompt_list:
            out_name = '{}_{}_DrugProt.jsonl'.format(prompt_name, data_type)
            prompt_list = []
            text_list = []
            for data in use_datas:
                text = data['text']
                text_list.append(text)
                prompt = prompt_format.format(
                    relation_list=trip_types,
                    entity_list = ent_types,
                    text=data['text']
                )
                prompt_list.append(prompt)
            prompt_list = get_apply_prompts(tokenizer, prompt_list)
            res_list = vllm_chat(tokenizer, model, prompt_list, sampling_params)
            out_datas = []
            for text, prompt, pred in zip(text_list, prompt_list, res_list):
                out_datas.append({
                    'text': text,
                    'prompt': prompt,
                    'output': pred
                })
            with jsonlines.open(os.path.join(now_out_dir, out_name), 'w') as f:
                for data in out_datas:
                    f.write(data)


if __name__ == '__main__':
    main()