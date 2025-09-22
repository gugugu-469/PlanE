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

c_to_hrt_prompt = '''You are currently a senior information extraction expert.
Your task is to extract event information from the given text. First, extract trigger–argument pairs from the given text. Then, for each extracted argument, identify the corresponding argument role type from the provided list of argument role types: {role_list}.
 Next, based on the trigger–argument pairs and their identified argument role types, choose the appropriate event type from the given list of event types: {event_list}.
The output format of this task is: (event trigger|| trigger|| event type|| argument|| argument role).
Given text: "{text}"
'''

c_to_hr_prompt = '''You are currently a senior expert in information extraction.
Your task is to identify potential event triggers from the given text.
The output format of the task is: (event trigger|| trigger).
Given text: "{text}"
'''
c_to_tr_prompt = '''You are currently a senior expert in information extraction.
Your task is to first extract the possible arguments from the given text. Then, for each extracted argument, determine its corresponding argument role from the given list of argument role types: {role_list}.
The output format of the task is: (argument|| argument role).
Given text: "{text}"
'''

sc_to_tr_prompt = '''You are currently a senior expert in information extraction.
Your task is to first extract the possible arguments from the given text and the event trigger. Then, for each extracted argument, determine its corresponding argument role from the given list of argument role types: {role_list}.
Next, based on the given text, event trigger, the extracted arguments, and their argument roles, identify the corresponding event types from the given list of event types: {event_list}.
The output format of the task is: (argument|| argument role|| event type).
Given text: "{text}"
Given event trigger: "{trigger}"
'''

sc_to_hr_prompt = '''You are currently a senior expert in information extraction.
Your task is to identify potential event triggers from the given text and the argument. Then, based on each trigger, determine the potential event type from the given list of event types {event_list}.
The input format of the argument is: (argument, argument role).
The output format of the task is: (event trigger|| trigger|| event type).
Given text: "{text}"
Given argument: ({arg_name}, {arg_role})
'''


with jsonlines.open('xx/ori/ACE05/ACE05-dev.jsonl', 'r') as f:
    dev_datas = [data for data in f]
with jsonlines.open('xx/ori/ACE05/ACE05-test.jsonl', 'r') as f:
    test_datas = [data for data in f]
with open('xx/ori/ACE05/labels.json','r') as f:
    trip_types_list = json.load(f)

event_types, role_types = trip_types_list
event_types = event_types.split(',')
event_types = [item.strip() for item in event_types]
role_types = role_types.split(',')
role_types = [item.strip() for item in role_types]
event_types = str(event_types).replace("'",'"')
role_types = str(role_types).replace("'",'"')

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
    data_type_list = ['test']
    use_datas_list = [test_datas]
    overall_prompt_list = [
        ('c_to_hrt', c_to_hrt_prompt),
        ('c_to_hr', c_to_hr_prompt),
        ('c_to_tr', c_to_tr_prompt),
        ('sc_to_tr', sc_to_tr_prompt),
        ('sc_to_hr', sc_to_hr_prompt),
    ]
    ori_task = ['c_to_hrt', 'c_to_hr', 'c_to_tr']
    for data_type, use_datas in zip(data_type_list, use_datas_list):
        for prompt_name, prompt_format in overall_prompt_list:
            out_name = '{}_{}_ACE05.jsonl'.format(prompt_name, data_type)
            prompt_list = []
            text_list = []

            if prompt_name in ori_task:
                for data in use_datas:
                    text = data['text']
                    text_list.append(text)
                    prompt = prompt_format.format(
                        role_list=role_types,event_list = event_types,
                        text=data['text']
                    )
                    prompt_list.append(prompt)
            elif prompt_name == 'sc_to_tr':
                read_file = os.path.join(now_out_dir, '{}_{}_ACE05.jsonl'.format('c_to_hr', data_type))
                with jsonlines.open(read_file, 'r') as f:
                    read_datas = [data for data in f]
                index_list = []
                hr_list = []
                for index,data in enumerate(read_datas):
                    text = data['text']
                    for hr in data['hr_list']:
                        text_list.append(text)
                        prompt = prompt_format.format(
                            role_list=role_types,event_list = event_types,
                            text=data['text'],
                            trigger = hr['subject'],
                        )
                        prompt_list.append(prompt)
                        index_list.append(index)
                        hr_list.append(hr)

            elif prompt_name == 'sc_to_hr':
                read_file = os.path.join(now_out_dir, '{}_{}_ACE05.jsonl'.format('c_to_tr', data_type))
                with jsonlines.open(read_file, 'r') as f:
                    read_datas = [data for data in f]
                index_list = []
                tr_list = []
                for index,data in enumerate(read_datas):
                    text = data['text']
                    for tr in data['tr_list']:
                        text_list.append(text)
                        prompt = prompt_format.format(
                            role_list=role_types,event_list = event_types,
                            text=data['text'],
                            arg_name = tr['object'],
                            arg_role = tr['object_type'],
                        )
                        prompt_list.append(prompt)
                        index_list.append(index)
                        tr_list.append(tr)
            else:
                raise ValueError('prompt name:{}'.format(prompt_name))
            prompt_list = get_apply_prompts(tokenizer, prompt_list)
            res_list = vllm_chat(tokenizer, model, prompt_list, sampling_params)
            if prompt_name == 'c_to_hrt':
                out_datas = []
                for text, prompt, pred in zip(text_list, prompt_list, res_list):
                    out_datas.append({
                        'text': text,
                        'prompt': prompt,
                        'output': pred
                    })
            elif prompt_name == 'c_to_hr':
                out_datas = []
                pattern = r'[（\(]([^|]+?)\s*\|\|\s*([^|]+?)[）\)]\s*\n'
                for text, prompt, pred in zip(text_list, prompt_list, res_list):
                    finds = re.findall(pattern, pred)
                    hr_list = []
                    for item in finds:
                        hr_list.append({
                            'subject': item[0],
                            'subject_type': item[1],
                        })
                    out_datas.append({
                        'text': text,
                        'prompt': prompt,
                        'output': pred,
                        'hr_list': hr_list
                    })
            elif prompt_name == 'c_to_tr':
                out_datas = []
                pattern = r'[（\(]([^|]+?)\s*\|\|\s*([^|]+?)[）\)]\s*\n'
                for text, prompt, pred in zip(text_list, prompt_list, res_list):
                    finds = re.findall(pattern, pred)
                    tr_list = []
                    for item in finds:
                        tr_list.append({
                            'object': item[0],
                            'object_type': item[1],
                        })
                    out_datas.append({
                        'text': text,
                        'prompt': prompt,
                        'output': pred,
                        'tr_list': tr_list
                    })
            elif prompt_name == 'sc_to_tr':
                out_datas = []
                pattern = r'[（\(]([^|]+?)\s*\|\|\s*([^|]+?)\s*\|\|\s*([^|]+?)[）\)]\s*\n'
                for text, prompt, pred, hr, index  in zip(text_list, prompt_list, res_list, hr_list, index_list):
                    finds = re.findall(pattern, pred)
                    tr_list = []
                    for item in finds:
                        tr_list.append({
                            'object': item[0],
                            'object_type': item[1],
                            'predicate': item[2],
                        })
                    out_datas.append({
                        'text': text,
                        'prompt': prompt,
                        'output': pred,
                        'tr_list': tr_list,
                        'hr': hr,
                        'index': index
                    })
                pass
            elif prompt_name == 'sc_to_hr':
                out_datas = []
                pattern = r'[（\(]([^|]+?)\s*\|\|\s*([^|]+?)\s*\|\|\s*([^|]+?)[）\)]\s*\n'
                for text, prompt, pred, tr, index  in zip(text_list, prompt_list, res_list, tr_list, index_list):
                    finds = re.findall(pattern, pred)
                    hr_list = []
                    for item in finds:
                        hr_list.append({
                            'subject': item[0],
                            'subject_type': item[1],
                            'predicate': item[2],
                        })
                    out_datas.append({
                        'text': text,
                        'prompt': prompt,
                        'output': pred,
                        'hr_list': hr_list,
                        'tr': tr,
                        'index': index
                    })
                pass
            else:
                raise ValueError('prompt name error:{}'.format(prompt_name))
            with jsonlines.open(os.path.join(now_out_dir, out_name), 'w') as f:
                for data in out_datas:
                    f.write(data)


if __name__ == '__main__':
    main()