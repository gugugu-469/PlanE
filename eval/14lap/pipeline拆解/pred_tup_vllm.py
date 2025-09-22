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
import time
from vllm.distributed.parallel_state import destroy_model_parallel
import argparse

c_to_hrt_prompt = '''You are currently a senior expert in fine-grained sentiment extraction.
Your task is to aspect sentiment triplets from the given text, each triplet contains aspect term, opinion term and sentiment type.
Given the list of sentiment types: ['NEG', 'NEU', 'POS']
The output format of the task is: (Aspect Term||Opinion Term||Sentiment Type) 
Given text: "{text}"
'''

c_to_ht_prompt = '''You are currently a senior expert in fine-grained sentiment extraction.
Your task is to extract all possible aspect-opinion term pairs from the given text. First, identify potential aspect terms. Then, based on the extracted aspect terms and the given text, extract the corresponding opinion terms.
The output format of the task is: (Aspect Term||Opinion Term) 
Given text: "{text}"
'''

c_to_r_prompt = '''You are currently a senior expert in fine-grained sentiment extraction.
Your task is to identify potential sentiment types based on the given text.
Given the list of sentiment types: ['NEG', 'NEU', 'POS']
The output format of the task is: (Sentiment Type) 
Given text: "{text}"
'''

rc_to_ht_prompt = '''You are currently a senior expert in fine-grained sentiment extraction.
Your task is to extract all aspect-opinion term pairs from the given text and the sentiment type. First, identify potential aspect terms. Then, based on the extracted aspect terms and the sentiment type, extract the corresponding opinion terms.
The output format of the task is: (Aspect Term||Opinion Term) 
Given text: "{text}"
Given sentiment type: "{sentiment_type}"
'''

htc_to_r_prompt = '''You are currently a senior expert in fine-grained sentiment extraction.
Your task is to identify the sentiment type from from the given aspect-opinion term pair.
Given the list of sentiment types: ['NEG', 'NEU', 'POS']
The input format of the aspect-opinion term pair is: (Aspect Term||Opinion Term) 
The output format of the task is: (Sentiment Type) 
Given text: "{text}"
Given aspect-opinion term pair: ({aspect_term}||{opinion_term})
'''

with jsonlines.open('xx/ori/14lap/processed_test.jsonl', 'r') as f:
    test_datas = [data for data in f]



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

def common_chat_qwen3(tokenizer, model, prompt_list, temperature, batch_size = 4):
    response_list = []
    # print('now common chat')
    for i in tqdm(range(0, len(prompt_list), batch_size)):
        batch_prompts = prompt_list[i:i+batch_size]
        # print('batch_prompts:{}'.format(batch_prompts))
        # Tokenize the batch
        model_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        # Generate responses for the batch
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample = True,
            temperature = temperature,
            top_p = 0.9
        )
        
        # Process each response in the batch
        for j in range(len(batch_prompts)):
            # Get the output for this specific prompt (skip input tokens)
            input_length = model_inputs.input_ids[j].shape[0]
            output_ids = generated_ids[j][input_length:].tolist()

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
                
            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            response_list.append(content)
            # print('thinking_content:{}'.format(thinking_content))
            # print('content:{}'.format(content))
            # print('all_content:{}'.format(all_content))
            
    return response_list

def common_chat(tokenizer, model, prompt_list, temperature, batch_size = 4):
    response_list = []
    # print('now common chat')
    for i in tqdm(range(0, len(prompt_list), batch_size)):
        batch_prompts = prompt_list[i:i+batch_size]
        # print('batch_prompts:{}'.format(batch_prompts))
        # Tokenize the batch
        model_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        # Generate responses for the batch
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample = True,
            temperature = temperature,
            top_p = 0.9
        )
        
        # Process each response in the batch
        for j in range(len(batch_prompts)):
            # Get the output for this specific prompt (skip input tokens)
            input_length = model_inputs.input_ids[j].shape[0]
            output_ids = generated_ids[j][input_length:].tolist()

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
                
            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            response_list.append(content)
            # print('thinking_content:{}'.format(thinking_content))
            # print('content:{}'.format(content))
            # print('all_content:{}'.format(all_content))
            
    return response_list


# 基本的输出路径
base_out_dir = './pred_result_0917_rl'
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
        gpu_memory_utilization = 0.82,
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
        ('c_to_ht', c_to_ht_prompt),
        ('c_to_r', c_to_r_prompt),
        ('rc_to_ht', rc_to_ht_prompt),
        ('htc_to_r', htc_to_r_prompt),
    ]
    ori_task = ['c_to_hrt', 'c_to_ht', 'c_to_r']
    start_time = time.time()
    for data_type, use_datas in zip(data_type_list, use_datas_list):
        for prompt_name, prompt_format in overall_prompt_list:
            out_name = '{}_{}_14lap.jsonl'.format(prompt_name, data_type)
            prompt_list = []
            text_list = []

            if prompt_name in ori_task:
                for data in use_datas:
                    text = data['text']
                    text_list.append(text)
                    prompt = prompt_format.format(
                        text=data['text']
                    )
                    prompt_list.append(prompt)
            elif prompt_name == 'rc_to_ht':
                read_file = os.path.join(now_out_dir, '{}_{}_14lap.jsonl'.format('c_to_r', data_type))
                with jsonlines.open(read_file, 'r') as f:
                    read_datas = [data for data in f]
                index_list = []
                sentiment_type_list = []
                for index,data in enumerate(read_datas):
                    text = data['text']
                    for sentiment_type in data['sentiment_type_list']:
                        text_list.append(text)
                        prompt = prompt_format.format(
                            text=data['text'],
                            sentiment_type = sentiment_type
                        )
                        prompt_list.append(prompt)
                        index_list.append(index)
                        sentiment_type_list.append(sentiment_type)

            elif prompt_name == 'htc_to_r':
                read_file = os.path.join(now_out_dir, '{}_{}_14lap.jsonl'.format('c_to_ht', data_type))
                with jsonlines.open(read_file, 'r') as f:
                    read_datas = [data for data in f]
                index_list = []
                sp_list = []
                for index,data in enumerate(read_datas):
                    text = data['text']
                    for sp in data['sp_list']:
                        text_list.append(text)
                        prompt = prompt_format.format(
                            text=data['text'],
                            aspect_term = sp['aspect_term'],
                            opinion_term = sp['opinion_term']
                        )
                        prompt_list.append(prompt)
                        index_list.append(index)
                        sp_list.append(sp)
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
            elif prompt_name == 'c_to_ht':
                out_datas = []
                pattern = r'[（\(]([^|]+?)\s*\|\|\s*([^|]+?)\s*[）\)]\s*\n'
                for text, prompt, pred in zip(text_list, prompt_list, res_list):
                    finds = re.findall(pattern, pred)
                    sp_list = []
                    for item in finds:
                        sp_list.append({
                            'aspect_term': item[0],
                            'opinion_term': item[1]
                        })
                    out_datas.append({
                        'text': text,
                        'prompt': prompt,
                        'output': pred,
                        'sp_list': sp_list
                    })
            elif prompt_name == 'c_to_r':
                out_datas = []
                pattern = r'[（\(]([^|]+?)[）\)]\s*\n'
                for text, prompt, pred in zip(text_list, prompt_list, res_list):
                    finds = re.findall(pattern, pred)
                    sentiment_type_list = []
                    for item in finds:
                        sentiment_type_list.append({
                            'predicate': item,
                        })
                    out_datas.append({
                        'text': text,
                        'prompt': prompt,
                        'output': pred,
                        'sentiment_type_list': sentiment_type_list
                    })
                pass
            elif prompt_name == 'rc_to_ht':
                out_datas = []
                pattern = r'[（\(]([^|]+?)\s*\|\|\s*([^|]+?)\s*[）\)]\s*\n'
                for text, prompt, pred, sentiment_type, index  in zip(text_list, prompt_list, res_list, sentiment_type_list, index_list):
                    finds = re.findall(pattern, pred)
                    sp_list = []
                    for item in finds:
                        sp_list.append({
                            'aspect_term': item[0],
                            'opinion_term': item[1]
                        })
                    out_datas.append({
                        'text': text,
                        'prompt': prompt,
                        'output': pred,
                        'sp_list': sp_list,
                        'sentiment_type': sentiment_type,
                        'index': index
                    })
                pass
            elif prompt_name == 'htc_to_r':
                out_datas = []
                pattern = r'[（\(]([^|]+?)[）\)]\s*\n'
                for text, prompt, pred, sp, index in zip(text_list, prompt_list, res_list, sp_list, index_list):
                    finds = re.findall(pattern, pred)
                    sentiment_type_list = []
                    for item in finds:
                        sentiment_type_list.append({
                            'predicate': item,
                        })
                    out_datas.append({
                        'text': text,
                        'prompt': prompt,
                        'output': pred,
                        'sentiment_type_list': sentiment_type_list,
                        'sp': sp,
                        'index': index
                    })
            else:
                raise ValueError('prompt name error:{}'.format(prompt_name))
            with jsonlines.open(os.path.join(now_out_dir, out_name), 'w') as f:
                for data in out_datas:
                    f.write(data)
    end_time = time.time()
    print('pipeline拆解_model_{}_time:{}'.format(model_path, end_time-start_time))


if __name__ == '__main__':
    main()