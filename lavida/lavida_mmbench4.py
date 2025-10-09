import os
import json
import time
import copy
import torch
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np
import gc
import math
import re

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append('./experiments/lavida')

from llava.model.builder import load_pretrained_model

from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from llava.model.language_model.llada.generate import generate as llada_generate
from llava.model.language_model.llada.log_likelyhood import get_logits as llada_get_logits

from datasets import load_dataset

# 옵션
ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]

answerPrompt = "Use the image and scene graph as context and answer the following question: "
sgPrompt = '''
For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question

Scene Graph:
'''

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() in ['nan', 'none']:
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

def judge_answer(text, choices):
    pattern = re.compile(r'\(([A-Za-z])\)')
    res = pattern.findall(text)
    if len(res) >= 1:
        pred = res[-1].upper()
    else:
        res = []
        for i, choice in enumerate(choices):
            if choice.lower() in text.lower():
                res.append(ALPHA_MAP[i])
        if len(res) >= 1:
            pred = res[-1]
        else:
            for i, choice in enumerate(choices):
                text = re.sub(r'[\n.,!?]', ' ', text)
                if ALPHA_MAP[i] in text.split(" "):
                    res.append(ALPHA_MAP[i])
            if len(res) >= 1:
                pred = res[-1]
            else:
                for i, choice in enumerate(choices):
                    text = re.sub(r'[\n.,!?]', ' ', text)
                    if ALPHA_MAP[i].lower() in text.split(" "):
                        res.append(ALPHA_MAP[i])
                if len(res) >= 1:
                    pred = res[-1]
                else:
                    pred = "FAILED"
    return pred

def run_eval(args, ccot=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_id = 'hbXNov/lavida-llada-reason' # jacklishufan/lavida-llada-v1.0-instruct  hbXNov/lavida-llada-reason
    model_name = 'llava_llada'
    vision_kwargs = dict(
        mm_vision_tower="google/siglip-so400m-patch14-384",
        mm_resampler_type=None,
        mm_projector_type='mlp2x_gelu',
        mm_hidden_size=1152,
        use_mm_proj=True
    )
    tokenizer, model, image_processor, max_length = load_pretrained_model(model_id, None, model_name, device_map='auto', vision_kwargs=vision_kwargs,torch_dtype='bfloat16')
    model.eval()

    ans_path = args.answers_file
    os.makedirs(os.path.dirname(ans_path), exist_ok=True)
    ans_file = open(ans_path, 'w')

    dataset = load_dataset("lmms-lab/MMBench_EN", split="dev")

    correct, total = 0, 0

    for data in tqdm(dataset):
        options = get_options(data, ["A", "B", "C", "D"])
        
        labels = ALPHA_MAP[:len(options)]

        image = data['image'].convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]

        # 질문/선택지 프롬프트 준비
        qs = f"Question: {data['question']}\nOptions\n"
        for i, c in zip(labels, options):
            qs += '{}. {}\n'.format(i, c)

        start_time = time.time()
        if ccot:
            # 1단계: Scene Graph 생성
            sg =  qs + "\n" + sgPrompt

            conv_template = "llada" 
            question = DEFAULT_IMAGE_TOKEN + '\n' + sg
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

            cont,hist = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=256,
                block_length=256,
                step_ratio=0.5,
                tokenizer=tokenizer,
                prefix_lm=True,
                verbose=True,
                schedule='shift',
                # use_cache=True,
            )

            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            sg_text = [text_output.lstrip('!') for text_output in text_outputs][0].strip()
            
            # 2단계: Scene Graph 기반 답변
            qs = "Scene Graph: " + sg_text + "\n\n" + answerPrompt + qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv_template = "llada" 
            question = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()  

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        else:
            # qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            # qs = qs + "\nLet's think step by step."
            question = qs + '\n\n' + "Please reason step by step, and answer the question with option letter from given choices in the format of Answer: <option letter>."
            # qs = qs + '\n\n' + 'Please reason step by step, and answer the question using a single word or phrase in the format of Answer: <answer>.'
            
            conv_template = "llada" 
            question = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()  

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        image_sizes = [image.size]

        max_new_tokens = 128
        
        cont,hist = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            block_length=max_new_tokens,
            step_ratio=0.5,
            tokenizer=tokenizer,
            prefix_lm=True,
            verbose=True,
            schedule='shift',
            remasking='low_confidence', # posaware  low_confidence  margin  entrophy
            pos_penalty_gamma=args.pos_penalty_gamma,
            pos_penalty_alpha=args.pos_penalty_alpha,
            vfg_scale=args.vfg_scale,
            # use_cache=True,
        )

        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        generated_text = [text_output.lstrip('!') for text_output in text_outputs][0]
        generated_text = generated_text.replace("\\boxed", '').replace('{', '').replace('}', '').strip()


        # qs = generated_text + '\n\n' + qs + '\n' + "Answer with the option's letter from the given choices directly."
        qs = qs + '\n\n' + generated_text + '\n' + "Answer with the option's letter from the given choices directly."

        conv_template = "llada" 
        question = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()  
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        cont,hist = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=2,
            block_length=2,
            step_ratio=0.5,
            tokenizer=tokenizer,
            prefix_lm=True,
            verbose=True,
            schedule='shift',
            remasking='low_confidence',
            vfg_scale=0.0,
            # use_cache=True,
        )
        end_time = time.time()
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        generated_text = [text_output.lstrip('!') for text_output in text_outputs][0]
        generated_text = generated_text.replace("\\boxed", '').replace('{', '').replace('}', '').strip()

        print(generated_text)

        ans_file.write(json.dumps({
            "prompt": qs,
            "text": generated_text,
            "label": data['answer'],
            "options": options,
            "time": end_time - start_time
        }) + "\n")
        ans_file.flush()

        total += 1
        pred = generated_text.strip().upper()
        if pred and pred[0] in ALPHA_MAP:
            pred = pred[0]
        else:
            pred = judge_answer(pred, options)

        if pred == data['answer']:
            correct += 1

        acc = correct / total
        print(f"[{total}] ACC: {acc:.4f}")


    ans_file.close()
    print(f"[{total}] ACC: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="results/mmada/mmada_mmbench.jsonl")
    parser.add_argument("--ccot", action="store_true")
    
    parser.add_argument("--pos_penalty_gamma", type=float, default=0.5)
    parser.add_argument("--pos_penalty_alpha", type=float, default=1.0)
    parser.add_argument("--vfg_scale", type=float, default=0.0)
    args = parser.parse_args()

    if args.ccot:
        run_eval(args, ccot=True)
    else:
        run_eval(args, ccot=False)
