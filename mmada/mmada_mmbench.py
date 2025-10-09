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
sys.path.append('./experiments/mmada')

from models import MAGVITv2, MMadaModelLM
from training.prompting_utils import UniversalPrompting
from training.utils import get_config, image_transform
from transformers import AutoTokenizer
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

def load_models(config, device):
    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.pretrained_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True
    )
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = MMadaModelLM.from_pretrained(
        config.model.mmada.pretrained_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

    return tokenizer, uni_prompting, vq_model, model

def run_eval(config, args, ccot=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, uni_prompting, vq_model, model = load_models(config, device)

    ans_path = args.answers_file
    os.makedirs(os.path.dirname(ans_path), exist_ok=True)
    ans_file = open(ans_path, 'w')

    dataset = load_dataset("lmms-lab/MMBench_EN", split="dev")

    correct, total = 0, 0

    for data in tqdm(dataset):
        options = get_options(data, ["A", "B", "C", "D"])
        
        labels = ALPHA_MAP[:len(options)]

        image_ori = data['image'].convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
        image = image.unsqueeze(0)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)

        # 질문/선택지 프롬프트 준비
        qs = f"Question: {data['question']}\nOptions\n"
        for i, c in zip(labels, options):
            qs += '{}. {}\n'.format(i, c)

        start_time = time.time()
        if ccot:
            # 1단계: Scene Graph 생성
            input_ids = torch.tensor(uni_prompting.text_tokenizer(['<|start_header_id|>user<|end_header_id|>\n' + qs + "\n" + sgPrompt  +'<eot_id><|start_header_id|>assistant<|end_header_id|>\n'])['input_ids']).to(device)
            input_ids = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                image_tokens,
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
                input_ids
            ], dim=1).long()

            sg_output_ids = model.mmu_generate(input_ids, max_new_tokens=256, steps=128, block_length=256)
            sg_text = uni_prompting.text_tokenizer.batch_decode(
                sg_output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0]

            # 2단계: Scene Graph 기반 답변
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            input_ids = torch.tensor(uni_prompting.text_tokenizer([
                '<|start_header_id|>user<|end_header_id|>\n' + 
                "Scene Graph: " + sg_text + "\n\n" + answerPrompt + qs +
                '<eot_id><|start_header_id|>assistant<|end_header_id|>\n'
            ])['input_ids']).to(device)
            input_ids = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                image_tokens,
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
                input_ids
            ], dim=1).long()

            del sg_output_ids
        else:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            input_ids = torch.tensor(uni_prompting.text_tokenizer([
                '<|start_header_id|>user<|end_header_id|>\n' + qs +
                '<eot_id><|start_header_id|>assistant<|end_header_id|>\n'
            ])['input_ids']).to(device)
            input_ids = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                image_tokens,
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
                input_ids
            ], dim=1).long()

        output_ids = model.mmu_generate(input_ids, max_new_tokens=2, steps=1, block_length=2, vfg_scale=2.0) # 0.5572
        end_time = time.time()

        generated_text = uni_prompting.text_tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()

        ans_file.write(json.dumps({
            "question_id": data["index"] if "index" in data else data["question_id"],
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

        del output_ids, input_ids

    ans_file.close()
    print(f"[{total}] ACC: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="results/mmada/mmada_mmbench.jsonl")
    parser.add_argument("--ccot", action="store_true")
    args = parser.parse_args()

    config = get_config()
    if args.ccot:
        run_eval(config, args, ccot=True)
    else:
        run_eval(config, args, ccot=False)
