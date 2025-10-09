import os
import json
import time
import copy
import torch
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append('./experiments/mmada')

from models import MAGVITv2, MMadaModelLM
from training.prompting_utils import UniversalPrompting
from training.utils import get_config, image_transform
from transformers import AutoTokenizer
from datasets import load_dataset


ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]

answerPrompt = "Use the image and scene graph as context and answer the following question: "
sgPrompt = '''
For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question

Scene Graph:
'''

import re
import math
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

def process_image(image_path, config, device):
    image_ori = Image.open(image_path).convert("RGB")
    image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
    image = image.unsqueeze(0)
    return image

def run_eval(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, uni_prompting, vq_model, model = load_models(config, device)

    ans_path = args.answers_file
    os.makedirs(os.path.dirname(ans_path), exist_ok=True)
    ans_file = open(ans_path, 'w')

    dataset = load_dataset("derek-thomas/ScienceQA", split="test")
    dataset = [x for x in dataset if x['image'] is not None]

    correct, total = 0, 0

    for data in tqdm(dataset):
        image_ori = data['image'].convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
        image = image.unsqueeze(0)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)

        labels = [chr(ord("A") + i) for i in range(len(data["choices"]))]
        options = labels
        answer = labels[data["answer"]]

        # 질문/선택지 프롬프트 준비
        if data['hint'] is not None:
            qs = f"Context: {data['hint']}\nQuestion: {data['question']}\nOptions\n"
        else:
            qs = f"Question: {data['question']}\nOptions\n"
        
        for i, c in zip(labels, data['choices']):
            qs += '{}. {}\n'.format(i, c)

        if args.ccot:
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
            input_ids = torch.tensor(uni_prompting.text_tokenizer(['<|start_header_id|>user<|end_header_id|>\n' + "Scene Graph: " + sg_text + "\n\n" + answerPrompt + qs  +'<eot_id><|start_header_id|>assistant<|end_header_id|>\n'])['input_ids']).to(device)
            input_ids = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                image_tokens,
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
                input_ids
            ], dim=1).long()
        else:
            # qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            think_prefix = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
            question = think_prefix + qs
            
            input_ids = torch.tensor(uni_prompting.text_tokenizer(['<|start_header_id|>user<|end_header_id|>\n' + qs  +'<eot_id><|start_header_id|>assistant<|end_header_id|>\n'])['input_ids']).to(device)
            question_prompt_ids = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                image_tokens,
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
                input_ids
            ], dim=1).long()

            input_ids = question_prompt_ids

        start_time = time.time()

        max_new_tokens = args.max_new_tokens
        output_ids = model.mmu_generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            steps=max_new_tokens // 2,
            block_length=max_new_tokens,
            remasking='low_confidence', # low_confidence random posaware
            pos_penalty_gamma=args.pos_penalty_gamma,
            pos_penalty_alpha=args.pos_penalty_alpha,
            vfg_scale=args.vfg_scale,
            vfg_start=args.vfg_start,
            vfg_end=args.vfg_end,
        )
        generated_text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        # qs = generated_text + '\n\n' + qs + '\n' + "Answer with the option's letter from the given choices directly."
        qs = qs + '\n\n' + generated_text + '\n' + "Answer with the option's letter from the given choices directly."

        prompt_text = '<|start_header_id|>user<|end_header_id|>\n' + qs + '<eot_id><|start_header_id|>assistant<|end_header_id|>\n'
        input_ids = torch.tensor(uni_prompting.text_tokenizer([prompt_text])['input_ids']).to(device)
        question_prompt_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
            image_tokens,
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
            input_ids
        ], dim=1).long()

        input_ids = question_prompt_ids

        output_ids = model.mmu_generate(
            input_ids,
            max_new_tokens=2,
            steps=1,
            block_length=2,
            remasking='low_confidence',
        )

        end_time = time.time()

        generated_text = uni_prompting.text_tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()

        ans_file.write(json.dumps({
            "prompt": qs,
            "text": generated_text,
            "label": data['answer'],
            "time": end_time - start_time
        }) + "\n")
        ans_file.flush()

        total += 1
        pred = generated_text.strip()
        if pred and pred[0] in ALPHA_MAP:
            pred = pred[0]
        else:
            pred = judge_answer(pred, options)


        if pred == answer:
            correct += 1

        acc = correct / total
        print(f"[{total}] ACC: {acc:.4f}")

        # 메모리 정리
        del output_ids, generated_text
        torch.cuda.empty_cache()

    ans_file.close()
    print(f"[{total}] ACC: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="test/mmada/mmada_m3cot.jsonl")
    parser.add_argument("--ccot", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    # NEW: 위치 인지 패널티 하이퍼파라미터
    parser.add_argument("--pos_penalty_gamma", type=float, default=0.5)
    parser.add_argument("--pos_penalty_alpha", type=float, default=1.0)
    parser.add_argument("--vfg_scale", type=float, default=0.0)
    parser.add_argument("--vfg_start", type=float, default=0.0)
    parser.add_argument("--vfg_end", type=float, default=1.0)
    args = parser.parse_args()

    config = get_config()
    run_eval(config, args)
