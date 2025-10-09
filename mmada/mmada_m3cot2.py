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

def judge_answer(text, choices):
    import re
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
    return pred # 0.2455

# def judge_answer(pred_text: str, choices) -> str | None:
#     import re
#     """pred 텍스트에서 최종 선택지(A~F) 한 글자를 최대한 안정적으로 추출."""
#     if not pred_text:
#         return None
#     t = pred_text.strip()

#     if t[-1].isdigit():
#         num = int(t[-1])
#         if 1 <= num <= len(choices):  # choices 길이에 맞춰 안전하게
#             return chr(ord('A') + num - 1)
    
#     # 1) "Answer: X" / "Ans: X" / "Final: X" / "Choice: X"
#     m = re.search(r'(?i)\b(ans(?:wer)?|final|choice)\s*[:=\-]\s*([A-F])\b', t)
#     if m:
#         return m.group(2).upper()

#     # 2) 줄 끝에 ' - X' / ': X' / '= X' 형태
#     m = re.search(r'[:=\-]\s*([A-F])\s*$', t, flags=re.I | re.M)
#     if m:
#         return m.group(1).upper()

#     # 3) 괄호 표기 "(X)"
#     m = re.search(r'\(([A-F])\)(?![a-z])', t, flags=re.I)
#     if m:
#         return m.group(1).upper()

#     # 4) 문장 중 단독 등장한 마지막 선택지 문자
#     letters = re.findall(r'\b([A-F])\b', t.upper())
#     if letters:
#         return letters[-1]

#     # 5) 옵션 텍스트를 직접 지칭하는 경우: "Option B", "Choice C"
#     m = re.search(r'(?i)\b(option|choice)\s*([A-F])\b', t)
#     if m:
#         return m.group(2).upper()

#     return None # 0.2450


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

    dataset = load_dataset("LightChen2333/M3CoT", split="test")
    dataset = [x for x in dataset if x['image'] is not None]

    correct, total = 0, 0

    for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
        
        image_ori = data['image'].convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
        image = image.unsqueeze(0)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)

        # 질문/선택지 프롬프트 준비
        qs = f"Question: {data['question']}\nOptions\n"
        labels = ALPHA_MAP[:len(data['choices'])]
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

            max_new_tokens = 64
            sg_output_ids = model.mmu_generate(input_ids, max_new_tokens=max_new_tokens, steps=max_new_tokens // 2, block_length=max_new_tokens)
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
            # qs = qs + "\nLet's think step by step."

            think_prefix = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
            question = think_prefix + qs
            
            # qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            prompt_text = '<|start_header_id|>user<|end_header_id|>\n' + question + '<eot_id><|start_header_id|>assistant<|end_header_id|>\n'
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

        start_time = time.time()
        
        # 새로운 위치 인지 패널티 적용 generate 함수 사용
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

        generated_text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        print(qs)
        print(generated_text)

        ans_file.write(json.dumps({
            "prompt": qs,
            "text": generated_text,
            "label": data['answer'],
            "time": end_time - start_time
        }) + "\n")
        ans_file.flush()

        total += 1
        pred = generated_text.strip().upper()
        if pred and pred[-1] in ALPHA_MAP:
            pred = pred[-1]
        else:
            pred = judge_answer(pred, data['choices'])

        if pred == data['answer']:
            correct += 1

        print('정답:', data['answer'], '예측:', pred)
        acc = correct / total
        print(f"[{total}] ACC: {acc:.4f}")

        del output_ids, generated_text
        torch.cuda.empty_cache()

    ans_file.close()
    print(f"[{total}] ACC: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="zzzzz/mmada/mmada_m3cot4.jsonl")
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
