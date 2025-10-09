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
    return pred

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

    dataset = load_dataset("LightChen2333/M3CoT", split="validation")
    dataset = [x for x in dataset if x['image'] is not None]

    correct, total = 0, 0
    history = []
    for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
        
        image = data['image'].convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]

        # 질문/선택지 프롬프트 준비
        qs = f"Question: {data['question']}\nOptions\n"
        labels = ALPHA_MAP[:len(data['choices'])]
        for i, c in zip(labels, data['choices']):
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

            conv_template = "llada" if model_name == 'llava_llada' else "dream"
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
            # qs = qs + '\n\n' + 'Please reason step by step, and answer the question using a single word or phrase in the format of Answer: <answer>.' # 0.3500

            conv_template = "llada" if model_name == 'llava_llada' else "dream"
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        image_sizes = [image.size]

        max_new_tokens = 256

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
            remasking='low_confidence', # posaware  low_confidence
            pos_penalty_gamma=args.pos_penalty_gamma,
            pos_penalty_alpha=args.pos_penalty_alpha,
            vfg_scale=1.0,
            # use_cache=True,
        )

        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        generated_text = [text_output.lstrip('!') for text_output in text_outputs][0]
        generated_text = generated_text.replace("\\boxed", '').replace('{', '').replace('}', '').strip()

        end_time = time.time()
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        generated_text = [text_output.lstrip('!') for text_output in text_outputs][0]
        generated_text = generated_text.replace("\\boxed", '').replace('{', '').replace('}', '').strip()

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

        history.append(hist)
        torch.cuda.empty_cache()
        

    ans_file.close()
    print(f"[{total}] ACC: {acc:.4f}")

    all_steps = sorted({k for h in history for k in h.keys()})
    avg_hist = {}

    for step in all_steps:
        vals = []
        for h in history:
            if step in h:
                v = h[step]
                if isinstance(v, list):
                    vals.extend(v)
                else:
                    vals.append(v)
        if len(vals) > 0:
            avg_hist[step] = float(np.mean(vals))

    with open(f'mean_diff{max_new_tokens}_pdm_val.jsonl', "w", encoding="utf-8") as f:
        json.dump(avg_hist, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="zzzzz/mmada/mmada_m3cot4.jsonl")
    parser.add_argument("--ccot", action="store_true")
    parser.add_argument("--think", action="store_true")
    # NEW: 위치 인지 패널티 하이퍼파라미터
    parser.add_argument("--pos_penalty_gamma", type=float, default=0.5)
    parser.add_argument("--pos_penalty_alpha", type=float, default=1.0)
    parser.add_argument("--vfg_scale", type=float, default=0.0)
    args = parser.parse_args()

    if args.ccot:
        run_eval(args, ccot=True)
    else:
        run_eval(args, ccot=False)
