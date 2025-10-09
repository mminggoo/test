import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def hellinger_distance(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    return (1 / np.sqrt(2)) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))


def total_variation_distance(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    return 0.5 * np.sum(np.abs(p - q))


def kl_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    return entropy(p, q) # KL(p||q)


# --- Prompt Dependency Measure (PDM) ---
def PDM(p_cond, p_uncond, dist_fn="hellinger"):
    """
    Compute PDM between two probability distributions.


    Args:
    p_cond: np.array, conditional distribution p(·|y<t, x, c)
    p_uncond: np.array, unconditional distribution p(·|y<t, x)
    dist_fn: str, distance function ["hellinger", "tv", "kl"]


    Returns:
    float: PDM value
    """
    # Normalize to ensure valid probability distributions
    p_cond = np.asarray(p_cond) / np.sum(p_cond)
    p_uncond = np.asarray(p_uncond) / np.sum(p_uncond)


    if dist_fn == "hellinger":
        return hellinger_distance(p_cond, p_uncond)
    elif dist_fn == "tv":
        return total_variation_distance(p_cond, p_uncond)
    elif dist_fn == "kl":
        return kl_divergence(p_cond, p_uncond)
    else:
        raise ValueError(f"Unknown distance function: {dist_fn}")

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def get_num_transfer_tokens_sch(mask_index, steps,schedule=None,schedule_kwargs=None):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    if schedule is None:
        return get_num_transfer_tokens(mask_index,steps)
    if schedule_kwargs is None:
        schedule_kwargs = {}
   
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = int(min(steps,mask_num[0]))
    t = torch.linspace(0, 1, steps+1)
    # at least one sample per step
    if schedule =='logit_normal':
      sigmas = sigmoid_normal_cdf(t)
    elif schedule =='shift':
      sigmas = logit_normal_schedule(schedule_kwargs.get('shift',3),t)
    elif schedule == 'cosine':
        sigmas = cosine_schedule(t)
    else:
      sigmas = t
    sigmas = sigmas.to(mask_num.device)
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
    
    for i in range(mask_num.size(0)):
      # print(sigmas.shape)
      sigmas_sample = (sigmas*mask_num[i]).to(torch.int64)
      # print(sigmas_sample)
      sigmas_sample = sigmas_sample[1:]-sigmas_sample[:-1]
      # print(sigmas_sample)
      # fix detal
      sigmas_sample = torch.clamp(sigmas_sample,1,None) # should only increase
      delta = sigmas_sample.sum() - mask_num[i]
    #   breakpoint()
      assert delta>=0
      j = 0
      
      while delta > 0:
        j = j % len(sigmas_sample) 
        if sigmas_sample[j] == 1:
          j += 1
          continue
        
        delta -= 1
        sigmas_sample[j] -= 1
        j += 1
    #   breakpoint()
      assert sigmas_sample.sum()==mask_num[i]
      num_transfer_tokens[i] = sigmas_sample#.to(torch.int64)
    return num_transfer_tokens.flip(-1)

def linear(y):
    return y

def cosine_schedule(x):
    """
    Cosine schedule mapping [0, 1] -> [1, 0]
    """
    x = np.clip(x, 0, 1)
    return 1-0.5 * (1 + np.cos(np.pi * x))

def sigmoid_normal_cdf(y):
    # y must be in (0, 1)
    logit_y = torch.log(y / (1 - y))
    return 0.5 * (1 + torch.erf(logit_y / torch.sqrt(torch.tensor(2.0))))
def logit_normal_schedule(shift,sigmas):
    # shift = 1 / shift
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return sigmas
import os
DEBUG_PRINT_OUTPUT = os.environ.get('DEBUG_PRINT_OUTPUT',False)
@ torch.no_grad()
def generate(model, prompt=None, steps=None, max_new_tokens=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,inputs_embeds=None, position_ids=None,attention_mask=None,
              tokenizer=None,
                verbose=False,
                step_per_block=None,
                prefix_lm=False,
                schedule=None,
                schedule_kwargs=None,
                draft_tokens=None,
                step_ratio=None,
                vfg_scale=0.,
                visual_index=None,
             **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    # breakpoint()
    # remasking = 
    # step_ratio = 0.5
    # block_length = 1024
    # steps = 1024
    steps = max_new_tokens # min(steps,max_new_tokens)
    # if step_ratio:
    #     steps = int(max_new_tokens*step_ratio)
    gen_length = max_new_tokens
    assert position_ids is None
    if prompt is None:
        assert inputs_embeds is not None
        bsz, seq_len = inputs_embeds.shape[:2]
        prompt = torch.full((bsz, seq_len), 0, dtype=torch.long).to(model.device)
    past_key_values = None
    if prefix_lm:
        if vfg_scale > 0.:
            mask_prompt = torch.full((1, 1), mask_id, dtype=torch.long).to(model.device)
            inputs_embeds_mask = model.transformer.wte(mask_prompt)
            inputs_embeds_unv = inputs_embeds.clone()
            inputs_embeds_unv[:, visual_index[0]:visual_index[1]+1] = inputs_embeds_mask
            past_key_values = model([None, None],input_embeddings=torch.cat([inputs_embeds, inputs_embeds_unv], dim=0),use_cache=True).attn_key_values
        else:
            past_key_values = model(None,input_embeddings=inputs_embeds,use_cache=True).attn_key_values
        # breakpoint()
        x = torch.full((bsz, gen_length), mask_id, dtype=torch.long).to(model.device)
        prompt = torch.full((bsz, 0), 0, dtype=torch.long).to(model.device)
        # x[:, :prompt.shape[1]] = prompt.clone()
    else:
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    # assert prompt.shape[0] == 1
    if draft_tokens is not None:
        assert draft_tokens.shape[1] <= gen_length
        x[:, prompt.shape[1]:prompt.shape[1]+draft_tokens.shape[1]] = draft_tokens.clone()

    # if block_length < gen_length:
    #    block_length = gen_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert ( steps % num_blocks == 0) or step_per_block is not None
    steps = steps // num_blocks
    if step_per_block:
        steps = min(step_per_block,block_length)
        assert step_ratio is None, 'Please do not pass both step_ratio and step_per_block'
    # step_ratio = 0.5
    # schedule = 'shift'
    # schedule_kwargs = dict(shift=3)
    # breakpoint()
    if step_ratio:
        steps = int(steps*step_ratio)
    
    # print(steps,step_per_block,block_length,draft_tokens.shape[-1])
    # NFE = 0
    if verbose:
        # history = []
        history = {}
    for num_block in range(num_blocks):
        
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_sch(block_mask_index, steps,schedule=schedule,schedule_kwargs=schedule_kwargs)
        if DEBUG_PRINT_OUTPUT:
            print(f"Block: {num_block + 1}/{num_blocks}, Steps per Block: {steps}, Block Length: {block_length}")
            print(f"Tokens generated per step {num_transfer_tokens[0]}")
        
        values = torch.linspace(0, 1, steps=steps)
        for i in range(steps):
            # print(i)
            
            mask_index = (x == mask_id)
            block_mask_index = mask_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:]
            # print(mask_index.sum())
            if block_mask_index.sum() == 0:
                continue
            # NFE += 2
            if cfg_scale > 0.:
                assert NotImplementedError('cfg_scale > 0. is not supported.')
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                #
                logits = model(x_,input_embeds_inference=[inputs_embeds,None]).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                if vfg_scale > 0.:
                    # inputs_embeds_curr = model.transformer.wte(x)
                    # logits_v = model([None,None],input_embeddings=torch.cat([inputs_embeds_curr, inputs_embeds_curr], dim=0),past_key_values=past_key_values).logits
                    # logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
                    # logits = logits_u + (values[i] + 1) * (logits_c - logits_u)

                    inputs_embeds_curr = model.transformer.wte(x)
                    logits_v = model([None,None],input_embeddings=torch.cat([inputs_embeds_curr, inputs_embeds_curr], dim=0),past_key_values=past_key_values).logits
                    logits, logits_u = torch.chunk(logits_v, 2, dim=0)
                else:
                    inputs_embeds_curr = model.transformer.wte(x)
                    #print(tokenizer.batch_decode(x)[0].replace('<|endoftext|>',''))
                    # print((x==mask_id).sum())
                    # breakpoint()
                    if prefix_lm:
                        # breakpoint()
                        logits = model(None,input_embeddings=inputs_embeds_curr,past_key_values=past_key_values).logits
                    else:
                        if inputs_embeds is not None:
                            inputs_embeds_curr[:,:inputs_embeds.shape[1]] = inputs_embeds
                        logits = model(None,input_embeddings=inputs_embeds_curr).logits
            # logits = logits.cpu()
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'entrophy':
                epsilon = 1e-10
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                log_probs = torch.log(probs + epsilon)
                x0_p = torch.sum(probs * log_probs, dim=-1)
            elif remasking == 'margin':
                ## similar to margin algo in Dream
                p = F.softmax(logits.to(torch.float64), dim=-1)
                sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                top1_probs = sorted_probs[:, :, 0] 
                top2_probs = sorted_probs[:, :, 1] 
                x0_p = top1_probs - top2_probs
            elif remasking == 'posaware':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # [B, L]

                pos_penalty_gamma = kwargs['pos_penalty_gamma']
                pos_penalty_alpha = kwargs['pos_penalty_alpha']
                
                # ---- Position-Aware Confidence Penalty ----
                if pos_penalty_gamma is not None and pos_penalty_gamma > 0.0:
                    block_start = prompt.shape[1] + num_block * block_length
                    block_end   = prompt.shape[1] + (num_block + 1) * block_length  # [start, end)

                    seq_len = x0_p.shape[1]
                    pos1d = torch.arange(seq_len, device=x0_p.device)

                    denom = max(block_length - 1, 1)
                    rel1d = (pos1d - block_start).clamp(min=0, max=denom) / denom  # [0,1] within block

                    # 진행률 (0,1], 초기일수록 큰 패널티
                    t = float(i + 1) / float(steps)
                    # penalty = 1 - gamma * (1 - t) * (rel^alpha)
                    penalty1d = 1.0 - float(pos_penalty_gamma) * (1.0 - t) * (rel1d ** float(pos_penalty_alpha))
                    penalty1d = penalty1d.clamp(min=0.0)

                    # 블록 외부는 패널티 1.0 유지
                    outside = (pos1d < block_start) | (pos1d >= block_end)
                    if outside.any():
                        penalty1d = torch.where(outside, torch.ones_like(penalty1d), penalty1d)

                    x0_p = x0_p * penalty1d.to(dtype=x0_p.dtype).unsqueeze(0)
                # -------------------------------------------

            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                try:
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                except:
                    breakpoint()
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            
            if verbose:
                # history.append(x.clone().cpu())
                history[i] = []
                c = 0
                for ixx in torch.nonzero(transfer_index, as_tuple=False):
                    # history[ixx[1].item()] = i


                    # import torch.nn as nn
                    # M = 0.5 * (nn.functional.softmax(logits[:, ixx[1].item()], dim=-1) + nn.functional.softmax(logits_u[:, ixx[1].item()], dim=-1))
                    # js = 0.5 * nn.functional.kl_div(nn.functional.log_softmax(logits[:, ixx[1].item()], dim=-1), M, reduction='batchmean') + 0.5 * nn.functional.kl_div(nn.functional.log_softmax(logits_u[:, ixx[1].item()], dim=-1), M, reduction='batchmean')
                    # c += js.item()
                    
                    # k = 5
                    # topk_vals_logits, topk_idx_logits = torch.topk(F.softmax(logits[:, ixx[1].item()], dim=-1), k=k, dim=-1)
                    # topk_vals_logits_u, topk_idx_logits_u = torch.topk(F.softmax(logits_u[:, ixx[1].item()], dim=-1), k=k, dim=-1)

                    # # 두 확률분포 차이 계산 (예: L1 거리)
                    # # 먼저 두 분포의 top-k 인덱스를 동일하게 맞춰야 함
                    # # 여기서는 단순히 같은 위치의 top-k 값끼리 비교
                    # topk_diff = torch.abs(topk_vals_logits - topk_vals_logits_u)

                    # # k개의 차이를 평균내서 스칼라로
                    # mean_topk_diff = topk_diff.mean()

                    # c += mean_topk_diff.item()


                    p_cond = F.softmax(logits[:, ixx[1].item()].to(torch.float32), dim=-1).detach().cpu().numpy().flatten()
                    p_uncond = F.softmax(logits_u[:, ixx[1].item()].to(torch.float32), dim=-1).detach().cpu().numpy().flatten()
                    pdm_value = PDM(p_cond, p_uncond, dist_fn="hellinger")
                    c += pdm_value.item()

                history[i].append(c/2)
    # breakpoint()
    # print(f"NFE: {NFE} Num Blocks: {num_blocks}")
    if verbose:
        return x,history
    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
   

if __name__ == '__main__':
    main()