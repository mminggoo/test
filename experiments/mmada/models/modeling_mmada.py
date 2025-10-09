# from __future__ import annotations

# import logging
# import math
# import sys
# from abc import abstractmethod
# from collections import defaultdict
# from functools import partial
# from typing import (
#     Callable,
#     Dict,
#     Iterable,
#     List,
#     NamedTuple,
#     Optional,
#     Sequence,
#     Set,
#     Tuple,
#     cast,
# )
# from dataclasses import fields
# from typing import List, Optional, Tuple, Union
# import numpy as np
# import torch
# import torch.backends.cuda
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import einsum
# from transformers import PreTrainedModel
# from transformers.modeling_outputs import CausalLMOutputWithPast
# from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
# from transformers.cache_utils import Cache
# from PIL import Image
# from .configuration_llada import (
#     LLaDAConfig,
#     StrEnum,
#     InitFnType,
#     ActivationType,
#     BlockType,
#     LayerNormType,
#     ModelConfig,
#     ActivationCheckpointingStrategy,
# )

# from .modeling_llada import LLaDAModelLM
# from .sampling import cosine_schedule, mask_by_random_topk
# from transformers import PretrainedConfig

# def add_gumbel_noise(logits, temperature):
#     '''
#     The Gumbel max is a method for sampling categorical distributions.
#     According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
#     Thus, we use float64.
#     '''
#     if temperature == 0:
#         return logits
#     logits = logits.to(torch.float64)
#     noise = torch.rand_like(logits, dtype=torch.float64)
#     gumbel_noise = (- torch.log(noise)) ** temperature
#     return logits.exp() / gumbel_noise


# def get_num_transfer_tokens(mask_index, steps):
#     '''
#     In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#     Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#     the expected number of tokens transitioned at each step should be consistent.

#     This function is designed to precompute the number of tokens that need to be transitioned at each step.
#     '''
#     mask_num = mask_index.sum(dim=1, keepdim=True)

#     base = mask_num // steps
#     remainder = mask_num % steps

#     num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, :remainder[i]] += 1

#     return num_transfer_tokens

# class MMadaConfig(PretrainedConfig):
#     model_type = "mmada"

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
        
#         allowed_keys = [
#             "vocab_size",
#             "llm_vocab_size",
#             "llm_model_path",
#             "codebook_size",
#             "num_vq_tokens",
#             "num_new_special_tokens",
#             "gradient_checkpointing",
#             "new_vocab_size",
#         ]

#         for key in allowed_keys:
#             if key in kwargs:
#                 setattr(self, key, kwargs[key])



# class MMadaModelLM(LLaDAModelLM):
#     config_class = MMadaConfig
#     base_model_prefix = "model"
#     def __init__(self, config: MMadaConfig, *args, **kwargs):
#         print(f"Initializing MMadaModelLM with config: {config}")
#         super().__init__(config, *args, **kwargs)

#         # # resize token embeddings
#         # print(f"Resizing token embeddings to {config.new_vocab_size}")
#         # self.resize_token_embeddings(config.new_vocab_size)

#     @torch.no_grad()
#     def t2i_generate(
#             self,
#             input_ids: torch.LongTensor = None,
#             uncond_input_ids: torch.LongTensor = None,
#             attention_mask=None,
#             uncond_attention_mask=None,
#             temperature=1.0,
#             timesteps=18,  # ideal number of steps is 18 in maskgit paper
#             guidance_scale=0,
#             noise_schedule=cosine_schedule,
#             generator: torch.Generator = None,
#             config=None,
#             seq_len=1024,
#             mask_token_id = 126336,
#             resolution = 512,
#             codebook_size = 8192,
#             **kwargs,
#     ):
#         """
#         Generate 1:1 similar to the original MaskGit repo
#         https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
#         """

#         # begin with all image token ids masked
#         # 计算有多少个mask token
#         mask_count = (input_ids == mask_token_id).sum().item()
#         num_vq_tokens = seq_len
#         num_new_special_tokens = 0
#         uni_prompting = kwargs.get("uni_prompting", None)
#         # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
#         input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
#         input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

#         # for classifier-free guidance
#         if uncond_input_ids is not None:
#             uncond_prefix = uncond_input_ids[:, :resolution + 1]

#         for step in range(timesteps):
#             if uncond_input_ids is not None and guidance_scale > 0:
#                 uncond_input_ids = torch.cat(
#                     [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
#                 model_input = torch.cat([input_ids, uncond_input_ids])
#                 all_attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
#                 attention_bias = (all_attention_mask[:, :, None] & all_attention_mask[:, None, :]).bool().unsqueeze(1)
#                 logits = self(model_input, attention_bias=attention_bias).logits 
#                 # print(f"logits.shape: {logits.shape}")
#                 cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
#                 # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
#                 # it seems that muse has a different cfg setting
#                 logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
#                 logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
#             else:
#                 attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
#                 logits = self(input_ids, attention_bias=attention_bias).logits
#                 logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

#             # logits: 1, 1024, 8192
#             # print(f"logits.shape: {logits.shape}")
#             probs = logits.softmax(dim=-1)
#             sampled = probs.reshape(-1, logits.size(-1))
#             # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
#             sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

#             unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
#             # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
#             sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
#             # Defines the mask ratio for the next round. The number to mask out is
#             # determined by mask_ratio * unknown_number_in_the_beginning.
#             ratio = 1.0 * (step + 1) / timesteps
#             mask_ratio = noise_schedule(torch.tensor(ratio))
#             # Computes the probabilities of each selected tokens.
#             selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
#             selected_probs = selected_probs.squeeze(-1)

#             # Ignores the tokens given in the input by overwriting their confidence.
#             selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
#             # Gets mask lens for each sample in the batch according to the mask ratio.
#             mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
#             # Keeps at least one of prediction in this round and also masks out at least
#             # one and for the next iteration
#             mask_len = torch.max(
#                 torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
#             )
#             # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
#             # Adds noise for randomness
#             temperature = temperature * (1.0 - ratio)
#             masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
#             # Masks tokens with lower confidence.
#             input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
#                                                           sampled_ids + len(uni_prompting.text_tokenizer)
#                                                           + num_new_special_tokens)
#             input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

#         return sampled_ids
    
#     def forward_process(
#             self,
#             input_ids, 
#             labels,
#             batch_size_t2i=0,
#             batch_size_lm=0,
#             batch_size_mmu=0,
#             max_seq_length=128,
#             p_mask_lm=None,
#             p_mask_mmu=None,
#             answer_lengths=None,
#             t2i_masks=None,
#             answer_lengths_lm=None
#             ):
#         # attention bias, True for batch_size, 1, seq_len, seq_len  
#         attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
#         attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
#         attention_bias[:batch_size_t2i] = attention_bias_t2i
#         logits = self(input_ids, attention_bias=attention_bias).logits 
#         self.output_size = logits.shape[-1]

#         if batch_size_t2i == 0:
#             loss_t2i = torch.tensor(0.0, device=input_ids.device)
#         else:
#             loss_t2i = F.cross_entropy(
#                 logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
#                 labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
#                 )
        
#         masked_indices = input_ids == self.config.mask_token_id 
#         masked_indices_lm = masked_indices[batch_size_t2i:batch_size_t2i + batch_size_lm]
#         masked_indices_mmu = masked_indices[-batch_size_mmu:]
#         p_mask_lm = p_mask_lm.to(masked_indices_lm.device)
#         p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)       
#         answer_lengths = answer_lengths.to(masked_indices_mmu.device) 
#         loss_lm = F.cross_entropy(
#             logits[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
#             labels[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
#             )/p_mask_lm[masked_indices_lm]

#         if answer_lengths_lm is not None:
#             loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0])  
#         else:
#             loss_lm = loss_lm.sum() / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0] * logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[1])     

#         loss_mmu = F.cross_entropy(
#             logits[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1, self.output_size),
#             labels[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
#             )/p_mask_mmu[masked_indices_mmu]
#         loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[-batch_size_mmu:].shape[0])
        
#         return logits, loss_t2i, loss_lm, loss_mmu

#     def forward_process_with_r2i(
#             self,
#             input_ids, 
#             labels,
#             t2i_masks=None,
#             max_seq_length=128,
#             batch_size_t2i=0,
#             batch_size_lm=0,
#             batch_size_mmu=0,
#             batch_size_r2i=0,
#             p_mask_lm=None,
#             p_mask_mmu=None,
#             p_mask_r2i=None,
#             answer_lengths=None,
#             answer_lengths_lm=None,
#             answer_lengths_r2i=None,
#             ):
#         # attention bias, True for batch_size, 1, seq_len, seq_len  
#         attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
#         attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
#         attention_bias[:batch_size_t2i] = attention_bias_t2i
#         logits = self(input_ids, attention_bias=attention_bias).logits 
#         # logits = self(input_ids).logits
#         self.output_size = logits.shape[-1]

#         if batch_size_t2i == 0:
#             loss_t2i = torch.tensor(0.0, device=input_ids.device)
#         else:
#             # t2i loss
#             loss_t2i = F.cross_entropy(
#                 logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
#                 labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
#                 )
        
#         # llada loss  

#         start_lm = batch_size_t2i
#         end_lm = start_lm + batch_size_lm
#         start_mmu = end_lm
#         end_mmu = start_mmu + batch_size_mmu
#         start_r2i = end_mmu
#         end_r2i = start_r2i + batch_size_r2i

#         masked_indices = input_ids == self.config.mask_token_id 
#         masked_indices_lm = masked_indices[start_lm:end_lm]
#         masked_indices_mmu = masked_indices[start_mmu:end_mmu]
#         masked_indices_r2i = masked_indices[start_r2i:end_r2i]

#         p_mask_lm = p_mask_lm.to(masked_indices_lm.device)
#         p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)
#         p_mask_r2i = p_mask_r2i.to(masked_indices_r2i.device)

#         answer_lengths = answer_lengths.to(masked_indices_mmu.device) 
#         answer_lengths_lm = answer_lengths_lm.to(masked_indices_lm.device)
#         answer_lengths_r2i = answer_lengths_r2i.to(masked_indices_r2i.device)

#         loss_lm = F.cross_entropy(
#             logits[start_lm:end_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
#             labels[start_lm:end_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
#             )/p_mask_lm[masked_indices_lm]

#         if answer_lengths_lm is not None:
#             loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[start_lm:end_lm].shape[0]) 
#         else:
#             loss_lm = loss_lm.sum() / (logits[start_lm:end_lm].shape[0] * logits[start_lm:end_lm].shape[1])

#         loss_mmu = F.cross_entropy(
#             logits[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1, self.output_size),
#             labels[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
#             )/p_mask_mmu[masked_indices_mmu]
#         loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[start_mmu:end_mmu].shape[0])
        
#         loss_r2i = F.cross_entropy(
#             logits[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1, self.output_size),
#             labels[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1), ignore_index=-100, reduction='none'
#             )/p_mask_r2i[masked_indices_r2i]
#         loss_r2i = torch.sum(loss_r2i/answer_lengths_r2i[masked_indices_r2i]) / (logits[start_r2i:end_r2i].shape[0])
        
#         return logits, loss_t2i, loss_lm, loss_mmu, loss_r2i


#     def forward_t2i(
#             self,
#             input_ids, 
#             labels,
#             batch_size_t2i=0,
#             max_seq_length=128,
#             t2i_masks=None
#             ):
#         # attention bias, True for batch_size, 1, seq_len, seq_len  
#         attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
#         attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
#         attention_bias[:batch_size_t2i] = attention_bias_t2i
#         logits = self(input_ids, attention_bias=attention_bias).logits 
#         # logits = self(input_ids).logits
#         self.output_size = logits.shape[-1]

#         # print(f"logits shape: {logits.shape}") B, 359, vocab_size

#         loss_t2i = F.cross_entropy(
#             logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
#             labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
#             )
        
#         return loss_t2i





#     @torch.no_grad()
#     def mmu_generate(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None, vfg_scale=0.0):
#         """
#         Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
#         the sequence max_new_tokens times, feeding the predictions back into the model each time.
#         Most likely you'll want to make sure to be in model.eval() mode of operation for this.
#         """

#         if attention_mask is not None and 0.0 in attention_mask:
#             attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
#             # print(f"attention_bias: {attention_bias}")
#         else:
#             attention_bias = None
#         try:
#             device = idx.device
#         except:
#             device = input_embeddings.device

#         result = []
#         batch_size = idx.shape[0]
#         x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
#         x[:, :idx.shape[1]] = idx.clone()
#         prompt_index = (x != mask_id)
        
#         # vfg용 시각 토큰 인덱스 (SOI/EOI)
#         soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
#         eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
#         start = soi_pos.min().item() if len(soi_pos) > 0 else 0
#         end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
#         visual_index = (start, end)

#         assert max_new_tokens % block_length == 0
#         num_blocks = max_new_tokens // block_length

#         assert steps % num_blocks == 0
#         steps = steps // num_blocks
        
#         # print(f"num_blocks: {num_blocks}, steps: {steps}")
#         # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
#         for num_block in range(num_blocks):
#             block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
#             num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#             # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
#             # print(f"num_transfer_tokens: {num_transfer_tokens}, num_transfer_tokens.shape: {num_transfer_tokens.shape}")
#             for i in range(steps):
#                 mask_index = (x == mask_id) 
#                 if cfg_scale > 0.0:
#                     un_x = x.clone()
#                     un_x[prompt_index] = mask_id
#                     x_ = torch.cat([x, un_x], dim=0)
#                     logits = self(x_).logits
#                     logits, un_logits = torch.chunk(logits, 2, dim=0)
#                     logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#                 else:
#                     logits = self(x, attention_bias=attention_bias).logits

#                 # vfg_scale 적용
#                 if vfg_scale > 0.0:
#                     un_x_v = x.clone()
#                     un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
#                     x_v = torch.cat([x, un_x_v], dim=0)
#                     logits_v = self(x_v).logits
#                     logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
#                     logits = logits_u + (vfg_scale + 1) * (logits_c - logits_u)
#                 else:
#                     logits = self(x, attention_bias=attention_bias).logits
                
#                 logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#                 x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
#                 if remasking == 'low_confidence':
#                     p = F.softmax(logits.to(torch.float64), dim=-1)
#                     x0_p = torch.squeeze(
#                         torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#                 elif remasking == 'random':
#                     x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#                 else:
#                     raise NotImplementedError(remasking)

#                 x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

#                 x0 = torch.where(mask_index, x0, x)
#                 confidence = torch.where(mask_index, x0_p, -np.inf)

#                 transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#                 for j in range(confidence.shape[0]):
#                     _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                     transfer_index[j, select_index] = True
#                 x[transfer_index] = x0[transfer_index]
                
#             # logits = logits[:, -1, :] / temperature
#             # # optionally crop the logits to only the top k options
#             # if top_k is not None:
#             #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#             #     logits[logits < v[:, [-1]]] = -float('Inf')
#             # # apply softmax to convert logits to (normalized) probabilities
#             # probs = F.softmax(logits, dim=-1)
#             # # sample from the distribution
#             # idx_next = torch.multinomial(probs, num_samples=1)
#             # result.append(idx_next[0][0])
#             # # append sampled index to the running sequence and continue
#             # if self.config.w_clip_vit:
#             #     idx_next_embeddings = self.mmada.model.embed_tokens(idx_next)
#             #     input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
#             # else:
#             #     idx = torch.cat((idx, idx_next), dim=1)

#             # if eot_token is not None and idx_next.cpu() == eot_token:
#             #     break

#         return x

#     @torch.no_grad()
#     def mmu_generate_measure(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128, block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None, vfg_scale=0.0):
#         """
#         mmu_generate 함수와 동일하지만 마지막 2개 토큰이 몇 번째 스텝에서 transfer되는지 측정하는 버전
#         """
#         if attention_mask is not None and 0.0 in attention_mask:
#             attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
#         else:
#             attention_bias = None
#         try:
#             device = idx.device
#         except:
#             device = input_embeddings.device

#         result = []
#         batch_size = idx.shape[0]
#         x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
#         x[:, :idx.shape[1]] = idx.clone()
#         prompt_index = (x != mask_id)
        
#         # vfg용 시각 토큰 인덱스 (SOI/EOI)
#         soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
#         eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
#         start = soi_pos.min().item() if len(soi_pos) > 0 else 0
#         end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
#         visual_index = (start, end)
        
#         assert max_new_tokens % block_length == 0
#         num_blocks = max_new_tokens // block_length

#         assert steps % num_blocks == 0
#         steps = steps // num_blocks
        
#         # 마지막 2개 토큰의 transfer 스텝을 추적하기 위한 변수들
#         last_two_transfer_step = [-1, -1]  # 마지막 2개 토큰이 transfer된 스텝
#         last_two_positions = None
        
#         for num_block in range(num_blocks):
#             block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
#             num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#             for i in range(steps):
#                 mask_index = (x == mask_id) 
#                 if cfg_scale > 0.0:
#                     un_x = x.clone()
#                     un_x[prompt_index] = mask_id
#                     x_ = torch.cat([x, un_x], dim=0)
#                     logits = self(x_).logits
#                     logits, un_logits = torch.chunk(logits, 2, dim=0)
#                     logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#                 else:
#                     logits = self(x, attention_bias=attention_bias).logits

#                 # vfg_scale 적용
#                 if vfg_scale > 0.0:
#                     un_x_v = x.clone()
#                     un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
#                     x_v = torch.cat([x, un_x_v], dim=0)
#                     logits_v = self(x_v).logits
#                     logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
#                     logits = logits_u + (vfg_scale + 1) * (logits_c - logits_u)
#                 
#                 logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#                 x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
#                 if remasking == 'low_confidence':
#                     p = F.softmax(logits.to(torch.float64), dim=-1)
#                     x0_p = torch.squeeze(
#                         torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#                 elif remasking == 'random':
#                     x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#                 else:
#                     raise NotImplementedError(remasking)

#                 x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

#                 x0 = torch.where(mask_index, x0, x)
#                 confidence = torch.where(mask_index, x0_p, -np.inf)

#                 transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#                 for j in range(confidence.shape[0]):
#                     _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                     transfer_index[j, select_index] = True
                    
#                     # 마지막 2개 토큰의 transfer 스텝 추적
#                     if last_two_positions is None:
#                         last_two_positions = torch.arange(x0.shape[1] - 2, x0.shape[1], device=x0.device)
                    
#                     # 모든 토큰의 confidence 출력
#                     print(f"[mmu_generate_measure] Block {num_block}, Step {i}:")
#                     print(f"  - Transfer할 토큰 수: {num_transfer_tokens[j, i]}")
#                     print(f"  - Transfer된 토큰 위치: {select_index.tolist()}")
#                     print(f"  - Transfer된 토큰 confidence: {confidence[j, select_index].tolist()}")
                    
#                     # 마지막 2개 토큰 정보
#                     if last_two_positions is None:
#                         last_two_positions = torch.arange(x0.shape[1] - 2, x0.shape[1], device=x0.device)
#                     print(f"  - 마지막 2개 토큰 위치: {last_two_positions.tolist()}")
#                     print(f"  - 마지막 2개 토큰 confidence: {confidence[j, last_two_positions].tolist()}")
                    
#                     # 마지막 2개 토큰이 transfer되었는지 확인
#                     last_two_in_transfer = any(pos in select_index for pos in last_two_positions)
#                     print(f"  - 마지막 2개 토큰이 transfer됨: {last_two_in_transfer}")
                    
#                     # 생성 영역의 confidence 분포 (input 이후부터)
#                     input_len = idx.shape[1]
#                     seq_len = confidence[j].shape[0]
#                     gen_len = seq_len - input_len
#                     print(f"  - Input 길이: {input_len}, 생성 영역 길이: {gen_len}")
                    
#                     if gen_len > 0:
#                         # 생성 영역의 처음 10개 토큰 confidence
#                         gen_start = input_len
#                         gen_first_10 = confidence[j, gen_start:min(gen_start + 10, seq_len)]
#                         print(f"  - 생성 영역 처음 10개 토큰 confidence (위치 {gen_start}-{min(gen_start + 9, seq_len-1)}): {gen_first_10.tolist()}")
                        
#                         # 생성 영역의 중간 10개 토큰 confidence
#                         if gen_len > 20:
#                             gen_mid_start = gen_start + (gen_len // 2 - 5)
#                             gen_mid_end = min(seq_len, gen_start + (gen_len // 2 + 5))
#                             gen_mid_10 = confidence[j, gen_mid_start:gen_mid_end]
#                             print(f"  - 생성 영역 중간 10개 토큰 confidence (위치 {gen_mid_start}-{gen_mid_end-1}): {gen_mid_10.tolist()}")
                        
#                         # 생성 영역의 마지막 10개 토큰 confidence
#                         if gen_len > 10:
#                             gen_last_10 = confidence[j, max(gen_start, seq_len-10):seq_len]
#                             print(f"  - 생성 영역 마지막 10개 토큰 confidence: {gen_last_10.tolist()}")
                        
#                         # 생성 영역의 confidence 통계
#                         gen_confidences = confidence[j, gen_start:seq_len]
#                         valid_gen_confidences = gen_confidences[gen_confidences > -np.inf]
#                         if len(valid_gen_confidences) > 0:
#                             print(f"  - 생성 영역 confidence 통계:")
#                             print(f"    * 최대값: {valid_gen_confidences.max().item():.6f}")
#                             print(f"    * 최소값: {valid_gen_confidences.min().item():.6f}")
#                             print(f"    * 평균값: {valid_gen_confidences.mean().item():.6f}")
#                             print(f"    * 표준편차: {valid_gen_confidences.std().item():.6f}")
#                             print(f"    * Mask 토큰 수: {(gen_confidences == -np.inf).sum().item()}")
#                             print(f"    * 유효 토큰 수: {len(valid_gen_confidences)}")
#                     
#                     print()
#                     
#                     # 마지막 2개 토큰이 transfer되었는지 확인
#                     for k, pos in enumerate(last_two_positions):
#                         if pos in select_index and last_two_transfer_step[k] == -1:
#                             last_two_transfer_step[k] = num_block * steps + i
#                             print(f"[mmu_generate_measure] 마지막 2개 토큰 중 {k+1}번째 토큰(위치 {pos})이 Block {num_block}, Step {i}에서 transfer됨")
#                 
#                 x[transfer_index] = x0[transfer_index]
#                 
#         # 최종 결과 출력
#         print(f"[mmu_generate_measure] 최종 결과:")
#         print(f"  - 마지막 2개 토큰 transfer 스텝: {last_two_transfer_step}")
#         print(f"  - 첫 번째 토큰: {last_two_transfer_step[0]}번째 스텝")
#         print(f"  - 두 번째 토큰: {last_two_transfer_step[1]}번째 스텝")
#         
#         return x, last_two_transfer_step

#     @torch.no_grad()
#     def mmu_generate_with_last_remasking(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128, block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None, vfg_scale=0.0):
#         """
#         mmu_generate 함수와 동일하지만 마지막 2개 토큰을 마지막 순간까지 remasking하는 버전
#         """
#         if attention_mask is not None and 0.0 in attention_mask:
#             attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
#         else:
#             attention_bias = None
#         try:
#             device = idx.device
#         except:
#             device = input_embeddings.device

#         result = []
#         batch_size = idx.shape[0]
#         x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
#         x[:, :idx.shape[1]] = idx.clone()
#         prompt_index = (x != mask_id)
        
#         # vfg용 시각 토큰 인덱스 (SOI/EOI)
#         soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
#         eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
#         start = soi_pos.min().item() if len(soi_pos) > 0 else 0
#         end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
#         visual_index = (start, end)
        
#         assert max_new_tokens % block_length == 0
#         num_blocks = max_new_tokens // block_length

#         assert steps % num_blocks == 0
#         steps = steps // num_blocks
        
#         for num_block in range(num_blocks):
#             block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
#             num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#             for i in range(steps):
#                 mask_index = (x == mask_id) 
#                 if cfg_scale > 0.0:
#                     un_x = x.clone()
#                     un_x[prompt_index] = mask_id
#                     x_ = torch.cat([x, un_x], dim=0)
#                     logits = self(x_).logits
#                     logits, un_logits = torch.chunk(logits, 2, dim=0)
#                     logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#                 else:
#                     logits = self(x, attention_bias=attention_bias).logits

#                 # vfg_scale 적용
#                 if vfg_scale > 0.0:
#                     un_x_v = x.clone()
#                     un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
#                     x_v = torch.cat([x, un_x_v], dim=0)
#                     logits_v = self(x_v).logits
#                     logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
#                     logits = logits_u + (vfg_scale + 1) * (logits_c - logits_u)
#                 
#                 logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#                 x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
#                 if remasking == 'low_confidence':
#                     p = F.softmax(logits.to(torch.float64), dim=-1)
#                     x0_p = torch.squeeze(
#                         torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#                 elif remasking == 'random':
#                     x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#                 else:
#                     raise NotImplementedError(remasking)

#                 # 마지막 2개 토큰은 마지막 순간까지 remasking
#                 if num_block == num_blocks - 1 and i < steps - 1:  # 마지막 블록이지만 마지막 스텝이 아닌 경우
#                     # 마지막 2개 토큰을 다시 mask로 설정하여 remasking되도록 함
#                     last_two_positions = torch.arange(x0_p.shape[1] - 2, x0_p.shape[1], device=x0_p.device)
#                     x[:, last_two_positions] = mask_id  # 마지막 2개 토큰을 다시 mask로 설정
#                     # confidence를 매우 낮춰서 remasking되도록 함 (0.001%로 설정)
#                     x0_p[:, last_two_positions] = x0_p[:, last_two_positions] * 0.00001  # confidence를 0.001%로 낮춤
#                 
#                 x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

#                 x0 = torch.where(mask_index, x0, x)
#                 confidence = torch.where(mask_index, x0_p, -np.inf)

#                 transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#                 for j in range(confidence.shape[0]):
#                     # 앞쪽부터 transfer하도록 수정
#                     mask_positions = torch.where(mask_index[j])[0]  # mask가 있는 위치들
#                     if len(mask_positions) > 0:
#                         # 앞쪽부터 num_transfer_tokens[j, i]개만큼 선택
#                         num_to_transfer = min(num_transfer_tokens[j, i], len(mask_positions))
#                         select_index = mask_positions[:num_to_transfer]  # 앞쪽부터 선택
#                         transfer_index[j, select_index] = True
#                 x[transfer_index] = x0[transfer_index]
#                 
#             if eot_token is not None:
#                 last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
#                 if last_token_index_in_current_block < x.shape[1]:
#                     tokens_at_block_end = x[:, last_token_index_in_current_block]
#                     if torch.all(tokens_at_block_end == eot_token):
#                         break
#         return x

#     @torch.no_grad()
#     def mmu_generate_fast_with_last_remasking(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128, block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None, vfg_scale=0.0):
#         """
#         mmu_generate_fast 함수와 동일하지만 마지막 2개 토큰을 마지막 순간까지 remasking하는 버전
#         """
#         if attention_mask is not None and 0.0 in attention_mask:
#             attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
#         else:
#             attention_bias = None
#         try:
#             device = idx.device
#         except:
#             device = input_embeddings.device

#         result = []
#         batch_size = idx.shape[0]
#         x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
#         x[:, :idx.shape[1]] = idx.clone()
#         prompt_index = (x != mask_id)
        
#         # vfg용 시각 토큰 인덱스 (SOI/EOI)
#         soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
#         eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
#         start = soi_pos.min().item() if len(soi_pos) > 0 else 0
#         end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
#         visual_index = (start, end)
        
#         assert max_new_tokens % block_length == 0
#         num_blocks = max_new_tokens // block_length

#         assert steps % num_blocks == 0
#         steps = steps // num_blocks
        
#         for num_block in range(num_blocks):
#             block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
#             num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#             for i in range(steps):
#                 mask_index = (x == mask_id) 
#                 if cfg_scale > 0.0:
#                     un_x = x.clone()
#                     un_x[prompt_index] = mask_id
#                     x_ = torch.cat([x, un_x], dim=0)
#                     logits = self(x_).logits
#                     logits, un_logits = torch.chunk(logits, 2, dim=0)
#                     logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#                 else:
#                     logits = self(x, attention_bias=attention_bias).logits

#                 # vfg_scale 적용
#                 if vfg_scale > 0.0:
#                     un_x_v = x.clone()
#                     un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
#                     x_v = torch.cat([x, un_x_v], dim=0)
#                     logits_v = self(x_v).logits
#                     logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
#                     logits = logits_u + (vfg_scale + 1) * (logits_c - logits_u)
#                 
#                 logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#                 x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
#                 if remasking == 'low_confidence':
#                     p = F.softmax(logits.to(torch.float64), dim=-1)
#                     x0_p = torch.squeeze(
#                         torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#                 elif remasking == 'random':
#                     x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#                 else:
#                     raise NotImplementedError(remasking)

#                 x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

#                 x0 = torch.where(mask_index, x0, x)
#                 confidence = torch.where(mask_index, x0_p, -np.inf)

#                 transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#                 for j in range(confidence.shape[0]):
#                     _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                     transfer_index[j, select_index] = True
#                 x[transfer_index] = x0[transfer_index]
#             if eot_token is not None:
#                 last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
#                 if last_token_index_in_current_block < x.shape[1]:
#                     tokens_at_block_end = x[:, last_token_index_in_current_block]
#                     if torch.all(tokens_at_block_end == eot_token):
#                         break
#         return x

#     @torch.no_grad()
#     def mmu_generate_fast(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None, vfg_scale=0.0):
#         """
#         Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
#         the sequence max_new_tokens times, feeding the predictions back into the model each time.
#         Most likely you'll want to make sure to be in model.eval() mode of operation for this.
#         """

#         if attention_mask is not None and 0.0 in attention_mask:
#             attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
#             # print(f"attention_bias: {attention_bias}")
#         else:
#             attention_bias = None
#         try:
#             device = idx.device
#         except:
#             device = input_embeddings.device

#         result = []
#         batch_size = idx.shape[0]
#         x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
#         x[:, :idx.shape[1]] = idx.clone()
#         prompt_index = (x != mask_id)
        
#         # vfg용 시각 토큰 인덱스 (SOI/EOI)
#         soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
#         eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
#         start = soi_pos.min().item() if len(soi_pos) > 0 else 0
#         end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
#         visual_index = (start, end)
        
#         assert max_new_tokens % block_length == 0
#         num_blocks = max_new_tokens // block_length

#         assert steps % num_blocks == 0
#         steps = steps // num_blocks
        
#         for num_block in range(num_blocks):
#             block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
#             num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#             for i in range(steps):
#                 mask_index = (x == mask_id) 
#                 if cfg_scale > 0.0:
#                     un_x = x.clone()
#                     un_x[prompt_index] = mask_id
#                     x_ = torch.cat([x, un_x], dim=0)
#                     logits = self(x_).logits
#                     logits, un_logits = torch.chunk(logits, 2, dim=0)
#                     logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#                 else:
#                     logits = self(x, attention_bias=attention_bias).logits

#                 # vfg_scale 적용
#                 if vfg_scale > 0.0:
#                     un_x_v = x.clone()
#                     un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
#                     x_v = torch.cat([x, un_x_v], dim=0)
#                     logits_v = self(x_v).logits
#                     logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
#                     logits = logits_u + (vfg_scale + 1) * (logits_c - logits_u)
#                 
#                 logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#                 x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
#                 if remasking == 'low_confidence':
#                     p = F.softmax(logits.to(torch.float64), dim=-1)
#                     x0_p = torch.squeeze(
#                         torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#                 elif remasking == 'random':
#                     x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#                 else:
#                     raise NotImplementedError(remasking)

#                 x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

#                 x0 = torch.where(mask_index, x0, x)
#                 confidence = torch.where(mask_index, x0_p, -np.inf)

#                 transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#                 for j in range(confidence.shape[0]):
#                     _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                     transfer_index[j, select_index] = True
#                 x[transfer_index] = x0[transfer_index]
#             if eot_token is not None:
#                 last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
#                 if last_token_index_in_current_block < x.shape[1]:
#                     tokens_at_block_end = x[:, last_token_index_in_current_block]
#                     if torch.all(tokens_at_block_end == eot_token):
#                         break
#         return x

#     @torch.no_grad()
#     def t2i_generate_decoding_stepwise(
#             self,
#             input_ids: torch.LongTensor = None,
#             uncond_input_ids: torch.LongTensor = None,
#             attention_mask=None,
#             uncond_attention_mask=None,
#             temperature=1.0,
#             timesteps=18,  # ideal number of steps is 18 in maskgit paper
#             guidance_scale=0,
#             noise_schedule=cosine_schedule,
#             generator: torch.Generator = None,
#             config=None,
#             seq_len=1024,
#             mask_token_id = 126336,
#             resolution = 512,
#             codebook_size = 8192,
#             vq_model = None,
#             **kwargs,
#     ):
#         """
#         Generate 1:1 similar to the original MaskGit repo
#         https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
#         """

#         # begin with all image token ids masked
#         # 计算有多少个mask token
#         mask_count = (input_ids == mask_token_id).sum().item()
#         num_vq_tokens = seq_len
#         num_new_special_tokens = 0
#         uni_prompting = kwargs.get("uni_prompting", None)
#         # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
#         input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
#         input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

#         # for classifier-free guidance
#         if uncond_input_ids is not None:
#             uncond_prefix = uncond_input_ids[:, :resolution + 1]

#         for step in range(timesteps):
#             if uncond_input_ids is not None and guidance_scale > 0:
#                 uncond_input_ids = torch.cat(
#                     [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
#                 model_input = torch.cat([input_ids, uncond_input_ids])
#                 attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
#                 attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
#                 logits = self(model_input, attention_bias=attention_bias).logits 
#                 # print(f"logits.shape: {logits.shape}")
#                 cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
#                 # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
#                 # it seems that muse has a different cfg setting
#                 logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
#                 logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
#             else:
#                 attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
#                 logits = self(input_ids, attention_bias=attention_bias).logits
#                 logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

#             # logits: 1, 1024, 8192
#             # print(f"logits.shape: {logits.shape}")
#             probs = logits.softmax(dim=-1)
#             sampled = probs.reshape(-1, logits.size(-1))
#             # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
#             sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

#             unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
#             # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
#             sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
#             # Defines the mask ratio for the next round. The number to mask out is
#             current_image_vq_indices = sampled_ids.clone()
#             # print(f"current_image_vq_indices: {current_image_vq_indices}")
#             current_image_vq_indices = torch.clamp(current_image_vq_indices, 0, 8192 - 1)
#             current_image = vq_model.decode_code(current_image_vq_indices)
#             images = torch.clamp((current_image + 1.0) / 2.0, min=0.0, max=1.0)
#             images *= 255.0
#             images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
#             pil_images = Image.fromarray(images[0]) 
#             yield pil_images, f"Step {step + 1}/{timesteps}"
#             # determined by mask_ratio * unknown_number_in_the_beginning.
#             ratio = 1.0 * (step + 1) / timesteps
#             mask_ratio = noise_schedule(torch.tensor(ratio))
#             # Computes the probabilities of each selected tokens.
#             selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
#             selected_probs = selected_probs.squeeze(-1)

#             # Ignores the tokens given in the input by overwriting their confidence.
#             selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
#             # Gets mask lens for each sample in the batch according to the mask ratio.
#             mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
#             # Keeps at least one of prediction in this round and also masks out at least
#             # one and for the next iteration
#             mask_len = torch.max(
#                 torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
#             )
#             # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
#             # Adds noise for randomness
#             temperature = temperature * (1.0 - ratio)
#             masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
#             # Masks tokens with lower confidence.
#             input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
#                                                           sampled_ids + len(uni_prompting.text_tokenizer)
#                                                           + num_new_special_tokens)
#             input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
            

#         return sampled_ids
    

#     # ----------------------------
#     # NEW: Position-Aware Confidence Penalty Generate
#     # ----------------------------
#     @torch.no_grad()
#     def mmu_generate_posaware(
#         self,
#         idx=None,
#         input_embeddings=None,
#         max_new_tokens=128,
#         steps=128,
#         block_length=128,
#         temperature=0.0,
#         top_k=None,
#         eot_token=None,
#         cfg_scale=0.0,
#         remasking='low_confidence',
#         mask_id=126336,
#         attention_mask=None,
#         pos_penalty_gamma: float = 0.5,
#         pos_penalty_alpha: float = 1.0,
#         vfg_scale: float = 0.0,
#     ):
#         """
#         Generate with Position-Aware Confidence Penalty.
#         뒤쪽(정답 영역) 토큰의 confidence를 초기 단계에서 위치 기반으로 감쇠하여
#         추론(앞부분) 토큰이 먼저 고정되도록 유도합니다.
#         기존 함수는 유지하고, 새 함수로만 변경사항을 적용합니다.
#         """
#         # attention bias 준비
#         if attention_mask is not None and 0.0 in attention_mask:
#             attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
#         else:
#             attention_bias = None

#         try:
#             device = idx.device
#         except:
#             device = input_embeddings.device

#         batch_size = idx.shape[0]
#         x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
#         x[:, :idx.shape[1]] = idx.clone()
#         prompt_index = (x != mask_id)

#         # vfg용 시각 토큰 인덱스 (SOI/EOI)
#         soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
#         eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
#         start = soi_pos.min().item() if len(soi_pos) > 0 else 0
#         end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
#         visual_index = (start, end)

#         assert max_new_tokens % block_length == 0
#         num_blocks = max_new_tokens // block_length

#         assert steps % num_blocks == 0
#         steps = steps // num_blocks

#         for num_block in range(num_blocks):
#             block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
#             num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

#             for i in range(steps):
#                 mask_index = (x == mask_id)

#                 # logits 계산
#                 if cfg_scale > 0.0:
#                     un_x = x.clone()
#                     un_x[prompt_index] = mask_id
#                     x_ = torch.cat([x, un_x], dim=0)
#                     logits = self(x_).logits
#                     logits, un_logits = torch.chunk(logits, 2, dim=0)
#                     logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#                 else:
#                     logits = self(x, attention_bias=attention_bias).logits

#                 # vfg_scale 적용
#                 if vfg_scale > 0.0:
#                     un_x_v = x.clone()
#                     un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
#                     x_v = torch.cat([x, un_x_v], dim=0)
#                     logits_v = self(x_v).logits
#                     logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
#                     logits = logits_u + (vfg_scale + 1) * (logits_c - logits_u)

#                 # 샘플링 후보 및 confidence 계산
#                 logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#                 x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

#                 if remasking == 'low_confidence':
#                     p = F.softmax(logits.to(torch.float64), dim=-1)
#                     x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # [B, L]
#                 elif remasking == 'random':
#                     x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#                 else:
#                     raise NotImplementedError(remasking)

#                 # ---- Position-Aware Confidence Penalty ----
#                 if pos_penalty_gamma is not None and pos_penalty_gamma > 0.0:
#                     block_start = idx.shape[1] + num_block * block_length
#                     block_end   = idx.shape[1] + (num_block + 1) * block_length  # [start, end)

#                     seq_len = x0_p.shape[1]
#                     pos1d = torch.arange(seq_len, device=x0_p.device)

#                     denom = max(block_length - 1, 1)
#                     rel1d = (pos1d - block_start).clamp(min=0, max=denom) / denom  # [0,1] within block

#                     # 진행률 (0,1], 초기일수록 큰 패널티
#                     t = float(i + 1) / float(steps)
#                     # penalty = 1 - gamma * (1 - t) * (rel^alpha)
#                     penalty1d = 1.0 - float(pos_penalty_gamma) * (1.0 - t) * (rel1d ** float(pos_penalty_alpha))
#                     penalty1d = penalty1d.clamp(min=0.0)

#                     # 블록 외부는 패널티 1.0 유지
#                     outside = (pos1d < block_start) | (pos1d >= block_end)
#                     if outside.any():
#                         penalty1d = torch.where(outside, torch.ones_like(penalty1d), penalty1d)

#                     x0_p = x0_p * penalty1d.to(dtype=x0_p.dtype).unsqueeze(0)
#                 # -------------------------------------------

#                 # 현재 블록 범위를 넘어서는 위치는 선택에서 제외
#                 x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

#                 # 후보 토큰/신뢰도 갱신
#                 x0 = torch.where(mask_index, x0, x)
#                 confidence = torch.where(mask_index, x0_p, -np.inf)

#                 # 상위 confidence 위치를 transfer
#                 transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#                 for j in range(confidence.shape[0]):
#                     _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                     transfer_index[j, select_index] = True

#                 x[transfer_index] = x0[transfer_index]

#         return x


# AutoConfig.register("mmada", MMadaConfig)
# AutoModelForCausalLM.register(MMadaConfig, MMadaModelLM)
# AutoModel.register(MMadaConfig, MMadaModelLM)


from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache
from PIL import Image
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

from .modeling_llada import LLaDAModelLM
from .sampling import cosine_schedule, mask_by_random_topk
from transformers import PretrainedConfig

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

class MMadaConfig(PretrainedConfig):
    model_type = "mmada"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        allowed_keys = [
            "vocab_size",
            "llm_vocab_size",
            "llm_model_path",
            "codebook_size",
            "num_vq_tokens",
            "num_new_special_tokens",
            "gradient_checkpointing",
            "new_vocab_size",
        ]

        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])



class MMadaModelLM(LLaDAModelLM):
    config_class = MMadaConfig
    base_model_prefix = "model"
    def __init__(self, config: MMadaConfig, *args, **kwargs):
        print(f"Initializing MMadaModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)

        # # resize token embeddings
        # print(f"Resizing token embeddings to {config.new_vocab_size}")
        # self.resize_token_embeddings(config.new_vocab_size)

    @torch.no_grad()
    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # 计算有多少个mask token
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                all_attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (all_attention_mask[:, :, None] & all_attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids
    
    def forward_process(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            max_seq_length=128,
            p_mask_lm=None,
            p_mask_mmu=None,
            answer_lengths=None,
            t2i_masks=None,
            answer_lengths_lm=None
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        self.output_size = logits.shape[-1]

        if batch_size_t2i == 0:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)
        else:
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
        
        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_lm = masked_indices[batch_size_t2i:batch_size_t2i + batch_size_lm]
        masked_indices_mmu = masked_indices[-batch_size_mmu:]
        p_mask_lm = p_mask_lm.to(masked_indices_lm.device)
        p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)       
        answer_lengths = answer_lengths.to(masked_indices_mmu.device) 
        loss_lm = F.cross_entropy(
            logits[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
            labels[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_lm[masked_indices_lm]

        if answer_lengths_lm is not None:
            loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0])  
        else:
            loss_lm = loss_lm.sum() / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0] * logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[1])     

        loss_mmu = F.cross_entropy(
            logits[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1, self.output_size),
            labels[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_mmu[masked_indices_mmu]
        loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[-batch_size_mmu:].shape[0])
        
        return logits, loss_t2i, loss_lm, loss_mmu

    def forward_process_with_r2i(
            self,
            input_ids, 
            labels,
            t2i_masks=None,
            max_seq_length=128,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            batch_size_r2i=0,
            p_mask_lm=None,
            p_mask_mmu=None,
            p_mask_r2i=None,
            answer_lengths=None,
            answer_lengths_lm=None,
            answer_lengths_r2i=None,
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        if batch_size_t2i == 0:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)
        else:
            # t2i loss
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
        
        # llada loss  

        start_lm = batch_size_t2i
        end_lm = start_lm + batch_size_lm
        start_mmu = end_lm
        end_mmu = start_mmu + batch_size_mmu
        start_r2i = end_mmu
        end_r2i = start_r2i + batch_size_r2i

        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_lm = masked_indices[start_lm:end_lm]
        masked_indices_mmu = masked_indices[start_mmu:end_mmu]
        masked_indices_r2i = masked_indices[start_r2i:end_r2i]

        p_mask_lm = p_mask_lm.to(masked_indices_lm.device)
        p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)
        p_mask_r2i = p_mask_r2i.to(masked_indices_r2i.device)

        answer_lengths = answer_lengths.to(masked_indices_mmu.device) 
        answer_lengths_lm = answer_lengths_lm.to(masked_indices_lm.device)
        answer_lengths_r2i = answer_lengths_r2i.to(masked_indices_r2i.device)

        loss_lm = F.cross_entropy(
            logits[start_lm:end_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
            labels[start_lm:end_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_lm[masked_indices_lm]

        if answer_lengths_lm is not None:
            loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[start_lm:end_lm].shape[0]) 
        else:
            loss_lm = loss_lm.sum() / (logits[start_lm:end_lm].shape[0] * logits[start_lm:end_lm].shape[1])

        loss_mmu = F.cross_entropy(
            logits[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1, self.output_size),
            labels[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_mmu[masked_indices_mmu]
        loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[start_mmu:end_mmu].shape[0])
        
        loss_r2i = F.cross_entropy(
            logits[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1, self.output_size),
            labels[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_r2i[masked_indices_r2i]
        loss_r2i = torch.sum(loss_r2i/answer_lengths_r2i[masked_indices_r2i]) / (logits[start_r2i:end_r2i].shape[0])
        
        return logits, loss_t2i, loss_lm, loss_mmu, loss_r2i


    def forward_t2i(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            max_seq_length=128,
            t2i_masks=None
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        # print(f"logits shape: {logits.shape}") B, 359, vocab_size

        loss_t2i = F.cross_entropy(
            logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
            labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )
        
        return loss_t2i






    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, 
        remasking='low_confidence', 
        mask_id=126336, 
        attention_mask=None,
        pos_penalty_gamma: float = 0.5,
        pos_penalty_alpha: float = 1.0,
        vfg_scale: float = 0.0,
        vfg_start=0.0,
        vfg_end=1.0,):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)

        # vfg용 시각 토큰 인덱스 (SOI/EOI)
        soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
        eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
        start = soi_pos.min().item() if len(soi_pos) > 0 else 0
        end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
        visual_index = (start, end)
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        # print(f"num_blocks: {num_blocks}, steps: {steps}")
        # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
            # print(f"num_transfer_tokens: {num_transfer_tokens}, num_transfer_tokens.shape: {num_transfer_tokens.shape}")
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    # vfg_scale 적용
                    if vfg_scale > 0.0 and int(vfg_start*steps) <= i <= int(vfg_end*steps):
                        un_x_v = x.clone()
                        un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
                        x_v = torch.cat([x, un_x_v], dim=0)
                        logits_v = self(x_v).logits
                        logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
                        logits = logits_u + (vfg_scale + 1) * (logits_c - logits_u)
                    else:
                        logits = self(x, attention_bias=attention_bias).logits
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                elif remasking == 'posaware':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                    
                    block_start = idx.shape[1] + num_block * block_length
                    block_end   = idx.shape[1] + (num_block + 1) * block_length  # [start, end)

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
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
                
            
            # logits = logits[:, -1, :] / temperature
            # # optionally crop the logits to only the top k options
            # if top_k is not None:
            #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            #     logits[logits < v[:, [-1]]] = -float('Inf')
            # # apply softmax to convert logits to (normalized) probabilities
            # probs = F.softmax(logits, dim=-1)
            # # sample from the distribution
            # idx_next = torch.multinomial(probs, num_samples=1)
            # result.append(idx_next[0][0])
            # # append sampled index to the running sequence and continue
            # if self.config.w_clip_vit:
            #     idx_next_embeddings = self.mmada.model.embed_tokens(idx_next)
            #     input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            # else:
            #     idx = torch.cat((idx, idx_next), dim=1)

            # if eot_token is not None and idx_next.cpu() == eot_token:
            #     break

        return x

    @torch.no_grad()
    def mmu_generate_measure(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128, block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None, vfg_scale=0.0):
        """
        mmu_generate 함수와 동일하지만 마지막 2개 토큰이 몇 번째 스텝에서 transfer되는지 측정하는 버전
        """
        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        # vfg용 시각 토큰 인덱스 (SOI/EOI)
        soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
        eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
        start = soi_pos.min().item() if len(soi_pos) > 0 else 0
        end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
        visual_index = (start, end)
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        # 마지막 2개 토큰의 transfer 스텝을 추적하기 위한 변수들
        last_two_transfer_step = [-1, -1]  # 마지막 2개 토큰이 transfer된 스텝
        last_two_positions = None
        
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits

                # vfg_scale 적용
                if vfg_scale > 0.0:
                    un_x_v = x.clone()
                    un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
                    x_v = torch.cat([x, un_x_v], dim=0)
                    logits_v = self(x_v).logits
                    logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
                    logits = logits_u + (vfg_scale + 1) * (logits_c - logits_u)
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                    
                    # 마지막 2개 토큰의 transfer 스텝 추적
                    if last_two_positions is None:
                        last_two_positions = torch.arange(x0.shape[1] - 2, x0.shape[1], device=x0.device)
                    
                    # 모든 토큰의 confidence 출력
                    print(f"[mmu_generate_measure] Block {num_block}, Step {i}:")
                    print(f"  - Transfer할 토큰 수: {num_transfer_tokens[j, i]}")
                    print(f"  - Transfer된 토큰 위치: {select_index.tolist()}")
                    print(f"  - Transfer된 토큰 confidence: {confidence[j, select_index].tolist()}")
                    
                    # 마지막 2개 토큰 정보
                    if last_two_positions is None:
                        last_two_positions = torch.arange(x0.shape[1] - 2, x0.shape[1], device=x0.device)
                    print(f"  - 마지막 2개 토큰 위치: {last_two_positions.tolist()}")
                    print(f"  - 마지막 2개 토큰 confidence: {confidence[j, last_two_positions].tolist()}")
                    
                    # 마지막 2개 토큰이 transfer되었는지 확인
                    last_two_in_transfer = any(pos in select_index for pos in last_two_positions)
                    print(f"  - 마지막 2개 토큰이 transfer됨: {last_two_in_transfer}")
                    
                    # 생성 영역의 confidence 분포 (input 이후부터)
                    input_len = idx.shape[1]
                    seq_len = confidence[j].shape[0]
                    gen_len = seq_len - input_len
                    print(f"  - Input 길이: {input_len}, 생성 영역 길이: {gen_len}")
                    
                    if gen_len > 0:
                        # 생성 영역의 처음 10개 토큰 confidence
                        gen_start = input_len
                        gen_first_10 = confidence[j, gen_start:min(gen_start + 10, seq_len)]
                        print(f"  - 생성 영역 처음 10개 토큰 confidence (위치 {gen_start}-{min(gen_start + 9, seq_len-1)}): {gen_first_10.tolist()}")
                        
                        # 생성 영역의 중간 10개 토큰 confidence
                        if gen_len > 20:
                            gen_mid_start = gen_start + (gen_len // 2 - 5)
                            gen_mid_end = min(seq_len, gen_start + (gen_len // 2 + 5))
                            gen_mid_10 = confidence[j, gen_mid_start:gen_mid_end]
                            print(f"  - 생성 영역 중간 10개 토큰 confidence (위치 {gen_mid_start}-{gen_mid_end-1}): {gen_mid_10.tolist()}")
                        
                        # 생성 영역의 마지막 10개 토큰 confidence
                        if gen_len > 10:
                            gen_last_10 = confidence[j, max(gen_start, seq_len-10):seq_len]
                            print(f"  - 생성 영역 마지막 10개 토큰 confidence: {gen_last_10.tolist()}")
                        
                        # 생성 영역의 confidence 통계
                        gen_confidences = confidence[j, gen_start:seq_len]
                        valid_gen_confidences = gen_confidences[gen_confidences > -np.inf]
                        if len(valid_gen_confidences) > 0:
                            print(f"  - 생성 영역 confidence 통계:")
                            print(f"    * 최대값: {valid_gen_confidences.max().item():.6f}")
                            print(f"    * 최소값: {valid_gen_confidences.min().item():.6f}")
                            print(f"    * 평균값: {valid_gen_confidences.mean().item():.6f}")
                            print(f"    * 표준편차: {valid_gen_confidences.std().item():.6f}")
                            print(f"    * Mask 토큰 수: {(gen_confidences == -np.inf).sum().item()}")
                            print(f"    * 유효 토큰 수: {len(valid_gen_confidences)}")
                    
                    print()
                    
                    # 마지막 2개 토큰이 transfer되었는지 확인
                    for k, pos in enumerate(last_two_positions):
                        if pos in select_index and last_two_transfer_step[k] == -1:
                            last_two_transfer_step[k] = num_block * steps + i
                            print(f"[mmu_generate_measure] 마지막 2개 토큰 중 {k+1}번째 토큰(위치 {pos})이 Block {num_block}, Step {i}에서 transfer됨")
                
                x[transfer_index] = x0[transfer_index]
                
        # 최종 결과 출력
        print(f"[mmu_generate_measure] 최종 결과:")
        print(f"  - 마지막 2개 토큰 transfer 스텝: {last_two_transfer_step}")
        print(f"  - 첫 번째 토큰: {last_two_transfer_step[0]}번째 스텝")
        print(f"  - 두 번째 토큰: {last_two_transfer_step[1]}번째 스텝")
        
        return x, last_two_transfer_step

    @torch.no_grad()
    def mmu_generate_with_last_remasking(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128, block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None, vfg_scale=0.0):
        """
        mmu_generate 함수와 동일하지만 마지막 2개 토큰을 마지막 순간까지 remasking하는 버전
        """
        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        # vfg용 시각 토큰 인덱스 (SOI/EOI)
        soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
        eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
        start = soi_pos.min().item() if len(soi_pos) > 0 else 0
        end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
        visual_index = (start, end)
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits

                # vfg_scale 적용
                if vfg_scale > 0.0:
                    un_x_v = x.clone()
                    un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
                    x_v = torch.cat([x, un_x_v], dim=0)
                    logits_v = self(x_v).logits
                    logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
                    logits = logits_u + (vfg_scale + 1) * (logits_c - logits_u)
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # 마지막 2개 토큰은 마지막 순간까지 remasking
                if num_block == num_blocks - 1 and i < steps - 1:  # 마지막 블록이지만 마지막 스텝이 아닌 경우
                    # 마지막 2개 토큰을 다시 mask로 설정하여 remasking되도록 함
                    last_two_positions = torch.arange(x0_p.shape[1] - 2, x0_p.shape[1], device=x0_p.device)
                    x[:, last_two_positions] = mask_id  # 마지막 2개 토큰을 다시 mask로 설정
                    # confidence를 매우 낮춰서 remasking되도록 함 (0.001%로 설정)
                    x0_p[:, last_two_positions] = x0_p[:, last_two_positions] * 0.00001  # confidence를 0.001%로 낮춤
                
                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    # 앞쪽부터 transfer하도록 수정
                    mask_positions = torch.where(mask_index[j])[0]  # mask가 있는 위치들
                    if len(mask_positions) > 0:
                        # 앞쪽부터 num_transfer_tokens[j, i]개만큼 선택
                        num_to_transfer = min(num_transfer_tokens[j, i], len(mask_positions))
                        select_index = mask_positions[:num_to_transfer]  # 앞쪽부터 선택
                        transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
                
            if eot_token is not None:
                last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
                if last_token_index_in_current_block < x.shape[1]:
                    tokens_at_block_end = x[:, last_token_index_in_current_block]
                    if torch.all(tokens_at_block_end == eot_token):
                        break
        return x

    @torch.no_grad()
    def mmu_generate_fast_with_last_remasking(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128, block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None, vfg_scale=0.0):
        """
        mmu_generate_fast 함수와 동일하지만 마지막 2개 토큰을 마지막 순간까지 remasking하는 버전
        """
        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        # vfg용 시각 토큰 인덱스 (SOI/EOI)
        soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
        eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
        start = soi_pos.min().item() if len(soi_pos) > 0 else 0
        end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
        visual_index = (start, end)
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    # vfg_scale 적용
                    if vfg_scale > 0.0:
                        un_x_v = x.clone()
                        un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
                        x_v = torch.cat([x, un_x_v], dim=0)
                        logits_v = self(x_v).logits
                        logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
                        logits = logits_u + (vfg_scale + 1) * (logits_c - logits_u)
                    else:
                        logits = self(x, attention_bias=attention_bias).logits
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
            if eot_token is not None:
                last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
                if last_token_index_in_current_block < x.shape[1]:
                    tokens_at_block_end = x[:, last_token_index_in_current_block]
                    if torch.all(tokens_at_block_end == eot_token):
                        break
        return x

    @torch.no_grad()
    def mmu_generate_fast(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None, vfg_scale=0.0):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        # vfg용 시각 토큰 인덱스 (SOI/EOI)
        soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
        eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
        start = soi_pos.min().item() if len(soi_pos) > 0 else 0
        end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
        visual_index = (start, end)
        
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits

                # vfg_scale 적용
                if vfg_scale > 0.0:
                    un_x_v = x.clone()
                    un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
                    x_v = torch.cat([x, un_x_v], dim=0)
                    logits_v = self(x_v).logits
                    logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
                    logits = logits_u + (vfg_scale + 1) * (logits_c - logits_u)
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                


                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
            if eot_token is not None:
                last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
                if last_token_index_in_current_block < x.shape[1]:
                    tokens_at_block_end = x[:, last_token_index_in_current_block]
                    if torch.all(tokens_at_block_end == eot_token):
                        break
        return x

    @torch.no_grad()
    def t2i_generate_decoding_stepwise(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            vq_model = None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # 计算有多少个mask token
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            current_image_vq_indices = sampled_ids.clone()
            # print(f"current_image_vq_indices: {current_image_vq_indices}")
            current_image_vq_indices = torch.clamp(current_image_vq_indices, 0, 8192 - 1)
            current_image = vq_model.decode_code(current_image_vq_indices)
            images = torch.clamp((current_image + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = Image.fromarray(images[0]) 
            yield pil_images, f"Step {step + 1}/{timesteps}"
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
            

        return sampled_ids
    

    # ----------------------------
    # NEW: Position-Aware Confidence Penalty Generate
    # ----------------------------
    # @torch.no_grad()
    # def mmu_generate_posaware(
    #     self,
    #     idx=None,
    #     input_embeddings=None,
    #     max_new_tokens=128,
    #     steps=128,
    #     block_length=128,
    #     temperature=0.0,
    #     top_k=None,
    #     eot_token=None,
    #     cfg_scale=0.0,
    #     remasking='low_confidence',
    #     mask_id=126336,
    #     attention_mask=None,
    #     pos_penalty_gamma: float = 0.5,
    #     pos_penalty_alpha: float = 1.0,
    #     vfg_scale: float = 0.0,
    # ):
    #     """
    #     Generate with Position-Aware Confidence Penalty.
    #     뒤쪽(정답 영역) 토큰의 confidence를 초기 단계에서 위치 기반으로 감쇠하여
    #     추론(앞부분) 토큰이 먼저 고정되도록 유도합니다.
    #     기존 함수는 유지하고, 새 함수로만 변경사항을 적용합니다.
    #     """
    #     # attention bias 준비
    #     if attention_mask is not None and 0.0 in attention_mask:
    #         attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
    #     else:
    #         attention_bias = None


    #     batch_size = idx.shape[0]
    #     x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
    #     x[:, :idx.shape[1]] = idx.clone()
    #     prompt_index = (x != mask_id)

    #     # vfg용 시각 토큰 인덱스 (SOI/EOI)
    #     soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
    #     eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
    #     start = soi_pos.min().item() if len(soi_pos) > 0 else 0
    #     end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
    #     visual_index = (start, end)

    #     assert max_new_tokens % block_length == 0
    #     num_blocks = max_new_tokens // block_length

    #     assert steps % num_blocks == 0
    #     steps = steps // num_blocks

    #     for num_block in range(num_blocks):
    #         block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
    #         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

    #         for i in range(steps):
    #             mask_index = (x == mask_id)

    #             # logits 계산
    #             if cfg_scale > 0.0:
    #                 un_x = x.clone()
    #                 un_x[prompt_index] = mask_id
    #                 x_ = torch.cat([x, un_x], dim=0)
    #                 logits = self(x_).logits
    #                 logits, un_logits = torch.chunk(logits, 2, dim=0)
    #                 logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    #             else:
    #                 # vfg_scale 적용
    #                 if vfg_scale > 0.0:
    #                     un_x_v = x.clone()
    #                     un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
    #                     x_v = torch.cat([x, un_x_v], dim=0)
    #                     logits_v = self(x_v).logits
    #                     logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
    #                     logits = logits_u + (vfg_scale + 1) * (logits_c - logits_u)
    #                 else:
    #                     logits = self(x, attention_bias=attention_bias).logits

                
    #             # 샘플링 후보 및 confidence 계산
    #             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    #             x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

    #             if remasking == 'low_confidence':
    #                 p = F.softmax(logits.to(torch.float64), dim=-1)
    #                 x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # [B, L]
    #             elif remasking == 'random':
    #                 x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    #             else:
    #                 raise NotImplementedError(remasking)

    #             # ---- Position-Aware Confidence Penalty ----
    #             if pos_penalty_gamma is not None and pos_penalty_gamma > 0.0:
    #                 block_start = idx.shape[1] + num_block * block_length
    #                 block_end   = idx.shape[1] + (num_block + 1) * block_length  # [start, end)

    #                 seq_len = x0_p.shape[1]
    #                 pos1d = torch.arange(seq_len, device=x0_p.device)

    #                 denom = max(block_length - 1, 1)
    #                 rel1d = (pos1d - block_start).clamp(min=0, max=denom) / denom  # [0,1] within block

    #                 # 진행률 (0,1], 초기일수록 큰 패널티
    #                 t = float(i + 1) / float(steps)
    #                 # penalty = 1 - gamma * (1 - t) * (rel^alpha)
    #                 penalty1d = 1.0 - float(pos_penalty_gamma) * (1.0 - t) * (rel1d ** float(pos_penalty_alpha))
    #                 penalty1d = penalty1d.clamp(min=0.0)

    #                 # 블록 외부는 패널티 1.0 유지
    #                 outside = (pos1d < block_start) | (pos1d >= block_end)
    #                 if outside.any():
    #                     penalty1d = torch.where(outside, torch.ones_like(penalty1d), penalty1d)

    #                 x0_p = x0_p * penalty1d.to(dtype=x0_p.dtype).unsqueeze(0)
    #             # -------------------------------------------

    #             # 현재 블록 범위를 넘어서는 위치는 선택에서 제외
    #             x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

    #             # 후보 토큰/신뢰도 갱신
    #             x0 = torch.where(mask_index, x0, x)
    #             confidence = torch.where(mask_index, x0_p, -np.inf)

    #             # 상위 confidence 위치를 transfer
    #             transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    #             for j in range(confidence.shape[0]):
    #                 _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
    #                 transfer_index[j, select_index] = True

    #             x[transfer_index] = x0[transfer_index]

    #         if eot_token is not None:
    #             last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
    #             if last_token_index_in_current_block < x.shape[1]:
    #                 tokens_at_block_end = x[:, last_token_index_in_current_block]
    #                 if torch.all(tokens_at_block_end == eot_token):
    #                     break

    #     return x

    @torch.no_grad()
    def mmu_generate_posaware(
        self,
        idx=None,
        input_embeddings=None,
        max_new_tokens=128,
        steps=128,
        block_length=128,
        temperature=0.0,
        top_k=None,
        eot_token=None,
        cfg_scale=0.0,
        remasking='low_confidence',
        mask_id=126336,
        attention_mask=None,
        pos_penalty_gamma: float = 0.5,
        pos_penalty_alpha: float = 1.0,
        vfg_scale: float = 0.0,
        post_token=None,
        post_idx=None
    ):
        """
        Generate with Position-Aware Confidence Penalty.
        뒤쪽(정답 영역) 토큰의 confidence를 초기 단계에서 위치 기반으로 감쇠하여
        추론(앞부분) 토큰이 먼저 고정되도록 유도합니다.
        기존 함수는 유지하고, 새 함수로만 변경사항을 적용합니다.
        """
        # attention bias 준비
        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        else:
            attention_bias = None


        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)

        if post_token is not None:
            x[:, post_idx-post_token.shape[1]:post_idx] = post_token.clone()

        # vfg용 시각 토큰 인덱스 (SOI/EOI)
        soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
        eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
        start = soi_pos.min().item() if len(soi_pos) > 0 else 0
        end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
        visual_index = (start, end)

        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        values = torch.linspace(0, 1, steps=steps)

        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

            for i in range(steps):
                mask_index = (x == mask_id)

                # logits 계산
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    # vfg_scale 적용
                    if vfg_scale > 0.0:
                        un_x_v = x.clone()
                        un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
                        x_v = torch.cat([x, un_x_v], dim=0)
                        logits_v = self(x_v).logits
                        logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
                        logits = logits_u + (values[i] + 1) * (logits_c - logits_u)
                    else:
                        logits = self(x, attention_bias=attention_bias).logits

                
                # 샘플링 후보 및 confidence 계산
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # [B, L]
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # ---- Position-Aware Confidence Penalty ----
                if pos_penalty_gamma is not None and pos_penalty_gamma > 0.0:
                    block_start = idx.shape[1] + num_block * block_length
                    block_end   = idx.shape[1] + (num_block + 1) * block_length  # [start, end)

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

                # 현재 블록 범위를 넘어서는 위치는 선택에서 제외
                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                # 후보 토큰/신뢰도 갱신
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                # 상위 confidence 위치를 transfer
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True

                x[transfer_index] = x0[transfer_index]

            if eot_token is not None:
                last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
                if last_token_index_in_current_block < x.shape[1]:
                    tokens_at_block_end = x[:, last_token_index_in_current_block]
                    if torch.all(tokens_at_block_end == eot_token):
                        break

        return x
    

    # # ----------------------------
    # # NEW: Position-Aware Confidence Penalty Generate
    # # ----------------------------
    # @torch.no_grad()
    # def mmu_generate_posaware(
    #     self,
    #     idx=None,
    #     input_embeddings=None,
    #     max_new_tokens=128,
    #     steps=128,
    #     block_length=128,
    #     temperature=0.0,
    #     top_k=None,
    #     eot_token=None,
    #     cfg_scale=0.0,
    #     remasking='low_confidence',
    #     mask_id=126336,
    #     attention_mask=None,
    #     pos_penalty_gamma: float = 0.5,
    #     pos_penalty_alpha: float = 1.0,
    #     vfg_scale: float = 0.0,
    # ):
    #     """
    #     Generate with Position-Aware Confidence Penalty.
    #     뒤쪽(정답 영역) 토큰의 confidence를 초기 단계에서 위치 기반으로 감쇠하여
    #     추론(앞부분) 토큰이 먼저 고정되도록 유도합니다.
    #     기존 함수는 유지하고, 새 함수로만 변경사항을 적용합니다.
    #     """
    #     # attention bias 준비
    #     if attention_mask is not None and 0.0 in attention_mask:
    #         attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
    #     else:
    #         attention_bias = None

    #     batch_size = idx.shape[0]
    #     x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
    #     x[:, :idx.shape[1]] = idx.clone()
    #     prompt_index = (x != mask_id)

    #     # vfg용 시각 토큰 인덱스 (SOI/EOI)
    #     soi_pos = (idx.cpu() == 126084).nonzero(as_tuple=True)[1]
    #     eoi_pos = (idx.cpu() == 126085).nonzero(as_tuple=True)[1]
    #     start = soi_pos.min().item() if len(soi_pos) > 0 else 0
    #     end = eoi_pos[eoi_pos >= start].min().item() if len(eoi_pos) > 0 else 0
    #     visual_index = (start, end)

    #     assert max_new_tokens % block_length == 0
    #     num_blocks = max_new_tokens // block_length

    #     assert steps % num_blocks == 0
    #     steps = steps // num_blocks

    #     repeat_token_id = 126081
    #     repeat_count = 0

    #     for num_block in range(num_blocks):
    #         block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
    #         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

    #         dynamic_patience = max(2, int(steps * 0.25))

    #         for i in range(steps):
    #             mask_index = (x == mask_id)

    #             # logits 계산
    #             if cfg_scale > 0.0:
    #                 un_x = x.clone()
    #                 un_x[prompt_index] = mask_id
    #                 x_ = torch.cat([x, un_x], dim=0)
    #                 logits = self(x_).logits
    #                 logits, un_logits = torch.chunk(logits, 2, dim=0)
    #                 logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    #             else:
    #                 # vfg_scale 적용
    #                 if vfg_scale > 0.0:
    #                     un_x_v = x.clone()
    #                     un_x_v[:, visual_index[0]:visual_index[1]+1] = mask_id
    #                     x_v = torch.cat([x, un_x_v], dim=0)
    #                     logits_v = self(x_v).logits
    #                     logits_c, logits_u = torch.chunk(logits_v, 2, dim=0)
    #                     logits = logits_u + (vfg_scale + 1) * (logits_c - logits_u)
    #                 else:
    #                     logits = self(x, attention_bias=attention_bias).logits

                
    #             # 샘플링 후보 및 confidence 계산
    #             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    #             x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

    #             if remasking == 'low_confidence':
    #                 p = F.softmax(logits.to(torch.float64), dim=-1)
    #                 x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # [B, L]
    #             elif remasking == 'random':
    #                 x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    #             else:
    #                 raise NotImplementedError(remasking)

    #             # ---- Position-Aware Confidence Penalty ----
    #             if pos_penalty_gamma is not None and pos_penalty_gamma > 0.0:
    #                 block_start = idx.shape[1] + num_block * block_length
    #                 block_end   = idx.shape[1] + (num_block + 1) * block_length  # [start, end)

    #                 seq_len = x0_p.shape[1]
    #                 pos1d = torch.arange(seq_len, device=x0_p.device)

    #                 denom = max(block_length - 1, 1)
    #                 rel1d = (pos1d - block_start).clamp(min=0, max=denom) / denom  # [0,1] within block

    #                 # 진행률 (0,1], 초기일수록 큰 패널티
    #                 t = float(i + 1) / float(steps)
    #                 # penalty = 1 - gamma * (1 - t) * (rel^alpha)
    #                 penalty1d = 1.0 - float(pos_penalty_gamma) * (1.0 - t) * (rel1d ** float(pos_penalty_alpha))
    #                 penalty1d = penalty1d.clamp(min=0.0)

    #                 # 블록 외부는 패널티 1.0 유지
    #                 outside = (pos1d < block_start) | (pos1d >= block_end)
    #                 if outside.any():
    #                     penalty1d = torch.where(outside, torch.ones_like(penalty1d), penalty1d)

    #                 x0_p = x0_p * penalty1d.to(dtype=x0_p.dtype).unsqueeze(0)
    #             # -------------------------------------------

    #             # 현재 블록 범위를 넘어서는 위치는 선택에서 제외
    #             x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

    #             # 후보 토큰/신뢰도 갱신
    #             x0 = torch.where(mask_index, x0, x)
    #             confidence = torch.where(mask_index, x0_p, -np.inf)

    #             # 상위 confidence 위치를 transfer
    #             transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    #             for j in range(confidence.shape[0]):
    #                 _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
    #                 transfer_index[j, select_index] = True

    #             x[transfer_index] = x0[transfer_index]

    #             generated_tokens = x[transfer_index]
    #             if torch.all(generated_tokens == repeat_token_id) and generated_tokens.numel() > 0:
    #                 repeat_count += 1
    #             else:
    #                 repeat_count = 0

    #             if repeat_count >= dynamic_patience:
    #                 print(f"Num steps: {i}.")
    #                 return x

    #         if eot_token is not None:
    #             last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
    #             if last_token_index_in_current_block < x.shape[1]:
    #                 tokens_at_block_end = x[:, last_token_index_in_current_block]
    #                 if torch.all(tokens_at_block_end == eot_token):
    #                     break

    #     return x


AutoConfig.register("mmada", MMadaConfig)
AutoModelForCausalLM.register(MMadaConfig, MMadaModelLM)
AutoModel.register(MMadaConfig, MMadaModelLM)
