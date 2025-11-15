import torch
from utils.util import forward_process_length, shift_logits,forward_process, cave_in_generate
import torch.nn.functional as F

def compute_loss_by_config(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
        config,
        tokenizer
):
    """Select different loss functions based on config file"""
    training_mode = config.get('training_mode', 'dream')
    
    if training_mode == 'llada':
        return compute_llada_loss(
            input_ids, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id, tokenizer
        )
    elif training_mode == 'dream':
        return compute_loss(
            input_ids, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id
        )
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

def remask_duplicates_(x0, mask_token_id):
    # NOT VECTORIZED, BAD CODE!!!!

    # make sure batch size is 1
    assert x0.shape[0] == 1

    x0 = x0[0]

    i = 0
    n = x0.shape[0]

    while i < n:
        search_token = x0[i].item()

        if search_token == mask_token_id:
            i += 1
            continue

        # use a while loop to fix python funkiness
        j = i + 1
        while j < n and x0[j] == search_token:
            j += 1

        # remask all duplicated tokens
        if j - i > 1:
            x0[i:j] = mask_token_id

        # skip to the next non-duplicated token
        i = j


def remask_duplicates(x0, mask_token_id=0):
    # is each element equal to the next?

    same_contig = (x0[:, 1:] == x0[:, :-1])

    same_n = torch.cat(
        [
            same_contig,
            torch.zeros_like(x0[:, -1:])   
        ], dim=1
    )
    # is each element equal to the prev?
    same_p = torch.cat(
        [
            torch.zeros_like(x0[:, -1:]),   
            same_contig
        ], dim=1
    )

    same = torch.logical_or(same_n, same_p)

    x0 = torch.where(same, mask_token_id, x0)
    return x0

def ntis_sample_tokens(logits, noisy_batch, mask_token_id=-1):
    with torch.no_grad():
        x0 = torch.argmax(logits, dim=-1)

        # assume we are not at the last ift step yet

        probs = torch.softmax(logits, dim=-1)
        x0_p = torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

        x0 = remask_duplicates(x0, mask_token_id)

        # keep tokens that are either 1) very high confidence or 2) were masked previously
        conf_cond = x0_p > 0.9
        prev_cond = (noisy_batch == mask_token_id)
        remask_cond = torch.logical_or(conf_cond, prev_cond)

        x0 = torch.where(remask_cond, x0, mask_token_id)

        # unlike regular sampling, we do not remask EOT tokens

        return x0

def ntis_rl_sample(logits, noisy_batch=None, input_ids=None, mask_token_id=None):
    """
    returns rl_noisy_batch for a second round of training
    """

    # set to zero for no temperature scaling
    temprature = 0.0
    if temprature > 0.0:
        logits = logits / temprature

    # B, L, V (where V is output vocab size)
    #prob = F.softmax(logits, dim=-1)

    # B, L
    top1 = torch.argmax(logits, dim=-1)

    # B, L
    #rl_next_batch = noisy_batch

    # keep tokens in the original noisy batch the same


    # keep it simple for now, just use top1
    return top1

def ntis_rl_sample_gt(logits, noisy_batch=None, input_ids=None, prompt_mask=None, mask_token_id=None, threshold=0.35):
    """
    returns rl_noisy_batch for a second round of training
    """

    # set to zero for no temperature scaling
    temprature = 0.0
    if temprature > 0.0:
        logits = logits / temprature

    # B, L, V (where V is output vocab size)
    probs = F.softmax(logits, dim=-1)

    # B, L
    gt_p = torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(input_ids, -1)), -1)

    # B, L
    top1 = torch.argmax(logits, dim=-1)

    # if gt above a certain confidence, select that
    # else, select top1
    # B, L
    next_noisy_batch = torch.where(gt_p > threshold, input_ids, top1)

    # protect prompt
    # B, L
    next_noisy_batch[prompt_mask] = input_ids[prompt_mask]

    if False:
        # then remask duplicates
        # B, L
        next_noisy_batch = remask_duplicates(next_noisy_batch, mask_token_id)

        # protect prompt again in case any numbers e.g. 00 were masked because of this
        # B, L
        next_noisy_batch[prompt_mask] = input_ids[prompt_mask]

    return next_noisy_batch

def ntis_rl_sample_qxi(logits, noisy_batch=None, input_ids=None, prompt_mask=None, mask_token_id=None, beta=1.2):
    """
    returns rl_noisy_batch for a second round of training
    """

    # set to zero for no temperature scaling
    temprature = 0.0
    if temprature > 0.0:
        logits = logits / temprature

    # B, L, V (where V is output vocab size)
    probs = F.softmax(logits, dim=-1)

    # B, L
    gt_p = torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(input_ids, -1)), -1) * beta

    keep_gt = torch.rand_like(gt_p) < gt_p
    
    removed_gt_logits = logits.scatter(-1, input_ids.unsqueeze(-1), float('-inf'))
    dist = torch.distributions.Categorical(logits=removed_gt_logits)

    # if gt above a certain confidence, select that
    # else, select top1
    # B, L
    next_noisy_batch = torch.where(keep_gt, input_ids, dist.sample())

    # protect prompt
    # B, L
    next_noisy_batch[prompt_mask] = input_ids[prompt_mask]

    return next_noisy_batch

def compute_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    attention_mask=build_custom_float_attention_mask(noisy_batch, question_length, block_size, device=noisy_batch.device)
    attention_mask=attention_mask.to(torch.float16)
    logits=denoiser(noisy_batch,attention_mask=attention_mask).logits
    logits=shift_logits(logits)

    # self align forces the block diffusion LLM to mirror the output of the bidirectional LLM
    # that is not what we are doing

    # ntis sample tokens using logits

    # recalculate new outputs
    # concatenate the batch

    # force calculation on all logits
    token_loss_2= F.cross_entropy(logits, input_ids, reduction='none')


        
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 
def compute_normal_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    logits=denoiser(noisy_batch).logits
    logits=shift_logits(logits)

    token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 
import torch
import time
import random
def compute_llada_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
        tokenizer
):
    mask_id=126336

    # 32k context length is way too much
    # I wonder how people train models with 1 million context length
    #input_ids = torch.ones(1, 32768, dtype=input_ids.dtype, device=input_ids.device)

    B, L = input_ids.shape

    token_positions = torch.arange(L, device=input_ids.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))

    use_randomized_masking = (random.random() < 10.1)

    if use_randomized_masking:
        noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
        noisy_batch = noisy_batch.to(denoiser.device)
        #print("Building attention mask...")
        start_attn_mask = time.time()
        attention_mask=build_custom_float_attention_mask(noisy_batch, question_length, block_size, device=noisy_batch.device)
        attention_mask=attention_mask.to(torch.float16)
        #print(f"Done, attn mask took {time.time() - start_attn_mask}s to build")
        noisy_batch[prompt_mask] = input_ids[prompt_mask]
    else:
        noisy_batch, input_ids, p_mask, masked_indices, need_unmask, attention_mask = cave_in_generate(
            input_ids=input_ids, mask_id=mask_id, block_size=block_size, prompt_lengths=prompt_length, eos_id=eos_id
        )


    custom_rope_pos = torch.arange(L, device=denoiser.device, dtype=torch.float)
    #print("NEW INFERENCE STEP -- START")

    logits=denoiser(noisy_batch,attention_bias=attention_mask, custom_rope_pos=custom_rope_pos).logits
    # logits=shift_logits(logits)

    # original D2F code
    if False:
        # logits=shift_logits(logits)
        if self_align and False:
            print("using self align")
            with torch.no_grad():
                with denoiser.disable_adapter():
                    # ref_model = denoiser
                # ref_model.eval()
                # print(type(ref_model))
                    ref_logits=denoiser(noisy_batch,attention_bias=torch.zeros([1,1,noisy_batch.shape[1],noisy_batch.shape[1]],dtype=torch.float16,device=denoiser.device)).logits
                    # ref_logits=shift_logits(ref_logits)
                    ref_logits = torch.nn.functional.softmax(ref_logits, dim=-1)
            token_loss_2 = F.cross_entropy(logits[masked_indices], ref_logits[masked_indices], reduction='none') / p_mask[masked_indices]
            # print("token_loss_2",token_loss_2.shape)
        else:
            token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
        losses = {
                    # 'loss_1': token_loss_2.mean() * 0,
                    'loss': token_loss_2.mean(),
                    'loss_masked': None,
                    'loss_unmasked': None
                }

        return losses 


    prompt_mask = prompt_mask.to(denoiser.device)

    all_logits = [logits]

    num_rl_iterations = 1
    for _ in range(num_rl_iterations):
        old_noisy_batch = noisy_batch

        # detach shouldn't matter here but idrc
        if use_randomized_masking:
            noisy_batch = ntis_rl_sample_qxi(
                logits=logits, 
                noisy_batch=noisy_batch, 
                input_ids=input_ids,
                prompt_mask=prompt_mask,
                mask_token_id=mask_id
            ).clone().detach()
        else:
            noisy_batch = torch.where(need_unmask, input_ids, noisy_batch)

        #import pdb;
        #pdb.set_trace()
        #print("NEW INFERENCE STEP -- RL")
        logits = denoiser(noisy_batch, attention_bias=attention_mask, custom_rope_pos=custom_rope_pos).logits
        all_logits.append(logits)



    all_logits = torch.cat(all_logits, dim=0).flatten(0, 1)
    all_input_ids = input_ids.repeat(num_rl_iterations + 1, 1)

    assert B == 1
    training_mask = ~prompt_mask
    training_mask = training_mask.expand_as(all_input_ids)
    masked_mask = masked_indices.expand_as(all_input_ids)
    all_p_mask = p_mask.expand_as(all_input_ids)

    # for the first iteration, the ground truth text is correct
    # not much to learn there, could reinforce bad habits like just setting prob to 1 if unmasked
    

    all_input_ids = all_input_ids.flatten()
    training_mask = training_mask.flatten()
    masked_mask = masked_mask.flatten()
    all_p_mask = all_p_mask.flatten()


    # calculate losses seperately, so they don't get "deleted" during averaging
    unmasked_mask = torch.logical_and(training_mask, ~masked_mask)




    token_loss_masked = F.cross_entropy(all_logits[masked_mask], all_input_ids[masked_mask], reduction='none') / all_p_mask[masked_mask]
    token_loss_masked = token_loss_masked.mean()

    token_loss_unmasked = F.cross_entropy(all_logits[unmasked_mask], all_input_ids[unmasked_mask], reduction='none') / all_p_mask[unmasked_mask]
    token_loss_unmasked = token_loss_unmasked.mean()

    combined_loss = token_loss_masked# + token_loss_unmasked

    losses = {
                'loss': combined_loss,
                'loss_masked': token_loss_masked,
                'loss_unmasked': token_loss_unmasked
            }

    return losses 

import math
def build_custom_float_attention_mask(input_ids, prompt_length, block_size, device=None):
    B,seq_len= input_ids.shape
    # 初始化为全 -inf
    attn_mask = torch.full((B,1,seq_len, seq_len), float('-inf'), dtype=torch.float32, device=device)
    # 1. Prompt部分：每个token可以注意整个prompt
    for i in range(B):
        attn_mask[i,:,:,:prompt_length[i]] = 0.0  # 允许所有 token 看 prompt


        # 2. 块划分：从 prompt_length 开始划分 block
        num_blocks = (seq_len - prompt_length[i] + block_size - 1) // block_size

        section_num_blocks = int(math.log2(block_size) + 1 + 0.5)
        total_num_sections = (num_blocks - section_num_blocks + 1)
        total_num_blocks = total_num_sections * num_blocks
        total_num_tokens = block_size * total_num_blocks

        #print(f"Prompt length is {prompt_length[i]}, num blocks is {num_blocks}, total length is {num_blocks * block_size}. Total num sections is {total_num_sections}, total num blocks is {total_num_blocks}, total num tokens is {total_num_tokens}")

        for b in range(num_blocks):
            block_start = prompt_length[i] + b * block_size
            # print(block_start,block_size,seq_len)
            block_end = min(block_start + block_size, seq_len)

            # 块内全注意
            attn_mask[i,:,block_start:block_end, block_start:block_end] = 0.0

            # 块之间因果注意（只能看前面块）
            for prev_b in range(b):
                prev_start = prompt_length[i] + prev_b * block_size
                prev_end = min(prev_start + block_size, seq_len)

                # 当前块可以看前面块
                attn_mask[i,:,block_start:block_end, prev_start:prev_end] = 0.0

    return attn_mask  # [seq_len, seq_len], float, 0.0 for allowed, -inf for disallowed
if __name__ == "__main__":
    seq_len = 10
    input_ids = torch.randint(0, 100, (2, seq_len))  # 示例输入
    block_size = 4
    prompt_length = torch.tensor([2, 4])  # 示例prompt长度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn_mask = build_custom_float_attention_mask(input_ids, prompt_length, block_size, device)
    print(attn_mask)