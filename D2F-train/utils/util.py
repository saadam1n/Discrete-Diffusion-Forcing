import torch
from torch.distributions import Uniform

def forward_process_block_fixed_p(x, mask_id, p_mask):
    B, L = x.shape
    if isinstance(p_mask, float):
        p_mask = torch.full((B, 1), p_mask, device=x.device)
    elif p_mask.ndim == 1:
        p_mask = p_mask[:, None]
    rand = torch.rand((B, L), device=x.device)
    mask = rand < p_mask
    x_masked = torch.where(mask, mask_id, x)
    return x_masked, mask

import torch

def generate_monotonic_pmasks(batch_size, max_blocks, device):
    """
    生成 shape (B, max_blocks) 的单调非降随机序列，每行第一个元素在[0,1]随机，后续不小于前一个
    """
    # 第一个block p_mask随机
    p0 = torch.rand(batch_size, 1, device=device)/2+0.2
    # print(p0)
    # 后续blocks生成增量 [0, 1]，加起来保证不超过1（之后用 clamp）
    increments = torch.rand(batch_size, max_blocks - 1, device=device) * (0.7 - p0)/ (max_blocks - 1)
    # print(increments)
    # 逐元素累加，保证非降
    cum_increments = torch.cumsum(increments, dim=1)
    # print(cum_increments)
    # 总 p_mask = p0 + 累积增量，保证不超过1
    p_masks = torch.cat([p0, p0 + cum_increments], dim=1)
    p_masks = torch.clamp(p_masks, max=1.0)
    # print(p_masks)
    return p_masks  # (B, max_blocks)


def forward_process_length(input_ids, mask_id, block_size, prompt_lengths,eos_id=None):
    """
    Args:
        input_ids: (B, L)
        prompt_lengths: (B,)
    Returns:
        noisy_batch, masked_indices, p_mask_tensor
    """
    B, L = input_ids.shape
    device = input_ids.device
    noisy_batch = input_ids.clone()
    eos_indices= (input_ids==eos_id)
    masked_indices = torch.zeros_like(input_ids,dtype=torch.bool)
    p_mask_tensor = torch.zeros((B, L), device=device)

    # 计算每个样本block数
    non_prompt_lens = L - prompt_lengths
    full_blocks = non_prompt_lens // block_size
    remainders = non_prompt_lens % block_size
    total_blocks = full_blocks + (remainders > 0).long()

    max_blocks = total_blocks.max().item()

    # 生成每个样本block的mask比率，单调非降且第一个随机
    p_masks = generate_monotonic_pmasks(B, max_blocks, device)  # shape (B, max_blocks)

    for i in range(B):
        prompt_len = prompt_lengths[i].item()
        num_blocks = total_blocks[i].item()
        start_block = torch.tensor([0])  # 随机选择一个block开始
        for block_idx in range(num_blocks):
            if block_idx < start_block:
                continue
            start = prompt_len + block_idx * block_size
            end = min(start + block_size, L)

            p_block = p_masks[i, block_idx-start_block].item()

            block = noisy_batch[i, start:end].unsqueeze(0)
            masked_block, mask = forward_process_block_fixed_p(block, mask_id, p_block)

            noisy_batch[i, start:end] = masked_block.squeeze(0)
            masked_indices[i, start:end] = mask.squeeze(0)
            # if torch.all(input_ids[i, start:end] == eos_id):
            #     masked_indices[i,start:end]== False
                # print("1")

            p_mask_tensor[i, start:end] = p_block

    return noisy_batch, masked_indices, p_mask_tensor

# def forward_process_length(input_ids, mask_id, block_size, prompt_lengths, p_min=0.2, p_max=0.9):
#     """
#     返回每个 token 的实际 mask 概率 tensor（非prompt区域），其余为0。
#     """
#     B, L = input_ids.shape
#     device = input_ids.device
#     noisy_batch = input_ids.clone()
#     masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
#     p_mask_tensor = torch.zeros((B, L), device=device)  # 最终返回值

#     for i in range(B):
#         prompt_len = prompt_lengths[i].item()
#         non_prompt_len = L - prompt_len
#         full_blocks = non_prompt_len // block_size
#         remainder = non_prompt_len % block_size
#         total_blocks = full_blocks + (1 if remainder > 0 else 0)

#         for block_idx in range(total_blocks):
#             start = prompt_len + block_idx * block_size
#             end = min(start + block_size, L)

#             # block的 mask 概率（线性递增）
#             if total_blocks > 1:
#                 p_block = p_min + (p_max - p_min) * (block_idx / (total_blocks - 1))
#             else:
#                 p_block = p_max

#             block = noisy_batch[i, start:end].unsqueeze(0)
#             masked_block, mask = forward_process_block_fixed_p(block, mask_id, p_block)
#             noisy_batch[i, start:end] = masked_block.squeeze(0)
#             masked_indices[i, start:end] = mask.squeeze(0)

#             # 记录 p_mask 到 tensor 中
#             p_mask_tensor[i, start:end] = p_block

#     return noisy_batch, masked_indices, p_mask_tensor
def forward_process(input_ids,mask_id ,t_max=1.0, eps=1e-4):
    B, L = input_ids.shape
    # t = torch.rand(B, device=input_ids.device)
    dist = Uniform(0., t_max)
    t = dist.sample((B,)).to(input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, L)
    masked_indices = torch.rand((B, L), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)

    return noisy_batch, masked_indices, p_mask
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def shift_logits(logits):
    shifted_logits = torch.zeros_like(logits)
    shifted_logits[:, 1:, :] = logits[:, :-1, :]
    shifted_logits[:, 0, :] = 1.0

    return shifted_logits

import math
import random
def cave_in_generate(input_ids, mask_id, block_size, prompt_lengths, eos_id):
    """
    Return the following:
    - new input ids
    - noisy batch
    - attention mask
    - which positions are going to be unmasked next
    - masked indices (so we can compute the loss)
    - pmask
    """

    # this is going to be a pain to vectorize
    # so let's not vectorize
    B, L = input_ids.shape

    device = input_ids.device

    non_prompt_lens = L - prompt_lengths
    full_blocks = non_prompt_lens // block_size
    remainders = non_prompt_lens % block_size
    total_blocks = full_blocks + (remainders > 0).long()

    global_section_num_blocks = int(math.log2(block_size) + 1 + 0.5)
    section_token_cost = global_section_num_blocks * block_size

    # what ratio of the original part of the context length dedicated to the response is repuprosed towards sections
    SECTION_BUDGET=0.5

    block_tok_pos = torch.arange(block_size, dtype=torch.int32, device=device)

    batch_input_ids = []
    batch_noisy_batch = []
    batch_mask_indices = []
    batch_need_unmask = []
    batch_attn_mask = []

    for b in range(B):
        prompt_len = prompt_lengths[b].item()
        num_blocks = total_blocks[b].item()

        non_prompt_len = non_prompt_lens[b].item()

        # how many sections can we use?
        num_sections = int((SECTION_BUDGET * non_prompt_len) // section_token_cost)
        if num_sections < 1:
            num_sections = 1



        # first, let's find the EOS block so we know where to stop generating tokens
        last_normal_block = num_blocks - 1
        for block_idx in range(num_blocks):

            start = prompt_len + block_idx * block_size
            end = min(start + block_size, L)

            block_ids = input_ids[b, start:end]

            if (block_ids == eos_id).any():
                last_normal_block = block_idx

        num_normal_blocks = last_normal_block + 1

        section_num_blocks = global_section_num_blocks
        if section_num_blocks > num_normal_blocks:
            section_num_blocks = num_normal_blocks

        total_num_sections = (num_normal_blocks - global_section_num_blocks + 1)
        if total_num_sections < 1:
            total_num_sections = 1

        if total_num_sections > num_sections:
            total_num_sections = num_sections

        section_start_list =  [i for i in range(total_num_sections)]
        random.shuffle(section_start_list)

        section_start_list = section_start_list[:num_sections]

        sample_raw_input_ids = input_ids[b]

        sample_input_ids = [sample_raw_input_ids]
        sample_noisy_batch = [sample_raw_input_ids]
        sample_mask_indices = [torch.zeros_like(sample_raw_input_ids)]
        sample_need_unmask = [torch.zeros_like(sample_raw_input_ids)]

        global_start_pos = L
        section_global_start_positions = []

        for start_block in section_start_list:
            start = prompt_len + start_block * block_size
            end = min(start + block_size * section_num_blocks, L)  

            translated_end = end - start

            section_input_ids = input_ids[b, start:end]

            section_noisy_batch = section_input_ids.clone()

            for block_idx in range(section_num_blocks):
                sublock_start = block_idx * block_size
                sublock_end = min(sublock_start + block_size, translated_end)

                sliced_block_tok_pos = block_tok_pos[:sublock_end - sublock_start]

                cave_in_divisor = 1 << (block_idx + 1)
                next_cave_in_divisor = 1 << block_idx

                # save this somewhere (used for loss calculations)
                remask = (sliced_block_tok_pos % cave_in_divisor != 0)

                next_remask = (sliced_block_tok_pos % next_cave_in_divisor != 0)

                # save this somewhere (used for unmasking algorithm)
                need_unmask = torch.logical_xor(remask, next_remask)



                sublock_noisy_batch = section_noisy_batch[sublock_start:sublock_end]

                sublock_noisy_batch[remask] = mask_id

                sample_mask_indices.append(remask)
                sample_need_unmask.append(need_unmask)

            # append this input ids and noisy batch to some list
            sample_input_ids.append(section_input_ids)
            sample_noisy_batch.append(section_noisy_batch)

            section_global_start_positions.append(global_start_pos)
            global_start_pos += section_input_ids.shape[-1]

        # now create a list of everything
        sample_input_ids = torch.cat(sample_input_ids, dim=0)
        sample_noisy_batch = torch.cat(sample_noisy_batch, dim=0)
        sample_mask_indices = torch.cat(sample_mask_indices, dim=0)
        sample_need_unmask = torch.cat(sample_need_unmask, dim=0)

        L_P = sample_input_ids.shape[-1]


        attn_mask = torch.full((L_P, L_P), float('-inf'), dtype=torch.float32, device=device)

        # first, the prompt block
        attn_mask[:prompt_len, :prompt_len] = 0

        # now, the fully unmasked response
        for block_idx in range(num_blocks):
            start = prompt_len + block_idx * block_size
            end = min(start + block_size, L)

            attn_mask[start:end, :end] = 0

        # for each section
        for section_idx in range(num_sections):
            # where does this section begin
            mask_start_pos = section_global_start_positions[section_idx]

            first_block_idx = section_start_list[section_idx]
            first_block_start = prompt_len + first_block_idx * block_idx

            for sublock_idx in range(section_num_blocks):
                # where does this block begin?

                start = first_block_start + sublock_idx * block_size
                end = min(start + block_size, L)

                # reset to local coords
                start -= first_block_start
                end -= first_block_start

                # translate to global coords
                start += mask_start_pos
                end += mask_start_pos

                print(f"POS {start} {end}")

                # two steps here:
                # 1) set unmasked prompt and response to 0
                # 2) set section to zer0

                attn_mask[start:end, :first_block_start] = 0
                attn_mask[start:end, mask_start_pos:end] = 0

        # unsqueeze for heads
        attn_mask = attn_mask.unsqueeze(0)

        batch_input_ids.append(sample_input_ids)
        batch_noisy_batch.append(sample_noisy_batch)
        batch_mask_indices.append(sample_mask_indices)
        batch_need_unmask.append(sample_need_unmask)
        batch_attn_mask.append(attn_mask)

    batch_input_ids = torch.stack(batch_input_ids)
    batch_noisy_batch = torch.stack(batch_noisy_batch)
    batch_mask_indices = torch.stack(batch_mask_indices)
    batch_need_unmask = torch.stack(batch_need_unmask)
    batch_attn_mask = torch.stack(batch_attn_mask)

    return batch_input_ids, batch_noisy_batch, batch_mask_indices, batch_need_unmask, batch_attn_mask

if __name__ == '__main__':
    """
    input_ids= torch.tensor([[1,5,4,3,25,6,7,9,5,8,7,6],[1,3,8,9,7,34,6,9,5,8,7,6]])
    mask_id=0
    block_size=3
    prompt_length=torch.tensor([2,1])
    noisy_batch, masked_indices,p_mask = forward_process_length(input_ids, mask_id, block_size, prompt_length)
    print("noisy_batch:", noisy_batch)
    print("masked_indices:", masked_indices)
    print("p_mask:", p_mask)
    """

    input_ids= torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1]])
    mask_id=0
    block_size=4
    prompt_length=torch.tensor([4])

    print(f"Length of L: {input_ids.shape}")
    batch_input_ids, batch_noisy_batch, batch_mask_indices, batch_need_unmask, batch_attn_mask = cave_in_generate(
        input_ids, mask_id, block_size, prompt_length, -1
    )

    print(batch_input_ids)
    print(batch_noisy_batch)
    print(batch_mask_indices)
    print(batch_need_unmask)
    print(batch_attn_mask)
