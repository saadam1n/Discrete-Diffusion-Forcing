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
    batch_p_mask = []
    batch_mask_indices = []
    batch_need_unmask = []
    batch_attn_mask = []

    for b in range(B):
        prompt_len = prompt_lengths[b].item()
        num_blocks = total_blocks[b].item()

        non_prompt_len = non_prompt_lens[b].item()

        # how many sections can we use?
        num_sections = int((SECTION_BUDGET * non_prompt_len) // section_token_cost)

        print(f"Non prompt length is {non_prompt_len}. Integer component of budget is {int(SECTION_BUDGET * non_prompt_len)}. Since section cost is {section_token_cost}, we will use {num_sections}")

        if num_sections < 1:
            print(f"Actually, I lied. We need to round up so we have to use at least one section")
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

        section_start_list =  [i for i in range(total_num_sections)]
        print(f"We found {num_blocks} blocks. Global section num blocks is {global_section_num_blocks}, but we are using {section_num_blocks}. We can use {total_num_sections} in our sample. Budget however is {num_sections}")
        print(f"Full list is {section_start_list}")
        random.shuffle(section_start_list)

        section_start_list = section_start_list[:num_sections]

        # not needed but should make this easier to debug
        section_start_list.sort()
        print(f"Using truncated list {section_start_list}")

        sample_raw_input_ids = input_ids[b]

        sample_input_ids = [sample_raw_input_ids]
        sample_noisy_batch = [sample_raw_input_ids]
        sample_p_mask = [torch.zeros_like(sample_raw_input_ids)]
        sample_mask_indices = [torch.zeros_like(sample_raw_input_ids)]
        sample_need_unmask = [torch.zeros_like(sample_raw_input_ids)]

        global_start_pos = L
        section_global_start_positions = []

        for start_block in section_start_list:
            start = prompt_len + start_block * block_size
            end = min(start + block_size * section_num_blocks, L)  

            print(f"BLOCK IDX {start_block}\tBLOCK START AT {start}\tAKA RESPONSE POS {start - prompt_len}\tTOKEN VALUE {input_ids[b, start].item()}")
            print(f"\tEND AT {end}")

            translated_end = end - start

            section_input_ids = input_ids[b, start:end]

            section_noisy_batch = section_input_ids.clone()

            for block_idx in range(section_num_blocks):
                sublock_start = block_idx * block_size
                sublock_end = min(sublock_start + block_size, translated_end)

                print(f"\tSUBLOCK {sublock_start} -> {sublock_end}")

                sliced_block_tok_pos = block_tok_pos[:sublock_end - sublock_start]

                cave_in_divisor = 1 << (block_idx + 1)
                next_cave_in_divisor = 1 << block_idx


                # only executes for the last block
                # for the last block we want everything unmasked 
                if cave_in_divisor > block_size:
                    sliced_block_tok_pos = sliced_block_tok_pos + 1


                # save this somewhere (used for loss calculations)
                remask = (sliced_block_tok_pos % cave_in_divisor != 0)

                next_remask = (sliced_block_tok_pos % next_cave_in_divisor != 0)

                # save this somewhere (used for unmasking algorithm)
                need_unmask = torch.logical_xor(remask, next_remask)



                sublock_noisy_batch = section_noisy_batch[sublock_start:sublock_end]

                sublock_noisy_batch[remask] = mask_id

                percentage_masked = torch.atleast_1d(remask.sum() / remask.numel())
                percentage_masked = percentage_masked.expand_as(remask)
                #print(f"{block_size} {cave_in_divisor} {next_cave_in_divisor} REMASK SIZE {remask.shape} {remask} PERCENRAGE MASK SIZE {percentage_masked.shape}")

                sample_p_mask.append(percentage_masked)
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
        sample_p_mask = torch.cat(sample_p_mask, dim=0)
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
            first_block_start = prompt_len + first_block_idx * block_size

            print(f"CREATING MASK FOR SECTION AT {section_idx}\t{first_block_idx}\t{first_block_start}\t{mask_start_pos}")

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

                print(f"\tPOS {sublock_idx}\t{start}\t{end}")

                # two steps here:
                # 1) set unmasked prompt and response to 0
                # 2) set section to zer0

                attn_mask[start:end, :first_block_start] = 0
                attn_mask[start:end, mask_start_pos:end] = 0

        # unsqueeze for heads
        attn_mask = attn_mask.unsqueeze(0)

        print(batch_p_mask)
        batch_input_ids.append(sample_input_ids)
        batch_noisy_batch.append(sample_noisy_batch)
        batch_p_mask.append(sample_p_mask)
        batch_mask_indices.append(sample_mask_indices)
        batch_need_unmask.append(sample_need_unmask)
        batch_attn_mask.append(attn_mask)

    batch_input_ids = torch.stack(batch_input_ids)
    batch_noisy_batch = torch.stack(batch_noisy_batch)
    batch_p_mask = torch.stack(batch_p_mask)
    batch_mask_indices = torch.stack(batch_mask_indices).bool()
    batch_need_unmask = torch.stack(batch_need_unmask).bool()
    batch_attn_mask = torch.stack(batch_attn_mask)

    return batch_input_ids, batch_noisy_batch, batch_p_mask, batch_mask_indices, batch_need_unmask, batch_attn_mask

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

    """
    mask_id = 0
    eos_id = -1

    normal_ctx_len = 128
    prompt_length = 14
    eos_len = 26

    input_ids = [i + 1 for i in range(prompt_length)] + [i + 1 for i in range(normal_ctx_len - prompt_length - eos_len)] + [-1 for _ in range(eos_len)]

    input_ids= torch.tensor([input_ids]) #[1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1]
    block_size=4
    prompt_length=torch.tensor([prompt_length])

    print(input_ids)

    print(f"Length of L: {input_ids.shape}")
    batch_input_ids, batch_noisy_batch, batch_mask_indices, batch_need_unmask, batch_attn_mask = cave_in_generate(
        input_ids, mask_id, block_size, prompt_length, eos_id
    )

    print(batch_input_ids)
    print(batch_noisy_batch)
    print(batch_mask_indices)
    print(batch_need_unmask)
    print(batch_attn_mask)

    # attention mask viz
    # code shamelessly copied from claude
    import matplotlib.pyplot as plt
    import numpy as np

    # Visualization
    # Convert attention mask to numpy for visualization
    attn_mask_np = batch_attn_mask.squeeze().cpu().numpy()

    # Create a binary mask for visualization (0 for valid positions, 1 for masked)
    # Since the mask has 0 and -inf, we'll map: 0 -> white (valid), -inf -> black (masked)
    vis_mask = np.where(np.isneginf(attn_mask_np), 1, 0)

    # Create figure with larger size for better visibility
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Raw values (showing -inf as a different color)
    im1 = ax1.imshow(attn_mask_np, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
    ax1.set_title('Attention Mask (Raw Values)\n0 = Valid, -inf = Masked', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Key Position', fontsize=12)
    ax1.set_ylabel('Query Position', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Mask Value')

    # Plot 2: Binary visualization (clearer view)
    im2 = ax2.imshow(vis_mask, cmap='binary', aspect='auto', interpolation='nearest')
    ax2.set_title('Attention Mask (Binary View)\nWhite = Valid (0), Black = Masked (-inf)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Key Position', fontsize=12)
    ax2.set_ylabel('Query Position', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='0=Valid, 1=Masked', ticks=[0, 1])

    # Add grid for better readability
    for ax in [ax1, ax2]:
        ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
        ax.set_xticks(range(0, attn_mask_np.shape[1], max(1, attn_mask_np.shape[1]//10)))
        ax.set_yticks(range(0, attn_mask_np.shape[0], max(1, attn_mask_np.shape[0]//10)))

    plt.tight_layout()
    plt.savefig('attention_mask.png', dpi=300, bbox_inches='tight')
    print("\nAttention mask visualization saved as 'attention_mask.png'")
    plt.show()

    # Optional: Create a zoomed-in view of a portion if the mask is large
    if attn_mask_np.shape[0] > 50:
        fig, ax = plt.subplots(figsize=(10, 10))
        zoom_size = min(50, attn_mask_np.shape[0])
        im = ax.imshow(vis_mask[:zoom_size, :zoom_size], cmap='binary', aspect='auto', interpolation='nearest')
        ax.set_title(f'Attention Mask (First {zoom_size}x{zoom_size} positions)\nWhite = Valid, Black = Masked', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Key Position', fontsize=10)
        ax.set_ylabel('Query Position', fontsize=10)
        plt.colorbar(im, ax=ax, label='0=Valid, 1=Masked', ticks=[0, 1])
        ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
        plt.tight_layout()
        plt.savefig('attention_mask_zoomed.png', dpi=300, bbox_inches='tight')
        print(f"Zoomed view saved as 'attention_mask_zoomed.png'")
        plt.show()
    """

    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from transformers import AutoTokenizer

    # Load your tokenizer
    model_path = "GSAI-ML/LLaDA-8B-Instruct"  # Replace with your actual model path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Sample conversation
    prompt = "What is the capital of France?"
    response = "The capital of France is Paris, a beautiful city known for its iconic Eiffel Tower, art museums, and rich history."

    # Create a chat-formatted prompt (adjust format based on your model's chat template)
    if hasattr(tokenizer, 'apply_chat_template'):
        # Use the model's chat template if available
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        # Fallback to simple concatenation
        full_text = f"User: {prompt}\nAssistant: {response}"

    print("="*80)
    print("FULL TEXT:")
    print("="*80)
    print(full_text)
    print("="*80)

    # Tokenize
    tokens = tokenizer.encode(full_text, add_special_tokens=True)
    input_ids = torch.tensor([tokens])

    print(f"\nTokenized sequence length: {len(tokens)}")
    print(f"First 20 tokens: {tokens[:20]}")
    print(f"Decoded first 20 tokens: {tokenizer.decode(tokens[:20])}")

    # Calculate prompt length (tokens up to the assistant's response)
    prompt_only = tokenizer.encode(
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], 
                                    tokenize=False, 
                                    add_generation_prompt=True) 
        if hasattr(tokenizer, 'apply_chat_template') 
        else f"User: {prompt}\nAssistant:",
        add_special_tokens=True
    )
    prompt_length = len(prompt_only)

    print(f"\nPrompt length (tokens): {prompt_length}")
    print(f"Response length (tokens): {len(tokens) - prompt_length}")

    # cave_in_generate parameters
    mask_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None else 0

    mask_id=126336

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    block_size = 4
    prompt_length_tensor = torch.tensor([prompt_length])

    print(f"\nMask token ID: {mask_id}")
    print(f"EOS token ID: {eos_id}")
    print(f"Block size: {block_size}")

    # Run cave_in_generate
    print("\n" + "="*80)
    print("RUNNING cave_in_generate")
    print("="*80)

    batch_input_ids, batch_noisy_batch, batch_p_mask, batch_mask_indices, batch_need_unmask, batch_attn_mask = cave_in_generate(
        input_ids, mask_id, block_size, prompt_length_tensor, eos_id
    )

    print(f"\nInput IDs shape: {batch_input_ids.shape}")
    print(f"Noisy batch shape: {batch_noisy_batch.shape}")
    print(f"Percentage shape: {batch_noisy_batch.shape}")
    print(f"Mask indices shape: {batch_mask_indices.shape}")
    print(f"Need unmask shape: {batch_need_unmask.shape}")
    print(f"Attention mask shape: {batch_attn_mask.shape}")

    print(f"\nNoisy batch (first 30 tokens): {batch_noisy_batch[0, :30].tolist()}")
    print(f"Mask indices (first 30): {batch_mask_indices[0, :30].tolist()}")
    print(f"Need unmask (first 30): {batch_need_unmask[0, :30].tolist()}")

    # Full text
    noisy_decoded = tokenizer.decode(batch_input_ids[0], skip_special_tokens=False)
    print("\n" + "="*80)
    print("FULL TEXT:")
    print("="*80)
    print(noisy_decoded)
    print("="*80)

    # Decode the noisy batch to see what it looks like
    noisy_decoded = tokenizer.decode(batch_noisy_batch[0], skip_special_tokens=False)
    print("\n" + "="*80)
    print("NOISY BATCH (with masks):")
    print("="*80)
    print(noisy_decoded)
    print("="*80)

    # Try one step of the algorithm to see what happens
    batch_noisy_batch_stepped = torch.where(batch_need_unmask.bool(), batch_input_ids, batch_noisy_batch)
    noisy_decoded = tokenizer.decode(batch_noisy_batch_stepped[0], skip_special_tokens=False)
    print("\n" + "="*80)
    print("NOISY BATCH AFTER PARTIAL DECODE (with masks):")
    print("="*80)
    print(noisy_decoded)
    print("="*80)

    print(f"PERCENTAGE MASKED:\n{batch_p_mask[0]}")

    # Visualization
    attn_mask_np = batch_attn_mask.squeeze().cpu().numpy()
    vis_mask = np.where(np.isneginf(attn_mask_np), 1, 0)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Full attention mask (binary)
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(vis_mask, cmap='binary', aspect='auto', interpolation='nearest')
    ax1.set_title(f'Full Attention Mask ({vis_mask.shape[0]}x{vis_mask.shape[1]})\nWhite = Valid, Black = Masked', 
                fontsize=14, fontweight='bold')
    ax1.set_xlabel('Key Position (Token Index)', fontsize=11)
    ax1.set_ylabel('Query Position (Token Index)', fontsize=11)
    ax1.axvline(x=prompt_length-0.5, color='red', linestyle='--', linewidth=2, label='Prompt End')
    ax1.axhline(y=prompt_length-0.5, color='red', linestyle='--', linewidth=2)
    ax1.legend(loc='upper right')
    plt.colorbar(im1, ax=ax1, label='0=Valid, 1=Masked', ticks=[0, 1])

    # Plot 2: Zoomed view of prompt region
    zoom_size = min(50, vis_mask.shape[0])
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(vis_mask[:zoom_size, :zoom_size], cmap='binary', aspect='auto', interpolation='nearest')
    ax2.set_title(f'Zoomed: First {zoom_size}x{zoom_size} Tokens', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Key Position', fontsize=10)
    ax2.set_ylabel('Query Position', fontsize=10)
    if prompt_length < zoom_size:
        ax2.axvline(x=prompt_length-0.5, color='red', linestyle='--', linewidth=2, label='Prompt End')
        ax2.axhline(y=prompt_length-0.5, color='red', linestyle='--', linewidth=2)
        ax2.legend(loc='upper right', fontsize=8)
    plt.colorbar(im2, ax=ax2, label='0=Valid, 1=Masked', ticks=[0, 1])

    # Plot 3: Mask pattern analysis
    ax3 = fig.add_subplot(gs[1, 1])
    mask_counts = vis_mask.sum(axis=1)  # Count masked positions per query
    ax3.plot(mask_counts, linewidth=2)
    ax3.set_title('Masked Positions per Query Token', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Query Position (Token Index)', fontsize=10)
    ax3.set_ylabel('Number of Masked Key Positions', fontsize=10)
    ax3.axvline(x=prompt_length, color='red', linestyle='--', linewidth=2, label='Prompt End')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Token information
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    info_text = f"""
    TOKEN INFORMATION:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Total Tokens: {len(tokens)}
    Prompt Tokens: {prompt_length}
    Response Tokens: {len(tokens) - prompt_length}

    Mask Token ID: {mask_id}
    EOS Token ID: {eos_id}
    Block Size: {block_size}

    Masked Tokens in Noisy Batch: {(batch_noisy_batch == mask_id).sum().item()}
    Tokens Needing Unmask: {batch_need_unmask.sum().item()}

    PROMPT: {prompt[:80]}{'...' if len(prompt) > 80 else ''}
    RESPONSE: {response[:80]}{'...' if len(response) > 80 else ''}
    """

    ax4.text(0.05, 0.5, info_text, fontsize=10, verticalalignment='center', 
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig('attention_mask_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'attention_mask_analysis.png'")
    plt.show()

    # Additional analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    print(f"Attention mask sparsity: {(vis_mask.sum() / vis_mask.size * 100):.2f}% masked")
    print(f"Average masked positions per query: {vis_mask.sum(axis=1).mean():.2f}")
    print(f"Max masked positions for any query: {vis_mask.sum(axis=1).max()}")
    print(f"Min masked positions for any query: {vis_mask.sum(axis=1).min()}")