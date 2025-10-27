import torch
import torch.nn.functional as F
import torch.distributions as dists
import transformers
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
import numpy as np
import random
import time
import os
from typing import List, Dict, Optional, Tuple, Iterator, Set
import gradio as gr

# Suppress some Hugging Face warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import necessary model classes
from model_cache.llada.modeling_llada import LLaDAModelLM
from model_cache.llada.configuration_llada import LLaDAConfig

# --- Helper Functions (Unchanged) ---
def set_seed(seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed); 
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
def create_full_block_attention_mask(prompt_length, max_length, block_size, device=None, dtype=None):
    if dtype is None: dtype = torch.bfloat16
    attention_mask = torch.full((1, 1, max_length, max_length), -torch.inf, device=device, dtype=dtype)
    attention_mask[:, :, :prompt_length, :prompt_length] = 0
    remaining_length = max_length - prompt_length
    num_blocks = (remaining_length + block_size - 1) // block_size
    for b in range(num_blocks):
        block_start = prompt_length + b * block_size; block_end = min(prompt_length + (b + 1) * block_size, max_length)
        attention_mask[:, :, block_start:block_end, :prompt_length] = 0
        for prev_b in range(b):
            prev_start = prompt_length + prev_b * block_size; prev_end = min(prompt_length + (prev_b + 1) * block_size, max_length)
            attention_mask[:, :, block_start:block_end, prev_start:prev_end] = 0
        attention_mask[:, :, block_start:block_end, block_start:block_end] = 0
    return attention_mask
def extract_attention_mask(full_mask, start_pos, input_length, cache_length):
    end_pos = start_pos + input_length; total_length = cache_length + input_length
    extracted_mask = torch.full((1, 1, input_length, total_length), -torch.inf, device=full_mask.device, dtype=full_mask.dtype)
    extracted_mask[:, :, :, :cache_length] = full_mask[:, :, start_pos:end_pos, :cache_length]
    extracted_mask[:, :, :, cache_length:] = full_mask[:, :, start_pos:end_pos, start_pos:end_pos]
    return extracted_mask
def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits
def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits
def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    if temperature > 0: logits = logits / temperature

    if top_p is not None and top_p < 1: logits = top_p_logits(logits, top_p)

    if top_k is not None: logits = top_k_logits(logits, top_k)

    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()

            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

        except: initial_confidence, x0 = probs.max(dim=-1)

    else: initial_confidence, x0 = probs.max(dim=-1)

    confidence = initial_confidence.clone()

    # ignore
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]

    if neg_entropy:
        epsilon = 1e-10
        confidence = torch.sum(probs * torch.log(probs + epsilon), dim=-1)

    return confidence, x0, initial_confidence


def remask_duplicates_(x0, mask_token_id):
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


def ntis_sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, num_ift_steps=-1, cur_ift_steps=-1, mask_token_id=-1, cur_x_t=None):
    if temperature > 0: logits = logits / temperature

    if top_p is not None and top_p < 1: logits = top_p_logits(logits, top_p)

    if top_k is not None: logits = top_k_logits(logits, top_k)

    x0 = torch.argmax(logits, dim=-1)

    # remasking for tokens if we are not at the last ift step
    if cur_ift_steps + 1 != num_ift_steps:
        probs = torch.softmax(logits, dim=-1)
        x0_p = torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

        #print(f"RMSK STR: {x0}")
        remask_duplicates_(x0, mask_token_id)
        #print(f"RMSK DUP: {x0}")


        # keep tokens that are either 1) very high confidence or 2) were masked previously
        conf_cond = x0_p > 0.9
        prev_cond = (cur_x_t == mask_token_id)
        remask_cond = torch.logical_or(conf_cond, prev_cond)

        x0 = torch.where(remask_cond, x0, mask_token_id)
        #print(f"RMSK LOW: {x0}")


        # remask endoftext or eot_id (==126348) tokens
        skip_end = torch.logical_or(x0 == mask_token_id, x0 == 126348) 
        x0 = torch.where(skip_end, cur_x_t, x0)
        #print(f"RMSK EOT: {x0}")


    return x0

class DreamLoRAInference:
    CSS = """
    /* Fixed height, scrollable visualization container */
    #viz-container {
        height: 500px;
        overflow-y: auto !important;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 10px;
        position: relative;
    }
    .block-container {
        display: inline-block; border: 2px solid transparent; border-radius: 8px;
        padding: 5px; margin: 4px 0; transition: border-color 0.3s, box-shadow 0.3s;
    }
    .block-updating {
        border-color: #FF4500 !important;
        box-shadow: 0 0 8px rgba(255, 69, 0, 0.7);
    }
    .token { padding: 2px 4px; margin: 2px; border-radius: 4px; display: inline-block; line-height: 1.4; font-family: monospace; }
    .token.prompt { background-color: #E5E7EB; color: #4B5563; }
    .token.gen-0 { background-color: #DBEAFE; color: #1E40AF; } /* Blue */
    .token.gen-1 { background-color: #D1FAE5; color: #065F46; } /* Green */
    .token.gen-2 { background-color: #FEF3C7; color: #92400E; } /* Yellow */
    .token.gen-3 { background-color: #FEE2E2; color: #991B1B; } /* Red */
    .token.gen-4 { background-color: #E0E7FF; color: #3730A3; } /* Indigo */
    .token.gen-5 { background-color: #F3E8FF; color: #6B21A8; } /* Purple */
    .token.mask { background-color: #F3F4F6; color: #9CA3AF; border: 1px dashed #D1D5DB; }

    /* Independent status box styles */
    #status-container {
        height: 300px;
        overflow-y: auto !important;
        margin-top: 10px; padding: 15px; border: 1px solid #E5E7EB; border-radius: 8px; background-color: #F9FAFB;
        position: relative;
    }
    #status-container h4 { margin-top: 0; }
    .status-line { font-family: monospace; font-size: 13px; margin-bottom: 5px; margin-top: 5px; padding: 2px 4px; border-radius: 3px;}
    #stats-output { padding: 15px; border: 1px solid #10B981; border-radius: 8px; background-color: #F0FDF4; margin-top: 10px; }
    
    /* Scroll anchor */
    .scroll-anchor {
        height: 1px;
        width: 100%;
    }
    
    /* Force scrollbar styles */
    #viz-container::-webkit-scrollbar, #status-container::-webkit-scrollbar {
        width: 10px !important;
        background-color: #f5f5f5 !important;
    }
    #viz-container::-webkit-scrollbar-thumb, #status-container::-webkit-scrollbar-thumb {
        background-color: #888 !important;
        border-radius: 5px !important;
    }
    #viz-container::-webkit-scrollbar-track, #status-container::-webkit-scrollbar-track {
        background-color: #f5f5f5 !important;
        border-radius: 5px !important;
    }
    
    /* Column height alignment */
    .left-column, .right-column {
        display: flex;
        flex-direction: column;
        height: auto !important;
        min-height: 800px;
    }
    
    .live-text-container, .viz-status-container {
        display: flex;
        flex-direction: column;
        flex: 1;
        overflow: visible;
    }
    
    #live-text-output, #stats-output {
        margin-bottom: 20px;
    }
    
    /* Fix for bottom content being cut off */
    .container {
        padding-bottom: 40px;
    }
    
    /* Make sure content is fully visible */
    .gradio-container {
        overflow-y: visible !important;
    }
    
    /* Add padding to bottom of page */
    .footer {
        margin-top: 30px;
        padding-bottom: 30px;
    }
    """

    def __init__(self, **kwargs):
        print("Initializing DreamLoRAInference...")
        self.device = torch.device(kwargs.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.__dict__.update(kwargs)
        if self.dtype == "bfloat16" and torch.cuda.is_bf16_supported(): self.target_dtype = torch.bfloat16
        elif self.dtype == "float16": self.target_dtype = torch.float16
        else: self.target_dtype = torch.float32
        self._setup_model(self.pretrained_path, self.lora_path)
        print("Model and tokenizer setup complete.")

    def _setup_model(self, pretrained_path, lora_path):
        config = LLaDAConfig.from_pretrained(pretrained_path)
        self.model = LLaDAModelLM.from_pretrained(pretrained_path, config=config, torch_dtype=self.target_dtype).eval()
        self.model = PeftModel.from_pretrained(self.model, lora_path)

        print(f" MOVING TO DEVICE {self.device}")
        print(f"Is cuda actually there? {torch.cuda.is_available()}")
        print(f"Info {pretrained_path} {lora_path}")

        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token

    def _apply_chat_template(self, prompt):
        chat_history = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)

    def _update_block_completion_states(self, block_states, decoded_token_threshold):
        for block_id in sorted(block_states.keys()):
            decoded_tokens = block_states[block_id]['total_masks'] - block_states[block_id]['mask_count']
            if block_states[block_id]['total_masks'] > 0:
                decode_ratio = decoded_tokens / block_states[block_id]['total_masks']
                if decode_ratio >= decoded_token_threshold:
                    if (next_block_id := block_id + 1) in block_states:
                        block_states[next_block_id]['is_complete'] = True

    # Render visualization part (excluding prompt status info)
    def _render_visualization_html(self, step: int, x_t: torch.Tensor, block_states: Dict, cache_length: int, updated_block_ids: Set[int]) -> str:
        timestamp = int(time.time() * 1000)
        
        html_parts = []
        for block_id in sorted(k for k in block_states.keys() if k > 0): # Only render generated part (block_id > 0)
            state = block_states[block_id]
            container_classes = ["block-container"]
            if block_id in updated_block_ids: container_classes.append("block-updating")
            html_parts.append(f'<div class="{" ".join(container_classes)}" id="block-{block_id}-{timestamp}">')
            block_tokens = x_t[0, state['start_pos']:state['end_pos']]
            for token_id in block_tokens:
                token_id_int = token_id.item()
                token_classes = ["token"]
                if token_id_int == self.mask_token_id:
                    token_str = '░'; token_classes.append("mask")
                else:
                    token_str = self.tokenizer.decode([token_id_int], skip_special_tokens=False)
                    token_str = token_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    token_classes.append(f"gen-{(block_id - 1) % 6}")
                html_parts.append(f'<span class="{" ".join(token_classes)}">{token_str}</span>')
            html_parts.append('</div>')
        
        html_parts.append(f'<div class="scroll-anchor" id="viz-anchor-{timestamp}"></div>')
        
        complete_html = f"""
        <div class="viz-content" id="viz-content-{timestamp}">
            {''.join(html_parts)}
        </div>
        
        <script>
        function executeVizScroll() {{
            const container = document.getElementById('viz-container');
            const anchor = document.getElementById('viz-anchor-{timestamp}');
            if (container && anchor) {{
                try {{
                    container.scrollTo(0, container.scrollHeight);
                    container.scrollTop = container.scrollHeight;
                    anchor.scrollIntoView({{behavior: 'auto', block: 'end'}});
                }} catch (e) {{
                    console.error('Scroll error:', e);
                }}
            }}
        }}
        
        setTimeout(executeVizScroll, 10);
        setTimeout(executeVizScroll, 50);
        setTimeout(executeVizScroll, 150);
        setTimeout(executeVizScroll, 300);
        
        try {{
            const vizContent = document.getElementById('viz-content-{timestamp}');
            const vizContainer = document.getElementById('viz-container');
            
            if (vizContent && vizContainer) {{
                const resizeObserver = new ResizeObserver(() => {{
                    executeVizScroll();
                }});
                resizeObserver.observe(vizContent);
                
                const mutationObserver = new MutationObserver(() => {{
                    executeVizScroll();
                }});
                mutationObserver.observe(vizContainer, {{ 
                    childList: true, 
                    subtree: true,
                    characterData: true 
                }});
            }}
        }} catch (e) {{
            console.error('Observer error:', e);
        }}
        </script>
        """
        
        return complete_html

    # Render status box part (only shows generation block information)
    def _render_status_html(self, step: int, block_states: Dict, cache_length: int) -> str:
        timestamp = int(time.time() * 1000)
        
        html_parts = []
        html_parts.append(f'<h4>Generation Block Status (Step: {step}, Cache Length: {cache_length})</h4>')
        for block_id in [k for k in sorted(block_states.keys()) if k > 0]:
            state = block_states[block_id]
            block_type = f"Block {block_id}"
            masks_filled = state['total_masks'] - state['mask_count']
            color_class = f"gen-{(block_id - 1) % 6}"
            status_line = f'<b>{block_type.ljust(8)}</b>: Pos=[{str(state["start_pos"]).rjust(4)}:{str(state["end_pos"]).ljust(4)}] | State=\'{state["state"].ljust(8)}\' | Filled={str(masks_filled).rjust(2)}/{state["total_masks"]}'
            html_parts.append(f'<p class="status-line token {color_class}" id="status-line-{block_id}-{timestamp}">{status_line}</p>')
        
        html_parts.append(f'<div class="scroll-anchor" id="status-anchor-{timestamp}"></div>')
        
        complete_html = f"""
        <div class="status-content" id="status-content-{timestamp}">
            {''.join(html_parts)}
        </div>
        
        <script>
        function executeStatusScroll() {{
            const container = document.getElementById('status-container');
            const anchor = document.getElementById('status-anchor-{timestamp}');
            if (container && anchor) {{
                try {{
                    container.scrollTo(0, container.scrollHeight);
                    container.scrollTop = container.scrollHeight;
                    anchor.scrollIntoView({{behavior: 'auto', block: 'end'}});
                }} catch (e) {{
                    console.error('Status scroll error:', e);
                }}
            }}
        }}
        
        setTimeout(executeStatusScroll, 10);
        setTimeout(executeStatusScroll, 50);
        setTimeout(executeStatusScroll, 150);
        setTimeout(executeStatusScroll, 300);
        
        try {{
            const statusContent = document.getElementById('status-content-{timestamp}');
            const statusContainer = document.getElementById('status-container');
            
            if (statusContent && statusContainer) {{
                const resizeObserver = new ResizeObserver(() => {{
                    executeStatusScroll();
                }});
                resizeObserver.observe(statusContent);
                
                const mutationObserver = new MutationObserver(() => {{
                    executeStatusScroll();
                }});
                mutationObserver.observe(statusContainer, {{ 
                    childList: true, 
                    subtree: true,
                    characterData: true 
                }});
            }}
        }} catch (e) {{
            console.error('Status observer error:', e);
        }}
        </script>
        """
        
        return complete_html

    @torch.inference_mode()
    def stream_and_capture_for_gradio(
        self,
        prompt_text: str,
        max_new_tokens: int,
        block_size: int,
        block_add_threshold: float,
        decoded_token_threshold: float,
        skip_threshold: float
    ) -> Iterator[Tuple[str, List[Tuple[str, str]], str, str, str]]:
        
        start_time = time.time()
        captured_frames: List[Tuple[str, str]] = []
        
        # Initialization
        input_ids = self.tokenizer(self._apply_chat_template(prompt_text), return_tensors="pt").input_ids.to(self.device)
        prompt_length = input_ids.shape[1]
        
        full_attention_mask = create_full_block_attention_mask(prompt_length, self.max_length, block_size, self.device, self.target_dtype)
        x_t = input_ids
        block_states = {0: {'start_pos': 0, 'end_pos': prompt_length, 'mask_count': 0, 'total_masks': prompt_length, 'state': 'to_cache', 'is_complete': True}}
        past_key_values, current_blocks, step, eos_detected, cache_length = None, 0, 0, False, 0
        
        # Capture initial state
        initial_viz_html = self._render_visualization_html(0, x_t, block_states, 0, set())
        initial_status_html = self._render_status_html(0, block_states, 0)
        captured_frames.append((initial_viz_html, initial_status_html))
        
        yield "", captured_frames, "Initializing generation process...", "Initializing visualization...", "Initializing block status..."

        num_ift_steps = 12
        cur_ift_step = num_ift_steps


        # Main generation loop
        while True:
            step += 1
            updated_block_ids: Set[int] = set()


            # logic for determining if we should add more blocks
            # we can just set block_add_threshold to 0.99 for this in our initial impl
            if len(block_states) - 1 < (max_new_tokens // block_size) and not eos_detected:
                # add new block if we have completed the number of ift steps for this iteration
                if cur_ift_step == num_ift_steps:
                    cur_ift_step = 0

                    # add new block
                    last_block_id = max(block_states.keys())
                    new_block_id = last_block_id + 1; new_start_pos = x_t.shape[1]
                    if new_start_pos + block_size <= self.max_length:
                        x_t = torch.cat([x_t, torch.full((1, block_size), self.mask_token_id, device=self.device, dtype=torch.long)], dim=1)
                        block_states[new_block_id] = {'start_pos': new_start_pos, 'end_pos': new_start_pos + block_size, 'mask_count': block_size, 'total_masks': block_size, 'state': 'active', 'is_complete': False}
                        current_blocks += 1


            # ntis sampler does not need to keep track of completion
            #self._update_block_completion_states(block_states, decoded_token_threshold)

            if (x_t == self.mask_token_id).sum() == 0 and current_blocks == 0: break
            #if (x_t == self.mask_token_id).sum() == 0 and current_blocks == 0: break

            print(f"Cur ift step: {cur_ift_step + 1}/{num_ift_steps}")

            blocks_to_cache = [bid for bid, state in block_states.items() if state['state'] == 'to_cache']
            if blocks_to_cache:
                # recompute kv cache for all blocks in this range
                # end pos is actually decieving here (it is just a temp var that matters in this scope)
                start_pos, end_pos = block_states[min(blocks_to_cache)]['start_pos'], block_states[max(blocks_to_cache)]['end_pos']
                update_kvcache = end_pos - start_pos

                input_seq, process_start_pos = x_t[:, start_pos:], start_pos
            else:
                update_kvcache = 0

                active_blocks = [bid for bid, state in block_states.items() if state['state'] == 'active' and state['start_pos'] >= cache_length]
                if not active_blocks: break

                start_pos = min(block_states[bid]['start_pos'] for bid in active_blocks)

                input_seq, process_start_pos = x_t[:, start_pos:], start_pos
            
            if input_seq.shape[1] == 0: break


            attention_mask = extract_attention_mask(full_attention_mask, process_start_pos, input_seq.shape[1], cache_length)
            outputs = self.model(input_seq, attention_bias=attention_mask, past_key_values=past_key_values, use_cache=True, update_kvcache=update_kvcache + cache_length)
            
            # active is currently being decoded
            # to_cache is we need to create kv_cache
            # in_cache is that kv_cache already created 
            if update_kvcache > 0:
                past_key_values = outputs.past_key_values
                for bid in blocks_to_cache: block_states[bid]['state'] = 'in_cache'


            blocks_to_deactivate = []
            for block_id, state in block_states.items():
                if state['state'] != 'active': continue

                true_start = state['start_pos'] - process_start_pos
                true_end = state['end_pos'] - process_start_pos

                block_all_logits = outputs.logits[0, true_start:true_end, :]
                cur_x_t = x_t[0, state['start_pos']:state['end_pos']]


                # for now, only one block will be active, so we don't need to keep track of ift iterations on a per-block basis
                x0 = ntis_sample_tokens(block_all_logits, self.temperature, self.top_p, self.top_k, num_ift_steps, cur_ift_step, self.mask_token_id, cur_x_t)
                x_t[0, state["start_pos"]:state["end_pos"]] = x0

                # update information
                updated_block_ids.add(block_id)
                state['mask_count'] = (x0 == self.mask_token_id).sum().item()

                if self.tokenizer.eos_token_id in x0:
                    print("\tEOT detected!")
                    eos_detected = True

                # move to deactivation if need be
                cur_ift_step += 1
                if cur_ift_step == num_ift_steps: 
                    print("\tWe have unmasked all tokens")
                    blocks_to_deactivate.append(block_id)
            


            # check if we can cache it
            for bid in blocks_to_deactivate:
                if block_states[bid]['state'] == 'active' and all(block_states.get(i, {}).get('state') != 'active' for i in range(bid)):
                    block_states[bid]['state'] = 'to_cache'; current_blocks -= 1
            if update_kvcache > 0: cache_length += update_kvcache
            
            # Capture current step's visualization and status frames
            generated_ids = x_t[0, prompt_length:]
            valid_ids = generated_ids[generated_ids != self.mask_token_id]
            live_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            
            current_viz_html = self._render_visualization_html(step, x_t, block_states, cache_length, updated_block_ids)
            current_status_html = self._render_status_html(step, block_states, cache_length)
            captured_frames.append((current_viz_html, current_status_html))
            
            yield live_text, captured_frames, "Generating...", "Generating...", "Generating..."
            

        print("Done generating text")

        # Final output
        total_time = time.time() - start_time
        final_generated_ids = x_t[0, prompt_length:]
        eos_positions = (final_generated_ids == self.tokenizer.eos_token_id).nonzero()
        if eos_positions.numel() > 0:
            final_generated_ids = final_generated_ids[:eos_positions[0, 0] + 1]

        final_text = self.tokenizer.decode(final_generated_ids, skip_special_tokens=True)
        final_viz_html = self._render_visualization_html(step, x_t, block_states, cache_length, set())
        final_status_html = self._render_status_html(step, block_states, cache_length)
        captured_frames.append((final_viz_html, final_status_html))
        
        tokens_incl_eos = len(final_generated_ids)
        tokens_excl_eos = len(final_generated_ids[final_generated_ids != self.tokenizer.eos_token_id])
        stats_text = f"""
        ### ✅ Generation Complete!
        ---
        - **Total time:** `{total_time:.2f} seconds`
        - **Tokens generated (incl. EOS):** `{tokens_incl_eos}`
        - **Tokens generated (excl. EOS):** `{tokens_excl_eos}`
        - **Tokens per second:** `{(tokens_incl_eos / total_time):.2f}`
        """
        
        yield final_text, captured_frames, stats_text, "Generation complete, playback starting soon", "Generation complete, playback starting soon"


# --- Gradio UI and Event Handlers ---
if __name__ == "__main__":
    config = {
        "pretrained_path": "GSAI-ML/LLaDA-8B-Instruct",
        "lora_path": "SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora",
        "device": "cuda", "dtype": "bfloat16", "max_length": 4096,
        "temperature": 0.0, "top_p": None, "top_k": None, "mask_token_id": 126336,
        "sampling_strategy": "default",
    }
    set_seed(42)
    inference_engine = DreamLoRAInference(**config)
    
    # Gradio helper for animation
    def animate_visualization(html_frames_list: List[Tuple[str, str]], delay: float) -> Iterator[Tuple[str, str]]:
        if not html_frames_list:
            yield "No visualization data captured", "No status data captured"
            return
        for viz_frame, status_frame in html_frames_list:
            yield viz_frame, status_frame
            time.sleep(delay)

    # Global auto-scroll JS
    auto_scroll_js = """
    <script>
    function globalForceScroll() {
        // Scroll visualization container
        var vizContainer = document.getElementById('viz-container');
        if (vizContainer) {
            vizContainer.scrollTop = vizContainer.scrollHeight;
        }
        
        // Scroll status container
        var statusContainer = document.getElementById('status-container');
        if (statusContainer) {
            statusContainer.scrollTop = statusContainer.scrollHeight;
        }
        
        // Scroll all anchors
        var anchors = document.querySelectorAll('.scroll-anchor');
        anchors.forEach(function(anchor) {
            try {
                anchor.scrollIntoView({behavior: 'auto', block: 'end'});
            } catch(e) {}
        });
    }
    
    // Periodic scrolling
    setInterval(globalForceScroll, 200);
    
    document.addEventListener('DOMContentLoaded', function() {
        // Monitor content changes
        var observer = new MutationObserver(function(mutations) {
            globalForceScroll();
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            characterData: true
        });
        
        // Initial scrolling
        setTimeout(globalForceScroll, 100);
        setTimeout(globalForceScroll, 500);
        setTimeout(globalForceScroll, 1000);
    });
    </script>
    """

    with gr.Blocks(css=DreamLoRAInference.CSS, theme=gr.themes.Soft()) as demo:
        html_frames_state = gr.State([])

        gr.Markdown("# ✨ D2F-LLaDA: Real-time Text vs. Slow-motion Visualization")
        gr.Markdown("Left side shows real-time streaming output. Right side plays back the decoding process visualization after generation completes.")
        
        # Inject global auto-scroll JS
        gr.HTML(auto_scroll_js)
        
        with gr.Row():
            # --- Left Column ---
            with gr.Column(scale=2, elem_classes=["left-column"]):
                prompt_input = gr.Textbox(label="Enter your question", placeholder="Example: Natalia sold clips to...", lines=5)
                generate_button = gr.Button("🚀 Generate & Visualize", variant="primary")
                with gr.Group(elem_classes=["live-text-container"]):
                    live_text_output = gr.Textbox(label="Real-time Generation Output", interactive=False, lines=25, elem_id="live-text-output")
                    stats_output = gr.Markdown(label="Generation Statistics", elem_id="stats-output")

            # --- Right Column ---
            with gr.Column(scale=3, elem_classes=["right-column"]):
                with gr.Accordion("⚙️ Parameter Settings", open=True):
                    with gr.Row():
                        max_new_tokens_slider = gr.Slider(minimum=64, maximum=2048, value=1024, step=64, label="Max Tokens to Generate")
                        block_size_slider = gr.Slider(minimum=16, maximum=128, value=32, step=16, label="Block Size")
                    with gr.Row():
                        block_add_thresh_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.05, label="Block Add Threshold")
                        decoded_token_thresh_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Decoding Completion Threshold")
                        skip_thresh_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.01, label="Skip Threshold")
                    delay_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.05, label="Playback Delay (seconds)", info="Adjust visualization playback speed.")
                
                with gr.Group(elem_classes=["viz-status-container"]):
                    visualization_output = gr.HTML(label="Generation Process Visualization", elem_id="viz-container")
                    status_output_html = gr.HTML(label="Generation Block Status", elem_id="status-container")

        gr.Examples(
            examples=[
                ["Solve the equation x² - 6x + 8 = 0. First, explain what a quadratic equation is and why it can have up to two solutions. Then solve this equation using three different methods: factoring, completing the square, and the quadratic formula. For each method, explain the mathematical reasoning behind it, show all steps in detail, and discuss when this particular method is most useful. Finally, verify your solutions by substituting them back into the original equation.", 1024, 32, 0.1, 0.55, 0.9, 0.1],
                
                ["A circular swimming pool has a diameter of 8 meters. Calculate the pool's circumference and area. First, explain the relationship between diameter, radius, circumference, and area of a circle, including the role of π in these formulas. Then perform the calculations using π ≈ 3.14159. Next, estimate how much water (in cubic meters) would be needed to fill this pool if it has a uniform depth of 1.5 meters. Finally, calculate how much it would cost to fill this pool if water costs $2.50 per cubic meter. Show all steps and include appropriate units in your answer.", 1024, 32, 0.1, 0.5, 0.9, 0.1],
                
                ["A movie theater offers a loyalty card that costs $15 and gives a 15% discount on all tickets. If a regular movie ticket costs $10, how many tickets would you need to buy to make the loyalty card worthwhile? First, explain the concept of a break-even point. Then set up an equation to find when the total cost with the card equals the total cost without the card. Solve this equation step by step, showing all your work. Finally, interpret your answer in the context of the problem.", 1024, 32, 0.1, 0.5, 0.9, 0.1],
            ],
            inputs=[
                prompt_input, max_new_tokens_slider, block_size_slider, block_add_thresh_slider,
                decoded_token_thresh_slider, skip_thresh_slider, delay_slider
            ],
            label="Examples (Math Problems)"
        )
                
        # --- Event Handling Chain ---
        inputs_list = [
            prompt_input, max_new_tokens_slider, block_size_slider,
            block_add_thresh_slider, decoded_token_thresh_slider, skip_thresh_slider
        ]
        
        generation_event = generate_button.click(
            fn=inference_engine.stream_and_capture_for_gradio,
            inputs=inputs_list,
            outputs=[live_text_output, html_frames_state, stats_output, visualization_output, status_output_html]
        )
        
        generation_event.then(
            fn=animate_visualization,
            inputs=[html_frames_state, delay_slider],
            outputs=[visualization_output, status_output_html]
        )

    demo.queue().launch(share=True)