import logging
import gc
import json
import time  # Add time module
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union, Dict
import torch
import torch.nn.functional as F
import torch.distributions as dists
import transformers
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datasets import Dataset
from packaging import version
from tqdm import tqdm
from peft import PeftConfig, PeftModel
import numpy as np  # Add numpy import
import os
import jinja2

# Import LLaDA model related modules
from model_cache.llada.modeling_llada import LLaDAModelLM
from model_cache.llada.configuration_llada import LLaDAConfig

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from lm_eval.__main__ import cli_evaluate

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="TemplateLM")

import random
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_full_block_attention_mask(prompt_length, max_length, block_size, device=None, dtype=None):
    """
    Creates a complete attention mask for the entire sequence with block-based causal attention.
    
    Args:
        prompt_length: Length of the prompt (first irregular block)
        max_length: Maximum total sequence length
        block_size: Size of each regular block
        device: Device to create tensor on
        dtype: Data type for the attention mask
        
    Returns:
        attention_mask: Tensor of shape [1, 1, max_length, max_length]
    """
    # Use the provided dtype or default to bfloat16
    if dtype is None:
        dtype = torch.bfloat16
    
    # Initialize mask with -inf (no attention)
    attention_mask = torch.full((1, 1, max_length, max_length), -torch.inf, device=device, dtype=dtype)
    
    # Block 0: Prompt (can see itself)
    attention_mask[:, :, :prompt_length, :prompt_length] = 0
    
    # Calculate the number of regular blocks after prompt
    remaining_length = max_length - prompt_length
    num_blocks = (remaining_length + block_size - 1) // block_size
    
    # Process each regular block
    for b in range(num_blocks):
        block_start = prompt_length + b * block_size
        block_end = min(prompt_length + (b + 1) * block_size, max_length)
        
        # Current block can see the prompt
        attention_mask[:, :, block_start:block_end, :prompt_length] = 0
        
        # Current block can see all previous regular blocks
        for prev_b in range(b):
            prev_start = prompt_length + prev_b * block_size
            prev_end = min(prompt_length + (prev_b + 1) * block_size, max_length)
            attention_mask[:, :, block_start:block_end, prev_start:prev_end] = 0
        
        # Current block can see itself (full attention within block)
        attention_mask[:, :, block_start:block_end, block_start:block_end] = 0
    
    return attention_mask

def extract_attention_mask(full_mask, start_pos, input_length, cache_length):
    """
    Extract the relevant portion of attention mask for current forward pass.
    
    Args:
        full_mask: Complete attention mask [1, 1, max_length, max_length]
        start_pos: Starting position in the full sequence
        input_length: Length of current input sequence
        cache_length: Length of cached sequence
        
    Returns:
        attention_mask: Extracted mask [1, 1, input_length, cache_length + input_length]
    """
    end_pos = start_pos + input_length
    total_length = cache_length + input_length
    
    # Extract the relevant rows (current input positions)
    # and columns (cache + current input positions)
    extracted_mask = torch.full((1, 1, input_length, total_length), -torch.inf, 
                               device=full_mask.device, dtype=full_mask.dtype)
    
    # Copy cache columns (0 to cache_length in the extracted mask corresponds to 0 to cache_length in full mask)
    extracted_mask[:, :, :, :cache_length] = full_mask[:, :, start_pos:end_pos, :cache_length]
    
    # Copy current input columns
    extracted_mask[:, :, :, cache_length:] = full_mask[:, :, start_pos:end_pos, start_pos:end_pos]
    
    return extracted_mask

def build_custom_float_attention_mask(input_ids, prompt_length, block_size, device=None, dtype=None):
    """
    Builds a custom float attention mask with block-based causal attention.

    Args:
        input_ids: Input token IDs.
        prompt_length: Length of the prompt for each sequence in the batch.
        block_size: Size of each regular block.
        device: Device to create tensor on.
        dtype: Data type for the attention mask.

    Returns:
        attn_mask: Tensor of shape [B, 1, seq_len, seq_len].
    """
    B, seq_len = input_ids.shape
    # Use the provided dtype or default to float32
    if dtype is None:
        dtype = torch.float32
    # Initialize to all -inf
    attn_mask = torch.full((B, 1, seq_len, seq_len), float('-inf'), dtype=dtype, device=device)
    # 1. Prompt section: each token can attend to the entire prompt
    for i in range(B):
        attn_mask[i, :, :, :prompt_length[i]] = 0.0  # Allow all tokens to see the prompt

        # 2. Block division: divide blocks starting from prompt_length
        num_blocks = (seq_len - prompt_length[i] + block_size - 1) // block_size

        for b in range(num_blocks):
            block_start = prompt_length[i] + b * block_size
            block_end = min(block_start + block_size, seq_len)

            # Full attention within the block
            attn_mask[i, :, block_start:block_end, block_start:block_end] = 0.0

            # Causal attention between blocks (can only see previous blocks)
            for prev_b in range(b):
                prev_start = prompt_length[i] + prev_b * block_size
                prev_end = min(prev_start + block_size, seq_len)

                # Current block can see previous blocks
                attn_mask[i, :, block_start:block_end, prev_start:prev_end] = 0.0

    return attn_mask

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            initial_confidence, x0 = probs.max(dim=-1)
    else:
        initial_confidence, x0 = probs.max(dim=-1)
    
    # Save initial confidence
    confidence = initial_confidence.clone()
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
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


def ntis_sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, num_ift_steps=-1, cur_ift_steps=-1, mask_token_id=-1, eot_token_id=-1, cur_x_t=None):
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
        skip_end = torch.logical_or(x0 == eot_token_id, x0 == 126348) 
        x0 = torch.where(skip_end, cur_x_t, x0)
        #print(f"RMSK EOT: {x0}")


    return x0

@register_model("dream_lora")
class DreamLoRA(TemplateLM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        lora_path: str,
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 128,
        max_length: Optional[int] = 4096,  # Updated to match example code
        add_bos_token: Optional[bool] = False,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        mc_num: Optional[int] = 128,
        classifier_free_guidance: Optional[float] = 1.0,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 128,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        temperature: Optional[float] = 0.2,  # Updated default value
        top_p: Optional[float] = None,  # Updated default value
        top_k: Optional[float] = None,
        alg: Optional[str] = "entropy",
        alg_temp: Optional[float] = 0.0,
        escape_until: Optional[bool] = False,
        block_size: Optional[int] = 4,  # Updated to match example code
        mask_token_id: Optional[int] = 126336,  # Added mask_token_id parameter
        block_add_threshold: Optional[float] = 0.5,  # Added block_add_threshold parameter
        decoded_token_threshold: Optional[float] = 0.9,  # Added decoded token threshold parameter
        skip_threshold: Optional[float] = 1.0,  # Added skip_threshold parameter
        sampling_strategy: Optional[str] = "default",  # Added sampling strategy parameter
        save_dir: Optional[str] = None,  # Added save directory parameter
        show_speed: Optional[bool] = True,  # Added speed statistics parameter
        **kwargs,
    ) -> None:
        super().__init__()

        # prepare for parallelism
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        # using one process with no model parallelism
        if not (parallelize or accelerator.num_processes > 1):
            # use user-passed device
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(
                    torch.__version__
                ) < version.parse("2.1"):
                    raise RuntimeError(
                        f"mps requires torch >= 2.1. You have {torch.__version__}"
                    )
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        else:  # Parallelism managed by accelerate
            if device != "cuda":
                eval_logger.info(
                    f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                )
            # TODO: include in warning that `load_in_8bit` etc. affect this too
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)
        
        # Save LoRA path and block_size
        self.lora_path = lora_path
        self.block_size = block_size
        self.block_add_threshold = block_add_threshold  # Added block_add_threshold attribute
        self.skip_threshold = skip_threshold  # Added skip_threshold attribute
        self.sampling_strategy = sampling_strategy  # Save sampling strategy parameter
        self.decoded_token_threshold = decoded_token_threshold  # Added decoded token threshold attribute
        
        # Save target_dtype for later use
        self.target_dtype = get_dtype(dtype)
        
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                # TODO: can remove this whole snippet except in the mps case, perhaps?
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    # place model onto device requested manually,
                    # if not using HF Accelerate or device_map
                    # or any other option that preloads model onto device
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        self.max_length = max_length
        self.add_bos_token = add_bos_token
        # generation params
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alg = alg
        self.alg_temp = alg_temp
        self.escape_until = escape_until
        self.block_size = block_size
        self.mask_token_id = mask_token_id

        # loglikelihood params
        self.nll_type = nll_type
        self.log_type = log_type
        self.mc_num = mc_num
        self.classifier_free_guidance = classifier_free_guidance
        self.sampling_eps = sampling_eps
        
        # Add backend attribute, consistent with LLaDA.py
        self.backend = "causal"
        
        # Add truncation attribute, consistent with LLaDA.py
        self.truncation = False

        self.save_dir = save_dir
        self.show_speed = show_speed

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def eot_token_id(self): 
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code):
        # Get correct data type
        target_dtype = get_dtype(dtype)
        
        # Load LLaDA model and configuration
        config = LLaDAConfig.from_pretrained(pretrained)
        self.model = LLaDAModelLM.from_pretrained(
            pretrained, 
            config=config,
            torch_dtype=target_dtype,
            trust_remote_code=False,
        ).eval()
        
        # Load LoRA configuration and model
        peft_config = PeftConfig.from_pretrained(self.lora_path)
        self.model = PeftModel.from_pretrained(self.model, self.lora_path)
        
        # Convert data type only when target_dtype is not None and not "auto"
        if target_dtype is not None and target_dtype != "auto":
            self.model = self.model.to(target_dtype)
        
        # Move to specified device
        self.model = self.model.to(self.device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs = {
                    "add_special_tokens": False or self.add_bos_token
                }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.backend == "causal":
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len:
            original_lengths = encoding["input_ids"].size(1)
            if original_lengths > left_truncate_len:
                eval_logger.warn(
                    f"Left truncation applied. Original sequence length was {original_lengths}, "
                    f"truncating to last {left_truncate_len} tokens. Some content will be lost.",
                )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    

    def _count_tokens_after_truncation(self, response_text: str, until_terms: List[str] = None) -> int:
        """
        Unified token counting function: calculates the number of non-126081 tokens after truncating the response.
        """
        # Apply truncation based on until parameters
        truncated_text = response_text
        if until_terms and not self.escape_until:
            for term in until_terms:
                if len(term) > 0:
                    truncated_text = truncated_text.split(term)[0]
        
        # Re-tokenize processed answer and count non-126081 tokens
        generated_answer_ids = torch.tensor(self.tokenizer(truncated_text)["input_ids"])
        return int((generated_answer_ids != 126081).sum())

    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the LM class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the LM class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )

        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _generate_block_single(self, prompt):
        """
        Generates a response for a single prompt using parallel block generation, based on KV cache, and uses pre-generated attention masks.
        Returns: generated_sequence (List[int]) - List of generated token IDs
        """
        self.model.eval()
        
        mask_id = self.mask_token_id
        block_size = self.block_size
        block_add_threshold = self.block_add_threshold
        skip_threshold = self.skip_threshold
        
        # Pre-generate the full attention mask, using the model's data type
        prompt_length = prompt.shape[1]
        full_attention_mask = create_full_block_attention_mask(
            prompt_length=prompt_length,
            max_length=self.max_length,
            block_size=block_size,
            device=self.device,
            dtype=self.target_dtype if self.target_dtype is not None and self.target_dtype != "auto" else torch.bfloat16
        )
        
        with torch.inference_mode():
            # Initialization
            x_t = prompt.to(self.device)
            
            # Track block states - states can be: 'active', 'to_cache', 'in_cache'
            # Added 'is_complete' field to indicate whether it's a complete state (True) or incomplete state (False)
            block_states = {
                0: {
                    'start_pos': 0,
                    'end_pos': prompt.shape[1],
                    'mask_count': 0,
                    'total_masks': prompt.shape[1],
                    'state': 'to_cache',  # Prompt is immediately ready for caching
                    'is_complete': True,  # Prompt is always in a complete state
                },
            }
            
            # Initialize cache
            past_key_values = None
            
            current_blocks = 0  # Number of active blocks
            step = 0
            eos_detected = False  # EOS detection flag
            cache_length = 0

            num_ift_steps = 12
            cur_ift_step = num_ift_steps

            while True:
                step += 1
                
                # Check if a new block needs to be added
                if len(block_states)-1 < (self.max_new_tokens // block_size) and not eos_detected:
                    if cur_ift_step == num_ift_steps:
                        cur_ift_step = 0
                        # Add new block
                        new_block_id = len(block_states)
                        new_start_pos = x_t.shape[1]
                        x_t = torch.cat([x_t, torch.tensor([[mask_id] * block_size]).to(self.device)], dim=1)
                        
                        block_states[new_block_id] = {
                            'start_pos': new_start_pos,
                            'end_pos': new_start_pos + block_size,
                            'mask_count': block_size,
                            'total_masks': block_size,
                            'state': 'active',
                            'is_complete': False,  # New block defaults to an incomplete state
                        }
                        current_blocks += 1
                
                # At the beginning of each loop, update the block's complete/incomplete states
                self._update_block_completion_states(block_states, self.decoded_token_threshold)
                # Check if there are still mask tokens
                mask_index = (x_t == mask_id)
                if mask_index.sum() == 0 and current_blocks == 0:
                    break
                
                # Determine which blocks need to be added to the cache
                blocks_to_cache = [bid for bid, state in block_states.items() 
                                if state['state'] == 'to_cache']
                    
                # Determine the part to be processed
                update_kvcache = 0
                if blocks_to_cache:
                    # Find the earliest block to be cached
                    earliest_block_id = min(blocks_to_cache)
                    earliest_pos = block_states[earliest_block_id]['start_pos']
                    
                    # Find the latest block to be cached
                    latest_block_id = max(blocks_to_cache)
                    latest_pos = block_states[latest_block_id]['end_pos']
                    
                    # Update the cache for all blocks within this range
                    update_kvcache = latest_pos - earliest_pos
                
                # Create input sequence for forward pass
                process_start_pos = cache_length
                
                if update_kvcache > 0:
                    # Need to update cache - use completed blocks
                    earliest_block_to_cache = min(blocks_to_cache)
                    input_seq = x_t[:, block_states[earliest_block_to_cache]['start_pos']:]
                    process_start_pos = block_states[earliest_block_to_cache]['start_pos']
                else:
                    # this code is starting to piss me off now

                    # Only process active blocks
                    active_blocks = [bid for bid, state in block_states.items() if state['state'] == 'active']
                    if active_blocks:
                        # Get all active blocks after caching
                        earliest_active_after_cache = float('inf')
                        for bid in active_blocks:
                            if block_states[bid]['start_pos'] >= cache_length:
                                earliest_active_after_cache = min(earliest_active_after_cache, block_states[bid]['start_pos'])
                        
                        if earliest_active_after_cache < float('inf'):
                            input_seq = x_t[:, earliest_active_after_cache:]
                            process_start_pos = earliest_active_after_cache
                        else:
                            # No active blocks after caching, this should not happen
                            input_seq = x_t[:, cache_length:]
                            # If cache length is already equal to or exceeds sequence length, exit
                            if cache_length >= x_t.shape[1]:
                                print(f"Cache length ({cache_length}) >= sequence length ({x_t.shape[1]}) at step {step}. Exiting generation loop.")
                                raise Exception("Cache length >= sequence length")
                    else:
                        # No active blocks, but blocks might need to be cached in the next iteration
                        # wait, if we break then we are done aren't we? 
                        break
                
                # Check if input_seq is empty
                if input_seq.shape[1] == 0:
                    print(f"Warning: input_seq is empty at step {step}. Breaking generation loop.")
                    raise Exception("input_seq is empty")
                
                # Extract the attention mask for the current input from the pre-generated full mask
                input_length = input_seq.shape[1]
                attention_mask = extract_attention_mask(
                    full_mask=full_attention_mask,
                    start_pos=process_start_pos,
                    input_length=input_length,
                    cache_length=cache_length
                )

                outputs = self.model(
                    input_seq,
                    attention_bias=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    update_kvcache=update_kvcache+cache_length,
                )
                
                # Get current logits - LLaDA model directly uses logits, no shifting needed
                logits = outputs.logits
                
                # Update cache if needed
                if update_kvcache > 0:
                    # Update cache
                    past_key_values = outputs.past_key_values
                    
                    # Mark blocks as cached
                    for block_id in blocks_to_cache:
                        block_states[block_id]['state'] = 'in_cache'
                
                # Process mask tokens for each active block
                blocks_to_deactivate = []
                
                for block_id, state in block_states.items():
                        if state['state'] != 'active': continue

                        true_start = state['start_pos'] - process_start_pos
                        true_end = state['end_pos'] - process_start_pos

                        block_all_logits = outputs.logits[0, true_start:true_end, :]
                        cur_x_t = x_t[0, state['start_pos']:state['end_pos']]


                        # for now, only one block will be active, so we don't need to keep track of ift iterations on a per-block basis
                        x0 = ntis_sample_tokens(block_all_logits, self.temperature, self.top_p, self.top_k, num_ift_steps, cur_ift_step, self.mask_token_id, self.tokenizer.eos_token_id, cur_x_t)
                        x_t[0, state["start_pos"]:state["end_pos"]] = x0

                        # update information
                        state['mask_count'] = (x0 == self.mask_token_id).sum().item()

                        if self.tokenizer.eos_token_id in x0:
                            print("\tEOT detected!")
                            eos_detected = True

                        # move to deactivation if need be
                        cur_ift_step += 1
                        if cur_ift_step == num_ift_steps: 
                            print("\tWe have unmasked all tokens")
                            blocks_to_deactivate.append(block_id)
                
                # Deactivate completed blocks and mark them for caching in the next iteration
                for block_id in blocks_to_deactivate:
                    if block_states[block_id]['state'] == 'active':
                        # Check if all preceding blocks are already in a non-active state
                        can_deactivate = True
                        for prev_block_id in range(block_id):
                            if prev_block_id in block_states and block_states[prev_block_id]['state'] == 'active':
                                can_deactivate = False
                                break
                        
                        # Only mark the current block as 'to_cache' if all preceding blocks are not active
                        if can_deactivate:
                            block_states[block_id]['state'] = 'to_cache'
                            current_blocks -= 1
                        # If there are active preceding blocks, keep the current block in active state (do nothing)

                if update_kvcache > 0:
                    cache_length += update_kvcache
                # Safety check
                if step > 10000:
                    print(f"WARNING: Hit safety check at step {step}. Exiting generation loop.")
                    break

                current_text = self.tokenizer.decode(x_t[0, prompt.shape[1]:].tolist(),skip_special_tokens=False)
        
        # Generate final answer
        generated_sequence = x_t[0, prompt.shape[1]:].tolist()
        
        return generated_sequence



    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        start_time = time.time()
        
        # Statistics variables
        num_tokens = 0
        num_nfe = 0
        
        bar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)), desc="Running generate_until requests")
        
        for i, req in enumerate(requests):
            question = req.args[0]
            # print("question:",question)
            # exit()
            gen_kwargs = req.args[1]
            
            # Process input in LLaDA.py style
            # print("Self.add_bos_token:", self.add_bos_token)
            contexts = [question]
            if self.add_bos_token:
                contexts = [self.tokenizer.bos_token + p for p in contexts]
            
            # Use the same tokenization method as LLaDA.py
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                truncation=self.truncation,
            )


            
            input_ids = context_enc[0].unsqueeze(0)  # Take the first one and add batch dimension
            
            # Add length check
            if input_ids.shape[1] > self.max_length - self.max_new_tokens:
                eval_logger.warning(f"Prompt length {input_ids.shape[1]} is larger than {self.max_length-self.max_new_tokens}, cutoff on the left side")
                input_ids = input_ids[:, -(self.max_length-self.max_new_tokens):]
            
            # Generate token IDs
            generated_answer = self._generate_block_single(input_ids)
            
            # Use tokenizer.batch_decode for decoding, consistent with LLaDA.py
            cont_toks_list = self.tokenizer.batch_decode([generated_answer], skip_special_tokens=True)
            s = cont_toks_list[0]  # Take the first (and only) result

            print(f"\tOutput is {s}")
            
            # Use unified token counting function
            if self.show_speed:
                num_tokens += self._count_tokens_after_truncation(s, gen_kwargs.get("until", []))
                num_nfe += 1  # NFE uses simplified statistics (fixed to 1)
            
            # Handle until truncation in LLaDA.py style
            if not self.escape_until:
                for term in gen_kwargs.get("until", []):
                    if len(term) > 0:
                        s = s.split(term)[0]
            
            res.append(s)
            bar.update(1)
        
        bar.close()
        
        # Save statistics only at the end
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            final_time = time.time()
            total_time = final_time - start_time
            
            final_stats = {
                "processed_samples": len(res),
                "total_samples": len(requests), 
                "total_tokens": int(num_tokens),
                "total_nfe": int(num_nfe),
                "total_time": total_time,
                "tokens_per_second": float(num_tokens) / total_time if total_time > 0 else 0.0,
                "nfe_per_token": float(num_nfe) / float(num_tokens) if num_tokens > 0 else 0.0,
                "timestamp": final_time
            }
            final_stats_path = os.path.join(self.save_dir, f'rank_{self.rank}_final_stats.json')
            with open(final_stats_path, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)
        
        if self.show_speed:
            final_time = time.time()
            total_time = final_time - start_time
            print(f"\n=== Final Statistics ===")
            print(f"Processed samples: {len(res)}")
            print(f"Total tokens: {num_tokens}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Throughput: {num_tokens / total_time:.2f} tokens/s")
            print(f"Total NFE: {num_nfe}")
        
        return res

    def _forward_process(self, batch):
        b, l = batch.shape
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps

        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        # always unmask bos and eos
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.mask_token_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        '''
        prompt_index : 1D bool tensor, length=batch.shape[1]
        '''
        if self.classifier_free_guidance > 1.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_token_id
            batch = torch.cat([batch, un_batch])

        input = batch

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(input).logits
            # since bos always unmask, the first logits will not be used
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

        if self.classifier_free_guidance > 1.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.cfg * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        if prefix is None:
            seq = target[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        
        if self.log_type == 'ftb':
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)

        loss_acc = []
        for _ in range(max(self.mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            # eval_logger.info("before noising")
            perturbed_seq_, p_mask = self._forward_process(seq)
            # eval_logger.info("end noising")
            if self.log_type == 'ftb':
                perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]
            elif self.log_type == 'btf':
                perturbed_seq[:, :len(prefix)] = perturbed_seq_[:, :len(prefix)]
            elif self.log_type == 'union':
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.mask_token_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0) # 1*l1, 1*l2
        assert self.log_type in ['ftb', 'btf']
        assert self.nll_type in ['ar_ftb', 'ar_btf']

        if self.log_type == 'ftb':
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        else:
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) >= prefix.shape[1]

        if self.log_type == 'ftb':
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous() # l2*l2
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous() # l1*l1

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        perturbed_[mask_index] = self.mask_token_id
        if self.log_type == 'ftb':
            perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)
        else:
            perturbed_seq = torch.cat([perturbed_, target.repeat(perturbed_.shape[0], 1)], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)
        mask_index[temp_index] = False
        if self.log_type == 'ftb':
            logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        else:
            logits_index = torch.cat([mask_index, torch.zeros((perturbed_.shape[1], target.shape[1]), dtype=torch.bool)], dim=-1)

        if self.log_type == 'ftb':
            loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().item()
        else:
            loss = F.cross_entropy(logits[logits_index], prefix[0], reduction='sum').cpu().item()
        return loss

    def _encode_pair(self, context, continuation):
        if self.add_bos_token:
            context = self.tokenizer.bos_token + context
            
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation) + [self.tokenizer.eos_token_id]
        context_enc = self.tokenizer.encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        # by default truncate on the left
        cutoff_length = max(len(whole_enc) - self.max_length, 0)
        if cutoff_length > 0:
            eval_logger.warning(f"Text length {len(whole_enc)} is larger than {self.max_length}, cutoff on the left side")
            context_remain = context_enc_len-cutoff_length
            if context_remain > 0:
                context_enc = context_enc[-context_remain:]
            else:
                eval_logger.warning(f"All context (prompt) is truncated.")
                context_enc = ""
                continuation_enc = whole_enc[-self.max_length:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        print(ds[0])
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                # likelihood calculations are modified from https://github.com/ML-GSAI/SMDM/blob/main/evaluate_diff.py
                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == 'union':
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type == 'ar_ftb' or self.nll_type == 'ar_btf':
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                # TODO: greedy decoding
                is_target_greedy_dec = False

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError


    def _update_block_completion_states(self, block_states, decoded_token_threshold):
        """
        Updates the complete/incomplete state of blocks.
        Iterates through blocks from front to back. If a block's decoded token count exceeds the threshold, the next block to its right (if it exists) is set to a complete state.
        """
        for block_id in sorted(block_states.keys()):
            # if block_id == 0:  # Skip prompt block
            #     continue
            
            # Calculate decoded tokens for the current block
            decoded_tokens = block_states[block_id]['total_masks'] - block_states[block_id]['mask_count']
            decode_ratio = decoded_tokens / block_states[block_id]['total_masks']
            # If current block's decoded token count exceeds the threshold, the next block (if exists) is set to a complete state
            # print("decode_ratio",decode_ratio)
            # print("decoded_token_threshold",decoded_token_threshold)
            if decode_ratio >= decoded_token_threshold:
                next_block_id = block_id + 1
                if next_block_id in block_states:
                    block_states[next_block_id]['is_complete'] = True


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()