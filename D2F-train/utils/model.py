import transformers
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig,get_peft_model
from model.modeling_llada import LLaDAModelLM
from model.configuration_llada import LLaDAConfig

def get_model_by_config(config):
    """Select different models based on config file"""
    training_mode = config.get('training_mode', 'dream')
    
    if training_mode == 'llada':
        return get_llada(config)
    elif training_mode == 'dream':
        return get_model(config)
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

def get_model(config):
    # Use path from config, use default path if no config
    model_path = config.paths.model if hasattr(config, 'paths') and hasattr(config.paths, 'model') else "/home/wx/data/model/Dream-org/Dream-v0-Base-7B"
    
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    # print(model.named_modules())
    # print(model,"model
    for param in model.parameters():
        param.requires_grad = False
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.1,target_modules=["q_proj", "v_proj","k_proj", "o_proj"],)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer

import torch
import torch.nn as nn
class RegisterEmbedding(nn.Module):
    """
    original mask id: 126336
    register token id: 126464
    """
    def __init__(self, wte, mask_token_id, pred_token_id):
        super(RegisterEmbedding, self).__init__()

        # keep hardcoded for now
        self.embedding_dim = wte.weight.shape[-1]
        self.mask_token_id = mask_token_id
        self.pred_token_id = pred_token_id

        self.wte = wte
        self.reg_modifier_table = nn.Embedding(num_embeddings=2, embedding_dim=self.embedding_dim)
        self.pred_token = nn.Embedding(num_embeddings=1, embedding_dim=self.embedding_dim)

    def forward(self, x):
        is_mask = (x == self.mask_token_id).int()
        is_pred = (x == self.pred_token_id)
    
        x_no_pred = torch.where(is_pred, self.mask_token_id, x)

        embedding = self.wte(x_no_pred) 

        embedding = embedding + self.reg_modifier_table(is_mask)

        prediction = self.pred_token(torch.zeros_like(x))

        embedding = torch.where(is_pred.unsqueeze(-1), prediction, embedding)

        return embedding

def patch_embedding(transformer: nn.Module):
    wte = transformer.wte
    remb = RegisterEmbedding(wte, 126336, 126464)

    transformer.wte = remb

    with torch.no_grad():
        remb.reg_modifier_table.weight.zero_()

        orig_mask_weight = wte.weight[126336, :].unsqueeze(0).clone()

        remb.pred_token.weight = nn.Parameter(orig_mask_weight + torch.randn_like(orig_mask_weight))

        remb.reg_modifier_table.weight.requires_grad = True
        remb.pred_token.weight.requires_grad = True
        # may re-enable grad calculation, we don't want that
        remb.wte.weight.requires_grad = False

def get_llada(config):
    # Use path from config, use default path if no config
    model_path = config.paths.model if hasattr(config, 'paths') and hasattr(config.paths, 'model') else "/data1/xck/models/llada-8b-instruct"
    
    config_obj=LLaDAConfig.from_pretrained(model_path)
    model = LLaDAModelLM.from_pretrained(model_path,config=config_obj)


    print([name for name, _ in model.named_modules()])
    print(model)
    patch_embedding(model.model.transformer)

    # print(model,"model
    # print(model)
    # exit()
    for param in model.parameters():
        param.requires_grad = False
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # unlike base D2F, we also train FFN layer
    peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.1,target_modules=["q_proj", "v_proj","k_proj", "attn_out", "ff_proj", "ff_out", "up_proj"], modules_to_save=["wte"])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer
# def create_attention_mask(input_ids, mask_id):
#     """
#     Create an attention mask based on the input_ids and mask_id.

#     Args:
#         input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
#         mask_id (int): The ID of the mask token.

#     Returns:
#         torch.Tensor: The attention mask of shape (batch_size, sequence_length, sequence_length).
