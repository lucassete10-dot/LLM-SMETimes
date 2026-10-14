import torch
import torch.nn as nn
from transformers import (
    OPTForCausalLM,
    AutoTokenizer
)
from utils.device import model_dtype, resolve_device
from utils.llm import resolve_llm_ckp_dir

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = resolve_device(configs)
        self.llm_ckp_dir = resolve_llm_ckp_dir(configs.llm_ckp_dir)
        print(self.device)

        self.opt = OPTForCausalLM.from_pretrained(
            self.llm_ckp_dir,
            dtype=model_dtype(configs, self.device),
            local_files_only=getattr(configs, "local_files_only", False),
        ).to(self.device)

        self.opt_tokenizer = AutoTokenizer.from_pretrained(
            self.llm_ckp_dir,
            local_files_only=getattr(configs, "local_files_only", False),
        )
        if self.opt_tokenizer.pad_token is None:
            self.opt_tokenizer.pad_token = self.opt_tokenizer.eos_token
        self.max_length = int(getattr(configs, "max_token_length", 256))

        self.vocab_size = self.opt_tokenizer.vocab_size
        self.hidden_dim_of_opt = self.opt.config.hidden_size

        for name, param in self.opt.named_parameters():
            param.requires_grad = False
        self.opt.eval()

    def tokenizer(self, text_list):
        texts = [text_list] if isinstance(text_list, str) else list(text_list)
        tokenized_output = self.opt_tokenizer(
            texts,
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.max_length,
        )
        
        input_ids = tokenized_output["input_ids"].to(self.device)
        attention_mask = tokenized_output["attention_mask"].to(self.device)
        embeddings = self.opt.get_input_embeddings()(input_ids)
        
        return embeddings, attention_mask
    
    def forecast(self, text_list):
        inputs_embeds, attention_mask = self.tokenizer(text_list)
        
        outputs = self.opt.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        text_outputs = outputs.last_hidden_state
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(text_outputs.shape[0], device=self.device)
        return text_outputs[batch_idx, last_token_idx, :]
    
    def forward(self, text_list):
        return self.forecast(text_list)
