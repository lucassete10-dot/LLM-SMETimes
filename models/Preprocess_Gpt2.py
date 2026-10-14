import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, AutoTokenizer
from utils.device import model_dtype, resolve_device
from utils.llm import resolve_llm_ckp_dir

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = resolve_device(configs)
        self.llm_ckp_dir = resolve_llm_ckp_dir(configs.llm_ckp_dir)
        print(self.device)
        
        self.gpt2 = GPT2Model.from_pretrained(
            self.llm_ckp_dir,
            dtype=model_dtype(configs, self.device),
            local_files_only=getattr(configs, "local_files_only", False),
        ).to(self.device)
        
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained(
            self.llm_ckp_dir,
            local_files_only=getattr(configs, "local_files_only", False),
        )
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.max_length = int(getattr(configs, "max_token_length", 256))
        self.vocab_size = self.gpt2_tokenizer.vocab_size
        self.hidden_dim_of_gpt2 = self.gpt2.config.hidden_size

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False
        self.gpt2.eval()

        self.encoder = nn.Linear(self.hidden_dim_of_gpt2, self.hidden_dim_of_gpt2)
        self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.vocab_size)

    def tokenizer(self, x):
        texts = [x] if isinstance(x, str) else list(x)
        tokens = self.gpt2_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True
        )
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        embeddings = self.gpt2.get_input_embeddings()(input_ids)
        return embeddings, attention_mask

    def forecast(self, x_mark_enc):
        embeddings, attention_mask = self.tokenizer(x_mark_enc)
        outputs = self.gpt2(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        ).last_hidden_state
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(outputs.shape[0], device=self.device)
        return outputs[batch_idx, last_token_idx, :]

    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)
