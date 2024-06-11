import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Literal
import pickle


class Actor_BC(nn.Module):
    def __init__(self, fan_in, fan_out, prefix_size: int = 2, hidden_dim: int = 4096):
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.fc1 = nn.Linear(fan_in, fan_in)
        self.fc2 = nn.Linear(fan_in, fan_out)
        self.fc3 = nn.Linear(fan_out, fan_out)

        self.upmapping = nn.Linear(fan_out, prefix_size * hidden_dim) # dimension of hidden state

    def act(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        return x

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = self.upmapping(x)
        return x

class Actor(nn.Module):
    def __init__(self, fan_in, fan_out, mean, std, upmapping=None, act_norm=.6, prefix_size=2, use_pca=True, hidden_dim=4096):
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.fc1 = nn.Linear(fan_in, fan_in)
        self.fc2 = nn.Linear(fan_in, fan_in)
        self.fc3 = nn.Linear(fan_in, fan_out)
        self.register_buffer(
            "state_scale", std.clone().detach().to(torch.float32)
        )
        self.register_buffer(
            "state_bias", mean.clone().detach().to(torch.float32)
        )
        if use_pca:
            # load redteaming_exp/llama3_8B_embed_pcas.pkl
            pcas =  pickle.load(open("redteaming_exp/llama3_8B_embed_pcas.pkl", "rb"))
            self.register_buffer(
                "upmapping", torch.tensor(pcas).to(torch.float32)[:fan_out]
            )
        self.act_clip = act_norm
        self.prefix_size = prefix_size

    def forward(self, x):
        x = (x - self.state_bias) / self.state_scale
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        x = x / torch.norm(x, dim=-1, keepdim=True) * self.act_clip
        return x
    

class ControlLLM(nn.Module):
    def __init__(self, model_name: str, prefix_size: int = 8, prefix_embedding_size: int = 64, prefix_pos: Literal ['start', 'mid', 'end'] = 'start'):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        # self.base_model = torch.compile(base_model, mode="reduce-overhead", fullgraph=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.prefix_pos = prefix_pos
        # freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # behavior cloning layer
        # transform the last hidden state to the prefix_embedding_size
        self.prefix_embedding_size = prefix_embedding_size
        self.prefix_size = prefix_size
        self.bc_layer = Actor_BC(self.base_model.config.hidden_size, prefix_embedding_size).to(self.base_model.device)

    def embed_action(self, action, input_ids):
        # project prefix to the same dimension as the input
        p = action.view(-1, self.prefix_size, self.base_model.config.hidden_size)

        input_embeddings = self.base_model.get_input_embeddings()
        embeddings = input_embeddings(input_ids)

        # concatenate prefix and input embeddings
        # skip padding tokens
        # find the first non-padding token in input_ids for all samples in the batch
        new_embeddings = torch.empty((input_ids.shape[0], input_ids.shape[1] + self.prefix_size, self.base_model.config.hidden_size), device=input_ids.device)
        if self.prefix_pos == 'start':
            # put the prefix tokens at the start of the input, just after the bos token
            for i in range(input_ids.shape[0]):
                new_embeddings[i] = torch.cat((embeddings[i, 0, :].unsqueeze(0), p[i], embeddings[i, 1:, :]), dim=0)

        elif self.prefix_pos == 'mid':
            # put the prefix tokens between instruction and conversation history
            mid_tokens = "\nConversation Starts"    # this is the token that separates instruction and conversation history
            for i in range(input_ids.shape[0]):
                # get input_ids for "Conversation History"
                target_input_ids = self.tokenizer.encode(mid_tokens, add_special_tokens=False, return_tensors='pt')[0, 1:].to(self.base_model.device)
                # find target input_ids in input_ids
                indices = (input_ids[i] == target_input_ids[0]).nonzero()
                for idx in indices:
                    if torch.equal(input_ids[i, idx:idx+len(target_input_ids)], target_input_ids):
                        # found the place to insert the prefix
                        new_embeddings[i] = torch.cat((embeddings[i, :idx, :], p[i], embeddings[i, idx:, :]), dim=0)
                        break

        else:
            # prepend the prefix tokens to the end of the input
            new_embeddings = torch.cat((embeddings, p), dim=1)
        
        return new_embeddings


    def _handle_prefix(self, input_ids, attention_mask):
        output = self.base_model(
            attention_mask=attention_mask,
            input_ids=input_ids,
            output_hidden_states=True,
        )
    
        # get the hidden state of the last layer of the last token of the input
        hidden_states = output.hidden_states
        last_hidden_state = hidden_states[-1][:, -1, :].to(torch.float32)

        # behavior cloning
        p = self.bc_layer(last_hidden_state)

        new_embeddings = self.embed_action(p, input_ids)

        new_embeddings = new_embeddings.to(dtype=torch.bfloat16)
        return new_embeddings

    def forward(self, input_ids, attention_mask, labels = None, **kwargs):
        new_embeddings = self._handle_prefix(input_ids, attention_mask)        
        first_non_padding = torch.ne(input_ids, self.tokenizer.pad_token_id).to(torch.long).argmax(dim=1)
        new_attention_masks = torch.empty((input_ids.shape[0], input_ids.shape[1] + self.prefix_size)).to(attention_mask.device)
        if labels is not None:
            new_labels = torch.empty((input_ids.shape[0], input_ids.shape[1] + self.prefix_size)).to(labels.device)
        for i in range(input_ids.shape[0]):
            new_attention_masks[i] = torch.cat((attention_mask[i, :first_non_padding[i]], torch.ones((self.prefix_size)).to(attention_mask.device), attention_mask[i, first_non_padding[i]:]), dim=0)
            if labels is not None:
                new_labels[i] = torch.cat((torch.full((self.prefix_size, ), -100).to(labels.device), labels[i]), dim=0)

        # convert to float16
        new_attention_masks = new_attention_masks.to(torch.bfloat16)
        if labels is not None:
            new_labels = new_labels.to(torch.long)
            
        return self.base_model(
            attention_mask=new_attention_masks,
            labels=new_labels,
            inputs_embeds=new_embeddings,
            output_hidden_states=True,
            **kwargs
        )
    
    def generate(self, input_ids, attention_mask, **kwargs):
        new_embeddings = self._handle_prefix(input_ids, attention_mask)
        
        return self.base_model.generate(
            inputs_embeds=new_embeddings,
            **kwargs
        )
    
    def load_bc_layer(self, weight_path):
        self.bc_layer.load_state_dict(torch.load(weight_path))

    def save_bc_layer(self, weight_path):
        torch.save(self.bc_layer.state_dict(), weight_path)