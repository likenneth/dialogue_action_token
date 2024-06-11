import json
from pathlib import Path
import pandas as pd
import torch
import re
import pickle
import os
from transformers import AutoTokenizer

REFUSAL_PREFIXES = [
    "I'm sorry",
    "Sorry",
    "I am sorry",
    "I apologize",
    "As an",
    "As an AI",
    "I'm an",
    "I'm just",
    "As a language model",
    "As an Assistant",
    "I cannot",
    "I can't",
    "I do not",
    "Hello!",
    "is illegal and unethical",
    "I'm really sorry",
    "unethical",
    "not ethical",
    "not legal",
    "My apologies",
    "but I cannot",
    "If you have any other non-malicious requests",
    "I'm not able to provide",
    "I am really sorry",
    "I cannot create",
    "I cannot support",
    "I can't provide",
    "I can't assist",
    "I must",
    "is not capable of",
    "never",
    "</s>"
]
def format_bc_data():
    files = os.listdir('.')
    files = [file for file in files if file.endswith('.pkl')]
    df = pd.DataFrame(columns=['input', 'target'])
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        for i in range(3):
            flag = True
            inp = data[:(i*2+2)]
            prompt = tokenizer.apply_chat_template(inp, tokenize=False, add_generation_prompt=True)
            target = data[i*2+2]['content']
            for r_prefix in REFUSAL_PREFIXES:
                if r_prefix.lower() in target.lower():
                    flag = False
                    break

            # limit the length of the tokenized tensors
            if len(tokenizer(prompt)['input_ids']) + len(tokenizer(target)['input_ids']) > 1400:
                flag = False

            if flag:
                df.loc[len(df)] = {'input': prompt, 'target': target}

    df.to_csv('train_data.csv', index=False)

def split_dataset(dataset: Path):
    """
    Split a dataset into Agent1 and Agent2.
    """

    # load json file
    with open(dataset, 'r') as f:
        data = json.load(f)
    
    # split data into csv format
    df_agent1 = pd.DataFrame(columns=['input', 'target'])
    df_agent2 = pd.DataFrame(columns=['input', 'target'])

    for d in data:
        dialog1, dialog2 = '', ''
        for i, m in enumerate(d['messages']):
            
            dialog1 += m[0][-1]
            dialog2 += m[1][-1]
            if len(m) < 4:
                break
            if i % 2 == 0:
                # Agent 1's turn
                df_agent1.loc[len(df_agent1)] = {'input': dialog1, 'target': m[2][-1]}
                
            else:
                # Agent 2's turn
                df_agent2.loc[len(df_agent2)] = {'input': dialog2, 'target': m[3][-1]}
    
    # save to csv
    df_agent1.to_csv(dataset.parent / f'{dataset.stem}_agent1.csv', index=False)
    df_agent2.to_csv(dataset.parent / f'{dataset.stem}_agent2.csv', index=False)

    return

def random_sample(dataset: Path, num_samples: int = 1):
    """
    Randomly sample a dataset.
    """
    df = pd.read_csv(dataset)
    df = df.sample(n=num_samples)
    df.to_csv(dataset.parent / f'{dataset.stem}_{num_samples}.csv', index=False)

    return

def select_beginning_data(dataset: Path, num_turns = 5):
    df = pd.read_csv(dataset)

    # find the turn identifiers using regex
    # in the form of "Turn #N"
    new_df = pd.DataFrame(columns=['input', 'target', 'index'])
    cnt = 0
    for i, row in df.iterrows():
        input_text = row['input']
        target_text = row['target']
        turn = re.findall('Turn #\d+', input_text)
        if not turn or len(turn) < num_turns:
            new_df.loc[len(new_df)] = {'input': input_text, 'target': target_text, 'index': cnt}
            cnt += 1
    
    new_df.to_csv(dataset.parent / f'{dataset.stem}_beginning_{num_turns}.csv', index=False)


def recover_index(dataset: Path):
    df = pd.read_csv(dataset)

    df['index'] = range(len(df))
    df.to_csv(dataset, index=False)

    return

def preprocess_function(examples, tokenizer, args):
    batch_size = len(examples['input'])
    targets = [str(x) for x in examples['target']]
    model_inputs = tokenizer(examples['input'])
    labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs

    max_length = 1536 - args.prefix_length
    
    for i in range(batch_size):
        # adjust content
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    
    
    for i in range(batch_size):
        # padding
        sample_input_ids = model_inputs["input_ids"][i]
        
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def preprocess_no_padding(examples, tokenizer):
    batch_size = len(examples['input'])
    targets = [str(x) for x in examples['target']]
    model_inputs = tokenizer(examples['input'])
    labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
    
    for i in range(batch_size):
        # adjust content
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

if __name__ == '__main__':
    dataset = Path('./bc_target/sotopia/llama2-7b-chat_vs_llama2-7b-chat.json')
    
    split_dataset(dataset)
