from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from utils import preprocess_no_padding
from ControlLLM import ControlLLM


def load_data(data_src: Path):
    dataset = load_dataset('csv', data_files=str(data_src), split='train')
    return dataset

def main(args):
    """
    Train a Prefix Tuning model to control it to generate expected output.
    """
    experiment_name = f"prefix={args.prefix_length}_dim={args.prefix_embedding_size}_{args.prefix_pos}"
    # experiment_name = f"prefix={args.prefix_length}_dim={args.prefix_embedding_size}_lr={args.lr}_bs={args.batch_size}_accum={args.grad_accum}"

    dataset = load_data(Path(args.dataset_path) / args.dataset)
    if args.debug:
        experiment_name = "debug_" + experiment_name
        dataset = dataset.select(range(args.grad_accum * 10))
        args.num_epochs = 20

    # load model
    model = ControlLLM(args.model_name, args.prefix_length, args.prefix_embedding_size, prefix_pos=args.prefix_pos)
    if args.load_model:
        print("loading model... Epoch: ", args.load_model_epoch)
        model.load_bc_layer(weight_path=f'weights/{experiment_name}/epoch={args.load_model_epoch}_bc.pth')

    # preprocessing
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    processed_dataset = dataset.map(
        preprocess_no_padding,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=['input', 'target'],
        fn_kwargs={'tokenizer': tokenizer},
        desc="Running tokenizer on dataset",
    )
    # initiate data loader
    train_dataloader = DataLoader(
        processed_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True
    )
    eval_dataset = load_data(Path(args.dataset_path) / args.eval_dataset)
    processed_eval_dataset = eval_dataset.map(
        preprocess_no_padding,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=['input', 'target'],
        fn_kwargs={'tokenizer': tokenizer},
        desc="Running tokenizer on dataset",
    )
    eval_dataloader = DataLoader(
        processed_eval_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True
    )

    # select parameters to optimize
    update_parameters = [
        *model.bc_layer.parameters(),
    ]

    optimizer = torch.optim.AdamW(update_parameters, lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(  # ken: maybe you need to take accum step into consideration
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == 'train':
        losses = []
        bleus = []
        blue_epochs = []
        for epoch in range(args.num_epochs):
            # print(f"Start training epoch {epoch}...")
            model.train()
            total_loss = 0
            pbar = tqdm(train_dataloader)
            for step, batch in enumerate(pbar):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                pbar.set_description(f"Train: S{step}E{args.load_model_epoch + epoch + 1}: loss {loss.item():.4f}")
                total_loss += loss.item()
                loss = loss / args.grad_accum
                loss.backward()

                # gradient accumulation
                if (step + 1) % args.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                lr_scheduler.step()
                pbar.update(1)

            # handle the residual loss
            if step % args.grad_accum != args.grad_accum - 1:
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch}: avg loss {avg_loss:.4f}")
            losses.append(avg_loss)

            # save the model
            folder_path = f'weights/{experiment_name}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            model.save_bc_layer(weight_path=f'weights/{experiment_name}/epoch={args.load_model_epoch + epoch + 1}_bc.pth')
            
            # inference after each training epoch
            if (epoch + 1) % args.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    if args.mode == 'train':
                        tot_bleu = 0
                        for _, batch in enumerate(tqdm(train_dataloader)):
                            batch = {k: v.to(device) for k, v in batch.items()}

                            outputs = model(**batch)
                            # decode outputs
                            outputs_ids = outputs.logits.argmax(dim=-1)
                            start_token = torch.argmax((batch['labels'] > -100).int()).item()
                            model_start_token = start_token + args.prefix_length - 1
                            outputs_text = tokenizer.batch_decode(outputs_ids[:, model_start_token:], skip_special_tokens=True)

                            reference = tokenizer.batch_decode(batch['labels'][:, start_token:], skip_special_tokens=True)
                            reference = reference[0].split()
                            
                            bleu_score = sentence_bleu([reference], outputs_text[0].split())
                            tot_bleu += bleu_score
                        avg_bleu = tot_bleu / len(train_dataloader)
                        bleus.append(avg_bleu)
                        blue_epochs.append(epoch)
                        print(f"Epoch {args.load_model_epoch + epoch + 1}: avg blue {avg_bleu:.4f}")    
                    else:
                        for _, batch in enumerate(tqdm(eval_dataloader)):
                            batch = {k: v.to(device) for k, v in batch.items()}

                            outputs = model(**batch)
                            # decode outputs
                            outputs_ids = outputs.logits.argmax(dim=-1)
                            start_token = torch.argmax((batch['labels'] > -100).int()).item()
                            model_start_token = start_token + args.prefix_length - 1
                            outputs_text = tokenizer.batch_decode(outputs_ids[:, model_start_token:], skip_special_tokens=True)

                            print(outputs_text)
                            with open(f"{args.mode}_outputs/prefix={args.prefix_length}.txt", "a") as f:
                                f.write("step: " + str(epoch) + '\n')
                                f.write(outputs_text[0] + '\n')


            # plotting the loss & the textual similarity
            
        plt.plot(losses, label = 'Loss')
        plt.ylabel('Avg. Loss')
        plt.legend(loc='upper left')
        # add another y axis to the right, for BLEU
        plt.twinx()
        plt.plot(blue_epochs, bleus, color='green', label='BLEU')
        # set the limit for the y axis
        plt.ylim(0, 1)
        plt.ylabel('Avg. BLEU')
        plt.xlabel('Epoch')
        plt.title(f'Prefix Length = {args.prefix_length}, Embedding Size = {args.prefix_embedding_size}')
        plt.legend(loc='upper right')
        
        plt.savefig(f'pics/{experiment_name}.png')
        plt.show()
    else:
        total_loss = 0
        tot_bleu = 0
        pbar = tqdm(eval_dataloader)
        with torch.no_grad():
            for _, batch in enumerate(pbar):
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()

                # decode outputs
                outputs_ids = outputs.logits.argmax(dim=-1)
                start_token = torch.argmax((batch['labels'] > -100).int()).item()
                model_start_token = start_token + args.prefix_length - 1
                outputs_text = tokenizer.batch_decode(outputs_ids[:, model_start_token:], skip_special_tokens=True)

                reference = tokenizer.batch_decode(batch['labels'][:, start_token:], skip_special_tokens=True)
                reference = reference[0].split()

                bleu_score = sentence_bleu([reference], outputs_text[0].split())
                tot_bleu += bleu_score
                pbar.set_description(f"Eval: loss {loss.item():.4f} bleu {bleu_score:.4f}")
        avg_bleu = tot_bleu / len(eval_dataloader)
        avg_loss = total_loss / len(eval_dataloader)
        print(f"Avg loss {avg_loss:.4f}")
        print(f"Avg blue {avg_bleu:.4f}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset_path', type=str, default='./bc_target')
    parser.add_argument('--dataset', type=str, default='cleaned_llama2-7b-chat_vs_llama2-7b-chat.csv')
    parser.add_argument('--eval_dataset', type=str, default='cleaned_llama2-7b-chat_vs_llama2-7b-chat.csv')
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--prefix_pos", type=str, default="start", choices=['start', 'mid', 'end'])

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=2)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--load_model_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--prefix_length', type=int, default=2)
    parser.add_argument('--prefix_embedding_size', type=int, default=128)

    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    args.job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    args.task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
    args.task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    print(args)
    # fix random seed
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


    main(args)