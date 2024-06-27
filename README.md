# Dialogue Action Tokens

This repository provides the code for the paper [Dialogue Action Tokens: Steering Language Models in Goal-Directed Dialogue with a Multi-Turn Planner](https://arxiv.org). It shows how to apply Dialogue Action Tokens (DAT) to a LLaMA3-8B model to strengthen goal-directed dialogue capabilities. 

Part of this project builds on top of [Sotopia](https://www.sotopia.world), [HarmBench](https://github.com/centerforaisafety/HarmBench) and [CleanRL](https://github.com/vwxyzjn/cleanrl).


## Abstract

> We present an approach called Dialogue Action Tokens (DAT) that adapts language model agents to plan goal-directed dialogues. The core idea is to treat each utterance as an action, thereby converting dialogues into games where existing approaches such as reinforcement learning can be applied. Specifically, we freeze a pretrained language model and train a small planner model that predicts a continuous action vector, used for controlled generation in each round. This design avoids the problem of language degradation under reward optimization. When evaluated on the Sotopia platform for social simulations, the DAT-steered LLaMA model surpasses GPT-4's performance. We also apply DAT to steer an attacker language model in a novel multi-turn red-teaming setting, revealing a potential new attack surface.

## Table of Contents
1. [Installation](#installation)
2. [Workflow](#workflow)
3. [An Example](#an-example)
4. [How to Cite](#how-to-cite)

## Installation

In this the root folder of this repo, run the following commands to set things up. 

This code base supports only sinlge-GPU experiments for now. Basically the GPU needs to load the dialogue model and potentially a judge model. It's tested that one A100 (40G) GPU is enough. 

```bash
conda create -n dat python=3.11
conda activate dat
pip install -r requirements.txt
pip install -r requirements_rl.txt
```

## Workflow

### Step1. Self-Cloning

First we need to generate training set for self-cloning.

```bash
cd dat
python pre_bc.py --env {sotopia,redteam} --runs 50 --epoch 1 --prefix_size 2 --prefix_embedding_size 64 --start_seed 1 --test_baseline --max_turns 6
```
<!-- *(TODO: transform the pickle files into csv format)* -->

We provide example dialog histories for behavior cloning [here](dat/bc_target/) so that you can skip this step and carry out the training as below. 

```bash
# Social Capability
python bc.py --dataset_path ./bc_target/sotopia --dataset cleaned_llama2-7b-chat_vs_llama2-7b-chat.csv --model_name meta-llama/Llama-2-7b-chat-hf --eval_dataset cleaned_llama2-7b-chat_vs_llama2-7b-chat.csv --prefix_embedding_size 64 --prefix_length 2 --prefix_pos start --num_epochs 100 --eval_every 10

# Red Teaming
python bc.py --dataset_path ./bc_target/harmbench --dataset train_data_small.csv --model_name meta-llama/Meta-Llama-3-8B-Instruct --eval_dataset train_data_small.csv --num_epochs 100 --eval_every 10
```

Meanwhile, the self-cloning step can be skipped by running a PCA over embedding matrix of the model with [this file](dat/calculate_upmapping.py). 

### Step2. Reinforcement Learning

First we need to generate episodes for offline RL training, run the following script with different `--seed` to collect the buffer.
```bash
cd dat
IDX=0
python td3.py --temperature 0. --learning_starts 1000000 --act_norm 1. --prefix_size 2 --action_dim 64 --env_idx $IDX --dialog_directory buffer_env${IDX}
```

Then we can start RL training for `IDX` from 0 to 158, the ASR will be logged in weight and biases.

```bash
python td3.py --buffer_dir buffer_env${IDX} --batch_size 1024 --learning_starts 0 --env_idx $IDX --temperature 0.7 --act_norm 1. --prefix_size 2 --action_dim 64 --total_timesteps 750 --track --wandb_entity <your_username>
```

## An Example

You can download pre-collected data from [here](https://drive.google.com/file/d/1m_TvCqssUye6kCyqNdKQwcYaydLtPjBr/view?usp=sharing) (1.76G, compressed), put it into `redteaming_exp`, and run

```python
python td3.py --alpha 0. --seed 43 --batch_size 1024 --learning_starts 0 --env_idx 0 --temperature 0.7 --act_norm 1. --prefix_size 2 --action_dim 128 --total_timesteps 500 --use_pca --buffer_dir buffer_ps2_ad128_env0 --buffer_size 80000 --track --wandb_entity <your_username>
```

Weight-and-bias logs can be found [here](https://api.wandb.ai/links/keli/hmdlsn3g).

## How to Cite

```
@article{li2024dialogue,
  title={Dialogue Action Tokens: Steering Language Models in Goal-Directed Dialogue with a Multi-Turn Planner},
  author={Li, Kenneth and Wang, Yiming and Vi{\'e}gas, Fernanda and Wattenberg, Martin},
  journal={arXiv preprint arXiv:2406.11978},
  year={2024}
}
```
