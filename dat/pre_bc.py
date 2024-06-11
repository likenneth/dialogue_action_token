import sys
sys.path.insert(0, '../envs')

import os
from sot_env import SotopiaEnv
from redteam_env import RedTeamEnv
import torch
import random
import numpy as np
from tqdm import tqdm
import json
import torch.nn as nn
from ControlLLM import Actor_BC
import argparse
from transformers import set_seed

@torch.no_grad()
def eval_actor(envs, actor, quick=False):
    actor.eval()
    old_temperature = envs.temperature
    old_max_new_tokens = envs.max_new_tokens
    envs.max_new_tokens = 512
    rewards = []
    if quick:
        indices = np.random.randint(0, len(envs.queries), 10)
    else:
        indices = range(len(envs.queries))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # states = []
    for i in tqdm(indices):
        state = envs.reset(i)
        rating_this_query = []
        # states.append(state)
        for j in range(3):
            actions = actor(torch.Tensor(state).to(device))
            state, reward, done, _ = envs.step(state, actions, actions, save_his=True)
            # states.append(state)
            rating_this_query.append(reward)
        rewards.append(1 if max(rating_this_query) > 0.5 else 0)
        print(f"Query {i}: {rewards[-1]}")
    actor.train()
    envs.temperature = old_temperature
    envs.max_new_tokens = old_max_new_tokens
    return np.mean(rewards).item()


def main(args):
    prefix_size = args.prefix_size
    prefix_embedding_size = args.prefix_embedding_size
    model_directory = os.path.join(args.directory, f'prefix={prefix_size}_dim={prefix_embedding_size}_{args.prefix_pos}')
    
    if args.actor_path is None:
        actor_wo_suffix = f"epoch={args.epoch}_bc"
    else:
        actor_wo_suffix = args.actor_path.split(".")[0]

    actor = Actor_BC(fan_in=4096, fan_out=prefix_embedding_size).to('cuda')
    if not args.test_baseline and not args.test_gpt:
        actor.load_state_dict(torch.load(f'{model_directory}/{actor_wo_suffix}.pth'))

    dialog_directory = f'{model_directory}/dialogs_noise{args.exploration_noise:.2f}_{actor_wo_suffix}'
    if args.opponent_model != "":
        dialog_directory += f"_vs_{args.opponent_model}"

    if args.test_baseline:
        dialog_directory += "_baseline"
    if args.test_gpt:
        dialog_directory += "_gpt"
    if not os.path.exists(dialog_directory):
        os.makedirs(dialog_directory)
    if args.env == 'sotopia':
        env = SotopiaEnv(
            model_name = 'meta-llama/Llama-2-7b-chat-hf',
            env_model = 'gpt-4',
            prefix_size = prefix_size,
            prefix_embedding_size = prefix_embedding_size, 
            temperature=args.temperature,
            test_baseline=args.test_baseline,
            test_gpt=args.test_gpt,
            saving_dir = dialog_directory,
            prefix_pos=args.prefix_pos, 
            max_turns=args.max_turns, 
        )
        # print(env.get_state())
        # print(env.get_current_prompt())
        # print(env.get_current_input_tensors())

        reward_container = []

        if args.load_old:
            for file in os.listdir(dialog_directory):
                if file.endswith(".txt"):
                    file_path = os.path.join(dialog_directory, file)
                    with open(file_path, "r") as f:
                        string = f.read()
                        dialog = json.loads(string)
                        reward_container.append(dialog["reward"][0])
            print(f"loaded {len(reward_container)} dialogs from previous runs...")

        bar = tqdm(range(args.start_seed, args.start_seed+args.runs))
        for seed in bar:
            set_seed(seed)
            if args.env_idx == -1:
                env_id = random.randint(0, 449)
            else:
                env_id = args.env_idx
            if args.actor_role == -1:
                actor_role = random.randint(1, 2)
            else:
                actor_role = args.actor_role

            state = env.reset(env_id = env_id, actor_role = f"agent{actor_role}")
            print(f"evaluating dialog {seed} with env_id {env_id} and actor_role {actor_role}...")

            done = False
            success = True
            while not done:
                state = state.to('cuda')
                with torch.no_grad():
                    action = actor(state).to('cuda')
                    if args.exploration_noise != 0:
                        noise = np.random.normal(0, args.exploration_noise, action.shape)
                        action += torch.tensor(noise).to('cuda')
                    try:
                        state, reward, done, _ = env.step(state, action)
                    except:
                        success = False
                        break
                    print(env.get_current_prompt())
                    print(reward)
            if not success:
                continue
            reward_container.append(reward)
            env.save_conversation_history(seed)
            bar.set_description(f"Mean reward: {np.mean(reward_container):.2f} | Std reward: {np.std(reward_container):.2f}")
    else:
        env = RedTeamEnv(
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
            prefix_size = args.prefix_size,
            prefix_embedding_size = args.prefix_embedding_size,
            temperature = args.temperature,
            opponent_model = args.opponent_model,
            max_turns = 6,
            max_new_tokens=512,
            saving_dir = dialog_directory, 
            test_baseline=args.test_baseline,
            mode="train", 
        )
        
        result = eval_actor(env, actor, quick=False)
        print(f"ASR: {result}")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='sotopia', choices=['sotopia', 'redteam'], help='Environment to run the dialog')
    parser.add_argument('--directory', type=str, default='weights', help='Directory to save the weights')
    parser.add_argument('--prefix_size', type=int, default=2, help='Size of the prefix')
    parser.add_argument('--prefix_embedding_size', type=int, default=64, help='Size of the prefix embedding')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling')
    parser.add_argument('--env_idx', type=int, default=-1, help='env index for the dialog')
    parser.add_argument('--actor_role', type=int, default=-1, help='role of BC in the dialog')
    parser.add_argument('--start_seed', type=int, default=1, help='run from this seed to seed+runs')
    parser.add_argument('--runs', type=int, default=1, help='run it many times')
    parser.add_argument('--max_turns', type=int, default=20, help='run it many times')
    parser.add_argument('--epoch', type=int, default=50, help='run it many times')
    parser.add_argument('--actor_path', type=str, default=None, help='Name of the actor weights file')
    parser.add_argument('--load_old', action='store_true', help='load old dialogs')
    parser.add_argument('--exploration_noise', type=float, default=0., help='use seed to add noise to the action')
    parser.add_argument('--test_baseline', action='store_true', help='disable actor in the env')
    parser.add_argument('--test_gpt', action='store_true', help='use gpt-4 as the actor')
    parser.add_argument('--opponent_model', type=str, default="", help="opponent model model card")
    parser.add_argument("--prefix_pos", type=str, default="start", choices=['start', 'mid', 'end'])
    parser.add_argument("--use_pca", action='store_true', help='use pca to reduce the dimension of the prefix')
    args = parser.parse_args()
    main(args)

# python evaluate_bc.py --prefix_embedding_size 8 --prefix_size 2 --runs 15