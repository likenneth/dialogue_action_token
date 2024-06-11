import sys
sys.path.append('../envs')

from redteam_env import RedTeamEnv, INDICES
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import torch
import glob
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import uuid
import pyrallis
from typing import Optional
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool, cpu_count


TensorBatch = List[torch.Tensor]

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "dat"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    test_baseline: bool = False

    # Environment specific arguments
    env_id: str = "StrongReject"
    """the id of the environment"""
    env_idx: int = -1
    """the id of the environment"""
    state_dim: int = 4096
    """the state dimension"""
    action_dim: int = 64
    """the action dimension"""
    model_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct'
    """the model name of the ControlLLM base model"""
    env_model: str = 'gpt-4'
    """the model name of the judging model"""
    temperature: float = 0.7
    """the temperature of the generation"""
    judge_temp: float = 10.0
    """the temperature of the judge sigmoid"""
    opponent_model: str = ''
    """the model name of the opponent model"""
    prefix_size: int = 2
    """the prefix size of the ControlLLM"""
    max_turns: int = 6
    """the maximum turns of the conversation"""
    prefix_pos: str = 'start'
    """the position of the inserted prefix"""
    model_directory: str = f'redteaming_exp'
    checkpoints_path: str = "runs"
    load_actor: str = ""
    # file name for loading a model, optional
    load_bc: str = ""
    # file name for loading BC model, optional
    # buffer directory
    buffer_dir: str = ""
    # dialog directory
    dialog_directory: str = ""
    """the directory of the model weights"""
    test_baseline: bool = False
    """whether to test the baseline model"""
    test_gpt: bool = False
    """whether to test the GPT model"""
    use_pca: bool = True
    """to use PCA or to use BC'ed model for the upmapping"""

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-5
    """the learning rate of the optimizer"""
    learning_rate_actor: float = 1e-6
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    alpha: float = 0
    """the alpah in TD3-BC"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    act_norm: float = 0.6
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25000
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    def __post_init__(self):
        uu = uuid.uuid4()
        self.name = f"{self.exp_name}-{self.env_id}-{str(uu)[:8]}"
        if self.checkpoints_path != "":
            self.checkpoints_path = os.path.join("runs", self.name)


@torch.no_grad()
def eval_actor(envs, actor, actor_bc = None, env_idx=-1, global_step=-1, add_residual=True):
    actor.eval()
    rewards = []
    if env_idx != -1:
        indices = [env_idx] * (16 if envs.temperature > 0. else 1)
    else:
        if False:
            indices = np.random.randint(0, len(envs.queries), 10)
        else:
            indices = INDICES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in tqdm(indices, disable=False):
        state = envs.reset(i)
        rating_this_query = []
        for j in range(3):
            with torch.no_grad():
                if actor_bc is None:
                    actions = actor(torch.Tensor(state).to(device))
                    prefixes = (actions @ actor.upmapping).repeat(actor.prefix_size, 1)
                else:
                    actions_bc = actor_bc.act(torch.Tensor(state).to(device))
                    actions = actor(torch.Tensor(state).to(device))
                    prefixes = actor_bc.upmapping(actions_bc + actions)
            state, reward, done, _ = envs.step(state, actions, prefixes, save_his=True)
            rating_this_query.append(reward)
        rewards.append(max(rating_this_query))
        print(f"eval/query idx {i}/{len(indices)}: reward {rewards[-1]}")
    actor.train()
    rewards = np.array(rewards)
    return np.mean(rewards).item(), np.max(rewards).item(), np.mean(rewards > 0.5).item()


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            data["observations"] = data["observations"][:self._buffer_size]
            data["actions"] = data["actions"][:self._buffer_size]
            data["rewards"] = data["rewards"][:self._buffer_size]
            data["next_observations"] = data["next_observations"][:self._buffer_size]
            data["terminals"] = data["terminals"][:self._buffer_size]
            n_transitions = self._buffer_size
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add(self, state, next_state, action, reward, done):
        # Use this method to add new data into the replay buffer 
        self._states[self._pointer] = state[0].to(self._device)
        self._actions[self._pointer] = action[0].to(self._device)
        self._rewards[self._pointer] = reward
        self._next_states[self._pointer] = next_state[0].to(self._device)
        self._dones[self._pointer] = done
        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)
        
from ControlLLM import Actor, Actor_BC

# ALGO LOGIC: initialize agent here:
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, mean, std):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, state_dim)
        self.fc2 = nn.Linear(state_dim, 256)
        self.fc3 = nn.Linear(256, 1)
        self.register_buffer(
            "state_scale", std.clone().detach().to(torch.float32)
        )
        self.register_buffer(
            "state_bias", mean.clone().detach().to(torch.float32)
        )

    def forward(self, x, a):
        x = (x - self.state_bias) / self.state_scale
        x = torch.cat([x, a], 1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)  # naturally [-1, 1]
        x = (x + 1) / 2
        return x  # will be MSE'ed to a [0,1] target


def load_pickle(pickle_file):
    with open(pickle_file, "rb") as f:
        pickle_data = pickle.load(f)
    return pickle_data

def load_dataset(buffer_dir: str, env_idx=-1):
    pickles = glob.glob(buffer_dir + "/*.pkl")
    if env_idx != -1:
        pickles = [_ for _ in pickles if (f"env_id={env_idx}_" in _)]
    elif env_idx == -1:
        pickles_new = []
        for idx in INDICES:
            pickles_new.extend([_ for _ in pickles if (f"env_id={idx}_" in _)])
        pickles = pickles_new
    num_processes = min(cpu_count(), len(pickles))

    rewards_list = []
    observations_list = []
    actions_list = []
    next_observations_list = []
    terminals_list = []

    with Pool(processes=num_processes) as pool:
        for pickle_data in tqdm(pool.imap_unordered(load_pickle, pickles), total=len(pickles), desc="Loading dataset"):
            rewards_list.append(pickle_data["rewards"])
            observations_list.append(pickle_data["observations"].squeeze(axis=1))
            actions_list.append(pickle_data["actions"].squeeze(axis=1))
            next_observations_list.append(pickle_data["next_observations"].squeeze(axis=1))
            terminals_list.append(pickle_data["terminals"])
            assert len(pickle_data["rewards"]) == len(pickle_data["observations"]) == len(pickle_data["actions"]) == len(pickle_data["next_observations"]) == len(pickle_data["terminals"])

    rewards = np.concatenate(rewards_list)
    observations = np.concatenate(observations_list)
    actions = np.concatenate(actions_list)
    next_observations = np.concatenate(next_observations_list)
    terminals = np.concatenate(terminals_list)
    dataset = dict(rewards=rewards, observations=observations, actions=actions, next_observations=next_observations, terminals=terminals)
    return dataset

if __name__ == "__main__":

    args = tyro.cli(Args)
    print(args)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{args.name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.dialog_directory == "":
        dialog_directory = args.checkpoints_path
        # dialog_directory = f'{args.model_directory}/breaks_{args.name}'
        # if args.opponent_model != "":
        #     opp_wo_prefix = args.opponent_model.split("/")[-1]
        #     dialog_directory += f"_vs_{opp_wo_prefix}"

        # if args.test_baseline:
        #     dialog_directory += "_baseline"
        # if args.test_gpt:
        #     dialog_directory += "_gpt"
    else:
        dialog_directory = f"{args.model_directory}/{args.dialog_directory}"

    if not os.path.exists(dialog_directory):
        os.makedirs(dialog_directory)

    # env setup
    envs = RedTeamEnv(
        model_name = args.model_name,
        env_model = args.env_model,
        prefix_size = args.prefix_size,
        prefix_embedding_size = args.action_dim,
        temperature = args.temperature,
        judge_temp=args.judge_temp,
        opponent_model = args.opponent_model,
        max_turns = args.max_turns,
        saving_dir = dialog_directory, 
        test_baseline = args.test_baseline,
        mode="train", 
    )


    state_means = torch.load(os.path.join(args.model_directory, 'mean_states.pth'))
    state_std = torch.load(os.path.join(args.model_directory, 'var_states.pth')).sqrt()
    actor_bc = None
    if args.use_pca:
        actor = Actor(args.state_dim, args.action_dim, state_means, state_std, act_norm=args.act_norm, prefix_size=args.prefix_size).to(device)
        if args.load_actor != "":
            checkpoint = torch.load(args.load_actor)
            actor.load_state_dict(checkpoint['actor'])
        target_actor = Actor(args.state_dim, args.action_dim, state_means, state_std, act_norm=args.act_norm, prefix_size=args.prefix_size).to(device)
    else:
        actor_bc = Actor_BC(args.state_dim, args.action_dim, args.prefix_size, args.state_dim).to(device)
        if args.load_bc != "":
            actor_bc.load_state_dict(torch.load(os.path.join(args.model_directory, args.load_bc), map_location=device))
        actor_bc.requires_grad_(False)
        actor = Actor(args.state_dim, args.action_dim, state_means, state_std, act_norm=args.act_norm, prefix_size=args.prefix_size, use_pca=False).to(device)
        target_actor = Actor(args.state_dim, args.action_dim, state_means, state_std, act_norm=args.act_norm, prefix_size=args.prefix_size, use_pca=False).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1 = Critic(args.state_dim, args.action_dim, state_means, state_std).to(device)
    if args.load_actor != "":
        qf1.load_state_dict(checkpoint['critic_1'])
    qf2 = Critic(args.state_dim, args.action_dim, state_means, state_std).to(device)
    if args.load_actor != "":
        qf2.load_state_dict(checkpoint['critic_2'])
    qf1_target = Critic(args.state_dim, args.action_dim, state_means, state_std).to(device)
    qf2_target = Critic(args.state_dim, args.action_dim, state_means, state_std).to(device)

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    if args.load_actor != "":
        q_optimizer.load_state_dict(checkpoint['critic_optimizer'])
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate_actor)  # removing the upmapping matrix
    if args.load_actor != "":
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])

    rb = ReplayBuffer(
        args.state_dim,
        args.action_dim,
        args.buffer_size,
        # device, 
    )

    if args.checkpoints_path != "":
        print(f"Checkpoints path: {args.checkpoints_path}")
        os.makedirs(args.checkpoints_path, exist_ok=True)
        with open(os.path.join(args.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(args, f)

    if args.buffer_dir != "":
        if not os.path.exists(os.path.join(args.model_directory, args.buffer_dir)):
            os.makedirs(os.path.join(args.model_directory, args.buffer_dir))
        if len(glob.glob(os.path.join(args.model_directory, args.buffer_dir) + "/*.pkl")):
            dataset = load_dataset(os.path.join(args.model_directory, args.buffer_dir), env_idx=args.env_idx)
            rb.load_d4rl_dataset(dataset)
    start_step = rb._size
    
    start_time = time.time()

    if args.test_baseline:
        episodic_returns, max_episodic_returns, asr = eval_actor(envs, actor, actor_bc, env_idx=args.env_idx, global_step=0)
        writer.close()
        sys.exit()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset(args.env_idx)
    for i, global_step in tqdm(enumerate(range(start_step, start_step + args.total_timesteps))):
        if args.learning_starts == args.total_timesteps:
            with torch.no_grad():
                if args.use_pca:
                    actions = actor(torch.Tensor(obs).to(device))
                    noises = torch.randn_like(actions)
                    if args.load_actor != "":
                        actions = actions / torch.norm(actions, dim=-1, keepdim=True)
                        actions = actions + noises * args.exploration_noise
                        actions = actions / torch.norm(actions, dim=-1, keepdim=True) * actor.act_clip
                    else:
                        actions = noises
                    actions = actions / torch.norm(actions, dim=-1, keepdim=True) * actor.act_clip
                    prefixes = (actions @ actor.upmapping).repeat(actor.prefix_size, 1)
                else:
                    actions_bc = actor_bc.act(torch.Tensor(obs).to(device))
                    actions = actor(torch.Tensor(obs).to(device))
                    clipped_noise = (torch.randn_like(actions_bc, device=device) * args.exploration_noise).clamp(-args.noise_clip, args.noise_clip)
                    actions = clipped_noise
                    prefixes = actor_bc.upmapping(actions_bc + actions)

            # TRY NOT TO MODIFY: execute the game and log data.
            state, reward, done, _ = envs.step(obs, actions, prefixes, log_time=False)
            writer.add_scalar("Charts/step_return", reward, global_step)

            real_next_obs = state.clone()
            rb.add(obs, real_next_obs, actions, reward, done)
            obs = state if not done else envs.reset(args.env_idx)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            states, actions, rewards, next_states, dones = [torch.Tensor(d).to(device) for d in data]
            with torch.no_grad():
                clipped_noise = (torch.randn_like(actions, device=device) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
                next_state_actions = (target_actor(next_states) + clipped_noise)
                next_state_actions = next_state_actions / torch.norm(next_state_actions, dim=-1, keepdim=True) * actor.act_clip
                qf1_next_target = qf1_target(next_states, next_state_actions)
                qf2_next_target = qf2_target(next_states, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = rewards.flatten() + (1 - dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(states, actions).view(-1)
            qf2_a_values = qf2(states, actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()
            
            writer.add_scalar("Charts/SPS", int((global_step - start_step) / (time.time() - start_time)), global_step)
            if global_step % args.policy_frequency == 0:
                pi = actor(states)
                if 0 == args.alpha:
                    actor_loss = -qf1(states, pi).mean()
                else:
                    q = qf1(states, pi)
                    lmbda = args.alpha / q.abs().mean().detach()
                    actor_loss = -lmbda * q.mean() + F.mse_loss(pi, actions)

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/action_norm", torch.norm(pi, dim=-1).mean().item(), global_step)
                writer.add_scalar("losses/qf1_value_var", qf1_a_values.std().item(), global_step)
                writer.add_scalar("losses/qf2_value_var", qf2_a_values.std().item(), global_step)

            if i % 50 == 0 and global_step >= start_step:
                episodic_returns, max_episodic_returns, asr = eval_actor(envs, actor, actor_bc, env_idx=args.env_idx, global_step=global_step)
                dialog = envs.cur_dialog
                attack1 = dialog[2]["content"]
                attack2 = dialog[4]["content"]
                attack3 = dialog[6]["content"]
                obs = envs.reset(args.env_idx)
                writer.add_scalar("eval/Mean Reward", episodic_returns, global_step)
                writer.add_scalar("eval/Max Reward", max_episodic_returns, global_step)
                writer.add_scalar("eval/ASR", asr, global_step)
                wandb.log({"attack1": wandb.Html(attack1)}, step=global_step)
                wandb.log({"attack2": wandb.Html(attack2)}, step=global_step)
                wandb.log({"attack3": wandb.Html(attack3)}, step=global_step)

            # if i % 50 == 0 and global_step >= start_step:
            #     if args.checkpoints_path != "":
            #         torch.save(
            #             {
            #                 "actor": actor.state_dict(),
            #                 "critic_1": qf1.state_dict(),
            #                 "critic_2": qf2.state_dict(),
            #                 "actor_optimizer": actor_optimizer.state_dict(),
            #                 "critic_optimizer": q_optimizer.state_dict(),
            #                 "total_it": global_step
            #             },
            #             os.path.join(args.checkpoints_path, f"checkpoint_{global_step}.pt"),
            #         )
            #     if asr == 1:
            #         break

    writer.close()