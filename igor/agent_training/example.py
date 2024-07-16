import sys

import gymnasium
import torch
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.make_env import BatchedVecEnv
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size

sys.path.append(".")

from ariande.agent_training.env.create_env import PatchedCrafterEnv
from ariande.agent_training.sequential_task_wrapper import SequentialTasksWrapper
from ariande.agent_training.main import parse_custom_args
from tabulate import tabulate


class Inference:
    def __init__(self):
        self.device = 'cuda'

        if self.device != 'cpu' and not torch.cuda.is_available():
            self.device = torch.device('cpu')
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

        # Loading config
        sys.argv.extend(['--experiment', 'scaled_reward'])
        self.cfg = parse_custom_args(evaluation=True)

        # Loading model
        env = PatchedCrafterEnv()
        actor_critic = create_actor_critic(self.cfg, env.observation_space, env.action_space)
        actor_critic.eval()
        actor_critic.model_to_device(self.device)
        policy_id = self.cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(self.cfg, policy_id), f"{name_prefix}_*")
        checkpoint_dict = Learner.load_checkpoint(checkpoints, self.device)
        actor_critic.load_state_dict(checkpoint_dict["model"])
        self.model = actor_critic

        # Initializing RNN state
        self.rnn_states = torch.zeros([env.num_agents, get_rnn_size(self.cfg)], dtype=torch.float32, device=self.device)

    def act(self, obs):
        with torch.no_grad():
            normalized_obs = prepare_and_normalize_obs(self.model, obs)
            policy_outputs = self.model(normalized_obs, self.rnn_states)

        self.rnn_states = policy_outputs['new_rnn_states']
        return policy_outputs['actions'].cpu().numpy()

    def reset_states(self):
        self.rnn_states.fill_(0)


def single_task():
    # Examples: collect_wood, collect_drink, collect_stone etc
    env = PatchedCrafterEnv(custom_task='collect_wood')
    env = BatchedVecEnv(env)

    algo = Inference()

    obs, info = env.reset()
    while True:
        obs, rew, terminated, truncated, infos = env.step(algo.act(obs))
        print(infos[0]['inventory'])
        if all(terminated) or all(truncated):
            break

    # Zeroing RNN hidden
    algo.reset_states()


class SupressTaskTerminationWrapper(gymnasium.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        dead = info['discount'] < 1
        if terminated and not dead:
            info['task_completed'] = True
            terminated = False
        else:
            terminated = dead

        return obs, reward, terminated, truncated, info


def sequential_tasks(list_of_tasks=None):
    if list_of_tasks is None:
        list_of_tasks = ['collect_wood'] + ['collect_stone'] * 10 + ['collect_wood'] * 5 + ['defeat_zombie']

    env = PatchedCrafterEnv()
    env = SupressTaskTerminationWrapper(env)
    env = BatchedVecEnv(env)

    obs, info = env.reset()
    algo = Inference()

    # Initialize a list to store all task results
    tasks_results = []

    for task in list_of_tasks:
        task_result = {'Task': task, 'Status': '', 'Achievements': [], 'Steps': 0}

        # Setting task
        env.set_task(task)

        # Zeroing RNN hidden
        algo.reset_states()

        while True:
            obs, rew, terminated, truncated, infos = env.step(algo.act(obs))
            task_result['Steps'] += 1
            if all(terminated) or all(truncated) or infos[0].get('task_completed'):
                break

        info = infos[0]

        if info['inventory']['health'] <= 0:
            task_result['Status'] = 'Agent is dead'
        elif all(truncated):
            task_result['Status'] = 'Truncated by time limit'
        elif info.get('task_completed'):
            task_result['Status'] = 'Completed'

        # Storing achievements
        for achievement, count in info['achievements'].items():
            if count > 0:
                task_result['Achievements'].append(f"{achievement}: {count}")

        tasks_results.append(task_result)

        if all(terminated) or all(truncated):
            break

    # Formatting the final results table
    final_table = []
    for result in tasks_results:
        final_table.append([result['Task'], '\n'.join(result['Achievements']), result['Steps'], result['Status']])

    print(tabulate(final_table, headers=['Task', 'Achievements', 'Steps', 'Status', ], tablefmt='grid'))


def sequential_tasks_v2():
    list_of_tasks = ['collect_wood'] + ['collect_stone'] * 10 + ['collect_wood'] * 5 + ['defeat_zombie']

    env = PatchedCrafterEnv()
    env = SequentialTasksWrapper(env, list_of_tasks=list_of_tasks)
    env = BatchedVecEnv(env)

    algo = Inference()

    obs, info = env.reset()
    while True:
        obs, rew, terminated, truncated, infos = env.step(algo.act(obs))
        print(infos[0]['inventory'])
        if all(terminated) or all(truncated):
            break
    # Zeroing RNN hidden
    algo.reset_states()


if __name__ == '__main__':
    sequential_tasks_v2()