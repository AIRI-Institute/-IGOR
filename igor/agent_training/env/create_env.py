import random

import crafter
import gymnasium
import numpy as np
from crafter import Env as CrafterEnv
from crafter import constants as crafter_constants


class PatchedCrafterEnv(gymnasium.Env):
    achievements_list = [
        'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron', 'collect_sapling', 'collect_stone',
        'collect_wood', 'defeat_skeleton', 'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
        'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword', 'make_wood_pickaxe', 'make_wood_sword',
        'place_furnace', 'place_plant', 'place_stone', 'place_table', 'wake_up',
    ]

    def __init__(self, custom_task=None, recorder_dir=None):
        self._inner_env = CrafterEnv()
        if recorder_dir:
            # VideoRecorder should be fixed, to work with this patched environment
            crafter.Recorder(self._inner_env, recorder_dir, save_video=True)
        self.num_agents = 1
        self._size = (64, 64)
        self._task_vector_size = 32
        self._subtask_extra_reward = 1.0
        self._main_task_reward_divisor = 10.0

        self._current_achievement_task = None
        self._previous_achievement_count = None
        self._tasks = None
        self._step = None

        self.custom_task = custom_task

    def render(self):
        self._inner_env.render()

    @property
    def observation_space(self):
        original_observation_space = gymnasium.spaces.Box(0, 255, tuple(self._size) + (3,), np.uint8)
        task_vector_space = gymnasium.spaces.Box(low=0, high=1, shape=(self._task_vector_size,), dtype=np.float32)
        return gymnasium.spaces.Dict({
            'original_observation': original_observation_space,
            'task_vector': task_vector_space
        })

    @property
    def action_space(self):
        # noinspection PyUnresolvedReferences
        return gymnasium.spaces.Discrete(len(crafter_constants.actions))

    def set_task(self, custom_task):
        self.custom_task = custom_task
        assert self.custom_task in self.achievements_list
        self._current_achievement_task = self.custom_task

        self._previous_achievement_count = None
        self._step = 0

    def _get_task_vector(self):
        # Create a one-hot encoded vector for the current achievement task
        task_vector = np.zeros(self._task_vector_size, dtype=np.float32)
        if self._current_achievement_task:
            task_index = self.achievements_list.index(self._current_achievement_task)
            task_vector[task_index] = 1
        return task_vector

    def step(self, action):
        obs, reward, done, info = self._inner_env.step(action)
        self._step += 1

        terminated = info['discount'] < 1
        truncated = done if not terminated else False

        reward /= self._main_task_reward_divisor
        # Code to handle additional reward for the current achievement task
        if self._current_achievement_task:
            cur_achievement_count = info['achievements'].get(self._current_achievement_task, 0)
            if self._previous_achievement_count is None:
                self._previous_achievement_count = cur_achievement_count
            if cur_achievement_count > self._previous_achievement_count:
                reward += self._subtask_extra_reward
                self._previous_achievement_count = cur_achievement_count
                terminated = True

        info['episode_extra_stats'] = info.get('episode_extra_stats', {})
        if terminated or truncated:
            self._update_episode_stats(info)

        augmented_obs = {
            'original_observation': obs,
            'task_vector': self._get_task_vector()
        }
        return augmented_obs, reward, terminated, truncated, info

    @staticmethod
    def _compute_scores(percents):
        scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
        return scores

    def _update_episode_stats(self, info, key='episode_extra_stats'):
        achievements = {'Achievements/' + ach: 100.0 if val > 0.0 else 0.0 for ach, val in
                        info['achievements'].items()}
        info[key] = achievements
        info[key]['Num_achievements'] = sum(val > 0 for val in achievements.values())
        info[key]['Score'] = self._compute_scores(np.array(list(achievements.values())))

        key_name = f'Task_achievements/{self._current_achievement_task}'
        info[key][key_name] = self._previous_achievement_count
        info[key]['TaskScore'] = self._previous_achievement_count
        if self._previous_achievement_count:
            info[key]['TaskStepsToSuccess'] = self._step

    def reset(self, *args, **kwargs):
        if self.custom_task:
            assert self.custom_task in self.achievements_list
            self._current_achievement_task = self.custom_task
        else:
            self._current_achievement_task = random.choice(self.achievements_list)
        self._previous_achievement_count = 0
        self._step = 0

        original_obs = self._inner_env.reset()
        augmented_obs = {
            'original_observation': original_obs,
            'task_vector': self._get_task_vector()
        }
        return augmented_obs, {}


def main():
    env = PatchedCrafterEnv()

    obs = env.reset()
    terminated, truncated, info = False, False, {}

    while not terminated and not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    print(info.get('episode_extra_stats'))


if __name__ == '__main__':
    main()