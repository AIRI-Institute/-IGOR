from dataclasses import dataclass

import numpy as np

from env.create_env import PatchedCrafterEnv


@dataclass
class TaskCurriculumStorage:
    alpha_for_delta: float = 0.2
    p_base: float = None
    running_delta: float = 0.0
    previous: float = 0.0

    alpha_for_sr: float = 0.5
    running_success_rate: float = 0.0

    threshold: float = 0.95
    high_value_div: float = 4.0


class CurriculumCrafterEnv(PatchedCrafterEnv):
    _info_stats_key = 'episode_extra_stats'

    def __init__(self, custom_task=None, recorder_dir=None):
        super().__init__(custom_task, recorder_dir)
        n = len(self.achievements_list)
        self._curriculum = {key: TaskCurriculumStorage(p_base=1.0) for key in self.achievements_list}
        self._forward_reset_info = None

    def step(self, action):
        obs, reward, done, info = self._inner_env.step(action)
        self._step += 1

        terminated = info['discount'] < 1
        truncated = done if not terminated else False

        reward /= self._main_task_reward_divisor
        # Code to handle additional reward for the current achievement task
        task_solved = False
        if self._current_achievement_task:
            cur_achievement_count = info['achievements'].get(self._current_achievement_task, 0)
            if self._previous_achievement_count is None:
                self._previous_achievement_count = cur_achievement_count
            if cur_achievement_count > self._previous_achievement_count:
                task_solved = True
                reward += self._subtask_extra_reward
                self._previous_achievement_count = cur_achievement_count
                terminated = True

        info['episode_extra_stats'] = info.get('episode_extra_stats', {})
        if terminated or truncated:
            q = self._curriculum[self._current_achievement_task]
            q.running_delta = q.running_delta * (1 - q.alpha_for_delta) + q.alpha_for_delta * abs(
                q.previous - task_solved)
            q.running_success_rate = q.running_success_rate * (1 - q.alpha_for_sr) + q.alpha_for_sr * task_solved
            q.previous = task_solved

            self._update_episode_stats(info)

        augmented_obs = {
            'original_observation': obs,
            'task_vector': self._get_task_vector()
        }
        return augmented_obs, reward, terminated, truncated, info

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

        if self._forward_reset_info:
            info['episode_extra_stats'].update(**self._forward_reset_info)
            self._forward_reset_info = None

    def select_task(self):
        probs = []
        for key in self.achievements_list:
            q = self._curriculum[key]
            if q.running_success_rate >= q.threshold:
                # probs.append(q.p_base / (q.high_value_div ** 2))
                probs.append(q.p_base / q.high_value_div)
            else:
                probs.append(q.p_base + q.running_delta * q.high_value_div)

        def softmax(x, temperature=1.0):
            x_adjusted = x / temperature
            e_x = np.exp(x_adjusted - np.max(x_adjusted))
            return e_x / e_x.sum()

        probs = softmax(np.array(probs))
        task_idx = np.random.choice(len(self.achievements_list), p=probs)
        task = self.achievements_list[task_idx]
        probability = probs[task_idx]

        return task, probability

    def reset(self, *args, **kwargs):

        if self.custom_task:
            assert self.custom_task in self.achievements_list
            self._current_achievement_task = self.custom_task
        else:
            self._current_achievement_task, probability = self.select_task()
            self._forward_reset_info = {'CurriculumProbability/' + self._current_achievement_task: probability}
        self._previous_achievement_count = 0
        self._step = 0

        original_obs = self._inner_env.reset()
        augmented_obs = {
            'original_observation': original_obs,
            'task_vector': self._get_task_vector()
        }

        return augmented_obs, {}