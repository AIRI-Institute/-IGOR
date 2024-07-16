import gymnasium

from igor.agent_training.env.create_env import PatchedCrafterEnv

def success_rate(info, target_tasks):
    crafter_info = info['achievements'].copy()
    solved_soft = 0
    for subtask in target_tasks:
        for key in crafter_info:
            if crafter_info[key] > 0:
                done_subtask = key
                if subtask in done_subtask:
                    solved_soft += 1
                    crafter_info[key] -= 1
    return int(solved_soft == len(target_tasks))
                
        
class SequentialTasksWrapper(gymnasium.Wrapper):
    def __init__(self, env, list_of_tasks=None, seed=None):
        if list_of_tasks is None:
            raise KeyError("List of tasks can't be empty")
        super().__init__(env)
        self.env._seed = seed
        self._list_of_tasks = list_of_tasks
        self._current_task_idx = None
        self._step = None
        self._seed = seed


    @property
    def task(self):

        task = self._list_of_tasks[self._current_task_idx]
        return task

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if len(self._list_of_tasks)==0:
            truncated = 1
            terminated = 1
            info['discount'] = 0
            return obs, rew, terminated, truncated, info
            
        self._step += 1
        dead = info['discount'] < 1

        if terminated and not dead:
            terminated = False
            self._current_task_idx += 1
            if self._current_task_idx >= len(self._list_of_tasks):
                terminated = True
            else:
                try:
                    self.env.set_task(self.task)
                except:
                    truncated = 1
                    terminated = 1
                    info['discount'] = 0

        if terminated or truncated:
            self.update_metrics(info)
        return obs, rew, terminated, truncated, info


    def update_metrics(self, info):
        SR = success_rate(info, self._list_of_tasks)
        info = info.get('episode_extra_stats', {})
        info['metrics/solved_tasks'] = self._current_task_idx / len(self._list_of_tasks)
        info['metrics/solved_fully'] = self._current_task_idx == len(self._list_of_tasks)
        info['metrics/len_list_of_task'] = len(self._list_of_tasks)
        info['metrics/count_solved'] = self._current_task_idx

        # not_solved_subtasks = self._list_of_tasks[self._current_task_idx:]
        # solved_soft = self._current_task_idx
        # for key in info:
        #     if 'Achievements' in key and info[key] > 0:
        #         additional_subtasks = key.replace("Achievements/", "")
        #         for subtasks in not_solved_subtasks:
        #             if subtasks in additional_subtasks:
        #                 solved_soft += 1
        info['metrics/solved_fully_soft'] = SR

        
            
        print(info)
       # exit()
        # ToDo: code to calculate metrics

    def reset(self, **kwargs):
        self._current_task_idx = 0
        self.env.set_task(self.task)
        self._step = 0
        return self.env.reset()


def make_crafter_sequential_env(full_env_name, cfg=None, _env_config=None, render_mode=None):
    return SequentialTasksWrapper(PatchedCrafterEnv(), list_of_tasks=cfg.list_of_tasks, seed = cfg.seed)