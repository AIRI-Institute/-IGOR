import time
from collections import deque

from sample_factory.envs.env_utils import register_env
from sample_factory.envs.env_wrappers import EpisodeCounterWrapper

from sample_factory.eval import _print_fps_stats, _print_eval_summaries
from sample_factory.cfg.arguments import checkpoint_override_defaults, parse_full_cfg, parse_sf_args

from sample_factory.algo.sampling.evaluation_sampling_api import EvalSamplingAPI, SamplingLoop
from sample_factory.algo.utils.env_info import obtain_env_info_in_a_separate_process
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log

import sys
sys.path.append(".")
from igor.agent_training.env.sequential_task_wrapper import make_crafter_sequential_env
from igor.agent_training.main import make_patched_crafter_env


def patched_eval(cfg: Config):
    cfg.episode_counter = True  # Always True for this script
    cfg.decorrelate_envs_on_one_worker = False  # Not needed in eval
    env_info = obtain_env_info_in_a_separate_process(cfg)

    sampler = EvalSamplingAPI(cfg, env_info)
    sampler.init()
    sampler.start()

    print_interval_sec = 1.0
    fps_stats = deque([(time.time(), 0, 0)], maxlen=10)
    episodes_sampled = 0
    last_print = time.time()

    while episodes_sampled < cfg.sample_env_episodes:
        try:
            if time.time() - last_print > print_interval_sec:
                policy_id = 0  # Only look at the first policy
                episodes_sampled = len(sampler.eval_episodes[policy_id])
                env_steps_sampled = sampler.total_samples

                fps_stats.append((time.time(), episodes_sampled, env_steps_sampled))
                _print_fps_stats(cfg, fps_stats)
                last_print = time.time()

                log.info(f"Progress: {episodes_sampled}/{cfg.sample_env_episodes} episodes sampled")
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt during evaluation")
            break

    status = sampler.stop()

    _print_eval_summaries(cfg, sampler.eval_stats)

    return status, sampler.eval_stats


def eval_plan(list_of_tasks, sample_env_episodes=100, seed=None):
    register_env("SequentialCrafterEnv-v0", make_crafter_sequential_env)
    register_env("PatchedCrafterEnv-v0", make_patched_crafter_env)

    sys.argv.extend(['--env', "SequentialCrafterEnv-v0"])
    sys.argv.extend(['--experiment', 'scaled_reward'])
    sys.argv.extend(['--train_dir', 'train_dir'])
    sys.argv.extend(['--num_workers', str(1)])
    sys.argv.extend(['--num_envs_per_worker', str(2)])
    sys.argv.extend(['--worker_num_splits', str(2)])
    sys.argv.extend(['--seed', str(seed)])
    sys.argv.extend(['--sample_env_episodes', str(sample_env_episodes)])
    # sys.argv.extend(['--serial_mode', "True"])

    parser, cfg = parse_sf_args(evaluation=True)

    checkpoint_override_defaults(cfg, parser)
    cfg = parse_full_cfg(parser)
    cfg.list_of_tasks = list_of_tasks
    status, eval_stats = patched_eval(cfg)

    _print_eval_summaries(cfg, eval_stats)

    return eval_stats


def main():
    stats = eval_plan(['collect_wood'] + ['collect_stone'] * 10 + ['collect_wood'] * 5, sample_env_episodes=100)
    print(stats)


if __name__ == "__main__":
    sys.exit(main())