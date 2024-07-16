import sys
from typing import Optional

import numpy as np
from sample_factory.algo.runners.runner import Runner, AlgoObserver
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
from sample_factory.envs.env_utils import register_env
from sample_factory.train import make_runner
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.utils.utils import log

from env.create_env import PatchedCrafterEnv
from env.curriculum_env import CurriculumCrafterEnv

# ENV_NAME = "PatchedCrafterEnv-v0"
ENV_NAME = "CurriculumCrafterEnv-v0"


def make_patched_crafter_env(full_env_name, cfg=None, _env_config=None, render_mode: Optional[str] = None):
    return CurriculumCrafterEnv()


def register_msg_handlers(cfg: Config, runner: Runner):
    class CustomExtraSummariesObserver(AlgoObserver):
        def extra_summaries(self, runner: Runner, policy_id: PolicyID, writer, env_steps: int) -> None:
            policy_avg_stats = runner.policy_avg_stats
            for key in policy_avg_stats:
                # Skipping default metrics
                if key in ['reward', 'len', 'true_reward', 'Done', 'true_objective']:
                    continue

                avg = np.mean(np.array(policy_avg_stats[key][policy_id]))
                writer.add_scalar(key, avg, env_steps)

                # Skipping printing of nested metrics
                if '/' not in key:
                    log.debug(f'{policy_id}-{key}: {round(float(avg), 3)}')

    runner.register_observer(CustomExtraSummariesObserver())


def override_default_params(parser):
    # Training, optimization, environment, and parallelization parameters
    parser.set_defaults(
        gamma=0.99,
        learning_rate=1e-4,
        recurrence=32,
        rollout=32,
        train_for_env_steps=1_000_000_000,  # One billion
        ppo_epochs=1,
        exploration_loss_coeff=0.003,
        batch_size=1024,
        max_grad_norm=0,
        num_envs_per_worker=10,
        num_workers=32,
    )

    # Model and architecture settings
    parser.set_defaults(
        use_rnn=True,
        encoder_conv_architecture='resnet_impala',
    )

    # Experiment control, reporting, and statistics
    parser.set_defaults(
        restart_behavior='overwrite',
        with_wandb=True,
        wandb_project='crafter-ppo2',
        heartbeat_reporting_interval=1000,
        experiment_summaries_interval=100,
        save_best_metric='Score',
        keep_checkpoints=1,
        stats_avg=1000,
    )


def parse_custom_args(argv=None, evaluation=False):
    sys.argv.extend(['--env', ENV_NAME])
    parser, cfg = parse_sf_args(argv, evaluation=evaluation)
    override_default_params(parser)
    final_cfg = parse_full_cfg(parser)
    return final_cfg


def main():
    register_env(ENV_NAME, make_patched_crafter_env)
    cfg = parse_custom_args()

    cfg, runner = make_runner(cfg)
    register_msg_handlers(cfg, runner)

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()
    return status


if __name__ == "__main__":
    sys.exit(main())