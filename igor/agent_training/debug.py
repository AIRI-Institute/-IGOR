import sys

from main import main as run_experiment


def main():
    sys.argv.extend(['--train_for_env_steps', str(5_000_000)])
    sys.argv.extend(['--stats_avg', str(100)])
    sys.argv.extend(['--wandb_project', 'debug'])
    sys.argv.extend(['--num_envs_per_worker', str(10)])
    sys.argv.extend(['--num_workers', str(4)])
    sys.argv.extend(['--experiment_summaries_interval', str(10)])
    sys.argv.extend(['--with_wandb', "False"])

    # Turn off parallelization for debug reasons
    # sys.argv.extend(['--serial_mode', "True"])

    run_experiment()


if __name__ == '__main__':
    sys.exit(main())
