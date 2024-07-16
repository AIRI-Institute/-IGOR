import json
import os
import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
import argparse
import yaml

sys.path.append(".")
import igor.agent_training.eval as agent
from igor.inference.llm_model import LanguageModel

sys.path.append("./utils/crafter")
from dataset_utils import title_case_to_snake_case, crafter_subtasks_list

sys.path.append("./datasets")
from crafter_dataset import CrafterDataset

DEPENDENCIES_PATH = "./datasets/tasks_dependencies.json"


class Validator():
    achievements_list = [
        'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron', 'collect_sapling', 'collect_stone',
        'collect_wood', 'defeat_skeleton', 'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
        'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword', 'make_wood_pickaxe', 'make_wood_sword',
        'place_furnace', 'place_plant', 'place_stone', 'place_table', 'wake_up',
    ]

    def __init__(self, data_path, llm_model):
        self.dataset = CrafterDataset(data_path).original_data[:100]
        self.results = [] * self.dataset.shape[0]
        self.llm_model = llm_model
        with open(DEPENDENCIES_PATH, "r") as file:
            self.all_tasks_dependencies = json.load(file)
            self.all_tasks_dependencies = {task["task"]: task["dependencies"] for task in self.all_tasks_dependencies}

    def llm_prediction(self, prompt):
        prediction = self.llm_model.predict(prompt)
        print(prediction)
        print("-----------")
        return prediction

    def _run_with_env(self, subtasks, N_evals, seed):
        keys = ["metrics/len_list_of_task", "metrics/count_solved", "metrics/solved_fully", "metrics/solved_fully_soft"]
        stats = agent.eval_plan(subtasks, sample_env_episodes=N_evals, seed=seed)

        results = {}
        for key in keys:
            results[f'{key}_mean'] = np.mean(stats[key][0])
            results[f'{key}_std'] = len(stats[key][0])

        return {
            'len_list_of_task': stats["metrics/len_list_of_task"],
            'count_solved': results["metrics/count_solved_mean"],
            'count_solved_std': results["metrics/count_solved_std"],
            'solved_fully': results["metrics/solved_fully_mean"],
            'solved_fully_soft': results["metrics/solved_fully_soft_mean"],
            'solved_fully_std': results["metrics/solved_fully_std"]
        }

    def filter_subtasks(self, predicted_subtasks):
        return [p for p in predicted_subtasks if p in self.achievements_list]

    def prepare_subtasks_for_env(self, subtasks):
        predicted = subtasks
        try:
            predicted = title_case_to_snake_case(predicted)
        except Exception as e:
            raise Exception(f"Subtask preparation error: {e}")
        predicted = self.filter_subtasks(predicted)
        if len(predicted) == 0:
            raise Exception(f"Subtask preparation error: no subtasks in list(!)")
        return predicted

    def get_subtasks(self, prompt):
        subtasks = self.llm_prediction(prompt)
        print(subtasks)
        print(crafter_subtasks_list(subtasks[1]))
        subtasks = crafter_subtasks_list(subtasks[1])
       # exit()
        if subtasks is not None:
            try:
                subtasks = self.prepare_subtasks_for_env(subtasks)
                return subtasks
            except:
                return ['wake_up']
        return None

    def run_with_env(self, N_evals, result_file_path, seed=None):
        results = {'target': [],
                   'description': [],
                   'predicted_target': []}

        for index, row in self.dataset.iterrows():
            prompt = row['description']
            target = row['subtasks']
            crafter_subtasks = eval(row['crafter_subtasks'])

            predicted_subtasks = self.get_subtasks(prompt)
            if predicted_subtasks is None:
                predicted_subtasks = crafter_subtasks

            results['predicted_target'].append(predicted_subtasks)
            results['target'].append(target)
            results['description'].append(prompt)

            metrics = self._run_with_env(predicted_subtasks, N_evals, seed)
            for key in metrics:
                if key not in results:
                    results[key] = []
                results[key].append(metrics[key])

            if index % 15 == 0:
                pd.DataFrame(results).to_csv(result_file_path)

        return pd.DataFrame(results)


class NotSeqValidator(Validator):
    def llm_prediction(self, prompt):
        return None, self.llm_model.predict(prompt)


class RowEnvValidator(Validator):
    def llm_prediction(self, prompt):
        return None, None


def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)  # Use yaml.safe_load to read the YAML file


def prepare_model(checkpoint, tokenizer_name="google/flan-t5-small"):
    """
    Prepare tokenizer and model based on the given checkpoint.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).cuda()
    return LanguageModel(model=model, tokenizer=tokenizer)


def process_model(model_name, run_index, config, use_llm=True):
    """
    Process a single model based on the configuration and save the results.
    """
    path = os.path.join(os.getcwd(), config['base_result_path'])

    if not os.path.exists(path):
        os.makedirs(path)

    underscore_suffix = '_'
    result_file_name = f"{model_name}{underscore_suffix}.csv".replace('/', '_')
    result_file_path = f"{config['base_result_path']}{result_file_name}"
    result_file_path = f"{config['base_result_path']}_{run_index}_{model_name}){underscore_suffix}.csv"

    if use_llm:
        lm = prepare_model(f"{config['base_model_path']}{model_name}")
        v = NotSeqValidator(data_path=config['data_path'], llm_model=lm)
    else:
        v = RowEnvValidator(data_path=config['data_path'], llm_model=False)

    results = v.run_with_env(N_evals=1, result_file_path=result_file_path, seed=run_index)
    results.to_csv(result_file_path)

    mean_result_file_path = f"{config['base_result_path']}_{run_index}_{model_name}){underscore_suffix}mean.csv".replace(
        '/', '_')
    results.mean().to_csv(mean_result_file_path, header=None)


def parse_arguments():
    """
    Parse command-line arguments to get the configuration file path.
    """
    parser = argparse.ArgumentParser(description="Process models based on a given configuration.")
    parser.add_argument('-c', '--config', default='config.yaml', help='Path to configuration file.')
    return parser.parse_known_args()[0]


def main():
    # args = parse_arguments()
    config = load_config("igor/configs/config_crafter/inference.yaml")

    # delattr(args, 'config')
    for run in range(config["num_runs"]):
        for model_name in config["model_names"]:
            process_model(model_name, run, config)


if __name__ == "__main__":
    main()
