import json
import tqdm
import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
from collections import Counter
sys.path.append(".")
import ariande.agent_training.example as agent 
from ariande.inference.llm_model import LanguageModel
sys.path.append("./utils")
from dataset_utils import concatenate_prompt, title_case_to_snake_case, crafter_subtasks_list
from metrics_rl import evaluate_episode_run
sys.path.append("./datasets")
from crafter_dataset import CrafterDataset

from scripts.crafter.make_reward_dataset import aggregate_subtask, extract_number_from_string
DEPENDENCIES_PATH = "./datasets/tasks_dependencies.json"

class Validator():
    achievements_list = [
        'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron', 'collect_sapling', 'collect_stone',
        'collect_wood', 'defeat_skeleton', 'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
        'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword', 'make_wood_pickaxe', 'make_wood_sword',
        'place_furnace', 'place_plant', 'place_stone', 'place_table', 'wake_up',
    ]
    def __init__(self, data_path, llm_model, idx = None):
        if idx is not None:
            print (f"======================= {idx} =========================")
            self.dataset = CrafterDataset(data_path).original_data[idx:idx+1]
        else:
            self.dataset = CrafterDataset(data_path).original_data
        self.results = []*self.dataset.shape[0]
        self.llm_model = llm_model
        with open(DEPENDENCIES_PATH, "r") as file:
            self.all_tasks_dependencies = json.load(file)
            self.all_tasks_dependencies = {task["task"]: task["dependencies"] for task in self.all_tasks_dependencies}

    def subtask_score(completed_rate, count_dependencies):
         return completed_rate * (count_dependencies+1)**0.12

    def run_with_env(self):
        results = {"results":[], 'task_name':[], 'target':[], 'description':[]}
        for index, row in self.dataset.iterrows():
            prompt = row['description']
            target = row['subtasks']
            predicted = crafter_subtasks_list(self.llm_model.predict(prompt))
            
            predicted = title_case_to_snake_case(predicted)
            predicted = [p for p in predicted if p in self.achievements_list]

            result_json = agent.sequential_tasks_json(predicted)
            print(result_json)
            metric = evaluate_episode_run(self.all_tasks_dependencies, result_json['results'])
            results['results'].append(result_json)
            results['task_name'].append(str(predicted))
            results['target'].append(str(target))
            results['description'].append(str(prompt))
        return self.make_reward_dataset(results)


    def make_reward_dataset(self, evaluated_data):
        dataset = {'prompt':[], 'predicted':[], 'completed_rate':[], 'count_dependencies':[], 'done':[]}
        print(evaluated_data['description'])
        for i in tqdm.tqdm(range(len(evaluated_data['description']))):
            description = evaluated_data['description'][i] + "; Done: []"
            title_case_goal = eval(evaluated_data['target'][i])
           # print(evaluated_data['target'][i], evaluated_data['task_name'][i])
            # tasks without repeat (Defeat zombie with count 3-> Defeat Zombie)
            snake_case_goal_no_repeats = [title_case_to_snake_case([task])[0] for task in title_case_goal] 
            snake_case_goal = evaluated_data['results'][i]['task']
        
            # If subtask repeats it is dificult to evaluate it so pass it
            if Counter(title_case_goal).most_common(1)[0][1]>1:
                continue
            
            results = {subtask:{'name':title_case_goal[i], 
                                'results':[], 
                                'count':extract_number_from_string(title_case_goal[i])} 
                       for i, subtask in enumerate(snake_case_goal_no_repeats)}
        
            episode_staistics = evaluated_data['results'][i]['results']
        
            #Add metrics for subtask
            for task_stats in episode_staistics:
                subtask_result = dict()
                subtask_result['completed_rate'] = task_stats['CompletedRate']/100
                subtask_result['dependencies'] = task_stats['DependenciesCount']
                subtask_result['done'] = int(task_stats['Status']=='Completed')
                try:
                    results[task_stats['Task']]['results'].append(subtask_result)
                except:
                    continue
        
            #Aggregate results if some subtask need to be perform several times
            # and make columns for dataset
            for subtask in results:
                subtask_name, completed_rate, count_dependencies, done = aggregate_subtask(results[subtask])
                dataset['prompt'].append(description)
                dataset['predicted'].append(subtask_name)
                dataset['completed_rate'].append(completed_rate)
                dataset['count_dependencies'].append(count_dependencies)
                dataset['done'].append(done)
        
                description = concatenate_prompt(description, subtask_name)        
        return pd.DataFrame(dataset)
        
def evaluate_score( dataset_pd):
    dataset_pd['score'] = dataset_pd['completed_rate'] * (dataset_pd['count_dependencies']+1)**0.12
    dataset_pd['score']= (dataset_pd['score'] - dataset_pd['score'].min())/(dataset_pd['score'].max() - dataset_pd['score'].min())
    return dataset_pd['score'].mean()

def run_with_env_n(model, tokenizer):
 
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    results = {'description':[], 'mean_s':[], 'std_s':[]}

    for i in range(25):
        v = Validator(data_path="./datasets/aug_test.csv", llm_model = lm, idx = i)
        v_description = v.dataset['description'].values[0]
        scores = [evaluate_score(v.run_with_env()) for _ in range(10)]
        mean_score = np.mean(scores)
        mean_std = np.std(scores)
        
        results['description'].append(v_description)
        results['mean_s'].append(mean_score)
        results['std_s'].append(mean_std)
    return pd.DataFrame(results)
        
    
def main( checkpoint, results_save_path):
    checkpoint = checkpoint #"./models/llm_models/aug_data/checkpoint-4500"
    result_save_path = results_save_path
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).cuda()

    results = run_with_env_n(model, tokenizer)
    results.to_csv(result_save_path)

    print(f"Statistics for {checkpoint}")
    print("----------------------------")
    print(results.mean())
    return results.mean()
    

if __name__ == "__main__":
   r1 = main("./models/llm_models/aug_data/checkpoint-4500","./results/base_checkpoint_15.csv")
   r2 = main("./models/llm_models/tuned_with_feedback_1.0", "./results/tuned_checkpoint_15.csv")
   r3 = main("./models/llm_models/tuned_with_feedback_1.0_5", "./results/tuned_checkpoint_v2_15.csv")
   r1.to_csv("./results/base_checkpoint_15_mean.csv", header=None)
   r2.to_csv("./results/tuned_checkpoint_15_mean.csv",header=None)
   r3.to_csv("./results/tuned_checkpoint_15_v2_mean.csv",header=None)

    


    
            
            
            
            
        

    
        
        
    
    