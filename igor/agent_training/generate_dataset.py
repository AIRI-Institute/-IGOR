import json
import random
import pandas as pd
import numpy as np
import tqdm
import logging
import ast


from example import sequential_tasks_json, format_json_with_tabulate

rl_logger = logging.getLogger("rl")
rl_logger.setLevel(logging.CRITICAL)

def transform_array(tasks):
    transformed_tasks = []
    for task in tasks:
        if "with count" in task:
            base_task, count_str = task.split(" with count ")
            count = int(count_str)
        else:
            base_task = task
            count = 1
        
        formatted_task = base_task.replace(" ", "_").lower()
        transformed_tasks.extend([formatted_task] * count)
    
    return transformed_tasks
    
def make_dataset(data_path, save_path):
    data = pd.read_csv(data_path)
    tasks = data['crafter_subtasks'].values
    results = {"results":[], 'task_name':[]}
    for i, task in enumerate(tqdm.tqdm(tasks)):
        subtask_shuffled = ast.literal_eval(data['subtasks'][i])
        random.shuffle(subtask_shuffled)
        subtask_shuffled_crafter = transform_array(subtask_shuffled)
        
        result = sequential_tasks_json(subtask_shuffled_crafter)
        print(format_json_with_tabulate(result))
        results['results'].append(result)
        results['task_name'].append(str(subtask_shuffled))
        if i%100 == 0:
            with open(save_path, 'w') as f:
                json.dump(results, f)
            

    with open(save_path, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    make_dataset("./datasets/aug_train.csv", '../../datasets/evaluated_data/evaluated_aug_train_3.json')
    make_dataset("../../datasets/aug_train.csv", '../../datasets/evaluated_data/evaluated_aug_train_4.json')
    make_dataset("../../datasets/aug_train.csv", '../../datasets/evaluated_data/evaluated_aug_train_5.json')
    make_dataset("../../datasets/aug_train.csv", '../../datasets/evaluated_data/evaluated_aug_train_6.json')
    make_dataset("../../datasets/aug_train.csv", '../../datasets/evaluated_data/evaluated_aug_train_7.json')
    make_dataset("../../datasets/aug_train.csv", '../../datasets/evaluated_data/evaluated_aug_train_8.json')
    make_dataset("../../datasets/aug_train.csv", '../../datasets/evaluated_data/evaluated_aug_train_9.json')
    make_dataset("../../datasets/aug_train.csv", '../../datasets/evaluated_data/evaluated_aug_train_10.json')
    
    
    

    