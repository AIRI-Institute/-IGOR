import pandas as pd
import tqdm
import random
import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


TASKS_json = """
[
    {"task": "Collect Wood"},
    {"task": "Place Table", "dependency": ["Collect Wood"], "count": 2},
    {"task": "Make Wood Pickaxe", "dependency": ["Place Table"]},
    {"task": "Collect Stone", "dependency": ["Make Wood Pickaxe"]},
    {"task": "Make Stone Pickaxe", "dependency": ["Collect Stone"]},
    {"task": "Place Stone", "dependency": ["Collect Stone"]},
    {"task": "Collect Coal", "dependency": ["Make Wood Pickaxe"]},
    {"task": "Make Iron Pickaxe", "dependency": ["Collect Coal", "Place Furnace"]},
    {"task": "Collect Iron", "dependency": ["Make Stone Pickaxe"]},
    {"task": "Make Iron Sword", "dependency": ["Collect Iron", "Place Furnace", "Collect Coal"]},
    {"task": "Collect Diamond", "dependency": ["Make Iron Pickaxe"]},
    {"task": "Place Furnace", "dependency": ["Collect Stone"], "count": 4},
    {"task": "Collect Drink"},
    {"task": "Defeat Zombie"},
    {"task": "Defeat Skeleton"},
    {"task": "Make Wood Sword", "dependency": ["Place Table"]},
    {"task": "Eat Cow"},
    {"task": "Collect Sapling"},
    {"task": "Place Plant", "dependency": ["Collect Sapling"]},
    {"task": "Eat Plant", "dependency": ["Place Plant"]},
    {"task": "Wake Up"},
    {"task": "Make Stone Sword", "dependency": ["Collect Stone"]}
]
"""

TASKS = json.loads(TASKS_json)

class DescriptionGenerator:
    def __init__(self, general_prompt, generation_args=None):
        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
        self.prompt = general_prompt

        if generation_args is None:
            self.generation_args = {'max_new_tokens':200}
        else:
            self.generation_args = generation_args

    def correctness(self, task):
        # text = f"Is the sentense <{task}> сontains list? Choose (yes/no)"
        # inputs = self.tokenizer(text, return_tensors="pt").to(0)
        
        # outputs = self.model.generate(**inputs, max_new_tokens=2)
        # answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # answer =  answer.replace(text, "").replace("\n", "")
        # correctness = 'no' in answer.lower()
        return True #correctness
        
    def stye(self, task):
        text = f"Rewrite the instruction in the style of Tolkien with the same SHORT sentence size adding cosuality. Instruction '{task}'. In the manner of Tolkien, the instruction with the same SHORT sentence size would be:"
        inputs = self.tokenizer(text, return_tensors="pt").to(0)
        
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.replace(text, "").replace("\n", "")
        
    def generate(self, tasks):
        text = f"{self.prompt} {tasks}"
        inputs = self.tokenizer(text, return_tensors="pt").to(0)
    
        outputs = self.model.generate(**inputs, **self.generation_args)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.replace(text, "").replace("\n", "")
        styled = self.stye(answer)
        return answer, styled, self.correctness(answer), self.correctness(styled)


class Task:
    def __init__(self, task, dependency=None, count=None):
        self.task = task
        self.dependency = dependency
        self.dependency_count = count

    def _find_dependent_task(self):
        for task in TASKS:
            if task['task'] == self.task and 'dependency' in task:
                for dep in task['dependency']:
                    for t in TASKS:
                        if t['task'] == dep:
                            return t
        return None

    def generate(self, use_dependency_p=0.5, use_count_p=0.2, count_range=(1, 3)):
        tasks = []
        
        dependency_p = random.random()
        if self.dependency and dependency_p < use_dependency_p:
            dependent_task_data = self._find_dependent_task()
            if dependent_task_data:
                depend_task = Task(**dependent_task_data)
                count_range = (self.dependency_count - 1, self.dependency_count + 2) if self.dependency_count else count_range
                tasks += depend_task.generate(use_dependency_p, use_count_p, 
                                              count_range=count_range)

        count_p = random.random()
        if count_p < use_count_p:
            count = random.randint(*count_range)
            task_with_count = f"{self.task} with count {count}"
            tasks.append(task_with_count)
        else:
            tasks.append(self.task)
        return tasks



class TaskGenerator:
    def __init__(self, tasks, use_dependency_p=0.5, use_count_p=0.2, count_range=(1, 3), general_prompt=""):
        self.tasks = tasks
        self.use_dependency_p = use_dependency_p
        self.use_count_p = use_count_p
        self.count_range = count_range
        self.description_generator = DescriptionGenerator(general_prompt=general_prompt)

    def generate_dataset(self, num_entries, csv_file_path):
        subtasks = []
        dataset = []
        logging.info("Start generate subtasks")
        for _ in range(num_entries):
            tasks = []
            for i in range(3):
                task_data = random.choice(self.tasks)
                task = Task(**task_data)
                tasks += task.generate(self.use_dependency_p, self.use_count_p, self.count_range)
                if len(tasks)>=3:
                    break
            subtasks.append(tasks)
            
        data = {"subtasks": [], "description": [], "styled": [], 'desc_correct':[], 'style_correct':[]}
        dataset = pd.DataFrame(data)
        logging.info("Start generate descriptions")
        for i, task in tqdm.tqdm(enumerate(subtasks, start=1)):
            description, style, c1, c2 =  self.description_generator.generate(task)  
            dataset = dataset.append({"subtasks": task, "description": description,
                                      "styled":style, 'desc_correct':c1, 'style_correct': c2}, ignore_index=True)
            if i % 2 == 0:
                dataset.to_csv(csv_file_path, index=False)
        
        dataset.to_csv(csv_file_path, index=False)
        return dataset

def main():
    general_prompt = "Transform the following tasks into descriptive and persuasive instructions, giving them context and internal logic. Tasks must be connected into a single plot. The instructions should be on behalf of the player who controls intelligent agents in some game. The instructions should be written in one SHORT paragraph and contain LESS than 10 words.."

    general_prompt = "Create a text instruction for a list of subtasks for navigating agent, combining multiple tasks into one script. Instructions should be written in a dry and concise style. Instructions should fit into one short paragraph around three sentence, combining all tasks into a logically connected story. !Dont use words and name of objects from Subtasks in your description! Subtasks: "

    general_prompt = "Rewrite the list of subtasks into one instructive sentence for an agent in a virtual environment, replacing words from the list with synonyms. Subtasks: "

    generator = TaskGenerator(TASKS, general_prompt = general_prompt)
    dataset = generator.generate_dataset(500, csv_file_path = "crafter_dataset_v_Tolkien2.csv")  # Генерация 10 наборов задач
    print(dataset.head)
    
if __name__ == "__main__":
    main()



        
    
        