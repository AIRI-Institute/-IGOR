import json
import sys
import glob
print(glob.glob("*"))
sys.path.append("./scripts")

from crafter_dataset_generator import TASKS_json
from typing import List, Union
import ast
import re

def get_allowed_token_ids(tokenizer):
    """
    List of the Crafter Allowed tokens
    """
    tasks = [
            " Collect Wood ", " Place Table ", " Make Wood Pickaxe ", " Collect Stone ",
            " Make Stone Pickaxe ", " Place Stone ", " Collect Coal ", " Make Iron Pickaxe ",
            " Collect Iron ", " Make Iron Sword ", " Collect Diamond ", " Place Furnace ",
            " Collect Drink ", "Defeat Zombie ", " Defeat Skeleton ", " Make Wood Sword ",
            " Eat Cow ", " Collect Sapling ", " Place Plant ", " Eat Plant ", " Wake Up ",
            " Make Stone Sword ", '"', "'", ',', '[', ']', " With Count ", " with count ", " 1 2 3 4 5 6 7 "
        ]
   
    task_tokens = [tokenizer.encode(task, add_special_tokens=False) for task in tasks]
    unique_tokens = set(token for tokens in task_tokens for token in tokens)

    all_token_ids = set(range(tokenizer.vocab_size))
    allowed_token_ids = list(unique_tokens) + [0, 784, 1]
    bad_token_ids = all_token_ids - set(allowed_token_ids)
    bad_words_ids = [[token] for token in bad_token_ids]

    return allowed_token_ids, bad_words_ids

def crafter_subtasks_list(str_subtasks):
    str_subtasks= str_subtasks.replace("'","")
    str_subtasks= str_subtasks.replace('"',"")
    str_subtasks= str_subtasks.replace('[',"")
    str_subtasks= str_subtasks.replace(']',"")
    str_subtasks= str_subtasks.replace('(',"")
    str_subtasks= str_subtasks.replace(')',"")
    list_subtasks = str_subtasks.split(", ")
    return list_subtasks
    
def remove_pad(s):
    s = s.replace("<pad>", '')
    s = s.replace("</s>", '')
    s = s.replace("[", '')
    s = s.replace("]", '')
    s = s.replace("'", '')
    return claen_str(s)
    
def claen_str(task):
    return " ".join(task.split())
    
def check_format_correctness(task):
    TASKS = json.loads(TASKS_json)
    task_name = claen_str(task.split("with count")[0])
    possible_tasks = [claen_str(task['task']) for task in TASKS]
    return task_name in possible_tasks

def concatenate_prompt(prompt: str, predicted_value: str) -> str:
    """
    Concatenates a predicted value to a prompt string before the last closing bracket,
    removing specific unwanted characters from the predicted value.

    Parameters:
    - prompt (str): The original prompt string.
    - predicted_value (str): The value to append to the prompt.

    Returns:
    - str: The modified prompt string with the predicted value appended.

    Example:
    >>> concatenate_prompt("Consume a bovine, craft a wooden tool for mining rock, and gather the mineral from the environment.; Done: []", "Eat cow")
    'Task: Consume a bovine, craft a wooden tool for mining rock, and gather the mineral from the environment.; Done: [Eat cow]'
    
    """
    return prompt.rsplit(']', 1)[0] + ', ' + predicted_value.strip(" <pad>['</s>]") + ']'

    
def title_case_to_snake_case(input_tasks: Union[str, List[str]], return_as_str: bool = False) -> Union[str, List[str]]:
    """
    Parameters:
    input_tasks (Union[str, List[str]]): A string representation of a task list or a direct list of task strings.
    return_as_str (bool): Flag to return the output as a string representation if True, or as a list if False.

    Returns:
    Union[str, List[str]]: The transformed tasks as a string representation or a list, depending on the parameter.

    Examples:
    >>> transform_array("['Eat Plant with count 2', 'Collect Drink']", return_as_str=True)
    "['eat_plant', 'eat_plant', 'collect_drink']"

    >>> transform_array(['Defeat Zombie with count 2', 'Collect Iron'], return_as_str=False)
    ['defeat_zombie', 'defeat_zombie', 'collect_iron']
    """
    if isinstance(input_tasks, str):
        tasks = ast.literal_eval(input_tasks)
    else:
        tasks = input_tasks
    
    transformed_tasks = []
    for task in tasks:
        count = 1
        if "with count" in task:
            # If cant parse, use count = 1. Need to DEBUG;
            try:
                base_task, count_str = task.split(" with count ")
                count = int(count_str)
            except:
                base_task = task
                count = 1
        else:
            base_task = task
            count = 1
        
        formatted_task = base_task.replace(" ", "_").lower()
        transformed_tasks.extend([formatted_task] * count)
    
    return str(transformed_tasks) if return_as_str else transformed_tasks


def snake_case_to_title_case(tasks: List[str]) -> List[str]:
    """
    Parameters:
    tasks (List[str]): A list of task strings in snake_case format.

    Returns:
    List[str]: A list of transformed task strings in Title Case, each appended 
    with a count if the task occurs more than once.

    Examples:
    >>> transform_tasks(['eat_plant', 'eat_plant', 'collect_drink'])
    ['Eat Plant with count 2', 'Collect Drink']

    >>> transform_tasks(['defeat_zombie', 'collect_iron', 'wake_up', 'wake_up'])
    ['Defeat Zombie', 'Collect Iron', 'Wake Up with count 2']

    >>> transform_tasks(['make_iron_pickaxe', 'eat_cow'])
    ['Make Iron Pickaxe', 'Eat Cow']
    """

    task_count: dict[str, int] = {}
    for task in tasks:
        if task in task_count:
            task_count[task] += 1
        else:
            task_count[task] = 1

    transformed_tasks: List[str] = []
    for task, count in task_count.items():
        readable_task = task.replace('_', ' ').title()
        if count > 1:
            transformed_tasks.append(f"{readable_task} with count {count}")
        else:
            transformed_tasks.append(readable_task)
    
    return transformed_tasks