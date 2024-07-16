import sys
from tqdm import tqdm
import pickle
import numpy as np
from gridworld.tasks import Task

sys.path.append("./datasets")
sys.path.append(".")
from utils.iglu._primitives import from_str, to_grid
from igor.inference.llm_model import LanguageModel


def coords_to_grid(coords):
    grid = np.zeros((9,11,11))
    try:
        Z,X,Y, colors = zip(*coords)
        grid[Z,X,Y]=colors
    except:
        return grid
    return grid

def grid_to_coords(grid):
    coords = np.where(grid)
    colors = grid[coords]
    return list(zip(*coords, colors))

def centralize(coords):
    coords = [list(coord) for coord in coords]
    dx = coords[0][1] - 5
    dy = coords[0][2] - 5
    for i, prim in enumerate(coords):
        coords[i][1] -= dx
        coords[i][2] -= dy
    coords = [tuple(coord) for coord in coords]
    return coords
    
def f1_based_reward(target_grid: np.ndarray, grid: np.ndarray, start_grid=np.zeros((9, 11, 11))):
    """
    Calculate metrics for the target grid and the current grid.
    :param target_grid: target grid numpy array.
    :param grid: current grid numpy array.
    :param start_grid: starting grid numpy array.
    :return: tuple of metrics.
    """
    task = Task('', target_grid=target_grid - start_grid)

    argmax = task.argmax_intersection(grid)
    builded = grid - start_grid
    maximal_intersection = task.get_intersection(builded, *argmax)

    target_size = task.target_size
    precision = maximal_intersection / (target_size + 1e-10)
    recall = maximal_intersection / (len(builded.nonzero()[0]) + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return f1

def parse_coords(subtasks):
    clear_subtasks = []
    for s in subtasks:
        try:
            clear_subtasks.append(eval(s))
        except:
            continue
    return clear_subtasks
    
def split_into_chunks(sequence, chunk_size=4):
    return [sequence[i:i+chunk_size] for i in range(0, len(sequence), chunk_size)]

class IgluValidator():
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data_test = pickle.load(f)
        self.dataset = []
        for dialogue in data_test:  
            for instruction, grid in dialogue:
                self.dataset.append((instruction, grid))

    def parse(self, output):
        new_output = f'[{output.replace("&&", "], [")}]'
        return eval(new_output)

    def subtasks_to_grid(self, subtasks):
        return coords_to_grid(subtasks)

    def aggregate_metrics(self, per_task_metrics):
        metrics_keys = list(per_task_metrics[0].keys())
        full_metrics = dict()
        avg_metrics = dict()
        per_task_metrics_wandb = dict()

        for i in range(len(per_task_metrics)):
            for key in metrics_keys:
                per_task_metrics_wandb[f"{i}/{key}"] = per_task_metrics[i][key]
            
        for key in metrics_keys:
            full_metrics[key] = []
            for i in range(len(per_task_metrics)):
                full_metrics[key].append(per_task_metrics[i][key])
                
        for key in metrics_keys:
            avg_metrics[key] = np.mean(full_metrics[key])

        per_task_metrics_wandb['avg/metrics'] = avg_metrics
        return per_task_metrics_wandb
                

    def metrics(self, predicted_subtasks, target_subtasks):
        task_metrics = {'on_plus':0, 'on_minus':0, 'diff_with_target':0, 'f1':0}
        
        # diff with target
        target_subtasks_count = len(target_subtasks)
        predicted_subtasks_count = len(predicted_subtasks)
        diff_with_target = target_subtasks_count - predicted_subtasks_count
        
        task_metrics['on_plus'] = int(diff_with_target < 0)
        task_metrics['on_minus'] = int(diff_with_target < 1)
        task_metrics['diff_with_target'] = abs(diff_with_target)

        # F1
        
        target_grid = coords_to_grid(target_subtasks)
        predicted_grid = self.subtasks_to_grid(predicted_subtasks)
        f1 = f1_based_reward(target_grid, predicted_grid)
        task_metrics['f1'] = f1

        return task_metrics
        
   
    def validate(self, model, tokenizer, temperature=1):
        lm = LanguageModel(model=model, tokenizer=tokenizer)
        metrics_full = []
        examples = []
        for (sentence, target_grid) in self.dataset:
            output = lm.predict(sentence, do_sample=False)
            examples.append((sentence, output))
            try:
                subtasks = self.parse(output)
                target_subtasks = grid_to_coords(target_grid)
                task_metrics = self.metrics(subtasks, target_subtasks)
                metrics_full.append(task_metrics)  
            except:
                task_metrics = {'on_plus':0, 'on_minus':0, 'diff_with_target':0, 'f1':0}
                metrics_full.append(task_metrics)  
                subtasks = []
                continue
        metrics = self.aggregate_metrics(metrics_full)
        metrics['examples'] = examples
        metrics['target'] = metrics['avg/metrics']['f1']
        return metrics

class IgluPrimValidator(IgluValidator):
    def parse(self, output):
        subtasks = from_str(output)
        return subtasks

    def subtasks_to_grid(self, subtasks):
        return to_grid(subtasks)
    
        