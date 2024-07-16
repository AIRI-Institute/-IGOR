from tqdm import tqdm
import numpy as np
from igor.inference.llm_model import LanguageModel
#from igor.rest.utils import calculate_reward


class CrafterValidator():
    def __init__(self, full_data, reward_model=None):
        self.full_data = full_data[:100]
        print(self.full_data)
        test_data = full_data['description'].values[:10]
        self.test_data = test_data
        self.reward_model = reward_model

    def sample_examples(self, model, tokenizer):
        lm = LanguageModel(model=model, tokenizer=tokenizer)
        return_ = []
        pbar = tqdm(self.test_data)
        examples = []
        for sentence in pbar:
            output = lm.predict(sentence, do_sample=False)
            output_temp = lm.predict(sentence, do_sample=True, temperature=2.7)
            examples.append((sentence, output, output_temp))
        return examples

    def check_correcness(self, right, predicted):
       # return calculate_reward(str(right), str(predicted), 'nocoh')
        predicted = [p.lower() for p in predicted]
        for subtask in right:
            if subtask.lower() not in predicted:
                return 0
        return 1

    def calculate_fake_reward(self, right, predicted):
        return calculate_reward(str(right), str(predicted), 'nocoh')

    def sample_per_one_task(self, model, tokenizer):
        lm = LanguageModel(model=model, tokenizer=tokenizer)
        return_ = []
        sentence = self.test_data[0]
        examples = []
        for i in range(50):
            #output = lm.predict(sentence, do_sample=False)
            output_temp = lm.predict(sentence, do_sample=True, temperature=1)
            examples.append(output_temp)
        return examples

    def reward_model_score(self, model, tokenizer):
        lm = LanguageModel(model=model, tokenizer=tokenizer)
        return_ = []
        pbar = tqdm(self.test_data)
    
        for sentence in pbar:
            _, subtasks = lm.sequentional_prediction(sentence, do_sample=False)
            mean_v = self.reward_model.evaluate_sequence(sentence, subtasks)
            return_.append(mean_v)
            current_mean = np.mean(return_)
            pbar.set_description(f"Current mean: {current_mean:.4f}")
    
        return np.mean(return_)

    def set_metrics_tags(self, metrics_tags):
        self.metrics_tags = metrics_tags

    def success_rate(self,  model, tokenizer):
        lm = LanguageModel(model=model, tokenizer=tokenizer)
        count_right = 0
        total_count = 0
        for i in range(self.full_data.shape[0]):
            right_subtasks = eval(self.full_data['subtasks'].values[i])
            instruction = self.full_data['description'].values[i]
            predicted_subtasks = lm.predict(instruction, do_sample=False)
            correctess = self.check_correcness(right_subtasks, predicted_subtasks.split(", "))
            count_right += correctess
            total_count += 1
        return count_right/total_count
        
    def validate(self, model, tokenizer, metrics_tags=None):
        metrics = dict()
        if metrics_tags is not None:
            self.set_metrics_tags(metrics_tags)
            
        if 'rm_score' in self.metrics_tags:
            metrics['target'] = self.reward_model_score(model, tokenizer)
        else:
            metrics['target'] = self.success_rate(model, tokenizer)
        if 'examples' in self.metrics_tags:
            metrics['examples'] = self.sample_examples(model, tokenizer)
           # metrics['one_sample'] = self.sample_per_one_task(model, tokenizer)
        
        return metrics
        
def validate_model(test_data, reward_model, model, tokenizer, temperature=1):
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    return_ = []
    pbar = tqdm(test_data)

    for sentence in pbar:
        _, subtasks = lm.sequentional_prediction(sentence, do_sample=False)
        mean_v = reward_model.evaluate_sequence(sentence, subtasks)
        return_.append(mean_v)
        current_mean = np.mean(return_)
        pbar.set_description(f"Current mean: {current_mean:.4f}")

    return np.mean(return_)

def sample_examples(test_data, model, tokenizer):
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    return_ = []
    pbar = tqdm(test_data)
    examples = []
    for sentence in pbar:
        output = lm.predict(sentence, do_sample=False)
       # output_temp = lm.predict(sentence, do_sample=True, temperature=1)
        examples.append((sentence, output))
    return examples
