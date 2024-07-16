import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
import numpy as np

        
class CrafterDataset():
    def __init__(self,path, to_sequence=True):
        self.original_data = pd.read_csv(path)
        if to_sequence:
            self.data = self._transform_data_to_sequence(self.original_data)
        else:
            self.data = self.original_data
        self.tokenizer = None

    def _transform_data_to_sequence(self, df):
        # Создание нового DataFrame для хранения результатов
        transformed_df = pd.DataFrame(columns=['subtasks', 'description'])
    
        # Итерация по каждой строке исходного DataFrame
        for _, row in df.iterrows():
            subtasks = eval(row['subtasks'])  # Преобразование строки в список
            description = row['description']
            
            # Создание промежуточных строк для каждого подзадания
            done_tasks = []
            for subtask in subtasks:
                done_str = ', '.join(done_tasks) if done_tasks else ''
                done_tasks.append(str(subtask))  # Добавляем текущее подзадание в список выполненных
                transformed_df = transformed_df.append({
                    'subtasks': str([subtask]),
                    'description': f"{description}; Done: [{done_str}]"
                }, ignore_index=True)
    
            # Добавление финальной строки с общим выполнением
            done_str = ', '.join([str(s) for s in subtasks])
            transformed_df = transformed_df.append({
                'subtasks': '<end>',
                'description': f"{description}; Done: [{done_str}]"
            }, ignore_index=True)
    
        return transformed_df

    def preprocess_function(self, row_inputs, padding="max_length"):
        # add prefix to the input for t5
        inputs_ = row_inputs['description']
        subtasks_ = row_inputs['subtasks']

        inputs = []
        subtasks = []
        for i, input_ in enumerate(inputs_):
            if not isinstance(input_, str):
                inputs.append("")
                subtasks.append("")
            else:
                inputs.append(input_)
                subtasks.append(subtasks_[i])
                
        
        model_inputs = self.tokenizer(inputs, truncation=True, padding='max_length', max_length=self.max_tokens_input)
        labels = self.tokenizer(text_target=subtasks, truncation=True, padding='max_length',  max_length=self.max_tokens_labels)
        
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
    
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def encode_and_decode(self, dataset):
        encoded = self.tokenizer(dataset['description'])
        return encoded

    def ppo_tokenize(self, sample):
        sample["input_ids"] = self.tokenizer.encode(sample["description"])
        sample["query"] = self.tokenizer.decode(sample["input_ids"])
        return sample

    def calculate_max_tokens(self, dataset, keys):
        tokenized_inputs = dataset.map(lambda x: self.tokenizer(x[keys[0]] if x else 0), batched=True)
        max_source_length = max([len(x) for x in tokenized_inputs[keys[1]]])
        return max_source_length
        
    def prepare_data(self, tokenizer, split=False, map=False):
        dataset = Dataset.from_pandas(self.data )
        self.tokenizer = tokenizer
        self.max_tokens_input = self.calculate_max_tokens(dataset, ('description', 'input_ids'))
        self.max_tokens_labels = self.calculate_max_tokens(dataset, ('subtasks', 'input_ids'))
        print("Max tokens: ", self.max_tokens_input, self.max_tokens_labels)

        if map:
            
            dataset = dataset.map(self.preprocess_function, batched=True)
            all_columns = dataset.column_names

            # Определяем колонки, которые хотим сохранить
            required_columns = ['input_ids', 'attention_mask', 'labels']
            
            # Формируем список колонок, которые нужно удалить
            columns_to_remove = [column for column in all_columns if column not in required_columns]
            
            # Удаляем ненужные колонки
            dataset = dataset.remove_columns(columns_to_remove)
           # dataset.set_format('torch')
          #  dataset = DataLoader(dataset, shuffle=True, batch_size=64)
            
           # print(dataset)
        else:
            dataset = dataset.map(self.ppo_tokenize, batched=False)
        if split:
            return dataset
        else:
            return dataset

