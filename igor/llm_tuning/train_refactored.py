import sys
import wandb
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

sys.path.append("./datasets")
sys.path.append("./utils")
sys.path.append(".")

from crafter_dataset import CrafterDataset
from parse_config import parse_args, parse_config
from iglu.validator import IgluValidator, IgluPrimValidator
from utils.crafter.validation import sample_examples, CrafterValidator
from igor.llm_tuning.custom_trainer import CustomSeq2SeqTrainer, AugmentSeq2SeqTrainer
import torch
from torch.nn.functional import cross_entropy, softmax


def create_wandb_table_from_tempered_example(tempered_example):
    if len(tempered_example[0]) == 3:
        table_columns = ["sentence", "subtasks", "tempered_subtasks"]
    else:
        table_columns = ["sentence", "subtasks"]
    table = wandb.Table(columns=table_columns)
    for entry in tempered_example:
        table.add_data(*entry)
    return table


def log_histogram(sentences):
    sentence_counts = {}
    for sentence in sentences:
        if sentence in sentence_counts:
            sentence_counts[sentence] += 1
        else:
            sentence_counts[sentence] = 1

    data = [[sentence, count] for sentence, count in sentence_counts.items()]
    columns = ["Sentence", "Count"]
    data_table = wandb.Table(columns=columns)
    for sentence, count in sentence_counts.items():
        data_table.add_data(sentence, count)

    return wandb.plot.bar(data_table, "sentence", "count",
                          title="Sentence Occurrences")


def entropy(data, model, tokenizer, epsilon=1e-9):
    print(data)
    labels = torch.tensor(data['test']['labels']).cuda()
    inputs = torch.tensor(data['test']['input_ids']).cuda()

    with torch.no_grad():
        outputs = model(inputs, labels=labels)
        logits = outputs.logits

        probs = softmax(logits, dim=-1)
        valid_labels_mask = labels != -100

        labels_one_hot = torch.nn.functional.one_hot(labels[valid_labels_mask], num_classes=probs.size(-1))

        valid_probs = probs[valid_labels_mask]

        relevant_probs = torch.sum(valid_probs * labels_one_hot, dim=-1)

        relevant_probs = relevant_probs.clamp_min(epsilon)
        entropy_loss = -(relevant_probs * relevant_probs.log()).mean()
    return entropy_loss


class Pipeline():
    def __init__(self, config):

        config_mapping = {
            'environment_setting': ['project', 'namespace', 'exp_name'],
            'experiment_setting': ['epochs', 'seed', 'learning_rate'],
            'paths': ['train_data', 'test_data', 'model_save_path'],
            'model_setting': ['base_model', 'use_castom_trainer', 'dataset_entropy_weight', 'training_args_config']
        }

        self.target_score = 0
        self.last_target_score = 0

        for config_key, attributes in config_mapping.items():
            for attr in attributes:
                setattr(self, attr, config[config_key].get(attr))

        wandb.init(project=self.project, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model)
        print("Trainer type: ", self.use_castom_trainer)

        self._init_data_loaders()
        self._init_trainer()
        self._init_validator()

    def _init_data_loaders(self, n_test=100):
        label_pad_token_id = -100
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
        )

        self.train_dataloader = CrafterDataset(self.train_data, False).prepare_data(self.tokenizer, split=True,
                                                                                    map=True)
        self.test_dataset = CrafterDataset(self.test_data, False).prepare_data(self.tokenizer, split=True, map=True)

    def _init_trainer(self):
        print("EPOCHS: ", self.epochs)

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"{self.model_save_path}/{self.exp_name}",
            num_train_epochs=25,
            seed=self.seed,
            learning_rate=float(self.learning_rate),
            **self.training_args_config
        )

        trainer_classes = {
            0: Trainer,
            1: AugmentSeq2SeqTrainer,
            2: CustomSeq2SeqTrainer
        }

        trainer_class = trainer_classes.get(self.use_castom_trainer)
        if not trainer_class:
            raise ValueError("Unsupported trainer type")

        common_params = {
            "model": self.model,
            "args": training_args,
            "tokenizer": self.tokenizer,
            "train_dataset": self.train_dataloader,
            "eval_dataset": self.train_dataloader,
            "data_collator": self.data_collator
        }

        if self.use_castom_trainer == 2:
            print("Trainer with entropy")
            common_params["dataset_entropy_weight"] = self.dataset_entropy_weight

        self.trainer = trainer_class(**common_params)

    def _init_validator(self):
        test_dataset = CrafterDataset(self.test_data, False)
        self.test_prompts = test_dataset.original_data['description'].values[:10]

        if self.namespace == 'iglu':
            self.validate = IgluValidator('./datasets/iglu/test_iglu.pickle').validate
        elif self.namespace == 'iglu_prim':
            self.validate = IgluPrimValidator('./datasets/iglu/test_iglu.pickle').validate
        else:
            c_validator = CrafterValidator(test_dataset.original_data)
            c_validator.set_metrics_tags('examples')
            self.validate = c_validator.validate

    def general_model_metrics(self):
        # entropy_score = entropy(self.test_dataset, self.trainer.model, self.tokenizer)
        return {"entropy": None}

    def save_chekpoint(self, epoch):
        if self.target_score > self.last_target_score:
            self.trainer.save_model(f"{self.model_save_path}/{self.exp_name}/_{epoch}")
            self.last_target_score = self.target_score

    def validation(self, epoch):
        specific_lore_metrics = self.validate(self.model, self.tokenizer)

        examples_val = specific_lore_metrics['examples'].copy()
        examples = create_wandb_table_from_tempered_example(examples_val)
        wandb.log({"examples": examples})

        specific_lore_metrics['log_time'] = epoch
        wandb.log({"specific_lore_metrics": specific_lore_metrics})

        if self.namespace == 'iglu':
            self.target_score = specific_lore_metrics['target']
        else:
            self.target_score = self.last_target_score + 1

    def run(self):
        res = None
        previos_score = 0
        # Train n_epochs with times N, log each epochs
        for i in range(0, self.epochs):
            self.validation(i)
            self.save_chekpoint(i)
            general_metrics = self.general_model_metrics()
            res = self.trainer.train()
        wandb.log({"results": res})
        general_metrics['epochs'] = i
        wandb.log({"general_metrics": general_metrics})

    def run_no_transformers(self):
        res = None
        previos_score = 0
        N = 10
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        for i in range(0, self.epochs):

            for batch in tqdm(self.train_dataloader):
                optimizer.zero_grad()
                # print(batch)
                batch = {k: v.to('cuda') for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            torch.cuda.empty_cache()

            if i % N == 0:
                self.validation(i // N)
                self.save_chekpoint(i // N)
            general_metrics = self.general_model_metrics()
        # res = self.trainer.train()

        wandb.log({"results": res})

        general_metrics['epochs'] = i
        wandb.log({"general_metrics": general_metrics})


def main():
    configs = parse_config("igor/configs/llm_model.yaml", use_args=True)
    print(configs)
    pipeline = Pipeline(configs)
    pipeline.run()


if __name__ == '__main__':
    main()
