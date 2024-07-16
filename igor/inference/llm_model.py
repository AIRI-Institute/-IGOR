import sys
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append("../../")
from utils.crafter.dataset_utils import concatenate_prompt, remove_pad


class LanguageModel():
    def __init__(self, model_path=None, model=None, tokenizer=None, generation_kwargs=None):
        if model is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        else:
            self.model = model
            self.tokenizer = tokenizer
        if generation_kwargs is None:
            self.generation_kwargs = {'max_new_tokens': 256}
        else:
            self.generation_kwargs = generation_kwargs

    def encode(self, sentence):
        model_inputs = torch.from_numpy(np.asarray(self.tokenizer([sentence])['input_ids']))
        return model_inputs

    def decode(self, tokens):
        return self.tokenizer.decode(tokens[0])

    def set_generation_kwargs(self, do_sample, temperature):
        self.generation_kwargs['do_sample'] = do_sample
        self.generation_kwargs['temperature'] = temperature

    def predict(self, sentence, set_kwargs=False, do_sample=False, temperature=1.0):
        tokens = self.encode(sentence).cuda()
        if set_kwargs:
            self.set_generation_kwargs(do_sample, temperature)
        next_subtasks_tokens = self.model.cuda().generate(tokens, **self.generation_kwargs)
        next_subtask = remove_pad(self.decode(next_subtasks_tokens))
        return next_subtask

    def sequentional_prediction(self, base_sentence, do_sample=False, temperature=1.0):
        prediction = ""
        next_subtask = ""
        sentence = base_sentence + "; Done: []"

        subtasks = []
        sentences = []
        sentences.append(sentence)
        with torch.no_grad():
            for _ in range(10):
                tokens = self.encode(sentence).cuda()
                next_subtasks_tokens = self.model.generate(tokens, do_sample=do_sample, temperature=temperature)
                next_subtask = remove_pad(self.decode(next_subtasks_tokens))

                if 'end' in next_subtask:
                    return sentences, subtasks
                subtasks.append(next_subtask)
                sentence = concatenate_prompt(sentence, f"({next_subtask})")
                sentences.append(sentence)
        return sentences, subtasks
