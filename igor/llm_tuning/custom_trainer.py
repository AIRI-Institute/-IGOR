import wandb

from transformers import Seq2SeqTrainer
import torch
from torch.nn.functional import cross_entropy, softmax
import wandb
import torch
from torch.nn.functional import cross_entropy, softmax
import random


class ColorChanger:
    def __init__(self, color_mapping=None):
        self.color_digit_mapping = {
            'blue': 1,
            'green': 2,
            'red': 3,
            'orange': 4,
            'purple': 5,
            'yellow': 6
        }
        self.digit_color_mapping = {v: k for k, v in self.color_digit_mapping.items()}

        if color_mapping is None:
            color_mapping = self.generate_random_color_mapping()
        index = random.randint(0, 5)
        self.colors = list(color_mapping.keys())
        self.color_mapping = color_mapping  # {colors[index]: color_mapping[colors[index]]}
        self.active_color_mapping = {self.colors[index]: color_mapping[self.colors[index]]}

    def set_main_color(self, color):
        if isinstance(color, int):
            color_str = self.digit_color_mapping[color]
        else:
            color_str = color

        index = self.colors.index(color_str)
        self.active_color_mapping = {self.colors[index]: self.color_mapping[self.colors[index]]}

    def augment_text(self, text):
        for original_color, new_color in self.active_color_mapping.items():
            text = text.replace(original_color, new_color)
        return text

    def possible_coords(self, block_coordinates):
        block_coordinates = eval(block_coordinates)
        colors_set = []
        for x, y, z, color_digit in block_coordinates:
            if color_digit not in colors_set:
                colors_set.append(color_digit)
        return colors_set

    def augment_coordinates(self, block_coordinates):
        block_coordinates = eval(block_coordinates)
        print(block_coordinates)
        new_coordinates = []
        for x, y, z, color_digit in block_coordinates:
            original_color = self.digit_color_mapping[color_digit]
            new_color = self.active_color_mapping.get(original_color, original_color)
            new_color_digit = self.color_digit_mapping[new_color]
            new_coordinates.append([x, y, z, new_color_digit])
        return str(new_coordinates)

    def generate_random_color_mapping(self):
        colors = list(self.color_digit_mapping.keys())
        shuffled_colors = random.sample(colors, len(colors))
        return dict(zip(colors, shuffled_colors))


class AugmentSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, tokenizer, augmentations=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.augmentations = augmentations or ['replace_red_with_green']

    def _prepare_input(self, data):
        try:
            inputs_decoded = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data['input_ids']]

            labels_decoded = []
            labels_masks = data['labels'] != -100
            for label_ids, mask in zip(data['labels'], labels_masks):
                label_ids_filtered = label_ids[mask]
                decoded_label = self.tokenizer.decode(label_ids_filtered, skip_special_tokens=True)
                labels_decoded.append(decoded_label)

            changer = ColorChanger()
            inputs_augmented = []
            labels_augmented = []

            for i in range(len(inputs_decoded)):
                p = random.random()
                if p < 0.01:
                    try:
                        label_color = changer.possible_coords(labels_decoded[i])
                        color_to_set = random.choice(label_color)
                        changer.set_main_color(color_to_set)

                        changed_instruction = changer.augment_text(inputs_decoded[i])
                        changed_subtasks = changer.augment_coordinates(labels_decoded[i])

                        inputs_augmented.append(changed_instruction)
                        labels_augmented.append(changed_subtasks)
                        print("AUGMENTATION!")
                    except:
                        print("NO AUGMENTATION!")
                        inputs_augmented.append(inputs_decoded[i])
                        labels_augmented.append(labels_decoded[i])
                else:
                    inputs_augmented.append(inputs_decoded[i])
                    labels_augmented.append(labels_decoded[i])

            data['input_ids'] = torch.stack([self.tokenizer.encode(text, return_tensors='pt', padding='max_length',
                                                                   truncation=True,
                                                                   max_length=data['input_ids'].shape[1]).squeeze(0) for
                                             text in inputs_augmented])
            new_labels = []
            for text, mask in zip(labels_augmented, labels_masks):
                encoded = self.tokenizer.encode(text, return_tensors='pt', padding='max_length', truncation=True,
                                                max_length=mask.size(0)).squeeze(0)
                encoded[~mask] = -100
                new_labels.append(encoded)
            data['labels'] = torch.stack(new_labels)
        except (ValueError, IndexError):
            # IF NOT DICT
            pass

        return super()._prepare_input(data)

    def apply_augmentations(self, text):
        for augmentation in self.augmentations:
            augmentation_func = getattr(self, augmentation, None)
            if augmentation_func and callable(augmentation_func):
                text = augmentation_func(text)
        return text

    @staticmethod
    def replace_red_with_green(text):
        return text.replace("red", "green")


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, dataset_entropy_weight=0.5, epsilon=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_entropy_weight = dataset_entropy_weight
        self.epsilon = epsilon
        self.pseudo_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        probs = softmax(logits, dim=-1)
        self.pseudo_steps += 1

        ce_loss = cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
        ce_loss = ce_loss[labels.view(-1) != -100].mean()

        indexes_tensor = torch.tensor([[val for val in row] for row in labels])
        indexes_tensor[labels == -100] = probs.size(-1)

        one_hot_encoded = torch.nn.functional.one_hot(indexes_tensor, num_classes=probs.size(-1) + 1)

        labels_one_hot = one_hot_encoded.sum(dim=1)[:, :probs.size(-1)].gt(0).int()
        new_shape = (labels_one_hot.shape[0], 1, labels_one_hot.shape[1])
        reshaped_labels_one_hot = labels_one_hot.reshape(new_shape).cuda()
        d2 = probs.shape[1]
        reshaped_labels_one_hot = reshaped_labels_one_hot.expand(-1, d2, -1)

        incorrect_probs = torch.sum(probs * reshaped_labels_one_hot, dim=-1)

        epsilon = 1e-8
        incorrect_probs = incorrect_probs.clamp_min(epsilon)

        entropy_loss = -(incorrect_probs.log()).mean()

        wandb.log({"ce": ce_loss, "entropy_loss": entropy_loss, 'loss_step': self.pseudo_steps})

        return ce_loss + entropy_loss
