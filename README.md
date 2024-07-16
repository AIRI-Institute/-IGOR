# IGOR – Instruction Following with Goal-Conditioned Reinforcement Learning in Virtual Environments


## Installation

### With Conda Virtual Environment (available only for LLM experiments)

1. **Create the environment:**

    ```bash
    conda create --name Igor python=3.9
    ```

2. **Activate the environment:**

    ```bash
    conda activate Igor
    ```

3. **Install the required packages:**

    ```bash
    pip install -r docker/requirements.txt
    ```
### With Docker

1. **Create the container:**
```bash
cd ./docker
sh build.sh
cd ../
 ```

2. **Run container:**
```bash
docker run --shm-size 20G --env WANDB_API_KEY=$WANDB_API_KEY --rm -it -v $(pwd):/code -w /code --gpus all igor bash
```




## 🤗 Links to Haggingface to LLM

###  [**Zoya/igor_iglu_prim_llm**](https://huggingface.co/Zoya/igor_iglu_prim_llm)

#### Example
- **Input:** `<Architect> Make 3 red blocks in the middle of the grid`
- **Output:** `[0, 5, 5], [1, 1, 3], skyeast, red`


###  [**Zoya/igor_crafter_llm**](https://huggingface.co/Zoya/igor_crafter_llm)

#### Example
- **Input:** `Vanquish the undead foe, gather a single unit
of metallic mineral, and forge an iron weapon`
- **Output:** `['Defeat Zombie', 'Collect Iron with count 1']`


## Datasets

### Crafter Dataset Generation

To generate the Crafter dataset, run:

```bash
python3 scripts/crafter_dataset_generator.py
```

- The datasets for the Crafter environment can be found at `./datasets/crafter`.

- The datasets for the IGLU environment, including its augmented and primitive versions, can be found at `./datasets/iglu`.

## Run Experiments with LLM

### Run LLM Tuning in Crafter Dataset

To run LLM tuning in the Crafter dataset, execute:

```bash
sh scripts/crafter/train_llm.sh
```

### Run LLM Tuning in IGLU Original Dataset

To run LLM tuning in the IGLU original dataset, execute:

```bash
sh scripts/iglu/train_llm.sh
```

### Run LLM Tuning in IGLU with Subtasks as Primitives

To run LLM tuning in IGLU with subtasks as primitives, execute:

```bash
sh scripts/iglu/train_llm_prim.sh
```

## Run Experiments with Rl

### Run RL traning

To run RL tuning in the Crafter dataset, execute:

```bash
sh scripts/crafter/train_rl.sh
```


**Paper:** [Instruction Following with Goal-Conditioned Reinforcement Learning in Virtual Environments](https://arxiv.org/abs/2407.09287)

**Citation:**

```bibtex
@article{volovikova2024instruction,
      title={Instruction Following with Goal-Conditioned Reinforcement Learning in Virtual Environments}, 
      author={Zoya Volovikova and Alexey Skrynnik and Petr Kuderov and Aleksandr I. Panov},
      year={2024},
      eprint={2407.09287},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.09287}, 
}
```
