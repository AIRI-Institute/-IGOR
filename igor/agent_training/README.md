# PPO for Goal-Based Crafter

## Installation

1. Install dependencies:
   ```bash
   pip3 install r docker/requirements.txt
   ```
2. Optionally, build the Docker image using the Dockerfile:
   ```bash
   cd docker && sh build.sh
   ```

## Inference Example

To run **PPO** with pre-trained weights:
  ```bash
  python3 example.py
  ```

## Local Training

Start training with default settings:
  ```bash
  python3 main.py
  ```
To adjust parameters, such as running with 10 sampling workers, use arguments:
  ```bash
  python3 main.py --num_workers 10
  ```
For a complete list of hyperparameters and their descriptions, visit [SampleFactory Configuration Parameters](https://www.samplefactory.dev/02-configuration/cfg-params/).

## Fine-tuning

To fine-tune existing weights specify experiment folder, e.g. for weights in ```finetune``` folder: 
```bash
python3 main.py --experiment finetune --restart_behavior resume
```

## Training on Server

1. Install the `crafting` package:
   ```shell
   pip install crafting
   ```
   Verify the installation and create a default configuration file:
     ```bash
     crafting test.yaml
     ```
   Run the created configuration:
     ```bash
     crafting test.yaml
     ```
2. If encountering issues such as a `command not found` error, add this line to `.bashrc` or `.bash_profile`:
   ```shell
   export PATH="$HOME/.local/bin:$PATH"
   ```
3. Add your wandb API key to `.bashrc`. It will be automatically ingested into the container by `crafting`.
4. Specify the desired GPU in `run.yaml`, e.g., `NVIDIA_VISIBLE_DEVICES=0`.
5. Start the training process, which will automatically run in Docker daemon mode (allowing you to stop logging while the container continues running):
   ```bash
   crafting run.yaml
   ```



## Local Debugging

For local debugging:
  ```bash
  python3 debug.py
  ```
