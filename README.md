# Project Training Scripts

This document outlines how to use the various training scripts included in this project: `feedforward_train.py`, `feedbackward_alt_train.py`, and `feedbackward_rec_alt_train.py`.

## 1. Feedforward Model Training (`feedforward_train.py`)

This script trains the base feedforward model. This model must be trained first, as its checkpoints are required for training the feedbackward models.

### Usage

```bash
python feedforward_train.py --active_dataset <dataset_name> [options]
```

### Arguments

*   `--config <path>`: Path to the configuration file. Defaults to `config.json`.
*   `--active_dataset <name>`: **(Required)** The name of the dataset to use for training (e.g., `gsm8k`, `wikitext`). This must be a key in the `datasets` section of your config file.
*   `--resume <bool>`: Set to `True` to resume training from the latest checkpoint in the output directory. Defaults to `False`.
*   `--checkpoint_dir <dir_name>`: The specific run directory inside `.out/<dataset_name>/` to resume from. If not provided when resuming, it will use the most recent run.
*   `--checkpoint_name <file_name>`: The specific checkpoint file (e.g., `epoch_ckpt_gsm8k.pt`) to load from within the checkpoint directory. If not provided when resuming, it will use the most recent checkpoint file.

### Example

To start a new training run for the `gsm8k` dataset:
```bash
python feedforward_train.py --active_dataset gsm8k
```

To resume the latest training run for `wikitext`:
```bash
python feedforward_train.py --active_dataset wikitext --resume True
```

## 2. Feedbackward Recurrent Model Training (`feedbackward_rec_alt_train.py`)

This script trains a *recurrent* feedbackward model. This model acts as a hypernetwork, generating LoRA-style delta weights to modify the behavior of a pre-trained feedforward model. It requires a checkpoint from the feedforward model to be trained.

### Usage

```bash
python feedbackward_rec_alt_train.py --active_dataset <dataset_name> --ff_checkpoint_dir <dir> --ff_checkpoint_name <file> --lora_rank <rank> [options]
```

### Arguments

*   `--config <path>`: Path to the configuration file. Defaults to `config.json`.
*   `--active_dataset <name>`: **(Required)** The name of the dataset to use.
*   `--ff_checkpoint_dir <dir_name>`: **(Required)** The directory of the pre-trained feedforward model checkpoint.
*   `--ff_checkpoint_name <file_name>`: **(Required)** The filename of the pre-trained feedforward model checkpoint.
*   `--lora_rank <int>`: **(Required)** The rank to use for the LoRA delta weights.
*   `--feedback_reverse <bool>`: If `True`, the feedback model generates delta weights in the reverse order of the feedforward model's blocks.
*   `--resume <bool>`: Set to `True` to resume training.
*   `--fb_checkpoint_dir <dir_name>`: The specific run directory for the feedbackward model to resume from.
*   `--fb_checkpoint_name <file_name>`: The specific checkpoint file for the feedbackward model to load.

### Example

To train a feedbackward recurrent model using a `gsm8k` feedforward checkpoint with a LoRA rank of 4:
```bash
python feedbackward_rec_alt_train.py \
    --active_dataset gsm8k \
    --ff_checkpoint_dir .out/gsm8k/ff_model_2025-08-17_10-00-00 \
    --ff_checkpoint_name epoch_ckpt_gsm8k.pt \
    --lora_rank 4
```

## 3. Feedbackward Model Training (`feedbackward_alt_train.py`)

This script trains a standard (non-recurrent) feedbackward model. Each block in this feedbackward model is distinct and corresponds to a block in the feedforward model. Like the recurrent version, it requires a pre-trained feedforward model checkpoint.

### Usage

```bash
python feedbackward_alt_train.py --active_dataset <dataset_name> --ff_checkpoint_dir <dir> --ff_checkpoint_name <file> --lora_rank <rank> [options]
```

### Arguments

The arguments are identical to `feedbackward_rec_alt_train.py`.

### Example

To train a standard feedbackward model using a `wikitext` feedforward checkpoint with a LoRA rank of 8:
```bash
python feedbackward_alt_train.py \
    --active_dataset wikitext \
    --ff_checkpoint_dir .out/wikitext/ff_model_2025-08-17_12-00-00 \
    --ff_checkpoint_name epoch_ckpt_wikitext.pt \
    --lora_rank 8
```
