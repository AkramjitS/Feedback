import sys
import os
import time

import json
import shutil
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from types import SimpleNamespace
import glob
from contextlib import nullcontext
from typing import Optional, List, Dict, Tuple, Union
import math

from custom_logger import Logger

from model import FeedforwardModel
from data_utils import get_dataloaders
from utils import forward_process, feedforward_iterative_generation_loop, count_parameters, get_lr_schedule, roundup_multiple, model_numel, average, save_model

@torch.no_grad()
def run_validation(model, dataloader, config, ctx, tokenizer, logger:Logger):
    """
    Runs validation loss calculation and generates a text sample.
    """
    model.eval()
    
    # --- 1. Calculate Validation Loss ---
    total_loss = 0.0
    num_batches = 0

    for x0 in dataloader:
        x0 = x0.to(config.training.device)
        
        noisy_batch, masked_indices, p_mask = forward_process(
            x0, tokenizer.mask_token_id, config.diffusion.eps
        )

        with ctx:
            logits = model(noisy_batch)
        
        # Calculate loss only on the masked tokens
        if masked_indices.any():
            token_loss = F.cross_entropy(logits[masked_indices], x0[masked_indices], reduction='none') / p_mask[masked_indices]
            loss = token_loss.sum() / (x0.shape[0] * x0.shape[1])
        else:
            loss = torch.tensor(0.0, device=config.training.device)

        total_loss += loss.item()
        num_batches += 1
        if num_batches >= 50: # Limit validation batches for speed
            break
    
    val_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # --- 2. Generate a Conditional Sample ---
    logger.log("\n--- Generating Sample ---")
    # Get a single sample from the validation set to use as context
    val_iterator = iter(dataloader)
    ground_truth_tokens = next(val_iterator)[0:1].to(config.training.device) # Take 1 sample
    
    # Create the context and the mask for generation
    context_len = ground_truth_tokens.shape[1] // 4 # Use first 25% as context
    context = ground_truth_tokens.clone()
    context[:, context_len:] = tokenizer.pad_token_id # Mask out the rest
    
    # True means "generate this token", False means "keep this token"
    generation_mask = torch.zeros_like(context, dtype=torch.bool)
    generation_mask[:, context_len:] = True

    # Generate the completion
    generated_tokens = feedforward_iterative_generation_loop(
        model, 
        context, 
        generation_mask, 
        config.diffusion.generation_steps, 
        tokenizer, 
        config.training.device
    )
    
    # Decode and log for qualitative assessment
    ground_truth_text = tokenizer.decode(ground_truth_tokens[0].tolist())
    context_text = tokenizer.decode(ground_truth_tokens[0, :context_len].tolist())
    generated_text = tokenizer.decode(generated_tokens[0].tolist())

    logger.log(f"\n--- Ground Truth ---\n{ground_truth_text}")
    logger.log(f"\n--- Provided Context ---\n{context_text}")
    logger.log(f"\n--- Model Generation ---\n{generated_text}\n")

    model.train()
    return val_loss

def main():
    logger = Logger()
    logger.log("Command line arguments:")
    for i, arg in enumerate(sys.argv):
        logger.log(f"Argument {i}: {arg}")
    # not logging env variables
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a Residual Diffusion LLM.")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file.')
    parser.add_argument('--active_dataset', type=str, help='Name of the active dataset.')
    parser.add_argument('--resume', type=bool, default=False, help='Whether to resume training from a checkpoint.')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Name of the checkpoint directory.')
    parser.add_argument('--checkpoint_name', type=str, default=None, help='Name of the checkpoint file.')
    args = parser.parse_args()

    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    # add active_dataset to config and ensure it is an allowed dataset
    assert args.active_dataset in config_dict['datasets'], f"Active dataset '{args.active_dataset}' not found in config."
    config_dict['active_dataset'] = args.active_dataset
    config_dict['training']['resume'] = args.resume
    config_dict['training']['checkpoint_dir'] = args.checkpoint_dir
    config_dict['training']['checkpoint_name'] = args.checkpoint_name
    config = json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))

    # --- Setup ---
    active_dataset_name = config.active_dataset
    dataset_config = getattr(config.datasets, active_dataset_name)
    
    torch.manual_seed(1337)
    if torch.cuda.is_available() and config.training.device == 'cuda':
        torch.cuda.manual_seed(1337)

    device = config.training.device
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.training.dtype]
    ctx = torch.amp.autocast(device_type=device, dtype=dtype) if device == 'cuda' else torch.amp.autocast(device_type='cpu', dtype=dtype)

    # --- Output Directory and Resuming Logic ---
    out_dir = None
    checkpoint_to_load = None
    
    just_resumed = False

    if config.training.resume:
        just_resumed = True
        
        base_dir = os.path.join(config.training.out_dir, config.active_dataset)
        run_prefix = config.training.ff_run_name
        # Get config directory or latest directory
        potential_dirs = []
        if config.training.checkpoint_dir:
            checkpoint_dir = os.path.join(base_dir, config.training.checkpoint_dir)
            assert os.path.isdir(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} does not exist."
            potential_dirs.append(checkpoint_dir)
            logger.log(f"Set output dir from config: {checkpoint_dir}")
        else:
            # Find all directories in base_dir that match the run_prefix
            potential_dirs = [d for d in glob.glob(os.path.join(base_dir, f'{run_prefix}_*')) if os.path.isdir(d)]
        if potential_dirs:
            latest_run_dir = max(potential_dirs, key=os.path.getctime)
            out_dir = latest_run_dir  # Continue in this directory
            
            # Get config checkpoint or latest checkpoint following 
            if config.training.checkpoint_name:
                checkpoint_to_load = os.path.join(out_dir, config.training.checkpoint_name)
                assert os.path.isfile(checkpoint_to_load), f"Checkpoint file {checkpoint_to_load} does not exist."
                logger.log(f"Resuming training from run: {out_dir}")
                logger.log(f"Loading checkpoint from config: {checkpoint_to_load}")
            else:
                # Find the latest intermediate checkpoint within that directory to resume from
                # Checks both ckpt's and epoch_ckpt's
                ckpt_pattern = os.path.join(out_dir, f'ckpt_{active_dataset_name}_*.pt')
                ckpt_list = glob.glob(ckpt_pattern)
                epoch_ckpt_pattern = os.path.join(out_dir, f'epoch_ckpt_{active_dataset_name}_*.pt')
                epoch_ckpt_list = glob.glob(epoch_ckpt_pattern)
                all_ckpts = ckpt_list + epoch_ckpt_list
                if all_ckpts:
                    checkpoint_to_load = max(all_ckpts, key=os.path.getctime)
                    logger.log(f"Resuming training from latest run: {out_dir}")
                    logger.log(f"Loading checkpoint: {checkpoint_to_load}")
                else:
                    logger.log(f"Found run directory {out_dir} but no checkpoint inside. Starting a new run.")
                    out_dir = None  # Force creation of a new directory
        else:
            logger.log(f"No previous run directories found with prefix '{run_prefix}'. Starting a new run.")

    # If not resuming or no suitable directory/checkpoint was found, create a new one
    if out_dir is None:
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        run_name_with_timestamp = f"{config.training.ff_run_name}_{timestamp}"
        out_dir = os.path.join(config.training.out_dir, config.active_dataset, run_name_with_timestamp)
        os.makedirs(out_dir, exist_ok=True)
        logger.log(f"Starting new run. Checkpoints and logs will be saved in: {out_dir}")

        # --- Save code and config for reproducibility ---
        logger.log("Saving configuration and code to output directory...")
        #shutil.copy2(args.config, os.path.join(out_dir, 'config.json'))
        shutil.copy2(args.config, os.path.join(out_dir, os.path.basename(args.config)))
        source_files_to_save = [
            'model.py', 'data_utils.py', 'utils.py', 'feedforward_train.py', 'custom_logger.py'
        ]
        for file_name in source_files_to_save:
            shutil.copy2(file_name, os.path.join(out_dir, file_name))
        logger.log("Done saving files.")
        
    # Setup log and csv files
    logger.set_log_file(os.path.join(out_dir, config.training.log_file_name))
    logger.set_csv_file(os.path.join(out_dir, config.training.csv_file_name), header=["type", "epoch", "loss", "lr"])

    # --- Data Loading ---
    train_loader, val_loader, tokenizer = get_dataloaders(dataset_config, config.training, logger)
    
    # Dynamically set the vocab_size from the tokenizer after it has been configured.
    # This is crucial because special tokens (like [MASK]) could be added, changing the vocab size.
    # Also need to ensure that the vocab_size is a mutliple of 128 for performance
    dataset_config.vocab_size = len(tokenizer)
    dataset_config.vocab_size = roundup_multiple(dataset_config.vocab_size, 128)
    logger.log(f"Vocab size: {dataset_config.vocab_size}")
    
    # --- Model and Optimizer ---
    model = FeedforwardModel(config.model, dataset_config)
    model.to(device=device, dtype=dtype)
    logger.log(f"Model has {count_parameters(model):,} trainable parameters.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    # --- Load state from checkpoint if resuming ---
    step = 0
    best_val_loss = float('inf')
    if checkpoint_to_load:
        checkpoint = torch.load(checkpoint_to_load, map_location=device)

        # The state dict may have been saved from a compiled model, which prefixes keys
        # with "_orig_mod.". We strip this prefix for robust loading.
        model_state_dict = checkpoint['model_state_dict']
        unwrapped_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(unwrapped_state_dict)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    if config.training.compile:
        logger.log("Compiling the model... (requires PyTorch 2.0+)")
        model = torch.compile(model)

    # --- Training Loop ---
    num_epochs = getattr(config.training.epochs, active_dataset_name)
    logger.log(f"Starting training for {num_epochs} epochs on dataset '{active_dataset_name}'...")

    total_training_steps = num_epochs * len(train_loader)
    logger.log(f"Total training steps: {total_training_steps}")
    
    logger.log(f"Total number of feedforward model parameters: {model_numel(model)}")
    
    warmup_steps = 0
    if config.training.decay_lr:
        warmup_steps = int(config.training.warmup_ratio * total_training_steps)
        logger.log(f"Warmup steps: {warmup_steps}")

    start_epoch = step // len(train_loader) if len(train_loader) > 0 else 0
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = []
        
        epoch_start_time = time.time()
        for i, x0 in enumerate(train_loader):
            if epoch == start_epoch and i < (step % len(train_loader) if len(train_loader) > 0 else 0):
                continue
            # Determine and set the learning rate for this iteration
            lr = get_lr_schedule(step, config, total_training_steps, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            x0 = x0.to(device)
            
            # LLaDA-style random sequence length adjustment
            if torch.rand(1) < 0.01:
                random_length = torch.randint(1, x0.shape[1] + 1, (1,))
                x0 = x0[:, :random_length]

            noisy_batch, masked_indices, p_mask = forward_process(
                x0, tokenizer.mask_token_id, config.diffusion.eps
            )

            with ctx:
                noisy_batch = noisy_batch.detach()
                logits = model(noisy_batch)

            # Calculate loss only on the masked tokens, weighted by inverse probability
            if masked_indices.any():
                token_loss = F.cross_entropy(logits[masked_indices], x0[masked_indices], reduction='none') / p_mask[masked_indices]
                loss = token_loss.sum() / (x0.shape[0] * x0.shape[1])
            else:
                # Handle case where no tokens are masked (can happen with low p_mask)
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()

            just_resumed = False

            epoch_loss.append(loss.item())
            step += 1
        
        epoch_duration = time.time() - epoch_start_time
        logger.log(f"Epoch {epoch+1} finished in {epoch_duration:.2f} seconds.")
        
        avg_e_loss = average(epoch_loss)
        logger.log(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_e_loss:.4f} | LR: {lr:.6f}")
        logger.log_csv(["train", epoch+1, avg_e_loss, lr])
        
        val_loss = run_validation(model, val_loader, config, ctx, tokenizer, logger)
        logger.log(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss:.4f} | LR: {lr:.6f}")
        logger.log_csv(["val", epoch+1, val_loss, lr])
        
        if (epoch + 1) % config.training.save_epoch_interval == 0:
            # save epoch model and best model based off of validation
            save_model(
                step,
                model, optimizer, 
                avg_e_loss, val_loss, best_val_loss,
                config.training.save_epoch, out_dir,
                f'epoch_ckpt_{active_dataset_name}_{epoch+1}.pt',
                'epoch checkpoint',
                logger
            )
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
                    
            save_model(
                step, 
                model, optimizer, 
                avg_e_loss, val_loss, best_val_loss, 
                config.training.save_best_ckpt, out_dir,
                f'best_ckpt_{active_dataset_name}.pt',
                'new best',
                logger
            )

if __name__ == "__main__":
    main()
