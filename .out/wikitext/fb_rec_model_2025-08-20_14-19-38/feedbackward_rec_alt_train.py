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

from model import FeedforwardModel, FeedbackwardRecurrentModel
from data_utils import get_dataloaders
from utils import save_model, forward_process, feedforward_iterative_generation_loop, feedbackward_iterative_generation_loop, count_parameters, get_lr_schedule, roundup_multiple, average, relative_change, model_numel

@torch.no_grad()
def run_validation(ff_model, fb_model, dataloader, config, ctx, tokenizer, logger:Logger):
    """
    Runs validation loss calculation and generates a text sample.
    """
    ff_model.eval()
    fb_model.eval()
    
    # --- 1. Calculate Validation Loss ---
    total_loss = 0.0
    total_loss_lora = 0.0
    num_batches = 0

    for x0 in dataloader:
        x0 = x0.to(config.training.device)
        
        noisy_batch, masked_indices, p_mask = forward_process(
            x0, tokenizer.mask_token_id, config.diffusion.eps
        )

        with ctx:
            delta_weights = fb_model(noisy_batch)
            logits_lora = ff_model(noisy_batch, delta_weights)
            logits = ff_model(noisy_batch)
        
        # Calculate loss only on the masked tokens
        if masked_indices.any():
            token_loss = F.cross_entropy(logits[masked_indices], x0[masked_indices], reduction='none') / p_mask[masked_indices]
            loss = token_loss.sum() / (x0.shape[0] * x0.shape[1])
            
            token_loss_lora = F.cross_entropy(logits_lora[masked_indices], x0[masked_indices], reduction='none') / p_mask[masked_indices]
            loss_lora = token_loss_lora.sum() / (x0.shape[0] * x0.shape[1])
        else:
            loss = torch.tensor(0.0, device=config.training.device)
            
            loss_lora = torch.tensor(0.0, device=config.training.device)

        total_loss += loss.item()
        total_loss_lora += loss_lora.item()
        num_batches += 1
        if num_batches >= 50: # Limit validation batches for speed
            break
    
    val_loss = total_loss / num_batches if num_batches > 0 else 0.0
    val_loss_lora = total_loss_lora / num_batches if num_batches > 0 else 0.0

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
        ff_model, 
        context, 
        generation_mask, 
        config.diffusion.generation_steps, 
        tokenizer, 
        config.training.device
    )
    
    generated_tokens_lora = feedbackward_iterative_generation_loop(
        ff_model, 
        fb_model,
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
    generated_text_lora = tokenizer.decode(generated_tokens_lora[0].tolist())

    logger.log(f"\n--- Ground Truth ---\n{ground_truth_text}")
    logger.log(f"\n--- Provided Context ---\n{context_text}")
    logger.log(f"\n--- Model Generation Original ---\n{generated_text}\n")
    logger.log(f"\n--- Model Generation With LoRA ---\n{generated_text_lora}\n")

    ff_model.train()
    fb_model.train()
    return val_loss, val_loss_lora

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
    parser.add_argument('--fb_checkpoint_dir', type=str, default=None, help='Name of the feedforward checkpoint directory.')
    parser.add_argument('--fb_checkpoint_name', type=str, default=None, help='Name of the feedforward checkpoint file.')
    parser.add_argument('--ff_checkpoint_dir', type=str, default=None, help='Name of the feedback checkpoint directory.')
    parser.add_argument('--ff_checkpoint_name', type=str, default=None, help='Name of the feedback checkpoint file.')
    parser.add_argument('--lora_rank', type=int, help='What rank to use for LoRA.')
    parser.add_argument('--feedback_reverse', type=bool, help='Whether to create delta weight blocks from the feedback model for the feedforward model in the reverse order for the feedforward model.')
    args = parser.parse_args()
    
    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    # add active_dataset to config and ensure it is an allowed dataset
    assert args.active_dataset in config_dict['datasets'], f"Active dataset '{args.active_dataset}' not found in config."
    assert args.ff_checkpoint_dir and os.path.isdir(args.ff_checkpoint_dir), f"Feedforward checkpoint directory {args.ff_checkpoint_dir} does not exist."
    assert args.ff_checkpoint_name and os.path.exists(os.path.join(args.ff_checkpoint_dir, args.ff_checkpoint_name)), f"Feedforward checkpoint file {args.ff_checkpoint_name} does not exist."
    
    config_dict['active_dataset'] = args.active_dataset
    config_dict['model']['lora_rank'] = args.lora_rank
    config_dict['model']['feedback_reverse'] = args.feedback_reverse
    config_dict['training']['resume'] = args.resume
    config_dict['training']['checkpoint_dir'] = args.fb_checkpoint_dir
    config_dict['training']['checkpoint_name'] = args.fb_checkpoint_name
    config_dict['feedforward'] = {}
    config_dict['feedforward']['checkpoint_dir'] = args.ff_checkpoint_dir
    config_dict['feedforward']['checkpoint_name'] = args.ff_checkpoint_name
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
        run_prefix = config.training.fb_rec_run_name
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
        run_name_with_timestamp = f"{config.training.fb_rec_run_name}_{timestamp}"
        out_dir = os.path.join(config.training.out_dir, config.active_dataset, run_name_with_timestamp)
        os.makedirs(out_dir, exist_ok=True)
        logger.log(f"Starting new run. Checkpoints and logs will be saved in: {out_dir}")

        # --- Save code and config for reproducibility ---
        logger.log("Saving configuration and code to output directory...")
        #shutil.copy2(args.config, os.path.join(out_dir, 'config.json'))
        shutil.copy2(args.config, os.path.join(out_dir, os.path.basename(args.config)))
        source_files_to_save = [
            'model.py', 'data_utils.py', 'utils.py', 'feedbackward_rec_alt_train.py', 'custom_logger.py'
        ]
        for file_name in source_files_to_save:
            shutil.copy2(file_name, os.path.join(out_dir, file_name))
        logger.log("Done saving files.")
        
    # Setup log and csv files
    logger.set_log_file(os.path.join(out_dir, config.training.log_file_name))
    # TODO update headers
    logger.set_csv_file(os.path.join(out_dir, config.training.csv_file_name), header=["type", "epoch", "loss", "loss_lora", "rel_change", "lr"])
    
    # --- Data Loading ---
    train_loader, val_loader, tokenizer = get_dataloaders(dataset_config, config.training, logger)
    
    # Dynamically set the vocab_size from the tokenizer after it has been configured.
    # This is crucial because special tokens (like [MASK]) could be added, changing the vocab size.
    # Also need to ensure that the vocab_size is a mutliple of 128 for performance
    dataset_config.vocab_size = len(tokenizer)
    dataset_config.vocab_size = roundup_multiple(dataset_config.vocab_size, 128)
    logger.log(f"Vocab size: {dataset_config.vocab_size}")

    # --- Model and Optimizer ---
    ff_model = FeedforwardModel(config.model, dataset_config)
    ff_model.to(device=device, dtype=dtype)
    fb_model = FeedbackwardRecurrentModel(config.model, dataset_config)
    fb_model.to(device=device, dtype=dtype)
    fb_optimizer = torch.optim.AdamW(fb_model.parameters(), lr=config.training.learning_rate)
    
    # --- Load state from checkpoint for ff_model
    ff_checkpoint = torch.load(os.path.join(config.feedforward.checkpoint_dir, config.feedforward.checkpoint_name), map_location=device)
    ff_model_state_dict = ff_checkpoint['model_state_dict']
    unwrapped_ff_state_dict = {k.replace('_orig_mod.', ''): v for k, v in ff_model_state_dict.items()}
    ff_model.load_state_dict(unwrapped_ff_state_dict)
    del ff_checkpoint, ff_model_state_dict, unwrapped_ff_state_dict
    
    # --- Load state from checkpoint if resuming ---
    step = 0
    best_val_loss = float('inf')
    if checkpoint_to_load:
        checkpoint = torch.load(checkpoint_to_load, map_location=device)

        # The state dict may have been saved from a compiled model, which prefixes keys
        # with "_orig_mod.". We strip this prefix for robust loading.
        model_state_dict = checkpoint['model_state_dict']
        unwrapped_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict.items()}
        fb_model.load_state_dict(unwrapped_state_dict)

        fb_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
    if config.training.compile:
        logger.log("Compiling the model... (requires PyTorch 2.0+)")
        fb_model = torch.compile(fb_model)
        ff_model = torch.compile(ff_model)
        
    # --- Training Loop ---
    num_epochs = getattr(config.training.epochs, active_dataset_name)
    logger.log(f"Starting training for {num_epochs} epochs on dataset '{active_dataset_name}'...")

    total_training_steps = num_epochs * len(train_loader)
    logger.log(f"Total training steps: {total_training_steps}")
    
    logger.log(f"Total number of feedbackward model parameters: {model_numel(fb_model)}")
    
    warmup_steps = 0
    if config.training.decay_lr:
        warmup_steps = int(config.training.warmup_ratio * total_training_steps)
        logger.log(f"Warmup steps: {warmup_steps}")

    start_epoch = step // len(train_loader) if len(train_loader) > 0 else 0
    
    for epoch in range(start_epoch, num_epochs):
        epoch_ff_loss = []
        epoch_ff_loss_lora = []
        
        epoch_start_time = time.time()
        for i, x0 in enumerate(train_loader):            
            if epoch == start_epoch and i < (step % len(train_loader) if len(train_loader) > 0 else 0):
                continue
            # Determine and set the learning rate for this iteration
            lr = get_lr_schedule(step, config, total_training_steps, warmup_steps)
            for param_group in fb_optimizer.param_groups:
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
                fb_noise_batch = noisy_batch.detach()
                fb_delta_weights = fb_model(fb_noise_batch)
                
                x0 = x0.detach()
                masked_indices = masked_indices.detach()
                p_mask = p_mask.detach()
                
                ff_noise_batch = noisy_batch.detach()
                
                ff_logits_lora = ff_model(ff_noise_batch, fb_delta_weights)
                with torch.no_grad():
                    ff_logits = ff_model(ff_noise_batch)

                # Calculate loss only on the masked tokens, weighted by inverse probability
                if masked_indices.any():
                    with torch.no_grad():
                        ff_token_loss = F.cross_entropy(ff_logits[masked_indices], x0[masked_indices], reduction='none') / p_mask[masked_indices]
                        ff_loss = ff_token_loss.sum() / (x0.shape[0] * x0.shape[1])
                    
                    ff_token_loss_lora = F.cross_entropy(ff_logits_lora[masked_indices], x0[masked_indices], reduction='none') / p_mask[masked_indices]
                    ff_loss_lora = ff_token_loss_lora.sum() / (x0.shape[0] * x0.shape[1])
                else:
                    # Handle case where no tokens are masked (can happen with low p_mask)
                    ff_loss = torch.tensor(0.0, device=device, requires_grad=False)
                    
                    ff_loss_lora = torch.tensor(0.0, device=device, requires_grad=True)
                    
                ff_loss_lora.backward()
                if config.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(fb_model.parameters(), config.training.grad_clip)
                fb_optimizer.step()
                fb_optimizer.zero_grad(set_to_none=True)
                ff_model.zero_grad(set_to_none=True)
                
                just_resumed = False

                epoch_ff_loss.append(ff_loss.item())
                epoch_ff_loss_lora.append(ff_loss_lora.item())
                step += 1
                
                del ff_logits_lora, ff_token_loss_lora, ff_loss_lora
                del ff_logits, ff_token_loss, ff_loss
                

            #del param_group
            del x0, noisy_batch, masked_indices, p_mask, fb_noise_batch, ff_noise_batch, fb_delta_weights

        epoch_duration = time.time() - epoch_start_time
        logger.log(f"Epoch {epoch+1} finished in {epoch_duration:.2f} seconds.")

        avg_e_ff_loss = average(epoch_ff_loss)
        avg_e_ff_loss_lora = average(epoch_ff_loss_lora)
        train_rel_change = relative_change(avg_e_ff_loss_lora, avg_e_ff_loss)
        logger.log(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_e_ff_loss:.4f} | Loss w/ LoRA: {avg_e_ff_loss_lora:.4f} | Rel Change: {train_rel_change:.4f} | LR: {lr:.6f}")
        logger.log_csv(["train", epoch+1, avg_e_ff_loss, avg_e_ff_loss_lora, train_rel_change, lr])
        
        val_loss, val_loss_lora = run_validation(ff_model, fb_model, val_loader, config, ctx, tokenizer, logger)
        val_rel_change = relative_change(val_loss_lora, val_loss)
        logger.log(f"Epoch {epoch+1}/{num_epochs} | Loss: {val_loss:.4f} | Loss w/ LoRA: {val_loss_lora:.4f} | Rel Change: {val_rel_change:.4f} | LR: {lr:.6f}")
        logger.log_csv(["val", epoch+1, val_loss, val_loss_lora, val_rel_change, lr])

        if (epoch + 1) % config.training.save_epoch_interval == 0:
            # save epoch model and best model based off of validation
            save_model(
                step,
                fb_model, fb_optimizer, 
                avg_e_ff_loss_lora, val_loss_lora, best_val_loss,
                config.training.save_epoch, out_dir,
                f'epoch_ckpt_{active_dataset_name}_{epoch+1}.pt',
                'epoch checkpoint',
                logger
            )
            
        if val_loss_lora < best_val_loss:
            best_val_loss = val_loss_lora
                    
            save_model(
                step, 
                fb_model, fb_optimizer, 
                avg_e_ff_loss_lora, val_loss_lora, best_val_loss, 
                config.training.save_best_ckpt, out_dir,
                f'best_ckpt_{active_dataset_name}.pt',
                'new best',
                logger
            )

if __name__ == "__main__":
    main()
