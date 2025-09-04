import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

from custom_logger import Logger

def forward_process(input_ids, mask_token_id, eps=1e-3):
    """
    Applies probabilistic masking to a batch of token IDs, as per the LLaDA pre-training.

    Args:
        input_ids (torch.Tensor): The original token IDs (B, L).
        mask_token_id (int): The ID for the [MASK] token.
        eps (float): A small epsilon to ensure p_mask is never zero.

    Returns:
        tuple: (noisy_batch, masked_indices, p_mask)
            - noisy_batch (torch.Tensor): The input_ids with some tokens replaced by [MASK].
            - masked_indices (torch.Tensor): A boolean tensor indicating which tokens were masked.
            - p_mask (torch.Tensor): The probability of masking used for each token.
    """
    b, l = input_ids.shape
    # Sample a random "time" t for each sequence in the batch
    t = torch.rand(b, device=input_ids.device)
    # Calculate the masking probability based on t
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    # Determine which tokens to mask based on the probability
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    
    # Create the noisy batch by replacing tokens with the MASK token ID
    noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)
    
    return noisy_batch, masked_indices, p_mask

@torch.no_grad()
def feedforward_iterative_generation_loop(model, context, context_mask, num_steps, tokenizer, device):
    """
    Iterative generation for a masked language model (e.g., Mask-Predict).
    Fills in [MASK] tokens over a series of steps.
    
    Args:
        model: The masked language model.
        context (torch.Tensor): The context tensor with known tokens (B, T).
        context_mask (torch.Tensor): A boolean mask where True indicates a token to be generated.
        num_steps (int): The number of generation steps.
        tokenizer: The tokenizer, used to get MASK and PAD token IDs.
        device: The device to run on.
    
    Returns:
        torch.Tensor: The generated token indices.
    """
    model.eval()
    
    B, T_seq = context.shape
    mask_token_id = tokenizer.mask_token_id
    
    # Start with the context and fill the generation area with [MASK] tokens
    unknown_tokens = torch.full_like(context, mask_token_id)
    generated_sequence = torch.where(context_mask, unknown_tokens, context)
    
    num_total_to_generate = context_mask.sum()

    # Use a tqdm progress bar only if there's more than one step
    for i in tqdm(range(num_steps), desc="Iterative Generation", total=num_steps, leave=False, disable=(num_steps==1)):
        # Find which tokens are currently [MASK]
        mask_indices = (generated_sequence == mask_token_id)
        if not mask_indices.any():
            break # Stop if no masks are left

        # Get model predictions for the current sequence
        logits = model(generated_sequence)
        masked_logits = logits[mask_indices]
        masked_probs = F.softmax(masked_logits, dim=-1)
        
        # Get the confidence (max probability) and the predicted token for each masked position
        confidence, predictions = torch.max(masked_probs, dim=-1)
        
        # Determine how many tokens to unmask in this step based on a cosine schedule (MaskGIT-style)
        ratio_to_keep_masked = math.cos(math.pi / 2 * (i + 1) / num_steps)
        num_to_keep_masked = math.floor(ratio_to_keep_masked * num_total_to_generate)
        num_to_unmask = mask_indices.sum() - num_to_keep_masked
        
        if num_to_unmask > 0:
            indices_to_unmask = torch.topk(confidence, k=min(num_to_unmask, len(confidence))).indices
            flat_mask_indices = torch.where(mask_indices.view(-1))[0]
            flat_indices_to_update = flat_mask_indices[indices_to_unmask]
            predictions_to_update = predictions[indices_to_unmask]

            generated_sequence.view(-1)[flat_indices_to_update] = predictions_to_update

    # Final pass to fill any remaining masks with the most likely token
    mask_indices = (generated_sequence == mask_token_id)
    if mask_indices.any():
        logits = model(generated_sequence)
        predictions = torch.argmax(logits, dim=-1)
        generated_sequence = torch.where(mask_indices, predictions, generated_sequence)

    # Ensure the original context is perfectly preserved
    generated_sequence = torch.where(context_mask, generated_sequence, context)
    
    return generated_sequence

@torch.no_grad()
def feedbackward_iterative_generation_loop(ff_model, fb_model, context, context_mask, num_steps, tokenizer, device):
    """
    Iterative generation for a masked language model (e.g., Mask-Predict).
    Fills in [MASK] tokens over a series of steps.
    
    Args:
        ff_model: The feedforward masked language model.
        fb_model: The feedbackward masked language hypernetwork model.
        context (torch.Tensor): The context tensor with known tokens (B, T).
        context_mask (torch.Tensor): A boolean mask where True indicates a token to be generated.
        num_steps (int): The number of generation steps.
        tokenizer: The tokenizer, used to get MASK and PAD token IDs.
        device: The device to run on.
    
    Returns:
        torch.Tensor: The generated token indices.
    """
    ff_model.eval()
    fb_model.eval()
    
    B, T_seq = context.shape
    mask_token_id = tokenizer.mask_token_id
    
    # Start with the context and fill the generation area with [MASK] tokens
    unknown_tokens = torch.full_like(context, mask_token_id)
    generated_sequence = torch.where(context_mask, unknown_tokens, context)
    
    num_total_to_generate = context_mask.sum()

    # Use a tqdm progress bar only if there's more than one step
    for i in tqdm(range(num_steps), desc="Iterative Generation", total=num_steps, leave=False, disable=(num_steps==1)):
        # Find which tokens are currently [MASK]
        mask_indices = (generated_sequence == mask_token_id)
        if not mask_indices.any():
            break # Stop if no masks are left

        # Get model predictions for the current sequence
        delta_weights = fb_model(generated_sequence)
        logits = ff_model(generated_sequence, delta_weights)
        masked_logits = logits[mask_indices]
        masked_probs = F.softmax(masked_logits, dim=-1)
        
        # Get the confidence (max probability) and the predicted token for each masked position
        confidence, predictions = torch.max(masked_probs, dim=-1)
        
        # Determine how many tokens to unmask in this step based on a cosine schedule (MaskGIT-style)
        ratio_to_keep_masked = math.cos(math.pi / 2 * (i + 1) / num_steps)
        num_to_keep_masked = math.floor(ratio_to_keep_masked * num_total_to_generate)
        num_to_unmask = mask_indices.sum() - num_to_keep_masked
        
        if num_to_unmask > 0:
            indices_to_unmask = torch.topk(confidence, k=min(num_to_unmask, len(confidence))).indices
            flat_mask_indices = torch.where(mask_indices.view(-1))[0]
            flat_indices_to_update = flat_mask_indices[indices_to_unmask]
            predictions_to_update = predictions[indices_to_unmask]

            generated_sequence.view(-1)[flat_indices_to_update] = predictions_to_update

    # Final pass to fill any remaining masks with the most likely token
    mask_indices = (generated_sequence == mask_token_id)
    if mask_indices.any():
        delta_weights = fb_model(generated_sequence)
        logits = ff_model(generated_sequence, delta_weights)
        predictions = torch.argmax(logits, dim=-1)
        generated_sequence = torch.where(mask_indices, predictions, generated_sequence)

    # Ensure the original context is perfectly preserved
    generated_sequence = torch.where(context_mask, generated_sequence, context)
    
    return generated_sequence

def count_parameters(model):
    """
    Counts the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_lr_schedule(step, config, total_steps, warmup_steps):
    """
    Calculates the learning rate if config has decay_lr set to True.
    Then:
        Calculates the learning rate for a given step based on a cosine decay schedule with warmup.
    Else:
        Returns the fixed learning rate.
    """
    if not config.training.decay_lr:
        return config.training.learning_rate
    
    # 1) Linear warmup for warmup_steps
    if step < warmup_steps:
        return config.training.learning_rate * step / warmup_steps
    
    # 2) If step > total_steps, return min learning rate
    if step > total_steps:
        return config.training.min_lr
        
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    assert 0. <= decay_ratio <= 1.
    # coeff starts at 1 and goes to 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    
    return config.training.min_lr + coeff * (config.training.learning_rate - config.training.min_lr)

@torch.no_grad()
def save_model(step, model, optimizer, loss, val_loss, best_val_loss, perform_save, out_dir, ckpt_name, save_type, logger:Logger):
    if not perform_save:
        return
    
    # Get the original model to save a clean state dict, free of compile-time artifacts.
    unwrapped_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    checkpoint = {
        'step': step,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item() if torch.is_tensor(loss) else loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
    }

    # Create a copy of the state dicts on the CPU for portability.
    # This is an out-of-place operation, so the live model/optimizer on the GPU are unaffected.
    cpu_checkpoint = {
        **checkpoint,
        'model_state_dict': {k: v.cpu() for k, v in checkpoint['model_state_dict'].items()},
        'optimizer_state_dict': {
            'state': {k_state: {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in v_state.items()} for k_state, v_state in checkpoint['optimizer_state_dict']['state'].items()},
            'param_groups': checkpoint['optimizer_state_dict']['param_groups']
        }
    }

    checkpoint_path = os.path.join(out_dir, ckpt_name)
    torch.save(cpu_checkpoint, checkpoint_path)
    logger.log(f"Saved {save_type} checkpoint to {checkpoint_path}")

def roundup_multiple(n, base):
    return ((n + (base - 1)) // base) * base

def average(lst):
    return sum(lst) / len(lst)

def relative_change(final, initial):
    return (final - initial) / abs(initial)

def model_numel(model):
    return sum(p.numel() for p in model.parameters())