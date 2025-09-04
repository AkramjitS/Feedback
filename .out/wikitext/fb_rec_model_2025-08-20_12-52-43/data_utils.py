import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial

class ChunkedDocumentDataset(Dataset):
    """
    A PyTorch Dataset that takes tokenized documents, chunks them to a specified
    block size, and serves these chunks. It respects document boundaries.
    The last chunk of a document can be smaller than block_size.
    """
    def __init__(self, tokenized_documents, block_size):
        super().__init__()
        self.chunks = []
        for doc in tokenized_documents:
            # Don't process empty documents
            if doc.size(0) == 0:
                continue
            # Split document into chunks of size `block_size`.
            self.chunks.extend(torch.split(doc, block_size))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

def pad_collate_fn(batch, pad_token_id):
    """
    A collate function to pad sequences in a batch to the same length.
    This is used for the DocumentDataset where documents have variable lengths.
    """
    max_len = max(len(x) for x in batch)
    padded_batch = []
    for x in batch:
        padded_batch.append(F.pad(x, (0, max_len - len(x)), 'constant', pad_token_id))
    return torch.stack(padded_batch)

def get_dataloaders(dataset_config, training_config, logger):
    """
    Loads a dataset and tokenizer from Hugging Face, tokenizes the data,
    splits it, and returns dataloaders.
    
    Args:
        dataset_config (SimpleNamespace): The specific config for the active dataset.
        training_config (SimpleNamespace): The main training config.
    """
    # --- 1. Load Tokenizer ---
    logger.log(f"Loading tokenizer: {dataset_config.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(dataset_config.tokenizer_name)
    
    # Add MASK and PAD tokens if they don't exist
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # --- 2. Load and Tokenize Dataset ---
    logger.log(f"Loading and tokenizing dataset: {dataset_config.name}...")
    dataset = load_dataset(dataset_config.name, dataset_config.config_name)
    
    # To avoid the tokenizer warning about long sequences, we process each text
    # individually and then concatenate the token IDs, rather than creating one
    # giant string first. We add an EOS token between documents as a separator.
    def tokenize_per_document(dataset_split, format_fn):
        tokenized_docs = []
        separator_tokens = [tokenizer.eos_token_id]
        for item in dataset_split:
            text = format_fn(item)
            if text and text.strip():
                tokens = tokenizer.encode(text, add_special_tokens=False)
                tokenized_docs.append(torch.tensor(tokens + separator_tokens, dtype=torch.long))
        return tokenized_docs

    if dataset_config.name == 'gsm8k':
        format_fn = lambda item: f"Question: {item['question']}\nAnswer: {item['answer']}"
        train_docs = tokenize_per_document(dataset['train'], format_fn)
        val_docs = tokenize_per_document(dataset['test'], format_fn) # Using test split for validation
    elif dataset_config.name == 'wikitext':
        format_fn = lambda item: item['text']
        train_docs = tokenize_per_document(dataset['train'], format_fn)
        val_docs = tokenize_per_document(dataset['validation'], format_fn)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_config.name}")

    logger.log(f"Train data has {len(train_docs)} documents.")
    logger.log(f"Validation data has {len(val_docs)} documents.")

    # --- 3. Create Datasets and DataLoaders ---
    train_dataset = ChunkedDocumentDataset(train_docs, dataset_config.block_size)
    val_dataset = ChunkedDocumentDataset(val_docs, dataset_config.block_size)
    
    logger.log(f"Created {len(train_dataset):,} training chunks.")
    logger.log(f"Created {len(val_dataset):,} validation chunks.")

    pad_fn = partial(pad_collate_fn, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=pad_fn)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=pad_fn)

    # Return the tokenizer object itself for decoding samples later
    return train_loader, val_loader, tokenizer
