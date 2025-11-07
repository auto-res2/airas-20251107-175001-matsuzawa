from typing import Tuple, List

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
from omegaconf import DictConfig

CACHE_DIR = ".cache/"

# -----------------------------------------------------------------------------
# Token-classification helpers
# -----------------------------------------------------------------------------

def _tokenize_align_labels(examples, tokenizer, label_all_tokens: bool):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=tokenizer.model_max_length,
    )
    labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_seq[word_idx])
            else:
                label_ids.append(label_seq[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized


# -----------------------------------------------------------------------------
# Data-loading entry point
# -----------------------------------------------------------------------------

def build_dataloaders(cfg: DictConfig, cache_dir: str):
    ds_cfg = cfg.dataset
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.host_model_name, cache_dir=cache_dir, use_fast=True
    )

    task_name = ds_cfg.name.lower()

    # Trial-mode dataset reduction -------------------------------------------
    limit = 64 if getattr(cfg, "mode", "full") == "trial" else None

    if task_name == "conll2003":
        raw = load_dataset(ds_cfg.hf_id, cache_dir=cache_dir)
        train_ds = raw[ds_cfg.splits.train]
        val_ds = raw[ds_cfg.splits.validation]
        if limit is not None:
            train_ds = train_ds.select(range(min(len(train_ds), limit)))
            val_ds = val_ds.select(range(min(len(val_ds), limit)))
        train_ds = train_ds.map(
            lambda ex: _tokenize_align_labels(ex, tokenizer, False),
            batched=True,
            remove_columns=train_ds.column_names,
        )
        val_ds = val_ds.map(
            lambda ex: _tokenize_align_labels(ex, tokenizer, False),
            batched=True,
            remove_columns=val_ds.column_names,
        )
        collator = DataCollatorForTokenClassification(tokenizer)
        # Try to get label list from dataset features, or use standard CoNLL-2003 labels
        try:
            label_list = raw["train"].features["ner_tags"].feature.names
        except AttributeError:
            # Standard CoNLL-2003 NER labels (IOB2 format)
            label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    else:
        raw = load_dataset(ds_cfg.hf_id, cache_dir=cache_dir)
        train_ds = raw[ds_cfg.splits.train]
        val_ds = raw[ds_cfg.splits.validation]
        if limit is not None:
            train_ds = train_ds.select(range(min(len(train_ds), limit)))
            val_ds = val_ds.select(range(min(len(val_ds), limit)))

        def tok_fn(ex):
            out = tokenizer(ex["text"], truncation=True, max_length=ds_cfg.max_length)
            out["labels"] = ex["label"]
            return out

        train_ds = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
        val_ds = val_ds.map(tok_fn, batched=True, remove_columns=val_ds.column_names)
        collator = DataCollatorWithPadding(tokenizer)
        label_list = raw["train"].features["label"].names

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        collate_fn=collator,
    )
    return train_loader, val_loader, label_list