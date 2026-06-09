"""
model_compare.py
================
对比 BERT / RoBERTa / ELECTRA 三种中文预训练模型在相同数据上的分类效果。

为了避免重复造轮子，这里直接复用 model_bert.py 的训练函数，
只是在调用时切换 model_name 即可。
"""

import os
import json
import time
import random
import warnings
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertModel,
    RobertaModel,
    ElectraModel,
    get_linear_schedule_with_warmup,
)

from config import (
    BERT_MODEL_NAME,
    ROBERTA_MODEL_NAME,
    ELECTRA_MODEL_NAME,
    MAX_LEN,
    BATCH_SIZE,
    EVAL_BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    DEVICE,
    SEED,
    CATEGORIES,
    RESULTS_DIR,
    MODEL_CACHE_DIR,
    USE_AMP,
    NUM_WORKERS,
)
from data_prepare import load_split, prepare_data
from evaluate import print_metrics


# 兼容不同模型的 AutoModel
def build_backbone(model_name: str):
    name = model_name.lower()
    if "roberta" in name:
        return RobertaModel.from_pretrained(
            model_name, local_files_only=True
        )
    if "electra" in name:
        return ElectraModel.from_pretrained(
            model_name, local_files_only=True
        )
    return BertModel.from_pretrained(
        model_name, local_files_only=True
    )


class UniversalClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.backbone = build_backbone(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        logits = self.classifier(cls)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fn(logits, labels)
            return loss, logits
        return logits


# 数据 / 训练过程 与 model_bert.py 几乎一致，这里单独再写一遍以保持独立
class TextDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_fn_factory(pad_id: int):
    """工厂函数：创建 collate_fn（避免 lambda 序列化问题）。"""
    def collate_fn(batch):
        max_len = max(b["input_ids"].size(0) for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.zeros((len(batch),), dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["input_ids"].size(0)
            input_ids[i, :n] = b["input_ids"]
            attn[i, :n] = b["attention_mask"]
            labels[i] = b["label"]
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "label": labels,
        }
    return collate_fn


def make_loader(samples, tokenizer, bs, shuffle):
    return DataLoader(
        TextDataset(samples, tokenizer, MAX_LEN),
        batch_size=bs, shuffle=shuffle, num_workers=NUM_WORKERS,
        pin_memory=True and DEVICE != 'cpu',
        collate_fn=collate_fn_factory(tokenizer.pad_token_id),
    )


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        logits = model(input_ids, attn)
        preds = logits.argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


def train_one(model_name: str, save_name: str, num_epochs=NUM_EPOCHS):
    set_seed()
    print(f"\n========== 对比实验: {model_name} ==========")
    tokenizer = BertTokenizer.from_pretrained(
        model_name, local_files_only=True
    )
    
    # 静默加载模型
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = UniversalClassifier(model_name, num_labels=len(CATEGORIES)).to(DEVICE)

    train_samples = load_split("train")
    dev_samples = load_split("dev")
    test_samples = load_split("test")
    train_loader = make_loader(train_samples, tokenizer, BATCH_SIZE, True)
    dev_loader = make_loader(dev_samples, tokenizer, EVAL_BATCH_SIZE, False)
    test_loader = make_loader(test_samples, tokenizer, EVAL_BATCH_SIZE, False)

    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=LEARNING_RATE)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    
    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    best_dev_acc = 0.0
    patience = 2
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        
        # 训练进度条
        progress_bar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch}/{num_epochs}",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            ncols=100
        )
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attn = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            
            if USE_AMP and scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, logits = model(input_ids, attn, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, logits = model(input_ids, attn, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{total_loss/(progress_bar.n+1):.4f}'})
        
        progress_bar.close()
        
        dev_true, dev_pred = evaluate(model, dev_loader)
        m = print_metrics(
            f"[{save_name}] Epoch {epoch} - Dev", dev_true, dev_pred, CATEGORIES
        )
        if m["accuracy"] > best_dev_acc:
            best_dev_acc = m["accuracy"]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⚠️ 早停触发！")
                break
        
        print(f"  耗时={time.time() - t0:.1f}s  loss={total_loss/len(train_loader):.4f}")

    test_true, test_pred = evaluate(model, test_loader)
    test_metrics = print_metrics(
        f"[{save_name}] Test", test_true, test_pred, CATEGORIES
    )

    out = {"model": model_name, "save_name": save_name, "best_dev_acc": best_dev_acc, **test_metrics}
    out_path = os.path.join(RESULTS_DIR, f"{save_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] {out_path}")
    return out


if __name__ == "__main__":
    if not os.path.exists(os.path.join("data", "THUCNews", "train.txt")):
        prepare_data()

    train_one(BERT_MODEL_NAME, "compare_bert")
    train_one(ROBERTA_MODEL_NAME, "compare_roberta")
    train_one(ELECTRA_MODEL_NAME, "compare_electra")
