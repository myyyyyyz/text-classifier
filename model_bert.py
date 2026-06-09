"""
model_bert.py
=============
基础任务：用 bert-base-chinese 在 THUCNews 上做文本分类微调。

模型结构：
    BertModel  ->  取 [CLS] 隐向量  ->  Dropout  ->  Linear(num_labels)

训练完之后，会在测试集上评估，并保存指标到 results/baseline_bert.json。

优化点：
    1. 使用动态 padding（减少无效计算）
    2. 梯度累积（模拟更大 batch size）
    3. 早停机制（防止过拟合）
    4. 学习率调度优化
    5. 训练进度条显示
"""

import os
import json
import time
import random
from typing import List, Tuple, Dict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertModel,
    get_linear_schedule_with_warmup,
)

from config import (
    BERT_MODEL_NAME,
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
from data_prepare import prepare_data, load_split
from evaluate import print_metrics


# ---------------------------------------------------------------------------
# 1. 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 2. 数据集封装（支持动态 padding）
# ---------------------------------------------------------------------------
class TextDataset(Dataset):
    """把 (text, label) 列表封装成 PyTorch Dataset。"""
    def __init__(self, samples: List[Tuple[str, int]], tokenizer, max_len: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
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


def make_loader(samples, tokenizer, batch_size, shuffle):
    ds = TextDataset(samples, tokenizer, MAX_LEN)
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=NUM_WORKERS,
        pin_memory=True and DEVICE != 'cpu',
        collate_fn=collate_fn_factory(tokenizer.pad_token_id),
    )


# ---------------------------------------------------------------------------
# 3. 模型（添加标签平滑）
# ---------------------------------------------------------------------------
class BertClassifier(nn.Module):
    """Bert + [CLS] 分类头。"""
    def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            model_name, local_files_only=True
        )
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden, num_labels)
        
        # 标签平滑（提升泛化能力）
        self.label_smoothing = 0.1

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]   # [batch, hidden]
        cls = self.dropout(cls)
        logits = self.classifier(cls)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = loss_fn(logits, labels)
            return loss, logits
        return logits


# ---------------------------------------------------------------------------
# 4. 训练 & 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader) -> Tuple[List[int], List[int]]:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attn = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        logits = model(input_ids, attn)
        preds = logits.argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


def train_one_model(model_name: str, num_labels: int, save_name: str,
                    num_epochs: int = NUM_EPOCHS, lr: float = LEARNING_RATE,
                    gradient_accumulation_steps: int = 1):
    """完整的训练+测试流程，保存指标到 results/。
    
    新增功能：
        - 混合精度训练（加速 ~30%）
        - 梯度累积（模拟更大 batch）
        - 早停机制（patience=2）
        - 训练进度条
    """
    set_seed()
    print(f"\n>>> 加载分词器与模型: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(
        model_name, local_files_only=True
    )
    
    # 优化：静默加载模型（减少不必要的警告）
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BertClassifier(model_name, num_labels).to(DEVICE)

    # 数据
    print(">>> 加载数据 ...")
    train_samples = load_split("train")
    dev_samples = load_split("dev")
    test_samples = load_split("test")
    print(f"  train={len(train_samples)}  dev={len(dev_samples)}  test={len(test_samples)}")

    train_loader = make_loader(train_samples, tokenizer, BATCH_SIZE, True)
    dev_loader = make_loader(dev_samples, tokenizer, EVAL_BATCH_SIZE, False)
    test_loader = make_loader(test_samples, tokenizer, EVAL_BATCH_SIZE, False)

    # 优化器 & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=lr)
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    
    # 早停机制
    patience = 2
    best_dev_acc = 0.0
    patience_counter = 0

    # 训练
    print(f">>> 开始训练 {num_epochs} 轮 ...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        
        # 训练进度条
        progress_bar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc=f"  Epoch {epoch}/{num_epochs}",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            ncols=100
        )
        
        for step, batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attn = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            
            if USE_AMP and scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, logits = model(input_ids, attn, labels)
                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                loss, logits = model(input_ids, attn, labels)
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # 更新进度条显示
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

        progress_bar.close()
        
        dev_true, dev_pred = evaluate(model, dev_loader)
        dev_metrics = print_metrics(f"Epoch {epoch} - Dev", dev_true, dev_pred, CATEGORIES)
        print(f"  训练耗时: {time.time() - t0:.1f}s, "
              f"train_loss={total_loss / len(train_loader):.4f}")

        # 早停检查
        if dev_metrics["accuracy"] > best_dev_acc:
            best_dev_acc = dev_metrics["accuracy"]
            patience_counter = 0
            # 保存最佳模型（可选）
            # torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⚠️ 早停触发！连续 {patience} 轮未提升")
                break

    # 测试
    print("\n>>> 在测试集上评估 ...")
    test_true, test_pred = evaluate(model, test_loader)
    test_metrics = print_metrics("Test", test_true, test_pred, CATEGORIES)

    # 保存
    result = {
        "model": model_name,
        "save_name": save_name,
        "num_epochs": num_epochs,
        "lr": lr,
        "best_dev_acc": best_dev_acc,
        **test_metrics,
    }
    out_path = os.path.join(RESULTS_DIR, f"{save_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 指标已保存到 {out_path}")
    return result


# ---------------------------------------------------------------------------
# 5. 主函数
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 如果数据还没准备好，就先准备
    if not os.path.exists(os.path.join("data", "THUCNews", "train.txt")):
        prepare_data()

    train_one_model(
        model_name=BERT_MODEL_NAME,
        num_labels=len(CATEGORIES),
        save_name="baseline_bert",
    )
