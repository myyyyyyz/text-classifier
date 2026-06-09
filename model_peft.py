"""
model_peft.py
=============
参数高效微调 (PEFT)。

实现两种典型方法：
    1) LoRA       : 在 BERT attention 的 Q/V 矩阵上加低秩分解矩阵
    2) P-Tuning v1: 在输入 embedding 前添加可学习的"软提示"向量

两者都"冻结"预训练模型主体，只训练少量新增参数。
最终会输出和"全量微调"基线的对比结果。
"""

import os
import json
import time
import warnings
from typing import List, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertModel,
    get_linear_schedule_with_warmup,
)

from utils import set_seed, count_trainable, count_total

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
from data_prepare import load_split, prepare_data
from evaluate import print_metrics


# ---------------------------------------------------------------------------
# 1. LoRA 实现（优化：支持更多层）
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """
    把 nn.Linear 替换为 y = x W^T + b + (alpha/r) * (x A^T) (B^T)
    其中 A, B 是可学习低秩矩阵，shape: A=[in, r], B=[r, out]
    """
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 32):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.A = nn.Parameter(torch.zeros(in_f, r))
        self.B = nn.Parameter(torch.randn(r, out_f) * 0.01)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.scale


def inject_lora(bert: BertModel, r: int = 8, alpha: int = 32):
    """把 BERT 所有 attention 的 Q/K/V 投影替换为 LoRALinear。"""
    replaced = 0
    for layer in bert.encoder.layer:
        attn = layer.attention.self
        attn.query = LoRALinear(attn.query, r=r, alpha=alpha)
        attn.key = LoRALinear(attn.key, r=r, alpha=alpha)
        attn.value = LoRALinear(attn.value, r=r, alpha=alpha)
        replaced += 3
    print(f"  - LoRA 已注入 {replaced} 个线性层（Q/K/V）")
    return bert


class LoRAClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            model_name, local_files_only=True
        )
        # 冻结 BERT 主体
        for p in self.bert.parameters():
            p.requires_grad = False
        # 注入 LoRA
        inject_lora(self.bert, r=r, alpha=alpha)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return self.classifier(cls)


# ---------------------------------------------------------------------------
# 2. P-Tuning v1 实现
# ---------------------------------------------------------------------------
class PTuningClassifier(nn.Module):
    """
    P-Tuning v1：在 embedding 序列最前面插入 K 个可学习的"prompt token"。
    训练时只更新 prompt 嵌入 + 分类头。
    """
    def __init__(self, model_name: str, num_labels: int, n_prompt_tokens: int = 20):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            model_name, local_files_only=True
        )
        # 冻结 BERT 主体
        for p in self.bert.parameters():
            p.requires_grad = False
        self.embed = self.bert.embeddings
        hidden = self.bert.config.hidden_size
        # 可学习 prompt embedding: [K, H]
        self.prompt_embeds = nn.Parameter(
            torch.randn(n_prompt_tokens, hidden) * 0.02
        )
        self.n_prompt = n_prompt_tokens
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        # 自己算 token + position embedding，再在最前面拼上 prompt
        embeds = self.embed(input_ids=input_ids)  # [B, L, H]
        bsz = embeds.size(0)
        # broadcast prompt
        prompt = self.prompt_embeds.unsqueeze(0).expand(bsz, -1, -1)
        # 拼接: [B, L+K, H]
        new_embeds = torch.cat([prompt, embeds], dim=1)
        # attention_mask 也要补 K 个 1
        new_attn = torch.cat([
            torch.ones(bsz, self.n_prompt, device=attention_mask.device, dtype=attention_mask.dtype),
            attention_mask,
        ], dim=1)
        # 简化：position_id 重新生成
        L = new_embeds.size(1)
        position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        new_embeds = new_embeds + self.embed.position_embeddings(position_ids)
        new_embeds = self.embed.LayerNorm(new_embeds)
        new_embeds = self.embed.dropout(new_embeds)

        out = self.bert(
            inputs_embeds=new_embeds,
            attention_mask=new_attn,
        )
        # 第一个 token（即 prompt 之后的 [CLS] 位置）的 hidden state
        cls_pos = self.n_prompt  # 第 K 个位置之后是 [CLS]
        cls = out.last_hidden_state[:, cls_pos, :]
        cls = self.dropout(cls)
        return self.classifier(cls)


# ---------------------------------------------------------------------------
# 通用数据加载 & 训练（优化：动态 padding + 混合精度）
# ---------------------------------------------------------------------------
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
            text, max_length=self.max_len, truncation=True, return_tensors="pt",
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


@torch.no_grad()
def evaluate(model, loader, model_type: str = "lora"):
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


def train_peft(model, train_samples, dev_samples, test_samples, save_name: str,
               num_epochs: int = NUM_EPOCHS, lr: float = 1e-3):
    set_seed()
    print(f"\n>>> 训练 {save_name} ...")
    tokenizer = BertTokenizer.from_pretrained(
        BERT_MODEL_NAME, local_files_only=True
    )
    print(f"  - 可训练参数: {count_trainable(model):,} / 总参数: {count_total(model):,}"
          f"  ({count_trainable(model)/max(count_total(model),1)*100:.2f}%)")

    train_loader = make_loader(train_samples, tokenizer, BATCH_SIZE, True)
    dev_loader = make_loader(dev_samples, tokenizer, EVAL_BATCH_SIZE, False)
    test_loader = make_loader(test_samples, tokenizer, EVAL_BATCH_SIZE, False)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
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
                    logits = model(input_ids, attn)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attn)
                loss = loss_fn(logits, labels)
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
    out = {
        "model": save_name,
        "trainable_params": count_trainable(model),
        "best_dev_acc": best_dev_acc,
        **test_metrics,
    }
    out_path = os.path.join(RESULTS_DIR, f"{save_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] {out_path}")
    return out


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_lora():
    model = LoRAClassifier(
        BERT_MODEL_NAME, num_labels=len(CATEGORIES), r=8, alpha=16
    ).to(DEVICE)
    train_peft(
        model,
        load_split("train"),
        load_split("dev"),
        load_split("test"),
        save_name="peft_lora",
    )


def run_ptuning():
    model = PTuningClassifier(
        BERT_MODEL_NAME, num_labels=len(CATEGORIES), n_prompt_tokens=20
    ).to(DEVICE)
    train_peft(
        model,
        load_split("train"),
        load_split("dev"),
        load_split("test"),
        save_name="peft_ptuning",
        lr=5e-3,  # prompt 参数一般用大学习率
    )


if __name__ == "__main__":
    if not os.path.exists(os.path.join("data", "THUCNews", "train.txt")):
        prepare_data()
    run_lora()
    run_ptuning()
