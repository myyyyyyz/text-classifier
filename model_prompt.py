"""
model_prompt.py
===============
提示学习 (Prompt Learning)：把分类任务改写成"完形填空"任务。

模板（Prompt Template）：
    "[CLS] 原文 [SEP] 这篇新闻属于 [MASK] 类 [SEP]"

[MASK] 位置预测的 token，就是分类的"标签词"（label word）。
例如模型预测 "[MASK]=财经" 的概率最大，就把这条新闻判为"财经"类。

支持 3 种模式：
    1) zero-shot：只用模板 + MLM 头，不训练，直接预测
    2) few-shot ：每个类别只用 FEW_SHOT_K 条样本微调模型
    3) full    ：用全量数据微调（作为 prompt-tuning 的上限）
"""

import os
import json
import time
import random
import warnings
from typing import List, Tuple, Dict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
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
    LABEL2ID,
    ID2LABEL,
    LABEL_WORDS,
    FEW_SHOT_K,
    RESULTS_DIR,
    MODEL_CACHE_DIR,
    USE_AMP,
    NUM_WORKERS,
)
from data_prepare import load_split, prepare_data
from evaluate import print_metrics


PROMPT_TEMPLATE = "这篇新闻属于[MASK]类。"  # 简单模板


# ---------------------------------------------------------------------------
# 工具
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
# 把 label word 编到 id
# ---------------------------------------------------------------------------
class LabelWordEncoder:
    """把"标签词"列表映射到 tokenizer 的 token id 列表（可能一个字对应多个 sub-token）。"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.label_word_ids: Dict[int, List[int]] = {}  # label_id -> 多个 token id
        for label, words in LABEL_WORDS.items():
            ids = set()
            for w in words:
                # add_special_tokens=False + 截断到合理长度
                toks = tokenizer.encode(w, add_special_tokens=False)
                # 只要第 1 个 sub-token（简化处理）
                if toks:
                    ids.add(toks[0])
            self.label_word_ids[LABEL2ID[label]] = list(ids)

    def get_label_word_ids(self, label_id: int) -> List[int]:
        return self.label_word_ids[label_id]


def collate_fn_factory(pad_id: int):
    """工厂函数：创建 collate_fn（避免 lambda 序列化问题）。"""
    def collate_fn(batch):
        max_len = max(b["input_ids"].size(0) for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.zeros((len(batch),), dtype=torch.long)
        mask_idx = torch.zeros((len(batch),), dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["input_ids"].size(0)
            input_ids[i, :n] = b["input_ids"]
            attn[i, :n] = b["attention_mask"]
            labels[i] = b["label"]
            mask_idx[i] = b["mask_index"]
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "label": labels,
            "mask_index": mask_idx,
        }
    return collate_fn


# ---------------------------------------------------------------------------
# 数据集（优化：缓存 tokenized 结果）
# ---------------------------------------------------------------------------
class PromptDataset(Dataset):
    """把样本构造成"原文 + prompt"格式，并记录 [MASK] 位置。"""
    def __init__(self, samples: List[Tuple[str, int]], tokenizer, max_len: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        # 预计算模板部分
        self.prompt_enc = tokenizer(
            PROMPT_TEMPLATE, add_special_tokens=False, return_tensors="pt"
        )
        self.prompt_ids = self.prompt_enc["input_ids"].squeeze(0)
        self.mask_id = tokenizer.mask_token_id
        mask_pos = (self.prompt_ids == self.mask_id).nonzero(as_tuple=True)[0]
        self.mask_pos_in_prompt = mask_pos[0].item() if len(mask_pos) > 0 else 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        
        # 截断原文
        text_enc = self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        )
        text_ids = text_enc["input_ids"].squeeze(0)

        max_text_len = self.max_len - len(self.prompt_ids) - 2
        if max_text_len < 8:
            max_text_len = 8
        text_ids = text_ids[:max_text_len]

        # 拼接: [CLS] text [SEP] prompt [SEP]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        input_ids = torch.cat([
            torch.tensor([cls_id]),
            text_ids,
            torch.tensor([sep_id]),
            self.prompt_ids,
            torch.tensor([sep_id]),
        ])
        attn = torch.ones_like(input_ids)
        
        # [MASK] 绝对下标
        mask_token_index = (input_ids == self.mask_id).nonzero(as_tuple=True)[0]
        mask_token_index = mask_token_index[0].item() if len(mask_token_index) > 0 else 0

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "label": torch.tensor(label, dtype=torch.long),
            "mask_index": torch.tensor(mask_token_index, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 模型：包装 BertForMaskedLM
# ---------------------------------------------------------------------------
class PromptModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.bert_mlm = BertForMaskedLM.from_pretrained(
            model_name, local_files_only=True
        )
        # 把 [MASK] 位置的 hidden state 取出来，过 vocab head
        self.config = self.bert_mlm.config

    def forward(self, input_ids, attention_mask, mask_index, label_word_mask):
        """
        label_word_mask: [num_labels, vocab_size]，标记哪些 token id 是该类的标签词
        """
        out = self.bert_mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = out.logits  # [B, L, V]
        bsz = logits.size(0)
        # 取 [MASK] 位置
        idx = mask_index.unsqueeze(1).unsqueeze(2).expand(bsz, 1, logits.size(-1))
        mask_logits = logits.gather(1, idx).squeeze(1)  # [B, V]
        # 与 label_word_mask 相乘再求和，得到每类的得分
        # mask_logits: [B, V] @ label_word_mask^T: [V, C] -> [B, C]
        scores = mask_logits @ label_word_mask.t()
        return scores


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_prompt(model, loader, label_word_mask) -> Tuple[List[int], List[int]]:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        mask_idx = batch["mask_index"].to(DEVICE)
        scores = model(input_ids, attn, mask_idx, label_word_mask)
        preds = scores.argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


# ---------------------------------------------------------------------------
# 训练主流程（优化：混合精度 + 早停 + 进度条）
# ---------------------------------------------------------------------------
def run(mode: str = "few_shot", k: int = FEW_SHOT_K):
    """
    mode ∈ {zero_shot, few_shot, full}
    """
    set_seed()
    print(f"\n>>> Prompt Learning | 模式: {mode} | k={k}")
    tokenizer = BertTokenizer.from_pretrained(
        BERT_MODEL_NAME, local_files_only=True
    )

    # 准备 label word mask: [num_labels, vocab_size]
    lw_enc = LabelWordEncoder(tokenizer)
    vocab_size = tokenizer.vocab_size
    label_word_mask = torch.zeros(len(CATEGORIES), vocab_size)
    for label_id, tok_ids in lw_enc.label_word_ids.items():
        for tid in tok_ids:
            label_word_mask[label_id, tid] = 1.0
    label_word_mask = label_word_mask.to(DEVICE)

    # 加载数据
    train_full = load_split("train")
    dev = load_split("dev")
    test = load_split("test")

    if mode == "zero_shot":
        model = PromptModel(BERT_MODEL_NAME).to(DEVICE)
        dev_loader = DataLoader(
            PromptDataset(dev, tokenizer, MAX_LEN),
            batch_size=EVAL_BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn_factory(tokenizer.pad_token_id),
            num_workers=NUM_WORKERS,
            pin_memory=True and DEVICE != 'cpu',
        )
        test_loader = DataLoader(
            PromptDataset(test, tokenizer, MAX_LEN),
            batch_size=EVAL_BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn_factory(tokenizer.pad_token_id),
            num_workers=NUM_WORKERS,
            pin_memory=True and DEVICE != 'cpu',
        )
        dev_true, dev_pred = evaluate_prompt(model, dev_loader, label_word_mask)
        test_true, test_pred = evaluate_prompt(model, test_loader, label_word_mask)
    else:
        # 切数据
        if mode == "few_shot":
            by_label = {i: [] for i in range(len(CATEGORIES))}
            for s in train_full:
                if len(by_label[s[1]]) < k:
                    by_label[s[1]].append(s)
            train = []
            for v in by_label.values():
                train.extend(v)
            print(f">>> Few-shot 训练样本数: {len(train)}")
        else:
            train = train_full
            print(f">>> Full 训练样本数: {len(train)}")

        model = PromptModel(BERT_MODEL_NAME).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

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
        epochs = NUM_EPOCHS if mode == "full" else max(NUM_EPOCHS, 5)
        train_loader = DataLoader(
            PromptDataset(train, tokenizer, MAX_LEN),
            batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn_factory(tokenizer.pad_token_id),
            num_workers=NUM_WORKERS,
            pin_memory=True and DEVICE != 'cpu',
        )
        dev_loader = DataLoader(
            PromptDataset(dev, tokenizer, MAX_LEN),
            batch_size=EVAL_BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn_factory(tokenizer.pad_token_id),
            num_workers=NUM_WORKERS,
            pin_memory=True and DEVICE != 'cpu',
        )
        test_loader = DataLoader(
            PromptDataset(test, tokenizer, MAX_LEN),
            batch_size=EVAL_BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn_factory(tokenizer.pad_token_id),
            num_workers=NUM_WORKERS,
            pin_memory=True and DEVICE != 'cpu',
        )
        total_steps = len(train_loader) * epochs
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

        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            total_loss = 0.0
            
            # 训练进度条
            progress_bar = tqdm(
                train_loader,
                desc=f"  Epoch {epoch}/{epochs}",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                ncols=100
            )
            
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
                attn = batch["attention_mask"].to(DEVICE, non_blocking=True)
                labels = batch["label"].to(DEVICE, non_blocking=True)
                mask_idx = batch["mask_index"].to(DEVICE, non_blocking=True)
                
                if USE_AMP and scaler is not None:
                    with torch.cuda.amp.autocast():
                        scores = model(input_ids, attn, mask_idx, label_word_mask)
                        loss = loss_fn(scores, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    scores = model(input_ids, attn, mask_idx, label_word_mask)
                    loss = loss_fn(scores, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                
                # 更新进度条
                progress_bar.set_postfix({'loss': f'{total_loss/(progress_bar.n+1):.4f}'})
            
            progress_bar.close()
            
            dev_true, dev_pred = evaluate_prompt(model, dev_loader, label_word_mask)
            dev_metrics = print_metrics(
                f"[{mode}] Epoch {epoch} - Dev", dev_true, dev_pred, CATEGORIES
            )
            print(f"  train_loss={total_loss / len(train_loader):.4f}  "
                  f"耗时={time.time() - t0:.1f}s")
            
            # 早停
            if dev_metrics["accuracy"] > best_dev_acc:
                best_dev_acc = dev_metrics["accuracy"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  ⚠️ 早停触发！")
                    break

        test_true, test_pred = evaluate_prompt(model, test_loader, label_word_mask)

    metrics = print_metrics(
        f"[{mode}] Test", test_true, test_pred, CATEGORIES
    )
    out = {
        "model": "Prompt-BERT",
        "mode": mode,
        "k": k if mode == "few_shot" else None,
        **metrics,
    }
    fname = f"prompt_{mode}.json" if mode != "few_shot" else f"prompt_few_shot_k{k}.json"
    out_path = os.path.join(RESULTS_DIR, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] 指标已保存到 {out_path}")
    return out


if __name__ == "__main__":
    if not os.path.exists(os.path.join("data", "THUCNews", "train.txt")):
        prepare_data()
    # 跑全部 3 种模式
    run("zero_shot")
    run("few_shot", k=FEW_SHOT_K)
    run("full")
