"""
data_augment.py
===============
数据增强：基于同义词替换。

策略：
    - 用 jieba 分词
    - 准备一份"通用中文同义词词典"（小规模演示用，可自行扩展）
    - 在每个句子里随机选 r% 的非停用词，替换为同义词
    - 注意：替换的词不能破坏类别语义（金融/体育/科技等关键词不替换）

增强完之后：
    - 把增强数据 + 原始数据一起当作新的训练集
    - 重新跑一次基础 BERT 微调，对比"无增强"和"有增强"的指标
"""

import os
import json
import random
import re
import time
from typing import List, Tuple

import jieba
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
    LABEL_WORDS,
    RESULTS_DIR,
    MODEL_CACHE_DIR,
    USE_AMP,
    NUM_WORKERS,
)
from data_prepare import load_split
from evaluate import print_metrics


# ---------------------------------------------------------------------------
# 1. 同义词词典（演示用，可自行扩展）
# ---------------------------------------------------------------------------
SYNONYM_DICT = {
    "公司": ["企业", "厂商"],
    "表示": ["指出", "称", "认为"],
    "认为": ["觉得", "以为", "指出"],
    "推出": ["发布", "上线", "面世"],
    "近日": ["最近", "日前", "不久前"],
    "网友": ["网民", "用户"],
    "平台": ["渠道", "站点"],
    "关注": ["关心", "重视"],
    "提高": ["提升", "增强"],
    "降低": ["减少", "下降"],
    "增加": ["增添", "加大"],
    "获得": ["赢得", "取得"],
    "开始": ["启动", "开启"],
    "结束": ["落幕", "收官"],
    "未来": ["以后", "今后"],
    "现在": ["当前", "目前"],
    "重要": ["关键", "要紧"],
    "发展": ["成长", "进步"],
    "问题": ["难题", "议题"],
    "解决": ["处理", "应对"],
    "会议": ["大会", "论坛"],
    "学生": ["学子", "学员"],
    "学校": ["院校", "学府"],
    "老师": ["教师", "先生"],
    "比赛": ["赛事", "竞赛"],
    "冠军": ["第一", "头名"],
    "球员": ["选手", "运动员"],
    "球队": ["俱乐部", "队伍"],
    "楼市": ["房地产市场", "房产市场"],
    "房价": ["房价", "楼盘价格"],
    "装修": ["装潢", "装饰"],
    "家具": ["家居", "家私"],
    "时装": ["服装", "服饰"],
    "化妆": ["妆容", "上妆"],
    "护肤": ["保养", "护理"],
    "电竞": ["电子竞技", "游戏比赛"],
    "玩家": ["用户", "游戏者"],
    "手游": ["手机游戏", "移动游戏"],
    "明星": ["艺人", "偶像"],
    "电影": ["影片", "大片"],
    "电视剧": ["剧集", "连续剧"],
    "导演": ["编剧", "执导者"],
}

# 停用词：不会被替换
STOPWORDS = set("的了是我在有不这那和与及等把被让使".split() + list("，。、！？；：""''《》（）"))


def build_protected_words() -> set:
    """类别关键词绝对不能被替换，否则会破坏类别语义。"""
    prot = set()
    for words in LABEL_WORDS.values():
        prot.update(words)
    return prot


PROTECTED = build_protected_words()


# ---------------------------------------------------------------------------
# 2. 替换函数（优化：并行处理）
# ---------------------------------------------------------------------------
def synonym_replace(text: str, replace_ratio: float = 0.15, rng=None) -> str:
    if rng is None:
        rng = random
    words = list(jieba.cut(text))
    new_words = []
    n = len(words)
    if n == 0:
        return text
    n_replace = max(1, int(n * replace_ratio))
    candidates = [i for i, w in enumerate(words)
                  if w.strip()
                  and w not in STOPWORDS
                  and w not in PROTECTED
                  and w in SYNONYM_DICT]
    rng.shuffle(candidates)
    to_replace = set(candidates[:n_replace])
    for i, w in enumerate(words):
        if i in to_replace and w in SYNONYM_DICT:
            new_words.append(rng.choice(SYNONYM_DICT[w]))
        else:
            new_words.append(w)
    return "".join(new_words)


def augment_dataset(samples: List[Tuple[str, int]],
                    aug_per_sample: int = 1,
                    replace_ratio: float = 0.15) -> List[Tuple[str, int]]:
    """对训练集做扩充：每条样本生成 aug_per_sample 个增强样本。"""
    from multiprocessing import Pool
    
    rng = random.Random(SEED)
    new_samples = list(samples)
    
    # 准备任务
    tasks = [(text, label, aug_per_sample, replace_ratio, SEED + i) 
             for i, (text, label) in enumerate(samples)]
    
    # 并行增强（加速）
    def _augment_single(task):
        text, label, aug_count, ratio, seed = task
        rng_local = random.Random(seed)
        augmented = []
        for _ in range(aug_count):
            new_text = synonym_replace(text, replace_ratio=ratio, rng=rng_local)
            if new_text != text:
                augmented.append((new_text, label))
        return augmented
    
    with Pool(processes=min(4, os.cpu_count() or 1)) as pool:
        results = pool.map(_augment_single, tasks)
    
    for aug_list in results:
        new_samples.extend(aug_list)
    
    return new_samples


# ---------------------------------------------------------------------------
# 3. 训练 (复用 model_bert 的结构)
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
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class BertClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            model_name, local_files_only=True
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return self.classifier(cls)


def make_loader(samples, tokenizer, bs, shuffle):
    return DataLoader(
        TextDataset(samples, tokenizer, MAX_LEN),
        batch_size=bs, shuffle=shuffle, num_workers=0,
    )


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


def collate_fn_factory(tokenizer):
    """工厂函数：创建 collate_fn（避免 lambda 序列化问题）。"""
    def collate_fn(batch):
        max_len = max(b["input_ids"].size(0) for b in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
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


def train_with_aug(aug_per_sample: int = 1, replace_ratio: float = 0.15):
    """使用数据增强训练并测试。"""
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("\n========== 数据增强实验 ==========")
    train_samples = load_split("train")
    dev_samples = load_split("dev")
    test_samples = load_split("test")
    print(f"  原始 train 样本数: {len(train_samples)}")
    aug_train = augment_dataset(
        train_samples, aug_per_sample=aug_per_sample,
        replace_ratio=replace_ratio
    )
    print(f"  增强后 train 样本数: {len(aug_train)}")
    print("  示例增强前: ", train_samples[0][0][:60])
    print("  示例增强后: ", aug_train[len(train_samples)][0][:60])

    tokenizer = BertTokenizer.from_pretrained(
        BERT_MODEL_NAME, local_files_only=True
    )
    model = BertClassifier(BERT_MODEL_NAME, len(CATEGORIES)).to(DEVICE)

    train_loader = DataLoader(
        TextDataset(aug_train, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True and DEVICE != 'cpu',
        collate_fn=collate_fn_factory(tokenizer),
    )
    dev_loader = DataLoader(
        TextDataset(dev_samples, tokenizer, MAX_LEN),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True and DEVICE != 'cpu',
        collate_fn=collate_fn_factory(tokenizer),
    )
    test_loader = DataLoader(
        TextDataset(test_samples, tokenizer, MAX_LEN),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True and DEVICE != 'cpu',
        collate_fn=collate_fn_factory(tokenizer),
    )

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
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for batch in train_loader:
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
        
        dev_true, dev_pred = evaluate(model, dev_loader)
        print_metrics(
            f"[aug] Epoch {epoch} - Dev", dev_true, dev_pred, CATEGORIES
        )
        print(f"  耗时={time.time() - t0:.1f}s  loss={total_loss/len(train_loader):.4f}")

    test_true, test_pred = evaluate(model, test_loader)
    test_metrics = print_metrics(
        "[aug] Test", test_true, test_pred, CATEGORIES
    )
    out = {
        "model": "BERT + 同义词替换",
        "save_name": "aug_bert",
        "aug_per_sample": aug_per_sample,
        "replace_ratio": replace_ratio,
        **test_metrics,
    }
    out_path = os.path.join(RESULTS_DIR, "aug_bert.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] {out_path}")
    return out


if __name__ == "__main__":
    train_with_aug(aug_per_sample=1, replace_ratio=0.15)
