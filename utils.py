"""
utils.py
========
工具函数集合。所有脚本都可能用到的零散工具放这里，避免重复。
"""

import os
import re
import random
from typing import List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. 随机种子
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """设置 random / numpy / torch 的随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 2. 文本清洗
# ---------------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    文本清洗：
    - 把各种空白字符（换行/制表/连续空格）压成单个空格
    - 去掉首尾空白
    """
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# 3. 参数统计
# ---------------------------------------------------------------------------
def count_trainable(model) -> int:
    """统计模型可训练参数数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model) -> int:
    """统计模型总参数数量。"""
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 4. 路径工具
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    """确保目录存在，返回路径。"""
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 5. 划分数据
# ---------------------------------------------------------------------------
def split_three(samples: List[Tuple], ratios=(0.8, 0.1, 0.1), seed: int = 42):
    """
    按 8:1:1 切分 train/dev/test。ratios 之和应为 1。
    """
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n = len(samples)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    train = [samples[i] for i in indices[:n_train]]
    dev = [samples[i] for i in indices[n_train:n_train + n_dev]]
    test = [samples[i] for i in indices[n_train + n_dev:]]
    return train, dev, test
