"""
evaluate.py
===========
评估指标：准确率、宏平均 F1、分类报告、混淆矩阵。
所有训练脚本都会调用这里的函数。
"""

from typing import List, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """计算准确率、宏平均 F1、加权 F1。"""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": round(acc * 100, 4),
        "macro_f1": round(macro_f1 * 100, 4),
        "weighted_f1": round(weighted_f1 * 100, 4),
    }


def classification_report_str(y_true, y_pred, target_names) -> str:
    """生成 sklearn 的分类报告（每类 precision/recall/f1）。"""
    return classification_report(
        y_true, y_pred, target_names=target_names, digits=4, zero_division=0
    )


def confusion_matrix_np(y_true, y_pred) -> np.ndarray:
    """返回 numpy 形式的混淆矩阵。"""
    return confusion_matrix(y_true, y_pred)


def print_metrics(name: str, y_true, y_pred, target_names):
    """一行一指标打印，便于在 PyCharm 控制台直接看。"""
    metrics = compute_metrics(y_true, y_pred)
    print(f"\n========== {name} ==========")
    for k, v in metrics.items():
        print(f"  {k:14s} = {v}")
    print("\n[分类报告]")
    print(classification_report_str(y_true, y_pred, target_names))
    return metrics
