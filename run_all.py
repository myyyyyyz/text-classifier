"""
run_all.py
==========
一键跑完全部实验，并把结果汇总成一张大表 + 一张柱状图。

实验顺序：
    1) baseline_bert         : 基础 BERT 微调（必做，基础任务）
    2) prompt_zero_shot      : 零样本 Prompt
    3) prompt_few_shot       : 少样本 Prompt（k=16）
    4) prompt_full           : 全量 Prompt
    5) peft_lora             : LoRA 高效微调
    6) peft_ptuning          : P-Tuning 高效微调
    7) compare_roberta       : RoBERTa
    8) compare_electra       : ELECTRA
    9) aug_bert              : BERT + 同义词替换

用法：
    python run_all.py                  # 跑全部
    python run_all.py --skip baseline  # 跳过某些实验
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR
from data_prepare import prepare_data


# 设置中文字体（防止图表中文乱码）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def print_progress_bar(current: int, total: int, experiment_name: str, 
                       elapsed_time: float = None, status: str = "运行中"):
    """打印进度条。
    
    Args:
        current: 当前实验序号（从1开始）
        total: 总实验数
        experiment_name: 当前实验名称
        elapsed_time: 已用时间（秒）
        status: 状态（运行中/完成/跳过/失败）
    """
    bar_length = 40
    progress = current / total
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # 格式化时间
    time_str = ""
    if elapsed_time is not None:
        if elapsed_time < 60:
            time_str = f" | ⏱️ {elapsed_time:.1f}s"
        elif elapsed_time < 3600:
            time_str = f" | ⏱️ {elapsed_time/60:.1f}min"
        else:
            time_str = f" | ⏱️ {elapsed_time/3600:.1f}h"
    
    # 状态图标
    status_icon = {
        "运行中": "🔄",
        "完成": "✅",
        "跳过": "⏭️",
        "失败": "❌"
    }.get(status, "•")
    
    print(f"\n{status_icon} [{bar}] {current}/{total} | {experiment_name}{time_str}")
    print("=" * 70)


def maybe_run(name: str, fn, skip_set: set, current: int, total: int):
    """如果该实验不在 skip 集合里就执行。"""
    if name in skip_set:
        print_progress_bar(current, total, name, status="跳过")
        print(f"[SKIP] {name}")
        return None
    
    print_progress_bar(current, total, name, status="运行中")
    start_time = time.time()
    
    try:
        result = fn()
        elapsed = time.time() - start_time
        print_progress_bar(current, total, name, elapsed_time=elapsed, status="完成")
        
        if result:
            acc = result.get('accuracy', 'N/A')
            f1 = result.get('macro_f1', 'N/A')
            print(f"✓ 准确率: {acc}% | 宏平均F1: {f1}% | 耗时: {elapsed:.1f}s")
        
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print_progress_bar(current, total, name, elapsed_time=elapsed, status="失败")
        print(f"[ERROR] {name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def collect_results() -> List[Dict]:
    """从 results/ 目录读所有 json 汇总。"""
    rows = []
    if not os.path.isdir(RESULTS_DIR):
        return rows
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(RESULTS_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            d["__file__"] = fn
            rows.append(d)
        except Exception as e:
            print(f"[WARN] 跳过 {fn}: {e}")
    return rows


def make_table(rows: List[Dict]) -> str:
    """把结果转成 markdown 表格。"""
    lines = ["| 实验 | 模型 | 准确率 | 宏平均F1 | 加权F1 | 备注 |",
             "|------|------|--------|----------|--------|------|"]
    for r in rows:
        name = r.get("save_name", r.get("model", "?"))
        model = r.get("model", "?")
        acc = r.get("accuracy", "-")
        mf1 = r.get("macro_f1", "-")
        wf1 = r.get("weighted_f1", "-")
        extra = ""
        if "mode" in r:
            extra = f"mode={r['mode']}"
        if "k" in r and r["k"]:
            extra += f" k={r['k']}"
        if "trainable_params" in r:
            extra += f" 可训练参数={r['trainable_params']:,}"
        if "aug_per_sample" in r:
            extra += f" aug×{r['aug_per_sample']}"
        lines.append(f"| {name} | {model} | {acc} | {mf1} | {wf1} | {extra} |")
    return "\n".join(lines)


def make_chart(rows: List[Dict], out_path: str):
    """画一张柱状图对比 acc 和 macro_f1。"""
    if not rows:
        print("[WARN] 没有结果，跳过画图")
        return
    names = [r.get("save_name", r.get("model", "?")) for r in rows]
    accs = [r.get("accuracy", 0) for r in rows]
    f1s = [r.get("macro_f1", 0) for r in rows]

    x = range(len(names))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))
    ax.bar([i - w / 2 for i in x], accs, width=w, label="Accuracy")
    ax.bar([i + w / 2 for i in x], f1s, width=w, label="Macro F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("THUCNews 中文新闻文本分类 - 实验对比")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] 柱状图已保存到 {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="跳过的实验名（save_name），多个用空格分隔",
    )
    parser.add_argument(
        "--only-data", action="store_true",
        help="只准备数据，不训练",
    )
    args = parser.parse_args()
    skip = set(args.skip)

    print("\n" + "=" * 70)
    print("  THUCNews 文本分类 - 全套实验流程")
    print("=" * 70)
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  设备: {('GPU (CUDA)' if 'cuda' in str(__import__('torch').device) else 'CPU')}")
    print("=" * 70)
    
    overall_start = time.time()

    # 1. 准备数据
    print("\n📦 步骤 1/2: 准备数据")
    print("-" * 70)
    if not os.path.exists(os.path.join("data", "THUCNews", "train.txt")):
        print("  数据不存在，开始准备...")
        prepare_data()
    else:
        print("  ✓ 数据已存在，跳过准备")
    print(f"  数据准备耗时: {time.time() - overall_start:.1f}s")

    if args.only_data:
        print("\n[OK] 数据准备完成！")
        return

    # 2. 各个实验
    print("\n\n🚀 步骤 2/2: 运行实验")
    print("=" * 70)
    
    from model_bert import train_one_model
    from model_prompt import run as run_prompt
    from model_peft import run_lora, run_ptuning
    from model_compare import train_one as compare_train
    from data_augment import train_with_aug
    from config import (
        BERT_MODEL_NAME, ROBERTA_MODEL_NAME, ELECTRA_MODEL_NAME,
        CATEGORIES, FEW_SHOT_K,
    )

    # 定义实验列表
    experiments = [
        ("baseline_bert", lambda: train_one_model(BERT_MODEL_NAME, len(CATEGORIES), "baseline_bert")),
        ("prompt_zero_shot", lambda: run_prompt("zero_shot")),
        ("prompt_few_shot", lambda: run_prompt("few_shot", k=FEW_SHOT_K)),
        ("prompt_full", lambda: run_prompt("full")),
        ("peft_lora", run_lora),
        ("peft_ptuning", run_ptuning),
        ("compare_roberta", lambda: compare_train(ROBERTA_MODEL_NAME, "compare_roberta")),
        ("compare_electra", lambda: compare_train(ELECTRA_MODEL_NAME, "compare_electra")),
        ("aug_bert", train_with_aug),
    ]
    
    # 过滤掉要跳过的实验
    active_experiments = [(name, fn) for name, fn in experiments if name not in skip]
    total_experiments = len(active_experiments)
    
    if total_experiments == 0:
        print("\n⚠️  所有实验都被跳过，退出。")
        return
    
    print(f"\n计划运行 {total_experiments} 个实验")
    if skip:
        print(f"已跳过 {len(skip)} 个实验: {', '.join(skip)}")
    print()
    
    results = []
    for idx, (name, fn) in enumerate(active_experiments, 1):
        result = maybe_run(name, fn, skip, idx, total_experiments)
        if result:
            results.append(result)
    
    overall_elapsed = time.time() - overall_start
    
    # 3. 汇总
    print("\n\n" + "=" * 70)
    print("  📊 实验结果汇总")
    print("=" * 70)
    
    rows = collect_results()
    table = make_table(rows)
    summary_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 实验结果汇总\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**总耗时**: {overall_elapsed/60:.1f} 分钟\n\n")
        f.write(table)
        f.write("\n\n---\n")
        f.write(f"*自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print("\n" + table)
    print("=" * 70)
    print(f"[OK] 汇总表已保存到 {summary_path}")
    
    chart_path = os.path.join(RESULTS_DIR, "comparison.png")
    make_chart(rows, chart_path)
    
    # 最终总结
    print("\n" + "=" * 70)
    print("  🎉 全部完成！")
    print("=" * 70)
    print(f"  总耗时: {overall_elapsed/60:.1f} 分钟 ({overall_elapsed:.1f} 秒)")
    print(f"  完成实验数: {len(results)}/{total_experiments}")
    print(f"  结果保存在: {RESULTS_DIR}")
    print(f"  - 汇总报告: {summary_path}")
    print(f"  - 对比图表: {chart_path}")
    print("=" * 70)
    
    # 显示最佳结果
    if rows:
        best_acc = max(rows, key=lambda x: x.get('accuracy', 0))
        best_f1 = max(rows, key=lambda x: x.get('macro_f1', 0))
        print(f"\n🏆 最佳准确率: {best_acc.get('save_name', '?')} - {best_acc.get('accuracy', 0)}%")
        print(f"🏆 最佳宏平均F1: {best_f1.get('save_name', '?')} - {best_f1.get('macro_f1', 0)}%")
    
    print()


if __name__ == "__main__":
    main()
