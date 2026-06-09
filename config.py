"""
config.py
=========
全局配置文件。所有超参、路径、类别标签都集中在这里，方便在 PyCharm 里调整。
"""

import os
import sys
import torch

# 强制 transformers 优先使用本地模型文件
# 用户必须先把模型放到 model_cache 目录下（用 modelscope snapshot_download 或手动下载）
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ---------------------------------------------------------------------------
# 1. 路径
# ---------------------------------------------------------------------------
# 项目根目录（当前文件所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据集所在目录（训练/验证/测试文本文件）
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 模型缓存目录（huggingface 下载的预训练权重会放在这里）
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "model_cache")

# 实验结果输出目录（保存指标、对比表、图片）
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. 类别标签
# ---------------------------------------------------------------------------
# THUCNews 全部 14 个类别（飞桨 AI Studio 上的官方划分）
THUCNEWS_ALL_CATEGORIES = [
    "财经", "彩票", "房产", "股票", "家居",
    "教育", "科技", "社会", "时尚", "时政",
    "体育", "星座", "游戏", "娱乐",
]

# 本次实验使用的类别子集。默认取最常用的 10 类（去掉容易混淆的彩票/股票/社会/星座）。
# 你可以改成 THUCNEWS_ALL_CATEGORIES 跑全 14 类。
CATEGORIES = [
    "财经", "体育", "科技", "时政", "娱乐",
    "家居", "教育", "时尚", "房产", "游戏",
]

# 兜底：如果 CATEGORIES 里有 THUCNEWS_ALL_CATEGORIES 没有的名字，会自动从 ALL 移除
CATEGORIES = [c for c in CATEGORIES if c in THUCNEWS_ALL_CATEGORIES]

# 标签 -> 整数 id  的映射，例如 {"财经": 0, "体育": 1, ...}
LABEL2ID = {label: idx for idx, label in enumerate(CATEGORIES)}
# 整数 id -> 标签
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# 每个类别对应的"标签词"：Prompt 学习里要用
# 例如分类头输出 [财经] 词对应的概率作为"财经"这一类的得分
# 注意：只包含 CATEGORIES 中实际使用的类别
LABEL_WORDS = {
    "财经": ["财经", "金融", "经济"],
    "体育": ["体育", "运动", "竞技"],
    "科技": ["科技", "技术", "数码"],
    "时政": ["时政", "政治", "政务"],
    "娱乐": ["娱乐", "明星", "影视"],
    "家居": ["家居", "家具", "装修"],
    "教育": ["教育", "学校", "学习"],
    "时尚": ["时尚", "潮流", "时装"],
    "房产": ["房产", "房地产", "楼市"],
    "游戏": ["游戏", "电竞", "网游"],
}

# ---------------------------------------------------------------------------
# 3. 数据规模
# ---------------------------------------------------------------------------
# 每类取多少条样本（增加到 3000 以获得更好效果）
SAMPLES_PER_CLASS = 3000
# 训练/验证/测试的划分比例
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
# TEST_RATIO = 0.1（剩余）

# ---------------------------------------------------------------------------
# 4. 模型超参数
# ---------------------------------------------------------------------------
# 预训练模型名（huggingface 上的标准名称）
BERT_MODEL_NAME = "bert-base-chinese"
ROBERTA_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
ELECTRA_MODEL_NAME = "hfl/chinese-electra-180g-base-discriminator"

# 最大序列长度（增加到 256 以捕获更多上下文）
MAX_LEN = 256

# 批大小（根据显存调整，如果 OOM 则减小）
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128

# 训练轮数（增加到 5 轮以获得更好收敛）
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# 随机种子
SEED = 42

# Few-shot 实验里每个类用多少条样本
FEW_SHOT_K = 16
ZERO_SHOT = True  # 是否也跑一次 zero-shot

# ---------------------------------------------------------------------------
# 5. 设备
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 混合精度训练标志（加速训练）
USE_AMP = torch.cuda.is_available()

# DataLoader 工作进程数（Windows 下建议设为 0 避免多进程问题）
NUM_WORKERS = 0 if sys.platform == 'win32' else min(4, os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# 6. 模型本地路径（ModelScope 缓存的目录）
# ---------------------------------------------------------------------------
# 用户可以把自己下载好的模型放到以下任一位置：
#   - model_cache/bert-base-chinese/  (推荐)
#   - model_cache/google-bert/bert-base-chinese/  (ModelScope 默认结构)
# 代码会优先尝试本地路径，找不到再用 huggingface 名称。
import glob as _glob
def _find_local(model_name: str) -> str:
    """在 model_cache 下递归查找名称匹配的子目录。
    匹配规则：尾部名称（bert-base-chinese）忽略大小写、不区分路径分隔符。
    """
    if not os.path.isdir(MODEL_CACHE_DIR):
        return model_name
    short = model_name.split("/")[-1].lower()  # bert-base-chinese
    # 1) 直接子目录
    cand1 = os.path.join(MODEL_CACHE_DIR, short)
    if os.path.isdir(cand1) and os.path.isfile(os.path.join(cand1, "config.json")):
        return os.path.normpath(cand1)
    # 2) 任意一级子目录（递归扫描）
    matches = _glob.glob(
        os.path.join(MODEL_CACHE_DIR, "**", short), recursive=True
    )
    for m in matches:
        if os.path.isfile(os.path.join(m, "config.json")):
            return os.path.normpath(m)
    return model_name

# 用本地路径替换原来的字符串
BERT_MODEL_NAME = _find_local("bert-base-chinese")
ROBERTA_MODEL_NAME = _find_local("chinese-roberta-wwm-ext")
ELECTRA_MODEL_NAME = _find_local("chinese-electra-180g-base-discriminator")

