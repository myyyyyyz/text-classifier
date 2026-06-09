# 基于提示词学习的中文新闻文本分类

> 课程大作业：使用 Transformer 预训练模型 + Prompt Learning + PEFT + 数据增强，做 THUCNews 中文新闻 10 分类。

---

## 0. 关于本目录

整个工程已经集中在本目录 `news_classification/` 里：

- **源码**：`config.py`、`data_prepare.py`、`model_*.py`、`data_augment.py`、`run_all.py` 等
- **数据软链**：`data/` → 自动指向 `data/THUCNews/{train,dev,test}.txt`
- **模型软链**：`model_cache/google-bert/bert-base-chinese/`
- **结果软链**：`results/` → 实验产出（JSON、汇总表、柱状图）

> 这些软链用的是 Windows **directory junction**（类似 Linux 的符号链接），不重复占空间。如果在别的机器上发现软链失效，重新跑 `python create_junctions.py` 即可重建。

PyCharm 打开 `news_classification/` 直接右键 `run_all.py` 就能跑。

---

## 1. 项目结构

```
news_classification/
├── config.py            # 全局配置（路径、超参、类别、标签词）
├── data_prepare.py      # 数据准备：读 THUCNews 原始数据 → train/dev/test
├── model_bert.py        # 基础 BERT 微调
├── model_prompt.py      # 提示学习（zero-shot / few-shot / full）
├── model_peft.py        # 参数高效微调：LoRA + P-Tuning
├── model_compare.py     # 模型对比：BERT / RoBERTa / ELECTRA
├── data_augment.py      # 数据增强：同义词替换
├── evaluate.py          # 评估指标
├── run_all.py           # 一键跑全部实验
├── create_junctions.py  # 软链重建脚本（跨机器时使用）
├── data/                # → 软链到父目录的 data/
├── model_cache/         # → 软链到父目录的 model_cache/
├── results/             # → 软链到父目录的 results/
├── requirements.txt     # 依赖列表
└── README.md            # 本文档
```

## 2. 环境安装

```bash
# 推荐 Python 3.10+，PyCharm 2026.1 直接打开即可
pip install -r requirements.txt
```

`requirements.txt` 内容：
```
torch>=2.0
transformers>=4.40
jieba>=0.42
scikit-learn>=1.0
matplotlib>=3.5
numpy
```

## 2.1 预训练模型下载

代码已**默认走本地模型路径**（避免运行时联网失败）。请先把模型放到 `model_cache/` 下。

方式 A：用 ModelScope（推荐，国内下载快）
```python
from modelscope import snapshot_download
# 中文 BERT
snapshot_download("google-bert/bert-base-chinese",
                  cache_dir="./model_cache")
# 中文 RoBERTa（可选，对比实验用）
snapshot_download("hfl/chinese-roberta-wwm-ext",
                  cache_dir="./model_cache")
# 中文 ELECTA（可选）
snapshot_download("hfl/chinese-electra-180g-base-discriminator",
                  cache_dir="./model_cache")
```

方式 B：手动从 HuggingFace 下载
- 浏览器打开 https://huggingface.co/bert-base-chinese/tree/main
- 下载 `config.json`、`pytorch_model.bin`、`tokenizer.json`、`tokenizer_config.json`、`vocab.txt` 全部文件
- 放到 `model_cache/bert-base-chinese/` 下
- 其他模型同理

`config.py` 里的 `_find_local()` 会自动在 `model_cache/` 下找模型，找不到才会用 huggingface 名称。

## 3. 数据准备

数据集：[飞桨 AI Studio - THUCNews 中文新闻文本分类](https://aistudio.baidu.com/projectdetail/1692440)

下载后请根据自己的解压方式，按下面三种结构之一放到 `data/` 下，代码会自动识别。

### 模式 A：按类分子文件夹（推荐）
解压后通常得到 14 个类别文件夹（每类一个文件夹，里面是若干 .txt 文件）。
把整个目录重命名为 `THUCNews_raw` 放到 `data/` 下：

```
data/THUCNews_raw/
    财经/  彩票/  房产/  股票/  家居/
    教育/  科技/  社会/  时尚/  时政/
    体育/  星座/  游戏/  娱乐/
        0.txt  1.txt  2.txt  ...   ← 每篇新闻一个 txt
```

> AI Studio 上部分版本本身就是这种结构。

### 模式 B：飞桨整合版（cnews.train.txt）
有些版本只提供整合好的 `cnews.train.txt` / `cnews.test.txt`，每行格式：

```
财经    小米宣布造车股价大涨    4月6日消息，小米集团...
体育    世界杯八强出炉          经过激烈角逐...
```

放进 `data/THUCNews_cnews/`：

```
data/THUCNews_cnews/
    cnews.train.txt   ← 18 万条
    cnews.test.txt    ← 1 万条
    cnews.val.txt     ← 可选
```

### 模式 C：THUCNews 预处理版（用户已有的数据 ⭐）
这是最常见也最推荐的格式。文件在 `../data/THUCNews/`（父目录），包含：

```
../data/THUCNews/
    Train_IDs.txt     ← 752,475 条训练，格式: "词id1,词id2,..."\t标签id
    Val_IDs.txt       ← 80,000 条验证
    Test_IDs.txt      ← 83,599 条测试（无标签）
    dict.txt          ← 字符→id 映射
    Train.txt         ← 可选：原始文本版
    Test.txt          ← 可选：原始文本版
```

`data_prepare.py` 会自动：
1. 加载 `dict.txt`（5307 字符表）建立 id→字符 反向映射
2. 读 `Train_IDs.txt` / `Val_IDs.txt`，把 id 序列还原成中文
3. 过滤到我们用的 10 个类别
4. 输出 train.txt / dev.txt / test.txt

**实际产出规模**（默认 `SAMPLES_PER_CLASS=2000`，10 类）：
- train: ~60,000 条
- dev:   ~9,700 条
- test:  ~9,700 条

跑通后 BERT 1 轮准确率就能上 88%（CPU 模式），3 轮可达 92%+。

### 模式 D：demo 兜底
如果三种都没有，脚本会用 `make_demo_dataset()` 自动生成"中文新闻风格"的演示数据，保证流程能跑通（适合先验证代码逻辑再下载真实数据）。

### 切换类别子集
`config.py` 顶部有 `CATEGORIES` 列表：
- 默认：10 类（去掉容易混的彩票/股票/社会/星座）
- 想跑全 14 类：把 `THUCNEWS_ALL_CATEGORIES` 赋给 `CATEGORIES` 即可

```python
# config.py
from config import THUCNEWS_ALL_CATEGORIES
CATEGORIES = THUCNEWS_ALL_CATEGORIES   # 全 14 类
```

## 4. 运行方式

### 方式 0：在 PyCharm 2026.1 里跑

1. 打开项目根目录
2. 右键 `run_all.py` → **Run 'run_all'**
3. 第一次运行会先做数据准备，再依次执行 10 个实验
4. 结果会输出到控制台，同时落到 `results/` 目录
5. 想跑单个实验：右键 `model_bert.py` → **Run** 即可

> PyCharm 运行前请确保已配好 Python 解释器（File → Settings → Project → Python Interpreter），推荐用项目专属 venv。

### 方式 A：一键跑全部（推荐，命令行）

```bash
python run_all.py
```

会自动执行 10 个实验，并在 `results/` 下生成：
- 每个实验一个 `.json`（含 acc / macro_f1 / weighted_f1）
- `summary.md`：Markdown 汇总表
- `comparison.png`：柱状图

### 方式 B：只跑某一个

```bash
python data_prepare.py            # 1. 准备数据
python model_bert.py              # 2. 基础 BERT 微调
python model_prompt.py            # 3. 提示学习
python model_peft.py              # 4. LoRA + P-Tuning
python model_compare.py           # 5. BERT/RoBERTa/ELECTRA 对比
python data_augment.py            # 6. 数据增强
```

### 方式 C：跳过某些实验

```bash
python run_all.py --skip compare_electra aug_bert
```

## 5. 核心思路

| 实验 | 思路 | 输出 |
|------|------|------|
| baseline_bert | BertModel + [CLS] + Linear，3 轮微调 | 全量微调基线 |
| prompt_xxx | 模板 `[文本] 这篇新闻属于[MASK]类`，把 label word 映射到类别；支持 zero/few/full | Prompt Learning |
| peft_lora | 冻结 BERT，在 attention Q/V 加 LoRA 矩阵 | 高效微调（参数少） |
| peft_ptuning | 冻结 BERT，输入前拼 K 个可学习 prompt 向量 | 高效微调 |
| compare_xxx | 切换 backbone (BERT/RoBERTa/ELECTRA) | 模型对比 |
| aug_bert | 同义词替换扩充训练集，再训练 BERT | 数据增强 |

## 6. 主要超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_LEN` | 128 | 文本最大长度 |
| `BATCH_SIZE` | 32 | 训练批大小 |
| `NUM_EPOCHS` | 3 | 训练轮数 |
| `LEARNING_RATE` | 2e-5 | 全量微调学习率 |
| `SAMPLES_PER_CLASS` | 2000 | 每类样本数 |
| `FEW_SHOT_K` | 16 | Few-shot 每类样本数 |

可在 `config.py` 自由修改。

## 7. 常见问题

**Q1：下载 bert-base-chinese 失败？**
A：把 `MODEL_CACHE_DIR` 改成国内可访问的路径，或提前手动下载放到 `~/.cache/huggingface/hub/` 下。

**Q2：GPU 显存不够？**
A：把 `BATCH_SIZE` 调到 16 或 8；把 `MAX_LEN` 调到 64。

**Q3：训练太慢？**
A：先把 `SAMPLES_PER_CLASS` 调到 500 跑通流程；或者用 GPU。

**Q4：零样本效果很差？**
A：正常现象，零样本没有训练过任何样本。改用 few-shot 即可。
