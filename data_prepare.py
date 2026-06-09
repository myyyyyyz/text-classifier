"""
data_prepare.py
===============
数据集准备。自动识别 4 种 THUCNews 格式，做划分并写到 train/dev/test.txt。

THUCNews 飞桨 AI Studio 版本（用户从 https://aistudio.baidu.com/projectdetail/1692440 下载）：

【模式 A：按类分子文件夹】（解压 THUCNews.zip 后会得到）
    data/THUCNews_raw/
        财经/  体育/  科技/  时政/  娱乐/
        家居/  教育/  时尚/  时政/  游戏/
        彩票/  房产/  股票/  社会/  星座/
            0.txt  1.txt  2.txt ...
        每篇新闻是一个独立 txt 文件，内容是纯文本。

【模式 B：飞桨整合版】（cnews.train.txt / cnews.test.txt）
    每行:  label + \\t + 标题 + \\t + 正文
    - 训练集 18 万条，测试集 1 万条
    - 文件名约定: cnews.train.txt, cnews.test.txt, cnews.val.txt
    - 路径: data/THUCNews_cnews/

【模式 C：父目录 THUCNews 整合版】（用户已有的数据！）
    路径: ../data/THUCNews/  (父目录)
    文件: Train.txt / Test.txt
    每行:  id + \\t + 类别名 + \\t + 文本
    - 训练集 ~75 万条，测试集 ~8 万条
    - 14 个类别齐全
    - 优先从父目录找（用户经常把大数据集放在项目根目录）

【模式 D：demo 兜底】
    自动用关键词组合生成"中文新闻风格"假数据，保证流程能跑通。

优先级: A > B > C > D
"""

import os
import random
from typing import List, Tuple

from config import (
    DATA_DIR,
    CATEGORIES,
    LABEL2ID,
    SAMPLES_PER_CLASS,
    SEED,
)
from utils import set_seed, clean_text, split_three


# ---------------------------------------------------------------------------
# 模式 A：按类分子文件夹
# ---------------------------------------------------------------------------
def read_local_thucnews(thucnews_root: str) -> List[Tuple[str, int]]:
    """
    读取本地 THUCNews 数据（按类分子文件夹版）。
    返回 [(text, label_id), ...]，每类最多取 SAMPLES_PER_CLASS 条。
    """
    samples = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(thucnews_root, cat)
        if not os.path.isdir(cat_dir):
            print(f"  [WARN] 找不到类别目录: {cat_dir}，跳过该类")
            continue
        files = [f for f in os.listdir(cat_dir)
                 if os.path.isfile(os.path.join(cat_dir, f))]
        random.shuffle(files)
        picked = 0
        for fn in files:
            if picked >= SAMPLES_PER_CLASS:
                break
            text = _read_text_safe(os.path.join(cat_dir, fn))
            if len(text) < 10:  # 过滤空文本
                continue
            samples.append((text, LABEL2ID[cat]))
            picked += 1
        print(f"  - 类别 [{cat}] 取了 {picked} 条")
    return samples


def _read_text_safe(path: str) -> str:
    """鲁棒地读文本：先 utf-8，失败用 gbk。"""
    for enc in ("utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return clean_text(f.read())
        except UnicodeDecodeError:
            continue
    return ""


# ---------------------------------------------------------------------------
# 模式 B：飞桨整合版（cnews.train.txt / cnews.test.txt）
# ---------------------------------------------------------------------------
def read_cnews_format(cnews_dir: str) -> List[Tuple[str, int]]:
    """
    读取飞桨 cnews 整合版。
    文件格式: label<TAB>title<TAB>content
    我们把 title + content 拼起来作为整篇新闻文本。

    文件命名约定:
        cnews.train.txt  - 训练集
        cnews.test.txt   - 测试集
        cnews.val.txt    - 验证集（可选）
    """
    # 三种文件名都试一下
    candidates = [
        ("train", "cnews.train.txt"),
        ("val", "cnews.val.txt"),
        ("test", "cnews.test.txt"),
    ]
    data_by_split = {}
    for split_name, fn in candidates:
        path = os.path.join(cnews_dir, fn)
        if not os.path.isfile(path):
            data_by_split[split_name] = []
            continue
        print(f"  - 正在读取: {path}")
        rows = _read_cnews_file(path)
        data_by_split[split_name] = rows
        print(f"    解析 {len(rows)} 条")

    # 各类按 SAMPLES_PER_CLASS 上限采样
    def _subsample(rows):
        per_label = {i: [] for i in range(len(CATEGORIES))}
        for text, lab in rows:
            per_label[lab].append((text, lab))
        out = []
        for lst in per_label.values():
            random.shuffle(lst)
            out.extend(lst[:SAMPLES_PER_CLASS])
        return out

    train_sub = _subsample(data_by_split["train"])
    val_sub = _subsample(data_by_split["val"]) if data_by_split["val"] else []
    test_sub = _subsample(data_by_split["test"]) if data_by_split["test"] else []

    # 训练集总量限制
    max_train = SAMPLES_PER_CLASS * len(CATEGORIES)
    if len(train_sub) > max_train:
        random.shuffle(train_sub)
        train_sub = train_sub[:max_train]

    print(f"  - train: {len(train_sub)} | val: {len(val_sub)} | test: {len(test_sub)}")
    return train_sub + val_sub + test_sub


def _read_cnews_file(path: str) -> List[Tuple[str, int]]:
    """读 cnews 文件，按 \\t 切分 3 段：label/title/content。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            label, title, content = parts[0], parts[1], parts[2]
            if label not in LABEL2ID:
                continue
            text = clean_text(title + "。" + content)
            if len(text) < 10:
                continue
            out.append((text, LABEL2ID[label]))
    return out


# ---------------------------------------------------------------------------
# 模式 C：父目录 THUCNews 整合版（用户已有数据！）
# ---------------------------------------------------------------------------
def read_parent_thucnews(parent_data_dir: str) -> List[Tuple[str, int]]:
    """
    读父目录 ../data/THUCNews/ 里的 THUCNews 数据。

    自动识别 2 种格式：

    【格式 1：原始文本版】(Train.txt / Test.txt)
        每行:  id\\t类别名\\t文本
        训练集 ~75 万条，测试集 ~8 万条

    【格式 2：字符 id 版】(Train_IDs.txt / Val_IDs.txt / Test_IDs.txt + dict.txt)
        Train_IDs.txt 每行:  "词id1,词id2,..."\\t标签id
        dict.txt 是个 Python dict 字符串:  {"字": id, ...}
        训练集 752K 条 + 验证集 80K 条（更标准）
        我们把 id 序列还原成原始汉字
    """
    train_txt = os.path.join(parent_data_dir, "Train.txt")
    test_txt = os.path.join(parent_data_dir, "Test.txt")
    train_ids = os.path.join(parent_data_dir, "Train_IDs.txt")
    test_ids = os.path.join(parent_data_dir, "Test_IDs.txt")
    val_ids = os.path.join(parent_data_dir, "Val_IDs.txt")
    dict_path = os.path.join(parent_data_dir, "dict.txt")

    # 优先：ID 版（标准 THUCNews 预处理格式）
    if os.path.isfile(train_ids) and os.path.isfile(dict_path):
        return _read_thucnews_ids(train_ids, val_ids, test_ids, dict_path)

    # 退而求其次：原始文本版
    if os.path.isfile(train_txt):
        return _read_thucnews_raw(train_txt, test_txt)

    raise FileNotFoundError(
        f"父目录 {parent_data_dir} 找不到 THUCNews 数据文件（需要 Train.txt 或 Train_IDs.txt）"
    )


def _read_thucnews_raw(train_path: str, test_path: str) -> List[Tuple[str, int]]:
    """读原始文本版：每行 id\\t类别名\\t文本"""
    def _read(path, split_name):
        if not os.path.isfile(path):
            return []
        print(f"  - 正在读取: {path}")
        out = []
        skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    skipped += 1
                    continue
                _, label_name, text = parts[0], parts[1], parts[2]
                if label_name not in LABEL2ID:
                    skipped += 1
                    continue
                text = clean_text(text)
                if len(text) < 10:
                    skipped += 1
                    continue
                out.append((text, LABEL2ID[label_name]))
        print(f"    {split_name}: 解析 {len(out)} 条，跳过 {skipped} 条")
        return out

    train_rows = _read(train_path, "Train")
    test_rows = _read(test_path, "Test")
    random.seed(SEED)
    random.shuffle(train_rows)
    n_train = int(len(train_rows) * 0.9)
    train_split = train_rows[:n_train]
    dev_split = train_rows[n_train:]
    test_split = test_rows
    print(f"  - 最终: 训练 {len(train_split)} | 验证 {len(dev_split)} | 测试 {len(test_split)}")
    return train_split + dev_split + test_split


def _read_thucnews_ids(train_path: str, val_path: str, test_path: str,
                       dict_path: str) -> List[Tuple[str, int]]:
    """
    读字符 id 版 THUCNews 数据。
    Train_IDs.txt: 每行 "id1,id2,..."\\tlabel_id
    dict.txt: 字符 → id 的 Python dict
    """
    import ast

    # 1. 加载字表
    print(f"  - 正在加载字表: {dict_path}")
    with open(dict_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    # 处理标准 Python dict 格式
    char_table = ast.literal_eval(content)
    # 反向：id → 字符
    id2char = {v: k for k, v in char_table.items()}
    print(f"    字表大小: {len(char_table)} 字符")

    def _ids_to_text(ids_str: str) -> str:
        """把 '3757,1147,3296' 转回原文 '中国' 这种"""
        chars = []
        for x in ids_str.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                idx = int(x)
                c = id2char.get(idx, "")
                chars.append(c)
            except ValueError:
                continue
        return "".join(chars)

    def _read_ids_file(path: str, split_name: str) -> List[Tuple[str, int]]:
        if not os.path.isfile(path):
            return []
        print(f"  - 正在读取: {path}")
        out = []
        skipped = 0
        no_label = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                # Test_IDs.txt 经常没有标签（只有 id 序列）
                if len(parts) == 1:
                    no_label += 1
                    continue  # 跳过没有标签的行
                if len(parts) < 2:
                    skipped += 1
                    continue
                ids_str, label_id_str = parts[0], parts[1]
                try:
                    label_id = int(label_id_str.strip())
                except ValueError:
                    skipped += 1
                    continue
                if label_id not in {v for v in range(len(LABEL2ID))}:
                    skipped += 1
                    continue
                text = clean_text(_ids_to_text(ids_str))
                if len(text) < 10:
                    skipped += 1
                    continue
                out.append((text, label_id))
        if no_label:
            print(f"    {split_name}: 解析 {len(out)} 条，无标签 {no_label} 条已跳过，跳过 {skipped} 条")
        else:
            print(f"    {split_name}: 解析 {len(out)} 条，跳过 {skipped} 条")
        return out

    train_rows = _read_ids_file(train_path, "Train")
    val_rows = _read_ids_file(val_path, "Val")
    test_rows = _read_ids_file(test_path, "Test")

    # 标准 THUCNews 划分：Train 做训练，Val 做验证，Test 做测试
    # 如果 Val 不存在，从 Train 里切 10% 当 Val
    if not val_rows and train_rows:
        random.seed(SEED)
        random.shuffle(train_rows)
        n_val = int(len(train_rows) * 0.1)
        val_rows = train_rows[:n_val]
        train_rows = train_rows[n_val:]
        print(f"  - Train 自动切分: {len(train_rows)} train | {len(val_rows)} val")

    # 默认只取每类前 SAMPLES_PER_CLASS * 3 条（避免内存爆）
    # 用户可以改 config.SAMPLES_PER_CLASS 来控制
    def _subsample(rows, per_class_cap):
        from collections import defaultdict
        per_label = defaultdict(list)
        for text, lab in rows:
            per_label[lab].append((text, lab))
        out = []
        for lab, lst in per_label.items():
            out.extend(lst[:per_class_cap])
        return out

    cap = SAMPLES_PER_CLASS * 3  # 默认最多 6000 条/类 × 10 类 = 6 万条
    train_sub = _subsample(train_rows, cap) if train_rows else []
    val_sub = _subsample(val_rows, cap) if val_rows else []
    test_sub = _subsample(test_rows, cap) if test_rows else []

    print(f"  - 最终: 训练 {len(train_sub)} | 验证 {len(val_sub)} | 测试 {len(test_sub)}")
    return train_sub + val_sub + test_sub


# ---------------------------------------------------------------------------
# 模式 D：demo 兜底
# ---------------------------------------------------------------------------
def make_demo_dataset() -> List[Tuple[str, int]]:
    """
    兜底方案：当用户没有本地 THUCNews 时，自动生成"中文新闻风格"的演示数据。
    """
    keyword_pool = {
        "财经": ["股市", "基金", "金融", "投资", "上市公司", "财报", "央行", "债券", "汇率", "宏观"],
        "彩票": ["双色球", "大乐透", "中奖", "彩票", "开奖", "投注", "福彩", "体彩", "号码", "奖金"],
        "房产": ["房价", "楼市", "开发商", "楼盘", "限购", "二手房", "中介", "租房", "调控", "学区房"],
        "股票": ["A股", "涨停", "跌停", "大盘", "个股", "散户", "操盘", "牛市", "熊市", "证监会"],
        "体育": ["足球", "篮球", "世界杯", "奥运", "联赛", "运动员", "冠军", "决赛", "球队", "比分"],
        "科技": ["人工智能", "芯片", "互联网", "算法", "云计算", "手机", "开源", "研发", "华为", "大模型"],
        "社会": ["民生", "街坊", "社区", "邻里", "志愿者", "基层", "救助", "公益", "事故", "调解"],
        "时政": ["国务院", "中央", "政策", "会议", "外交", "领导", "政府", "改革", "国家", "讲话"],
        "娱乐": ["明星", "电影", "电视剧", "综艺", "演唱会", "演员", "导演", "粉丝", "票房", "选秀"],
        "家居": ["装修", "家具", "客厅", "厨房", "设计", "家居", "品牌", "沙发", "卫浴", "户型"],
        "教育": ["学校", "学生", "老师", "高考", "大学", "课程", "教材", "家长", "招生", "教学"],
        "时尚": ["时装", "潮流", "搭配", "服装", "品牌", "街拍", "走秀", "设计师", "化妆", "护肤"],
        "星座": ["运势", "占卜", "天蝎座", "双子座", "十二星座", "本周", "水逆", "上升", "月亮", "塔罗"],
        "游戏": ["电竞", "网游", "手游", "主机", "玩家", "Steam", "原神", "英雄联盟", "皮肤", "段位"],
    }
    templates = [
        "近日关于{cat}领域的报道：{w1}与{w2}的结合，引发业内广泛关注。",
        "{w1}最近的表现成为{cat}行业热议的焦点，多家媒体跟踪报道。",
        "据{cat}行业内部人士透露，{w1}即将迎来重大调整，{w2}或成关键变量。",
        "在{cat}领域，{w1}与{w2}的竞争愈发激烈，专家给出深度解读。",
        "{w1}持续升温，{cat}板块整体呈现新的发展趋势，{w2}成最大亮点。",
        "权威机构发布最新{cat}报告：{w1}数据创新高，{w2}表现稳定。",
        "{cat}观察：{w1}、{w2}等多个维度发生变化，相关从业者表达看法。",
        "针对{cat}行业的最新动态，{w1}成为讨论热点，{w2}也备受关注。",
    ]

    samples = []
    rng = random.Random(SEED)
    for cat in CATEGORIES:
        kws = keyword_pool.get(cat, [cat, "领域", "报道", "新闻"])
        generated = set()
        attempts = 0
        while len(generated) < SAMPLES_PER_CLASS and attempts < SAMPLES_PER_CLASS * 5:
            attempts += 1
            tpl = rng.choice(templates)
            w1, w2 = rng.sample(kws, 2)
            text = tpl.format(cat=cat, w1=w1, w2=w2)
            if text in generated:
                continue
            generated.add(text)
            samples.append((text, LABEL2ID[cat]))
    return samples


# ---------------------------------------------------------------------------
# 保存 & 读取
# ---------------------------------------------------------------------------
def save_split(splits):
    """把划分结果写到 txt 文件。每行：label<TAB>text"""
    train, dev, test = splits
    out_dir = os.path.join(DATA_DIR, "THUCNews")
    os.makedirs(out_dir, exist_ok=True)
    for name, data in [("train", train), ("dev", dev), ("test", test)]:
        path = os.path.join(out_dir, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for text, label in data:
                # 用 \\t 分隔，文本里的换行/制表提前清洗
                text = clean_text(text).replace("\t", " ")
                f.write(f"{label}\t{text}\n")
        print(f"  - 已写入 {path}，共 {len(data)} 条")


def load_split(name: str) -> List[Tuple[str, int]]:
    """读取 train/dev/test。"""
    path = os.path.join(DATA_DIR, "THUCNews", f"{name}.txt")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, text = line.split("\t", 1)
            data.append((text, int(label)))
    return data


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def prepare_data():
    """主入口。自动按 模式A > 模式B > 模式C > 模式D 顺序尝试数据源。"""
    print("=" * 60)
    print("  准备 THUCNews 数据集")
    print(f"  目标类别 ({len(CATEGORIES)}): {' / '.join(CATEGORIES)}")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # 候选路径
    thucnews_raw = os.path.join(DATA_DIR, "THUCNews_raw")         # 模式 A
    thucnews_cnews = os.path.join(DATA_DIR, "THUCNews_cnews")     # 模式 B
    # 模式 C：父目录的整合版（用户经常把大数据集放在项目根目录）
    # DATA_DIR = news_classification/data（已经通过 junction 软链到 ../data）
    # 所以要往上 2 级才是真正的项目根目录（新闻分类/）
    project_root = os.path.dirname(os.path.dirname(DATA_DIR))
    parent_thucnews = os.path.join(project_root, "data", "THUCNews")

    samples = []
    source = None
    sampled = False  # 是否已经采样（模式 A/B/D 都做了 SAMPLES_PER_CLASS 上限采样）

    # 模式 A 优先
    if os.path.isdir(thucnews_raw):
        print(f"[INFO] 模式 A：检测到按类分子文件夹 {thucnews_raw}")
        samples = read_local_thucnews(thucnews_raw)
        source = "raw"
        sampled = True
    # 模式 B
    elif os.path.isdir(thucnews_cnews) and os.path.isfile(
        os.path.join(thucnews_cnews, "cnews.train.txt")
    ):
        print(f"[INFO] 模式 B：检测到飞桨 cnews 整合版 {thucnews_cnews}")
        samples = read_cnews_format(thucnews_cnews)
        source = "cnews"
        sampled = True
    # 模式 C：父目录的整合版（用户经常把大数据集放在项目根目录）
    # 触发条件：父目录 data/THUCNews/ 下有 Train.txt 或 Train_IDs.txt
    elif (
        os.path.isdir(parent_thucnews)
        and (
            os.path.isfile(os.path.join(parent_thucnews, "Train.txt"))
            or os.path.isfile(os.path.join(parent_thucnews, "Train_IDs.txt"))
        )
    ):
        print(f"[INFO] 模式 C：检测到父目录 THUCNews 整合版 {parent_thucnews}")
        samples = read_parent_thucnews(parent_thucnews)
        source = "parent_thucnews"
        sampled = True  # 函数内部已做了 subsample
    # 模式 D 兜底
    else:
        print(f"[WARN] 都没找到，改用 demo 数据")
        print("        想用真实数据请按 README 3 的说明放到合适位置。")
        samples = make_demo_dataset()
        source = "demo"
        sampled = True

    print(f"\n[INFO] 数据源: {source}, 样本总数: {len(samples)}")

    if sampled:
        # 重新切分
        splits = split_three(samples, ratios=(0.8, 0.1, 0.1), seed=SEED)
    else:
        # 模式 C 已经在 read_parent_thucnews 里切分好了
        # 还原为 (train, dev, test) 三元组
        random.seed(SEED)
        # 重新洗牌按比例切（保险起见）
        random.shuffle(samples)
        n = len(samples)
        n_train = int(n * 0.8)
        n_dev = int(n * 0.1)
        splits = (
            samples[:n_train],
            samples[n_train:n_train + n_dev],
            samples[n_train + n_dev:],
        )

    save_split(splits)
    print("\n[OK] 数据准备完成！")
    return splits


if __name__ == "__main__":
    prepare_data()
