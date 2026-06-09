"""
create_junctions.py
在 news_classification/ 下创建指向父目录的目录软链 (junction)
这样新闻分类文件夹就有完整工程结构，但模型/数据不重复占空间。
"""
import os
import ctypes

ROOT = r"C:\Users\27578\Desktop\文本分类"
DEST = os.path.join(ROOT, "news_classification")


def create_junction(link_path: str, target_path: str) -> bool:
    """用 Windows API 直接创建 junction。"""
    if not os.path.isdir(target_path):
        print(f"  [skip] 源不存在: {target_path}")
        return False
    if os.path.exists(link_path):
        print(f"  [skip] 已存在: {link_path}")
        return True
    # 确保 link 所在父目录存在
    os.makedirs(os.path.dirname(link_path), exist_ok=True)

    # 调用 CreateSymbolicLinkW / CreateJunction
    # 简化：使用 mklink via ctypes
    import subprocess
    # 不用 cmd，直接用 PowerShell 的 New-Item -ItemType Junction
    ps_cmd = (
        f'$link = "{link_path}"; '
        f'$target = "{target_path}"; '
        f'New-Item -ItemType Junction -Path $link -Target $target | Out-Null; '
        f'Write-Host "OK: $link -> $target"'
    )
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_cmd],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print(f"  junction OK: {os.path.basename(link_path)} -> {target_path}")
        return True
    print(f"  [warn] junction 失败: {r.stderr.strip()}")
    return False


if __name__ == "__main__":
    print("=" * 60)
    print(" 创建目录软链 (junction)")
    print("=" * 60)
    for folder in ["data", "model_cache", "results"]:
        src = os.path.join(ROOT, folder)
        dst = os.path.join(DEST, folder)
        create_junction(dst, src)

    print("\n=== news_classification/ 目录树 ===")
    for fn in sorted(os.listdir(DEST)):
        full = os.path.join(DEST, fn)
        flag = "<DIR>" if os.path.isdir(full) else ""
        print(f"  {fn:30s} {flag}")
