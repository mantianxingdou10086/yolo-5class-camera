# 02_清洗并划分数据集.py —— 去掉重复/损坏/太小的图，按 9:1 分成 train/val
# 用法：py 02_清洗并划分数据集.py
import os, hashlib, random, shutil

根目录 = os.path.dirname(os.path.abspath(__file__))
原始目录 = os.path.join(根目录, "dataset", "raw")
数据目录 = os.path.join(根目录, "dataset")
验证集比例 = 0.1              # 10% 做验证集
类别表 = ["bottle", "cup", "phone", "keyboard", "mouse"]   # 顺序就是类别 id 0~4

random.seed(0)


def 读图(path):
    import cv2
    import numpy as np
    数据 = open(path, "rb").read()
    图 = cv2.imdecode(np.frombuffer(数据, dtype="uint8"), cv2.IMREAD_COLOR)
    return 图, 数据


if __name__ == "__main__":
    # 先清空旧的 images/labels，避免重复残留
    for 子 in ["images/train", "images/val", "labels/train", "labels/val"]:
        p = os.path.join(数据目录, *子.split("/"))
        if os.path.isdir(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)

    全局指纹 = set()
    统计 = {}
    for 类别 in 类别表:
        源目录 = os.path.join(原始目录, 类别)
        if not os.path.isdir(源目录):
            print(f"[跳过] 没有 {类别} 目录")
            continue
        文件列表 = sorted(os.listdir(源目录))
        random.shuffle(文件列表)

        好的文件 = []
        for 文件名 in 文件列表:
            路径 = os.path.join(源目录, 文件名)
            try:
                图, 数据 = 读图(路径)
            except Exception:
                continue
            if 图 is None:
                continue
            高, 宽 = 图.shape[:2]
            if 宽 < 250 or 高 < 250:
                continue
            指纹 = hashlib.md5(数据).hexdigest()
            if 指纹 in 全局指纹:      # 全局去重（跨类别也去，防止同一张图混进两类）
                continue
            全局指纹.add(指纹)
            好的文件.append((路径, 文件名, 后缀 := os.path.splitext(文件名)[1]))

        随机序 = 好的文件[:]
        random.shuffle(随机序)
        验证数 = max(1, int(len(随机序) * 验证集比例))
        验证集 = set(名字 for _, 名字, _ in 随机序[:验证数])

        # 全部先复制进 train，验证集那些再挪到 val（保证文件名唯一：类别_原名）
        for 路径, 文件名, 后缀 in 好的文件:
            新名 = f"{类别}_{文件名.rsplit('.', 1)[0]}{后缀}"
            子集 = "val" if 文件名 in 验证集 else "train"
            目标 = os.path.join(数据目录, "images", 子集, 新名)
            shutil.copy2(路径, 目标)

        统计[类别] = (len(好的文件) - 验证数, 验证数)
        print(f"[{类别}] 有效 {len(好的文件)} 张 -> train {len(好的文件) - 验证数} / val {验证数}")

    # 写 data.yaml（YOLO 训练用的配置文件）
    yaml路径 = os.path.join(数据目录, "data.yaml")
    with open(yaml路径, "w", encoding="utf-8") as f:
        f.write(f"path: {数据目录.replace(os.sep, '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(类别表)}\n")
        f.write("names:\n")
        for i, 类别 in enumerate(类别表):
            f.write(f"  {i}: {类别}\n")

    总train = sum(v[0] for v in 统计.values())
    总val = sum(v[1] for v in 统计.values())
    print(f"\n划分完成：train {总train} 张 / val {总val} 张")
    print(f"配置文件：{yaml路径}")
