# 08_做测试集.py —— 爬一批"全新"图片当外部测试集，用 yolo11x 打教师标签
# 目的：v1 和 v2 的训练数据不同，各自验证集也不同，没法直接比。
#       干脆单独做一份两个模型都没见过、标签标准统一的测试集来公平评估。
# 用法：py 08_做测试集.py
import os, sys, glob, importlib, shutil

import torch
from ultralytics import YOLO

根目录 = os.path.dirname(os.path.abspath(__file__))
测试目录 = os.path.join(根目录, "dataset", "test")
原图目录 = os.path.join(根目录, "dataset", "test_raw")
每类张数 = 80
类别表 = ["bottle", "cup", "phone", "keyboard", "mouse"]
COCO名字 = {"bottle": "bottle", "cup": "cup", "phone": "cell phone",
            "keyboard": "keyboard", "mouse": "mouse"}

# 故意换成和训练时不一样的关键词，尽量不重复抓到训练用过的图
测试关键词 = {
    "bottle":   ["瓶装水 货架", "矿泉水 瓶子 实物", "water bottle isolated", "纯净水瓶"],
    "cup":      ["陶瓷杯 实物", "咖啡纸杯 手持", "mug isolated", "玻璃杯 饮料"],
    "phone":    ["手机 正面 黑色", "smartphone on desk", "手机 实拍", "全面屏手机"],
    "keyboard": ["键盘 黑色 实物", "mechanical keyboard white background", "办公键盘", "键盘 斜拍"],
    "mouse":    ["鼠标 黑色 实物", "wireless mouse white background", "光电鼠标", "鼠标 斜拍"],
}


def 爬测试图():
    爬虫 = importlib.import_module("01_爬图")
    爬虫.类别关键词 = 测试关键词
    爬虫.每类目标张数 = 每类张数
    爬虫.原始目录 = 原图目录
    for 类 in 类别表:
        print(f"\n===== 测试集爬图：{类} =====")
        爬虫.爬一个类别(类)


def 打教师标签():
    设备 = 0 if torch.cuda.is_available() else "cpu"
    模型 = YOLO(os.path.join(根目录, "weights", "yolo11x.pt"))
    名字到id = {v: k for k, v in 模型.names.items()}
    目标id = {名字到id[n]: 类别表.index(类) for 类, n in COCO名字.items()}
    图目录 = os.path.join(测试目录, "images")
    签目录 = os.path.join(测试目录, "labels")
    os.makedirs(图目录, exist_ok=True)
    os.makedirs(签目录, exist_ok=True)
    保留 = 0
    for 类 in 类别表:
        原图们 = sorted(glob.glob(os.path.join(原图目录, 类, "*")))
        for 路径 in 原图们:
            结果 = 模型.predict(路径, imgsz=1024, conf=0.25, iou=0.45,
                               device=设备, verbose=False)[0]
            高, 宽 = 结果.orig_shape
            行们 = []
            if 结果.boxes is not None:
                for 框 in 结果.boxes:
                    cid = int(框.cls.item())
                    if cid not in 目标id:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in 框.xyxy[0]]
                    面积占比 = ((x2 - x1) * (y2 - y1)) / (宽 * 高)
                    if 面积占比 < 0.003 or 面积占比 > 0.95:
                        continue
                    新id = 目标id[cid]
                    行们.append(f"{新id} {(x1 + x2) / 2 / 宽:.6f} {(y1 + y2) / 2 / 高:.6f} "
                              f"{(x2 - x1) / 宽:.6f} {(y2 - y1) / 高:.6f}")
            if not 行们:
                continue
            新名 = f"{类}_{os.path.basename(路径).rsplit('.', 1)[0]}"
            后缀 = os.path.splitext(路径)[1]
            shutil.copy2(路径, os.path.join(图目录, 新名 + 后缀))
            open(os.path.join(签目录, 新名 + ".txt"), "w", encoding="utf-8").write("\n".join(行们) + "\n")
            保留 += 1
    print(f"测试集保留 {保留} 张")

    yaml路径 = os.path.join(测试目录, "test.yaml")
    with open(yaml路径, "w", encoding="utf-8") as f:
        f.write(f"path: {测试目录.replace(os.sep, '/')}\n")
        f.write("train: images\n")
        f.write("val: images\n\n")
        f.write(f"nc: {len(类别表)}\n")
        f.write("names:\n")
        for i, 类 in enumerate(类别表):
            f.write(f"  {i}: {类}\n")
    print("测试集配置：", yaml路径)
    return 保留


if __name__ == "__main__":
    爬测试图()
    打教师标签()
    print("\n外部测试集做好：" + 测试目录)
