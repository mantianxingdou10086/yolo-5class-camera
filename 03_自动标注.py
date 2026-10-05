# 03_自动标注.py —— 用 COCO 预训练的 YOLO11x 给爬来的图自动打框，生成 YOLO 格式标签
# 原理：COCO 数据集里本来就有 bottle / cup / cell phone / keyboard / mouse 这 5 类，
#       所以拿官方预训练模型跑一遍，把命中这 5 类的框写下来，就是（伪）标注。
# v2 改进：① NMS 阈值调低，减少同一物体上的重复框
#          ② 同一类里高度重叠的框只留置信度最高的那个（v1 的键盘被框了两遍，就是这么来的）
# 用法：py 03_自动标注.py
import os, glob

import torch
from ultralytics import YOLO

根目录 = os.path.dirname(os.path.abspath(__file__))
数据目录 = os.path.join(根目录, "dataset")
预训练权重 = os.path.join(根目录, "weights", "yolo11x.pt")

类别表 = ["bottle", "cup", "phone", "keyboard", "mouse"]
COCO名字 = {"bottle": "bottle", "cup": "cup", "phone": "cell phone",
            "keyboard": "keyboard", "mouse": "mouse"}

置信度阈值 = 0.25
IOU阈值 = 0.45           # 调低：让 NMS 更狠一点，压掉重叠框
推理尺寸 = 1024
最小框面积占比 = 0.003
最大框面积占比 = 0.95
同类别重叠阈值 = 0.55    # 同一类里两个框重叠超过这个比例，只留分数高的


def 重叠比例(框甲, 框乙):
    """算两个框的交并比 IoU"""
    甲左, 甲上, 甲右, 甲下 = 框甲[1], 框甲[2], 框甲[3], 框甲[4]
    乙左, 乙上, 乙右, 乙下 = 框乙[1], 框乙[2], 框乙[3], 框乙[4]
    交宽 = max(0.0, min(甲右, 乙右) - max(甲左, 乙左))
    交高 = max(0.0, min(甲下, 乙下) - max(甲上, 乙上))
    交面积 = 交宽 * 交高
    甲面积 = (甲右 - 甲左) * (甲下 - 甲上)
    乙面积 = (乙右 - 乙左) * (乙下 - 乙上)
    并面积 = 甲面积 + 乙面积 - 交面积
    return 交面积 / 并面积 if 并面积 > 0 else 0.0


def 主流程():
    设备 = 0 if torch.cuda.is_available() else "cpu"
    print(f"使用的设备：{设备}（{torch.cuda.get_device_name(0) if 设备 == 0 else 'CPU'}）")
    模型 = YOLO(预训练权重)
    名字表 = 模型.names
    名字到id = {v: k for k, v in 名字表.items()}
    目标id = {名字到id[n]: 类别表.index(类) for 类, n in COCO名字.items() if n in 名字到id}
    print("要保留的 COCO 类别：", {名字表[k]: 类别表[v] for k, v in 目标id.items()})

    统计 = {类: 0 for 类 in 类别表}
    丢掉的图 = 0
    去重掉的框 = 0
    总图数 = 0

    for 子集 in ["train", "val"]:
        图片目录 = os.path.join(数据目录, "images", 子集)
        标签目录 = os.path.join(数据目录, "labels", 子集)
        os.makedirs(标签目录, exist_ok=True)
        图片列表 = sorted(glob.glob(os.path.join(图片目录, "*")))
        print(f"\n[{子集}] 共 {len(图片列表)} 张图，开始自动标注……")
        本子集丢掉 = 0

        for i, 图片路径 in enumerate(图片列表):
            总图数 += 1
            结果 = 模型.predict(图片路径, imgsz=推理尺寸, conf=置信度阈值,
                               iou=IOU阈值, device=设备, verbose=False)[0]
            高, 宽 = 结果.orig_shape

            # 先收集所有候选框：(新类别id, x1, y1, x2, y2, 分数)
            候选 = []
            if 结果.boxes is not None:
                for 框 in 结果.boxes:
                    cid = int(框.cls.item())
                    if cid not in 目标id:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in 框.xyxy[0]]
                    面积占比 = ((x2 - x1) * (y2 - y1)) / (宽 * 高)
                    if 面积占比 < 最小框面积占比 or 面积占比 > 最大框面积占比:
                        continue
                    候选.append((目标id[cid], x1, y1, x2, y2, float(框.conf.item())))

            # 同一类里重叠太多的框，只留分数最高的
            候选.sort(key=lambda b: -b[5])
            保留 = []
            for 框 in 候选:
                if any(框[0] == 已有[0] and 重叠比例(框, 已有) > 同类别重叠阈值 for 已有 in 保留):
                    去重掉的框 += 1
                    continue
                保留.append(框)

            行们 = []
            for 新id, x1, y1, x2, y2, _ in 保留:
                # YOLO 标签格式：类别 cx cy w h（都是相对整图的 0~1 小数）
                行们.append(f"{新id} {(x1 + x2) / 2 / 宽:.6f} {(y1 + y2) / 2 / 高:.6f} "
                          f"{(x2 - x1) / 宽:.6f} {(y2 - y1) / 高:.6f}")
                统计[类别表[新id]] += 1

            if 行们:
                标签路径 = os.path.join(标签目录,
                                    os.path.splitext(os.path.basename(图片路径))[0] + ".txt")
                open(标签路径, "w", encoding="utf-8").write("\n".join(行们) + "\n")
            else:
                os.remove(图片路径)      # 一个目标都没有的图直接删，免得变成噪声
                丢掉的图 += 1
                本子集丢掉 += 1

            if (i + 1) % 200 == 0:
                print(f"  已处理 {i + 1}/{len(图片列表)} 张（本子集丢 {本子集丢掉} 张）")

    print(f"\n自动标注完成：处理 {总图数} 张，丢掉无目标 {丢掉的图} 张，"
          f"去重重叠框 {去重掉的框} 个")
    print("各类目标框数量：")
    for 类, 数量 in 统计.items():
        print(f"  {类:9s} {数量} 个框")


if __name__ == "__main__":
    主流程()
