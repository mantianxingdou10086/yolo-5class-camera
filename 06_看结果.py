# 06_看结果.py —— 打印每类精度，并挑几张验证集图片生成带中文框的预览图
# 用法：py 06_看结果.py                          （默认看 runs/5class）
#       py 06_看结果.py runs/5class_v2/weights/best.pt   （指定别的模型）
import os, glob, random, sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

根目录 = os.path.dirname(os.path.abspath(__file__))
# 命令行给了权重就用给的，没给就用 runs/5class 的
权重 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(根目录, "runs", "5class", "weights", "best.pt")
if not os.path.isabs(权重):
    权重 = os.path.join(根目录, 权重)
预览目录 = os.path.join(os.path.dirname(os.path.dirname(权重)), "预览图")
类别中文 = {"bottle": "水瓶", "cup": "杯子", "phone": "手机",
            "keyboard": "键盘", "mouse": "鼠标"}
字体 = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)

if __name__ == "__main__":
    模型 = YOLO(权重)

    # 1) 在验证集上算一遍指标，看看每类准不准
    print("在验证集上评估……")
    指标 = 模型.val(data=os.path.join(根目录, "dataset", "data.yaml"),
                  split="val", verbose=False)
    print(f"\n总体：mAP50 = {指标.box.map50:.3f}   mAP50-95 = {指标.box.map:.3f}")
    print("每一类：")
    for i, 序号 in enumerate(指标.ap_class_index):
        print(f"  {类别中文.get(模型.names[序号], 模型.names[序号]):4s} "
              f"mAP50 = {指标.box.ap50[i]:.3f}   mAP50-95 = {指标.box.ap[i]:.3f}")

    # 2) 挑 12 张验证集图片画框，存成预览图方便肉眼看
    os.makedirs(预览目录, exist_ok=True)
    图片们 = glob.glob(os.path.join(根目录, "dataset", "images", "val", "*"))
    random.seed(1)
    random.shuffle(图片们)
    做好的 = 0
    for 图片路径 in 图片们[:12]:
        图 = cv2.imread(图片路径)
        结果 = 模型.predict(图, conf=0.35, verbose=False)[0]
        img = Image.fromarray(cv2.cvtColor(图, cv2.COLOR_BGR2RGB))
        画笔 = ImageDraw.Draw(img)
        for 框 in 结果.boxes:
            x1, y1, x2, y2 = [int(v) for v in 框.xyxy[0]]
            名字 = 模型.names[int(框.cls.item())]
            文字 = f"{类别中文.get(名字, 名字)} {float(框.conf.item()):.2f}"
            画笔.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            左, 上, 右, 下 = 画笔.textbbox((x1, max(0, y1 - 28)), 文字, font=字体)
            画笔.rectangle([左 - 3, 上 - 2, 右 + 3, 下 + 2], fill=(0, 255, 0))
            画笔.text((x1, max(0, y1 - 28)), 文字, font=字体, fill=(0, 0, 0))
        保存路径 = os.path.join(预览目录, f"预览_{做好的:02d}.jpg")
        cv2.imwrite(保存路径, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        做好的 += 1

    print(f"\n已生成 {做好的} 张预览图：{预览目录}")
    print(f"最终模型：{权重}")
