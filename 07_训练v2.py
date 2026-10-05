# 07_训练v2.py —— 第二轮训练：用了补爬的更多键盘/鼠标数据 + 更干净的标签
# 用法：py 07_训练v2.py
import os
from ultralytics import YOLO

根目录 = os.path.dirname(os.path.abspath(__file__))
数据配置 = os.path.join(根目录, "dataset", "data.yaml")
预训练权重 = os.path.join(根目录, "weights", "yolo11s.pt")

amp文件 = os.path.join(根目录, "weights", "yolo26n.pt")
能用amp = False
if os.path.exists(amp文件):
    try:
        import torch
        torch.load(amp文件, map_location="cpu", weights_only=False)
        能用amp = True
    except Exception:
        os.remove(amp文件)

if __name__ == "__main__":
    模型 = YOLO(预训练权重)
    模型.train(
        data=数据配置,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=25,
        amp=能用amp,
        seed=0,
        project=os.path.join(根目录, "runs"),
        name="5class_v2",
        exist_ok=True,
        plots=True,
    )
    print("\nv2 训练结束：runs/5class_v2/weights/best.pt")
