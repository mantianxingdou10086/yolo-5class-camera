# 04_训练.py —— 用自动标注好的数据集训练 YOLO
# 用法：py 04_训练.py
import os
from ultralytics import YOLO

根目录 = os.path.dirname(os.path.abspath(__file__))
数据配置 = os.path.join(根目录, "dataset", "data.yaml")
预训练权重 = os.path.join(根目录, "weights", "yolo11s.pt")   # s 版：8G 显存够用，速度快

轮数 = 100           # 训练 100 轮
尺寸 = 640           # 输入图片边长
批大小 = 16          # 一批 16 张，8G 显存刚好
# AMP 自检会去网上下 yolo26n.pt，本机下不到就直接关掉（速度慢一点点，不影响结果）
amp文件 = os.path.join(根目录, "weights", "yolo26n.pt")
能用amp = False
if os.path.exists(amp文件):
    try:
        import torch
        torch.load(amp文件, map_location="cpu", weights_only=False)   # 能加载才说明文件是完整的
        能用amp = True
    except Exception:
        os.remove(amp文件)           # 残缺文件会让训练直接报错，先删掉

if __name__ == "__main__":
    print(f"数据配置：{数据配置}")
    print(f"预训练权重：{预训练权重}（存在={os.path.exists(预训练权重)}）")
    print(f"AMP 加速：{'开' if 能用amp else '关'}")

    模型 = YOLO(预训练权重)      # 从 COCO 预训练权重接着学，比从零快得多也准得多
    模型.train(
        data=数据配置,
        epochs=轮数,
        imgsz=尺寸,
        batch=批大小,
        device=0,                # 用第 0 号 GPU
        workers=4,
        patience=25,             # 25 轮没进步就早停
        amp=能用amp,
        seed=0,
        project=os.path.join(根目录, "runs"),
        name="5class",
        exist_ok=True,
        plots=True,
    )
    print("\n训练结束，最好的权重在：runs/5class/weights/best.pt")
