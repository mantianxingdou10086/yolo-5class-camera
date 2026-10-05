# 05_摄像头识别.py —— 打开摄像头实时识别（按 q 退出）
# 用法：py 05_摄像头识别.py                     （用最新训练好的 v2 模型）
#       py 05_摄像头识别.py runs/5class/weights/best.pt   （指定别的模型）
import os, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

根目录 = os.path.dirname(os.path.abspath(__file__))
# 命令行给了权重就用给的，没给就用 v2 的（没有 v2 就退回 v1）
权重 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(根目录, "runs", "5class_v2", "weights", "best.pt")
if not os.path.isabs(权重):
    权重 = os.path.join(根目录, 权重)
if not os.path.exists(权重):
    权重 = os.path.join(根目录, "runs", "5class", "weights", "best.pt")
置信度 = 0.35           # 低于这个分数不显示
摄像头编号 = 0          # 0 一般是笔记本自带摄像头

# 中文标签要用 PIL 画（OpenCV 的 putText 不支持中文）
中文字体 = "C:/Windows/Fonts/msyh.ttc"
字体 = ImageFont.truetype(中文字体, 22) if os.path.exists(中文字体) else ImageFont.load_default()

if __name__ == "__main__":
    print(f"加载模型：{权重}（存在={os.path.exists(权重)}）")
    模型 = YOLO(权重)

    摄像头 = cv2.VideoCapture(摄像头编号)
    if not 摄像头.isOpened():
        raise SystemExit("打不开摄像头，检查是不是被别的程序占用了")

    print("摄像头已打开，按 q 键退出")
    while True:
        成功, 画面 = 摄像头.read()
        if not 成功:
            print("读不到画面")
            break

        结果 = 模型.predict(画面, conf=置信度, verbose=False)[0]

        # 把画面交给 PIL，用中文字体画标签
        图 = Image.fromarray(cv2.cvtColor(画面, cv2.COLOR_BGR2RGB))
        画笔 = ImageDraw.Draw(图)

        for 框 in 结果.boxes:
            x1, y1, x2, y2 = [int(v) for v in 框.xyxy[0]]
            类别名 = 模型.names[int(框.cls.item())]
            分数 = float(框.conf.item())
            文字 = f"{类别名} {分数:.2f}"
            颜色 = (0, 255, 0)

            画笔.rectangle([x1, y1, x2, y2], outline=颜色, width=3)
            # 文字背景条，避免字看不清
            左, 上, 右, 下 = 画笔.textbbox((x1, max(0, y1 - 26)), 文字, font=字体)
            画笔.rectangle([左 - 3, 上 - 2, 右 + 3, 下 + 2], fill=颜色)
            画笔.text((x1, max(0, y1 - 26)), 文字, font=字体, fill=(0, 0, 0))

        画面 = cv2.cvtColor(np.array(图), cv2.COLOR_RGB2BGR)
        cv2.imshow("YOLO 摄像头识别（按 q 退出）", 画面)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    摄像头.release()
    cv2.destroyAllWindows()
