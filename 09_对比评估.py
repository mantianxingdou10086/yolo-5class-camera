# 09_对比评估.py —— 在同一个外部测试集上评估多个模型，公平对比
# 用法：py 09_对比评估.py runs/5class/weights/best.pt runs/5class_v2/weights/best.pt
import os, sys

from ultralytics import YOLO

根目录 = os.path.dirname(os.path.abspath(__file__))
测试配置 = os.path.join(根目录, "dataset", "test", "test.yaml")
类别表 = ["bottle", "cup", "phone", "keyboard", "mouse"]
中文 = {"bottle": "水瓶", "cup": "杯子", "phone": "手机", "keyboard": "键盘", "mouse": "鼠标"}

if __name__ == "__main__":
    权重们 = sys.argv[1:] or ["runs/5class/weights/best.pt", "runs/5class_v2/weights/best.pt"]
    结果表 = {}
    for w in 权重们:
        路径 = w if os.path.isabs(w) else os.path.join(根目录, w)
        print(f"\n======== 评估 {w} ========")
        模型 = YOLO(路径)
        指标 = 模型.val(data=测试配置, split="val", verbose=False, project=os.path.join(根目录, "runs"),
                     name="test_eval_" + os.path.basename(os.path.dirname(os.path.dirname(路径))))
        每类 = {}
        for i, 序号 in enumerate(指标.ap_class_index):
            每类[模型.names[序号]] = (指标.box.ap50[i], 指标.box.ap[i])
        结果表[w] = {"总": (指标.box.map50, 指标.box.map), "类": 每类}
        print(f"  mAP50 = {指标.box.map50:.3f}   mAP50-95 = {指标.box.map:.3f}")

    print("\n\n======== 公平对比（同一个外部测试集） ========")
    表头 = f"{'模型':<34}{'mAP50':>8}{'mAP50-95':>11}"
    for 类 in 类别表:
        表头 += f"{中文[类]:>8}"
    print(表头)
    for w, r in 结果表.items():
        行 = f"{w:<34}{r['总'][0]:>8.3f}{r['总'][1]:>11.3f}"
        for 类 in 类别表:
            行 += f"{r['类'].get(类, (0, 0))[0]:>8.3f}"
        print(行)
