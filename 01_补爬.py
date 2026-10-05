# 01_补爬.py —— 给样本偏少的类别多爬点图（键盘/鼠标），其余逻辑和 01_爬图.py 一样
# 用法：py 01_补爬.py
import os, sys

# 复用主爬虫，只把关键词和数量换掉
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib

主爬虫 = importlib.import_module("01_爬图")

主爬虫.类别关键词 = {
    "keyboard": ["机械键盘 桌面", "键盘 俯拍", "keyboard desk", "薄膜键盘",
                 "键盘 白色背景", "客制化键盘", "键盘 特写 黑色", "laptop keyboard"],
    "mouse":    ["鼠标 白色背景", "鼠标 俯拍", "mouse desk", "有线鼠标",
                 "鼠标 特写 黑色", "游戏鼠标 灯", "computer mouse top view", "办公鼠标"],
}
主爬虫.每类目标张数 = 450

if __name__ == "__main__":
    主爬虫.原始目录 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "raw")
    总 = 0
    for 类 in 主爬虫.类别关键词:
        print(f"\n===== 补爬：{类} =====")
        总 += 主爬虫.爬一个类别(类)
    print(f"\n补爬结束，共 {总} 张")
