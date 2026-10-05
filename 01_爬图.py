# 01_爬图.py —— 从必应图片搜索批量下载训练图片（多线程版）
# 用法：py 01_爬图.py
import os, re, time, random, urllib.request, urllib.parse, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== 要爬的 5 个类别：文件夹名 -> 搜索关键词列表 =====
# 关键词给多一些、每类翻页少一些：必应首页的图片链接质量最高，深页大多是垃圾外链
类别关键词 = {
    "bottle":   ["矿泉水瓶", "塑料水瓶", "矿泉水瓶 特写", "water bottle", "plastic bottle",
                 "饮料瓶", "宝特瓶", "水瓶 商品"],
    "cup":      ["马克杯", "纸杯", "咖啡杯", "coffee cup", "mug",
                 "一次性纸杯", "杯子 特写", "陶瓷杯"],
    "phone":    ["手机", "智能手机", "smartphone", "iphone",
                 "手机 特写", "安卓手机", "mobile phone", "手机 桌面"],
    "keyboard": ["键盘", "电脑键盘", "机械键盘", "keyboard",
                 "笔记本键盘", "键盘 特写", "游戏键盘", "keyboard close up"],
    "mouse":    ["鼠标", "电脑鼠标", "无线鼠标", "computer mouse",
                 "gaming mouse", "鼠标 特写", "游戏鼠标", "mouse close up"],
}
每类目标张数 = 350          # 每类至少拿到这么多张原图
每页数量 = 35               # 必应一页给 35 个图片链接
下载线程数 = 64             # 并发下载线程（实测：图床响应慢，靠线程数堆吞吐）
原始目录 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "raw")
浏览器标识 = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
锁 = __import__("threading").Lock()


# 抓一个网址的内容（图片返回 bytes，网页返回 str）
def 抓取(url, 超时=8):
    请求 = urllib.request.Request(url, headers={
        "User-Agent": 浏览器标识,
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Referer": "https://www.bing.com/images/",
    })
    return urllib.request.urlopen(请求, timeout=超时).read()


# 用必应 async 接口拿到一页图片链接
def 搜一页(关键词, 起始位置):
    q = urllib.parse.quote(关键词)
    网址 = (f"https://www.bing.com/images/async?q={q}"
            f"&first={起始位置}&count={每页数量}&mmasync=1")
    html = 抓取(网址, 超时=15).decode("utf-8", "ignore")
    链接 = re.findall(r'murl&quot;:&quot;(.*?)&quot;', html)
    if not 链接:
        链接 = re.findall(r'"murl":"(.*?)"', html)
    return [u for u in 链接 if re.search(r"\.(jpg|jpeg|png|webp)(\?|$)", u, re.I)]


# 校验图片：能解码、宽高都不小于 250、长宽比不过分
def 是合格图片(数据):
    try:
        import cv2
        import numpy as np
        图 = cv2.imdecode(np.frombuffer(数据, dtype="uint8"), cv2.IMREAD_COLOR)
        if 图 is None:
            return False
        高, 宽 = 图.shape[:2]
        return 宽 >= 250 and 高 >= 250 and max(宽 / 高, 高 / 宽) <= 4.0
    except Exception:
        return False


# 下载单张图片：成功返回 (指纹, 后缀, 数据)，失败返回 None
def 下载一张(链接):
    try:
        数据 = 抓取(链接, 超时=8)
    except Exception:
        return None
    if len(数据) < 8000 or not 是合格图片(数据):
        return None
    后缀 = os.path.splitext(urllib.parse.urlparse(链接).path)[1].lower()
    if 后缀 not in (".jpg", ".jpeg", ".png", ".webp"):
        后缀 = ".jpg"
    return hashlib.md5(数据).hexdigest(), 后缀, 数据


def 爬一个类别(类别):
    保存目录 = os.path.join(原始目录, 类别)
    os.makedirs(保存目录, exist_ok=True)
    已存指纹 = {f.split("_")[0] for f in os.listdir(保存目录)}
    已下载数 = len(已存指纹)
    print(f"  已有 {已下载数} 张")

    # ---- 第一步：搜索，攒够 3 倍的候选链接 ----
    链接池 = []
    for 关键词 in 类别关键词[类别]:
        for 起始 in range(0, 每页数量 * 5, 每页数量):     # 每个关键词只翻 5 页
            前 = len(链接池)
            try:
                链接池 += 搜一页(关键词, 起始)
            except Exception as e:
                print(f"    [警告] 搜索失败 {关键词} first={起始}: {e}")
                break
            if len(链接池) == 前:      # 这页没有新东西了，换关键词
                break
            time.sleep(0.3)
        if len(链接池) >= (每类目标张数 - 已下载数) * 3:
            break
    random.shuffle(链接池)
    print(f"  候选链接 {len(链接池)} 条，开始并发下载")

    # ---- 第二步：12 线程并发下载 ----
    成功 = 已下载数
    用过的链接 = 0
    with ThreadPoolExecutor(max_workers=下载线程数) as 池:
        # 分批提交，边下边判断够没够，避免下太多
        for 批起点 in range(0, len(链接池), 60):
            if 成功 >= 每类目标张数:
                break
            批 = 链接池[批起点:批起点 + 60]
            用过的链接 += len(批)
            for 结果 in 池.map(下载一张, 批):
                if 结果 is None or 成功 >= 每类目标张数:
                    continue
                指纹, 后缀, 数据 = 结果
                with 锁:
                    if 指纹 in 已存指纹:
                        continue
                    已存指纹.add(指纹)
                    open(os.path.join(保存目录, f"{指纹}_{成功}{后缀}"), "wb").write(数据)
                    成功 += 1
                    if 成功 % 50 == 0:
                        print(f"  [{类别}] 已下载 {成功} 张")

    print(f"[完成] {类别}: {成功} 张（试了 {用过的链接} 条链接）-> {保存目录}")
    return 成功


if __name__ == "__main__":
    总数 = 0
    for 类别 in 类别关键词:
        print(f"\n===== 开始爬：{类别} =====")
        总数 += 爬一个类别(类别)
    print(f"\n全部完成，raw 目录共 {总数} 张原图")
