"""
nametag_bg_to_green.py

把 nametag 图片里的“绿色系背景”统一替换成引擎要求的纯绿 (0,255,0)。

背景（例如豆包/截图工具给的偏黄亮绿）往往并不是精确的 RGB(0,255,0)，
而引擎里 `get_mask(img, (0,255,0))` 只会忽略【精确纯绿】像素，差一点点
都会被当成前景参与匹配，导致模板匹配跑偏。本脚本用 HSV 阈值把所有“绿色
调”的像素判定为背景并刷成纯绿，其余（黑名牌框、白字、蓝色称号条、红色
勋章）保持不变。

用法：
  python3 tools/nametag_bg_to_green.py image.png
  python3 tools/nametag_bg_to_green.py image.png -o nametag/fulilian.png

可选参数用于微调绿色判定范围（默认值已覆盖常见亮绿/黄绿背景）：
  --h-lo / --h-hi   OpenCV 色相范围(0-179)，绿色大约 35-85
  --s-min           最小饱和度(0-255)，过滤发灰的绿
  --v-min           最小明度(0-255)，过滤很暗的绿
  --preview out.png 额外导出一张“背景标红”的可视化图，便于确认判定是否正确

背景目标色（重要，取决于引擎里用的 nametag.mode）：
  --bg green  背景刷成纯绿 (0,255,0)。配合 mode: "grayscale"，引擎用
              get_mask((0,255,0)) 忽略背景。缺点：对半透明底色+变化背景不鲁棒。
  --bg black  背景刷成黑色 (0,0,0)。配合 mode: "white_mask"（推荐）——
              white_mask 只保留白字(亮度150-255)、忽略黑底与一切背景变化，
              角色走到不同背景前分数也稳定。注意：纯绿转灰≈150 会被 white_mask
              误当成白字，所以 white_mask 必须用 black 背景而不是 green。
"""

import argparse
import os
import sys

import cv2
import numpy as np


PURE_GREEN_BGR = (0, 255, 0)  # OpenCV 用 BGR，纯绿同样是 (0,255,0)
BLACK_BGR = (0, 0, 0)

BG_COLORS = {
    "green": PURE_GREEN_BGR,
    "black": BLACK_BGR,
}


def build_green_mask(img_bgr, h_lo, h_hi, s_min, v_min):
    """返回“绿色背景”掩码（255=背景绿，0=保留的前景）。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_lo, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_hi, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # 轻度形态学清理：填掉字缝里零星的非绿点，让背景更干净。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def main():
    ap = argparse.ArgumentParser(
        description="Replace green-ish nametag background with pure green (0,255,0)."
    )
    ap.add_argument("input", help="输入 nametag 图片路径，例如 image.png")
    ap.add_argument("-o", "--output", default=None,
                    help="输出路径（默认在原名后加 _green）")
    ap.add_argument("--h-lo", type=int, default=35, help="绿色色相下限(0-179)")
    ap.add_argument("--h-hi", type=int, default=90, help="绿色色相上限(0-179)")
    ap.add_argument("--s-min", type=int, default=60, help="最小饱和度(0-255)")
    ap.add_argument("--v-min", type=int, default=60, help="最小明度(0-255)")
    ap.add_argument("--bg", choices=["green", "black"], default="green",
                    help="背景目标色：green=纯绿(配 grayscale)，black=黑(配 white_mask，推荐)")
    ap.add_argument("--preview", default=None,
                    help="可选：导出把背景标红的可视化图路径")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 找不到输入文件: {args.input}", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] 无法读取图片(格式不支持?): {args.input}", file=sys.stderr)
        sys.exit(1)

    mask = build_green_mask(img, args.h_lo, args.h_hi, args.s_min, args.v_min)

    bg_color = BG_COLORS[args.bg]
    out = img.copy()
    out[mask == 255] = bg_color

    if args.output:
        out_path = args.output
    else:
        root, ext = os.path.splitext(args.input)
        out_path = f"{root}_green{ext or '.png'}"

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ok = cv2.imwrite(out_path, out)
    if not ok:
        print(f"[ERROR] 写出失败: {out_path}", file=sys.stderr)
        sys.exit(1)

    total = mask.size
    bg = int((mask == 255).sum())
    print(f"[OK] 背景像素 {bg}/{total} ({bg / total:.1%}) 已刷为 {args.bg} {bg_color}")
    print(f"[OK] 已保存: {out_path}")

    if args.preview:
        preview = img.copy()
        preview[mask == 255] = (0, 0, 255)  # 背景标红，便于人工确认
        cv2.imwrite(args.preview, preview)
        print(f"[OK] 可视化(背景标红)已保存: {args.preview}")


if __name__ == "__main__":
    main()
