#!/usr/bin/env python3
"""ホールのティー・グリーン位置を読み取るための、方眼つき画像を作る。

使い方:  python3 grid.py WORKDIR [--step 100] [--crop x0,y0,x1,y1] [--out grid.jpg]
方眼の数字は aerial.jpg のピクセル座標。holes.json の座標はこの数字で書く。
--crop で一部を拡大すると、細かい位置を読みやすい。
"""
import argparse, os, cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--crop", default=None)
    ap.add_argument("--out", default="grid.jpg")
    ap.add_argument("--maxw", type=int, default=1400)
    a = ap.parse_args()
    im = cv2.imread(os.path.join(a.workdir, "aerial.jpg"))
    x0, y0, x1, y1 = 0, 0, im.shape[1], im.shape[0]
    if a.crop:
        x0, y0, x1, y1 = map(int, a.crop.split(","))
    im = im[y0:y1, x0:x1].copy()
    s = min(1.0, a.maxw / im.shape[1])
    im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
    st = a.step
    for gx in range((x0 // st + 1) * st, x1, st):
        X = int((gx - x0) * s)
        cv2.line(im, (X, 0), (X, im.shape[0]), (0, 255, 255), 1)
        cv2.putText(im, str(gx), (X + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(im, str(gx), (X + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    for gy in range((y0 // st + 1) * st, y1, st):
        Y = int((gy - y0) * s)
        cv2.line(im, (0, Y), (im.shape[1], Y), (0, 255, 255), 1)
        cv2.putText(im, str(gy), (2, Y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(im, str(gy), (2, Y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(a.workdir, a.out), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(os.path.join(a.workdir, a.out), im.shape[1], im.shape[0])


if __name__ == "__main__":
    main()
