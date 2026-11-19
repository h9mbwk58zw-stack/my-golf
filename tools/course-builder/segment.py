#!/usr/bin/env python3
"""航空写真を色と模様で塗り分ける（芝・フェアウェイ・池・バンカー・林）。

使い方:  python3 segment.py WORKDIR
出力:
  WORKDIR/masks.npz      各地物のマスク（turf, fw, water, bunker, area）
  WORKDIR/segment.jpg    左：写真 / 右：塗り分け結果（確認用）
しきい値は写真の季節や明るさで変わるので、結果を見て必要なら調整する。
"""
import sys, os
import numpy as np, cv2
from scipy import ndimage as ndi


def keep_big(mask, min_px, max_px=None):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sz = ndi.sum(mask, lab, range(1, n + 1))
    ok = sz > min_px if max_px is None else (sz > min_px) & (sz < max_px)
    return np.isin(lab, 1 + np.where(ok)[0])


def segment(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = cv2.blur(g, (9, 9))
    tex = np.sqrt(np.maximum(cv2.blur(g * g, (9, 9)) - m * m, 0))
    green = (H > 28) & (H < 85) & (S > 40)
    # 芝（ラフ＋フェアウェイ＋グリーン）：緑で模様が少なく明るい。林は模様が多いか暗い
    turf = green & (tex < 9) & (V > 95)
    turf = ndi.binary_closing(ndi.binary_opening(turf, iterations=2), iterations=4)
    turf = keep_big(turf, 3000)
    trees = green & ~turf
    # コースの範囲：芝のまとまりの外形
    # 道路などで分かれた部分も含めるため、大きなまとまりごとに外形をとる
    grown = ndi.binary_dilation(turf, iterations=40)
    lab, n = ndi.label(grown)
    sz = ndi.sum(grown, lab, range(1, n + 1))
    area = np.zeros(turf.shape, np.uint8)
    for i in np.where(sz >= 0.02 * sz.max())[0]:
        part = ndi.binary_erosion(lab == i + 1, iterations=25)
        if part.any():
            hull = cv2.convexHull(np.argwhere(part)[:, ::-1].astype(np.int32))
            cv2.fillPoly(area, [hull], 1)
    area = area.astype(bool)
    turf &= area
    # フェアウェイ：芝の中でも明るい部分
    fw = turf & (L > np.percentile(L[turf], 45))
    fw = keep_big(ndi.binary_closing(ndi.binary_opening(fw, iterations=3), iterations=5), 1500)
    # 池：暗くて模様が少ない。芝か林に接しているものだけ
    water0 = (V < 110) & (tex < 7) & (S < 140)
    water0 = keep_big(ndi.binary_opening(water0, iterations=3), 800)
    water0 = ndi.binary_fill_holes(ndi.binary_closing(water0, iterations=3))
    lw, n = ndi.label(water0)
    # コースの端にある池も範囲に入れる（池に接している範囲は、池ごと外形をとり直す）
    near = np.unique(lw[ndi.binary_dilation(area, iterations=15) & water0]); near = near[near > 0]
    grown = area | np.isin(lw, near)
    lab2, n2 = ndi.label(grown)
    area = np.zeros(turf.shape, np.uint8)
    for i in range(1, n2 + 1):
        hull = cv2.convexHull(np.argwhere(lab2 == i)[:, ::-1].astype(np.int32))
        cv2.fillPoly(area, [hull], 1)
    area = area.astype(bool)
    t = np.unique(lw[area & water0 & ndi.binary_dilation(turf | trees, iterations=3)])
    water = keep_big(np.isin(lw, t[t > 0]) & area, 1500)
    # バンカー：明るく彩度の低い砂。芝の近くだけ
    bk = (V > 165) & (S < 80) & area & ~water
    bk = ndi.binary_closing(ndi.binary_opening(bk, iterations=1), iterations=2)
    bk = keep_big(bk, 20, 25000) & ndi.binary_dilation(turf, iterations=10)
    return dict(turf=turf, fw=fw, water=water, bunker=bk, area=area)


def preview(im, M):
    vis = np.full_like(im, (36, 33, 30))
    vis[M["area"]] = (40, 95, 40)
    vis[M["turf"]] = (80, 165, 80)
    vis[M["fw"]] = (120, 225, 130)
    vis[M["water"]] = (215, 140, 60)
    vis[M["bunker"]] = (170, 225, 245)
    w, h = im.shape[1] // 2, im.shape[0] // 2
    return np.hstack([cv2.resize(im, (w, h)), np.full((h, 12, 3), 255, np.uint8),
                      cv2.resize(vis, (w, h), interpolation=cv2.INTER_AREA)])


if __name__ == "__main__":
    wd = sys.argv[1]
    im = cv2.imread(os.path.join(wd, "aerial.jpg"))
    M = segment(im)
    np.savez_compressed(os.path.join(wd, "masks.npz"), **M)
    cv2.imwrite(os.path.join(wd, "segment.jpg"), preview(im, M), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print({k: round(float(v.mean()), 4) for k, v in M.items()})
