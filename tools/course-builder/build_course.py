#!/usr/bin/env python3
"""holes.json（ホールの番号・パー・距離・ティー〜グリーンの位置）と塗り分け結果から、
MY GOLF のコースデータ（ホールごとの座標、ヤード単位）を作る。

使い方:  python3 build_course.py WORKDIR
入力:  WORKDIR/holes.json, WORKDIR/masks.npz, WORKDIR/meta.json, WORKDIR/aerial.jpg
出力:
  WORKDIR/course.json   insert_course.py に渡すコース情報（raw はアプリの形式）
  WORKDIR/check.jpg     航空写真にホールの線と番号を重ねた確認用画像
  WORKDIR/holes.jpg     各ホールをアプリと同じ向き（ティーが下・グリーンが上）で描いた確認用画像

holes.json の書き方（座標は aerial.jpg のピクセル。grid.py の方眼で読む）:
{
 "id": "minori", "name": "○○ゴルフクラブ", "short": "○○GC", "pref": "茨城県",
 "tees": [{"id":"regular","name":"レギュラー","color":"#f4f4f4","yd":[18ホール分]}],
 "defTee": "regular", "teeNote": "ティーの距離は○○のスコアカード。",
 "credit": "コース図は国土地理院の空中写真をもとに作成した目安です。",
 "holes": [{"n":1, "par":4, "line":[[ティーx,y],[曲がり角x,y],[グリーンx,y]]}, ...]
}
"line" はティーから順に、ドッグレッグの曲がり角（あれば）、最後にグリーンの中心。
"""
import json, math, os, sys
import numpy as np, cv2
from scipy import ndimage as ndi

CW_YD = 45        # アプリと同じ：センターラインから左右45ydがそのホールの範囲
YD = 0.9144
GREEN_R_YD = 13   # グリーンの半径の目安
TYPES = [("A", "water", 3.0, 300), ("P", None, 3.0, 400), ("F", "fw", 2.5, 250), ("B", "bunker", 1.2, 15)]


def snap(pt, mask, r):
    x, y = int(pt[0]), int(pt[1])
    y0, y1, x0, x1 = max(0, y - r), min(mask.shape[0], y + r + 1), max(0, x - r), min(mask.shape[1], x + r + 1)
    sub = mask[y0:y1, x0:x1]
    if sub[y - y0, x - x0] or not sub.any():
        return [x, y]
    ys, xs = np.nonzero(sub)
    i = np.argmin((xs + x0 - x) ** 2 + (ys + y0 - y) ** 2)
    return [int(xs[i] + x0), int(ys[i] + y0)]


def frame(line_px, mpp):
    """ピクセル → アプリの座標（ティー=原点、グリーン=+y、右=+x、ヤード）"""
    t, g = np.array(line_px[0], float), np.array(line_px[-1], float)
    v = (g - t) * [1, -1]                       # 北が+yになるように反転
    u = v / np.linalg.norm(v)
    r = np.array([u[1], -u[0]])                 # 進行方向の右
    def tf(p):
        d = (np.array(p, float) - t) * [1, -1] * mpp / YD
        return [float(d @ r), float(d @ u)]
    return tf


def polys(mask, eps_m, min_m2, mpp, tf, prefix):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if cv2.contourArea(c) * mpp * mpp < min_m2:
            continue
        a = cv2.approxPolyDP(c, max(1.0, eps_m / mpp), True)[:, 0, :]
        if len(a) < 3:
            continue
        pts = [tf(p) for p in a]
        out.append(prefix + " ".join(f"{round(x)},{round(y)}" for x, y in pts))
    return out


def build(wd):
    meta = json.load(open(os.path.join(wd, "meta.json")))
    spec = json.load(open(os.path.join(wd, "holes.json")))
    M = dict(np.load(os.path.join(wd, "masks.npz")))
    mpp = meta["mpp"]
    im = cv2.imread(os.path.join(wd, "aerial.jpg"))
    check = im.copy()
    tees = spec["tees"]
    deft = next(t for t in tees if t["id"] == spec["defTee"])
    lines, sheets = [], []
    corr_px = int(CW_YD * YD / mpp)
    for h in sorted(spec["holes"], key=lambda h: h["n"]):
        n = h["n"]
        lp = [list(map(int, p)) for p in h["line"]]
        lp[0] = snap(lp[0], M["turf"], 30)
        lp[-1] = snap(lp[-1], M["turf"], 30)
        tf = frame(lp, mpp)
        corr = np.zeros(M["turf"].shape, np.uint8)
        cv2.polylines(corr, [np.array(lp, np.int32)], False, 1, thickness=2 * corr_px)
        corr = corr.astype(bool)
        lmask = np.zeros_like(corr, np.uint8)
        cv2.polylines(lmask, [np.array(lp, np.int32)], False, 1, thickness=max(3, int(8 / mpp)))
        gcirc = np.zeros_like(lmask)
        cv2.circle(gcirc, tuple(lp[-1]), int(GREEN_R_YD * YD / mpp), 1, -1)
        # プレー範囲 = 芝のうちホールの線に触れているまとまり＋グリーン
        p = (M["turf"] | gcirc.astype(bool)) & corr
        lab, k = ndi.label(p)
        hit = np.unique(lab[lmask.astype(bool) & p]); hit = hit[hit > 0]
        P = ndi.binary_closing(np.isin(lab, hit), iterations=3) & corr
        near = ndi.binary_dilation(P, iterations=int(6 / mpp)) & corr
        masks = {"P": P, "F": M["fw"] & P, "B": M["bunker"] & near,
                 "A": M["water"] & ndi.binary_dilation(corr, iterations=int(10 / mpp))}
        feats = []
        for t, _, eps, mn in TYPES:
            feats += polys(masks[t], eps, mn, mpp, tf, t)
        # グリーン（円の目安）
        gx, gy = tf(lp[-1])
        rg = GREEN_R_YD
        feats.append("G" + " ".join(f"{round(gx + rg * math.cos(a))},{round(gy + rg * 0.85 * math.sin(a))}" for a in np.linspace(0, 2 * math.pi, 14, endpoint=False)))
        ln = [tf(q) for q in lp]
        L = sum(math.dist(ln[i], ln[i + 1]) for i in range(len(ln) - 1))
        lines.append(f"{n};{h['par']};{deft['yd'][n - 1]};{round(L)};" + " ".join(f"{round(x)},{round(y)}" for x, y in ln) + ";" + "|".join(feats))
        # 確認用：航空写真に線と番号
        cv2.polylines(check, [np.array(lp, np.int32)], False, (0, 255, 255), 3)
        cv2.circle(check, tuple(lp[0]), 7, (255, 255, 255), -1)
        cv2.circle(check, tuple(lp[-1]), 9, (40, 40, 230), -1)
        cv2.putText(check, str(n), (lp[0][0] + 8, lp[0][1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
        cv2.putText(check, str(n), (lp[0][0] + 8, lp[0][1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        sheets.append(render_hole(n, h["par"], deft["yd"][n - 1], round(L), lines[-1]))
    raw = "\n".join(lines)
    entry = {k: spec[k] for k in ["id", "name", "short", "pref", "tees", "defTee", "teeNote", "credit"]}
    for k in ["notes", "noteSrc", "elev"]:
        if k in spec:
            entry[k] = spec[k]
    json.dump({"entry": entry, "raw": raw}, open(os.path.join(wd, "course.json"), "w"), ensure_ascii=False, indent=1)
    s = min(1.0, 1400 / check.shape[1])
    cv2.imwrite(os.path.join(wd, "check.jpg"), cv2.resize(check, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cols = 6
    rows = [np.hstack(sheets[i:i + cols] + [np.zeros_like(sheets[0])] * (cols - len(sheets[i:i + cols]))) for i in range(0, len(sheets), cols)]
    cv2.imwrite(os.path.join(wd, "holes.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(lines)} holes -> course.json, check.jpg, holes.jpg")


COL = {"P": (70, 140, 70), "F": (110, 210, 120), "B": (170, 225, 245), "A": (215, 140, 60), "G": (150, 255, 170)}


def render_hole(n, par, yd, L, row):
    W, H = 220, 420
    img = np.full((H, W, 3), (35, 60, 35), np.uint8)
    fs = row.split(";")[5].split("|")
    ln = [list(map(float, p.split(","))) for p in row.split(";")[4].split()]
    allp = [list(map(float, q.split(","))) for f in fs for q in f[1:].split()] + ln
    xs, ys = [p[0] for p in allp], [p[1] for p in allp]
    s = min((W - 20) / (max(xs) - min(xs) + 1), (H - 50) / (max(ys) - min(ys) + 1))
    cx = (max(xs) + min(xs)) / 2
    tp = lambda p: (int(W / 2 + (p[0] - cx) * s), int(H - 15 - (p[1] - min(ys)) * s))
    for t in "PFABG":
        for f in fs:
            if f[0] == t:
                cv2.fillPoly(img, [np.array([tp(list(map(float, q.split(",")))) for q in f[1:].split()], np.int32)], COL[t])
    cv2.polylines(img, [np.array([tp(p) for p in ln], np.int32)], False, (255, 255, 255), 1)
    cv2.putText(img, f"{n}  Par{par}  {yd}yd (line {L})", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


if __name__ == "__main__":
    build(sys.argv[1])
