#!/usr/bin/env python3
"""切り抜いたフェアウェイから「ホールの候補」を自動で見つけ、番号（記号）を振った画像を作る。
候補ごとに両端の座標を出すので、どちらがティーかと何番ホールかを決めて holes.json に写す。

使い方:  python3 candidates.py WORKDIR [--min-len 90]
出力:  WORKDIR/candidates.json, WORKDIR/candidates.jpg
"""
import argparse, json, os, string
import numpy as np, cv2
from scipy import ndimage as ndi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--min-len", type=float, default=90, help="候補にする最小の長さ（m）")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.workdir, "meta.json")))
    M = np.load(os.path.join(a.workdir, "masks.npz"))
    mpp = meta["mpp"]
    turf = ndi.binary_opening(M["turf"], iterations=int(4 / mpp))
    # 芝を細い林の切れ目で分けるため、少し削ってからまとまりを数える
    core = ndi.binary_erosion(turf, iterations=int(5 / mpp))
    lab, n = ndi.label(core)
    cands = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) * mpp * mpp < 1500:
            continue
        pts = np.c_[xs, ys].astype(float)
        c = pts.mean(0)
        w, v = np.linalg.eigh(np.cov((pts - c).T))
        ax = v[:, np.argmax(w)]
        proj = (pts - c) @ ax
        lo, hi = np.percentile(proj, 1), np.percentile(proj, 99)
        length = (hi - lo) * mpp
        if length < a.min_len:
            continue
        e1, e2 = c + ax * lo, c + ax * hi
        # 曲がりを見るため、軸に沿って3区間の中心も出す
        mid = []
        for f in (1 / 3, 2 / 3):
            q = lo + (hi - lo) * f
            sel = np.abs(proj - q) < (hi - lo) / 12
            if sel.any():
                mid.append([int(v_) for v_ in pts[sel].mean(0)])
        cands.append({"ends": [[int(e1[0]), int(e1[1])], [int(e2[0]), int(e2[1])]], "mid": mid,
                      "len_yd": round(length / 0.9144), "area_m2": round(len(xs) * mpp * mpp)})
    cands.sort(key=lambda c: (c["ends"][0][1] + c["ends"][1][1]))
    labels = (string.ascii_uppercase + string.ascii_lowercase)
    im = cv2.imread(os.path.join(a.workdir, "aerial.jpg"))
    for k, c in enumerate(cands):
        c["id"] = labels[k]
        line = [c["ends"][0]] + c["mid"] + [c["ends"][1]]
        cv2.polylines(im, [np.array(line, np.int32)], False, (0, 255, 255), 3)
        for e, col in zip(c["ends"], [(255, 255, 255), (255, 0, 255)]):
            cv2.circle(im, tuple(e), 8, col, -1)
        m = np.mean(line, 0).astype(int)
        txt = f"{c['id']} {c['len_yd']}y"
        cv2.putText(im, txt, tuple(m + [6, 0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
        cv2.putText(im, txt, tuple(m + [6, 0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    s = min(1.0, 1400 / im.shape[1])
    cv2.imwrite(os.path.join(a.workdir, "candidates.jpg"), cv2.resize(im, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(cands, open(os.path.join(a.workdir, "candidates.json"), "w"), indent=1)
    print(len(cands), "candidates;", "白丸=端1 / 桃丸=端2")


if __name__ == "__main__":
    main()
