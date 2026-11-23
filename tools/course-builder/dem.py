#!/usr/bin/env python3
"""国土地理院の標高タイル（DEM5A＝5mメッシュ、なければDEM5B・DEM10B）を取得して、
aerial.jpg（ズーム17）のピクセル位置の標高（m）を返す。

使い方（単体）:  python3 dem.py WORKDIR   → WORKDIR/dem.npy（aerial.jpg と同じ大きさの標高、m）
出典: 国土地理院 標高タイル（https://maps.gsi.go.jp/development/ichiran.html#dem）
"""
import json, os, sys, urllib.request
import numpy as np, cv2
from scipy import ndimage as ndi

URL = "https://cyberjapandata.gsi.go.jp/xyz/{name}/{z}/{x}/{y}.txt"
ZD = 15  # 標高タイルのズーム（DEM5Aは15まで）


def _tile(cache, name, x, y):
    f = os.path.join(cache, f"{name}_{ZD}_{x}_{y}.txt")
    if not os.path.exists(f):
        try:
            req = urllib.request.Request(URL.format(name=name, z=ZD, x=x, y=y), headers={"User-Agent": "MYGOLF-course-builder"})
            with urllib.request.urlopen(req, timeout=30) as r, open(f, "wb") as o:
                o.write(r.read())
        except Exception:
            open(f, "w").close()
    t = open(f).read().strip()
    if not t:
        return None
    a = np.array([[np.nan if v == "e" else float(v) for v in row.split(",")] for row in t.split("\n")])
    return a if a.shape == (256, 256) else None


def dem_for(wd):
    meta = json.load(open(os.path.join(wd, "meta.json")))
    s = 2 ** (meta["zoom"] - ZD)
    x0, y0 = meta["px0"] / s, meta["py0"] / s
    x1, y1 = (meta["px0"] + meta["w"]) / s, (meta["py0"] + meta["h"]) / s
    tx0, ty0, tx1, ty1 = int(x0 // 256), int(y0 // 256), int(x1 // 256), int(y1 // 256)
    cache = os.path.join(wd, "tiles")
    os.makedirs(cache, exist_ok=True)
    big = np.full(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.nan)
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            for name in ("dem5a", "dem5b", "dem10b"):
                a = _tile(cache, name, tx, ty)
                if a is not None:
                    sl = big[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256]
                    m = np.isnan(sl)
                    sl[m] = a[m]          # 細かいデータで埋まらない所だけ次のデータで埋める
                    if not np.isnan(sl).any():
                        break
    # 池などの欠けた所は、まわりの値で埋める
    if np.isnan(big).any():
        idx = ndi.distance_transform_edt(np.isnan(big), return_distances=False, return_indices=True)
        big = big[tuple(idx)]
    # ズーム17のピクセルに合わせて切り出し・拡大（なめらかに補間）
    ox, oy = x0 - tx0 * 256, y0 - ty0 * 256
    M = np.float32([[1 / s, 0, ox], [0, 1 / s, oy]])
    out = cv2.warpAffine(big.astype(np.float32), M, (meta["w"], meta["h"]),
                         flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return cv2.GaussianBlur(out, (0, 0), 2.0)


if __name__ == "__main__":
    wd = sys.argv[1]
    d = dem_for(wd)
    np.save(os.path.join(wd, "dem.npy"), d)
    print(f"dem.npy {d.shape}  標高 {np.nanmin(d):.1f}〜{np.nanmax(d):.1f} m")
