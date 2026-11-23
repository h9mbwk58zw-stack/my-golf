#!/usr/bin/env python3
"""航空写真を色と模様で塗り分ける（芝・フェアウェイ・池・バンカー・林）。

使い方:  python3 segment.py WORKDIR [--hi WORKDIR18]
  --hi に fetch_aerial.py --zoom 18 で取った同じ範囲のフォルダを渡すと、
  バンカーを2倍細かい写真の砂の色（明るく、緑っぽさがない）と、まわりの芝との明るさの差で見分ける
出力:
  WORKDIR/masks.npz      各地物のマスク（turf, fw, water, bunker, area）
  WORKDIR/segment.jpg    左：写真 / 右：塗り分け結果（確認用）
しきい値は写真の季節や明るさで変わるので、結果を見て必要なら調整する。
"""
import sys, os
import numpy as np, cv2
from scipy import ndimage as ndi


HERE = os.path.dirname(os.path.abspath(__file__))


def match_color(im, ref_name="ref_color.jpg"):
    """写真の色の分布を、基準の写真（PGM石岡のズーム17/18）に合わせる。
    もやがかかった写真や季節の違う写真でも、同じしきい値で判別できるようにするため"""
    from skimage.exposure import match_histograms
    ref = cv2.imread(os.path.join(HERE, ref_name))
    if ref is None:
        return im
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    rlab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    out = match_histograms(lab, rlab, channel_axis=-1)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def is_hazy(im):
    hsv = cv2.cvtColor(cv2.resize(im, (400, 400)), cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[..., 1])) < 60 or float(np.percentile(hsv[..., 2], 2)) > 45


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


def fw_score_hi(im18):
    """ズーム18でのフェアウェイらしさ。緑が濃いほど・模様が少ないほど高い（芝の中で比べて使う）"""
    lab = cv2.cvtColor(im18, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    m = cv2.blur(L, (7, 7))
    tex = cv2.GaussianBlur(np.sqrt(np.maximum(cv2.blur(L * L, (7, 7)) - m * m, 0)), (0, 0), 4)
    Ab, Bb = cv2.GaussianBlur(A, (0, 0), 4), cv2.GaussianBlur(B, (0, 0), 4)
    return -(Ab - 116) - 0.5 * (Bb - 146) - 0.35 * np.minimum(tex, 12)


def sand_hi(im18, mpp18):
    """ズーム18の写真から砂（バンカー）を見分ける。明るく、緑っぽさがなく、まわりより目立って明るいもの"""
    lab = cv2.cvtColor(im18, cv2.COLOR_BGR2LAB).astype(np.float32)
    Ls, As, Bs = [cv2.GaussianBlur(lab[..., i], (0, 0), 1.2) for i in range(3)]
    sand = (Ls > 162) & (As > 119.5) & (Bs > 128) & (Bs < 160)
    sand = ndi.binary_fill_holes(ndi.binary_closing(ndi.binary_opening(sand, iterations=1), iterations=2))
    lab_, n = ndi.label(sand)
    if not n:
        return sand
    sz = ndi.sum(sand, lab_, range(1, n + 1)) * mpp18 * mpp18
    r = max(2, int(2.5 / mpp18))
    keep = []
    for i, sl in enumerate(ndi.find_objects(lab_), 1):
        if not (6 < sz[i - 1] < 1500):  # 1500m²より大きい砂地は造成地・資材置き場
            continue
        y0, y1 = max(0, sl[0].start - r - 2), sl[0].stop + r + 2
        x0, x1 = max(0, sl[1].start - r - 2), sl[1].stop + r + 2
        mk = lab_[y0:y1, x0:x1] == i
        ring = ndi.binary_dilation(mk, iterations=r) & ~mk
        if not ring.any():
            continue
        dL = np.median(Ls[y0:y1, x0:x1][mk]) - np.median(Ls[y0:y1, x0:x1][ring])
        if dL > 28 and np.median(As[y0:y1, x0:x1][ring]) < 125:   # 枯れ芝はまわりとの差が小さい
            keep.append(i)
    return np.isin(lab_, keep)


def forest_score(im):
    """森らしさ：細かいでこぼこ（木のこずえ）が多く、暗いほど高い"""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.8), cv2.CV_32F, ksize=3))
    V = cv2.blur(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32), (21, 21))
    return cv2.blur(lap, (21, 21)) / np.maximum(V / 120, 0.2)


def course_open(im, seed, mpp):
    """見つかった芝から、森ではない開けた場所を広げてコース全体をとらえる。
    コースの外周はたいてい濃い森なので、森との境目で止まる"""
    from skimage.filters import threshold_otsu
    sc = forest_score(im)
    roi = ndi.binary_dilation(seed, iterations=int(120 / mpp))
    if not roi.any():
        return np.zeros_like(seed), sc
    T = threshold_otsu(sc[roi])
    op = ndi.binary_opening(sc < T, iterations=3)
    lab, n = ndi.label(op)
    ids = np.unique(lab[seed & op]); ids = ids[ids > 0]
    if not len(ids):
        return np.zeros_like(seed), sc
    cnt = ndi.sum(seed & op, lab, ids); tot = ndi.sum(op, lab, ids)
    course = np.isin(lab, ids[(cnt > 200 / mpp ** 2) & (cnt / tot > 0.05)])
    course = ndi.binary_fill_holes(ndi.binary_closing(course, iterations=4))
    # 一番大きなまとまりから離れたもの（畑・隣のゴルフ場）は外す
    grp, k = ndi.label(ndi.binary_dilation(course, iterations=int(30 / mpp)))
    if k > 1:
        sz = ndi.sum(course, grp, range(1, k + 1))
        course &= grp == 1 + int(np.argmax(sz))
    return course, sc


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
    import json
    wd = sys.argv[1]
    im = cv2.imread(os.path.join(wd, "aerial.jpg"))
    hazy = is_hazy(im) or "--match" in sys.argv
    if hazy:
        print("写真の色を基準に合わせます（もや・色かぶりの補正）")
        im = match_color(im)
        cv2.imwrite(os.path.join(wd, "aerial_matched.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    M = segment(im)
    meta = json.load(open(os.path.join(wd, "meta.json")))
    op, _ = course_open(im, M["turf"], meta["mpp"])
    if op.any():
        # 開けた場所（ラフも含む芝）をコースの芝として使い、範囲も広げる
        M["turf"] = (M["turf"] | op) & ~M["water"]
        lab4, n4 = ndi.label(op | (M["area"] & M["turf"]))
        area = np.zeros(op.shape, np.uint8)
        for i in range(1, n4 + 1):
            part = lab4 == i
            if part.sum() * meta["mpp"] ** 2 < 2000:
                continue
            hull = cv2.convexHull(np.argwhere(part)[:, ::-1].astype(np.int32))
            cv2.fillPoly(area, [hull], 1)
        M["area"] = area.astype(bool) | (M["area"] & ndi.binary_dilation(op, iterations=60))
        M["turf"] &= M["area"]
        L_ = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        fw = M["turf"] & (L_ > np.percentile(L_[M["turf"]], 45))
        M["fw"] = keep_big(ndi.binary_closing(ndi.binary_opening(fw, iterations=3), iterations=5), 1500)
    if "--hi" in sys.argv:
        hd = sys.argv[sys.argv.index("--hi") + 1]
        im18 = cv2.imread(os.path.join(hd, "aerial.jpg"))
        if hazy:
            im18 = match_color(im18, "ref_color18.jpg")
        m17, m18 = json.load(open(os.path.join(wd, "meta.json"))), json.load(open(os.path.join(hd, "meta.json")))
        s18 = sand_hi(im18, m18["mpp"]).astype(np.float32)
        # ズーム18の位置をズーム17の画像に合わせる（ズーム18のピクセル = ズーム17の2倍）
        ox, oy = m17["px0"] * 2 - m18["px0"], m17["py0"] * 2 - m18["py0"]
        canvas = np.zeros((im.shape[0] * 2, im.shape[1] * 2), np.float32)
        ys, xs = slice(max(0, -oy), min(canvas.shape[0], s18.shape[0] - oy)), slice(max(0, -ox), min(canvas.shape[1], s18.shape[1] - ox))
        canvas[ys, xs] = s18[ys.start + oy:ys.stop + oy, xs.start + ox:xs.stop + ox]
        b17 = cv2.resize(canvas, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_AREA) > 0.4
        # フェアウェイらしさ（ズーム18）：芝の中で、緑が濃く（a*が低い・b*が低い）、刈り込みで模様が少ないところ
        sc18 = fw_score_hi(im18)
        cs = np.zeros(canvas.shape, np.float32)
        cs[ys, xs] = sc18[ys.start + oy:ys.stop + oy, xs.start + ox:xs.stop + ox]
        sc = cv2.resize(cs, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_AREA)
        t_ = M["turf"]
        sc = (sc - np.median(sc[t_])) / (np.std(sc[t_]) + 1e-6)
        M["fwscore"] = sc.astype(np.float16)
        fwg = t_ & (sc > 0.35)
        fwg = ndi.binary_closing(ndi.binary_opening(fwg, iterations=2), iterations=3)
        M["fw"] = keep_big(ndi.binary_fill_holes(fwg), 600)
        # コースの範囲の端にあるグリーンまわりのバンカーも拾う（範囲を少し広げて判定し、範囲もバンカーまで広げる）
        M["bunker"] = b17 & ndi.binary_dilation(M["area"] | M["turf"], iterations=15) & ~M["water"]
        lab3, n3 = ndi.label(M["area"] | ndi.binary_dilation(M["bunker"], iterations=6))
        area = np.zeros(M["area"].shape, np.uint8)
        for i in range(1, n3 + 1):
            hull = cv2.convexHull(np.argwhere(lab3 == i)[:, ::-1].astype(np.int32))
            cv2.fillPoly(area, [hull], 1)
        M["area"] = area.astype(bool)
        M["turf"] &= ~M["bunker"]
        M["fw"] &= ~M["bunker"]
    np.savez_compressed(os.path.join(wd, "masks.npz"), **M)
    cv2.imwrite(os.path.join(wd, "segment.jpg"), preview(im, M), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print({k: round(float(v.mean()), 4) for k, v in M.items()})
