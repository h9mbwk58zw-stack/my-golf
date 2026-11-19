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
 "fitTee": "black",   ← 省略可。このティーの距離に合わせて、線の最初の点（ティー）を前後にずらす
 "holes": [{"n":1, "par":4, "line":[[ティーx,y],[曲がり角x,y],[グリーンx,y]]}, ...]
}
fitTee には一番後ろのティーを指定する（アプリは前のティーを、距離の差だけ線に沿って前に置く）。
ずらす量が大きいホール（MAX_FIT_YD 以上）は、番号違いの疑いがあるので調整せず警告を出す。
"line" はティーから順に、ドッグレッグの曲がり角（あれば）、最後にグリーンの中心。
"""
import json, math, os, sys
import numpy as np, cv2
from scipy import ndimage as ndi

MAX_FIT_YD = 100   # これ以上ずらす必要があるホールは調整しない（番号違いの疑い）
CW_YD = 45        # アプリと同じ：センターラインから左右45ydがそのホールの範囲
YD = 0.9144
GREEN_R_YD = 13   # グリーンの半径の目安
TYPES = [("P", None, 3.0, 400), ("W", None, 3.5, 250), ("A", "water", 3.0, 300), ("F", "fw", 2.5, 250), ("B", "bunker", 1.2, 15)]
AREA_HALF_YD = 45   # 楕円の短い方の半径（ホールの左右の幅）
AREA_PAD_YD = 22    # 楕円をティーの後ろとグリーンの奥に伸ばす長さ
CTX_YD = 70         # 楕円の外にどこまで周りの景色（隣のホール・林・池）を描くか


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


def fit_tee(lp, target_yd, mpp):
    """線の長さが target_yd になるように、最初の点を最初の区間の向きに沿って前後に動かす"""
    k = YD / mpp  # 1ydあたりのピクセル
    seg = [math.dist(lp[i], lp[i + 1]) for i in range(len(lp) - 1)]
    cur = sum(seg) / k
    d = target_yd - cur                        # +なら後ろへ延ばす、-なら前へ縮める
    if abs(d) > MAX_FIT_YD:
        return lp, cur, None
    a, b = np.array(lp[0], float), np.array(lp[1], float)
    u = (a - b) / (np.linalg.norm(a - b) or 1)
    if d < 0 and -d * k >= seg[0] - 10 * k:   # 最初の区間より縮めるときは曲がり角を残してその手前まで
        return lp, cur, None
    t = a + u * d * k
    return [[int(round(t[0])), int(round(t[1]))]] + lp[1:], cur, d


def ellipse_area(lp, shape, mpp):
    """ホールの線に沿った楕円（曲がるホールは区間ごとの楕円をなめらかにつないだ形）"""
    k = YD / mpp
    m = np.zeros(shape, np.uint8)
    n = len(lp) - 1
    for i in range(n):
        a, b = np.array(lp[i], float), np.array(lp[i + 1], float)
        L = np.linalg.norm(b - a)
        pa = AREA_PAD_YD * k if i == 0 else 0
        pb = AREA_PAD_YD * k if i == n - 1 else 0
        u = (b - a) / (L or 1)
        a2, b2 = a - u * pa, b + u * pb
        c = (a2 + b2) / 2
        ang = math.degrees(math.atan2(u[1], u[0]))
        semi = (np.linalg.norm(b2 - a2) / 2) + (AREA_HALF_YD * k * 0.35 if n > 1 else 0)
        cv2.ellipse(m, (int(c[0]), int(c[1])), (int(semi), int(AREA_HALF_YD * k)), ang, 0, 360, 1, -1)
    r = int(12 * k) | 1
    m = cv2.GaussianBlur(m.astype(np.float32), (r, r), 0) > 0.5
    return m


def build(wd):
    meta = json.load(open(os.path.join(wd, "meta.json")))
    spec = json.load(open(os.path.join(wd, "holes.json")))
    M = dict(np.load(os.path.join(wd, "masks.npz")))
    mpp = meta["mpp"]
    im = cv2.imread(os.path.join(wd, "aerial.jpg"))
    check = im.copy()
    tees = spec["tees"]
    deft = next(t for t in tees if t["id"] == spec["defTee"])
    lines, sheets, report = [], [], []
    corr_px = int(CW_YD * YD / mpp)
    for h in sorted(spec["holes"], key=lambda h: h["n"]):
        n = h["n"]
        lp = [list(map(int, p)) for p in h["line"]]
        boxes = h.get("teeBoxes")
        if boxes:
            lp[0] = list(map(int, boxes[0]))   # 写真で確かめた一番後ろのティー台から線を始める
        else:
            lp[0] = snap(lp[0], M["turf"], 30)
        lp[-1] = snap(lp[-1], M["turf"], 30)
        if spec.get("fitTee") and not boxes:
            ft = next(t for t in tees if t["id"] == spec["fitTee"])
            lp, before, d = fit_tee(lp, ft["yd"][n - 1], mpp)
            # ティー台は小さくて芝として取れないことが多いので、芝から15m以内なら良しとする
            r_ = int(15 / mpp); x_, y_ = lp[0]
            on = bool(M["turf"][max(0, y_ - r_):y_ + r_ + 1, max(0, x_ - r_):x_ + r_ + 1].any())
            report.append({"n": n, "target": ft["yd"][n - 1], "before": round(before), "moved": None if d is None else round(d),
                           "tee_on_turf": on})
        tf = frame(lp, mpp)
        # ティー台：写真で確かめた位置、なければ各ティーの距離の位置を一番近い芝に寄せる
        k_ = YD / mpp
        if not boxes:
            Lt = sum(math.dist(lp[i], lp[i + 1]) for i in range(len(lp) - 1)) / k_
            back = max(t["yd"][n - 1] for t in tees)
            boxes = []
            for t in sorted(tees, key=lambda t: -t["yd"][n - 1]):
                d = (Lt - t["yd"][n - 1]) if Lt >= back else (back - t["yd"][n - 1]) * Lt / back
                rest, q = max(0.0, d) * k_, None
                for i in range(len(lp) - 1):
                    a_, b_ = np.array(lp[i], float), np.array(lp[i + 1], float); sl = np.linalg.norm(b_ - a_)
                    if rest <= sl:
                        q = a_ + (b_ - a_) * (rest / sl); break
                    rest -= sl
                q = lp[-1] if q is None else q
                q = snap([int(q[0]), int(q[1])], M["turf"], int(15 / mpp))
                if all(math.dist(q, b) > 9 * k_ for b in boxes):
                    boxes.append(q)
        tee_mask = np.zeros(M["turf"].shape, np.uint8)
        tee_polys = []
        nxt = np.array(lp[1], float)
        for b in boxes:
            c = np.array(b, float)
            u = nxt - c; u = u / (np.linalg.norm(u) or 1)
            ang = math.degrees(math.atan2(u[1], u[0]))
            cv2.ellipse(tee_mask, (int(c[0]), int(c[1])), (int(7 * k_), int(5 * k_)), ang, 0, 360, 1, -1)
            # 角丸の四角（長さ14yd×幅10yd）をアプリの座標で
            cx_, cy_ = tf(c)
            fu = np.array(tf(c + u)) - np.array([cx_, cy_]); fu = fu / (np.linalg.norm(fu) or 1); fr = np.array([fu[1], -fu[0]])
            pts_ = []
            for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                ca, sa = math.cos(a), math.sin(a)
                ex = math.copysign(abs(ca) ** 0.5, ca) * 7; ey = math.copysign(abs(sa) ** 0.5, sa) * 5
                pt = np.array([cx_, cy_]) + fu * ex + fr * ey
                pts_.append(f"{round(pt[0])},{round(pt[1])}")
            tee_polys.append("T" + " ".join(pts_))
        tee_mask = tee_mask.astype(bool)
        corr = np.zeros(M["turf"].shape, np.uint8)
        cv2.polylines(corr, [np.array(lp, np.int32)], False, 1, thickness=2 * corr_px)
        corr = corr.astype(bool)
        lmask = np.zeros_like(corr, np.uint8)
        cv2.polylines(lmask, [np.array(lp, np.int32)], False, 1, thickness=max(3, int(8 / mpp)))
        gcirc = np.zeros_like(lmask)
        cv2.circle(gcirc, tuple(lp[-1]), int(GREEN_R_YD * YD / mpp), 1, -1)
        if spec.get("areaShape", "ellipse") == "ellipse":
            # 楕円で切り取る：隣のホールが少し入っても、形のきれいさを優先
            E = ellipse_area(lp, M["turf"].shape, mpp)
            # 芝のまとまり（線にかかるもの）＋線にかかる池＋グリーンを少し広げて、なめらかな形にする
            base = (M["turf"] | gcirc.astype(bool)) & E
            lab_b, _ = ndi.label(base)
            hb = np.unique(lab_b[lmask.astype(bool) & base]); hb = hb[hb > 0]
            base = np.isin(lab_b, hb)
            lab_w, _ = ndi.label(M["water"] & E)
            line20 = np.zeros_like(lmask)
            cv2.polylines(line20, [np.array(lp, np.int32)], False, 1, thickness=int(2 * 20 * YD / mpp))
            hw = np.unique(lab_w[line20.astype(bool) & (lab_w > 0)]); hw = hw[hw > 0]
            base = base | np.isin(lab_w, hw) | line20.astype(bool) | ndi.binary_dilation(tee_mask, iterations=int(5 * YD / mpp))
            base = ndi.binary_dilation(base, iterations=int(7 * YD / mpp))
            sig = 9 / mpp
            P = (cv2.GaussianBlur(base.astype(np.float32), (0, 0), sig) > 0.42) & E
            P = ndi.binary_fill_holes(P)
            A = M["water"] & P
            B = M["bunker"] & P & ~A
            turf = (M["turf"] | gcirc.astype(bool)) & P
            F = M["fw"] & P & ~A
            on = ndi.binary_erosion(turf & ~A & ~B, iterations=int(8 * YD / mpp))
            if (F & corr).sum() < 0.15 * max(1, (turf & corr).sum()):
                F = F | (on & corr)
            # 林：楕円の中で芝・池・バンカー以外の場所
            # ティー台のまわりと、ティーから最初の区間の通り道は林にしない（明るいラフで見やすく）
            apron = np.zeros_like(lmask)
            for bx in boxes:
                cv2.line(apron, tuple(map(int, bx)), tuple(map(int, lp[1])), 1, thickness=int(2 * 9 * YD / mpp))
            apron = apron.astype(bool) & (ndi.distance_transform_edt(~tee_mask) * mpp / YD < 70)
            apron |= ndi.binary_dilation(tee_mask, iterations=int(8 * YD / mpp))
            Wd = P & ~ndi.binary_dilation(turf, iterations=int(2 / mpp)) & ~A & ~B & ~apron
            Wd = ndi.binary_opening(Wd, iterations=int(3 / mpp))
            masks = {"P": P, "W": Wd, "F": F, "B": B, "A": A}
            feats = []
            for t, _, eps, mn in TYPES:
                feats += polys(masks[t], eps, mn, mpp, tf, t)
            masks = None
            # 周りの景色（楕円の外）：アプリで薄く描く背景。プレーの計算には使わない
            big = cv2.dilate(P.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * CTX_YD * YD / mpp) | 1,) * 2)).astype(bool)
            out = big & ~ndi.binary_erosion(P, iterations=int(3 / mpp))
            cturf = M["turf"] & out
            cW = out & ~ndi.binary_dilation(M["turf"], iterations=int(2 / mpp)) & ~M["water"] & ~M["bunker"]
            cW = ndi.binary_opening(cW, iterations=int(3 / mpp))
            cmasks = {"W": cW, "F": M["fw"] & out & ~M["water"], "A": M["water"] & out, "B": M["bunker"] & out}
            ctx = []
            for t in "WFAB":
                ctx += polys(cmasks[t], 4.0 if t != "B" else 1.5, 400 if t != "B" else 20, mpp, tf, t)
            ctx_str = "|".join(ctx)
        else:
            ctx_str = ""
            masks = {}
        # プレー範囲 = 芝のうちホールの線に触れているまとまり＋グリーン
        p = (M["turf"] | gcirc.astype(bool)) & corr
        lab, k = ndi.label(p)
        hit = np.unique(lab[lmask.astype(bool) & p]); hit = hit[hit > 0]
        P = ndi.binary_closing(np.isin(lab, hit), iterations=3) & corr
        near = ndi.binary_dilation(P, iterations=int(6 / mpp)) & corr
        B = M["bunker"] & near
        A = M["water"] & ndi.binary_dilation(corr, iterations=int(10 / mpp))
        # 線から20yd以内にかかる池（池越え・池沿い）は、アプリで描けるようにプレー範囲に含める
        lab_a, na = ndi.label(A)
        line20 = np.zeros_like(lmask)
        cv2.polylines(line20, [np.array(lp, np.int32)], False, 1, thickness=int(2 * 20 * YD / mpp))
        hitA = np.unique(lab_a[line20.astype(bool) & A]); hitA = hitA[hitA > 0]
        P = P | (np.isin(lab_a, hitA) & corr) | B
        P = ndi.binary_closing(P, iterations=2) & corr
        F = M["fw"] & P & ~A
        # フェアウェイが自動で取れなかったホールは、プレー範囲の内側（端から8yd）をフェアウェイとみなす
        if F.sum() < 0.15 * max(1, (P & ~A & ~B).sum()):
            F = ndi.binary_erosion(P & ~A & ~B, iterations=int(8 * YD / mpp))
        if masks is not None:
            masks = {"P": P, "W": np.zeros_like(P), "F": F, "B": B, "A": A}
            feats = []
            for t, _, eps, mn in TYPES:
                feats += polys(masks[t], eps, mn, mpp, tf, t)
        feats += tee_polys
        # グリーン（円の目安）
        gx, gy = tf(lp[-1])
        rg = GREEN_R_YD
        feats.append("G" + " ".join(f"{round(gx + rg * math.cos(a))},{round(gy + rg * 0.85 * math.sin(a))}" for a in np.linspace(0, 2 * math.pi, 14, endpoint=False)))
        ln = [tf(q) for q in lp]
        L = sum(math.dist(ln[i], ln[i + 1]) for i in range(len(ln) - 1))
        lines.append(f"{n};{h['par']};{deft['yd'][n - 1]};{round(L)};" + " ".join(f"{round(x)},{round(y)}" for x, y in ln) + ";" + "|".join(feats) + (";" + ctx_str if ctx_str else ""))
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
    if report:
        json.dump(report, open(os.path.join(wd, "fit_report.json"), "w"), ensure_ascii=False, indent=1)
        for r in report:
            st = "調整せず（ずれが大きい：番号を確認）" if r["moved"] is None else f"{r['moved']:+d}yd ずらした" + ("" if r["tee_on_turf"] else "（ティーが芝の外：位置を確認）")
            print(f"{r['n']:>2}番  目標{r['target']}yd  元の線{r['before']}yd  {st}")
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
