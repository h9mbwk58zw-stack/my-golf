#!/usr/bin/env python3
"""国土地理院の航空写真（シームレス空中写真）を取得してつなぎ合わせる。

使い方:
  python3 fetch_aerial.py WORKDIR --lat 36.2097 --lon 140.3716 [--radius 900] [--zoom 17]

出力:
  WORKDIR/aerial.jpg   つなぎ合わせた航空写真（北が上）
  WORKDIR/meta.json    画像の位置情報（ズーム、左上のピクセル座標、1ピクセルのメートル数）
出典: 国土地理院（https://maps.gsi.go.jp/development/ichiran.html）
"""
import argparse, json, math, os, urllib.request
from PIL import Image

URL = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"


def lonlat_to_px(lat, lon, z):
    n = 256 * 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def px_to_lonlat(px, py, z):
    n = 256 * 2 ** z
    lon = px / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * py / n))))
    return lat, lon


def meters_per_px(lat, z):
    return 156543.03392 * math.cos(math.radians(lat)) / 2 ** z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--radius", type=float, default=900, help="中心からの範囲（m）")
    ap.add_argument("--zoom", type=int, default=17)
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    cache = os.path.join(a.workdir, "tiles")
    os.makedirs(cache, exist_ok=True)
    mpp = meters_per_px(a.lat, a.zoom)
    cx, cy = lonlat_to_px(a.lat, a.lon, a.zoom)
    r = a.radius / mpp
    x0, y0, x1, y1 = int(cx - r), int(cy - r), int(cx + r), int(cy + r)
    tx0, ty0, tx1, ty1 = x0 // 256, y0 // 256, x1 // 256, y1 // 256
    img = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            f = os.path.join(cache, f"{a.zoom}_{tx}_{ty}.jpg")
            if not os.path.exists(f):
                req = urllib.request.Request(URL.format(z=a.zoom, x=tx, y=ty), headers={"User-Agent": "MYGOLF-course-builder"})
                with urllib.request.urlopen(req, timeout=30) as r_, open(f, "wb") as o:
                    o.write(r_.read())
            img.paste(Image.open(f), ((tx - tx0) * 256, (ty - ty0) * 256))
    img = img.crop((x0 - tx0 * 256, y0 - ty0 * 256, x1 - tx0 * 256, y1 - ty0 * 256))
    img.save(os.path.join(a.workdir, "aerial.jpg"), quality=92)
    meta = {"zoom": a.zoom, "px0": x0, "py0": y0, "w": img.width, "h": img.height,
            "lat": a.lat, "lon": a.lon, "mpp": mpp, "source": "国土地理院 シームレス空中写真"}
    json.dump(meta, open(os.path.join(a.workdir, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print(f"aerial.jpg {img.width}x{img.height}px, 1px={mpp:.2f}m")


if __name__ == "__main__":
    main()
