#!/usr/bin/env python3
"""build_course.py で作った course.json を MY GOLF の HTML に組み込む（同じidがあれば置き換え）。

使い方:  python3 insert_course.py WORKDIR/course.json index.html [claude-version/my-golf-claude.html ...]
"""
import json, re, sys


def insert(html, c):
    e, raw = c["entry"], c["raw"]
    cid = e["id"]
    var = re.sub(r"\W", "_", cid).upper() + "_RAW"
    assert "`" not in raw
    rawjs = f"/*course-raw:{cid}*/const {var}=`\n{raw}\n`;/*end-course-raw:{cid}*/\n"
    fields = ",".join(f"{k}:{json.dumps(v, ensure_ascii=False)}" for k, v in e.items() if k != "id")
    entry = f"/*course:{cid}*/{json.dumps(cid)}:{{id:{json.dumps(cid)},raw:{var},{fields}}}/*end-course:{cid}*/"
    html = re.sub(r"/\*course-raw:%s\*/.*?/\*end-course-raw:%s\*/\n" % (re.escape(cid), re.escape(cid)), "", html, flags=re.S)
    html = re.sub(r",\n  /\*course:%s\*/.*?/\*end-course:%s\*/" % (re.escape(cid), re.escape(cid)), "", html, flags=re.S)
    i = html.index("const COURSES={")
    html = html[:i] + rawjs + html[i:]
    end = "\n};\nlet COURSE_ID="
    assert html.count(end) == 1
    return html.replace(end, ",\n  " + entry + end)


if __name__ == "__main__":
    c = json.load(open(sys.argv[1]))
    for f in sys.argv[2:]:
        s = open(f, encoding="utf-8").read()
        open(f, "w", encoding="utf-8").write(insert(s, c))
        print("updated", f)
