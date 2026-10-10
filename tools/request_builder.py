"""Interactive new-song request helpers, driven by tools/new_song.bat.

Subcommands:
  init <path>      write an editable UTF-8 JSON template
  validate <path>  strict-check against yue2.protocol.SongRequest
  show <path>      print a human-readable summary
  outdir <path>    print outputs/<id>_<timestamp> (safe for `for /f`)
"""
import datetime
import json
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TEMPLATE = {
    "id": "untitled_song",
    "style": "Mandarin, <语言/曲风/主要乐器/情绪/BPM; 人声特征也写这里, 例如: Mandarin, 温暖钢琴流行, 女声清亮带气声, 88 BPM>",
    "lyrics": ("[Verse]\n"
               "<主歌歌词, 每行一句; 想写第二段主歌就再写一个 [Verse] 段>\n"
               "<下一行>\n"
               "\n"
               "[Pre-Chorus]\n"
               "<预副歌(可选, 不要这个段落就把整段连同段标一起删掉)>\n"
               "\n"
               "[Chorus]\n"
               "<副歌; 想重复唱就再写一个 [Chorus] 段, 模型会安排重复>\n"
               "\n"
               "[Bridge]\n"
               "<桥段(可选)>\n"
               "\n"
               "[Outro]\n"
               "<尾声(可选): 写一两句渐弱歌词=人声收尾; 不写=模型器乐淡出>"
               ", 常见顺序: Verse -> Pre-Chorus -> Chorus -> Verse -> "
               "Pre-Chorus -> Chorus -> Bridge -> Outro"),
    "cot": "full",
    "abc": None,
    "cfg_scale": None
}

def load(path):
    with open(path, encoding="utf-8-sig") as f:   # 记事本可能写 BOM
        data = json.load(f)
    if "seed" not in data:
        data["seed"] = random.randint(0, 2**31)
    return data

def cmd_init(path):
    data = dict(TEMPLATE)
    data["seed"] = random.randint(1, 999999)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"模板已创建: {path}")
    return 0

def cmd_validate(path):
    errors, warns = [], []
    try:
        data = load(path)
    except Exception as e:
        print(f"❌ JSON 解析失败: {e}")
        print("   (检查是否漏了逗号/引号; 建议整段复制原模板修改)")
        return 1
    try:
        from yue2.protocol import SongRequest
        req = SongRequest(style=data["style"], lyrics=data["lyrics"], **{k: v for k, v in data.items() if k not in ("style", "lyrics")})
        print(f"✔ 字段合法 (id={req.id} cot={req.cot} seed={req.seed})")
    except Exception as e:
        print(f"❌ 字段校验失败: {e}")
        return 1
    if "<" in data["style"] or "<" in data["lyrics"]:
        warns.append("style/lyrics 看起来还是模板占位内容, 记得替换")
    for w in warns:
        print(f"⚠️ {w}")
    return 0

def cmd_show(path):
    data = load(path)
    print("=" * 46)
    for k in ("id", "style", "cot", "seed"):
        print(f"{k:6}: {data.get(k)}")
    print("lyrics:")
    for ln in data["lyrics"].splitlines():
        print("       ", ln)
    print("=" * 46)
    return 0

def cmd_outdir(path, name=None):
    data = load(path)
    base = data["id"]
    if name:
        slug = "".join(c for c in name.strip() if c not in '\\/:*?"<>|')
        slug = slug.strip()
        if slug:
            base = slug
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"outputs/{base}_{ts}")
    return 0

if __name__ == "__main__":
    cmd = sys.argv[1]
    sys.exit({"init": cmd_init, "validate": cmd_validate, "show": cmd_show,
              "outdir": lambda a: cmd_outdir(a, sys.argv[3] if len(sys.argv) > 3 else None)}[cmd](sys.argv[2]))