"""Generate a synced .lrc next to the YuE2 output FLAC.

Timing is derived from the Vocal bars in outputs/<dir>/score.abc. The timeline
is calibrated either by two real vocal-onset anchors (recommended: measured in
the player or auto-detected by audio_align.py) or, without anchors, by evenly
stretching the score over the audio duration. Each lyric line lands at the
start of its evenly divided block of that section's bars.

Usage (CLI): python tools/make_lrc.py outputs/waltz-v2 [verse_sec chorus_sec]
"""
import json
import os
import re
import sys

UNIT_PER_BAR = 24  # L:1/32, one 3/4 or 6/8 bar = 24 units


def line_bars(line):
    """Bar unit totals (None = rest bar) for one voice line of ABC."""
    bars = []
    line = re.sub(r'"[^"]*"', "", line).strip()
    if not line or line.startswith("%"):
        return bars
    for tok in line.split("|"):
        tok = tok.strip()
        if not tok:
            continue
        units, zbars = 0, 0
        for m in re.finditer(r"(Z)(\d+)?|(z|[a-gA-G][',+]*)(-?)(\d+)?", tok):
            if m.group(1):
                zbars += int(m.group(2) or 1)
            elif m.group(3):
                units += int(m.group(5) or 1)
        if zbars:
            bars.extend([None] * zbars)
        if units:
            bars.append(units)
    return bars


def parse_sections(score):
    headers = [(m.group(1), m.start()) for m in re.finditer(r"^% (\w+)", score, re.M)]
    sections = {}
    for i, (name, pos) in enumerate(headers):
        end = headers[i + 1][1] if i + 1 < len(headers) else len(score)
        bars, take = [], False
        for line in score[pos:end].splitlines():
            s = line.strip()
            if s.startswith("V:"):
                take = s == "V: Vocal"
                continue
            if take:
                bars.extend(line_bars(s))
        sections[name] = bars
    return sections


def lyric_blocks(lyrics):
    """{section_lower: [lines]} from the request lyrics."""
    blocks, current = {}, None
    for ln in lyrics.splitlines():
        s = ln.strip()
        m = re.match(r"\[\s*(verse|chorus)\s*\]$", s, re.I)
        if m:
            current = m.group(1).lower()
        elif s and current:
            blocks.setdefault(current, []).append(s)
    return blocks


def fmt(sec):
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def build_lrc(out_dir, verse_sec=None, chorus_sec=None):
    """Writes audio.lrc and returns {path, lines, calibrated, diagnostics}."""
    request = json.load(open(os.path.join(out_dir, "request.json"), encoding="utf-8"))
    audio_sec = json.load(open(os.path.join(out_dir, "result.json"), encoding="utf-8"))["audio_seconds"]
    scores = parse_sections(open(os.path.join(out_dir, "score.abc"), encoding="utf-8").read())

    bars, offsets = [], {}
    for name in ("intro", "verse", "chorus", "outro"):
        if name in scores:
            offsets[name] = len(bars)
            bars.extend(scores[name])
    bad = sorted({b for b in bars if b is not None and b != UNIT_PER_BAR})

    if verse_sec is not None and chorus_sec is not None:
        a = (chorus_sec - verse_sec) / (offsets["chorus"] - offsets["verse"])
        b = verse_sec - a * offsets["verse"]
        mode = "anchors"
    else:
        a, b, mode = audio_sec / len(bars), 0.0, "total-length (approximate)"

    blocks = lyric_blocks(request["lyrics"])
    lrc = ["[by:YuE2]"]
    lines = []
    for key in ("verse", "chorus"):
        if key not in offsets or not blocks.get(key):
            continue
        nxt = min((v for k, v in offsets.items() if v > offsets[key]), default=len(bars))
        sect = bars[offsets[key]:nxt]
        step = len(sect) // len(blocks[key])
        for i, text in enumerate(blocks[key]):
            sec = b + (offsets[key] + i * step) * a
            line = f"[{fmt(sec)}]{text}"
            lrc.append(line)
            lines.append(line)
    path = os.path.join(out_dir, "audio.lrc")
    open(path, "w", encoding="utf-8").write("\n".join(lrc) + "\n")
    return {"path": path, "lines": lines, "mode": mode,
            "bar_sec": round(a, 4), "offset_sec": round(b, 3),
            "bad_bar_lengths": bad, "total_bars": len(bars)}


def main():
    out_dir = sys.argv[1]
    verse = float(sys.argv[2]) if len(sys.argv) > 2 else None
    chorus = float(sys.argv[3]) if len(sys.argv) > 3 else None
    info = build_lrc(out_dir, verse, chorus)
    print(f"mode: {info['mode']} | bar_sec={info['bar_sec']} offset={info['offset_sec']}s "
          f"| bad_bar_lengths={info['bad_bar_lengths'] or 'none'}")
    for line in info["lines"]:
        print(line)
    print(f"✅ {info['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())