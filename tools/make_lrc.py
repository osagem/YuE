"""Generate a synced .lrc next to the YuE2 output FLAC.

Timing is derived from the Vocal bars in outputs/<dir>/score.abc, calibrated by
two real vocal-onset anchors (verse and first chorus) or, without anchors, by
evenly stretching the score over the audio duration. Section headers repeat in
modern checkpoints (verse/pre-chorus/chorus/interlude/...), so the timeline is
built from the header ORDER and each lyric block is matched to a section
occurrence by name; sections without lyrics (interlude/outro) are skipped and
unmatched lyric blocks are interpolated between their neighbours.

Usage (CLI): python tools/make_lrc.py outputs/<dir> [verse_sec chorus_sec]
"""
import bisect
import json
import os
import re
import sys

UNIT_PER_BAR = 24  # L:1/32, one 3/4, 6/8 or 4/4 bar = 24 units


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


def norm(name):
    return re.sub(r"[^a-z\-]", "", name.lower())


def parse_sections(score):
    """Ordered [(name, [bar units]) ...] for every % header occurrence."""
    headers = [(m.group(1), m.start()) for m in re.finditer(r"^% ([\w-]+)", score, re.M)]
    out = []
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
        out.append((norm(name), bars))
    return out


def lyric_blocks_ordered(lyrics):
    """Ordered [(tag_lower, [lines]), ...] for every [tag] header, repeats kept."""
    out, current = [], None
    for ln in lyrics.splitlines():
        s = ln.strip()
        m = re.match(r"\[\s*([A-Za-z][A-Za-z0-9 \-]*)\]\s*$", s)
        if m:
            current = m.group(1).strip().lower()
            out.append([current, []])
        elif s and current is not None:
            out[-1][1].append(s)
    return [(norm(tag), lines) for tag, lines in out if lines]


def lyric_blocks(lyrics):
    """{section_lower: [lines]} (first occurrence per tag)."""
    blocks = {}
    for tag, lines in lyric_blocks_ordered(lyrics):
        blocks.setdefault(tag, lines)
    return blocks


def fmt(sec):
    total = round(sec, 2)   # 先舍入再进位, 避免 59.996 → "60.00" 的非法时间戳
    return f"{int(total // 60):02d}:{total % 60:05.2f}"


def build_lrc(out_dir, verse_sec=None, chorus_sec=None, points=None):
    """Writes audio.lrc; points=[(lyric_text, measured_sec), ...] adds
    piecewise calibration, since the renderer is expressive: one linear
    bar->seconds map cannot fit every section when the tempo drifts."""
    """Writes audio.lrc and returns {path, lines, mode, diagnostics}."""
    request = json.load(open(os.path.join(out_dir, "request.json"), encoding="utf-8"))
    audio_sec = json.load(open(os.path.join(out_dir, "result.json"), encoding="utf-8"))["audio_seconds"]
    heads, bars, spans = [], [], {}
    for name, sbars in parse_sections(open(os.path.join(out_dir, "score.abc"), encoding="utf-8").read()):
        spans.setdefault(name, []).append((len(bars), len(bars) + len(sbars)))
        heads.append((name, len(bars), len(sbars)))
        bars.extend(sbars)
    offsets = {}
    for name, bs in spans.items():
        if name in ("verse", "chorus"):
            offsets[name] = bs[0][0]
    bad = sorted({b for b in bars if b is not None and b != UNIT_PER_BAR})

    if verse_sec is not None and chorus_sec is not None:
        a = (chorus_sec - verse_sec) / (offsets["chorus"] - offsets["verse"])
        b = verse_sec - a * offsets["verse"]
        mode = "anchors"
    else:
        a, b, mode = audio_sec / len(bars), 0.0, "total-length (approximate)"

    ordered = lyric_blocks_ordered(request["lyrics"])
    tags = [t for t, _ in ordered]
    lrc, lines, mode_extra = ["[by:YuE2]"], [], ""

    if tags == ["verse", "chorus"]:
        # Exact, ear-validated path: exactly one [Verse] then one [Chorus].
        blocks = lyric_blocks(request["lyrics"])
        for key in ("verse", "chorus"):
            if key not in offsets or not blocks.get(key):
                continue
            start, end = spans[key][0]
            step = (end - start) // len(blocks[key])
            for i, text in enumerate(blocks[key]):
                sec = b + (start + i * step) * a
                line = f"[{fmt(sec)}]{text}"
                lrc.append(line)
                lines.append(line)
    else:
        # Multi-section path: match each lyric block to a section occurrence
        # by name; unmatched lyric blocks fill the bar gap between their
        # matched neighbours; sections without lyrics are skipped.
        used = set()
        assigns = []  # (block_index, kind, lo, hi)
        for bi, (tag, _ls) in enumerate(ordered):
            match = None
            for k, (name, start, nbars) in enumerate(heads):
                if name == tag and (tag, k) not in used:
                    match = (k, start, start + nbars)
                    break
            if match is not None:
                used.add((tag, match[0]))
            assigns.append([bi, "matched" if match else "interp",
                            match[1] if match else None, match[2] if match else None])
        # Unmatched lyric blocks: fill the bar gap between their matched
        # neighbours. The gap can degenerate to zero when the model sang the
        # block inside a neighbouring section (e.g. pre-chorus folded into a
        # long % verse) — in that case borrow up to half of the previous
        # span so the lines still spread musically instead of collapsing.
        for row in assigns:
            if row[1] != "interp":
                continue
            lo_ = next((r[3] for r in reversed(assigns[:row[0]])
                        if r[1] == "matched" and r[3] is not None),
                       offsets.get("verse", 0))
            hi_ = next((r[2] for r in assigns[row[0] + 1:]
                        if r[1] == "matched" and r[2] is not None), len(bars))
            n_lines = len(ordered[row[0]][1])
            if hi_ - lo_ < n_lines and lo_ - hi_ < n_lines:
                prev = next((r for r in reversed(assigns[:row[0]])
                             if r[1] == "matched" and r[3] is not None), None)
                if prev is not None and (prev[3] - prev[2]) >= 2:
                    borrow = min((prev[3] - prev[2]) // 2, max(1, n_lines))
                    prev[3] -= borrow
                    lo_ = prev[3]
                    row[2], row[3] = lo_, lo_ + max(1, borrow)
            else:
                row[2], row[3] = lo_, hi_
        # Piecewise calibration: measured lyric-line timings become exact
        # (bar, seconds) knots; segments between knots interpolate and the
        # outer segments extrapolate with their own slope. The verse/chorus
        # anchors serve as base knots unless a measurement replaces them.
        tmap = None
        if points:
            def bar_of(text):
                t = text.strip()
                for bi, kind, lo, hi in assigns:
                    ls = ordered[bi][1]
                    step = (hi - lo) / max(1, len(ls))
                    for i, x in enumerate(ls):
                        if x.strip() == t or t in x:
                            return lo + i * step
                return None
            knots = {}
            for text, sec in points:
                bb = bar_of(text)
                if bb is not None:
                    knots.setdefault(round(bb, 3), float(sec))
            if verse_sec is not None and chorus_sec is not None:
                knots.setdefault(round(offsets["verse"], 3), float(verse_sec))
                knots.setdefault(round(offsets["chorus"], 3), float(chorus_sec))
            xs = sorted(knots.items())
            if len(xs) >= 2:
                bs = [b for b, _ in xs]
                def tmap(x, xs=xs, bs=bs):
                    if len(xs) == 1:
                        return xs[0][1]
                    j = bisect.bisect_right(bs, x) - 1
                    if j < 0:                     # before the first knot
                        (b0, s0), (b1, s1) = xs[0], xs[1]
                    elif j >= len(xs) - 1:        # after the last knot
                        (b0, s0), (b1, s1) = xs[-2], xs[-1]
                    else:                          # between two knots
                        (b0, s0), (b1, s1) = xs[j], xs[j + 1]
                    return s0 + (x - b0) * (s1 - s0) / (b1 - b0) if b1 != b0 else s0
                mode_extra += f" (piecewise×{len(xs)})"
        for bi, kind, lo, hi in assigns:
            _tag, ls = ordered[bi]
            step = (hi - lo) / max(1, len(ls))
            for i, text in enumerate(ls):
                bar = lo + i * step
                sec = tmap(bar) if tmap is not None else b + bar * a
                line = f"[{fmt(sec)}]{text}"
                lrc.append(line)
                lines.append(line)
        mode_extra = " (multi-section name-matched)" + mode_extra
    with open(os.path.join(out_dir, "grid_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"verse_sec": verse_sec, "chorus_sec": chorus_sec}, f)
    path = os.path.join(out_dir, "audio.lrc")
    open(path, "w", encoding="utf-8").write("\n".join(lrc) + "\n")
    return {"path": path, "lines": lines, "mode": mode + mode_extra,
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