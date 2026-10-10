"""Align lyric lines to real vocal timings via local Faster-Whisper-XXL.

Runs faster-whisper-xxl.exe (CPU int8, word timestamps, lyrics seeded into the
initial prompt) on the rendered FLAC, matches every lyric line against the
word stream with fuzzy similarity and a monotonic pointer, and rewrites
audio.lrc with the measured starts. Lines the ASR cannot match keep the
grid-fallback time from the existing audio.lrc (two-anchor build_lrc output).

Env: YUE2_WHISPER_XXL   path to faster-whisper-xxl.exe (optional)
Usage: python tools/lyrics_align.py outputs/<dir>
"""
import json
import os
import subprocess
import sys
import tempfile
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_lrc import build_lrc  # noqa: E402

XXL_DEFAULT = r"H:/ai/FasterWhisper/Faster-Whisper-XXL_r192.3.4/faster-whisper-xxl.exe"
MODEL_DEFAULT = "large-v3"  # XXL prefixes _models\faster-whisper-<name> itself; junction links our _models
ATTEMPTS = [("cuda", "bfloat16"),  # 5090: cublas 12.9 + bf16 passes; fp16 gemm lacks sm_120
            ("cpu", "int8")]

def norm_chars(text):
    out = []
    for ch in unicodedata.normalize("NFC", text):
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            out.append(ch.lower())
    return "".join(out)

def run_asr(audio_path, lyrics, attempts=None):
    exe = os.environ.get("YUE2_WHISPER_XXL", XXL_DEFAULT)
    out_dir = tempfile.mkdtemp(prefix="yue2_align_")
    prompt = "歌词: " + lyrics[:256]
    audio_path = os.path.abspath(audio_path)
    exe = os.path.abspath(exe)
    detail = ""
    for device, compute in (attempts or ATTEMPTS):
        cmd = [exe, "--model", MODEL_DEFAULT, "--device", device, "--compute_type", compute,
               "--threads", "12", "--language", "zh", "--temperature", "0",
               "--word_timestamps", "True",
               "--output_format", "json", "--output_dir", out_dir, "--vad_filter", "True",
               "--initial_prompt", prompt, audio_path]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        found = [n for n in os.listdir(out_dir) if n.endswith(".json")] if os.path.isdir(out_dir) else []
        if r.returncode == 0 and found:
            print(f"ASR: {device}/{compute}", flush=True)
            with open(os.path.join(out_dir, found[0]), encoding="utf-8") as f:
                data = json.load(f)
            segments = [(float(s["start"]), float(s["end"]), norm_chars(s["text"]))
                        for s in data.get("segments", [])]
            words = []
            for seg in data.get("segments", []):
                for w in seg.get("words") or []:
                    if w.get("start") is None or w["start"] < 0:
                        continue
                    chars = norm_chars(w["word"])
                    if chars:
                        words.append((float(w["start"]), float(w["end"]), chars, w["word"]))
            return words, segments
        detail = f"{r.stdout[-300:]} {r.stderr[-300:]}"
    raise RuntimeError(f"whisper-xxl produced no json\n{detail}")

def align(words, lines, segments=None):
    """lines = [(text, fallback_sec)]; returns [(sec, matched)] monotonic.

    Segment-level greedy assignment. A segment may serve SEVERAL consecutive
    lyric lines (whisper merges lines); selection is monotonic in both
    (segment index, word position) and in seconds, which stops repeated
    chorus lines from grabbing a later chorus instance.
    """
    lchars = [norm_chars(t) for t, _ in lines]
    sp, last_j, last_pos, prev_sec, result = 0, -1, -1, -1.0, []
    for i, (text, fallback) in enumerate(lines):
        chars = lchars[i]
        n = len(chars)
        best = None
        for j in range(max(0, sp), min(sp + 6, len(segments))):
            start0, end0 = segments[j][0], segments[j][1]
            idxs = [k for k, w in enumerate(words) if start0 - 0.01 <= w[0] <= end0 + 0.01]
            if not idxs:
                continue
            for s0 in range(0, len(idxs) - 1):
                for ln in sorted({max(2, int(n * k)) for k in (0.7, 1.0, 1.3)}):
                    e0 = s0 + ln
                    if e0 > len(idxs):
                        continue
                    window = "".join(words[k][2] for k in idxs[s0:e0])
                    import difflib
                    ratio = difflib.SequenceMatcher(None, chars, window).ratio()
                    if ratio < 0.5:
                        continue
                    sec = words[idxs[s0]][0]
                    progress = (j > last_j) or (j == last_j and s0 > last_pos)
                    if progress and sec >= prev_sec - 0.5:
                        cand = (ratio, j, s0, e0, sec)
                        if best is None or ratio > best[0]:
                            best = cand
        if best is not None:
            ratio, j, s0, e0, sec = best
            result.append((sec, True, round(ratio, 2)))
            prev_sec = sec
            last_j, last_pos = j, s0
            sp = j
        else:
            result.append((float(fallback), False, 0.0))
    for i in range(1, len(result)):
        if result[i][0] < result[i-1][0]:
            result[i] = (result[i-1][0], result[i][1], result[i][2])
    return result

def align_words_only(words, lines):
    """Legacy flat-word matcher kept as fallback when segments are absent."""
    wc = [w[2] for w in words]
    n_words = len(wc)
    pointer, prev_sec, result = 0, -1.0, []
    for text, fallback in lines:
        chars = norm_chars(text)
        n = len(chars)
        best = (0.0, 0, 0)
        from_start = max(0, pointer - n // 2)
        for start in range(from_start, min(pointer + 2 * n + 20, n_words)):
            for size in sorted({max(1, int(n * k)) for k in (0.5, 0.75, 1.0, 1.25, 1.5)}):
                end = start + size
                if end > n_words:
                    break
                import difflib
                ratio = difflib.SequenceMatcher(None, chars, "".join(wc[start:end])).ratio()
                if ratio > best[0]:
                    best = (ratio, start, size)
        if best[0] >= 0.5:
            start, size = best[1], best[2]
            sec = words[start][0]
            if sec >= prev_sec - 0.5:
                result.append((sec, True, round(best[0], 2)))
                prev_sec = sec
                pointer = start + max(2, size // 2)
                continue
        result.append((float(fallback), False, 0.0))
    for i in range(1, len(result)):
        if result[i][0] < result[i-1][0]:
            result[i] = (result[i-1][0], result[i][1], result[i][2])
    return result

def fmt_from(sec):
    total = round(max(0.0, sec), 2)
    return f"{int(total // 60):02d}:{total % 60:05.2f}"

def run_asr_json(json_path):
    """Words/segments straight from an existing XXL json (no ASR rerun)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = [(float(s["start"]), float(s["end"]), norm_chars(s["text"]))
                for s in data.get("segments", [])]
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words") or []:
            if w.get("start") is None or w["start"] < 0:
                continue
            chars = norm_chars(w["word"])
            if chars:
                words.append((float(w["start"]), float(w["end"]), chars, w["word"]))
    return words, segments

def align_dir(out_dir, verbose=True, asr_json=None):
    """Runs ASR alignment for an output dir that already has a grid audio.lrc.
    asr_json: use an existing XXL json instead of transcribing again.
    Returns (matched, total) after rewriting audio.lrc in place."""
    request = json.load(open(os.path.join(out_dir, "request.json"), encoding="utf-8"))
    # regenerate a CLEAN grid first: audio.lrc may hold results of an earlier
    # (possibly clamped/frozen) alignment and must never seed the fallbacks
    anchors = {"verse_sec": None, "chorus_sec": None}
    mpath = os.path.join(out_dir, "grid_manifest.json")
    if os.path.exists(mpath):
        anchors.update(json.load(open(mpath, encoding="utf-8")))
    build_lrc(out_dir, anchors["verse_sec"], anchors["chorus_sec"])
    lines = []
    for line in open(os.path.join(out_dir, "audio.lrc"), encoding="utf-8").read().splitlines():
        import re
        m = re.match(r"\[(\d+):(\d\d\.\d\d)\](.*)", line)
        if m:
            lines.append((m.group(3).strip(),
                          int(m.group(1)) * 60 + float(m.group(2))))
    audio = os.path.join(out_dir, "audio.flac")
    if asr_json:
        words, segments = run_asr_json(asr_json)
    else:
        words, segments = run_asr(audio, request["lyrics"], [ATTEMPTS[0]])
    print(f"ASR words: {len(words)} | segments: {len(segments)} | lyric lines: {len(lines)}")
    result = align(words, lines, segments=segments)
    matched = sum(1 for _, m, _ in result if m)
    if matched < (len(lines) // 2) and not asr_json:   # GPU nondeterminism: retry the CPU path
        words2, segments2 = run_asr(audio, request["lyrics"], [ATTEMPTS[1]])
        result2 = align(words2, lines, segments=segments2)
        matched2 = sum(1 for _, m, _ in result2 if m)
        print(f"cpu retry: {matched2}/{len(lines)}")
        if matched2 > matched:
            result, matched = result2, matched2
    # Final merge: strong ASR lines (>=0.6) become piecewise knots; weak
    # matches and unmatched lines interpolate on the calibrated bar grid.
    knots = [(text, sec) for (sec, m, ratio), (text, _) in zip(result, lines)
             if m and ratio >= 0.6]
    info = build_lrc(out_dir, anchors["verse_sec"], anchors["chorus_sec"], points=knots)
    for line in info["lines"]:
        if verbose:
            print(f"    {line}")
    print(f"✅ aligned: {matched}/{len(lines)} lines from ASR ({len(knots)} knots) "
          f"-> piecewise lrc -> {info['path']}")
    return matched, len(lines)

def main():
    asr = sys.argv[2] if len(sys.argv) > 2 else None
    return align_dir(sys.argv[1], asr_json=asr)[1] and 0

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())