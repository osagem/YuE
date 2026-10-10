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

    Segment-level greedy assignment against the ASR segment list (segment
    text is nearly 1:1 with a lyric line; hallucinated filler segments match
    nothing and get skipped). Word timestamps place lines inside merged or
    split segments. Fallbacks keep the grid time; a final clamp restores
    strict monotonicity.
    """
    if not segments:
        # rebuild segment boundaries from the flat word list is unreliable;
        # caller passes segments
        return align_words_only(words, lines)
    lchars = [norm_chars(t) for t, _ in lines]
    fchars = [norm_chars(s[2]) for s in segments]
    sp, prev_sec, result = 0, -1.0, []
    for i, (text, fallback) in enumerate(lines):
        n = len(lchars[i])
        best = (0.0, sp)
        for j in range(sp, min(sp + 6, len(segments))):
            import difflib
            ratio = difflib.SequenceMatcher(None, lchars[i], fchars[j]).ratio()
            if ratio > best[0]:
                best = (ratio, j)
        ratio, j = best
        if ratio >= 0.5:
            seg = segments[j]
            sec, ok, _ = place_inside(seg, words, chars=lchars[i], min_sec=prev_sec - 0.5)
            if ok and sec >= prev_sec - 0.5:
                result.append((sec, True, round(ratio, 2)))
                prev_sec = sec
                sp = j + 1
                continue
        result.append((float(fallback), False, 0.0))
    for i in range(1, len(result)):
        if result[i][0] < result[i-1][0]:
            result[i] = (result[i-1][0], result[i][1], result[i][2])
    return result

def place_inside(seg, words, *, chars, min_sec):
    """Locate one line's chars among a segment's words; returns (sec, ok, ratio).
    The word list here is the GLOBAL word stream around the segment, sliced to
    the segment so merged segments resolve line-by-line with word timings.
    """
    import difflib
    n = len(chars)
    in_seg = [(w[0], w[2]) for w in words if seg[0] - 0.01 <= w[0] <= seg[1] + 0.01]
    if not in_seg:
        return seg[0], True, 0.0
    joined = [c for _, c in in_seg]
    best = (0.0, 0, len(joined))
    for start0 in range(0, max(1, len(joined) - 2)):
        for end0 in range(start0 + 1, len(joined) + 1):
            window = "".join(joined[start0:end0])
            if abs(len(window) - n) > max(4, n // 2):
                continue
            ratio = difflib.SequenceMatcher(None, chars, window).ratio()
            if ratio > best[0]:
                best = (ratio, start0, end0)
    ratio, s0, e0 = best
    if ratio >= 0.5 and in_seg[s0][0] >= min_sec:
        return in_seg[s0][0], True, ratio
    if ratio >= 0.75:
        return seg[0], True, ratio
    return seg[0], False, ratio

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

def align_dir(out_dir, verbose=True):
    """Runs ASR alignment for an output dir that already has a grid audio.lrc.
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
    words, segments = run_asr(audio, request["lyrics"], [ATTEMPTS[0]])
    print(f"ASR words: {len(words)} | segments: {len(segments)} | lyric lines: {len(lines)}")
    result = align(words, lines, segments=segments)
    matched = sum(1 for _, m, _ in result if m)
    if matched < (len(lines) // 2):   # GPU nondeterminism: retry the CPU path
        words2, segments2 = run_asr(audio, request["lyrics"], [ATTEMPTS[1]])
        result2 = align(words2, lines, segments=segments2)
        matched2 = sum(1 for _, m, _ in result2 if m)
        print(f"cpu retry: {matched2}/{len(lines)}")
        if matched2 > matched:
            result, matched = result2, matched2
    lrc = ["[by:YuE2]"]
    for (sec, m, ratio), (text, _) in zip(result, lines):
        lrc.append(f"[{fmt_from(sec)}]{text}")
        if verbose:
            flag = f"ASR({ratio})" if m else "grid"
            print(f"[{fmt_from(sec):>9}]{text}   <- {flag}")
    path = os.path.join(out_dir, "audio.lrc")
    open(path, "w", encoding="utf-8").write("\n".join(lrc) + "\n")
    print(f"✅ aligned: {matched}/{len(lines)} lines from ASR, rest kept grid -> {path}")
    return matched, len(lines)

def main():
    return align_dir(sys.argv[1])[1] and 0

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())