"""Auto-calibrate LRC anchors against the rendered audio.

The voice band's (250-4000 Hz) onset stream bursts where singing enters — the
legato accompaniment produces almost no onsets — but the flux burst lags the
audible attack by ~0.6 s. Lyric lines recur at the 4-bar lyric-block period,
so autocorrelating the onset stream inside the sung region gives the line
period directly ("bar period" = line period / 4); that beats the score's
nominal tempo because the NAR renderer's timing is expressive. The chorus
entry is re-detected with the same burst search around its predicted time.
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_lrc import parse_sections  # noqa: E402


def load_audio(out_dir):
    import soundfile as sf
    y, sr = sf.read(os.path.join(out_dir, "audio.flac"), dtype="float32", always_2d=True)
    return y.mean(axis=1), sr


def stft_mag(y, sr, n_fft=2048, hop=512):
    frames = max(0, (len(y) - n_fft) // hop)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    win = y[idx] * np.hanning(n_fft)[None, :]
    return np.abs(np.fft.rfft(win, axis=1)), sr, hop


def flux_env(mag):
    f = np.concatenate([[0.0], np.maximum(mag[1:] - mag[:-1], 0.0).sum(axis=1)])
    return np.convolve(f, np.ones(3) / 3, mode="same")


def score_grid(out_dir):
    audio_sec = json.load(open(os.path.join(out_dir, "result.json"), encoding="utf-8"))["audio_seconds"]
    score = open(os.path.join(out_dir, "score.abc"), encoding="utf-8").read()
    sections = parse_sections(score)
    bars, offsets = [], {}
    for name in ("intro", "verse", "chorus", "outro"):
        if name in sections:
            offsets[name] = len(bars)
            bars.extend(sections[name])
    q = re.search(r"Q:1/4=(\d+)", score)
    bpm = int(q.group(1)) if q else 133
    bar_theo = (60.0 / bpm) / 8.0 * 24.0   # L:1/32, 24 units per 6/8 or 3/4 bar
    return bars, offsets, bar_theo, audio_sec


def detect_anchors(out_dir):
    """v5: two-mode vocal entry detection on 0.5 s voice-band onset bins.

    Abrupt attacks (vocal pickup on quiet accompaniment) show a jump >=4x from
    the previous bin; crescendo entries show a gradual rise into a sustained
    plateau. In both cases the audible attack sits ~0.2 s before the detected
    bin start (half a bin). The line/bar period is only diagnostic: the chorus
    entry is re-detected with the same search around a coarse prediction.
    """
    bars, offsets, bar_theo, audio_sec = score_grid(out_dir)
    if "verse" not in offsets or "chorus" not in offsets:
        raise RuntimeError("score lacks verse/chorus sections for anchor detection")
    y, sr = load_audio(out_dir)
    mag, sr, hop = stft_mag(y, sr)
    bins = np.fft.rfftfreq(2 * mag.shape[1] - 2, d=1 / sr)
    frame_sec = hop / sr

    band = mag[:, (bins >= 250) & (bins <= 4000)]
    on = (flux_env(band) > np.percentile(flux_env(band), 75)).astype(float)
    binw = 0.5
    nb = int(mag.shape[0] * frame_sec / binw)
    cnt = np.array([on[int(b * binw / frame_sec):int((b + 1) * binw / frame_sec)].sum()
                    for b in range(nb)])
    bt = np.arange(nb) * binw

    def find_entry(t_lo, t_hi):
        w = (bt >= t_lo) & (bt < t_hi)
        wmax = float(cnt[w].max())
        base = float(np.median(cnt[w][cnt[w] <= np.percentile(cnt[w], 30)])) if w.any() else 0.0
        t_abrupt, t_plateau = None, None
        prev = np.concatenate([[0.0], cnt[w][:-1]])
        for k, b in enumerate(np.where(w)[0]):
            if cnt[b] >= max(5.0, 4.0 * base) and cnt[b] >= 4.0 * max(prev[k], 0.5) and t_abrupt is None:
                t_abrupt = bt[b]
            if cnt[b] >= 0.55 * wmax and t_plateau is None:
                t_plateau = bt[b]
        if t_abrupt is not None:
            return t_abrupt - 0.2, "abrupt"
        return t_plateau - 0.2, "plateau"

    theo_verse = offsets["verse"] * bar_theo
    verse_sec, verse_mode = find_entry(0.85 * theo_verse, 1.12 * theo_verse)

    est = verse_sec + (offsets["chorus"] - offsets["verse"]) * (bar_theo * 0.97)
    chorus_sec, chorus_mode = find_entry(est - 2.0, est + 4.0)

    # line period (diagnostic): autocorr of onset counts in the sung region
    a0 = int((verse_sec + 0.5) / binw)
    a1 = max(a0 + 1, int((verse_sec + 1 + 20 * bar_theo) / binw))
    seg = cnt[a0:a1].astype(float)
    seg = seg - seg.mean()
    best_v, line_p = 0.0, bar_theo * 4.0
    for L_b in np.arange(4.0, 7.0, binw):
        L = int(L_b / binw)
        if L < len(seg) and len(seg) > L:
            v = float(np.dot(seg[:-L], seg[L:])) / max(float(np.dot(seg, seg)), 1e-9)
            if v > best_v:
                best_v, line_p = v, L_b
    conf = max(0.0, min(1.0, best_v * 1.2 + 0.35))
    return {"verse_sec": round(verse_sec, 2), "chorus_sec": round(chorus_sec, 2),
            "verse_mode": verse_mode, "chorus_mode": chorus_mode,
            "line_period": round(line_p, 2), "line_ac": round(best_v, 3),
            "bar_theoretical": round(bar_theo, 4), "confidence": round(conf, 3),
            "audio_seconds": audio_sec, "total_bars": len(bars)}


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    import pprint
    pprint.pprint(detect_anchors(sys.argv[1]))