"""Probe: syllabic onset density per 3s window in the vocal band (300-3300 Hz).

Expect a clear jump where the singing enters (user-measured verse ~26.3s,
chorus ~49.8s for outputs/waltz-v2).
"""
import os
import sys

import numpy as np
import soundfile as sf

def main(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    mag, sr, hop = stft(y, sr)
    frame_sec = hop / sr
    bins = np.fft.rfftfreq(2 * mag.shape[1] - 2, d=1 / sr)
    band = mag[:, (bins >= 300) & (bins <= 3300)]
    flux = np.concatenate([[0.0], np.maximum(band[1:] - band[:-1], 0.0).sum(axis=1)])
    thr = np.percentile(flux, 75)
    onsets = flux > thr
    t = np.arange(len(onsets)) * frame_sec
    win, step = 3.0, 1.0
    print("win_start  density  (onsets/s)")
    for w0 in np.arange(0, len(y) / sr - win, step):
        m = (t >= w0) & (t < w0 + win)
        d = onsets[m].sum() / win
        bar = "#" * int(d)
        print(f"{w0:7.1f}   {d:5.2f}  {bar}")

def stft(y, sr, n_fft=2048, hop=512):
    frames = max(0, (len(y) - n_fft) // hop)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    win = y[idx] * np.hanning(n_fft)[None, :]
    return np.abs(np.fft.rfft(win, axis=1)), sr, hop

if __name__ == "__main__":
    main(sys.argv[1])