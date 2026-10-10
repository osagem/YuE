"""Embed lyrics/metadata Vorbis comments into a YuE2 output FLAC (ffmpeg -c copy).

Reads request.json from the output dir; keeps the audio bitstream identical.
Usage (CLI): python tools/tag_song.py outputs/waltz-v2 [title]
"""
import json
import os
import subprocess


def tag(out_dir, title=None):
    """Returns (path, list of tag key=value strings written)."""
    req = json.load(open(os.path.join(out_dir, "request.json"), encoding="utf-8"))
    lyr = req.get("lyrics", "")
    src = os.path.join(out_dir, "audio.flac")
    tmp = src + ".tagged.flac"
    pairs = [
        ("TITLE", title or req.get("id", "YuE2 Song")),
        ("ARTIST", "YuE2"),
        ("ALBUM", "YuE2 Generations"),
        ("DATE", req.get("created", "") or "2026-10-11"),
        ("GENRE", "Cinematic Waltz"),
        ("LYRICS", lyr),
        ("UNSYNCEDLYRICS", lyr),
        ("DESCRIPTION", req.get("style", "")[:300]),
    ]
    cmd = ["ffmpeg", "-y", "-i", src, "-map_metadata", "-1"]
    for k, v in pairs:
        cmd += ["-metadata", f"{k}={v}"]
    cmd += ["-c", "copy", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-800:]}")
    os.replace(tmp, src)
    return src, pairs


def main():
    out_dir = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else None
    src, pairs = tag(out_dir, title)
    print(f"✅ tagged: {src} ({len(pairs)} tags)")
    return 0


import sys

if __name__ == "__main__":
    raise SystemExit(main())