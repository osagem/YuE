"""One-command YuE2 generation: audio + embedded lyrics tags + synced LRC.

Runs examples/generate.py, then embeds plain-text lyrics / title metadata into
the FLAC (ffmpeg stream copy) and writes audio.lrc. LRC anchors are
auto-detected from the rendered audio (audio_align.detect_anchors); pass
--verse-sec/--chorus-sec to calibrate manually instead, or --lrc off to skip.
Any other flag (e.g. --cot, --abc-file, --model) is passed through to
examples/generate.py.

Usage: python tools/generate_full.py --output outputs/my-song --title "名称"
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    ap.add_argument("--request", default="examples/song.json")
    ap.add_argument("--title", help="TITLE tag; default = request id")
    ap.add_argument("--lrc", choices=("auto", "off"), default="auto",
                    help="synced audio.lrc: auto-detected anchors")
    ap.add_argument("--verse-sec", type=float, help="manual anchor: verse vocal onset")
    ap.add_argument("--chorus-sec", type=float, help="manual anchor: chorus vocal onset")
    ap.add_argument("--no-tag", action="store_true", help="skip FLAC metadata embedding")
    args, extra = ap.parse_known_args()
    if extra:
        print(f"[+] passthrough flags: {' '.join(extra)}")

    print("[1/3] generating (examples/generate.py)...")
    gen = subprocess.run([sys.executable, "examples/generate.py",
                          "--request", args.request, "--output", args.output] + extra)
    if gen.returncode != 0:
        return gen.returncode
    result = json.load(open(os.path.join(args.output, "result.json"), encoding="utf-8"))
    if any(result["truncated"].values()):
        print("⚠️ generation truncated:", result["truncated"])

    if not args.no_tag:
        print("[2/3] embedding lyrics/metadata tags...")
        from tag_song import tag
        src, pairs = tag(args.output, args.title)
        print(f"    ✅ {len(pairs)} tags -> {os.path.basename(src)}")
    else:
        print("[2/3] tags skipped")
        src = os.path.join(args.output, "audio.flac")

    print("[3/3] synced lyrics (audio.lrc)...")
    if args.lrc == "off":
        print("    skipped (--lrc off)")
        return 0
    if args.verse_sec is not None and args.chorus_sec is not None:
        info = build_lrc_safe(args.output, args.verse_sec, args.chorus_sec)
        print(f"    anchors: manual verse={args.verse_sec}s chorus={args.chorus_sec}s")
    else:
        from audio_align import detect_anchors
        a = detect_anchors(args.output)
        print(f"    detected: verse={a['verse_sec']}s ({a['verse_mode']}), "
              f"chorus={a['chorus_sec']}s ({a['chorus_mode']}), "
              f"line_period={a['line_period']}s ac={a['line_ac']}, conf={a['confidence']}")
        if a["confidence"] < 0.35:
            print("    ⚠️ low confidence; re-run with --verse-sec/--chorus-sec if lines drift")
        info = build_lrc_safe(args.output, a["verse_sec"], a["chorus_sec"])
    for line in info["lines"]:
        print(f"    {line}")
    print(f"    ✅ {info['path']} (mode: {info['mode']}, bar_sec={info['bar_sec']})")
    print(f"✅ done: {src} + {info.get('path', 'no lrc')} "
          f"({result['audio_seconds']:.1f}s audio, e2e {result['timing']['e2e_seconds']:.0f}s)")
    return 0


def build_lrc_safe(out_dir, verse, chorus):
    from make_lrc import build_lrc
    return build_lrc(out_dir, verse, chorus)


if __name__ == "__main__":
    raise SystemExit(main())