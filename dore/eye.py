#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eye.py — unified CLI over dore/vision.py.

Plain-text reports by default; `--json` emits ONE json object carrying the
same text plus structured numbers, so external drivers (e.g. the Doré vision
plugin for DeepSeek Harness) can consume the toolkit through a single entry
point. Dependencies: Pillow + numpy only, no network, no model.

Examples:
  python3 dore/eye.py report  DORE_INFERNO/Chant_I_La_Lumiere_Divine/DORE_LA_LUMIERE_DIVINE.png
  python3 dore/eye.py ascii   <img> --crop 950 560 1250 860
  python3 dore/eye.py pixel   <img> --crop 1050 1150 1150 1250 --step 1
  python3 dore/eye.py ink     <img> --paper-lum 232 --grid 8
  python3 dore/eye.py metrics <img> --region glow,950,560,1250,860 \
      --point core,1100,700 --profile-axis 0 --bins 32 \
      --fft-region 100,100,2100,900
  python3 dore/eye.py critic  <img> --canto 1
"""
import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vision as v  # noqa: E402


def _capture(fn, *args, **kwargs):
    """Run a printing function; return (text, return_value)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        value = fn(*args, **kwargs)
    return buf.getvalue().rstrip("\n"), value


def _central_band(a):
    """Central ~60% band of the plate — robust for any image size."""
    H, W = a.shape
    return a[H // 4: 3 * H // 4, W // 5: 4 * W // 5]


# ----------------------------------------------------------------------------
# ops — each returns (text, extra_dict)
# ----------------------------------------------------------------------------
def op_report(path, fast=False, width=110):
    img, a = v.load(path)
    W, H = img.size
    parts = [f"plate {W}x{H}  mean lum={a.mean():.1f}  std={a.std():.1f}", ""]
    txt, _ = _capture(v.ascii_view, img, 64 if fast else width, "full")
    parts.append(txt)
    parts.append("")
    txt, prof = _capture(v.luminance_profile, a, 32, 0)
    parts.append(txt)
    parts.append("")
    txt, ratio = _capture(v.fft_regularity, _central_band(a), "central band")
    parts.append(txt)
    txt, counts = _capture(v.histogram, a, 16)
    parts.append(txt)
    return ("\n".join(parts),
            {"width": W, "height": H,
             "mean_lum": float(a.mean()), "std_lum": float(a.std()),
             "profile": [float(x) for x in prof],
             "fft_ratio": ratio, "histogram": [int(c) for c in counts]})


def op_ascii(path, crop=None, width=110):
    img, a = v.load(path)
    if crop is None:
        txt, _ = _capture(v.ascii_view, img, width, "full")
    else:
        x0, y0, x1, y1 = crop
        arr = v.crop(a, x0, y0, x1, y1, width)
        txt, _ = _capture(v.ascii_arr, arr, width, f"crop {x0},{y0}-{x1},{y1}")
    return txt, {}


def op_pixel(path, crop, step=2):
    if crop is None:
        raise SystemExit("pixel requires --crop x0,y0,x1,y1")
    img, a = v.load(path)
    x0, y0, x1, y1 = crop
    txt, _ = _capture(v.pixel_window, a, x0, y0, x1, y1, step)
    return txt, {}


def op_ink(path, paper_lum=232.0, width=110, grid=8):
    img, a = v.load(path)
    parts = []
    txt, _ = _capture(v.ink_ascii, a, paper_lum, width, "ink map")
    parts.append(txt)
    parts.append("")
    txt, g = _capture(v.ink_grid, a, grid, paper_lum)
    parts.append(txt)
    return ("\n".join(parts),
            {"ink_mean": float(v.ink_map(a, paper_lum).mean()),
             "ink_grid": [[float(x) for x in row] for row in g]})


def op_metrics(path, regions, points, profile_axis, bins, fft_region):
    img, a = v.load(path)
    W, H = img.size
    parts = [f"plate {W}x{H}  mean lum={a.mean():.1f}  std={a.std():.1f}"]
    stats_out, prof, ratio, counts = [], None, None, None
    if regions:
        parts.append("--- region stats ---")
        for name, x0, y0, x1, y1 in regions:
            txt, (lum, dark, bright) = _capture(v.region_stats, a, x0, y0, x1, y1, name)
            parts.append(txt)
            stats_out.append({"name": name, "lum": lum, "dark": dark, "bright": bright})
    if profile_axis is not None:
        parts.append("")
        txt, prof = _capture(v.luminance_profile, a, bins, profile_axis)
        parts.append(txt)
    if fft_region is not None:
        parts.append("")
        x0, y0, x1, y1 = fft_region
        txt, ratio = _capture(v.fft_regularity, a[y0:y1, x0:x1], "selected region")
        parts.append(txt)
    if points:
        parts.append("")
        txt, _ = _capture(v.probe, a, [(name, int(x), int(y)) for name, x, y in points])
        parts.append(txt)
    if not (regions or points or profile_axis is not None or fft_region is not None):
        parts.append("")
        txt, counts = _capture(v.histogram, a, bins)
        parts.append(txt)
    extra = {"regions": stats_out, "profile": prof, "fft_ratio": ratio}
    if counts is not None:
        extra["histogram"] = [int(c) for c in counts]
    return "\n".join(parts), extra


def op_critic(path, canto):
    suffix = "" if canto == 1 else str(canto)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"critic{suffix}.py")
    if not os.path.exists(script):
        raise SystemExit(f"no critic for canto {canto} ({script})")
    proc = subprocess.run([sys.executable, script, path],
                          capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise SystemExit(f"critic{suffix} failed: {proc.stderr.strip()[:800]}")
    return proc.stdout.rstrip("\n"), {"canto": canto}


# ----------------------------------------------------------------------------
# cli
# ----------------------------------------------------------------------------
def _parse_region(s):
    bits = s.split(",")
    if len(bits) != 5:
        raise SystemExit(f"--region expects NAME,X0,Y0,X1,Y1; got {s!r}")
    return bits[0], *[int(x) for x in bits[1:]]


def _parse_point(s):
    bits = s.split(",")
    if len(bits) != 3:
        raise SystemExit(f"--point expects NAME,X,Y; got {s!r}")
    return bits[0], int(bits[1]), int(bits[2])


_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
          "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}


def _guess_canto(path):
    m = re.search(r"chant[_ ]?([ivx]+)", path.lower())
    return _ROMAN.get(m.group(1)) if m else None


def main(argv=None):
    p = argparse.ArgumentParser(description="Doré invented-vision toolkit CLI")
    p.add_argument("op", choices=["report", "ascii", "pixel", "ink", "metrics", "critic"])
    p.add_argument("path", help="image path (PNG/JPG)")
    p.add_argument("--crop", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--width", type=int, default=None, help="ASCII width in chars")
    p.add_argument("--step", type=int, default=2, help="pixel-window stride")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--paper-lum", type=float, default=232.0)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--profile-axis", type=int, default=None, choices=[0, 1])
    p.add_argument("--region", action="append", default=[],
                   metavar="NAME,X0,Y0,X1,Y1", help="named rectangle; repeatable")
    p.add_argument("--point", action="append", default=[],
                   metavar="NAME,X,Y", help="exact sample point; repeatable")
    p.add_argument("--fft-region", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--canto", type=int, default=None, choices=range(1, 11))
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    ok, error, text, extra = True, None, "", {}
    try:
        if args.op == "report":
            text, extra = op_report(args.path, fast=args.fast, width=args.width or 110)
        elif args.op == "ascii":
            text, extra = op_ascii(args.path, tuple(args.crop) if args.crop else None,
                                   width=args.width or 90)
        elif args.op == "pixel":
            text, extra = op_pixel(args.path, tuple(args.crop) if args.crop else None,
                                   step=args.step)
        elif args.op == "ink":
            text, extra = op_ink(args.path, paper_lum=args.paper_lum,
                                 width=args.width or 110, grid=args.grid)
        elif args.op == "metrics":
            regions = [_parse_region(s) for s in args.region] or None
            points = [_parse_point(s) for s in args.point] or None
            text, extra = op_metrics(args.path, regions, points, args.profile_axis,
                                     args.bins,
                                     tuple(args.fft_region) if args.fft_region else None)
        else:  # critic
            canto = args.canto or _guess_canto(args.path)
            if canto is None:
                raise SystemExit("critic needs --canto 1..5 or a path containing 'chant_N'")
            text, extra = op_critic(args.path, canto)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — the CLI reports, never re-raises
        ok, text, extra, error = False, "", {}, f"{type(exc).__name__}: {exc}"

    if args.json:
        print(json.dumps({"ok": ok, "op": args.op, "path": args.path,
                          "error": error, "text": text, **extra}))
    else:
        print(text if ok else f"error: {error}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
