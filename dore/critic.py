#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Critic for Chant I plates — numeric + ASCII inspection via vision.py.

Presence checks: vortex rings, rays, pilgrim silhouette, frame.
"""
import sys

from vision import (CHARS, ascii_view, fft_regularity, load,
                    luminance_profile, region_stats)


def main(path, fast=False):
    img, a = load(path)
    W, H = img.size
    print(f"plate {W}x{H}  mean lum={a.mean():.1f}  std={a.std():.1f}")
    print()
    ascii_view(img, 110 if not fast else 64, "full")
    print()
    luminance_profile(a, 32, axis=0)
    print()
    print("--- region stats ---")
    region_stats(a, 950, 560, 1250, 860, "glow core")
    region_stats(a, 1040, 1200, 1160, 1500, "light corridor")
    region_stats(a, 250, 300, 700, 600, "outer sky L")
    region_stats(a, 1600, 400, 2000, 700, "outer sky R")
    region_stats(a, 200, 1900, 420, 2100, "left cliff")
    region_stats(a, 1820, 1960, 2050, 2150, "right rock")
    region_stats(a, 420, 1900, 960, 2150, "gulf L")
    region_stats(a, 1290, 1900, 1800, 2150, "gulf R")
    region_stats(a, 1060, 2000, 1210, 2150, "pilgrim area")
    region_stats(a, 250, 1440, 470, 1720, "tree area")
    region_stats(a, 200, 2290, 2000, 2380, "caption band")
    print()
    sub = a[1985:2150, 1080:1190]
    print(f"  pilgrim box ink%={(sub < 120).mean()*100:.1f} (expect ~15-45)")
    print(f"  targets: corridor ~190-215 (brightest), sky L/R ~150-190, cliffs/gulf 90-150")
    corr = a[1020:1500, 1020:1180].mean()
    outr = a[300:600, 1600:2000].mean()
    print(f"  corridor lum={corr:.0f} vs outer sky lum={outr:.0f}  (corridor should be brighter)")
    print()
    fft_regularity(a[380:900, 500:1700], "sky")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/v01.png",
         fast="--fast" in sys.argv)
