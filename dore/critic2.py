#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Critic for Canto II plates — via the shared vision toolkit (vision.py)."""
import sys

from vision import (CHARS, ascii_view, load, luminance_profile, region_stats)


def main(path):
    img, a = load(path)
    W, H = img.size
    print(f"plate {W}x{H}  mean lum={a.mean():.1f}  std={a.std():.1f}")
    print()
    ascii_view(img, 110, "full")
    print()
    luminance_profile(a, 32, axis=0)
    print()
    print("--- region stats ---")
    region_stats(a, 500, 200, 900, 500, "sky top L")
    region_stats(a, 1300, 250, 1900, 550, "sky top R")
    region_stats(a, 1020, 1170, 1180, 1330, "glow")
    region_stats(a, 1040, 1380, 1160, 1540, "path upper")
    region_stats(a, 950, 1950, 1250, 2150, "path lower")
    region_stats(a, 300, 2050, 620, 2200, "ground L")
    region_stats(a, 1580, 2050, 1900, 2200, "ground R")
    region_stats(a, 200, 1400, 460, 2000, "tree L1")
    region_stats(a, 1760, 1400, 2000, 2050, "tree R1")
    region_stats(a, 940, 1870, 1190, 2050, "figures box")
    region_stats(a, 700, 1580, 1500, 1770, "mist band")
    region_stats(a, 200, 2290, 2000, 2380, "caption band")
    print()
    sub = a[1860:2040, 960:1180]
    print(f"  figures ink%={(sub < 130).mean()*100:.1f} (expect ~8-25)")
    pl = a[1950:2150, 950:1250].mean()
    gl = a[2050:2200, 300:620].mean()
    print(f"  path lum={pl:.0f} vs ground lum={gl:.0f}  (path brighter)")
    print(f"  sky top lum={a[200:500, 500:900].mean():.0f} vs glow lum={a[1170:1330, 1020:1180].mean():.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/c2_v01.png")
