#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Critic for Canto III plates — via the shared vision toolkit (vision.py)."""
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
    region_stats(a, 200, 160, 620, 900, "rock upper L")
    region_stats(a, 1580, 160, 2000, 900, "rock upper R")
    region_stats(a, 200, 1500, 600, 2100, "rock lower L")
    region_stats(a, 1600, 1500, 2000, 2100, "rock lower R")
    region_stats(a, 660, 1250, 1560, 1320, "lintel")
    region_stats(a, 900, 1500, 1300, 1850, "gate interior")
    region_stats(a, 1020, 1760, 1180, 2000, "interior glow")
    region_stats(a, 950, 2040, 1200, 2160, "figures zone")
    region_stats(a, 300, 2100, 700, 2220, "floor L")
    region_stats(a, 1520, 2100, 1900, 2220, "floor R")
    region_stats(a, 200, 2290, 2000, 2380, "caption band")
    print()
    sub = a[1255:1318, 700:1500]
    print(f"  lintel ink%={(sub < 160).mean()*100:.1f} (carved letters expect ~5-15)")
    sub = a[2020:2140, 950:1200]
    print(f"  figures ink%={(sub < 130).mean()*100:.1f} (expect ~10-30)")
    print(f"  glow lum={a[1760:2000, 1020:1180].mean():.0f} vs rock lum={a[1500:2100, 200:600].mean():.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "DORE_INFERNO/Chant_III_La_Porte_de_l_Enfer/c3_v01.png")
