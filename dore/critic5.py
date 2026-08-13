#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Critic for Canto V plates — via the shared vision toolkit (vision.py)."""
import sys

from vision import (CHARS, ascii_view, fft_regularity, load,
                    luminance_profile, region_stats)


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
    region_stats(a, 200, 150, 700, 700, "storm upper L")
    region_stats(a, 1500, 150, 2000, 700, "storm upper R")
    region_stats(a, 800, 900, 1400, 1500, "eye zone")
    region_stats(a, 1030, 1100, 1170, 1270, "the pair")
    region_stats(a, 300, 800, 900, 1500, "bodies L")
    region_stats(a, 1300, 800, 1900, 1500, "bodies R")
    region_stats(a, 300, 2050, 900, 2200, "ground L")
    region_stats(a, 1300, 2050, 1900, 2200, "ground R")
    region_stats(a, 900, 1980, 1220, 2180, "dante+virgil")
    region_stats(a, 200, 2290, 2000, 2380, "caption band")
    print()
    # the pair should be the brightest patch in the storm
    pair = a[1100:1270, 1030:1170].mean()
    storm = a[150:900, 1500:2000].mean()
    print(f"  pair lum={pair:.0f} vs storm lum={storm:.0f}  (pair brighter)")
    # bodies: pale marks on dark storm → count bright pixels in a storm band
    band = a[400:1500, 300:1900]
    print(f"  bright marks in storm: {(band > 210).mean()*100:.1f}% (pale bodies, expect ~0.5-2)")
    # figures presence
    sub = a[2000:2160, 950:1200]
    print(f"  figures ink%={(sub < 130).mean()*100:.1f} (expect ~10-30)")
    fft_regularity(a[300:1500, 300:1900], "storm")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "DORE_INFERNO/Chant_V_Paolo_et_Francesca/c5_v01.png")
