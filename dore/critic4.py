#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Critic for Canto IV plates — via the shared vision toolkit (vision.py)."""
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
    region_stats(a, 200, 150, 700, 700, "ceiling L")
    region_stats(a, 1500, 150, 2000, 700, "ceiling R")
    region_stats(a, 860, 1000, 1340, 1200, "castle zone")
    region_stats(a, 990, 1190, 1210, 1540, "castle body")
    region_stats(a, 900, 1180, 1300, 1400, "castle glow")
    region_stats(a, 1040, 1540, 1160, 1610, "bridge")
    region_stats(a, 250, 1600, 800, 2000, "plain L")
    region_stats(a, 1400, 1600, 1950, 2000, "plain R")
    region_stats(a, 960, 1950, 1240, 2200, "path+figures")
    region_stats(a, 250, 1470, 900, 1700, "spirits L")
    region_stats(a, 200, 2290, 2000, 2380, "caption band")
    print()
    win = a[1290:1364, 1082:1118]
    print(f"  keep window lum={win.mean():.0f} (expect >190)")
    body = a[1250:1500, 1000:1200]
    print(f"  keep wall lum={body.mean():.0f} (expect 110-160)")
    sub = a[2000:2150, 970:1200]
    print(f"  figures ink%={(sub < 130).mean()*100:.1f} (expect ~8-25)")
    print(f"  glow lum={a[1180:1400, 900:1300].mean():.0f} vs ceiling lum={a[150:700, 1500:2000].mean():.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "DORE_INFERNO/Chant_IV_Le_Noble_Chateau/c4_v01.png")
