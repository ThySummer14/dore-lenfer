#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch health-check across all nine canto plates.

Checks per plate:
  - caption-zone contamination: strokes that crossed the content boundary
    (y > 2220) other than the letterpress text itself
  - caption band luminance (should be ~190+ on the 0.24-coefficient plates
    once fixed; text rows excluded)
  - global mean / std
  - FFT regularity on a central band
  - per-plate key region quick stats
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from vision import load, fft_regularity

PLATES = [
    ("Chant I", "DORE_INFERNO/Chant_I_La_Lumiere_Divine/DORE_LA_LUMIERE_DIVINE.png"),
    ("Chant II", "DORE_INFERNO/Chant_II_La_Foret_Obscure/DORE_LA_FORET_OBSCURE.png"),
    ("Chant III", "DORE_INFERNO/Chant_III_La_Porte_de_l_Enfer/DORE_LA_PORTE_DE_L_ENFER.png"),
    ("Chant IV", "DORE_INFERNO/Chant_IV_Le_Noble_Chateau/DORE_LE_NOBLE_CHATEAU.png"),
    ("Chant V", "DORE_INFERNO/Chant_V_Paolo_et_Francesca/DORE_PAOLO_ET_FRANCESCA.png"),
    ("Chant VI", "DORE_INFERNO/Chant_VI_Cerbere/DORE_CERBERE.png"),
    ("Chant VII", "DORE_INFERNO/Chant_VII_Les_Avares/DORE_LES_AVARES.png"),
    ("Chant VIII", "DORE_INFERNO/Chant_VIII_La_Barque_de_Phlegyas/DORE_LA_BARQUE_DE_PHLEGYAS.png"),
    ("Chant IX", "DORE_INFERNO/Chant_IX_L_Ange_ouvre_les_Portes/DORE_L_ANGE_OUVRE_LES_PORTES.png"),
]


def check(path):
    img, a = load(path)
    # caption zones (text rows excluded: title ~2290-2360, sub ~2380-2440, cred ~2500-2560)
    seam = a[2218:2232, 140:2060]        # content bottom edge (dark content here is fine)
    gap1 = a[2232:2280, 140:2060]        # above the title — must be clean
    gap2 = a[2445:2495, 140:2060]        # between sub and credits — must be clean
    gap3 = a[2520:2560, 140:2060]        # below credits, above frame — must be clean
    rows = (gap1, gap2, gap3)
    ink = [(r < 150).mean() * 100 for r in rows]
    lum = [float(r.mean()) for r in rows]
    clean = max(ink) < 3.0
    status = "OK  " if clean else "DIRTY"
    print(f"[{status}] {path.split('/')[1]}  mean={a.mean():.1f}")
    print(f"      caption gaps lum={['%.0f' % v for v in lum]}  ink%={['%.1f' % v for v in ink]}")
    fft = fft_regularity(a[100:900, 300:1900], "central band")
    return clean, max(ink), min(lum)


def main():
    print("=== nine-plate health check ===")
    all_clean = True
    for name, path in PLATES:
        res = check(path)
        clean = res[0]
        all_clean = all_clean and clean
    print()
    print("ALL CLEAN" if all_clean else "ISSUES FOUND — see DIRTY rows above")


if __name__ == "__main__":
    main()
