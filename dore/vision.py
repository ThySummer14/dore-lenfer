#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Doré critic's eye — a self-contained "invented vision" for inspecting
generated engraving plates when no image-capable model is attached.

Everything is grayscale-luminance based. The toolkit offers:

  load(path)                      open any image as a float32 luminance array
  ascii_view(img, ...)            multi-scale ASCII render of a whole image
  ascii_arr(arr, ...)             same, from an array (for crops)
  crop(arr, x0, y0, x1, y1, w)    downsample a crop into an array for ascii_arr
  ink_map(a, paper_lum)           paper-minus-luminance map: where the ink went
  ink_ascii(a, ...)               ASCII render of the ink map ('@' = heavy ink)
  ink_grid(a, n, paper_lum)       n x n mean-ink grid: locates missing/extra ink
  histogram(a, bins)              luminance histogram (percent per bin)
  region_stats(a, x0, y0, x1, y1, name)
                                  tone statistics of a named rectangle
  luminance_profile(a, bins, axis)  binned mean profile (0 = vertical)
  fft_regularity(a, label)        FFT peak ratio: mechanical repetition / moiré
  probe(a, points)                sample luminance at exact pixel coordinates
  pixel_window(a, x0, y0, x1, y1, step)
                                  1:1 rendering of a small window (every nth px)

Conventions that matter when reading the ASCII output:
  CHARS = " .:-=+*#%@" maps luminance 0..255 onto 10 levels:
  " " = brightest (0-25), "@" = darkest (230-255). So a dark figure prints
  as ".", a sun prints as "@".

Dependencies: Pillow, numpy. No network, no model — pure local computation.
"""
import numpy as np
from PIL import Image

CHARS = " .:-=+*#%@"


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
def load(path):
    """Open any image and return (rgb_image, float32 luminance array)."""
    img = Image.open(path).convert("RGB")
    a = np.asarray(img.convert("L")).astype(np.float32)
    return img, a


# ----------------------------------------------------------------------------
# ASCII rendering
# ----------------------------------------------------------------------------
def _print_ascii(arr, label):
    print(f"--- ASCII {label} ({arr.shape[1]}x{arr.shape[0]}) ---")
    for row in arr:
        print("".join(CHARS[min(9, int(v / 25.6))] for v in row))


def ascii_arr(arr, width=110, label=""):
    """ASCII render of a luminance array, downsampled to `width` cols."""
    h = max(1, int(arr.shape[0] / arr.shape[1] * width * 0.5))
    g = Image.fromarray(arr.astype(np.uint8)).resize((width, h), Image.BILINEAR)
    _print_ascii(np.asarray(g), label or f"array {arr.shape[1]}x{arr.shape[0]}")


def ascii_view(img, width=110, label="full"):
    """ASCII render of a PIL image."""
    g = img.convert("L").resize((width, int(img.height / img.width * width * 0.5)), Image.BILINEAR)
    _print_ascii(np.asarray(g), label)


def crop(arr, x0, y0, x1, y1, width=90):
    """Extract a crop of the luminance array and downsample it for ASCII."""
    sub = arr[y0:y1, x0:x1]
    h = max(1, int((y1 - y0) / (x1 - x0) * width * 0.5))
    g = Image.fromarray(sub.astype(np.uint8)).resize((width, h), Image.BILINEAR)
    return np.asarray(g)


def pixel_window(arr, x0, y0, x1, y1, step=2):
    """1:1 (pixel-level) ASCII of a small window, printing every `step`-th px."""
    sub = arr[y0:y1, x0:x1]
    print(f"--- 1:1 window x{x0}-{x1} y{y0}-{y1} (every {step}px) ---")
    for row in sub[::step]:
        print("".join(CHARS[min(9, int(v / 25.6))] for v in row[::step]))


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def ink_map(a, paper_lum=232.0):
    """paper minus luminance → where the ink landed (255 = full ink)."""
    return np.clip(paper_lum - a, 0, 255)


def ink_ascii(a, paper_lum=232.0, width=110, label="ink map"):
    """ASCII render of the ink map: '@' = heavy ink, ' ' = clean paper.
    Returns the downsampled ink array."""
    m = ink_map(a, paper_lum)
    h = max(1, int(m.shape[0] / m.shape[1] * width * 0.5))
    g = Image.fromarray(m.astype(np.uint8)).resize((width, h), Image.BILINEAR)
    _print_ascii(np.asarray(g), label)
    return np.asarray(g)


def ink_grid(a, n=8, paper_lum=232.0):
    """Mean ink per cell of an n x n grid — locates missing/extra ink.
    Prints a coarse ASCII density map and the numeric cell means; returns
    the n x n float32 array."""
    m = ink_map(a, paper_lum)
    H, W = m.shape
    grid = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(n):
            grid[i, j] = m[i * H // n:(i + 1) * H // n, j * W // n:(j + 1) * W // n].mean()
    print(f"--- ink grid {n}x{n} (mean ink per cell, 0=clean 255=full) ---")
    for row in grid:
        print("".join(CHARS[min(9, int(v / 25.6))] for v in row))
    print(" ".join(f"{v:4.0f}" for v in grid.ravel()))
    return grid


def histogram(a, bins=16, label="luminance histogram"):
    """Luminance histogram over 0..255: prints percent per bin and returns
    the raw counts."""
    counts, edges = np.histogram(a, bins=bins, range=(0, 256))
    total = max(1, int(a.size))
    print(f"--- {label} ({bins} bins of 0..255, percent per bin) ---")
    print(" ".join(f"{100.0 * c / total:4.1f}" for c in counts))
    return counts


def region_stats(a, x0, y0, x1, y1, name=""):
    """Mean / dark% / bright% of a rectangle; prints and returns the values."""
    sub = a[y0:y1, x0:x1]
    lum = float(sub.mean())
    dark = float((sub < 110).mean())
    bright = float((sub > 200).mean())
    print(f"  {name:22s} lum={lum:6.1f}  dark%={dark*100:5.1f}  bright%={bright*100:5.1f}")
    return lum, dark, bright


def luminance_profile(a, bins=32, axis=0):
    """Binned mean profile. axis=0 → vertical (top→bottom), axis=1 → horizontal."""
    n = a.shape[axis]
    idx = np.linspace(0, n, bins + 1).astype(int)
    prof = []
    for i in range(bins):
        sl = [slice(None), slice(None)]
        sl[axis] = slice(idx[i], idx[i + 1])
        prof.append(float(a[tuple(sl)].mean()))
    print("--- luminance profile (" + ("top->bottom" if axis == 0 else "left->right") + ") ---")
    print("".join(CHARS[min(9, int((255 - v) / 25.6))] for v in prof))
    print(" ".join(f"{v:3.0f}" for v in prof))
    return prof


def fft_regularity(a, label="region"):
    """Ratio of the strongest non-DC FFT peak: high values hint mechanical
    repetition or moiré. Prints and returns the ratio in percent."""
    f = np.fft.fftshift(np.fft.fft2(a - a.mean()))
    mag = np.abs(f)
    ch, cw = mag.shape[0] // 2, mag.shape[1] // 2
    mag[ch - 2:ch + 3, cw - 2:cw + 3] = 0
    peak = float(mag.max())
    dc = float(np.abs(f).sum())
    ratio = peak / dc * 100.0
    print(f"  FFT regularity peak ratio ({label})={ratio:.2f}%  (>0.35% hints mechanical repetition/moiré)")
    return ratio


def probe(a, points):
    """Print luminance at exact pixel coordinates. points = [(name, x, y), ...]"""
    for name, x, y in points:
        print(f"  {name:16s} at ({x},{y}): lum={a[int(y), int(x)]:.0f}")


# ----------------------------------------------------------------------------
# CLI — works on any image path
# ----------------------------------------------------------------------------
def main(path, fast=False):
    img, a = load(path)
    W, H = img.size
    print(f"plate {W}x{H}  mean lum={a.mean():.1f}  std={a.std():.1f}")
    print()
    ascii_view(img, 110 if not fast else 64, "full")
    print()
    luminance_profile(a, 32, axis=0)
    print()
    fft_regularity(a[100:900, 100:2100], "central band")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "out/v01.png",
         fast="--fast" in sys.argv)
