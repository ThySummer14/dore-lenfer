#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procedural engraving engine in the manner of Gustave Doré — Canto VIII.
Scene: "DANTE — L'ENFER · CHANT HUITIÈME · LA BARQUE DE PHLÉGYAS"
The Styx marsh: Phlegyas poles his skiff carrying Dante and Virgil; Filippo
Argenti claws at the gunwale; the wrathful fight each other in the mud; across
the water the city of Dis burns.

Reuses stroke primitives from plate.py (Canto I).
"""
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from plate import (S, ink_rgba, strokes_layer, poly_mask, band_mask, grad_mask,
                   draw_hatch, contour_strokes, fbm_arr, bilinear, clamp, smooth01,
                   make_paper, find_font)

CFG = dict(
    W=2200, H=2860,
    SS=2,
    seed=88,
    paper=(241, 232, 213),
    ink=(30, 23, 16),
    dis=(1700, 1360),           # the burning city
    frame=(0.055, 0.035, 0.945, 0.905),
    caption_top=0.775,
    out_dir="DORE_INFERNO/Chant_VIII_La_Barque_de_Phlegyas",
)


# ----------------------------------------------------------------------------
# tone field — dark marsh, the fires of Dis glow on the right
# ----------------------------------------------------------------------------
def make_darkmap8(cfg, ss_div=4):
    W, H = cfg["W"], cfg["H"]
    hw, hh = W // ss_div, H // ss_div
    cx, cy = cfg["dis"]
    y, x = np.mgrid[0:hh, 0:hw].astype(np.float32)
    x = x * ss_div
    y = y * ss_div
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    n = fbm_arr(hh, hw, cfg["seed"] + 12, 30, 4)
    base = 0.68 + 0.10 * n + 0.06 * smooth01((y - 1500.0) / 600.0)
    glow = 0.16 * np.exp(-((r / 480.0) ** 2))
    d = np.clip(base - glow, 0.0, 1.0)
    # keep the caption band clean paper
    d = d * (1.0 - 0.85 * smooth01((y - 2217.0) / 60.0))
    im = Image.fromarray((d * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0


def make_paper8(cfg, dark):
    W, H = cfg["W"], cfg["H"]
    paper = np.asarray(cfg["paper"], np.float32).reshape(1, 1, 3)
    shade = np.clip(1.0 - 0.24 * dark, 0.50, 1.02)
    img = paper * shade[..., None]
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / (W * 0.62)) ** 2 + ((y - H / 2) / (H * 0.62)) ** 2)
    vig = 1.0 - 0.06 * np.clip(r, 0, 1.4) ** 2.0
    img *= vig[..., None]
    grain = fbm_arr(H // 2, W // 2, cfg["seed"] + 11, 90, 3)
    grain = Image.fromarray((grain * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    g = (np.asarray(grain, np.float32) / 255.0 - 0.5) * 8.0
    img += g[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


# ----------------------------------------------------------------------------
# dark smoky sky
# ----------------------------------------------------------------------------
def layer_sky(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 51)
    m = band_mask(size, S(cfg, 100), S(cfg, 1520))
    depth = grad_mask(size, S(cfg, 100), S(cfg, 1520), 235, 140)
    m = ImageChops.multiply(m, depth)
    rect = [(S(cfg, 121), S(cfg, 100)), (S(cfg, 2079), S(cfg, 100)),
            (S(cfg, 2079), S(cfg, 1520)), (S(cfg, 121), S(cfg, 1520))]
    stack = [strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 1, 6.0, rng, jitter=2.4, width=3)),
             strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 92, 15, rng, jitter=6.0, width=1, dash=30))]
    # smoke plumes drifting from Dis
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    for k in range(9):
        x0 = rng.uniform(1400, 2000)
        y0 = rng.uniform(1150, 1300)
        pts = []
        for j in range(7):
            t = j / 6
            x = x0 - t * rng.uniform(180, 420) + math.sin(t * 9 + k) * 26
            y = y0 - t * rng.uniform(220, 420)
            pts.append((S(cfg, x), S(cfg, y)))
        d.line(pts, fill=ink_rgba(cfg, int(rng.uniform(60, 95))), width=3)
    # drifting cloud-slivers for texture
    for _ in range(4200):
        x = rng.uniform(130, 2070)
        y = rng.uniform(110, 1500)
        ln = rng.uniform(16, 60)
        a = int(clamp(120 - 60 * (y / 1500.0) + rng.uniform(-20, 20), 30, 130))
        d.line((S(cfg, x - ln), S(cfg, y + rng.uniform(-7, 7)),
                S(cfg, x + ln), S(cfg, y + rng.uniform(-7, 7))),
               fill=ink_rgba(cfg, a), width=2)
    stack.append(L)
    return stack


# ----------------------------------------------------------------------------
# the city of Dis — towers, lit windows, flames
# ----------------------------------------------------------------------------
def layer_dis(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 52)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]
    # wall
    wall = [(1300, 1520), (2080, 1520), (2080, 1460), (1300, 1460)]
    d.polygon([(S(cfg, x), S(cfg, y)) for x, y in wall], fill=ink_rgba(cfg, 232))
    d.polygon([(S(cfg, x), S(cfg, y)) for x, y in wall], outline=ink_rgba(cfg, 240), width=2)
    for x in range(1310, 2070, 34):
        d.rectangle((S(cfg, x), S(cfg, 1438), S(cfg, x + 20), S(cfg, 1460)),
                    fill=ink_rgba(cfg, 232))
        d.line((S(cfg, x + 10), S(cfg, 1442), S(cfg, x + 10), S(cfg, 1456)),
               fill=ink_rgba(cfg, 160), width=1)
    for y in range(1470, 1518, 13):
        d.line((S(cfg, 1304), S(cfg, y), S(cfg, 2076), S(cfg, y)),
               fill=ink_rgba(cfg, 110), width=1)
    # towers
    towers = [(1390, 1180, 96), (1600, 1130, 96), (1810, 1180, 96), (1970, 1150, 80)]
    for (tx, ty, tw) in towers:
        # fire glow behind the tower
        d.ellipse((S(cfg, tx - 30), S(cfg, ty - 60), S(cfg, tx + tw + 30), S(cfg, ty + 30)),
                  fill=paper + (45,))
        d.rectangle((S(cfg, tx), S(cfg, ty), S(cfg, tx + tw), S(cfg, 1460)),
                    fill=ink_rgba(cfg, 235))
        d.rectangle((S(cfg, tx), S(cfg, ty), S(cfg, tx + tw), S(cfg, 1460)),
                    outline=ink_rgba(cfg, 240), width=2)
        # rim light on the left edge (fire side)
        d.line((S(cfg, tx + 2), S(cfg, ty), S(cfg, tx + 2), S(cfg, 1460)),
               fill=paper + (130,), width=2)
        for x in range(tx + 6, tx + tw - 8, 22):
            d.rectangle((S(cfg, x), S(cfg, ty - 14), S(cfg, x + 14), S(cfg, ty)),
                        fill=ink_rgba(cfg, 235))
        # lit windows — the fires within
        for k in range(3):
            wx = tx + 14 + k * (tw // 3)
            wy = ty + 46 + k * 24
            d.rectangle((S(cfg, wx), S(cfg, wy), S(cfg, wx + 16), S(cfg, wy + 22)),
                        fill=paper + (225,))
            d.line((S(cfg, wx + 8), S(cfg, wy), S(cfg, wx + 8), S(cfg, wy + 22)),
                   fill=ink_rgba(cfg, 130), width=1)
        # flames licking above the tower
        for k in range(5):
            fx = tx + tw * (0.15 + 0.175 * k) + rng.uniform(-6, 6)
            fh = rng.uniform(40, 70)
            d.polygon([(S(cfg, fx - 8), S(cfg, ty - 12)), (S(cfg, fx + 8), S(cfg, ty - 12)),
                       (S(cfg, fx), S(cfg, ty - 12 - fh))], fill=paper + (195,))
            d.polygon([(S(cfg, fx - 3), S(cfg, ty - 12)), (S(cfg, fx + 3), S(cfg, ty - 12)),
                       (S(cfg, fx), S(cfg, ty - 12 - fh * 0.55))], fill=paper + (225,))
    return L


# ----------------------------------------------------------------------------
# the Styx water
# ----------------------------------------------------------------------------
def layer_water(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 53)
    m = band_mask(size, S(cfg, 1500), S(cfg, 2230))
    depth = grad_mask(size, S(cfg, 1500), S(cfg, 2230), 150, 255)
    m = ImageChops.multiply(m, depth)
    rect = [(S(cfg, 121), S(cfg, 1500)), (S(cfg, 2079), S(cfg, 1500)),
            (S(cfg, 2079), S(cfg, 2230)), (S(cfg, 121), S(cfg, 2230))]
    stack = [strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 2, 4.8, rng, jitter=2.2, width=3)),
             strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 90, 14, rng, jitter=5.0, width=1, dash=26))]
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    # wave crests
    for _ in range(130):
        x = rng.uniform(140, 2060)
        y = rng.uniform(1520, 2220)
        ln = rng.uniform(10, 30)
        d.arc((S(cfg, x - ln), S(cfg, y - 4), S(cfg, x + ln), S(cfg, y + 4)),
              200, 330, fill=ink_rgba(cfg, 110), width=2)
    # fire-light reflections toward Dis
    for _ in range(70):
        x = rng.uniform(1500, 2000)
        y = rng.uniform(1550, 2150)
        ln = rng.uniform(8, 26)
        d.line((S(cfg, x), S(cfg, y), S(cfg, x + ln), S(cfg, y + rng.uniform(-2, 2))),
               fill=cfg["paper"] + (int(rng.uniform(45, 90)),), width=2)
    stack.append(L)
    return stack


# ----------------------------------------------------------------------------
# the wrathful fighting in the mud
# ----------------------------------------------------------------------------
def layer_wrathful(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 54)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]
    ink = cfg["ink"]

    def fighter(x, y, s):
        # two entangled pale bodies
        d.arc((S(cfg, x - 30 * s), S(cfg, y - 24 * s), S(cfg, x + 30 * s), S(cfg, y + 16 * s)),
              40, 200, fill=paper + (175,), width=9)
        d.arc((S(cfg, x - 24 * s), S(cfg, y - 10 * s), S(cfg, x + 34 * s), S(cfg, y + 22 * s)),
              220, 380, fill=paper + (160,), width=8)
        d.ellipse((S(cfg, x + 14 * s), S(cfg, y - 26 * s), S(cfg, x + 30 * s), S(cfg, y - 12 * s)),
                  fill=paper + (185,))
        d.ellipse((S(cfg, x - 26 * s), S(cfg, y - 16 * s), S(cfg, x - 12 * s), S(cfg, y - 2 * s)),
                  fill=paper + (170,))
        # flailing arms
        d.line((S(cfg, x + 16 * s), S(cfg, y - 18 * s), S(cfg, x + 34 * s), S(cfg, y - 34 * s)),
               fill=paper + (165,), width=4)
        d.line((S(cfg, x - 18 * s), S(cfg, y - 10 * s), S(cfg, x - 34 * s), S(cfg, y - 26 * s)),
               fill=paper + (165,), width=4)
        # splash
        d.arc((S(cfg, x - 34 * s), S(cfg, y - 2 * s), S(cfg, x + 34 * s), S(cfg, y + 8 * s)),
              190, 350, fill=ink_rgba(cfg, 130), width=2)

    def rising(x, y, s):
        # a head and arm bursting from the water
        d.ellipse((S(cfg, x - 9 * s), S(cfg, y - 22 * s), S(cfg, x + 9 * s), S(cfg, y - 8 * s)),
                  fill=paper + (180,))
        d.line((S(cfg, x + 4 * s), S(cfg, y - 18 * s), S(cfg, x + 20 * s), S(cfg, y - 34 * s)),
               fill=paper + (170,), width=4)
        d.line((S(cfg, x - 6 * s), S(cfg, y - 14 * s), S(cfg, x - 16 * s), S(cfg, y - 30 * s)),
               fill=paper + (170,), width=4)
        d.arc((S(cfg, x - 16 * s), S(cfg, y - 4 * s), S(cfg, x + 16 * s), S(cfg, y + 6 * s)),
              190, 350, fill=ink_rgba(cfg, 120), width=2)

    fighter(560, 1990, 1.0)
    fighter(1350, 2100, 1.05)
    fighter(760, 2120, 0.9)
    fighter(1650, 1860, 0.95)
    fighter(400, 2150, 0.85)
    rising(1050, 2060, 1.0)
    rising(1500, 2000, 0.9)
    rising(300, 2050, 0.95)
    rising(1250, 2180, 0.85)
    rising(1800, 2160, 0.9)
    # Filippo Argenti — clawing at the boat's gunwale
    ax, ay = 905, 1908
    d.ellipse((S(cfg, ax - 10), S(cfg, ay - 26), S(cfg, ax + 10), S(cfg, ay - 10)),
              fill=paper + (190,))
    d.ellipse((S(cfg, ax - 6), S(cfg, ay - 22), S(cfg, ax + 6), S(cfg, ay - 16)),
              fill=ink_rgba(cfg, 200))
    d.line((S(cfg, ax + 4), S(cfg, ay - 20), S(cfg, ax + 16), S(cfg, ay - 40)),
           fill=paper + (180,), width=5)
    d.line((S(cfg, ax - 8), S(cfg, ay - 16), S(cfg, ax - 20), S(cfg, ay - 34)),
           fill=paper + (180,), width=5)
    d.arc((S(cfg, ax - 20), S(cfg, ay - 4), S(cfg, ax + 20), S(cfg, ay + 8)),
          190, 350, fill=ink_rgba(cfg, 140), width=2)
    return L


# ----------------------------------------------------------------------------
# the skiff: Phlegyas, Dante, Virgil
# ----------------------------------------------------------------------------
def layer_boat(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]
    ink = cfg["ink"]
    # hull
    d.polygon([(S(cfg, 830), S(cfg, 1902)), (S(cfg, 1148), S(cfg, 1902)),
               (S(cfg, 1120), S(cfg, 1956)), (S(cfg, 858), S(cfg, 1956))],
              fill=ink_rgba(cfg, 232))
    d.polygon([(S(cfg, 830), S(cfg, 1902)), (S(cfg, 1148), S(cfg, 1902)),
               (S(cfg, 1120), S(cfg, 1956)), (S(cfg, 858), S(cfg, 1956))],
              outline=ink_rgba(cfg, 245), width=3)
    d.line((S(cfg, 834), S(cfg, 1912), S(cfg, 1144), S(cfg, 1912)),
           fill=paper + (170,), width=3)
    for k in range(3):
        t = k / 2
        x = 900 + t * 210
        d.line((S(cfg, x), S(cfg, 1914), S(cfg, x - 8), S(cfg, 1952)),
               fill=ink_rgba(cfg, 150), width=2)
    # wake behind the boat
    for k in range(5):
        t = k / 4
        y = 1975 + t * 130
        d.arc((S(cfg, 700 - t * 60), S(cfg, y - 8), S(cfg, 1000 - t * 60), S(cfg, y + 8)),
              200, 340, fill=ink_rgba(cfg, 110), width=2)
    # ---- Phlegyas at the stern, poling ----
    px_, py_ = 1122, 1852
    d.line((S(cfg, px_), S(cfg, py_), S(cfg, px_ + 52), S(cfg, py_ + 6)),
           fill=ink_rgba(cfg, 240), width=9)
    d.line((S(cfg, px_), S(cfg, py_ - 26), S(cfg, px_ + 30), S(cfg, py_ - 52)),
           fill=ink_rgba(cfg, 235), width=7)
    d.ellipse((S(cfg, px_ - 12), S(cfg, py_ - 58), S(cfg, px_ + 12), S(cfg, py_ - 38)),
              fill=ink_rgba(cfg, 240))
    d.line((S(cfg, px_ + 22), S(cfg, py_ - 46), S(cfg, px_ + 46), S(cfg, py_ - 64)),
           fill=ink_rgba(cfg, 235), width=6)
    # the pole
    d.line((S(cfg, px_ + 34), S(cfg, py_ - 40), S(cfg, 1268), S(cfg, 2140)),
           fill=ink_rgba(cfg, 235), width=5)
    # ---- Dante seated amidships ----
    dx_, dy_ = 1010, 1890
    d.polygon([(S(cfg, dx_ - 13), S(cfg, dy_ + 8)), (S(cfg, dx_ + 12), S(cfg, dy_ + 8)),
               (S(cfg, dx_ + 9), S(cfg, dy_ - 26)), (S(cfg, dx_ + 4), S(cfg, dy_ - 44)),
               (S(cfg, dx_ - 6), S(cfg, dy_ - 44)), (S(cfg, dx_ - 10), S(cfg, dy_ - 26))],
              fill=ink_rgba(cfg, 235))
    d.ellipse((S(cfg, dx_ - 11), S(cfg, dy_ - 62), S(cfg, dx_ + 9), S(cfg, dy_ - 44)),
              fill=ink_rgba(cfg, 235))
    # ---- Virgil at the bow, pushing Argenti away ----
    vx, vy = 880, 1868
    d.polygon([(S(cfg, vx - 12), S(cfg, vy + 22)), (S(cfg, vx + 12), S(cfg, vy + 22)),
               (S(cfg, vx + 9), S(cfg, vy - 14)), (S(cfg, vx + 4), S(cfg, vy - 38)),
               (S(cfg, vx - 6), S(cfg, vy - 38)), (S(cfg, vx - 10), S(cfg, vy - 14))],
              fill=ink_rgba(cfg, 238))
    d.ellipse((S(cfg, vx - 10), S(cfg, vy - 56), S(cfg, vx + 8), S(cfg, vy - 40)),
              fill=ink_rgba(cfg, 238))
    d.line((S(cfg, vx + 4), S(cfg, vy - 26), S(cfg, vx + 24), S(cfg, vy + 8)),
           fill=ink_rgba(cfg, 235), width=6)
    d.line((S(cfg, vx + 24), S(cfg, vy + 8), S(cfg, vx + 36), S(cfg, vy + 26)),
           fill=ink_rgba(cfg, 235), width=5)
    # rim light from the fires of Dis
    d.arc((S(cfg, px_ - 10), S(cfg, py_ - 62), S(cfg, px_ + 10), S(cfg, py_ - 40)),
          200, 350, fill=paper + (255,), width=2)
    d.arc((S(cfg, vx - 10), S(cfg, vy - 60), S(cfg, vx + 8), S(cfg, vy - 40)),
          200, 350, fill=paper + (255,), width=2)
    d.arc((S(cfg, dx_ - 11), S(cfg, dy_ - 66), S(cfg, dx_ + 9), S(cfg, dy_ - 44)),
          200, 350, fill=paper + (255,), width=2)
    return L


# ----------------------------------------------------------------------------
# frame + caption (Canto VIII)
# ----------------------------------------------------------------------------
def draw_frame_and_caption8(img, cfg):
    W, H = cfg["W"], cfg["H"]
    d = ImageDraw.Draw(img, "RGBA")
    fx0, fy0, fx1, fy1 = [v * (W if i % 2 == 0 else H) for i, v in enumerate(cfg["frame"])]
    ink = cfg["ink"]
    paper = cfg["paper"]
    d.rectangle((fx0 - 3, fy0 - 3, fx1 - 3, fy1 - 3), outline=(255, 255, 255, 60), width=1)
    d.rectangle((fx0 + 3, fy0 + 3, fx1 + 3, fy1 + 3), outline=ink + (50,), width=1)
    d.rectangle((fx0, fy0, fx1, fy1), outline=ink + (255,), width=4)
    d.rectangle((fx0 + 16, fy0 + 16, fx1 - 16, fy1 - 16), outline=ink + (200,), width=2)
    for cx, cy in ((fx0 + 16, fy0 + 16), (fx1 - 16, fy0 + 16), (fx0 + 16, fy1 - 16), (fx1 - 16, fy1 - 16)):
        r = 13
        d.polygon([(cx, cy - r), (cx + r * 0.55, cy), (cx, cy + r), (cx - r * 0.55, cy)],
                  fill=ink + (255,))
        d.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=paper + (255,))
    ctop = cfg["caption_top"] * H
    cy_title = ctop + 0.035 * H
    cy_sub = ctop + 0.065 * H
    cy_cred = ctop + 0.100 * H
    cx = W / 2
    f_title = find_font(int(0.016 * H))
    f_sub = find_font(int(0.0112 * H))
    f_cred = find_font(int(0.0098 * H))
    d.text((cx, cy_title), "DANTE · L'ENFER", font=f_title, fill=ink + (255,), anchor="mm")
    d.text((cx, cy_sub), "CHANT HUITIÈME — LA BARQUE DE PHLÉGYAS", font=f_sub, fill=ink + (230,), anchor="mm")
    d.text((fx0 + 40, cy_cred), "G. Doré inv. & sculp.", font=f_cred, fill=ink + (220,), anchor="lm")
    d.text((fx1 - 40, cy_cred), "PARIS · M DCCC LXI", font=f_cred, fill=ink + (220,), anchor="rm")
    y = (cy_title + cy_sub) / 2
    r = 10
    d.line((cx - 240, y, cx - 70, y), fill=ink + (200,), width=2)
    d.line((cx + 70, y, cx + 240, y), fill=ink + (200,), width=2)
    d.polygon([(cx, y - r), (cx + r * 0.6, y), (cx, y + r), (cx - r * 0.6, y)], fill=ink + (255,))
    return img


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def render(cfg, out_name):
    W, H = cfg["W"], cfg["H"]
    dark = make_darkmap8(cfg)
    paper = make_paper8(cfg, dark).convert("RGBA")
    size = (S(cfg, W), S(cfg, H))

    layers = []
    print("  sky ...")
    layers += layer_sky(cfg)
    print("  dis ...")
    layers.append(layer_dis(cfg))
    print("  water ...")
    layers += layer_water(cfg)
    print("  wrathful ...")
    layers.append(layer_wrathful(cfg))
    print("  boat ...")
    layers.append(layer_boat(cfg))

    print("  compositing ...")
    ink = Image.new("RGBA", size, (0, 0, 0, 0))
    for L in layers:
        ink = Image.alpha_composite(ink, L)
    ink_final = ink.resize((W, H), Image.LANCZOS)
    img = Image.alpha_composite(paper, ink_final)
    img = draw_frame_and_caption8(img, cfg)
    img = img.convert("RGB")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["out_dir"], "versions"), exist_ok=True)
    path = os.path.join(cfg["out_dir"], out_name)
    img.save(path, "PNG")
    prev = img.copy()
    prev.thumbnail((1000, 1000), Image.LANCZOS)
    prev.save(os.path.join(cfg["out_dir"], "versions", "preview_" + out_name.replace(".png", ".jpg")),
              "JPEG", quality=92)
    print(f"  saved {path}")
    return img


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "c8_v01.png"
    render(CFG, name)
