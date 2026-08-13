#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procedural engraving engine in the manner of Gustave Doré — Canto IX.
Scene: "DANTE — L'ENFER · CHANT NEUVIÈME · L'ANGE OUVRE LES PORTES DE DIT"
The angel of heaven strides over the Styx and opens the iron gate of Dis with
a wand; a blade of light splits the doors; the fog parts; demons flee; the
Furies rage on the wall; Virgil shields Dante's eyes.

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
    seed=99,
    paper=(241, 232, 213),
    ink=(30, 23, 16),
    gate=(1100, 1400),          # centre of the gate crack
    angel=(700, 1950),          # the angel striding on the water
    frame=(0.055, 0.035, 0.945, 0.905),
    caption_top=0.775,
    out_dir="DORE_INFERNO/Chant_IX_L_Ange_ouvre_les_Portes",
)


# ----------------------------------------------------------------------------
# tone field — gloom + the corridor of light from angel to gate
# ----------------------------------------------------------------------------
def make_darkmap9(cfg, ss_div=4):
    W, H = cfg["W"], cfg["H"]
    hw, hh = W // ss_div, H // ss_div
    gx, gy = cfg["gate"]
    ax, ay = cfg["angel"]
    y, x = np.mgrid[0:hh, 0:hw].astype(np.float32)
    x = x * ss_div
    y = y * ss_div
    n = fbm_arr(hh, hw, cfg["seed"] + 14, 30, 4)
    base = 0.68 + 0.10 * n + 0.05 * smooth01((y - 1600.0) / 500.0)
    # distance to the angel-gate segment (the corridor of light)
    dx, dy = gx - ax, gy - ay
    t = np.clip(((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy), 0, 1)
    px, py = ax + t * dx, ay + t * dy
    seg = np.sqrt((x - px) ** 2 + (y - py) ** 2)
    corridor = 0.20 * np.exp(-((seg / 160.0) ** 2))
    gate_glow = 0.22 * np.exp(-((np.sqrt((x - gx) ** 2 + (y - gy) ** 2) / 260.0) ** 2))
    angel_glow = 0.12 * np.exp(-((np.sqrt((x - ax) ** 2 + (y - ay) ** 2) / 220.0) ** 2))
    d = np.clip(base - corridor - gate_glow - angel_glow, 0.0, 1.0)
    # keep the caption band clean paper
    d = d * (1.0 - 0.85 * smooth01((y - 2217.0) / 60.0))
    im = Image.fromarray((d * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0


def make_paper9(cfg, dark):
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
# dark sky + parting fog banks
# ----------------------------------------------------------------------------
def layer_sky_fog(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 61)
    m = band_mask(size, S(cfg, 100), S(cfg, 2217))
    depth = grad_mask(size, S(cfg, 100), S(cfg, 2217), 235, 150)
    m = ImageChops.multiply(m, depth)
    rect = [(S(cfg, 121), S(cfg, 100)), (S(cfg, 2079), S(cfg, 100)),
            (S(cfg, 2079), S(cfg, 2217)), (S(cfg, 121), S(cfg, 2217))]
    stack = [strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 1, 6.2, rng, jitter=2.4, width=3)),
             strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 92, 16, rng, jitter=6.0, width=1, dash=30))]
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    # cloud slivers
    for _ in range(3600):
        x = rng.uniform(130, 2070)
        y = rng.uniform(110, 2220)
        ln = rng.uniform(16, 56)
        a = int(clamp(120 - 50 * (y / 2240.0) + rng.uniform(-18, 18), 25, 125))
        d.line((S(cfg, x - ln), S(cfg, y + rng.uniform(-6, 6)),
                S(cfg, x + ln), S(cfg, y + rng.uniform(-6, 6))),
               fill=ink_rgba(cfg, a), width=2)
    # dense fog banks in the two lower corners, parted by the light
    for cx0, flip in ((350, -1), (1850, 1)):
        for _ in range(420):
            x = cx0 + flip * rng.uniform(60, 420)
            y = rng.uniform(1750, 2220)
            ln = rng.uniform(30, 90)
            a = int(rng.uniform(60, 110))
            d.line((S(cfg, x - ln), S(cfg, y + rng.uniform(-8, 8)),
                    S(cfg, x + ln), S(cfg, y + rng.uniform(-8, 8))),
                   fill=ink_rgba(cfg, a), width=2)
    stack.append(L)
    return stack


# ----------------------------------------------------------------------------
# the water, the corridor of light on it
# ----------------------------------------------------------------------------
def layer_water(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 62)
    m = band_mask(size, S(cfg, 1620), S(cfg, 2230))
    depth = grad_mask(size, S(cfg, 1620), S(cfg, 2230), 150, 255)
    m = ImageChops.multiply(m, depth)
    rect = [(S(cfg, 121), S(cfg, 1620)), (S(cfg, 2079), S(cfg, 1620)),
            (S(cfg, 2079), S(cfg, 2230)), (S(cfg, 121), S(cfg, 2230))]
    stack = [strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 2, 4.8, rng, jitter=2.2, width=3))]
    # the corridor of light from the angel to the gate
    poly = [(620, 1985), (780, 1985), (1120, 1640), (1040, 1640)]
    mp = poly_mask(size, [(S(cfg, x), S(cfg, y)) for x, y in poly])
    er = Image.new("RGBA", size, (0, 0, 0, 0))
    er.paste(cfg["paper"] + (110,), (0, 0), mp)
    stack.append(er)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    # ripples along the corridor
    for _ in range(60):
        x = rng.uniform(650, 1100)
        y = rng.uniform(1660, 1970)
        ln = rng.uniform(8, 22)
        d.arc((S(cfg, x - ln), S(cfg, y - 3), S(cfg, x + ln), S(cfg, y + 3)),
              200, 330, fill=ink_rgba(cfg, 90), width=2)
    # wave crests elsewhere
    for _ in range(80):
        x = rng.uniform(140, 2060)
        y = rng.uniform(1640, 2220)
        ln = rng.uniform(10, 28)
        d.arc((S(cfg, x - ln), S(cfg, y - 4), S(cfg, x + ln), S(cfg, y + 4)),
              200, 330, fill=ink_rgba(cfg, 110), width=2)
    stack.append(L)
    return stack


# ----------------------------------------------------------------------------
# the wall of Dis, the towers, the iron gate split by light
# ----------------------------------------------------------------------------
def layer_wall_gate(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 63)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]
    # wall mass — pale stone against the gloom
    wall = [(700, 1660), (1500, 1660), (1500, 950), (700, 950)]
    d.polygon([(S(cfg, x), S(cfg, y)) for x, y in wall], fill=paper + (75,))
    d.polygon([(S(cfg, x), S(cfg, y)) for x, y in wall], outline=ink_rgba(cfg, 240), width=3)
    for y in range(966, 1656, 14):
        d.line((S(cfg, 704), S(cfg, y), S(cfg, 1496), S(cfg, y)),
               fill=ink_rgba(cfg, 130), width=1)
    for y in range(966, 1656, 14):
        for x in range(720, 1490, 26):
            off = 13 if (y // 14) % 2 else 0
            d.line((S(cfg, x + off), S(cfg, y - 7), S(cfg, x + off), S(cfg, y + 7)),
                   fill=ink_rgba(cfg, 100), width=1)
    # battlements on top
    for x in range(706, 1490, 34):
        d.rectangle((S(cfg, x), S(cfg, 928), S(cfg, x + 22), S(cfg, 950)),
                    fill=paper + (75,), outline=ink_rgba(cfg, 200), width=1)
    # towers
    for (tx, ty, tw) in ((760, 860, 110), (1330, 860, 110)):
        d.ellipse((S(cfg, tx - 24), S(cfg, ty - 50), S(cfg, tx + tw + 24), S(cfg, ty + 24)),
                  fill=paper + (40,))
        d.rectangle((S(cfg, tx), S(cfg, ty), S(cfg, tx + tw), S(cfg, 950)),
                    fill=paper + (60,))
        d.rectangle((S(cfg, tx), S(cfg, ty), S(cfg, tx + tw), S(cfg, 950)),
                    outline=ink_rgba(cfg, 240), width=2)
        d.line((S(cfg, tx + 2), S(cfg, ty), S(cfg, tx + 2), S(cfg, 950)),
               fill=paper + (160,), width=2)
        for x in range(tx + 8, tx + tw - 10, 24):
            d.rectangle((S(cfg, x), S(cfg, ty - 16), S(cfg, x + 16), S(cfg, ty)),
                        fill=paper + (60,), outline=ink_rgba(cfg, 200), width=1)
        for k in range(3):
            wx = tx + 16 + k * (tw // 3)
            wy = ty + 60 + k * 30
            d.rectangle((S(cfg, wx), S(cfg, wy), S(cfg, wx + 14), S(cfg, wy + 20)),
                        fill=paper + (215,))
        for k in range(3):
            fx = tx + tw * (0.25 + 0.25 * k) + rng.uniform(-5, 5)
            fh = rng.uniform(30, 52)
            d.polygon([(S(cfg, fx - 6), S(cfg, ty - 14)), (S(cfg, fx + 6), S(cfg, ty - 14)),
                       (S(cfg, fx), S(cfg, ty - 14 - fh))], fill=paper + (190,))
    # the great iron gate
    gate_arch = [(990, 1600), (990, 1280), (1000, 1210), (1020, 1160), (1050, 1120),
                 (1100, 1098), (1150, 1120), (1180, 1160), (1200, 1210), (1210, 1280),
                 (1210, 1600)]
    d.polygon([(S(cfg, x), S(cfg, y)) for x, y in gate_arch], fill=ink_rgba(cfg, 238))
    d.polygon([(S(cfg, x), S(cfg, y)) for x, y in gate_arch], outline=paper + (110,), width=2)
    # iron bands + studs
    for y in (1250, 1360, 1470):
        d.line((S(cfg, 994), S(cfg, y), S(cfg, 1206), S(cfg, y)),
               fill=ink_rgba(cfg, 175), width=6)
        for x in range(1010, 1200, 38):
            d.ellipse((S(cfg, x - 3), S(cfg, y - 3), S(cfg, x + 3), S(cfg, y + 3)),
                      fill=paper + (150,))
    # the central seam, split by a blade of light
    d.polygon([(S(cfg, 1094), S(cfg, 1102)), (S(cfg, 1106), S(cfg, 1102)),
               (S(cfg, 1110), S(cfg, 1240)), (S(cfg, 1114), S(cfg, 1600)),
               (S(cfg, 1096), S(cfg, 1600)), (S(cfg, 1090), S(cfg, 1240))],
              fill=paper + (235,))
    d.ellipse((S(cfg, 1030), S(cfg, 1220), S(cfg, 1170), S(cfg, 1400)),
              fill=paper + (115,))
    return L


# ----------------------------------------------------------------------------
# the Furies on the wall
# ----------------------------------------------------------------------------
def layer_furies(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 64)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    for k, (x, y) in enumerate(((985, 990), (1100, 980), (1215, 990))):
        s = 1.0
        d.line((S(cfg, x), S(cfg, y), S(cfg, x), S(cfg, y - 26)),
               fill=ink_rgba(cfg, 235), width=7)
        d.ellipse((S(cfg, x - 8), S(cfg, y - 40), S(cfg, x + 8), S(cfg, y - 26)),
                  fill=ink_rgba(cfg, 235))
        d.line((S(cfg, x - 4), S(cfg, y - 30), S(cfg, x - 16), S(cfg, y - 46)),
               fill=ink_rgba(cfg, 230), width=4)
        d.line((S(cfg, x + 4), S(cfg, y - 30), S(cfg, x + 16), S(cfg, y - 46)),
               fill=ink_rgba(cfg, 230), width=4)
        # snaky hair
        for hk in range(3):
            d.line((S(cfg, x - 6 + hk * 6), S(cfg, y - 38),
                    S(cfg, x - 8 + hk * 6), S(cfg, y - 48 + rng.uniform(-4, 4))),
                   fill=ink_rgba(cfg, 190), width=1)
        # rim light from the gate crack
        d.line((S(cfg, x + 7), S(cfg, y - 38), S(cfg, x + 7), S(cfg, y - 4)),
               fill=cfg["paper"] + (140,), width=1)
    return L


# ----------------------------------------------------------------------------
# the angel — striding on the water, wand raised
# ----------------------------------------------------------------------------
def layer_angel(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]
    x, y = 700, 1950
    # ripples under his feet
    for r0, a in ((58, 110), (44, 130)):
        d.arc((S(cfg, x - r0), S(cfg, y - 5), S(cfg, x + r0), S(cfg, y + 7)),
              190, 350, fill=ink_rgba(cfg, a), width=2)
    # flowing robe, swept back by the stride
    d.polygon([(S(cfg, x - 52), S(cfg, y)), (S(cfg, x + 22), S(cfg, y)),
               (S(cfg, x + 28), S(cfg, y - 84)), (S(cfg, x + 20), S(cfg, y - 152)),
               (S(cfg, x - 4), S(cfg, y - 196)), (S(cfg, x - 26), S(cfg, y - 168)),
               (S(cfg, x - 44), S(cfg, y - 96))],
              fill=paper + (225,))
    for ox in (-30, -12, 6):
        d.line((S(cfg, x + ox), S(cfg, y - 150), S(cfg, x + ox * 0.6), S(cfg, y - 8)),
               fill=ink_rgba(cfg, 130), width=2)
    # head
    d.ellipse((S(cfg, x - 2), S(cfg, y - 224), S(cfg, x + 20), S(cfg, y - 202)),
              fill=paper + (225,))
    # halo
    d.ellipse((S(cfg, x - 22), S(cfg, y - 236), S(cfg, x + 38), S(cfg, y - 190)),
              outline=paper + (200,), width=3)
    for k in range(9):
        t = -0.6 + k * 0.35
        d.line((S(cfg, x + 8 + math.cos(t) * 30), S(cfg, y - 213 + math.sin(t) * 23),
                S(cfg, x + 8 + math.cos(t) * 42), S(cfg, y - 213 + math.sin(t) * 32)),
               fill=paper + (170,), width=2)
    # wings
    for k, ln in ((0, 96), (1, 78), (2, 58)):
        d.arc((S(cfg, x - 34 - k * 6), S(cfg, y - 250 + k * 10),
               S(cfg, x + 60 - k * 6), S(cfg, y - 60 + k * 10)),
              200, 330, fill=paper + (190 - k * 35,), width=4)
    # raised arm with the wand
    d.line((S(cfg, x + 14), S(cfg, y - 170), S(cfg, x + 46), S(cfg, y - 226)),
           fill=paper + (220,), width=7)
    d.line((S(cfg, x + 46), S(cfg, y - 226), S(cfg, x + 68), S(cfg, y - 248)),
           fill=ink_rgba(cfg, 200), width=3)
    d.line((S(cfg, x + 68), S(cfg, y - 248), S(cfg, x + 76), S(cfg, y - 256)),
           fill=ink_rgba(cfg, 200), width=2)
    d.ellipse((S(cfg, x + 76), S(cfg, y - 262), S(cfg, x + 84), S(cfg, y - 254)),
              fill=paper + (230,))
    # stride legs
    d.line((S(cfg, x - 10), S(cfg, y), S(cfg, x - 18), S(cfg, y + 6)),
           fill=paper + (215,), width=6)
    d.line((S(cfg, x + 8), S(cfg, y), S(cfg, x + 20), S(cfg, y + 6)),
           fill=paper + (215,), width=6)
    # rays toward the gate
    for k in range(9):
        t = -0.35 + k * 0.09
        x0 = x + 60
        y0 = y - 180
        ang = math.atan2(1400 - y0, 1100 - x0) + t
        d.line((S(cfg, x0), S(cfg, y0),
                S(cfg, x0 + math.cos(ang) * 480), S(cfg, y0 + math.sin(ang) * 480)),
               fill=ink_rgba(cfg, 14), width=2)
    return L


# ----------------------------------------------------------------------------
# demons fleeing, and the poets at the edge
# ----------------------------------------------------------------------------
def layer_demons(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 65)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    for (x, y, s, dirn) in ((480, 2000, 1.0, -1), (530, 2070, 0.85, -1),
                            (1630, 1990, 1.0, 1), (1670, 2070, 0.85, 1), (1500, 2130, 0.9, 1)):
        # hunched fleeing figure
        d.polygon([(S(cfg, x - 14 * s), S(cfg, y)), (S(cfg, x + 14 * s), S(cfg, y)),
                   (S(cfg, x + 10 * s), S(cfg, y - 30 * s)), (S(cfg, x + 2 * s), S(cfg, y - 48 * s)),
                   (S(cfg, x - 10 * s), S(cfg, y - 40 * s)), (S(cfg, x - 16 * s), S(cfg, y - 16 * s))],
                  fill=ink_rgba(cfg, 238))
        d.ellipse((S(cfg, x - 12 * s), S(cfg, y - 62 * s), S(cfg, x + 6 * s), S(cfg, y - 46 * s)),
                  fill=ink_rgba(cfg, 238))
        d.line((S(cfg, x + 2 * s), S(cfg, y - 50 * s), S(cfg, x + 18 * s + dirn * 6), S(cfg, y - 62 * s)),
               fill=ink_rgba(cfg, 232), width=5)
        d.line((S(cfg, x - 10 * s), S(cfg, y - 44 * s), S(cfg, x - 22 * s + dirn * 6), S(cfg, y - 56 * s)),
               fill=ink_rgba(cfg, 232), width=5)
        # splashes
        d.arc((S(cfg, x - 20 * s), S(cfg, y - 3), S(cfg, x + 20 * s), S(cfg, y + 7)),
              190, 350, fill=ink_rgba(cfg, 120), width=2)
    return L


def layer_poets(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]
    # Virgil shielding Dante's eyes
    vx, vy = S(cfg, 330), S(cfg, 2145)
    s = S(cfg, 0.95)
    d.polygon([(vx - 14 * s, vy), (vx + 14 * s, vy),
               (vx + 16 * s, vy - 46 * s), (vx + 10 * s, vy - 84 * s),
               (vx + 3 * s, vy - 104 * s), (vx - 7 * s, vy - 104 * s),
               (vx - 11 * s, vy - 84 * s), (vx - 15 * s, vy - 44 * s)],
              fill=ink_rgba(cfg, 238))
    d.ellipse((vx - 9 * s, vy - 122 * s, vx + 8 * s, vy - 104 * s),
              fill=ink_rgba(cfg, 238))
    # his hand over Dante's eyes
    d.line((vx + 6 * s, vy - 88 * s, vx + 22 * s, vy - 66 * s),
           fill=ink_rgba(cfg, 235), width=6)
    # Dante cowering, hands to his face
    dx_, dy_ = S(cfg, 452), S(cfg, 2125)
    ds = S(cfg, 0.85)
    d.polygon([(dx_ - 12 * ds, dy_), (dx_ + 11 * ds, dy_),
               (dx_ + 14 * ds, dy_ - 42 * ds), (dx_ + 8 * ds, dy_ - 76 * ds),
               (dx_ + 2 * ds, dy_ - 96 * ds), (dx_ - 7 * ds, dy_ - 96 * ds),
               (dx_ - 10 * ds, dy_ - 76 * ds), (dx_ - 13 * ds, dy_ - 40 * ds)],
              fill=ink_rgba(cfg, 236))
    d.ellipse((dx_ - 10 * ds, dy_ - 116 * ds, dx_ + 7 * ds, dy_ - 96 * ds),
              fill=ink_rgba(cfg, 236))
    d.line((dx_ + 2 * ds, dy_ - 84 * ds, dx_ - 6 * ds, dy_ - 100 * ds),
           fill=ink_rgba(cfg, 230), width=5)
    d.line((dx_ - 6 * ds, dy_ - 84 * ds, dx_ - 14 * ds, dy_ - 98 * ds),
           fill=ink_rgba(cfg, 230), width=5)
    d.arc((vx - 10 * s, vy - 126 * s, vx + 9 * s, vy - 102 * s), 200, 350,
          fill=paper + (255,), width=2)
    d.arc((dx_ - 11 * ds, dy_ - 120 * ds, dx_ + 8 * ds, dy_ - 94 * ds), 200, 350,
          fill=paper + (255,), width=2)
    return L


# ----------------------------------------------------------------------------
# frame + caption (Canto IX)
# ----------------------------------------------------------------------------
def draw_frame_and_caption9(img, cfg):
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
    d.text((cx, cy_sub), "CHANT NEUVIÈME — L'ANGE OUVRE LES PORTES DE DIT", font=f_sub, fill=ink + (230,), anchor="mm")
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
    dark = make_darkmap9(cfg)
    paper = make_paper9(cfg, dark).convert("RGBA")
    size = (S(cfg, W), S(cfg, H))

    layers = []
    print("  sky + fog ...")
    layers += layer_sky_fog(cfg)
    print("  water ...")
    layers += layer_water(cfg)
    print("  wall + gate ...")
    layers.append(layer_wall_gate(cfg))
    print("  furies ...")
    layers.append(layer_furies(cfg))
    print("  angel ...")
    layers.append(layer_angel(cfg))
    print("  demons ...")
    layers.append(layer_demons(cfg))
    print("  poets ...")
    layers.append(layer_poets(cfg))

    print("  compositing ...")
    ink = Image.new("RGBA", size, (0, 0, 0, 0))
    for L in layers:
        ink = Image.alpha_composite(ink, L)
    ink_final = ink.resize((W, H), Image.LANCZOS)
    img = Image.alpha_composite(paper, ink_final)
    img = draw_frame_and_caption9(img, cfg)
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
    name = sys.argv[1] if len(sys.argv) > 1 else "c9_v01.png"
    render(CFG, name)
