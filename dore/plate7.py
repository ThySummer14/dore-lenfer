#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procedural engraving engine in the manner of Gustave Doré — Canto VII.
Scene: "DANTE — L'ENFER · CHANT SEPTIÈME · LES AVARES ET LES PRODIGUES"
The fourth circle: two crowds of pale figures pushing huge weights in opposite
semicircles, colliding at the middle; Plutus raging on his rock ledge; the
endless line of pushers on a distant shelf; Dante and Virgil watching.

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
    seed=77,
    paper=(241, 232, 213),
    ink=(30, 23, 16),
    frame=(0.055, 0.035, 0.945, 0.905),
    caption_top=0.775,
    out_dir="DORE_INFERNO/Chant_VII_Les_Avares",
)


# ----------------------------------------------------------------------------
# tone field — dark rocky arena
# ----------------------------------------------------------------------------
def make_darkmap7(cfg, ss_div=4):
    W, H = cfg["W"], cfg["H"]
    hw, hh = W // ss_div, H // ss_div
    y, x = np.mgrid[0:hh, 0:hw].astype(np.float32)
    x = x * ss_div
    y = y * ss_div
    n = fbm_arr(hh, hw, cfg["seed"] + 10, 30, 4)
    base = 0.70 + 0.10 * n + 0.08 * smooth01((y - 1600.0) / 500.0)
    # dim glow over the arena centre
    light = 0.08 * np.exp(-(((x - 1100) / 900.0) ** 2 + ((y - 1500) / 800.0) ** 2))
    d = np.clip(base - light, 0.0, 1.0)
    # keep the caption band clean paper
    d = d * (1.0 - 0.85 * smooth01((y - 2217.0) / 60.0))
    im = Image.fromarray((d * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0


# ----------------------------------------------------------------------------
# darker paper for the dark arena (stronger shade than the default make_paper)
# ----------------------------------------------------------------------------
def make_paper7(cfg, dark):
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
# rocky walls, the shelf, the distant endless line
# ----------------------------------------------------------------------------
def layer_walls(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 41)
    left = [(121, 100), (560, 100), (620, 300), (700, 700), (720, 1100),
            (700, 1500), (640, 1680), (121, 1680)]
    right = [(2079, 100), (1640, 100), (1580, 300), (1500, 700), (1480, 1100),
             (1500, 1500), (1560, 1680), (2079, 1680)]
    centre = [(640, 1370), (700, 1680), (1500, 1680), (1560, 1370)]
    m = Image.new("L", size, 0)
    md = ImageDraw.Draw(m)
    for poly in (left, right):
        md.polygon([(S(cfg, x), S(cfg, y)) for x, y in poly], fill=255)
    mc = Image.new("L", size, 0)
    md2 = ImageDraw.Draw(mc)
    md2.polygon([(S(cfg, x), S(cfg, y)) for x, y in centre], fill=255)
    depth = Image.fromarray(
        np.repeat((np.linspace(0.80, 0.95, size[1]) * 255).astype(np.uint8)[:, None],
                  size[0], axis=1))
    depthc = Image.fromarray(
        np.repeat((np.linspace(0.94, 1.0, size[1]) * 255).astype(np.uint8)[:, None],
                  size[0], axis=1))
    m = ImageChops.multiply(m, depth)
    mc = ImageChops.multiply(mc, depthc)
    stack = []
    for poly in (left, right):
        stack.append(strokes_layer(size, cfg["ink"], m, lambda d, p=poly:
                     draw_hatch(d, [(S(cfg, x), S(cfg, y)) for x, y in p], 5, 6.2, rng, jitter=2.4, width=4)))
        stack.append(strokes_layer(size, cfg["ink"], m, lambda d, p=poly:
                     draw_hatch(d, [(S(cfg, x), S(cfg, y)) for x, y in p], 96, 10, rng, jitter=4.0, width=2, dash=24)))
    stack.append(strokes_layer(size, cfg["ink"], mc, lambda d, p=centre:
                 draw_hatch(d, [(S(cfg, x), S(cfg, y)) for x, y in p], 5, 7.4, rng, jitter=2.4, width=4)))
    stack.append(strokes_layer(size, cfg["ink"], mc, lambda d, p=centre:
                 draw_hatch(d, [(S(cfg, x), S(cfg, y)) for x, y in p], 96, 11, rng, jitter=4.0, width=2, dash=26)))
    # strata
    sm = Image.new("L", size, 0)
    sd = ImageDraw.Draw(sm)
    for k in range(24):
        y = S(cfg, 140 + k * 64 + rng.uniform(-14, 14))
        amp = rng.uniform(4, 12)
        wav = rng.uniform(0.008, 0.016)
        pts = [(x, y + amp * math.sin(x * wav)) for x in range(S(cfg, 130), S(cfg, 2070), 30)]
        sd.line(pts, fill=255, width=2)
    sm = ImageChops.multiply(sm, ImageChops.lighter(m, mc))
    s3 = Image.new("RGBA", size, (0, 0, 0, 0))
    s3.paste(cfg["ink"] + (150,), (0, 0), sm)
    stack.append(s3)
    # the shelf with the endless line of distant pushers
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    d.polygon([(S(cfg, 700), S(cfg, 1340)), (S(cfg, 1500), S(cfg, 1340)),
               (S(cfg, 1500), S(cfg, 1370)), (S(cfg, 700), S(cfg, 1370))],
              fill=ink_rgba(cfg, 225))
    d.line((S(cfg, 700), S(cfg, 1336), S(cfg, 1500), S(cfg, 1336)),
           fill=ink_rgba(cfg, 200), width=3)
    for k in range(9):
        x = 780 + k * 82
        # tiny pusher silhouette
        d.line((S(cfg, x), S(cfg, 1330), S(cfg, x - 4), S(cfg, 1312)),
               fill=ink_rgba(cfg, 150), width=3)
        d.line((S(cfg, x - 4), S(cfg, 1312), S(cfg, x - 12), S(cfg, 1306)),
               fill=ink_rgba(cfg, 140), width=2)
        d.ellipse((S(cfg, x - 7), S(cfg, 1304), S(cfg, x - 1), S(cfg, 1310)),
                  fill=ink_rgba(cfg, 150))
        d.line((S(cfg, x - 14), S(cfg, 1328), S(cfg, x - 18), S(cfg, 1330)),
               fill=ink_rgba(cfg, 140), width=2)
        # tiny weight
        d.ellipse((S(cfg, x + 6), S(cfg, 1318), S(cfg, x + 22), S(cfg, 1332)),
                  outline=ink_rgba(cfg, 150), width=2)
    stack.append(L)
    # rim light on the inner edges of the walls
    R = Image.new("RGBA", size, (0, 0, 0, 0))
    rd = ImageDraw.Draw(R)
    for poly in (left, right):
        inner = [(x - 2, y - 2) for x, y in poly[2:6]]
        rd.line([(S(cfg, x), S(cfg, y)) for x, y in inner],
                fill=cfg["paper"] + (120,), width=2)
    stack.append(R)
    return stack


# ----------------------------------------------------------------------------
# arena floor
# ----------------------------------------------------------------------------
def layer_floor(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 42)
    m = band_mask(size, S(cfg, 1660), S(cfg, 2230))
    depth = grad_mask(size, S(cfg, 1660), S(cfg, 2230), 130, 255)
    m = ImageChops.multiply(m, depth)
    rect = [(S(cfg, 121), S(cfg, 1660)), (S(cfg, 2079), S(cfg, 1660)),
            (S(cfg, 2079), S(cfg, 2230)), (S(cfg, 121), S(cfg, 2230))]
    stack = [strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 2, 4.8, rng, jitter=2.2, width=4))]
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    for _ in range(24):
        x = rng.uniform(140, 2060)
        y = rng.uniform(1700, 2220)
        r = rng.uniform(6, 18)
        d.ellipse((S(cfg, x - r), S(cfg, y - r * 0.45), S(cfg, x + r), S(cfg, y + r * 0.45)),
                  outline=ink_rgba(cfg, 120), width=2)
    stack.append(L)
    return stack


# ----------------------------------------------------------------------------
# weights — huge bags and boulders
# ----------------------------------------------------------------------------
def draw_bag(d, cfg, x, y, s, rng):
    poly = [(x - 45 * s, y - 26 * s), (x + 45 * s, y - 26 * s), (x + 50 * s, y + 6 * s),
            (x + 34 * s, y + 34 * s), (x - 34 * s, y + 34 * s), (x - 50 * s, y + 6 * s)]
    d.polygon([(S(cfg, px), S(cfg, py)) for px, py in poly], fill=ink_rgba(cfg, 235))
    d.polygon([(S(cfg, px), S(cfg, py)) for px, py in poly], outline=cfg["paper"] + (150,), width=2)
    # tied neck
    d.polygon([(S(cfg, x - 12 * s), S(cfg, y - 26 * s)), (S(cfg, x + 12 * s), S(cfg, y - 26 * s)),
               (S(cfg, x + 8 * s), S(cfg, y - 40 * s)), (S(cfg, x - 8 * s), S(cfg, y - 40 * s))],
              fill=ink_rgba(cfg, 235))
    d.arc((S(cfg, x - 34 * s), S(cfg, y - 14 * s), S(cfg, x + 8 * s), S(cfg, y + 22 * s)),
          120, 260, fill=cfg["paper"] + (150,), width=3)


def draw_boulder(d, cfg, x, y, s, rng):
    pts = []
    for k in range(8):
        ang = k * math.pi / 4 + rng.uniform(-0.2, 0.2)
        r = (30 + rng.uniform(-6, 8)) * s
        pts.append((x + math.cos(ang) * r * 1.2, y + math.sin(ang) * r))
    d.polygon([(S(cfg, px), S(cfg, py)) for px, py in pts], fill=ink_rgba(cfg, 230))
    d.polygon([(S(cfg, px), S(cfg, py)) for px, py in pts], outline=cfg["paper"] + (130,), width=2)
    d.line((S(cfg, x - 14 * s), S(cfg, y - 10 * s), S(cfg, x + 6 * s), S(cfg, y - 16 * s)),
           fill=ink_rgba(cfg, 160), width=2)
    d.arc((S(cfg, x - 22 * s), S(cfg, y - 18 * s), S(cfg, x + 2 * s), S(cfg, y - 2 * s)),
          140, 250, fill=cfg["paper"] + (140,), width=3)


# ----------------------------------------------------------------------------
# a pusher — pale bent figure straining against the weight
# ----------------------------------------------------------------------------
def draw_pusher(d, cfg, x, y, direction, s=1.0, rng=None):
    """direction: +1 pushes right, -1 pushes left. Body bent into the weight."""
    paper = cfg["paper"]
    ink = cfg["ink"]
    dd = direction
    # legs braced behind
    d.line((S(cfg, x - dd * 6 * s), S(cfg, y), S(cfg, x - dd * 22 * s), S(cfg, y - 6 * s)),
           fill=ink_rgba(cfg, 165), width=4)
    d.line((S(cfg, x + dd * 4 * s), S(cfg, y), S(cfg, x + dd * 4 * s), S(cfg, y - 14 * s)),
           fill=ink_rgba(cfg, 165), width=4)
    # torso bent forward into the weight
    d.arc((S(cfg, x - 26 * s), S(cfg, y - 66 * s), S(cfg, x + 26 * s), S(cfg, y + 2 * s)),
          40, 180, fill=paper + (185,), width=11)
    # head low, against the weight
    hx0, hx1 = x + dd * 16 * s, x + dd * 34 * s
    d.ellipse((S(cfg, min(hx0, hx1)), S(cfg, y - 58 * s), S(cfg, max(hx0, hx1)), S(cfg, y - 42 * s)),
              fill=paper + (190,))
    # arms straining forward
    d.line((S(cfg, x + dd * 12 * s), S(cfg, y - 44 * s), S(cfg, x + dd * 34 * s), S(cfg, y - 34 * s)),
           fill=paper + (175,), width=4)
    d.line((S(cfg, x + dd * 8 * s), S(cfg, y - 52 * s), S(cfg, x + dd * 30 * s), S(cfg, y - 40 * s)),
           fill=paper + (175,), width=4)
    # mouth open in the strain
    mx0, mx1 = x + dd * 22 * s, x + dd * 28 * s
    d.ellipse((S(cfg, min(mx0, mx1)), S(cfg, y - 50 * s), S(cfg, max(mx0, mx1)), S(cfg, y - 46 * s)),
              fill=ink_rgba(cfg, 190))


# ----------------------------------------------------------------------------
# the two semicircular crowds
# ----------------------------------------------------------------------------
def layer_crowd(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 43)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    # left group — pushing rightward, rising toward the centre
    left_path = [(280, 2160), (470, 2010), (655, 1880), (845, 1800), (1010, 1775)]
    # right group — pushing leftward
    right_path = [(1920, 2160), (1730, 2010), (1545, 1880), (1355, 1800), (1190, 1775)]
    for k, (x, y) in enumerate(left_path):
        s = 0.9 + 0.15 * (k / 4)
        draw_bag(d, cfg, x + 62, y - 14, 0.75 + 0.1 * (k / 4), rng) if k % 2 == 0 else \
            draw_boulder(d, cfg, x + 62, y - 14, 0.8 + 0.1 * (k / 4), rng)
        draw_pusher(d, cfg, x - 8, y, 1, s, rng)
    for k, (x, y) in enumerate(right_path):
        s = 0.9 + 0.15 * (k / 4)
        if k % 2 == 1:
            draw_bag(d, cfg, x - 62, y - 14, 0.75 + 0.1 * (k / 4), rng)
        else:
            draw_boulder(d, cfg, x - 62, y - 14, 0.8 + 0.1 * (k / 4), rng)
        draw_pusher(d, cfg, x + 8, y, -1, s, rng)
    # the collision — two pairs face to face over shared weights
    draw_pusher(d, cfg, 1016, 1852, 1, 1.05, rng)
    draw_pusher(d, cfg, 1184, 1852, -1, 1.05, rng)
    draw_bag(d, cfg, 1100, 1838, 1.0, rng)
    draw_pusher(d, cfg, 1020, 1950, 1, 0.95, rng)
    draw_pusher(d, cfg, 1180, 1950, -1, 0.95, rng)
    draw_boulder(d, cfg, 1100, 1936, 0.95, rng)
    # a few stragglers further back
    for x, y in ((760, 1930), (1440, 1930)):
        dd = 1 if x < 1100 else -1
        draw_pusher(d, cfg, x, y, dd, 0.8, rng)
        draw_bag(d, cfg, x + dd * 52, y - 12, 0.6, rng)
    return L


# ----------------------------------------------------------------------------
# Plutus on his ledge
# ----------------------------------------------------------------------------
def layer_plutus(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]
    x, y = 860, 1260
    # bloated body
    d.ellipse((S(cfg, x - 46), S(cfg, y - 52), S(cfg, x + 46), S(cfg, y + 44)),
              fill=ink_rgba(cfg, 238))
    # squatting legs
    d.line((S(cfg, x - 24), S(cfg, y + 40), S(cfg, x - 34), S(cfg, y + 58)),
           fill=ink_rgba(cfg, 235), width=12)
    d.line((S(cfg, x + 24), S(cfg, y + 40), S(cfg, x + 34), S(cfg, y + 58)),
           fill=ink_rgba(cfg, 235), width=12)
    # head, mouth wide open in the cry
    d.ellipse((S(cfg, x - 20), S(cfg, y - 78), S(cfg, x + 20), S(cfg, y - 44)),
              fill=ink_rgba(cfg, 240))
    d.ellipse((S(cfg, x - 8), S(cfg, y - 70), S(cfg, x + 14), S(cfg, y - 52)),
              fill=paper + (200,))
    d.ellipse((S(cfg, x - 8), S(cfg, y - 68), S(cfg, x + 10), S(cfg, y - 56)),
              fill=ink_rgba(cfg, 230))
    # horns
    d.polygon([(S(cfg, x - 14), S(cfg, y - 72)), (S(cfg, x - 22), S(cfg, y - 96)),
               (S(cfg, x - 6), S(cfg, y - 76))], fill=ink_rgba(cfg, 235))
    d.polygon([(S(cfg, x + 10), S(cfg, y - 74)), (S(cfg, x + 20), S(cfg, y - 98)),
               (S(cfg, x + 22), S(cfg, y - 74))], fill=ink_rgba(cfg, 235))
    # raised arms raging
    d.line((S(cfg, x - 30), S(cfg, y - 34), S(cfg, x - 46), S(cfg, y - 70)),
           fill=ink_rgba(cfg, 235), width=8)
    d.line((S(cfg, x - 46), S(cfg, y - 70), S(cfg, x - 52), S(cfg, y - 88)),
           fill=ink_rgba(cfg, 235), width=6)
    d.line((S(cfg, x + 30), S(cfg, y - 34), S(cfg, x + 46), S(cfg, y - 66)),
           fill=ink_rgba(cfg, 235), width=8)
    # rim light from above
    d.arc((S(cfg, x - 40), S(cfg, y - 72), S(cfg, x + 40), S(cfg, y - 12)),
          200, 330, fill=paper + (150,), width=2)
    return L


# ----------------------------------------------------------------------------
# Dante and Virgil watching from the edge
# ----------------------------------------------------------------------------
def layer_poets(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]
    # Virgil — pointing at the crowds
    vx, vy = S(cfg, 452), S(cfg, 2130)
    s = S(cfg, 0.95)
    d.polygon([(vx - 16 * s, vy), (vx + 15 * s, vy),
               (vx + 18 * s, vy - 50 * s), (vx + 11 * s, vy - 90 * s),
               (vx + 3 * s, vy - 112 * s), (vx - 8 * s, vy - 112 * s),
               (vx - 12 * s, vy - 90 * s), (vx - 17 * s, vy - 48 * s)],
              fill=ink_rgba(cfg, 238))
    d.ellipse((vx - 10 * s, vy - 130 * s, vx + 9 * s, vy - 112 * s),
              fill=ink_rgba(cfg, 238))
    d.line((vx + 2 * s, vy - 96 * s, vx + 24 * s, vy - 128 * s),
           fill=ink_rgba(cfg, 235), width=6)
    d.line((vx + 24 * s, vy - 128 * s, vx + 40 * s, vy - 142 * s),
           fill=ink_rgba(cfg, 235), width=5)
    for ox in (-8, 0, 7):
        d.line((vx + ox * s, vy - 100 * s, vx + ox * s * 0.8, vy - 12 * s),
               fill=ink_rgba(cfg, 140), width=2)
    # Dante — hands clasped, watching
    dx_, dy_ = S(cfg, 566), S(cfg, 2115)
    ds = S(cfg, 0.85)
    d.polygon([(dx_ - 13 * ds, dy_), (dx_ + 12 * ds, dy_),
               (dx_ + 15 * ds, dy_ - 48 * ds), (dx_ + 9 * ds, dy_ - 86 * ds),
               (dx_ + 2 * ds, dy_ - 108 * ds), (dx_ - 7 * ds, dy_ - 108 * ds),
               (dx_ - 11 * ds, dy_ - 86 * ds), (dx_ - 14 * ds, dy_ - 46 * ds)],
              fill=ink_rgba(cfg, 236))
    d.ellipse((dx_ - 11 * ds, dy_ - 128 * ds, dx_ + 8 * ds, dy_ - 108 * ds),
              fill=ink_rgba(cfg, 236))
    for ox in (-6, 0, 6):
        d.line((dx_ + ox * ds, dy_ - 92 * ds, dx_ + ox * ds * 0.8, dy_ - 10 * ds),
               fill=ink_rgba(cfg, 140), width=2)
    d.arc((vx - 11 * s, vy - 134 * s, vx + 10 * s, vy - 110 * s), 250, 350,
          fill=paper + (255,), width=2)
    d.arc((dx_ - 12 * ds, dy_ - 132 * ds, dx_ + 9 * ds, dy_ - 106 * ds), 250, 350,
          fill=paper + (255,), width=2)
    return L


# ----------------------------------------------------------------------------
# frame + caption (Canto VII)
# ----------------------------------------------------------------------------
def draw_frame_and_caption7(img, cfg):
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
    d.text((cx, cy_sub), "CHANT SEPTIÈME — LES AVARES ET LES PRODIGUES", font=f_sub, fill=ink + (230,), anchor="mm")
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
    dark = make_darkmap7(cfg)
    paper = make_paper7(cfg, dark).convert("RGBA")
    size = (S(cfg, W), S(cfg, H))

    layers = []
    print("  walls ...")
    layers += layer_walls(cfg)
    print("  floor ...")
    layers += layer_floor(cfg)
    print("  crowd ...")
    layers.append(layer_crowd(cfg))
    print("  plutus ...")
    layers.append(layer_plutus(cfg))
    print("  poets ...")
    layers.append(layer_poets(cfg))

    print("  compositing ...")
    ink = Image.new("RGBA", size, (0, 0, 0, 0))
    for L in layers:
        ink = Image.alpha_composite(ink, L)
    ink_final = ink.resize((W, H), Image.LANCZOS)
    img = Image.alpha_composite(paper, ink_final)
    img = draw_frame_and_caption7(img, cfg)
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
    name = sys.argv[1] if len(sys.argv) > 1 else "c7_v01.png"
    render(CFG, name)
