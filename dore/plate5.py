#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procedural engraving engine in the manner of Gustave Doré — Canto V.
Scene: "DANTE — L'ENFER · CHANT CINQUIÈME · PAOLO ET FRANCESCA"
The eternal whirlwind of the second circle: spiral wind-lines, pale bodies
swept around the eye of the storm, Paolo and Francesca embracing at the
centre, Dante overcome and Virgil steadying him below.

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
    seed=55,
    paper=(241, 232, 213),
    ink=(30, 23, 16),
    eye=(1100, 1180),           # centre of the whirlwind — the lovers
    frame=(0.055, 0.035, 0.945, 0.905),
    caption_top=0.775,
    out_dir="DORE_INFERNO/Chant_V_Paolo_et_Francesca",
)


# ----------------------------------------------------------------------------
# tone field — dark storm, faint light in the eye of the whirlwind
# ----------------------------------------------------------------------------
def make_darkmap5(cfg, ss_div=4):
    W, H = cfg["W"], cfg["H"]
    hw, hh = W // ss_div, H // ss_div
    cx, cy = cfg["eye"]
    y, x = np.mgrid[0:hh, 0:hw].astype(np.float32)
    x = x * ss_div
    y = y * ss_div
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    n = fbm_arr(hh, hw, cfg["seed"] + 6, 30, 4)
    base = 0.72 + 0.10 * n + 0.06 * smooth01((y - 1950.0) / 300.0)
    glow = 0.26 * np.exp(-((r / 300.0) ** 2)) + 0.09 * np.exp(-((r / 700.0) ** 2))
    d = np.clip(base - glow, 0.0, 1.0)
    im = Image.fromarray((d * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0


# ----------------------------------------------------------------------------
# the whirlwind — spiral arms + tangential slivers
# ----------------------------------------------------------------------------
def layer_wind(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 21)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    cx, cy = cfg["eye"]
    k = 0.155
    for i in range(110):
        t0 = rng.uniform(0, 2 * math.pi)
        pts = []
        for j in range(96):
            t = t0 + j * 5.0 * math.pi / 95
            r = 240 * math.exp(k * (t - t0))
            rr = r * (1.0 + 0.06 * (rng.random() - 0.5))
            px = cx + rr * math.cos(t)
            py = cy + rr * math.sin(t) * 0.86
            if py > 1940:
                break
            pts.append((S(cfg, px), S(cfg, py)))
        a = int(clamp(75 + 135 * (r / 1000.0), 45, 200))
        if len(pts) > 2:
            d.line(pts, fill=ink_rgba(cfg, a), width=3 if r > 450 else 2)
    # tangential slivers — the rushing air
    for _ in range(8500):
        t = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(250, 1100)
        px = cx + r * math.cos(t)
        py = cy + r * math.sin(t) * 0.86
        if py > 1950:
            continue
        tx = math.cos(t) - k * math.sin(t)
        ty = math.sin(t) + k * math.cos(t)
        tl = math.hypot(tx, ty) or 1.0
        tx, ty = tx / tl, ty / tl
        ln = rng.uniform(20, 70)
        a = int(clamp(80 + 130 * (r / 900.0) - 30 * smooth01((py - 900) / 900.0), 30, 180))
        wob = rng.uniform(-0.6, 0.6)
        d.line((S(cfg, px - tx * ln), S(cfg, py - ty * ln * 0.86),
                S(cfg, px + tx * ln + wob * ln), S(cfg, py + ty * ln * 0.86)),
               fill=ink_rgba(cfg, a), width=2)
    return L


# ----------------------------------------------------------------------------
# pale bodies swept in the storm
# ----------------------------------------------------------------------------
def layer_bodies(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 22)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    cx, cy = cfg["eye"]
    k = 0.155
    paper = cfg["paper"]
    ink = cfg["ink"]
    for _ in range(54):
        t = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(240, 1020)
        px = cx + r * math.cos(t)
        py = cy + r * math.sin(t) * 0.86
        if py < 150 or py > 1920 or math.hypot(px - cx, py - cy) < 200:
            continue
        tx = math.cos(t) - k * math.sin(t)
        ty = math.sin(t) + k * math.cos(t)
        tl = math.hypot(tx, ty) or 1.0
        tx, ty = tx / tl, ty / tl
        flail = rng.uniform(-1.3, 1.3)
        sz = rng.uniform(0.7, 1.3)
        x0, y0 = px - tx * 12 * sz, py - ty * 12 * sz
        x1, y1 = px + tx * 12 * sz, py + ty * 12 * sz
        d.line((S(cfg, x0), S(cfg, y0), S(cfg, x1), S(cfg, y1)),
               fill=paper + (int(rng.uniform(145, 195)),), width=max(3, int(S(cfg, 3.0 * sz))))
        # head
        hx, hy = x1 + tx * 5 * sz, y1 + ty * 5 * sz
        d.ellipse((S(cfg, hx - 3 * sz), S(cfg, hy - 3 * sz), S(cfg, hx + 3 * sz), S(cfg, hy + 3 * sz)),
                  fill=paper + (200,))
        # flailing limbs
        for _ in range(2):
            ang = math.atan2(ty, tx) + flail + rng.uniform(-0.8, 0.8)
            ln = rng.uniform(8, 15) * sz
            d.line((S(cfg, px), S(cfg, py),
                    S(cfg, px + math.cos(ang) * ln), S(cfg, py + math.sin(ang) * ln * 0.86)),
                   fill=ink_rgba(cfg, 130), width=2)
        # trailing drapery
        for _ in range(2):
            ang = math.atan2(ty, tx) + flail * 0.5 + rng.uniform(-0.5, 0.5)
            ln = rng.uniform(10, 22) * sz
            d.line((S(cfg, px - tx * 8 * sz), S(cfg, py - ty * 8 * sz),
                    S(cfg, px - tx * 8 * sz + math.cos(ang) * ln),
                    S(cfg, py - ty * 8 * sz + math.sin(ang) * ln)),
                   fill=ink_rgba(cfg, 110), width=1)
    return L


# ----------------------------------------------------------------------------
# Paolo and Francesca — the embracing lovers in the eye
# ----------------------------------------------------------------------------
def layer_pair(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 23)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    cx, cy = cfg["eye"]
    paper = cfg["paper"]
    # luminous eye behind the pair
    d.ellipse((S(cfg, cx - 160), S(cfg, cy - 140), S(cfg, cx + 160), S(cfg, cy + 140)),
              fill=paper + (90,))
    d.ellipse((S(cfg, cx - 95), S(cfg, cy - 85), S(cfg, cx + 95), S(cfg, cy + 85)),
              fill=paper + (115,))
    # wind wraps around the eye
    for r0, a in ((170, 130), (150, 110), (132, 95)):
        pts = [(S(cfg, cx + r0 * math.cos(t)), S(cfg, cy + r0 * math.sin(t) * 0.86))
               for t in np.linspace(0.3, 6.0, 60)]
        d.line(pts, fill=ink_rgba(cfg, a), width=3)
    # --- Francesca (right, leaning back, hair streaming up) ---
    fx, fy = 1122, 1182
    # ink under-stroke as outline
    d.arc((S(cfg, fx - 52), S(cfg, fy - 40), S(cfg, fx + 52), S(cfg, fy + 40)),
          150, 320, fill=ink_rgba(cfg, 130), width=13)
    d.arc((S(cfg, fx - 52), S(cfg, fy - 40), S(cfg, fx + 52), S(cfg, fy + 40)),
          150, 320, fill=paper + (225,), width=9)
    # head tilted back
    d.ellipse((S(cfg, fx - 2), S(cfg, fy - 46), S(cfg, fx + 14), S(cfg, fy - 30)),
              fill=paper + (225,))
    # hair streaming with the wind
    for hk in range(4):
        t = 0.5 + hk * 0.55
        x0 = fx + 8 + hk * 2
        y0 = fy - 40
        d.line((S(cfg, x0), S(cfg, y0),
                S(cfg, x0 + 26 + hk * 8 + rng.uniform(-4, 4)),
                S(cfg, y0 - 6 - hk * 6 + rng.uniform(-3, 3))),
               fill=ink_rgba(cfg, 175), width=2)
    # --- Paolo (left, bent around her) ---
    px_, py_ = 1078, 1188
    d.arc((S(cfg, px_ - 55), S(cfg, py_ - 38), S(cfg, px_ + 55), S(cfg, py_ + 38)),
          320, 500, fill=ink_rgba(cfg, 130), width=13)
    d.arc((S(cfg, px_ - 55), S(cfg, py_ - 38), S(cfg, px_ + 55), S(cfg, py_ + 38)),
          320, 500, fill=paper + (225,), width=9)
    d.ellipse((S(cfg, px_ + 26), S(cfg, py_ - 30), S(cfg, px_ + 42), S(cfg, py_ - 14)),
              fill=paper + (225,))
    # Paolo's hair
    for hk in range(3):
        x0 = px_ + 34
        y0 = py_ - 26 + hk * 4
        d.line((S(cfg, x0), S(cfg, y0),
                S(cfg, x0 + 12 + hk * 5), S(cfg, y0 + 10 + hk * 4)),
               fill=ink_rgba(cfg, 170), width=2)
    # arms wrapped around each other
    d.line((S(cfg, 1096), S(cfg, 1156), S(cfg, 1126), S(cfg, 1180)),
           fill=paper + (215,), width=6)
    d.line((S(cfg, 1104), S(cfg, 1192), S(cfg, 1092), S(cfg, 1162)),
           fill=paper + (215,), width=6)
    # legs trailing downward
    d.line((S(cfg, 1078), S(cfg, 1204), S(cfg, 1062), S(cfg, 1240)),
           fill=paper + (200,), width=5)
    d.line((S(cfg, 1122), S(cfg, 1202), S(cfg, 1138), S(cfg, 1238)),
           fill=paper + (200,), width=5)
    # long drapery sweeping to the left
    for dk in range(5):
        t = dk / 4
        x0 = 1090 - dk * 8
        y0 = 1196 + dk * 6
        d.line((S(cfg, x0), S(cfg, y0),
                S(cfg, x0 - 60 - dk * 30 + rng.uniform(-6, 6)),
                S(cfg, y0 + 16 + dk * 18 + rng.uniform(-5, 5))),
               fill=ink_rgba(cfg, 145), width=2)
    return L


# ----------------------------------------------------------------------------
# ground strip, Dante overcome, Virgil steadying him
# ----------------------------------------------------------------------------
def layer_ground(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 24)
    m = band_mask(size, S(cfg, 1970), S(cfg, 2230))
    depth = grad_mask(size, S(cfg, 1970), S(cfg, 2230), 150, 255)
    m = ImageChops.multiply(m, depth)
    stack = [strokes_layer(size, cfg["ink"], m, lambda d:
                           draw_hatch(d, [(S(cfg, 121), S(cfg, 1970)), (S(cfg, 2079), S(cfg, 1970)),
                                          (S(cfg, 2079), S(cfg, 2230)), (S(cfg, 121), S(cfg, 2230))],
                                      3, 5.4, rng, jitter=2.2, width=3)),
             strokes_layer(size, cfg["ink"], m, lambda d:
                           draw_hatch(d, [(S(cfg, 121), S(cfg, 1970)), (S(cfg, 2079), S(cfg, 1970)),
                                          (S(cfg, 2079), S(cfg, 2230)), (S(cfg, 121), S(cfg, 2230))],
                                      91, 14, rng, jitter=5.0, width=1, dash=30))]
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    for _ in range(10):
        x = rng.uniform(180, 2020)
        y = rng.uniform(2030, 2210)
        r = rng.uniform(7, 20)
        d.ellipse((S(cfg, x - r), S(cfg, y - r * 0.45), S(cfg, x + r), S(cfg, y + r * 0.45)),
                  outline=ink_rgba(cfg, 135), width=2)
    stack.append(L)
    return stack


def layer_figures(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]
    # --- Dante: overcome, sagging, hand to his brow ---
    dx_, dy_ = S(cfg, 1012), S(cfg, 2120)
    s = S(cfg, 1.0)
    d.polygon([(dx_ - 16 * s, dy_), (dx_ + 14 * s, dy_),
               (dx_ + 18 * s, dy_ - 50 * s), (dx_ + 12 * s, dy_ - 92 * s),
               (dx_ + 4 * s, dy_ - 116 * s), (dx_ - 8 * s, dy_ - 116 * s),
               (dx_ - 13 * s, dy_ - 90 * s), (dx_ - 17 * s, dy_ - 48 * s)],
              fill=ink_rgba(cfg, 240))
    d.ellipse((dx_ - 13 * s, dy_ - 138 * s, dx_ + 9 * s, dy_ - 116 * s),
              fill=ink_rgba(cfg, 240))
    # arm raised to the brow
    d.line((dx_ - 6 * s, dy_ - 104 * s, dx_ - 16 * s, dy_ - 128 * s),
           fill=ink_rgba(cfg, 235), width=6)
    # sagging folds
    for ox in (-9, 0, 8):
        d.line((dx_ + ox * s, dy_ - 100 * s, dx_ + ox * s * 0.8, dy_ - 12 * s),
               fill=ink_rgba(cfg, 145), width=2)
    # --- Virgil: upright, steadying him ---
    vx, vy = S(cfg, 1126), S(cfg, 2136)
    d.polygon([(vx - 15 * s, vy), (vx + 14 * s, vy),
               (vx + 17 * s, vy - 54 * s), (vx + 11 * s, vy - 98 * s),
               (vx + 3 * s, vy - 122 * s), (vx - 8 * s, vy - 122 * s),
               (vx - 12 * s, vy - 98 * s), (vx - 16 * s, vy - 52 * s)],
              fill=ink_rgba(cfg, 242))
    d.ellipse((vx - 9 * s, vy - 142 * s, vx + 10 * s, vy - 122 * s),
              fill=ink_rgba(cfg, 242))
    # supporting arm reaching to Dante
    d.line((vx - 12 * s, vy - 96 * s, vx - 30 * s, vy - 60 * s),
           fill=ink_rgba(cfg, 240), width=6)
    for ox in (-8, 0, 7):
        d.line((vx + ox * s, vy - 106 * s, vx + ox * s * 0.8, vy - 12 * s),
               fill=ink_rgba(cfg, 150), width=2)
    # rim light from the storm's eye
    d.arc((dx_ - 13 * s, dy_ - 142 * s, dx_ + 10 * s, dy_ - 114 * s), 190, 340,
          fill=paper + (255,), width=2)
    d.arc((vx - 10 * s, vy - 146 * s, vx + 11 * s, vy - 120 * s), 190, 340,
          fill=paper + (255,), width=2)
    return L


# ----------------------------------------------------------------------------
# frame + caption (Canto V)
# ----------------------------------------------------------------------------
def draw_frame_and_caption5(img, cfg):
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
    d.text((cx, cy_sub), "CHANT CINQUIÈME — PAOLO ET FRANCESCA", font=f_sub, fill=ink + (230,), anchor="mm")
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
    dark = make_darkmap5(cfg)
    paper = make_paper(cfg, dark).convert("RGBA")
    size = (S(cfg, W), S(cfg, H))

    layers = []
    print("  wind ...")
    layers.append(layer_wind(cfg))
    print("  bodies ...")
    layers.append(layer_bodies(cfg))
    print("  the pair ...")
    layers.append(layer_pair(cfg))
    print("  ground ...")
    layers += layer_ground(cfg)
    print("  figures ...")
    layers.append(layer_figures(cfg))

    print("  compositing ...")
    ink = Image.new("RGBA", size, (0, 0, 0, 0))
    for L in layers:
        ink = Image.alpha_composite(ink, L)
    ink_final = ink.resize((W, H), Image.LANCZOS)
    img = Image.alpha_composite(paper, ink_final)
    img = draw_frame_and_caption5(img, cfg)
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
    name = sys.argv[1] if len(sys.argv) > 1 else "c5_v01.png"
    render(CFG, name)
