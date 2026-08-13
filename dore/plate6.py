#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procedural engraving engine in the manner of Gustave Doré — Canto VI.
Scene: "DANTE — L'ENFER · CHANT SIXIÈME · CERBÈRE"
The third circle: the three-headed dog Cerberus over the mire of the gluttons,
under the eternal cold rain. Virgil casts earth into one of the three gullets,
Dante cowers behind him; Ciacco sits up from the mud.

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
    seed=66,
    paper=(241, 232, 213),
    ink=(30, 23, 16),
    frame=(0.055, 0.035, 0.945, 0.905),
    caption_top=0.775,
    out_dir="DORE_INFERNO/Chant_VI_Cerbere",
)


# ----------------------------------------------------------------------------
# tone field — dim rainy air, darker mud below
# ----------------------------------------------------------------------------
def make_darkmap6(cfg, ss_div=4):
    W, H = cfg["W"], cfg["H"]
    hw, hh = W // ss_div, H // ss_div
    y, x = np.mgrid[0:hh, 0:hw].astype(np.float32)
    x = x * ss_div
    y = y * ss_div
    n = fbm_arr(hh, hw, cfg["seed"] + 8, 30, 4)
    base = 0.72 + 0.10 * n + 0.10 * smooth01((y - 1800.0) / 300.0)
    # faint sickly light from above-left
    light = 0.12 * np.exp(-(((x - 900) / 900.0) ** 2 + ((y - 300) / 700.0) ** 2))
    d = np.clip(base - light, 0.0, 1.0)
    # keep the caption band clean paper
    d = d * (1.0 - 0.85 * smooth01((y - 2217.0) / 60.0))
    im = Image.fromarray((d * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0


# ----------------------------------------------------------------------------
# darker paper for the rainy mire (stronger shade than the default make_paper)
# ----------------------------------------------------------------------------
def make_paper6(cfg, dark):
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
# the eternal rain
# ----------------------------------------------------------------------------
def layer_rain(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 31)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    ang = math.radians(76)
    dx, dy = math.cos(ang), math.sin(ang)
    for _ in range(3400):
        x = rng.uniform(110, 2090)
        y = rng.uniform(100, 2230)
        ln = rng.uniform(50, 150)
        a = int(rng.uniform(26, 50))
        d.line((S(cfg, x), S(cfg, y), S(cfg, x + dx * ln), S(cfg, y + dy * ln)),
               fill=ink_rgba(cfg, a), width=2)
    return L


# ----------------------------------------------------------------------------
# the mire of the gluttons
# ----------------------------------------------------------------------------
def layer_mud(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 32)
    m = band_mask(size, S(cfg, 1880), S(cfg, 2230))
    depth = grad_mask(size, S(cfg, 1880), S(cfg, 2230), 140, 255)
    m = ImageChops.multiply(m, depth)
    rect = [(S(cfg, 121), S(cfg, 1880)), (S(cfg, 2079), S(cfg, 1880)),
            (S(cfg, 2079), S(cfg, 2230)), (S(cfg, 121), S(cfg, 2230))]
    stack = [strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 3, 4.8, rng, jitter=2.2, width=3)),
             strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 92, 13, rng, jitter=5.0, width=1, dash=28))]
    # puddle glints + mud lumps
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    for _ in range(40):
        x = rng.uniform(150, 2050)
        y = rng.uniform(1900, 2220)
        ln = rng.uniform(10, 34)
        d.line((S(cfg, x), S(cfg, y), S(cfg, x + ln), S(cfg, y)),
               fill=cfg["paper"] + (55,), width=2)
    for _ in range(30):
        x = rng.uniform(150, 2050)
        y = rng.uniform(1900, 2220)
        r = rng.uniform(4, 10)
        d.ellipse((S(cfg, x - r), S(cfg, y - r * 0.5), S(cfg, x + r), S(cfg, y + r * 0.5)),
                  outline=ink_rgba(cfg, 110), width=2)
    stack.append(L)
    return stack


def layer_gluttons(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 33)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]
    # bodies lying half-submerged
    for _ in range(8):
        x = rng.uniform(220, 1880)
        y = rng.uniform(1970, 2180)
        if 700 < x < 1100 and y > 1950:   # keep clear of Cerberus' forelegs
            x = rng.uniform(1150, 1880)
        ln = rng.uniform(26, 46)
        a = int(rng.uniform(130, 180))
        d.arc((S(cfg, x - ln), S(cfg, y - 8), S(cfg, x + ln), S(cfg, y + 8)),
              200, 330, fill=paper + (a,), width=7)
        d.ellipse((S(cfg, x + ln - 6), S(cfg, y - 12), S(cfg, x + ln + 6), S(cfg, y)),
                  fill=paper + (a,))
        # one arm raised out of the mud
        if rng.random() < 0.4:
            d.line((S(cfg, x + ln * 0.4), S(cfg, y - 2),
                    S(cfg, x + ln * 0.4 + rng.uniform(-10, 10)), S(cfg, y - rng.uniform(14, 24))),
                   fill=paper + (a - 20,), width=3)
    # Ciacco — sitting up, arm raised toward the poets
    cx_, cy_ = 1170, 2020
    d.arc((S(cfg, cx_ - 30), S(cfg, cy_ - 26), S(cfg, cx_ + 30), S(cfg, cy_ + 26)),
          230, 380, fill=paper + (185,), width=11)
    d.ellipse((S(cfg, cx_ - 14), S(cfg, cy_ - 52), S(cfg, cx_ + 10), S(cfg, cy_ - 26)),
              fill=paper + (185,))
    d.line((S(cfg, cx_ + 18), S(cfg, cy_ - 38), S(cfg, cx_ + 46), S(cfg, cy_ - 62)),
           fill=paper + (180,), width=5)
    d.line((S(cfg, cx_ + 4), S(cfg, cy_ + 14), S(cfg, cx_ + 14), S(cfg, cy_ + 40)),
           fill=paper + (140,), width=4)
    return L


# ----------------------------------------------------------------------------
# Cerberus — the three-headed dog
# ----------------------------------------------------------------------------
def layer_cerberus(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 34)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]
    ink = cfg["ink"]

    def R(x0, y0, ang, px, py, s=1.0):
        c, sn = math.cos(ang), math.sin(ang)
        return (x0 + (px * c - py * sn) * s, y0 + (px * sn + py * c) * s)

    def seg(x0, y0, ang, p0, p1, s=1.0, **kw):
        (ax, ay) = R(x0, y0, ang, *p0, s)
        (bx, by) = R(x0, y0, ang, *p1, s)
        d.line((S(cfg, ax), S(cfg, ay), S(cfg, bx), S(cfg, by)), **kw)

    def poly(x0, y0, ang, pts, s=1.0, **kw):
        q = []
        for px, py in pts:
            rx, ry = R(x0, y0, ang, px, py, s)
            q.append((S(cfg, rx), S(cfg, ry)))
        d.polygon(q, **kw)

    def head(x, y, ang, s=1.0):
        # skull
        poly(x, y, ang, [(-34, -16), (-24, -27), (4, -30), (28, -20), (36, -4),
                         (30, 9), (8, 18), (-22, 16), (-34, 6)], s, fill=ink_rgba(cfg, 240))
        # upper jaw
        poly(x, y, ang, [(14, -10), (56, -14), (52, -3), (16, -3)], s, fill=ink_rgba(cfg, 240))
        # lower jaw (dropped open)
        poly(x, y, ang, [(12, 6), (46, 4), (42, 16), (14, 15)], s, fill=ink_rgba(cfg, 235))
        # mouth interior
        poly(x, y, ang, [(14, -4), (46, -6), (42, 8), (16, 5)], s, fill=ink_rgba(cfg, 250))
        # teeth
        for k in range(4):
            tx = 18 + k * 7
            poly(x, y, ang, [(tx, -6), (tx + 3, -1), (tx + 6, -6)], s, fill=paper + (215,))
            poly(x, y, ang, [(tx + 1, 8), (tx + 4, 3), (tx + 7, 8)], s, fill=paper + (210,))
        # eye
        (ex0, ey0) = R(x, y, ang, -2, -18, s)
        (ex1, ey1) = R(x, y, ang, 6, -10, s)
        d.ellipse((S(cfg, min(ex0, ex1)), S(cfg, min(ey0, ey1)),
                   S(cfg, max(ex0, ex1)), S(cfg, max(ey0, ey1))),
                  fill=paper + (230,))
        (px_, py_) = R(x, y, ang, 2, -14, s)
        d.ellipse((S(cfg, px_ - 1.5), S(cfg, py_ - 1.5), S(cfg, px_ + 1.5), S(cfg, py_ + 1.5)),
                  fill=ink_rgba(cfg, 255))
        # ear
        poly(x, y, ang, [(-16, -24), (-6, -20), (-14, -38)], s, fill=ink_rgba(cfg, 235))
        # mane behind the skull (follows the head's angle)
        for k in range(6):
            seg(x, y, ang, (-26, -8 - k * 1.5), (-52 - rng.uniform(0, 14), -6 - k * 3),
                s, fill=ink_rgba(cfg, 200), width=3)
        # drool
        seg(x, y, ang, (40, 14), (44, 30 + rng.uniform(0, 10)), s,
            fill=ink_rgba(cfg, 120), width=1)

    # serpent tail
    tail = [(990, 1930), (1085, 1885), (1130, 1950), (1220, 1895), (1265, 1965)]
    d.line([(S(cfg, x), S(cfg, y)) for x, y in tail], fill=ink_rgba(cfg, 225), width=13)
    d.line([(S(cfg, x), S(cfg, y)) for x, y in tail], fill=ink_rgba(cfg, 255), width=9)
    hx, hy = tail[-1]
    d.ellipse((S(cfg, hx - 12), S(cfg, hy - 7), S(cfg, hx + 14), S(cfg, hy + 9)),
              fill=ink_rgba(cfg, 240))
    d.line((S(cfg, hx + 10), S(cfg, hy - 2), S(cfg, hx + 22), S(cfg, hy - 8)),
           fill=ink_rgba(cfg, 180), width=1)
    d.line((S(cfg, hx + 10), S(cfg, hy + 2), S(cfg, hx + 22), S(cfg, hy + 8)),
           fill=ink_rgba(cfg, 180), width=1)

    # body — crouching mass
    body = [(640, 1980), (700, 1830), (765, 1725), (835, 1670), (905, 1710),
            (965, 1790), (1005, 1905), (985, 2005), (900, 2055), (755, 2065), (655, 2035)]
    d.polygon([(S(cfg, x), S(cfg, y)) for x, y in body], fill=ink_rgba(cfg, 238))
    # back haunch
    d.ellipse((S(cfg, 870), S(cfg, 1860), S(cfg, 1015), S(cfg, 2040)),
              fill=ink_rgba(cfg, 238))
    # front legs
    for lx0, ly0, lx1, ly1 in ((715, 1975, 700, 2115), (905, 1990, 925, 2115)):
        d.line((S(cfg, lx0), S(cfg, ly0), S(cfg, lx1), S(cfg, ly1)),
               fill=ink_rgba(cfg, 240), width=40)
        for k in range(3):
            d.line((S(cfg, lx1 - 10 + k * 9), S(cfg, ly1), S(cfg, lx1 - 6 + k * 9), S(cfg, ly1 + 18)),
                   fill=ink_rgba(cfg, 235), width=4)
    # necks
    for (nx0, ny0, nx1, ny1, w) in ((700, 1765, 648, 1375, 38),
                                    (822, 1705, 812, 1255, 40),
                                    (942, 1755, 1022, 1395, 36)):
        d.line((S(cfg, nx0), S(cfg, ny0), S(cfg, nx1), S(cfg, ny1)),
               fill=ink_rgba(cfg, 240), width=S(cfg, w))
    # heads
    head(648, 1375, math.radians(-58), 1.00)   # left — glaring down at the mire
    head(812, 1255, math.radians(22), 1.25)    # centre — jaws at Virgil
    head(1022, 1395, math.radians(165), 0.95)  # right — howling up
    # fur on the body
    m = poly_mask(size, [(S(cfg, x), S(cfg, y)) for x, y in body])
    stack = strokes_layer(size, cfg["ink"], m, lambda dd:
                          draw_hatch(dd, body_bbox(), 8, 7.5, rng, jitter=3.0, width=2, dash=16))
    L = Image.alpha_composite(L, stack)
    # rim light along the back (sickly light from above-left)
    for (ax, ay), (bx, by) in zip(body[:6], body[1:7]):
        d.line((S(cfg, ax - 2), S(cfg, ay - 3), S(cfg, bx - 2), S(cfg, by - 3)),
               fill=paper + (185,), width=3)
    for nx0, ny0, nx1, ny1, w in ((700, 1765, 648, 1375, 38), (822, 1705, 812, 1255, 40)):
        d.line((S(cfg, nx0 - 3), S(cfg, ny0 - 2), S(cfg, nx1 - 3), S(cfg, ny1 - 2)),
               fill=paper + (130,), width=2)
    return L


def body_bbox():
    return [(S(CFG, 640), S(CFG, 1670)), (S(CFG, 1015), S(CFG, 1670)),
            (S(CFG, 1015), S(CFG, 2065)), (S(CFG, 640), S(CFG, 2065))]


# ----------------------------------------------------------------------------
# Virgil casting earth, Dante cowering
# ----------------------------------------------------------------------------
def layer_poets(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 35)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]

    # ---- Virgil, profile, casting earth toward the centre head ----
    vx, vy = S(cfg, 1400), S(cfg, 2060)
    s = S(cfg, 1.05)
    d.polygon([(vx - 20 * s, vy), (vx + 20 * s, vy),
               (vx + 17 * s, vy - 48 * s), (vx + 11 * s, vy - 84 * s),
               (vx - 1 * s, vy - 106 * s), (vx - 13 * s, vy - 92 * s),
               (vx - 18 * s, vy - 48 * s)],
              fill=ink_rgba(cfg, 240))
    d.ellipse((vx - 5 * s, vy - 120 * s, vx + 9 * s, vy - 105 * s),
              fill=ink_rgba(cfg, 240))
    # casting arm up-left toward the centre head
    d.line((vx - 6 * s, vy - 88 * s, vx - 30 * s, vy - 132 * s),
           fill=ink_rgba(cfg, 240), width=7)
    d.line((vx - 30 * s, vy - 132 * s, vx - 48 * s, vy - 158 * s),
           fill=ink_rgba(cfg, 240), width=5)
    for ox in (-8, 0, 7):
        d.line((vx + ox * s, vy - 96 * s, vx + ox * s * 0.8, vy - 12 * s),
               fill=ink_rgba(cfg, 145), width=2)
    d.arc((vx - 22 * s, vy - 10 * s, vx + 22 * s, vy + 6 * s), 190, 350,
          fill=ink_rgba(cfg, 150), width=2)
    # the thrown earth — small dashes arcing toward the jaws
    for k in range(5):
        t = k / 4
        x = 1350 - t * 430
        y = 1930 - t * 640
        d.line((S(cfg, x), S(cfg, y), S(cfg, x + 8 + k * 2), S(cfg, y + 5)),
               fill=ink_rgba(cfg, 150), width=2)

    # ---- Dante, cowering behind Virgil ----
    dx_, dy_ = S(cfg, 1522), S(cfg, 2040)
    ds = S(cfg, 0.9)
    d.polygon([(dx_ - 14 * ds, dy_), (dx_ + 13 * ds, dy_),
               (dx_ + 17 * ds, dy_ - 46 * ds), (dx_ + 11 * ds, dy_ - 82 * ds),
               (dx_ + 4 * ds, dy_ - 104 * ds), (dx_ - 7 * ds, dy_ - 104 * ds),
               (dx_ - 12 * ds, dy_ - 82 * ds), (dx_ - 16 * ds, dy_ - 44 * ds)],
              fill=ink_rgba(cfg, 238))
    d.ellipse((dx_ - 12 * ds, dy_ - 124 * ds, dx_ + 9 * ds, dy_ - 103 * ds),
              fill=ink_rgba(cfg, 238))
    # shielding arm
    d.line((dx_ - 4 * ds, dy_ - 92 * ds, dx_ - 16 * ds, dy_ - 110 * ds),
           fill=ink_rgba(cfg, 235), width=6)
    for ox in (-6, 0, 6):
        d.line((dx_ + ox * ds, dy_ - 88 * ds, dx_ + ox * ds * 0.8, dy_ - 10 * ds),
               fill=ink_rgba(cfg, 140), width=2)
    # rim light
    d.arc((vx - 6 * s, vy - 124 * s, vx + 10 * s, vy - 104 * s), 250, 350,
          fill=paper + (255,), width=2)
    d.arc((dx_ - 12 * ds, dy_ - 128 * ds, dx_ + 10 * ds, dy_ - 102 * ds), 250, 350,
          fill=paper + (255,), width=2)
    return L


# ----------------------------------------------------------------------------
# frame + caption (Canto VI)
# ----------------------------------------------------------------------------
def draw_frame_and_caption6(img, cfg):
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
    d.text((cx, cy_sub), "CHANT SIXIÈME — CERBÈRE", font=f_sub, fill=ink + (230,), anchor="mm")
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
    dark = make_darkmap6(cfg)
    paper = make_paper6(cfg, dark).convert("RGBA")
    size = (S(cfg, W), S(cfg, H))

    layers = []
    print("  rain ...")
    layers.append(layer_rain(cfg))
    print("  mud ...")
    layers += layer_mud(cfg)
    print("  gluttons ...")
    layers.append(layer_gluttons(cfg))
    print("  cerberus ...")
    layers.append(layer_cerberus(cfg))
    print("  poets ...")
    layers.append(layer_poets(cfg))

    print("  compositing ...")
    ink = Image.new("RGBA", size, (0, 0, 0, 0))
    for L in layers:
        ink = Image.alpha_composite(ink, L)
    ink_final = ink.resize((W, H), Image.LANCZOS)
    img = Image.alpha_composite(paper, ink_final)
    img = draw_frame_and_caption6(img, cfg)
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
    name = sys.argv[1] if len(sys.argv) > 1 else "c6_v01.png"
    render(CFG, name)
