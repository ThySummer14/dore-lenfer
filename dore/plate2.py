#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procedural engraving engine in the manner of Gustave Doré — Canto II.
Scene: "DANTE — L'ENFER · CHANT DEUXIÈME · LA FORÊT OBSCURE"
Night falls on the dark wood: towering trees, a winding path toward the last
light, Virgil pointing the way, Dante following.

Reuses the stroke primitives and tone machinery of plate.py (Canto I).
"""
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from plate import (S, ink_rgba, strokes_layer, poly_mask, band_mask, grad_mask,
                   draw_hatch, contour_strokes, fbm_arr, bilinear, clamp, smooth01,
                   make_paper, find_font)

CFG = dict(
    W=2200, H=2860,
    SS=2,
    seed=77,
    paper=(241, 232, 213),
    ink=(30, 23, 16),
    glow=(1100, 1250),        # the last light at the end of the path
    moon=(380, 360),
    frame=(0.055, 0.035, 0.945, 0.905),
    caption_top=0.775,
    out_dir="out",
)


# ----------------------------------------------------------------------------
# tone field — dusk, darkest above, a small glow where the path ends
# ----------------------------------------------------------------------------
def make_darkmap2(cfg, ss_div=4):
    W, H = cfg["W"], cfg["H"]
    hw, hh = W // ss_div, H // ss_div
    cx, cy = cfg["glow"]
    y, x = np.mgrid[0:hh, 0:hw].astype(np.float32)
    x = x * ss_div
    y = y * ss_div
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    n = fbm_arr(hh, hw, cfg["seed"] + 3, 36, 4)
    base = 0.48 + 0.26 * smooth01(y / 1500.0) + 0.10 * n
    # glow hole around the distant light
    glow = (0.30 * np.exp(-((r / 150.0) ** 2)) + 0.12 * np.exp(-((r / 420.0) ** 2)))
    # forest floor brighter than the sky (open glade), but stop at the caption
    floor = 0.10 * smooth01((y - 1900.0) / 260.0) * (1.0 - smooth01((y - 2230.0) / 40.0))
    d = np.clip(base - glow * 0.8 + floor, 0.0, 1.0)
    im = Image.fromarray((d * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0


# ----------------------------------------------------------------------------
# sky — evening cloud bands, slivers, moon, the distant glow, mist
# ----------------------------------------------------------------------------
def layer_sky(cfg, dark_low):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 11)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    # wavy horizontal cloud bands, darker toward the top
    cx, cy = cfg["glow"]
    for k in range(84):
        y = 110 + k * 11.5 + rng.uniform(-6, 6)
        phase = rng.uniform(0, 6.28)
        amp = rng.uniform(6, 20)
        wav = rng.uniform(0.011, 0.024)
        pts = [(x, y + amp * math.sin(x * wav + phase)) for x in range(90, 2110, 40)]
        a = int(clamp(235 - 140 * (y / 1250.0), 55, 240))
        segs, cur = [], []
        for p in pts:
            if math.hypot(p[0] - cx, p[1] - cy) < 150:
                if len(cur) > 1:
                    segs.append(cur)
                    cur = []
            else:
                cur.append(p)
        if len(cur) > 1:
            segs.append(cur)
        for sg in segs:
            d.line(sg, fill=ink_rgba(cfg, a), width=3)
    # slivers — short horizontal cloud strokes
    for _ in range(9000):
        x = rng.uniform(100, 2100)
        y = rng.uniform(120, 1500)
        if math.hypot(x - cx, y - cy) < 150:
            continue
        a = int(clamp(90 + 165 * bilinear(dark_low, x / 4.0, y / 4.0) - 40 * smooth01((y - 1150) / 250.0),
                      12, 210))
        ln = rng.uniform(16, 80)
        d.line((S(cfg, x - ln), S(cfg, y + rng.uniform(-8, 8)),
                S(cfg, x + ln), S(cfg, y + rng.uniform(-8, 8))),
               fill=ink_rgba(cfg, a), width=2)
    # clean luminous hole at the end of the path
    paper = cfg["paper"]
    d.ellipse((S(cfg, cx - 108), S(cfg, cy - 108), S(cfg, cx + 108), S(cfg, cy + 108)),
              fill=paper + (170,))
    d.ellipse((S(cfg, cx - 56), S(cfg, cy - 56), S(cfg, cx + 56), S(cfg, cy + 56)),
              fill=paper + (225,))
    # halo arcs + thin streaks around the distant glow
    for hr, ha in ((46, 70), (76, 48), (108, 32)):
        for g0, g1 in ((0.2, 1.5), (1.8, 3.4), (4.0, 5.6)):
            pts = [(S(cfg, cx + hr * math.cos(t)), S(cfg, cy + hr * math.sin(t)))
                   for t in np.linspace(g0, g1, 20)]
            d.line(pts, fill=ink_rgba(cfg, ha), width=2)
    for _ in range(16):
        ang = math.radians(rng.uniform(55, 125))
        r0 = rng.uniform(90, 160)
        ln = rng.uniform(260, 700)
        d.line((S(cfg, cx + r0 * math.sin(ang)), S(cfg, cy + r0 * math.cos(ang)),
                S(cfg, cx + (r0 + ln) * math.sin(ang)), S(cfg, cy + (r0 + ln) * math.cos(ang))),
               fill=ink_rgba(cfg, int(rng.uniform(10, 22))), width=2)
    # stars coming out in the dusk sky
    for _ in range(34):
        x = rng.uniform(110, 2090)
        y = rng.uniform(140, 640)
        if math.hypot(x - cfg["moon"][0], y - cfg["moon"][1]) < 80:
            continue
        if math.hypot(x - cx, y - cy) < 200:
            continue
        d.rectangle((S(cfg, x), S(cfg, y), S(cfg, x) + 2, S(cfg, y) + 2),
                    fill=ink_rgba(cfg, int(rng.uniform(140, 220))))
    for _ in range(7):
        x = rng.uniform(130, 2060)
        y = rng.uniform(170, 560)
        if math.hypot(x - cfg["moon"][0], y - cfg["moon"][1]) < 90 or math.hypot(x - cx, y - cy) < 220:
            continue
        ln = 6
        a = int(rng.uniform(150, 210))
        d.line((S(cfg, x - ln), S(cfg, y), S(cfg, x + ln), S(cfg, y)), fill=ink_rgba(cfg, a), width=2)
        d.line((S(cfg, x), S(cfg, y - ln), S(cfg, x), S(cfg, y + ln)), fill=ink_rgba(cfg, a), width=2)
    # crescent moon
    mx, my = cfg["moon"]
    d.arc((S(cfg, mx - 46), S(cfg, my - 46), S(cfg, mx + 46), S(cfg, my + 46)),
          20, 330, fill=ink_rgba(cfg, 195), width=3)
    return L


def layer_moon_erase(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    mx, my = cfg["moon"]
    d.ellipse((S(cfg, mx - 40), S(cfg, my - 52), S(cfg, mx + 34), S(cfg, my + 30)),
              fill=cfg["paper"] + (255,))
    return L


# ----------------------------------------------------------------------------
# distant treeline silhouette behind the mist
# ----------------------------------------------------------------------------
def layer_treeline(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 22)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    y0 = 1390
    for x in range(220, 2000, 22):
        h = rng.uniform(40, 110)
        d.polygon([(S(cfg, x - 10), S(cfg, y0)), (S(cfg, x + 10), S(cfg, y0)),
                   (S(cfg, x), S(cfg, y0 - h))], fill=ink_rgba(cfg, 225))
    for x in range(232, 1990, 30):
        h = rng.uniform(22, 60)
        d.polygon([(S(cfg, x - 7), S(cfg, y0 - 4)), (S(cfg, x + 7), S(cfg, y0 - 4)),
                   (S(cfg, x), S(cfg, y0 - 4 - h))], fill=ink_rgba(cfg, 160))
    d.rectangle((S(cfg, 200), S(cfg, y0 - 6), S(cfg, 2010), S(cfg, y0 + 4)),
                fill=ink_rgba(cfg, 225))
    return L


# ----------------------------------------------------------------------------
# the winding path of last light
# ----------------------------------------------------------------------------
def path_centerline():
    return [(1106, 2240), (1090, 2140), (1032, 2045), (1006, 1955), (1060, 1870),
            (1012, 1785), (1052, 1690), (1018, 1600), (1068, 1505), (1100, 1400),
            (1100, 1300)]


def path_polygon():
    cl = path_centerline()
    poly = []
    widths = [96, 88, 76, 68, 60, 58, 50, 45, 39, 31, 29]
    for k, ((x, y), w) in enumerate(zip(cl, widths)):
        dx, dy = 0.0, -1.0
        if k < len(cl) - 1:
            dx, dy = cl[k + 1][0] - x, cl[k + 1][1] - y
        ln = math.hypot(dx, dy) or 1.0
        px, py = -dy / ln, dx / ln
        poly.append((x + px * w, y + py * w))
    for k in range(len(cl) - 1, -1, -1):
        x, y, w = cl[k][0], cl[k][1], widths[k]
        dx, dy = 0.0, -1.0
        if k > 0:
            dx, dy = x - cl[k - 1][0], y - cl[k - 1][1]
        ln = math.hypot(dx, dy) or 1.0
        px, py = -dy / ln, dx / ln
        poly.append((x - px * w, y - py * w))
    return poly


def layer_path(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 33)
    cl = path_centerline()
    sp = [(S(cfg, x), S(cfg, y)) for x, y in path_polygon()]
    # erase to paper (the path of light)
    m = poly_mask(size, sp)
    er = Image.new("RGBA", size, (0, 0, 0, 0))
    er.paste(cfg["paper"] + (170,), (0, 0), m)
    # flow lines along the path
    fl = Image.new("L", size, 0)
    fd = ImageDraw.Draw(fl)
    for k in range(len(cl) - 1):
        x0, y0 = cl[k]
        x1, y1 = cl[k + 1]
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy) or 1.0
        px, py = -dy / ln, dx / ln
        w = 96 * (1 - k / len(cl)) + 26
        for off in np.linspace(-w, w, int(w / 8)):
            jx = rng.uniform(-8, 8)
            jy = rng.uniform(-8, 8)
            fd.line((S(cfg, x0 + px * off + jx), S(cfg, y0 + py * off + jy),
                     S(cfg, x1 + px * off + jx * 0.4), S(cfg, y1 + py * off + jy * 0.4)),
                    fill=255, width=2)
    fl = ImageChops.multiply(fl, m)
    er2 = Image.new("RGBA", size, (0, 0, 0, 0))
    er2.paste(cfg["ink"] + (95,), (0, 0), fl)
    # path edges + stones
    ed = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ed)
    d.line(sp + [sp[0]], fill=ink_rgba(cfg, 110), width=3)
    for _ in range(26):
        t = rng.random()
        x = sum(cl[int(t * (len(cl) - 1))][0] for _ in range(1))
        # interpolate along the centreline
        tt = t * (len(cl) - 1)
        i = int(tt)
        f = tt - i
        if i >= len(cl) - 1:
            i = len(cl) - 2
            f = 1.0
        x = cl[i][0] + (cl[i + 1][0] - cl[i][0]) * f
        y = cl[i][1] + (cl[i + 1][1] - cl[i][1]) * f
        dx, dy = cl[i + 1][0] - cl[i][0], cl[i + 1][1] - cl[i][1]
        ln = math.hypot(dx, dy) or 1.0
        px, py = -dy / ln, dx / ln
        w = 130 * (1 - t) + 36
        off = rng.uniform(-w, w) * 0.7
        rx = rng.uniform(5, 13)
        d.ellipse((S(cfg, x + px * off - rx), S(cfg, y + py * off - rx * 0.5),
                   S(cfg, x + px * off + rx), S(cfg, y + py * off + rx * 0.5)),
                  outline=ink_rgba(cfg, 115), width=2)
    return [er, er2, ed]


# ----------------------------------------------------------------------------
# trees — solid tapered trunks, bark, roots, bare branches
# ----------------------------------------------------------------------------
def taper_poly(pts, w0, w1):
    """Offset a polyline into a tapered polygon."""
    left, right = [], []
    for k, (x, y) in enumerate(pts):
        t = k / max(1, len(pts) - 1)
        w = w0 + (w1 - w0) * t
        if k == 0:
            dx, dy = pts[1][0] - x, pts[1][1] - y
        elif k == len(pts) - 1:
            dx, dy = x - pts[k - 1][0], y - pts[k - 1][1]
        else:
            dx, dy = pts[k + 1][0] - pts[k - 1][0], pts[k + 1][1] - pts[k - 1][1]
        ln = math.hypot(dx, dy) or 1.0
        px, py = -dy / ln, dx / ln
        left.append((x + px * w, y + py * w))
        right.append((x - px * w, y - py * w))
    return left + right[::-1]


def layer_trees(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 44)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    # (base_x, base_y, top_x, top_y, base_w, bend, alpha, bark_seed)
    trees = [
        (250, 2270, 470, 820, 46, 60, 240, 1),
        (168, 2260, 300, 1160, 34, 40, 205, 2),
        (2040, 2270, 1740, 880, 44, -70, 240, 3),
        (1930, 2260, 2060, 1250, 30, -30, 200, 4),
        (706, 2190, 780, 1490, 18, 20, 165, 5),
    ]
    trunk_masks = []
    for (bx, by, tx, ty, w0, bend, alpha, bs) in trees:
        n = 26
        pts = []
        for k in range(n + 1):
            t = k / n
            x = bx + (tx - bx) * t + bend * math.sin(t * math.pi) * 0.55
            y = by + (ty - by) * t
            pts.append((S(cfg, x), S(cfg, y)))
        poly = taper_poly(pts, S(cfg, w0), S(cfg, max(6, w0 * 0.18)))
        d.polygon(poly, fill=ink_rgba(cfg, alpha))
        trunk_masks.append((poly_mask(size, poly), pts, w0, bs))
        # roots
        for ra, rl in ((-0.9, 60), (0.9, 55), (-1.4, 40), (1.35, 44)):
            d.line((S(cfg, bx), S(cfg, by - 8),
                    S(cfg, bx + math.cos(ra) * rl), S(cfg, by - 8 + math.sin(ra) * rl * 0.5 + 26)),
                   fill=ink_rgba(cfg, alpha - 20), width=5)
        # branches
        def branch(x, y, ang, ln, w, depth):
            if depth == 0 or ln < 14:
                return
            x2 = x + math.cos(ang) * ln
            y2 = y + math.sin(ang) * ln
            d.line((S(cfg, x), S(cfg, y), S(cfg, x2), S(cfg, y2)),
                   fill=ink_rgba(cfg, alpha - 8), width=max(2, int(S(cfg, w))))
            offs = [-0.5 + 0.25 * rng.random(), 0.4 + 0.35 * rng.random()]
            for k2, off in enumerate(offs):
                branch(x2, y2, ang + off + (rng.random() - 0.5) * 0.35,
                       ln * (0.66 - 0.07 * k2), w * 0.58, depth - 1)
        for j in range(4):
            t = 0.62 + 0.09 * j
            bx2 = bx + (tx - bx) * t + bend * math.sin(t * math.pi) * 0.55
            by2 = by + (ty - by) * t
            ang0 = -1.15 + (0.55 if bx < 1100 else -0.55) + rng.uniform(-0.25, 0.25)
            branch(bx2, by2, ang0, 130 - 20 * j, 5.5 - j, 3)
    # bark hatch clipped to trunks
    for mask, pts, w0, bs in trunk_masks:
        ang_deg = math.degrees(math.atan2(pts[-1][1] - pts[0][1], pts[-1][0] - pts[0][0]))
        stack = strokes_layer(size, cfg["ink"], mask,
                              lambda dd, a=ang_deg, b=bs: draw_hatch(dd, _bbox_of(pts), a + 90, 7, random.Random(900 + b), jitter=2.5, width=2))
        L = Image.alpha_composite(L, stack)
    return L


def _bbox_of(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]


# ----------------------------------------------------------------------------
# forest floor
# ----------------------------------------------------------------------------
def layer_ground(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 55)
    m = band_mask(size, S(cfg, 1980), S(cfg, 2230))
    depth = grad_mask(size, S(cfg, 1980), S(cfg, 2230), 150, 255)
    m = ImageChops.multiply(m, depth)
    stack = [strokes_layer(size, cfg["ink"], m, lambda d:
                           draw_hatch(d, [(S(cfg, 121), S(cfg, 1980)), (S(cfg, 2080), S(cfg, 1980)),
                                          (S(cfg, 2080), S(cfg, 2230)), (S(cfg, 121), S(cfg, 2230))],
                                      7, 4.6, rng, jitter=2.4, width=2)),
             strokes_layer(size, cfg["ink"], m, lambda d:
                           draw_hatch(d, [(S(cfg, 121), S(cfg, 1980)), (S(cfg, 2080), S(cfg, 1980)),
                                          (S(cfg, 2080), S(cfg, 2230)), (S(cfg, 121), S(cfg, 2230))],
                                      94, 15, rng, jitter=6.0, width=1, dash=30))]
    # leaf litter + stones
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    for _ in range(900):
        x = rng.uniform(130, 2070)
        y = rng.uniform(1990, 2225)
        ln = rng.uniform(4, 12)
        a = rng.uniform(60, 150)
        d.line((S(cfg, x), S(cfg, y), S(cfg, x + ln * rng.uniform(-1, 1)), S(cfg, y + ln * 0.4)),
               fill=ink_rgba(cfg, int(a)), width=2)
    for _ in range(14):
        x = rng.uniform(200, 2000)
        y = rng.uniform(2060, 2220)
        r = rng.uniform(8, 22)
        d.ellipse((S(cfg, x - r), S(cfg, y - r * 0.55), S(cfg, x + r), S(cfg, y + r * 0.55)),
                  outline=ink_rgba(cfg, 140), width=2)
        d.line((S(cfg, x - r * 0.5), S(cfg, y + r * 0.3), S(cfg, x + r * 0.6), S(cfg, y + r * 0.3)),
               fill=ink_rgba(cfg, 90), width=1)
    # fallen log on the left of the path
    d.line((S(cfg, 830), S(cfg, 2162), S(cfg, 965), S(cfg, 2186)),
           fill=ink_rgba(cfg, 205), width=11)
    for k in range(7):
        t = k / 6
        x = 830 + 135 * t
        y = 2162 + 24 * t
        d.line((S(cfg, x), S(cfg, y - 9), S(cfg, x + 4), S(cfg, y + 7)),
               fill=ink_rgba(cfg, 130), width=2)
    d.line((S(cfg, 912), S(cfg, 2178), S(cfg, 918), S(cfg, 2158)),
           fill=ink_rgba(cfg, 190), width=5)
    d.line((S(cfg, 952), S(cfg, 2176), S(cfg, 966), S(cfg, 2160)),
           fill=ink_rgba(cfg, 190), width=4)
    stack.append(L)
    return stack


# ----------------------------------------------------------------------------
# mist bands between the trunks
# ----------------------------------------------------------------------------
def layer_mist(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 66)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    for y, a, count in ((1545, 34, 46), (1640, 40, 52), (1735, 46, 56), (1830, 30, 40)):
        for _ in range(count):
            ln = rng.uniform(110, 320)
            x0 = rng.choice([rng.uniform(140, 900), rng.uniform(1300, 2060)])
            yj = rng.uniform(-16, 16)
            d.line((S(cfg, x0 - ln), S(cfg, y + yj), S(cfg, x0 + ln), S(cfg, y + yj * 0.4)),
                   fill=ink_rgba(cfg, int(rng.uniform(a * 0.7, a * 1.4))), width=2)
    return L


# ----------------------------------------------------------------------------
# the two figures — Virgil pointing, Dante following
# ----------------------------------------------------------------------------
def layer_figures(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    s = S(cfg, 1.0)

    # ---- Virgil (in profile, facing the light, arm raised pointing) ----
    vx, vy = S(cfg, 1006), S(cfg, 2032)
    # toga body
    d.polygon([(vx - 26 * s, vy), (vx + 24 * s, vy),
               (vx + 20 * s, vy - 52 * s), (vx + 14 * s, vy - 92 * s),
               (vx - 2 * s, vy - 118 * s), (vx - 16 * s, vy - 100 * s),
               (vx - 22 * s, vy - 54 * s)],
              fill=ink_rgba(cfg, 240))
    # head in profile (facing right/up)
    d.ellipse((vx - 6 * s, vy - 132 * s, vx + 10 * s, vy - 116 * s),
              fill=ink_rgba(cfg, 240))
    # nose
    d.line((vx + 8 * s, vy - 126 * s, vx + 15 * s, vy - 122 * s),
           fill=ink_rgba(cfg, 200), width=2)
    # laurel wreath
    for k in range(5):
        t = -0.8 + k * 0.4
        d.ellipse((vx - 9 * s + k * 3.4 * s, vy - 137 * s, vx - 1 * s + k * 3.4 * s, vy - 127 * s),
                  outline=ink_rgba(cfg, 190), width=2)
    # raised arm pointing toward the glow (up-right)
    d.line((vx + 10 * s, vy - 96 * s, vx + 34 * s, vy - 138 * s),
           fill=ink_rgba(cfg, 240), width=7)
    d.line((vx + 34 * s, vy - 138 * s, vx + 52 * s, vy - 152 * s),
           fill=ink_rgba(cfg, 240), width=5)
    # toga folds
    for ox in (-12, -2, 8):
        d.line((vx + ox * s, vy - 108 * s, vx + ox * s * 0.7, vy - 16 * s),
               fill=ink_rgba(cfg, 140), width=2)
    # hem
    d.arc((vx - 28 * s, vy - 12 * s, vx + 28 * s, vy + 6 * s), 190, 350,
          fill=ink_rgba(cfg, 150), width=2)

    # ---- Dante (hooded, following, head bowed) ----
    dx_, dy_ = S(cfg, 1152), S(cfg, 2010)
    k = 0.9
    ds = S(cfg, k)
    d.polygon([(dx_ - 13 * ds, dy_), (dx_ + 12 * ds, dy_),
               (dx_ + 16 * ds, dy_ - 58 * ds), (dx_ + 10 * ds, dy_ - 98 * ds),
               (dx_ + 3 * ds, dy_ - 120 * ds), (dx_ - 8 * ds, dy_ - 120 * ds),
               (dx_ - 12 * ds, dy_ - 98 * ds), (dx_ - 15 * ds, dy_ - 56 * ds)],
              fill=ink_rgba(cfg, 238))
    d.ellipse((dx_ - 12 * ds, dy_ - 140 * ds, dx_ + 10 * ds, dy_ - 117 * ds),
              fill=ink_rgba(cfg, 238))
    for ox, lean in ((-7, -4), (0, -1), (7, 4)):
        d.line((dx_ + ox * ds, dy_ - 108 * ds, dx_ + ox * ds + lean * ds, dy_ - 14 * ds),
               fill=ink_rgba(cfg, 150), width=2)
    d.arc((dx_ - 18 * ds, dy_ - 13 * ds, dx_ + 20 * ds, dy_ + 4 * ds), 200, 340,
          fill=ink_rgba(cfg, 150), width=2)

    # rim light from the distant glow (upper-right of both figures)
    paper = cfg["paper"] + (255,)
    d.arc((vx - 8 * s, vy - 138 * s, vx + 12 * s, vy - 112 * s), 260, 350,
          fill=paper, width=2)
    d.line((vx + 18 * s, vy - 96 * s, vx + 24 * s, vy - 30 * s), fill=paper, width=2)
    d.arc((dx_ - 13 * ds, dy_ - 146 * ds, dx_ + 12 * ds, dy_ - 116 * ds), 250, 350,
          fill=paper, width=2)
    return L


# ----------------------------------------------------------------------------
# frame + caption (Canto II)
# ----------------------------------------------------------------------------
def draw_frame_and_caption2(img, cfg):
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
    d.text((cx, cy_sub), "CHANT DEUXIÈME — LA FORÊT OBSCURE", font=f_sub, fill=ink + (230,), anchor="mm")
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
    dark = make_darkmap2(cfg)
    dark_low = np.asarray(
        Image.fromarray((dark * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR),
        np.float32) / 255.0
    paper = make_paper(cfg, dark).convert("RGBA")
    size = (S(cfg, W), S(cfg, H))

    layers = []
    print("  sky ...")
    layers.append(layer_sky(cfg, dark_low))
    layers.append(layer_moon_erase(cfg))
    print("  treeline ...")
    layers.append(layer_treeline(cfg))
    print("  ground ...")
    layers += layer_ground(cfg)
    print("  path ...")
    layers += layer_path(cfg)
    print("  trees ...")
    layers.append(layer_trees(cfg))
    print("  mist ...")
    layers.append(layer_mist(cfg))
    print("  figures ...")
    layers.append(layer_figures(cfg))

    print("  compositing ...")
    ink = Image.new("RGBA", size, (0, 0, 0, 0))
    for L in layers:
        ink = Image.alpha_composite(ink, L)
    ink_final = ink.resize((W, H), Image.LANCZOS)
    img = Image.alpha_composite(paper, ink_final)
    img = draw_frame_and_caption2(img, cfg)
    img = img.convert("RGB")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    path = os.path.join(cfg["out_dir"], out_name)
    img.save(path, "PNG")
    prev = img.copy()
    prev.thumbnail((1000, 1000), Image.LANCZOS)
    prev.save(os.path.join(cfg["out_dir"], "preview_" + out_name.replace(".png", ".jpg")),
              "JPEG", quality=92)
    print(f"  saved {path}")
    return img


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "c2_v01.png"
    render(CFG, name)
