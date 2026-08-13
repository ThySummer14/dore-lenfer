#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procedural engraving engine in the manner of Gustave Doré — v2.
Scene: "DANTE — L'ENFER · LA LUMIÈRE DIVINE"
A pilgrim on a rocky spur gazes into a vortex of divine light above a dark gulf.

All scene coordinates are final-plate coords; ink is drawn at SS supersampling.
"""
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

CFG = dict(
    W=2200, H=2860,
    SS=2,
    seed=42,
    paper=(241, 232, 213),
    ink=(30, 23, 16),
    glow=(1100, 700),
    r_core=175,
    r_max=1520,
    ring_r0=235,
    ring_k=0.0112,
    frame=(0.055, 0.035, 0.945, 0.905),
    caption_top=0.775,
    out_dir="out",
)

# ----------------------------------------------------------------------------
# math / noise
# ----------------------------------------------------------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def fbm_arr(h, w, seed, base_res, octaves=4):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp = 1.0
    tot = 0.0
    f = base_res
    for o in range(octaves):
        gh = max(2, int(round(h / f)))
        gw = max(2, int(round(w / f)))
        g = rng.random((gh + 1, gw + 1)).astype(np.float32)
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        acc += np.asarray(im, np.float32) / 255.0 * amp
        tot += amp
        amp *= 0.5
        f = max(2.0, f / 2.0)
    return acc / tot


def bilinear(arr, x, y):
    h, w = arr.shape
    x = clamp(x, 0, w - 1.001)
    y = clamp(y, 0, h - 1.001)
    x0 = int(x)
    y0 = int(y)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    a = arr[y0, x0] * (1 - fx) + arr[y0, x1] * fx
    b = arr[y1, x0] * (1 - fx) + arr[y1, x1] * fx
    return float(a * (1 - fy) + b * fy)


# ----------------------------------------------------------------------------
# tone field + paper
# ----------------------------------------------------------------------------
def make_darkmap(cfg, ss_div=4):
    W, H = cfg["W"], cfg["H"]
    hw, hh = W // ss_div, H // ss_div
    cx, cy = cfg["glow"]
    y, x = np.mgrid[0:hh, 0:hw].astype(np.float32)
    x = x * ss_div
    y = y * ss_div
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ang = np.degrees(np.arctan2(x - cx, y - cy))  # 0 = straight down
    n = fbm_arr(hh, hw, cfg["seed"] + 7, 40, 4)
    base = smooth01((r - 300.0) / 950.0)
    base = base * (0.58 + 0.42 * n)
    openf = smooth01(np.clip((np.abs(ang) - 4.0) / 14.0, 0, 1))
    base = base * (0.58 + 0.42 * openf)
    # light corridor dies before the gulf; sky near the horizon darkens (silhouette)
    fade = 1.0 - smooth01((y - 1300.0) / 430.0)
    base = base * (0.44 + 0.56 * fade)
    glow_core = (0.11 * np.exp(-((r / 235.0) ** 2)) + 0.04 * np.exp(-((r / 600.0) ** 2)))
    glow_core *= 0.5 + 0.5 * np.exp(-((ang / 15.0) ** 2))
    d = np.clip(base - glow_core * 0.45, 0.0, 1.0)
    im = Image.fromarray((d * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0


def make_paper(cfg, dark):
    W, H = cfg["W"], cfg["H"]
    paper = np.asarray(cfg["paper"], np.float32).reshape(1, 1, 3)
    shade = np.clip(1.0 - 0.16 * dark, 0.56, 1.05)
    img = paper * shade[..., None]
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / (W * 0.62)) ** 2 + ((y - H / 2) / (H * 0.62)) ** 2)
    vig = 1.0 - 0.08 * np.clip(r, 0, 1.4) ** 2.0
    img *= vig[..., None]
    grain = fbm_arr(H // 2, W // 2, cfg["seed"] + 11, 90, 3)
    grain = Image.fromarray((grain * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    g = (np.asarray(grain, np.float32) / 255.0 - 0.5) * 8.0
    img += g[..., None]
    fib = fbm_arr(H // 4, W // 4, cfg["seed"] + 17, 60, 2)
    fib = Image.fromarray((fib * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    fb = (np.asarray(fib, np.float32) / 255.0 - 0.5) * 4.0
    img += fb[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


# ----------------------------------------------------------------------------
# stroke primitives (SS space)
# ----------------------------------------------------------------------------
def S(cfg, v):
    return v * cfg["SS"]


def ink_rgba(cfg, a=255):
    return cfg["ink"] + (a,)


def strokes_layer(size, color, mask, fn):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    fn(d)
    if mask is not None:
        m = ImageChops.multiply(m, mask)
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    out.paste(color + (255,), (0, 0), m)
    return out


def poly_mask(size, pts):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return m


def band_mask(size, y0, y1):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle([0, y0, size[0], y1], fill=255)
    return m


def grad_mask(size, y0, y1, a0, a1):
    rows = np.linspace(a0, a1, max(1, y1 - y0)).astype(np.uint8)
    arr = np.zeros((size[1], size[0]), np.uint8)
    arr[y0:y1, :] = rows[:, None]
    return Image.fromarray(arr)


def draw_hatch(d, poly, angle_deg, spacing, rng, jitter=0.0, width=2, dash=None, margin=100):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ang = math.radians(angle_deg)
    dx, dy = math.cos(ang), math.sin(ang)
    px, py = -dy, dx
    span = math.hypot(x1 - x0, y1 - y0) / 2.0 + margin
    t = -span
    while t <= span:
        bx, by = cx + t * px, cy + t * py
        ax, ay = bx - dx * span, by - dy * span
        bx2, by2 = bx + dx * span, by + dy * span
        if jitter > 0:
            j1 = rng.uniform(-jitter, jitter)
            j2 = rng.uniform(-jitter, jitter)
            ax += px * j1
            ay += py * j1
            bx2 += px * j2
            by2 += py * j2
        if dash:
            L = math.hypot(bx2 - ax, by2 - ay)
            s = 0.0
            while s < L:
                e = min(s + dash, L)
                d.line((ax + dx * s, ay + dy * s, ax + dx * e, ay + dy * e),
                       fill=255, width=width)
                s = e + dash * 0.55
        else:
            d.line((ax, ay, bx2, by2), fill=255, width=width)
        t += spacing


def contour_strokes(d, poly, count, shrink=0.035, width=2):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    for i in range(1, count + 1):
        f = 1.0 - shrink * i
        if f <= 0.1:
            break
        pts = [(cx + (px - cx) * f, cy + (py - cy) * f) for px, py in poly]
        d.line(pts + [pts[0]], fill=255, width=width)


# ----------------------------------------------------------------------------
# scene geometry (final coords)
# ----------------------------------------------------------------------------
def scene(cfg):
    W, H = cfg["W"], cfg["H"]
    return dict(
        left_cliff=[(121, 2230), (138, 1960), (176, 1810), (238, 1710), (316, 1685),
                    (372, 1740), (402, 1825), (392, 1925), (360, 2045), (322, 2160),
                    (280, 2230), (121, 2230)],
        right_rock=[(2080, 2230), (2058, 2020), (2008, 1940), (1932, 1908), (1860, 1918),
                    (1818, 1980), (1842, 2065), (1888, 2150), (1942, 2230), (2080, 2230)],
        spur=[(972, 2230), (1032, 2170), (1088, 2142), (1160, 2144), (1232, 2180),
              (1288, 2230), (972, 2230)],
        horizon=[(760, 1790), (810, 1798), (1220, 1796), (1280, 1790), (1160, 1812), (800, 1812)],
        towers=[(812, 1762, 26), (918, 1744, 38), (1042, 1756, 30), (1190, 1770, 22)],
        pilgrim=(1132, 2142),
        tree_root=(310, 1712),
    )


# ----------------------------------------------------------------------------
# ink layers
# ----------------------------------------------------------------------------
def layer_vortex(cfg, dark_low, n_low):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    rng = random.Random(cfg["seed"] + 101)
    cx, cy = cfg["glow"]
    ex, ey = 1.0, 1.24
    rot = math.radians(-6.0)
    i = 0
    while True:
        r = cfg["ring_r0"] * math.exp(cfg["ring_k"] * i)
        if r > cfg["r_max"]:
            break
        phase = i * 0.23
        N = 300
        pts = []
        for j in range(N + 1):
            th = 2 * math.pi * j / N
            px = cx + r * math.cos(th + rot) * ex
            py = cy + r * math.sin(th + rot) * ey
            nr = bilinear(n_low, px / 4.0, py / 4.0)
            arm = 1.0 + 0.05 * math.cos(2 * (th + phase)) + 0.028 * math.cos(3 * th + phase)
            rr = r * arm * (1.0 + 0.05 * (nr - 0.5))
            pts.append((S(cfg, cx + rr * math.cos(th + rot) * ex),
                        S(cfg, cy + rr * math.sin(th + rot) * ey)))
        dr = bilinear(dark_low, cx / 4.0, cy / 4.0 + r / 4.0)
        ratio = (r - cfg["ring_r0"]) / (cfg["r_max"] - cfg["ring_r0"])
        base = (60 + 195 * ratio ** 1.15) * (0.62 + 0.38 * dr)
        a = int(clamp(base, 0, 255))
        if a < 50 or r < 290:
            i += 1
            continue
        gapw = 0.16 + 0.20 * bilinear(n_low, cx / 4.0, (cy + r * 0.5) / 4.0)
        gapdir = (1.2 + i * 0.17) % (2 * math.pi)
        ga = gapdir + gapw
        gb = gapdir + 2 * math.pi - gapw
        width = 3 if r > 0.5 * cfg["r_max"] else 2
        # corridor of light in ring-param space: final-space |angle| <= 20 deg
        c0 = (2 * math.pi - 0.244)
        c1 = 0.454
        cur = []
        for j in range(N + 1):
            th = 2 * math.pi * j / N
            is_gap = (th >= ga and th <= gb) if ga <= gb else (th >= ga or th <= gb)
            if is_gap or j == N:
                if len(cur) > 2:
                    t0 = 2 * math.pi * (j - len(cur)) / N
                    t1 = 2 * math.pi * (j - 1) / N
                    tmid = (t0 + t1) / 2
                    in_corr = (tmid >= c0 or tmid <= c1)
                    aa = int(a * 0.7) if in_corr else a
                    d.line(cur, fill=ink_rgba(cfg, aa), width=width)
                cur = []
            if not is_gap and j < N:
                cur.append(pts[j])
        i += 1
    # halo ringlets inside the glow core
    for hr, ha in ((58, 60), (92, 45), (128, 32)):
        for k, (g0, g1) in enumerate(((0.2, 1.5), (1.8, 3.4), (4.0, 5.6))):
            pts = [(S(cfg, cx + hr * math.cos(t)), S(cfg, cy + hr * math.sin(t) * ey))
                   for t in np.linspace(g0, g1, 24)]
            d.line(pts, fill=ink_rgba(cfg, ha), width=2)
    # tangential slivers — cloud texture in the dark sky
    for _ in range(11000):
        th = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(500, cfg["r_max"])
        px = cx + r * math.cos(th + rot) * ex
        py = cy + r * math.sin(th + rot) * ey
        ang_deg = math.degrees(math.atan2(px - cx, py - cy))
        in_corr = abs(ang_deg) < 20 and r < 1100
        dr = bilinear(dark_low, px / 4.0, py / 4.0)
        a = int(clamp((16 if in_corr else 90) + 160 * dr, 10, 200))
        ln = rng.uniform(16, 60)
        tx = -math.sin(th + rot) * ey
        ty = math.cos(th + rot) * ex
        tl = math.hypot(tx, ty) or 1.0
        tx, ty = tx / tl * ln, ty / tl * ln
        wob = rng.uniform(-0.5, 0.5)
        d.line((S(cfg, px - tx), S(cfg, py - ty), S(cfg, px + tx + wob * ln), S(cfg, py + ty + wob * ln)),
               fill=ink_rgba(cfg, a), width=2)
    return L


def layer_rays(cfg, dark_low):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    rng = random.Random(cfg["seed"] + 202)
    cx, cy = cfg["glow"]
    for _ in range(140):
        ang = math.radians(rng.uniform(46, 134))
        bend = rng.uniform(-70, 70)
        a = rng.uniform(30, 60)
        r_start = 260 + rng.uniform(0, 170)
        segs = 6
        pts = []
        for k in range(segs + 1):
            t = k / segs
            r = r_start + t * 1850
            off = bend * math.sin(t * math.pi) * (r / 800.0)
            px = cx + r * math.sin(ang) + off * math.cos(ang)
            py = cy + r * math.cos(ang) - off * math.sin(ang)
            pts.append((px, py))
        for k in range(segs):
            dr = bilinear(dark_low, pts[k][0] / 4.0, pts[k][1] / 4.0)
            r0 = math.hypot(pts[k][0] - cx, pts[k][1] - cy)
            fade_in = smooth01((r0 - 150) / 350.0)
            taper = 1.0 - 0.30 * (k / segs)
            aa = int(clamp(a * taper * (1.0 - 0.7 * dr) * fade_in, 3, 255))
            if aa >= 4:
                d.line((S(cfg, pts[k][0]), S(cfg, pts[k][1]),
                        S(cfg, pts[k + 1][0]), S(cfg, pts[k + 1][1])),
                       fill=ink_rgba(cfg, aa), width=2)
    for _ in range(16):
        ang = math.radians(rng.uniform(62, 118))
        r0, r1 = 245, 2250
        bend = rng.uniform(-40, 40)
        midx = cx + (r0 + r1) / 2 * math.sin(ang) + bend
        midy = cy + (r0 + r1) / 2 * math.cos(ang)
        d.line((S(cfg, cx + r0 * math.sin(ang)), S(cfg, cy + r0 * math.cos(ang)),
                S(cfg, midx), S(cfg, midy)),
               fill=ink_rgba(cfg, 10), width=2)
        d.line((S(cfg, midx), S(cfg, midy),
                S(cfg, cx + r1 * math.sin(ang)), S(cfg, cy + r1 * math.cos(ang))),
               fill=ink_rgba(cfg, 8), width=2)
    # illuminated cloud wisps inside the corridor of light
    for _ in range(380):
        th = rng.uniform(-0.22, 0.22)          # near the downward axis
        r0 = rng.uniform(350, 1250)
        bend = rng.uniform(-0.16, 0.16)
        seg = 4
        pts = []
        for k in range(seg + 1):
            t = k / seg
            r = r0 + t * rng.uniform(220, 900)
            a_eff = th + bend * math.sin(t * math.pi)
            pts.append((cx + r * math.sin(a_eff), cy + r * math.cos(a_eff)))
        a = rng.uniform(20, 42)
        for k in range(seg):
            d.line((S(cfg, pts[k][0]), S(cfg, pts[k][1]),
                    S(cfg, pts[k + 1][0]), S(cfg, pts[k + 1][1])),
                   fill=ink_rgba(cfg, int(a)), width=2)
    # fine radiating sparkle inside the sun core
    for _ in range(90):
        th = rng.uniform(0, 2 * math.pi)
        r0 = rng.uniform(30, 155)
        ln = rng.uniform(24, 66)
        d.line((S(cfg, cx + r0 * math.sin(th)), S(cfg, cy + r0 * math.cos(th)),
                S(cfg, cx + (r0 + ln) * math.sin(th)), S(cfg, cy + (r0 + ln) * math.cos(th))),
               fill=ink_rgba(cfg, int(rng.uniform(24, 44))), width=2)
    return L


def layer_cliffs(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 303)
    g = scene(cfg)
    noise = Image.fromarray((fbm_arr(H // 4, W // 4, cfg["seed"] + 31, 26, 3) * 255)
                            .astype(np.uint8)).resize(size, Image.BILINEAR)
    noise = noise.point(lambda v: int(255 * (0.68 + 0.36 * v / 255.0)))
    stack = []
    for name in ("left_cliff", "right_rock"):
        poly = g[name]
        sp = [(S(cfg, x), S(cfg, y)) for x, y in poly]
        m = poly_mask(size, sp)
        m = ImageChops.multiply(m, noise)
        top = min(y for _, y in sp)
        bottom = max(y for _, y in sp)
        # base hatch
        stack.append(strokes_layer(size, cfg["ink"], m, lambda d, p=sp:
                     draw_hatch(d, p, 24, 4.2, rng, jitter=2.0, width=2)))
        # shadow: gradient cross-hatch, strongest at the bottom
        gm = grad_mask(size, int(top + (bottom - top) * 0.40), bottom, 55, 245)
        gm = ImageChops.multiply(gm, m)
        stack.append(strokes_layer(size, cfg["ink"], gm, lambda d, p=sp:
                     draw_hatch(d, p, 114, 6.0, rng, jitter=3.0, width=2)))
        # dashes for rock texture
        stack.append(strokes_layer(size, cfg["ink"], m, lambda d, p=sp:
                     draw_hatch(d, p, 74, 9, rng, jitter=4.0, width=2, dash=24)))
        # contour form-lines
        stack.append(strokes_layer(size, cfg["ink"], m, lambda d, p=sp:
                     contour_strokes(d, p, 7, shrink=0.04, width=2)))
        # crest roughness
        crest = [pt for pt in sp if pt[1] <= min(y for _, y in sp) + S(cfg, 110)]
        stack.append(strokes_layer(size, cfg["ink"], m, lambda d, c=crest:
                     [d.line((x + rng.uniform(-8, 8), y - rng.uniform(4, 18),
                              x + rng.uniform(-8, 8), y + rng.uniform(8, 38)), fill=255, width=2)
                      for x, y in c for _ in range(3)]))
    # rocky spur
    sp = [(S(cfg, x), S(cfg, y)) for x, y in g["spur"]]
    m = ImageChops.multiply(poly_mask(size, sp), noise)
    stack.append(strokes_layer(size, cfg["ink"], m, lambda d, p=sp:
                 draw_hatch(d, p, 30, 4.0, rng, jitter=2.0, width=2)))
    gm = grad_mask(size, S(cfg, 2160), S(cfg, 2230), 60, 210)
    gm = ImageChops.multiply(gm, m)
    stack.append(strokes_layer(size, cfg["ink"], gm, lambda d, p=sp:
                 draw_hatch(d, p, 116, 8, rng, jitter=3.0, width=2)))
    stack.append(strokes_layer(size, cfg["ink"], m, lambda d, p=sp:
                 contour_strokes(d, p, 4, shrink=0.05, width=2)))
    return stack


def layer_gulf(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 404)
    g = scene(cfg)
    m = band_mask(size, S(cfg, 1750), S(cfg, 2230))
    md = ImageDraw.Draw(m)
    for name in ("left_cliff", "right_rock", "spur"):
        md.polygon([(S(cfg, x), S(cfg, y)) for x, y in g[name]], fill=0)
    md.polygon([(S(cfg, x), S(cfg, y)) for x, y in g["horizon"]], fill=235)
    depth = grad_mask(size, S(cfg, 1750), S(cfg, 2230), 150, 255)
    m = ImageChops.multiply(m, depth)
    rect = [(S(cfg, 121), S(cfg, 1750)), (S(cfg, 2080), S(cfg, 1750)),
            (S(cfg, 2080), S(cfg, 2230)), (S(cfg, 121), S(cfg, 2230))]
    stack = [strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 92, 3.0, rng, jitter=2.4, width=2)),
             strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 4, 14, rng, jitter=6.0, width=1, dash=38))]
    t = Image.new("RGBA", size, (0, 0, 0, 0))
    td = ImageDraw.Draw(t)
    for tx, ty, th in g["towers"]:
        tx, ty, th = S(cfg, tx), S(cfg, ty), S(cfg, th)
        td.polygon([(tx - 5, ty), (tx + 5, ty), (tx + 3, ty - th), (tx - 3, ty - th)],
                   fill=ink_rgba(cfg, 235))
        td.polygon([(tx - 7, ty - th), (tx + 7, ty - th), (tx, ty - th - 13)],
                   fill=ink_rgba(cfg, 235))
    stack.append(t)
    return stack


def layer_chains(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]
    for (x0, y0, x1, y1) in [(1832, 2000, 1788, 2162), (1866, 1990, 1828, 2162)]:
        seg = 36
        pts = []
        for k in range(seg + 1):
            t = k / seg
            px = x0 + (x1 - x0) * t + math.sin(t * 7.3) * 9
            py = y0 + (y1 - y0) * t
            pts.append((S(cfg, px), S(cfg, py)))
        # paper rim so the chain reads against the dark gulf
        d.line([(x + 3, y + 3) for x, y in pts], fill=paper + (150,), width=3)
        d.line(pts, fill=ink_rgba(cfg, 220), width=2)
        for k in range(0, seg, 2):
            px, py = pts[k]
            nx, ny = pts[min(k + 1, seg)]
            dx, dy = nx - px, ny - py
            ln = math.hypot(dx, dy) or 1.0
            d.line((px - dy / ln * 8, py + dx / ln * 8,
                    px + dy / ln * 8, py - dx / ln * 8),
                   fill=ink_rgba(cfg, 195), width=3)
    return L


def layer_pilgrim(cfg):
    """Tiny hooded pilgrim seen from behind, staff in hand. Final coords scaled by SS."""
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    fx, fy = cfg["_pilgrim"]
    k = 1.35  # figure scale
    fx, fy = S(cfg, fx), S(cfg, fy)
    s = S(cfg, k)
    ink = cfg["ink"]
    # cast shadow streak on the rock (toward viewer-left)
    d.polygon([(fx - 30 * s, fy + 2 * s), (fx + 26 * s, fy + 2 * s),
               (fx + 10 * s, fy + 9 * s), (fx - 44 * s, fy + 9 * s)],
              fill=ink_rgba(cfg, 110))
    # cloak, blown slightly right
    d.polygon([(fx - 15 * s, fy), (fx + 13 * s, fy),
               (fx + 19 * s, fy - 64 * s), (fx + 12 * s, fy - 108 * s),
               (fx + 4 * s, fy - 132 * s), (fx - 8 * s, fy - 132 * s),
               (fx - 13 * s, fy - 108 * s), (fx - 17 * s, fy - 62 * s)],
              fill=ink_rgba(cfg, 238))
    # hood
    d.ellipse((fx - 13 * s, fy - 155 * s, fx + 12 * s, fy - 129 * s),
              fill=ink_rgba(cfg, 238))
    # face shadow (turned away, profile hint)
    d.ellipse((fx + 3 * s, fy - 148 * s, fx + 10 * s, fy - 135 * s),
              fill=ink_rgba(cfg, 120))
    # cloak folds
    for ox, lean in ((-8, -5), (0, -1), (8, 4)):
        d.line((fx + ox * s, fy - 122 * s, fx + ox * s + lean * s, fy - 16 * s),
               fill=ink_rgba(cfg, 155), width=2)
    # hem
    d.arc((fx - 21 * s, fy - 15 * s, fx + 23 * s, fy + 5 * s), 200, 340,
          fill=ink_rgba(cfg, 155), width=2)
    # staff
    d.line((fx + 28 * s, fy - 94 * s, fx + 43 * s, fy + 5 * s),
           fill=ink_rgba(cfg, 215), width=3)
    # knob + hand
    d.ellipse((fx + 25 * s, fy - 100 * s, fx + 32 * s, fy - 92 * s),
              fill=ink_rgba(cfg, 215))
    d.ellipse((fx + 23 * s, fy - 97 * s, fx + 31 * s, fy - 89 * s),
              fill=ink_rgba(cfg, 130))
    return L


def layer_birds(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 606)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    for _ in range(13):
        x = S(cfg, rng.uniform(860, 1420))
        y = S(cfg, rng.uniform(620, 1420))
        s = S(cfg, rng.uniform(7, 16))
        a = int(rng.uniform(100, 190))
        d.arc((x - s, y - s, x + s, y + s), 200, 310, fill=ink_rgba(cfg, a), width=2)
        d.arc((x, y - s, x + 2 * s, y + s), 230, 340, fill=ink_rgba(cfg, a), width=2)
    return L


def layer_tree(cfg):
    """Dead, gnarled tree on the left cliff crest — Doré's favourite gothic motif."""
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 707)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)

    def branch(x, y, ang, ln, w, depth, alpha=252):
        if depth == 0 or ln < 12:
            return
        x2 = x + math.cos(ang) * ln
        y2 = y + math.sin(ang) * ln
        d.line((S(cfg, x), S(cfg, y), S(cfg, x2), S(cfg, y2)),
               fill=ink_rgba(cfg, alpha), width=max(2, int(S(cfg, w))))
        offs = [-0.55 + 0.22 * rng.random(), 0.45 + 0.3 * rng.random()]
        for k, off in enumerate(offs):
            na = ang + off + (rng.random() - 0.5) * 0.3
            nl = ln * (0.68 - 0.08 * k)
            branch(x2, y2, na, nl, w * 0.62, depth - 1, alpha - 10)

    rx, ry = cfg["_tree_root"]
    # trunk leaning right over the gulf
    branch(rx, ry, -1.42, 160, 7.5, 6)
    # second trunk
    branch(rx + 6, ry + 4, -2.35, 78, 5.0, 5)
    # roots
    for ra in (-2.9, 2.9):
        d.line((S(cfg, rx), S(cfg, ry),
                S(cfg, rx + math.cos(ra) * 26), S(cfg, ry + math.sin(ra) * 18)),
               fill=ink_rgba(cfg, 230), width=4)
    return L


def layer_rimlight(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    g = scene(cfg)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"] + (255,)
    for poly in (g["left_cliff"], g["right_rock"]):
        sp = [(S(cfg, x) - 3, S(cfg, y) - 3) for x, y in poly]
        d.line(sp + [sp[0]], fill=paper, width=2)
    fx, fy = cfg["_pilgrim"]
    s = S(cfg, 1.35)
    d.arc((S(cfg, fx) - 15 * s, S(cfg, fy) - 162 * s, S(cfg, fx) + 14 * s, S(cfg, fy) - 126 * s),
          190, 340, fill=paper, width=2)
    # lit shoulder (light falls from above)
    d.arc((S(cfg, fx) - 2 * s, S(cfg, fy) - 118 * s, S(cfg, fx) + 24 * s, S(cfg, fy) - 84 * s),
          250, 320, fill=paper, width=2)
    # pool of light on the rock around the pilgrim
    for pr, pa in ((100, 20), (72, 30), (48, 38), (28, 46)):
        d.ellipse((S(cfg, fx) - pr * s, S(cfg, fy - 12) - pr * s * 0.55,
                   S(cfg, fx) + pr * s, S(cfg, fy - 12) + pr * s * 0.55),
                  outline=paper, width=2)
        d.ellipse((S(cfg, fx) - pr * s + 4, S(cfg, fy - 12) - pr * s * 0.55 + 3,
                   S(cfg, fx) + pr * s - 4, S(cfg, fy - 12) + pr * s * 0.55 - 3),
                  outline=cfg["paper"] + (pa,), width=2)
    return L


# ----------------------------------------------------------------------------
# frame + caption
# ----------------------------------------------------------------------------
def find_font(size, bold=False):
    cands = [
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Baskerville.ttc",
        "/System/Library/Fonts/Supplemental/Didot.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Songti.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ]
    for c in cands:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def draw_frame_and_caption(img, cfg):
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
    d.text((cx, cy_sub), "CHANT PREMIER — LA LUMIÈRE DIVINE", font=f_sub, fill=ink + (230,), anchor="mm")
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
    dark = make_darkmap(cfg)
    dark_low = np.asarray(
        Image.fromarray((dark * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR),
        np.float32) / 255.0
    n_low = fbm_arr(H // 4, W // 4, cfg["seed"] + 21, 30, 4)
    g = scene(cfg)
    cfg["_pilgrim"] = g["pilgrim"]
    cfg["_tree_root"] = g["tree_root"]
    paper = make_paper(cfg, dark).convert("RGBA")
    size = (S(cfg, W), S(cfg, H))

    layers = []
    print("  vortex ...")
    layers.append(layer_vortex(cfg, dark_low, n_low))
    print("  rays ...")
    layers.append(layer_rays(cfg, dark_low))
    print("  gulf + ridge ...")
    layers += layer_gulf(cfg)
    print("  cliffs ...")
    layers += layer_cliffs(cfg)
    print("  chains ...")
    layers.append(layer_chains(cfg))
    print("  pilgrim ...")
    layers.append(layer_pilgrim(cfg))
    print("  tree ...")
    layers.append(layer_tree(cfg))
    print("  birds ...")
    layers.append(layer_birds(cfg))
    print("  rim light ...")
    layers.append(layer_rimlight(cfg))

    print("  compositing ...")
    ink = Image.new("RGBA", size, (0, 0, 0, 0))
    for L in layers:
        ink = Image.alpha_composite(ink, L)
    ink_final = ink.resize((W, H), Image.LANCZOS)
    img = Image.alpha_composite(paper, ink_final)
    img = draw_frame_and_caption(img, cfg)
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
    name = sys.argv[1] if len(sys.argv) > 1 else "v02.png"
    render(CFG, name)
