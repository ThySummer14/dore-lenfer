#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procedural engraving engine in the manner of Gustave Doré — Canto III.
Scene: "DANTE — L'ENFER · CHANT TROISIÈME · LA PORTE DE L'ENFER"
The gate of Hell: a monumental rock arch, the inscription carved on the lintel,
a dim glow from within, Virgil pointing at the words, Dante reading them.

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
    seed=13,
    paper=(241, 232, 213),
    ink=(30, 23, 16),
    arch_c=(1100, 1750),        # springline centre
    arch_rx=430,
    arch_ry=470,
    lintel=(620, 1230, 1580, 1300),
    glow=(1100, 1880),          # dim light deep inside the gate
    frame=(0.055, 0.035, 0.945, 0.905),
    caption_top=0.775,
    out_dir="DORE_INFERNO/Chant_III_La_Porte_de_l_Enfer",
)


# ----------------------------------------------------------------------------
# tone field — all rock, a weak glow inside the gate
# ----------------------------------------------------------------------------
def make_darkmap3(cfg, ss_div=4):
    W, H = cfg["W"], cfg["H"]
    hw, hh = W // ss_div, H // ss_div
    cx, cy = cfg["glow"]
    y, x = np.mgrid[0:hh, 0:hw].astype(np.float32)
    x = x * ss_div
    y = y * ss_div
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    n = fbm_arr(hh, hw, cfg["seed"] + 5, 30, 4)
    base = 0.60 + 0.14 * n + 0.06 * smooth01((y - 2050.0) / 200.0)
    glow = (0.30 * np.exp(-((r / 260.0) ** 2)) + 0.10 * np.exp(-((r / 520.0) ** 2)))
    d = np.clip(base - glow, 0.0, 1.0)
    im = Image.fromarray((d * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def arch_points(cfg, n=90):
    """Arch curve from left springline over the crown to the right springline."""
    cx, cy = cfg["arch_c"]
    pts = []
    for k in range(n + 1):
        th = math.pi + math.pi * k / n
        pts.append((cx + cfg["arch_rx"] * math.cos(th), cy + cfg["arch_ry"] * math.sin(th)))
    return pts


def opening_polygon(cfg):
    cx, cy = cfg["arch_c"]
    poly = arch_points(cfg)
    poly += [(cx + cfg["arch_rx"], 2230), (cx - cfg["arch_rx"], 2230)]
    return poly


# ----------------------------------------------------------------------------
# the rock mass around the gate
# ----------------------------------------------------------------------------
def layer_rock(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 71)
    m = band_mask(size, S(cfg, 100), S(cfg, 2230))
    md = ImageDraw.Draw(m)
    md.polygon([(S(cfg, x), S(cfg, y)) for x, y in opening_polygon(cfg)], fill=0)
    # depth: darker toward the bottom and the outer edges
    depth = Image.fromarray(
        (np.clip((np.linspace(0.85, 1.0, size[1]))[:, None]
                 * (1.0 - 0.50 * np.clip(np.abs(np.linspace(-1, 1, size[0])) ** 1.4, 0, 1)[None, :])
                 * 255, 0, 255)).astype(np.uint8))
    m = ImageChops.multiply(m, depth)
    noise = Image.fromarray((fbm_arr(H // 4, W // 4, cfg["seed"] + 37, 24, 3) * 255)
                            .astype(np.uint8)).resize(size, Image.BILINEAR)
    noise = noise.point(lambda v: int(255 * (0.85 + 0.22 * v / 255.0)))
    m = ImageChops.multiply(m, noise)
    rect = [(S(cfg, 121), S(cfg, 100)), (S(cfg, 2079), S(cfg, 100)),
            (S(cfg, 2079), S(cfg, 2230)), (S(cfg, 121), S(cfg, 2230))]
    stack = [strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 4, 5.4, rng, jitter=2.4, width=4)),
             strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 92, 9.5, rng, jitter=4.0, width=2, dash=30))]
    # strata emphasis lines, clipped to the rock mask
    sm = Image.new("L", size, 0)
    sd = ImageDraw.Draw(sm)
    for k in range(26):
        y = S(cfg, 140 + k * 78 + rng.uniform(-20, 20))
        amp = rng.uniform(4, 14)
        wav = rng.uniform(0.006, 0.014)
        pts = [(x, y + amp * math.sin(x * wav)) for x in range(S(cfg, 130), S(cfg, 2070), 30)]
        sd.line(pts, fill=255, width=2)
    sm = ImageChops.multiply(sm, m)
    s3 = Image.new("RGBA", size, (0, 0, 0, 0))
    s3.paste(cfg["ink"] + (130,), (0, 0), sm)
    stack.append(s3)
    # cracks
    cracks = Image.new("RGBA", size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(cracks)
    for _ in range(14):
        th = rng.uniform(math.pi * 0.15, math.pi * 0.85)
        rx0 = cfg["arch_c"][0] + (cfg["arch_rx"] + rng.uniform(10, 60)) * math.cos(th)
        ry0 = cfg["arch_c"][1] + (cfg["arch_ry"] + rng.uniform(10, 60)) * math.sin(th)
        ang = th + math.pi + rng.uniform(-0.3, 0.3)
        x, y = rx0, ry0
        pts = [(S(cfg, x), S(cfg, y))]
        for _ in range(6):
            ang += rng.uniform(-0.35, 0.35)
            x += math.cos(ang) * rng.uniform(26, 64)
            y += math.sin(ang) * rng.uniform(20, 52)
            pts.append((S(cfg, x), S(cfg, y)))
        cd.line(pts, fill=ink_rgba(cfg, 165), width=2)
    # drips of damp below the lintel and springline
    for _ in range(30):
        x = rng.uniform(760, 1440)
        ln = rng.uniform(20, 90)
        cd.line((S(cfg, x), S(cfg, rng.uniform(1330, 1600)),
                 S(cfg, x + rng.uniform(-6, 6)), S(cfg, rng.uniform(1330, 1600) + ln)),
                fill=ink_rgba(cfg, 90), width=2)
    stack.append(cracks)
    return stack


# ----------------------------------------------------------------------------
# arch voussoirs, lintel with the inscription, interior, floor, figures
# ----------------------------------------------------------------------------
def layer_arch_edge(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 72)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    cx, cy = cfg["arch_c"]
    # voussoir ticks radiating around the arch
    for k in range(0, 91, 5):
        th = math.pi + math.pi * k / 90
        r0 = cfg["arch_rx"] * cfg["arch_ry"] / math.hypot(cfg["arch_rx"] * math.sin(th),
                                                          cfg["arch_ry"] * math.cos(th))
        x0 = cx + math.cos(th) * cfg["arch_rx"] * (1 + 8 / r0)
        y0 = cy + math.sin(th) * cfg["arch_ry"] * (1 + 8 / r0)
        x1 = cx + math.cos(th) * cfg["arch_rx"] * (1 + 34 / r0)
        y1 = cy + math.sin(th) * cfg["arch_ry"] * (1 + 34 / r0)
        d.line((S(cfg, x0), S(cfg, y0), S(cfg, x1), S(cfg, y1)),
               fill=ink_rgba(cfg, 150), width=2)
    # arch outline (double)
    ap = [(S(cfg, x), S(cfg, y)) for x, y in arch_points(cfg)]
    d.line(ap, fill=ink_rgba(cfg, 225), width=4)
    inner = [(S(cfg, cx + (x - cx) * 0.96), S(cfg, cy + (y - cy) * 0.96)) for x, y in arch_points(cfg)]
    d.line(inner, fill=ink_rgba(cfg, 110), width=2)
    # springline sides
    for sx in (cx - cfg["arch_rx"], cx + cfg["arch_rx"]):
        d.line((S(cfg, sx), S(cfg, cy), S(cfg, sx), S(cfg, 2230)),
               fill=ink_rgba(cfg, 210), width=3)
    return L


def layer_lintel(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    x0, y0, x1, y1 = cfg["lintel"]
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    # lighter stone band
    d.rectangle((S(cfg, x0), S(cfg, y0), S(cfg, x1), S(cfg, y1)),
                fill=cfg["paper"] + (150,))
    d.rectangle((S(cfg, x0), S(cfg, y0), S(cfg, x1), S(cfg, y1)),
                outline=ink_rgba(cfg, 220), width=3)
    d.rectangle((S(cfg, x0 + 8), S(cfg, y0 + 8), S(cfg, x1 - 8), S(cfg, y1 - 8)),
                outline=ink_rgba(cfg, 120), width=1)
    # carved inscription — three lines
    lines = [
        "PER ME SI VA NE LA CITTÀ DOLENTE,",
        "PER ME SI VA NE L'ETTERNO DOLORE,",
        "LASCIATE OGNE SPERANZA, VOI CH'INTRATE.",
    ]
    font = find_font(int((y1 - y0) * 0.30))
    cy0 = (y0 + y1) / 2 - (len(lines) - 1) * 8
    for k, text in enumerate(lines):
        yy = cy0 + k * 16
        total = sum(d.textlength(ch, font=font) for ch in text) + 1.5 * (len(text) - 1)
        xx = (x0 + x1) / 2 - total / 2
        for ch in text:
            d.text((S(cfg, xx), S(cfg, yy)), ch, font=font, fill=ink_rgba(cfg, 235))
            xx += d.textlength(ch, font=font) + 1.5
    # corner rosettes
    for cx_, cy_ in ((x0 + 26, y0 + 26), (x1 - 26, y0 + 26)):
        d.ellipse((S(cfg, cx_ - 5), S(cfg, cy_ - 5), S(cfg, cx_ + 5), S(cfg, cy_ + 5)),
                  outline=ink_rgba(cfg, 160), width=2)
    return L


def layer_interior(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 73)
    cx, cy = cfg["glow"]
    m = poly_mask(size, [(S(cfg, x), S(cfg, y)) for x, y in opening_polygon(cfg)])
    # vertical hatch, darker toward the opening edges
    prof = np.clip(0.65 + 0.35 * np.abs(np.linspace(-1, 1, size[0])) ** 2.0, 0, 1)
    depth = Image.fromarray(np.repeat((prof * 255).astype(np.uint8)[None, :], size[1], axis=0))
    depth = ImageChops.multiply(depth, m)
    rect = [(S(cfg, 670), S(cfg, 1280)), (S(cfg, 1530), S(cfg, 1280)),
            (S(cfg, 1530), S(cfg, 2230)), (S(cfg, 670), S(cfg, 2230))]
    stack = [strokes_layer(size, cfg["ink"], depth, lambda d, r=rect:
                           draw_hatch(d, r, 90, 4.6, rng, jitter=2.8, width=3)),
             strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 2, 13, rng, jitter=5.0, width=1, dash=36))]
    # the dim glow inside
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    for pr, pa in ((340, 70), (230, 120), (155, 165), (92, 200)):
        d.ellipse((S(cfg, cx - pr), S(cfg, cy - pr * 0.62), S(cfg, cx + pr), S(cfg, cy + pr * 0.62)),
                  fill=cfg["paper"] + (pa,))
    # faint radiance streaks
    for _ in range(22):
        th = rng.uniform(0, 2 * math.pi)
        r0 = rng.uniform(90, 240)
        ln = rng.uniform(120, 420)
        d.line((S(cfg, cx + r0 * math.sin(th)), S(cfg, cy + r0 * math.cos(th) * 0.62),
                S(cfg, cx + (r0 + ln) * math.sin(th)), S(cfg, cy + (r0 + ln) * math.cos(th) * 0.62)),
               fill=ink_rgba(cfg, 10), width=2)
    # hanging lamp at the crown
    lx, ly = cfg["arch_c"][0], 1375
    d.line((S(cfg, lx), S(cfg, ly), S(cfg, lx), S(cfg, ly + 46)),
           fill=ink_rgba(cfg, 200), width=2)
    d.ellipse((S(cfg, lx - 14), S(cfg, ly + 42), S(cfg, lx + 14), S(cfg, ly + 74)),
              outline=ink_rgba(cfg, 190), width=3)
    d.ellipse((S(cfg, lx - 6), S(cfg, ly + 52), S(cfg, lx + 6), S(cfg, ly + 64)),
              fill=cfg["paper"] + (200,))
    stack.append(L)
    return stack


def layer_floor(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 74)
    # floor runs through the doorway
    m = band_mask(size, S(cfg, 2060), S(cfg, 2230))
    depth = grad_mask(size, S(cfg, 2060), S(cfg, 2230), 130, 255)
    m = ImageChops.multiply(m, depth)
    stack = [strokes_layer(size, cfg["ink"], m, lambda d:
                           draw_hatch(d, [(S(cfg, 121), S(cfg, 2060)), (S(cfg, 2079), S(cfg, 2060)),
                                          (S(cfg, 2079), S(cfg, 2230)), (S(cfg, 121), S(cfg, 2230))],
                                      3, 6.6, rng, jitter=2.2, width=3))]
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    for _ in range(16):
        x = rng.uniform(150, 2050)
        y = rng.uniform(2090, 2225)
        r = rng.uniform(9, 26)
        d.ellipse((S(cfg, x - r), S(cfg, y - r * 0.5), S(cfg, x + r), S(cfg, y + r * 0.5)),
                  outline=ink_rgba(cfg, 150), width=2)
        d.line((S(cfg, x - r * 0.4), S(cfg, y + r * 0.28), S(cfg, x + r * 0.5), S(cfg, y + r * 0.28)),
               fill=ink_rgba(cfg, 90), width=1)
    # small skull resting by a stone on the left
    sx, sy = 880, 2158
    d.ellipse((S(cfg, sx - 15), S(cfg, sy - 13), S(cfg, sx + 15), S(cfg, sy + 13)),
              outline=ink_rgba(cfg, 200), width=2)
    d.ellipse((S(cfg, sx - 9), S(cfg, sy + 9), S(cfg, sx + 9), S(cfg, sy + 25)),
              outline=ink_rgba(cfg, 190), width=2)
    d.ellipse((S(cfg, sx - 8), S(cfg, sy - 4), S(cfg, sx - 3), S(cfg, sy + 1)),
              fill=ink_rgba(cfg, 180))
    d.ellipse((S(cfg, sx + 3), S(cfg, sy - 4), S(cfg, sx + 8), S(cfg, sy + 1)),
              fill=ink_rgba(cfg, 180))
    stack.append(L)
    return stack


def layer_figures(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)

    # ---- Virgil: in profile, arm raised pointing at the inscription ----
    vx, vy = S(cfg, 1000), S(cfg, 2128)
    s = S(cfg, 1.05)
    d.polygon([(vx - 25 * s, vy), (vx + 22 * s, vy),
               (vx + 19 * s, vy - 50 * s), (vx + 13 * s, vy - 88 * s),
               (vx - 2 * s, vy - 112 * s), (vx - 15 * s, vy - 95 * s),
               (vx - 21 * s, vy - 52 * s)],
              fill=ink_rgba(cfg, 242))
    d.ellipse((vx - 6 * s, vy - 126 * s, vx + 9 * s, vy - 110 * s),
              fill=ink_rgba(cfg, 242))
    d.line((vx + 7 * s, vy - 120 * s, vx + 14 * s, vy - 116 * s),
           fill=ink_rgba(cfg, 200), width=2)
    for k in range(5):
        d.ellipse((vx - 8 * s + k * 3.2 * s, vy - 131 * s, vx - 1 * s + k * 3.2 * s, vy - 121 * s),
                  outline=ink_rgba(cfg, 190), width=2)
    # raised arm — steep, toward the lintel
    d.line((vx + 8 * s, vy - 92 * s, vx + 26 * s, vy - 150 * s),
           fill=ink_rgba(cfg, 242), width=7)
    d.line((vx + 26 * s, vy - 150 * s, vx + 40 * s, vy - 172 * s),
           fill=ink_rgba(cfg, 242), width=5)
    for ox in (-11, -2, 7):
        d.line((vx + ox * s, vy - 102 * s, vx + ox * s * 0.7, vy - 14 * s),
               fill=ink_rgba(cfg, 145), width=2)
    d.arc((vx - 27 * s, vy - 11 * s, vx + 26 * s, vy + 6 * s), 190, 350,
          fill=ink_rgba(cfg, 150), width=2)

    # ---- Dante: hooded, head tilted up reading the inscription ----
    dx_, dy_ = S(cfg, 1128), S(cfg, 2114)
    ds = S(cfg, 0.92)
    d.polygon([(dx_ - 13 * ds, dy_), (dx_ + 12 * ds, dy_),
               (dx_ + 16 * ds, dy_ - 56 * ds), (dx_ + 10 * ds, dy_ - 94 * ds),
               (dx_ + 3 * ds, dy_ - 116 * ds), (dx_ - 8 * ds, dy_ - 116 * ds),
               (dx_ - 12 * ds, dy_ - 94 * ds), (dx_ - 15 * ds, dy_ - 54 * ds)],
              fill=ink_rgba(cfg, 240))
    # hood tilted back (looking up at the words)
    d.ellipse((dx_ - 12 * ds, dy_ - 138 * ds, dx_ + 10 * ds, dy_ - 113 * ds),
              fill=ink_rgba(cfg, 240))
    d.ellipse((dx_ + 1 * ds, dy_ - 134 * ds, dx_ + 8 * ds, dy_ - 121 * ds),
              fill=ink_rgba(cfg, 125))
    for ox, lean in ((-7, -4), (0, -1), (7, 4)):
        d.line((dx_ + ox * ds, dy_ - 104 * ds, dx_ + ox * ds + lean * ds, dy_ - 13 * ds),
               fill=ink_rgba(cfg, 150), width=2)
    d.arc((dx_ - 18 * ds, dy_ - 12 * ds, dx_ + 20 * ds, dy_ + 4 * ds), 200, 340,
          fill=ink_rgba(cfg, 150), width=2)

    # rim light from the gate's interior
    paper = cfg["paper"] + (255,)
    d.arc((vx - 8 * s, vy - 132 * s, vx + 11 * s, vy - 106 * s), 250, 350,
          fill=paper, width=2)
    d.line((vx + 16 * s, vy - 96 * s, vx + 22 * s, vy - 26 * s), fill=paper, width=2)
    d.arc((dx_ - 13 * ds, dy_ - 144 * ds, dx_ + 12 * ds, dy_ - 112 * ds), 250, 350,
          fill=paper, width=2)
    return L


# ----------------------------------------------------------------------------
# frame + caption (Canto III)
# ----------------------------------------------------------------------------
def draw_frame_and_caption3(img, cfg):
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
    d.text((cx, cy_sub), "CHANT TROISIÈME — LA PORTE DE L'ENFER", font=f_sub, fill=ink + (230,), anchor="mm")
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
    dark = make_darkmap3(cfg)
    paper = make_paper(cfg, dark).convert("RGBA")
    size = (S(cfg, W), S(cfg, H))

    layers = []
    print("  rock ...")
    layers += layer_rock(cfg)
    print("  interior ...")
    layers += layer_interior(cfg)
    print("  arch edge ...")
    layers.append(layer_arch_edge(cfg))
    print("  lintel ...")
    layers.append(layer_lintel(cfg))
    print("  floor ...")
    layers += layer_floor(cfg)
    print("  figures ...")
    layers.append(layer_figures(cfg))

    print("  compositing ...")
    ink = Image.new("RGBA", size, (0, 0, 0, 0))
    for L in layers:
        ink = Image.alpha_composite(ink, L)
    ink_final = ink.resize((W, H), Image.LANCZOS)
    img = Image.alpha_composite(paper, ink_final)
    img = draw_frame_and_caption3(img, cfg)
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
    name = sys.argv[1] if len(sys.argv) > 1 else "c3_v01.png"
    render(CFG, name)
