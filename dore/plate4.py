#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procedural engraving engine in the manner of Gustave Doré — Canto IV.
Scene: "DANTE — L'ENFER · CHANT QUATRIÈME · LE NOBLE CHÂTEAU"
Limbo: a vast dark cavern, and the noble castle of the sages glowing alone —
keep, twin towers, battlements, lit windows, moat and bridge. Dante and Virgil
walk toward the gate across the dim plain; spirits drift in the dark.

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
    seed=404,
    paper=(241, 232, 213),
    ink=(30, 23, 16),
    castle=(1100, 1400),         # centre of the castle glow
    frame=(0.055, 0.035, 0.945, 0.905),
    caption_top=0.775,
    out_dir="DORE_INFERNO/Chant_IV_Le_Noble_Chateau",
)


# ----------------------------------------------------------------------------
# tone field — dark cavern, a pool of light around the castle
# ----------------------------------------------------------------------------
def make_darkmap4(cfg, ss_div=4):
    W, H = cfg["W"], cfg["H"]
    hw, hh = W // ss_div, H // ss_div
    cx, cy = cfg["castle"]
    y, x = np.mgrid[0:hh, 0:hw].astype(np.float32)
    x = x * ss_div
    y = y * ss_div
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    n = fbm_arr(hh, hw, cfg["seed"] + 9, 34, 4)
    base = 0.62 + 0.10 * n
    glow = (0.26 * np.exp(-((r / 320.0) ** 2)) + 0.12 * np.exp(-((r / 700.0) ** 2))
            + 0.05 * np.exp(-((r / 1200.0) ** 2)))
    d = np.clip(base - glow, 0.0, 1.0)
    im = Image.fromarray((d * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0


# ----------------------------------------------------------------------------
# the cavern ceiling
# ----------------------------------------------------------------------------
def layer_ceiling(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 11)
    m = band_mask(size, S(cfg, 100), S(cfg, 1200))
    depth = grad_mask(size, S(cfg, 100), S(cfg, 1200), 235, 120)
    m = ImageChops.multiply(m, depth)
    rect = [(S(cfg, 121), S(cfg, 100)), (S(cfg, 2079), S(cfg, 100)),
            (S(cfg, 2079), S(cfg, 1200)), (S(cfg, 121), S(cfg, 1200))]
    stack = [strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 1, 6.0, rng, jitter=2.2, width=3)),
             strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 88, 14, rng, jitter=5.0, width=1, dash=34))]
    # hanging stalactites
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    for _ in range(9):
        x = rng.uniform(200, 2000)
        ln = rng.uniform(40, 110)
        w = rng.uniform(10, 26)
        d.polygon([(S(cfg, x - w), S(cfg, 100)), (S(cfg, x + w), S(cfg, 100)),
                   (S(cfg, x), S(cfg, 100 + ln))], fill=ink_rgba(cfg, 170))
        d.line((S(cfg, x), S(cfg, 100 + ln), S(cfg, x + rng.uniform(-4, 4)), S(cfg, 100 + ln + 26)),
               fill=ink_rgba(cfg, 120), width=2)
    stack.append(L)
    return stack


# ----------------------------------------------------------------------------
# castle glow halo behind the masonry
# ----------------------------------------------------------------------------
def layer_glow(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 12)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    cx, cy = cfg["castle"]
    paper = cfg["paper"]
    for prx, pry, pa in ((540, 410, 40), (380, 290, 70), (260, 205, 100), (150, 125, 130)):
        d.ellipse((S(cfg, cx - prx), S(cfg, cy - pry), S(cfg, cx + prx), S(cfg, cy + pry)),
                  fill=paper + (pa,))
    for _ in range(20):
        th = rng.uniform(0, 2 * math.pi)
        r0 = rng.uniform(150, 330)
        ln = rng.uniform(260, 640)
        d.line((S(cfg, cx + r0 * math.sin(th)), S(cfg, cy + r0 * math.cos(th)),
                S(cfg, cx + (r0 + ln) * math.sin(th)), S(cfg, cy + (r0 + ln) * math.cos(th))),
               fill=ink_rgba(cfg, 8), width=2)
    return L


# ----------------------------------------------------------------------------
# the castle
# ----------------------------------------------------------------------------
def crenels(cfg, cx0, cx1, top_y, n, d):
    total = cx1 - cx0
    w = total / (n * 1.55)
    gap = w * 0.55
    h = w * 1.1
    x = cx0 + (total - (n * w + (n - 1) * gap)) / 2
    for k in range(n):
        d.rectangle((S(cfg, x), S(cfg, top_y - h), S(cfg, x + w), S(cfg, top_y)),
                    fill=cfg["paper"] + (150,), outline=ink_rgba(cfg, 205), width=2)
        x += w + gap


def layer_castle(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 13)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    paper = cfg["paper"]
    ink = cfg["ink"]

    blocks = [
        (880, 985, 1140, 1540),   # left tower
        (985, 1215, 1190, 1540),  # keep
        (1215, 1320, 1140, 1540), # right tower
    ]
    walls = []
    for (x0, x1, y0, y1) in blocks:
        sp = [(S(cfg, x0), S(cfg, y0)), (S(cfg, x1), S(cfg, y0)),
              (S(cfg, x1), S(cfg, y1)), (S(cfg, x0), S(cfg, y1))]
        # pale stone walls — the castle glows against the dark cavern
        d.polygon(sp, fill=paper + (125,))
        d.polygon(sp, outline=ink_rgba(cfg, 240), width=3)
        walls.append((x0, x1, y0, y1))
        # stone courses
        for y in range(y0 + 12, y1, 13):
            d.line((S(cfg, x0 + 2), S(cfg, y), S(cfg, x1 - 2), S(cfg, y)),
                   fill=ink_rgba(cfg, 115), width=2)
        # vertical joints
        for y in range(y0 + 12, y1, 13):
            for x in range(x0 + 24, x1 - 10, 22):
                if (y // 13) % 2 == 0:
                    d.line((S(cfg, x), S(cfg, y - 6), S(cfg, x), S(cfg, y + 6)),
                           fill=ink_rgba(cfg, 95), width=1)
                else:
                    d.line((S(cfg, x + 11), S(cfg, y - 6), S(cfg, x + 11), S(cfg, y + 6)),
                           fill=ink_rgba(cfg, 95), width=1)
    # battlements
    crenels(cfg, 880, 985, 1140, 3, d)
    crenels(cfg, 985, 1215, 1190, 5, d)
    crenels(cfg, 1215, 1320, 1140, 3, d)
    # lit windows
    wins = [
        (908, 934, 1268, 1326), (913, 928, 1202, 1234),          # left tower
        (1012, 1048, 1290, 1364), (1082, 1118, 1290, 1364), (1152, 1188, 1290, 1364),  # keep
        (1266, 1292, 1268, 1326), (1272, 1287, 1202, 1234),      # right tower
    ]
    for (x0, x1, y0, y1) in wins:
        d.rectangle((S(cfg, x0), S(cfg, y0), S(cfg, x1), S(cfg, y1)),
                    fill=paper + (228,))
        d.rectangle((S(cfg, x0), S(cfg, y0), S(cfg, x1), S(cfg, y1)),
                    outline=ink_rgba(cfg, 210), width=2)
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        d.line((S(cfg, mx), S(cfg, y0 + 2), S(cfg, mx), S(cfg, y1 - 2)),
               fill=ink_rgba(cfg, 150), width=1)
        d.line((S(cfg, x0 + 2), S(cfg, my), S(cfg, x1 - 2), S(cfg, my)),
               fill=ink_rgba(cfg, 150), width=1)
    # gate with portcullis
    d.polygon([(S(cfg, 1060), S(cfg, 1540)), (S(cfg, 1140), S(cfg, 1540)),
               (S(cfg, 1140), S(cfg, 1475)), (S(cfg, 1122), S(cfg, 1432)),
               (S(cfg, 1078), S(cfg, 1432)), (S(cfg, 1060), S(cfg, 1475))],
              fill=ink_rgba(cfg, 240))
    d.polygon([(S(cfg, 1060), S(cfg, 1540)), (S(cfg, 1140), S(cfg, 1540)),
               (S(cfg, 1140), S(cfg, 1475)), (S(cfg, 1122), S(cfg, 1432)),
               (S(cfg, 1078), S(cfg, 1432)), (S(cfg, 1060), S(cfg, 1475))],
              outline=paper + (150,), width=2)
    for gx in range(1070, 1131, 12):
        d.line((S(cfg, gx), S(cfg, 1442), S(cfg, gx), S(cfg, 1536)),
               fill=ink_rgba(cfg, 160), width=3)
    d.line((S(cfg, 1066), S(cfg, 1466), S(cfg, 1134), S(cfg, 1466)),
           fill=ink_rgba(cfg, 160), width=3)
    return L


def layer_moat_bridge(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 14)
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    # moat water
    m = band_mask(size, S(cfg, 1544), S(cfg, 1608))
    water = strokes_layer(size, cfg["ink"], m, lambda dd:
                          draw_hatch(dd, [(S(cfg, 800), S(cfg, 1544)), (S(cfg, 1400), S(cfg, 1544)),
                                          (S(cfg, 1400), S(cfg, 1608)), (S(cfg, 800), S(cfg, 1608))],
                                     1, 4.6, rng, jitter=2.0, width=3))
    L = Image.alpha_composite(L, water)
    for _ in range(26):
        y = rng.uniform(1552, 1600)
        x = rng.uniform(840, 1360)
        d.line((S(cfg, x), S(cfg, y), S(cfg, x + rng.uniform(14, 40)), S(cfg, y)),
               fill=ink_rgba(cfg, 60), width=2)
    # light of the castle reflected on the water
    for _ in range(18):
        y = rng.uniform(1550, 1604)
        d.line((S(cfg, 1100 - 40 + rng.uniform(-12, 12)), S(cfg, y),
                S(cfg, 1100 + 40 + rng.uniform(-12, 12)), S(cfg, y + rng.uniform(-2, 2))),
               fill=cfg["paper"] + (70,), width=3)
    # bridge
    d.polygon([(S(cfg, 1046), S(cfg, 1608)), (S(cfg, 1154), S(cfg, 1608)),
               (S(cfg, 1144), S(cfg, 1540)), (S(cfg, 1056), S(cfg, 1540))],
              fill=cfg["paper"] + (120,))
    d.polygon([(S(cfg, 1046), S(cfg, 1608)), (S(cfg, 1154), S(cfg, 1608)),
               (S(cfg, 1144), S(cfg, 1540)), (S(cfg, 1056), S(cfg, 1540))],
              outline=ink_rgba(cfg, 190), width=2)
    for k in range(6):
        t = k / 5
        y = 1544 + t * 64
        d.line((S(cfg, 1052 + t * 6), S(cfg, y), S(cfg, 1148 - t * 6), S(cfg, y)),
               fill=ink_rgba(cfg, 120), width=1)
    return L


# ----------------------------------------------------------------------------
# the plain, spirits, mist, path, figures
# ----------------------------------------------------------------------------
def layer_plain(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 15)
    cx, cy = cfg["castle"]
    m = band_mask(size, S(cfg, 1440), S(cfg, 2230))
    # darker far from the castle
    y, x = np.mgrid[0:size[1], 0:size[0]].astype(np.float32)
    rr = np.sqrt((x - S(cfg, cx)) ** 2 + (y - S(cfg, cy)) ** 2)
    prof = np.clip(0.5 + 0.5 * (1 - np.exp(-(rr / S(cfg, 900)) ** 2)), 0, 1)
    depth = Image.fromarray((prof * 255).astype(np.uint8))
    m = ImageChops.multiply(m, depth)
    rect = [(S(cfg, 121), S(cfg, 1440)), (S(cfg, 2079), S(cfg, 1440)),
            (S(cfg, 2079), S(cfg, 2230)), (S(cfg, 121), S(cfg, 2230))]
    stack = [strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 2, 5.4, rng, jitter=2.2, width=3)),
             strokes_layer(size, cfg["ink"], m, lambda d, r=rect:
                           draw_hatch(d, r, 92, 15, rng, jitter=6.0, width=1, dash=32))]
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    # stones
    for _ in range(22):
        x = rng.uniform(160, 2040)
        y = rng.uniform(1620, 2210)
        r = rng.uniform(6, 18)
        d.ellipse((S(cfg, x - r), S(cfg, y - r * 0.45), S(cfg, x + r), S(cfg, y + r * 0.45)),
                  outline=ink_rgba(cfg, 130), width=2)
    # grass tufts
    for _ in range(160):
        x = rng.uniform(150, 2050)
        y = rng.uniform(1560, 2220)
        n = rng.randint(2, 4)
        for k in range(n):
            d.line((S(cfg, x + k * 3 - 4), S(cfg, y), S(cfg, x + k * 3 - 4 + rng.uniform(-2, 2)),
                    S(cfg, y - rng.uniform(6, 14))), fill=ink_rgba(cfg, 100), width=1)
    # wandering spirits — small hooded silhouettes
    for _ in range(30):
        x = rng.uniform(220, 1980)
        y = rng.uniform(1470, 1750)
        if abs(x - cx) < 260 and y > 1520:
            continue
        h = rng.uniform(9, 17)
        a = int(rng.uniform(90, 160))
        d.line((S(cfg, x), S(cfg, y), S(cfg, x), S(cfg, y - h)),
               fill=ink_rgba(cfg, a), width=3)
        d.ellipse((S(cfg, x - 3), S(cfg, y - h - 4), S(cfg, x + 3), S(cfg, y - h + 2)),
                  fill=ink_rgba(cfg, a))
    # four poets at the gate — slightly larger marks
    for x, y in ((1000, 1532), (1200, 1532), (1030, 1562), (1170, 1562)):
        d.line((S(cfg, x), S(cfg, y), S(cfg, x), S(cfg, y - 14)),
               fill=ink_rgba(cfg, 190), width=4)
        d.ellipse((S(cfg, x - 3), S(cfg, y - 18), S(cfg, x + 3), S(cfg, y - 12)),
                  fill=ink_rgba(cfg, 190))
    stack.append(L)
    return stack


def layer_path(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    rng = random.Random(cfg["seed"] + 16)
    cx = cfg["castle"][0]
    poly = [(cx - 130, 2230), (cx + 130, 2230), (cx + 70, 1608), (cx - 70, 1608)]
    m = poly_mask(size, [(S(cfg, x), S(cfg, y)) for x, y in poly])
    er = Image.new("RGBA", size, (0, 0, 0, 0))
    er.paste(cfg["paper"] + (85,), (0, 0), m)
    fl = Image.new("L", size, 0)
    fd = ImageDraw.Draw(fl)
    for k in range(14):
        t = k / 13
        y = 2230 - t * 622
        w = 130 - 60 * t
        for off in np.linspace(-w, w, max(3, int(w / 9))):
            fd.line((S(cfg, cx + off + rng.uniform(-6, 6)), S(cfg, y),
                     S(cfg, cx + off * 0.6 + rng.uniform(-6, 6)), S(cfg, y - 44)),
                    fill=255, width=2)
    fl = ImageChops.multiply(fl, m)
    er2 = Image.new("RGBA", size, (0, 0, 0, 0))
    er2.paste(cfg["ink"] + (50,), (0, 0), fl)
    ed = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ed)
    d.line([(S(cfg, cx - 130), S(cfg, 2230)), (S(cfg, cx - 70), S(cfg, 1608))],
           fill=ink_rgba(cfg, 80), width=2)
    d.line([(S(cfg, cx + 130), S(cfg, 2230)), (S(cfg, cx + 70), S(cfg, 1608))],
           fill=ink_rgba(cfg, 80), width=2)
    return [er, er2, ed]


def layer_figures(cfg):
    W, H = cfg["W"], cfg["H"]
    size = (S(cfg, W), S(cfg, H))
    L = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(L)

    def walker(x, y, s, hooded, alpha=242):
        x, y = S(cfg, x), S(cfg, y)
        s = S(cfg, s)
        # cloak / toga from behind, slight forward lean
        d.polygon([(x - 14 * s, y), (x + 13 * s, y),
                   (x + 16 * s, y - 56 * s), (x + 10 * s, y - 96 * s),
                   (x + 3 * s, y - 118 * s), (x - 8 * s, y - 118 * s),
                   (x - 12 * s, y - 96 * s), (x - 15 * s, y - 54 * s)],
                  fill=ink_rgba(cfg, alpha))
        d.ellipse((x - 12 * s, y - 136 * s, x + 10 * s, y - 115 * s),
                  fill=ink_rgba(cfg, alpha))
        for ox in (-6, 0, 6):
            d.line((x + ox * s, y - 104 * s, x + ox * s, y - 12 * s),
                   fill=ink_rgba(cfg, 150), width=2)
        # walking legs
        d.line((x - 6 * s, y, x - 10 * s, y + 12 * s), fill=ink_rgba(cfg, alpha - 20), width=4)
        d.line((x + 6 * s, y, x + 11 * s, y + 9 * s), fill=ink_rgba(cfg, alpha - 20), width=4)
        # rim light on the hood from the castle ahead
        d.arc((x - 12 * s, y - 140 * s, x + 11 * s, y - 114 * s), 190, 350,
              fill=cfg["paper"] + (255,), width=2)

    walker(1016, 2136, 1.05, False)   # Virgil
    walker(1148, 2122, 0.92, True)    # Dante
    return L


# ----------------------------------------------------------------------------
# frame + caption (Canto IV)
# ----------------------------------------------------------------------------
def draw_frame_and_caption4(img, cfg):
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
    d.text((cx, cy_sub), "CHANT QUATRIÈME — LE NOBLE CHÂTEAU", font=f_sub, fill=ink + (230,), anchor="mm")
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
    dark = make_darkmap4(cfg)
    paper = make_paper(cfg, dark).convert("RGBA")
    size = (S(cfg, W), S(cfg, H))

    layers = []
    print("  ceiling ...")
    layers += layer_ceiling(cfg)
    print("  glow ...")
    layers.append(layer_glow(cfg))
    print("  castle ...")
    layers.append(layer_castle(cfg))
    print("  moat + bridge ...")
    layers.append(layer_moat_bridge(cfg))
    print("  plain ...")
    layers += layer_plain(cfg)
    print("  path ...")
    layers += layer_path(cfg)
    print("  figures ...")
    layers.append(layer_figures(cfg))

    print("  compositing ...")
    ink = Image.new("RGBA", size, (0, 0, 0, 0))
    for L in layers:
        ink = Image.alpha_composite(ink, L)
    ink_final = ink.resize((W, H), Image.LANCZOS)
    img = Image.alpha_composite(paper, ink_final)
    img = draw_frame_and_caption4(img, cfg)
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
    name = sys.argv[1] if len(sys.argv) > 1 else "c4_v01.png"
    render(CFG, name)
