# DORÉ — L'ENFER

A procedurally generated engraving suite in the manner of Gustave Doré, made
entirely on this machine — no image model, no neural network, no external API.
Every line is a mathematical decision: exponential vortex rings, jittered
cross-hatching, radial god-rays, gradient depth shading.

## The gallery — `DORE_INFERNO/`

All plates live in one master folder, one subfolder per canto (each with the
final plate, detail crops and a `versions/` iteration history):

- **Chant I** — `Chant_I_La_Lumiere_Divine/DORE_LA_LUMIERE_DIVINE.png`
- **Chant II** — `Chant_II_La_Foret_Obscure/DORE_LA_FORET_OBSCURE.png`
- **Chant III** — `Chant_III_La_Porte_de_l_Enfer/DORE_LA_PORTE_DE_L_ENFER.png`
- **Chant IV** — `Chant_IV_Le_Noble_Chateau/DORE_LE_NOBLE_CHATEAU.png`
- **Chant V** — `Chant_V_Paolo_et_Francesca/DORE_PAOLO_ET_FRANCESCA.png`
- **Chant VI** — `Chant_VI_Cerbere/DORE_CERBERE.png`

Gallery index: `DORE_INFERNO/README.md`.

## The piece

**`DORE_INFERNO/Chant_I_La_Lumiere_Divine/DORE_LA_LUMIERE_DIVINE.png`** —
2200 × 2860, drawn at 2× supersampling (effective 4400 × 5720 ink resolution),
warm ivory paper, letterpress caption:

> DANTE · L'ENFER — CHANT PREMIER — LA LUMIÈRE DIVINE
> G. Doré inv. & sculp. — PARIS · M DCCC LXI

Scene: a hooded pilgrim on a rocky spur, staff in hand, before a vortex of
divine light that opens above a dark gulf; a gnarled dead tree on the left
cliff, chains hanging into the abyss, the ruined towers of a distant city
silhouetted on the horizon, birds crossing the light.

Detail crops: `DORE_INFERNO/Chant_I_La_Lumiere_Divine/detail_*.png`.
Iteration history: `DORE_INFERNO/Chant_I_La_Lumiere_Divine/versions/`.

## Canto II — `dore/plate2.py`

**`DORE_INFERNO/Chant_II_La_Foret_Obscure/DORE_LA_FORET_OBSCURE.png`** —
2200 × 2860, same plate format, new scene: night falls on the dark wood. A
winding path of last light climbs toward a distant glow between towering
trees; Virgil in profile points the way, Dante follows hooded; crescent moon
and first stars over hatched dusk clouds, mist bands between the trunks, a
treeline silhouette, leaf litter and a fallen log.

> DANTE · L'ENFER — CHANT DEUXIÈME — LA FORÊT OBSCURE
> G. Doré inv. & sculp. — PARIS · M DCCC LXI

Detail crops: `DORE_INFERNO/Chant_II_La_Foret_Obscure/detail_*.png`.
Iteration history: `DORE_INFERNO/Chant_II_La_Foret_Obscure/versions/`.
Inspector: `dore/critic2.py` (same invented vision — ASCII, ink maps, region
stats — retuned for the forest scene).

Bugs the vision caught on this plate: Virgil's anchor was left in final coords
(he floated as a stray dark blob at (503, 1016) in the sky), a too-bright flat
sky, an over-white path, a dim glow, and ground darkening bleeding into the
caption band.

## Canto III — `dore/plate3.py`

**`DORE_INFERNO/Chant_III_La_Porte_de_l_Enfer/DORE_LA_PORTE_DE_L_ENFER.png`** —
2200 × 2860: the gate of Hell. Monumental rock arch, voussoirs and strata
hatching, the inscription carved on the lintel ("PER ME SI VA NE LA CITTÀ
DOLENTE…"), a dim glow and hanging lamp inside the gate, Virgil pointing at
the words, Dante reading them, a skull among the stones at the threshold.

> DANTE · L'ENFER — CHANT TROISIÈME — LA PORTE DE L'ENFER
> G. Doré inv. & sculp. — PARIS · M DCCC LXI

Detail crops: `DORE_INFERNO/Chant_III_La_Porte_de_l_Enfer/detail_*.png`.
Iteration history: `DORE_INFERNO/Chant_III_La_Porte_de_l_Enfer/versions/`.
Inspector: `dore/critic3.py`.

Bugs the vision caught here: single-pixel hatch lines lose ~half their alpha
in the LANCZOS downscale (the rock refused to turn dark until strokes went
2 px wide), a too-bright rock mass, an over-dark floor, and an inscription
that overflowed its lintel band.

## Canto IV — `dore/plate4.py`

**`DORE_INFERNO/Chant_IV_Le_Noble_Chateau/DORE_LE_NOBLE_CHATEAU.png`** —
2200 × 2860: the noble castle of Limbo. A vast dark cavern whose sole light is
the castle of the sages — keep, twin towers, crenellations, glowing windows,
moat and drawbridge; souls wander the plain as silhouettes, the four great
poets wait at the gate, Dante and Virgil approach along the path of light;
stalactites hang from the cavern roof.

> DANTE · L'ENFER — CHANT QUATRIÈME — LE NOBLE CHÂTEAU
> G. Doré inv. & sculp. — PARIS · M DCCC LXI

Detail crops: `DORE_INFERNO/Chant_IV_Le_Noble_Chateau/detail_*.png`.
Iteration history: `DORE_INFERNO/Chant_IV_Le_Noble_Chateau/versions/`.
Inspector: `dore/critic4.py`.

## Canto V — `dore/plate5.py`

**`DORE_INFERNO/Chant_V_Paolo_et_Francesca/DORE_PAOLO_ET_FRANCESCA.png`** —
2200 × 2860: the eternal whirlwind of the second circle. 110 spiral wind-lines
and ~8500 wind-scratches build the storm; 54 pale bodies tumble around the
eye, Paolo and Francesca embrace at the centre (long hair, entwined arms,
falling legs, robe streaming left); below on the ground, Dante overcome with
hand to brow, Virgil steadying him.

> DANTE · L'ENFER — CHANT CINQUIÈME — PAOLO ET FRANCESCA
> G. Doré inv. & sculp. — PARIS · M DCCC LXI

Detail crops: `DORE_INFERNO/Chant_V_Paolo_et_Francesca/detail_*.png`.
Iteration history: `DORE_INFERNO/Chant_V_Paolo_et_Francesca/versions/`.
Inspector: `dore/critic5.py`.

## The engine — `dore/plate.py`

Pipeline: tone field (darkmap) → paper (tint, grain, vignette) → ink layers
drawn at 2× supersampling → LANCZOS downscale → composite → plate frame +
caption. Layers: vortex tunnel (elliptic rings with rotating gaps, spiral-arm
radius modulation, tangential slivers), god-rays with source fade-in and bend,
gulf hatching with depth gradient, cliff masses (base hatch + gradient
cross-hatch + dash texture + contour form-lines + crest roughness), chains,
pilgrim, tree, birds, rim-light erasures.

Run: `python3 dore/plate.py <out>.png`

## The invented sight — `dore/vision.py`

The working session's model cannot see images, so the work was inspected with
a self-built critic. The toolkit lives in **`dore/vision.py`** — a
self-contained module (dependencies: Pillow + numpy only, no network, no
model):

- `load(path)` — image → float32 luminance array
- `ascii_view(img, width, label)` / `ascii_arr(arr, ...)` — multi-scale ASCII
  luminance render (composition check); `CHARS = " .:-=+*#%@"`, " " = brightest,
  "@" = darkest
- `crop(arr, x0, y0, x1, y1, width)` — downsample a crop for ASCII zoom
- `pixel_window(arr, x0, y0, x1, y1, step)` — 1:1 pixel window (figure detail)
- `ink_map(a, paper_lum)` — paper-minus-luminance: where the ink landed
- `ink_ascii(a, paper_lum, width)` — ASCII render of the ink map (locate ink)
- `ink_grid(a, n, paper_lum)` — n × n mean-ink grid (locate missing/extra ink)
- `histogram(a, bins)` — luminance histogram (percent per bin)
- `region_stats(a, x0, y0, x1, y1, name)` — mean / dark% / bright% per region
- `luminance_profile(a, bins, axis)` — binned mean profile (vertical/horizontal)
- `fft_regularity(a, label)` — FFT peak ratio: mechanical repetition / moiré
- `probe(a, points)` — sample luminance at exact coordinates

Per-canto critics import from it: `dore/critic.py` (I), `dore/critic2.py` (II),
`dore/critic3.py` (III), `dore/critic4.py` (IV), `dore/critic5.py` (V). Run any
of them as `python3 dore/criticN.py <path-to-png>`; `python3 dore/vision.py
<path>` gives the generic report.

**`dore/eye.py`** is the unified CLI over the toolkit — every op in one entry
point, with a `--json` mode that returns the text plus structured numbers
(global stats, per-region stats, profile, FFT ratio, ink grid, histogram).
It is what the agent plugin drives, and it is pip-installable as `dore-eye`:

```bash
python3 dore/eye.py report  <img> [--fast] [--width 110]
python3 dore/eye.py ascii   <img> --crop 950 560 1250 860
python3 dore/eye.py pixel   <img> --crop 1080 1180 1120 1200 --step 1
python3 dore/eye.py ink     <img> --paper-lum 232 --grid 8
python3 dore/eye.py metrics <img> --region glow,950,560,1250,860 \
    --point core,1100,700 --profile-axis 0 --bins 32 --fft-region 100 100 2100 900
python3 dore/eye.py critic  <img> --canto 5   # canto auto-guessed from the path
```

## The vision plugin — `plugin/`

`plugin/dore-vision-plugin.js` packages this toolkit as a dynamic Cordis
plugin for DeepSeek Harness. It registers five Host tools —
`dore_eye_report`, `dore_eye_zoom`, `dore_eye_metrics`, `dore_eye_ink`,
`dore_eye_critic` — each of which shells out to `dore/eye.py --json` through
the harness `shell` service and returns the report text to the model. Mounting
instructions live in `plugin/README.md`.

Golden reference: when the session model can accept image input, skip the
ASCII path — read the plate directly with the model's own `read_image` tool
and use the numeric diagnostics (`dore_eye_metrics` / `dore_eye_ink`) as
cross-checks. The ASCII views remain the fallback for image-blind models,
which is what this toolkit was invented for.

The render engines live in `dore/plate.py` … `dore/plate5.py` (one per canto;
each is a standalone script sharing the stroke primitives of `plate.py`).
`plate.py` also hosts the shared primitives: supersampling helper `S`, ink
layers via `strokes_layer`/`poly_mask`/`grad_mask`, `draw_hatch` (jittered,
dashed, full two-sided coverage), `contour_strokes`, fbm noise, tone-field
(`make_darkmap`), paper (`make_paper`), fonts and frame/caption drawing.

Run: `python3 dore/plateN.py <out>.png`

## Bugs the vision caught (and their fixes)

1. **Coordinate-scale bug** — vortex and rays were drawn in final coords on the
   2× canvas; the pilgrim never appeared on the spur. → scale every scene coord.
2. **Hatch one-sided coverage** — `draw_hatch` anchored its loop to the absolute
   perpendicular coordinate of the bbox centre, so strokes covered only one side
   of every mass (v03: only the left gulf hatched; v04: only the right).
   → iterate offsets from `-span` to `+span`.
3. **Ring-gap wrap truncation** — arcs crossing the 2π seam were sliced short.
   → wrap-aware arc assembly.
4. **Transposed darkmap** — `hw, hh = H//4, W//4` swapped axes, so the light
   centre sat at (847, 910) instead of (1100, 700) and the sky was 65 % ink
   asymmetric. → one-line fix restored symmetry (205.2 / 202.4).
5. **Bullseye sun** — dense dark rings around the glow core; rays starting as a
   black cluster. → ring alpha grows with radius², rings skip r < 290, rays
   fade in from the source, halo arcs softened.
6. **Washed-out corridor** — the god-light was 99 % pure white. → corridor-
   modulated ring alpha (×0.7), 140 tapered rays, 380 cloud wisps, sun sparkle.
7. **Invisible chains** — dark ink on dark rock. → moved over the gulf, added a
   paper-coloured rim highlight.

Final balance: glow core 227.7 → corridor 214 → sky edges ~192 → cliffs ~118 →
gulf ~110 → pilgrim 92 (the darkest point of the plate, standing in a pool of
light).
