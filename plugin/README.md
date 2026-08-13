# dore-vision-plugin

Dynamic Cordis plugin for [DeepSeek Harness](https://github.com/deepseek-ai/DeepSeek-Harness)
that gives a session model the Doré "invented sight": ASCII luminance views,
ink maps, region statistics, profiles, FFT regularity and per-canto critics,
computed locally from `dore/vision.py` (Pillow + numpy; no network, no model).

## Files

- `dore-vision-plugin.js` — the Host-half Package body (`return { apply(ctx) … }`).
  It registers five model tools and shells out to `dore/eye.py --json` through
  the harness `shell` service:

  | Tool | What it does |
  | --- | --- |
  | `dore_eye_report` | full report: ASCII view, vertical profile, FFT regularity, histogram |
  | `dore_eye_zoom` | ASCII crop view or 1:1 pixel window (stroke-level detail) |
  | `dore_eye_metrics` | named region stats, profile, FFT, exact-point probes |
  | `dore_eye_ink` | ink deposition map + n × n mean-ink grid |
  | `dore_eye_critic` | run the per-canto critic (critics exist for cantos 1–5; other cantos fall back to the generic report) |

## Mounting (dynamic, per session)

Prerequisites: the repo's `dore/` directory must sit at
`<session workspace>/dore`, and `python3` with numpy + Pillow on `PATH`.

1. In the harness GUI, open the session and use the `cordis_define` tool with
   `plugin: { kind: "new", idPrefix: "dorev" }` and `code.host` set to the
   body of `dore-vision-plugin.js` (everything after `// dore-vision-plugin.js …`
   header, starting at `return {`).
2. `cordis_run` the returned `pluginId`/`packageId` with mode `run`.
3. The five `dore_eye_*` tools become callable in the next model step.
   Updates: append a new Package via `cordis_define` `kind: "existing"` and
   switch with mode `update`.

## Conventions

- ASCII character ramp `" .:-=+*#%@"`: `" "` = brightest (lum 0–25),
  `"@"` = darkest (lum 230–255). A dark figure prints as `@`, a sun as ` `.
- Coordinates are in final-image pixels (2200 × 2860 plates).
- When the session model accepts image input, use its own `read_image` tool as
  the golden reference and the numeric diagnostics here as cross-checks.
