// dore-vision-plugin.js — dynamic Cordis plugin for DeepSeek Harness (Host half)
//
// Wraps dore/eye.py (the unified CLI of dore/vision.py) as five model tools:
//   dore_eye_report  full ASCII + profile + FFT + histogram report
//   dore_eye_zoom    ASCII crop view or 1:1 pixel window
//   dore_eye_metrics named region stats, profile, FFT, exact-point probes
//   dore_eye_ink     ink deposition map + n x n mean-ink grid
//   dore_eye_critic  per-canto critic (critic1..critic5, canto guessed or given)
//
// This file is the exact `code.host` body of the dynamic Cordis Package. It is
// NOT a standalone Node module: it runs inside the harness, which provides
// `harness` and `ctx` as builtins. Mount it with the cordis_define / cordis_run
// tools (see plugin/README.md), or paste it into a cordis.yml composition.
//
// Prerequisites: the dore/ directory must sit at <session workspace>/dore and
// `python3` with numpy + Pillow must be on PATH. CHARS " .:-=+*#%@":
// " " = brightest, "@" = darkest.

return {
  apply(ctx) {
    const shell = ctx.get('shell')
    if (shell === undefined) return
    const policy = ctx.get('sandboxPolicy')
    const root = policy && typeof policy.workspaceRoot === 'string' ? policy.workspaceRoot : ''
    const doreDir = root ? root.replace(/\/+$/, '') + '/dore' : 'dore'
    const eye = doreDir + '/eye.py'

    function q(v) {
      return "'" + String(v).replace(/'/g, "'\\''") + "'"
    }

    async function runEye(op, args) {
      const parts = ['python3', q(eye), op]
      for (const a of args) {
        if (typeof a === 'number') parts.push(String(a))
        else parts.push(String(a).startsWith('--') ? String(a) : q(String(a)))
      }
      parts.push('--json')
      let spec
      try {
        spec = shell.resolve({ command: parts.join(' '), workdir: doreDir, timeoutMs: 120000, stdoutMaxBytes: 262144 })
      } catch (err) {
        return 'dore eye resolve failed: ' + String(err && err.message ? err.message : err)
      }
      let res
      try {
        res = await shell.run(spec)
      } catch (err) {
        return 'dore eye run failed: ' + String(err && err.message ? err.message : err)
      }
      const out = res && res.stdout && typeof res.stdout.text === 'string' ? res.stdout.text : ''
      let parsed = null
      try { parsed = JSON.parse(out) } catch (err) { parsed = null }
      if (parsed && typeof parsed === 'object' && parsed.ok === true) {
        return typeof parsed.text === 'string' ? parsed.text : JSON.stringify(parsed)
      }
      if (parsed && typeof parsed === 'object' && parsed.error) {
        return 'dore eye error: ' + String(parsed.error)
      }
      const code = res && typeof res.exitCode === 'number' ? res.exitCode : 'unknown'
      const errText = res && res.stderr && typeof res.stderr.text === 'string' ? res.stderr.text : ''
      return 'dore eye failed (exit ' + code + '): ' + String(errText || out).slice(0, 2000)
    }

    const stringOut = {
      schema: { type: 'string' },
      render(_args, value) {
        return [{ type: 'text', text: String(value) }]
      },
    }

    harness.registerTool(ctx, harness.defineTool({
      name: 'dore_eye_report',
      description: 'Full invented-vision report on one image (PNG/JPG): multi-scale ASCII luminance render for composition checks, vertical luminance profile, FFT regularity ratio (mechanical repetition/moire), luminance histogram, global stats. CHARS " .:-=+*#%@": " " = brightest, "@" = darkest. Pure local Pillow+numpy computation, no model, no network.',
      parameters: {
        path: { type: 'string', required: true, description: 'Image path (PNG or JPG).' },
        width: { type: 'integer', description: 'ASCII width in characters; default 110, use ~64 for a quicker look.' },
        fast: { type: 'boolean', description: 'When true, use a 64-column ASCII view for speed.' },
      },
      output: stringOut,
      async execute(args) {
        const extra = []
        if (typeof args.width === 'number' && args.width > 0) extra.push('--width', args.width)
        if (args.fast === true) extra.push('--fast')
        return runEye('report', [args.path].concat(extra))
      },
    }))

    harness.registerTool(ctx, harness.defineTool({
      name: 'dore_eye_zoom',
      description: 'ASCII zoom into a rectangle of an image. mode "ascii": downsampled crop view (composition of a region). mode "pixel": 1:1 pixel window printing every step-th pixel (stroke-level detail). CHARS " .:-=+*#%@": " " = brightest, "@" = darkest.',
      parameters: {
        path: { type: 'string', required: true, description: 'Image path (PNG or JPG).' },
        x0: { type: 'integer', required: true, description: 'Left edge of the rectangle.' },
        y0: { type: 'integer', required: true, description: 'Top edge of the rectangle.' },
        x1: { type: 'integer', required: true, description: 'Right edge of the rectangle.' },
        y1: { type: 'integer', required: true, description: 'Bottom edge of the rectangle.' },
        mode: { type: 'string', enum: ['ascii', 'pixel'], description: 'ascii = downsampled crop view (default); pixel = 1:1 window.' },
        width: { type: 'integer', description: 'ASCII width for mode=ascii (default 90).' },
        step: { type: 'integer', description: 'Stride for mode=pixel (default 2; 1 = every pixel).' },
      },
      output: stringOut,
      async execute(args) {
        const mode = args.mode === 'pixel' ? 'pixel' : 'ascii'
        const extra = ['--crop', args.x0, args.y0, args.x1, args.y1]
        if (mode === 'pixel') {
          if (typeof args.step === 'number' && args.step > 0) extra.push('--step', args.step)
        } else if (typeof args.width === 'number' && args.width > 0) {
          extra.push('--width', args.width)
        }
        return runEye(mode, [args.path].concat(extra))
      },
    }))

    harness.registerTool(ctx, harness.defineTool({
      name: 'dore_eye_metrics',
      description: 'Numeric tone diagnostics on one image: named rectangle stats (mean luminance, dark% below 110, bright% above 200), optional vertical/horizontal luminance profile, FFT regularity ratio on a selected rectangle, and exact-coordinate probes.',
      parameters: {
        path: { type: 'string', required: true, description: 'Image path (PNG or JPG).' },
        regions: {
          type: 'array',
          description: 'Named rectangles to measure.',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              name: { type: 'string', required: true, description: 'Region label.' },
              x0: { type: 'integer', required: true },
              y0: { type: 'integer', required: true },
              x1: { type: 'integer', required: true },
              y1: { type: 'integer', required: true },
            },
          },
        },
        points: {
          type: 'array',
          description: 'Exact sample points.',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              name: { type: 'string', required: true, description: 'Point label.' },
              x: { type: 'integer', required: true },
              y: { type: 'integer', required: true },
            },
          },
        },
        profileAxis: { type: 'integer', enum: [0, 1], description: '0 = vertical profile (top to bottom), 1 = horizontal (left to right).' },
        bins: { type: 'integer', description: 'Profile bins (default 32).' },
        fftRegion: {
          type: 'object',
          additionalProperties: false,
          properties: {
            x0: { type: 'integer', required: true },
            y0: { type: 'integer', required: true },
            x1: { type: 'integer', required: true },
            y1: { type: 'integer', required: true },
          },
          description: 'Rectangle for the FFT regularity check.'
        },
      },
      output: stringOut,
      async execute(args) {
        const extra = []
        if (Array.isArray(args.regions)) {
          for (const r of args.regions) {
            if (r && typeof r === 'object') extra.push('--region', [r.name, r.x0, r.y0, r.x1, r.y1].join(','))
          }
        }
        if (Array.isArray(args.points)) {
          for (const p of args.points) {
            if (p && typeof p === 'object') extra.push('--point', [p.name, p.x, p.y].join(','))
          }
        }
        if (args.profileAxis === 0 || args.profileAxis === 1) extra.push('--profile-axis', args.profileAxis)
        if (typeof args.bins === 'number' && args.bins > 0) extra.push('--bins', args.bins)
        if (args.fftRegion && typeof args.fftRegion === 'object') {
          extra.push('--fft-region', args.fftRegion.x0, args.fftRegion.y0, args.fftRegion.x1, args.fftRegion.y1)
        }
        return runEye('metrics', [args.path].concat(extra))
      },
    }))

    harness.registerTool(ctx, harness.defineTool({
      name: 'dore_eye_ink',
      description: 'Ink deposition map of an image (paper luminance minus image): ASCII ink map plus an n x n mean-ink grid to locate missing or excess ink. "@" = heavy ink, " " = clean paper.',
      parameters: {
        path: { type: 'string', required: true, description: 'Image path (PNG or JPG).' },
        paperLum: { type: 'number', description: 'Paper luminance to subtract (default 232).' },
        width: { type: 'integer', description: 'ASCII width (default 110).' },
        grid: { type: 'integer', description: 'Grid cells per side (default 8).' },
      },
      output: stringOut,
      async execute(args) {
        const extra = []
        if (typeof args.paperLum === 'number') extra.push('--paper-lum', args.paperLum)
        if (typeof args.width === 'number' && args.width > 0) extra.push('--width', args.width)
        if (typeof args.grid === 'number' && args.grid > 0) extra.push('--grid', args.grid)
        return runEye('ink', [args.path].concat(extra))
      },
    }))

    harness.registerTool(ctx, harness.defineTool({
      name: 'dore_eye_critic',
      description: 'Run the per-canto critic script on an image: full ASCII view plus scene-specific region checks with expected tone ranges. Critics exist for cantos 1..5; any other canto (or image without a critic) falls back to the generic full report.',
      parameters: {
        path: { type: 'string', required: true, description: 'Image path (PNG or JPG).' },
        canto: { type: 'integer', description: 'Canto number; when omitted, guessed from a chant_N part of the path.' },
      },
      output: stringOut,
      async execute(args) {
        const extra = []
        if (typeof args.canto === 'number') extra.push('--canto', args.canto)
        return runEye('critic', [args.path].concat(extra))
      },
    }))
  },
}
