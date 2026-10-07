# UNFILED-IP
# ECZIS Microscope Lab (experimental)

Desktop app: live view -> burst capture -> support field on a marked region -> question
certificate -> next-capture guidance. Built on `eczis.experimental`; never changes production
M/R/I/U. Synthetic software reduction-to-practice until validated on a calibrated physical
microscope.

## Run

```powershell
uv sync --all-groups
uv run python -m eczis.lab.app --simulated --glyph 8      # simulated microscope (default)
uv run python tools/probe_camera.py --camera 0             # real device: modes + lockable settings
uv run python -m eczis.lab.app --camera 0 --profile my_profile.json
```

A real camera is opened only when `--camera` is given. Without a calibration profile the
analysis refuses with `UNCALIBRATED`.

## Agent control (MCP)

`uv run python -m eczis.lab.mcp_server` (stdio). Registered for this repo in `.mcp.json` as
`eczis-lab`. Tools (all prefixed `eczis_lab_`): `list_devices` (no device is opened),
`open_simulator`, `open_camera` (explicit opt-in), `close`, `status`, `set_profile`,
`load_profile`, `set_region`, `use_demo_region`, `get_frame` (quality + optional PNG),
`capture_burst`, `clear_bursts`, `send_command` (validated protocol), `connect_serial`,
`analyze`, `investigate` (control level gates execution). Errors are returned as
`{"error": "<what to do next>"}`. Frames returned as images are sent to the calling model.

## 3D canal scan (experimental)

`eczis.lab.scan3d`: one moving camera -> tracked corners -> two-view start + PnP -> triangulated
points graded R/I/U by views, triangulation angle and reprojection error -> wall coverage and
next-motion advice -> PLY export coloured by support. MCP: `eczis_lab_scan3d_simulate`,
`_record`, `_reconstruct`, `_export`. Synthetic ear canal (r0 3.5 mm): R points 0.13 mm median
wall error (60-frame sweep), I 0.22 mm, U 0.37 mm. Metric scale needs one known baseline
(insertion mark / marker); a real camera without a calibrated focal length is reported
UNCALIBRATED. Measurement record only, not a diagnosis.

Scan window: `uv run python -m eczis.lab.scan_app` (camera 1) or `--simulate`. Buttons: Record
sweep (3 s countdown), Snapshot (full-resolution colour PNG), Ultra-zoom (click the live image;
8x pixel view, labelled as adding no detail), Open 3D / Open folder; reconstruction runs on a
worker thread.

Negative result (2026-10-06): masking "probe-fixed" regions (low temporal variance, optionally
textured) plus dropping tracks that never move rescued a synthetic fixed-tip case (R error 0.24 ->
0.11 mm) but degraded clean scans (0.13 -> 0.30-0.35 mm) and collapsed the first real ear sweep
(58 -> 2 frames registered). Reverted. The real sweep's main limit was too little motion.

## MASt3R-SLAM on DGX Spark (2026-10-07, first run)

Built on Spark-1 (GB10, aarch64, CUDA 13, torch 2.14) in `~/eczis-3d/`; setup recorded in
`tools/spark_mast3r_setup.sh` (six compatibility fixes). Same 120-frame ear sweep as the
corner tracker (23:51:47), calibration-free, `--no-viz`:

| | corner tracker (eczis.lab.scan3d) | MASt3R-SLAM |
|---|---|---|
| frames placed | 120/120 | ~90/120 (lost tracking at frame ~90, 6 relocalization attempts failed) |
| points | 132 R / 252 I / 1540 U (sparse) | 103,559 (dense, no evidence grading) |
| keyframes | n/a | 6 |
| time | 1-2 s CPU | 41.6 s wall on GB10 (~4.4 fps) |
| scale | arbitrary | arbitrary (metric head, uncalibrated lens) |

Dense output is a far better surface for coverage and cross-sections; it is also confidently
dense everywhere, including regions seen once, so ECZIS must still grade each point by its own
evidence (views, angle, reprojection) before anything is labelled R. Outputs stay in gitignored
`artifacts/earscan/<time>/mast3r/`. Next: use MASt3R poses as the tracker, keep ECZIS grading.

## Tested devices

- USB endoscope "USB2.0 PC CAMERA" (VID 1908, PID 2311): uncompressed YUYV up to 640x480@30.
  Opens with the default Media Foundation backend as camera 1 on the owner's PC
  (`--camera 1 --size 640x480`); DirectShow could not open it by index. Close the Windows
  Camera app first. Fixed focus: magnification changes with distance (use fixed standoffs).

## Workflow

1. **Mark Region** (drag around the feature) or **Use demo region**.
2. **Analyze**: result is `CERTIFIED: feature present/absent` or `UNDETERMINED` with an answer
   range, the ranked capture options (digital zoom is always listed with zero gain) and the
   per-band support summary.
3. **Investigate** runs the loop at the selected control level:

| Level | Behaviour |
|---|---|
| MANUAL | shows the instruction, executes nothing |
| ASSISTED | same as MANUAL in this version (detection of reached focus/position is planned) |
| SEMI_AUTOMATIC | same as MANUAL in this version |
| AUTOMATIC / AUTONOMOUS | sends the commands, captures, re-analyses until certified or no action helps |

## Command protocol (serial, one line each)

`FOCUS +12` · `STAGE_X -30` · `STAGE_Y 5` · `LIGHT SIDE_LEFT 70` (RING, SIDE_LEFT, SIDE_RIGHT,
TOP, BOTTOM, POLARIZED; 0–100) · `ZOOM 2` · `CAPTURE 16`. Malformed or unknown lines are rejected.
There is deliberately no digital-zoom command: it adds no measurement.

## Calibration profile (JSON)

```json
{"psf_sigma_lr": 0.7, "read_noise": 0.004, "shot_electrons": 700.0,
 "quantization_step": 0.00392, "zoom_levels": [1, 2, 4], "source": "measured:2026-10-06"}
```

## Known limits

- Intervals do not yet include registration uncertainty: decisions were correct in 12/12 seeds but
  one interval missed the truth (strict xfail `test_interval_coverage_includes_registration_uncertainty`).
- Exact posterior per 32x32 region only; cross-zoom alignment assumed from calibration.
- Real USB microscopes usually deliver MJPEG with auto exposure: use the probe tool, prefer
  uncompressed modes, and lock exposure/white balance before trusting any certificate.
- Focus Stack and Compare are planned, not implemented.
