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

A real camera is opened only when `--camera` is given (every entry point: `eczis.lab.app`,
`eczis.lab.scan_app`, `tools/ear_scan.py`, `tools/probe_camera.py`; the scan tools error out
without `--camera` or `--simulate`). Without a calibration profile the analysis refuses with
`UNCALIBRATED`. CERTIFIED results reported for the simulated microscope use a calibration profile
identical to the simulator's own parameters (matched model); they show the mechanism, not
robustness to a mis-specified profile. "Digital zoom adds no measurement" is stated by
construction in the ranking (see `pipeline.suggest`), derived in the support-field certificate
matrix, not re-computed per analysis.

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
corner tracker (`artifacts/earscan/20261006-225147`, 22:51:47), calibration-free, `--no-viz`:

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

## Evidence grading of MASt3R output (2026-10-07)

**Lab "R" is not production R.** In this section R/I/U mean multi-keyframe self-consistent /
single-keyframe-supported / unsupported geometry inside the lab tooling. No real input is ever
promoted to production R or M; the real-sweep fractions below are unvalidated output until a
phantom with known geometry has graded them (benchmark protocol at the end of this file).

`eczis.lab.grade_evidence` treats the reconstructor as a proposal to cross-check: the depth test
and calibration-free intrinsics come from the reconstructor's own per-keyframe data (so they are
self-consistency checks), while colour agreement and parallax are independent of it. Per-keyframe
exports (points, confidence, image, Sim3 pose; `tools/spark_mast3r_evidence_patch.py`) are
reprojected into every other keyframe and checked for depth and colour agreement and parallax.
R needs >= 2 other keyframes agreeing with >= 2 deg parallax; a depth conflict (another keyframe
sees a nearer surface) forces U; keyframes after a declared tracking loss cap at I. Reconstructor
confidence is reported, never used to promote. Statuses stay independent: geometry R/I/U,
`lens_calibration: UNCALIBRATED`, `metric_scale: NO_SCALE`.

Real 22:51 sweep (`artifacts/earscan/20261006-225147/mast3r/grading_report.json`; 6 keyframes, loss after frame 90): 294,912 points -> R 50% / I 16% / U 34%
(71k single-view, 50k depth-conflict = drifted sheets, 6k parallax too small). The
`comparison.png` shows the main wall patch R and the overlapping sheets U.

Synthetic-canal validation: **not achieved.** MASt3R's own reconstruction of the synthetic
canal is wrong independent of grading: every keyframe's point map sits ~3 mm from the true wall
(radius ~0.3-0.7 mm vs true 3.5 mm), with Sim3 scales drifting 1.0 -> 0.22, so no class ordering
R < I < U can appear (all ~3.1 mm). The corner tracker gets 0.13 mm on the same frames. Likely
cause: the synthetic frames (320x240, flat texture, near-field LED shading) are far from the
model's training distribution. The grader is instead validated on synthetic multi-keyframe
scenes with known geometry (`tests/lab/test_grade_evidence.py`): consistent wall 72% R,
single view / duplicate cameras never R, floating sheet demoted, post-loss capped.
Open: a synthetic canal MASt3R can actually reconstruct (richer texture, higher resolution).

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

## Physical benchmark protocol (prepared 2026-10-07, not yet run)

No physical run has been made yet; the tools exist and are unit-tested, the physical runs are
the next step and will be recorded here with printer/paper/scaling and the raw JSON.

Diagnostic runs so far (synthetic, not physical): the first benchmark version fitted the tube
axis by PCA of the R points and, on the synthetic canal's short wall patch, chose an axis 90 deg
off the true one (|cos| 0.006, found by independent review) while still printing R < I < U
under `--profile` -- a wrong-axis result that looked like a pass. The benchmark now takes the
axis from the camera trajectory (`--trajectory`) and refuses PCA unless the patch is clearly
longer than wide and wraps the tube; on the same cloud with the true trajectory: R 0.176 /
I 0.272 / U 0.461 mm median physical error, fitted R radius 3.50 mm vs true 3.50
(`artifacts/scan3d/phantom_benchmark.json`). `scale_from_insertion.py --single` on the real
22:51 ear sweep gives path/straight 3.37: too wobbly for a scale, as expected for a freehand
ear scan.

Run order (pre-registered 2026-10-07; thresholds are not changed after seeing results):

0. **Freeze the policy** `uv run python tools/freeze_policy.py` BEFORE measuring the tube or
   looking at any error. Blind-analysis control: every benchmark output records whether the
   grading files/constants/commit still match the freeze (`BLIND` / `CHANGED` / `NOT_FROZEN`).
   Freeze log (the freeze file lives under the ignored `artifacts/`, so the digest is recorded here):
   - 2026-10-07T13:40 — commit 0755a4f, digest `a0667b40…` — before the first physical run.
   - 2026-10-08 ~03:10 — re-frozen at dc6ce4f, digest `7e0f2b69…`, status BLIND. **Constants identical** to
     13:40 (no threshold moved after the 2026-10-07 refusals; 0.16 vs 0.15 was not loosened). File changes
     since: `grade_dense.read_ply` empty-PLY guard; `phantom_benchmark` gained declared axis modes
     (`path` | `optical` | `+fit`, cylinder refinement capped at 15°) and a declared ROI window, both added
     after the capped-can refusal was seen. **Pre-registration:** the formal run names its axis mode and ROI
     in the run command before the tube diameter is entered; a mode picked after seeing an error unblinds
     that run. The 13:40 record is kept beside it as `policy_freeze.2026-10-07T1340.json`.
1. **Print** `uv run python tools/checkerboard_print.py` at 100% (no fit-to-page). Measure the
   20 mm bar in BOTH directions and record it:
   `uv run python tools/print_record.py --printer .. --paper .. --viewer .. --scaling .. --h-mm .. --v-mm ..`
   (> 1% axis mismatch is refused: fix the print, never compensate in software). The calibrator
   reads the record and uses the measured scale for the square size.
2. **Calibrate** `uv run python tools/calibrate_camera.py --camera 1 --square 2.0`: 8-12 views
   that vary position, tilt, rotation and apparent size. The profile records the model,
   intrinsics, distortion, aggregate and per-view RMS, **held-out** RMS (each view scored by a
   lens solved without it), view diversity, accepted count, camera identity, capture mode and
   the print record. `ok` = >= 8 views and RMS <= 1.0 px; unstable held-out or weak diversity
   is printed as a warning and the profile says so.
3. **Measure and record the tube**: inner diameter (calipers), material/surface, visible
   length, probe translation between marks, LED setting, centred or not. Record each sweep
   with its role: `uv run python tools/phantom_record.py <folder> --role straight|twist|repeat-twist|low-motion-control|high-glare-control|wrong-calibration-control --diameter-mm .. --material .. --length-mm .. --insertion-mm .. --led .. --duration-s .. --centred yes|no`
   (binds the frames.npz hash).
4. **Capture** five sweeps: straight, twist, repeat-twist, deliberately low-motion, deliberately
   high-glare. Reconstruct each: `uv run python tools/spark_scan.py <folder> --lens artifacts/calib/lens_camera1.json`.
   Glare/dark sweeps are refused before reconstruction (`SweepRefused`, result.json carries the
   reason); low-motion sweeps raise `NotEnoughMotion`. The wrong-calibration control re-runs one
   valid sweep with `--lens-perturb 0.10` into `mast3r_perturbed/` (statuses say
   `PERTURBED_CONTROL`); errors must rise or R support weaken.
5. **Scale** `uv run python tools/scale_from_insertion.py <folder> --moved-mm <D> --single`
   turns the marked insertion distance into `--scale-units`; path/straight above ~1.3 means the
   sweep wobbled and the operator refuses the scale.
6. **Benchmark** each valid sweep in both modes (global cylinder, and `--profile` which adds the
   per-slice column and the profile-vs-global disagreement):
   `uv run python tools/phantom_benchmark.py <folder>/mast3r/graded.ply --diameter-mm <d> --trajectory <folder>/mast3r/<name>.txt --scale-mm <D> --scale-units <u> --profile`
7. **Assemble the record** `uv run python tools/benchmark_report.py <sweep folders...> --lens artifacts/calib/lens_camera1.json --print-record artifacts/calib/print_record.json`
   -> `artifacts/benchmark/report-<time>/report.{md,json}`: the class table per sweep (points,
   median, P90, P95, within 0.5 mm), diameter error by slice, every pre-registered criterion
   with its evidence, the controls' disposition, and the bindings (frames/cloud/lens hashes,
   MASt3R checkpoint and commit hashes, torch version, repo commit). Missing receipts are listed
   as missing. Every run is preserved, including failures.

Acceptance (pre-registered): calibration >= 8 diverse views with stable held-out error; known
scale recovered (fitted diameter within 0.3 mm of measured, absolute mode); R < I < U physical
error at the median AND at P90/P95 on every valid sweep; twist increases R wall coverage over
straight; repeated twisting sweeps give compatible fitted diameters (<= 0.3 mm apart); the
low-motion and high-glare controls fail explicitly; the wrong-calibration control degrades error
or support; lens_calibration and metric_scale stay independent statuses; the policy freeze
matches. The synthetic 0.2 mm R target is NOT required on the first physical run; the first run
establishes the actual class-conditioned error. Interpretation table: ordering only on the
fitted scan -> leakage/overfitting suspected; similar errors across classes -> the grader lacks
physical discrimination; large R tail -> add metric uncertainty or demote unstable points;
global and profile disagree -> tube/model mismatch; twist improves coverage but worsens error ->
tracking or calibration degraded; a failure control that still yields R -> evidence policy
needs repair.

## Experiment runner (2026-10-08): one declared change at a time, losers kept

Borrowed from the public integer-multiplication exponent race (owner decision 2026-10-08): pin the inputs,
change one knob per candidate, run the negative control on every candidate, keep every result, and test whether
two winners still win together. `src/eczis/lab/experiments.py`, CLI `tools/experiment.py`, tests
`tests/lab/test_experiments.py` (synthetic tube + not-a-tube, end to end through the real benchmark).

- **Pin** a dataset once: files and SHA-256 recorded under `artifacts/experiments/datasets/<name>.json`; a pinned
  file that changes refuses every later run. Kinds: `phantom` (known diameter) or `negative` (not a tube; any
  candidate that accepts it is `CONTROL_BROKEN` and unusable).
- **Candidate** = benchmark knobs only (`axis_from`, `axial_window`, `max_radius_units`, `profile`, `slice_mm`,
  `scale_mm`, `scale_units`); policy constants are not knobs. A candidate may differ from the baseline in exactly
  one knob; `combine a b` is the only compound run (the compatibility check: together must be at least as good as
  the better single, else `incompatible`).
- **Record** per run (append-only `artifacts/experiments/log.jsonl`, full output under `runs/`): dataset digest,
  declared change, git commit, policy-freeze status (`BLIND` required for a formal claim), status
  (`accepted` / `ordering_failed` / `refused` + gate message / `error_N`), the R/I/U medians, seconds.
- **Scorer** (separate decision layer, not a second measurement — an independent re-measurement of the cloud is a
  listed gap): `better` / `same` / `worse` on the R median physical error with a 0.05 mm margin, and only
  when both runs were BLIND, share the policy digest (which fingerprints the benchmark code) and the pinned
  dataset, and both held the strict ordering; anything else is `not_comparable`; negative datasets must be
  refused (`controls_hold`). `combine` calls two winners compatible when together they are within the same
  margin of the better single.
- Declare the candidate BEFORE looking at its result; one chosen after seeing an error is unblinded and should
  say so in its name.

First real runs (policy BLIND, digest `7e0f2b69…`), datasets `can66-endon` (capped can, phantom, 4 files pinned)
and `can-open-end` (negative): baseline `path` refused (residual 0.24), `optical` refused (0.22), `path+fit`
refused (tilt 40.5°), `optical+fit` refused (31.4°), `axial_window=0:1.95` on the path axis kept 0 R points
(error, so last night's window only existed relative to the optical axis), `max_radius_units=1.2` refused
(0.26); the negative control refused under the baseline (0.50). Eight records, all kept: record 5 (`…-can66-endon-window`) was written by the first version of the runner, which called a plain benchmark error "ordering_failed" with empty metrics; the bug was fixed in the same commit and the candidate re-run as `window2` (record 8, `error_1`). The mislabelled line stays in the append-only log as a record of that bug. The phantom is still the
blocker (a can seen end-on is not a tube the camera moved along); the runner is ready for the longer slick tube.
