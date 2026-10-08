# UNFILED-IP
# Capability doors — things the bench showed are needed, with the evidence that showed it

A door is logged when work makes a previously unavailable capability possible or shows it is
required. Each entry keeps the triggering evidence, the prior art looked at, and the routing
decision. Evidence words: observed / reproduced / measured / verified. "Novel" is never claimed
without a prior-art search recorded here.

## DOOR-20261007-01 — Smooth / specular interiors need projected texture and specular control

**Triggering evidence (observed, 2026-10-07 live bench, real endoscope, policy frozen 13:40):**
- Sweep `20261007-212758`, aluminium tube: refused at the image-quality gate, clipped-bright
  median 12.3 % (threshold 5 %). Record: `summary.json` status `REFUSED: GLARE`.
- Sweep `20261007-213024`, matte package interior: passed glare (3.3 %), refused for motion,
  tracked points moved 3.5 px (threshold 25). Record: `summary.json` status `REFUSED: NOT ENOUGH MOTION`.
- Sweep `20261007-215934`, same interior, enough motion: local tracker placed 25 of 120 frames in
  3 segments, R 6 / I 50 / U 989; MASt3R-SLAM on the Spark with the calibrated lens returned a
  **0-vertex** cloud and 11 poses with ~0.004 units of total translation. Record:
  `mast3r/result.json` status `REFUSED: reconstructor returned 0 points`.
- The operator's stated target is plumbing: "pipes are smooth and slick".

**Why (physics):** every camera-only method here (LK tracking, MASt3R) needs wall appearance
that differs from place to place. A slick wall gives saturated specular hotspots, featureless
regions between them, and reflections that move with the camera (violating the static-scene
assumption). The refusals are correct behaviour; the imaging is starved.

**Prior art (looked at, not exhaustive):** industrial measurement borescopes project laser
dots/lines or a shadow stripe from the tip for metric 3D on reflective interiors (Olympus
IPLEX stereo/laser tips, Waygate Mentor Visual iQ 3D phase measurement); cross-polarised
illumination is standard for suppressing specular reflection in dermatology and machine
vision; structured light (dot grids, ring projectors) is standard in pipe profiling. None of
this is novel; what ECZIS adds is per-point evidence grading on top of whatever texture source
is used, and explicit refusal when there is none.

**What it would take:**
1. Projected texture at the tip: a laser dot-grid/ring cap or a separate laser pointer with a
   grid diffraction cap shone down the pipe (cheap test, no code change: the motion gate,
   Spark reconstruction, grading and phantom benchmark all run as-is on the dots).
2. Specular control: diffuser on the LED ring; cross-polarised LED + lens polariser.
3. Later: use the specular highlight curve on a cylinder as a geometric cue (weaker, model-based).

**Value:** inside this project it is the difference between "works on textured canals" and
"works on pipes"; outside it, the same probe + grading applies to any reflective bore
(hydraulic lines, heat-exchanger tubes, gun barrels, ducts).

**Routing (owner decision pending):** immediate cheap test with a laser pointer + grid cap and
LED tape on the aluminium tube; if dots track, add a projected-texture mode to the protocol
(record the projector as part of the capture mode) before any claim about pipes.

## DOOR-20261007-02 — Evidence-labelled image clean-up from a burst (text on packaging)

**Trigger (observed):** operator snapshots of a drink-mix package: large print readable, fine
print not, single frames defocused (lens focus 1-3 cm). Request: "clean up how the writing looks".

**Built:** `tools/burst_enhance.py` — ECC registration of a recorded burst to a reference frame,
mean fusion, per-pixel R/I/U support by frame agreement, optional Richardson-Lucy deblur labelled
as a re-rendering that adds no evidence. No generative model.

**Measured (2026-10-07 22:04, burst `artifacts/earscan/20261007-220435`, 120 frames of the
package's warning text, real endoscope):** 120/120 frames aligned (ECC correlation median
0.998); support R 99.0 % / I 0.9 % / U 0.1 % of pixels; noise removed from the reference frame
s.d. 0.0137 (intensity units). Fine print that is grainy in the single frame reads cleanly in
`fused.png` / `fused_deblur.png` ("Do not exceed 2 packets per day", "consult your ... practitioner
before use if ... pregnant, nursing, taking medication", "out of reach of children", "B6 (as
pyridoxine ...)"). The same burst is REFUSED for 3D (glare 17.2 %): image clean-up and 3D have
different gates, and the record says which applied. Operator-side artifacts stay local (gitignored).

**Also measured on a moving subject (operator's own eye, 120-frame burst `20261007-220638`):**
frame-to-frame motion median 10 px, p90 39 px; with the default acceptance only 27/120 frames
aligned (cc >= 0.97), R 60 % / I 27 % / U 12 %. Found and fixed a flaw: the corneal LED hotspot
is constant relative to the camera and was counting as supported; clipped-bright pixels are now
U regardless of agreement. The tool reports frames rejected and the forced-U fraction.

## DOOR-20261007-03 — Corneal reflection imaging (forensic): method works, this sensor cannot

**Trigger (observed):** operator eye shots show two reflections in the cornea; question raised
whether what the subject was looking at can be recovered ("could be major in crime labs").

**Prior art:** Jenkins & Kerr 2013 (PLoS ONE) recovered identifiable bystander faces from
corneal reflections in 39-megapixel portraits; the cornea acts as a convex mirror of radius
~7.8 mm (focal length ~3.9 mm), so an object of size S at distance d forms an image of
S x 3.9 / (d + 3.9) mm on the cornea.

**Measured here (640x480 endoscope, calibrated fx 816 px, ~23 mm from the eye assuming an
11.7 mm iris -> 35 px/mm at the cornea):** LED-ring highlight ~50 px; a second reflection
~50 px across, identical colour in two shots (RGB 67/90/110), at a near-constant offset from
the highlight. My inference from size/offset was "the probe's own tip"; the operator's ground
truth is **the monitor: an LG 45-inch curved ultrawide with a water wallpaper behind open
terminal windows** (the water explains the teal; the windows the dark structure), and the
white disc is the endoscope. Enlarged 8x, the reflection shows the screen's tilted rectangle
and a brighter band, no window detail: one terminal text line is under a pixel on this sensor. Lesson recorded: the reflection
geometry alone was ambiguous at this resolution and a wrong inference was stated with more
confidence than the data allowed; ground truth from the scene must be asked for before any
identification is claimed. Predicted corneal images
at this camera: 27-inch monitor at 0.7 m ~116 px (a blob), a face at 0.6 m ~45 px (no identity),
a hand at 0.4 m ~61 px. Conclusion: the analysis path (locate iris, locate highlight, crop at
native pixels, labelled enhancement, convex-mirror geometry) is sound and cheap; recovering
scene identity needs a sensor with roughly 10-50x the pixels on the cornea. Not novel; a
possible ECZIS angle is the evidence labelling of such enhancements (what is supported by how
many frames) rather than the recovery itself. Routing: roadmap note only; no claim.

## DOOR-20261007-04 — The phone as the sensor: live bridge, LiDAR scale, deflectometry for slick pipes

**Trigger:** operator question "could you ever see what my phone shows in a live camera view";
phone is an iPhone with LiDAR.

**Built tonight:** `eczis.lab.phone_cam` — a local HTTPS page (self-signed, LAN only) the phone
opens in its browser; frames stream to the PC over a WebSocket, the coach caption is shown on
the phone and spoken by the phone; `scan_app --phone` runs the whole scan window on it, with
full-resolution stills on Snapshot. `tools/lidar_stream.py` — Record3D USB receiver (colour,
LiDAR depth in metres, camera pose, phone-reported intrinsics) writing the lab's layout; written
before being run against a phone, so unverified.

**Why it matters:** (1) the phone's depth and pose are measurements INDEPENDENT of the
reconstructor, so the evidence grader can check MASt3R against them instead of against itself;
(2) metric scale and intrinsics arrive with every frame, closing NO_SCALE/UNCALIBRATED without a
ruler; (3) **deflectometry** — showing a known stripe pattern on the monitor/phone and recording
its reflection in a shiny surface is the standard way to measure mirror-like surfaces; the
reflection that broke the matcher becomes the measurement (prior art: phase-measuring
deflectometry, Knauer/Kaminski/Häusler 2004; used industrially for car bodies and optics); the
corneal-reflection result above is the same physics on a convex mirror. Also: the phone can watch
the endoscope from outside and give the probe's insertion distance from its own tracking.

**Routing:** build order per the operator's plan: web bridge (done) -> Record3D depth/pose
(needs the app on the phone + USB) -> deflectometry test on the aluminium tube with phase-shifted
stripes on the LG monitor -> corneal reflection at phone resolution. Keep compressed live video
for guidance only; evidence comes from stills/RAW/depth.
