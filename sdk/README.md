# Eye Tracker SDK

Windows x64 inference/capture, portable Rust calibration core, local execution.
The SDK reports gaze; it never moves the OS mouse or sends keyboard events.

## Native library and packaging

When you choose to build the SDK:

```powershell
cargo build --release --lib --no-default-features --features camera
.\scripts\package-sdk.ps1
```

The packager copies an **existing** DLL, wrappers, C header and notices into
`dist/eye-tracker-sdk`. Models/runtimes are embedded and hash-checked on extraction.
Set `EYE_TRACKER_LIBRARY` to an absolute DLL path, or place `eyeos.dll` in
`sdk/native` beside the wrappers. No build runs when a tracker is created.

## Python

```python
from eye_tracker import Tracker

with Tracker({"screen_size": {"x": 1920, "y": 1080},
              "screen_width_mm": 530, "screen_height_mm": 300,
              "viewing_distance_mm": 600}) as tracker:
    tracker.start_camera()
    tracker.start_calibration()
    # In your UI loop, draw calibration_progress()['target'] in physical screen
    # pixels. Poll every ~16ms. Targets include explicit head-position prompts.
    for event in tracker.poll():
        estimate = event["estimate"]
        if estimate and estimate["state"] == "Tracking":
            position = estimate["filtered"]
```

`python/example_overlay.py` is a complete fullscreen calibration/preview UI using
Tk. `example_rgb.py` demonstrates caller-owned RGB frames with optional OpenCV.
Install the wrapper with `pip install ./sdk/python`; it has no runtime Python dependencies.
Use `pip install ./sdk/python[examples]` only for the OpenCV example.

## Node.js / Electron

Run `npm ci` in `sdk/node`, then import `Tracker` from `index.mjs`.

```js
import { Tracker } from './sdk/node/index.mjs';
const tracker = new Tracker({screen_size: {x: 1920, y: 1080}});
tracker.startCamera();
// Host renders calibrationProgress().target after startCalibration().
const events = tracker.poll();
tracker.close();
```

`node/example_camera.mjs` prints five seconds of status/observation events.
`examples/node-overlay` contains a fullscreen Electron calibration and gaze preview.
Install its dependencies separately, set `SCREEN_WIDTH_MM`, `SCREEN_HEIGHT_MM`,
`VIEWING_DISTANCE_MM`, and run `npm start` there. In packaged distributions that
example remains in the source repository. Run capture/inference in an Electron
main process or dedicated worker, never expose the native API directly to webpages.

## Rust and C

Use `eye-tracker-core` alone for a camera/UI/OS-independent engine. Implement
`InferenceBackend` or `CaptureSource` to adapt your own vision/capture code.
The Windows `eyeos::EyeTracker` owns the native adapter and optional webcam.
`examples/tracker.rs` demonstrates the camera lifecycle. The GUI hosts in Python
and Electron demonstrate the complete calibration rendering contract.

The C ABI is defined in `include/eye_tracker.h`. Configuration, observations,
profiles, events and errors are UTF-8 JSON. `et_create` accepts an empty input for
defaults; optional `runtime_root` chooses the managed extraction directory.
Use `et_timestamp_ms` before capture when supplying RGB frames. Frames are RGB24,
with explicit width, height and stride; input data is consumed during the call.
Owned webcam and caller-frame modes cannot be mixed on one tracker.

All calls on a handle must use its creating thread. Read `et_last_error` immediately
after a nonzero result. Release every successful `EyeBuffer` exactly once using
`et_buffer_free`; never free it with a host allocator. Rust panics are caught at
the ABI boundary; recreate a tracker after a panic. Invalid foreign pointers
remain a caller error and cannot be made safe by panic handling.

## Calibration, status and coordinates

Default setup uses nine targets, three centre fixations with head-position prompts,
then thirteen independently collected validation targets. Stable tracking at
30 FPS normally permits the sequence within 45 seconds; bad tracking may take
longer or require a retry. Display each target **immediately** when progress changes,
in unmirrored physical pixels on the configured display. Camera images must be
unmirrored RGB. Convert these coordinates explicitly when your UI uses logical pixels.

The engine fits standardized robust ridge regression using Householder QR.
Calibration retrieval corrects residuals with up to eight distinct fixation groups.
Selection uses leave-target-out folds (including all repeated fixations at that target),
with fold-local scaling/fitting/indexes. Retrieval must improve median error without
worsening p95. Final validation samples never enter fitting/retrieval.

Validation scores individual raw predictions before smoothing/clamping. Reports
include per-region median/p95 pixels, coverage, jitter and inference latency.
Angular errors are **estimates** based on supplied physical dimensions and viewing
distance, assuming a centred observer and a flat screen. Missing measurements do
not produce an angular score. Precision requires ≤1° median, ≤2° p95, ≥80% valid
coverage, and all thirteen validation targets. Pixel previews can still be returned
with `CalibrationRequired`; host applications must gate fine actions on both
`state == "Tracking"` and `precision_validated`.

`TrackingLost`/`StaleFrame` clear the filter and return no coordinate. Observations
outside calibrated pose/feature coverage require recalibration. `quality_score`
is an observable image/geometry heuristic, **not** a calibrated probability. Missing
landmark scores stay `null`. `estimated_error_px` is the independent validation p95,
not a guaranteed per-frame bound. `latency_ms` includes capture-to-consumption delay.

On failure, call `extendCalibration` / `extend_calibration` to collect newly labeled
fixations at suggested weak regions, then repeat independent validation. No implicit
learning from clicks, generated predictions, or validation labels occurs.

Exported profiles are versioned and bound to camera, display, model, configuration
and feature/preprocessing version. Owned-camera profiles also bind camera metadata
and resolution; RGB profiles bind frame resolution. Old EyeOS maps need fresh calibration. Generic
SDK JSON export is plaintext for host-managed storage; EyeOS wraps the same profile
in Windows DPAPI. Raw frames are neither exported nor retained in profiles.

## Tests and real accuracy evaluation

```powershell
cargo test --workspace --all-targets
cargo test -p eye-tracker-core
$env:EYE_TRACKER_LIBRARY = (Resolve-Path target\debug\deps\eyeos.dll).Path
python -m unittest discover -s sdk/python -p test_sdk.py
node --test sdk/node/test.mjs
```

`examples/evaluate.rs` compares the corrected baseline and selected retrieval
mapping on separately recorded `LabeledObservation[]` JSON from an independent
session. Include rejected observations in the dataset to measure coverage honestly.
Collect consented sessions with glasses, different lighting, edges and head movement;
do not treat synthetic tests or model initialization as proof of webcam accuracy.
No real-user accuracy dataset ships with this repository.

Camera shutdown signals its worker promptly. A Windows camera driver may block
inside a frame-read call; the SDK does not hang the host waiting for that call.
The worker closes the camera when the driver returns. Avoid reopening the same
camera immediately after stopping a blocked driver.
