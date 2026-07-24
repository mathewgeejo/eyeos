# EyeOS

EyeOS is a safety-first, offline Windows 11 desktop eye-control prototype written in Rust.

It is deliberately scoped to the normal signed-in desktop. It does not bypass Windows UAC,
the lock screen, or the secure desktop. Raw camera frames, calibration samples, and text
predictions remain on the device.

## Launch behaviour

Running `eyeos.exe` opens only a small transparent control blob in the bottom-left corner.
There is no startup dashboard or menu. With a valid local tracker and saved calibration, it
starts tracking immediately. An 800 ms dwell on the blob opens the compact 3×3 action pad;
choosing an action returns to the blob. The gaze keyboard is also a compact bottom overlay and
supports direct dwell selection, without requiring a mouse click.

Caregiver-only surfaces remain available through `--setup` and `--training`. The latter always
uses dry-run input. Live desktop input is enabled only after a reviewed local model and saved
calibration are present; if either is absent, EyeOS stays paused instead of guessing.

The gaze pipeline accepts 478-point face/iris landmarks through `vision::LandmarkFrame`.
The repository intentionally does not bundle a third-party model weight until its source,
licence, conversion reproducibility, and evaluation are recorded in `assets/models/NOTICE.md`.
This avoids pretending that a placeholder model is safe or accurate enough for assistive use.

## Build

```powershell
cargo test
cargo run -- --training
cargo build --release
```

`eyeos.exe` is at `target\release\eyeos.exe`. The first build needs Rust stable, Visual
Studio Build Tools with the C++ desktop workload, and the Windows 11 SDK.

## Commands

```text
eyeos.exe                  Start the floating control blob.
eyeos.exe --training       Open the safe training environment.
eyeos.exe --setup          Open caregiver calibration/setup.
eyeos.exe --install-autostart
eyeos.exe --reset-profile
```

For developer verification only, `eyeos.exe --simulate-gaze` maps the physical mouse position
through the gaze state machine in dry-run mode. It never sends input to another application.

## Safety

- Control starts paused and dry-run enabled.
- Lost tracking releases any held button and pauses control.
- Live Windows input never runs on the Windows secure desktop and is limited by normal Windows
  integrity rules.
- A real user evaluation is required before independent use.
