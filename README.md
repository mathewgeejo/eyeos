# EyeOS

EyeOS is a safety-first, offline Windows 11 desktop eye-control prototype written in Rust.

It is deliberately scoped to the normal signed-in desktop. It does not bypass Windows UAC,
the lock screen, or the secure desktop. Raw camera frames, calibration samples, and text
predictions remain on the device.

## Current prototype

The executable provides the native floating control blob, a safe training environment,
calibration math, dwell/drag/click state machine, local preferences, and a Windows input
backend. It starts in **dry-run mode**: actions are logged to the in-app activity view, not
sent to Windows. A caregiver can explicitly enable live desktop input after training.

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

## Safety

- Control starts paused and dry-run enabled.
- Lost tracking releases any held button and pauses control.
- Live Windows input never runs on the Windows secure desktop and is limited by normal Windows
  integrity rules.
- A real user evaluation is required before independent use.

