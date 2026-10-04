# Redistribution

The engine and bindings use the repository's MIT OR Apache-2.0 license. The MIT
license text is included; the Apache-2.0 text shipped with MediaPipe is also included.

The Windows DLL embeds the existing hash-pinned MediaPipe Face Landmarker task,
MediaPipe runtime, OpenVINO runtime and Open Model Zoo networks. Packaging copies
their existing provenance notices; extraction verifies the hashes before loading.
The SDK never downloads a model during tracking.

MediaPipe/OpenVINO runtime source and Open Model Zoo models are Apache-2.0 as
described in the bundled notices. Preserve model source URLs, model cards and
third-party notices when redistributing. The MediaPipe model task is a separately
distributed Google asset; the original repository notice does not grant blanket
rights to every downstream distribution. Check its applicable model terms for
your release. Packaging alone does not certify those rights or gaze accuracy.

The Node wrapper uses Koffi (MIT); retain its license when bundling npm dependencies.
Electron and optional OpenCV examples have their own dependency notices. The
packager does not copy node_modules or install dependencies.
