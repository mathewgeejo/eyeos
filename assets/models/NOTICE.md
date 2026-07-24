# Face Landmarker model

EyeOS embeds this exact local task bundle. It is passed to MediaPipe from memory and is never
downloaded at runtime.

| Field | Value |
| --- | --- |
| Model | MediaPipe Face Landmarker, `float16/1` |
| Source | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` |
| SHA-256 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| Landmarks | 478 normalized face/iris landmarks |
| Runtime | MediaPipe C API 0.10.35, CPU/XNNPACK |

The source code and C API headers used by EyeOS are Apache-2.0. The model asset is distributed
by Google separately; retain the source URL and review its current model-card and distribution
terms before redistribution or a production release.

`assets/runtime/mediapipe/LICENSE` contains the license shipped with the pinned MediaPipe Windows
runtime. The runtime is embedded in the EXE, hash-checked, then extracted only to the current
user's EyeOS runtime directory so Windows can load it.
