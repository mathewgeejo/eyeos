# Vision model notice

EyeOS uses a model adapter designed for a 478-point face-and-iris landmark model. Model weights
are not committed in this prototype. Before enabling the ONNX feature, add a redistributable,
Apache-2.0-compatible model here, record its immutable source URL and SHA-256 in this file, and
validate it with the intended user on the target webcam.

The application must not download model weights at runtime. This preserves the offline-only
privacy guarantee and makes the deployed model reviewable.
