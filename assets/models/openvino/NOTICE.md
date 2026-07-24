# Local OpenVINO gaze-estimation assets

EyeOS embeds these CPU models and extracts them only to its per-user managed runtime folder when
the program starts. They are never downloaded at runtime and no camera frames leave the device.

| Model | Source | Format | Purpose |
| --- | --- | --- | --- |
| `head-pose-estimation-adas-0001` | Open Model Zoo 2022.1 | FP16 IR | Yaw, pitch, and roll from a face crop |
| `gaze-estimation-adas-0002` | Open Model Zoo 2022.1 | FP16 IR | Binocular 3D gaze vector from rotated eye crops and head pose |

Source repository: `https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/`.
The Open Model Zoo models and the OpenVINO runtime are Apache-2.0. The runtime DLLs are taken
from the pinned official OpenVINO Windows wheel and are hash-checked before extraction.
