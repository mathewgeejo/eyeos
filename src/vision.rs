//! Local vision boundary. Camera frames and model output never leave this process. The model
//! adapter is deliberately separate from the control engine so replacing model weights cannot
//! accidentally change dwell or input safety rules.

use crate::gaze::GazeSample;

pub const LANDMARK_COUNT: usize = 478;
const LEFT_IRIS: std::ops::Range<usize> = 468..473;
const RIGHT_IRIS: std::ops::Range<usize> = 473..478;
const LEFT_EYE_OUTER: usize = 33;
const LEFT_EYE_INNER: usize = 133;
const RIGHT_EYE_INNER: usize = 362;
const RIGHT_EYE_OUTER: usize = 263;

#[derive(Debug, Clone, Copy, Default)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct LandmarkFrame {
    pub landmarks: Vec<Landmark>,
    pub timestamp_ms: u64,
    pub face_confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeFeatures {
    pub x: f64,
    pub y: f64,
    pub confidence: f32,
    pub one_eye_fallback: bool,
}

/// Translate iris position into a head-pose-tolerant normalized feature. Screen calibration maps
/// this feature to pixels; it is not treated as a universal gaze coordinate.
pub fn extract_eye_features(frame: &LandmarkFrame) -> Option<EyeFeatures> {
    if frame.landmarks.len() < LANDMARK_COUNT {
        return None;
    }
    let left = normalized_iris(&frame.landmarks, LEFT_IRIS, LEFT_EYE_OUTER, LEFT_EYE_INNER);
    let right = normalized_iris(
        &frame.landmarks,
        RIGHT_IRIS,
        RIGHT_EYE_OUTER,
        RIGHT_EYE_INNER,
    );
    match (left, right) {
        (Some(left), Some(right)) => Some(EyeFeatures {
            x: (left.0 + right.0) / 2.0,
            y: (left.1 + right.1) / 2.0,
            confidence: frame.face_confidence.min(left.2).min(right.2),
            one_eye_fallback: false,
        }),
        (Some(eye), None) | (None, Some(eye)) => Some(EyeFeatures {
            x: eye.0,
            y: eye.1,
            confidence: frame.face_confidence.min(eye.2) * 0.85,
            one_eye_fallback: true,
        }),
        (None, None) => None,
    }
}

fn normalized_iris(
    landmarks: &[Landmark],
    iris: std::ops::Range<usize>,
    outer_index: usize,
    inner_index: usize,
) -> Option<(f64, f64, f32)> {
    let eye_outer = landmarks.get(outer_index)?;
    let eye_inner = landmarks.get(inner_index)?;
    let width = (eye_outer.x - eye_inner.x).hypot(eye_outer.y - eye_inner.y);
    if width < 1e-4 {
        return None;
    }
    let points = &landmarks[iris];
    let iris_x = points.iter().map(|point| point.x).sum::<f32>() / points.len() as f32;
    let iris_y = points.iter().map(|point| point.y).sum::<f32>() / points.len() as f32;
    let confidence = points
        .iter()
        .map(|point| point.confidence)
        .fold(1.0_f32, f32::min);
    let eye_left = eye_outer.x.min(eye_inner.x);
    let eye_center_y = (eye_outer.y + eye_inner.y) / 2.0;
    Some((
        f64::from((iris_x - eye_left) / width),
        f64::from((iris_y - eye_center_y) / width),
        confidence,
    ))
}

pub fn calibrated_sample(
    features: EyeFeatures,
    calibration: &crate::CalibrationProfile,
    timestamp_ms: u64,
) -> GazeSample {
    GazeSample {
        position: calibration.map(features.x, features.y),
        confidence: features.confidence,
        timestamp_ms,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CameraStatus {
    NotStarted,
    Available { devices: usize },
    Unavailable(String),
    ModelMissing,
}

pub fn detect_camera_status() -> CameraStatus {
    #[cfg(feature = "camera")]
    {
        use nokhwa::utils::ApiBackend;
        match nokhwa::query(ApiBackend::MediaFoundation) {
            Ok(devices) if devices.is_empty() => {
                CameraStatus::Unavailable("No Windows camera was detected.".to_owned())
            }
            Ok(devices) => CameraStatus::Available {
                devices: devices.len(),
            },
            Err(error) => {
                CameraStatus::Unavailable(format!("Camera access is unavailable: {error}"))
            }
        }
    }
    #[cfg(not(feature = "camera"))]
    {
        CameraStatus::Unavailable("Camera support was excluded at build time.".to_owned())
    }
}

/// Model weights are opt-in. This type prevents the application from silently falling back to a
/// cloud API or a non-reviewable model when the local asset is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    NotBundled,
    Ready,
}

pub fn model_status() -> ModelStatus {
    // The included NOTICE documents why model weights are intentionally not downloaded at run
    // time. Enabling the `onnx` feature is not enough: the reviewed asset must also be bundled.
    ModelStatus::NotBundled
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_with_eyes() -> LandmarkFrame {
        let mut landmarks = vec![
            Landmark {
                confidence: 1.0,
                ..Landmark::default()
            };
            LANDMARK_COUNT
        ];
        landmarks[LEFT_EYE_OUTER].x = 0.2;
        landmarks[LEFT_EYE_INNER].x = 0.4;
        landmarks[RIGHT_EYE_INNER].x = 0.6;
        landmarks[RIGHT_EYE_OUTER].x = 0.8;
        landmarks[LEFT_EYE_OUTER].y = 0.5;
        landmarks[LEFT_EYE_INNER].y = 0.5;
        landmarks[RIGHT_EYE_INNER].y = 0.5;
        landmarks[RIGHT_EYE_OUTER].y = 0.5;
        for point in &mut landmarks[LEFT_IRIS] {
            point.x = 0.3;
            point.y = 0.5;
        }
        for point in &mut landmarks[RIGHT_IRIS] {
            point.x = 0.7;
            point.y = 0.5;
        }
        LandmarkFrame {
            landmarks,
            timestamp_ms: 1,
            face_confidence: 0.95,
        }
    }

    #[test]
    fn calculates_average_iris_features() {
        let features = extract_eye_features(&frame_with_eyes()).unwrap();
        assert!((features.x - 0.5).abs() < 1e-6);
        assert!((features.y - 0.0).abs() < 1e-6);
        assert!(!features.one_eye_fallback);
    }

    #[test]
    fn rejects_incomplete_landmarks() {
        assert!(
            extract_eye_features(&LandmarkFrame {
                landmarks: vec![],
                timestamp_ms: 0,
                face_confidence: 1.0
            })
            .is_none()
        );
    }
}
