use serde::{Deserialize, Serialize};

pub const PROFILE_VERSION: u32 = 2;
pub const FEATURE_COUNT: usize = 16;
pub const MODEL_ID: &str = "mediapipe-64184e229b26/adas-0002/preprocess-v2";

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}
impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    pub fn distance_to(self, other: Self) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }
    pub fn lerp(self, other: Self, alpha: f64) -> Self {
        Self::new(
            self.x + (other.x - self.x) * alpha,
            self.y + (other.y - self.y) * alpha,
        )
    }
    pub fn finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

/// Scores describe observable image/geometry quality, not a neural posterior probability.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Quality {
    pub landmark_confidence: Option<f32>,
    pub image_score: f32,
    pub eye_opening: [f32; 2],
    pub crops_valid: bool,
}
impl Default for Quality {
    fn default() -> Self {
        Self {
            landmark_confidence: None,
            image_score: 1.0,
            eye_opening: [0.2; 2],
            crops_valid: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Observation {
    pub x: f64,
    pub y: f64,
    pub gaze_direction: [f64; 3],
    /// Yaw, pitch, roll in degrees, before crop alignment.
    pub head_pose: [f64; 3],
    pub left_iris: Option<Point>,
    pub right_iris: Option<Point>,
    pub face_center: Point,
    pub face_scale: f64,
    pub confidence: f32,
    pub quality: Quality,
    pub one_eye_fallback: bool,
    pub blink: bool,
    pub timestamp_ms: u64,
    pub inference_latency_ms: f64,
}
impl Observation {
    pub fn features(self) -> [f64; FEATURE_COUNT] {
        let left = self.left_iris.unwrap_or_default();
        let right = self.right_iris.unwrap_or_default();
        [
            self.x,
            self.y,
            self.x * self.x,
            self.x * self.y,
            self.y * self.y,
            self.head_pose[0],
            self.head_pose[1],
            self.head_pose[2],
            self.face_center.x,
            self.face_center.y,
            self.face_scale,
            left.x,
            left.y,
            right.x,
            right.y,
            if self.left_iris.is_some() && self.right_iris.is_some() {
                1.0
            } else {
                0.0
            },
        ]
    }
    pub fn usable(self, minimum: f32) -> bool {
        self.features().iter().all(|v| v.is_finite())
            && self.gaze_direction.iter().all(|v| v.is_finite())
            && self.confidence.is_finite()
            && self.confidence >= minimum
            && self.confidence <= 1.0
            && !self.blink
            && !self.one_eye_fallback
            && self.quality.crops_valid
            && self.quality.image_score.is_finite()
            && self.quality.image_score >= 0.35
            && self
                .quality
                .eye_opening
                .iter()
                .all(|v| v.is_finite() && *v >= 0.075)
            && self
                .quality
                .landmark_confidence
                .is_none_or(|v| v.is_finite() && (0.0..=1.0).contains(&v))
            && self.inference_latency_ms.is_finite()
            && self.inference_latency_ms >= 0.0
            && self.head_pose[0].abs() <= 45.0
            && self.head_pose[1].abs() <= 35.0
            && self.head_pose[2].abs() <= 35.0
    }
}

/// Borrowed RGB24; padding between rows is allowed. Timestamps use one monotonic clock.
pub struct FrameView<'a> {
    pub rgb: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub stride_bytes: usize,
    pub timestamp_ms: u64,
}
impl FrameView<'_> {
    pub fn validate(&self) -> Result<(), String> {
        let row = (self.width as usize)
            .checked_mul(3)
            .ok_or("frame dimensions overflow")?;
        let bytes = self
            .stride_bytes
            .checked_mul(self.height as usize)
            .ok_or("frame dimensions overflow")?;
        if self.width < 2
            || self.height < 2
            || self.width > 8192
            || self.height > 8192
            || self.stride_bytes < row
            || self.rgb.len() < bytes
        {
            return Err("invalid RGB24 frame dimensions, stride, or length".into());
        }
        Ok(())
    }
}
pub trait InferenceBackend {
    fn model_id(&self) -> &str;
    fn infer(&mut self, frame: FrameView<'_>) -> Result<Option<Observation>, String>;
}
pub trait CaptureSource {
    fn camera_id(&self) -> &str;
    fn next_frame(&mut self) -> Result<Option<FrameView<'_>>, String>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TrackerConfig {
    pub screen_size: Point,
    pub screen_width_mm: Option<f64>,
    pub screen_height_mm: Option<f64>,
    pub viewing_distance_mm: Option<f64>,
    pub camera_id: String,
    pub display_id: String,
    pub model_id: String,
    pub minimum_quality: f32,
    pub maximum_frame_age_ms: u64,
    pub median_target_deg: f64,
    pub p95_target_deg: f64,
}
impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            screen_size: Point::new(1920.0, 1080.0),
            screen_width_mm: None,
            screen_height_mm: None,
            viewing_distance_mm: None,
            camera_id: "webcam:0".into(),
            display_id: "primary".into(),
            model_id: MODEL_ID.into(),
            minimum_quality: 0.72,
            maximum_frame_age_ms: 200,
            median_target_deg: 1.0,
            p95_target_deg: 2.0,
        }
    }
}
impl TrackerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.screen_size.finite()
            || self.screen_size.x <= 0.0
            || self.screen_size.y <= 0.0
            || !self.minimum_quality.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_quality)
            || self.maximum_frame_age_ms == 0
            || self.maximum_frame_age_ms > 5000
            || !self.median_target_deg.is_finite()
            || !self.p95_target_deg.is_finite()
            || self.median_target_deg <= 0.0
            || self.p95_target_deg < self.median_target_deg
            || [&self.camera_id, &self.display_id, &self.model_id]
                .iter()
                .any(|v| v.is_empty())
            || [
                self.screen_width_mm,
                self.screen_height_mm,
                self.viewing_distance_mm,
            ]
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err("invalid tracker configuration".into());
        }
        Ok(())
    }
    pub fn angular_error(&self, predicted: Point, target: Point) -> Option<f64> {
        let (w, h, d) = (
            self.screen_width_mm?,
            self.screen_height_mm?,
            self.viewing_distance_mm?,
        );
        let ray = |p: Point| {
            [
                (p.x / self.screen_size.x - 0.5) * w,
                (p.y / self.screen_size.y - 0.5) * h,
                d,
            ]
        };
        let a = ray(predicted);
        let b = ray(target);
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm = |v: [f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        Some(
            (dot / (norm(a) * norm(b)))
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees(),
        )
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrackingState {
    Tracking,
    Uncalibrated,
    CalibrationRequired,
    TrackingLost,
    StaleFrame,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GazeEstimate {
    pub timestamp_ms: u64,
    pub raw: Option<Point>,
    pub filtered: Option<Point>,
    pub quality: Quality,
    pub quality_score: f32,
    pub estimated_error_px: Option<f64>,
    pub state: TrackingState,
    pub precision_validated: bool,
    pub retrieval_used: bool,
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegionError {
    pub region: String,
    pub median_px: f64,
    pub p95_px: f64,
    pub samples: usize,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AccuracyReport {
    pub median_error_px: f64,
    pub p95_error_px: f64,
    pub median_error_deg: Option<f64>,
    pub p95_error_deg: Option<f64>,
    pub valid_sample_coverage: f64,
    pub jitter_px: f64,
    pub median_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub regions: Vec<RegionError>,
    pub samples: usize,
    pub precision_passed: bool,
}
