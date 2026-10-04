//! Camera-, UI-, and OS-independent gaze tracking. No network or desktop input.
mod calibration;
mod engine;
mod math;
mod preprocessing;
mod session;
mod types;

pub use calibration::{CalibrationPoint, CalibrationProfile, Fixation, Regression};
pub use engine::EyeTracker;
pub use preprocessing::normalize_gaze_vector;
pub use session::{
    CALIBRATION_SAMPLES_PER_TARGET, CalibrationOutcome, CalibrationSession, LabeledObservation,
    evaluate,
};
pub use types::*;
