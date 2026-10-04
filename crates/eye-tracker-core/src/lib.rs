//! Camera-, UI-, and OS-independent gaze tracking. No network or desktop input.
mod calibration;
mod engine;
mod math;
mod session;
mod types;

pub use calibration::{CalibrationPoint, CalibrationProfile, Fixation, Regression};
pub use engine::EyeTracker;
pub use session::{CalibrationOutcome, CalibrationSession, CALIBRATION_SAMPLES_PER_TARGET};
pub use types::*;

