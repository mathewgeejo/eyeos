//! EyeOS core.  The state machine is intentionally independent of the camera and UI so it can
//! be tested without moving the real mouse or saving camera frames.

pub mod calibration;
#[cfg(all(windows, feature = "camera"))]
mod capture;
pub mod config;
pub mod ffi;
pub mod gaze;
#[cfg(all(windows, feature = "native"))]
pub mod gaze_estimator;
pub mod input;
#[cfg(feature = "desktop")]
pub mod persistence;
pub mod sdk;
#[cfg(all(windows, feature = "native"))]
pub mod tracker;
#[cfg(all(windows, feature = "native"))]
pub mod vision;

pub use eye_tracker_core::EyeTracker as TrackingEngine;
pub use eye_tracker_core::{
    AccuracyReport, CalibrationSession, CaptureSource, FrameView, GazeEstimate, InferenceBackend,
    Observation, Quality, TrackerConfig, TrackingState,
};
pub use sdk::EyeTracker;

pub use calibration::{CalibrationPoint, CalibrationProfile};
pub use gaze::{ControlEngine, EngineEvent, GazeSample, InteractionMode, Point, SafetyState};
pub use input::{InputAction, InputController, InputSink};
