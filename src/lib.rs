//! EyeOS core.  The state machine is intentionally independent of the camera and UI so it can
//! be tested without moving the real mouse or saving camera frames.

pub mod calibration;
pub mod config;
pub mod gaze;
pub mod input;
pub mod persistence;
pub mod tracker;
pub mod vision;

pub use calibration::{CalibrationPoint, CalibrationProfile};
pub use gaze::{ControlEngine, EngineEvent, GazeSample, InteractionMode, Point, SafetyState};
pub use input::{InputAction, InputController, InputSink};
