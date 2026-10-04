//! Minimal native camera/calibration/gaze example; UI hosts render progress.target.
use eyeos::{EyeTracker, TrackerConfig};
use std::{
    thread,
    time::{Duration, Instant},
};
fn main() -> Result<(), String> {
    let config = TrackerConfig::default();
    let mut tracker = EyeTracker::new(config, std::env::temp_dir().join("eye-tracker-example"))?;
    tracker.start_camera(0)?;
    let until = Instant::now() + Duration::from_secs(5);
    while Instant::now() < until {
        for event in tracker.poll()? {
            println!("{}", serde_json::to_string(&event).unwrap());
        }
        thread::sleep(Duration::from_millis(16));
    }
    // A graphical host calls start_calibration(), draws calibration_progress().target
    // in physical screen pixels, and waits for Completed's accuracy report.
    tracker.stop_camera();
    Ok(())
}
