//! Windows adapter for the portable engine. Does not depend on the EyeOS UI and
//! never injects input. A tracker is used on one owning thread.
#[cfg(all(windows, feature = "native"))]
use crate::{
    gaze_estimator::GazeEstimator,
    tracker::{
        LocalTracker, MediaPipeFaceLandmarker, TrackerEvent, TrackerStatus, extract_runtime,
    },
};
use eye_tracker_core::{CalibrationOutcome, EyeTracker as Engine, *};
use serde::Serialize;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
#[cfg(not(all(windows, feature = "native")))]
#[derive(Debug, Clone, Serialize)]
pub enum TrackerStatus {
    GazeUnavailable { detail: String },
}

#[cfg(all(windows, feature = "native"))]
pub struct NativeBackend {
    estimator: GazeEstimator,
    landmarker: MediaPipeFaceLandmarker,
    last_timestamp: Option<u64>,
}
#[cfg(all(windows, feature = "native"))]
impl NativeBackend {
    pub fn load(root: &Path) -> Result<Self, String> {
        let runtime = extract_runtime(root).map_err(|e| e.to_string())?;
        Ok(Self {
            estimator: GazeEstimator::load(root).map_err(|e| e.to_string())?,
            landmarker: MediaPipeFaceLandmarker::load(&runtime).map_err(|e| e.to_string())?,
            last_timestamp: None,
        })
    }
}
#[cfg(all(windows, feature = "native"))]
impl InferenceBackend for NativeBackend {
    fn model_id(&self) -> &str {
        MODEL_ID
    }
    fn infer(&mut self, frame: FrameView<'_>) -> Result<Option<Observation>, String> {
        frame.validate()?;
        if self.last_timestamp.is_some_and(|t| frame.timestamp_ms <= t) {
            return Err("frame timestamps must strictly increase".into());
        }
        self.last_timestamp = Some(frame.timestamp_ms);
        let packed;
        let rgb = if frame.stride_bytes == frame.width as usize * 3 {
            &frame.rgb[..frame.stride_bytes * frame.height as usize]
        } else {
            packed = frame
                .rgb
                .chunks(frame.stride_bytes)
                .take(frame.height as usize)
                .flat_map(|row| row[..frame.width as usize * 3].iter().copied())
                .collect::<Vec<_>>();
            &packed
        };
        let started = Instant::now();
        let landmarks = self
            .landmarker
            .detect_rgb(
                rgb,
                frame.width as i32,
                frame.height as i32,
                frame.timestamp_ms,
            )
            .map_err(|e| e.to_string())?;
        match landmarks {
            Some(landmarks) => {
                match self
                    .estimator
                    .estimate(rgb, frame.width, frame.height, &landmarks)
                {
                    Ok(mut observation) => {
                        observation.inference_latency_ms = started.elapsed().as_secs_f64() * 1000.0;
                        Ok(Some(observation))
                    }
                    Err(_) => Ok(None), // Quality rejection is tracking loss, not a fabricated gaze.
                }
            }
            None => Ok(None),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationProgress {
    pub target: Option<Point>,
    pub phase: String,
    pub instruction: String,
    pub completed: usize,
    pub total: usize,
    pub stable_samples: usize,
    pub suggested_targets: Vec<Point>,
}
#[derive(Debug, Clone, Serialize)]
pub struct SdkEvent {
    pub estimate: Option<GazeEstimate>,
    pub observation: Option<Observation>,
    pub calibration: Option<CalibrationOutcome>,
    pub progress: Option<CalibrationProgress>,
    pub status: Option<TrackerStatus>,
}

pub struct EyeTracker {
    engine: Engine,
    root: PathBuf,
    #[cfg(all(windows, feature = "native"))]
    backend: Option<NativeBackend>,
    #[cfg(all(windows, feature = "native"))]
    camera: Option<LocalTracker>,
    session: Option<CalibrationSession>,
    clock: Instant,
    last_frame_ms: Option<u64>,
    loss_reported: bool,
    capture_identity: Option<String>,
}
impl EyeTracker {
    pub fn new(config: TrackerConfig, runtime_root: PathBuf) -> Result<Self, String> {
        if config.model_id != MODEL_ID {
            return Err("bundled backend/model identity mismatch".into());
        }
        Ok(Self {
            engine: Engine::new(config)?,
            root: runtime_root,
            #[cfg(all(windows, feature = "native"))]
            backend: None,
            #[cfg(all(windows, feature = "native"))]
            camera: None,
            session: None,
            clock: Instant::now(),
            last_frame_ms: None,
            loss_reported: false,
            capture_identity: None,
        })
    }
    pub fn config(&self) -> &TrackerConfig {
        self.engine.config()
    }
    pub fn timestamp_ms(&self) -> u64 {
        self.clock.elapsed().as_millis() as u64
    }
    pub fn clock(&self) -> Instant {
        self.clock
    }
    pub fn engine(&self) -> &Engine {
        &self.engine
    }
    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }
    fn bind_capture(&mut self, identity: String) {
        if self
            .capture_identity
            .as_ref()
            .is_some_and(|old| old != &identity)
        {
            self.session = None;
        }
        if self
            .engine
            .profile()
            .is_some_and(|p| p.capture_identity.as_ref() != Some(&identity))
        {
            self.engine.clear_profile();
        }
        self.capture_identity = Some(identity);
    }
    pub fn import_profile(&mut self, json: &str) -> Result<(), String> {
        if self.session.as_ref().is_some_and(|s| s.target().is_some()) {
            return Err("cancel calibration before importing a profile".into());
        }
        let profile: CalibrationProfile = serde_json::from_str(json).map_err(|e| e.to_string())?;
        if self
            .capture_identity
            .as_ref()
            .is_some_and(|id| profile.capture_identity.as_ref() != Some(id))
        {
            return Err(
                "camera identity or capture dimensions changed; recalibration required".into(),
            );
        }
        self.engine.set_profile(profile)?;
        self.session = None;
        Ok(())
    }
    pub fn reconfigure(&mut self, config: TrackerConfig) -> Result<(), String> {
        if config.model_id != MODEL_ID || config.camera_id != self.config().camera_id {
            return Err(
                "stop and recreate the tracker to change its backend/camera identity".into(),
            );
        }
        self.engine = Engine::new(config)?;
        self.session = None;
        Ok(())
    }
    pub fn start_camera(&mut self, index: u32) -> Result<(), String> {
        #[cfg(all(windows, feature = "camera"))]
        {
            if self.backend.is_some() {
                return Err("caller-supplied frames and owned-camera modes cannot be mixed".into());
            }
            if self.camera.is_some() {
                return Err("camera already running".into());
            }
            if self.config().camera_id != format!("webcam:{index}") {
                return Err("camera index differs from configured camera identity".into());
            }
            self.camera = Some(
                LocalTracker::start_at(self.root.clone(), index, self.clock)
                    .map_err(|e| e.to_string())?,
            );
            Ok(())
        }
        #[cfg(not(all(windows, feature = "camera")))]
        {
            let _ = index;
            Err("Windows camera backend is not enabled".into())
        }
    }
    pub fn stop_camera(&mut self) {
        #[cfg(all(windows, feature = "native"))]
        {
            self.camera = None;
        }
        self.engine.reset_filter();
        self.last_frame_ms = None;
    }
    pub fn has_camera(&self) -> bool {
        #[cfg(all(windows, feature = "native"))]
        {
            self.camera.is_some()
        }
        #[cfg(not(all(windows, feature = "native")))]
        {
            false
        }
    }
    pub fn start_calibration(&mut self) -> Result<(), String> {
        self.session = Some(CalibrationSession::new(self.config().clone())?);
        self.engine.clear_profile();
        Ok(())
    }
    pub fn extend_calibration(&mut self) -> Result<(), String> {
        self.session
            .as_mut()
            .ok_or("no calibration session")?
            .extend()?;
        self.engine.clear_profile();
        Ok(())
    }
    pub fn cancel_calibration(&mut self) {
        self.session = None;
        self.engine.reset_filter();
    }
    pub fn calibration_progress(&self) -> Option<CalibrationProgress> {
        self.session.as_ref().map(|session| {
            let (completed, total) = session.progress();
            CalibrationProgress {
                target: session.target(),
                phase: session.phase_label().into(),
                instruction: session.instruction().into(),
                completed,
                total,
                stable_samples: session.sample_progress(),
                suggested_targets: session.suggested_targets(),
            }
        })
    }
    pub fn observe(&mut self, observation: Observation, now_ms: u64) -> Result<SdkEvent, String> {
        let timely = observation.timestamp_ms <= now_ms
            && now_ms - observation.timestamp_ms <= self.config().maximum_frame_age_ms;
        if timely {
            self.last_frame_ms = Some(observation.timestamp_ms);
            self.loss_reported = false;
        }
        let mut outcome = if timely {
            self.session
                .as_mut()
                .and_then(|s| s.observe(observation, observation.timestamp_ms))
        } else {
            None
        };
        if let Some(CalibrationOutcome::Completed { profile, .. }) = &mut outcome {
            profile.capture_identity = self.capture_identity.clone();
            self.engine.set_profile(profile.clone())?;
        }
        let estimate = self.engine.process(observation, now_ms);
        Ok(SdkEvent {
            estimate: Some(estimate),
            observation: Some(observation),
            calibration: outcome,
            progress: self.calibration_progress(),
            status: None,
        })
    }
    fn lost_event(&mut self, status: TrackerStatus) -> SdkEvent {
        self.loss_reported = true;
        let now = self.timestamp_ms();
        let invalid = Observation {
            timestamp_ms: now,
            ..Observation::default()
        };
        let mut outcome = self.session.as_mut().and_then(|s| s.observe(invalid, now));
        if let Some(CalibrationOutcome::Completed { profile, .. }) = &mut outcome {
            profile.capture_identity = self.capture_identity.clone();
            let _ = self.engine.set_profile(profile.clone());
        }
        SdkEvent {
            estimate: Some(self.engine.lost(now)),
            observation: None,
            calibration: outcome,
            progress: self.calibration_progress(),
            status: Some(status),
        }
    }
    pub fn poll(&mut self) -> Result<Vec<SdkEvent>, String> {
        #[cfg(all(windows, feature = "native"))]
        {
            let events = self.camera.as_ref().map(|c| c.drain()).unwrap_or_default();
            let mut result = Vec::new();
            for event in events {
                match event {
                    TrackerEvent::Features { features, .. } => {
                        let now = self.timestamp_ms();
                        result.push(self.observe(features, now)?);
                    }
                    TrackerEvent::Status(status) => {
                        if let TrackerStatus::CameraReady {
                            width,
                            height,
                            device_id,
                            ..
                        } = &status
                        {
                            self.bind_capture(format!("camera:{device_id}:{width}x{height}"));
                        }
                        if matches!(status, TrackerStatus::CameraReady { .. }) {
                            self.last_frame_ms = Some(self.timestamp_ms());
                            self.loss_reported = false;
                        }
                        if matches!(
                            status,
                            TrackerStatus::NoFace
                                | TrackerStatus::GazeUnavailable { .. }
                                | TrackerStatus::CameraRetrying { .. }
                                | TrackerStatus::Failed(_)
                                | TrackerStatus::Stopped
                        ) {
                            result.push(self.lost_event(status));
                        } else {
                            result.push(SdkEvent {
                                estimate: None,
                                observation: None,
                                calibration: None,
                                progress: self.calibration_progress(),
                                status: Some(status),
                            });
                        }
                    }
                }
            }
            let now = self.timestamp_ms();
            if self.has_camera()
                && !self.loss_reported
                && self
                    .last_frame_ms
                    .is_some_and(|t| now.saturating_sub(t) > self.config().maximum_frame_age_ms)
            {
                result.push(self.lost_event(TrackerStatus::GazeUnavailable {
                    detail: "camera/inference stream stalled".into(),
                }));
            }
            Ok(result)
        }
        #[cfg(not(all(windows, feature = "native")))]
        {
            Ok(vec![])
        }
    }
    /// Capture timestamp must come from `timestamp_ms()` before capture, not
    /// after inference. This makes latency/staleness comparable across SDKs.
    pub fn process_frame(&mut self, frame: FrameView<'_>) -> Result<SdkEvent, String> {
        #[cfg(all(windows, feature = "native"))]
        {
            if self.camera.is_some() {
                return Err("owned camera is running; poll its events instead".into());
            }
            frame.validate()?;
            self.bind_capture(format!("rgb:{}x{}", frame.width, frame.height));
            let timestamp = frame.timestamp_ms;
            let now = self.timestamp_ms();
            if timestamp > now || now - timestamp > self.config().maximum_frame_age_ms {
                return self.observe(
                    Observation {
                        timestamp_ms: timestamp,
                        ..Observation::default()
                    },
                    now,
                );
            }
            if self.backend.is_none() {
                self.backend = Some(NativeBackend::load(&self.root)?);
            }
            match self.backend.as_mut().unwrap().infer(frame)? {
                Some(observation) => {
                    let now = self.timestamp_ms();
                    self.observe(observation, now)
                }
                None => Ok(self.lost_event(TrackerStatus::GazeUnavailable {
                    detail: "no usable binocular gaze in frame".into(),
                })),
            }
        }
        #[cfg(not(all(windows, feature = "native")))]
        {
            let _ = frame;
            Err("RGB inference requires the Windows native backend".into())
        }
    }
}
