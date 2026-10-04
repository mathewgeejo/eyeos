use crate::*;

pub struct EyeTracker {
    pub config: TrackerConfig,
    profile: Option<CalibrationProfile>,
    last_timestamp: Option<u64>,
    filtered: Option<Point>,
    previous_raw: Option<Point>,
}
impl EyeTracker {
    pub fn new(config: TrackerConfig) -> Result<Self,String> {
        config.validate()?;
        Ok(Self { config, profile: None, last_timestamp: None, filtered: None, previous_raw: None })
    }
    pub fn profile(&self) -> Option<&CalibrationProfile> { self.profile.as_ref() }
    pub fn set_profile(&mut self, profile: CalibrationProfile) -> Result<(),String> {
        profile.validate(&self.config)?; self.profile=Some(profile); self.reset_filter(); Ok(())
    }
    pub fn clear_profile(&mut self) { self.profile=None; self.reset_filter(); }
    pub fn reset_filter(&mut self) { self.filtered=None; self.previous_raw=None; }
    pub fn export_profile(&self) -> Result<String,String> {
        serde_json::to_string(self.profile.as_ref().ok_or("no calibration profile")?).map_err(|e| e.to_string())
    }
    pub fn import_profile(&mut self, json: &str) -> Result<(),String> {
        if json.len()>4*1024*1024 { return Err("profile exceeds 4 MiB limit".into()); }
        self.set_profile(serde_json::from_str(json).map_err(|e| format!("profile needs recalibration: {e}"))?)
    }
    pub fn lost(&mut self, timestamp_ms: u64) -> GazeEstimate {
        self.reset_filter();
        GazeEstimate { timestamp_ms, raw: None, filtered: None, quality: Quality::default(),
            quality_score: 0.0, estimated_error_px: None, state: TrackingState::TrackingLost,
            precision_validated: false, retrieval_used: false, latency_ms: 0.0 }
    }
    pub fn process(&mut self, observation: Observation, now_ms: u64) -> GazeEstimate {
        let mut result=GazeEstimate { timestamp_ms: observation.timestamp_ms, raw: None, filtered: None,
            quality: observation.quality, quality_score: observation.confidence,
            estimated_error_px: None, state: TrackingState::Uncalibrated,
            precision_validated: false, retrieval_used: false,
            latency_ms: now_ms.saturating_sub(observation.timestamp_ms) as f64 };
        if observation.timestamp_ms>now_ms || now_ms-observation.timestamp_ms>self.config.maximum_frame_age_ms
            || self.last_timestamp.is_some_and(|t| observation.timestamp_ms<=t) {
            self.reset_filter(); result.state=TrackingState::StaleFrame; return result;
        }
        let dt=self.last_timestamp.map(|t| (observation.timestamp_ms-t) as f64/1000.0).unwrap_or(1.0/30.0);
        self.last_timestamp=Some(observation.timestamp_ms);
        if !observation.usable(self.config.minimum_quality) {
            self.reset_filter(); result.state=TrackingState::TrackingLost; return result;
        }
        let Some(profile)=&self.profile else { self.reset_filter(); return result; };
        let Some((raw,retrieval_used))=profile.predict(observation) else {
            self.reset_filter(); result.state=TrackingState::CalibrationRequired; return result;
        };
        result.estimated_error_px=profile.accuracy.as_ref().map(|a| a.p95_error_px);
        result.precision_validated=profile.validation_passed;
        if dt>self.config.maximum_frame_age_ms as f64/1000.0 { self.reset_filter(); }
        let speed=self.previous_raw.map(|p| p.distance_to(raw)/dt.max(0.001)).unwrap_or(0.0);
        let cutoff=1.5+0.012*speed;
        let alpha=1.0-(-std::f64::consts::TAU*cutoff*dt.max(0.001)).exp();
        let filtered=self.filtered.map(|p| p.lerp(raw,alpha)).unwrap_or(raw);
        self.filtered=Some(filtered); self.previous_raw=Some(raw);
        result.raw=Some(raw);
        result.filtered=Some(Point::new(filtered.x.clamp(0.0,self.config.screen_size.x-1.0),
            filtered.y.clamp(0.0,self.config.screen_size.y-1.0)));
        result.retrieval_used=retrieval_used;
        result.state=if result.precision_validated { TrackingState::Tracking } else { TrackingState::CalibrationRequired };
        result
    }
    pub fn process_frame(&mut self, backend: &mut dyn InferenceBackend, frame: FrameView<'_>, now_ms: u64) -> Result<GazeEstimate,String> {
        frame.validate()?;
        if backend.model_id()!=self.config.model_id { return Err("backend/model identity differs from tracker configuration".into()); }
        let timestamp=frame.timestamp_ms;
        match backend.infer(frame) {
            Ok(Some(observation)) => Ok(self.process(observation,now_ms)),
            Ok(None) => Ok(self.lost(timestamp)),
            Err(error) => { self.reset_filter(); Err(error) }
        }
    }
}
