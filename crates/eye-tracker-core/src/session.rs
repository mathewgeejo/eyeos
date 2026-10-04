use crate::{math::quantile, *};
use serde::{Deserialize, Serialize};

pub const CALIBRATION_SAMPLES_PER_TARGET: usize = 12;
const SETTLE_MS: u64 = 650;
const WINDOW_MS: u64 = 350;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledObservation {
    pub observation: Observation,
    pub target: Point,
    pub group_id: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationOutcome {
    Completed {
        profile: CalibrationProfile,
        report: AccuracyReport,
    },
    Rejected(String),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Mapping,
    Validation,
    Complete,
}

/// Labels come exclusively from displayed targets. Validation samples are never
/// inserted into the retrieval index, even when requesting extra calibration.
pub struct CalibrationSession {
    config: TrackerConfig,
    targets: Vec<Point>,
    validation_targets: Vec<Point>,
    current: usize,
    phase: Phase,
    target_started: Option<u64>,
    last_timestamp: Option<u64>,
    feature_samples: Vec<Observation>,
    fixations: Vec<Fixation>,
    candidate: Option<CalibrationProfile>,
    validation: Vec<LabeledObservation>,
    attempted: usize,
    failed_targets: Vec<Point>,
    report: Option<AccuracyReport>,
    extended: bool,
}
impl CalibrationSession {
    pub fn new(config: TrackerConfig) -> Result<Self, String> {
        config.validate()?;
        let size = config.screen_size;
        let mut targets = Vec::new();
        for y in [0.08, 0.5, 0.92] {
            for x in [0.08, 0.5, 0.92] {
                targets.push(Point::new(size.x * x, size.y * y));
            }
        }
        targets.extend([Point::new(size.x * 0.5, size.y * 0.5); 3]);
        let validation_targets = [
            (0.03, 0.03),
            (0.97, 0.03),
            (0.03, 0.97),
            (0.97, 0.97),
            (0.25, 0.25),
            (0.75, 0.25),
            (0.25, 0.75),
            (0.75, 0.75),
            (0.5, 0.18),
            (0.5, 0.82),
            (0.18, 0.5),
            (0.82, 0.5),
            (0.5, 0.5),
        ]
        .map(|(x, y)| Point::new(size.x * x, size.y * y))
        .to_vec();
        Ok(Self {
            config,
            targets,
            validation_targets,
            current: 0,
            phase: Phase::Mapping,
            target_started: None,
            last_timestamp: None,
            feature_samples: vec![],
            fixations: vec![],
            candidate: None,
            validation: vec![],
            attempted: 0,
            failed_targets: vec![],
            report: None,
            extended: false,
        })
    }
    pub fn target(&self) -> Option<Point> {
        match self.phase {
            Phase::Mapping => self.targets.get(self.current).copied(),
            Phase::Validation => self.validation_targets.get(self.current).copied(),
            Phase::Complete => None,
        }
    }
    pub fn progress(&self) -> (usize, usize) {
        (
            self.current,
            if self.phase == Phase::Mapping {
                self.targets.len()
            } else {
                self.validation_targets.len()
            },
        )
    }
    pub fn sample_progress(&self) -> usize {
        self.feature_samples.len()
    }
    pub fn phase_label(&self) -> &'static str {
        match self.phase {
            Phase::Mapping => "Calibration",
            Phase::Validation => "Independent validation",
            Phase::Complete => "Complete",
        }
    }
    pub fn instruction(&self) -> &'static str {
        if self.phase == Phase::Mapping && !self.extended {
            match self.current {
                9 => "Keep looking at the dot; turn your head slightly left",
                10 => "Keep looking at the dot; turn your head slightly right",
                11 => "Keep looking at the dot; move slightly closer",
                _ => "Look at the dot and hold still",
            }
        } else {
            "Look at the dot and hold still"
        }
    }
    pub fn report(&self) -> Option<&AccuracyReport> {
        self.report.as_ref()
    }
    pub fn suggested_targets(&self) -> Vec<Point> {
        let mut targets = self.failed_targets.clone();
        if let Some(profile) = &self.candidate {
            let mut errors: Vec<_> = self
                .validation
                .iter()
                .filter_map(|s| {
                    profile
                        .predict(s.observation)
                        .map(|(p, _)| (p.distance_to(s.target), s.target))
                })
                .collect();
            errors.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (_, target) in errors {
                if !targets.iter().any(|p| p.distance_to(target) < 1.0) {
                    targets.push(target);
                }
                if targets.len() >= 5 {
                    break;
                }
            }
        }
        targets.truncate(5);
        targets
    }
    pub fn extend(&mut self) -> Result<(), String> {
        if self.phase != Phase::Complete {
            return Err("complete validation before requesting extra calibration".into());
        }
        if self.fixations.len() >= 128 {
            return Err("calibration is too large; start a fresh session".into());
        }
        let targets = self.suggested_targets();
        if targets.is_empty() {
            return Err("no extra targets available".into());
        }
        self.targets = targets;
        self.current = 0;
        self.phase = Phase::Mapping;
        self.extended = true;
        self.target_started = None;
        self.feature_samples.clear();
        self.validation.clear();
        self.attempted = 0;
        self.failed_targets.clear();
        self.report = None;
        self.candidate = None;
        Ok(())
    }
    pub fn observe(
        &mut self,
        mut observation: Observation,
        timestamp_ms: u64,
    ) -> Option<CalibrationOutcome> {
        if self.phase == Phase::Complete || self.last_timestamp.is_some_and(|t| timestamp_ms <= t) {
            return None;
        }
        self.last_timestamp = Some(timestamp_ms);
        observation.timestamp_ms = timestamp_ms;
        let target = self.target()?;
        let started = *self.target_started.get_or_insert(timestamp_ms);
        let elapsed = timestamp_ms.saturating_sub(started);
        if elapsed < SETTLE_MS {
            return None;
        }
        if self.phase == Phase::Validation {
            self.attempted += 1;
        }
        if !observation.usable(self.config.minimum_quality) {
            self.feature_samples.clear();
            if elapsed >= 4000 {
                return self.timeout(target, timestamp_ms);
            }
            return None;
        }
        if self.phase == Phase::Validation
            && self.candidate.as_ref()?.predict(observation).is_none()
        {
            if elapsed >= 4000 {
                return self.timeout(target, timestamp_ms);
            }
            return None;
        }
        self.feature_samples.push(observation);
        let first = self.feature_samples.first()?.timestamp_ms;
        if self.feature_samples.len() < CALIBRATION_SAMPLES_PER_TARGET
            || timestamp_ms - first < WINDOW_MS
        {
            return None;
        }
        let x = quantile(
            &self.feature_samples.iter().map(|s| s.x).collect::<Vec<_>>(),
            0.5,
        );
        let y = quantile(
            &self.feature_samples.iter().map(|s| s.y).collect::<Vec<_>>(),
            0.5,
        );
        let spreads: Vec<_> = self
            .feature_samples
            .iter()
            .map(|s| (s.x - x).hypot(s.y - y))
            .collect();
        if quantile(&spreads, 0.8) > 0.025 {
            self.feature_samples.clear();
            if elapsed >= 4000 {
                return self.timeout(target, timestamp_ms);
            }
            return None;
        }
        if self.phase == Phase::Mapping {
            let samples: Vec<_> = self
                .feature_samples
                .iter()
                .zip(&spreads)
                .filter(|(_, d)| **d <= 0.025)
                .map(|(s, _)| *s)
                .collect();
            if samples.len() < 8 {
                self.feature_samples.clear();
                return None;
            }
            let mut aggregate = samples[0];
            let med = |f: fn(&Observation) -> f64| {
                quantile(&samples.iter().map(f).collect::<Vec<_>>(), 0.5)
            };
            aggregate.x = med(|s| s.x);
            aggregate.y = med(|s| s.y);
            for i in 0..3 {
                aggregate.head_pose[i] = quantile(
                    &samples.iter().map(|s| s.head_pose[i]).collect::<Vec<_>>(),
                    0.5,
                );
                aggregate.gaze_direction[i] = quantile(
                    &samples
                        .iter()
                        .map(|s| s.gaze_direction[i])
                        .collect::<Vec<_>>(),
                    0.5,
                );
            }
            aggregate.face_center = Point::new(med(|s| s.face_center.x), med(|s| s.face_center.y));
            aggregate.face_scale = med(|s| s.face_scale);
            let iris = |left| {
                let points: Vec<Point> = samples
                    .iter()
                    .filter_map(|s| if left { s.left_iris } else { s.right_iris })
                    .collect();
                (points.len() == samples.len()).then(|| {
                    Point::new(
                        quantile(&points.iter().map(|p| p.x).collect::<Vec<_>>(), 0.5),
                        quantile(&points.iter().map(|p| p.y).collect::<Vec<_>>(), 0.5),
                    )
                })
            };
            aggregate.left_iris = iris(true);
            aggregate.right_iris = iris(false);
            self.fixations.push(Fixation {
                group_id: self.fixations.len() as u64,
                observation: aggregate,
                target,
                frames: samples.len(),
            });
        } else {
            // Score every accepted frame, never an averaged fixation or cursor.
            self.validation
                .extend(self.feature_samples.iter().map(|s| LabeledObservation {
                    observation: *s,
                    target,
                    group_id: self.current as u64,
                }));
        }
        self.advance(timestamp_ms)
    }
    fn timeout(&mut self, target: Point, timestamp_ms: u64) -> Option<CalibrationOutcome> {
        if self.phase == Phase::Mapping {
            self.phase = Phase::Complete;
            Some(CalibrationOutcome::Rejected(
                "Could not collect a stable fixation. Improve lighting/camera position and retry."
                    .into(),
            ))
        } else {
            self.failed_targets.push(target);
            self.advance(timestamp_ms)
        }
    }
    fn advance(&mut self, timestamp_ms: u64) -> Option<CalibrationOutcome> {
        self.current += 1;
        self.feature_samples.clear();
        self.target_started = Some(timestamp_ms);
        if self.phase == Phase::Mapping && self.current == self.targets.len() {
            match CalibrationProfile::fit_fixations(self.config.clone(), self.fixations.clone()) {
                Ok(profile) => {
                    self.candidate = Some(profile);
                    self.phase = Phase::Validation;
                    self.current = 0;
                    None
                }
                Err(error) => {
                    self.phase = Phase::Complete;
                    Some(CalibrationOutcome::Rejected(error))
                }
            }
        } else if self.phase == Phase::Validation && self.current == self.validation_targets.len() {
            let mut profile = self.candidate.take()?;
            let mut report = evaluate(&profile, &self.validation, self.attempted);
            report.precision_passed &= self.failed_targets.is_empty();
            profile.validation_passed = report.precision_passed;
            profile.validation_median_error_px = report.median_error_px;
            profile.accuracy = Some(report.clone());
            self.report = Some(report.clone());
            self.candidate = Some(profile.clone());
            self.phase = Phase::Complete;
            Some(CalibrationOutcome::Completed { profile, report })
        } else {
            None
        }
    }
}

pub fn evaluate(
    profile: &CalibrationProfile,
    samples: &[LabeledObservation],
    attempted: usize,
) -> AccuracyReport {
    let mut errors = Vec::new();
    let mut angles = Vec::new();
    let mut latencies = Vec::new();
    let mut predictions = Vec::new();
    let mut regions: Vec<(String, Vec<f64>)> = Vec::new();
    let mut groups = Vec::new();
    for sample in samples {
        let Some((point, _)) = profile.predict(sample.observation) else {
            continue;
        };
        let error = point.distance_to(sample.target);
        errors.push(error);
        if let Some(deg) = profile.config.angular_error(point, sample.target) {
            angles.push(deg);
        }
        latencies.push(sample.observation.inference_latency_ms);
        predictions.push((sample.group_id, point));
        if !groups.contains(&sample.group_id) {
            groups.push(sample.group_id);
        }
        let x = (sample.target.x / profile.config.screen_size.x * 3.0)
            .floor()
            .clamp(0.0, 2.0) as u32;
        let y = (sample.target.y / profile.config.screen_size.y * 3.0)
            .floor()
            .clamp(0.0, 2.0) as u32;
        let name = format!("row{}-col{}", y + 1, x + 1);
        if let Some((_, v)) = regions.iter_mut().find(|(n, _)| n == &name) {
            v.push(error);
        } else {
            regions.push((name, vec![error]));
        }
    }
    let mut jitter = Vec::new();
    for group in &groups {
        let points: Vec<_> = predictions
            .iter()
            .filter(|(g, _)| g == group)
            .map(|(_, p)| *p)
            .collect();
        let center = Point::new(
            quantile(&points.iter().map(|p| p.x).collect::<Vec<_>>(), 0.5),
            quantile(&points.iter().map(|p| p.y).collect::<Vec<_>>(), 0.5),
        );
        jitter.extend(points.iter().map(|p| p.distance_to(center)));
    }
    let mut report = AccuracyReport {
        median_error_px: quantile(&errors, 0.5),
        p95_error_px: quantile(&errors, 0.95),
        median_error_deg: (!angles.is_empty()).then(|| quantile(&angles, 0.5)),
        p95_error_deg: (!angles.is_empty()).then(|| quantile(&angles, 0.95)),
        valid_sample_coverage: errors.len() as f64 / attempted.max(samples.len()).max(1) as f64,
        jitter_px: quantile(&jitter, 0.5),
        median_latency_ms: quantile(&latencies, 0.5),
        p95_latency_ms: quantile(&latencies, 0.95),
        regions: regions
            .into_iter()
            .map(|(region, v)| RegionError {
                region,
                median_px: quantile(&v, 0.5),
                p95_px: quantile(&v, 0.95),
                samples: v.len(),
            })
            .collect(),
        samples: errors.len(),
        precision_passed: false,
    };
    report.precision_passed = groups.len() >= 13
        && report.valid_sample_coverage >= 0.8
        && report
            .median_error_deg
            .is_some_and(|v| v <= profile.config.median_target_deg)
        && report
            .p95_error_deg
            .is_some_and(|v| v <= profile.config.p95_target_deg);
    report
}
