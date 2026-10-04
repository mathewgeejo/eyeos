use crate::{
    AccuracyReport, FEATURE_COUNT, Observation, PROFILE_VERSION, Point, TrackerConfig,
    math::{least_squares, quantile},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CalibrationPoint {
    pub feature_x: f64,
    pub feature_y: f64,
    pub screen_x: f64,
    pub screen_y: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fixation {
    pub group_id: u64,
    pub observation: Observation,
    pub target: Point,
    pub frames: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Regression {
    pub mean: [f64; FEATURE_COUNT],
    pub scale: [f64; FEATURE_COUNT],
    pub coefficients: Vec<[f64; 2]>,
}
impl Regression {
    fn row(&self, observation: Observation) -> Vec<f64> {
        let mut row = vec![1.0];
        row.extend(
            observation
                .features()
                .iter()
                .enumerate()
                .map(|(i, x)| (x - self.mean[i]) / self.scale[i]),
        );
        row
    }
    pub fn predict(&self, observation: Observation) -> Point {
        let row = self.row(observation);
        let sum = |axis: usize| -> f64 {
            row.iter()
                .zip(&self.coefficients)
                .map(|(v, c)| v * c[axis])
                .sum()
        };
        Point::new(sum(0), sum(1))
    }
    fn fit(samples: &[Fixation]) -> Option<Self> {
        if samples.len() < 6 {
            return None;
        }
        let mut model = Self {
            mean: [0.0; FEATURE_COUNT],
            scale: [1.0; FEATURE_COUNT],
            coefficients: vec![],
        };
        for i in 0..FEATURE_COUNT {
            model.mean[i] = samples
                .iter()
                .map(|s| s.observation.features()[i])
                .sum::<f64>()
                / samples.len() as f64;
            let variance = samples
                .iter()
                .map(|s| (s.observation.features()[i] - model.mean[i]).powi(2))
                .sum::<f64>()
                / samples.len() as f64;
            model.scale[i] = variance.sqrt().max(feature_floor(i));
        }
        let rows: Vec<_> = samples.iter().map(|s| model.row(s.observation)).collect();
        let mut weights = vec![1.0_f64; samples.len()];
        for _ in 0..4 {
            let mut a = Vec::new();
            let mut b = Vec::new();
            for ((s, row), weight) in samples.iter().zip(&rows).zip(&weights) {
                let w = weight.sqrt();
                a.push(row.iter().map(|v| v * w).collect());
                b.push([s.target.x * w, s.target.y * w]);
            }
            // Penalise features, not the intercept. One row per fixation gives each
            // fixation equal weight regardless of camera FPS or sample count.
            for i in 1..=FEATURE_COUNT {
                let mut row = vec![0.0; FEATURE_COUNT + 1];
                row[i] = 0.01_f64.sqrt();
                a.push(row);
                b.push([0.0; 2]);
            }
            model.coefficients = least_squares(a, b, FEATURE_COUNT + 1)?;
            let errors: Vec<_> = samples
                .iter()
                .map(|s| model.predict(s.observation).distance_to(s.target))
                .collect();
            let center = quantile(&errors, 0.5);
            let mad = quantile(
                &errors
                    .iter()
                    .map(|v| (v - center).abs())
                    .collect::<Vec<_>>(),
                0.5,
            );
            let cutoff = (center + 2.5 * 1.4826 * mad).max(2.0);
            weights = errors
                .iter()
                .map(|e| (cutoff / e.max(1e-9)).min(1.0))
                .collect();
        }
        Some(model)
    }
}

fn feature_floor(i: usize) -> f64 {
    match i {
        5..=7 => 3.0,
        8..=10 => 0.025,
        15 => 1.0,
        _ => 0.025,
    }
}
fn distance(model: &Regression, a: Observation, b: Observation) -> f64 {
    let a = a.features();
    let b = b.features();
    [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        .iter()
        .map(|&i| ((a[i] - b[i]) / model.scale[i]).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationProfile {
    pub version: u32,
    pub config: TrackerConfig,
    #[serde(default)]
    pub capture_identity: Option<String>,
    pub regression: Regression,
    pub fixations: Vec<Fixation>,
    pub retrieval_enabled: bool,
    pub retrieval_radius: f64,
    pub baseline_cv_median_px: f64,
    pub baseline_cv_p95_px: f64,
    pub retrieval_cv_median_px: f64,
    pub retrieval_cv_p95_px: f64,
    pub sample_count: usize,
    pub median_error_px: f64,
    pub validation_median_error_px: f64,
    pub validation_passed: bool,
    pub accuracy: Option<AccuracyReport>,
}
impl CalibrationProfile {
    pub fn fit_fixations(config: TrackerConfig, fixations: Vec<Fixation>) -> Result<Self, String> {
        config.validate()?;
        if fixations.len() < 9
            || fixations.len() > 512
            || fixations.iter().any(|s| {
                !s.observation.usable(config.minimum_quality) || !s.target.finite() || s.frames == 0
            })
        {
            return Err("need at least nine usable labeled fixations".into());
        }
        let range = |i| {
            let min = fixations
                .iter()
                .map(|s| s.observation.features()[i])
                .fold(f64::INFINITY, f64::min);
            let max = fixations
                .iter()
                .map(|s| s.observation.features()[i])
                .fold(f64::NEG_INFINITY, f64::max);
            max - min
        };
        if range(0) < 0.02 || range(1) < 0.015 {
            return Err("gaze did not cover both screen axes; follow each target".into());
        }
        let regression = Regression::fit(&fixations).ok_or("could not fit a stable gaze map")?;
        let mut profile = Self {
            version: PROFILE_VERSION,
            config,
            capture_identity: None,
            regression,
            fixations,
            retrieval_enabled: false,
            retrieval_radius: 0.0,
            baseline_cv_median_px: 0.0,
            baseline_cv_p95_px: 0.0,
            retrieval_cv_median_px: 0.0,
            retrieval_cv_p95_px: 0.0,
            sample_count: 0,
            median_error_px: 0.0,
            validation_median_error_px: 0.0,
            validation_passed: false,
            accuracy: None,
        };
        profile.sample_count = profile.fixations.iter().map(|s| s.frames).sum();
        let fit_errors: Vec<_> = profile
            .fixations
            .iter()
            .map(|s| {
                profile
                    .regression
                    .predict(s.observation)
                    .distance_to(s.target)
            })
            .collect();
        profile.median_error_px = quantile(&fit_errors, 0.5);
        let mut baseline = Vec::new();
        let mut retrieved = Vec::new();
        let mut nearest = Vec::new();
        // Hold out ALL fixations at the same target, including head-position repeats.
        // Neither scaling, fitting nor the retrieval index can see the held-out target.
        for held in &profile.fixations {
            let training: Vec<_> = profile
                .fixations
                .iter()
                .filter(|s| s.target.distance_to(held.target) > 0.01)
                .cloned()
                .collect();
            let Some(model) = Regression::fit(&training) else {
                continue;
            };
            let base = model.predict(held.observation);
            let (corrected, near) = retrieve(&model, &training, held.observation, f64::INFINITY);
            baseline.push(base.distance_to(held.target));
            retrieved.push(corrected.distance_to(held.target));
            nearest.push(near);
        }
        if baseline.len() != profile.fixations.len() {
            return Err("insufficient independent targets for cross-validation".into());
        }
        profile.baseline_cv_median_px = quantile(&baseline, 0.5);
        profile.baseline_cv_p95_px = quantile(&baseline, 0.95);
        profile.retrieval_radius = quantile(&nearest, 0.95).max(0.5);
        // Evaluate the same radius that will actually be used at runtime.
        retrieved.clear();
        for held in &profile.fixations {
            let training: Vec<_> = profile
                .fixations
                .iter()
                .filter(|s| s.target.distance_to(held.target) > 0.01)
                .cloned()
                .collect();
            let model = Regression::fit(&training).ok_or("cross-validation failed")?;
            let (p, _) = retrieve(
                &model,
                &training,
                held.observation,
                profile.retrieval_radius,
            );
            retrieved.push(p.distance_to(held.target));
        }
        profile.retrieval_cv_median_px = quantile(&retrieved, 0.5);
        profile.retrieval_cv_p95_px = quantile(&retrieved, 0.95);
        profile.retrieval_enabled = profile.retrieval_cv_median_px + 0.01
            < profile.baseline_cv_median_px
            && profile.retrieval_cv_p95_px <= profile.baseline_cv_p95_px;
        Ok(profile)
    }
    pub fn supports(&self, observation: Observation) -> bool {
        let query = observation.features();
        [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            .iter()
            .all(|&i| {
                let min = self
                    .fixations
                    .iter()
                    .map(|s| s.observation.features()[i])
                    .fold(f64::INFINITY, f64::min);
                let max = self
                    .fixations
                    .iter()
                    .map(|s| s.observation.features()[i])
                    .fold(f64::NEG_INFINITY, f64::max);
                let margin = ((max - min) * 0.20).max(feature_floor(i) * 2.0);
                query[i] >= min - margin && query[i] <= max + margin
            })
    }
    /// Unsmoothed, unclamped predictions for independent evaluation.
    pub fn predict(&self, observation: Observation) -> Option<(Point, bool)> {
        if !observation.usable(self.config.minimum_quality) || !self.supports(observation) {
            return None;
        }
        let (p, used) = if self.retrieval_enabled {
            let (p, nearest) = retrieve(
                &self.regression,
                &self.fixations,
                observation,
                self.retrieval_radius,
            );
            (p, nearest <= self.retrieval_radius)
        } else {
            (self.regression.predict(observation), false)
        };
        p.finite().then_some((p, used))
    }
    pub fn validate(&self, config: &TrackerConfig) -> Result<(), String> {
        if self.version != PROFILE_VERSION || &self.config != config {
            return Err("profile is incompatible; recalibration required".into());
        }
        config.validate()?;
        if self.fixations.len() < 9
            || self.fixations.len() > 512
            || self.regression.coefficients.len() != FEATURE_COUNT + 1
            || self.regression.mean.iter().any(|v| !v.is_finite())
            || self
                .regression
                .scale
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            || self
                .regression
                .coefficients
                .iter()
                .flatten()
                .any(|v| !v.is_finite())
            || [
                self.retrieval_radius,
                self.median_error_px,
                self.validation_median_error_px,
                self.baseline_cv_median_px,
                self.baseline_cv_p95_px,
                self.retrieval_cv_median_px,
                self.retrieval_cv_p95_px,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
            || self.retrieval_radius <= 0.0
            || self.fixations.iter().any(|s| {
                !s.observation.usable(config.minimum_quality) || !s.target.finite() || s.frames == 0
            })
            || self.sample_count != self.fixations.iter().map(|s| s.frames).sum::<usize>()
        {
            return Err("invalid calibration profile".into());
        }
        if self.retrieval_enabled
            && !(self.retrieval_cv_median_px + 0.01 < self.baseline_cv_median_px
                && self.retrieval_cv_p95_px <= self.baseline_cv_p95_px)
        {
            return Err("retrieval has not passed grouped cross-validation".into());
        }
        if self.validation_passed
            && !self.accuracy.as_ref().is_some_and(|r| {
                r.precision_passed
                    && r.samples >= 13
                    && r.valid_sample_coverage >= 0.8
                    && r.median_error_deg
                        .is_some_and(|v| v.is_finite() && v <= config.median_target_deg)
                    && r.p95_error_deg
                        .is_some_and(|v| v.is_finite() && v <= config.p95_target_deg)
            })
        {
            return Err("profile lacks independent precision validation".into());
        }
        Ok(())
    }
}

fn retrieve(
    model: &Regression,
    samples: &[Fixation],
    query: Observation,
    radius: f64,
) -> (Point, f64) {
    let base = model.predict(query);
    let mut neighbors: Vec<_> = samples
        .iter()
        .map(|s| (distance(model, query, s.observation), s))
        .collect();
    neighbors.sort_by(|a, b| a.0.total_cmp(&b.0));
    let nearest = neighbors.first().map(|s| s.0).unwrap_or(f64::INFINITY);
    if nearest > radius {
        return (base, nearest);
    }
    let mut groups = Vec::new();
    let mut correction = Point::default();
    let mut weight_sum = 0.0;
    for (dist, sample) in neighbors {
        if groups.contains(&sample.group_id) || dist > radius {
            continue;
        }
        groups.push(sample.group_id);
        let weight = 1.0 / (dist * dist + 0.05);
        let p = model.predict(sample.observation);
        correction.x += (sample.target.x - p.x) * weight;
        correction.y += (sample.target.y - p.y) * weight;
        weight_sum += weight;
        if groups.len() == 8 {
            break;
        }
    }
    if weight_sum > 0.0 {
        (
            Point::new(
                base.x + correction.x / weight_sum,
                base.y + correction.y / weight_sum,
            ),
            nearest,
        )
    } else {
        (base, f64::INFINITY)
    }
}
