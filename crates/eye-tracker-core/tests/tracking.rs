use eye_tracker_core::*;

fn config() -> TrackerConfig {
    TrackerConfig {
        screen_width_mm: Some(530.0),
        screen_height_mm: Some(300.0),
        viewing_distance_mm: Some(600.0),
        ..TrackerConfig::default()
    }
}
fn observation(x: f64, y: f64, timestamp_ms: u64) -> Observation {
    let norm = (x * x + y * y + 1.0).sqrt();
    Observation {
        x,
        y,
        gaze_direction: [x / norm, y / norm, -1.0 / norm],
        confidence: 1.0,
        timestamp_ms,
        inference_latency_ms: 12.0,
        ..Observation::default()
    }
}
fn fixations(config: &TrackerConfig) -> Vec<Fixation> {
    let mut samples = Vec::new();
    for y in [0.08, 0.5, 0.92] {
        for x in [0.08, 0.5, 0.92] {
            samples.push(Fixation {
                group_id: samples.len() as u64,
                observation: observation(x, y, 0),
                target: Point::new(x * config.screen_size.x, y * config.screen_size.y),
                frames: 12,
            });
        }
    }
    samples
}
fn profile() -> CalibrationProfile {
    CalibrationProfile::fit_fixations(config(), fixations(&config())).unwrap()
}

#[test]
fn vector_scale_is_not_confidence_and_roll_is_restored_once() {
    let a = normalize_gaze_vector([0.1, 0.2, -1.0], 0.0).unwrap();
    let b = normalize_gaze_vector([0.9, 1.8, -9.0], 0.0).unwrap();
    assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-7));
    let rotated = normalize_gaze_vector([1.0, 0.0, -1.0], 90.0).unwrap();
    assert!(rotated[0].abs() < 1e-8 && rotated[1] < 0.0 && rotated[2] < 0.0);
    assert!(normalize_gaze_vector([0.0; 3], 0.0).is_none());
    assert!(normalize_gaze_vector([0.0, 0.0, 1.0], 0.0).is_none());
    assert!(normalize_gaze_vector([1.0, 0.0, -0.001], 0.0).is_none());
    assert!(normalize_gaze_vector([f32::NAN, 0.0, 1.0], 0.0).is_none());
}

#[test]
fn native_forward_gaze_survives_normalization_and_engine_quality_checks() {
    // Intel's reference computes horizontal degrees as 90 + atan2(z, x).
    // Therefore (0, 0, -1), not positive Z, is a straight-ahead gaze.
    let direction = normalize_gaze_vector([0.0, 0.0, -0.9], 0.0).unwrap();
    assert_eq!(direction, [0.0, 0.0, -1.0]);
    let mut sample = observation(0.0, 0.0, 100);
    sample.gaze_direction = direction;
    assert!(sample.usable(0.72));
    let mut tracker = EyeTracker::new(config()).unwrap();
    assert_eq!(
        tracker.process(sample, 100).state,
        TrackingState::Uncalibrated
    );
    sample.timestamp_ms = 133;
    sample.gaze_direction[2] = 1.0;
    assert_eq!(
        tracker.process(sample, 133).state,
        TrackingState::TrackingLost
    );
}

#[test]
fn image_quality_uses_one_floor_without_claiming_validated_precision() {
    let c = config();
    let mut tracker = EyeTracker::new(c.clone()).unwrap();
    let mut sample = observation(0.5, 0.5, 100);
    sample.quality.image_score = 0.5;
    sample.confidence = 0.5;
    assert!(sample.usable(c.minimum_quality));
    let estimate = tracker.process(sample, 100);
    assert_eq!(estimate.state, TrackingState::Uncalibrated);
    assert!(!estimate.precision_validated);
    sample.timestamp_ms = 133;
    sample.quality.image_score = MINIMUM_IMAGE_QUALITY - 0.01;
    assert_eq!(
        tracker.process(sample, 133).state,
        TrackingState::TrackingLost
    );
}

#[test]
fn regression_predicts_unseen_points_and_retrieval_requires_cv_improvement() {
    let p = profile();
    let target = Point::new(1920.0 * 0.25, 1080.0 * 0.75);
    let (predicted, _) = p.predict(observation(0.25, 0.75, 1)).unwrap();
    assert!(predicted.distance_to(target) < 15.0, "{predicted:?}");
    assert!(p.baseline_cv_p95_px.is_finite());
    if p.retrieval_enabled {
        assert!(p.retrieval_cv_median_px + 0.01 < p.baseline_cv_median_px);
        assert!(p.retrieval_cv_p95_px <= p.baseline_cv_p95_px);
    }
}

#[test]
fn independent_validation_is_not_calibration_data_and_quick_setup_is_under_45s() {
    let mut session = CalibrationSession::new(config()).unwrap();
    let mut now = 0;
    let mut outcome = None;
    while let Some(target) = session.target() {
        let progress = session.progress();
        let phase = session.phase_label();
        let x = target.x / 1920.0;
        let y = target.y / 1080.0;
        for _ in 0..100 {
            now += 33;
            let next = session.observe(observation(x, y, now), now);
            if next.is_some() {
                outcome = next;
            }
            if session.progress() != progress || session.phase_label() != phase {
                break;
            }
        }
        assert!(now < 45_000, "quick calibration exceeded 45 seconds");
    }
    let CalibrationOutcome::Completed { profile, report } = outcome.unwrap() else {
        panic!("expected measured validation");
    };
    assert_eq!(profile.fixations.len(), 12);
    assert_eq!(report.regions.len(), 9);
    assert!(report.samples >= 13 * 12);
    assert!(report.precision_passed, "{report:?}");
    assert!(profile.validation_passed);
    profile.validate(&config()).unwrap();
    session.extend().unwrap();
    assert!(session.target().is_some());
}

#[test]
fn missing_geometry_never_claims_angular_accuracy() {
    let c = TrackerConfig::default();
    assert!(
        c.angular_error(Point::default(), Point::new(10.0, 10.0))
            .is_none()
    );
    let p = CalibrationProfile::fit_fixations(c.clone(), fixations(&c)).unwrap();
    let samples: Vec<_> = (0..13)
        .map(|i| LabeledObservation {
            observation: observation(0.5, 0.5, i),
            target: Point::new(960.0, 540.0),
            group_id: i,
        })
        .collect();
    let report = evaluate(&p, &samples, samples.len());
    assert!(report.median_error_deg.is_none());
    assert!(!report.precision_passed);
}

#[test]
fn stale_nan_blink_pose_and_tracking_loss_reset_the_filter() {
    let mut tracker = EyeTracker::new(config()).unwrap();
    tracker.set_profile(profile()).unwrap();
    let a = tracker.process(observation(0.4, 0.4, 100), 100);
    assert_eq!(a.raw, a.filtered);
    assert_eq!(
        tracker.process(observation(0.5, 0.5, 100), 100).state,
        TrackingState::StaleFrame
    );
    let b = tracker.process(observation(0.6, 0.6, 133), 133);
    assert_eq!(b.raw, b.filtered);
    assert_eq!(
        tracker.process(observation(0.5, 0.5, 134), 500).state,
        TrackingState::StaleFrame
    );
    let mut invalid = observation(0.5, 0.5, 600);
    invalid.confidence = f32::NAN;
    assert_eq!(
        tracker.process(invalid, 600).state,
        TrackingState::TrackingLost
    );
    invalid = observation(0.5, 0.5, 633);
    invalid.blink = true;
    assert_eq!(
        tracker.process(invalid, 633).state,
        TrackingState::TrackingLost
    );
    let mut pose = observation(0.5, 0.5, 666);
    pose.head_pose[0] = 30.0;
    assert_eq!(
        tracker.process(pose, 666).state,
        TrackingState::CalibrationRequired
    );
    tracker.lost(700);
    let recovered = tracker.process(observation(0.5, 0.5, 733), 733);
    assert_eq!(recovered.raw, recovered.filtered);
}

#[test]
fn profiles_reject_old_versions_wrong_devices_and_malformed_models() {
    let mut tracker = EyeTracker::new(config()).unwrap();
    assert!(
        tracker
            .import_profile(r#"{"x_coefficients":[1,2,3]}"#)
            .is_err()
    );
    let mut p = profile();
    p.version = 1;
    assert!(tracker.set_profile(p).is_err());
    let mut p = profile();
    p.config.camera_id = "another camera".into();
    assert!(tracker.set_profile(p).is_err());
    let mut p = profile();
    p.config.model_id = "mediapipe-64184e229b26/adas-0002/preprocess-v2".into();
    assert!(tracker.set_profile(p).is_err());
    let mut p = profile();
    p.regression.coefficients.clear();
    assert!(tracker.set_profile(p).is_err());
    let mut p = profile();
    p.validation_passed = true;
    assert!(tracker.set_profile(p).is_err());
    let p = profile();
    tracker.set_profile(p.clone()).unwrap();
    let json = tracker.export_profile().unwrap();
    tracker.clear_profile();
    tracker.import_profile(&json).unwrap();
    assert_eq!(tracker.profile(), Some(&p));
}

#[test]
fn frame_validation_checks_padding_and_overflow() {
    let rgb = [0; 16];
    assert!(
        FrameView {
            rgb: &rgb,
            width: 2,
            height: 2,
            stride_bytes: 8,
            timestamp_ms: 1
        }
        .validate()
        .is_ok()
    );
    assert!(
        FrameView {
            rgb: &rgb,
            width: 2,
            height: 2,
            stride_bytes: 5,
            timestamp_ms: 1
        }
        .validate()
        .is_err()
    );
    assert!(
        FrameView {
            rgb: &rgb,
            width: 2,
            height: 2,
            stride_bytes: usize::MAX,
            timestamp_ms: 1
        }
        .validate()
        .is_err()
    );
}

#[test]
fn stationary_gaze_and_invalid_labels_cannot_make_a_profile() {
    let mut s = fixations(&config());
    for f in &mut s {
        f.observation = observation(0.5, 0.5, 0);
    }
    assert!(CalibrationProfile::fit_fixations(config(), s).is_err());
    let mut s = fixations(&config());
    s[0].target.x = f64::NAN;
    assert!(CalibrationProfile::fit_fixations(config(), s).is_err());
}

#[test]
fn validation_counts_bad_frames_in_coverage_and_does_not_smooth_errors() {
    let p = profile();
    let samples: Vec<_> = (0..13)
        .map(|i| LabeledObservation {
            observation: observation(0.5, 0.5, i),
            target: Point::new(700.0, 700.0),
            group_id: i,
        })
        .collect();
    let report = evaluate(&p, &samples, 26);
    assert_eq!(report.valid_sample_coverage, 0.5);
    assert!(!report.precision_passed);
    assert!(report.median_error_px > 250.0);
}
