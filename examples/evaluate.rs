//! `cargo run --example evaluate -- profile.json held-out.json`.
//! held-out.json contains LabeledObservation[] from a SEPARATE evaluation session.
use eye_tracker_core::{CalibrationProfile, LabeledObservation, evaluate};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: evaluate profile.json held-out.json".into());
    }
    let mut profile: CalibrationProfile = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    profile
        .validate(&profile.config)
        .map_err(std::io::Error::other)?;
    let observations: Vec<LabeledObservation> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    profile.retrieval_enabled = false;
    let baseline = evaluate(&profile, &observations, observations.len());
    let retrieval_allowed = profile.retrieval_cv_median_px + 0.01 < profile.baseline_cv_median_px
        && profile.retrieval_cv_p95_px <= profile.baseline_cv_p95_px;
    profile.retrieval_enabled = retrieval_allowed;
    let selected = evaluate(&profile, &observations, observations.len());
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
        "baseline":baseline,"selected":selected,"retrieval_selected":retrieval_allowed }))?
    );
    Ok(())
}
