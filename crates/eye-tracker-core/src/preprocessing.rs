/// Normalize ADAS-0002 output and restore camera roll exactly once.
/// Intel's reference gazeVectorToGazeAngles maps (0, 0, -1) to straight ahead.
/// Preserve that native convention: screen-facing gaze has negative Z. Never
/// use abs(Z), which would also admit a genuinely reversed direction.
pub fn normalize_gaze_vector(vector: [f32; 3], roll_degrees: f32) -> Option<[f64; 3]> {
    if !roll_degrees.is_finite() || vector.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let v = vector.map(f64::from);
    let magnitude = v.iter().map(|v| v * v).sum::<f64>().sqrt();
    if magnitude <= 1e-9 {
        return None;
    }
    let roll = f64::from(roll_degrees).to_radians();
    let direction = [
        (v[0] * roll.cos() + v[1] * roll.sin()) / magnitude,
        (-v[0] * roll.sin() + v[1] * roll.cos()) / magnitude,
        v[2] / magnitude,
    ];
    (direction[2] <= -0.05).then_some(direction)
}
