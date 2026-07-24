use serde::{Deserialize, Serialize};

use crate::gaze::Point;

const BASIS_SIZE: usize = 6;

/// A screen target and normalized binocular-gaze feature observed during fixation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CalibrationPoint {
    pub feature_x: f64,
    pub feature_y: f64,
    pub screen_x: f64,
    pub screen_y: f64,
}

/// Per-user quadratic gaze map. The quadratic terms model the screen-edge curvature that an
/// affine fit cannot represent on an ordinary webcam. Profiles with the previous three-term
/// map are deliberately rejected by deserialization; they need a fresh, measured calibration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationProfile {
    pub x_coefficients: [f64; BASIS_SIZE],
    pub y_coefficients: [f64; BASIS_SIZE],
    pub sample_count: usize,
    pub median_error_px: f64,
}

impl CalibrationProfile {
    pub fn fit(samples: &[CalibrationPoint]) -> Option<Self> {
        if samples.len() < BASIS_SIZE {
            return None;
        }
        let mut normal = [[0.0; BASIS_SIZE]; BASIS_SIZE];
        let mut x_rhs = [0.0; BASIS_SIZE];
        let mut y_rhs = [0.0; BASIS_SIZE];
        for sample in samples {
            let row = basis(sample.feature_x, sample.feature_y);
            for i in 0..BASIS_SIZE {
                x_rhs[i] += row[i] * sample.screen_x;
                y_rhs[i] += row[i] * sample.screen_y;
                for j in 0..BASIS_SIZE {
                    normal[i][j] += row[i] * row[j];
                }
            }
        }
        // Regularisation prevents a noisy webcam calibration from producing an unstable edge.
        for (index, row) in normal.iter_mut().enumerate() {
            row[index] += 1e-5;
        }
        let x_coefficients = solve(normal, x_rhs)?;
        let y_coefficients = solve(normal, y_rhs)?;
        let errors = samples
            .iter()
            .map(|sample| {
                let predicted = Point::new(
                    evaluate(x_coefficients, sample.feature_x, sample.feature_y),
                    evaluate(y_coefficients, sample.feature_x, sample.feature_y),
                );
                predicted.distance_to(Point::new(sample.screen_x, sample.screen_y))
            })
            .collect();
        Some(Self {
            x_coefficients,
            y_coefficients,
            sample_count: samples.len(),
            median_error_px: median(errors),
        })
    }

    pub fn map(&self, feature_x: f64, feature_y: f64) -> Point {
        Point::new(
            evaluate(self.x_coefficients, feature_x, feature_y),
            evaluate(self.y_coefficients, feature_x, feature_y),
        )
    }
}

fn basis(x: f64, y: f64) -> [f64; BASIS_SIZE] {
    [1.0, x, y, x * x, x * y, y * y]
}

fn evaluate(coefficients: [f64; BASIS_SIZE], x: f64, y: f64) -> f64 {
    basis(x, y)
        .into_iter()
        .zip(coefficients)
        .map(|(feature, coefficient)| feature * coefficient)
        .sum()
}

fn solve(
    mut matrix: [[f64; BASIS_SIZE]; BASIS_SIZE],
    mut rhs: [f64; BASIS_SIZE],
) -> Option<[f64; BASIS_SIZE]> {
    for column in 0..BASIS_SIZE {
        let pivot = (column..BASIS_SIZE)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))?;
        if matrix[pivot][column].abs() < 1e-10 {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let divisor = matrix[column][column];
        for value in &mut matrix[column][column..] {
            *value /= divisor;
        }
        rhs[column] /= divisor;
        let pivot_values = matrix[column];
        for row in 0..BASIS_SIZE {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for (value, pivot) in matrix[row][column..]
                .iter_mut()
                .zip(&pivot_values[column..])
            {
                *value -= factor * pivot;
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    Some(rhs)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_quadratic_calibration() {
        let samples = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (0.5, 0.2),
            (0.2, 0.7),
            (0.8, 0.4),
            (0.4, 0.9),
        ]
        .map(|(x, y)| CalibrationPoint {
            feature_x: x,
            feature_y: y,
            screen_x: 10.0 + 100.0 * x + 20.0 * y + 15.0 * x * x,
            screen_y: -3.0 + 10.0 * x + 200.0 * y + 8.0 * x * y,
        });
        let profile = CalibrationProfile::fit(&samples).expect("profile");
        let mapped = profile.map(0.5, 0.5);
        assert!((mapped.x - 73.75).abs() < 1e-3, "mapped = {mapped:?}");
        assert!((mapped.y - 103.0).abs() < 1e-3, "mapped = {mapped:?}");
    }
}
