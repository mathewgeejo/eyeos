use serde::{Deserialize, Serialize};

use crate::gaze::Point;

/// A screen target and the normalized eye feature observed while the user fixates it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CalibrationPoint {
    pub feature_x: f64,
    pub feature_y: f64,
    pub screen_x: f64,
    pub screen_y: f64,
}

/// A regularised affine mapping. Keeping it small makes calibration explainable and avoids a
/// model that drifts unexpectedly between sessions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationProfile {
    pub x_coefficients: [f64; 3],
    pub y_coefficients: [f64; 3],
    pub sample_count: usize,
    pub median_error_px: f64,
}

impl CalibrationProfile {
    pub fn fit(samples: &[CalibrationPoint]) -> Option<Self> {
        if samples.len() < 3 {
            return None;
        }

        let mut normal = [[0.0; 3]; 3];
        let mut x_rhs = [0.0; 3];
        let mut y_rhs = [0.0; 3];

        for sample in samples {
            let row = [1.0, sample.feature_x, sample.feature_y];
            for i in 0..3 {
                x_rhs[i] += row[i] * sample.screen_x;
                y_rhs[i] += row[i] * sample.screen_y;
                for j in 0..3 {
                    normal[i][j] += row[i] * row[j];
                }
            }
        }

        // Tikhonov regularisation makes nearly collinear calibration samples harmless.
        for (index, row) in normal.iter_mut().enumerate() {
            row[index] += 1e-6;
        }

        let x_coefficients = solve_3x3(normal, x_rhs)?;
        let y_coefficients = solve_3x3(normal, y_rhs)?;
        let errors = samples
            .iter()
            .map(|sample| {
                let predicted = Point::new(
                    evaluate(x_coefficients, sample.feature_x, sample.feature_y),
                    evaluate(y_coefficients, sample.feature_x, sample.feature_y),
                );
                predicted.distance_to(Point::new(sample.screen_x, sample.screen_y))
            })
            .collect::<Vec<_>>();

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

fn evaluate(coefficients: [f64; 3], x: f64, y: f64) -> f64 {
    coefficients[0] + coefficients[1] * x + coefficients[2] * y
}

fn solve_3x3(mut matrix: [[f64; 3]; 3], mut rhs: [f64; 3]) -> Option<[f64; 3]> {
    for column in 0..3 {
        let pivot = (column..3).max_by(|&a, &b| {
            matrix[a][column]
                .abs()
                .partial_cmp(&matrix[b][column].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if matrix[pivot][column].abs() < 1e-12 {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);

        let divisor = matrix[column][column];
        for cell in &mut matrix[column][column..] {
            *cell /= divisor;
        }
        rhs[column] /= divisor;
        let pivot_values = matrix[column];

        for row in 0..3 {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for (target, pivot) in matrix[row][column..]
                .iter_mut()
                .zip(pivot_values[column..].iter())
            {
                *target -= factor * pivot;
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    Some(rhs)
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
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
    fn solves_an_affine_calibration() {
        let samples = [
            CalibrationPoint {
                feature_x: 0.0,
                feature_y: 0.0,
                screen_x: 10.0,
                screen_y: -3.0,
            },
            CalibrationPoint {
                feature_x: 1.0,
                feature_y: 0.0,
                screen_x: 110.0,
                screen_y: 7.0,
            },
            CalibrationPoint {
                feature_x: 0.0,
                feature_y: 1.0,
                screen_x: 30.0,
                screen_y: 197.0,
            },
            CalibrationPoint {
                feature_x: 1.0,
                feature_y: 1.0,
                screen_x: 130.0,
                screen_y: 207.0,
            },
        ];
        let profile = CalibrationProfile::fit(&samples).expect("profile");
        let mapped = profile.map(0.5, 0.5);
        assert!((mapped.x - 70.0).abs() < 1e-3, "mapped = {mapped:?}");
        assert!((mapped.y - 102.0).abs() < 1e-3, "mapped = {mapped:?}");
        assert!(profile.median_error_px < 1e-3);
    }

    #[test]
    fn requires_three_distinct_samples() {
        assert!(CalibrationProfile::fit(&[]).is_none());
        assert!(
            CalibrationProfile::fit(&[CalibrationPoint {
                feature_x: 0.0,
                feature_y: 0.0,
                screen_x: 0.0,
                screen_y: 0.0,
            }])
            .is_none()
        );
    }
}
