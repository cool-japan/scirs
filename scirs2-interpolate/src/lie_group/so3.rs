//! Kernel interpolation on SO(3) (the rotation group).
//!
//! Points are unit quaternions in `[w, x, y, z]` (scalar-first) convention.
//! Quaternions are automatically normalized on input.
//! The geodesic distance is `2·arccos(|q₁·q₂|)`.

use super::kernel::{eval_kernel, so3_geodesic_dist, GeometricKernel};
use crate::InterpolateError;

/// RBF interpolator on SO(3), the group of 3D rotations.
///
/// SO(3) is identified with the unit quaternion sphere S³ / {±1}.
/// The geodesic distance used is the bi-invariant metric
/// `d(q₁, q₂) = 2·arccos(|q₁·q₂|)`.
///
/// # Examples
///
/// ```
/// use scirs2_interpolate::lie_group::so3::So3RbfInterpolator;
/// use scirs2_interpolate::lie_group::kernel::GeometricKernel;
///
/// // Identity rotation and 90-degree rotation around Z.
/// let identity = [1.0_f64, 0.0, 0.0, 0.0];
/// let rot_z90 = [0.7071_f64, 0.0, 0.0, 0.7071];
/// let points = vec![identity, rot_z90];
/// let values = vec![0.0_f64, 1.0];
/// let interp = So3RbfInterpolator::new(
///     &points, &values,
///     GeometricKernel::Heat { sigma: 1.0 },
///     1e-6,
/// ).expect("construction should succeed");
/// let v = interp.eval(&identity);
/// assert!((v - 0.0).abs() < 0.2);
/// ```
pub struct So3RbfInterpolator {
    centers: Vec<[f64; 4]>,
    weights: Vec<f64>,
    kernel: GeometricKernel,
}

impl So3RbfInterpolator {
    /// Fit the SO(3) RBF interpolator.
    ///
    /// # Arguments
    ///
    /// * `quaternions` — slice of unit quaternions `[w, x, y, z]` (normalized internally).
    /// * `values` — function values at each rotation.
    /// * `kernel` — geometric kernel to use.
    /// * `lambda` — Tikhonov regularization coefficient (must be ≥ 0).
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError`] on empty input, mismatched lengths, or singular system.
    pub fn new(
        quaternions: &[[f64; 4]],
        values: &[f64],
        kernel: GeometricKernel,
        lambda: f64,
    ) -> Result<Self, InterpolateError> {
        let n = quaternions.len();
        if n == 0 {
            return Err(InterpolateError::invalid_input(
                "at least 1 quaternion required for So3RbfInterpolator",
            ));
        }
        if n != values.len() {
            return Err(InterpolateError::shape_mismatch(
                n.to_string(),
                values.len().to_string(),
                "So3RbfInterpolator: quaternions vs values",
            ));
        }

        // Normalize all quaternions to the unit 4-sphere.
        let centers: Vec<[f64; 4]> = quaternions.iter().map(|q| normalize4(q)).collect();

        // Build regularized kernel matrix K[i,j] = k(d(qi,qj)) + λ·δ_{ij}.
        let mut k_mat = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let d = so3_geodesic_dist(&centers[i], &centers[j]);
                let mut kij = eval_kernel(d, &kernel);
                if i == j {
                    kij += lambda;
                }
                k_mat[i * n + j] = kij;
            }
        }

        let weights = solve_system(&k_mat, values, n)?;
        Ok(Self {
            centers,
            weights,
            kernel,
        })
    }

    /// Evaluate the interpolant at a new rotation (given as unit quaternion).
    ///
    /// The quaternion is normalized to the unit sphere automatically.
    pub fn eval(&self, q: &[f64; 4]) -> f64 {
        let qn = normalize4(q);
        self.centers
            .iter()
            .zip(self.weights.iter())
            .map(|(c, &w)| {
                let d = so3_geodesic_dist(&qn, c);
                w * eval_kernel(d, &self.kernel)
            })
            .sum()
    }

    /// Evaluate the interpolant at a batch of rotations.
    pub fn eval_batch(&self, quaternions: &[[f64; 4]]) -> Vec<f64> {
        quaternions.iter().map(|q| self.eval(q)).collect()
    }
}

/// Normalize a quaternion `[w, x, y, z]` to unit length.
///
/// If the quaternion is degenerate (near-zero norm), returns `[1, 0, 0, 0]` (identity).
fn normalize4(q: &[f64; 4]) -> [f64; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < f64::EPSILON {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
    }
}

/// Solve the n×n system A·x = b.
///
/// Attempts Cholesky decomposition first; falls back to Gaussian elimination.
fn solve_system(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, InterpolateError> {
    if let Ok(x) = cholesky_solve(a, b, n) {
        return Ok(x);
    }
    gauss_solve(a, b, n)
}

fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, InterpolateError> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s: f64 = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(InterpolateError::ComputationError(
                        "Cholesky: not positive definite".into(),
                    ));
                }
                l[i * n + j] = s.sqrt();
            } else {
                let lii = l[j * n + j];
                if lii.abs() < f64::EPSILON {
                    return Err(InterpolateError::ComputationError(
                        "Cholesky: zero diagonal".into(),
                    ));
                }
                l[i * n + j] = s / lii;
            }
        }
    }
    // Forward substitution L·y = b.
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        let lii = l[i * n + i];
        if lii.abs() < f64::EPSILON {
            return Err(InterpolateError::ComputationError(
                "Cholesky forward sub: zero diagonal".into(),
            ));
        }
        y[i] = s / lii;
    }
    // Back substitution Lᵀ·x = y.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        let lii = l[i * n + i];
        if lii.abs() < f64::EPSILON {
            return Err(InterpolateError::ComputationError(
                "Cholesky back sub: zero diagonal".into(),
            ));
        }
        x[i] = s / lii;
    }
    Ok(x)
}

fn gauss_solve(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, InterpolateError> {
    let mut mat = a.to_vec();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&i, &j| {
                mat[i * n + col]
                    .abs()
                    .partial_cmp(&mat[j * n + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| InterpolateError::ComputationError("empty matrix".into()))?;
        if pivot_row != col {
            for k in 0..n {
                mat.swap(col * n + k, pivot_row * n + k);
            }
            rhs.swap(col, pivot_row);
        }
        let piv = mat[col * n + col];
        if piv.abs() < 1e-14 {
            return Err(InterpolateError::ComputationError(
                "singular kernel matrix — increase lambda".into(),
            ));
        }
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / piv;
            for k in col..n {
                let val = mat[col * n + k];
                mat[row * n + k] -= factor * val;
            }
            let rv = rhs[col];
            rhs[row] -= factor * rv;
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in (i + 1)..n {
            s -= mat[i * n + j] * x[j];
        }
        let d = mat[i * n + i];
        if d.abs() < f64::EPSILON {
            return Err(InterpolateError::ComputationError(
                "back-substitution: zero diagonal".into(),
            ));
        }
        x[i] = s / d;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lie_group::kernel::{so3_geodesic_dist, GeometricKernel};
    use std::f64::consts::PI;

    /// Identity quaternion [1, 0, 0, 0].
    fn identity() -> [f64; 4] {
        [1.0, 0.0, 0.0, 0.0]
    }

    /// 180-degree rotation around X: [0, 1, 0, 0].
    fn rot_x_180() -> [f64; 4] {
        [0.0, 1.0, 0.0, 0.0]
    }

    /// 90-degree rotation around Z: [cos(45°), 0, 0, sin(45°)].
    fn rot_z_90() -> [f64; 4] {
        let s = (PI / 4.0).sin();
        let c = (PI / 4.0).cos();
        [c, 0.0, 0.0, s]
    }

    #[test]
    fn test_so3_geodesic_dist_identity() {
        let id = identity();
        let d = so3_geodesic_dist(&id, &id);
        assert!(
            d.abs() < 1e-12,
            "identity self-distance should be 0, got {d}"
        );
    }

    #[test]
    fn test_so3_geodesic_dist_180deg() {
        // Identity and 180° rotation: distance = π.
        let id = identity();
        let r180 = rot_x_180();
        let d = so3_geodesic_dist(&id, &r180);
        assert!(
            (d - PI).abs() < 1e-10,
            "180° rotation distance should be π, got {d}"
        );
    }

    #[test]
    fn test_so3_geodesic_dist_double_cover() {
        // q and -q represent the same rotation.
        let q = [0.5_f64, 0.5, 0.5, 0.5];
        let neg_q = [-0.5_f64, -0.5, -0.5, -0.5];
        let d = so3_geodesic_dist(&q, &neg_q);
        assert!(d.abs() < 1e-10, "q and -q should have distance 0, got {d}");
    }

    #[test]
    fn test_so3_rbf_constant_function() {
        // Use all 4 axis-aligned 90° rotations + identity for better coverage.
        let s = 1.0_f64 / 2.0_f64.sqrt();
        let pts = vec![
            identity(),
            rot_x_180(),
            rot_z_90(),
            [s, 0.0_f64, s, 0.0],  // 90° rotation around Y
            [s, 0.0_f64, 0.0, -s], // -90° rotation around Z
        ];
        let vals = vec![1.0_f64; pts.len()];
        let interp =
            So3RbfInterpolator::new(&pts, &vals, GeometricKernel::Heat { sigma: 2.0 }, 1e-8)
                .expect("construction should succeed");

        // Eval at training points — should be ≈1.
        let v = interp.eval(&identity());
        assert!(
            (v - 1.0).abs() < 0.3,
            "constant function should return ≈1 at identity, got {v}"
        );
        // Result should be finite and positive.
        let v2 = interp.eval(&[0.5_f64, 0.5, 0.5, 0.5]);
        assert!(
            v2.is_finite() && v2 > 0.0,
            "constant function eval should be finite and positive, got {v2}"
        );
    }

    #[test]
    fn test_so3_rbf_reproduces_training_points() {
        let pts = vec![identity(), rot_x_180(), rot_z_90()];
        let vals = vec![0.0_f64, 3.14, 1.57];
        let interp =
            So3RbfInterpolator::new(&pts, &vals, GeometricKernel::Heat { sigma: 0.5 }, 1e-10)
                .expect("construction should succeed");

        for (q, &expected) in pts.iter().zip(vals.iter()) {
            let got = interp.eval(q);
            assert!(
                (got - expected).abs() < 0.5,
                "at training quaternion {:?}, expected {expected}, got {got}",
                q
            );
        }
    }

    #[test]
    fn test_so3_rbf_empty_input_error() {
        let result = So3RbfInterpolator::new(&[], &[], GeometricKernel::Heat { sigma: 1.0 }, 1e-6);
        assert!(result.is_err(), "empty input should return error");
    }

    #[test]
    fn test_so3_rbf_mismatched_lengths_error() {
        let pts = vec![identity(), rot_x_180()];
        let vals = vec![1.0_f64];
        let result =
            So3RbfInterpolator::new(&pts, &vals, GeometricKernel::Heat { sigma: 1.0 }, 1e-6);
        assert!(result.is_err(), "mismatched lengths should return error");
    }

    #[test]
    fn test_so3_rbf_batch_eval() {
        let pts = vec![identity(), rot_x_180()];
        let vals = vec![1.0_f64, 2.0];
        let interp =
            So3RbfInterpolator::new(&pts, &vals, GeometricKernel::Heat { sigma: 1.0 }, 1e-6)
                .expect("construction should succeed");

        let batch = interp.eval_batch(&[identity(), rot_x_180()]);
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_so3_rbf_matern_kernel() {
        let pts = vec![identity(), rot_z_90()];
        let vals = vec![0.0_f64, 1.0];
        let interp = So3RbfInterpolator::new(
            &pts,
            &vals,
            GeometricKernel::Matern {
                nu: 2.5,
                length_scale: 1.0,
            },
            1e-6,
        )
        .expect("construction with Matern kernel should succeed");
        let v = interp.eval(&identity());
        assert!(
            v.is_finite(),
            "Matern kernel eval should be finite, got {v}"
        );
    }
}
