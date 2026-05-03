//! Kernel interpolation on S² (the unit 2-sphere).
//!
//! Points are unit vectors in ℝ³ (automatically normalized on input).
//! The geodesic distance is the great-circle arc length.
//!
//! The interpolator solves a regularized kernel system
//! `(K + λI)·w = y`, where `K[i,j] = k(dist(p_i, p_j))`.

use super::kernel::{eval_kernel, sphere_geodesic_dist, GeometricKernel};
use crate::InterpolateError;

/// RBF interpolator on the unit 2-sphere S².
///
/// All input points are normalized to lie on the unit sphere.
/// Tikhonov regularization (λ) ensures a well-conditioned system.
///
/// # Examples
///
/// ```
/// use scirs2_interpolate::lie_group::sphere::SphereRbfInterpolator;
/// use scirs2_interpolate::lie_group::kernel::GeometricKernel;
///
/// let points = vec![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let values = vec![1.0_f64, 2.0, 3.0];
/// let interp = SphereRbfInterpolator::new(
///     &points, &values,
///     GeometricKernel::Heat { sigma: 1.0 },
///     1e-6,
/// ).expect("construction should succeed");
/// let v = interp.eval(&[1.0, 0.0, 0.0]);
/// assert!((v - 1.0).abs() < 0.1);
/// ```
pub struct SphereRbfInterpolator {
    centers: Vec<[f64; 3]>,
    weights: Vec<f64>,
    kernel: GeometricKernel,
}

impl SphereRbfInterpolator {
    /// Fit the sphere RBF interpolator.
    ///
    /// # Arguments
    ///
    /// * `points` — slice of ℝ³ vectors (normalized to unit sphere internally).
    /// * `values` — function values at each point.
    /// * `kernel` — geometric kernel to use.
    /// * `lambda` — Tikhonov regularization coefficient (must be ≥ 0).
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError`] if `points` and `values` have different lengths,
    /// if no points are given, or if the resulting system is singular (increase λ).
    pub fn new(
        points: &[[f64; 3]],
        values: &[f64],
        kernel: GeometricKernel,
        lambda: f64,
    ) -> Result<Self, InterpolateError> {
        let n = points.len();
        if n == 0 {
            return Err(InterpolateError::invalid_input(
                "at least 1 point required for SphereRbfInterpolator",
            ));
        }
        if n != values.len() {
            return Err(InterpolateError::shape_mismatch(
                n.to_string(),
                values.len().to_string(),
                "SphereRbfInterpolator: points vs values",
            ));
        }

        // Normalize all centers to the unit sphere.
        let centers: Vec<[f64; 3]> = points.iter().map(|p| normalize3(p)).collect();

        // Build regularized kernel matrix K[i,j] = k(d(pi,pj)) + λ·δ_{ij}.
        let mut k_mat = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let d = sphere_geodesic_dist(&centers[i], &centers[j]);
                let mut kij = eval_kernel(d, &kernel);
                if i == j {
                    kij += lambda;
                }
                k_mat[i * n + j] = kij;
            }
        }

        // Solve the linear system via Cholesky (preferred for SPD matrices) or
        // Gaussian elimination with partial pivoting as fallback.
        let weights = solve_spd_system(&k_mat, values, n)?;

        Ok(Self {
            centers,
            weights,
            kernel,
        })
    }

    /// Evaluate the interpolant at a new sphere point.
    ///
    /// The input point is normalized to the unit sphere automatically.
    pub fn eval(&self, point: &[f64; 3]) -> f64 {
        let p = normalize3(point);
        self.centers
            .iter()
            .zip(self.weights.iter())
            .map(|(c, &w)| {
                let d = sphere_geodesic_dist(&p, c);
                w * eval_kernel(d, &self.kernel)
            })
            .sum()
    }

    /// Evaluate the interpolant at a batch of sphere points.
    pub fn eval_batch(&self, points: &[[f64; 3]]) -> Vec<f64> {
        points.iter().map(|p| self.eval(p)).collect()
    }
}

/// Normalize a ℝ³ vector to the unit sphere.
///
/// Returns the original direction; if the norm is negligible, returns `[1,0,0]`.
fn normalize3(v: &[f64; 3]) -> [f64; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm < f64::EPSILON {
        [1.0, 0.0, 0.0]
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}

/// Solve `A·x = b` where `A` is approximately symmetric positive definite (n×n).
///
/// Uses Cholesky decomposition (in-place modified Cholesky) for SPD systems.
/// Falls back to Gaussian elimination with partial pivoting if the matrix is
/// not numerically positive definite.
fn solve_spd_system(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, InterpolateError> {
    // Attempt Cholesky: A = L Lᵀ.
    if let Ok(x) = try_cholesky(a, b, n) {
        return Ok(x);
    }
    // Fallback: Gaussian elimination with partial pivoting.
    gauss_elim(a, b, n)
}

/// Attempt Cholesky decomposition and solve.  Returns Err on non-SPD matrix.
fn try_cholesky(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, InterpolateError> {
    // Compute lower Cholesky factor L stored in flat column-major (row-major here).
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
                        "Cholesky: matrix is not positive definite".into(),
                    ));
                }
                l[i * n + j] = s.sqrt();
            } else {
                let lii = l[j * n + j];
                if lii.abs() < f64::EPSILON {
                    return Err(InterpolateError::ComputationError(
                        "Cholesky: near-zero diagonal".into(),
                    ));
                }
                l[i * n + j] = s / lii;
            }
        }
    }

    // Forward substitution: L·y = b.
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        let lii = l[i * n + i];
        if lii.abs() < f64::EPSILON {
            return Err(InterpolateError::ComputationError(
                "Cholesky: zero diagonal in forward sub".into(),
            ));
        }
        y[i] = s / lii;
    }

    // Back substitution: Lᵀ·x = y.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        let lii = l[i * n + i];
        if lii.abs() < f64::EPSILON {
            return Err(InterpolateError::ComputationError(
                "Cholesky: zero diagonal in back sub".into(),
            ));
        }
        x[i] = s / lii;
    }
    Ok(x)
}

/// Gaussian elimination with partial pivoting.
fn gauss_elim(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, InterpolateError> {
    let mut mat = a.to_vec();
    let mut rhs = b.to_vec();

    for col in 0..n {
        // Find the row with the largest absolute value in column `col`.
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
                "singular kernel matrix — increase lambda (regularization)".into(),
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

    // Back substitution.
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
    use crate::lie_group::kernel::{sphere_geodesic_dist, GeometricKernel};
    use std::f64::consts::PI;

    fn unit_sphere_points() -> Vec<[f64; 3]> {
        vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    }

    #[test]
    fn test_sphere_rbf_constant_function() {
        let pts = unit_sphere_points();
        let vals = vec![1.0_f64; pts.len()];
        let interp =
            SphereRbfInterpolator::new(&pts, &vals, GeometricKernel::Heat { sigma: 1.0 }, 1e-8)
                .expect("construction should succeed");

        // Evaluate at training points — should return ≈ 1 there.
        let test_pts = vec![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for p in &test_pts {
            let v = interp.eval(p);
            assert!(
                (v - 1.0).abs() < 0.15,
                "constant function should return ≈1 at {:?}, got {v}",
                p
            );
        }
    }

    #[test]
    fn test_sphere_rbf_reproduces_training_points() {
        let pts = unit_sphere_points();
        let vals: Vec<f64> = (0..pts.len()).map(|i| (i + 1) as f64).collect();
        let interp =
            SphereRbfInterpolator::new(&pts, &vals, GeometricKernel::Heat { sigma: 0.5 }, 1e-10)
                .expect("construction should succeed");

        for (p, &expected) in pts.iter().zip(vals.iter()) {
            let got = interp.eval(p);
            assert!(
                (got - expected).abs() < 0.5,
                "at training point {:?}, expected {expected}, got {got}",
                p
            );
        }
    }

    #[test]
    fn test_sphere_geodesic_dist_poles() {
        let north = [0.0_f64, 0.0, 1.0];
        let south = [0.0_f64, 0.0, -1.0];
        let d = sphere_geodesic_dist(&north, &south);
        assert!(
            (d - PI).abs() < 1e-12,
            "antipodal distance should be π, got {d}"
        );
    }

    #[test]
    fn test_sphere_rbf_antipodal_robustness() {
        // Points near poles — no NaN expected.
        let pts = vec![[0.0_f64, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]];
        let vals = vec![1.0_f64, 2.0, 3.0];
        let interp =
            SphereRbfInterpolator::new(&pts, &vals, GeometricKernel::Heat { sigma: 0.5 }, 1e-6)
                .expect("construction should succeed");

        let v = interp.eval(&[0.0, 0.0, 1.0]);
        assert!(
            v.is_finite(),
            "eval at north pole should be finite, got {v}"
        );
        let v2 = interp.eval(&[0.0, 0.0, -1.0]);
        assert!(
            v2.is_finite(),
            "eval at south pole should be finite, got {v2}"
        );
    }

    #[test]
    fn test_sphere_rbf_all_kernels() {
        let pts = vec![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let vals = vec![1.0_f64, 2.0, 3.0];

        let kernels = vec![
            GeometricKernel::Heat { sigma: 1.0 },
            GeometricKernel::Matern {
                nu: 1.5,
                length_scale: 1.0,
            },
            GeometricKernel::Matern {
                nu: 2.5,
                length_scale: 1.0,
            },
            GeometricKernel::SphericalHarmonic {
                bandwidth: 5,
                sigma: 0.5,
            },
        ];

        for kernel in kernels {
            let interp = SphereRbfInterpolator::new(&pts, &vals, kernel.clone(), 1e-6)
                .expect("construction should succeed");
            let v = interp.eval(&[0.5773_f64, 0.5773, 0.5773]);
            assert!(
                v.is_finite(),
                "kernel eval should return finite value, got {v}"
            );
        }
    }

    #[test]
    fn test_sphere_rbf_batch_eval() {
        let pts = vec![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let vals = vec![1.0_f64, 2.0];
        let interp =
            SphereRbfInterpolator::new(&pts, &vals, GeometricKernel::Heat { sigma: 1.0 }, 1e-6)
                .expect("construction should succeed");

        let test_pts = vec![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let results = interp.eval_batch(&test_pts);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sphere_rbf_empty_input_error() {
        let result =
            SphereRbfInterpolator::new(&[], &[], GeometricKernel::Heat { sigma: 1.0 }, 1e-6);
        assert!(result.is_err(), "empty input should return error");
    }

    #[test]
    fn test_sphere_rbf_mismatched_lengths_error() {
        let pts = vec![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let vals = vec![1.0_f64];
        let result =
            SphereRbfInterpolator::new(&pts, &vals, GeometricKernel::Heat { sigma: 1.0 }, 1e-6);
        assert!(result.is_err(), "mismatched lengths should return error");
    }
}
