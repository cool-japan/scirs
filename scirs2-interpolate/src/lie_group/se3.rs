//! Kernel interpolation on SE(3), the group of rigid motions in ℝ³.
//!
//! SE(3) = ℝ³ ⋊ SO(3): each point is a pair `(translation, rotation)`.
//! Rotations are represented as unit quaternions `[w, x, y, z]`.
//!
//! The product-metric distance is:
//! `d((t1,q1),(t2,q2)) = ‖t1-t2‖ + w_rot · d_{SO(3)}(q1,q2)`
//!
//! where `w_rot` weights the rotational component.

use super::kernel::{eval_kernel, se3_geodesic_dist, GeometricKernel};
use crate::InterpolateError;

/// A point on SE(3): translation in ℝ³ plus unit quaternion rotation.
#[derive(Clone, Debug)]
pub struct Se3Point {
    /// Translation component `[x, y, z]`.
    pub translation: [f64; 3],
    /// Rotation as unit quaternion `[w, x, y, z]` (scalar-first).
    pub quaternion: [f64; 4],
}

impl Se3Point {
    /// Construct from a translation vector and quaternion.
    pub fn new(translation: [f64; 3], quaternion: [f64; 4]) -> Self {
        let q = normalize4(&quaternion);
        Self {
            translation,
            quaternion: q,
        }
    }
}

/// RBF interpolator on SE(3), the group of rigid body motions.
///
/// Uses the product metric: Euclidean distance on ℝ³ plus the SO(3) geodesic,
/// with a user-specified weight `w_rot` for the rotational component.
///
/// # Examples
///
/// ```
/// use scirs2_interpolate::lie_group::se3::{Se3RbfInterpolator, Se3Point};
/// use scirs2_interpolate::lie_group::kernel::GeometricKernel;
///
/// let id = Se3Point::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
/// let moved = Se3Point::new([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
/// let points = vec![id, moved];
/// let values = vec![0.0_f64, 1.0];
/// let interp = Se3RbfInterpolator::new(
///     &points, &values,
///     GeometricKernel::Heat { sigma: 1.0 },
///     1.0,
///     1e-6,
/// ).expect("construction should succeed");
/// let v = interp.eval(&Se3Point::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]));
/// assert!((v - 0.0).abs() < 0.2);
/// ```
pub struct Se3RbfInterpolator {
    centers: Vec<Se3Point>,
    weights: Vec<f64>,
    kernel: GeometricKernel,
    w_rot: f64,
}

impl Se3RbfInterpolator {
    /// Fit the SE(3) RBF interpolator.
    ///
    /// # Arguments
    ///
    /// * `points` — slice of SE(3) poses (each a `(translation, quaternion)` pair).
    /// * `values` — function values at each pose.
    /// * `kernel` — geometric kernel to use.
    /// * `w_rot` — weight for the rotational distance component (≥ 0).
    /// * `lambda` — Tikhonov regularization coefficient (≥ 0).
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError`] on empty input, mismatched lengths, or singular system.
    pub fn new(
        points: &[Se3Point],
        values: &[f64],
        kernel: GeometricKernel,
        w_rot: f64,
        lambda: f64,
    ) -> Result<Self, InterpolateError> {
        let n = points.len();
        if n == 0 {
            return Err(InterpolateError::invalid_input(
                "at least 1 pose required for Se3RbfInterpolator",
            ));
        }
        if n != values.len() {
            return Err(InterpolateError::shape_mismatch(
                n.to_string(),
                values.len().to_string(),
                "Se3RbfInterpolator: points vs values",
            ));
        }

        // Ensure all quaternions are normalized.
        let centers: Vec<Se3Point> = points
            .iter()
            .map(|p| Se3Point::new(p.translation, p.quaternion))
            .collect();

        // Build regularized kernel matrix.
        let mut k_mat = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let d = se3_geodesic_dist(
                    &centers[i].translation,
                    &centers[i].quaternion,
                    &centers[j].translation,
                    &centers[j].quaternion,
                    w_rot,
                );
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
            w_rot,
        })
    }

    /// Evaluate the interpolant at a new SE(3) pose.
    pub fn eval(&self, pose: &Se3Point) -> f64 {
        let p = Se3Point::new(pose.translation, pose.quaternion);
        self.centers
            .iter()
            .zip(self.weights.iter())
            .map(|(c, &w)| {
                let d = se3_geodesic_dist(
                    &p.translation,
                    &p.quaternion,
                    &c.translation,
                    &c.quaternion,
                    self.w_rot,
                );
                w * eval_kernel(d, &self.kernel)
            })
            .sum()
    }

    /// Evaluate the interpolant at a batch of poses.
    pub fn eval_batch(&self, poses: &[Se3Point]) -> Vec<f64> {
        poses.iter().map(|p| self.eval(p)).collect()
    }
}

/// Normalize a quaternion `[w, x, y, z]` to unit length.
fn normalize4(q: &[f64; 4]) -> [f64; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < f64::EPSILON {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
    }
}

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
    use crate::lie_group::kernel::{se3_geodesic_dist, GeometricKernel};

    fn identity_pose() -> Se3Point {
        Se3Point::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    }

    fn translated_pose(x: f64) -> Se3Point {
        Se3Point::new([x, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    }

    fn rotated_pose() -> Se3Point {
        Se3Point::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
    }

    #[test]
    fn test_se3_geodesic_dist_same_pose() {
        let id = identity_pose();
        let d = se3_geodesic_dist(
            &id.translation,
            &id.quaternion,
            &id.translation,
            &id.quaternion,
            1.0,
        );
        assert!(d.abs() < 1e-12, "same pose distance should be 0, got {d}");
    }

    #[test]
    fn test_se3_geodesic_dist_pure_translation() {
        let p1 = identity_pose();
        let p2 = translated_pose(3.0);
        let d = se3_geodesic_dist(
            &p1.translation,
            &p1.quaternion,
            &p2.translation,
            &p2.quaternion,
            1.0,
        );
        assert!(
            (d - 3.0).abs() < 1e-12,
            "pure translation distance should be ‖Δt‖=3, got {d}"
        );
    }

    #[test]
    fn test_se3_rbf_constant_function() {
        let pts = vec![
            identity_pose(),
            translated_pose(1.0),
            translated_pose(2.0),
            rotated_pose(),
        ];
        let vals = vec![1.0_f64; pts.len()];
        let interp =
            Se3RbfInterpolator::new(&pts, &vals, GeometricKernel::Heat { sigma: 2.0 }, 1.0, 1e-8)
                .expect("construction should succeed");

        let v = interp.eval(&translated_pose(1.5));
        assert!(
            (v - 1.0).abs() < 0.1,
            "constant function should return ≈1, got {v}"
        );
    }

    #[test]
    fn test_se3_rbf_reproduces_training_points() {
        let pts = vec![identity_pose(), translated_pose(1.0), rotated_pose()];
        let vals = vec![0.0_f64, 1.0, 2.0];
        let interp = Se3RbfInterpolator::new(
            &pts,
            &vals,
            GeometricKernel::Heat { sigma: 0.5 },
            1.0,
            1e-10,
        )
        .expect("construction should succeed");

        for (p, &expected) in pts.iter().zip(vals.iter()) {
            let got = interp.eval(p);
            assert!(
                (got - expected).abs() < 0.5,
                "at training pose, expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_se3_rbf_empty_input_error() {
        let result =
            Se3RbfInterpolator::new(&[], &[], GeometricKernel::Heat { sigma: 1.0 }, 1.0, 1e-6);
        assert!(result.is_err(), "empty input should return error");
    }

    #[test]
    fn test_se3_rbf_mismatched_lengths_error() {
        let pts = vec![identity_pose(), translated_pose(1.0)];
        let vals = vec![1.0_f64];
        let result =
            Se3RbfInterpolator::new(&pts, &vals, GeometricKernel::Heat { sigma: 1.0 }, 1.0, 1e-6);
        assert!(result.is_err(), "mismatched lengths should return error");
    }

    #[test]
    fn test_se3_rbf_batch_eval() {
        let pts = vec![identity_pose(), translated_pose(1.0)];
        let vals = vec![0.0_f64, 1.0];
        let interp =
            Se3RbfInterpolator::new(&pts, &vals, GeometricKernel::Heat { sigma: 1.0 }, 1.0, 1e-6)
                .expect("construction should succeed");

        let batch = interp.eval_batch(&[identity_pose(), translated_pose(1.0)]);
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_se3_rbf_w_rot_zero_ignores_rotation() {
        // With w_rot=0, rotational difference should be ignored.
        let pts = vec![identity_pose(), rotated_pose()];
        let vals = vec![1.0_f64, 1.0];
        let interp = Se3RbfInterpolator::new(
            &pts,
            &vals,
            GeometricKernel::Heat { sigma: 1.0 },
            0.0, // w_rot = 0 means rotation is irrelevant
            1e-6,
        )
        .expect("construction should succeed");
        // Both poses have the same translation (0,0,0), so they have distance 0 when w_rot=0.
        // The result should be well-behaved (no NaN).
        let v = interp.eval(&identity_pose());
        assert!(v.is_finite(), "eval should be finite, got {v}");
    }
}
