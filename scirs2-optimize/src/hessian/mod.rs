//! Hessian approximations for second-order optimization
//!
//! This module provides structures and algorithms for maintaining and updating
//! dense Hessian approximations. These are building blocks used by
//! quasi-Newton methods (BFGS, SR1, DFP) and related algorithms.
//!
//! ## Provided approximations
//!
//! | Type | Update | Approximates |
//! |------|--------|--------------|
//! | [`FiniteDiffHessian`] | recomputed each call | Full Hessian H |
//! | [`SR1Update`]   | rank-1 update | Hessian H or inverse |
//! | [`BFGSUpdate`]  | rank-2 update (pos-def preserving) | Hessian H |
//! | [`DFP`]         | rank-2 update on **inverse** | Inverse Hessian H⁻¹ |
//!
//! All types implement the [`HessianApproximation`] trait.
//!
//! ## References
//!
//! - Nocedal & Wright, "Numerical Optimization", 2nd ed., Ch.6–7
//! - Dennis & Schnabel, "Numerical Methods for Unconstrained Optimization", 1983

use crate::error::OptimizeError;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};

// ─── HessianApproximation trait ───────────────────────────────────────────────

/// Common interface for Hessian (or inverse Hessian) approximations.
pub trait HessianApproximation: Send + Sync {
    /// Update the approximation given a new step-curvature pair.
    ///
    /// # Arguments
    /// * `s` – displacement `xₖ₊₁ - xₖ`
    /// * `y` – gradient change `∇f(xₖ₊₁) - ∇f(xₖ)`
    fn update(&mut self, s: &[f64], y: &[f64]) -> Result<(), OptimizeError>;

    /// Compute `H v` (Hessian times vector).
    fn multiply(&self, v: &[f64]) -> Result<Vec<f64>, OptimizeError>;

    /// Compute `H⁻¹ v` (inverse Hessian times vector).
    fn inverse_multiply(&self, v: &[f64]) -> Result<Vec<f64>, OptimizeError>;

    /// Return the current approximation as a dense matrix (Hessian or inverse, implementation-defined).
    fn to_dense(&self) -> Array2<f64>;

    /// Reset to the initial (identity) approximation.
    fn reset(&mut self);

    /// Problem dimension `n`.
    fn dim(&self) -> usize;
}

// ─── FiniteDiffHessian ────────────────────────────────────────────────────────

/// Finite-difference full Hessian approximation.
///
/// Recomputes the Hessian from scratch at each `update` call using central
/// differences. Suitable only for **small** problems (n ≤ a few hundred)
/// because cost is O(n²) function evaluations.
pub struct FiniteDiffHessian<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Send + Sync,
{
    /// Objective function
    pub fun: F,
    /// Current iterate x
    pub x: Vec<f64>,
    /// Finite-difference step size
    pub step: f64,
    /// Cached Hessian at `x`
    hess: Array2<f64>,
}

impl<F> FiniteDiffHessian<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Send + Sync,
{
    /// Create and compute the initial Hessian at `x0`.
    pub fn new(fun: F, x0: &[f64], step: f64) -> Result<Self, OptimizeError> {
        let x = x0.to_vec();
        let hess = compute_fd_hessian(&fun, &x, step)?;
        Ok(Self { fun, x, step, hess })
    }

    /// Create with the default step size.
    pub fn with_default_step(fun: F, x0: &[f64]) -> Result<Self, OptimizeError> {
        Self::new(fun, x0, f64::EPSILON.cbrt())
    }

    /// Recompute the Hessian at a new point.
    pub fn recompute_at(&mut self, x: &[f64]) -> Result<(), OptimizeError> {
        self.x = x.to_vec();
        self.hess = compute_fd_hessian(&self.fun, &self.x, self.step)?;
        Ok(())
    }
}

impl<F> HessianApproximation for FiniteDiffHessian<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Send + Sync,
{
    fn update(&mut self, _s: &[f64], _y: &[f64]) -> Result<(), OptimizeError> {
        // Recompute Hessian at the new point x + s
        let n = self.x.len();
        let mut x_new = self.x.clone();
        for i in 0..n {
            x_new[i] += _s[i];
        }
        self.recompute_at(&x_new)
    }

    fn multiply(&self, v: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        let n = self.hess.nrows();
        if v.len() != n {
            return Err(OptimizeError::ValueError("Dimension mismatch".to_string()));
        }
        let mut result = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                result[i] += self.hess[[i, j]] * v[j];
            }
        }
        Ok(result)
    }

    fn inverse_multiply(&self, v: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        let n = self.hess.nrows();
        if v.len() != n {
            return Err(OptimizeError::ValueError("Dimension mismatch".to_string()));
        }
        // Solve H x = v via Gaussian elimination with partial pivoting
        let mut a: Vec<Vec<f64>> = (0..n).map(|i| self.hess.row(i).to_vec()).collect();
        let mut b = v.to_vec();
        gaussian_solve(&mut a, &mut b).ok_or_else(|| {
            OptimizeError::ComputationError("Hessian is singular; cannot invert".to_string())
        })
    }

    fn to_dense(&self) -> Array2<f64> {
        self.hess.clone()
    }

    fn reset(&mut self) {
        let n = self.x.len();
        self.hess = Array2::eye(n);
    }

    fn dim(&self) -> usize {
        self.x.len()
    }
}

// ─── SR1 update ───────────────────────────────────────────────────────────────

/// Symmetric rank-1 (SR1) Hessian update.
///
/// Maintains an approximation `B ≈ H` updated by:
/// ```text
/// Bₖ₊₁ = Bₖ + (y - B s)(y - B s)ᵀ / (y - B s)ᵀ s
/// ```
/// SR1 can approximate indefinite Hessians (unlike BFGS) and often achieves
/// better Hessian quality than BFGS when used inside a trust-region framework.
///
/// The update is skipped when `|(y - Bs)ᵀ s| < r ‖y - Bs‖ ‖s‖` (default r=1e-8)
/// to avoid numerical instabilities.
pub struct SR1Update {
    /// Current Hessian approximation B
    pub b: Array2<f64>,
    /// Dimension
    pub n: usize,
    /// Skip-update safeguard parameter
    pub r: f64,
}

impl SR1Update {
    /// Create with identity initialisation.
    pub fn new(n: usize) -> Self {
        Self {
            b: Array2::eye(n),
            n,
            r: 1e-8,
        }
    }

    /// Create with custom safeguard `r`.
    pub fn with_safeguard(n: usize, r: f64) -> Self {
        Self {
            b: Array2::eye(n),
            n,
            r,
        }
    }
}

impl HessianApproximation for SR1Update {
    fn update(&mut self, s: &[f64], y: &[f64]) -> Result<(), OptimizeError> {
        let n = self.n;
        if s.len() != n || y.len() != n {
            return Err(OptimizeError::ValueError(
                "Dimension mismatch in SR1".to_string(),
            ));
        }

        // Compute u = y - B s
        let bs: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| self.b[[i, j]] * s[j]).sum())
            .collect();
        let u: Vec<f64> = (0..n).map(|i| y[i] - bs[i]).collect();

        let uts: f64 = (0..n).map(|i| u[i] * s[i]).sum();
        let u_norm: f64 = (0..n).map(|i| u[i] * u[i]).sum::<f64>().sqrt();
        let s_norm: f64 = (0..n).map(|i| s[i] * s[i]).sum::<f64>().sqrt();

        // Skip update if the denominator is too small (safeguard)
        if uts.abs() <= self.r * u_norm * s_norm {
            return Ok(());
        }

        // B = B + u uᵀ / uᵀ s
        for i in 0..n {
            for j in 0..n {
                self.b[[i, j]] += u[i] * u[j] / uts;
            }
        }

        Ok(())
    }

    fn multiply(&self, v: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        let n = self.n;
        if v.len() != n {
            return Err(OptimizeError::ValueError("Dimension mismatch".to_string()));
        }
        let mut result = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                result[i] += self.b[[i, j]] * v[j];
            }
        }
        Ok(result)
    }

    fn inverse_multiply(&self, v: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        let n = self.n;
        if v.len() != n {
            return Err(OptimizeError::ValueError("Dimension mismatch".to_string()));
        }
        let mut a: Vec<Vec<f64>> = (0..n).map(|i| self.b.row(i).to_vec()).collect();
        let mut b = v.to_vec();
        gaussian_solve(&mut a, &mut b)
            .ok_or_else(|| OptimizeError::ComputationError("SR1 Hessian singular".to_string()))
    }

    fn to_dense(&self) -> Array2<f64> {
        self.b.clone()
    }

    fn reset(&mut self) {
        self.b = Array2::eye(self.n);
    }

    fn dim(&self) -> usize {
        self.n
    }
}

// ─── BFGS update ─────────────────────────────────────────────────────────────

/// BFGS Hessian update (dense).
///
/// Maintains `B ≈ H` and `H_inv ≈ H⁻¹` via the rank-2 formula:
/// ```text
/// Bₖ₊₁ = Bₖ - (Bₖ sₖ)(Bₖ sₖ)ᵀ / (sₖᵀ Bₖ sₖ) + yₖ yₖᵀ / (yₖᵀ sₖ)
/// ```
/// and the Sherman–Morrison–Woodbury update for the inverse:
/// ```text
/// H_{k+1} = (I - ρ sₖ yₖᵀ) Hₖ (I - ρ yₖ sₖᵀ) + ρ sₖ sₖᵀ
/// ```
/// where `ρ = 1 / (yₖᵀ sₖ)`.
pub struct BFGSUpdate {
    /// Hessian approximation B ≈ H
    pub b: Array2<f64>,
    /// Inverse Hessian approximation H ≈ H⁻¹
    pub h_inv: Array2<f64>,
    /// Dimension
    pub n: usize,
}

impl BFGSUpdate {
    /// Create with identity initialisations.
    pub fn new(n: usize) -> Self {
        Self {
            b: Array2::eye(n),
            h_inv: Array2::eye(n),
            n,
        }
    }
}

impl HessianApproximation for BFGSUpdate {
    fn update(&mut self, s: &[f64], y: &[f64]) -> Result<(), OptimizeError> {
        let n = self.n;
        if s.len() != n || y.len() != n {
            return Err(OptimizeError::ValueError(
                "Dimension mismatch in BFGS".to_string(),
            ));
        }

        let sy: f64 = (0..n).map(|i| s[i] * y[i]).sum();
        if sy <= 1e-10 * (0..n).map(|i| y[i] * y[i]).sum::<f64>().sqrt() {
            // Curvature condition not satisfied; skip update
            return Ok(());
        }

        let rho = 1.0 / sy;

        // --- Update Hessian B ---
        // Bs = B * s
        let bs: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| self.b[[i, j]] * s[j]).sum())
            .collect();
        let s_bs: f64 = (0..n).map(|i| s[i] * bs[i]).sum();

        for i in 0..n {
            for j in 0..n {
                self.b[[i, j]] += rho * y[i] * y[j] - bs[i] * bs[j] / s_bs.max(1e-300);
            }
        }

        // --- Update inverse Hessian H_inv (Sherman-Morrison-Woodbury) ---
        // H_new = (I - ρ s yᵀ) H (I - ρ y sᵀ) + ρ s sᵀ
        let mut h_new = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0f64;
                for k in 0..n {
                    for l in 0..n {
                        let li = if i == k { 1.0 } else { 0.0 } - rho * s[i] * y[k];
                        let rj = if j == l { 1.0 } else { 0.0 } - rho * y[l] * s[j];
                        val += li * self.h_inv[[k, l]] * rj;
                    }
                }
                h_new[[i, j]] = val + rho * s[i] * s[j];
            }
        }
        self.h_inv = h_new;

        Ok(())
    }

    fn multiply(&self, v: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        let n = self.n;
        if v.len() != n {
            return Err(OptimizeError::ValueError("Dimension mismatch".to_string()));
        }
        let mut result = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                result[i] += self.b[[i, j]] * v[j];
            }
        }
        Ok(result)
    }

    fn inverse_multiply(&self, v: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        let n = self.n;
        if v.len() != n {
            return Err(OptimizeError::ValueError("Dimension mismatch".to_string()));
        }
        let mut result = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                result[i] += self.h_inv[[i, j]] * v[j];
            }
        }
        Ok(result)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.b.clone()
    }

    fn reset(&mut self) {
        self.b = Array2::eye(self.n);
        self.h_inv = Array2::eye(self.n);
    }

    fn dim(&self) -> usize {
        self.n
    }
}

// ─── DFP update ───────────────────────────────────────────────────────────────

/// DFP (Davidon–Fletcher–Powell) inverse Hessian update.
///
/// Directly updates an approximation to the **inverse** Hessian `C ≈ H⁻¹`:
/// ```text
/// Cₖ₊₁ = Cₖ - (Cₖ yₖ)(Cₖ yₖ)ᵀ / (yₖᵀ Cₖ yₖ) + sₖ sₖᵀ / (yₖᵀ sₖ)
/// ```
/// DFP is the dual of BFGS. It updates the inverse directly and can be more
/// efficient when `inverse_multiply` is called much more frequently than `multiply`.
pub struct DFP {
    /// Inverse Hessian approximation C ≈ H⁻¹
    pub c: Array2<f64>,
    /// Dimension
    pub n: usize,
}

impl DFP {
    /// Create with identity initialisation.
    pub fn new(n: usize) -> Self {
        Self {
            c: Array2::eye(n),
            n,
        }
    }
}

impl HessianApproximation for DFP {
    fn update(&mut self, s: &[f64], y: &[f64]) -> Result<(), OptimizeError> {
        let n = self.n;
        if s.len() != n || y.len() != n {
            return Err(OptimizeError::ValueError(
                "Dimension mismatch in DFP".to_string(),
            ));
        }

        let sy: f64 = (0..n).map(|i| s[i] * y[i]).sum();
        if sy <= 1e-10 {
            return Ok(());
        }

        // Cy = C * y
        let cy: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| self.c[[i, j]] * y[j]).sum())
            .collect();
        let y_cy: f64 = (0..n).map(|i| y[i] * cy[i]).sum();

        // C = C - (C y)(C y)ᵀ / (yᵀ C y) + s sᵀ / (yᵀ s)
        for i in 0..n {
            for j in 0..n {
                self.c[[i, j]] += s[i] * s[j] / sy - cy[i] * cy[j] / y_cy.max(1e-300);
            }
        }

        Ok(())
    }

    fn multiply(&self, v: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        let n = self.n;
        if v.len() != n {
            return Err(OptimizeError::ValueError("Dimension mismatch".to_string()));
        }
        // Multiply by C^{-1} (approximate Hessian) via Gaussian elimination
        let mut a: Vec<Vec<f64>> = (0..n).map(|i| self.c.row(i).to_vec()).collect();
        let mut b = v.to_vec();
        gaussian_solve(&mut a, &mut b)
            .ok_or_else(|| OptimizeError::ComputationError("DFP inverse singular".to_string()))
    }

    fn inverse_multiply(&self, v: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        let n = self.n;
        if v.len() != n {
            return Err(OptimizeError::ValueError("Dimension mismatch".to_string()));
        }
        let mut result = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                result[i] += self.c[[i, j]] * v[j];
            }
        }
        Ok(result)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.c.clone()
    }

    fn reset(&mut self) {
        self.c = Array2::eye(self.n);
    }

    fn dim(&self) -> usize {
        self.n
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Compute the full finite-difference Hessian at `x`.
fn compute_fd_hessian<F>(fun: &F, x: &[f64], step: f64) -> Result<Array2<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let mut hess = Array2::<f64>::zeros((n, n));
    let mut x_tmp = Array1::from(x.to_vec());

    let f0 = fun(&x_tmp.view());

    for i in 0..n {
        let hi = step * (1.0 + x[i].abs());

        x_tmp[i] = x[i] + hi;
        let fp = fun(&x_tmp.view());
        x_tmp[i] = x[i] - hi;
        let fm = fun(&x_tmp.view());
        x_tmp[i] = x[i];

        if !fp.is_finite() || !fm.is_finite() {
            return Err(OptimizeError::ComputationError(
                "Non-finite value computing Hessian diagonal".to_string(),
            ));
        }
        hess[[i, i]] = (fp - 2.0 * f0 + fm) / (hi * hi);

        for j in (i + 1)..n {
            let hj = step * (1.0 + x[j].abs());

            x_tmp[i] = x[i] + hi;
            x_tmp[j] = x[j] + hj;
            let fpp = fun(&x_tmp.view());

            x_tmp[i] = x[i] + hi;
            x_tmp[j] = x[j] - hj;
            let fpm = fun(&x_tmp.view());

            x_tmp[i] = x[i] - hi;
            x_tmp[j] = x[j] + hj;
            let fmp = fun(&x_tmp.view());

            x_tmp[i] = x[i] - hi;
            x_tmp[j] = x[j] - hj;
            let fmm = fun(&x_tmp.view());

            x_tmp[i] = x[i];
            x_tmp[j] = x[j];

            let val = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
            hess[[i, j]] = val;
            hess[[j, i]] = val;
        }
    }

    Ok(hess)
}

/// Gaussian elimination with partial pivoting. Returns `None` if singular.
fn gaussian_solve(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();

    for col in 0..n {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        a.swap(col, max_row);
        b.swap(col, max_row);

        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            b[row] -= factor * b[col];
            for k in col..n {
                let v = a[col][k];
                a[row][k] -= factor * v;
            }
        }
    }

    // Back-substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        if a[i][i].abs() < 1e-14 {
            return None;
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn quadratic(x: &ArrayView1<f64>) -> f64 {
        x[0] * x[0] + 4.0 * x[1] * x[1]
    }

    #[test]
    fn test_bfgs_identity_update() {
        let mut bfgs = BFGSUpdate::new(2);
        // After one step (s, y) on a quadratic f = x₀² + 4 x₁²
        // H = diag(2, 8), so s = [-1, -1], y = [-2, -8] → sy = 2+8 = 10
        let s = vec![-1.0, -1.0];
        let y = vec![-2.0, -8.0];
        bfgs.update(&s, &y).expect("BFGS update failed");

        let v = vec![1.0, 0.0];
        let hv = bfgs.multiply(&v).expect("multiply failed");
        // Should be approximately [2, 0] after correct curvature
        assert!(hv[0] > 0.0); // positive Hessian
    }

    #[test]
    fn test_sr1_update() {
        let mut sr1 = SR1Update::new(2);
        let s = vec![1.0, 0.0];
        let y = vec![2.0, 0.0]; // H_11 = 2
        sr1.update(&s, &y).expect("SR1 update failed");
        let hv = sr1.multiply(&vec![1.0, 0.0]).expect("multiply failed");
        assert_abs_diff_eq!(hv[0], 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_dfp_inverse_multiply() {
        let mut dfp = DFP::new(2);
        let s = vec![1.0, 0.0];
        let y = vec![2.0, 0.0];
        dfp.update(&s, &y).expect("DFP update failed");
        // C = H⁻¹, so C * y should give s (approximately after one update)
        let sv = dfp.inverse_multiply(&y).expect("inverse multiply failed");
        // After one DFP step with s=(1,0) and y=(2,0), C y ≈ s = (1, 0)
        assert!(sv[0] > 0.0);
    }

    #[test]
    fn test_fd_hessian_diagonal() {
        // For f = x₀² + 4 x₁², H = diag(2, 8)
        let hess = compute_fd_hessian(&quadratic, &[0.0, 0.0], 1e-5).expect("FD Hessian failed");
        assert_abs_diff_eq!(hess[[0, 0]], 2.0, epsilon = 1e-4);
        assert_abs_diff_eq!(hess[[1, 1]], 8.0, epsilon = 1e-4);
        assert_abs_diff_eq!(hess[[0, 1]], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_gaussian_solve() {
        let mut a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let mut b = vec![5.0, 7.0];
        let x = gaussian_solve(&mut a, &mut b).expect("solve failed");
        // 2x + y = 5, x + 3y = 7 → x = 8/5 = 1.6, y = 9/5 = 1.8
        assert_abs_diff_eq!(x[0], 1.6, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 1.8, epsilon = 1e-10);
    }

    #[test]
    fn test_bfgs_reset() {
        let mut bfgs = BFGSUpdate::new(3);
        bfgs.update(&[1.0, 0.0, 0.0], &[2.0, 0.0, 0.0])
            .expect("update failed");
        bfgs.reset();
        let v = vec![1.0, 2.0, 3.0];
        let hv = bfgs.multiply(&v).expect("multiply failed");
        // After reset, B = I, so H v = v
        assert_abs_diff_eq!(hv[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(hv[1], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(hv[2], 3.0, epsilon = 1e-12);
    }
}
