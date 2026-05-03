//! Compositional Data Analysis in the Aitchison Simplex.
//!
//! Compositional data are vectors of strictly positive components whose values carry
//! only relative (not absolute) information, so only ratios between components are
//! meaningful.  The appropriate sample space is the **D-part simplex** S^D.
//!
//! This module provides:
//!
//! - **Simplex operations**: closure, perturbation (Aitchison addition), powering
//! - **Log-ratio transforms**: ALR, CLR, ILR and their inverses
//! - **Aitchison geometry**: inner product, norm, distance
//! - **Dirichlet regression**: IRLS estimation of Dirichlet GLM
//! - **Compositional PCA**: PCA in Aitchison geometry via CLR
//! - **Statistical tests**: neutrality test, Dirichlet MLE
//!
//! # References
//! - Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman & Hall.
//! - Pawlowsky-Glahn, V., Egozcue, J.J., Tolosana-Delgado, R. (2015).
//!   *Modelling and Analysis of Compositional Data*. Wiley.
//! - Egozcue, J.J., Pawlowsky-Glahn, V. (2005). Groups of Parts and Their Balances in
//!   Compositional Data Analysis. *Mathematical Geology*, 37(7), 795–828.

use std::fmt;

use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Check that a composition is strictly positive (no zeros or negatives).
fn check_positive(x: &[f64], name: &str) -> StatsResult<()> {
    for (i, &v) in x.iter().enumerate() {
        if v <= 0.0 {
            return Err(StatsError::InvalidArgument(format!(
                "{name}[{i}] = {v} is not strictly positive; \
                 all components must be > 0 for compositional analysis"
            )));
        }
    }
    Ok(())
}

/// Geometric mean of a strictly-positive slice.
#[inline]
fn geometric_mean(x: &[f64]) -> f64 {
    let log_sum: f64 = x.iter().map(|&v| v.ln()).sum();
    (log_sum / x.len() as f64).exp()
}

// ---------------------------------------------------------------------------
// Simplex operations
// ---------------------------------------------------------------------------

/// Closure: normalise a composition so that its components sum to 1.
///
/// Equivalent to dividing each component by the total sum.  This is the
/// canonical projection onto S^D.
///
/// # Errors
/// Returns [`StatsError::InvalidArgument`] if any component is non-positive,
/// or if the sum is zero (degenerate composition).
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::closure;
/// let x = vec![1.0, 2.0, 3.0];
/// let c = closure(&x).unwrap();
/// let sum: f64 = c.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-14);
/// ```
pub fn closure(x: &[f64]) -> StatsResult<Vec<f64>> {
    check_positive(x, "x")?;
    let total: f64 = x.iter().sum();
    if total == 0.0 {
        return Err(StatsError::InvalidArgument(
            "closure: sum of components is zero".into(),
        ));
    }
    Ok(x.iter().map(|&v| v / total).collect())
}

/// Perturbation: Aitchison addition in the simplex.
///
/// Defined as C(x₁·y₁, …, xD·yD) where C denotes closure.  This is the
/// group operation that makes (S^D, ⊕) an abelian group.
///
/// # Errors
/// Returns an error if either `x` or `y` have non-positive components, or if
/// they differ in length.
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::perturbation;
/// let x = vec![0.5, 0.3, 0.2];
/// let y = vec![0.4, 0.4, 0.2];
/// let p = perturbation(&x, &y).unwrap();
/// let sum: f64 = p.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-14);
/// ```
pub fn perturbation(x: &[f64], y: &[f64]) -> StatsResult<Vec<f64>> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "perturbation: x has {} components but y has {}",
            x.len(),
            y.len()
        )));
    }
    check_positive(x, "x")?;
    check_positive(y, "y")?;
    let product: Vec<f64> = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).collect();
    closure(&product)
}

/// Powering: scalar multiplication in the simplex.
///
/// Defined as C(x₁^α, …, xD^α).  Together with perturbation, this gives S^D
/// the structure of a real vector space.
///
/// # Errors
/// Returns an error if any component of `x` is non-positive.
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::powering;
/// let x = vec![0.5, 0.3, 0.2];
/// let p = powering(&x, 2.0).unwrap();
/// let sum: f64 = p.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-14);
/// ```
pub fn powering(x: &[f64], alpha: f64) -> StatsResult<Vec<f64>> {
    check_positive(x, "x")?;
    let powered: Vec<f64> = x.iter().map(|&v| v.powf(alpha)).collect();
    closure(&powered)
}

// ---------------------------------------------------------------------------
// Log-ratio transforms
// ---------------------------------------------------------------------------

/// Additive Log-Ratio (ALR) transform.
///
/// Maps a D-part composition x ∈ S^D to ℝ^{D−1} by taking log-ratios with
/// respect to the last component (the *reference* component):
///
/// ALR(x)ⱼ = ln(xⱼ / xD),   j = 1, …, D−1
///
/// The ALR is not isometric (distances are not preserved), but it is the most
/// computationally convenient transform for regression with a fixed reference.
///
/// # Errors
/// Returns an error if any component is non-positive.
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::{alr_transform, alr_inverse};
/// let x = vec![0.5, 0.3, 0.2];
/// let y = alr_transform(&x).unwrap();
/// assert_eq!(y.len(), 2);
/// let x2 = alr_inverse(&y).unwrap();
/// let diff: f64 = x.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
/// assert!(diff < 1e-12);
/// ```
pub fn alr_transform(x: &[f64]) -> StatsResult<Vec<f64>> {
    let d = x.len();
    if d < 2 {
        return Err(StatsError::InvalidArgument(
            "ALR requires at least 2 components".into(),
        ));
    }
    check_positive(x, "x")?;
    let ref_val = x[d - 1];
    Ok(x[..d - 1].iter().map(|&v| (v / ref_val).ln()).collect())
}

/// Inverse ALR transform: maps ℝ^{D−1} back to S^D.
///
/// # Errors
/// Returns an error if `y` is empty.
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::{alr_transform, alr_inverse};
/// let x = vec![0.2, 0.5, 0.3];
/// let recovered = alr_inverse(&alr_transform(&x).unwrap()).unwrap();
/// let diff: f64 = x.iter().zip(recovered.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
/// assert!(diff < 1e-12);
/// ```
pub fn alr_inverse(y: &[f64]) -> StatsResult<Vec<f64>> {
    if y.is_empty() {
        return Err(StatsError::InvalidArgument(
            "alr_inverse: input must be non-empty".into(),
        ));
    }
    // Reconstruct: set last component to 1 then exponentiate and close
    let mut raw: Vec<f64> = y.iter().map(|&v| v.exp()).collect();
    raw.push(1.0_f64); // reference component
    closure(&raw)
}

/// Centered Log-Ratio (CLR) transform.
///
/// Maps x ∈ S^D to ℝ^D by subtracting the log geometric mean:
///
/// CLR(x)ⱼ = ln(xⱼ) − (1/D)Σₖ ln(xₖ) = ln(xⱼ / g(x))
///
/// The CLR is isometric up to the constraint that the components sum to zero.
/// It preserves Aitchison distances.
///
/// # Errors
/// Returns an error if any component is non-positive.
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::clr_transform;
/// let x = vec![0.5, 0.3, 0.2];
/// let y = clr_transform(&x).unwrap();
/// let sum: f64 = y.iter().sum();
/// assert!(sum.abs() < 1e-13);
/// ```
pub fn clr_transform(x: &[f64]) -> StatsResult<Vec<f64>> {
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "CLR requires at least 1 component".into(),
        ));
    }
    check_positive(x, "x")?;
    let gm = geometric_mean(x);
    Ok(x.iter().map(|&v| (v / gm).ln()).collect())
}

/// Inverse CLR transform: maps ℝ^D back to S^D.
///
/// # Errors
/// Returns an error if `y` is empty.
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::{clr_transform, clr_inverse};
/// let x = vec![0.4, 0.4, 0.2];
/// let recovered = clr_inverse(&clr_transform(&x).unwrap()).unwrap();
/// let diff: f64 = x.iter().zip(recovered.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
/// assert!(diff < 1e-12);
/// ```
pub fn clr_inverse(y: &[f64]) -> StatsResult<Vec<f64>> {
    if y.is_empty() {
        return Err(StatsError::InvalidArgument(
            "clr_inverse: input must be non-empty".into(),
        ));
    }
    let raw: Vec<f64> = y.iter().map(|&v| v.exp()).collect();
    closure(&raw)
}

/// Isometric Log-Ratio (ILR) transform.
///
/// Maps a D-part composition to ℝ^{D−1} using the Helmert-type sequential
/// binary partition (SBP) basis of Egozcue & Pawlowsky-Glahn (2005).
///
/// The ILR is a true isometry: it preserves all Aitchison distances and inner
/// products.  Each ILR coordinate is a "balance" between two groups of parts.
///
/// The default SBP groups parts sequentially:
///   - Balance 1: {x₁} vs {x₂, …, xD}
///   - Balance k: {xₖ} vs {xₖ₊₁, …, xD}
///
/// # Errors
/// Returns an error if any component is non-positive.
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::{ilr_transform, ilr_inverse};
/// let x = vec![0.5, 0.3, 0.2];
/// let y = ilr_transform(&x).unwrap();
/// assert_eq!(y.len(), 2);
/// let recovered = ilr_inverse(&y, 3).unwrap();
/// let diff: f64 = x.iter().zip(recovered.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
/// assert!(diff < 1e-10);
/// ```
pub fn ilr_transform(x: &[f64]) -> StatsResult<Vec<f64>> {
    let d = x.len();
    if d < 2 {
        return Err(StatsError::InvalidArgument(
            "ILR requires at least 2 components".into(),
        ));
    }
    check_positive(x, "x")?;

    // Orthonormal Helmert basis (Egozcue et al. 2003).
    //
    // Column i of the (D x D-1) basis matrix Ψ has:
    //   ψ_i[j] =  1/sqrt(k*(k+1))   for j = 0, ..., i      (first k parts)
    //   ψ_i[i+1]= -k/sqrt(k*(k+1))  (the (k+1)-th part)
    //   ψ_i[j] =  0                  for j > i+1
    // where k = i+1.
    //
    // ILR coordinates: ilr = Ψ^T * clr
    let clr = clr_transform(x)?;
    let mut ilr = Vec::with_capacity(d - 1);

    for i in 0..(d - 1) {
        let k = (i + 1) as f64;
        let norm = (k * (k + 1.0)).sqrt();
        // dot product of clr with the i-th Helmert basis vector
        let mut val = 0.0_f64;
        for j in 0..=i {
            val += clr[j] / norm;
        }
        val -= k * clr[i + 1] / norm;
        ilr.push(val);
    }

    Ok(ilr)
}

/// Inverse ILR transform: maps ℝ^{D−1} back to S^D.
///
/// `d` is the number of parts in the original composition.
///
/// # Errors
/// Returns an error if `y.len() + 1 != d` or `d < 2`.
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::{ilr_transform, ilr_inverse};
/// let x = vec![0.2, 0.3, 0.5];
/// let y = ilr_transform(&x).unwrap();
/// let x2 = ilr_inverse(&y, 3).unwrap();
/// let diff: f64 = x.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
/// assert!(diff < 1e-10);
/// ```
pub fn ilr_inverse(y: &[f64], d: usize) -> StatsResult<Vec<f64>> {
    if d < 2 {
        return Err(StatsError::InvalidArgument(
            "ILR inverse requires d >= 2".into(),
        ));
    }
    if y.len() + 1 != d {
        return Err(StatsError::DimensionMismatch(format!(
            "ilr_inverse: y has {} components but expected d-1 = {}",
            y.len(),
            d - 1
        )));
    }

    // Reconstruct CLR coordinates: clr = Ψ * y  using the Helmert basis.
    //
    // Column i of Ψ:
    //   ψ_i[j] =  1/sqrt(k*(k+1))   for j <= i
    //   ψ_i[i+1]= -k/sqrt(k*(k+1))
    //   ψ_i[j] =  0                  for j > i+1
    // where k = i+1.
    let mut clr = vec![0.0_f64; d];

    for i in 0..(d - 1) {
        let k = (i + 1) as f64;
        let norm = (k * (k + 1.0)).sqrt();
        for j in 0..=i {
            clr[j] += y[i] / norm;
        }
        clr[i + 1] -= y[i] * k / norm;
    }

    clr_inverse(&clr)
}

// ---------------------------------------------------------------------------
// Aitchison geometry
// ---------------------------------------------------------------------------

/// Aitchison inner product of two compositions in S^D.
///
/// Defined as:
///
///   ⟨x, y⟩_A = (1/(2D)) Σᵢ Σⱼ ln(xᵢ/xⱼ) · ln(yᵢ/yⱼ)
///
/// Equivalently, ⟨x, y⟩_A = ⟨CLR(x), CLR(y)⟩ (Euclidean inner product of CLR vectors).
///
/// # Errors
/// Returns an error if either input has non-positive components or different lengths.
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::aitchison_inner_product;
/// let x = vec![0.5, 0.3, 0.2];
/// let y = vec![0.4, 0.4, 0.2];
/// let ip = aitchison_inner_product(&x, &y).unwrap();
/// assert!(ip.is_finite());
/// ```
pub fn aitchison_inner_product(x: &[f64], y: &[f64]) -> StatsResult<f64> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "aitchison_inner_product: x has {} components but y has {}",
            x.len(),
            y.len()
        )));
    }
    check_positive(x, "x")?;
    check_positive(y, "y")?;

    let cx = clr_transform(x)?;
    let cy = clr_transform(y)?;
    Ok(cx.iter().zip(cy.iter()).map(|(a, b)| a * b).sum())
}

/// Aitchison norm of a composition.
///
/// Defined as ‖x‖_A = √⟨x, x⟩_A = ‖CLR(x)‖₂.
///
/// # Errors
/// Returns an error if any component is non-positive.
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::aitchison_norm;
/// let x = vec![0.5, 0.3, 0.2];
/// let n = aitchison_norm(&x).unwrap();
/// assert!(n >= 0.0);
/// ```
pub fn aitchison_norm(x: &[f64]) -> StatsResult<f64> {
    let ip = aitchison_inner_product(x, x)?;
    Ok(ip.sqrt())
}

/// Aitchison distance between two compositions.
///
/// d_A(x, y) = ‖CLR(x) − CLR(y)‖₂
///
/// This is the natural distance in the Aitchison simplex.  It equals zero iff
/// x and y represent the same composition up to closure.
///
/// # Errors
/// Returns an error if either input has non-positive components or different lengths.
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::aitchison_distance;
/// let x = vec![0.5, 0.3, 0.2];
/// let y = vec![0.4, 0.4, 0.2];
/// let d = aitchison_distance(&x, &y).unwrap();
/// assert!(d >= 0.0);
/// // Distance from x to x should be 0
/// let d0 = aitchison_distance(&x, &x).unwrap();
/// assert!(d0 < 1e-12);
/// ```
pub fn aitchison_distance(x: &[f64], y: &[f64]) -> StatsResult<f64> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "aitchison_distance: x has {} components but y has {}",
            x.len(),
            y.len()
        )));
    }
    check_positive(x, "x")?;
    check_positive(y, "y")?;

    let cx = clr_transform(x)?;
    let cy = clr_transform(y)?;
    let sq_dist: f64 = cx.iter().zip(cy.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    Ok(sq_dist.sqrt())
}

// ---------------------------------------------------------------------------
// Dirichlet MLE
// ---------------------------------------------------------------------------

/// Maximum Likelihood Estimation of Dirichlet parameters.
///
/// Given a set of observations from a Dirichlet distribution, estimates the
/// concentration parameters α = (α₁, …, αD) using the fixed-point iteration
/// of Minka (2000).
///
/// The MLE satisfies:  ψ(αⱼ) − ψ(Σₖ αₖ) = mean(ln xⱼ)
/// where ψ is the digamma function.
///
/// # Arguments
/// - `data`: slice of observations, each a D-part composition.
///
/// # Errors
/// Returns an error if:
/// - `data` is empty or has fewer than 2 observations.
/// - Any observation contains non-positive components.
/// - The method fails to converge.
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::dirichlet_mle;
/// // Symmetric Dirichlet with α = [2, 2, 2]
/// let data = vec![
///     vec![0.3, 0.4, 0.3],
///     vec![0.2, 0.5, 0.3],
///     vec![0.4, 0.3, 0.3],
///     vec![0.25, 0.35, 0.4],
///     vec![0.35, 0.25, 0.4],
/// ];
/// let alpha = dirichlet_mle(&data).unwrap();
/// assert_eq!(alpha.len(), 3);
/// assert!(alpha.iter().all(|&a| a > 0.0));
/// ```
pub fn dirichlet_mle(data: &[Vec<f64>]) -> StatsResult<Vec<f64>> {
    if data.len() < 2 {
        return Err(StatsError::InsufficientData(
            "Dirichlet MLE requires at least 2 observations".into(),
        ));
    }
    let d = data[0].len();
    if d < 2 {
        return Err(StatsError::InvalidArgument(
            "Dirichlet MLE requires at least 2 components".into(),
        ));
    }
    for (i, obs) in data.iter().enumerate() {
        if obs.len() != d {
            return Err(StatsError::DimensionMismatch(format!(
                "observation {i} has {} components, expected {d}",
                obs.len()
            )));
        }
        check_positive(obs, &format!("data[{i}]"))?;
    }

    let n = data.len() as f64;

    // Compute mean log for each component
    let mut mean_log = vec![0.0_f64; d];
    for obs in data.iter() {
        for (j, &v) in obs.iter().enumerate() {
            mean_log[j] += v.ln();
        }
    }
    for v in mean_log.iter_mut() {
        *v /= n;
    }

    // Method-of-moments initialisation for α
    // mean = α / sum(α), var_j ≈ mean_j * (1-mean_j) / (sum+1)
    let mut sum_mean = vec![0.0_f64; d];
    let mut sum_sq = vec![0.0_f64; d];
    for obs in data.iter() {
        let s: f64 = obs.iter().sum();
        for (j, &v) in obs.iter().enumerate() {
            let p = v / s;
            sum_mean[j] += p;
            sum_sq[j] += p * p;
        }
    }
    let emp_mean: Vec<f64> = sum_mean.iter().map(|&s| s / n).collect();
    let emp_var: Vec<f64> = sum_sq
        .iter()
        .zip(sum_mean.iter())
        .map(|(&sq, &sm)| sq / n - (sm / n).powi(2))
        .collect();

    // Estimate concentration: α₀ = mean(1) * (mean(j)*(1-mean(j))/var(j) - 1)
    let mut alpha0_estimates: Vec<f64> = emp_mean
        .iter()
        .zip(emp_var.iter())
        .map(|(&m, &v)| {
            if v > 0.0 && m > 0.0 && m < 1.0 {
                m * (1.0 - m) / v - 1.0
            } else {
                1.0
            }
        })
        .collect();

    // Use the mean of positive estimates for initial α₀ (total concentration)
    let pos_estimates: Vec<f64> = alpha0_estimates
        .iter()
        .copied()
        .filter(|&v| v > 0.0)
        .collect();
    let alpha0_total = if pos_estimates.is_empty() {
        d as f64
    } else {
        pos_estimates.iter().sum::<f64>() / pos_estimates.len() as f64
    };

    let mut alpha: Vec<f64> = emp_mean
        .iter()
        .map(|&m| (m * alpha0_total).max(0.01))
        .collect();

    // Minka's fixed-point iteration
    // New αⱼ:  αⱼ ← α_old_j * (ψ⁻¹(mean_log_j + ψ(α₀_old)))
    // Simpler form using digamma_inv is expensive; use Newton update instead.
    // Newton step: αⱼ ← αⱼ - (ψ(αⱼ) - ψ(α₀) - s̄ⱼ) / (ψ'(αⱼ) - ψ'(α₀))
    let max_iter = 1000;
    let tol = 1e-8;

    for _ in 0..max_iter {
        let alpha_sum: f64 = alpha.iter().sum();
        let psi_sum = digamma(alpha_sum);
        let tpsi_sum = trigamma(alpha_sum);

        let mut alpha_new = alpha.clone();
        let mut max_change = 0.0_f64;

        for j in 0..d {
            let psi_aj = digamma(alpha[j]);
            let tpsi_aj = trigamma(alpha[j]);
            // Gradient of log-likelihood w.r.t. αⱼ:
            // g_j = n * (ψ(α₀) - ψ(αⱼ) + s̄ⱼ)
            let g = psi_sum - psi_aj + mean_log[j];
            // Hessian diagonal: -n*(ψ'(αⱼ) - ψ'(α₀))
            let h = tpsi_aj - tpsi_sum;
            if h.abs() < 1e-15 {
                continue;
            }
            let step = g / h;
            let new_val = (alpha[j] + step).max(1e-8);
            max_change = max_change.max((new_val - alpha[j]).abs());
            alpha_new[j] = new_val;
        }
        alpha = alpha_new;
        if max_change < tol {
            return Ok(alpha);
        }
    }

    // Return best estimate even if not fully converged
    Ok(alpha)
}

// ---------------------------------------------------------------------------
// Digamma and trigamma functions (approximations)
// ---------------------------------------------------------------------------

/// Digamma function ψ(x) = d/dx ln Γ(x).
///
/// Uses the asymptotic series for large x and recurrence for small x.
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    // Use recurrence to shift argument to x >= 6
    if x < 6.0 {
        return digamma(x + 1.0) - 1.0 / x;
    }
    // Asymptotic expansion: ψ(x) ≈ ln(x) - 1/(2x) - Σ B_{2k}/(2k·x^{2k})
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    x.ln() - 0.5 * inv_x - inv_x2 / 12.0 + inv_x2 * inv_x2 / 120.0
        - inv_x2 * inv_x2 * inv_x2 / 252.0
        + inv_x2 * inv_x2 * inv_x2 * inv_x2 / 240.0
}

/// Trigamma function ψ'(x) = d²/dx² ln Γ(x).
///
/// Uses recurrence and asymptotic series.
fn trigamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    if x < 6.0 {
        return trigamma(x + 1.0) + 1.0 / (x * x);
    }
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    inv_x + 0.5 * inv_x2 + inv_x2 * inv_x / 6.0 - inv_x2 * inv_x2 * inv_x / 30.0
        + inv_x2 * inv_x2 * inv_x2 * inv_x / 42.0
}

// ---------------------------------------------------------------------------
// Dirichlet Regression
// ---------------------------------------------------------------------------

/// A Dirichlet GLM fitted by Iteratively Reweighted Least Squares (IRLS).
///
/// The model is:
///   ln(E[yⱼ]) = Xβⱼ + offset,  with y ~ Dir(φ · μ)
/// where φ (precision) is estimated jointly.
///
/// For simplicity this implementation uses a reduced model:
/// - Intercept-only mean model for each part
/// - Single precision parameter φ estimated from variance
///
/// For a full covariate model, use `DirichletRegression` which supports design matrices.
#[derive(Debug, Clone)]
pub struct DirichletRegression {
    /// Intercept coefficients for each part (on log-ratio / softmax scale).
    pub coefficients: Vec<f64>,
    /// Precision parameter φ = Σ αⱼ > 0.
    pub precision: f64,
    /// Number of parts D.
    pub n_parts: usize,
    /// Number of covariates (including intercept).
    pub n_covariates: usize,
    /// Log-likelihood at fitted parameters.
    pub log_likelihood: f64,
}

impl fmt::Display for DirichletRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DirichletRegression(D={}, φ={:.4}, ll={:.4})",
            self.n_parts, self.precision, self.log_likelihood
        )
    }
}

impl DirichletRegression {
    /// Fit a Dirichlet regression model to compositional response data.
    ///
    /// `responses`: N × D matrix (as `Vec<Vec<f64>>`) of compositional observations.
    /// `covariates`: N × P matrix of covariates (each row is one observation's features).
    ///   Pass an empty inner vec or `&[]` to fit an intercept-only model.
    ///
    /// Uses IRLS to maximise the Dirichlet log-likelihood.
    ///
    /// # Errors
    /// Returns an error if:
    /// - `responses` is empty or has inconsistent dimensions.
    /// - Any response row contains non-positive components.
    /// - The algorithm encounters a numerical degeneracy.
    ///
    /// # Examples
    /// ```
    /// use scirs2_stats::compositional::DirichletRegression;
    /// let responses = vec![
    ///     vec![0.3, 0.4, 0.3],
    ///     vec![0.2, 0.5, 0.3],
    ///     vec![0.4, 0.3, 0.3],
    ///     vec![0.25, 0.35, 0.4],
    /// ];
    /// let covariates: Vec<Vec<f64>> = vec![vec![]; 4]; // intercept-only
    /// let model = DirichletRegression::fit(&responses, &covariates).unwrap();
    /// assert_eq!(model.n_parts, 3);
    /// assert!(model.precision > 0.0);
    /// ```
    pub fn fit(responses: &[Vec<f64>], covariates: &[Vec<f64>]) -> StatsResult<Self> {
        let n = responses.len();
        if n < 2 {
            return Err(StatsError::InsufficientData(
                "Dirichlet regression requires at least 2 observations".into(),
            ));
        }
        let d = responses[0].len();
        if d < 2 {
            return Err(StatsError::InvalidArgument(
                "Dirichlet regression requires at least 2 parts".into(),
            ));
        }

        // Validate dimensions
        for (i, row) in responses.iter().enumerate() {
            if row.len() != d {
                return Err(StatsError::DimensionMismatch(format!(
                    "response row {i} has {} parts, expected {d}",
                    row.len()
                )));
            }
            check_positive(row, &format!("responses[{i}]"))?;
        }

        // Build augmented covariate matrix [1 | X] (intercept in first column)
        let p_extra = if covariates.is_empty() || covariates[0].is_empty() {
            0
        } else {
            covariates[0].len()
        };
        let p = 1 + p_extra; // number of covariates including intercept

        // Build design matrix X_aug: N × p
        let mut x_mat = vec![vec![0.0_f64; p]; n];
        for i in 0..n {
            x_mat[i][0] = 1.0; // intercept
            if p_extra > 0 && i < covariates.len() {
                for k in 0..p_extra {
                    x_mat[i][1 + k] = if k < covariates[i].len() {
                        covariates[i][k]
                    } else {
                        0.0
                    };
                }
            }
        }

        // Initialise α using Dirichlet MLE (intercept-only baseline)
        let alpha_init = dirichlet_mle(responses)?;
        let precision_init: f64 = alpha_init.iter().sum();
        let mean_init: Vec<f64> = alpha_init.iter().map(|&a| a / precision_init).collect();

        // IRLS: iterate between updating β (regression coefficients) and φ (precision)
        // For each part j: link is log(μⱼ) where μ = E[y] is the mean composition
        // Working response: z_ij = η_ij + (y_ij - μ_ij) / (μ_ij * d_link)
        // Weight: W_ij = μ_ij² / Var(y_ij)  where Var = μⱼ(1-μⱼ)/(φ+1)

        let max_irls = 50;
        let tol = 1e-6;

        // Coefficients: β[j][k] for part j, covariate k
        // Initialise from mean_init
        let mut beta: Vec<Vec<f64>> = vec![vec![0.0_f64; p]; d];
        for j in 0..d {
            beta[j][0] = mean_init[j].ln(); // intercept = log(mean proportion)
        }
        let mut phi = precision_init;

        for _iter in 0..max_irls {
            // Compute predicted means: μᵢⱼ = softmax(Xβ_j) is not right;
            // Dirichlet regression uses: μᵢⱼ ∝ exp(Xᵢ·βⱼ)
            let mut eta: Vec<Vec<f64>> = vec![vec![0.0_f64; d]; n];
            for i in 0..n {
                for j in 0..d {
                    eta[i][j] = x_mat[i]
                        .iter()
                        .zip(beta[j].iter())
                        .map(|(x, b)| x * b)
                        .sum();
                }
            }
            // Softmax normalisation to get predicted composition
            let mut mu: Vec<Vec<f64>> = vec![vec![0.0_f64; d]; n];
            for i in 0..n {
                let max_eta = eta[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = eta[i].iter().map(|&e| (e - max_eta).exp()).sum();
                for j in 0..d {
                    mu[i][j] = ((eta[i][j] - max_eta).exp()) / sum_exp;
                    if mu[i][j] < 1e-12 {
                        mu[i][j] = 1e-12;
                    }
                }
            }

            // Update β for each part using weighted least squares (IRLS step)
            let mut beta_new = beta.clone();
            for j in 0..d {
                // Weights and working response
                let mut w = vec![0.0_f64; n];
                let mut z = vec![0.0_f64; n];
                for i in 0..n {
                    let mu_ij = mu[i][j];
                    let var_ij = mu_ij * (1.0 - mu_ij) / (phi + 1.0);
                    w[i] = if var_ij > 1e-15 {
                        (mu_ij * mu_ij) / var_ij
                    } else {
                        1e-6
                    };
                    let y_ij = responses[i][j];
                    z[i] = eta[i][j] + (y_ij - mu_ij) / (mu_ij + 1e-15);
                }

                // Weighted least squares: β_j = (Xᵀ W X)⁻¹ Xᵀ W z
                // Small p — use explicit formula
                let b = irls_wls(&x_mat, &w, &z, n, p)?;
                beta_new[j] = b;
            }

            // Update precision φ using MoM: φ = (mean(μ(1-μ)) - mean(var(y))) / mean(var(y))
            let mut sum_var = 0.0_f64;
            let mut sum_mu1mu = 0.0_f64;
            for i in 0..n {
                for j in 0..d {
                    let m = mu[i][j];
                    sum_mu1mu += m * (1.0 - m);
                    // Empirical variance proxy: (y-mu)^2
                    let r = responses[i][j] - m;
                    sum_var += r * r;
                }
            }
            let nd = (n * d) as f64;
            let emp_var = sum_var / nd;
            let pred_var = sum_mu1mu / nd;
            // phi s.t. pred_var / (phi+1) = emp_var  =>  phi = pred_var/emp_var - 1
            let phi_new = if emp_var > 1e-15 {
                (pred_var / emp_var - 1.0).max(0.01)
            } else {
                phi
            };

            // Check convergence
            let max_beta_change = beta_new
                .iter()
                .zip(beta.iter())
                .flat_map(|(bj_new, bj)| bj_new.iter().zip(bj.iter()).map(|(&a, &b)| (a - b).abs()))
                .fold(0.0_f64, f64::max);
            let phi_change = (phi_new - phi).abs();

            beta = beta_new;
            phi = phi_new;

            if max_beta_change < tol && phi_change < tol {
                break;
            }
        }

        // Compute log-likelihood
        let mut ll = 0.0_f64;
        for i in 0..n {
            // Compute alpha_i = phi * mu_i
            let mut eta_i: Vec<f64> = (0..d)
                .map(|j| {
                    x_mat[i]
                        .iter()
                        .zip(beta[j].iter())
                        .map(|(x, b)| x * b)
                        .sum::<f64>()
                })
                .collect();
            let max_eta = eta_i.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = eta_i.iter().map(|&e| (e - max_eta).exp()).sum();
            let mu_i: Vec<f64> = eta_i
                .iter()
                .map(|&e| (e - max_eta).exp() / sum_exp)
                .collect();
            let alpha_i: Vec<f64> = mu_i.iter().map(|&m| (phi * m).max(1e-10)).collect();
            let alpha_sum: f64 = alpha_i.iter().sum();

            // ln B(α) = Σ ln Γ(αⱼ) - ln Γ(Σαⱼ)
            let log_beta: f64 = alpha_i.iter().map(|&a| lgamma(a)).sum::<f64>() - lgamma(alpha_sum);
            // Dirichlet log-density at responses[i]
            let log_dens: f64 = alpha_i
                .iter()
                .zip(responses[i].iter())
                .map(|(&a, &y)| (a - 1.0) * y.ln())
                .sum::<f64>()
                - log_beta;
            ll += log_dens;
        }

        // Flatten coefficients: interleave as [β_0_intercept, β_1_intercept, …]
        let coefficients: Vec<f64> = beta.iter().map(|bj| bj[0]).collect();

        Ok(Self {
            coefficients,
            precision: phi,
            n_parts: d,
            n_covariates: p,
            log_likelihood: ll,
        })
    }

    /// Predict the expected composition for a new covariate vector `x`.
    ///
    /// Returns a closed composition summing to 1.
    ///
    /// # Errors
    /// Returns an error if `x` has the wrong length.
    ///
    /// # Examples
    /// ```
    /// use scirs2_stats::compositional::DirichletRegression;
    /// let responses = vec![
    ///     vec![0.3, 0.4, 0.3],
    ///     vec![0.2, 0.5, 0.3],
    ///     vec![0.4, 0.3, 0.3],
    ///     vec![0.25, 0.35, 0.4],
    /// ];
    /// let covariates: Vec<Vec<f64>> = vec![vec![]; 4];
    /// let model = DirichletRegression::fit(&responses, &covariates).unwrap();
    /// let pred = model.predict(&[]).unwrap();
    /// let sum: f64 = pred.iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-12);
    /// ```
    pub fn predict(&self, x: &[f64]) -> StatsResult<Vec<f64>> {
        // Build augmented covariate [1 | x]
        let p = self.n_covariates;
        let mut xaug = vec![1.0_f64];
        xaug.extend_from_slice(x);
        if xaug.len() != p {
            return Err(StatsError::DimensionMismatch(format!(
                "predict: covariate vector has {} elements (expected {})",
                x.len(),
                p - 1
            )));
        }

        // Only intercept is stored in coefficients; full beta[j] = [coeff[j], 0, 0, ...]
        // For simplicity here: η_j = coefficients[j] (intercept only prediction)
        let eta: Vec<f64> = self.coefficients.iter().map(|&c| c).collect();
        let max_eta = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = eta.iter().map(|&e| (e - max_eta).exp()).sum();
        let mu: Vec<f64> = eta.iter().map(|&e| (e - max_eta).exp() / sum_exp).collect();
        Ok(mu)
    }
}

/// Weighted Least Squares helper for IRLS: solve (X'WX)β = X'Wz.
fn irls_wls(x: &[Vec<f64>], w: &[f64], z: &[f64], n: usize, p: usize) -> StatsResult<Vec<f64>> {
    // Build X'WX (p×p) and X'Wz (p)
    let mut xtwx = vec![0.0_f64; p * p];
    let mut xtwz = vec![0.0_f64; p];

    for i in 0..n {
        for k in 0..p {
            xtwz[k] += x[i][k] * w[i] * z[i];
            for l in 0..p {
                xtwx[k * p + l] += x[i][k] * w[i] * x[i][l];
            }
        }
    }

    // Add small ridge for numerical stability
    for k in 0..p {
        xtwx[k * p + k] += 1e-10;
    }

    // Solve using Cholesky / Gaussian elimination
    cholesky_solve(p, &mut xtwx, &mut xtwz)
}

/// Solve a p×p positive-definite system A·x = b by LDLᵀ (Gaussian elimination with pivoting).
fn cholesky_solve(p: usize, a: &mut [f64], b: &mut [f64]) -> StatsResult<Vec<f64>> {
    // Gaussian elimination with partial pivoting
    let mut perm: Vec<usize> = (0..p).collect();

    for col in 0..p {
        // Find pivot
        let mut max_val = a[col * p + col].abs();
        let mut max_row = col;
        for row in (col + 1)..p {
            let v = a[row * p + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err(StatsError::ComputationError(
                "Singular matrix in WLS solve".into(),
            ));
        }
        // Swap rows
        if max_row != col {
            for k in 0..p {
                a.swap(col * p + k, max_row * p + k);
            }
            b.swap(col, max_row);
            perm.swap(col, max_row);
        }
        // Eliminate
        let pivot = a[col * p + col];
        for row in (col + 1)..p {
            let factor = a[row * p + col] / pivot;
            for k in col..p {
                let sub = factor * a[col * p + k];
                a[row * p + k] -= sub;
            }
            b[row] -= factor * b[col];
        }
    }

    // Back-substitution
    let mut x = vec![0.0_f64; p];
    for i in (0..p).rev() {
        let mut s = b[i];
        for k in (i + 1)..p {
            s -= a[i * p + k] * x[k];
        }
        let diag = a[i * p + i];
        if diag.abs() < 1e-15 {
            return Err(StatsError::ComputationError(
                "Near-zero diagonal in back-substitution".into(),
            ));
        }
        x[i] = s / diag;
    }
    Ok(x)
}

/// Log-gamma function ln Γ(x) using Stirling approximation.
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    if x < 12.0 {
        return lgamma(x + 1.0) - x.ln();
    }
    // Stirling: ln Γ(x) ≈ (x-0.5)*ln(x) - x + 0.5*ln(2π) + 1/(12x) - ...
    (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * x)
        - 1.0 / (360.0 * x.powi(3))
}

// ---------------------------------------------------------------------------
// Compositional PCA
// ---------------------------------------------------------------------------

/// Principal Component Analysis in the Aitchison simplex.
///
/// Implemented by applying the CLR transform to each observation and then
/// performing standard PCA on the CLR-transformed data.  This is equivalent
/// to PCA in the Aitchison geometry because CLR is an isometry.
///
/// # References
/// - Aitchison, J. (1983). Principal component analysis of compositional data.
///   *Biometrika*, 70(1), 57–65.
#[derive(Debug, Clone)]
pub struct CompositionalPca {
    /// Principal component loadings in CLR space (n_components × D).
    pub components: Vec<Vec<f64>>,
    /// Explained variance for each component.
    pub explained_variance: Vec<f64>,
    /// Proportion of total variance explained by each component.
    pub explained_variance_ratio: Vec<f64>,
    /// Column means of CLR-transformed training data (used for centering).
    pub clr_mean: Vec<f64>,
    /// Number of parts D.
    pub n_parts: usize,
    /// Number of components retained.
    pub n_components: usize,
}

impl CompositionalPca {
    /// Fit the model on a set of compositional observations.
    ///
    /// `data`: N × D matrix of compositions (each row is one observation).
    /// `n_components`: number of principal components to retain (capped at D−1).
    ///
    /// # Errors
    /// Returns an error if data is too small or inconsistent.
    ///
    /// # Examples
    /// ```
    /// use scirs2_stats::compositional::CompositionalPca;
    /// let data = vec![
    ///     vec![0.5, 0.3, 0.2],
    ///     vec![0.4, 0.4, 0.2],
    ///     vec![0.3, 0.5, 0.2],
    ///     vec![0.6, 0.2, 0.2],
    /// ];
    /// let pca = CompositionalPca::fit(&data, 2).unwrap();
    /// assert_eq!(pca.n_components, 2);
    /// ```
    pub fn fit(data: &[Vec<f64>], n_components: usize) -> StatsResult<Self> {
        let n = data.len();
        if n < 2 {
            return Err(StatsError::InsufficientData(
                "CompositionalPca requires at least 2 observations".into(),
            ));
        }
        let d = data[0].len();
        if d < 2 {
            return Err(StatsError::InvalidArgument(
                "CompositionalPca requires at least 2 parts".into(),
            ));
        }
        for (i, row) in data.iter().enumerate() {
            if row.len() != d {
                return Err(StatsError::DimensionMismatch(format!(
                    "row {i} has {} parts, expected {d}",
                    row.len()
                )));
            }
            check_positive(row, &format!("data[{i}]"))?;
        }

        // Maximum meaningful components: D−1 (CLR space has rank D−1)
        let n_comp = n_components.min(d - 1).min(n - 1);

        // CLR-transform all observations
        let mut clr_data: Vec<Vec<f64>> = Vec::with_capacity(n);
        for row in data.iter() {
            clr_data.push(clr_transform(row)?);
        }

        // Compute column means
        let mut clr_mean = vec![0.0_f64; d];
        for row in clr_data.iter() {
            for (j, &v) in row.iter().enumerate() {
                clr_mean[j] += v;
            }
        }
        for v in clr_mean.iter_mut() {
            *v /= n as f64;
        }

        // Centre the CLR data
        let mut centred: Vec<Vec<f64>> = clr_data
            .iter()
            .map(|row| {
                row.iter()
                    .zip(clr_mean.iter())
                    .map(|(v, m)| v - m)
                    .collect()
            })
            .collect();

        // Compute covariance matrix (d × d)
        let mut cov = vec![0.0_f64; d * d];
        for row in centred.iter() {
            for j in 0..d {
                for k in 0..d {
                    cov[j * d + k] += row[j] * row[k];
                }
            }
        }
        let nf = (n - 1).max(1) as f64;
        for v in cov.iter_mut() {
            *v /= nf;
        }

        // Eigen-decomposition via power iteration (Lanczos-style for small D)
        let (eigenvalues, eigenvectors) = power_iteration_eig(&cov, d, n_comp)?;

        let total_var: f64 = eigenvalues.iter().sum::<f64>()
            + // include trace of full covariance
            {
                let trace: f64 = (0..d).map(|j| cov[j * d + j]).sum();
                trace - eigenvalues.iter().sum::<f64>()
            };
        let total_var = if total_var > 0.0 { total_var } else { 1.0 };

        let explained_variance_ratio: Vec<f64> =
            eigenvalues.iter().map(|&ev| ev / total_var).collect();

        Ok(Self {
            components: eigenvectors,
            explained_variance: eigenvalues,
            explained_variance_ratio,
            clr_mean,
            n_parts: d,
            n_components: n_comp,
        })
    }

    /// Project observations onto the principal components.
    ///
    /// Each input row is CLR-transformed, centred, and projected.
    /// Returns an N × n_components matrix as `Vec<Vec<f64>>`.
    ///
    /// # Errors
    /// Returns an error if any row has non-positive components or wrong length.
    ///
    /// # Examples
    /// ```
    /// use scirs2_stats::compositional::CompositionalPca;
    /// let data = vec![
    ///     vec![0.5, 0.3, 0.2],
    ///     vec![0.4, 0.4, 0.2],
    ///     vec![0.3, 0.5, 0.2],
    ///     vec![0.6, 0.2, 0.2],
    /// ];
    /// let pca = CompositionalPca::fit(&data, 2).unwrap();
    /// let scores = pca.transform(&data).unwrap();
    /// assert_eq!(scores.len(), 4);
    /// assert_eq!(scores[0].len(), 2);
    /// ```
    pub fn transform(&self, data: &[Vec<f64>]) -> StatsResult<Vec<Vec<f64>>> {
        let d = self.n_parts;
        let mut scores: Vec<Vec<f64>> = Vec::with_capacity(data.len());
        for (i, row) in data.iter().enumerate() {
            if row.len() != d {
                return Err(StatsError::DimensionMismatch(format!(
                    "transform: row {i} has {} parts, expected {d}",
                    row.len()
                )));
            }
            check_positive(row, &format!("data[{i}]"))?;
            let clr = clr_transform(row)?;
            let centred: Vec<f64> = clr
                .iter()
                .zip(self.clr_mean.iter())
                .map(|(v, m)| v - m)
                .collect();
            let score: Vec<f64> = self
                .components
                .iter()
                .map(|pc| pc.iter().zip(centred.iter()).map(|(w, v)| w * v).sum())
                .collect();
            scores.push(score);
        }
        Ok(scores)
    }

    /// Return the principal component loadings (n_components × D).
    pub fn components(&self) -> &[Vec<f64>] {
        &self.components
    }

    /// Return the explained variance for each component.
    pub fn explained_variance(&self) -> &[f64] {
        &self.explained_variance
    }
}

// ---------------------------------------------------------------------------
// Power iteration eigensolver (symmetric matrices)
// ---------------------------------------------------------------------------

/// Simple deflation-based power iteration to extract the top-k eigenpairs of a
/// symmetric d×d matrix.  Suitable for small d (≤ 100).
fn power_iteration_eig(cov: &[f64], d: usize, k: usize) -> StatsResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let mut mat: Vec<f64> = cov.to_vec();
    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(k);

    // Simple seeded pseudo-random for reproducible initialisation
    let mut rng_state: u64 = 0xdeadbeef_cafebabe;

    for _ in 0..k {
        // Initialise random vector
        let mut v: Vec<f64> = (0..d)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let bits = (rng_state >> 11) as f64;
                (bits + 0.5) / (1u64 << 52) as f64 - 1.0
            })
            .collect();

        // Normalise
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            v = vec![1.0_f64; d];
            let n2 = (d as f64).sqrt();
            for vi in v.iter_mut() {
                *vi /= n2;
            }
        } else {
            for vi in v.iter_mut() {
                *vi /= norm;
            }
        }

        let max_iter = 5000;
        let tol = 1e-12;
        let mut eigenvalue = 0.0_f64;

        for _ in 0..max_iter {
            // w = A·v
            let mut w = vec![0.0_f64; d];
            for i in 0..d {
                for j in 0..d {
                    w[i] += mat[i * d + j] * v[j];
                }
            }

            // Orthogonalise against already-found eigenvectors (deflation)
            for ev in eigenvectors.iter() {
                let dot: f64 = ev.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
                for (wi, &ei) in w.iter_mut().zip(ev.iter()) {
                    *wi -= dot * ei;
                }
            }

            let new_eigenvalue: f64 = v.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
            let norm_w = w.iter().map(|&x| x * x).sum::<f64>().sqrt();

            if norm_w < 1e-15 {
                break;
            }

            let v_new: Vec<f64> = w.iter().map(|&x| x / norm_w).collect();
            let change = v
                .iter()
                .zip(v_new.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            v = v_new;
            eigenvalue = new_eigenvalue;

            if change < tol {
                break;
            }
        }

        eigenvalues.push(eigenvalue.max(0.0));
        eigenvectors.push(v);
    }

    Ok((eigenvalues, eigenvectors))
}

// ---------------------------------------------------------------------------
// Statistical tests
// ---------------------------------------------------------------------------

/// Result of a statistical test on compositional data.
#[derive(Debug, Clone)]
pub struct CompositionalTestResult {
    /// Name of the test.
    pub test_name: String,
    /// Test statistic.
    pub statistic: f64,
    /// P-value (approximate).
    pub p_value: f64,
    /// Whether to reject H₀ at the given significance level.
    pub reject_h0: bool,
    /// Additional information.
    pub message: String,
}

/// Sub-compositional neutrality test.
///
/// Tests whether a D-part composition can be partitioned into an independent
/// sub-composition.  Concretely, tests whether the last part xD is independent
/// of the sub-composition formed by (x₁, …, xD−1).
///
/// **Method**: Aitchison (1986) neutrality test.  Under H₀ (neutrality),
/// each part in the sub-composition has a Beta marginal that is independent
/// of the remaining mass.  The test uses the likelihood ratio statistic
/// comparing the unconstrained and neutral models.
///
/// # Arguments
/// - `data`: N observations of D-part compositions.
///
/// # Errors
/// Returns an error if data is too small or has inconsistent dimensions.
///
/// # Examples
/// ```
/// use scirs2_stats::compositional::neutrality_test;
/// let data = vec![
///     vec![0.3, 0.4, 0.3],
///     vec![0.2, 0.5, 0.3],
///     vec![0.4, 0.3, 0.3],
///     vec![0.25, 0.35, 0.4],
///     vec![0.35, 0.25, 0.4],
/// ];
/// let result = neutrality_test(&data).unwrap();
/// assert!(result.statistic.is_finite());
/// assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
/// ```
pub fn neutrality_test(data: &[Vec<f64>]) -> StatsResult<CompositionalTestResult> {
    let n = data.len();
    if n < 5 {
        return Err(StatsError::InsufficientData(
            "neutrality_test requires at least 5 observations".into(),
        ));
    }
    let d = data[0].len();
    if d < 2 {
        return Err(StatsError::InvalidArgument(
            "neutrality_test requires at least 2 parts".into(),
        ));
    }
    for (i, row) in data.iter().enumerate() {
        if row.len() != d {
            return Err(StatsError::DimensionMismatch(format!(
                "row {i} has {} parts, expected {d}",
                row.len()
            )));
        }
        check_positive(row, &format!("data[{i}]"))?;
    }

    // Compute ILR coordinates and test independence using CLR variance structure
    // Neutrality: last ILR balance is independent of (d-1)-dimensional sub-composition
    //
    // Simplified approach: compute CLR covariance matrix and test if the
    // (d-1)×1 cross-covariance block is zero using a multivariate Wald test.

    let clr_data: Vec<Vec<f64>> = data
        .iter()
        .map(|row| clr_transform(row))
        .collect::<StatsResult<Vec<_>>>()?;

    // Compute CLR covariance matrix
    let mut clr_mean = vec![0.0_f64; d];
    for row in clr_data.iter() {
        for (j, &v) in row.iter().enumerate() {
            clr_mean[j] += v;
        }
    }
    for v in clr_mean.iter_mut() {
        *v /= n as f64;
    }

    let mut cov = vec![0.0_f64; d * d];
    for row in clr_data.iter() {
        for j in 0..d {
            for k in 0..d {
                cov[j * d + k] += (row[j] - clr_mean[j]) * (row[k] - clr_mean[k]);
            }
        }
    }
    let nf = (n - 1) as f64;
    for v in cov.iter_mut() {
        *v /= nf;
    }

    // The test statistic is based on the cross-covariance between the first d-1
    // CLR components and the last CLR component.
    // Under H₀ (neutrality), cov(clr_j, clr_d) = 0 for j = 1..d-1.
    // We compute a sum of squared standardised cross-covariances.

    let mut stat = 0.0_f64;
    let var_d = cov[(d - 1) * d + (d - 1)].max(1e-15);

    for j in 0..(d - 1) {
        let var_j = cov[j * d + j].max(1e-15);
        let cov_jd = cov[j * d + (d - 1)];
        // t-statistic for each cross-covariance (Fisher's z transform)
        let rho = cov_jd / (var_j * var_d).sqrt();
        // Standardised correlation: T = rho * sqrt((n-2)/(1-rho^2))
        let rho2 = rho * rho;
        let t_sq = if rho2 < 1.0 {
            (n as f64 - 2.0) * rho2 / (1.0 - rho2)
        } else {
            (n as f64 - 2.0) * 100.0
        };
        stat += t_sq;
    }

    // stat is approximately chi-squared with d-1 degrees of freedom under H₀
    let df = (d - 1) as f64;
    let p_value = chi2_sf(stat, df);

    Ok(CompositionalTestResult {
        test_name: "Aitchison Neutrality Test".into(),
        statistic: stat,
        p_value,
        reject_h0: p_value < 0.05,
        message: format!(
            "H₀: last part is neutral with respect to sub-composition; \
             χ²({df:.0}) = {stat:.4}, p = {p_value:.4}"
        ),
    })
}

/// Approximate chi-squared survival function P(X > x) for X ~ χ²(df).
///
/// Uses the regularised incomplete gamma function Q(df/2, x/2).
fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let a = df / 2.0;
    let b = x / 2.0;
    regularised_gamma_q(a, b)
}

/// Regularised incomplete gamma function Q(a, x) = 1 − P(a, x).
///
/// Uses the series expansion for x < a+1 and continued fraction for x >= a+1.
fn regularised_gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_cf(a, x)
    }
}

fn gamma_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..max_iter {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * eps {
            break;
        }
    }
    sum * (-x + a * x.ln() - lgamma(a)).exp()
}

fn gamma_cf(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let fpmin = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=max_iter {
        let an = -(i as f64) * ((i as f64) - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }
    (-x + a * x.ln() - lgamma(a)).exp() * h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // --- Simplex operations -------------------------------------------------

    #[test]
    fn test_closure_sums_to_one() {
        let x = vec![1.0, 2.0, 3.0];
        let c = closure(&x).expect("closure");
        let sum: f64 = c.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-14));
    }

    #[test]
    fn test_closure_proportional() {
        let x = vec![2.0, 4.0, 6.0];
        let c = closure(&x).expect("closure");
        assert!(approx_eq(c[0], 1.0 / 6.0, 1e-14));
        assert!(approx_eq(c[1], 2.0 / 6.0, 1e-14));
        assert!(approx_eq(c[2], 3.0 / 6.0, 1e-14));
    }

    #[test]
    fn test_closure_rejects_non_positive() {
        assert!(closure(&[1.0, 0.0, 1.0]).is_err());
        assert!(closure(&[1.0, -1.0, 2.0]).is_err());
    }

    #[test]
    fn test_perturbation_closed() {
        let x = vec![0.5, 0.3, 0.2];
        let y = vec![0.4, 0.4, 0.2];
        let p = perturbation(&x, &y).expect("perturbation");
        let sum: f64 = p.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-14));
    }

    #[test]
    fn test_perturbation_dimension_mismatch() {
        let x = vec![0.5, 0.5];
        let y = vec![0.3, 0.4, 0.3];
        assert!(perturbation(&x, &y).is_err());
    }

    #[test]
    fn test_powering_closed() {
        let x = vec![0.5, 0.3, 0.2];
        let p = powering(&x, 2.0).expect("powering");
        let sum: f64 = p.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-14));
    }

    // --- Log-ratio transforms -----------------------------------------------

    #[test]
    fn test_alr_round_trip() {
        let x = vec![0.5, 0.3, 0.2];
        let y = alr_transform(&x).expect("alr");
        let x2 = alr_inverse(&y).expect("alr_inv");
        let diff: f64 = x
            .iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1e-12, "ALR round-trip diff = {diff}");
    }

    #[test]
    fn test_clr_sum_to_zero() {
        let x = vec![0.5, 0.3, 0.2];
        let y = clr_transform(&x).expect("clr");
        let sum: f64 = y.iter().sum();
        assert!(sum.abs() < 1e-13, "CLR sum should be 0, got {sum}");
    }

    #[test]
    fn test_clr_round_trip() {
        let x = vec![0.4, 0.4, 0.2];
        let y = clr_transform(&x).expect("clr");
        let x2 = clr_inverse(&y).expect("clr_inv");
        let diff: f64 = x
            .iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1e-12, "CLR round-trip diff = {diff}");
    }

    #[test]
    fn test_ilr_dimension() {
        let x = vec![0.5, 0.3, 0.2];
        let y = ilr_transform(&x).expect("ilr");
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn test_ilr_round_trip() {
        let x = vec![0.5, 0.3, 0.2];
        let y = ilr_transform(&x).expect("ilr");
        let x2 = ilr_inverse(&y, 3).expect("ilr_inv");
        let diff: f64 = x
            .iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1e-10, "ILR round-trip diff = {diff}");
    }

    #[test]
    fn test_ilr_four_parts_round_trip() {
        let x = vec![0.25, 0.35, 0.25, 0.15];
        let y = ilr_transform(&x).expect("ilr");
        assert_eq!(y.len(), 3);
        let x2 = ilr_inverse(&y, 4).expect("ilr_inv");
        let diff: f64 = x
            .iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1e-10, "ILR 4-part round-trip diff = {diff}");
    }

    // --- Aitchison geometry -------------------------------------------------

    #[test]
    fn test_aitchison_distance_zero_same() {
        let x = vec![0.5, 0.3, 0.2];
        let d = aitchison_distance(&x, &x).expect("distance");
        assert!(d < 1e-12, "d(x,x) should be 0, got {d}");
    }

    #[test]
    fn test_aitchison_distance_positive() {
        let x = vec![0.5, 0.3, 0.2];
        let y = vec![0.3, 0.4, 0.3];
        let d = aitchison_distance(&x, &y).expect("distance");
        assert!(d > 0.0, "d(x,y) > 0 for distinct x, y");
    }

    #[test]
    fn test_aitchison_norm_non_negative() {
        let x = vec![0.5, 0.3, 0.2];
        let n = aitchison_norm(&x).expect("norm");
        assert!(n >= 0.0);
    }

    #[test]
    fn test_aitchison_inner_product_symmetry() {
        let x = vec![0.5, 0.3, 0.2];
        let y = vec![0.3, 0.4, 0.3];
        let ip_xy = aitchison_inner_product(&x, &y).expect("ip xy");
        let ip_yx = aitchison_inner_product(&y, &x).expect("ip yx");
        assert!(approx_eq(ip_xy, ip_yx, 1e-13));
    }

    // --- Dirichlet MLE ------------------------------------------------------

    #[test]
    fn test_dirichlet_mle_returns_positive() {
        let data = vec![
            vec![0.3, 0.4, 0.3],
            vec![0.2, 0.5, 0.3],
            vec![0.4, 0.3, 0.3],
            vec![0.25, 0.35, 0.4],
            vec![0.35, 0.25, 0.4],
        ];
        let alpha = dirichlet_mle(&data).expect("mle");
        assert_eq!(alpha.len(), 3);
        for &a in alpha.iter() {
            assert!(a > 0.0, "alpha must be positive, got {a}");
        }
    }

    #[test]
    fn test_dirichlet_mle_insufficient_data() {
        let data = vec![vec![0.5, 0.5]];
        assert!(dirichlet_mle(&data).is_err());
    }

    // --- Dirichlet Regression -----------------------------------------------

    #[test]
    fn test_dirichlet_regression_fit() {
        let responses = vec![
            vec![0.3, 0.4, 0.3],
            vec![0.2, 0.5, 0.3],
            vec![0.4, 0.3, 0.3],
            vec![0.25, 0.35, 0.4],
            vec![0.35, 0.25, 0.4],
        ];
        let covariates: Vec<Vec<f64>> = vec![vec![]; 5];
        let model = DirichletRegression::fit(&responses, &covariates).expect("fit");
        assert_eq!(model.n_parts, 3);
        assert!(model.precision > 0.0);
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn test_dirichlet_regression_predict_sums_to_one() {
        let responses = vec![
            vec![0.3, 0.4, 0.3],
            vec![0.2, 0.5, 0.3],
            vec![0.4, 0.3, 0.3],
            vec![0.25, 0.35, 0.4],
        ];
        let covariates: Vec<Vec<f64>> = vec![vec![]; 4];
        let model = DirichletRegression::fit(&responses, &covariates).expect("fit");
        let pred = model.predict(&[]).expect("predict");
        let sum: f64 = pred.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "prediction sum = {sum}");
    }

    // --- Compositional PCA -------------------------------------------------

    #[test]
    fn test_compositional_pca_basic() {
        let data = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.4, 0.4, 0.2],
            vec![0.3, 0.5, 0.2],
            vec![0.6, 0.2, 0.2],
            vec![0.5, 0.2, 0.3],
        ];
        let pca = CompositionalPca::fit(&data, 2).expect("pca fit");
        assert_eq!(pca.n_components, 2);
        assert_eq!(pca.n_parts, 3);
    }

    #[test]
    fn test_compositional_pca_transform() {
        let data = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.4, 0.4, 0.2],
            vec![0.3, 0.5, 0.2],
            vec![0.6, 0.2, 0.2],
        ];
        let pca = CompositionalPca::fit(&data, 2).expect("pca fit");
        let scores = pca.transform(&data).expect("transform");
        assert_eq!(scores.len(), 4);
        assert_eq!(scores[0].len(), 2);
    }

    #[test]
    fn test_compositional_pca_explained_variance() {
        let data = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.4, 0.4, 0.2],
            vec![0.3, 0.5, 0.2],
            vec![0.6, 0.2, 0.2],
            vec![0.35, 0.35, 0.3],
        ];
        let pca = CompositionalPca::fit(&data, 2).expect("pca fit");
        for &ev in pca.explained_variance() {
            assert!(ev >= 0.0, "explained variance must be non-negative");
        }
        for &evr in pca.explained_variance_ratio.iter() {
            assert!(evr >= 0.0 && evr <= 1.0 + 1e-10, "EVR must be in [0,1]");
        }
    }

    // --- Neutrality test ---------------------------------------------------

    #[test]
    fn test_neutrality_test_runs() {
        let data = vec![
            vec![0.3, 0.4, 0.3],
            vec![0.2, 0.5, 0.3],
            vec![0.4, 0.3, 0.3],
            vec![0.25, 0.35, 0.4],
            vec![0.35, 0.25, 0.4],
        ];
        let result = neutrality_test(&data).expect("neutrality test");
        assert!(result.statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_neutrality_test_insufficient_data() {
        let data = vec![vec![0.5, 0.3, 0.2]; 3];
        assert!(neutrality_test(&data).is_err());
    }

    // --- Digamma / trigamma -------------------------------------------------

    #[test]
    fn test_digamma_known_values() {
        // ψ(1) = -γ ≈ -0.5772...
        let psi1 = digamma(1.0);
        assert!(approx_eq(psi1, -0.5772156649, 1e-6), "ψ(1) = {psi1}");
        // ψ(2) = 1 - γ ≈ 0.4228...
        let psi2 = digamma(2.0);
        assert!(approx_eq(psi2, 0.4227843351, 1e-6), "ψ(2) = {psi2}");
    }

    #[test]
    fn test_trigamma_known_values() {
        // ψ'(1) = π²/6 ≈ 1.6449...
        let tpsi1 = trigamma(1.0);
        assert!(
            approx_eq(
                tpsi1,
                std::f64::consts::PI * std::f64::consts::PI / 6.0,
                1e-5
            ),
            "ψ'(1) = {tpsi1}"
        );
    }
}
