//! Markov chain analysis and stochastic matrix tools
//!
//! This module provides a comprehensive set of tools for analysing discrete-time
//! Markov chains represented by row-stochastic transition matrices P where each
//! row sums to 1 and all entries are non-negative.
//!
//! ## Functions provided
//!
//! - [`stationary_distribution`] – Power-iteration to find the unique stationary
//!   distribution π satisfying π = π P, ‖π‖₁ = 1.
//! - [`is_stochastic`] / [`is_stochastic_f`] – Validate stochasticity.
//! - [`mean_first_passage_time`] – Matrix M where M[i,j] is the expected steps to
//!   reach state j starting from state i.
//! - [`fundamental_matrix`] – Z = (I − P + Π)⁻¹ where Π = 1·π^T.
//! - [`mixing_time`] – Estimate the ε-mixing time from the spectral gap.
//! - [`analyze_absorbing_chain`] – Detect absorbing and transient states.
//! - [`absorption_probabilities`] – B matrix of transient→absorbing absorption
//!   probabilities.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::solve::solve_multiple;
use crate::validation::validate_squarematrix;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean norm of a 1-D array.
fn norm2_1d<F: Float + Sum>(v: &Array1<F>) -> F {
    v.iter()
        .map(|&x| x * x)
        .fold(F::zero(), |a, b| a + b)
        .sqrt()
}

/// L1 norm of a 1-D array (sum of absolute values).
fn norm1_1d<F: Float + Sum>(v: &Array1<F>) -> F {
    v.iter().map(|&x| x.abs()).fold(F::zero(), |a, b| a + b)
}

/// Compute π^T · P (left multiplication of row vector by matrix).
fn left_multiply_row<F>(pi: &Array1<F>, p: &ArrayView2<F>) -> Array1<F>
where
    F: Float + NumAssign + ScalarOperand,
{
    let n = p.ncols();
    let mut result = Array1::zeros(n);
    for j in 0..n {
        let mut sum = F::zero();
        for i in 0..n {
            sum += pi[i] * p[[i, j]];
        }
        result[j] = sum;
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Stochasticity validation
// ─────────────────────────────────────────────────────────────────────────────

/// Check whether a matrix is row-stochastic (rows sum to 1, all entries ≥ 0).
///
/// Uses an absolute tolerance `tol` around 1.0 for row sums and 0.0 for
/// non-negativity.
///
/// # Examples
///
/// ```
/// use scirs2_linalg::matrix_functions::is_stochastic;
/// use scirs2_core::ndarray::array;
///
/// let p = array![[0.7_f64, 0.3], [0.4, 0.6]];
/// assert!(is_stochastic(&p.view(), 1e-10));
/// ```
pub fn is_stochastic<F>(p: &ArrayView2<F>, tol: F) -> bool
where
    F: Float + Sum + NumAssign + ScalarOperand,
{
    if p.nrows() != p.ncols() {
        return false;
    }
    let n = p.nrows();
    for i in 0..n {
        // Non-negativity
        for j in 0..n {
            if p[[i, j]] < -tol {
                return false;
            }
        }
        // Row sum ≈ 1
        let row_sum: F = (0..n).map(|j| p[[i, j]]).fold(F::zero(), |a, b| a + b);
        if (row_sum - F::one()).abs() > tol {
            return false;
        }
    }
    true
}

/// Convenience overload for `f64` with a hard-coded tolerance check.
///
/// # Examples
///
/// ```
/// use scirs2_linalg::matrix_functions::is_stochastic_f64;
/// use scirs2_core::ndarray::array;
///
/// let p = array![[0.5_f64, 0.5], [0.3, 0.7]];
/// assert!(is_stochastic_f64(&p.view(), 1e-10));
/// ```
pub fn is_stochastic_f64(p: &ArrayView2<f64>, tol: f64) -> bool {
    is_stochastic(p, tol)
}

// ─────────────────────────────────────────────────────────────────────────────
// Stationary distribution
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the stationary distribution of an ergodic Markov chain by power
/// iteration.
///
/// Iterates π^(k+1) = π^(k) · P until ‖π^(k+1) − π^(k)‖₂ < `tol`, then
/// normalises so that ‖π‖₁ = 1.
///
/// # Arguments
///
/// * `transition_matrix` – Row-stochastic square transition matrix P.
/// * `tol` – Convergence tolerance for successive π iterates.
/// * `max_iter` – Maximum number of power iterations.
///
/// # Returns
///
/// Stationary distribution vector π with ∑ πᵢ = 1.
///
/// # Errors
///
/// Returns [`LinalgError::ConvergenceError`] if the iteration does not converge,
/// or [`LinalgError::InvalidInputError`] if P is not square or not stochastic.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::stationary_distribution;
///
/// let p = array![[0.7_f64, 0.3], [0.4, 0.6]];
/// let pi = stationary_distribution(&p.view(), 1e-12, 10_000)
///     .expect("Must converge for ergodic chain");
/// // For this chain: π = [4/7, 3/7] ≈ [0.5714, 0.4286]
/// assert!((pi[0] - 4.0/7.0).abs() < 1e-8);
/// ```
pub fn stationary_distribution<F>(
    transition_matrix: &ArrayView2<F>,
    tol: F,
    max_iter: usize,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static + std::fmt::Display,
{
    validate_squarematrix(transition_matrix, "Stationary distribution")?;

    let n = transition_matrix.nrows();

    // Validate stochasticity
    if !is_stochastic(
        transition_matrix,
        F::from(1e-6).unwrap_or(F::epsilon() * F::from(1000.0).unwrap_or(F::one())),
    ) {
        return Err(LinalgError::InvalidInputError(
            "Transition matrix must be row-stochastic (rows sum to 1, all entries non-negative)"
                .to_string(),
        ));
    }

    if max_iter == 0 {
        return Err(LinalgError::InvalidInputError(
            "max_iter must be positive".to_string(),
        ));
    }

    // Initialise uniform distribution
    let inv_n = F::one()
        / F::from(n).ok_or_else(|| {
            LinalgError::ComputationError("Cannot convert n to float".to_string())
        })?;
    let mut pi: Array1<F> = Array1::from_elem(n, inv_n);

    for _ in 0..max_iter {
        let pi_new = left_multiply_row(&pi, transition_matrix);

        // Check convergence
        let diff: Array1<F> = Array1::from_iter(pi_new.iter().zip(pi.iter()).map(|(&a, &b)| a - b));
        let change = norm2_1d(&diff);

        pi = pi_new;

        // Renormalise to counteract floating-point drift
        let sum: F = pi.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum > F::epsilon() {
            pi.mapv_inplace(|v| v / sum);
        }

        if change < tol {
            return Ok(pi);
        }
    }

    Err(LinalgError::ConvergenceError(format!(
        "Stationary distribution did not converge in {max_iter} iterations"
    )))
}

// ─────────────────────────────────────────────────────────────────────────────
// Fundamental matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the fundamental matrix Z = (I − P + Π)⁻¹ where Π = 1·π^T.
///
/// The fundamental matrix encodes the expected number of times the chain
/// passes through each state. It is used to derive the mean first-passage
/// time and variance of absorption times.
///
/// # Arguments
///
/// * `transition_matrix` – Row-stochastic square transition matrix P.
///
/// # Returns
///
/// The n×n fundamental matrix Z.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::fundamental_matrix;
///
/// let p = array![[0.7_f64, 0.3], [0.4, 0.6]];
/// let z = fundamental_matrix(&p.view()).expect("Must succeed for ergodic chain");
/// assert_eq!(z.nrows(), 2);
/// ```
pub fn fundamental_matrix<F>(transition_matrix: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static + std::fmt::Display,
{
    validate_squarematrix(transition_matrix, "Fundamental matrix")?;

    let n = transition_matrix.nrows();

    // First compute stationary distribution
    let pi = stationary_distribution(
        transition_matrix,
        F::from(1e-12).unwrap_or(F::epsilon()),
        50_000,
    )?;

    // Build A = I - P + Π where Π[i,j] = pi[j]
    let mut a = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let delta_ij = if i == j { F::one() } else { F::zero() };
            a[[i, j]] = delta_ij - transition_matrix[[i, j]] + pi[j];
        }
    }

    // Solve A · Z = I  ⟺  Z = A⁻¹
    // Build identity right-hand side
    let identity = Array2::eye(n);

    let z = solve_multiple(&a.view(), &identity.view(), None)?;

    Ok(z)
}

// ─────────────────────────────────────────────────────────────────────────────
// Mean first passage time
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the mean first passage time (MFPT) matrix from the fundamental
/// matrix.
///
/// M[i,j] = expected number of steps to reach state j for the first time
/// starting from state i. For i = j this is the mean recurrence time 1/πⱼ.
///
/// Formula: M[i,j] = (Z[j,j] − Z[i,j]) / π[j].
///
/// # Arguments
///
/// * `transition_matrix` – Row-stochastic square transition matrix P.
///
/// # Returns
///
/// The n×n mean first passage time matrix M.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::mean_first_passage_time;
///
/// let p = array![[0.7_f64, 0.3], [0.4, 0.6]];
/// let mfpt = mean_first_passage_time(&p.view())
///     .expect("Must succeed for ergodic chain");
/// assert_eq!(mfpt.nrows(), 2);
/// // Mean recurrence time for state 0: 1/pi[0] = 7/4 ≈ 1.75
/// assert!((mfpt[[0, 0]] - 7.0/4.0).abs() < 1e-6);
/// ```
pub fn mean_first_passage_time<F>(transition_matrix: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static + std::fmt::Display,
{
    validate_squarematrix(transition_matrix, "Mean first passage time")?;

    let n = transition_matrix.nrows();

    let pi = stationary_distribution(
        transition_matrix,
        F::from(1e-12).unwrap_or(F::epsilon()),
        50_000,
    )?;
    let z = fundamental_matrix(transition_matrix)?;

    let mut m = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if pi[j] <= F::epsilon() {
                m[[i, j]] = F::infinity();
            } else {
                m[[i, j]] = (z[[j, j]] - z[[i, j]]) / pi[j];
            }
        }
    }

    // Diagonal: mean recurrence time = 1/π[j]
    for j in 0..n {
        if pi[j] > F::epsilon() {
            m[[j, j]] = F::one() / pi[j];
        } else {
            m[[j, j]] = F::infinity();
        }
    }

    Ok(m)
}

// ─────────────────────────────────────────────────────────────────────────────
// Mixing time
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the ε-mixing time of an ergodic Markov chain from its spectral gap.
///
/// The mixing time is estimated using the bound:
/// τ(ε) ≈ ⌈ln(1/ε) / (1 − |λ₂|)⌉
///
/// where λ₂ is the second-largest eigenvalue modulus. A larger spectral gap
/// (1 − |λ₂|) means faster mixing.
///
/// # Arguments
///
/// * `transition_matrix` – Row-stochastic square transition matrix P.
/// * `epsilon` – Target total variation distance from stationarity.
///
/// # Returns
///
/// Estimated number of steps to achieve ε-close convergence to stationary
/// distribution.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::mixing_time;
///
/// let p = array![[0.7_f64, 0.3], [0.4, 0.6]];
/// let t_mix = mixing_time(&p.view(), 0.01).expect("Must succeed");
/// assert!(t_mix >= 1);
/// ```
pub fn mixing_time<F>(transition_matrix: &ArrayView2<F>, epsilon: F) -> LinalgResult<usize>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static + std::fmt::Display,
{
    validate_squarematrix(transition_matrix, "Mixing time")?;

    if epsilon <= F::zero() || epsilon >= F::one() {
        return Err(LinalgError::InvalidInputError(
            "epsilon must be in (0, 1)".to_string(),
        ));
    }

    let n = transition_matrix.nrows();

    // Compute eigenvalues of P to find the spectral gap.
    // We use power iteration on P^T to find the dominant eigenvalue (= 1),
    // then deflate and find the sub-dominant one.
    //
    // For ergodic chains the eigenvalue 1 is simple; all others have |λ| < 1.

    // Build matrix P^T P (for use in eigenvalue iteration)
    // Instead use direct power iteration on (P - π 1^T) to find λ₂.

    let pi_res = stationary_distribution(
        transition_matrix,
        F::from(1e-12).unwrap_or(F::epsilon()),
        50_000,
    );
    let pi =
        pi_res.unwrap_or_else(|_| Array1::from_elem(n, F::one() / F::from(n).unwrap_or(F::one())));

    // Build P_centered = P - Π  (Π[i,j] = π[j])
    // λ₂ of P_centered = λ₂ of P (shifted to remove eigenvalue 1 component)
    // Actually we estimate |λ₂| via power iteration on (P − 1·π^T).

    // Compute ||P^k v - π||₂ → 0 at rate |λ₂|^k
    // Use inverse iteration proxy: iterate v_{k+1} = P^T v_k / ‖ ‖ and track
    // convergence of Rayleigh quotient.

    // Simple approach: compute the spectral gap using the second-largest
    // singular value of (P - Π).
    let mut p_centered = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            p_centered[[i, j]] = transition_matrix[[i, j]] - pi[j];
        }
    }

    // Power iteration on p_centered^T to find |λ₂|
    use crate::decomposition::svd;
    let (_, s, _) = svd(&p_centered.view(), false, None)?;

    let lambda2 = if s.len() > 1 { s[0] } else { F::zero() };

    // spectral gap
    let gap = F::one() - lambda2;

    if gap <= F::epsilon() {
        return Err(LinalgError::ComputationError(
            "Spectral gap is zero or negative; chain may not be ergodic".to_string(),
        ));
    }

    // τ(ε) ≈ ceil(ln(1/ε) / gap)
    let ln_inv_eps = (-epsilon.ln()).max(F::zero());
    let t_f = ln_inv_eps / gap;

    let t = t_f.ceil();
    let t_usize = t.to_usize().unwrap_or(usize::MAX).max(1);

    Ok(t_usize)
}

// ─────────────────────────────────────────────────────────────────────────────
// Absorbing chain analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Describes the canonical partition of an absorbing Markov chain.
#[derive(Debug, Clone)]
pub struct AbsorbingMarkovChain {
    /// Indices of absorbing states (states where P[i,i] = 1).
    pub absorbing_states: Vec<usize>,
    /// Indices of transient states (all non-absorbing states).
    pub transient_states: Vec<usize>,
}

/// Identify absorbing and transient states in a Markov chain.
///
/// A state i is *absorbing* if P[i,i] = 1 (equivalently P[i,j] = 0 for j ≠ i).
/// All other states are *transient*.
///
/// # Arguments
///
/// * `transition_matrix` – Row-stochastic square transition matrix P.
///
/// # Returns
///
/// An [`AbsorbingMarkovChain`] struct listing absorbing and transient states.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::analyze_absorbing_chain;
///
/// // State 0 and 2 are absorbing; state 1 is transient
/// let p = array![
///     [1.0_f64, 0.0, 0.0],
///     [0.3,     0.4, 0.3],
///     [0.0,     0.0, 1.0],
/// ];
/// let chain = analyze_absorbing_chain(&p.view()).expect("Must succeed");
/// assert_eq!(chain.absorbing_states, vec![0, 2]);
/// assert_eq!(chain.transient_states, vec![1]);
/// ```
pub fn analyze_absorbing_chain<F>(
    transition_matrix: &ArrayView2<F>,
) -> LinalgResult<AbsorbingMarkovChain>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static + std::fmt::Display,
{
    validate_squarematrix(transition_matrix, "Absorbing chain analysis")?;

    let n = transition_matrix.nrows();

    // Validate stochasticity
    let tol_stoch = F::from(1e-6).unwrap_or(F::epsilon());
    for i in 0..n {
        let row_sum: F = (0..n)
            .map(|j| transition_matrix[[i, j]])
            .fold(F::zero(), |a, b| a + b);
        if (row_sum - F::one()).abs() > tol_stoch {
            return Err(LinalgError::InvalidInputError(format!(
                "Row {i} of transition matrix does not sum to 1 (sum = {row_sum})"
            )));
        }
    }

    let tol = F::from(1e-10).unwrap_or(F::epsilon());

    let mut absorbing = Vec::new();
    let mut transient = Vec::new();

    for i in 0..n {
        // Check if this is an absorbing state: P[i,i] ≈ 1
        if (transition_matrix[[i, i]] - F::one()).abs() <= tol {
            // Verify all off-diagonal entries are zero
            let off_diag_sum: F = (0..n)
                .filter(|&j| j != i)
                .map(|j| transition_matrix[[i, j]].abs())
                .fold(F::zero(), |a, b| a + b);
            if off_diag_sum <= tol {
                absorbing.push(i);
                continue;
            }
        }
        transient.push(i);
    }

    Ok(AbsorbingMarkovChain {
        absorbing_states: absorbing,
        transient_states: transient,
    })
}

/// Compute absorption probabilities for an absorbing Markov chain.
///
/// Returns the matrix B where B[i,j] is the probability that a chain starting
/// in transient state `transient_states[i]` is eventually absorbed by
/// absorbing state `absorbing_states[j]`.
///
/// Formula: B = (I − Q)⁻¹ R  where Q is the transient-to-transient sub-matrix
/// and R is the transient-to-absorbing sub-matrix.
///
/// # Arguments
///
/// * `transition_matrix` – Row-stochastic square transition matrix P.
/// * `absorbing_chain` – Pre-computed [`AbsorbingMarkovChain`] (from
///   [`analyze_absorbing_chain`]).
///
/// # Returns
///
/// Matrix B of shape `(num_transient × num_absorbing)`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::{analyze_absorbing_chain, absorption_probabilities};
///
/// let p = array![
///     [1.0_f64, 0.0, 0.0],
///     [0.3,     0.4, 0.3],
///     [0.0,     0.0, 1.0],
/// ];
/// let chain = analyze_absorbing_chain(&p.view()).expect("Must succeed");
/// let b = absorption_probabilities(&p.view(), &chain).expect("Must succeed");
/// // State 1 is absorbed by 0 with prob 0.5 and by 2 with prob 0.5
/// assert!((b[[0, 0]] - 0.5).abs() < 1e-8);
/// assert!((b[[0, 1]] - 0.5).abs() < 1e-8);
/// ```
pub fn absorption_probabilities<F>(
    transition_matrix: &ArrayView2<F>,
    absorbing_chain: &AbsorbingMarkovChain,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static + std::fmt::Display,
{
    validate_squarematrix(transition_matrix, "Absorption probabilities")?;

    let t_states = &absorbing_chain.transient_states;
    let a_states = &absorbing_chain.absorbing_states;

    if t_states.is_empty() {
        return Err(LinalgError::InvalidInputError(
            "No transient states found in the chain".to_string(),
        ));
    }
    if a_states.is_empty() {
        return Err(LinalgError::InvalidInputError(
            "No absorbing states found in the chain".to_string(),
        ));
    }

    let nt = t_states.len();
    let na = a_states.len();

    // Q: transient → transient sub-matrix
    let mut q = Array2::zeros((nt, nt));
    for (i, &ti) in t_states.iter().enumerate() {
        for (j, &tj) in t_states.iter().enumerate() {
            q[[i, j]] = transition_matrix[[ti, tj]];
        }
    }

    // R: transient → absorbing sub-matrix
    let mut r = Array2::zeros((nt, na));
    for (i, &ti) in t_states.iter().enumerate() {
        for (j, &aj) in a_states.iter().enumerate() {
            r[[i, j]] = transition_matrix[[ti, aj]];
        }
    }

    // Build (I − Q)
    let mut i_minus_q: Array2<F> = Array2::eye(nt);
    for i in 0..nt {
        for j in 0..nt {
            i_minus_q[[i, j]] -= q[[i, j]];
        }
    }

    // B = (I − Q)⁻¹ · R
    let b = solve_multiple(&i_minus_q.view(), &r.view(), None)?;

    Ok(b)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    // ── helpers ──

    fn two_state_chain() -> Array2<f64> {
        // P = [[0.7, 0.3], [0.4, 0.6]]
        // Stationary: π = [4/7, 3/7]
        array![[0.7_f64, 0.3], [0.4, 0.6]]
    }

    fn three_state_ergodic() -> Array2<f64> {
        array![[0.5_f64, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.2, 0.5]]
    }

    fn absorbing_chain_3state() -> Array2<f64> {
        // States 0 and 2 are absorbing; state 1 is transient
        array![[1.0_f64, 0.0, 0.0], [0.3, 0.4, 0.3], [0.0, 0.0, 1.0],]
    }

    // ════════════════════════════════════════════════════
    // is_stochastic
    // ════════════════════════════════════════════════════

    #[test]
    fn test_is_stochastic_valid() {
        let p = two_state_chain();
        assert!(is_stochastic(&p.view(), 1e-10));
    }

    #[test]
    fn test_is_stochastic_invalid_row_sum() {
        let p = array![[0.7_f64, 0.4], [0.4, 0.6]]; // row 0 sums to 1.1
        assert!(!is_stochastic(&p.view(), 1e-10));
    }

    #[test]
    fn test_is_stochastic_negative_entry() {
        let p = array![[0.7_f64, 0.3], [-0.1, 1.1]];
        assert!(!is_stochastic(&p.view(), 1e-10));
    }

    #[test]
    fn test_is_stochastic_non_square() {
        let p = array![[0.5_f64, 0.5, 0.0], [0.3, 0.4, 0.3]];
        assert!(!is_stochastic(&p.view(), 1e-10));
    }

    // ════════════════════════════════════════════════════
    // stationary_distribution
    // ════════════════════════════════════════════════════

    #[test]
    fn test_stationary_distribution_two_state() {
        let p = two_state_chain();
        let pi = stationary_distribution(&p.view(), 1e-12, 10_000).expect("Must converge");
        assert_relative_eq!(pi[0], 4.0 / 7.0, epsilon = 1e-8);
        assert_relative_eq!(pi[1], 3.0 / 7.0, epsilon = 1e-8);
    }

    #[test]
    fn test_stationary_distribution_three_state() {
        let p = three_state_ergodic();
        let pi = stationary_distribution(&p.view(), 1e-12, 10_000).expect("Must converge");
        // Sum must be 1
        let sum: f64 = pi.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        // All entries must be positive
        for &v in pi.iter() {
            assert!(v > 0.0);
        }
        // Must satisfy π = π P
        let pi_p: Array1<f64> = {
            let n = 3;
            let mut r = Array1::zeros(n);
            for j in 0..n {
                r[j] = (0..n).map(|i| pi[i] * p[[i, j]]).sum::<f64>();
            }
            r
        };
        for i in 0..3 {
            assert_relative_eq!(pi[i], pi_p[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_stationary_distribution_uniform() {
        // Doubly stochastic → uniform stationary distribution
        let p = array![[0.5_f64, 0.5], [0.5, 0.5]];
        let pi = stationary_distribution(&p.view(), 1e-12, 10_000).expect("Must converge");
        assert_relative_eq!(pi[0], 0.5, epsilon = 1e-8);
        assert_relative_eq!(pi[1], 0.5, epsilon = 1e-8);
    }

    // ════════════════════════════════════════════════════
    // fundamental_matrix
    // ════════════════════════════════════════════════════

    #[test]
    fn test_fundamental_matrix_two_state() {
        let p = two_state_chain();
        let z = fundamental_matrix(&p.view()).expect("Must succeed");
        assert_eq!(z.nrows(), 2);
        assert_eq!(z.ncols(), 2);
    }

    #[test]
    fn test_fundamental_matrix_identity_row() {
        // Z · (I - P + Π) = I (by construction)
        let p = two_state_chain();
        let z = fundamental_matrix(&p.view()).expect("Must succeed");
        let pi = stationary_distribution(&p.view(), 1e-12, 10_000).expect("failed to create pi");
        let n = 2;
        // Build A = I - P + Π
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = (if i == j { 1.0 } else { 0.0 }) - p[[i, j]] + pi[j];
            }
        }
        // Z · A should be ≈ I
        let mut za = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    za[[i, j]] += z[[i, k]] * a[[k, j]];
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(za[[i, j]], expected, epsilon = 1e-8);
            }
        }
    }

    // ════════════════════════════════════════════════════
    // mean_first_passage_time
    // ════════════════════════════════════════════════════

    #[test]
    fn test_mfpt_diagonal_is_recurrence_time() {
        let p = two_state_chain();
        let mfpt = mean_first_passage_time(&p.view()).expect("Must succeed");
        let pi = stationary_distribution(&p.view(), 1e-12, 10_000).expect("failed to create pi");
        // Mean recurrence time M[i,i] = 1/π[i]
        assert_relative_eq!(mfpt[[0, 0]], 1.0 / pi[0], epsilon = 1e-6);
        assert_relative_eq!(mfpt[[1, 1]], 1.0 / pi[1], epsilon = 1e-6);
    }

    #[test]
    fn test_mfpt_positive_entries() {
        let p = two_state_chain();
        let mfpt = mean_first_passage_time(&p.view()).expect("Must succeed");
        for i in 0..2 {
            for j in 0..2 {
                assert!(mfpt[[i, j]] > 0.0, "MFPT entry [{i},{j}] must be positive");
            }
        }
    }

    // ════════════════════════════════════════════════════
    // mixing_time
    // ════════════════════════════════════════════════════

    #[test]
    fn test_mixing_time_positive() {
        let p = two_state_chain();
        let t = mixing_time(&p.view(), 0.01).expect("Must succeed");
        assert!(t >= 1);
    }

    #[test]
    fn test_mixing_time_faster_for_large_gap() {
        // Identity permutation matrix has spectral gap = 0 → skip
        // Instead: compare two chains with different gaps
        // Fast chain: nearly uniform
        let p_fast = array![[0.5_f64, 0.5], [0.5, 0.5]];
        // Slow chain: highly persistent
        let p_slow = array![[0.99_f64, 0.01], [0.01, 0.99]];
        let t_fast = mixing_time(&p_fast.view(), 0.05).expect("Must succeed");
        let t_slow = mixing_time(&p_slow.view(), 0.05).expect("Must succeed");
        assert!(
            t_slow > t_fast,
            "Slower chain should need more steps: {t_slow} vs {t_fast}"
        );
    }

    #[test]
    fn test_mixing_time_invalid_epsilon() {
        let p = two_state_chain();
        assert!(mixing_time(&p.view(), 0.0).is_err());
        assert!(mixing_time(&p.view(), 1.0).is_err());
        assert!(mixing_time(&p.view(), -0.1).is_err());
    }

    // ════════════════════════════════════════════════════
    // analyze_absorbing_chain
    // ════════════════════════════════════════════════════

    #[test]
    fn test_analyze_absorbing_chain_basic() {
        let p = absorbing_chain_3state();
        let chain = analyze_absorbing_chain(&p.view()).expect("Must succeed");
        assert_eq!(chain.absorbing_states, vec![0, 2]);
        assert_eq!(chain.transient_states, vec![1]);
    }

    #[test]
    fn test_analyze_absorbing_chain_no_absorbing() {
        let p = two_state_chain();
        let chain = analyze_absorbing_chain(&p.view()).expect("Must succeed");
        assert!(chain.absorbing_states.is_empty());
        assert_eq!(chain.transient_states.len(), 2);
    }

    #[test]
    fn test_analyze_absorbing_chain_all_absorbing() {
        // 3×3 identity: all absorbing
        let p: Array2<f64> = Array2::eye(3);
        let chain = analyze_absorbing_chain(&p.view()).expect("Must succeed");
        assert_eq!(chain.absorbing_states.len(), 3);
        assert!(chain.transient_states.is_empty());
    }

    // ════════════════════════════════════════════════════
    // absorption_probabilities
    // ════════════════════════════════════════════════════

    #[test]
    fn test_absorption_probs_symmetric() {
        // Due to symmetry, transient state 1 should be absorbed equally
        let p = absorbing_chain_3state();
        let chain = analyze_absorbing_chain(&p.view()).expect("Must succeed");
        let b = absorption_probabilities(&p.view(), &chain).expect("Must succeed");
        // b[0, 0] = P(absorbed by state 0 | start at state 1)
        // By symmetry (R[0]=R[1]=0.3): each absorbing state gets 0.5
        assert_relative_eq!(b[[0, 0]], 0.5, epsilon = 1e-8);
        assert_relative_eq!(b[[0, 1]], 0.5, epsilon = 1e-8);
    }

    #[test]
    fn test_absorption_probs_row_sum_to_one() {
        // Each row of B must sum to 1 (eventual certain absorption)
        let p = absorbing_chain_3state();
        let chain = analyze_absorbing_chain(&p.view()).expect("Must succeed");
        let b = absorption_probabilities(&p.view(), &chain).expect("Must succeed");
        for i in 0..b.nrows() {
            let row_sum: f64 = b.row(i).iter().sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_absorption_probs_shape() {
        let p = absorbing_chain_3state();
        let chain = analyze_absorbing_chain(&p.view()).expect("Must succeed");
        let b = absorption_probabilities(&p.view(), &chain).expect("Must succeed");
        // 1 transient × 2 absorbing
        assert_eq!(b.nrows(), chain.transient_states.len());
        assert_eq!(b.ncols(), chain.absorbing_states.len());
    }

    #[test]
    fn test_absorption_probs_asymmetric() {
        // State 1 transitions: to 0 with 0.7, to 2 with 0.3 (no self-loop)
        let p = array![[1.0_f64, 0.0, 0.0], [0.7, 0.0, 0.3], [0.0, 0.0, 1.0],];
        let chain = analyze_absorbing_chain(&p.view()).expect("Must succeed");
        let b = absorption_probabilities(&p.view(), &chain).expect("Must succeed");
        assert_relative_eq!(b[[0, 0]], 0.7, epsilon = 1e-8);
        assert_relative_eq!(b[[0, 1]], 0.3, epsilon = 1e-8);
    }
}
