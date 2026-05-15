//! ANOVA functional decomposition for sparse grids.
//!
//! Classical ANOVA decomposition:
//! ```text
//! f(x) = f₀ + Σᵢ fᵢ(xᵢ) + Σᵢ<ⱼ fᵢⱼ(xᵢ,xⱼ) + …
//! ```
//! where each term is defined by orthogonal projection (marginal integration).
//!
//! Also provides Sobol' sensitivity indices: Sᵢ = Var(fᵢ) / Var(f).
//!
//! # References
//! - Saltelli et al., "Global Sensitivity Analysis: The Primer" (2008)
//! - Sobol', "Sensitivity estimates for nonlinear mathematical models" (1993)

use scirs2_core::ndarray::Array2;

/// Configuration for ANOVA decomposition.
#[derive(Debug, Clone)]
pub struct AnovaConfig {
    /// Maximum interaction order (default 2). Orders above this are truncated.
    pub max_order: usize,
    /// Number of Gauss-Legendre quadrature points per dimension (capped at 5 internally).
    pub n_quad_points: usize,
}

impl Default for AnovaConfig {
    fn default() -> Self {
        Self {
            max_order: 2,
            n_quad_points: 10,
        }
    }
}

/// Result of an ANOVA functional decomposition.
#[derive(Debug, Clone)]
pub struct AnovaDecomposition {
    /// f₀ = E\[f\], the constant (mean) term.
    pub mean: f64,
    /// Variance of each main-effect term fᵢ: Var(fᵢ) for dim i.
    pub main_effects: Vec<f64>,
    /// Variance of each pairwise interaction term fᵢⱼ: `interaction_effects[[i,j]]`.
    /// Symmetric, diagonal is zero.
    pub interaction_effects: Array2<f64>,
    /// First-order Sobol' index Sᵢ = Var(fᵢ) / Var(f).
    pub sobol_indices: Vec<f64>,
    /// Total Sobol' index Tᵢ = 1 - Var(E[f|x_{−i}]) / Var(f).
    pub total_sobol_indices: Vec<f64>,
    /// Total variance Var(f).
    pub total_variance: f64,
}

// ─── Gauss-Legendre nodes and weights on [0, 1] ───────────────────────────────

/// Returns (nodes, weights) for Gauss-Legendre quadrature mapped to [0, 1].
///
/// Supports n ∈ {1, 2, 3, 4, 5}.  For any other `n` the function silently
/// falls back to n = 5 (accuracy up to degree-9 polynomials).
fn gauss_legendre_01(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.5], vec![1.0]),
        2 => (
            vec![0.211_324_865_405_187_1, 0.788_675_134_594_812_9],
            vec![0.5, 0.5],
        ),
        3 => (
            vec![0.112_701_665_379_258_31, 0.5, 0.887_298_334_620_741_69],
            vec![
                0.277_777_777_777_777_79,
                0.444_444_444_444_444_42,
                0.277_777_777_777_777_79,
            ],
        ),
        4 => (
            vec![
                0.069_431_844_202_973_713,
                0.330_009_478_207_571_87,
                0.669_990_521_792_428_13,
                0.930_568_155_797_026_29,
            ],
            vec![
                0.173_927_422_568_726_93,
                0.326_072_577_431_273_07,
                0.326_072_577_431_273_07,
                0.173_927_422_568_726_93,
            ],
        ),
        // n == 5 or any unsupported n
        _ => (
            vec![
                0.046_910_077_781_151_27,
                0.230_765_345_653_031_44,
                0.5,
                0.769_234_654_346_968_56,
                0.953_089_922_218_848_73,
            ],
            vec![
                0.118_463_442_528_094_54,
                0.239_314_335_249_683_23,
                0.284_444_444_444_444_44,
                0.239_314_335_249_683_23,
                0.118_463_442_528_094_54,
            ],
        ),
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Compute an ANOVA functional decomposition of a black-box function
/// `f: [0,1]^d → ℝ`.
///
/// # Arguments
/// * `f_fn` – The function to decompose.
/// * `d`    – Number of input dimensions.
/// * `config` – Quadrature and truncation settings.
///
/// # Returns
/// [`AnovaDecomposition`] with mean, main-effect variances, pairwise interaction
/// variances, and first-order / total Sobol' indices.
pub fn anova_decompose<F>(
    f_fn: &F,
    d: usize,
    config: &AnovaConfig,
) -> Result<AnovaDecomposition, AnovaError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    if d == 0 {
        return Err(AnovaError::ZeroDimension);
    }

    // Cap quadrature points to the table size (max 5).
    let q = config.n_quad_points.clamp(1, 5);
    let (nodes, weights) = gauss_legendre_01(q);

    // For large d reduce the effective grid to avoid n^d explosion.
    let effective_q = if d <= 3 { q } else { 3.min(q) };
    let nodes = &nodes[..effective_q];
    let weights = &weights[..effective_q];

    // f₀ = E[f]
    let mean = compute_mean(f_fn, d, nodes, weights);

    // E[f²] → total variance
    let mean_sq = compute_mean_of_square(f_fn, d, nodes, weights);
    let total_variance = mean_sq - mean * mean;

    // Main-effect variances: Var(fᵢ) for each i
    let main_effects: Vec<f64> = (0..d)
        .map(|i| compute_main_effect_variance(f_fn, d, i, mean, nodes, weights))
        .collect();

    // First-order Sobol' indices
    let sobol_indices: Vec<f64> = if total_variance > 1e-12 {
        main_effects.iter().map(|v| v / total_variance).collect()
    } else {
        vec![0.0; d]
    };

    // Total Sobol' indices: Tᵢ = 1 − Var(E[f | x_{−i}]) / Var(f)
    let total_sobol_indices: Vec<f64> = (0..d)
        .map(|i| {
            let complement_var =
                compute_complement_effect_variance(f_fn, d, i, mean, nodes, weights);
            if total_variance > 1e-12 {
                1.0 - complement_var / total_variance
            } else {
                0.0
            }
        })
        .collect();

    // Pairwise interaction variances (max_order >= 2 and d ≤ 8 to stay tractable)
    let mut interaction_effects = Array2::zeros((d, d));
    if config.max_order >= 2 && d <= 8 {
        for i in 0..d {
            for j in (i + 1)..d {
                let v_ij = compute_interaction_variance(
                    f_fn,
                    d,
                    i,
                    j,
                    mean,
                    main_effects[i],
                    main_effects[j],
                    nodes,
                    weights,
                );
                interaction_effects[[i, j]] = v_ij;
                interaction_effects[[j, i]] = v_ij;
            }
        }
    }

    Ok(AnovaDecomposition {
        mean,
        main_effects,
        interaction_effects,
        sobol_indices,
        total_sobol_indices,
        total_variance,
    })
}

// ─── Quadrature helpers ───────────────────────────────────────────────────────

/// E[f] via full tensor-product Gauss-Legendre quadrature.
fn compute_mean<F: Fn(&[f64]) -> f64>(f_fn: &F, d: usize, nodes: &[f64], weights: &[f64]) -> f64 {
    let nq = nodes.len();
    let total_pts = nq.pow(d as u32);
    let mut x = vec![0.0f64; d];
    let mut total = 0.0f64;
    for idx in 0..total_pts {
        let mut w = 1.0f64;
        let mut tmp = idx;
        for xi in x.iter_mut().take(d) {
            let k = tmp % nq;
            tmp /= nq;
            *xi = nodes[k];
            w *= weights[k];
        }
        total += w * f_fn(&x);
    }
    total
}

/// E[f²] via full tensor-product Gauss-Legendre quadrature.
fn compute_mean_of_square<F: Fn(&[f64]) -> f64>(
    f_fn: &F,
    d: usize,
    nodes: &[f64],
    weights: &[f64],
) -> f64 {
    let nq = nodes.len();
    let total_pts = nq.pow(d as u32);
    let mut x = vec![0.0f64; d];
    let mut total = 0.0f64;
    for idx in 0..total_pts {
        let mut w = 1.0f64;
        let mut tmp = idx;
        for xi in x.iter_mut().take(d) {
            let k = tmp % nq;
            tmp /= nq;
            *xi = nodes[k];
            w *= weights[k];
        }
        let val = f_fn(&x);
        total += w * val * val;
    }
    total
}

/// Var(fᵢ) = E[(E[f | xᵢ] − f₀)²].
fn compute_main_effect_variance<F: Fn(&[f64]) -> f64>(
    f_fn: &F,
    d: usize,
    dim_i: usize,
    mean: f64,
    nodes: &[f64],
    weights: &[f64],
) -> f64 {
    nodes
        .iter()
        .zip(weights.iter())
        .map(|(&xi, &wi)| {
            let cond_mean = compute_conditional_mean(f_fn, d, dim_i, xi, nodes, weights);
            wi * (cond_mean - mean).powi(2)
        })
        .sum()
}

/// E[f | x_{fixed_dim} = fixed_val] — integrate out all other dimensions.
fn compute_conditional_mean<F: Fn(&[f64]) -> f64>(
    f_fn: &F,
    d: usize,
    fixed_dim: usize,
    fixed_val: f64,
    nodes: &[f64],
    weights: &[f64],
) -> f64 {
    let nq = nodes.len();
    let d_rest = d - 1;
    let total_pts = nq.pow(d_rest as u32);
    let mut x = vec![0.0f64; d];
    x[fixed_dim] = fixed_val;

    let other_dims: Vec<usize> = (0..d).filter(|&dim| dim != fixed_dim).collect();

    let mut total = 0.0f64;
    for idx in 0..total_pts {
        let mut w = 1.0f64;
        let mut tmp = idx;
        for &dim in &other_dims {
            let k = tmp % nq;
            tmp /= nq;
            x[dim] = nodes[k];
            w *= weights[k];
        }
        total += w * f_fn(&x);
    }
    total
}

/// Var(E[f | x_{−i}]) — variance of the conditional expectation with dim i
/// integrated out.  Used for total Sobol' indices.
fn compute_complement_effect_variance<F: Fn(&[f64]) -> f64>(
    f_fn: &F,
    d: usize,
    dim_i: usize,
    mean: f64,
    nodes: &[f64],
    weights: &[f64],
) -> f64 {
    let nq = nodes.len();
    let d_rest = d - 1;
    let total_complement_pts = nq.pow(d_rest as u32);

    let other_dims: Vec<usize> = (0..d).filter(|&dim| dim != dim_i).collect();
    let mut x = vec![0.0f64; d];
    let mut variance = 0.0f64;

    for idx in 0..total_complement_pts {
        let mut w_complement = 1.0f64;
        let mut tmp = idx;
        for &dim in &other_dims {
            let k = tmp % nq;
            tmp /= nq;
            x[dim] = nodes[k];
            w_complement *= weights[k];
        }
        // Integrate out dim_i
        let cond_mean_complement: f64 = nodes
            .iter()
            .zip(weights.iter())
            .map(|(&xi, &wi)| {
                x[dim_i] = xi;
                wi * f_fn(&x)
            })
            .sum();
        variance += w_complement * (cond_mean_complement - mean).powi(2);
    }
    variance
}

/// Pure second-order interaction variance:
/// Var(fᵢⱼ) = Var(E[f | xᵢ, xⱼ]) − Var(fᵢ) − Var(fⱼ), clamped to ≥ 0.
fn compute_interaction_variance<F: Fn(&[f64]) -> f64>(
    f_fn: &F,
    d: usize,
    dim_i: usize,
    dim_j: usize,
    mean: f64,
    var_i: f64,
    var_j: f64,
    nodes: &[f64],
    weights: &[f64],
) -> f64 {
    let nq = nodes.len();
    let d_rest = if d >= 2 { d - 2 } else { 0 };
    let mut x = vec![0.0f64; d];
    let other_dims: Vec<usize> = (0..d).filter(|&dim| dim != dim_i && dim != dim_j).collect();

    let mut v_ij_plus = 0.0f64;
    for (ki, &xi) in nodes.iter().enumerate() {
        for (kj, &xj) in nodes.iter().enumerate() {
            x[dim_i] = xi;
            x[dim_j] = xj;
            // E[f | xᵢ, xⱼ] — integrate out remaining dims
            let total_rest = nq.pow(d_rest as u32);
            let mut acc = 0.0f64;
            for idx in 0..total_rest {
                let mut w = 1.0f64;
                let mut tmp = idx;
                for &dim in &other_dims {
                    let k = tmp % nq;
                    tmp /= nq;
                    x[dim] = nodes[k];
                    w *= weights[k];
                }
                acc += w * f_fn(&x);
            }
            v_ij_plus += weights[ki] * weights[kj] * (acc - mean).powi(2);
        }
    }
    (v_ij_plus - var_i - var_j).max(0.0)
}

// ─── Error type ───────────────────────────────────────────────────────────────

/// Error returned by [`anova_decompose`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnovaError {
    /// `d = 0` is not supported.
    ZeroDimension,
}

impl std::fmt::Display for AnovaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnovaError::ZeroDimension => write!(f, "ANOVA requires d ≥ 1"),
        }
    }
}

impl std::error::Error for AnovaError {}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_legendre_weights_sum_to_one() {
        for n in 1..=5 {
            let (_, weights) = gauss_legendre_01(n);
            let sum: f64 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "n={n}: weights sum = {sum}");
        }
    }

    #[test]
    fn constant_function_mean_correct() {
        let result =
            anova_decompose(&|_x: &[f64]| 7.0_f64, 2, &AnovaConfig::default()).expect("anova");
        assert!((result.mean - 7.0).abs() < 1e-8, "mean={}", result.mean);
        assert!(result.total_variance < 1e-8);
    }
}
