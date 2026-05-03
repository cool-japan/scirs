//! Anchored-ANOVA decomposition around a fixed anchor point.
//!
//! Unlike the classical ANOVA (which requires marginal integration), the
//! anchored decomposition is *evaluation-only*: terms are defined by
//! substituting the anchor point `c` into all dimensions except those under
//! study.
//!
//! ```text
//! f(x) = f(c)
//!       + Σᵢ  [f(xᵢ, c_{−i}) − f(c)]
//!       + Σᵢ<ⱼ [f(xᵢ, xⱼ, c_{−ij}) − fᵢ(xᵢ) − fⱼ(xⱼ) − f(c)]
//!       + …
//! ```
//!
//! **Important:** anchored-ANOVA terms are **not** mutually orthogonal
//! (unlike classical ANOVA), so the sum of their variance estimates may
//! exceed the total variance of `f`.
//!
//! # References
//! - Kuo, Sloan, Wasilkowski & Woźniakowski (2010), "On decompositions of
//!   multivariate functions", Math. Comput. 79, 953–966.

/// Result of an anchored-ANOVA decomposition.
#[derive(Debug, Clone)]
pub struct AnchoredAnovaDecomposition {
    /// f(c) — the value at the anchor point.
    pub anchor_value: f64,
    /// First-order terms: (dimension, empirical variance of fᵢ).
    pub main_effects: Vec<(usize, f64)>,
    /// Second-order terms: (dim_i, dim_j, empirical variance of fᵢⱼ).
    pub interaction_effects: Vec<(usize, usize, f64)>,
    /// Highest interaction order included (1 or 2).
    pub max_order_reached: usize,
}

/// Compute an anchored-ANOVA decomposition of `f: [0,1]^d → ℝ`.
///
/// # Arguments
/// * `f_fn`          – Black-box function.
/// * `d`             – Input dimensionality.
/// * `anchor`        – Anchor point `c ∈ [0,1]^d`.  Defaults to `[0.5; d]`.
/// * `max_order`     – Highest order of terms to compute (currently 1 or 2).
/// * `n_eval_per_dim` – Number of equispaced evaluation points per dimension
///   for variance estimation.
pub fn anchored_anova_decompose<F>(
    f_fn: &F,
    d: usize,
    anchor: Option<&[f64]>,
    max_order: usize,
    n_eval_per_dim: usize,
) -> Result<AnchoredAnovaDecomposition, AnchoredAnovaError>
where
    F: Fn(&[f64]) -> f64,
{
    let default_anchor = vec![0.5f64; d];
    let c: &[f64] = anchor.unwrap_or(&default_anchor);
    if c.len() != d {
        return Err(AnchoredAnovaError::AnchorDimMismatch);
    }

    let anchor_value = f_fn(c);

    // Equispaced evaluation points on (0, 1) — midpoints of n equal intervals.
    let eval_pts: Vec<f64> = (0..n_eval_per_dim)
        .map(|k| (k as f64 + 0.5) / n_eval_per_dim as f64)
        .collect();

    // ── 1st-order terms: fᵢ(xᵢ) = f(xᵢ, c_{−i}) − f(c) ──────────────────
    let mut main_effects: Vec<(usize, f64)> = Vec::with_capacity(d);
    // Store raw fi values for use when computing interaction terms.
    let mut main_effect_vals: Vec<Vec<f64>> = Vec::with_capacity(d);

    for i in 0..d {
        let mut x = c.to_vec();
        let fi_vals: Vec<f64> = eval_pts
            .iter()
            .map(|&xi| {
                x[i] = xi;
                let val = f_fn(&x) - anchor_value;
                x[i] = c[i]; // restore anchor
                val
            })
            .collect();
        let fi_mean = fi_vals.iter().sum::<f64>() / fi_vals.len() as f64;
        let fi_var =
            fi_vals.iter().map(|v| (v - fi_mean).powi(2)).sum::<f64>() / fi_vals.len() as f64;
        main_effects.push((i, fi_var));
        main_effect_vals.push(fi_vals);
    }

    // ── 2nd-order terms (optional) ─────────────────────────────────────────
    let mut interaction_effects: Vec<(usize, usize, f64)> = Vec::new();

    if max_order >= 2 {
        let nq = n_eval_per_dim;
        for i in 0..d {
            for j in (i + 1)..d {
                let mut x = c.to_vec();
                let mut fij_vals: Vec<f64> = Vec::with_capacity(nq * nq);

                for (qi, &xi) in eval_pts.iter().enumerate() {
                    for (qj, &xj) in eval_pts.iter().enumerate() {
                        x[i] = xi;
                        x[j] = xj;
                        // fᵢⱼ(xᵢ,xⱼ) = f(xᵢ,xⱼ,c_{−ij}) − f(c) − fᵢ(xᵢ) − fⱼ(xⱼ)
                        let fij = f_fn(&x)
                            - anchor_value
                            - main_effect_vals[i][qi]
                            - main_effect_vals[j][qj];
                        fij_vals.push(fij);
                        x[i] = c[i];
                        x[j] = c[j]; // restore anchor
                    }
                }

                let fij_mean = fij_vals.iter().sum::<f64>() / fij_vals.len() as f64;
                let fij_var = fij_vals.iter().map(|v| (v - fij_mean).powi(2)).sum::<f64>()
                    / fij_vals.len() as f64;

                if fij_var > 1e-12 {
                    interaction_effects.push((i, j, fij_var));
                }
            }
        }
    }

    let max_order_reached = max_order.min(2);
    Ok(AnchoredAnovaDecomposition {
        anchor_value,
        main_effects,
        interaction_effects,
        max_order_reached,
    })
}

/// Adaptive variant: start with order-1 decomposition and add order-2 terms
/// only when the interaction variance is large relative to the main-effect
/// variance (threshold: `tolerance`).
///
/// Returns the order-1 result if `interaction_var < tolerance * main_var`,
/// otherwise returns the order-2 result.
pub fn adaptive_anchored_anova_refinement<F>(
    f_fn: &F,
    d: usize,
    anchor: Option<&[f64]>,
    tolerance: f64,
    n_eval: usize,
) -> Result<AnchoredAnovaDecomposition, AnchoredAnovaError>
where
    F: Fn(&[f64]) -> f64,
{
    let result_1 = anchored_anova_decompose(f_fn, d, anchor, 1, n_eval)?;
    let total_var_1: f64 = result_1.main_effects.iter().map(|(_, v)| v).sum();

    let result_2 = anchored_anova_decompose(f_fn, d, anchor, 2, n_eval)?;
    let interaction_var: f64 = result_2.interaction_effects.iter().map(|(_, _, v)| v).sum();

    if interaction_var < tolerance * (total_var_1 + 1e-12) {
        Ok(result_1)
    } else {
        Ok(result_2)
    }
}

// ─── Error type ───────────────────────────────────────────────────────────────

/// Error returned by [`anchored_anova_decompose`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnchoredAnovaError {
    /// Length of the supplied `anchor` slice does not match `d`.
    AnchorDimMismatch,
}

impl std::fmt::Display for AnchoredAnovaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnchoredAnovaError::AnchorDimMismatch => {
                write!(f, "anchor slice length does not match dimension d")
            }
        }
    }
}

impl std::error::Error for AnchoredAnovaError {}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_value_matches_function_at_default_anchor() {
        let f = |x: &[f64]| x[0] + x[1];
        let result = anchored_anova_decompose(&f, 2, None, 1, 10).expect("decompose");
        // Default anchor is (0.5, 0.5) → f(c) = 1.0
        assert!((result.anchor_value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn dim_mismatch_returns_error() {
        let bad_anchor = vec![0.5; 3];
        let err =
            anchored_anova_decompose(&|_x: &[f64]| 0.0, 2, Some(&bad_anchor), 1, 10).unwrap_err();
        assert_eq!(err, AnchoredAnovaError::AnchorDimMismatch);
    }
}
