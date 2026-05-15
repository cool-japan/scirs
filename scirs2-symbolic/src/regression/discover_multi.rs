//! Multi-output symbolic regression.
//!
//! Discover separate formulas for each output dimension of vector-valued
//! data. Useful for systems like Lorenz attractors, Lotka-Volterra, etc.,
//! where the right-hand sides of multiple equations need recovery.
//!
//! Phase 1 first cut: per-output independent discovery via [`fn@discover`].
//! v0.4.5 will add JOINT discovery (shared sub-expressions across outputs
//! via structural-hash CSE).
//!
//! # Examples
//!
//! ```no_run
//! use ndarray::arr2;
//! use scirs2_symbolic::regression::{discover_multi, SrConfig};
//!
//! let features = arr2(&[[1.0], [2.0], [3.0]]);
//! let targets = arr2(&[[1.0, 2.0], [4.0, 4.0], [9.0, 6.0]]); // [x², 2x]
//! let config = SrConfig::default();
//! let results = discover_multi(features.view(), targets.view(), &config);
//! // results[0] should approximate x²; results[1] should approximate 2x.
//! ```
//!
//! # Limitations (Phase 1)
//!
//! - Outputs are searched **independently** — there is no sharing of
//!   sub-expressions across output dimensions. A future v0.4.5 release will
//!   add JOINT search where common sub-trees are amortised across outputs
//!   via structural-hash CSE (Common Subexpression Elimination).
//! - Search budget (`config.max_iter`, `config.beam_width`, etc.) is
//!   applied **per output**; total work scales linearly with `n_outputs`.

use crate::regression::{discover, DiscoveredFormula, SrConfig};
use ndarray::ArrayView2;

/// Discover formulas for each output dimension of vector-valued targets.
///
/// Phase 1 implementation: per-output **independent** discovery — for each
/// column of `targets`, [`fn@discover`] is invoked with the same `config`.
/// Joint search with cross-output sub-expression sharing is planned for
/// v0.4.5 (see module-level docs).
///
/// # Arguments
/// - `features`: shape `(n_samples, n_features)` 2D array of input data.
/// - `targets`: shape `(n_samples, n_outputs)` 2D array of vector-valued
///   targets.
/// - `config`: search configuration (applied identically per output).
///
/// # Returns
/// `Vec<Vec<DiscoveredFormula>>` — outer index is output dimension `i`;
/// inner `Vec` is up to `config.top_n` formulas for output `i` ranked by
/// fitness (best first).
///
/// Returns an empty `Vec` if input shapes are incompatible (sample count
/// mismatch, zero samples, or zero outputs).
///
/// # Example
///
/// ```no_run
/// use ndarray::arr2;
/// use scirs2_symbolic::regression::{discover_multi, SrConfig};
///
/// let features = arr2(&[[1.0], [2.0], [3.0]]);
/// let targets = arr2(&[[1.0, 2.0], [4.0, 4.0], [9.0, 6.0]]); // [x², 2x]
/// let config = SrConfig::default();
/// let results = discover_multi(features.view(), targets.view(), &config);
/// assert_eq!(results.len(), 2);
/// ```
pub fn discover_multi(
    features: ArrayView2<'_, f64>,
    targets: ArrayView2<'_, f64>,
    config: &SrConfig,
) -> Vec<Vec<DiscoveredFormula>> {
    let (n_samples_f, _) = features.dim();
    let (n_samples_t, n_outputs) = targets.dim();

    if n_samples_f != n_samples_t || n_samples_f == 0 || n_outputs == 0 {
        return Vec::new();
    }

    (0..n_outputs)
        .map(|out_idx| {
            let target_col = targets.column(out_idx);
            discover(features, target_col, config)
        })
        .collect()
}

/// Convenience: extract the single best formula per output dimension.
///
/// Equivalent to calling [`discover_multi`] and taking the first element
/// of each per-output result vector. Returns `None` for any output where
/// the search produced no candidates (e.g. all candidates produced
/// non-finite predictions).
///
/// # Arguments
/// See [`discover_multi`].
///
/// # Returns
/// `Vec<Option<DiscoveredFormula>>` of length `n_outputs` (empty if input
/// shapes are incompatible — see [`discover_multi`]).
///
/// # Example
///
/// ```no_run
/// use ndarray::arr2;
/// use scirs2_symbolic::regression::{discover_multi_best, SrConfig};
///
/// let features = arr2(&[[1.0], [2.0], [3.0]]);
/// let targets = arr2(&[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]); // [x, 2x]
/// let config = SrConfig::default();
/// let bests = discover_multi_best(features.view(), targets.view(), &config);
/// assert_eq!(bests.len(), 2);
/// ```
pub fn discover_multi_best(
    features: ArrayView2<'_, f64>,
    targets: ArrayView2<'_, f64>,
    config: &SrConfig,
) -> Vec<Option<DiscoveredFormula>> {
    discover_multi(features, targets, config)
        .into_iter()
        .map(|mut v| {
            if v.is_empty() {
                None
            } else {
                Some(v.remove(0))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn discover_multi_two_outputs() {
        // outputs: [x, x²] — output 0 fits perfectly via the identity
        // `Var(0)` already in the initial population; output 1 needs the
        // first-generation expansion `Mul(Var(0), Var(0))`.
        let xs: Vec<f64> = (0..30).map(|i| (i as f64 - 15.0) * 0.2).collect();
        let features = Array2::from_shape_vec((30, 1), xs.clone()).expect("shape");
        let mut targets_data: Vec<f64> = Vec::new();
        for &x in &xs {
            targets_data.push(x);
            targets_data.push(x * x);
        }
        let targets = Array2::from_shape_vec((30, 2), targets_data).expect("shape");

        let config = SrConfig::default().with_max_iter(20);
        let results = discover_multi(features.view(), targets.view(), &config);

        assert_eq!(results.len(), 2);
        assert!(!results[0].is_empty());
        assert!(!results[1].is_empty());
        // Output 0 = x: should fit perfectly (identity formula in initial pop).
        assert!(
            results[0][0].fitness.mse < 1e-10,
            "output 0 MSE = {}",
            results[0][0].fitness.mse
        );
        // Output 1 = x²: should fit well (Mul(Var, Var) generation 1).
        assert!(
            results[1][0].fitness.r_squared > 0.9,
            "output 1 R² = {}",
            results[1][0].fitness.r_squared
        );
    }

    #[test]
    fn discover_multi_handles_empty() {
        let features = Array2::<f64>::zeros((0, 1));
        let targets = Array2::<f64>::zeros((0, 2));
        let config = SrConfig::default();
        let results = discover_multi(features.view(), targets.view(), &config);
        assert!(results.is_empty());
    }

    #[test]
    fn discover_multi_handles_zero_outputs() {
        let features = Array2::from_shape_vec((10, 1), vec![0.0; 10]).expect("shape");
        let targets = Array2::<f64>::zeros((10, 0));
        let config = SrConfig::default();
        let results = discover_multi(features.view(), targets.view(), &config);
        assert!(results.is_empty());
    }

    #[test]
    fn discover_multi_shape_mismatch() {
        let features = Array2::from_shape_vec((10, 1), vec![0.0; 10]).expect("shape");
        let targets = Array2::<f64>::zeros((20, 2));
        let config = SrConfig::default();
        let results = discover_multi(features.view(), targets.view(), &config);
        assert!(results.is_empty());
    }

    #[test]
    fn discover_multi_best_returns_top_per_output() {
        // Both outputs are identical (= x); each `Some` should be the
        // identity formula `Var(0)` from the initial population.
        let xs: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1).collect();
        let features = Array2::from_shape_vec((20, 1), xs.clone()).expect("shape");
        let mut targets_data: Vec<f64> = Vec::new();
        for &x in &xs {
            targets_data.push(x);
            targets_data.push(x);
        }
        let targets = Array2::from_shape_vec((20, 2), targets_data).expect("shape");

        let config = SrConfig::default().with_max_iter(10);
        let bests = discover_multi_best(features.view(), targets.view(), &config);

        assert_eq!(bests.len(), 2);
        assert!(bests[0].is_some());
        assert!(bests[1].is_some());
    }
}
