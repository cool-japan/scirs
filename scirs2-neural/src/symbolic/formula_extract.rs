//! Extract symbolic formulas from callable functions using symbolic regression.
//!
//! Given any callable `f: ArrayView2<f64> → Array2<f64>`, this module runs SR
//! on the callable's outputs over a user-supplied input grid to discover a
//! compact symbolic approximation.
//!
//! Using a generic `F: Fn(...)` callable (rather than a `Layer` trait) keeps
//! this module decoupled from the neural layer hierarchy and works equally well
//! with `Dense` forwards, closure wrappers, or arbitrary black-box functions.
//!
//! # Stochasticity note
//!
//! Symbolic regression is non-deterministic.  Tests should only assert that
//! the call returns `Ok` with a non-empty result vector, not that a specific
//! formula is recovered.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_symbolic::regression::{discover, DiscoveredFormula, SrConfig};
use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`extract_formula_from_callable`].
#[derive(Debug, Clone)]
pub struct FormulaExtractionConfig {
    /// Maximum formula tree size (node count).
    pub max_complexity: usize,
    /// Beam width — candidates retained per generation.
    pub population_size: usize,
    /// Number of search iterations.
    pub n_generations: usize,
    /// How many top formulas to return.
    pub n_results: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for FormulaExtractionConfig {
    fn default() -> Self {
        Self {
            max_complexity: 10,
            population_size: 150,
            n_generations: 30,
            n_results: 3,
            seed: 42,
        }
    }
}

impl FormulaExtractionConfig {
    /// Convert to [`SrConfig`].
    fn to_sr_config(&self) -> SrConfig {
        SrConfig::default()
            .with_max_nodes(self.max_complexity)
            .with_beam_width(self.population_size)
            .with_max_iter(self.n_generations)
            .with_top_n(self.n_results)
            .with_seed(self.seed)
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`extract_formula_from_callable`].
#[derive(Debug)]
pub enum FormulaExtractionError {
    /// The `input_grid` has zero rows.
    EmptyGrid,
    /// The callable returned an array with a different number of rows than the
    /// input grid, or more than one column (only scalar output is supported).
    DimensionMismatch,
    /// An error propagated from the symbolic regression engine (e.g. the
    /// engine was given incompatible data).
    SymbolicError(String),
}

impl fmt::Display for FormulaExtractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormulaExtractionError::EmptyGrid => {
                write!(f, "input_grid has zero rows — nothing to extract from")
            }
            FormulaExtractionError::DimensionMismatch => write!(
                f,
                "callable returned an array with wrong shape: \
                 expected (n_samples, 1), check callable output dimensions"
            ),
            FormulaExtractionError::SymbolicError(msg) => {
                write!(f, "symbolic regression error: {msg}")
            }
        }
    }
}

impl std::error::Error for FormulaExtractionError {}

// ---------------------------------------------------------------------------
// Main function
// ---------------------------------------------------------------------------

/// Extract symbolic formula(s) from a scalar-valued callable by running SR on
/// its outputs.
///
/// The callable `f` is invoked once on the full `input_grid`; its output must
/// have shape `(n_samples, 1)`.  The single output column is used as the SR
/// target.
///
/// # Arguments
///
/// * `f` — callable mapping `ArrayView2<f64>` → `Array2<f64>`.  The output
///   must have shape `(n_samples, 1)`.
/// * `input_grid` — shape `(n_samples, n_inputs)`.
/// * `config` — search configuration.
///
/// # Returns
///
/// `Vec<DiscoveredFormula>` of the top-`config.n_results` formulas ranked by
/// combined fitness (best first).  May be empty if the search finds no finite-
/// fitness candidate (e.g. the callable returns all-NaN).
pub fn extract_formula_from_callable<F>(
    f: F,
    input_grid: ArrayView2<'_, f64>,
    config: &FormulaExtractionConfig,
) -> Result<Vec<DiscoveredFormula>, FormulaExtractionError>
where
    F: Fn(ArrayView2<'_, f64>) -> Array2<f64>,
{
    let n_samples = input_grid.shape()[0];

    // 1. Validate grid is non-empty.
    if n_samples == 0 {
        return Err(FormulaExtractionError::EmptyGrid);
    }

    // 2. Invoke the callable.
    let output = f(input_grid);

    // 3. Validate output shape: must be (n_samples, 1).
    if output.shape()[0] != n_samples || output.shape().len() < 2 || output.shape()[1] != 1 {
        return Err(FormulaExtractionError::DimensionMismatch);
    }

    // 4. Extract the target column.
    let targets = output.column(0).to_owned();

    // 5. Run symbolic regression.
    let sr_cfg = config.to_sr_config();
    let results = discover(input_grid, targets.view(), &sr_cfg);

    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests (unit — run as part of the crate test suite)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn linear_grid(n: usize, scale: f64) -> Array2<f64> {
        Array2::from_shape_fn((n, 1), |(i, _)| i as f64 * scale / n as f64)
    }

    #[test]
    fn test_empty_grid_returns_err() {
        let grid = Array2::<f64>::zeros((0, 1));
        let cfg = FormulaExtractionConfig::default();
        let result = extract_formula_from_callable(
            |x: ArrayView2<'_, f64>| Array2::zeros((x.shape()[0], 1)),
            grid.view(),
            &cfg,
        );
        assert!(
            matches!(result, Err(FormulaExtractionError::EmptyGrid)),
            "expected EmptyGrid, got {:?}",
            result.err().map(|e| e.to_string())
        );
    }

    #[test]
    fn test_dim_mismatch_returns_err() {
        let grid = Array2::<f64>::zeros((4, 1));
        let cfg = FormulaExtractionConfig::default();
        // Callable returns wrong shape (n_samples=3 instead of 4).
        let result = extract_formula_from_callable(
            |_x: ArrayView2<'_, f64>| Array2::zeros((3, 1)),
            grid.view(),
            &cfg,
        );
        assert!(
            matches!(result, Err(FormulaExtractionError::DimensionMismatch)),
            "expected DimensionMismatch, got {:?}",
            result.err().map(|e| e.to_string())
        );
    }
}
