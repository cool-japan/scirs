//! Symbolic formula → linear layer weight initialisation via least-squares.
//!
//! Given a [`LoweredOp`] formula and a sample grid of input points, this
//! module fits the linear model `W · x + b ≈ formula(x)` by solving a
//! least-squares problem over the grid.
//!
//! # Algorithm
//!
//! 1. Evaluate the formula at every row of `sample_grid` to obtain targets `y`.
//! 2. Augment each input row with a bias column `[x_0, …, x_{n-1}, 1.0]`.
//! 3. Solve `X_aug · θ = y` in the least-squares sense via
//!    [`scirs2_linalg::lstsq`] (QR-based).
//! 4. Split the solution into `W = θ[0..n_inputs]` and `b = θ[n_inputs]`.
//!
//! The returned `W` has shape `[1, n_inputs]` and `b` has shape `[1]` so they
//! can be loaded directly into a scalar-output linear layer.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::LoweredOp;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`init_weights_from_formula`].
#[derive(Debug)]
pub enum InitFromFormulaError {
    /// Evaluation of the formula at a sample point failed.
    EvalError(String),
    /// The `sample_grid` is empty (zero rows or zero columns).
    EmptyGrid,
    /// The least-squares solve failed (e.g. the design matrix is singular).
    SingularDesignMatrix,
    /// Internal ndarray shape error (should not occur under normal use).
    ShapeError(String),
}

impl std::fmt::Display for InitFromFormulaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitFromFormulaError::EvalError(msg) => {
                write!(f, "formula evaluation error: {msg}")
            }
            InitFromFormulaError::EmptyGrid => {
                write!(f, "sample_grid has zero rows or zero columns")
            }
            InitFromFormulaError::SingularDesignMatrix => {
                write!(
                    f,
                    "design matrix is singular — cannot compute least-squares weights"
                )
            }
            InitFromFormulaError::ShapeError(msg) => {
                write!(f, "internal shape error: {msg}")
            }
        }
    }
}

impl std::error::Error for InitFromFormulaError {}

// ---------------------------------------------------------------------------
// Main function
// ---------------------------------------------------------------------------

/// Initialise linear layer weights `(W, b)` by least-squares fitting a
/// symbolic formula over a sample grid.
///
/// The formula is treated as a scalar function `f: ℝ^{n_inputs} → ℝ`.
/// We find `W ∈ ℝ^{1 × n_inputs}` and `b ∈ ℝ` that minimise
/// `‖W · x + b − f(x)‖₂` over the `n_samples` rows of `sample_grid`.
///
/// # Arguments
///
/// * `formula` — symbolic expression; `Var(k)` addresses `sample_grid[row, k]`.
/// * `sample_grid` — shape `(n_samples, n_inputs)`.
///
/// # Returns
///
/// `(W, b)` where `W` has shape `[1, n_inputs]` and `b` has shape `[1]`.
///
/// # Errors
///
/// * [`InitFromFormulaError::EmptyGrid`] — `sample_grid` has zero rows or
///   zero columns.
/// * [`InitFromFormulaError::EvalError`] — a formula evaluation failed.
/// * [`InitFromFormulaError::SingularDesignMatrix`] — QR back-substitution
///   found a zero on the diagonal.
pub fn init_weights_from_formula(
    formula: &Arc<LoweredOp>,
    sample_grid: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array1<f64>), InitFromFormulaError> {
    let n_samples = sample_grid.shape()[0];
    let n_inputs = sample_grid.shape()[1];

    if n_samples == 0 || n_inputs == 0 {
        return Err(InitFromFormulaError::EmptyGrid);
    }

    // -----------------------------------------------------------------------
    // Step 1 — Build target vector y by evaluating the formula at each row.
    // -----------------------------------------------------------------------
    let mut y_vec: Vec<f64> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let row: Vec<f64> = (0..n_inputs).map(|j| sample_grid[(i, j)]).collect();
        let ctx = EvalCtx::new(&row);
        let val = eval_real(formula, &ctx)
            .map_err(|e| InitFromFormulaError::EvalError(format!("{e:?}")))?;
        y_vec.push(val);
    }

    // -----------------------------------------------------------------------
    // Step 2 — Build augmented design matrix X with bias column.
    //   Each row: [x_0, x_1, ..., x_{n-1}, 1.0]   shape (n_samples, n_inputs+1)
    // -----------------------------------------------------------------------
    let n_aug = n_inputs + 1;
    let mut x_data: Vec<f64> = Vec::with_capacity(n_samples * n_aug);
    for i in 0..n_samples {
        for j in 0..n_inputs {
            x_data.push(sample_grid[(i, j)]);
        }
        x_data.push(1.0); // bias column
    }
    let x_aug = Array2::from_shape_vec((n_samples, n_aug), x_data)
        .map_err(|e| InitFromFormulaError::ShapeError(e.to_string()))?;
    let y_arr = Array1::from_vec(y_vec);

    // -----------------------------------------------------------------------
    // Step 3 — Solve via scirs2_linalg::lstsq (QR-based).
    // -----------------------------------------------------------------------
    // lstsq may return LinalgError::Singular or similar; map all linalg errors
    // to SingularDesignMatrix (the only plausible failure mode here).
    let lstsq_result = scirs2_linalg::lstsq(&x_aug.view(), &y_arr.view(), None)
        .map_err(|_e| InitFromFormulaError::SingularDesignMatrix)?;
    let theta: Array1<f64> = lstsq_result.x;

    // -----------------------------------------------------------------------
    // Step 4 — Extract W and b.
    // -----------------------------------------------------------------------
    let w_slice: Vec<f64> = (0..n_inputs).map(|j| theta[j]).collect();
    let b_val = theta[n_inputs];

    let w = Array2::from_shape_vec((1, n_inputs), w_slice)
        .map_err(|e| InitFromFormulaError::ShapeError(e.to_string()))?;
    let b = Array1::from_vec(vec![b_val]);

    Ok((w, b))
}
