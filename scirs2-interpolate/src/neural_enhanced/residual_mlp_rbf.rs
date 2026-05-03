//! Residual MLP-RBF interpolator.
//!
//! Fits an RBF interpolant on the data, computes residuals, then trains a
//! small MLP to correct the residuals.  At prediction time:
//!
//! ```text
//! predict(x) = rbf(x) + mlp(x)
//! ```
//!
//! Because the MLP output layer is zero-initialised, using `epochs == 0`
//! is equivalent to pure-RBF interpolation.
//!
//! # Design decisions
//!
//! - Uses `ScatteredRbf<f64>` (from `rbf_interpolation`) as the base model.
//! - The MLP operates on `f32` inputs/outputs for efficiency; inputs are the
//!   same coordinates as the RBF, scaled to `[-1, 1]` per dimension.
//! - Backpropagation is pure-ndarray analytic (same pattern as Wave 46 SimCSE).

use crate::error::{InterpolateError, InterpolateResult};
use crate::neural_enhanced::tiny_mlp::{Activation, TinyMlp};
use crate::rbf_interpolation::{RbfKernel, ScatteredRbf};
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for `ResidualMlpRbf`.
#[derive(Debug, Clone)]
pub struct ResidualMlpRbfConfig {
    /// Hidden layer sizes. Default: `[32, 16]`.
    pub hidden_sizes: Vec<usize>,
    /// Activation function. Default: `Tanh`.
    pub activation: Activation,
    /// Number of training epochs. Default: `200`.
    pub epochs: usize,
    /// Learning rate for SGD. Default: `1e-3`.
    pub lr: f32,
    /// Mini-batch size. Default: `16`.
    pub batch_size: usize,
    /// L2 regularisation strength. Default: `1e-4`.
    pub l2: f32,
    /// RNG seed for MLP initialisation. Default: `42`.
    pub seed: u64,
    /// RBF kernel. Default: `Gaussian`.
    pub rbf_kernel: RbfKernel,
    /// RBF shape parameter.  `None` = auto-select.
    pub rbf_epsilon: Option<f64>,
    /// Diagonal regularisation (nugget) added to the RBF matrix before solving.
    ///
    /// A positive `rbf_nugget` makes the RBF a *smoothing* interpolant rather
    /// than exact interpolation; training residuals are non-zero so the MLP has
    /// something to learn.  Set to `0.0` for exact interpolation (which makes
    /// the MLP correction identically zero at training points).
    ///
    /// Default: `1e-3`.
    pub rbf_nugget: f64,
}

impl Default for ResidualMlpRbfConfig {
    fn default() -> Self {
        Self {
            hidden_sizes: vec![32, 16],
            activation: Activation::Tanh,
            epochs: 200,
            lr: 1e-3,
            batch_size: 16,
            l2: 1e-4,
            seed: 42,
            rbf_kernel: RbfKernel::Gaussian,
            rbf_epsilon: None,
            rbf_nugget: 1e-3,
        }
    }
}

// ---------------------------------------------------------------------------
// ResidualMlpRbf
// ---------------------------------------------------------------------------

/// Residual MLP + RBF interpolator.
///
/// # Example
///
/// ```rust
/// use scirs2_interpolate::neural_enhanced::residual_mlp_rbf::{
///     ResidualMlpRbf, ResidualMlpRbfConfig,
/// };
/// use scirs2_core::ndarray::{array, Array2};
///
/// let mut model = ResidualMlpRbf::new(ResidualMlpRbfConfig::default());
///
/// // 1-D training data: y = sin(x)
/// let mut pts = Array2::<f64>::zeros((5, 1));
/// let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
/// let ys: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
/// for (i, &x) in xs.iter().enumerate() { pts[[i, 0]] = x; }
/// let values = scirs2_core::ndarray::Array1::from(ys);
///
/// model.fit(&pts, &values).expect("fit");
/// let pred = model.predict(&array![2.5]).expect("predict");
/// println!("sin(2.5) ≈ {pred:.4}");
/// ```
#[derive(Debug)]
pub struct ResidualMlpRbf {
    config: ResidualMlpRbfConfig,
    rbf: Option<ScatteredRbf<f64>>,
    mlp: Option<TinyMlp>,
    /// Per-dimension min values for normalisation.
    x_min: Vec<f64>,
    /// Per-dimension max values (used for scale).
    x_max: Vec<f64>,
    is_fitted: bool,
}

impl ResidualMlpRbf {
    /// Create a new (unfitted) `ResidualMlpRbf`.
    pub fn new(config: ResidualMlpRbfConfig) -> Self {
        Self {
            config,
            rbf: None,
            mlp: None,
            x_min: Vec::new(),
            x_max: Vec::new(),
            is_fitted: false,
        }
    }

    /// Fit the model to `(points, values)`.
    ///
    /// 1. Fit base RBF on `(points, values)`.
    /// 2. Compute residuals `r_i = values[i] - rbf(points[i])`.
    /// 3. Train MLP on `(normalised_points, residuals)`.
    ///
    /// # Arguments
    ///
    /// * `points` — `(n, d)` matrix of training coordinates.
    /// * `values` — `n`-vector of function values.
    pub fn fit(&mut self, points: &Array2<f64>, values: &Array1<f64>) -> InterpolateResult<()> {
        let n = points.nrows();
        let d = points.ncols();

        if n == 0 {
            return Err(InterpolateError::empty_data("ResidualMlpRbf::fit"));
        }
        if values.len() != n {
            return Err(InterpolateError::ShapeMismatch {
                expected: format!("{n}"),
                actual: format!("{}", values.len()),
                object: "values".to_string(),
            });
        }

        // 1. Fit base RBF (optionally with nugget regularisation).
        let rbf = ScatteredRbf::<f64>::new_with_nugget(
            points,
            values,
            self.config.rbf_kernel,
            self.config.rbf_epsilon,
            self.config.rbf_nugget,
        )?;

        // 2. Compute residuals.
        let mut residuals = Array1::<f64>::zeros(n);
        for i in 0..n {
            let pt: Vec<f64> = (0..d).map(|k| points[[i, k]]).collect();
            let rbf_val = rbf.evaluate(&pt)?;
            residuals[i] = values[i] - rbf_val;
        }

        // 3. Compute per-dimension normalisation constants.
        let mut x_min = vec![f64::MAX; d];
        let mut x_max = vec![f64::MIN; d];
        for i in 0..n {
            for k in 0..d {
                let v = points[[i, k]];
                if v < x_min[k] {
                    x_min[k] = v;
                }
                if v > x_max[k] {
                    x_max[k] = v;
                }
            }
        }
        // Avoid divide-by-zero for constant dimensions.
        for k in 0..d {
            if (x_max[k] - x_min[k]).abs() < 1e-12 {
                x_max[k] = x_min[k] + 1.0;
            }
        }

        // 4. Build and train MLP.
        let mut layer_sizes = vec![d];
        layer_sizes.extend_from_slice(&self.config.hidden_sizes);
        layer_sizes.push(1); // scalar output

        let mut mlp = TinyMlp::new(&layer_sizes, self.config.activation, self.config.seed)?;

        if self.config.epochs > 0 {
            // Simple SGD loop with mini-batches.
            let bs = self.config.batch_size.max(1).min(n);
            let lr = self.config.lr;
            let l2 = self.config.l2;

            // Build a shuffled index list using XorShift.
            for _epoch in 0..self.config.epochs {
                // Process all samples in chunks (simple sequential, no true shuffle).
                let mut start = 0;
                while start < n {
                    let end = (start + bs).min(n);
                    for i in start..end {
                        let x_norm = self.normalise_point(points, i, &x_min, &x_max, d);
                        let target = residuals[i] as f32;
                        mlp.train_step(&x_norm, target, lr, l2)?;
                    }
                    start = end;
                }
            }
        }

        self.rbf = Some(rbf);
        self.mlp = Some(mlp);
        self.x_min = x_min;
        self.x_max = x_max;
        self.is_fitted = true;
        Ok(())
    }

    /// Predict at a single query point.
    ///
    /// Returns `rbf(x) + mlp(x)`.
    ///
    /// # Arguments
    ///
    /// * `x` — 1-D array of length `d` (spatial dimension).
    pub fn predict(&self, x: &Array1<f64>) -> InterpolateResult<f64> {
        if !self.is_fitted {
            return Err(InterpolateError::InvalidState(
                "ResidualMlpRbf is not fitted; call fit() first".to_string(),
            ));
        }
        let rbf = self
            .rbf
            .as_ref()
            .ok_or_else(|| InterpolateError::InvalidState("RBF not fitted".to_string()))?;
        let mlp = self
            .mlp
            .as_ref()
            .ok_or_else(|| InterpolateError::InvalidState("MLP not fitted".to_string()))?;

        let d = x.len();
        let pt: Vec<f64> = x.to_vec();
        let rbf_val = rbf.evaluate(&pt)?;

        // Normalise input.
        let x_norm: Array1<f32> = Array1::from_iter((0..d).map(|k| {
            let range = self.x_max[k] - self.x_min[k];
            ((x[k] - self.x_min[k]) / range * 2.0 - 1.0) as f32
        }));
        let mlp_out = mlp.forward(&x_norm)?;
        let correction = mlp_out[0] as f64;

        Ok(rbf_val + correction)
    }

    /// Normalise a training point for MLP input into `[-1, 1]^d`.
    fn normalise_point(
        &self,
        points: &Array2<f64>,
        i: usize,
        x_min: &[f64],
        x_max: &[f64],
        d: usize,
    ) -> Array1<f32> {
        Array1::from_iter((0..d).map(|k| {
            let range = x_max[k] - x_min[k];
            ((points[[i, k]] - x_min[k]) / range * 2.0 - 1.0) as f32
        }))
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    /// Helper: build 1-D sin data.
    fn sin_data(n: usize) -> (Array2<f64>, Array1<f64>) {
        let mut pts = Array2::<f64>::zeros((n, 1));
        let mut vals = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x = i as f64 / (n - 1) as f64 * std::f64::consts::PI * 2.0;
            pts[[i, 0]] = x;
            vals[i] = x.sin();
        }
        (pts, vals)
    }

    #[test]
    fn residual_rbf_is_fitted_after_fit() {
        let (pts, vals) = sin_data(10);
        let mut model = ResidualMlpRbf::new(ResidualMlpRbfConfig::default());
        assert!(!model.is_fitted());
        model.fit(&pts, &vals).expect("fit");
        assert!(model.is_fitted());
    }

    #[test]
    fn residual_rbf_predict_before_fit_returns_error() {
        let model = ResidualMlpRbf::new(ResidualMlpRbfConfig::default());
        let x = Array1::from(vec![1.0f64]);
        assert!(model.predict(&x).is_err());
    }
}
