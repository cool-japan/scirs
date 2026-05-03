//! Advanced Neural Time Series Forecasting Models
//!
//! This module provides production-quality implementations of state-of-the-art
//! neural network architectures for time series forecasting:
//!
//! - **Temporal Fusion Transformer (TFT)**: Interpretable multi-horizon forecasting
//!   with variable selection networks, gated residual networks, and static covariate encoding.
//!
//! - **N-BEATS**: Neural Basis Expansion Analysis with trend/seasonality/generic stacks,
//!   backcast/forecast decomposition, and interpretable signal components.
//!
//! - **DeepAR**: Probabilistic autoregressive LSTM-based forecasting with Gaussian and
//!   negative binomial output distributions, yielding full prediction intervals.
//!
//! - **Time Series Transformer**: Encoder-decoder transformer with learned positional
//!   encoding, feature embedding, and multi-step ahead forecasting.
//!
//! All models implement a common [`NeuralForecastModel`] trait providing `fit()`,
//! `predict()`, and `predict_interval()` methods.

pub mod crossformer;
pub mod deepar;
pub mod dlinear;
pub mod itransformer;
pub mod nbeats;
pub mod nhits;
pub mod patchtst;
pub mod simple_api;
pub mod tft;
pub mod timesnet;
pub mod ts_transformer;

// Re-exports for convenience
pub use deepar::{DeepARConfig, DeepARModel, OutputDistribution};
pub use nbeats::{NBEATSConfig, NBEATSModel, StackType as NBeatsStackType};
pub use tft::{TFTConfig, TFTModel};
pub use ts_transformer::{TSTransformerConfig, TSTransformerModel};

use crate::error::Result;
use crate::forecasting::ForecastResult;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

/// Common trait for all neural forecasting models.
///
/// All models in this module implement this trait, providing a unified
/// interface for training, prediction, and probabilistic forecasting.
pub trait NeuralForecastModel<F: Float + Debug + FromPrimitive> {
    /// Train the model on univariate time series data.
    ///
    /// # Arguments
    /// * `data` - 1-D array of the target time series
    ///
    /// # Errors
    /// Returns error if data is too short for the configured lookback window + horizon.
    fn fit(&mut self, data: &Array1<F>) -> Result<()>;

    /// Train the model with exogenous covariates.
    ///
    /// # Arguments
    /// * `data` - 1-D array of the target time series
    /// * `covariates` - 2-D array of shape `(time_steps, num_features)` with covariates
    ///
    /// # Errors
    /// Returns error if dimensions are inconsistent or data is insufficient.
    fn fit_with_covariates(&mut self, data: &Array1<F>, covariates: &Array2<F>) -> Result<()>;

    /// Forecast `steps` future values.
    ///
    /// # Arguments
    /// * `steps` - Number of future time steps to predict
    ///
    /// # Errors
    /// Returns error if the model has not been trained.
    fn predict(&self, steps: usize) -> Result<ForecastResult<F>>;

    /// Forecast with prediction intervals at the given confidence level.
    ///
    /// # Arguments
    /// * `steps` - Number of future time steps
    /// * `confidence` - Confidence level in `(0, 1)`, e.g. 0.95 for 95% intervals
    ///
    /// # Errors
    /// Returns error if the model has not been trained or confidence is out of range.
    fn predict_interval(&self, steps: usize, confidence: f64) -> Result<ForecastResult<F>>;

    /// Return the training loss history (empty before training).
    fn loss_history(&self) -> &[F];
}

/// Shared utility functions for all neural forecast models.
pub(crate) mod nn_utils {
    use crate::error::{Result, TimeSeriesError};
    use scirs2_core::ndarray::{Array1, Array2, Axis};
    use scirs2_core::numeric::{Float, FromPrimitive};

    /// Initialise a weight matrix with Xavier/Glorot-like pseudo-random values.
    ///
    /// Uses a deterministic linear-congruential generator seeded by `seed`,
    /// producing values in `[-std_dev, +std_dev]`.
    pub fn xavier_matrix<F: Float + FromPrimitive>(
        rows: usize,
        cols: usize,
        seed: u32,
    ) -> Array2<F> {
        let fan_avg = F::from((rows + cols) as f64 / 2.0).unwrap_or_else(|| F::one());
        let std_dev = (F::from(2.0).unwrap_or_else(|| F::one()) / fan_avg).sqrt();
        let mut matrix = Array2::zeros((rows, cols));
        let mut s = seed;
        for i in 0..rows {
            for j in 0..cols {
                s = s.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
                let u = F::from(s as f64 / 2_147_483_647.0).unwrap_or_else(|| F::zero());
                let centered = (u - F::from(0.5).unwrap_or_else(|| F::zero()))
                    * F::from(2.0).unwrap_or_else(|| F::one());
                matrix[[i, j]] = centered * std_dev;
            }
        }
        matrix
    }

    /// Initialise a bias vector of zeros.
    pub fn zero_bias<F: Float>(size: usize) -> Array1<F> {
        Array1::zeros(size)
    }

    /// Dense (fully-connected) forward pass: `output = input * W^T + bias`.
    ///
    /// * `input`  - shape `(batch, in_dim)`
    /// * `weight` - shape `(out_dim, in_dim)`
    /// * `bias`   - shape `(out_dim,)`
    ///
    /// Returns shape `(batch, out_dim)`.
    pub fn dense_forward<F: Float>(
        input: &Array2<F>,
        weight: &Array2<F>,
        bias: &Array1<F>,
    ) -> Array2<F> {
        let (batch, _in_dim) = input.dim();
        let out_dim = weight.nrows();
        let mut output = Array2::zeros((batch, out_dim));
        for b in 0..batch {
            for o in 0..out_dim {
                let mut acc = bias[o];
                for k in 0..weight.ncols() {
                    acc = acc + input[[b, k]] * weight[[o, k]];
                }
                output[[b, o]] = acc;
            }
        }
        output
    }

    /// Dense forward for a single vector.
    pub fn dense_forward_vec<F: Float>(
        input: &Array1<F>,
        weight: &Array2<F>,
        bias: &Array1<F>,
    ) -> Array1<F> {
        let out_dim = weight.nrows();
        let mut output = Array1::zeros(out_dim);
        for o in 0..out_dim {
            let mut acc = bias[o];
            for k in 0..weight.ncols() {
                acc = acc + input[k] * weight[[o, k]];
            }
            output[o] = acc;
        }
        output
    }

    /// Element-wise ReLU activation on a 2-D array.
    pub fn relu_2d<F: Float>(x: &Array2<F>) -> Array2<F> {
        x.mapv(|v| v.max(F::zero()))
    }

    /// Element-wise ReLU activation on a 1-D array.
    pub fn relu_1d<F: Float>(x: &Array1<F>) -> Array1<F> {
        x.mapv(|v| v.max(F::zero()))
    }

    /// Element-wise sigmoid on a 2-D array.
    pub fn sigmoid_2d<F: Float>(x: &Array2<F>) -> Array2<F> {
        x.mapv(|v| F::one() / (F::one() + (-v).exp()))
    }

    /// Element-wise sigmoid on a 1-D array.
    pub fn sigmoid_1d<F: Float>(x: &Array1<F>) -> Array1<F> {
        x.mapv(|v| F::one() / (F::one() + (-v).exp()))
    }

    /// Element-wise tanh on a 2-D array.
    pub fn tanh_2d<F: Float>(x: &Array2<F>) -> Array2<F> {
        x.mapv(|v| v.tanh())
    }

    /// Element-wise tanh on a 1-D array.
    pub fn tanh_1d<F: Float>(x: &Array1<F>) -> Array1<F> {
        x.mapv(|v| v.tanh())
    }

    /// GELU approximation element-wise on 1-D array.
    pub fn gelu_1d<F: Float + FromPrimitive>(x: &Array1<F>) -> Array1<F> {
        let half = F::from(0.5).unwrap_or_else(|| F::zero());
        let sqrt_2_pi = F::from(0.7978845608).unwrap_or_else(|| F::one());
        let coeff = F::from(0.044715).unwrap_or_else(|| F::zero());
        x.mapv(|v| half * v * (F::one() + (sqrt_2_pi * (v + coeff * v * v * v)).tanh()))
    }

    /// Softmax over the last axis of a 2-D array (row-wise).
    pub fn softmax_rows<F: Float>(x: &Array2<F>) -> Array2<F> {
        let (rows, cols) = x.dim();
        let mut out = Array2::zeros((rows, cols));
        for r in 0..rows {
            let row_max = x.row(r).iter().cloned().fold(F::neg_infinity(), F::max);
            let mut sum_exp = F::zero();
            for c in 0..cols {
                let e = (x[[r, c]] - row_max).exp();
                out[[r, c]] = e;
                sum_exp = sum_exp + e;
            }
            if sum_exp > F::zero() {
                for c in 0..cols {
                    out[[r, c]] = out[[r, c]] / sum_exp;
                }
            }
        }
        out
    }

    /// Layer normalisation over the feature (last) axis of a 2-D array.
    pub fn layer_norm<F: Float + FromPrimitive>(
        x: &Array2<F>,
        gamma: &Array1<F>,
        beta: &Array1<F>,
    ) -> Array2<F> {
        let eps = F::from(1e-5).unwrap_or_else(|| F::zero());
        let (rows, cols) = x.dim();
        let cols_f = F::from(cols).unwrap_or_else(|| F::one());
        let mut out = Array2::zeros((rows, cols));
        for r in 0..rows {
            let mut mean = F::zero();
            for c in 0..cols {
                mean = mean + x[[r, c]];
            }
            mean = mean / cols_f;
            let mut var = F::zero();
            for c in 0..cols {
                let diff = x[[r, c]] - mean;
                var = var + diff * diff;
            }
            var = var / cols_f;
            let inv_std = F::one() / (var + eps).sqrt();
            for c in 0..cols {
                out[[r, c]] = (x[[r, c]] - mean) * inv_std * gamma[c] + beta[c];
            }
        }
        out
    }

    /// Create sliding windows from time series data.
    ///
    /// Returns `(X, Y)` where `X` has shape `(n_samples, lookback)` and
    /// `Y` has shape `(n_samples, horizon)`.
    pub fn create_sliding_windows<F: Float>(
        data: &Array1<F>,
        lookback: usize,
        horizon: usize,
    ) -> Result<(Array2<F>, Array2<F>)> {
        if lookback == 0 || horizon == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "lookback and horizon must be positive".to_string(),
            ));
        }
        let n = data.len();
        if n < lookback + horizon {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for sliding windows".to_string(),
                required: lookback + horizon,
                actual: n,
            });
        }
        let n_samples = n - lookback - horizon + 1;
        let mut x = Array2::zeros((n_samples, lookback));
        let mut y = Array2::zeros((n_samples, horizon));
        for i in 0..n_samples {
            for j in 0..lookback {
                x[[i, j]] = data[i + j];
            }
            for j in 0..horizon {
                y[[i, j]] = data[i + lookback + j];
            }
        }
        Ok((x, y))
    }

    /// Min-max normalisation returning `(normalised, min, max)`.
    pub fn normalize<F: Float + FromPrimitive>(data: &Array1<F>) -> Result<(Array1<F>, F, F)> {
        if data.is_empty() {
            return Err(TimeSeriesError::InvalidInput("Data is empty".to_string()));
        }
        let min_val = data.iter().cloned().fold(data[0], F::min);
        let max_val = data.iter().cloned().fold(data[0], F::max);
        let range = max_val - min_val;
        if range == F::zero() {
            return Err(TimeSeriesError::InvalidInput(
                "Data has no variance".to_string(),
            ));
        }
        let normed = data.mapv(|v| (v - min_val) / range);
        Ok((normed, min_val, max_val))
    }

    /// Inverse min-max normalisation.
    pub fn denormalize<F: Float>(data: &Array1<F>, min_val: F, max_val: F) -> Array1<F> {
        let range = max_val - min_val;
        data.mapv(|v| v * range + min_val)
    }

    /// z-score for a confidence level.
    pub fn z_score_for_confidence<F: Float + FromPrimitive>(confidence: f64) -> F {
        let z = if confidence >= 0.99 {
            2.576
        } else if confidence >= 0.975 {
            2.241
        } else if confidence >= 0.95 {
            1.96
        } else if confidence >= 0.90 {
            1.645
        } else if confidence >= 0.80 {
            1.282
        } else {
            1.0
        };
        F::from(z).unwrap_or_else(|| F::one())
    }

    /// Compute MSE between two 1-D arrays of equal length.
    pub fn mse<F: Float + FromPrimitive>(a: &Array1<F>, b: &Array1<F>) -> F {
        let n = a.len();
        if n == 0 {
            return F::zero();
        }
        let mut sum = F::zero();
        for i in 0..n {
            let d = a[i] - b[i];
            sum = sum + d * d;
        }
        sum / F::from(n).unwrap_or_else(|| F::one())
    }
}
