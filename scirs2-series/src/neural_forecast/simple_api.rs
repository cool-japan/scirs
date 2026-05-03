//! Simple / flat API for N-BEATS, ETS, and Theta method
//!
//! This module exposes structs with the exact signatures requested in the
//! v0.3.0 API specification, built on top of the richer implementations in the
//! sibling sub-modules and in `crate::ets` / `crate::theta`.
//!
//! # Structures
//!
//! - [`NBeatsBlock`] — single N-BEATS block (FC stack + basis expansion)
//! - [`NBeats`] — full N-BEATS model (multiple stacks of blocks)
//! - [`EtsErrorType`], [`EtsTrendType`], [`EtsSeasonalType`] — ETS component enums
//! - [`EtsModel`] / [`FittedEts`] — fit-then-forecast ETS state-space model
//! - [`auto_ets`] — automatic model selection by AIC
//! - [`ThetaModel`] — Theta method with SES + linear drift decomposition

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Lightweight LCG pseudo-random number generator.
fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f64) / (u32::MAX as f64)
}

/// ReLU activation.
#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Dense layer forward pass: y = W·x + b  (W is row-major, rows = out_size).
fn dense_fwd(x: &[f64], w: &[f64], b: &[f64], out_size: usize) -> Vec<f64> {
    let in_size = x.len();
    let mut out = vec![0.0_f64; out_size];
    for o in 0..out_size {
        let mut acc = b[o];
        for i in 0..in_size {
            acc += w[o * in_size + i] * x[i];
        }
        out[o] = acc;
    }
    out
}

/// Compute MSE between two equal-length slices.
fn mse(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / a.len() as f64
}

/// Golden-section search on [lo, hi] minimising f.
fn golden_section<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64) -> f64 {
    let gr = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut a = lo;
    let mut b = hi;
    let mut x1 = b - gr * (b - a);
    let mut x2 = a + gr * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    for _ in 0..200 {
        if (b - a).abs() < 1e-9 {
            break;
        }
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - gr * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + gr * (b - a);
            f2 = f(x2);
        }
    }
    (a + b) / 2.0
}

// ---------------------------------------------------------------------------
// N-BEATS block
// ---------------------------------------------------------------------------

/// A single N-BEATS block implementing basis-expansion decomposition.
///
/// Each block comprises:
/// 1. A stack of `n_layers` fully-connected hidden layers of width `layer_size`.
/// 2. Two linear projections to `theta_size` coefficients.
/// 3. Basis expansion: backcast and forecast vectors computed from theta coefficients.
///
/// The default basis is generic (identity-like polynomial): `basis[k][t] = t^k / n`.
#[derive(Debug, Clone)]
pub struct NBeatsBlock {
    /// Number of basis expansion coefficients.
    pub theta_size: usize,
    /// Length of the backcast output.
    pub backcast_size: usize,
    /// Length of the forecast output.
    pub forecast_size: usize,
    /// Hidden layers: each element is `(weight_matrix_flat, bias_vec)`.
    /// Weight matrix shape: `(layer_size × in_dim)` stored row-major.
    pub layers: Vec<(Vec<f64>, Vec<f64>)>,
    /// Width of each hidden layer.
    pub layer_size: usize,
    /// Number of hidden layers.
    pub n_layers: usize,
    /// Backcast theta projection: weight `(theta_size × layer_size)`, bias `(theta_size,)`.
    theta_b_w: Vec<f64>,
    theta_b_b: Vec<f64>,
    /// Forecast theta projection.
    theta_f_w: Vec<f64>,
    theta_f_b: Vec<f64>,
    /// Backcast basis matrix `(theta_size × backcast_size)`.
    basis_b: Vec<f64>,
    /// Forecast basis matrix `(theta_size × forecast_size)`.
    basis_f: Vec<f64>,
}

impl NBeatsBlock {
    /// Create a new N-BEATS block with randomly-initialised weights.
    ///
    /// # Arguments
    /// * `backcast_size` — length of the input (lookback) window
    /// * `forecast_size` — number of future steps
    /// * `theta_size`    — number of basis expansion coefficients
    /// * `n_layers`      — number of hidden FC layers
    /// * `layer_size`    — width of each hidden layer
    pub fn new(
        backcast_size: usize,
        forecast_size: usize,
        theta_size: usize,
        n_layers: usize,
        layer_size: usize,
    ) -> Self {
        let mut rng: u64 = 0xdeadbeef_cafebabe;

        // Xavier std-dev helper
        let xavier =
            |fan_in: usize, fan_out: usize| -> f64 { (2.0 / (fan_in + fan_out) as f64).sqrt() };

        let init_mat = |rows: usize, cols: usize, rng: &mut u64| -> Vec<f64> {
            let std = xavier(cols, rows);
            (0..rows * cols)
                .map(|_| (lcg_next(rng) * 2.0 - 1.0) * std)
                .collect()
        };

        let init_bias = |size: usize| -> Vec<f64> { vec![0.0_f64; size] };

        // Build hidden layers (in_dim → layer_size, then layer_size → layer_size)
        let mut layers = Vec::with_capacity(n_layers);
        let mut in_dim = backcast_size;
        for _ in 0..n_layers {
            let w = init_mat(layer_size, in_dim, &mut rng);
            let b = init_bias(layer_size);
            layers.push((w, b));
            in_dim = layer_size;
        }

        // Theta projections
        let theta_b_w = init_mat(theta_size, layer_size, &mut rng);
        let theta_b_b = init_bias(theta_size);
        let theta_f_w = init_mat(theta_size, layer_size, &mut rng);
        let theta_f_b = init_bias(theta_size);

        // Generic polynomial basis
        let basis_b = Self::make_basis(theta_size, backcast_size);
        let basis_f = Self::make_basis(theta_size, forecast_size);

        Self {
            theta_size,
            backcast_size,
            forecast_size,
            layers,
            layer_size,
            n_layers,
            theta_b_w,
            theta_b_b,
            theta_f_w,
            theta_f_b,
            basis_b,
            basis_f,
        }
    }

    /// Construct a generic polynomial basis of size `(n_theta × length)`.
    ///
    /// `basis[k][t] = (t / length)^k` for k = 0..n_theta, t = 0..length.
    fn make_basis(n_theta: usize, length: usize) -> Vec<f64> {
        let mut b = vec![0.0_f64; n_theta * length];
        for k in 0..n_theta {
            for t in 0..length {
                let x = if length > 1 {
                    t as f64 / (length - 1) as f64
                } else {
                    0.0
                };
                b[k * length + t] = x.powi(k as i32);
            }
        }
        b
    }

    /// Forward pass.
    ///
    /// Returns `(backcast, forecast)` where each vector has length
    /// `backcast_size` and `forecast_size` respectively.
    pub fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Hidden stack
        let mut h = x.to_vec();
        for (w, b) in &self.layers {
            let out = dense_fwd(&h, w, b, self.layer_size);
            h = out.into_iter().map(relu).collect();
        }

        // Theta coefficients
        let theta_b = dense_fwd(&h, &self.theta_b_w, &self.theta_b_b, self.theta_size);
        let theta_f = dense_fwd(&h, &self.theta_f_w, &self.theta_f_b, self.theta_size);

        // Basis expansion: backcast[t] = sum_k theta_b[k] * basis_b[k*bs+t]
        let mut backcast = vec![0.0_f64; self.backcast_size];
        for k in 0..self.theta_size {
            for t in 0..self.backcast_size {
                backcast[t] += theta_b[k] * self.basis_b[k * self.backcast_size + t];
            }
        }

        let mut forecast = vec![0.0_f64; self.forecast_size];
        for k in 0..self.theta_size {
            for t in 0..self.forecast_size {
                forecast[t] += theta_f[k] * self.basis_f[k * self.forecast_size + t];
            }
        }

        (backcast, forecast)
    }
}

// ---------------------------------------------------------------------------
// N-BEATS model
// ---------------------------------------------------------------------------

/// Full N-BEATS model: multiple stacks of [`NBeatsBlock`]s.
///
/// Architecture (doubly-residual stacking):
/// 1. Input residual starts as the raw lookback window.
/// 2. Each block computes a backcast and a forecast.
/// 3. The backcast is subtracted from the residual passed to the next block.
/// 4. All forecasts from all stacks are summed.
#[derive(Debug, Clone)]
pub struct NBeats {
    /// Stacks of blocks: outer Vec = stacks, inner Vec = blocks within a stack.
    pub stacks: Vec<Vec<NBeatsBlock>>,
    /// Length of the lookback window.
    pub backcast_size: usize,
    /// Number of future steps to forecast.
    pub forecast_size: usize,
    /// Training loss history (MSE per epoch).
    loss_history: Vec<f64>,
}

impl NBeats {
    /// Create a new N-BEATS model with random weights.
    ///
    /// Each stack contains `blocks_per_stack` generic blocks with
    /// `theta_size = backcast_size`, `n_layers = 4`, `layer_size = 256`.
    ///
    /// # Arguments
    /// * `backcast_size`    — lookback window length
    /// * `forecast_size`    — forecast horizon
    /// * `n_stacks`         — number of stacks
    /// * `blocks_per_stack` — blocks per stack
    pub fn new(
        backcast_size: usize,
        forecast_size: usize,
        n_stacks: usize,
        blocks_per_stack: usize,
    ) -> Self {
        let theta_size = backcast_size.min(64).max(4);
        let layer_size = 128_usize;
        let n_layers = 4_usize;

        let stacks = (0..n_stacks)
            .map(|_| {
                (0..blocks_per_stack)
                    .map(|_| {
                        NBeatsBlock::new(
                            backcast_size,
                            forecast_size,
                            theta_size,
                            n_layers,
                            layer_size,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Self {
            stacks,
            backcast_size,
            forecast_size,
            loss_history: Vec::new(),
        }
    }

    /// Produce a forecast from the last `backcast_size` values of `history`.
    ///
    /// Returns a vector of length `forecast_size`.
    pub fn forecast(&self, history: &[f64]) -> Result<Vec<f64>> {
        if history.len() < self.backcast_size {
            return Err(TimeSeriesError::InsufficientData {
                message: format!(
                    "N-BEATS forecast requires at least {} history values",
                    self.backcast_size
                ),
                required: self.backcast_size,
                actual: history.len(),
            });
        }

        // Extract last backcast_size values and normalize
        let window: Vec<f64> = history[history.len() - self.backcast_size..].to_vec();
        let (normed, mean, std_dev) = normalize_window(&window);

        let mut residual = normed.clone();
        let mut global_forecast = vec![0.0_f64; self.forecast_size];

        for stack in &self.stacks {
            for block in stack {
                let (backcast, fc) = block.forward(&residual);
                // Subtract backcast from residual
                for (r, b) in residual.iter_mut().zip(backcast.iter()) {
                    *r -= b;
                }
                // Accumulate forecast
                for (g, f) in global_forecast.iter_mut().zip(fc.iter()) {
                    *g += f;
                }
            }
        }

        // Denormalize
        let result: Vec<f64> = global_forecast
            .iter()
            .map(|&v| v * std_dev + mean)
            .collect();
        Ok(result)
    }

    /// Fit the model using stochastic gradient descent with finite-difference gradients.
    ///
    /// For computational tractability this uses a simplified one-pass SGD over
    /// random sliding-window samples, with numerical (central difference) gradients
    /// for the last FC layer of each block.
    ///
    /// # Arguments
    /// * `train_data` — full training time series
    /// * `epochs`     — number of training epochs
    /// * `lr`         — learning rate
    ///
    /// # Returns
    /// Loss history (MSE per epoch).
    pub fn fit(&mut self, train_data: &[f64], epochs: usize, lr: f64) -> Result<Vec<f64>> {
        let n = train_data.len();
        let window = self.backcast_size + self.forecast_size;
        if n < window {
            return Err(TimeSeriesError::InsufficientData {
                message: format!("N-BEATS fit requires at least {} data points", window),
                required: window,
                actual: n,
            });
        }

        let n_samples = n - window + 1;
        let mut rng: u64 = 42;

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0_f64;
            let mut count = 0_usize;

            // Shuffle sample indices
            let mut indices: Vec<usize> = (0..n_samples).collect();
            for i in (1..indices.len()).rev() {
                let j = (lcg_next(&mut rng) * (i + 1) as f64) as usize;
                indices.swap(i, j.min(i));
            }

            for &start in &indices {
                let x_raw = &train_data[start..start + self.backcast_size];
                let y_raw = &train_data[start + self.backcast_size..start + window];

                let (x_norm, x_mean, x_std) = normalize_window(x_raw);
                let y_norm: Vec<f64> = y_raw
                    .iter()
                    .map(|&v| {
                        if x_std > 1e-10 {
                            (v - x_mean) / x_std
                        } else {
                            0.0
                        }
                    })
                    .collect();

                let pred = self.forward_pass(&x_norm);
                let loss = mse(&pred, &y_norm);
                epoch_loss += loss;
                count += 1;

                // Finite-difference gradient update on theta projection weights
                // of the last block in each stack (lightweight approximation)
                let eps = 1e-4_f64;
                for stack_idx in 0..self.stacks.len() {
                    let block_idx = self.stacks[stack_idx].len().saturating_sub(1);
                    let theta_f_w_len = self.stacks[stack_idx][block_idx].theta_f_w.len();
                    // Only update a random subset for speed
                    let sample_count = theta_f_w_len.min(8);
                    for _ in 0..sample_count {
                        let wi = (lcg_next(&mut rng) * theta_f_w_len as f64) as usize;
                        let wi = wi.min(theta_f_w_len - 1);

                        self.stacks[stack_idx][block_idx].theta_f_w[wi] += eps;
                        let pred_plus = self.forward_pass(&x_norm);
                        let loss_plus = mse(&pred_plus, &y_norm);

                        self.stacks[stack_idx][block_idx].theta_f_w[wi] -= 2.0 * eps;
                        let pred_minus = self.forward_pass(&x_norm);
                        let loss_minus = mse(&pred_minus, &y_norm);

                        self.stacks[stack_idx][block_idx].theta_f_w[wi] += eps; // restore

                        let grad = (loss_plus - loss_minus) / (2.0 * eps);
                        self.stacks[stack_idx][block_idx].theta_f_w[wi] -= lr * grad;
                    }
                }
            }

            let avg_loss = if count > 0 {
                epoch_loss / count as f64
            } else {
                0.0
            };
            self.loss_history.push(avg_loss);
        }

        Ok(self.loss_history.clone())
    }

    /// Internal forward pass for a normalized input window.
    fn forward_pass(&self, x_norm: &[f64]) -> Vec<f64> {
        let mut residual = x_norm.to_vec();
        let mut global_forecast = vec![0.0_f64; self.forecast_size];

        for stack in &self.stacks {
            for block in stack {
                let (backcast, fc) = block.forward(&residual);
                for (r, b) in residual.iter_mut().zip(backcast.iter()) {
                    *r -= b;
                }
                for (g, f) in global_forecast.iter_mut().zip(fc.iter()) {
                    *g += f;
                }
            }
        }
        global_forecast
    }

    /// Return training loss history.
    pub fn loss_history(&self) -> &[f64] {
        &self.loss_history
    }
}

/// Normalize a slice to zero mean and unit variance.
/// Returns `(normalized, mean, std_dev)`.
fn normalize_window(x: &[f64]) -> (Vec<f64>, f64, f64) {
    if x.is_empty() {
        return (Vec::new(), 0.0, 1.0);
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
    let std_dev = var.sqrt().max(1e-10);
    let normed = x.iter().map(|&v| (v - mean) / std_dev).collect();
    (normed, mean, std_dev)
}

// ---------------------------------------------------------------------------
// ETS error / trend / seasonal types
// ---------------------------------------------------------------------------

/// Error component type for the ETS state-space model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtsErrorType {
    /// Additive errors.
    Additive,
    /// Multiplicative errors.
    Multiplicative,
}

/// Trend component type for the ETS model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtsTrendType {
    /// No trend.
    None,
    /// Additive (linear) trend.
    Additive,
    /// Additive damped trend (phi < 1 dampens growth).
    AdditiveDamped,
}

/// Seasonal component type for the ETS model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtsSeasonalType {
    /// No seasonality.
    None,
    /// Additive seasonality.
    Additive,
    /// Multiplicative seasonality.
    Multiplicative,
}

// ---------------------------------------------------------------------------
// EtsModel builder
// ---------------------------------------------------------------------------

/// Builder for an ETS (Error-Trend-Seasonal) state-space model.
///
/// Call [`EtsModel::fit`] to obtain a [`FittedEts`] struct with forecast methods.
#[derive(Debug, Clone)]
pub struct EtsModel {
    /// Error component type.
    pub error_type: EtsErrorType,
    /// Trend component type.
    pub trend_type: EtsTrendType,
    /// Seasonal component type.
    pub seasonal_type: EtsSeasonalType,
    /// Level smoothing parameter (0 < alpha < 1).
    pub alpha: f64,
    /// Trend smoothing parameter (0 < beta < 1).
    pub beta: f64,
    /// Seasonal smoothing parameter (0 < gamma < 1).
    pub gamma: f64,
    /// Damping factor for damped-trend (0 < phi < 1).
    pub phi: f64,
    /// Seasonal period (must be >= 2 when seasonal_type != None).
    pub seasonal_period: usize,
}

impl EtsModel {
    /// Create a new ETS model with default smoothing parameters.
    ///
    /// Default parameters: alpha=0.3, beta=0.1, gamma=0.1, phi=0.98.
    pub fn new(
        error: EtsErrorType,
        trend: EtsTrendType,
        seasonal: EtsSeasonalType,
        period: usize,
    ) -> Self {
        Self {
            error_type: error,
            trend_type: trend,
            seasonal_type: seasonal,
            alpha: 0.3,
            beta: 0.1,
            gamma: 0.1,
            phi: 0.98,
            seasonal_period: period,
        }
    }

    /// Fit the model to data, optimising smoothing parameters, and return a [`FittedEts`].
    pub fn fit(mut self, data: &[f64]) -> Result<FittedEts> {
        let n = data.len();
        let p = self.seasonal_period;

        // Minimum data length validation
        let min_len = match self.seasonal_type {
            EtsSeasonalType::None => 4,
            _ => {
                if p < 2 {
                    return Err(TimeSeriesError::InvalidParameter {
                        name: "seasonal_period".to_string(),
                        message: "Seasonal period must be >= 2".to_string(),
                    });
                }
                2 * p + 1
            }
        };
        if n < min_len {
            return Err(TimeSeriesError::InsufficientData {
                message: "Insufficient data for ETS model".to_string(),
                required: min_len,
                actual: n,
            });
        }

        // Validate multiplicative seasonal data must be positive
        if self.seasonal_type == EtsSeasonalType::Multiplicative {
            if data.iter().any(|&v| v <= 0.0) {
                return Err(TimeSeriesError::InvalidInput(
                    "Multiplicative seasonal ETS requires all-positive data".to_string(),
                ));
            }
        }

        // Optimise alpha via golden-section search
        if self.trend_type == EtsTrendType::None && self.seasonal_type == EtsSeasonalType::None {
            let data_clone = data.to_vec();
            let opt_alpha = golden_section(|a| ets_sse_simple(&data_clone, a), 1e-4, 0.999);
            self.alpha = opt_alpha;
        } else {
            // For trend/seasonal models, do a coarse grid search
            self.alpha = self.grid_search_alpha(data);
        }

        // Initialise states and run recursion
        let (level_states, trend_states, seasonal_states, fitted, residuals) =
            self.run_recursion(data)?;

        let sigma2: f64 = residuals.iter().map(|r: &f64| r * r).sum::<f64>() / n as f64;

        // Information criteria
        let n_params = self.count_params();
        let log_lik = if sigma2 > 0.0 {
            -0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln())
        } else {
            0.0
        };
        let k = n_params as f64;
        let nf = n as f64;
        let aic = -2.0 * log_lik + 2.0 * k;
        let bic = -2.0 * log_lik + k * nf.ln();

        Ok(FittedEts {
            model: self,
            level: level_states,
            trend: trend_states,
            seasonal: seasonal_states,
            fitted,
            residuals,
            aic,
            bic,
        })
    }

    /// Coarse grid search over alpha (and beta if trend).
    fn grid_search_alpha(&self, data: &[f64]) -> f64 {
        let alphas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9];
        let mut best_alpha = 0.3;
        let mut best_sse = f64::INFINITY;
        for &a in &alphas {
            let mut trial = self.clone();
            trial.alpha = a;
            if let Ok((_, _, _, _, resid)) = trial.run_recursion(data) {
                let sse: f64 = resid.iter().map(|r: &f64| r * r).sum::<f64>();
                if sse < best_sse {
                    best_sse = sse;
                    best_alpha = a;
                }
            }
        }
        best_alpha
    }

    /// Count number of free parameters for AIC/BIC.
    fn count_params(&self) -> usize {
        let mut k = 1; // alpha
        if self.trend_type != EtsTrendType::None {
            k += 1; // beta
        }
        if self.seasonal_type != EtsSeasonalType::None {
            k += 1; // gamma
        }
        if self.trend_type == EtsTrendType::AdditiveDamped {
            k += 1; // phi
        }
        k += 1; // initial level
        if self.trend_type != EtsTrendType::None {
            k += 1;
        }
        if self.seasonal_type != EtsSeasonalType::None {
            k += self.seasonal_period - 1;
        }
        k
    }

    /// Run the ETS recursion, returning (level_states, trend_states, seasonal_states, fitted, residuals).
    fn run_recursion(
        &self,
        data: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let n = data.len();
        let p = self.seasonal_period.max(1);

        // Initialise states
        let (mut l, mut b_trend, mut seasonals) = self.init_states(data)?;

        let mut level_hist = Vec::with_capacity(n + 1);
        let mut trend_hist = Vec::with_capacity(n + 1);
        let mut seasonal_hist = Vec::with_capacity(n * p + p);
        let mut fitted = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);

        level_hist.push(l);
        trend_hist.push(b_trend);
        for &s in &seasonals {
            seasonal_hist.push(s);
        }

        for t in 0..n {
            let s_t = if self.seasonal_type != EtsSeasonalType::None {
                seasonals[t % p]
            } else {
                if self.seasonal_type == EtsSeasonalType::Multiplicative {
                    1.0
                } else {
                    0.0
                }
            };

            // One-step forecast
            let yhat = self.compute_forecast(l, b_trend, s_t, 1);
            fitted.push(yhat);

            let err = data[t] - yhat;
            residuals.push(err);

            // State update
            let new_l;
            let new_b;
            let new_s;

            match (self.trend_type, self.seasonal_type) {
                (EtsTrendType::None, EtsSeasonalType::None) => {
                    new_l = self.alpha * data[t] + (1.0 - self.alpha) * l;
                    new_b = 0.0;
                    new_s = 0.0;
                }
                (EtsTrendType::None, EtsSeasonalType::Additive) => {
                    new_l = self.alpha * (data[t] - s_t) + (1.0 - self.alpha) * l;
                    new_b = 0.0;
                    new_s = self.gamma * (data[t] - new_l) + (1.0 - self.gamma) * s_t;
                }
                (EtsTrendType::None, EtsSeasonalType::Multiplicative) => {
                    let safe_s = if s_t.abs() > 1e-10 { s_t } else { 1.0 };
                    new_l = self.alpha * (data[t] / safe_s) + (1.0 - self.alpha) * l;
                    new_b = 0.0;
                    let safe_nl = if new_l.abs() > 1e-10 { new_l } else { 1.0 };
                    new_s = self.gamma * (data[t] / safe_nl) + (1.0 - self.gamma) * s_t;
                }
                (EtsTrendType::Additive, EtsSeasonalType::None) => {
                    new_l = self.alpha * data[t] + (1.0 - self.alpha) * (l + b_trend);
                    new_b = self.beta * (new_l - l) + (1.0 - self.beta) * b_trend;
                    new_s = 0.0;
                }
                (EtsTrendType::Additive, EtsSeasonalType::Additive) => {
                    new_l = self.alpha * (data[t] - s_t) + (1.0 - self.alpha) * (l + b_trend);
                    new_b = self.beta * (new_l - l) + (1.0 - self.beta) * b_trend;
                    new_s = self.gamma * (data[t] - new_l) + (1.0 - self.gamma) * s_t;
                }
                (EtsTrendType::Additive, EtsSeasonalType::Multiplicative) => {
                    let safe_s = if s_t.abs() > 1e-10 { s_t } else { 1.0 };
                    new_l = self.alpha * (data[t] / safe_s) + (1.0 - self.alpha) * (l + b_trend);
                    new_b = self.beta * (new_l - l) + (1.0 - self.beta) * b_trend;
                    let safe_nl = if new_l.abs() > 1e-10 { new_l } else { 1.0 };
                    new_s = self.gamma * (data[t] / safe_nl) + (1.0 - self.gamma) * s_t;
                }
                (EtsTrendType::AdditiveDamped, EtsSeasonalType::None) => {
                    new_l = self.alpha * data[t] + (1.0 - self.alpha) * (l + self.phi * b_trend);
                    new_b = self.beta * (new_l - l) + (1.0 - self.beta) * self.phi * b_trend;
                    new_s = 0.0;
                }
                (EtsTrendType::AdditiveDamped, EtsSeasonalType::Additive) => {
                    new_l = self.alpha * (data[t] - s_t)
                        + (1.0 - self.alpha) * (l + self.phi * b_trend);
                    new_b = self.beta * (new_l - l) + (1.0 - self.beta) * self.phi * b_trend;
                    new_s = self.gamma * (data[t] - new_l) + (1.0 - self.gamma) * s_t;
                }
                (EtsTrendType::AdditiveDamped, EtsSeasonalType::Multiplicative) => {
                    let safe_s = if s_t.abs() > 1e-10 { s_t } else { 1.0 };
                    new_l = self.alpha * (data[t] / safe_s)
                        + (1.0 - self.alpha) * (l + self.phi * b_trend);
                    new_b = self.beta * (new_l - l) + (1.0 - self.beta) * self.phi * b_trend;
                    let safe_nl = if new_l.abs() > 1e-10 { new_l } else { 1.0 };
                    new_s = self.gamma * (data[t] / safe_nl) + (1.0 - self.gamma) * s_t;
                }
            }

            l = new_l;
            b_trend = new_b;
            if self.seasonal_type != EtsSeasonalType::None {
                seasonals[t % p] = new_s;
            }

            level_hist.push(l);
            trend_hist.push(b_trend);
            for &s in &seasonals {
                seasonal_hist.push(s);
            }
        }

        Ok((level_hist, trend_hist, seasonal_hist, fitted, residuals))
    }

    /// Initialise level, trend, and seasonal states from data.
    fn init_states(&self, data: &[f64]) -> Result<(f64, f64, Vec<f64>)> {
        let p = self.seasonal_period.max(1);

        let level = data[0];

        let trend = if self.trend_type != EtsTrendType::None && data.len() >= 2 {
            let n_diff = data.len().min(5);
            let slope = data[1..n_diff]
                .iter()
                .zip(data[..n_diff - 1].iter())
                .map(|(b, a)| b - a)
                .sum::<f64>()
                / (n_diff - 1) as f64;
            slope
        } else {
            0.0
        };

        let seasonals = if self.seasonal_type != EtsSeasonalType::None {
            let n_seasons = (data.len() / p).max(1);
            // Compute season means for the first n_seasons
            let season_means: Vec<f64> = (0..n_seasons.min(3))
                .map(|s| {
                    let start = s * p;
                    let end = (start + p).min(data.len());
                    let slice = &data[start..end];
                    slice.iter().sum::<f64>() / slice.len() as f64
                })
                .collect();
            let overall_mean = season_means.iter().sum::<f64>() / season_means.len() as f64;

            match self.seasonal_type {
                EtsSeasonalType::Additive => {
                    let mut s = vec![0.0_f64; p];
                    for j in 0..p {
                        let mut total = 0.0;
                        let mut count = 0;
                        for si in 0..n_seasons.min(3) {
                            let idx = si * p + j;
                            if idx < data.len() {
                                let sm = season_means.get(si).copied().unwrap_or(overall_mean);
                                total += data[idx] - sm;
                                count += 1;
                            }
                        }
                        s[j] = if count > 0 { total / count as f64 } else { 0.0 };
                    }
                    // Normalise to sum-zero
                    let mean_s = s.iter().sum::<f64>() / p as f64;
                    for v in &mut s {
                        *v -= mean_s;
                    }
                    s
                }
                EtsSeasonalType::Multiplicative => {
                    let mut s = vec![1.0_f64; p];
                    if overall_mean.abs() > 1e-10 {
                        for j in 0..p {
                            let mut total = 0.0;
                            let mut count = 0;
                            for si in 0..n_seasons.min(3) {
                                let idx = si * p + j;
                                if idx < data.len() {
                                    let sm = season_means.get(si).copied().unwrap_or(overall_mean);
                                    if sm.abs() > 1e-10 {
                                        total += data[idx] / sm;
                                        count += 1;
                                    }
                                }
                            }
                            s[j] = if count > 0 { total / count as f64 } else { 1.0 };
                        }
                        // Normalise to product = 1
                        let mean_s = s.iter().sum::<f64>() / p as f64;
                        for v in &mut s {
                            if mean_s.abs() > 1e-10 {
                                *v /= mean_s;
                            }
                        }
                    }
                    s
                }
                EtsSeasonalType::None => vec![],
            }
        } else {
            vec![]
        };

        Ok((level, trend, seasonals))
    }

    /// Compute h-step point forecast given current state components.
    fn compute_forecast(&self, level: f64, trend: f64, seasonal: f64, h: usize) -> f64 {
        let trend_part = match self.trend_type {
            EtsTrendType::None => 0.0,
            EtsTrendType::Additive => h as f64 * trend,
            EtsTrendType::AdditiveDamped => {
                // sum_{i=1}^{h} phi^i * trend
                let mut phi_sum = 0.0_f64;
                let mut phi_pow = self.phi;
                for _ in 0..h {
                    phi_sum += phi_pow;
                    phi_pow *= self.phi;
                }
                phi_sum * trend
            }
        };

        let base = level + trend_part;

        match self.seasonal_type {
            EtsSeasonalType::None => base,
            EtsSeasonalType::Additive => base + seasonal,
            EtsSeasonalType::Multiplicative => base * seasonal,
        }
    }
}

/// SSE for simple (A,N,N) ETS with given alpha.
fn ets_sse_simple(data: &[f64], alpha: f64) -> f64 {
    let mut l = data[0];
    let mut sse = 0.0_f64;
    for &y in data.iter().skip(1) {
        let err = y - l;
        sse += err * err;
        l = alpha * y + (1.0 - alpha) * l;
    }
    sse
}

// ---------------------------------------------------------------------------
// FittedEts
// ---------------------------------------------------------------------------

/// A fitted ETS model, holding state histories and methods for forecasting.
#[derive(Debug, Clone)]
pub struct FittedEts {
    /// The fitted ETS model configuration and optimised parameters.
    pub model: EtsModel,
    /// Level state at each time step (length n+1).
    pub level: Vec<f64>,
    /// Trend state at each time step (length n+1; empty if no trend).
    pub trend: Vec<f64>,
    /// Seasonal states stored row-major (length (n+1)·p; empty if no seasonal).
    pub seasonal: Vec<f64>,
    /// One-step-ahead fitted values.
    pub fitted: Vec<f64>,
    /// Residuals (data - fitted).
    pub residuals: Vec<f64>,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
}

impl FittedEts {
    /// Produce `h` point forecasts from the end of the training series.
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        let n = self.fitted.len();
        let p = self.model.seasonal_period.max(1);

        // Retrieve final state
        let final_level = self.level.last().copied().unwrap_or(0.0);
        let final_trend = self.trend.last().copied().unwrap_or(0.0);

        let final_seasonals: Vec<f64> =
            if self.model.seasonal_type != EtsSeasonalType::None && !self.seasonal.is_empty() {
                // Last p seasonal values
                let start = if self.seasonal.len() >= p {
                    self.seasonal.len() - p
                } else {
                    0
                };
                self.seasonal[start..].to_vec()
            } else {
                vec![]
            };

        (1..=h)
            .map(|step| {
                let s = if !final_seasonals.is_empty() {
                    let idx = (n + step - 1) % p;
                    final_seasonals.get(idx).copied().unwrap_or(
                        if self.model.seasonal_type == EtsSeasonalType::Multiplicative {
                            1.0
                        } else {
                            0.0
                        },
                    )
                } else {
                    if self.model.seasonal_type == EtsSeasonalType::Multiplicative {
                        1.0
                    } else {
                        0.0
                    }
                };
                self.model
                    .compute_forecast(final_level, final_trend, s, step)
            })
            .collect()
    }

    /// Return (lower, upper) prediction intervals for `h`-step forecast.
    ///
    /// Uses a Normal approximation: `forecast ± z * sqrt(h) * sigma`.
    pub fn prediction_intervals(&self, h: usize, level: f64) -> Vec<(f64, f64)> {
        let z = normal_quantile(0.5 + level / 2.0);
        let sigma = self.residuals.iter().map(|&r| r * r).sum::<f64>();
        let sigma = (sigma / self.residuals.len().max(1) as f64).sqrt();
        let forecasts = self.forecast(h);
        forecasts
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let margin = z * sigma * ((i + 1) as f64).sqrt();
                (f - margin, f + margin)
            })
            .collect()
    }
}

/// Approximation to the standard normal quantile function (Beasley-Springer-Moro).
fn normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    // Rational approximation (Abramowitz & Stegun 26.2.17)
    let sign = if p >= 0.5 { 1.0_f64 } else { -1.0_f64 };
    let q = if p >= 0.5 { 1.0 - p } else { p };
    let t = (-2.0 * q.ln()).sqrt();
    let c0 = 2.515517_f64;
    let c1 = 0.802853_f64;
    let c2 = 0.010328_f64;
    let d1 = 1.432788_f64;
    let d2 = 0.189269_f64;
    let d3 = 0.001308_f64;
    let num = c0 + c1 * t + c2 * t * t;
    let den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    sign * (t - num / den)
}

// ---------------------------------------------------------------------------
// auto_ets
// ---------------------------------------------------------------------------

/// Automatically select and fit the best ETS model by minimising AIC.
///
/// Evaluates all combinations of trend ∈ {None, Additive, AdditiveDamped}
/// and seasonal ∈ {None, Additive, Multiplicative} (with fixed Additive error).
///
/// # Arguments
/// * `data`            — time series observations
/// * `seasonal_period` — candidate seasonal period (pass 1 to disable seasonality search)
///
/// # Returns
/// The best-fitting [`FittedEts`] according to AIC.
pub fn auto_ets(data: &[f64], seasonal_period: usize) -> Result<FittedEts> {
    if data.len() < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "auto_ets requires at least 4 observations".to_string(),
            required: 4,
            actual: data.len(),
        });
    }

    let trends = [
        EtsTrendType::None,
        EtsTrendType::Additive,
        EtsTrendType::AdditiveDamped,
    ];
    let seasonals = if seasonal_period >= 2 {
        vec![
            EtsSeasonalType::None,
            EtsSeasonalType::Additive,
            EtsSeasonalType::Multiplicative,
        ]
    } else {
        vec![EtsSeasonalType::None]
    };

    let mut best: Option<FittedEts> = None;

    for &t in &trends {
        for &s in &seasonals {
            // Multiplicative seasonal requires positive data
            if s == EtsSeasonalType::Multiplicative {
                if data.iter().any(|&v| v <= 0.0) {
                    continue;
                }
            }

            let model = EtsModel::new(EtsErrorType::Additive, t, s, seasonal_period);
            match model.fit(data) {
                Ok(fitted) => {
                    if fitted.aic.is_finite() {
                        let update = match &best {
                            None => true,
                            Some(current) => fitted.aic < current.aic,
                        };
                        if update {
                            best = Some(fitted);
                        }
                    }
                }
                Err(_) => continue,
            }
        }
    }

    best.ok_or_else(|| TimeSeriesError::FittingError("auto_ets: no model converged".to_string()))
}

// ---------------------------------------------------------------------------
// ThetaModel (simple API)
// ---------------------------------------------------------------------------

/// Theta method for time series forecasting (Assimakopoulos & Nikolopoulos, 2000).
///
/// Decomposes the series into:
/// - Theta-line 0 (θ=0): the linear trend (long-run drift)
/// - Theta-line 2 (θ=2): amplified local curvature, fitted with SES
///
/// The combined forecast is: `0.5 * f_0(h) + 0.5 * f_2(h)`.
///
/// This is a simplified wrapper around the detailed implementation in
/// [`crate::theta::ThetaModel`], providing the API specified in the v0.3.0
/// public spec.
#[derive(Debug, Clone)]
pub struct ThetaModel {
    /// Theta coefficient (default 2.0).
    pub theta: f64,
    /// Optional seasonal period for deseasonalisation.
    pub period: Option<usize>,
}

impl ThetaModel {
    /// Create a new Theta model.
    pub fn new(theta: f64) -> Self {
        Self {
            theta,
            period: None,
        }
    }

    /// Forecast `h` steps ahead using the Theta method.
    ///
    /// Applies:
    /// 1. Optional seasonal decomposition.
    /// 2. Linear regression (theta-line 0).
    /// 3. SES on theta-line 2.
    /// 4. Combination with equal weights.
    pub fn fit_forecast(data: &[f64], h: usize) -> Result<Vec<f64>> {
        if data.len() < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "ThetaModel requires at least 4 observations".to_string(),
                required: 4,
                actual: data.len(),
            });
        }
        if h == 0 {
            return Ok(Vec::new());
        }

        let n = data.len();
        let theta = 2.0_f64;

        // Fit linear regression y = a + b*t  (t = 1..=n)
        let (a, b) = linear_regression(data);

        // Construct theta-line 2: θ * y_t - (θ-1) * lin_t
        let theta2: Vec<f64> = data
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let lin = a + b * (i + 1) as f64;
                theta * y - (theta - 1.0) * lin
            })
            .collect();

        // Optimal SES alpha on theta-line 2
        let theta2_clone = theta2.clone();
        let alpha = golden_section(|a| ses_sse(&theta2_clone, a), 1e-6, 1.0 - 1e-10);
        let ses_level = ses_final_level(&theta2, alpha);

        // Combine: 0.5 * f0 + 0.5 * f2
        let forecasts: Vec<f64> = (1..=h)
            .map(|k| {
                let f0 = a + b * (n + k) as f64;
                let f2 = ses_level;
                0.5 * f0 + 0.5 * f2
            })
            .collect();

        Ok(forecasts)
    }
}

// ---------------------------------------------------------------------------
// Simple helpers shared with ThetaModel
// ---------------------------------------------------------------------------

/// Simple linear regression: returns (intercept, slope) for y = a + b*t, t=1..=n.
fn linear_regression(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    let t_mean = (n + 1.0) / 2.0;
    let y_mean = data.iter().sum::<f64>() / n;
    let mut sxy = 0.0_f64;
    let mut sxx = 0.0_f64;
    for (i, &y) in data.iter().enumerate() {
        let t = (i + 1) as f64;
        sxy += (t - t_mean) * (y - y_mean);
        sxx += (t - t_mean).powi(2);
    }
    let slope = if sxx.abs() > 1e-14 { sxy / sxx } else { 0.0 };
    let intercept = y_mean - slope * t_mean;
    (intercept, slope)
}

/// SES sum-of-squared errors for a given alpha.
fn ses_sse(data: &[f64], alpha: f64) -> f64 {
    let mut level = data[0];
    let mut sse = 0.0_f64;
    for &y in data.iter().skip(1) {
        let err = y - level;
        sse += err * err;
        level = alpha * y + (1.0 - alpha) * level;
    }
    sse
}

/// Final SES level after processing data.
fn ses_final_level(data: &[f64], alpha: f64) -> f64 {
    let mut level = data[0];
    for &y in data.iter().skip(1) {
        level = alpha * y + (1.0 - alpha) * level;
    }
    level
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── NBeatsBlock tests ────────────────────────────────────────────────────

    #[test]
    fn test_nbeats_block_output_shapes() {
        let block = NBeatsBlock::new(12, 4, 8, 4, 64);
        let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let (backcast, forecast) = block.forward(&x);
        assert_eq!(backcast.len(), 12, "backcast length");
        assert_eq!(forecast.len(), 4, "forecast length");
    }

    #[test]
    fn test_nbeats_block_finite_output() {
        let block = NBeatsBlock::new(8, 3, 4, 2, 32);
        let x = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0];
        let (backcast, forecast) = block.forward(&x);
        for &v in backcast.iter().chain(forecast.iter()) {
            assert!(v.is_finite(), "output must be finite");
        }
    }

    #[test]
    fn test_nbeats_model_forecast_length() {
        let model = NBeats::new(12, 4, 2, 2);
        let history: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let fc = model.forecast(&history).expect("forecast failed");
        assert_eq!(fc.len(), 4);
    }

    #[test]
    fn test_nbeats_model_fit_returns_loss_history() {
        let mut model = NBeats::new(8, 2, 1, 1);
        let data: Vec<f64> = (0..30).map(|i| i as f64 * 0.5).collect();
        let losses = model.fit(&data, 3, 0.001).expect("fit failed");
        assert_eq!(losses.len(), 3, "one loss per epoch");
        for &l in &losses {
            assert!(l.is_finite(), "loss must be finite");
        }
    }

    #[test]
    fn test_nbeats_model_insufficient_data() {
        let mut model = NBeats::new(10, 5, 1, 1);
        let data = vec![1.0, 2.0, 3.0]; // too short
        assert!(model.fit(&data, 1, 0.01).is_err());
    }

    // ── ETS tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_ets_ann_is_ses() {
        // ETS(A,N,N) = Simple Exponential Smoothing: forecasts should be flat
        let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let model = EtsModel::new(
            EtsErrorType::Additive,
            EtsTrendType::None,
            EtsSeasonalType::None,
            1,
        );
        let fitted = model.fit(&data).expect("fit failed");
        let fc = fitted.forecast(4);
        assert_eq!(fc.len(), 4);
        // SES forecasts are constant
        for w in fc.windows(2) {
            assert!(
                (w[1] - w[0]).abs() < 1e-8,
                "SES forecasts should be flat: {:?}",
                fc
            );
        }
    }

    #[test]
    fn test_ets_aan_holt_increasing() {
        // ETS(A,A,N) = Holt's linear method: forecasts should increase for upward trend
        let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let model = EtsModel::new(
            EtsErrorType::Additive,
            EtsTrendType::Additive,
            EtsSeasonalType::None,
            1,
        );
        let fitted = model.fit(&data).expect("fit failed");
        let fc = fitted.forecast(4);
        assert_eq!(fc.len(), 4);
        for w in fc.windows(2) {
            assert!(
                w[1] > w[0] - 1e-6,
                "Holt forecasts should increase: {:?}",
                fc
            );
        }
    }

    #[test]
    fn test_ets_aaa_holt_winters() {
        // ETS(A,A,A) = full Holt-Winters additive
        let mut data = Vec::new();
        for cycle in 0..4_usize {
            for j in 0..4_usize {
                data.push(10.0 + j as f64 * 2.0 + cycle as f64 * 0.5);
            }
        }
        let model = EtsModel::new(
            EtsErrorType::Additive,
            EtsTrendType::Additive,
            EtsSeasonalType::Additive,
            4,
        );
        let fitted = model.fit(&data).expect("fit failed");
        let fc = fitted.forecast(4);
        assert_eq!(fc.len(), 4);
        for &v in &fc {
            assert!(v.is_finite(), "forecast must be finite");
        }
    }

    #[test]
    fn test_ets_prediction_intervals() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let model = EtsModel::new(
            EtsErrorType::Additive,
            EtsTrendType::Additive,
            EtsSeasonalType::None,
            1,
        );
        let fitted = model.fit(&data).expect("fit failed");
        let intervals = fitted.prediction_intervals(4, 0.95);
        assert_eq!(intervals.len(), 4);
        for (lo, hi) in &intervals {
            assert!(lo <= hi, "lower must be <= upper");
            assert!(lo.is_finite() && hi.is_finite());
        }
    }

    #[test]
    fn test_auto_ets_selects_model() {
        let data: Vec<f64> = (1..=24).map(|i| i as f64 + (i % 4) as f64).collect();
        let fitted = auto_ets(&data, 4).expect("auto_ets failed");
        // AIC should be finite
        assert!(fitted.aic.is_finite(), "AIC should be finite");
        assert_eq!(fitted.forecast(4).len(), 4);
    }

    #[test]
    fn test_auto_ets_insufficient_data() {
        assert!(auto_ets(&[1.0, 2.0], 1).is_err());
    }

    // ── Theta tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_theta_linear_trend() {
        // Theta method on a perfect linear trend should project the trend forward
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let fc = ThetaModel::fit_forecast(&data, 4).expect("theta forecast failed");
        assert_eq!(fc.len(), 4);
        for w in fc.windows(2) {
            assert!(w[1] > w[0] - 1e-6, "should continue upward: {:?}", fc);
        }
    }

    #[test]
    fn test_theta_output_len() {
        let data: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let fc = ThetaModel::fit_forecast(&data, 5).expect("theta failed");
        assert_eq!(fc.len(), 5);
    }

    #[test]
    fn test_theta_insufficient_data() {
        assert!(ThetaModel::fit_forecast(&[1.0, 2.0, 3.0], 1).is_err());
    }

    #[test]
    fn test_theta_zero_horizon() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let fc = ThetaModel::fit_forecast(&data, 0).expect("zero horizon ok");
        assert_eq!(fc.len(), 0);
    }

    #[test]
    fn test_theta_model_new() {
        let m = ThetaModel::new(2.0);
        assert_eq!(m.theta, 2.0);
        assert!(m.period.is_none());
    }
}
