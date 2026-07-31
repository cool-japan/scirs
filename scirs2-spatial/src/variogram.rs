//! Variogram analysis for spatial statistics
//!
//! This module provides tools for variogram analysis, which is fundamental to
//! geostatistics and spatial interpolation methods like kriging.
//!
//! # Features
//!
//! * **Experimental variogram computation** - Calculate empirical variograms from data
//! * **Theoretical variogram models** - Fit standard models (spherical, exponential, Gaussian, etc.)
//! * **Variogram fitting** - Optimize model parameters
//! * **Directional variograms** - Anisotropic spatial correlation analysis
//! * **Cross-variograms** - Multivariate spatial correlation
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_spatial::variogram::{experimental_variogram, VariogramModel, fit_variogram};
//!
//! // Create spatial data
//! let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//! let values = array![1.0, 2.0, 1.5, 2.5];
//!
//! // Compute experimental variogram
//! let (lags, gamma) = experimental_variogram(
//!     &coords.view(),
//!     &values.view(),
//!     10,   // number of lag bins
//!     None  // automatic lag tolerance
//! ).expect("Failed to compute variogram");
//!
//! // Fit theoretical model
//! let model = fit_variogram(&lags, &gamma, VariogramModel::Spherical)
//!     .expect("Failed to fit model");
//!
//! println!("Fitted parameters: range={:.2}, sill={:.2}, nugget={:.2}",
//!          model.range, model.sill, model.nugget);
//! ```

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use std::f64::consts::PI;

/// Variogram model types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VariogramModel {
    /// Spherical model: γ(h) = nugget + sill * [1.5(h/range) - 0.5(h/range)³] for h < range
    Spherical,
    /// Exponential model: γ(h) = nugget + sill * [1 - exp(-h/range)]
    Exponential,
    /// Gaussian model: γ(h) = nugget + sill * [1 - exp(-(h/range)²)]
    Gaussian,
    /// Linear model: γ(h) = nugget + slope * h
    Linear,
    /// Power model: γ(h) = nugget + scale * h^power
    Power,
    /// Matérn model with smoothness parameter
    Matern,
}

/// Fitted variogram parameters
#[derive(Debug, Clone)]
pub struct FittedVariogram<T: Float> {
    /// Model type
    pub model: VariogramModel,
    /// Range parameter (distance at which correlation becomes negligible)
    pub range: T,
    /// Sill parameter (variance at infinite distance)
    pub sill: T,
    /// Nugget parameter (variance at zero distance, measurement error)
    pub nugget: T,
    /// Additional parameters for specific models
    pub extra_params: Vec<T>,
    /// Goodness of fit (R²)
    pub r_squared: T,
}

impl<T: Float> FittedVariogram<T> {
    /// Evaluate the variogram model at a given distance
    pub fn evaluate(&self, distance: T) -> T {
        match self.model {
            VariogramModel::Spherical => {
                if distance >= self.range {
                    self.nugget + self.sill
                } else {
                    let h_over_r = distance / self.range;
                    let three_halves = T::from(1.5).expect("conversion failed");
                    let half = T::from(0.5).expect("conversion failed");
                    self.nugget
                        + self.sill
                            * (three_halves * h_over_r - half * h_over_r * h_over_r * h_over_r)
                }
            }
            VariogramModel::Exponential => {
                self.nugget + self.sill * (T::one() - (-distance / self.range).exp())
            }
            VariogramModel::Gaussian => {
                let h_over_r = distance / self.range;
                self.nugget + self.sill * (T::one() - (-(h_over_r * h_over_r)).exp())
            }
            VariogramModel::Linear => {
                let slope = if !self.extra_params.is_empty() {
                    self.extra_params[0]
                } else {
                    self.sill / self.range
                };
                self.nugget + slope * distance
            }
            VariogramModel::Power => {
                let power = if !self.extra_params.is_empty() {
                    self.extra_params[0]
                } else {
                    T::from(0.5).expect("conversion failed")
                };
                self.nugget + self.sill * distance.powf(power)
            }
            VariogramModel::Matern => {
                let nu = if !self.extra_params.is_empty() {
                    self.extra_params[0]
                } else {
                    T::from(1.5).expect("conversion failed") // Default smoothness
                };
                // Simplified Matérn for common nu values
                if distance.is_zero() {
                    self.nugget
                } else {
                    let scaled_dist = distance / self.range
                        * T::from(2.0).expect("conversion failed")
                        * nu.sqrt();
                    // Approximation for nu = 1.5 (most common)
                    let term = (T::one() + scaled_dist) * (-scaled_dist).exp();
                    self.nugget + self.sill * (T::one() - term)
                }
            }
        }
    }
}

/// Compute experimental (empirical) variogram from spatial data
///
/// # Arguments
///
/// * `coordinates` - Spatial coordinates of observations (n × d array)
/// * `values` - Observed values at each location (n-vector)
/// * `n_lags` - Number of lag distance bins
/// * `lag_tolerance` - Tolerance for binning distances (None for automatic)
///
/// # Returns
///
/// * Tuple of (lag distances, variogram values)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::variogram::experimental_variogram;
///
/// let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
/// let values = array![1.0, 2.0, 1.5, 2.5];
///
/// let (lags, gamma) = experimental_variogram(&coords.view(), &values.view(), 10, None)
///     .expect("Failed to compute variogram");
/// ```
pub fn experimental_variogram<T: Float>(
    coordinates: &ArrayView2<T>,
    values: &ArrayView1<T>,
    n_lags: usize,
    lag_tolerance: Option<T>,
) -> SpatialResult<(Array1<T>, Array1<T>)> {
    let n = coordinates.shape()[0];

    if n != values.len() {
        return Err(SpatialError::DimensionError(
            "Number of coordinates must match number of values".to_string(),
        ));
    }

    if n < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 points for variogram".to_string(),
        ));
    }

    // Compute all pairwise distances and squared differences
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist_sq = T::zero();
            for k in 0..coordinates.shape()[1] {
                let diff = coordinates[[i, k]] - coordinates[[j, k]];
                dist_sq = dist_sq + diff * diff;
            }
            let distance = dist_sq.sqrt();

            let value_diff = values[i] - values[j];
            let gamma = value_diff * value_diff / (T::one() + T::one()); // γ = 0.5 * (z_i - z_j)²

            pairs.push((distance, gamma));
        }
    }

    if pairs.is_empty() {
        return Err(SpatialError::ValueError("No valid pairs found".to_string()));
    }

    // Find maximum distance
    let max_distance = pairs
        .iter()
        .map(|(d, _)| *d)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| SpatialError::ValueError("Failed to find max distance".to_string()))?;

    // Determine lag size
    let lag_size = max_distance / T::from(n_lags).expect("conversion failed");
    let tolerance = lag_tolerance.unwrap_or(lag_size / (T::one() + T::one()));

    // Bin pairs into lags
    let mut lag_bins: Vec<Vec<T>> = vec![Vec::new(); n_lags];
    let mut lag_centers = Array1::zeros(n_lags);

    for i in 0..n_lags {
        let lag_center = lag_size
            * (T::from(i).expect("conversion failed") + T::from(0.5).expect("conversion failed"));
        lag_centers[i] = lag_center;

        for &(distance, gamma) in &pairs {
            if (distance - lag_center).abs() <= tolerance {
                lag_bins[i].push(gamma);
            }
        }
    }

    // Compute mean variogram value for each lag
    let mut gamma_values = Array1::zeros(n_lags);
    let mut valid_lags = Vec::new();
    let mut valid_gammas = Vec::new();

    for i in 0..n_lags {
        if !lag_bins[i].is_empty() {
            let sum: T = lag_bins[i]
                .iter()
                .copied()
                .fold(T::zero(), |acc, x| acc + x);
            let mean = sum / T::from(lag_bins[i].len()).expect("conversion failed");
            gamma_values[i] = mean;
            valid_lags.push(lag_centers[i]);
            valid_gammas.push(mean);
        }
    }

    if valid_lags.is_empty() {
        return Err(SpatialError::ValueError(
            "No valid lags computed".to_string(),
        ));
    }

    // Convert to arrays
    let lags_array = Array1::from_vec(valid_lags);
    let gamma_array = Array1::from_vec(valid_gammas);

    Ok((lags_array, gamma_array))
}

/// Fit a theoretical variogram model to experimental data
///
/// Uses nonlinear least squares (Levenberg-Marquardt) to find the model
/// parameters -- nugget, sill, range, and any model-specific extra
/// parameter -- that minimize the sum of squared residuals between the
/// model and the experimental variogram values. The heuristic estimates
/// computed up front only seed the optimizer's initial guess; they are
/// refined by the nonlinear solver before being returned.
///
/// # Arguments
///
/// * `lags` - Lag distances from experimental variogram
/// * `gamma` - Variogram values at each lag
/// * `model` - Type of variogram model to fit
///
/// # Returns
///
/// * Fitted variogram with optimized parameters
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::variogram::{fit_variogram, VariogramModel};
///
/// let lags = array![0.5, 1.0, 1.5, 2.0];
/// let gamma = array![0.1, 0.4, 0.7, 0.9];
///
/// let fitted = fit_variogram(&lags, &gamma, VariogramModel::Spherical)
///     .expect("Failed to fit");
/// ```
pub fn fit_variogram<T: Float>(
    lags: &Array1<T>,
    gamma: &Array1<T>,
    model: VariogramModel,
) -> SpatialResult<FittedVariogram<T>> {
    if lags.len() != gamma.len() {
        return Err(SpatialError::DimensionError(
            "Lags and gamma must have same length".to_string(),
        ));
    }

    if lags.is_empty() {
        return Err(SpatialError::ValueError(
            "Need at least one lag-gamma pair".to_string(),
        ));
    }

    let zero = T::zero();
    let one = T::one();

    // Initial parameter estimates -- these only seed the nonlinear
    // optimizer below; they are never returned as the "fitted" values.
    let max_lag = lags
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| SpatialError::ValueError("Failed to find max lag".to_string()))?;

    let max_gamma = gamma
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| SpatialError::ValueError("Failed to find max gamma".to_string()))?;

    // Scale references used to build a strictly-positive lower bound for
    // any parameter (range, sill, power, ...) that appears as a divisor or
    // under `powf` in `FittedVariogram::evaluate` -- letting those hit
    // exactly zero during optimization would produce NaN/Inf.
    let safe_max_lag = if max_lag > zero { max_lag } else { one };
    let safe_max_gamma = if max_gamma > zero { max_gamma } else { one };
    let positive_floor =
        (safe_max_lag.min(safe_max_gamma) * float_lit(1e-6, zero)).max(float_lit(1e-30, zero));

    let initial_range = (max_lag * float_lit(0.7, zero)).max(positive_floor);
    let initial_sill = (max_gamma * float_lit(0.9, zero)).max(positive_floor);
    let initial_nugget = (gamma[0] * float_lit(0.1, zero)).max(zero);

    let (mut params, lower_bounds) = initial_params(
        model,
        initial_nugget,
        initial_sill,
        initial_range,
        positive_floor,
    );

    levenberg_marquardt_fit(model, lags, gamma, &mut params, &lower_bounds);

    let mut fitted = params_to_fitted(model, &params);

    // Compute R² for goodness of fit against the *converged* parameters.
    let mean_gamma = gamma.sum() / T::from(gamma.len()).unwrap_or(one);
    let mut ss_res = zero;
    let mut ss_tot = zero;

    for i in 0..lags.len() {
        let predicted = fitted.evaluate(lags[i]);
        let residual = gamma[i] - predicted;
        ss_res = ss_res + residual * residual;

        let deviation = gamma[i] - mean_gamma;
        ss_tot = ss_tot + deviation * deviation;
    }

    fitted.r_squared = if ss_tot > zero {
        one - ss_res / ss_tot
    } else {
        zero
    };

    Ok(fitted)
}

/// Convert an `f64` literal to `T`, falling back to `fallback` instead of
/// panicking in the (practically unreachable for `f32`/`f64`) case the
/// conversion fails.
fn float_lit<T: Float>(x: f64, fallback: T) -> T {
    T::from(x).unwrap_or(fallback)
}

/// Build the initial nonlinear-least-squares parameter vector for `model`
/// from the heuristic starting estimates, along with the lower bound each
/// parameter must stay at or above throughout optimization.
///
/// The parameter vector layout depends on the model: most models optimize
/// `[nugget, sill, range]` directly, while `Linear` (which has no `range`
/// in `evaluate`) optimizes `[nugget, slope]` and `Power` (whose exponent
/// is the meaningful free parameter) optimizes `[nugget, scale, power]`.
fn initial_params<T: Float>(
    model: VariogramModel,
    nugget0: T,
    sill0: T,
    range0: T,
    positive_floor: T,
) -> (Vec<T>, Vec<T>) {
    let zero = T::zero();
    match model {
        VariogramModel::Linear => {
            let slope0 = if range0 > zero { sill0 / range0 } else { sill0 };
            (
                vec![nugget0, slope0.max(positive_floor)],
                vec![zero, positive_floor],
            )
        }
        VariogramModel::Power => {
            let power0 = float_lit(0.5, T::one());
            (
                vec![nugget0, sill0.max(positive_floor), power0],
                vec![zero, positive_floor, positive_floor],
            )
        }
        _ => (
            vec![
                nugget0,
                sill0.max(positive_floor),
                range0.max(positive_floor),
            ],
            vec![zero, positive_floor, positive_floor],
        ),
    }
}

/// Reconstruct a [`FittedVariogram`] from the internal optimizer parameter
/// vector produced by [`initial_params`] / refined by
/// [`levenberg_marquardt_fit`]. `r_squared` is left at zero here; the
/// caller recomputes it once against the converged parameters.
fn params_to_fitted<T: Float>(model: VariogramModel, params: &[T]) -> FittedVariogram<T> {
    match model {
        VariogramModel::Linear => FittedVariogram {
            model,
            range: T::one(),
            sill: params[1],
            nugget: params[0],
            extra_params: vec![params[1]],
            r_squared: T::zero(),
        },
        VariogramModel::Power => FittedVariogram {
            model,
            range: T::one(),
            sill: params[1],
            nugget: params[0],
            extra_params: vec![params[2]],
            r_squared: T::zero(),
        },
        _ => FittedVariogram {
            model,
            range: params[2],
            sill: params[1],
            nugget: params[0],
            extra_params: vec![],
            r_squared: T::zero(),
        },
    }
}

/// Refine `params` in place via Levenberg-Marquardt nonlinear least
/// squares, minimizing `sum((gamma_i - model(lags_i; params))^2)` subject
/// to each parameter staying at or above the matching entry in
/// `lower_bounds`.
///
/// Uses a central-difference Jacobian of the model function (cheap here:
/// at most 3 free parameters and typically a few dozen lag bins) together
/// with the standard Marquardt diagonal-scaling damping strategy, and
/// projects each accepted step back onto the box constraints.
fn levenberg_marquardt_fit<T: Float>(
    model: VariogramModel,
    lags: &Array1<T>,
    gamma: &Array1<T>,
    params: &mut Vec<T>,
    lower_bounds: &[T],
) {
    let n = params.len();
    let m = lags.len();
    if m == 0 || n == 0 {
        return;
    }

    let zero = T::zero();
    let one = T::one();

    let evaluate_cost = |p: &[T]| -> (T, Vec<T>) {
        let candidate = params_to_fitted(model, p);
        let mut residuals = Vec::with_capacity(m);
        let mut sse = zero;
        for i in 0..m {
            let r = gamma[i] - candidate.evaluate(lags[i]);
            sse = sse + r * r;
            residuals.push(r);
        }
        (sse, residuals)
    };

    let relative_step = float_lit(1e-6, zero);
    let mut lambda = float_lit(1e-3, zero);
    let lambda_up = float_lit(10.0, one);
    let lambda_down = float_lit(0.1, one);
    let min_lambda = float_lit(1e-12, zero);
    let max_iters = 200;
    let max_lambda_attempts = 30;

    let (mut current_cost, mut residuals) = evaluate_cost(params);

    for _ in 0..max_iters {
        // Central-difference Jacobian: jac[i * n + j] = d(model_i)/d(param_j)
        let mut jac = vec![zero; m * n];
        for j in 0..n {
            let base = params[j].abs();
            let step = if base > zero {
                base * relative_step
            } else {
                relative_step
            };
            let mut p_plus = params.clone();
            let mut p_minus = params.clone();
            p_plus[j] = p_plus[j] + step;
            p_minus[j] = p_minus[j] - step;

            let cand_plus = params_to_fitted(model, &p_plus);
            let cand_minus = params_to_fitted(model, &p_minus);
            let two_step = step + step;
            for i in 0..m {
                let f_plus = cand_plus.evaluate(lags[i]);
                let f_minus = cand_minus.evaluate(lags[i]);
                jac[i * n + j] = (f_plus - f_minus) / two_step;
            }
        }

        // Normal equations: (JᵀJ + λ·diag(JᵀJ)) δ = Jᵀ·residuals
        let mut jtj = vec![zero; n * n];
        let mut jtr = vec![zero; n];
        for i in 0..m {
            for a in 0..n {
                jtr[a] = jtr[a] + jac[i * n + a] * residuals[i];
                for b in 0..n {
                    jtj[a * n + b] = jtj[a * n + b] + jac[i * n + a] * jac[i * n + b];
                }
            }
        }

        let mut attempt_lambda = lambda;
        let mut improved = false;

        for _ in 0..max_lambda_attempts {
            let mut damped = jtj.clone();
            for d in 0..n {
                let diag = jtj[d * n + d];
                damped[d * n + d] = if diag > zero {
                    diag * (one + attempt_lambda)
                } else {
                    attempt_lambda.max(min_lambda)
                };
            }

            if let Some(delta) = solve_linear_system(&damped, &jtr, n) {
                let mut candidate_params = params.clone();
                for (k, slot) in candidate_params.iter_mut().enumerate() {
                    let updated = *slot + delta[k];
                    *slot = if updated < lower_bounds[k] {
                        lower_bounds[k]
                    } else {
                        updated
                    };
                }

                let (candidate_cost, candidate_residuals) = evaluate_cost(&candidate_params);
                if candidate_cost < current_cost {
                    *params = candidate_params;
                    residuals = candidate_residuals;
                    current_cost = candidate_cost;
                    lambda = (attempt_lambda * lambda_down).max(min_lambda);
                    improved = true;
                    break;
                }
            }

            attempt_lambda = attempt_lambda * lambda_up;
        }

        if !improved {
            // No damping level improved the fit any further: a stationary
            // point (up to numerical precision) has been reached.
            break;
        }
    }
}

/// Solve the dense `n x n` linear system `a * x = b` via Gaussian
/// elimination with partial pivoting. `a` is row-major. Returns `None` if
/// the system is numerically singular.
fn solve_linear_system<T: Float>(a: &[T], b: &[T], n: usize) -> Option<Vec<T>> {
    let zero = T::zero();
    let pivot_floor = float_lit(1e-300, zero);
    let mut aug = vec![zero; n * (n + 1)];
    for i in 0..n {
        aug[i * (n + 1)..i * (n + 1) + n].copy_from_slice(&a[i * n..i * n + n]);
        aug[i * (n + 1) + n] = b[i];
    }

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = row;
            }
        }
        if pivot_val <= pivot_floor {
            return None;
        }
        if pivot_row != col {
            for k in 0..(n + 1) {
                aug.swap(col * (n + 1) + k, pivot_row * (n + 1) + k);
            }
        }

        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            if factor.is_zero() {
                continue;
            }
            for k in col..(n + 1) {
                let sub = factor * aug[col * (n + 1) + k];
                aug[row * (n + 1) + k] = aug[row * (n + 1) + k] - sub;
            }
        }
    }

    let mut x = vec![zero; n];
    for row in (0..n).rev() {
        let mut sum = aug[row * (n + 1) + n];
        for (col, &xc) in x.iter().enumerate().take(n).skip(row + 1) {
            sum = sum - aug[row * (n + 1) + col] * xc;
        }
        let diag = aug[row * (n + 1) + row];
        if diag.abs() <= pivot_floor {
            return None;
        }
        x[row] = sum / diag;
    }

    Some(x)
}

/// Compute directional (anisotropic) variogram
///
/// Computes experimental variogram for a specific direction, useful for
/// detecting directional trends in spatial correlation.
///
/// # Arguments
///
/// * `coordinates` - Spatial coordinates (must be 2D)
/// * `values` - Observed values
/// * `direction` - Direction angle in radians (0 = East, π/2 = North)
/// * `tolerance` - Angular tolerance in radians
/// * `n_lags` - Number of lag bins
///
/// # Returns
///
/// * Tuple of (lag distances, variogram values)
pub fn directional_variogram<T: Float>(
    coordinates: &ArrayView2<T>,
    values: &ArrayView1<T>,
    direction: T,
    tolerance: T,
    n_lags: usize,
) -> SpatialResult<(Array1<T>, Array1<T>)> {
    let n = coordinates.shape()[0];

    if coordinates.shape()[1] != 2 {
        return Err(SpatialError::DimensionError(
            "Directional variogram requires 2D coordinates".to_string(),
        ));
    }

    if n != values.len() {
        return Err(SpatialError::DimensionError(
            "Number of coordinates must match number of values".to_string(),
        ));
    }

    // Compute pairwise distances and angles
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coordinates[[j, 0]] - coordinates[[i, 0]];
            let dy = coordinates[[j, 1]] - coordinates[[i, 1]];

            let distance = (dx * dx + dy * dy).sqrt();
            let angle = dy.atan2(dx); // atan2(dy, dx) gives angle from East

            // Check if angle matches direction within tolerance
            let angle_diff = (angle - direction).abs();
            let pi_t = T::from(PI).expect("conversion failed");
            let angle_diff_wrapped = if angle_diff > pi_t {
                (T::one() + T::one()) * pi_t - angle_diff
            } else {
                angle_diff
            };

            if angle_diff_wrapped <= tolerance {
                let value_diff = values[i] - values[j];
                let gamma = value_diff * value_diff / (T::one() + T::one());
                pairs.push((distance, gamma));
            }
        }
    }

    if pairs.is_empty() {
        return Err(SpatialError::ValueError(
            "No pairs found in specified direction".to_string(),
        ));
    }

    // Rest of variogram computation similar to experimental_variogram
    // (simplified for brevity)
    let max_distance = pairs
        .iter()
        .map(|(d, _)| *d)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| SpatialError::ValueError("Failed to find max distance".to_string()))?;

    let lag_size = max_distance / T::from(n_lags).expect("conversion failed");
    let mut lag_bins: Vec<Vec<T>> = vec![Vec::new(); n_lags];
    let mut lag_centers = Array1::zeros(n_lags);

    for i in 0..n_lags {
        let lag_center = lag_size
            * (T::from(i).expect("conversion failed") + T::from(0.5).expect("conversion failed"));
        lag_centers[i] = lag_center;

        for &(distance, gamma) in &pairs {
            let lag_tolerance = lag_size / (T::one() + T::one());
            if (distance - lag_center).abs() <= lag_tolerance {
                lag_bins[i].push(gamma);
            }
        }
    }

    let mut valid_lags = Vec::new();
    let mut valid_gammas = Vec::new();

    for i in 0..n_lags {
        if !lag_bins[i].is_empty() {
            let sum: T = lag_bins[i]
                .iter()
                .copied()
                .fold(T::zero(), |acc, x| acc + x);
            let mean = sum / T::from(lag_bins[i].len()).expect("conversion failed");
            valid_lags.push(lag_centers[i]);
            valid_gammas.push(mean);
        }
    }

    Ok((Array1::from_vec(valid_lags), Array1::from_vec(valid_gammas)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_experimental_variogram() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let values = array![1.0, 2.0, 1.5, 2.5];

        let result = experimental_variogram(&coords.view(), &values.view(), 5, None);
        assert!(result.is_ok());

        let (lags, gamma) = result.expect("computation failed");
        assert!(!lags.is_empty());
        assert_eq!(lags.len(), gamma.len());

        // All gamma values should be non-negative
        for &g in gamma.iter() {
            assert!(g >= 0.0);
        }
    }

    #[test]
    fn test_fit_spherical_variogram() {
        let lags = array![0.5, 1.0, 1.5, 2.0, 2.5];
        let gamma = array![0.1, 0.3, 0.6, 0.85, 0.95];

        let fitted = fit_variogram(&lags, &gamma, VariogramModel::Spherical);
        assert!(fitted.is_ok());

        let model = fitted.expect("fitting failed");
        assert!(model.range > 0.0);
        assert!(model.sill > 0.0);
        assert!(model.nugget >= 0.0);
    }

    #[test]
    fn test_fit_exponential_variogram() {
        let lags = array![0.5, 1.0, 1.5, 2.0, 2.5];
        let gamma = array![0.2, 0.4, 0.6, 0.75, 0.85];

        let fitted = fit_variogram(&lags, &gamma, VariogramModel::Exponential);
        assert!(fitted.is_ok());

        let model = fitted.expect("fitting failed");
        assert!(model.range > 0.0);
        assert!(model.sill > 0.0);
    }

    /// Regression test for the fitting stub: `fit_variogram` used to return
    /// `range = max_lag * 0.7`, `sill = max_gamma * 0.9`, and
    /// `nugget = gamma[0] * 0.1` verbatim, completely ignoring the shape of
    /// the (non-constant) input data. Here the "experimental" data is
    /// generated exactly from a known spherical model, so a real
    /// least-squares fit must recover parameters close to the ground
    /// truth -- while the old heuristic would have returned range=21.0
    /// (vs. true 12.0) and nugget=~0.047 (vs. true 0.5).
    #[test]
    fn test_fit_variogram_recovers_known_parameters() {
        let true_model = FittedVariogram {
            model: VariogramModel::Spherical,
            range: 12.0,
            sill: 4.0,
            nugget: 0.5,
            extra_params: vec![],
            r_squared: 1.0,
        };

        let lags = array![
            0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 18.0, 22.0, 30.0
        ];
        let gamma = lags.mapv(|h| true_model.evaluate(h));

        let fitted = fit_variogram(&lags, &gamma, VariogramModel::Spherical)
            .expect("fit should succeed on noiseless synthetic data");

        // The old heuristic-only stub would give range = max_lag*0.7 = 21.0,
        // sill = max_gamma*0.9, nugget = gamma[0]*0.1 -- all far from the
        // ground truth used to generate this data.
        assert_relative_eq!(fitted.range, 12.0, epsilon = 0.3);
        assert_relative_eq!(fitted.sill, 4.0, epsilon = 0.1);
        assert_relative_eq!(fitted.nugget, 0.5, epsilon = 0.1);
        assert!(
            fitted.r_squared > 0.999,
            "expected near-perfect fit on noiseless data, got r_squared = {}",
            fitted.r_squared
        );
    }

    #[test]
    fn test_variogram_evaluate() {
        let fitted = FittedVariogram {
            model: VariogramModel::Spherical,
            range: 2.0,
            sill: 1.0,
            nugget: 0.1,
            extra_params: vec![],
            r_squared: 0.95,
        };

        // At zero distance, should be close to nugget
        let gamma_0 = fitted.evaluate(0.0);
        assert_relative_eq!(gamma_0, 0.1, epsilon = 0.01);

        // At range, should approach nugget + sill
        let gamma_range = fitted.evaluate(2.0);
        assert!(gamma_range >= 1.0);
        assert!(gamma_range <= 1.2);

        // Beyond range, should be nugget + sill
        let gamma_beyond = fitted.evaluate(5.0);
        assert_relative_eq!(gamma_beyond, 1.1, epsilon = 0.01);
    }

    #[test]
    fn test_directional_variogram() {
        let coords = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0]
        ];
        let values = array![1.0, 1.5, 2.0, 1.2, 1.7, 2.2];

        // East direction (0 radians)
        let result = directional_variogram(
            &coords.view(),
            &values.view(),
            0.0,
            std::f64::consts::PI / 4.0, // 45 degree tolerance
            5,
        );

        assert!(result.is_ok());
        let (lags, gamma) = result.expect("computation failed");
        assert!(!lags.is_empty());
    }

    #[test]
    fn test_variogram_models() {
        let models = vec![
            VariogramModel::Spherical,
            VariogramModel::Exponential,
            VariogramModel::Gaussian,
            VariogramModel::Linear,
        ];

        for model in models {
            let fitted = FittedVariogram {
                model,
                range: 1.0,
                sill: 1.0,
                nugget: 0.0,
                extra_params: vec![],
                r_squared: 0.0,
            };

            // All models should be monotonically increasing
            let gamma_1 = fitted.evaluate(0.5);
            let gamma_2 = fitted.evaluate(1.0);
            assert!(gamma_2 >= gamma_1);
        }
    }
}
