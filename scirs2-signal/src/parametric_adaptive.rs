use scirs2_core::ndarray::s;
// Adaptive parametric spectral estimation methods
//
// This module implements advanced adaptive algorithms for parametric spectral
// estimation including:
// - Recursive Least Squares (RLS) for time-varying spectra
// - Kalman filter-based parameter estimation
// - Variable order selection
// - Online model updating

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::validation::{check_finite, check_positive};

#[allow(unused_imports)]
/// Adaptive AR model with time-varying parameters
#[derive(Debug, Clone)]
pub struct AdaptiveArModel {
    /// Current AR coefficients
    pub coefficients: Array1<f64>,
    /// Model order
    pub order: usize,
    /// Estimation error variance
    pub variance: f64,
    /// Forgetting factor (0 < lambda <= 1)
    pub forgetting_factor: f64,
    /// Adaptation gain
    pub gain: Array2<f64>,
    /// Internal state buffer
    state_buffer: Vec<f64>,
    /// Adaptation method
    method: AdaptiveMethod,
}

/// Adaptive estimation method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdaptiveMethod {
    /// Recursive Least Squares
    RLS,
    /// Kalman filter
    Kalman,
    /// Least Mean Squares
    LMS,
    /// Normalized LMS
    NLMS,
}

/// Configuration for adaptive estimation
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Initial model order
    pub initial_order: usize,
    /// Maximum model order
    pub max_order: usize,
    /// Forgetting factor for RLS (0.95-0.99 typical)
    pub forgetting_factor: f64,
    /// Adaptation step size for LMS
    pub step_size: f64,
    /// Estimation method
    pub method: AdaptiveMethod,
    /// Enable variable order selection
    pub variable_order: bool,
    /// Order selection threshold
    pub order_threshold: f64,
    /// Process noise covariance (for Kalman)
    pub process_noise: f64,
    /// Measurement noise covariance (for Kalman)
    pub measurement_noise: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            initial_order: 4,
            max_order: 20,
            forgetting_factor: 0.98,
            step_size: 0.01,
            method: AdaptiveMethod::RLS,
            variable_order: false,
            order_threshold: 0.01,
            process_noise: 1e-4,
            measurement_noise: 1.0,
        }
    }
}

/// Initialize adaptive AR model
///
/// # Arguments
///
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * Initialized adaptive model
#[allow(dead_code)]
pub fn initialize_adaptive_ar(config: &AdaptiveConfig) -> SignalResult<AdaptiveArModel> {
    check_positive(_config.initial_order, "initial_order")?;
    check_positive(_config.forgetting_factor, "forgetting_factor")?;

    if config.forgetting_factor > 1.0 {
        return Err(SignalError::ValueError(
            "Forgetting factor must be <= 1.0".to_string(),
        ));
    }

    let order = config.initial_order;
    let coefficients = Array1::zeros(order);

    // Initialize gain matrix based on method
    let gain = match config.method {
        AdaptiveMethod::RLS => {
            // P = delta * I (large initial value)
            Array2::eye(order) * 1000.0
        }
        AdaptiveMethod::Kalman => {
            // Initialize with process noise covariance
            Array2::eye(order) * config.process_noise
        }
        _ => Array2::zeros((order, order)), // Not used for LMS/NLMS
    };

    Ok(AdaptiveArModel {
        coefficients,
        order,
        variance: 1.0,
        forgetting_factor: config.forgetting_factor,
        gain,
        state_buffer: vec![0.0; config.max_order],
        method: config.method,
    })
}

/// Update adaptive AR model with new sample
///
/// # Arguments
///
/// * `model` - Adaptive AR model
/// * `sample` - New signal sample
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * Prediction error
#[allow(dead_code)]
pub fn update_adaptive_ar(
    model: &mut AdaptiveArModel,
    sample: f64,
    config: &AdaptiveConfig,
) -> SignalResult<f64> {
    check_finite(sample, "sample value")?;

    // Predict using current model
    let prediction = predict_next_sample(model)?;
    let error = sample - prediction;

    // Update state buffer
    for i in (1..model.state_buffer.len()).rev() {
        model.state_buffer[i] = model.state_buffer[i - 1];
    }
    model.state_buffer[0] = sample;

    // Update model parameters based on method
    match model.method {
        AdaptiveMethod::RLS => update_rls(model, error, config.forgetting_factor)?,
        AdaptiveMethod::Kalman => update_kalman(model, error, config)?,
        AdaptiveMethod::LMS => update_lms(model, error, config.step_size)?,
        AdaptiveMethod::NLMS => update_nlms(model, error, config.step_size)?,
    }

    // Update error variance estimate
    let alpha = 0.01; // Smoothing factor
    model.variance = (1.0 - alpha) * model.variance + alpha * error * error;

    // Variable order selection if enabled
    if config.variable_order {
        update_model_order(model, config)?;
    }

    Ok(error)
}

/// Predict next sample using current model
#[allow(dead_code)]
fn predict_next_sample(model: &AdaptiveArModel) -> SignalResult<f64> {
    let mut prediction = 0.0;

    for i in 0.._model.order {
        if i < model.state_buffer.len() {
            prediction += model.coefficients[i] * model.state_buffer[i];
        }
    }

    Ok(prediction)
}

/// Update using Recursive Least Squares
#[allow(dead_code)]
fn update_rls(model: &mut AdaptiveArModel, error: f64, lambda: f64) -> SignalResult<()> {
    let order = model.order;
    let x = Array1::from_vec(_model.state_buffer[..order].to_vec());

    // Compute gain vector: k = P * x / (lambda + x' * P * x)
    let px = model.gain.dot(&x);
    let denominator = lambda + x.dot(&px);

    if denominator.abs() < 1e-10 {
        return Ok(()); // Skip update if denominator too small
    }

    let k = px / denominator;

    // Update coefficients: a = a + k * error
    model.coefficients = &_model.coefficients + &k * error;

    // Update gain matrix: P = (P - k * x' * P) / lambda
    let outer_product = k
        .view()
        .insert_axis(scirs2_core::ndarray::Axis(1))
        .dot(&x.view().insert_axis(scirs2_core::ndarray::Axis(0)));
    model.gain = (&_model.gain - &outer_product.dot(&_model.gain)) / lambda;

    Ok(())
}

/// Update using Kalman filter
#[allow(dead_code)]
fn update_kalman(
    model: &mut AdaptiveArModel,
    error: f64,
    config: &AdaptiveConfig,
) -> SignalResult<()> {
    let order = model.order;
    let x = Array1::from_vec(model.state_buffer[..order].to_vec());

    // Prediction step
    // P = P + Q (add process noise)
    for i in 0..order {
        model.gain[[i, i]] += config.process_noise;
    }

    // Update step
    // K = P * H' / (H * P * H' + R)
    let px = model.gain.dot(&x);
    let s = x.dot(&px) + config.measurement_noise;

    if s.abs() < 1e-10 {
        return Ok(());
    }

    let k = px / s;

    // Update state estimate
    model.coefficients = &model.coefficients + &k * error;

    // Update error covariance
    let i_minus_kh = Array2::eye(order)
        - k.view()
            .insert_axis(scirs2_core::ndarray::Axis(1))
            .dot(&x.view().insert_axis(scirs2_core::ndarray::Axis(0)));
    model.gain = i_minus_kh.dot(&model.gain);

    Ok(())
}

/// Update using Least Mean Squares
#[allow(dead_code)]
fn update_lms(_model: &mut AdaptiveArModel, error: f64, stepsize: f64) -> SignalResult<()> {
    let order = model.order;
    let x = Array1::from_vec(_model.state_buffer[..order].to_vec());

    // Simple gradient update: a = a + mu * error * x
    model.coefficients = &_model.coefficients + step_size * error * &x;

    Ok(())
}

/// Update using Normalized LMS
#[allow(dead_code)]
fn update_nlms(_model: &mut AdaptiveArModel, error: f64, stepsize: f64) -> SignalResult<()> {
    let order = model.order;
    let x = Array1::from_vec(_model.state_buffer[..order].to_vec());

    // Compute normalization factor
    let x_view = x.view();
    let norm_sq = f64::simd_dot(&x_view, &x_view);
    let epsilon = 1e-10;

    // Normalized update: a = a + (mu / (||x||^2 + epsilon)) * error * x
    let normalized_step = step_size / (norm_sq + epsilon);
    model.coefficients = &_model.coefficients + normalized_step * error * &x;

    Ok(())
}

/// Update model order based on prediction performance
#[allow(dead_code)]
fn update_model_order(model: &mut AdaptiveArModel, config: &AdaptiveConfig) -> SignalResult<()> {
    // Simple criterion: increase order if variance too high, decrease if coefficients small

    if model.variance > config.order_threshold && model.order < config.max_order {
        // Increase order
        let new_order = (_model.order + 1).min(config.max_order);
        resize_model(_model, new_order)?;
    } else if model.order > config.initial_order {
        // Check if higher-order coefficients are significant
        let tail_energy: f64 = _model
            .coefficients
            .slice(scirs2_core::ndarray::s![_model.order - 2..])
            .iter()
            .map(|&c| c * c)
            .sum();

        if tail_energy < config.order_threshold * 0.1 {
            // Decrease order
            let new_order = (_model.order - 1).max(config.initial_order);
            resize_model(_model, new_order)?;
        }
    }

    Ok(())
}

/// Resize model to new order
#[allow(dead_code)]
fn resize_model(_model: &mut AdaptiveArModel, neworder: usize) -> SignalResult<()> {
    if new_order == model._order {
        return Ok(());
    }

    // Resize coefficients
    let mut new_coeffs = Array1::zeros(new_order);
    let copy_len = new_order.min(_model._order);
    for i in 0..copy_len {
        new_coeffs[i] = model.coefficients[i];
    }
    model.coefficients = new_coeffs;

    // Resize gain matrix
    let mut new_gain = Array2::zeros((new_order, new_order));
    for i in 0..copy_len {
        for j in 0..copy_len {
            new_gain[[i, j]] = model.gain[[i, j]];
        }
    }
    // Initialize new elements
    for i in copy_len..new_order {
        new_gain[[i, i]] = 1000.0; // Large initial value for RLS
    }
    model.gain = new_gain;

    model._order = new_order;
    Ok(())
}

/// Compute time-varying spectrum from adaptive model
///
/// # Arguments
///
/// * `model` - Adaptive AR model
/// * `n_freq` - Number of frequency points
///
/// # Returns
///
/// * (frequencies, power spectral density)
#[allow(dead_code)]
pub fn adaptive_spectrum(
    model: &AdaptiveArModel,
    n_freq: usize,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    check_positive(n_freq, "n_freq")?;

    let mut frequencies = Vec::with_capacity(n_freq);
    let mut psd = Vec::with_capacity(n_freq);

    for i in 0..n_freq {
        let f = i as f64 / (2.0 * n_freq as f64);
        frequencies.push(f);

        // Compute transfer function at this frequency
        let omega = 2.0 * PI * f;
        let mut h = Complex64::new(1.0, 0.0);

        for k in 0..model.order {
            let z = Complex64::new(0.0, -omega * (k + 1) as f64).exp();
            h -= model.coefficients[k] * z;
        }

        // PSD = variance / |H(f)|^2
        let power = model.variance / h.norm_sqr();
        psd.push(power);
    }

    Ok((frequencies, psd))
}

/// Track spectral peaks over time
#[derive(Debug, Clone)]
pub struct SpectralPeakTracker {
    /// Current peak frequencies
    pub peak_frequencies: Vec<f64>,
    /// Peak amplitudes
    pub peak_amplitudes: Vec<f64>,
    /// Peak tracking history
    pub history: Vec<Vec<f64>>,
    /// Maximum number of peaks to track
    max_peaks: usize,
}

impl SpectralPeakTracker {
    /// Create new peak tracker
    pub fn new(_maxpeaks: usize) -> Self {
        Self {
            peak_frequencies: Vec::new(),
            peak_amplitudes: Vec::new(),
            history: Vec::new(),
            max_peaks,
        }
    }

    /// Update tracked peaks from spectrum
    pub fn update(&mut self, frequencies: &[f64], psd: &[f64]) -> SignalResult<()> {
        if frequencies.len() != psd.len() {
            return Err(SignalError::ShapeMismatch(
                "Frequencies and PSD must have same length".to_string(),
            ));
        }

        // Find local maxima
        let mut peaks = Vec::new();

        for i in 1..psd.len() - 1 {
            if psd[i] > psd[i - 1] && psd[i] > psd[i + 1] {
                peaks.push((frequencies[i], psd[i]));
            }
        }

        // Sort by amplitude and keep top peaks
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        peaks.truncate(self.max_peaks);

        // Update tracked peaks
        self.peak_frequencies = peaks.iter().map(|(f_, _)| *f_).collect();
        self.peak_amplitudes = peaks.iter().map(|(_, a)| *a).collect();

        // Add to history
        self.history.push(self.peak_frequencies.clone());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_ar_initialization() {
        let config = AdaptiveConfig::default();
        let model = initialize_adaptive_ar(&config).unwrap();

        assert_eq!(model.order, config.initial_order);
        assert_eq!(model.coefficients.len(), config.initial_order);
    }

    #[test]
    fn test_adaptive_update() {
        let config = AdaptiveConfig {
            method: AdaptiveMethod::LMS,
            step_size: 0.01,
            ..Default::default()
        };

        let mut model = initialize_adaptive_ar(&config).unwrap();

        // Generate simple AR signal
        let signal = vec![1.0, 0.5, 0.25, 0.125, 0.0625];

        for &sample in &signal {
            let error = update_adaptive_ar(&mut model, sample, &config).unwrap();
            assert!(error.is_finite());
        }

        assert!(model.variance > 0.0);
    }

    #[test]
    fn test_adaptive_spectrum() {
        let model = AdaptiveArModel {
            coefficients: Array1::from_vec(vec![0.5, -0.3]),
            order: 2,
            variance: 1.0,
            forgetting_factor: 0.98,
            gain: Array2::eye(2),
            state_buffer: vec![0.0; 10],
            method: AdaptiveMethod::RLS,
        };

        let (freqs, psd) = adaptive_spectrum(&model, 128).unwrap();

        assert_eq!(freqs.len(), 128);
        assert_eq!(psd.len(), 128);
        assert!(psd.iter().all(|&p| p > 0.0));
    }
}
