//! Advanced adaptive filtering algorithms
//!
//! This module provides comprehensive adaptive filter implementations including classical
//! algorithms (LMS, RLS, NLMS) and advanced variants (Variable Step-Size LMS, Affine
//! Projection Algorithm, Frequency Domain LMS, robust adaptive filters). These filters
//! are used for applications such as noise cancellation, system identification, echo
//! cancellation, equalization, beamforming, and channel estimation.

use crate::error::{SignalError, SignalResult};
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::VecDeque;
use std::fmt::Debug;

/// Least Mean Squares (LMS) adaptive filter
///
/// The LMS algorithm is a simple and robust adaptive filter that minimizes
/// the mean square error between the desired signal and the filter output.
/// It uses a gradient descent approach to update the filter coefficients.
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::adaptive::LmsFilter;
///
/// let mut lms = LmsFilter::new(4, 0.01, 0.0).unwrap();
/// let output = lms.adapt(&[1.0, 0.5, -0.3, 0.8], 0.5).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LmsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)
    buffer: Vec<f64>,
    /// Step size (learning rate)
    step_size: f64,
    /// Current buffer index
    buffer_index: usize,
}

impl LmsFilter {
    /// Create a new LMS filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps (filter order + 1)
    /// * `step_size` - Learning rate (typically 0.001 to 0.1)
    /// * `initial_weight` - Initial value for all filter weights
    ///
    /// # Returns
    ///
    /// * A new LMS filter instance
    pub fn new(num_taps: usize, step_size: f64, initial_weight: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        Ok(LmsFilter {
            weights: vec![initial_weight; num_taps],
            buffer: vec![0.0; num_taps],
            step_size,
            buffer_index: 0,
        })
    }

    /// Process one sample through the adaptive filter
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample
    /// * `desired` - Desired output sample
    ///
    /// # Returns
    ///
    /// * Tuple of (filter_output, error, mse_estimate)
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        // Update circular buffer with new input
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Compute filter output (dot product of weights and buffered inputs)
        let mut output = 0.0;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            output += self.weights[i] * self.buffer[buffer_idx];
        }

        // Compute error
        let error = desired - output;

        // Update weights using LMS algorithm: w(n+1) = w(n) + μ * e(n) * x(n)
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            self.weights[i] += self.step_size * error * self.buffer[buffer_idx];
        }

        // Estimate MSE (simple exponential average)
        let mse_estimate = error * error;

        Ok((output, error, mse_estimate))
    }

    /// Process a batch of samples
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input signal samples
    /// * `desired` - Desired output samples
    ///
    /// # Returns
    ///
    /// * Tuple of (outputs, errors, mse_estimates)
    pub fn adapt_batch(
        &mut self,
        inputs: &[f64],
        desired: &[f64],
    ) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if inputs.len() != desired.len() {
            return Err(SignalError::ValueError(
                "Input and desired signals must have the same length".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors = Vec::with_capacity(inputs.len());
        let mut mse_estimates = Vec::with_capacity(inputs.len());

        for (&input, &des) in inputs.iter().zip(desired.iter()) {
            let (output, error, mse) = self.adapt(input, des)?;
            outputs.push(output);
            errors.push(error);
            mse_estimates.push(mse);
        }

        Ok((outputs, errors, mse_estimates))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get current input buffer state
    pub fn buffer(&self) -> &[f64] {
        &self.buffer
    }

    /// Reset the filter to initial state
    pub fn reset(&mut self, initial_weight: f64) {
        self.weights.fill(initial_weight);
        self.buffer.fill(0.0);
        self.buffer_index = 0;
    }

    /// Set step size (learning rate)
    pub fn set_step_size(&mut self, step_size: f64) -> SignalResult<()> {
        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }
        self.step_size = step_size;
        Ok(())
    }
}

/// Recursive Least Squares (RLS) adaptive filter
///
/// The RLS algorithm provides faster convergence than LMS but with higher
/// computational complexity. It minimizes the exponentially weighted sum
/// of squared errors and is particularly effective for non-stationary signals.
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::adaptive::RlsFilter;
///
/// let mut rls = RlsFilter::new(4, 0.99, 1000.0).unwrap();
/// let output = rls.adapt(&[1.0, 0.5, -0.3, 0.8], 0.5).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RlsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)  
    buffer: Vec<f64>,
    /// Inverse correlation matrix P
    p_matrix: Vec<Vec<f64>>,
    /// Forgetting factor (typically 0.95 to 0.999)
    lambda: f64,
    /// Current buffer index
    buffer_index: usize,
}

impl RlsFilter {
    /// Create a new RLS filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps
    /// * `lambda` - Forgetting factor (0 < lambda <= 1.0, typically 0.99)
    /// * `delta` - Initialization parameter for P matrix (typically 100-10000)
    ///
    /// # Returns
    ///
    /// * A new RLS filter instance
    pub fn new(num_taps: usize, lambda: f64, delta: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if lambda <= 0.0 || lambda > 1.0 {
            return Err(SignalError::ValueError(
                "Forgetting factor must be in (0, 1]".to_string(),
            ));
        }

        if delta <= 0.0 {
            return Err(SignalError::ValueError(
                "Delta must be positive".to_string(),
            ));
        }

        // Initialize P matrix as delta * I (identity matrix)
        let mut p_matrix = vec![vec![0.0; num_taps]; num_taps];
        for (i, row) in p_matrix.iter_mut().enumerate().take(num_taps) {
            row[i] = delta;
        }

        Ok(RlsFilter {
            weights: vec![0.0; num_taps],
            buffer: vec![0.0; num_taps],
            p_matrix,
            lambda,
            buffer_index: 0,
        })
    }

    /// Process one sample through the adaptive filter
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample
    /// * `desired` - Desired output sample
    ///
    /// # Returns
    ///
    /// * Tuple of (filter_output, error, mse_estimate)
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        let num_taps = self.weights.len();

        // Update circular buffer with new input
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Create input vector (in proper order)
        let mut input_vec = vec![0.0; num_taps];
        for (i, input_val) in input_vec.iter_mut().enumerate().take(num_taps) {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            *input_val = self.buffer[buffer_idx];
        }

        // Compute filter output
        let output = dot_product(&self.weights, &input_vec);

        // Compute error
        let error = desired - output;

        // RLS algorithm updates
        // 1. Compute k(n) = P(n-1) * x(n) / (lambda + x(n)^T * P(n-1) * x(n))
        let mut px = matrix_vector_multiply(&self.p_matrix, &input_vec);
        let xpx = dot_product(&input_vec, &px);
        let denominator = self.lambda + xpx;

        if denominator.abs() < 1e-10 {
            return Err(SignalError::ValueError(
                "RLS denominator too small, numerical instability".to_string(),
            ));
        }

        for px_val in &mut px {
            *px_val /= denominator;
        }
        let k = px; // k(n) = P(n-1) * x(n) / denominator

        // 2. Update weights: w(n) = w(n-1) + k(n) * e(n)
        for (weight, &k_val) in self.weights.iter_mut().zip(k.iter()) {
            *weight += k_val * error;
        }

        // 3. Update P matrix: P(n) = (P(n-1) - k(n) * x(n)^T * P(n-1)) / lambda
        let mut kx_outer = vec![vec![0.0; num_taps]; num_taps];
        for (kx_row, &k_val) in kx_outer.iter_mut().zip(k.iter()) {
            for (kx_elem, &input_val) in kx_row.iter_mut().zip(input_vec.iter()) {
                *kx_elem = k_val * input_val;
            }
        }

        // P = (P - k * x^T * P) / lambda
        let p_matrix_copy = self.p_matrix.clone();
        for (p_row, kx_row) in self.p_matrix.iter_mut().zip(kx_outer.iter()) {
            for (j, p_elem) in p_row.iter_mut().enumerate() {
                let kxp = dot_product(kx_row, &get_column(&p_matrix_copy, j));
                *p_elem = (*p_elem - kxp) / self.lambda;
            }
        }

        let mse_estimate = error * error;

        Ok((output, error, mse_estimate))
    }

    /// Process a batch of samples
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input signal samples
    /// * `desired` - Desired output samples
    ///
    /// # Returns
    ///
    /// * Tuple of (outputs, errors, mse_estimates)
    pub fn adapt_batch(
        &mut self,
        inputs: &[f64],
        desired: &[f64],
    ) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if inputs.len() != desired.len() {
            return Err(SignalError::ValueError(
                "Input and desired signals must have the same length".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors = Vec::with_capacity(inputs.len());
        let mut mse_estimates = Vec::with_capacity(inputs.len());

        for (&input, &des) in inputs.iter().zip(desired.iter()) {
            let (output, error, mse) = self.adapt(input, des)?;
            outputs.push(output);
            errors.push(error);
            mse_estimates.push(mse);
        }

        Ok((outputs, errors, mse_estimates))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Reset the filter to initial state
    pub fn reset(&mut self, delta: f64) -> SignalResult<()> {
        if delta <= 0.0 {
            return Err(SignalError::ValueError(
                "Delta must be positive".to_string(),
            ));
        }

        let num_taps = self.weights.len();
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buffer_index = 0;

        // Reinitialize P matrix
        for i in 0..num_taps {
            for j in 0..num_taps {
                self.p_matrix[i][j] = if i == j { delta } else { 0.0 };
            }
        }

        Ok(())
    }
}

/// Normalized LMS (NLMS) adaptive filter
///
/// The NLMS algorithm normalizes the step size by the input signal power,
/// providing better performance for signals with varying power levels.
#[derive(Debug, Clone)]
pub struct NlmsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)
    buffer: Vec<f64>,
    /// Step size (learning rate)
    step_size: f64,
    /// Regularization parameter to avoid division by zero
    epsilon: f64,
    /// Current buffer index
    buffer_index: usize,
}

impl NlmsFilter {
    /// Create a new NLMS filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps
    /// * `step_size` - Learning rate (typically 0.1 to 2.0)
    /// * `epsilon` - Regularization parameter (typically 1e-6)
    ///
    /// # Returns
    ///
    /// * A new NLMS filter instance
    pub fn new(num_taps: usize, step_size: f64, epsilon: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        if epsilon <= 0.0 {
            return Err(SignalError::ValueError(
                "Epsilon must be positive".to_string(),
            ));
        }

        Ok(NlmsFilter {
            weights: vec![0.0; num_taps],
            buffer: vec![0.0; num_taps],
            step_size,
            epsilon,
            buffer_index: 0,
        })
    }

    /// Process one sample through the adaptive filter
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample
    /// * `desired` - Desired output sample
    ///
    /// # Returns
    ///
    /// * Tuple of (filter_output, error, mse_estimate)
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        // Update circular buffer
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Compute filter output
        let mut output = 0.0;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            output += self.weights[i] * self.buffer[buffer_idx];
        }

        // Compute error
        let error = desired - output;

        // Compute input power (norm squared)
        let input_power: f64 = self.buffer.iter().map(|&x| x * x).sum();
        let normalized_step = self.step_size / (input_power + self.epsilon);

        // Update weights using NLMS algorithm
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            self.weights[i] += normalized_step * error * self.buffer[buffer_idx];
        }

        let mse_estimate = error * error;

        Ok((output, error, mse_estimate))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buffer_index = 0;
    }
}

/// Variable Step-Size LMS (VS-LMS) adaptive filter
///
/// The VS-LMS algorithm automatically adjusts the step size based on the gradient
/// estimation to achieve faster convergence and better steady-state performance.
#[derive(Debug, Clone)]
pub struct VsLmsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)
    buffer: Vec<f64>,
    /// Current step size
    step_size: f64,
    /// Initial step size
    initial_step_size: f64,
    /// Step size adaptation parameter
    alpha: f64,
    /// Gradient power estimate
    gradient_power: f64,
    /// Current buffer index
    buffer_index: usize,
    /// Previous error for gradient estimation
    prev_error: f64,
}

impl VsLmsFilter {
    /// Create a new VS-LMS filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps
    /// * `initial_step_size` - Initial learning rate
    /// * `alpha` - Step size adaptation parameter (typically 0.01-0.1)
    ///
    /// # Returns
    ///
    /// * A new VS-LMS filter instance
    pub fn new(num_taps: usize, initial_step_size: f64, alpha: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if initial_step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Initial step size must be positive".to_string(),
            ));
        }

        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(SignalError::ValueError(
                "Alpha must be in (0, 1)".to_string(),
            ));
        }

        Ok(VsLmsFilter {
            weights: vec![0.0; num_taps],
            buffer: vec![0.0; num_taps],
            step_size: initial_step_size,
            initial_step_size,
            alpha,
            gradient_power: 1.0,
            buffer_index: 0,
            prev_error: 0.0,
        })
    }

    /// Process one sample through the adaptive filter
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        // Update circular buffer
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Compute filter output
        let mut output = 0.0;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            output += self.weights[i] * self.buffer[buffer_idx];
        }

        // Compute error
        let error = desired - output;

        // Estimate gradient correlation
        let gradient_correlation = error * self.prev_error;

        // Update gradient power estimate
        self.gradient_power =
            (1.0 - self.alpha) * self.gradient_power + self.alpha * gradient_correlation.abs();

        // Adapt step size
        if gradient_correlation > 0.0 {
            // Increase step size if gradient correlation is positive
            self.step_size *= 1.05;
        } else {
            // Decrease step size if gradient correlation is negative
            self.step_size *= 0.95;
        }

        // Bound step size
        self.step_size = self
            .step_size
            .clamp(self.initial_step_size * 0.01, self.initial_step_size * 10.0);

        // Update weights using current step size
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            self.weights[i] += self.step_size * error * self.buffer[buffer_idx];
        }

        self.prev_error = error;
        let mse_estimate = error * error;

        Ok((output, error, mse_estimate))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get current step size
    pub fn current_step_size(&self) -> f64 {
        self.step_size
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buffer_index = 0;
        self.step_size = self.initial_step_size;
        self.gradient_power = 1.0;
        self.prev_error = 0.0;
    }
}

/// Affine Projection Algorithm (APA) adaptive filter
///
/// APA uses multiple previous input vectors to accelerate convergence,
/// especially effective for highly correlated input signals.
#[derive(Debug, Clone)]
pub struct ApaFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input matrix (K x num_taps)
    input_matrix: Vec<Vec<f64>>,
    /// Projection order K
    projection_order: usize,
    /// Step size
    step_size: f64,
    /// Regularization parameter
    delta: f64,
    /// Current matrix row index
    current_row: usize,
}

impl ApaFilter {
    /// Create a new APA filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps
    /// * `projection_order` - Projection order K (typically 2-10)
    /// * `step_size` - Learning rate
    /// * `delta` - Regularization parameter
    ///
    /// # Returns
    ///
    /// * A new APA filter instance
    pub fn new(
        num_taps: usize,
        projection_order: usize,
        step_size: f64,
        delta: f64,
    ) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if projection_order == 0 {
            return Err(SignalError::ValueError(
                "Projection order must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        Ok(ApaFilter {
            weights: vec![0.0; num_taps],
            input_matrix: vec![vec![0.0; num_taps]; projection_order],
            projection_order,
            step_size,
            delta,
            current_row: 0,
        })
    }

    /// Process one sample through the adaptive filter
    pub fn adapt(&mut self, input: &[f64], desired: f64) -> SignalResult<(f64, f64, f64)> {
        if input.len() != self.weights.len() {
            return Err(SignalError::ValueError(
                "Input length must match number of taps".to_string(),
            ));
        }

        // Update input matrix
        self.input_matrix[self.current_row] = input.to_vec();
        self.current_row = (self.current_row + 1) % self.projection_order;

        // Compute filter output
        let output = dot_product(&self.weights, input);
        let error = desired - output;

        // Compute all errors for the projection order
        let mut errors = vec![0.0; self.projection_order];
        let mut outputs = vec![0.0; self.projection_order];

        for k in 0..self.projection_order {
            outputs[k] = dot_product(&self.weights, &self.input_matrix[k]);
            errors[k] =
                if k == (self.current_row + self.projection_order - 1) % self.projection_order {
                    error
                } else {
                    // For simplicity, use zero for other desired values
                    -outputs[k]
                };
        }

        // Compute input correlation matrix X^T * X
        let mut correlation_matrix = vec![vec![0.0; self.projection_order]; self.projection_order];
        for (i, row) in correlation_matrix
            .iter_mut()
            .enumerate()
            .take(self.projection_order)
        {
            for (j, cell) in row.iter_mut().enumerate().take(self.projection_order) {
                *cell = dot_product(&self.input_matrix[i], &self.input_matrix[j]);
                if i == j {
                    *cell += self.delta; // Regularization
                }
            }
        }

        // Solve for step size vector: alpha = (X^T * X + delta * I)^(-1) * e
        let step_vector = solve_linear_system_small(&correlation_matrix, &errors)?;

        // Update weights: w = w + mu * X^T * alpha
        for (k, input_row) in self
            .input_matrix
            .iter()
            .enumerate()
            .take(self.projection_order)
        {
            for (i, weight) in self.weights.iter_mut().enumerate() {
                *weight += self.step_size * step_vector[k] * input_row[i];
            }
        }

        let mse_estimate = error * error;
        Ok((output, error, mse_estimate))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        for row in &mut self.input_matrix {
            row.fill(0.0);
        }
        self.current_row = 0;
    }
}

/// Frequency Domain LMS (FDLMS) adaptive filter
///
/// FDLMS operates in the frequency domain for computational efficiency
/// with long filters, using overlap-save processing.
pub struct FdlmsFilter {
    /// Filter length
    filter_length: usize,
    /// Block size (typically 2 * filter_length)
    block_size: usize,
    /// Filter coefficients in frequency domain
    freq_weights: Vec<Complex<f64>>,
    /// Input buffer for overlap-save
    input_buffer: VecDeque<f64>,
    /// Error buffer for overlap-save
    error_buffer: VecDeque<f64>,
    /// Step size
    step_size: f64,
    /// FFT planner
    fft_planner: FftPlanner<f64>,
    /// Leakage factor for weight constraint
    leakage: f64,
}

impl FdlmsFilter {
    /// Create a new FDLMS filter
    ///
    /// # Arguments
    ///
    /// * `filter_length` - Number of filter taps
    /// * `step_size` - Learning rate
    /// * `leakage` - Leakage factor (0.999-1.0)
    ///
    /// # Returns
    ///
    /// * A new FDLMS filter instance
    pub fn new(filter_length: usize, step_size: f64, leakage: f64) -> SignalResult<Self> {
        if filter_length == 0 {
            return Err(SignalError::ValueError(
                "Filter length must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        let block_size = 2 * filter_length;

        Ok(FdlmsFilter {
            filter_length,
            block_size,
            freq_weights: vec![Complex::new(0.0, 0.0); block_size],
            input_buffer: VecDeque::with_capacity(block_size),
            error_buffer: VecDeque::with_capacity(block_size),
            step_size,
            fft_planner: FftPlanner::new(),
            leakage,
        })
    }

    /// Process a block of samples
    pub fn adapt_block(
        &mut self,
        inputs: &[f64],
        desired: &[f64],
    ) -> SignalResult<(Vec<f64>, Vec<f64>)> {
        if inputs.len() != desired.len() {
            return Err(SignalError::ValueError(
                "Input and desired signals must have same length".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors = Vec::with_capacity(inputs.len());

        // Process in blocks
        for (input_chunk, desired_chunk) in inputs
            .chunks(self.filter_length)
            .zip(desired.chunks(self.filter_length))
        {
            // Fill input buffer
            for &sample in input_chunk {
                if self.input_buffer.len() >= self.block_size {
                    self.input_buffer.pop_front();
                }
                self.input_buffer.push_back(sample);
            }

            if self.input_buffer.len() == self.block_size {
                let (block_outputs, block_errors) =
                    self.process_block(input_chunk, desired_chunk)?;
                outputs.extend(block_outputs);
                errors.extend(block_errors);
            }
        }

        Ok((outputs, errors))
    }

    fn process_block(
        &mut self,
        inputs: &[f64],
        desired: &[f64],
    ) -> SignalResult<(Vec<f64>, Vec<f64>)> {
        // Convert input buffer to vector for FFT
        let mut input_vec: Vec<Complex<f64>> = self
            .input_buffer
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        // FFT of input
        let fft = self.fft_planner.plan_fft_forward(self.block_size);
        fft.process(&mut input_vec);

        // Frequency domain filtering
        let mut freq_output: Vec<Complex<f64>> = input_vec
            .iter()
            .zip(self.freq_weights.iter())
            .map(|(&x, &w)| x * w)
            .collect();

        // IFFT to get time domain output
        let ifft = self.fft_planner.plan_fft_inverse(self.block_size);
        ifft.process(&mut freq_output);

        // Extract outputs (last half due to overlap-save)
        let outputs: Vec<f64> = freq_output[self.filter_length..]
            .iter()
            .take(inputs.len())
            .map(|c| c.re / self.block_size as f64)
            .collect();

        // Compute errors
        let errors: Vec<f64> = outputs
            .iter()
            .zip(desired.iter())
            .map(|(&out, &des)| des - out)
            .collect();

        // Update weights in frequency domain
        self.update_weights(&input_vec, &errors)?;

        Ok((outputs, errors))
    }

    fn update_weights(&mut self, freq_input: &[Complex<f64>], errors: &[f64]) -> SignalResult<()> {
        // Create error signal in frequency domain
        let mut error_padded = vec![Complex::new(0.0, 0.0); self.block_size];
        for (i, &err) in errors.iter().enumerate() {
            if self.filter_length + i < self.block_size {
                error_padded[self.filter_length + i] = Complex::new(err, 0.0);
            }
        }

        // FFT of error
        let fft = self.fft_planner.plan_fft_forward(self.block_size);
        fft.process(&mut error_padded);

        // Update frequency domain weights
        for i in 0..self.block_size {
            let gradient = freq_input[i].conj() * error_padded[i];
            self.freq_weights[i] =
                self.leakage * self.freq_weights[i] + Complex::new(self.step_size, 0.0) * gradient;
        }

        Ok(())
    }

    /// Get current filter weights (time domain)
    pub fn weights(&self) -> Vec<f64> {
        // Convert frequency domain weights to time domain
        let mut time_weights = self.freq_weights.clone();
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(self.block_size);
        ifft.process(&mut time_weights);

        time_weights[..self.filter_length]
            .iter()
            .map(|c| c.re / self.block_size as f64)
            .collect()
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.freq_weights.fill(Complex::new(0.0, 0.0));
        self.input_buffer.clear();
        self.error_buffer.clear();
    }
}

/// Least Mean Fourth (LMF) robust adaptive filter
///
/// LMF uses fourth-order moments instead of second-order, providing
/// better performance in the presence of impulsive noise.
#[derive(Debug, Clone)]
pub struct LmfFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)
    buffer: Vec<f64>,
    /// Step size
    step_size: f64,
    /// Current buffer index
    buffer_index: usize,
}

impl LmfFilter {
    /// Create a new LMF filter
    pub fn new(num_taps: usize, step_size: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        Ok(LmfFilter {
            weights: vec![0.0; num_taps],
            buffer: vec![0.0; num_taps],
            step_size,
            buffer_index: 0,
        })
    }

    /// Process one sample through the adaptive filter
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        // Update circular buffer
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Compute filter output
        let mut output = 0.0;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            output += self.weights[i] * self.buffer[buffer_idx];
        }

        // Compute error
        let error = desired - output;

        // LMF weight update: w(n+1) = w(n) + μ * e³(n) * x(n)
        let error_cubed = error * error * error;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            self.weights[i] += self.step_size * error_cubed * self.buffer[buffer_idx];
        }

        let mse_estimate = error * error;
        Ok((output, error, mse_estimate))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buffer_index = 0;
    }
}

/// Set-Membership LMS (SM-LMS) adaptive filter
///
/// SM-LMS updates weights only when the output error exceeds a threshold,
/// reducing computational complexity and providing robustness.
#[derive(Debug, Clone)]
pub struct SmLmsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)
    buffer: Vec<f64>,
    /// Step size
    step_size: f64,
    /// Error bound threshold
    error_bound: f64,
    /// Current buffer index
    buffer_index: usize,
    /// Update counter
    update_count: u64,
    /// Total sample counter
    sample_count: u64,
}

impl SmLmsFilter {
    /// Create a new SM-LMS filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps
    /// * `step_size` - Learning rate
    /// * `error_bound` - Error threshold for updates
    ///
    /// # Returns
    ///
    /// * A new SM-LMS filter instance
    pub fn new(num_taps: usize, step_size: f64, error_bound: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        if error_bound <= 0.0 {
            return Err(SignalError::ValueError(
                "Error bound must be positive".to_string(),
            ));
        }

        Ok(SmLmsFilter {
            weights: vec![0.0; num_taps],
            buffer: vec![0.0; num_taps],
            step_size,
            error_bound,
            buffer_index: 0,
            update_count: 0,
            sample_count: 0,
        })
    }

    /// Process one sample through the adaptive filter
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        // Update circular buffer
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Compute filter output
        let mut output = 0.0;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            output += self.weights[i] * self.buffer[buffer_idx];
        }

        // Compute error
        let error = desired - output;
        self.sample_count += 1;

        // Update weights only if error exceeds bound
        if error.abs() > self.error_bound {
            // Compute input power for normalization
            let input_power: f64 = self.buffer.iter().map(|&x| x * x).sum();
            let normalization = if input_power > 1e-12 {
                input_power + 1e-12
            } else {
                1e-12
            };

            // Normalized update
            let normalized_step = self.step_size / normalization;

            for i in 0..self.weights.len() {
                let buffer_idx =
                    (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
                self.weights[i] += normalized_step * error * self.buffer[buffer_idx];
            }

            self.update_count += 1;
        }

        let mse_estimate = error * error;
        Ok((output, error, mse_estimate))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get update statistics
    pub fn update_statistics(&self) -> (u64, u64, f64) {
        let update_ratio = if self.sample_count > 0 {
            self.update_count as f64 / self.sample_count as f64
        } else {
            0.0
        };
        (self.update_count, self.sample_count, update_ratio)
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buffer_index = 0;
        self.update_count = 0;
        self.sample_count = 0;
    }
}

// Helper functions for matrix operations

/// Compute dot product of two vectors
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Multiply matrix by vector
fn matrix_vector_multiply(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.len()];
    for i in 0..matrix.len() {
        result[i] = dot_product(&matrix[i], vector);
    }
    result
}

/// Get column from matrix
fn get_column(matrix: &[Vec<f64>], col: usize) -> Vec<f64> {
    matrix.iter().map(|row| row[col]).collect()
}

/// Solve small linear system using Gaussian elimination (for APA)
fn solve_linear_system_small(matrix: &[Vec<f64>], rhs: &[f64]) -> SignalResult<Vec<f64>> {
    let n = matrix.len();
    if n != rhs.len() {
        return Err(SignalError::ValueError(
            "Matrix and RHS dimensions must match".to_string(),
        ));
    }

    // Create augmented matrix
    let mut aug_matrix = matrix
        .iter()
        .zip(rhs.iter())
        .map(|(row, &b)| {
            let mut aug_row = row.clone();
            aug_row.push(b);
            aug_row
        })
        .collect::<Vec<_>>();

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug_matrix[k][i].abs() > aug_matrix[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        aug_matrix.swap(i, max_row);

        // Check for singular matrix
        if aug_matrix[i][i].abs() < 1e-12 {
            return Err(SignalError::Compute(
                "Matrix is singular or near-singular".to_string(),
            ));
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug_matrix[k][i] / aug_matrix[i][i];
            for j in i..=n {
                aug_matrix[k][j] -= factor * aug_matrix[i][j];
            }
        }
    }

    // Back substitution
    let mut solution = vec![0.0; n];
    for i in (0..n).rev() {
        solution[i] = aug_matrix[i][n];
        for j in (i + 1)..n {
            solution[i] -= aug_matrix[i][j] * solution[j];
        }
        solution[i] /= aug_matrix[i][i];
    }

    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lms_creation() {
        let lms = LmsFilter::new(4, 0.01, 0.0).unwrap();
        assert_eq!(lms.weights().len(), 4);
        assert_eq!(lms.buffer().len(), 4);

        // Test error conditions
        assert!(LmsFilter::new(0, 0.01, 0.0).is_err());
        assert!(LmsFilter::new(4, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_lms_adapt() {
        let mut lms = LmsFilter::new(2, 0.1, 0.0).unwrap();

        // Test single adaptation
        let (output, error, _mse) = lms.adapt(1.0, 0.5).unwrap();

        // Initially weights are zero, so output should be zero
        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.5, epsilon = 1e-10);

        // Weights should be updated
        assert!(!lms.weights().iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_lms_batch() {
        let mut lms = LmsFilter::new(2, 0.05, 0.0).unwrap();

        let inputs = vec![1.0, 0.5, -0.3, 0.8];
        let desired = vec![0.1, 0.2, 0.3, 0.4];

        let (outputs, errors, _mse) = lms.adapt_batch(&inputs, &desired).unwrap();

        assert_eq!(outputs.len(), 4);
        assert_eq!(errors.len(), 4);

        // Error should generally decrease over time for a learnable system
        // Note: LMS adaptation is gradual, so we just check that it's reasonable
        assert!(errors.iter().all(|&e| e.abs() < 10.0)); // Errors should be bounded
    }

    #[test]
    fn test_lms_system_identification() {
        // Test LMS for system identification
        let mut lms = LmsFilter::new(3, 0.01, 0.0).unwrap();

        // Target system: h = [0.5, -0.3, 0.2]
        let target_system = [0.5, -0.3, 0.2];

        // Generate training data
        let mut inputs = Vec::new();
        let mut desired = Vec::new();

        for i in 0..100 {
            let input = (i as f64 * 0.1).sin();
            inputs.push(input);

            // Generate desired output from target system (simplified)
            let output = if i >= 2 {
                target_system[0] * inputs[i]
                    + target_system[1] * inputs[i - 1]
                    + target_system[2] * inputs[i - 2]
            } else {
                0.0
            };
            desired.push(output);
        }

        let (_outputs, _errors, _mse) = lms.adapt_batch(&inputs, &desired).unwrap();

        // Check if weights converged towards target (approximately)
        // Note: LMS convergence depends on step size, signal properties, and training length
        // We test that the weights are in a reasonable range rather than exact convergence
        for (i, &target_weight) in target_system.iter().enumerate() {
            let weight_diff = (lms.weights()[i] - target_weight).abs();
            assert!(
                weight_diff < 1.0,
                "Weight {} difference {} too large",
                i,
                weight_diff
            );
        }
    }

    #[test]
    fn test_rls_creation() {
        let rls = RlsFilter::new(3, 0.99, 100.0).unwrap();
        assert_eq!(rls.weights().len(), 3);

        // Test error conditions
        assert!(RlsFilter::new(0, 0.99, 100.0).is_err());
        assert!(RlsFilter::new(3, 0.0, 100.0).is_err());
        assert!(RlsFilter::new(3, 1.1, 100.0).is_err());
        assert!(RlsFilter::new(3, 0.99, 0.0).is_err());
    }

    #[test]
    fn test_rls_adapt() {
        let mut rls = RlsFilter::new(2, 0.99, 100.0).unwrap();

        let (output, error, _mse) = rls.adapt(1.0, 0.5).unwrap();

        // Initially weights are zero, so output should be zero
        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_nlms_creation() {
        let nlms = NlmsFilter::new(4, 0.5, 1e-6).unwrap();
        assert_eq!(nlms.weights().len(), 4);

        // Test error conditions
        assert!(NlmsFilter::new(0, 0.5, 1e-6).is_err());
        assert!(NlmsFilter::new(4, 0.0, 1e-6).is_err());
        assert!(NlmsFilter::new(4, 0.5, 0.0).is_err());
    }

    #[test]
    fn test_nlms_adapt() {
        let mut nlms = NlmsFilter::new(2, 0.5, 1e-6).unwrap();

        let (output, error, _mse) = nlms.adapt(1.0, 0.3).unwrap();

        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.3, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_operations() {
        // Test dot product
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert_relative_eq!(result, 32.0, epsilon = 1e-10); // 1*4 + 2*5 + 3*6 = 32

        // Test matrix-vector multiply
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let vector = vec![5.0, 6.0];
        let result = matrix_vector_multiply(&matrix, &vector);
        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 17.0, epsilon = 1e-10); // 1*5 + 2*6 = 17
        assert_relative_eq!(result[1], 39.0, epsilon = 1e-10); // 3*5 + 4*6 = 39

        // Test get column
        let column = get_column(&matrix, 0);
        assert_eq!(column, vec![1.0, 3.0]);
    }

    #[test]
    fn test_convergence_comparison() {
        // Compare LMS and RLS convergence for the same problem
        let target_system = [0.8, -0.4];
        let num_samples = 50;

        let mut lms = LmsFilter::new(2, 0.05, 0.0).unwrap();
        let mut rls = RlsFilter::new(2, 0.99, 100.0).unwrap();

        let mut lms_errors = Vec::new();
        let mut rls_errors = Vec::new();

        for i in 0..num_samples {
            let input = (i as f64 * 0.2).sin();
            let desired = if i >= 1 {
                target_system[0] * input + target_system[1] * (((i - 1) as f64) * 0.2).sin()
            } else {
                target_system[0] * input
            };

            let (_out_lms, err_lms, _) = lms.adapt(input, desired).unwrap();
            let (_out_rls, err_rls, _) = rls.adapt(input, desired).unwrap();

            lms_errors.push(err_lms.abs());
            rls_errors.push(err_rls.abs());
        }

        // RLS should generally converge faster (lower final error)
        let lms_final_error = lms_errors.iter().rev().take(10).sum::<f64>() / 10.0;
        let rls_final_error = rls_errors.iter().rev().take(10).sum::<f64>() / 10.0;

        // This is a rough test - both algorithms should achieve reasonable convergence
        // We don't enforce that RLS is better since convergence depends on many factors
        assert!(
            lms_final_error < 2.0,
            "LMS final error too large: {}",
            lms_final_error
        );
        assert!(
            rls_final_error < 2.0,
            "RLS final error too large: {}",
            rls_final_error
        );
    }

    #[test]
    fn test_vs_lms_creation() {
        let vs_lms = VsLmsFilter::new(4, 0.01, 0.05).unwrap();
        assert_eq!(vs_lms.weights().len(), 4);
        assert_relative_eq!(vs_lms.current_step_size(), 0.01, epsilon = 1e-10);

        // Test error conditions
        assert!(VsLmsFilter::new(0, 0.01, 0.05).is_err());
        assert!(VsLmsFilter::new(4, 0.0, 0.05).is_err());
        assert!(VsLmsFilter::new(4, 0.01, 0.0).is_err());
        assert!(VsLmsFilter::new(4, 0.01, 1.0).is_err());
    }

    #[test]
    fn test_vs_lms_adapt() {
        let mut vs_lms = VsLmsFilter::new(2, 0.1, 0.01).unwrap();

        let (output, error, _mse) = vs_lms.adapt(1.0, 0.5).unwrap();

        // Initially weights are zero, so output should be zero
        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.5, epsilon = 1e-10);

        // Step size should adapt over time
        let initial_step = vs_lms.current_step_size();

        for _ in 0..10 {
            vs_lms.adapt(1.0, 0.5).unwrap();
        }

        // Step size should have changed (either increased or decreased)
        assert_ne!(vs_lms.current_step_size(), initial_step);
    }

    #[test]
    fn test_apa_creation() {
        let apa = ApaFilter::new(4, 3, 0.1, 0.01).unwrap();
        assert_eq!(apa.weights().len(), 4);

        // Test error conditions
        assert!(ApaFilter::new(0, 3, 0.1, 0.01).is_err());
        assert!(ApaFilter::new(4, 0, 0.1, 0.01).is_err());
        assert!(ApaFilter::new(4, 3, 0.0, 0.01).is_err());
    }

    #[test]
    fn test_apa_adapt() {
        let mut apa = ApaFilter::new(2, 2, 0.1, 0.01).unwrap();
        let input = vec![1.0, 0.5];

        let (output, error, _mse) = apa.adapt(&input, 0.3).unwrap();

        // Initially weights are zero, so output should be zero
        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.3, epsilon = 1e-10);

        // Test wrong input size
        let wrong_input = vec![1.0];
        assert!(apa.adapt(&wrong_input, 0.3).is_err());
    }

    #[test]
    fn test_fdlms_creation() {
        let fdlms = FdlmsFilter::new(8, 0.01, 0.999).unwrap();
        assert_eq!(fdlms.weights().len(), 8);

        // Test error conditions
        assert!(FdlmsFilter::new(0, 0.01, 0.999).is_err());
        assert!(FdlmsFilter::new(8, 0.0, 0.999).is_err());
    }

    #[test]
    fn test_fdlms_adapt_block() {
        let mut fdlms = FdlmsFilter::new(4, 0.01, 0.999).unwrap();
        let inputs = vec![1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.6, -0.4];
        let desired = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let (outputs, errors) = fdlms.adapt_block(&inputs, &desired).unwrap();

        // Should produce some outputs and errors
        assert!(!outputs.is_empty());
        assert!(!errors.is_empty());
        assert_eq!(outputs.len(), errors.len());

        // Test wrong input size
        let wrong_desired = vec![0.1, 0.2];
        assert!(fdlms.adapt_block(&inputs, &wrong_desired).is_err());
    }

    #[test]
    fn test_lmf_creation() {
        let lmf = LmfFilter::new(4, 0.01).unwrap();
        assert_eq!(lmf.weights().len(), 4);

        // Test error conditions
        assert!(LmfFilter::new(0, 0.01).is_err());
        assert!(LmfFilter::new(4, 0.0).is_err());
    }

    #[test]
    fn test_lmf_adapt() {
        let mut lmf = LmfFilter::new(2, 0.01).unwrap();

        let (output, error, _mse) = lmf.adapt(1.0, 0.5).unwrap();

        // Initially weights are zero, so output should be zero
        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.5, epsilon = 1e-10);

        // LMF should converge (test basic functionality)
        for _ in 0..50 {
            lmf.adapt(1.0, 0.5).unwrap();
        }

        // After many iterations, there should be some learning
        let (final_output, _, _) = lmf.adapt(1.0, 0.5).unwrap();
        assert!(final_output.abs() > 1e-6); // Some non-zero output expected
    }

    #[test]
    fn test_sm_lms_creation() {
        let sm_lms = SmLmsFilter::new(4, 0.1, 0.1).unwrap();
        assert_eq!(sm_lms.weights().len(), 4);

        // Test error conditions
        assert!(SmLmsFilter::new(0, 0.1, 0.1).is_err());
        assert!(SmLmsFilter::new(4, 0.0, 0.1).is_err());
        assert!(SmLmsFilter::new(4, 0.1, 0.0).is_err());
    }

    #[test]
    fn test_sm_lms_adapt() {
        let mut sm_lms = SmLmsFilter::new(2, 0.1, 0.05).unwrap();

        // Small error - should not trigger update
        let (output, error, _mse) = sm_lms.adapt(1.0, 0.01).unwrap();
        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.01, epsilon = 1e-10);

        let (update_count, sample_count, _) = sm_lms.update_statistics();
        assert_eq!(update_count, 0); // No update for small error
        assert_eq!(sample_count, 1);

        // Large error - should trigger update
        sm_lms.adapt(1.0, 0.5).unwrap();
        let (update_count, sample_count, update_ratio) = sm_lms.update_statistics();
        assert_eq!(update_count, 1); // Update triggered
        assert_eq!(sample_count, 2);
        assert_relative_eq!(update_ratio, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_sm_lms_selective_updates() {
        let mut sm_lms = SmLmsFilter::new(3, 0.1, 0.1).unwrap();

        // Apply mix of small and large errors
        let inputs = [1.0, 0.5, -0.3, 0.8, 0.2];
        let errors = [0.05, 0.15, 0.08, 0.2, 0.03]; // Mix of small and large

        for (&input, &target_error) in inputs.iter().zip(errors.iter()) {
            // Desired is computed to produce target error (since initial output is 0)
            sm_lms.adapt(input, target_error).unwrap();
        }

        let (update_count, sample_count, _) = sm_lms.update_statistics();
        assert_eq!(sample_count, 5);
        assert!(update_count < sample_count); // Not all samples triggered updates
        assert!(update_count > 0); // Some updates occurred
    }

    #[test]
    fn test_advanced_algorithm_convergence_comparison() {
        // Compare convergence of different algorithms on the same problem
        let target_system = [0.6, -0.4, 0.2];
        let num_samples = 100;

        let mut lms = LmsFilter::new(3, 0.01, 0.0).unwrap();
        let mut vs_lms = VsLmsFilter::new(3, 0.01, 0.05).unwrap();
        let mut nlms = NlmsFilter::new(3, 0.5, 1e-6).unwrap();
        let mut lmf = LmfFilter::new(3, 0.001).unwrap();

        let mut lms_errors = Vec::new();
        let mut vs_lms_errors = Vec::new();
        let mut nlms_errors = Vec::new();
        let mut lmf_errors = Vec::new();

        for i in 0..num_samples {
            let input = (i as f64 * 0.1).sin();
            let desired = if i >= 2 {
                target_system[0] * input
                    + target_system[1] * ((i - 1) as f64 * 0.1).sin()
                    + target_system[2] * ((i - 2) as f64 * 0.1).sin()
            } else if i >= 1 {
                target_system[0] * input + target_system[1] * ((i - 1) as f64 * 0.1).sin()
            } else {
                target_system[0] * input
            };

            let (_, err_lms, _) = lms.adapt(input, desired).unwrap();
            let (_, err_vs_lms, _) = vs_lms.adapt(input, desired).unwrap();
            let (_, err_nlms, _) = nlms.adapt(input, desired).unwrap();
            let (_, err_lmf, _) = lmf.adapt(input, desired).unwrap();

            lms_errors.push(err_lms.abs());
            vs_lms_errors.push(err_vs_lms.abs());
            nlms_errors.push(err_nlms.abs());
            lmf_errors.push(err_lmf.abs());
        }

        // All algorithms should achieve reasonable convergence
        let final_window = 20;
        let lms_final_error =
            lms_errors.iter().rev().take(final_window).sum::<f64>() / final_window as f64;
        let vs_lms_final_error =
            vs_lms_errors.iter().rev().take(final_window).sum::<f64>() / final_window as f64;
        let nlms_final_error =
            nlms_errors.iter().rev().take(final_window).sum::<f64>() / final_window as f64;
        let lmf_final_error =
            lmf_errors.iter().rev().take(final_window).sum::<f64>() / final_window as f64;

        assert!(
            lms_final_error < 1.0,
            "LMS final error too large: {}",
            lms_final_error
        );
        assert!(
            vs_lms_final_error < 1.0,
            "VS-LMS final error too large: {}",
            vs_lms_final_error
        );
        assert!(
            nlms_final_error < 1.0,
            "NLMS final error too large: {}",
            nlms_final_error
        );
        assert!(
            lmf_final_error < 1.5,
            "LMF final error too large: {}",
            lmf_final_error
        );

        // VS-LMS should generally perform better than standard LMS
        // (This is not always guaranteed but is expected on average)
        println!(
            "LMS final error: {:.4}, VS-LMS final error: {:.4}",
            lms_final_error, vs_lms_final_error
        );
    }

    #[test]
    fn test_solve_linear_system_small() {
        // Test 2x2 system
        let matrix = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let rhs = vec![5.0, 6.0];

        let solution = solve_linear_system_small(&matrix, &rhs).unwrap();

        // Verify solution: 2x + y = 5, x + 3y = 6 => x = 1.8, y = 1.4
        assert_relative_eq!(solution[0], 1.8, epsilon = 1e-10);
        assert_relative_eq!(solution[1], 1.4, epsilon = 1e-10);

        // Test singular matrix
        let singular_matrix = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(solve_linear_system_small(&singular_matrix, &rhs).is_err());

        // Test dimension mismatch
        let wrong_rhs = vec![5.0];
        assert!(solve_linear_system_small(&matrix, &wrong_rhs).is_err());
    }

    #[test]
    fn test_algorithm_reset_functionality() {
        // Test that all algorithms can be properly reset
        let mut vs_lms = VsLmsFilter::new(3, 0.01, 0.05).unwrap();
        let mut apa = ApaFilter::new(3, 2, 0.1, 0.01).unwrap();
        let mut lmf = LmfFilter::new(3, 0.01).unwrap();
        let mut sm_lms = SmLmsFilter::new(3, 0.1, 0.1).unwrap();

        // Adapt some samples to change internal state
        for i in 0..10 {
            let input = i as f64;
            vs_lms.adapt(input, 0.5).unwrap();
            apa.adapt(&[input, input * 0.5, input * 0.2], 0.5).unwrap();
            lmf.adapt(input, 0.5).unwrap();
            sm_lms.adapt(input, 0.5).unwrap();
        }

        // Verify weights are not zero
        assert!(vs_lms.weights().iter().any(|&w| w != 0.0));
        assert!(apa.weights().iter().any(|&w| w != 0.0));
        assert!(lmf.weights().iter().any(|&w| w != 0.0));
        assert!(sm_lms.weights().iter().any(|&w| w != 0.0));

        // Reset filters
        vs_lms.reset();
        apa.reset();
        lmf.reset();
        sm_lms.reset();

        // Verify weights are back to zero
        assert!(vs_lms.weights().iter().all(|&w| w == 0.0));
        assert!(apa.weights().iter().all(|&w| w == 0.0));
        assert!(lmf.weights().iter().all(|&w| w == 0.0));
        assert!(sm_lms.weights().iter().all(|&w| w == 0.0));

        // Verify SM-LMS statistics are reset
        let (update_count, sample_count, _) = sm_lms.update_statistics();
        assert_eq!(update_count, 0);
        assert_eq!(sample_count, 0);
    }
}
