//! K-FAC layer state management
//!
//! This module contains the layer-specific state for K-FAC optimization,
//! including covariance matrices and their inverses.

use ndarray::{s, Array1, Array2};
use num_traits::Float;
use super::config::LayerInfo;
use crate::error::{OptimError, Result};

/// K-FAC optimizer state for a single layer
#[derive(Debug, Clone)]
pub struct KFACLayerState<T: Float> {
    /// Input covariance matrix A = E[a a^T]
    pub a_cov: Array2<T>,

    /// Output gradient covariance matrix G = E[g g^T]
    pub g_cov: Array2<T>,

    /// Inverse of input covariance matrix
    pub a_cov_inv: Option<Array2<T>>,

    /// Inverse of output gradient covariance matrix
    pub g_cov_inv: Option<Array2<T>>,

    /// Number of updates performed
    pub num_updates: usize,

    /// Last update step for covariance matrices
    pub last_cov_update: usize,

    /// Last update step for inverse matrices
    pub last_inv_update: usize,

    /// Damping values for this layer
    pub damping_a: T,
    pub damping_g: T,

    /// Layer information
    pub layerinfo: LayerInfo,

    /// Precomputed Kronecker factors for bias
    pub bias_correction: Option<Array1<T>>,

    /// Moving average statistics
    pub running_mean_a: Option<Array1<T>>,
    pub running_mean_g: Option<Array1<T>>,
}

impl<T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive> KFACLayerState<T> {
    /// Create a new layer state for the given layer
    pub fn new(layer_info: LayerInfo, initial_damping: T) -> Self {
        let input_size = layer_info.input_cov_size();
        let output_size = layer_info.output_cov_size();

        Self {
            a_cov: Array2::eye(input_size),
            g_cov: Array2::eye(output_size),
            a_cov_inv: None,
            g_cov_inv: None,
            num_updates: 0,
            last_cov_update: 0,
            last_inv_update: 0,
            damping_a: initial_damping,
            damping_g: initial_damping,
            layerinfo: layer_info,
            bias_correction: None,
            running_mean_a: None,
            running_mean_g: None,
        }
    }

    /// Initialize moving average statistics
    pub fn init_running_stats(&mut self) {
        let input_size = self.layerinfo.input_cov_size();
        let output_size = self.layerinfo.output_cov_size();

        self.running_mean_a = Some(Array1::zeros(input_size));
        self.running_mean_g = Some(Array1::zeros(output_size));
    }

    /// Update the input covariance matrix with new activations
    pub fn update_input_covariance(&mut self, activations: &Array2<T>, decay: T) {
        let batch_size = activations.nrows();
        if batch_size == 0 {
            return;
        }

        // Add bias term if needed
        let input_data = if self.layerinfo.has_bias {
            self.add_bias_column(activations)
        } else {
            activations.clone()
        };

        // Compute sample covariance
        let batch_cov = self.compute_sample_covariance(&input_data);

        // Update running covariance with exponential moving average
        self.a_cov = &self.a_cov * decay + &batch_cov * (T::one() - decay);
        self.num_updates += 1;
    }

    /// Update the output gradient covariance matrix
    pub fn update_output_covariance(&mut self, gradients: &Array2<T>, decay: T) {
        let batch_size = gradients.nrows();
        if batch_size == 0 {
            return;
        }

        // Compute sample covariance
        let batch_cov = self.compute_sample_covariance(gradients);

        // Update running covariance with exponential moving average
        self.g_cov = &self.g_cov * decay + &batch_cov * (T::one() - decay);
    }

    /// Compute the inverse of covariance matrices with regularization
    pub fn compute_inverses(&mut self, damping_a: T, damping_g: T) -> Result<()> {
        self.damping_a = damping_a;
        self.damping_g = damping_g;

        // Compute regularized inverse of input covariance
        let mut a_reg = self.a_cov.clone();
        for i in 0..a_reg.nrows() {
            a_reg[[i, i]] = a_reg[[i, i]] + damping_a;
        }

        self.a_cov_inv = Some(self.compute_matrix_inverse(&a_reg)?);

        // Compute regularized inverse of output gradient covariance
        let mut g_reg = self.g_cov.clone();
        for i in 0..g_reg.nrows() {
            g_reg[[i, i]] = g_reg[[i, i]] + damping_g;
        }

        self.g_cov_inv = Some(self.compute_matrix_inverse(&g_reg)?);

        self.last_inv_update = self.num_updates;
        Ok(())
    }

    /// Get the condition number estimate of covariance matrices
    pub fn condition_number_estimate(&self) -> (T, T) {
        let a_cond = self.estimate_condition_number(&self.a_cov);
        let g_cond = self.estimate_condition_number(&self.g_cov);
        (a_cond, g_cond)
    }

    /// Check if the layer state is ready for optimization (inverses computed)
    pub fn is_ready(&self) -> bool {
        self.a_cov_inv.is_some() && self.g_cov_inv.is_some()
    }

    /// Get memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        let float_size = std::mem::size_of::<T>();
        let mut size = 0;

        // Covariance matrices
        size += self.a_cov.len() * float_size;
        size += self.g_cov.len() * float_size;

        // Inverse matrices
        if let Some(ref inv) = self.a_cov_inv {
            size += inv.len() * float_size;
        }
        if let Some(ref inv) = self.g_cov_inv {
            size += inv.len() * float_size;
        }

        // Running statistics
        if let Some(ref mean) = self.running_mean_a {
            size += mean.len() * float_size;
        }
        if let Some(ref mean) = self.running_mean_g {
            size += mean.len() * float_size;
        }

        // Bias correction
        if let Some(ref bias) = self.bias_correction {
            size += bias.len() * float_size;
        }

        size
    }

    /// Reset the layer state
    pub fn reset(&mut self) {
        let input_size = self.layerinfo.input_cov_size();
        let output_size = self.layerinfo.output_cov_size();

        self.a_cov = Array2::eye(input_size);
        self.g_cov = Array2::eye(output_size);
        self.a_cov_inv = None;
        self.g_cov_inv = None;
        self.num_updates = 0;
        self.last_cov_update = 0;
        self.last_inv_update = 0;
        self.bias_correction = None;

        if self.running_mean_a.is_some() {
            self.running_mean_a = Some(Array1::zeros(input_size));
        }
        if self.running_mean_g.is_some() {
            self.running_mean_g = Some(Array1::zeros(output_size));
        }
    }

    // Private helper methods

    fn add_bias_column(&self, activations: &Array2<T>) -> Array2<T> {
        let (batch_size, input_dim) = activations.dim();
        let mut result = Array2::ones((batch_size, input_dim + 1));
        result.slice_mut(s![.., ..input_dim]).assign(activations);
        result
    }

    fn compute_sample_covariance(&self, data: &Array2<T>) -> Array2<T> {
        let batch_size = data.nrows() as f64;
        if batch_size <= 1.0 {
            return Array2::eye(data.ncols());
        }

        let batch_size_t = T::from(batch_size).unwrap();

        // Center the data
        let mean = data.mean_axis(ndarray::Axis(0)).unwrap();
        let centered = data - &mean;

        // Compute covariance: (1/(n-1)) * X^T * X
        let cov = centered.t().dot(&centered) / (batch_size_t - T::one());

        cov
    }

    fn compute_matrix_inverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        // Simple matrix inversion using LU decomposition
        // In practice, you would use a more robust method like SVD
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(OptimError::InvalidParameter("Matrix must be square".to_string()));
        }

        // For now, use a simple identity matrix as a placeholder
        // In a real implementation, you would use a proper matrix inversion library
        let mut inv = Array2::eye(n);

        // Add small regularization to ensure numerical stability
        let reg_term = T::from(1e-8).unwrap();
        for i in 0..n {
            inv[[i, i]] = inv[[i, i]] + reg_term;
        }

        Ok(inv)
    }

    fn estimate_condition_number(&self, matrix: &Array2<T>) -> T {
        let mut max_diag = T::zero();
        let mut min_diag = T::infinity();

        for i in 0..matrix.nrows() {
            let diag = matrix[[i, i]];
            if diag > max_diag {
                max_diag = diag;
            }
            if diag < min_diag {
                min_diag = diag;
            }
        }

        if min_diag > T::zero() {
            max_diag / min_diag
        } else {
            T::infinity()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::second_order::kfac::config::{LayerInfo, LayerType};

    #[test]
    fn test_layer_state_creation() {
        let layer_info = LayerInfo {
            name: "test_layer".to_string(),
            input_dim: 128,
            output_dim: 64,
            layer_type: LayerType::Dense,
            has_bias: true,
        };

        let state = KFACLayerState::<f32>::new(layer_info, 0.001);

        assert_eq!(state.a_cov.nrows(), 129); // +1 for bias
        assert_eq!(state.g_cov.nrows(), 64);
        assert!(!state.is_ready()); // No inverses computed yet
    }

    #[test]
    fn test_covariance_update() {
        let layer_info = LayerInfo {
            name: "test_layer".to_string(),
            input_dim: 4,
            output_dim: 2,
            layer_type: LayerType::Dense,
            has_bias: false,
        };

        let mut state = KFACLayerState::<f64>::new(layer_info, 0.001);
        let activations = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        state.update_input_covariance(&activations, 0.95);

        assert_eq!(state.num_updates, 1);
        assert!(state.a_cov[[0, 0]] != 1.0); // Should have changed from identity
    }

    #[test]
    fn test_condition_number_estimation() {
        let layer_info = LayerInfo {
            name: "test_layer".to_string(),
            input_dim: 3,
            output_dim: 3,
            layer_type: LayerType::Dense,
            has_bias: false,
        };

        let state = KFACLayerState::<f32>::new(layer_info, 0.001);
        let (a_cond, g_cond) = state.condition_number_estimate();

        // Identity matrix should have condition number 1
        assert!((a_cond - 1.0).abs() < 1e-6);
        assert!((g_cond - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_memory_usage() {
        let layer_info = LayerInfo {
            name: "test_layer".to_string(),
            input_dim: 100,
            output_dim: 50,
            layer_type: LayerType::Dense,
            has_bias: true,
        };

        let state = KFACLayerState::<f64>::new(layer_info, 0.001);
        let memory_usage = state.memory_usage();

        assert!(memory_usage > 0);
        // Should at least include the covariance matrices
        let expected_minimum = (101 * 101 + 50 * 50) * std::mem::size_of::<f64>();
        assert!(memory_usage >= expected_minimum);
    }
}