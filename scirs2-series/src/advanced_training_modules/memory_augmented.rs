//! Memory-Augmented Neural Networks
//!
//! This module implements Memory-Augmented Neural Networks (MANNs) for few-shot learning
//! and other tasks that require external memory capabilities.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::few_shot::FewShotEpisode;
use crate::error::Result;

/// Memory-Augmented Neural Network (MANN)
#[derive(Debug)]
pub struct MANN<F: Float + Debug + ndarray::ScalarOperand> {
    /// Controller network parameters
    controller_params: Array2<F>,
    /// External memory matrix
    memory: Array2<F>,
    /// Memory dimensions
    memory_size: usize,
    memory_width: usize,
    /// Controller dimensions
    controller_input_dim: usize,
    controller_hidden_dim: usize,
    controller_output_dim: usize,
    /// Read/write head parameters
    #[allow(dead_code)]
    read_head_params: Array2<F>,
    #[allow(dead_code)]
    write_head_params: Array2<F>,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> MANN<F> {
    /// Create new Memory-Augmented Neural Network
    pub fn new(
        memory_size: usize,
        memory_width: usize,
        controller_input_dim: usize,
        controller_hidden_dim: usize,
        controller_output_dim: usize,
    ) -> Self {
        // Initialize controller parameters
        let controller_param_count = controller_input_dim * controller_hidden_dim
            + controller_hidden_dim
            + controller_hidden_dim * controller_output_dim
            + controller_output_dim;

        let mut controller_params = Array2::zeros((1, controller_param_count));
        let scale =
            F::from(2.0).unwrap() / F::from(controller_input_dim + controller_output_dim).unwrap();
        let std_dev = scale.sqrt();

        for i in 0..controller_param_count {
            let val = ((i * 67) % 1000) as f64 / 1000.0 - 0.5;
            controller_params[[0, i]] = F::from(val).unwrap() * std_dev;
        }

        // Initialize memory
        let memory = Array2::zeros((memory_size, memory_width));

        // Initialize read/write head parameters
        let head_param_count = memory_width * 2 + 3; // key, beta, gate, shift, gamma
        let mut read_head_params = Array2::zeros((1, head_param_count));
        let mut write_head_params = Array2::zeros((1, head_param_count));

        for i in 0..head_param_count {
            let val1 = ((i * 71) % 1000) as f64 / 1000.0 - 0.5;
            let val2 = ((i * 73) % 1000) as f64 / 1000.0 - 0.5;
            read_head_params[[0, i]] = F::from(val1).unwrap() * F::from(0.1).unwrap();
            write_head_params[[0, i]] = F::from(val2).unwrap() * F::from(0.1).unwrap();
        }

        Self {
            controller_params,
            memory,
            memory_size,
            memory_width,
            controller_input_dim,
            controller_hidden_dim,
            controller_output_dim,
            read_head_params,
            write_head_params,
        }
    }

    /// Forward pass through MANN
    pub fn forward(&mut self, input: &Array1<F>) -> Result<Array1<F>> {
        // Read from memory
        let read_vector = self.memory_read()?;

        // Combine input with read vector
        let mut controller_input = Array1::zeros(self.controller_input_dim);
        for i in 0..input.len().min(self.controller_input_dim) {
            controller_input[i] = input[i];
        }

        // Add read vector to controller input
        let read_start = input.len().min(self.controller_input_dim);
        for i in 0..read_vector.len() {
            if read_start + i < self.controller_input_dim {
                controller_input[read_start + i] = read_vector[i];
            }
        }

        // Controller forward pass
        let controller_output = self.controller_forward(&controller_input)?;

        // Write to memory
        self.memory_write(&controller_output)?;

        Ok(controller_output)
    }

    /// Controller neural network forward pass
    fn controller_forward(&self, input: &Array1<F>) -> Result<Array1<F>> {
        let (w1, b1, w2, b2) = self.extract_controller_weights();

        // Hidden layer
        let mut hidden = Array1::zeros(self.controller_hidden_dim);
        for i in 0..self.controller_hidden_dim {
            let mut sum = b1[i];
            for j in 0..input.len().min(w1.ncols()) {
                sum = sum + input[j] * w1[[i, j]];
            }
            hidden[i] = self.tanh(sum);
        }

        // Output layer
        let mut output = Array1::zeros(self.controller_output_dim);
        for i in 0..self.controller_output_dim {
            let mut sum = b2[i];
            for j in 0..self.controller_hidden_dim {
                sum = sum + hidden[j] * w2[[i, j]];
            }
            output[i] = sum;
        }

        Ok(output)
    }

    /// Read from external memory
    fn memory_read(&self) -> Result<Array1<F>> {
        // Simplified memory read - return average of memory rows
        let mut read_vector = Array1::zeros(self.memory_width);

        for i in 0..self.memory_size {
            for j in 0..self.memory_width {
                read_vector[j] = read_vector[j] + self.memory[[i, j]];
            }
        }

        let size = F::from(self.memory_size).unwrap();
        for j in 0..self.memory_width {
            read_vector[j] = read_vector[j] / size;
        }

        Ok(read_vector)
    }

    /// Write to external memory
    fn memory_write(&mut self, controller_output: &Array1<F>) -> Result<()> {
        // Simplified memory write - update first row with controller _output
        for i in 0..controller_output.len().min(self.memory_width) {
            self.memory[[0, i]] = controller_output[i];
        }

        Ok(())
    }

    /// Extract controller weights from parameters
    fn extract_controller_weights(&self) -> (Array2<F>, Array1<F>, Array2<F>, Array1<F>) {
        let param_vec = self.controller_params.row(0);
        let mut idx = 0;

        // W1: controller_input_dim x controller_hidden_dim
        let mut w1 = Array2::zeros((self.controller_hidden_dim, self.controller_input_dim));
        for i in 0..self.controller_hidden_dim {
            for j in 0..self.controller_input_dim {
                if idx < param_vec.len() {
                    w1[[i, j]] = param_vec[idx];
                    idx += 1;
                }
            }
        }

        // b1: controller_hidden_dim
        let mut b1 = Array1::zeros(self.controller_hidden_dim);
        for i in 0..self.controller_hidden_dim {
            if idx < param_vec.len() {
                b1[i] = param_vec[idx];
                idx += 1;
            }
        }

        // W2: controller_hidden_dim x controller_output_dim
        let mut w2 = Array2::zeros((self.controller_output_dim, self.controller_hidden_dim));
        for i in 0..self.controller_output_dim {
            for j in 0..self.controller_hidden_dim {
                if idx < param_vec.len() {
                    w2[[i, j]] = param_vec[idx];
                    idx += 1;
                }
            }
        }

        // b2: controller_output_dim
        let mut b2 = Array1::zeros(self.controller_output_dim);
        for i in 0..self.controller_output_dim {
            if idx < param_vec.len() {
                b2[i] = param_vec[idx];
                idx += 1;
            }
        }

        (w1, b1, w2, b2)
    }

    /// Reset memory
    pub fn reset_memory(&mut self) {
        self.memory = Array2::zeros((self.memory_size, self.memory_width));
    }

    /// Train MANN on few-shot learning task
    pub fn train_few_shot(&mut self, episodes: &[FewShotEpisode<F>]) -> Result<F> {
        let mut total_loss = F::zero();

        for episode in episodes {
            self.reset_memory();

            // Present support set
            for i in 0..episode.support_x.nrows() {
                let input_row = episode.support_x.row(i).to_owned();
                let _output = self.forward(&input_row)?;
            }

            // Test on query set
            let mut episode_loss = F::zero();
            for i in 0..episode.query_x.nrows() {
                let input_row = episode.query_x.row(i).to_owned();
                let prediction = self.forward(&input_row)?;

                // Compute loss (simplified)
                if i < episode.query_y.len() {
                    let target = F::from(episode.query_y[i]).unwrap();
                    if !prediction.is_empty() {
                        let diff = prediction[0] - target;
                        episode_loss = episode_loss + diff * diff;
                    }
                }
            }

            total_loss = total_loss + episode_loss;
        }

        Ok(total_loss / F::from(episodes.len()).unwrap())
    }

    /// Get current memory state
    pub fn get_memory(&self) -> &Array2<F> {
        &self.memory
    }

    /// Set memory state
    pub fn set_memory(&mut self, memory: Array2<F>) -> Result<()> {
        if memory.dim() != (self.memory_size, self.memory_width) {
            return Err(crate::error::TimeSeriesError::InvalidOperation(
                "Memory dimensions do not match".to_string(),
            ));
        }
        self.memory = memory;
        Ok(())
    }

    /// Get controller parameters
    pub fn get_controller_params(&self) -> &Array2<F> {
        &self.controller_params
    }

    /// Set controller parameters
    pub fn set_controller_params(&mut self, params: Array2<F>) -> Result<()> {
        if params.dim() != self.controller_params.dim() {
            return Err(crate::error::TimeSeriesError::InvalidOperation(
                "Controller parameter dimensions do not match".to_string(),
            ));
        }
        self.controller_params = params;
        Ok(())
    }

    /// Get memory dimensions
    pub fn memory_dimensions(&self) -> (usize, usize) {
        (self.memory_size, self.memory_width)
    }

    /// Get controller dimensions
    pub fn controller_dimensions(&self) -> (usize, usize, usize) {
        (
            self.controller_input_dim,
            self.controller_hidden_dim,
            self.controller_output_dim,
        )
    }

    /// Process a sequence of inputs
    pub fn process_sequence(&mut self, inputs: &[Array1<F>]) -> Result<Vec<Array1<F>>> {
        let mut outputs = Vec::new();

        for input in inputs {
            let output = self.forward(input)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Compute attention weights for memory addressing (simplified)
    pub fn compute_attention_weights(&self, key: &Array1<F>) -> Result<Array1<F>> {
        let mut weights = Array1::zeros(self.memory_size);

        for i in 0..self.memory_size {
            let memory_row = self.memory.row(i);
            let mut similarity = F::zero();

            for j in 0..key.len().min(memory_row.len()) {
                similarity = similarity + key[j] * memory_row[j];
            }

            weights[i] = similarity;
        }

        // Apply softmax
        let max_weight = weights.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
        let mut sum = F::zero();

        for weight in weights.iter_mut() {
            *weight = (*weight - max_weight).exp();
            sum = sum + *weight;
        }

        for weight in weights.iter_mut() {
            *weight = *weight / sum;
        }

        Ok(weights)
    }

    /// Hyperbolic tangent activation
    fn tanh(&self, x: F) -> F {
        x.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mann_creation() {
        let mann = MANN::<f64>::new(10, 8, 12, 16, 6);
        let (memory_size, memory_width) = mann.memory_dimensions();
        let (input_dim, hidden_dim, output_dim) = mann.controller_dimensions();

        assert_eq!(memory_size, 10);
        assert_eq!(memory_width, 8);
        assert_eq!(input_dim, 12);
        assert_eq!(hidden_dim, 16);
        assert_eq!(output_dim, 6);
    }

    #[test]
    fn test_mann_forward() {
        let mut mann = MANN::<f64>::new(5, 4, 8, 10, 3);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let output = mann.forward(&input).unwrap();
        assert_eq!(output.len(), 3);

        // Check that output is finite
        for &val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_mann_memory_operations() {
        let mut mann = MANN::<f64>::new(3, 2, 4, 6, 2);

        // Test memory read (should be zeros initially)
        let read_vector = mann.memory_read().unwrap();
        assert_eq!(read_vector.len(), 2);
        for &val in read_vector.iter() {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
        }

        // Test memory write
        let write_data = Array1::from_vec(vec![1.0, 2.0]);
        mann.memory_write(&write_data).unwrap();

        // Check that memory was updated
        let memory = mann.get_memory();
        assert_abs_diff_eq!(memory[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(memory[[0, 1]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mann_reset_memory() {
        let mut mann = MANN::<f64>::new(3, 2, 4, 6, 2);

        // Write some data
        let write_data = Array1::from_vec(vec![5.0, 10.0]);
        mann.memory_write(&write_data).unwrap();

        // Reset memory
        mann.reset_memory();

        // Check that memory is reset to zeros
        let memory = mann.get_memory();
        for &val in memory.iter() {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mann_process_sequence() {
        let mut mann = MANN::<f64>::new(4, 3, 6, 8, 2);
        let inputs = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        ];

        let outputs = mann.process_sequence(&inputs).unwrap();
        assert_eq!(outputs.len(), 3);

        for output in outputs {
            assert_eq!(output.len(), 2);
            for &val in output.iter() {
                assert!(val.is_finite());
            }
        }
    }

    #[test]
    fn test_mann_attention_weights() {
        let mut mann = MANN::<f64>::new(3, 4, 6, 8, 2);

        // Set some values in memory
        let memory_data = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        mann.set_memory(memory_data).unwrap();

        // Compute attention with a key
        let key = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let weights = mann.compute_attention_weights(&key).unwrap();

        assert_eq!(weights.len(), 3);

        // The sum of attention weights should be approximately 1
        let sum: f64 = weights.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);

        // All weights should be non-negative
        for &weight in weights.iter() {
            assert!(weight >= 0.0);
        }

        // The first memory location should have the highest weight
        // since the key matches the first row
        let max_weight = weights.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert_abs_diff_eq!(weights[0], max_weight, epsilon = 1e-10);
    }

    #[test]
    fn test_mann_controller_forward() {
        let mann = MANN::<f64>::new(4, 3, 6, 8, 2);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let output = mann.controller_forward(&input).unwrap();
        assert_eq!(output.len(), 2);

        for &val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_mann_set_get_params() {
        let mut mann = MANN::<f64>::new(2, 2, 4, 4, 2);

        let original_params = mann.get_controller_params().clone();
        let new_params = Array2::zeros(original_params.dim());

        mann.set_controller_params(new_params.clone()).unwrap();
        let retrieved_params = mann.get_controller_params();

        assert_eq!(retrieved_params.dim(), new_params.dim());
        for (&a, &b) in retrieved_params.iter().zip(new_params.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mann_memory_dimensions_validation() {
        let mut mann = MANN::<f64>::new(3, 2, 4, 6, 2);

        // Try to set memory with wrong dimensions
        let wrong_memory = Array2::zeros((2, 3)); // Wrong dimensions
        let result = mann.set_memory(wrong_memory);
        assert!(result.is_err());

        // Set memory with correct dimensions
        let correct_memory = Array2::zeros((3, 2));
        let result = mann.set_memory(correct_memory);
        assert!(result.is_ok());
    }
}
