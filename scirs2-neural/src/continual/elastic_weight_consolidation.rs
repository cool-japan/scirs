//! Elastic Weight Consolidation (EWC) implementation
//!
//! EWC helps prevent catastrophic forgetting by adding a regularization term
//! that penalizes changes to important parameters.

use crate::error::Result;
use crate::models::sequential::Sequential;
use scirs2_core::ndarray::prelude::*;
use scirs2_core::ndarray::ArrayView1;

/// EWC configuration
#[derive(Debug, Clone)]
pub struct EWCConfig {
    /// Regularization strength (lambda)
    pub lambda: f32,
    /// Number of samples for Fisher information estimation
    pub num_samples: usize,
    /// Use diagonal Fisher approximation
    pub diagonal_fisher: bool,
    /// Decay factor for old tasks
    pub decay_factor: f32,
    /// Online EWC (accumulate Fisher information)
    pub online: bool,
}

impl Default for EWCConfig {
    fn default() -> Self {
        Self {
            lambda: 1000.0,
            num_samples: 200,
            diagonal_fisher: true,
            decay_factor: 1.0,
            online: false,
        }
    }
}

/// Elastic Weight Consolidation
pub struct EWC {
    config: EWCConfig,
    /// Fisher information matrices for each task
    fisher_matrices: Vec<FisherMatrix>,
    /// Optimal parameters for each task
    optimal_params: Vec<ModelParameters>,
    /// Current task index
    pub current_task: usize,
}

/// Fisher information matrix representation
#[derive(Clone)]
pub(crate) struct FisherMatrix {
    #[allow(dead_code)]
    task_id: usize,
    /// Diagonal Fisher approximation
    diagonal: Option<Vec<Array1<f32>>>,
    /// Full Fisher matrix (if not using diagonal approximation)
    full: Option<Vec<Array2<f32>>>,
}

/// Model parameters snapshot
#[derive(Clone)]
struct ModelParameters {
    #[allow(dead_code)]
    task_id: usize,
    parameters: Vec<Array2<f32>>,
}

impl EWC {
    /// Create a new EWC instance
    pub fn new(config: EWCConfig) -> Self {
        Self {
            config,
            fisher_matrices: Vec::new(),
            optimal_params: Vec::new(),
            current_task: 0,
        }
    }

    /// Compute EWC loss for current parameters
    pub fn compute_loss(&self, current_params: &[Array2<f32>]) -> Result<f32> {
        if self.current_task == 0 {
            return Ok(0.0);
        }
        let mut total_loss = 0.0;
        for (task_idx, (fisher, optimal)) in self
            .fisher_matrices
            .iter()
            .zip(&self.optimal_params)
            .enumerate()
        {
            let task_weight = self
                .config
                .decay_factor
                .powi((self.current_task - task_idx) as i32);
            let task_loss = self.compute_task_loss(current_params, &optimal.parameters, fisher)?;
            total_loss += task_weight * task_loss;
        }
        Ok(self.config.lambda * total_loss)
    }

    /// Compute loss for a single task
    fn compute_task_loss(
        &self,
        current_params: &[Array2<f32>],
        optimal_params: &[Array2<f32>],
        fisher: &FisherMatrix,
    ) -> Result<f32> {
        let mut loss = 0.0;
        if self.config.diagonal_fisher {
            // Diagonal Fisher approximation
            if let Some(ref diagonal) = fisher.diagonal {
                for (i, (curr, opt)) in current_params.iter().zip(optimal_params).enumerate() {
                    if i >= diagonal.len() {
                        continue;
                    }
                    let diff = curr - opt;
                    let fisher_diag = &diagonal[i];
                    let diff_flat = diff.iter().copied().collect::<Vec<_>>();
                    let fisher_flat = fisher_diag.iter().copied().collect::<Vec<_>>();
                    let min_len = diff_flat.len().min(fisher_flat.len());
                    for k in 0..min_len {
                        loss += fisher_flat[k] * diff_flat[k] * diff_flat[k];
                    }
                }
            }
        } else {
            // Full Fisher matrix
            if let Some(ref full) = fisher.full {
                for (i, (curr, opt)) in current_params.iter().zip(optimal_params).enumerate() {
                    if i >= full.len() {
                        continue;
                    }
                    let diff = curr - opt;
                    let fisher_mat = &full[i];
                    let diff_flat = Array1::from_vec(diff.iter().copied().collect());
                    let fisher_diff = fisher_mat.dot(&diff_flat);
                    loss += diff_flat.dot(&fisher_diff);
                }
            }
        }
        Ok(loss / 2.0)
    }

    /// Update Fisher information after training on a task
    pub fn update_fisher_information(
        &mut self,
        model: &Sequential<f32>,
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<()> {
        let num_samples = self.config.num_samples.min(data.shape()[0]);
        let indices: Vec<usize> = (0..data.shape()[0]).collect();
        let sample_indices = &indices[..num_samples];

        // Get model parameters
        let params = self.extract_parameters(model)?;
        let num_params = params.len();

        // Initialize Fisher matrix
        let mut fisher = if self.config.diagonal_fisher {
            FisherMatrix {
                task_id: self.current_task,
                diagonal: Some(vec![Array1::zeros(1); num_params]),
                full: None,
            }
        } else {
            FisherMatrix {
                task_id: self.current_task,
                diagonal: None,
                full: Some(vec![Array2::zeros((1, 1)); num_params]),
            }
        };

        // Estimate Fisher information
        for &idx in sample_indices {
            let sample_data = data.row(idx);
            let sample_label = labels[idx];
            // Compute gradients (simplified placeholder)
            let gradients = self.compute_gradients(model, &sample_data, sample_label)?;
            // Accumulate Fisher information
            self.accumulate_fisher(&mut fisher, &gradients)?;
        }

        // Normalize by number of samples
        self.normalize_fisher(&mut fisher, num_samples as f32)?;

        // Store or update Fisher matrix
        if self.config.online && self.current_task > 0 {
            // Online EWC: merge with previous Fisher
            self.merge_fisher_matrices(&mut fisher)?;
        } else {
            self.fisher_matrices.push(fisher);
        }

        // Store optimal parameters
        self.optimal_params.push(ModelParameters {
            task_id: self.current_task,
            parameters: params,
        });
        self.current_task += 1;
        Ok(())
    }

    /// Extract parameters from model (placeholder)
    fn extract_parameters(&self, _model: &Sequential<f32>) -> Result<Vec<Array2<f32>>> {
        Ok(vec![
            Array2::from_elem((10, 10), 0.5),
            Array2::from_elem((10, 5), 0.3),
        ])
    }

    /// Compute gradients for a sample (placeholder)
    fn compute_gradients(
        &self,
        _model: &Sequential<f32>,
        _data: &ArrayView1<f32>,
        _label: usize,
    ) -> Result<Vec<Array2<f32>>> {
        Ok(vec![
            Array2::from_elem((10, 10), 0.1),
            Array2::from_elem((10, 5), 0.05),
        ])
    }

    /// Accumulate Fisher information from gradients
    #[allow(private_interfaces)]
    pub(crate) fn accumulate_fisher(
        &self,
        fisher: &mut FisherMatrix,
        gradients: &[Array2<f32>],
    ) -> Result<()> {
        if self.config.diagonal_fisher {
            if let Some(ref mut diagonal) = fisher.diagonal {
                for (i, grad) in gradients.iter().enumerate() {
                    if i >= diagonal.len() {
                        diagonal.push(Array1::zeros(grad.len()));
                    }
                    let grad_flat = Array1::from_vec(grad.iter().copied().collect::<Vec<_>>());
                    let grad_sq = &grad_flat * &grad_flat;
                    // Resize diagonal[i] if needed
                    if diagonal[i].len() != grad_sq.len() {
                        diagonal[i] = Array1::zeros(grad_sq.len());
                    }
                    diagonal[i] = &diagonal[i] + &grad_sq;
                }
            }
        } else if let Some(ref mut full) = fisher.full {
            for (i, grad) in gradients.iter().enumerate() {
                let grad_flat = Array1::from_vec(grad.iter().copied().collect::<Vec<_>>());
                let n = grad_flat.len();
                let outer_product =
                    Array2::from_shape_fn((n, n), |(r, c)| grad_flat[r] * grad_flat[c]);
                if i >= full.len() {
                    full.push(outer_product);
                } else {
                    full[i] = &full[i] + &outer_product;
                }
            }
        }
        Ok(())
    }

    /// Normalize Fisher matrix
    fn normalize_fisher(&self, fisher: &mut FisherMatrix, num_samples: f32) -> Result<()> {
        if let Some(ref mut diagonal) = fisher.diagonal {
            for diag in diagonal.iter_mut() {
                *diag /= num_samples;
            }
        }
        if let Some(ref mut full) = fisher.full {
            for mat in full.iter_mut() {
                *mat /= num_samples;
            }
        }
        Ok(())
    }

    /// Merge Fisher matrices for online EWC
    fn merge_fisher_matrices(&mut self, new_fisher: &mut FisherMatrix) -> Result<()> {
        if let Some(last_fisher) = self.fisher_matrices.last_mut() {
            if self.config.diagonal_fisher {
                if let (Some(ref mut last_diag), Some(ref new_diag)) =
                    (&mut last_fisher.diagonal, &new_fisher.diagonal)
                {
                    for (last, new) in last_diag.iter_mut().zip(new_diag) {
                        *last = &*last + new;
                    }
                }
            } else {
                if let (Some(ref mut last_full), Some(ref new_full)) =
                    (&mut last_fisher.full, &new_fisher.full)
                {
                    for (last, new) in last_full.iter_mut().zip(new_full) {
                        *last = &*last + new;
                    }
                }
            }
        } else {
            self.fisher_matrices.push(new_fisher.clone());
        }
        Ok(())
    }

    /// Get importance scores for parameters
    pub fn get_parameter_importance(&self) -> Result<Vec<Array1<f32>>> {
        let mut importance_scores = Vec::new();
        for fisher in &self.fisher_matrices {
            if let Some(ref diagonal) = fisher.diagonal {
                for diag in diagonal {
                    importance_scores.push(diag.clone());
                }
            } else if let Some(ref full) = fisher.full {
                for mat in full {
                    let diag = mat.diag().to_owned();
                    importance_scores.push(diag);
                }
            }
        }
        Ok(importance_scores)
    }

    /// Reset for new task sequence
    pub fn reset(&mut self) {
        self.fisher_matrices.clear();
        self.optimal_params.clear();
        self.current_task = 0;
    }
}

/// EWC regularizer for integration with training loops
pub struct EWCRegularizer {
    ewc: EWC,
    enabled: bool,
}

impl EWCRegularizer {
    /// Create a new EWC regularizer
    pub fn new(config: EWCConfig) -> Self {
        Self {
            ewc: EWC::new(config),
            enabled: true,
        }
    }

    /// Enable/disable regularization
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get regularization loss
    pub fn get_loss(&self, current_params: &[Array2<f32>]) -> Result<f32> {
        if self.enabled {
            self.ewc.compute_loss(current_params)
        } else {
            Ok(0.0)
        }
    }

    /// Update after task completion
    pub fn task_finished(
        &mut self,
        model: &Sequential<f32>,
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<()> {
        self.ewc.update_fisher_information(model, data, labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewc_config_default() {
        let config = EWCConfig::default();
        assert_eq!(config.lambda, 1000.0);
        assert!(config.diagonal_fisher);
    }

    #[test]
    fn test_ewc_initialization() {
        let config = EWCConfig::default();
        let ewc = EWC::new(config);
        assert_eq!(ewc.current_task, 0);
        assert!(ewc.fisher_matrices.is_empty());
    }

    #[test]
    fn test_fisher_matrix_accumulation() {
        let config = EWCConfig::default();
        let mut ewc = EWC::new(config);
        let grad = vec![Array2::from_elem((3, 3), 0.1)];
        let mut fisher = FisherMatrix {
            task_id: 0,
            diagonal: Some(vec![Array1::zeros(9)]),
            full: None,
        };
        ewc.accumulate_fisher(&mut fisher, &grad)
            .expect("accumulate_fisher failed");
        if let Some(ref diagonal) = fisher.diagonal {
            assert!(diagonal[0].iter().all(|&x| x >= 0.0));
        }
    }

    #[test]
    fn test_ewc_regularizer() {
        let config = EWCConfig::default();
        let mut regularizer = EWCRegularizer::new(config);
        regularizer.set_enabled(false);
        let params = vec![Array2::from_elem((5, 5), 1.0)];
        let loss = regularizer.get_loss(&params).expect("get_loss failed");
        assert_eq!(loss, 0.0);
        regularizer.set_enabled(true);
        let loss2 = regularizer.get_loss(&params).expect("get_loss failed");
        assert_eq!(loss2, 0.0); // No previous tasks yet
    }
}
