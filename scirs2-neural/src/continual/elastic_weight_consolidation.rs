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
/// Elastic Weight Consolidation
pub struct EWC {
    config: EWCConfig,
    /// Fisher information matrices for each task
    fisher_matrices: Vec<FisherMatrix>,
    /// Optimal parameters for each task
    optimal_params: Vec<ModelParameters>,
    /// Current task index
    current_task: usize,
/// Fisher information matrix representation
#[derive(Clone)]
struct FisherMatrix {
    task_id: usize,
    /// Diagonal Fisher approximation
    diagonal: Option<Vec<Array1<f32>>>,
    /// Full Fisher matrix (if not using diagonal approximation)
    full: Option<Vec<Array2<f32>>>,
/// Model parameters snapshot
struct ModelParameters {
    parameters: Vec<Array2<f32>>,
impl EWC {
    /// Create a new EWC instance
    pub fn new(config: EWCConfig) -> Self {
            config,
            fisher_matrices: Vec::new(),
            optimal_params: Vec::new(),
            current_task: 0,
    /// Compute EWC loss for current parameters
    pub fn compute_loss(&self, currentparams: &[Array2<f32>]) -> Result<f32> {
        if self.current_task == 0 {
            return Ok(0.0);
        let mut total_loss = 0.0;
        for (task_idx, (fisher, optimal)) in self
            .fisher_matrices
            .iter()
            .zip(&self.optimal_params)
            .enumerate()
        {
            let task_weight = self
                ._config
                .decay_factor
                .powi((self.current_task - task_idx) as i32);
            let task_loss = self.compute_task_loss(current_params, &optimal.parameters, fisher)?;
            total_loss += task_weight * task_loss;
        Ok(self._config.lambda * total_loss)
    /// Compute loss for a single task
    fn compute_task_loss(
        &self,
        current_params: &[Array2<f32>],
        optimal_params: &[Array2<f32>],
        fisher: &FisherMatrix,
    ) -> Result<f32> {
        let mut loss = 0.0;
        if self._config.diagonal_fisher {
            // Diagonal Fisher approximation
            if let Some(ref diagonal) = fisher.diagonal {
                for (i, (curr, opt)) in current_params.iter().zip(optimal_params).enumerate() {
                    let diff = curr - opt;
                    let fisher_diag = &diagonal[i];
                    // Flatten parameters and compute weighted squared difference
                    let diff_flat = diff.as_slice().unwrap();
                    let fisher_flat = fisher_diag.as_slice().unwrap();
                    for (d, f) in diff_flat.iter().zip(fisher_flat) {
                        loss += f * d * d;
                    }
                }
            }
        } else {
            // Full Fisher matrix
            if let Some(ref full) = fisher.full {
                    let fisher_mat = &full[i];
                    // Quadratic form: diff^T * F * diff
                    let diff_flat = Array1::from_vec(diff.as_slice().unwrap().to_vec());
                    let fisher_diff = fisher_mat.dot(&diff_flat);
                    loss += diff_flat.dot(&fisher_diff);
        Ok(loss / 2.0)
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
                diagonal: None,
                full: Some(vec![Array2::zeros((1, 1)); num_params]),
        };
        // Estimate Fisher information
        for &idx in sample_indices {
            let sample_data = data.row(idx);
            let sample_label = labels[idx];
            // Compute gradients (simplified - would use actual autograd)
            let gradients = self.compute_gradients(model, &sample_data, sample_label)?;
            // Accumulate Fisher information
            self.accumulate_fisher(&mut fisher, &gradients)?;
        // Normalize by number of samples
        self.normalize_fisher(&mut fisher, num_samples as f32)?;
        // Store or update Fisher matrix
        if self.config.online && self.current_task > 0 {
            // Online EWC: merge with previous Fisher
            self.merge_fisher_matrices(&mut fisher)?;
            self.fisher_matrices.push(fisher);
        // Store optimal parameters
        self.optimal_params.push(ModelParameters {
            task_id: self.current_task,
            parameters: params,
        });
        self.current_task += 1;
        Ok(())
    /// Extract parameters from model
    fn extract_parameters(&self, model: &Sequential<f32>) -> Result<Vec<Array2<f32>>> {
        // Simplified parameter extraction
        // In practice, would extract actual model weights
        Ok(vec![
            Array2::from_elem((10, 10), 0.5),
            Array2::from_elem((10, 5), 0.3),
        ])
    /// Compute gradients for a sample
    fn compute_gradients(
        data: &ArrayView1<f32>,
        label: usize,
    ) -> Result<Vec<Array2<f32>>> {
        // Simplified gradient computation
        // In practice, would compute actual gradients
            Array2::from_elem((10, 10), 0.1),
            Array2::from_elem((10, 5), 0.05),
    /// Accumulate Fisher information from gradients
    fn accumulate_fisher(
        fisher: &mut FisherMatrix,
        gradients: &[Array2<f32>],
            if let Some(ref mut diagonal) = fisher.diagonal {
                // Update diagonal elements
                for (i, grad) in gradients.iter().enumerate() {
                    if i >= diagonal.len() {
                        diagonal.push(Array1::zeros(grad.len()));
                    let grad_flat = Array1::from_vec(grad.as_slice().unwrap().to_vec());
                    diagonal[i] = &diagonal[i] + &(&grad_flat * &grad_flat);
        } else if let Some(ref mut full) = fisher.full {
            // Update full Fisher matrix
            for (i, grad) in gradients.iter().enumerate() {
                let grad_flat = Array1::from_vec(grad.as_slice().unwrap().to_vec());
                let outer_product = grad_flat
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&grad_flat.insert_axis(Axis(0)));
                if i >= full.len() {
                    full.push(outer_product);
                } else {
                    full[i] = &full[i] + &outer_product;
    /// Normalize Fisher matrix
    fn normalize_fisher(&self, fisher: &mut FisherMatrix, numsamples: f32) -> Result<()> {
                for diag in diagonal.iter_mut() {
                    *diag /= num_samples;
            if let Some(ref mut full) = fisher.full {
                for mat in full.iter_mut() {
                    *mat /= num_samples;
    /// Merge Fisher matrices for online EWC
    fn merge_fisher_matrices(&mut self, newfisher: &mut FisherMatrix) -> Result<()> {
        if let Some(last_fisher) = self.fisher_matrices.last_mut() {
            if self.config.diagonal_fisher {
                if let (Some(ref mut last_diag), Some(ref new_diag)) =
                    (&mut last_fisher.diagonal, &new_fisher.diagonal)
                {
                    for (last, new) in last_diag.iter_mut().zip(new_diag) {
                        *last = last + new;
            } else {
                if let (Some(ref mut last_full), Some(ref new_full)) =
                    (&mut last_fisher.full, &new_fisher.full)
                    for (last, new) in last_full.iter_mut().zip(new_full) {
            self.fisher_matrices.push(new_fisher.clone());
    /// Get importance scores for parameters
    pub fn get_parameter_importance(&self) -> Result<Vec<Array1<f32>>> {
        let mut importance_scores = Vec::new();
        for fisher in &self.fisher_matrices {
                for diag in diagonal {
                    importance_scores.push(diag.clone());
            } else if let Some(ref full) = fisher.full {
                // Extract diagonal from full Fisher matrix
                for mat in full {
                    let diag = mat.diag().to_owned();
                    importance_scores.push(diag);
        Ok(importance_scores)
    /// Reset for new task sequence
    pub fn reset(&mut self) {
        self.fisher_matrices.clear();
        self.optimal_params.clear();
        self.current_task = 0;
/// EWC regularizer for integration with training loops
pub struct EWCRegularizer {
    ewc: EWC,
    enabled: bool,
impl EWCRegularizer {
    /// Create a new EWC regularizer
            ewc: EWC::new(config),
            enabled: true,
    /// Enable/disable regularization
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    /// Get regularization loss
    pub fn get_loss(&self, currentparams: &[Array2<f32>]) -> Result<f32> {
        if self.enabled {
            self.ewc.compute_loss(current_params)
            Ok(0.0)
    /// Update after task completion
    pub fn task_finished(
        self.ewc.update_fisher_information(model, data, labels)
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ewc_config_default() {
        let config = EWCConfig::default();
        assert_eq!(config.lambda, 1000.0);
        assert!(config.diagonal_fisher);
    fn test_ewc_initialization() {
        let ewc = EWC::new(config);
        assert_eq!(ewc.current_task, 0);
        assert!(ewc.fisher_matrices.is_empty());
    fn test_fisher_matrix_accumulation() {
        let mut ewc = EWC::new(config);
        let grad = vec![Array2::from_elem((3, 3), 0.1)];
        let mut fisher = FisherMatrix {
            task_id: 0,
            diagonal: Some(vec![Array1::zeros(9)]),
            full: None,
        ewc.accumulate_fisher(&mut fisher, &grad).unwrap();
        if let Some(ref diagonal) = fisher.diagonal {
            assert!(diagonal[0].iter().all(|&x| x > 0.0));
    fn test_ewc_regularizer() {
        let mut regularizer = EWCRegularizer::new(config);
        regularizer.set_enabled(false);
        let params = vec![Array2::from_elem((5, 5), 1.0)];
        let loss = regularizer.get_loss(&params).unwrap();
        assert_eq!(loss, 0.0);
        regularizer.set_enabled(true);
        assert_eq!(loss, 0.0); // No previous tasks yet
