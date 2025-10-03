//! Sparse training for reduced model size and computation

use crate::error::Result;
use scirs2_core::ndarray::prelude::*;
use std::collections::HashSet;
/// Sparse training configuration
pub struct SparseTrainer {
    /// Target sparsity level (0.0 to 1.0)
    target_sparsity: f32,
    /// Sparsity schedule
    schedule: SparsitySchedule,
    /// Pruning method
    pruning_method: PruningMethod,
    /// Enable structured sparsity
    structured: bool,
    /// Minimum granularity for structured sparsity
    granularity: usize,
}
impl SparseTrainer {
    /// Create a new sparse trainer
    pub fn new(_targetsparsity: f32, schedule: SparsitySchedule) -> Self {
        Self {
            target_sparsity,
            schedule,
            pruning_method: PruningMethod::Magnitude,
            structured: false,
            granularity: 1,
        }
    }
    
    /// Apply sparsity to weights
    pub fn apply_sparsity(
        &self,
        weights: &mut ArrayViewMut2<f32>,
        step: usize,
        layer_name: &str,
    ) -> Result<SparsityStats> {
        let current_sparsity = self.schedule.get_sparsity(step, self.target_sparsity);
        
        match self.pruning_method {
            PruningMethod::Magnitude => self.magnitude_pruning(weights, current_sparsity),
            PruningMethod::Gradient => self.gradient_based_pruning(weights, current_sparsity),
            PruningMethod::Random => self.random_pruning(weights, current_sparsity),
            PruningMethod::Structured => self.structured_pruning(weights, current_sparsity),
    /// Magnitude-based pruning
    fn magnitude_pruning(
        sparsity: f32,
        let total_params = weights.len();
        let params_to_prune = (total_params as f32 * sparsity) as usize;
        // Get absolute values and sort
        let mut weight_magnitudes: Vec<(f32, (usize, usize))> = weights
            .indexed_iter()
            .map(|((i, j), &w)| (w.abs(), (i, j)))
            .collect();
        weight_magnitudes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // Prune smallest magnitude weights
        let mut pruned_count = 0;
        for i in 0..params_to_prune.min(weight_magnitudes.len()) {
            let (_, (row, col)) = weight_magnitudes[i];
            weights[[row, col]] = 0.0;
            pruned_count += 1;
        Ok(SparsityStats {
            total_params,
            pruned_params: pruned_count,
            sparsity: pruned_count as f32 / total_params as f32,
        })
    /// Gradient-based pruning
    fn gradient_based_pruning(
        // Simplified: use magnitude pruning as placeholder
        // In practice, would use gradient information
        self.magnitude_pruning(weights, sparsity)
    /// Random pruning
    fn random_pruning(
        use scirs2_core::random::seq::SliceRandom;
        let mut rng = rng();
        // Get all indices
        let mut indices: Vec<(usize, usize)> = weights
            .map(|((i, j)_)| (i, j))
        indices.shuffle(&mut rng);
        // Prune random weights
        for i in 0..params_to_prune.min(indices.len()) {
            let (row, col) = indices[i];
            pruned_params: params_to_prune,
            sparsity: params_to_prune as f32 / total_params as f32,
    /// Structured pruning (e.g., channel pruning)
    fn structured_pruning(
        let (rows, cols) = weights.dim();
        let channels_to_prune = (cols as f32 * sparsity) as usize;
        // Calculate channel importance (L2 norm)
        let mut channel_importance: Vec<(f32, usize)> = (0..cols)
            .map(|c| {
                let norm = weights.column(c).iter().map(|x| x * x).sum::<f32>().sqrt();
                (norm, c)
            })
        channel_importance.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // Prune least important channels
        let mut pruned_channels = 0;
        for i in 0..channels_to_prune.min(channel_importance.len()) {
            let (_, channel) = channel_importance[i];
            for r in 0..rows {
                weights[[r, channel]] = 0.0;
            }
            pruned_channels += 1;
            total_params: weights.len(),
            pruned_params: pruned_channels * rows,
            sparsity: (pruned_channels * rows) as f32 / weights.len() as f32,
            structured: true,
    /// Get sparse mask
    pub fn get_mask(&self, weights: &ArrayView2<f32>) -> Array2<bool> {
        weights.mapv(|w| w != 0.0)
    /// Apply mask to gradients
    pub fn mask_gradients(
        gradients: &mut ArrayViewMut2<f32>,
        mask: &ArrayView2<bool>,
    ) -> Result<()> {
        gradients.zip_mut_with(mask, |g, &m| {
            if !m {
                *g = 0.0;
        });
        Ok(())
/// Sparsity schedule
#[derive(Debug, Clone)]
pub enum SparsitySchedule {
    /// Constant sparsity
    Constant,
    /// Linear increase
    Linear {
        start_step: usize,
        end_step: usize,
    },
    /// Polynomial increase
    Polynomial {
        power: f32,
    /// Exponential increase
    Exponential {
impl SparsitySchedule {
    /// Get sparsity at current step
    pub fn get_sparsity(&self, step: usize, target: f32) -> f32 {
        match self {
            SparsitySchedule::Constant => target,
            SparsitySchedule::Linear { start_step, end_step } => {
                if step < *start_step {
                    0.0
                } else if step >= *end_step {
                    target
                } else {
                    let progress = (step - start_step) as f32 / (end_step - start_step) as f32;
                    target * progress
                }
            },
            SparsitySchedule::Polynomial { start_step, end_step, power } => {
                    target * progress.powf(*power)
            SparsitySchedule::Exponential { start_step, end_step } => {
                    target * (1.0 - (-5.0 * progress).exp())
/// Pruning method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningMethod {
    Magnitude,
    Gradient,
    Random,
    /// Structured pruning
    Structured,
/// Sparsity statistics
pub struct SparsityStats {
    /// Total number of parameters
    pub total_params: usize,
    /// Number of pruned parameters
    pub pruned_params: usize,
    /// Actual sparsity level
    pub sparsity: f32,
    /// Whether structured sparsity was applied
    pub structured: bool,
/// Dynamic sparse networks
pub struct DynamicSparseNetwork {
    /// Prune and grow ratio
    prune_grow_ratio: f32,
    /// Growth method
    growth_method: GrowthMethod,
    /// Connection history
    connection_history: ConnectionHistory,
impl DynamicSparseNetwork {
    /// Create a new dynamic sparse network
    pub fn new(_prune_growratio: f32) -> Self {
            prune_grow_ratio,
            growth_method: GrowthMethod::Gradient,
            connection_history: ConnectionHistory::new(),
    /// Update connections (prune and grow)
    pub fn update_connections(
        &mut self,
        gradients: &ArrayView2<f32>,
        let num_connections = weights.iter().filter(|&&w| w != 0.0).count();
        let num_to_update = (num_connections as f32 * self._prune_grow_ratio) as usize;
        // Prune weakest connections
        self.prune_connections(weights, num_to_update)?;
        // Grow new connections
        self.grow_connections(weights, gradients, num_to_update)?;
        // Update history
        self.connection_history.update(weights, step);
    /// Prune connections
    fn prune_connections(
        num_to_prune: usize,
        let mut active_weights: Vec<(f32, (usize, usize))> = weights
            .filter(|(_, &w)| w != 0.0)
        activeweights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for i in 0..num_to_prune.min(activeweights.len()) {
            let (_, (row, col)) = active_weights[i];
    /// Grow connections
    fn grow_connections(
        num_to_grow: usize,
        match self.growth_method {
            GrowthMethod::Random => self.random_growth(weights, num_to_grow),
            GrowthMethod::Gradient => self.gradient_based_growth(weights, gradients, num_to_grow),
    /// Random growth
    fn random_growth(
        let mut zero_indices: Vec<(usize, usize)> = weights
            .filter(|(_, &w)| w == 0.0)
        zero_indices.shuffle(&mut rng);
        for i in 0..num_to_grow.min(zero_indices.len()) {
            let (row, col) = zero_indices[i];
            weights[[row, col]] = 0.001 * scirs2_core::random::random::<f32>(); // Small random initialization
    /// Gradient-based growth
    fn gradient_based_growth(
        let mut gradient_magnitudes: Vec<(f32, (usize, usize))> = weights
            .map(|((i, j)_)| (gradients[[i, j]].abs(), (i, j)))
        gradient_magnitudes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for i in 0..num_to_grow.min(gradient_magnitudes.len()) {
            let (_, (row, col)) = gradient_magnitudes[i];
            weights[[row, col]] = 0.001 * gradients[[row, col]].signum();
/// Growth method for dynamic sparse networks
enum GrowthMethod {
/// Connection history for dynamic sparse networks
struct ConnectionHistory {
    history: Vec<HashSet<(usize, usize)>>,
    max_history: usize,
impl ConnectionHistory {
    fn new() -> Self {
            history: Vec::new(),
            max_history: 100,
    fn update(&mut self, weights: &ArrayView2<f32>, step: usize) {
        let active_connections: HashSet<(usize, usize)> = weights
        self.history.push(active_connections);
        if self.history.len() > self.max_history {
            self.history.remove(0);
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_magnitude_pruning() {
        let trainer = SparseTrainer::new(0.5, SparsitySchedule::Constant);
        let mut weights = Array2::from_shape_vec(
            (2, 3),
            vec![0.1, -0.5, 0.2, -0.3, 0.4, -0.6]
        ).unwrap();
        let stats = trainer.magnitude_pruning(&mut weights.view_mut(), 0.5).unwrap();
        assert_eq!(stats.pruned_params, 3);
        assert!((stats.sparsity - 0.5).abs() < 0.01);
        // Check that smallest magnitude values are pruned
        assert_eq!(weights[[0, 0]], 0.0); // 0.1 was smallest
        assert_eq!(weights[[0, 2]], 0.0); // 0.2 was second smallest
        assert_eq!(weights[[1, 0]], 0.0); // -0.3 was third smallest
    fn test_sparsity_schedule() {
        let linear = SparsitySchedule::Linear {
            start_step: 0,
            end_step: 100,
        };
        assert_eq!(linear.get_sparsity(0, 0.9), 0.0);
        assert_eq!(linear.get_sparsity(50, 0.9), 0.45);
        assert_eq!(linear.get_sparsity(100, 0.9), 0.9);
        assert_eq!(linear.get_sparsity(150, 0.9), 0.9);
    fn test_structured_pruning() {
        let mut trainer = SparseTrainer::new(0.5, SparsitySchedule::Constant);
        trainer.structured = true;
            (3, 4),
            vec![0.1, 0.2, 0.3, 0.4,
                 0.5, 0.6, 0.7, 0.8,
                 0.9, 1.0, 1.1, 1.2]
        let stats = trainer.structured_pruning(&mut weights.view_mut(), 0.5).unwrap();
        assert!(stats.structured);
        // Should prune entire columns
        assert_eq!(weights.column(0), Array1::zeros(3));
        assert_eq!(weights.column(1), Array1::zeros(3));
