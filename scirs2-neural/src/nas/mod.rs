//! Neural Architecture Search (NAS) module
//!
//! This module provides automated neural architecture search capabilities,
//! allowing for automatic discovery of optimal network architectures for
//! specific tasks.

pub mod architecture_encoding;
pub mod controller;
pub mod enas;
pub mod evaluator;
pub mod hardware_aware;
pub mod multi_objective;
pub mod performance_estimation;
pub mod progressive_search;
pub mod search_algorithms;
pub mod search_space;
pub use architecture_encoding::{ArchitectureEncoding, GraphEncoding, SequentialEncoding};
pub use controller::{ControllerConfig, NASController};
pub use enas::{ENASController, ENASTrainer, SuperNetwork};
pub use evaluator::{ArchitectureEvaluator, EvaluationMetrics};
pub use hardware_aware::{HardwareAwareSearch, HardwareConstraints, LatencyPredictor};
pub use multi_objective::{
    MultiObjectiveAlgorithm, MultiObjectiveConfig, MultiObjectiveOptimizer, MultiObjectiveSolution,
    Objective,
};
pub use performance_estimation::{
    EarlyStoppingEstimator, LearningCurveEstimator, MultiFidelityEstimator, PerformanceEstimator,
    SuperNetEstimator, ZeroCostEstimator,
pub use progressive_search::{ProgressiveConfig, ProgressiveSearch};
pub use search_algorithms::{
    BayesianOptimization, DifferentiableSearch, EvolutionarySearch, RandomSearch,
    ReinforcementSearch, SearchAlgorithm,
pub use search_space::{SearchSpace, SearchSpaceConfig};
use crate::error::Result;
use crate::models::sequential::Sequential;
use scirs2_core::ndarray::prelude::*;
use std::sync::Arc;
use scirs2_core::ndarray::ArrayView1;
/// Configuration for Neural Architecture Search
pub struct NASConfig {
    /// Search space configuration
    pub search_space: SearchSpaceConfig,
    /// Search algorithm to use
    pub search_algorithm: Box<dyn SearchAlgorithm>,
    /// Performance estimation strategy
    pub performance_estimator: Box<dyn PerformanceEstimator>,
    /// Maximum number of architectures to evaluate
    pub max_evaluations: usize,
    /// Time budget in seconds
    pub time_budget: Option<u64>,
    /// GPU memory budget in MB
    pub memory_budget: Option<usize>,
    /// Target metric to optimize
    pub target_metric: String,
    /// Whether to minimize or maximize the target metric
    pub minimize: bool,
    /// Number of parallel evaluations
    pub parallel_evaluations: usize,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
}
impl Default for NASConfig {
    fn default() -> Self {
        Self {
            search_space: SearchSpaceConfig::default(),
            search_algorithm: Box::new(RandomSearch::new()),
            performance_estimator: Box::new(EarlyStoppingEstimator::new(10)),
            max_evaluations: 100,
            time_budget: None,
            memory_budget: None,
            target_metric: "validation_accuracy".to_string(),
            minimize: false,
            parallel_evaluations: 1,
            early_stopping_patience: Some(10),
        }
    }
/// Main Neural Architecture Search engine
pub struct NeuralArchitectureSearch {
    config: NASConfig,
    controller: NASController,
    evaluator: ArchitectureEvaluator,
    best_architecture: Option<Arc<dyn ArchitectureEncoding>>,
    search_history: Vec<SearchResult>,
    /// Multi-objective optimizer for handling multiple objectives
    multi_objective_optimizer: Option<MultiObjectiveOptimizer>,
    /// Progressive search for adaptive search space
    progressive_search: Option<ProgressiveSearch>,
    /// Hardware-aware constraints
    hardware_constraints: Option<HardwareConstraints>,
/// Result of a single architecture evaluation
#[derive(Clone)]
pub struct SearchResult {
    /// The evaluated architecture
    pub architecture: Arc<dyn ArchitectureEncoding>,
    /// Performance metrics
    pub metrics: EvaluationMetrics,
    /// Training time in seconds
    pub training_time: f64,
    /// Model parameters count
    pub parameter_count: usize,
    /// Model FLOPs
    pub flops: Option<usize>,
impl NeuralArchitectureSearch {
    /// Create a new NAS instance
    pub fn new(config: NASConfig) -> Result<Self> {
        let controller = NASController::new(_config.search_space.clone())?;
        let evaluator = ArchitectureEvaluator::new(ControllerConfig::default())?;
        Ok(Self {
            config,
            controller,
            evaluator,
            best_architecture: None,
            search_history: Vec::new(),
        })
    /// Run the architecture search
    pub fn search(
        &mut self,
        train_data: &ArrayView2<f32>,
        train_labels: &ArrayView1<usize>,
        val_data: &ArrayView2<f32>,
        val_labels: &ArrayView1<usize>,
    ) -> Result<Arc<dyn ArchitectureEncoding>> {
        let start_time = std::time::Instant::now();
        let mut evaluations = 0;
        let mut no_improvement_count = 0;
        let mut best_metric = if self.config.minimize {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        while evaluations < self.config.max_evaluations {
            // Check time budget
            if let Some(budget) = self.config.time_budget {
                if start_time.elapsed().as_secs() > budget {
                    break;
                }
            }
            // Generate architectures to evaluate
            let architectures = self.config.search_algorithm.propose_architectures(
                &self.search_history,
                self.config
                    .parallel_evaluations
                    .min(self.config.max_evaluations - evaluations),
            )?;
            // Evaluate architectures in parallel
            let results: Vec<SearchResult> = architectures
                .into_iter()
                .map(|arch| {
                    self.evaluate_architecture(arch, train_data, train_labels, val_data, val_labels)
                })
                .collect::<Result<Vec<_>>>()?;
            // Update search history and best architecture
            for result in results {
                evaluations += 1;
                self.search_history.push(result.clone());
                let current_metric =
                    result
                        .metrics
                        .get(&self.config.target_metric)
                        .ok_or_else(|| {
                            crate::error::NeuralError::InvalidArgument(format!(
                                "Target metric {} not found",
                                self.config.target_metric
                            ))
                        })?;
                let is_better = if self.config.minimize {
                    current_metric < best_metric
                } else {
                    current_metric > best_metric
                };
                if is_better {
                    best_metric = current_metric;
                    self.best_architecture = Some(result.architecture.clone());
                    no_improvement_count = 0;
                    no_improvement_count += 1;
            // Early stopping
            if let Some(patience) = self.config.early_stopping_patience {
                if no_improvement_count >= patience {
        self.best_architecture.clone().ok_or_else(|| {
            crate::error::NeuralError::InvalidArchitecture("No architecture found".to_string())
    /// Evaluate a single architecture
    fn evaluate_architecture(
        &self,
        architecture: Arc<dyn ArchitectureEncoding>,
    ) -> Result<SearchResult> {
        // Build model from architecture encoding
        let model = self.controller.build_model(&architecture)?;
        // Estimate performance using the configured strategy
        let metrics = self.config.performance_estimator.estimate(
            &model,
            train_data,
            train_labels,
            val_data,
            val_labels,
        )?;
        let training_time = start_time.elapsed().as_secs_f64();
        let parameter_count = self.controller.count_parameters(&model)?;
        let flops = self.controller.estimate_flops(&model, train_data.shape())?;
        Ok(SearchResult {
            architecture,
            metrics,
            training_time,
            parameter_count,
            flops: Some(flops),
    /// Get the best architecture found
    pub fn best_architecture(&self) -> Option<&Arc<dyn ArchitectureEncoding>> {
        self.best_architecture.as_ref()
    /// Get the search history
    pub fn search_history(&self) -> &[SearchResult] {
        &self.search_history
    /// Build a model from the best architecture
    pub fn build_best_model(&self) -> Result<Sequential<f32>> {
        let arch = self.best_architecture.as_ref().ok_or_else(|| {
            crate::error::NeuralError::InvalidArchitecture("No best architecture found".to_string())
        })?;
        self.controller.build_model(arch)
    /// Export search results to a file
    pub fn export_results(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;
        let mut file = File::create(path)?;
        writeln!(file, "# Neural Architecture Search Results")?;
        writeln!(file, "## Configuration")?;
        writeln!(file, "- Max evaluations: {}", self.config.max_evaluations)?;
        writeln!(file, "- Target metric: {}", self.config.target_metric)?;
        writeln!(file, "- Minimize: {}", self.config.minimize)?;
        writeln!(file)?;
        writeln!(file, "## Search History")?;
        for (i, result) in self.search_history.iter().enumerate() {
            writeln!(file, "### Architecture {}", i + 1)?;
            writeln!(file, "- Training time: {:.2}s", result.training_time)?;
            writeln!(file, "- Parameters: {}", result.parameter_count)?;
            if let Some(flops) = result.flops {
                writeln!(file, "- FLOPs: {}", flops)?;
            writeln!(file, "- Metrics:")?;
            for (metric, value) in result.metrics.iter() {
                writeln!(file, "  - {}: {:.4}", metric, value)?;
            writeln!(file)?;
        if let Some(best) = &self.best_architecture {
            writeln!(file, "## Best Architecture")?;
            writeln!(file, "{}", best.to_string())?;
        Ok(())
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_nas_config_default() {
        let config = NASConfig::default();
        assert_eq!(config.max_evaluations, 100);
        assert_eq!(config.target_metric, "validation_accuracy");
        assert!(!config.minimize);
