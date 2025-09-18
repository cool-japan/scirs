//! Neural architecture search for optimizer design
//!
//! This module provides comprehensive neural architecture search functionality
//! specifically tailored for discovering optimal learned optimizer architectures.

pub mod architecture;
pub mod search;
pub mod evaluation;
pub mod space;
pub mod population;
pub mod history;

use std::collections::HashMap;
use num_traits::Float;

// Re-export key types
pub use architecture::*;
pub use search::*;
pub use evaluation::*;
pub use space::*;
pub use population::*;
pub use history::*;

/// Main Neural Architecture Search system for optimizer design
pub struct NeuralArchitectureSearch<T: Float> {
    /// Search configuration
    config: NASConfig,

    /// Architecture search space
    search_space: ArchitectureSearchSpace,

    /// Search strategy
    search_strategy: SearchStrategy<T>,

    /// Architecture evaluator
    evaluator: ArchitectureEvaluator,

    /// Population manager (for evolutionary search)
    population_manager: PopulationManager<T>,

    /// Performance predictor
    performance_predictor: Option<PerformancePredictor>,

    /// Search history
    search_history: SearchHistory<T>,

    /// Resource manager
    resource_manager: ResourceManager,

    /// Multi-objective optimizer
    multi_objective_optimizer: Option<MultiObjectiveOptimizer<T>>,
}

/// NAS configuration
#[derive(Debug, Clone, Default)]
pub struct NASConfig {
    /// Search strategy type
    pub search_strategy: SearchStrategyType,

    /// Maximum search iterations
    pub max_iterations: usize,

    /// Population size (for evolutionary strategies)
    pub population_size: usize,

    /// Number of top architectures to keep
    pub elite_size: usize,

    /// Mutation rate for evolutionary search
    pub mutation_rate: f64,

    /// Crossover rate for evolutionary search
    pub crossover_rate: f64,

    /// Early stopping patience
    pub early_stopping_patience: usize,

    /// Evaluation budget (computational resources)
    pub evaluation_budget: usize,

    /// Multi-objective weights
    pub objective_weights: Vec<f64>,

    /// Enable performance prediction
    pub enable_performance_prediction: bool,

    /// Enable progressive search
    pub progressive_search: bool,

    /// Search space constraints
    pub constraints: SearchConstraints,

    /// Parallelization level
    pub parallelization_level: usize,

    /// Enable transfer learning
    pub enable_transfer_learning: bool,

    /// Warm start from existing architectures
    pub warm_start_architectures: Vec<String>,
}

/// Resource manager for computational resources
pub struct ResourceManager {
    /// Available compute budget
    pub compute_budget: f64,

    /// Used compute resources
    pub compute_used: f64,

    /// Memory budget (GB)
    pub memory_budget: f64,

    /// Used memory
    pub memory_used: f64,

    /// Time budget (seconds)
    pub time_budget: f64,

    /// Time used
    pub time_used: f64,

    /// Resource utilization history
    pub utilization_history: Vec<ResourceUtilization>,
}

/// Resource utilization record
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// Timestamp
    pub timestamp: std::time::Instant,

    /// Compute utilization (0.0-1.0)
    pub compute_util: f64,

    /// Memory utilization (0.0-1.0)
    pub memory_util: f64,

    /// Current operation
    pub operation: String,
}

/// Multi-objective optimizer
pub struct MultiObjectiveOptimizer<T: Float> {
    /// Objective functions
    pub objectives: Vec<ObjectiveFunction<T>>,

    /// Pareto front
    pub pareto_front: Vec<ArchitectureCandidate>,

    /// Multi-objective algorithm
    pub algorithm: MultiObjectiveAlgorithm,

    /// Hypervolume history
    pub hypervolume_history: Vec<f64>,
}

/// Objective function
pub struct ObjectiveFunction<T: Float> {
    /// Function name
    pub name: String,

    /// Function implementation
    pub function: Box<dyn Fn(&ArchitectureCandidate) -> T>,

    /// Weight in multi-objective optimization
    pub weight: f64,

    /// Minimization or maximization
    pub maximize: bool,
}

impl<T: Float + Default + std::fmt::Debug + Clone + From<f64> + Into<f64>> NeuralArchitectureSearch<T> {
    /// Create new NAS system
    pub fn new(config: NASConfig) -> Self {
        let search_space = ArchitectureSearchSpace::default();
        let search_strategy = SearchStrategy::new(config.search_strategy, SearchConfig {
            budget: config.evaluation_budget,
            population_size: config.population_size,
            max_iterations: config.max_iterations,
            convergence_threshold: 0.001,
            early_stopping_patience: config.early_stopping_patience,
            random_seed: None,
            strategy_params: HashMap::new(),
        });

        let evaluator = ArchitectureEvaluator {
            config: EvaluationConfig::default(),
            predictor: if config.enable_performance_prediction {
                Some(PerformancePredictor {
                    predictor_type: PredictorType::NeuralNetwork,
                    training_data: Vec::new(),
                    accuracy: 0.0,
                })
            } else {
                None
            },
            evaluation_history: Vec::new(),
        };

        let population_manager = PopulationManager::new(PopulationConfig {
            size: config.population_size,
            elite_size: config.elite_size,
            maintain_diversity: true,
            min_diversity: 0.1,
            max_generations: config.max_iterations,
        });

        let search_history = SearchHistory::new(HistoryConfig::default());

        Self {
            config,
            search_space,
            search_strategy,
            evaluator,
            population_manager,
            performance_predictor: None,
            search_history,
            resource_manager: ResourceManager::new(),
            multi_objective_optimizer: None,
        }
    }

    /// Run the neural architecture search
    pub fn search(&mut self) -> Result<Vec<ArchitectureCandidate>, NASError> {
        // Initialize search
        self.initialize_search()?;

        // Main search loop
        for iteration in 0..self.config.max_iterations {
            // Check resource constraints
            if !self.resource_manager.has_resources() {
                break;
            }

            // Generate candidate architecture
            let candidate = self.search_strategy.generate_candidate()?;

            // Evaluate candidate
            let performance = self.evaluate_candidate(&candidate)?;

            // Update search strategy with result
            self.search_strategy.update_with_result(&candidate, performance);

            // Record in history
            self.search_history.add_record(
                candidate,
                performance,
                iteration,
                format!("{:?}", self.config.search_strategy),
            );

            // Check convergence
            if self.search_history.stats.convergence.converged {
                break;
            }

            // Early stopping check
            if self.should_early_stop() {
                break;
            }
        }

        // Return best architectures found
        Ok(self.search_history.best_architectures.clone())
    }

    /// Initialize search
    fn initialize_search(&mut self) -> Result<(), NASError> {
        // Initialize population if using evolutionary strategy
        if matches!(self.config.search_strategy, SearchStrategyType::Evolutionary) {
            self.population_manager.initialize(&self.search_space);
        }

        // Warm start if enabled
        if self.config.enable_transfer_learning && !self.config.warm_start_architectures.is_empty() {
            self.warm_start()?;
        }

        Ok(())
    }

    /// Evaluate architecture candidate
    fn evaluate_candidate(&mut self, candidate: &ArchitectureCandidate) -> Result<T, NASError> {
        let start_time = std::time::Instant::now();

        // Use performance predictor if available and configured
        if let Some(ref predictor) = self.evaluator.predictor {
            if self.config.enable_performance_prediction && predictor.accuracy > 0.8 {
                return self.predict_performance(candidate);
            }
        }

        // Full evaluation (simplified - would run actual training/validation)
        let performance = self.full_evaluation(candidate)?;

        // Record evaluation
        let eval_record = EvaluationRecord {
            architecture_id: candidate.id.clone(),
            performance: PerformanceMetrics {
                optimization_performance: performance.into(),
                convergence_speed: 0.8,
                generalization: 0.7,
                robustness: 0.6,
                transfer_performance: 0.5,
                multitask_performance: 0.4,
                stability: 0.9,
            },
            resource_usage: ResourceUsage::default(),
            evaluation_time: start_time.elapsed(),
            evaluation_type: EvaluationType::Full,
        };

        self.evaluator.evaluation_history.push(eval_record);

        // Update resource usage
        self.resource_manager.update_usage(start_time.elapsed().as_secs_f64(), "evaluation");

        Ok(performance)
    }

    /// Predict performance using predictor
    fn predict_performance(&self, _candidate: &ArchitectureCandidate) -> Result<T, NASError> {
        // Simplified prediction
        Ok(num_traits::cast::cast(0.7).unwrap_or_else(|| T::zero())) // Placeholder
    }

    /// Full evaluation of architecture
    fn full_evaluation(&self, candidate: &ArchitectureCandidate) -> Result<T, NASError> {
        // Simplified evaluation based on architecture properties
        let param_count = candidate.architecture.parameter_count();
        let layer_count = candidate.architecture.layers.len();

        // Simple heuristic: balance between capacity and efficiency
        let capacity_score = (param_count as f64 / 1000000.0).min(1.0); // Normalize to [0,1]
        let efficiency_score = 1.0 / (layer_count as f64 / 10.0 + 1.0); // Prefer fewer layers

        let performance = 0.3 * capacity_score + 0.7 * efficiency_score;
        Ok(num_traits::cast::cast(performance).unwrap_or_else(|| T::zero()))
    }

    /// Check if should early stop
    fn should_early_stop(&self) -> bool {
        // Early stopping based on convergence
        if self.search_history.stats.convergence.converged {
            return true;
        }

        // Early stopping based on resource exhaustion
        if !self.resource_manager.has_resources() {
            return true;
        }

        // Early stopping based on stagnation
        let stagnation_limit = self.config.early_stopping_patience;
        if self.search_history.stats.convergence.stagnation_count >= stagnation_limit {
            return true;
        }

        false
    }

    /// Warm start with existing architectures
    fn warm_start(&mut self) -> Result<(), NASError> {
        // Simplified warm start - would load actual architectures
        for arch_id in &self.config.warm_start_architectures {
            let arch = self.search_space.sample_random(); // Placeholder
            let candidate = ArchitectureCandidate::new(arch_id.clone(), arch);
            let performance = self.evaluate_candidate(&candidate)?;
            
            self.search_history.add_record(
                candidate,
                performance,
                0,
                "warm_start".to_string(),
            );
        }
        
        Ok(())
    }

    /// Get search statistics
    pub fn get_statistics(&self) -> SearchSummary<T> {
        self.search_history.get_summary()
    }

    /// Get best architectures
    pub fn get_best_architectures(&self, n: usize) -> Vec<&ArchitectureCandidate> {
        self.search_history.best_architectures.iter().take(n).collect()
    }
}

impl ResourceManager {
    /// Create new resource manager
    pub fn new() -> Self {
        Self {
            compute_budget: 1000.0, // 1000 compute units
            compute_used: 0.0,
            memory_budget: 32.0,    // 32 GB
            memory_used: 0.0,
            time_budget: 3600.0,    // 1 hour
            time_used: 0.0,
            utilization_history: Vec::new(),
        }
    }

    /// Check if resources are available
    pub fn has_resources(&self) -> bool {
        self.compute_used < self.compute_budget
            && self.memory_used < self.memory_budget
            && self.time_used < self.time_budget
    }

    /// Update resource usage
    pub fn update_usage(&mut self, time_delta: f64, operation: &str) {
        self.time_used += time_delta;
        self.compute_used += time_delta * 0.1; // Simplified compute usage
        self.memory_used += 0.1;               // Simplified memory usage

        self.utilization_history.push(ResourceUtilization {
            timestamp: std::time::Instant::now(),
            compute_util: self.compute_used / self.compute_budget,
            memory_util: self.memory_used / self.memory_budget,
            operation: operation.to_string(),
        });
    }
}

/// NAS-specific errors
#[derive(Debug)]
pub enum NASError {
    SearchError(SearchError),
    EvaluationError(String),
    ResourceError(String),
    ConfigurationError(String),
    InitializationError(String),
}

impl From<SearchError> for NASError {
    fn from(err: SearchError) -> Self {
        NASError::SearchError(err)
    }
}

impl std::fmt::Display for NASError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NASError::SearchError(err) => write!(f, "Search error: {}", err),
            NASError::EvaluationError(msg) => write!(f, "Evaluation error: {}", msg),
            NASError::ResourceError(msg) => write!(f, "Resource error: {}", msg),
            NASError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            NASError::InitializationError(msg) => write!(f, "Initialization error: {}", msg),
        }
    }
}

impl std::error::Error for NASError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nas_creation() {
        let config = NASConfig::default();
        let nas: NeuralArchitectureSearch<f64> = NeuralArchitectureSearch::new(config);
        
        assert_eq!(nas.config.search_strategy, SearchStrategyType::Random);
        assert!(!nas.search_history.records.is_empty() == false);
    }

    #[test]
    fn test_resource_manager() {
        let mut manager = ResourceManager::new();
        assert!(manager.has_resources());
        
        manager.update_usage(100.0, "test");
        assert!(manager.time_used > 0.0);
    }

    #[test]
    fn test_search_initialization() {
        let config = NASConfig {
            search_strategy: SearchStrategyType::Evolutionary,
            population_size: 20,
            ..Default::default()
        };
        
        let mut nas: NeuralArchitectureSearch<f64> = NeuralArchitectureSearch::new(config);
        let result = nas.initialize_search();
        assert!(result.is_ok());
    }
}