//! Asynchronous parallel optimization for varying evaluation times
//!
//! This module provides optimization algorithms that can efficiently handle
//! function evaluations with highly variable execution times using asynchronous
//! parallel execution.

use crate::error::OptimizeError;
use crate::unconstrained::OptimizeResult;
use ndarray::{Array1, Array2};
use rand::Rng;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task::JoinHandle;

/// Configuration for asynchronous parallel optimization
#[derive(Debug, Clone)]
pub struct AsyncOptimizationConfig {
    /// Maximum number of concurrent function evaluations
    pub max_workers: usize,
    /// Timeout for individual function evaluations
    pub evaluation_timeout: Option<Duration>,
    /// Maximum time to wait for completing evaluations before termination
    pub completion_timeout: Option<Duration>,
    /// Strategy for handling slow evaluations
    pub slow_evaluation_strategy: SlowEvaluationStrategy,
    /// Minimum number of evaluations to complete before considering termination
    pub min_evaluations: usize,
}

/// Strategy for handling slow function evaluations
#[derive(Debug, Clone)]
pub enum SlowEvaluationStrategy {
    /// Wait for all evaluations to complete
    WaitAll,
    /// Cancel slow evaluations after timeout
    CancelSlow { timeout: Duration },
    /// Use partial results if enough fast evaluations complete
    UsePartial { min_fraction: f64 },
}

/// Evaluation request containing point and metadata
#[derive(Debug, Clone)]
pub struct EvaluationRequest {
    pub id: usize,
    pub point: Array1<f64>,
    pub generation: usize,
    pub submitted_at: Instant,
}

/// Evaluation result with timing information
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub id: usize,
    pub point: Array1<f64>,
    pub value: f64,
    pub generation: usize,
    pub evaluation_time: Duration,
    pub completed_at: Instant,
}

/// Statistics for async optimization
#[derive(Debug, Clone)]
pub struct AsyncOptimizationStats {
    /// Total number of evaluations submitted
    pub total_submitted: usize,
    /// Total number of evaluations completed
    pub total_completed: usize,
    /// Total number of evaluations cancelled/timed out
    pub total_cancelled: usize,
    /// Average evaluation time
    pub avg_evaluation_time: Duration,
    /// Minimum evaluation time
    pub min_evaluation_time: Duration,
    /// Maximum evaluation time
    pub max_evaluation_time: Duration,
    /// Current number of active workers
    pub active_workers: usize,
    /// Total optimization time
    pub total_time: Duration,
}

/// Async differential evolution optimizer
pub struct AsyncDifferentialEvolution {
    config: AsyncOptimizationConfig,
    population_size: usize,
    dimensions: usize,
    bounds: Option<(Array1<f64>, Array1<f64>)>,
    mutation_factor: f64,
    crossover_probability: f64,
    max_generations: usize,
    tolerance: f64,
}

impl Default for AsyncOptimizationConfig {
    fn default() -> Self {
        // Use number of logical CPUs or fallback to 4
        let max_workers = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        Self {
            max_workers,
            evaluation_timeout: Some(Duration::from_secs(300)), // 5 minutes
            completion_timeout: Some(Duration::from_secs(60)),  // 1 minute
            slow_evaluation_strategy: SlowEvaluationStrategy::UsePartial { min, fraction: 0.8 },
            min_evaluations: 10,
        }
    }
}

impl AsyncDifferentialEvolution {
    /// Create new async differential evolution optimizer
    pub fn new(
        dimensions: usize,
        population_size: Option<usize>,
        config: Option<AsyncOptimizationConfig>,
    ) -> Self {
        let pop_size = population_size.unwrap_or(std::cmp::max(4, dimensions * 10));

        Self {
            config: config.unwrap_or_default(),
            population_size: pop_size,
            dimensions,
            bounds: None,
            mutation_factor: 0.8,
            crossover_probability: 0.7,
            max_generations: 1000,
            tolerance: 1e-6,
        }
    }

    /// Set bounds for optimization variables
    pub fn with_bounds(
        mut self,
        lower: Array1<f64>,
        upper: Array1<f64>,
    ) -> Result<Self, OptimizeError> {
        if lower.len() != self.dimensions || upper.len() != self.dimensions {
            return Err(OptimizeError::ValueError(
                "Bounds dimensions must match problem dimensions".to_string(),
            ));
        }

        for (&l, &u) in lower.iter().zip(upper.iter()) {
            if l >= u {
                return Err(OptimizeError::ValueError(
                    "Lower bounds must be less than upper bounds".to_string(),
                ));
            }
        }

        self.bounds = Some((lower, upper));
        Ok(self)
    }

    /// Set differential evolution parameters
    pub fn with_parameters(
        mut self,
        mutation_factor: f64,
        crossover_probability: f64,
        max_generations: usize,
        tolerance: f64,
    ) -> Self {
        self.mutation_factor = mutation_factor;
        self.crossover_probability = crossover_probability;
        self.max_generations = max_generations;
        self.tolerance = tolerance;
        self
    }

    /// Run async differential evolution optimization
    pub async fn optimize<F, Fut>(
        &self,
        objective_fn: F,
    ) -> Result<(OptimizeResult<f64>, AsyncOptimizationStats), OptimizeError>
    where
        F: Fn(Array1<f64>) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = f64> + Send + 'static,
    {
        let start_time = Instant::now();

        // Initialize population
        let mut population = self.initialize_population();
        let mut fitness_values = vec![f64::INFINITY; self.population_size];

        // Create channels for communication
        let (request_tx, request_rx) = mpsc::unbounded_channel::<EvaluationRequest>();
        let (result_tx, mut result_rx) = mpsc::unbounded_channel::<EvaluationResult>();

        // Shared state
        let stats = Arc::new(RwLock::new(AsyncOptimizationStats {
            total_submitted: 0,
            total_completed: 0,
            total_cancelled: 0,
            avg_evaluation_time: Duration::from_millis(0),
            min_evaluation_time: Duration::from_secs(u64::MAX),
            max_evaluation_time: Duration::from_millis(0),
            active_workers: 0,
            total_time: Duration::from_millis(0),
        }));

        // Worker pool for async evaluations
        let worker_handles = self
            .spawn_workers(objective_fn, request_rx, result_tx.clone(), stats.clone())
            .await;

        // Evaluate initial population
        let mut request_id = 0;
        for individual in population.outer_iter() {
            let request = EvaluationRequest {
                id: request_id,
                point: individual.to_owned(),
                generation: 0,
                submitted_at: Instant::now(),
            };

            request_tx.send(request)?;
            request_id += 1;
        }

        // Track pending evaluations
        let mut pending_evaluations = std::collections::HashMap::new();
        let mut best_individual = Array1::zeros(self.dimensions);
        let mut best_fitness = f64::INFINITY;
        let mut generation = 0;
        let mut completed_in_generation = 0;

        // Main optimization loop
        while generation < self.max_generations {
            // Collect completed evaluations with timeout
            let timeout_duration = self
                .config
                .completion_timeout
                .unwrap_or(Duration::from_secs(60));

            match tokio::time::timeout(timeout_duration, result_rx.recv()).await {
                Ok(Some(result)) => {
                    // Update fitness if this was initial population evaluation
                    if result.generation == generation
                        && pending_evaluations.contains_key(&result.id)
                    {
                        if let Some(individual_index) = pending_evaluations.remove(&result.id) {
                            fitness_values[individual_index] = result.value;
                            completed_in_generation += 1;

                            // Update best solution
                            if result.value < best_fitness {
                                best_fitness = result.value;
                                best_individual = result.point.clone();
                            }

                            // Update statistics
                            self.update_stats(&stats, &result).await;
                        }
                    }

                    // Check if generation is complete
                    if completed_in_generation >= self.population_size
                        || self
                            .should_proceed_with_partial_results(&stats, completed_in_generation)
                            .await
                    {
                        // Handle missing evaluations
                        if completed_in_generation < self.population_size {
                            self.handle_incomplete_generation(
                                &mut fitness_values,
                                completed_in_generation,
                            );
                        }

                        // Check convergence
                        if self.check_convergence(&fitness_values) {
                            break;
                        }

                        // Generate new population for next generation
                        generation += 1;
                        completed_in_generation = 0;

                        let new_population =
                            self.generate_next_population(&population, &fitness_values);
                        population = new_population;

                        // Submit evaluations for new generation
                        for (i, individual) in population.outer_iter().enumerate() {
                            let request = EvaluationRequest {
                                id: request_id,
                                point: individual.to_owned(),
                                generation,
                                submitted_at: Instant::now(),
                            };

                            pending_evaluations.insert(request_id, i);
                            request_tx.send(request)?;
                            request_id += 1;
                        }

                        // Reset fitness values for new generation
                        fitness_values.fill(f64::INFINITY);
                    }
                }
                Ok(None) => {
                    // Channel closed
                    break;
                }
                Err(_) => {
                    // Timeout - handle based on strategy
                    match self.config.slow_evaluation_strategy {
                        SlowEvaluationStrategy::WaitAll => continue,
                        SlowEvaluationStrategy::CancelSlow { .. }
                        | SlowEvaluationStrategy::UsePartial { .. } => {
                            if completed_in_generation >= self.config.min_evaluations {
                                self.handle_incomplete_generation(
                                    &mut fitness_values,
                                    completed_in_generation,
                                );
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Cleanup workers
        drop(request_tx);
        for handle in worker_handles {
            let _ = handle.await;
        }

        // Final statistics
        let final_stats = {
            let mut stats_guard = stats.write().await;
            stats_guard.total_time = start_time.elapsed();
            stats_guard.clone()
        };

        let result = OptimizeResult {
            x: best_individual,
            fun: best_fitness,
            nit: generation,
            func_evals: final_stats.total_completed,
            nfev: final_stats.total_completed,
            jacobian: None,
            hessian: None,
            success: best_fitness.is_finite(),
            message: format!(
                "Async differential evolution completed after {} generations",
                generation
            ),
        };

        Ok((result, final_stats))
    }

    /// Initialize random population
    fn initialize_population(&self) -> Array2<f64> {
        let mut population = Array2::zeros((self.population_size, self.dimensions));
        let mut rng = rand::rng();

        if let Some((ref lower, ref upper)) = self.bounds {
            for mut individual in population.outer_iter_mut() {
                for (j, gene) in individual.iter_mut().enumerate() {
                    *gene = lower[j] + rng.gen::<f64>() * (upper[j] - lower[j]);
                }
            }
        } else {
            for mut individual in population.outer_iter_mut() {
                for gene in individual.iter_mut() {
                    *gene = rng.gen::<f64>() * 2.0 - 1.0; // [-1, 1]
                }
            }
        }

        population
    }

    /// Spawn worker tasks for async evaluation
    async fn spawn_workers<F, Fut>(
        &self,
        objective_fn: F,
        request_rx: mpsc::UnboundedReceiver<EvaluationRequest>,
        result_tx: mpsc::UnboundedSender<EvaluationResult>,
        stats: Arc<RwLock<AsyncOptimizationStats>>,
    ) -> Vec<JoinHandle<()>>
    where
        F: Fn(Array1<f64>) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = f64> + Send + 'static,
    {
        let request_rx = Arc::new(Mutex::new(request_rx));
        let mut handles = Vec::new();

        for _worker_id in 0..self.config.max_workers {
            let objective_fn = objective_fn.clone();
            let request_rx = request_rx.clone();
            let result_tx = result_tx.clone();
            let stats = stats.clone();
            let config = self.config.clone();

            let handle = tokio::spawn(async move {
                loop {
                    // Get next evaluation request
                    let request = {
                        let mut _rx = request_rx.lock().await;
                        rx.recv().await
                    };

                    match request {
                        Some(req) => {
                            // Update active workers count
                            {
                                let mut stats_guard = stats.write().await;
                                stats_guard.active_workers += 1;
                                stats_guard.total_submitted += 1;
                            }

                            let start_time = Instant::now();

                            // Evaluate with timeout
                            let evaluation_result = if let Some(timeout) = config.evaluation_timeout
                            {
                                tokio::time::timeout(timeout, objective_fn(req.point.clone())).await
                            } else {
                                Ok(objective_fn(req.point.clone()).await)
                            };

                            let evaluation_time = start_time.elapsed();

                            match evaluation_result {
                                Ok(value) => {
                                    let result = EvaluationResult {
                                        id: req.id,
                                        point: req.point,
                                        value,
                                        generation: req.generation,
                                        evaluation_time,
                                        completed_at: Instant::now(),
                                    };

                                    if result_tx.send(result).is_err() {
                                        break; // Channel closed
                                    }
                                }
                                Err(_) => {
                                    // Timeout occurred
                                    let mut stats_guard = stats.write().await;
                                    stats_guard.total_cancelled += 1;
                                }
                            }

                            // Update active workers count
                            {
                                let mut stats_guard = stats.write().await;
                                stats_guard.active_workers =
                                    stats_guard.active_workers.saturating_sub(1);
                            }
                        }
                        None => break, // Channel closed
                    }
                }
            });

            handles.push(handle);
        }

        handles
    }

    /// Update optimization statistics
    async fn update_stats(
        &self,
        stats: &Arc<RwLock<AsyncOptimizationStats>>,
        result: &EvaluationResult,
    ) {
        let mut stats_guard = stats.write().await;

        stats_guard.total_completed += 1;

        let total_time = stats_guard.avg_evaluation_time * (stats_guard.total_completed - 1) as u32
            + result.evaluation_time;
        stats_guard.avg_evaluation_time = if stats_guard.total_completed > 0 {
            total_time / stats_guard.total_completed as u32
        } else {
            Duration::ZERO // No evaluations completed yet
        };

        if result.evaluation_time < stats_guard.min_evaluation_time {
            stats_guard.min_evaluation_time = result.evaluation_time;
        }

        if result.evaluation_time > stats_guard.max_evaluation_time {
            stats_guard.max_evaluation_time = result.evaluation_time;
        }
    }

    /// Check if we should proceed with partial results
    async fn should_proceed_with_partial_results(
        self_stats: &Arc<RwLock<AsyncOptimizationStats>>,
        completed: usize,
    ) -> bool {
        match self.config.slow_evaluation_strategy {
            SlowEvaluationStrategy::UsePartial { min_fraction } => {
                let fraction = if self.population_size > 0 {
                    completed as f64 / self.population_size as f64
                } else {
                    0.0 // Invalid population size
                };
                fraction >= min_fraction && completed >= self.config.min_evaluations
            }
            _ => false,
        }
    }

    /// Handle incomplete generation by filling missing fitness values
    fn handle_incomplete_generation(&self, fitnessvalues: &mut [f64], completed: usize) {
        // Fill incomplete evaluations with a penalty value
        let max_completed_fitness = fitness_values[..completed]
            .iter()
            .filter(|&&f| f.is_finite())
            .fold(f64::NEG_INFINITY, |acc, &f| acc.max(f));

        let penalty = if max_completed_fitness.is_finite() {
            max_completed_fitness * 2.0
        } else {
            1e6
        };

        for fitness in fitness_values[completed..].iter_mut() {
            *fitness = penalty;
        }
    }

    /// Check convergence based on fitness variance
    fn check_convergence(&self, fitnessvalues: &[f64]) -> bool {
        let finite_fitness: Vec<f64> = fitness_values
            .iter()
            .filter(|&&f| f.is_finite())
            .cloned()
            .collect();

        if finite_fitness.len() < 2 {
            return false;
        }

        let mean = if !finite_fitness.is_empty() {
            finite_fitness.iter().sum::<f64>() / finite_fitness.len() as f64
        } else {
            return false; // No finite fitness _values to check
        };
        let variance = if !finite_fitness.is_empty() {
            finite_fitness
                .iter()
                .map(|&f| (f - mean).powi(2))
                .sum::<f64>()
                / finite_fitness.len() as f64
        } else {
            0.0 // No variance with empty set
        };

        // Safe sqrt - variance should be non-negative by construction
        let std_dev = if variance >= 0.0 {
            variance.sqrt()
        } else {
            0.0 // Handle numerical errors that might produce small negative _values
        };
        std_dev < self.tolerance
    }

    /// Generate next population using differential evolution
    fn generate_next_population(
        &self,
        current_population: &Array2<f64>,
        _fitness_values: &[f64],
    ) -> Array2<f64> {
        let mut new_population = Array2::zeros((self.population_size, self.dimensions));
        let mut rng = rand::rng();

        for i in 0..self.population_size {
            // Select three random individuals (different from current)
            let mut indices = Vec::new();
            while indices.len() < 3 {
                let idx = rng.random_range(0..self.population_size);
                if idx != i && !indices.contains(&idx) {
                    indices.push(idx);
                }
            }

            // Create mutant vector
            let mut mutant = Array1::zeros(self.dimensions);
            for j in 0..self.dimensions {
                mutant[j] = current_population[[indices[0]..j]]
                    + self.mutation_factor
                        * (current_population[[indices[1], j]]
                            - current_population[[indices[2], j]]);
            }

            // Apply bounds
            if let Some((ref lower, ref upper)) = self.bounds {
                for (j, value) in mutant.iter_mut().enumerate() {
                    *value = value.max(lower[j]).min(upper[j]);
                }
            }

            // Crossover
            let mut trial = current_population.row(i).to_owned();
            let r = rng.random_range(0..self.dimensions);

            for j in 0..self.dimensions {
                if j == r || rng.gen::<f64>() < self.crossover_probability {
                    trial[j] = mutant[j];
                }
            }

            new_population.row_mut(i).assign(&trial);
        }

        new_population
    }
}

impl From<mpsc::error::SendError<EvaluationRequest>> for OptimizeError {
    fn from(_: mpsc::error::SendError<EvaluationRequest>) -> Self {
        OptimizeError::ValueError("Failed to send evaluation request".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_async_differential_evolution_simple() {
        // Simple quadratic function
        let objective = |x: Array1<f64>| async move {
            // Simulate some computation time
            sleep(Duration::from_millis(10)).await;
            x.iter().map(|&xi| xi.powi(2)).sum::<f64>()
        };

        let bounds_lower = Array1::from_vec(vec![-5.0..-5.0]);
        let bounds_upper = Array1::from_vec(vec![5.0, 5.0]);

        let optimizer = AsyncDifferentialEvolution::new(2, Some(20), None)
            .with_bounds(bounds_lower, bounds_upper)
            .expect("Setting bounds should succeed for valid dimensions")
            .with_parameters(0.8, 0.7, 50, 1e-6);

        let (result, stats) = optimizer
            .optimize(objective)
            .await
            .expect("Optimization should complete successfully");

        assert!(result.success);
        assert!(result.fun < 1e-3);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-1);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-1);
        assert!(stats.total_completed > 0);
    }

    #[tokio::test]
    async fn test_async_optimization_with_varying_times() {
        use rand::Rng;

        // Function with varying evaluation times
        let objective = |x: Array1<f64>| async move {
            // Simulate varying computation times (10ms to 100ms)
            let delay = rand::rng().random_range(10..=100);
            sleep(Duration::from_millis(delay)).await;

            // Rosenbrock function (2D)
            let a = 1.0 - x[0];
            let b = x[1] - x[0].powi(2);
            a.powi(2) + 100.0 * b.powi(2)
        };

        let bounds_lower = Array1::from_vec(vec![-2.0..-2.0]);
        let bounds_upper = Array1::from_vec(vec![2.0, 2.0]);

        let config = AsyncOptimizationConfig {
            max_workers: 4,
            evaluation_timeout: Some(Duration::from_millis(200)),
            completion_timeout: Some(Duration::from_secs(5)),
            slow_evaluation_strategy: SlowEvaluationStrategy::UsePartial { min, fraction: 0.7 },
            min_evaluations: 5,
        };

        let optimizer = AsyncDifferentialEvolution::new(2, Some(20), Some(config))
            .with_bounds(bounds_lower, bounds_upper)
            .expect("Setting bounds should succeed for valid dimensions")
            .with_parameters(0.8, 0.7, 30, 1e-4);

        let (result, stats) = optimizer
            .optimize(objective)
            .await
            .expect("Optimization should complete successfully");

        assert!(result.success);
        assert!(result.fun < 1.0); // Should get reasonably close to minimum
        assert!(stats.total_completed > 0);
        assert!(stats.avg_evaluation_time > Duration::from_millis(0));

        println!("Async DE Results:");
        println!("  Final solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
        println!("  Final cost: {:.6}", result.fun);
        println!("  Generations: {}", result.nit);
        println!("  Total evaluations: {}", stats.total_completed);
        println!("  Average eval time: {:?}", stats.avg_evaluation_time);
    }

    #[tokio::test]
    async fn test_timeout_handling() {
        // Function that sometimes takes too long
        let objective = |x: Array1<f64>| async move {
            // 50% chance of taking too long
            if rand::rng().gen::<f64>() < 0.5 {
                sleep(Duration::from_secs(1)).await; // Too long
            } else {
                sleep(Duration::from_millis(10)).await; // Normal
            }
            x.iter().map(|&xi| xi.powi(2)).sum::<f64>()
        };

        let config = AsyncOptimizationConfig {
            max_workers: 2,
            evaluation_timeout: Some(Duration::from_millis(100)), // Short timeout
            completion_timeout: Some(Duration::from_millis(500)),
            slow_evaluation_strategy: SlowEvaluationStrategy::CancelSlow {
                timeout: Duration::from_millis(100),
            },
            min_evaluations: 3,
        };

        let bounds_lower = Array1::from_vec(vec![-1.0, -1.0]);
        let bounds_upper = Array1::from_vec(vec![1.0, 1.0]);

        let optimizer = AsyncDifferentialEvolution::new(2, Some(10), Some(config))
            .with_bounds(bounds_lower, bounds_upper)
            .expect("Setting bounds should succeed for valid dimensions")
            .with_parameters(0.8, 0.7, 10, 1e-3);

        let (result, stats) = optimizer
            .optimize(objective)
            .await
            .expect("Optimization should complete successfully");

        // Should still succeed despite timeouts
        assert!(result.success);
        assert!(stats.total_cancelled > 0); // Some evaluations should be cancelled
        assert!(stats.total_completed > 0); // Some should complete
    }
}
