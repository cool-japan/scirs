//! Multi-objective optimization algorithms
//!
//! This module provides implementations of various multi-objective optimization algorithms:
//! - NSGA-II (Non-dominated Sorting Genetic Algorithm II)
//! - NSGA-III (Non-dominated Sorting Genetic Algorithm III)
//! - MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
//! - SPEA2 (Strength Pareto Evolutionary Algorithm 2)

pub mod moead;
pub mod nsga2;
pub mod nsga3;
pub mod spea2;

pub use moead::MOEAD;
pub use nsga2::NSGAII;
pub use nsga3::NSGAIII;
pub use spea2::SPEA2;

use super::solutions::{MultiObjectiveResult, MultiObjectiveSolution, Population};
use crate::error::OptimizeError;
use ndarray::{Array1, ArrayView1};

/// Configuration for multi-objective optimization algorithms
#[derive(Debug, Clone)]
pub struct MultiObjectiveConfig {
    /// Population size
    pub population_size: usize,
    /// Maximum number of generations
    pub max_generations: usize,
    /// Maximum number of function evaluations
    pub max_evaluations: Option<usize>,
    /// Crossover probability
    pub crossover_probability: f64,
    /// Mutation probability
    pub mutation_probability: f64,
    /// Mutation strength (eta for polynomial mutation)
    pub mutation_eta: f64,
    /// Crossover distribution index (eta for SBX)
    pub crossover_eta: f64,
    /// Convergence tolerance for hypervolume
    pub tolerance: f64,
    /// Reference point for hypervolume calculation
    pub reference_point: Option<Array1<f64>>,
    /// Variable bounds (lower, upper)
    pub bounds: Option<(Array1<f64>, Array1<f64>)>,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Archive size (for algorithms that use archives)
    pub archive_size: Option<usize>,
    /// Neighborhood size (for MOEA/D)
    pub neighborhood_size: Option<usize>,
}

impl Default for MultiObjectiveConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_generations: 250,
            max_evaluations: None,
            crossover_probability: 0.9,
            mutation_probability: 0.1,
            mutation_eta: 20.0,
            crossover_eta: 15.0,
            tolerance: 1e-6,
            reference_point: None,
            bounds: None,
            random_seed: None,
            archive_size: None,
            neighborhood_size: None,
        }
    }
}

/// Trait for multi-objective optimization algorithms
pub trait MultiObjectiveOptimizer {
    /// Optimize the problem
    fn optimize<F>(&mut self, objective_function: F) -> Result<MultiObjectiveResult, OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync;

    /// Initialize the population
    fn initialize_population(&mut self) -> Result<(), OptimizeError>;

    /// Perform one generation/iteration
    fn evolve_generation<F>(&mut self, objective_function: &F) -> Result<(), OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync;

    /// Check convergence criteria
    fn check_convergence(&self) -> bool;

    /// Get current population
    fn get_population(&self) -> &Population;

    /// Get current generation number
    fn get_generation(&self) -> usize;

    /// Get number of function evaluations
    fn get_evaluations(&self) -> usize;

    /// Get algorithm name
    fn name(&self) -> &str;
}

/// Wrapper enum for different multi-objective optimizers
pub enum MultiObjectiveOptimizerWrapper {
    NSGAII(NSGAII),
    NSGAIII(NSGAIII),
    MOEAD(MOEAD),
    SPEA2(SPEA2),
}

impl MultiObjectiveOptimizerWrapper {
    pub fn optimize<F>(
        &mut self,
        objective_function: F,
    ) -> Result<MultiObjectiveResult, OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        match self {
            Self::NSGAII(opt) => opt.optimize(objective_function),
            Self::NSGAIII(opt) => opt.optimize(objective_function),
            Self::MOEAD(opt) => opt.optimize(objective_function),
            Self::SPEA2(opt) => opt.optimize(objective_function),
        }
    }
}

/// Factory for creating multi-objective optimizers
pub struct OptimizerFactory;

impl OptimizerFactory {
    /// Create NSGA-II optimizer
    pub fn create_nsga2(
        config: MultiObjectiveConfig,
        n_objectives: usize,
        n_variables: usize,
    ) -> Result<NSGAII, OptimizeError> {
        NSGAII::new(config, n_objectives, n_variables)
    }

    /// Create NSGA-III optimizer
    pub fn create_nsga3(
        config: MultiObjectiveConfig,
        n_objectives: usize,
        n_variables: usize,
        reference_points: Option<Vec<Array1<f64>>>,
    ) -> Result<NSGAIII, OptimizeError> {
        // TODO: Use reference_points when NSGA-III is fully implemented
        Ok(NSGAIII::new(
            config.population_size,
            n_objectives,
            n_variables,
        ))
    }

    /// Create optimizer by name
    pub fn create_by_name(
        algorithm: &str,
        config: MultiObjectiveConfig,
        n_objectives: usize,
        n_variables: usize,
    ) -> Result<MultiObjectiveOptimizerWrapper, OptimizeError> {
        match algorithm.to_lowercase().as_str() {
            "nsga2" | "nsga-ii" => Ok(MultiObjectiveOptimizerWrapper::NSGAII(Self::create_nsga2(
                config,
                n_objectives,
                n_variables,
            )?)),
            "nsga3" | "nsga-iii" => Ok(MultiObjectiveOptimizerWrapper::NSGAIII(
                Self::create_nsga3(config, n_objectives, n_variables, None)?,
            )),
            _ => Err(OptimizeError::InvalidInput(format!(
                "Unknown algorithm: {}",
                algorithm
            ))),
        }
    }
}

/// Utility functions for multi-objective optimization
pub mod utils {
    use super::*;
    use ndarray::Array2;

    /// Generate Das-Dennis reference points for NSGA-III
    pub fn generate_das_dennis_points(
        n_objectives: usize,
        n_partitions: usize,
    ) -> Vec<Array1<f64>> {
        if n_objectives == 1 {
            return vec![Array1::from_vec(vec![1.0])];
        }

        let mut points = Vec::new();
        generate_das_dennis_recursive(
            &mut points,
            Array1::zeros(n_objectives),
            0,
            n_objectives,
            n_partitions,
            n_partitions,
        );

        // Normalize points
        for point in &mut points {
            let sum: f64 = point.sum();
            if sum > 0.0 {
                *point /= sum;
            }
        }

        points
    }

    fn generate_das_dennis_recursive(
        points: &mut Vec<Array1<f64>>,
        mut current_point: Array1<f64>,
        index: usize,
        n_objectives: usize,
        n_partitions: usize,
        remaining: usize,
    ) {
        if index == n_objectives - 1 {
            current_point[index] = remaining as f64;
            points.push(current_point);
        } else {
            for i in 0..=remaining {
                current_point[index] = i as f64;
                generate_das_dennis_recursive(
                    points,
                    current_point.clone(),
                    index + 1,
                    n_objectives,
                    n_partitions,
                    remaining - i,
                );
            }
        }
    }

    /// Calculate hypervolume indicator
    pub fn calculate_hypervolume(
        pareto_front: &[MultiObjectiveSolution],
        reference_point: &Array1<f64>,
    ) -> f64 {
        if pareto_front.is_empty() {
            return 0.0;
        }

        // For 2D case, use simple calculation
        if reference_point.len() == 2 {
            return calculate_hypervolume_2d(pareto_front, reference_point);
        }

        // For higher dimensions, use Monte Carlo approximation
        calculate_hypervolume_monte_carlo(pareto_front, reference_point, 10000)
    }

    fn calculate_hypervolume_2d(
        pareto_front: &[MultiObjectiveSolution],
        reference_point: &Array1<f64>,
    ) -> f64 {
        let mut points: Vec<_> = pareto_front
            .iter()
            .map(|sol| (sol.objectives[0], sol.objectives[1]))
            .collect();

        // Sort by first objective in descending order for correct hypervolume calculation
        points.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut hypervolume = 0.0;
        let mut prev_x = reference_point[0];

        for (x, y) in points {
            if x < reference_point[0] && y < reference_point[1] {
                hypervolume += (prev_x - x) * (reference_point[1] - y);
                prev_x = x;
            }
        }

        hypervolume
    }

    fn calculate_hypervolume_monte_carlo(
        pareto_front: &[MultiObjectiveSolution],
        reference_point: &Array1<f64>,
        n_samples: usize,
    ) -> f64 {
        use rand::prelude::*;
        let mut rng = rand::rng();
        let n_objectives = reference_point.len();

        // Find bounds for sampling
        let mut min_bounds = Array1::from_elem(n_objectives, f64::INFINITY);
        for sol in pareto_front {
            for (i, &obj) in sol.objectives.iter().enumerate() {
                min_bounds[i] = min_bounds[i].min(obj);
            }
        }

        let mut dominated_count = 0;
        for _ in 0..n_samples {
            // Generate random point
            let mut point = Array1::zeros(n_objectives);
            for i in 0..n_objectives {
                point[i] = rng.gen_range(min_bounds[i]..reference_point[i]);
            }

            // Check if point is dominated by any solution in Pareto front
            for sol in pareto_front {
                let mut dominates = true;
                for (i, &obj) in sol.objectives.iter().enumerate() {
                    if obj >= point[i] {
                        dominates = false;
                        break;
                    }
                }
                if dominates {
                    dominated_count += 1;
                    break;
                }
            }
        }

        // Calculate volume
        let total_volume: f64 = (0..n_objectives)
            .map(|i| reference_point[i] - min_bounds[i])
            .product();

        total_volume * (dominated_count as f64 / n_samples as f64)
    }

    /// Calculate diversity metric (spacing)
    pub fn calculate_spacing(pareto_front: &[MultiObjectiveSolution]) -> f64 {
        if pareto_front.len() < 2 {
            return 0.0;
        }

        let distances: Vec<f64> = pareto_front
            .iter()
            .map(|sol1| {
                pareto_front
                    .iter()
                    .filter(|sol2| sol1 as *const _ != *sol2 as *const _)
                    .map(|sol2| sol1.objective_distance(sol2))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance = distances
            .iter()
            .map(|d| (d - mean_distance).powi(2))
            .sum::<f64>()
            / distances.len() as f64;

        variance.sqrt()
    }

    /// Calculate convergence metric (average distance to true Pareto front)
    pub fn calculate_convergence(
        pareto_front: &[MultiObjectiveSolution],
        true_pareto_front: &[MultiObjectiveSolution],
    ) -> f64 {
        if pareto_front.is_empty() || true_pareto_front.is_empty() {
            return f64::INFINITY;
        }

        let total_distance: f64 = pareto_front
            .iter()
            .map(|sol1| {
                true_pareto_front
                    .iter()
                    .map(|sol2| sol1.objective_distance(sol2))
                    .fold(f64::INFINITY, f64::min)
            })
            .sum();

        total_distance / pareto_front.len() as f64
    }

    /// Generate random initial population
    pub fn generate_random_population(
        size: usize,
        n_variables: usize,
        bounds: &Option<(Array1<f64>, Array1<f64>)>,
    ) -> Vec<Array1<f64>> {
        use rand::prelude::*;
        let mut rng = rand::rng();
        let mut population = Vec::with_capacity(size);

        let (lower, upper) = match bounds {
            Some((l, u)) => (l.clone(), u.clone()),
            None => (Array1::zeros(n_variables), Array1::ones(n_variables)),
        };

        for _ in 0..size {
            let mut individual = Array1::zeros(n_variables);
            for j in 0..n_variables {
                individual[j] = rng.gen_range(lower[j]..upper[j]);
            }
            population.push(individual);
        }

        population
    }

    /// Apply bounds to a solution
    pub fn apply_bounds(individual: &mut Array1<f64>, bounds: &Option<(Array1<f64>, Array1<f64>)>) {
        if let Some((lower, upper)) = bounds {
            for (i, value) in individual.iter_mut().enumerate() {
                *value = value.max(lower[i]).min(upper[i]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_config_default() {
        let config = MultiObjectiveConfig::default();
        assert_eq!(config.population_size, 100);
        assert_eq!(config.max_generations, 250);
        assert_eq!(config.crossover_probability, 0.9);
    }

    #[test]
    fn test_das_dennis_points_2d() {
        let points = utils::generate_das_dennis_points(2, 3);
        assert!(!points.is_empty());

        // Check that points are normalized
        for point in &points {
            let sum: f64 = point.sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_hypervolume_2d() {
        let solutions = vec![
            MultiObjectiveSolution::new(array![1.0], array![1.0, 3.0]),
            MultiObjectiveSolution::new(array![2.0], array![2.0, 2.0]),
            MultiObjectiveSolution::new(array![3.0], array![3.0, 1.0]),
        ];

        let reference_point = array![4.0, 4.0];
        let hv = utils::calculate_hypervolume(&solutions, &reference_point);
        assert!(hv > 0.0);
    }

    #[test]
    fn test_spacing_calculation() {
        let solutions = vec![
            MultiObjectiveSolution::new(array![1.0], array![1.0, 3.0]),
            MultiObjectiveSolution::new(array![2.0], array![2.0, 2.0]),
            MultiObjectiveSolution::new(array![3.0], array![3.0, 1.0]),
        ];

        let spacing = utils::calculate_spacing(&solutions);
        assert!(spacing >= 0.0);
    }

    #[test]
    fn test_random_population_generation() {
        let bounds = Some((array![0.0, -1.0], array![1.0, 1.0]));
        let population = utils::generate_random_population(10, 2, &bounds);

        assert_eq!(population.len(), 10);
        assert_eq!(population[0].len(), 2);

        // Check bounds
        for individual in &population {
            assert!(individual[0] >= 0.0 && individual[0] <= 1.0);
            assert!(individual[1] >= -1.0 && individual[1] <= 1.0);
        }
    }

    #[test]
    fn test_apply_bounds() {
        let bounds = Some((array![0.0, -1.0], array![1.0, 1.0]));
        let mut individual = array![-0.5, 2.0];

        utils::apply_bounds(&mut individual, &bounds);

        assert_eq!(individual[0], 0.0); // Clamped to lower bound
        assert_eq!(individual[1], 1.0); // Clamped to upper bound
    }

    #[test]
    fn test_optimizer_factory() {
        let config = MultiObjectiveConfig::default();

        let nsga2 = OptimizerFactory::create_by_name("nsga2", config.clone(), 2, 3);
        assert!(nsga2.is_ok());

        let nsga3 = OptimizerFactory::create_by_name("nsga3", config.clone(), 2, 3);
        assert!(nsga3.is_ok());

        let unknown = OptimizerFactory::create_by_name("unknown", config, 2, 3);
        assert!(unknown.is_err());
    }
}
