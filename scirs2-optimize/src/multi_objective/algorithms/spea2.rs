//! SPEA2 (Strength Pareto Evolutionary Algorithm 2)
//!
//! An improved version of SPEA with better fitness assignment and archive truncation.

use super::{MultiObjectiveOptimizer, MultiObjectiveResult};
use crate::error::OptimizeError;
use crate::multi_objective::solutions::{Population, Solution};
use scirs2_core::ndarray::{Array1, ArrayView1};

/// SPEA2 optimizer
#[derive(Debug, Clone)]
pub struct SPEA2 {
    population_size: usize,
    archive_size: usize,
    n_objectives: usize,
    n_variables: usize,
    archive: Vec<Solution>,
    population: Population,
    generation: usize,
    evaluations: usize,
}

impl SPEA2 {
    /// Create new SPEA2 optimizer
    pub fn new(population_size: usize, n_objectives: usize, n_variables: usize) -> Self {
        Self {
            population_size,
            archive_size: population_size,
            n_objectives,
            n_variables,
            archive: Vec::new(),
            population: Population::with_capacity(population_size, n_objectives, n_variables),
            generation: 0,
            evaluations: 0,
        }
    }

    /// Calculate strength values
    fn calculate_strength(&self, population: &[Solution]) -> Vec<usize> {
        let mut strengths = vec![0; population.len()];

        for (i, sol_i) in population.iter().enumerate() {
            for (j, sol_j) in population.iter().enumerate() {
                if i != j && self.dominates(sol_i, sol_j) {
                    strengths[i] += 1;
                }
            }
        }

        strengths
    }

    /// Check if solution a dominates solution b
    fn dominates(&self, a: &Solution, b: &Solution) -> bool {
        let mut at_least_one_better = false;

        for i in 0..self.n_objectives {
            if a.objectives[i] > b.objectives[i] {
                return false;
            }
            if a.objectives[i] < b.objectives[i] {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Calculate k-th nearest neighbor distance
    fn kth_nearest_distance(&self, index: usize, population: &[Solution], k: usize) -> f64 {
        let mut distances = Vec::new();
        let current = &population[index];

        for (i, other) in population.iter().enumerate() {
            if i != index {
                let dist = self.euclidean_distance(
                    current.objectives.as_slice().unwrap(),
                    other.objectives.as_slice().unwrap(),
                );
                distances.push(dist);
            }
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distances
            .get(k.min(distances.len() - 1))
            .copied()
            .unwrap_or(0.0)
    }

    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl MultiObjectiveOptimizer for SPEA2 {
    fn optimize<F>(&mut self, objective_function: F) -> Result<MultiObjectiveResult, OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        // TODO: Implement SPEA2 algorithm
        Ok(MultiObjectiveResult {
            pareto_front: Vec::new(),
            population: Vec::new(),
            n_evaluations: 0,
            n_generations: 0,
            success: true,
            message: "SPEA2 not yet implemented".to_string(),
            hypervolume: Some(0.0),
            metrics: Default::default(),
        })
    }

    fn evolve_generation<F>(&mut self, _objective_function: &F) -> Result<(), OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        self.generation += 1;
        Ok(())
    }

    fn initialize_population(&mut self) -> Result<(), OptimizeError> {
        // TODO: Implement population initialization
        Ok(())
    }

    fn check_convergence(&self) -> bool {
        // TODO: Implement convergence criteria
        false
    }

    fn get_population(&self) -> &Population {
        &self.population
    }

    fn get_generation(&self) -> usize {
        self.generation
    }

    fn get_evaluations(&self) -> usize {
        self.evaluations
    }

    fn name(&self) -> &str {
        "SPEA2"
    }
}
