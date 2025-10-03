//! Multi-objective solution representations and utilities
//!
//! This module provides core data structures for multi-objective optimization:
//! - Solution representation with objectives and constraints
//! - Population management utilities
//! - Pareto front analysis
//! - Solution comparison and ranking

use scirs2_core::ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

/// Type alias for backward compatibility
pub type Solution = MultiObjectiveSolution;

/// Represents a solution in multi-objective optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveSolution {
    /// Decision variables
    pub variables: Array1<f64>,
    /// Objective function values
    pub objectives: Array1<f64>,
    /// Constraint violation (0.0 if feasible)
    pub constraint_violation: f64,
    /// Rank in the population (for NSGA-II/III)
    pub rank: usize,
    /// Crowding distance (for NSGA-II)
    pub crowding_distance: f64,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

impl MultiObjectiveSolution {
    /// Create a new solution
    pub fn new(variables: Array1<f64>, objectives: Array1<f64>) -> Self {
        Self {
            variables,
            objectives,
            constraint_violation: 0.0,
            rank: 0,
            crowding_distance: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Create a new solution with constraint violation
    pub fn new_with_constraints(
        variables: Array1<f64>,
        objectives: Array1<f64>,
        constraint_violation: f64,
    ) -> Self {
        Self {
            variables,
            objectives,
            constraint_violation,
            rank: 0,
            crowding_distance: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Check if this solution dominates another
    pub fn dominates(&self, other: &Self) -> bool {
        // First check constraint violations
        if self.constraint_violation < other.constraint_violation {
            return true;
        }
        if self.constraint_violation > other.constraint_violation {
            return false;
        }

        // If both are equally feasible/infeasible, compare objectives
        let mut at_least_one_better = false;

        for (obj1, obj2) in self.objectives.iter().zip(other.objectives.iter()) {
            if obj1 > obj2 {
                return false; // Assuming minimization
            }
            if obj1 < obj2 {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Check if this solution is dominated by another
    pub fn is_dominated_by(&self, other: &Self) -> bool {
        other.dominates(self)
    }

    /// Check if solutions are non-dominated with respect to each other
    pub fn is_non_dominated_with(&self, other: &Self) -> bool {
        !self.dominates(other) && !other.dominates(self)
    }

    /// Calculate distance to another solution in objective space
    pub fn objective_distance(&self, other: &Self) -> f64 {
        self.objectives
            .iter()
            .zip(other.objectives.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate distance to another solution in variable space
    pub fn variable_distance(&self, other: &Self) -> f64 {
        self.variables
            .iter()
            .zip(other.variables.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Get number of objectives
    pub fn n_objectives(&self) -> usize {
        self.objectives.len()
    }

    /// Get number of variables
    pub fn n_variables(&self) -> usize {
        self.variables.len()
    }

    /// Check if solution is feasible
    pub fn is_feasible(&self) -> bool {
        self.constraint_violation <= 1e-10
    }

    /// Set metadata value
    pub fn set_metadata(&mut self, key: String, value: f64) {
        self.metadata.insert(key, value);
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).copied()
    }

    /// Clone with new objectives (useful for constraint handling)
    pub fn with_objectives(&self, objectives: Array1<f64>) -> Self {
        Self {
            variables: self.variables.clone(),
            objectives,
            constraint_violation: self.constraint_violation,
            rank: self.rank,
            crowding_distance: self.crowding_distance,
            metadata: self.metadata.clone(),
        }
    }

    /// Clone with new variables
    pub fn with_variables(&self, variables: Array1<f64>) -> Self {
        Self {
            variables,
            objectives: self.objectives.clone(),
            constraint_violation: self.constraint_violation,
            rank: self.rank,
            crowding_distance: self.crowding_distance,
            metadata: self.metadata.clone(),
        }
    }
}

impl PartialEq for MultiObjectiveSolution {
    fn eq(&self, other: &Self) -> bool {
        self.variables == other.variables && self.objectives == other.objectives
    }
}

impl PartialOrd for MultiObjectiveSolution {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Compare by rank first, then by crowding distance
        match self.rank.cmp(&other.rank) {
            Ordering::Equal => {
                // Higher crowding distance is better (reversed comparison)
                other.crowding_distance.partial_cmp(&self.crowding_distance)
            }
            other_order => Some(other_order),
        }
    }
}

/// Result of multi-objective optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveResult {
    /// Pareto front solutions
    pub pareto_front: Vec<MultiObjectiveSolution>,
    /// All final population solutions
    pub population: Vec<MultiObjectiveSolution>,
    /// Number of function evaluations
    pub n_evaluations: usize,
    /// Number of generations/iterations
    pub n_generations: usize,
    /// Success flag
    pub success: bool,
    /// Convergence message
    pub message: String,
    /// Hypervolume indicator (if reference point provided)
    pub hypervolume: Option<f64>,
    /// Additional metrics
    pub metrics: OptimizationMetrics,
}

impl MultiObjectiveResult {
    /// Create a new result
    pub fn new(
        pareto_front: Vec<MultiObjectiveSolution>,
        population: Vec<MultiObjectiveSolution>,
        n_evaluations: usize,
        n_generations: usize,
    ) -> Self {
        Self {
            pareto_front,
            population,
            n_evaluations,
            n_generations,
            success: true,
            message: "Optimization completed successfully".to_string(),
            hypervolume: None,
            metrics: OptimizationMetrics::default(),
        }
    }

    /// Create a failed result
    pub fn failed(message: String, n_evaluations: usize, n_generations: usize) -> Self {
        Self {
            pareto_front: vec![],
            population: vec![],
            n_evaluations,
            n_generations,
            success: false,
            message,
            hypervolume: None,
            metrics: OptimizationMetrics::default(),
        }
    }

    /// Get best solution for a specific objective
    pub fn best_for_objective(&self, objective_index: usize) -> Option<&MultiObjectiveSolution> {
        self.pareto_front.iter().min_by(|a, b| {
            a.objectives[objective_index]
                .partial_cmp(&b.objectives[objective_index])
                .unwrap_or(Ordering::Equal)
        })
    }

    /// Get solution closest to ideal point
    pub fn closest_to_ideal(&self, ideal_point: &Array1<f64>) -> Option<&MultiObjectiveSolution> {
        self.pareto_front.iter().min_by(|a, b| {
            let dist_a = a
                .objectives
                .iter()
                .zip(ideal_point.iter())
                .map(|(obj, ideal)| (obj - ideal).powi(2))
                .sum::<f64>();
            let dist_b = b
                .objectives
                .iter()
                .zip(ideal_point.iter())
                .map(|(obj, ideal)| (obj - ideal).powi(2))
                .sum::<f64>();
            dist_a.partial_cmp(&dist_b).unwrap_or(Ordering::Equal)
        })
    }

    /// Get number of solutions in Pareto front
    pub fn pareto_front_size(&self) -> usize {
        self.pareto_front.len()
    }

    /// Check if result contains feasible solutions
    pub fn has_feasible_solutions(&self) -> bool {
        self.pareto_front.iter().any(|sol| sol.is_feasible())
    }

    /// Get all feasible solutions
    pub fn feasible_solutions(&self) -> Vec<&MultiObjectiveSolution> {
        self.pareto_front
            .iter()
            .filter(|sol| sol.is_feasible())
            .collect()
    }
}

/// Optimization metrics for tracking performance
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizationMetrics {
    /// Convergence history (hypervolume over generations)
    pub convergence_history: Vec<f64>,
    /// Diversity metrics over generations
    pub diversity_history: Vec<f64>,
    /// Average objective values over generations
    pub average_objectives: Vec<Array1<f64>>,
    /// Best objective values over generations
    pub best_objectives: Vec<Array1<f64>>,
    /// Population statistics
    pub population_stats: PopulationStatistics,
}

/// Population statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationStatistics {
    /// Mean objective values
    pub mean_objectives: Array1<f64>,
    /// Standard deviation of objectives
    pub std_objectives: Array1<f64>,
    /// Minimum objective values
    pub min_objectives: Array1<f64>,
    /// Maximum objective values
    pub max_objectives: Array1<f64>,
    /// Feasibility ratio
    pub feasibility_ratio: f64,
    /// Average constraint violation
    pub avg_constraint_violation: f64,
}

impl Default for PopulationStatistics {
    fn default() -> Self {
        Self {
            mean_objectives: Array1::zeros(0),
            std_objectives: Array1::zeros(0),
            min_objectives: Array1::zeros(0),
            max_objectives: Array1::zeros(0),
            feasibility_ratio: 0.0,
            avg_constraint_violation: 0.0,
        }
    }
}

/// Population management utilities
#[derive(Debug, Clone)]
pub struct Population {
    solutions: Vec<MultiObjectiveSolution>,
}

impl Population {
    /// Create a new empty population
    pub fn new() -> Self {
        Self { solutions: vec![] }
    }

    /// Create a new population with specified capacity
    pub fn with_capacity(
        population_size: usize,
        _n_objectives: usize,
        _n_variables: usize,
    ) -> Self {
        Self {
            solutions: Vec::with_capacity(population_size),
        }
    }

    /// Create population from solutions
    pub fn from_solutions(solutions: Vec<MultiObjectiveSolution>) -> Self {
        Self { solutions }
    }

    /// Add a solution to the population
    pub fn add(&mut self, solution: MultiObjectiveSolution) {
        self.solutions.push(solution);
    }

    /// Get all solutions
    pub fn solutions(&self) -> &[MultiObjectiveSolution] {
        &self.solutions
    }

    /// Get mutable reference to solutions
    pub fn solutions_mut(&mut self) -> &mut Vec<MultiObjectiveSolution> {
        &mut self.solutions
    }

    /// Get population size
    pub fn size(&self) -> usize {
        self.solutions.len()
    }

    /// Check if population is empty
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Clear the population
    pub fn clear(&mut self) {
        self.solutions.clear();
    }

    /// Extract Pareto front from population
    pub fn extract_pareto_front(&self) -> Vec<MultiObjectiveSolution> {
        let mut pareto_front: Vec<MultiObjectiveSolution> = Vec::new();

        for candidate in &self.solutions {
            let mut is_dominated = false;

            // Check if candidate is dominated by any existing solution in Pareto front
            for existing in &pareto_front {
                if existing.dominates(candidate) {
                    is_dominated = true;
                    break;
                }
            }

            if !is_dominated {
                // Remove solutions from Pareto front that are dominated by candidate
                pareto_front.retain(|existing| !candidate.dominates(existing));
                pareto_front.push(candidate.clone());
            }
        }

        pareto_front
    }

    /// Perform non-dominated sorting (for NSGA-II)
    pub fn non_dominated_sort(&mut self) -> Vec<Vec<usize>> {
        let n = self.solutions.len();
        let mut fronts = Vec::new();
        let mut domination_counts = vec![0; n];
        let mut dominated_solutions: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Calculate domination relationships
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    if self.solutions[i].dominates(&self.solutions[j]) {
                        dominated_solutions[i].push(j);
                    } else if self.solutions[j].dominates(&self.solutions[i]) {
                        domination_counts[i] += 1;
                    }
                }
            }
        }

        // Find first front
        let mut current_front = Vec::new();
        for i in 0..n {
            if domination_counts[i] == 0 {
                self.solutions[i].rank = 0;
                current_front.push(i);
            }
        }

        let mut front_number = 0;

        while !current_front.is_empty() {
            fronts.push(current_front.clone());
            let mut next_front = Vec::new();

            for &i in &current_front {
                for &j in &dominated_solutions[i] {
                    domination_counts[j] -= 1;
                    if domination_counts[j] == 0 {
                        self.solutions[j].rank = front_number + 1;
                        next_front.push(j);
                    }
                }
            }

            current_front = next_front;
            front_number += 1;
        }

        fronts
    }

    /// Calculate crowding distances for solutions in the same front
    pub fn calculate_crowding_distances(&mut self, front_indices: &[usize]) {
        let front_size = front_indices.len();
        if front_size <= 2 {
            // Set infinite distance for boundary solutions
            for &i in front_indices {
                self.solutions[i].crowding_distance = f64::INFINITY;
            }
            return;
        }

        let n_objectives = self.solutions[front_indices[0]].n_objectives();

        // Initialize distances to zero
        for &i in front_indices {
            self.solutions[i].crowding_distance = 0.0;
        }

        // Calculate distance for each objective
        for obj in 0..n_objectives {
            // Sort by current objective
            let mut sorted_indices = front_indices.to_vec();
            sorted_indices.sort_by(|&a, &b| {
                self.solutions[a].objectives[obj]
                    .partial_cmp(&self.solutions[b].objectives[obj])
                    .unwrap_or(Ordering::Equal)
            });

            // Set boundary solutions to infinite distance
            self.solutions[sorted_indices[0]].crowding_distance = f64::INFINITY;
            self.solutions[sorted_indices[front_size - 1]].crowding_distance = f64::INFINITY;

            // Calculate objective range
            let obj_min = self.solutions[sorted_indices[0]].objectives[obj];
            let obj_max = self.solutions[sorted_indices[front_size - 1]].objectives[obj];
            let obj_range = obj_max - obj_min;

            if obj_range > 0.0 {
                // Calculate distances for intermediate solutions
                for i in 1..front_size - 1 {
                    if self.solutions[sorted_indices[i]].crowding_distance != f64::INFINITY {
                        let distance = (self.solutions[sorted_indices[i + 1]].objectives[obj]
                            - self.solutions[sorted_indices[i - 1]].objectives[obj])
                            / obj_range;
                        self.solutions[sorted_indices[i]].crowding_distance += distance;
                    }
                }
            }
        }
    }

    /// Select best solutions using NSGA-II crowded comparison
    pub fn select_best(&mut self, target_size: usize) -> Vec<MultiObjectiveSolution> {
        if self.solutions.len() <= target_size {
            return self.solutions.clone();
        }

        // Perform non-dominated sorting
        let fronts = self.non_dominated_sort();

        // Calculate crowding distances for each front
        for front in &fronts {
            self.calculate_crowding_distances(front);
        }

        let mut selected = Vec::new();

        // Add complete fronts until we reach capacity
        for front in &fronts {
            if selected.len() + front.len() <= target_size {
                for &i in front {
                    selected.push(self.solutions[i].clone());
                }
            } else {
                // Add partial front based on crowding distance
                let remaining = target_size - selected.len();
                let mut front_solutions: Vec<_> =
                    front.iter().map(|&i| &self.solutions[i]).collect();

                // Sort by crowding distance (descending)
                front_solutions.sort_by(|a, b| {
                    b.crowding_distance
                        .partial_cmp(&a.crowding_distance)
                        .unwrap_or(Ordering::Equal)
                });

                for solution in front_solutions.iter().take(remaining) {
                    selected.push((*solution).clone());
                }
                break;
            }
        }

        selected
    }

    /// Calculate population statistics
    pub fn calculate_statistics(&self) -> PopulationStatistics {
        if self.solutions.is_empty() {
            return PopulationStatistics::default();
        }

        let n_objectives = self.solutions[0].n_objectives();
        let n_solutions = self.solutions.len();

        let mut mean_objectives = Array1::zeros(n_objectives);
        let mut min_objectives = Array1::from_elem(n_objectives, f64::INFINITY);
        let mut max_objectives = Array1::from_elem(n_objectives, f64::NEG_INFINITY);

        let mut feasible_count = 0;
        let mut total_violation = 0.0;

        // Calculate means, mins, maxs
        for solution in &self.solutions {
            for (i, &obj) in solution.objectives.iter().enumerate() {
                mean_objectives[i] += obj;
                min_objectives[i] = min_objectives[i].min(obj);
                max_objectives[i] = max_objectives[i].max(obj);
            }

            if solution.is_feasible() {
                feasible_count += 1;
            }
            total_violation += solution.constraint_violation;
        }

        mean_objectives /= n_solutions as f64;

        // Calculate standard deviations
        let mut std_objectives = Array1::zeros(n_objectives);
        for solution in &self.solutions {
            for (i, &obj) in solution.objectives.iter().enumerate() {
                let diff = obj - mean_objectives[i];
                std_objectives[i] += diff * diff;
            }
        }
        std_objectives = std_objectives.mapv(|x: f64| (x / n_solutions as f64).sqrt());

        PopulationStatistics {
            mean_objectives,
            std_objectives,
            min_objectives,
            max_objectives,
            feasibility_ratio: feasible_count as f64 / n_solutions as f64,
            avg_constraint_violation: total_violation / n_solutions as f64,
        }
    }
}

impl Default for Population {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_solution_creation() {
        let variables = array![1.0, 2.0, 3.0];
        let objectives = array![4.0, 5.0];
        let solution = MultiObjectiveSolution::new(variables.clone(), objectives.clone());

        assert_eq!(solution.variables, variables);
        assert_eq!(solution.objectives, objectives);
        assert_eq!(solution.constraint_violation, 0.0);
        assert!(solution.is_feasible());
    }

    #[test]
    fn test_domination() {
        let sol1 = MultiObjectiveSolution::new(array![1.0], array![1.0, 2.0]);
        let sol2 = MultiObjectiveSolution::new(array![2.0], array![2.0, 3.0]);
        let sol3 = MultiObjectiveSolution::new(array![3.0], array![0.5, 3.5]);

        assert!(sol1.dominates(&sol2)); // sol1 is better in both objectives
        assert!(!sol2.dominates(&sol1));
        assert!(sol1.is_non_dominated_with(&sol3)); // Neither dominates the other
    }

    #[test]
    fn test_constraint_domination() {
        let feasible =
            MultiObjectiveSolution::new_with_constraints(array![1.0], array![2.0, 3.0], 0.0);
        let infeasible =
            MultiObjectiveSolution::new_with_constraints(array![2.0], array![1.0, 2.0], 1.0);

        assert!(feasible.dominates(&infeasible)); // Feasible dominates infeasible
        assert!(!infeasible.dominates(&feasible));
    }

    #[test]
    fn test_population_pareto_front() {
        let mut population = Population::new();
        population.add(MultiObjectiveSolution::new(array![1.0], array![1.0, 3.0]));
        population.add(MultiObjectiveSolution::new(array![2.0], array![2.0, 2.0]));
        population.add(MultiObjectiveSolution::new(array![3.0], array![3.0, 1.0]));
        population.add(MultiObjectiveSolution::new(array![4.0], array![2.5, 2.5])); // Dominated

        let pareto_front = population.extract_pareto_front();
        assert_eq!(pareto_front.len(), 3); // First three solutions are non-dominated
    }

    #[test]
    fn test_non_dominated_sorting() {
        let mut population = Population::new();
        population.add(MultiObjectiveSolution::new(array![1.0], array![1.0, 3.0]));
        population.add(MultiObjectiveSolution::new(array![2.0], array![3.0, 1.0]));
        population.add(MultiObjectiveSolution::new(array![3.0], array![2.0, 2.0]));
        population.add(MultiObjectiveSolution::new(array![4.0], array![4.0, 4.0])); // Dominated

        let fronts = population.non_dominated_sort();

        assert_eq!(fronts.len(), 2); // Two fronts
        assert_eq!(fronts[0].len(), 3); // First front has 3 solutions
        assert_eq!(fronts[1].len(), 1); // Second front has 1 solution

        // Check ranks
        for &i in &fronts[0] {
            assert_eq!(population.solutions[i].rank, 0);
        }
        for &i in &fronts[1] {
            assert_eq!(population.solutions[i].rank, 1);
        }
    }

    #[test]
    fn test_crowding_distance() {
        let mut population = Population::new();
        population.add(MultiObjectiveSolution::new(array![1.0], array![1.0, 3.0]));
        population.add(MultiObjectiveSolution::new(array![2.0], array![2.0, 2.0]));
        population.add(MultiObjectiveSolution::new(array![3.0], array![3.0, 1.0]));

        let front = vec![0, 1, 2];
        population.calculate_crowding_distances(&front);

        // Boundary solutions should have infinite distance
        assert_eq!(population.solutions[0].crowding_distance, f64::INFINITY);
        assert_eq!(population.solutions[2].crowding_distance, f64::INFINITY);
        // Middle solution should have finite distance
        assert!(population.solutions[1].crowding_distance.is_finite());
        assert!(population.solutions[1].crowding_distance > 0.0);
    }

    #[test]
    fn test_population_statistics() {
        let mut population = Population::new();
        population.add(MultiObjectiveSolution::new(array![1.0], array![1.0, 3.0]));
        population.add(MultiObjectiveSolution::new(array![2.0], array![2.0, 2.0]));
        population.add(MultiObjectiveSolution::new(array![3.0], array![3.0, 1.0]));

        let stats = population.calculate_statistics();

        assert_eq!(stats.mean_objectives[0], 2.0); // (1+2+3)/3
        assert_eq!(stats.mean_objectives[1], 2.0); // (3+2+1)/3
        assert_eq!(stats.min_objectives[0], 1.0);
        assert_eq!(stats.max_objectives[0], 3.0);
        assert_eq!(stats.feasibility_ratio, 1.0); // All feasible
    }
}
