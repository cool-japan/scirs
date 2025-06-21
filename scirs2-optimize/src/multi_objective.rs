//! Multi-objective optimization algorithms
//!
//! This module provides algorithms for solving multi-objective optimization problems
//! where multiple conflicting objectives need to be optimized simultaneously.

use crate::error::OptimizeError;
use ndarray::{Array1, Array2, ArrayView1};
use rand::{prelude::*, rng};
use std::cmp::Ordering;
use std::collections::HashMap;

/// Represents a solution in multi-objective optimization
#[derive(Debug, Clone)]
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

/// Result of multi-objective optimization
#[derive(Debug, Clone)]
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
}

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
        }
    }
}

/// NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation
pub struct NSGAII {
    config: MultiObjectiveConfig,
    n_objectives: usize,
    n_variables: usize,
    population: Vec<MultiObjectiveSolution>,
    generation: usize,
    n_evaluations: usize,
}

/// NSGA-III (Non-dominated Sorting Genetic Algorithm III) implementation
/// Designed for many-objective optimization (4+ objectives)
pub struct NSGAIII {
    config: MultiObjectiveConfig,
    n_objectives: usize,
    n_variables: usize,
    population: Vec<MultiObjectiveSolution>,
    reference_points: Array2<f64>,
    generation: usize,
    n_evaluations: usize,
}

impl NSGAII {
    /// Create new NSGA-II optimizer
    pub fn new(
        n_variables: usize,
        n_objectives: usize,
        config: Option<MultiObjectiveConfig>,
    ) -> Self {
        let config = config.unwrap_or_default();

        Self {
            config,
            n_objectives,
            n_variables,
            population: Vec::new(),
            generation: 0,
            n_evaluations: 0,
        }
    }

    /// Set variable bounds
    pub fn with_bounds(
        mut self,
        lower: Array1<f64>,
        upper: Array1<f64>,
    ) -> Result<Self, OptimizeError> {
        if lower.len() != self.n_variables || upper.len() != self.n_variables {
            return Err(OptimizeError::ValueError(
                "Bounds dimensions must match number of variables".to_string(),
            ));
        }

        for (&l, &u) in lower.iter().zip(upper.iter()) {
            if l >= u {
                return Err(OptimizeError::ValueError(
                    "Lower bounds must be less than upper bounds".to_string(),
                ));
            }
        }

        self.config.bounds = Some((lower, upper));
        Ok(self)
    }

    /// Optimize multiple objectives
    pub fn optimize<F, C>(
        &mut self,
        mut objective_fn: F,
        mut constraint_fn: Option<C>,
    ) -> Result<MultiObjectiveResult, OptimizeError>
    where
        F: FnMut(&ArrayView1<f64>) -> Array1<f64>,
        C: FnMut(&ArrayView1<f64>) -> f64,
    {
        // Initialize population
        self.initialize_population(&mut objective_fn, constraint_fn.as_mut())?;

        let mut prev_hypervolume = 0.0;
        let mut stagnation_count = 0;

        for generation in 0..self.config.max_generations {
            self.generation = generation;

            // Check termination criteria
            if let Some(max_evals) = self.config.max_evaluations {
                if self.n_evaluations >= max_evals {
                    break;
                }
            }

            // Create offspring through selection, crossover, and mutation
            let offspring = self.create_offspring(&mut objective_fn, constraint_fn.as_mut())?;

            // Combine parent and offspring populations
            let mut combined_population = self.population.clone();
            combined_population.extend(offspring);

            // Environmental selection (non-dominated sorting + crowding distance)
            self.population = self.environmental_selection(combined_population);

            // Check convergence
            if let Some(ref reference_point) = self.config.reference_point {
                let current_hypervolume = self.calculate_hypervolume(reference_point)?;

                if (current_hypervolume - prev_hypervolume).abs() < self.config.tolerance {
                    stagnation_count += 1;
                    if stagnation_count >= 10 {
                        break;
                    }
                } else {
                    stagnation_count = 0;
                }

                prev_hypervolume = current_hypervolume;
            }
        }

        // Extract Pareto front (rank 0 solutions)
        let pareto_front: Vec<MultiObjectiveSolution> = self
            .population
            .iter()
            .filter(|sol| sol.rank == 0)
            .cloned()
            .collect();

        let hypervolume = if let Some(ref reference_point) = self.config.reference_point {
            Some(self.calculate_hypervolume(reference_point)?)
        } else {
            None
        };

        Ok(MultiObjectiveResult {
            pareto_front,
            population: self.population.clone(),
            n_evaluations: self.n_evaluations,
            n_generations: self.generation + 1,
            success: true,
            message: "NSGA-II optimization completed successfully".to_string(),
            hypervolume,
        })
    }

    /// Initialize random population
    fn initialize_population<F, C>(
        &mut self,
        objective_fn: &mut F,
        mut constraint_fn: Option<&mut C>,
    ) -> Result<(), OptimizeError>
    where
        F: FnMut(&ArrayView1<f64>) -> Array1<f64>,
        C: FnMut(&ArrayView1<f64>) -> f64,
    {
        self.population.clear();
        let mut rng = rng();

        for _ in 0..self.config.population_size {
            let variables = if let Some((ref lower, ref upper)) = self.config.bounds {
                Array1::from_shape_fn(self.n_variables, |i| {
                    lower[i] + rng.random::<f64>() * (upper[i] - lower[i])
                })
            } else {
                Array1::from_shape_fn(self.n_variables, |_| rng.random::<f64>() * 2.0 - 1.0)
            };

            let objectives = objective_fn(&variables.view());
            self.n_evaluations += 1;

            let constraint_violation = if let Some(ref mut constraint_fn) = constraint_fn {
                constraint_fn(&variables.view()).max(0.0)
            } else {
                0.0
            };

            let solution = MultiObjectiveSolution {
                variables,
                objectives,
                constraint_violation,
                rank: 0,
                crowding_distance: 0.0,
                metadata: HashMap::new(),
            };

            self.population.push(solution);
        }

        // Perform initial ranking
        self.population = self.environmental_selection(self.population.clone());

        Ok(())
    }

    /// Create offspring through selection, crossover, and mutation
    fn create_offspring<F, C>(
        &mut self,
        objective_fn: &mut F,
        mut constraint_fn: Option<&mut C>,
    ) -> Result<Vec<MultiObjectiveSolution>, OptimizeError>
    where
        F: FnMut(&ArrayView1<f64>) -> Array1<f64>,
        C: FnMut(&ArrayView1<f64>) -> f64,
    {
        let mut offspring = Vec::new();
        let mut rng = rng();

        while offspring.len() < self.config.population_size {
            // Tournament selection
            let parent1 = self.tournament_selection(&mut rng);
            let parent2 = self.tournament_selection(&mut rng);

            // Crossover
            let (mut child1_vars, mut child2_vars) = if rng.random::<f64>()
                < self.config.crossover_probability
            {
                self.simulated_binary_crossover(&parent1.variables, &parent2.variables, &mut rng)
            } else {
                (parent1.variables.clone(), parent2.variables.clone())
            };

            // Mutation
            if rng.random::<f64>() < self.config.mutation_probability {
                self.polynomial_mutation(&mut child1_vars, &mut rng);
            }
            if rng.random::<f64>() < self.config.mutation_probability {
                self.polynomial_mutation(&mut child2_vars, &mut rng);
            }

            // Apply bounds
            if let Some((ref lower, ref upper)) = self.config.bounds {
                self.apply_bounds(&mut child1_vars, lower, upper);
                self.apply_bounds(&mut child2_vars, lower, upper);
            }

            // Evaluate offspring
            for child_vars in [child1_vars, child2_vars] {
                if offspring.len() >= self.config.population_size {
                    break;
                }

                let objectives = objective_fn(&child_vars.view());
                self.n_evaluations += 1;

                let constraint_violation = if let Some(ref mut constraint_fn) = constraint_fn {
                    constraint_fn(&child_vars.view()).max(0.0)
                } else {
                    0.0
                };

                let solution = MultiObjectiveSolution {
                    variables: child_vars,
                    objectives,
                    constraint_violation,
                    rank: 0,
                    crowding_distance: 0.0,
                    metadata: HashMap::new(),
                };

                offspring.push(solution);
            }
        }

        Ok(offspring)
    }

    /// Tournament selection
    fn tournament_selection(&self, rng: &mut impl Rng) -> &MultiObjectiveSolution {
        let tournament_size = 2;
        let mut best_idx = rng.random_range(0..self.population.len());

        for _ in 1..tournament_size {
            let idx = rng.random_range(0..self.population.len());
            if self.dominates_or_better(&self.population[idx], &self.population[best_idx]) {
                best_idx = idx;
            }
        }

        &self.population[best_idx]
    }

    /// Check if solution a dominates or is better than solution b
    fn dominates_or_better(&self, a: &MultiObjectiveSolution, b: &MultiObjectiveSolution) -> bool {
        // First check constraint dominance
        if a.constraint_violation < b.constraint_violation {
            return true;
        } else if a.constraint_violation > b.constraint_violation {
            return false;
        }

        // If both are feasible or both infeasible with same violation, check rank and crowding
        match a.rank.cmp(&b.rank) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Equal => a.crowding_distance > b.crowding_distance,
            std::cmp::Ordering::Greater => false,
        }
    }

    /// Simulated Binary Crossover (SBX)
    fn simulated_binary_crossover(
        &self,
        parent1: &Array1<f64>,
        parent2: &Array1<f64>,
        rng: &mut impl Rng,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();
        let eta = self.config.crossover_eta;

        for i in 0..self.n_variables {
            if rng.random::<f64>() <= 0.5 {
                let y1 = parent1[i].min(parent2[i]);
                let y2 = parent1[i].max(parent2[i]);

                if (y2 - y1).abs() > 1e-14 {
                    let u = rng.random::<f64>();
                    let beta = if u <= 0.5 {
                        (2.0 * u).powf(1.0 / (eta + 1.0))
                    } else {
                        (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
                    };

                    child1[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1));
                    child2[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1));
                }
            }
        }

        (child1, child2)
    }

    /// Polynomial mutation
    fn polynomial_mutation(&self, individual: &mut Array1<f64>, rng: &mut impl Rng) {
        let eta = self.config.mutation_eta;

        for i in 0..self.n_variables {
            if rng.random::<f64>() <= 1.0 / self.n_variables as f64 {
                let u = rng.random::<f64>();
                let delta = if u < 0.5 {
                    (2.0 * u).powf(1.0 / (eta + 1.0)) - 1.0
                } else {
                    1.0 - (2.0 * (1.0 - u)).powf(1.0 / (eta + 1.0))
                };

                if let Some((ref lower, ref upper)) = self.config.bounds {
                    let range = upper[i] - lower[i];
                    individual[i] += delta * range * 0.1; // Scale mutation
                } else {
                    individual[i] += delta * 0.1;
                }
            }
        }
    }

    /// Apply variable bounds
    fn apply_bounds(&self, individual: &mut Array1<f64>, lower: &Array1<f64>, upper: &Array1<f64>) {
        for (i, value) in individual.iter_mut().enumerate() {
            *value = value.max(lower[i]).min(upper[i]);
        }
    }

    /// Environmental selection using non-dominated sorting and crowding distance
    fn environmental_selection(
        &self,
        mut population: Vec<MultiObjectiveSolution>,
    ) -> Vec<MultiObjectiveSolution> {
        // Non-dominated sorting
        let fronts = self.non_dominated_sorting(&mut population);

        let mut new_population = Vec::new();
        let mut front_idx = 0;

        // Add fronts until population limit is reached
        while front_idx < fronts.len()
            && new_population.len() + fronts[front_idx].len() <= self.config.population_size
        {
            new_population.extend(fronts[front_idx].clone());
            front_idx += 1;
        }

        // If there's a partial front to add
        if front_idx < fronts.len() && new_population.len() < self.config.population_size {
            let mut last_front = fronts[front_idx].clone();
            self.calculate_crowding_distance(&mut last_front);

            // Sort by crowding distance (descending)
            last_front.sort_by(|a, b| {
                b.crowding_distance
                    .partial_cmp(&a.crowding_distance)
                    .unwrap_or(Ordering::Equal)
            });

            let remaining = self.config.population_size - new_population.len();
            new_population.extend(last_front.into_iter().take(remaining));
        }

        new_population
    }

    /// Non-dominated sorting
    fn non_dominated_sorting(
        &self,
        population: &mut [MultiObjectiveSolution],
    ) -> Vec<Vec<MultiObjectiveSolution>> {
        let n = population.len();
        let mut domination_count = vec![0; n];
        let mut dominated_solutions = vec![Vec::new(); n];
        let mut fronts = Vec::new();
        let mut current_front = Vec::new();

        // Calculate domination relationships
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    match self.compare_dominance(&population[i], &population[j]) {
                        Ordering::Less => {
                            dominated_solutions[i].push(j);
                        }
                        Ordering::Greater => {
                            domination_count[i] += 1;
                        }
                        Ordering::Equal => {}
                    }
                }
            }

            if domination_count[i] == 0 {
                population[i].rank = 0;
                current_front.push(population[i].clone());
            }
        }

        let mut rank = 0;
        while !current_front.is_empty() {
            fronts.push(current_front.clone());
            let mut next_front = Vec::new();

            for i in 0..n {
                if population[i].rank == rank {
                    for &j in &dominated_solutions[i] {
                        domination_count[j] -= 1;
                        if domination_count[j] == 0 {
                            population[j].rank = rank + 1;
                            next_front.push(population[j].clone());
                        }
                    }
                }
            }

            current_front = next_front;
            rank += 1;
        }

        fronts
    }

    /// Compare two solutions for dominance
    fn compare_dominance(
        &self,
        a: &MultiObjectiveSolution,
        b: &MultiObjectiveSolution,
    ) -> Ordering {
        // Handle constraint dominance
        if a.constraint_violation < b.constraint_violation {
            return Ordering::Less;
        } else if a.constraint_violation > b.constraint_violation {
            return Ordering::Greater;
        }

        // Both solutions have same constraint status, check objective dominance
        let mut a_better = false;
        let mut b_better = false;

        for i in 0..self.n_objectives {
            if a.objectives[i] < b.objectives[i] {
                a_better = true;
            } else if a.objectives[i] > b.objectives[i] {
                b_better = true;
            }
        }

        if a_better && !b_better {
            Ordering::Less
        } else if b_better && !a_better {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    /// Calculate crowding distance for a front
    fn calculate_crowding_distance(&self, front: &mut [MultiObjectiveSolution]) {
        let n = front.len();

        // Initialize crowding distances
        for solution in front.iter_mut() {
            solution.crowding_distance = 0.0;
        }

        if n <= 2 {
            for solution in front.iter_mut() {
                solution.crowding_distance = f64::INFINITY;
            }
            return;
        }

        // Calculate crowding distance for each objective
        for obj in 0..self.n_objectives {
            // Sort by objective value
            front.sort_by(|a, b| {
                a.objectives[obj]
                    .partial_cmp(&b.objectives[obj])
                    .unwrap_or(Ordering::Equal)
            });

            // Set boundary points to infinity
            front[0].crowding_distance = f64::INFINITY;
            front[n - 1].crowding_distance = f64::INFINITY;

            let obj_range = front[n - 1].objectives[obj] - front[0].objectives[obj];

            if obj_range > 0.0 {
                for i in 1..n - 1 {
                    front[i].crowding_distance +=
                        (front[i + 1].objectives[obj] - front[i - 1].objectives[obj]) / obj_range;
                }
            }
        }
    }

    /// Calculate hypervolume indicator
    fn calculate_hypervolume(&self, reference_point: &Array1<f64>) -> Result<f64, OptimizeError> {
        if reference_point.len() != self.n_objectives {
            return Err(OptimizeError::ValueError(
                "Reference point dimension must match number of objectives".to_string(),
            ));
        }

        let pareto_front: Vec<&MultiObjectiveSolution> = self
            .population
            .iter()
            .filter(|sol| sol.rank == 0 && sol.constraint_violation == 0.0)
            .collect();

        if pareto_front.is_empty() {
            return Ok(0.0);
        }

        // For 2D problems, use simple geometric calculation
        if self.n_objectives == 2 {
            let mut points: Vec<_> = pareto_front
                .iter()
                .map(|sol| (sol.objectives[0], sol.objectives[1]))
                .collect();

            // Sort by first objective in descending order for hypervolume calculation
            points.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

            let mut volume = 0.0;
            let mut prev_y = reference_point[1];

            for &(x, y) in &points {
                if x < reference_point[0] && y < reference_point[1] && y < prev_y {
                    volume += (reference_point[0] - x) * (prev_y - y);
                    prev_y = y;
                }
            }

            Ok(volume)
        } else {
            // For higher dimensions, use Monte Carlo approximation
            self.hypervolume_monte_carlo(reference_point, &pareto_front)
        }
    }

    /// Monte Carlo hypervolume calculation for higher dimensions
    fn hypervolume_monte_carlo(
        &self,
        reference_point: &Array1<f64>,
        pareto_front: &[&MultiObjectiveSolution],
    ) -> Result<f64, OptimizeError> {
        let n_samples = 10000;
        let mut rng = rng();
        let mut dominated_count = 0;

        // Find bounds for sampling
        let mut min_bounds = reference_point.clone();
        for sol in pareto_front {
            for i in 0..self.n_objectives {
                min_bounds[i] = min_bounds[i].min(sol.objectives[i]);
            }
        }

        for _ in 0..n_samples {
            // Generate random point
            let mut point = Array1::zeros(self.n_objectives);
            for i in 0..self.n_objectives {
                point[i] =
                    min_bounds[i] + rng.random::<f64>() * (reference_point[i] - min_bounds[i]);
            }

            // Check if point is dominated by any solution in Pareto front
            let mut is_dominated = false;
            for sol in pareto_front {
                let mut dominates = true;
                for i in 0..self.n_objectives {
                    if sol.objectives[i] >= point[i] {
                        dominates = false;
                        break;
                    }
                }
                if dominates {
                    is_dominated = true;
                    break;
                }
            }

            if is_dominated {
                dominated_count += 1;
            }
        }

        // Calculate volume
        let mut total_volume = 1.0;
        for i in 0..self.n_objectives {
            total_volume *= reference_point[i] - min_bounds[i];
        }

        Ok(total_volume * dominated_count as f64 / n_samples as f64)
    }
}

impl NSGAIII {
    /// Create new NSGA-III optimizer
    pub fn new(
        n_variables: usize,
        n_objectives: usize,
        config: Option<MultiObjectiveConfig>,
    ) -> Self {
        let config = config.unwrap_or_default();
        let reference_points = Self::generate_reference_points(n_objectives, 12); // Default 12 divisions

        Self {
            config,
            n_objectives,
            n_variables,
            population: Vec::new(),
            reference_points,
            generation: 0,
            n_evaluations: 0,
        }
    }

    /// Create NSGA-III with custom reference points
    pub fn with_reference_points(
        n_variables: usize,
        n_objectives: usize,
        reference_points: Array2<f64>,
        config: Option<MultiObjectiveConfig>,
    ) -> Result<Self, OptimizeError> {
        if reference_points.ncols() != n_objectives {
            return Err(OptimizeError::ValueError(
                "Reference points must have same number of columns as objectives".to_string(),
            ));
        }

        let config = config.unwrap_or_default();

        Ok(Self {
            config,
            n_objectives,
            n_variables,
            population: Vec::new(),
            reference_points,
            generation: 0,
            n_evaluations: 0,
        })
    }

    /// Set variable bounds
    pub fn with_bounds(
        mut self,
        lower: Array1<f64>,
        upper: Array1<f64>,
    ) -> Result<Self, OptimizeError> {
        if lower.len() != self.n_variables || upper.len() != self.n_variables {
            return Err(OptimizeError::ValueError(
                "Bounds dimensions must match number of variables".to_string(),
            ));
        }

        for (&l, &u) in lower.iter().zip(upper.iter()) {
            if l >= u {
                return Err(OptimizeError::ValueError(
                    "Lower bounds must be less than upper bounds".to_string(),
                ));
            }
        }

        self.config.bounds = Some((lower, upper));
        Ok(self)
    }

    /// Generate structured reference points using Das and Dennis method
    pub fn generate_reference_points(n_objectives: usize, n_divisions: usize) -> Array2<f64> {
        let n_points = Self::binomial_coefficient(n_objectives + n_divisions - 1, n_divisions);
        let mut points = Array2::zeros((n_points, n_objectives));
        let mut point_idx = 0;

        Self::generate_points_recursive(
            &mut points,
            &mut point_idx,
            n_objectives,
            n_divisions,
            0,
            Array1::zeros(n_objectives),
            n_divisions,
        );

        // Normalize points (they should sum to 1)
        for mut row in points.outer_iter_mut() {
            let sum: f64 = row.sum();
            if sum > 0.0 {
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
        }

        points
    }

    /// Recursive helper for generating reference points
    fn generate_points_recursive(
        points: &mut Array2<f64>,
        point_idx: &mut usize,
        n_objectives: usize,
        _n_divisions: usize,
        current_objective: usize,
        mut current_point: Array1<f64>,
        remaining_sum: usize,
    ) {
        if current_objective == n_objectives - 1 {
            current_point[current_objective] = remaining_sum as f64;
            if *point_idx < points.nrows() {
                points.row_mut(*point_idx).assign(&current_point);
                *point_idx += 1;
            }
            return;
        }

        for i in 0..=remaining_sum {
            current_point[current_objective] = i as f64;
            Self::generate_points_recursive(
                points,
                point_idx,
                n_objectives,
                _n_divisions,
                current_objective + 1,
                current_point.clone(),
                remaining_sum - i,
            );
        }
    }

    /// Calculate binomial coefficient
    fn binomial_coefficient(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let k = k.min(n - k); // Take advantage of symmetry
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// Optimize multiple objectives using NSGA-III
    pub fn optimize<F, C>(
        &mut self,
        mut objective_fn: F,
        mut constraint_fn: Option<C>,
    ) -> Result<MultiObjectiveResult, OptimizeError>
    where
        F: FnMut(&ArrayView1<f64>) -> Array1<f64>,
        C: FnMut(&ArrayView1<f64>) -> f64,
    {
        // Initialize population
        self.initialize_population(&mut objective_fn, constraint_fn.as_mut())?;

        for generation in 0..self.config.max_generations {
            self.generation = generation;

            // Check termination criteria
            if let Some(max_evals) = self.config.max_evaluations {
                if self.n_evaluations >= max_evals {
                    break;
                }
            }

            // Create offspring
            let offspring = self.create_offspring(&mut objective_fn, constraint_fn.as_mut())?;

            // Combine parent and offspring populations
            let mut combined_population = self.population.clone();
            combined_population.extend(offspring);

            // Environmental selection using reference point association
            self.population = self.environmental_selection_nsga3(combined_population);
        }

        // Extract Pareto front (rank 0 solutions)
        let pareto_front: Vec<MultiObjectiveSolution> = self
            .population
            .iter()
            .filter(|sol| sol.rank == 0)
            .cloned()
            .collect();

        let hypervolume = if let Some(ref reference_point) = self.config.reference_point {
            Some(self.calculate_hypervolume(reference_point)?)
        } else {
            None
        };

        Ok(MultiObjectiveResult {
            pareto_front,
            population: self.population.clone(),
            n_evaluations: self.n_evaluations,
            n_generations: self.generation + 1,
            success: true,
            message: "NSGA-III optimization completed successfully".to_string(),
            hypervolume,
        })
    }

    /// Initialize random population (similar to NSGA-II but reused here)
    fn initialize_population<F, C>(
        &mut self,
        objective_fn: &mut F,
        mut constraint_fn: Option<&mut C>,
    ) -> Result<(), OptimizeError>
    where
        F: FnMut(&ArrayView1<f64>) -> Array1<f64>,
        C: FnMut(&ArrayView1<f64>) -> f64,
    {
        self.population.clear();
        let mut rng = rng();

        for _ in 0..self.config.population_size {
            let variables = if let Some((ref lower, ref upper)) = self.config.bounds {
                Array1::from_shape_fn(self.n_variables, |i| {
                    lower[i] + rng.random::<f64>() * (upper[i] - lower[i])
                })
            } else {
                Array1::from_shape_fn(self.n_variables, |_| rng.random::<f64>() * 2.0 - 1.0)
            };

            let objectives = objective_fn(&variables.view());
            self.n_evaluations += 1;

            let constraint_violation = if let Some(ref mut constraint_fn) = constraint_fn {
                constraint_fn(&variables.view()).max(0.0)
            } else {
                0.0
            };

            let solution = MultiObjectiveSolution {
                variables,
                objectives,
                constraint_violation,
                rank: 0,
                crowding_distance: 0.0,
                metadata: HashMap::new(),
            };

            self.population.push(solution);
        }

        Ok(())
    }

    /// Create offspring (reuses NSGA-II logic)
    fn create_offspring<F, C>(
        &mut self,
        objective_fn: &mut F,
        mut constraint_fn: Option<&mut C>,
    ) -> Result<Vec<MultiObjectiveSolution>, OptimizeError>
    where
        F: FnMut(&ArrayView1<f64>) -> Array1<f64>,
        C: FnMut(&ArrayView1<f64>) -> f64,
    {
        let mut offspring = Vec::new();
        let mut rng = rng();

        while offspring.len() < self.config.population_size {
            // Tournament selection
            let parent1 = self.tournament_selection(&mut rng);
            let parent2 = self.tournament_selection(&mut rng);

            // Crossover
            let (mut child1_vars, mut child2_vars) = if rng.random::<f64>()
                < self.config.crossover_probability
            {
                self.simulated_binary_crossover(&parent1.variables, &parent2.variables, &mut rng)
            } else {
                (parent1.variables.clone(), parent2.variables.clone())
            };

            // Mutation
            if rng.random::<f64>() < self.config.mutation_probability {
                self.polynomial_mutation(&mut child1_vars, &mut rng);
            }
            if rng.random::<f64>() < self.config.mutation_probability {
                self.polynomial_mutation(&mut child2_vars, &mut rng);
            }

            // Apply bounds
            if let Some((ref lower, ref upper)) = self.config.bounds {
                self.apply_bounds(&mut child1_vars, lower, upper);
                self.apply_bounds(&mut child2_vars, lower, upper);
            }

            // Evaluate offspring
            for child_vars in [child1_vars, child2_vars] {
                if offspring.len() >= self.config.population_size {
                    break;
                }

                let objectives = objective_fn(&child_vars.view());
                self.n_evaluations += 1;

                let constraint_violation = if let Some(ref mut constraint_fn) = constraint_fn {
                    constraint_fn(&child_vars.view()).max(0.0)
                } else {
                    0.0
                };

                let solution = MultiObjectiveSolution {
                    variables: child_vars,
                    objectives,
                    constraint_violation,
                    rank: 0,
                    crowding_distance: 0.0,
                    metadata: HashMap::new(),
                };

                offspring.push(solution);
            }
        }

        Ok(offspring)
    }

    /// Environmental selection specific to NSGA-III using reference points
    fn environmental_selection_nsga3(
        &self,
        mut population: Vec<MultiObjectiveSolution>,
    ) -> Vec<MultiObjectiveSolution> {
        // Step 1: Non-dominated sorting
        let fronts = self.non_dominated_sorting(&mut population);

        let mut new_population = Vec::new();
        let mut front_idx = 0;

        // Step 2: Add complete fronts until we reach capacity
        while front_idx < fronts.len()
            && new_population.len() + fronts[front_idx].len() <= self.config.population_size
        {
            new_population.extend(fronts[front_idx].clone());
            front_idx += 1;
        }

        // Step 3: If there's a partial front to add, use reference point association
        if front_idx < fronts.len() && new_population.len() < self.config.population_size {
            let remaining = self.config.population_size - new_population.len();
            let partial_front = &fronts[front_idx];

            // Normalize objectives for the combined population
            let all_solutions: Vec<_> = new_population.iter().chain(partial_front.iter()).collect();
            let normalized_objectives = self.normalize_objectives(&all_solutions);

            // Associate solutions with reference points
            let associations = self.associate_with_reference_points(
                &normalized_objectives,
                &new_population,
                partial_front,
            );

            // Select solutions based on reference point niching
            let selected = self.reference_point_selection(partial_front, associations, remaining);
            new_population.extend(selected);
        }

        new_population
    }

    /// Normalize objectives using ideal and nadir points
    fn normalize_objectives(&self, solutions: &[&MultiObjectiveSolution]) -> Array2<f64> {
        let n_solutions = solutions.len();
        let mut normalized = Array2::zeros((n_solutions, self.n_objectives));

        // Find ideal point (minimum in each objective)
        let mut ideal_point = Array1::from_elem(self.n_objectives, f64::INFINITY);
        let mut nadir_point = Array1::from_elem(self.n_objectives, f64::NEG_INFINITY);

        for solution in solutions {
            for (i, &obj_val) in solution.objectives.iter().enumerate() {
                ideal_point[i] = ideal_point[i].min(obj_val);
                nadir_point[i] = nadir_point[i].max(obj_val);
            }
        }

        // Normalize objectives
        for (sol_idx, solution) in solutions.iter().enumerate() {
            for (obj_idx, &obj_val) in solution.objectives.iter().enumerate() {
                let range = nadir_point[obj_idx] - ideal_point[obj_idx];
                normalized[[sol_idx, obj_idx]] = if range > 1e-12 {
                    (obj_val - ideal_point[obj_idx]) / range
                } else {
                    0.0
                };
            }
        }

        normalized
    }

    /// Associate solutions with reference points
    fn associate_with_reference_points(
        &self,
        normalized_objectives: &Array2<f64>,
        new_population: &[MultiObjectiveSolution],
        partial_front: &[MultiObjectiveSolution],
    ) -> Vec<Vec<usize>> {
        let n_ref_points = self.reference_points.nrows();
        let mut associations = vec![Vec::new(); n_ref_points];

        // Count associations for already selected solutions
        let mut niche_count = vec![0; n_ref_points];
        for (sol_idx, _) in new_population.iter().enumerate() {
            let ref_point_idx = self.find_closest_reference_point(sol_idx, normalized_objectives);
            niche_count[ref_point_idx] += 1;
        }

        // Associate partial front solutions with reference points
        let start_idx = new_population.len();
        for (i, _) in partial_front.iter().enumerate() {
            let sol_idx = start_idx + i;
            let ref_point_idx = self.find_closest_reference_point(sol_idx, normalized_objectives);
            associations[ref_point_idx].push(i);
        }

        associations
    }

    /// Find closest reference point for a solution
    fn find_closest_reference_point(
        &self,
        solution_idx: usize,
        normalized_objectives: &Array2<f64>,
    ) -> usize {
        let mut min_distance = f64::INFINITY;
        let mut closest_ref_point = 0;

        for (ref_idx, ref_point) in self.reference_points.outer_iter().enumerate() {
            let distance =
                self.perpendicular_distance(&normalized_objectives.row(solution_idx), &ref_point);

            if distance < min_distance {
                min_distance = distance;
                closest_ref_point = ref_idx;
            }
        }

        closest_ref_point
    }

    /// Calculate perpendicular distance from point to reference direction
    fn perpendicular_distance(
        &self,
        point: &ArrayView1<f64>,
        reference_point: &ArrayView1<f64>,
    ) -> f64 {
        // Calculate projection length
        let dot_product: f64 = point
            .iter()
            .zip(reference_point.iter())
            .map(|(&p, &r)| p * r)
            .sum();
        let ref_norm_sq: f64 = reference_point.iter().map(|&r| r * r).sum();

        if ref_norm_sq < 1e-12 {
            return point.iter().map(|&p| p * p).sum::<f64>().sqrt();
        }

        let projection_length = dot_product / ref_norm_sq.sqrt();

        // Calculate perpendicular distance
        let point_norm: f64 = point.iter().map(|&p| p * p).sum::<f64>().sqrt();

        if projection_length >= point_norm {
            0.0
        } else {
            (point_norm * point_norm - projection_length * projection_length).sqrt()
        }
    }

    /// Select solutions using reference point niching
    fn reference_point_selection(
        &self,
        partial_front: &[MultiObjectiveSolution],
        associations: Vec<Vec<usize>>,
        k: usize,
    ) -> Vec<MultiObjectiveSolution> {
        let mut selected = Vec::new();
        let mut niche_count = vec![0; self.reference_points.nrows()];

        // Calculate current niche counts (should be 0 for partial front)

        for _ in 0..k {
            // Find reference point with minimum niche count that has associated solutions
            let mut min_niche_count = usize::MAX;
            let mut selected_ref_points = Vec::new();

            for (ref_idx, associated_solutions) in associations.iter().enumerate() {
                if !associated_solutions.is_empty() && niche_count[ref_idx] < min_niche_count {
                    min_niche_count = niche_count[ref_idx];
                    selected_ref_points.clear();
                    selected_ref_points.push(ref_idx);
                } else if !associated_solutions.is_empty()
                    && niche_count[ref_idx] == min_niche_count
                {
                    selected_ref_points.push(ref_idx);
                }
            }

            if selected_ref_points.is_empty() {
                break;
            }

            // Randomly select one reference point
            let mut rng = rng();
            let ref_point_idx = selected_ref_points[rng.random_range(0..selected_ref_points.len())];

            // Select solution from this reference point
            let associated_indices = &associations[ref_point_idx];
            if !associated_indices.is_empty() {
                let sol_idx = if niche_count[ref_point_idx] == 0 {
                    // If niche is empty, select closest solution
                    associated_indices[0] // Simplified: could compute actual closest
                } else {
                    // Random selection
                    associated_indices[rng.random_range(0..associated_indices.len())]
                };

                selected.push(partial_front[sol_idx].clone());
                niche_count[ref_point_idx] += 1;

                // Remove selected solution from future consideration
                // (This is simplified - in practice would modify associations)
            }
        }

        selected
    }

    /// Non-dominated sorting (reuse from NSGA-II)
    fn non_dominated_sorting(
        &self,
        population: &mut [MultiObjectiveSolution],
    ) -> Vec<Vec<MultiObjectiveSolution>> {
        let n = population.len();
        let mut domination_count = vec![0; n];
        let mut dominated_solutions = vec![Vec::new(); n];
        let mut fronts = Vec::new();
        let mut current_front = Vec::new();

        // Calculate domination relationships
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    match self.compare_dominance(&population[i], &population[j]) {
                        Ordering::Less => {
                            dominated_solutions[i].push(j);
                        }
                        Ordering::Greater => {
                            domination_count[i] += 1;
                        }
                        Ordering::Equal => {}
                    }
                }
            }

            if domination_count[i] == 0 {
                population[i].rank = 0;
                current_front.push(population[i].clone());
            }
        }

        let mut rank = 0;
        while !current_front.is_empty() {
            fronts.push(current_front.clone());
            let mut next_front = Vec::new();

            for i in 0..n {
                if population[i].rank == rank {
                    for &j in &dominated_solutions[i] {
                        domination_count[j] -= 1;
                        if domination_count[j] == 0 {
                            population[j].rank = rank + 1;
                            next_front.push(population[j].clone());
                        }
                    }
                }
            }

            current_front = next_front;
            rank += 1;
        }

        fronts
    }

    /// Compare two solutions for dominance (reuse from NSGA-II logic)
    fn compare_dominance(
        &self,
        a: &MultiObjectiveSolution,
        b: &MultiObjectiveSolution,
    ) -> Ordering {
        // Handle constraint dominance
        if a.constraint_violation < b.constraint_violation {
            return Ordering::Less;
        } else if a.constraint_violation > b.constraint_violation {
            return Ordering::Greater;
        }

        // Both solutions have same constraint status, check objective dominance
        let mut a_better = false;
        let mut b_better = false;

        for i in 0..self.n_objectives {
            if a.objectives[i] < b.objectives[i] {
                a_better = true;
            } else if a.objectives[i] > b.objectives[i] {
                b_better = true;
            }
        }

        if a_better && !b_better {
            Ordering::Less
        } else if b_better && !a_better {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    /// Tournament selection (reuse logic from NSGA-II)
    fn tournament_selection(&self, rng: &mut impl Rng) -> &MultiObjectiveSolution {
        let tournament_size = 2;
        let mut best_idx = rng.random_range(0..self.population.len());

        for _ in 1..tournament_size {
            let idx = rng.random_range(0..self.population.len());
            if self.dominates_or_better(&self.population[idx], &self.population[best_idx]) {
                best_idx = idx;
            }
        }

        &self.population[best_idx]
    }

    /// Check if solution a dominates or is better than solution b
    fn dominates_or_better(&self, a: &MultiObjectiveSolution, b: &MultiObjectiveSolution) -> bool {
        // First check constraint dominance
        if a.constraint_violation < b.constraint_violation {
            return true;
        } else if a.constraint_violation > b.constraint_violation {
            return false;
        }

        // If both are feasible or both infeasible with same violation, check rank and crowding
        match a.rank.cmp(&b.rank) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Equal => a.crowding_distance > b.crowding_distance,
            std::cmp::Ordering::Greater => false,
        }
    }

    /// Simulated Binary Crossover (reuse from NSGA-II)
    fn simulated_binary_crossover(
        &self,
        parent1: &Array1<f64>,
        parent2: &Array1<f64>,
        rng: &mut impl Rng,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();
        let eta = self.config.crossover_eta;

        for i in 0..self.n_variables {
            if rng.random::<f64>() <= 0.5 {
                let y1 = parent1[i].min(parent2[i]);
                let y2 = parent1[i].max(parent2[i]);

                if (y2 - y1).abs() > 1e-14 {
                    let u = rng.random::<f64>();
                    let beta = if u <= 0.5 {
                        (2.0 * u).powf(1.0 / (eta + 1.0))
                    } else {
                        (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
                    };

                    child1[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1));
                    child2[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1));
                }
            }
        }

        (child1, child2)
    }

    /// Polynomial mutation (reuse from NSGA-II)
    fn polynomial_mutation(&self, individual: &mut Array1<f64>, rng: &mut impl Rng) {
        let eta = self.config.mutation_eta;

        for i in 0..self.n_variables {
            if rng.random::<f64>() <= 1.0 / self.n_variables as f64 {
                let u = rng.random::<f64>();
                let delta = if u < 0.5 {
                    (2.0 * u).powf(1.0 / (eta + 1.0)) - 1.0
                } else {
                    1.0 - (2.0 * (1.0 - u)).powf(1.0 / (eta + 1.0))
                };

                if let Some((ref lower, ref upper)) = self.config.bounds {
                    let range = upper[i] - lower[i];
                    individual[i] += delta * range * 0.1;
                } else {
                    individual[i] += delta * 0.1;
                }
            }
        }
    }

    /// Apply variable bounds (reuse from NSGA-II)
    fn apply_bounds(&self, individual: &mut Array1<f64>, lower: &Array1<f64>, upper: &Array1<f64>) {
        for (i, value) in individual.iter_mut().enumerate() {
            *value = value.max(lower[i]).min(upper[i]);
        }
    }

    /// Calculate hypervolume (reuse basic version, could be enhanced for many objectives)
    fn calculate_hypervolume(&self, reference_point: &Array1<f64>) -> Result<f64, OptimizeError> {
        if reference_point.len() != self.n_objectives {
            return Err(OptimizeError::ValueError(
                "Reference point dimension must match number of objectives".to_string(),
            ));
        }

        let pareto_front: Vec<&MultiObjectiveSolution> = self
            .population
            .iter()
            .filter(|sol| sol.rank == 0 && sol.constraint_violation == 0.0)
            .collect();

        if pareto_front.is_empty() {
            return Ok(0.0);
        }

        // For many objectives, use Monte Carlo approximation
        self.hypervolume_monte_carlo(reference_point, &pareto_front)
    }

    /// Monte Carlo hypervolume calculation
    fn hypervolume_monte_carlo(
        &self,
        reference_point: &Array1<f64>,
        pareto_front: &[&MultiObjectiveSolution],
    ) -> Result<f64, OptimizeError> {
        let n_samples = 100000; // More samples for many objectives
        let mut rng = rng();
        let mut dominated_count = 0;

        // Find bounds for sampling
        let mut min_bounds = reference_point.clone();
        for sol in pareto_front {
            for i in 0..self.n_objectives {
                min_bounds[i] = min_bounds[i].min(sol.objectives[i]);
            }
        }

        for _ in 0..n_samples {
            // Generate random point
            let mut point = Array1::zeros(self.n_objectives);
            for i in 0..self.n_objectives {
                point[i] =
                    min_bounds[i] + rng.random::<f64>() * (reference_point[i] - min_bounds[i]);
            }

            // Check if point is dominated by any solution in Pareto front
            let mut is_dominated = false;
            for sol in pareto_front {
                let mut dominates = true;
                for i in 0..self.n_objectives {
                    if sol.objectives[i] >= point[i] {
                        dominates = false;
                        break;
                    }
                }
                if dominates {
                    is_dominated = true;
                    break;
                }
            }

            if is_dominated {
                dominated_count += 1;
            }
        }

        // Calculate volume
        let mut total_volume = 1.0;
        for i in 0..self.n_objectives {
            total_volume *= reference_point[i] - min_bounds[i];
        }

        Ok(total_volume * dominated_count as f64 / n_samples as f64)
    }
}

/// Scalarization methods for converting multi-objective to single-objective
pub mod scalarization {
    use super::*;

    /// Weighted sum scalarization
    pub fn weighted_sum<F>(objectives_fn: F, weights: &Array1<f64>, x: &ArrayView1<f64>) -> f64
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64>,
    {
        let objectives = objectives_fn(x);
        objectives
            .iter()
            .zip(weights.iter())
            .map(|(&obj, &w)| w * obj)
            .sum()
    }

    /// Weighted Tchebycheff scalarization
    pub fn weighted_tchebycheff<F>(
        objectives_fn: F,
        weights: &Array1<f64>,
        ideal_point: &Array1<f64>,
        x: &ArrayView1<f64>,
    ) -> f64
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64>,
    {
        let objectives = objectives_fn(x);
        objectives
            .iter()
            .zip(weights.iter())
            .zip(ideal_point.iter())
            .map(|((&obj, &w), &ideal)| w * (obj - ideal).abs())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Achievement Scalarizing Function (ASF)
    pub fn achievement_scalarizing<F>(
        objectives_fn: F,
        weights: &Array1<f64>,
        reference_point: &Array1<f64>,
        x: &ArrayView1<f64>,
    ) -> f64
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64>,
    {
        let objectives = objectives_fn(x);
        let rho = 1e-6; // Small augmentation parameter

        let max_term = objectives
            .iter()
            .zip(weights.iter())
            .zip(reference_point.iter())
            .map(|((&obj, &w), &ref_val)| (obj - ref_val) / w)
            .fold(f64::NEG_INFINITY, f64::max);

        let sum_term: f64 = objectives
            .iter()
            .zip(reference_point.iter())
            .map(|(&obj, &ref_val)| obj - ref_val)
            .sum();

        max_term + rho * sum_term
    }

    /// ε-constraint method: minimize one objective subject to constraints on others
    pub struct EpsilonConstraint {
        /// Index of objective to minimize
        pub primary_objective: usize,
        /// Constraint bounds for other objectives
        pub epsilon_constraints: Array1<f64>,
    }

    impl EpsilonConstraint {
        pub fn new(primary_objective: usize, epsilon_constraints: Array1<f64>) -> Self {
            Self {
                primary_objective,
                epsilon_constraints,
            }
        }

        /// Convert to single-objective problem with penalty for constraint violations
        pub fn scalarize<F>(
            &self,
            objectives_fn: F,
            penalty_weight: f64,
        ) -> impl Fn(&ArrayView1<f64>) -> f64
        where
            F: Fn(&ArrayView1<f64>) -> Array1<f64> + Clone,
        {
            let primary_obj = self.primary_objective;
            let constraints = self.epsilon_constraints.clone();

            move |x: &ArrayView1<f64>| -> f64 {
                let objectives = objectives_fn(x);
                let mut result = objectives[primary_obj];

                // Add penalty for constraint violations
                for (i, &eps) in constraints.iter().enumerate() {
                    let obj_idx = if i >= primary_obj { i + 1 } else { i };
                    if obj_idx < objectives.len() && objectives[obj_idx] > eps {
                        result += penalty_weight * (objectives[obj_idx] - eps).powi(2);
                    }
                }

                result
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_nsga2_basic() {
        // Simple bi-objective problem: minimize (x^2, (x-1)^2)
        let objective_fn = |x: &ArrayView1<f64>| array![x[0].powi(2), (x[0] - 1.0).powi(2)];

        let config = MultiObjectiveConfig {
            population_size: 50,
            max_generations: 50,
            ..Default::default()
        };

        let mut optimizer = NSGAII::new(1, 2, Some(config))
            .with_bounds(array![-2.0], array![2.0])
            .unwrap();

        let result = optimizer
            .optimize(objective_fn, None::<fn(&ArrayView1<f64>) -> f64>)
            .unwrap();

        assert!(result.success);
        assert!(!result.pareto_front.is_empty());
        assert!(result.n_evaluations > 0);

        // Check that Pareto front solutions are reasonable
        for solution in &result.pareto_front {
            assert!(solution.variables[0] >= -2.0 && solution.variables[0] <= 2.0);
            assert!(solution.constraint_violation == 0.0);
            assert!(solution.rank == 0);
        }
    }

    #[test]
    fn test_dominance_comparison() {
        let optimizer = NSGAII::new(2, 2, None);

        let sol1 = MultiObjectiveSolution {
            variables: array![0.0, 0.0],
            objectives: array![1.0, 2.0],
            constraint_violation: 0.0,
            rank: 0,
            crowding_distance: 0.0,
            metadata: HashMap::new(),
        };

        let sol2 = MultiObjectiveSolution {
            variables: array![1.0, 1.0],
            objectives: array![2.0, 1.0],
            constraint_violation: 0.0,
            rank: 0,
            crowding_distance: 0.0,
            metadata: HashMap::new(),
        };

        let sol3 = MultiObjectiveSolution {
            variables: array![0.5, 0.5],
            objectives: array![0.5, 1.5],
            constraint_violation: 0.0,
            rank: 0,
            crowding_distance: 0.0,
            metadata: HashMap::new(),
        };

        // sol3 dominates sol1 but not sol2
        assert_eq!(optimizer.compare_dominance(&sol3, &sol1), Ordering::Less);
        assert_eq!(optimizer.compare_dominance(&sol3, &sol2), Ordering::Equal);

        // sol1 and sol2 are non-dominated with respect to each other
        assert_eq!(optimizer.compare_dominance(&sol1, &sol2), Ordering::Equal);
    }

    #[test]
    fn test_scalarization_methods() {
        let objectives_fn = |x: &ArrayView1<f64>| array![x[0].powi(2), x[1].powi(2)];
        let x = array![1.0, 2.0];
        let weights = array![0.5, 0.5];
        let ideal_point = array![0.0, 0.0];

        // Test weighted sum
        let weighted_result = scalarization::weighted_sum(objectives_fn, &weights, &x.view());
        assert_abs_diff_eq!(weighted_result, 2.5, epsilon = 1e-10); // 0.5*1 + 0.5*4

        // Test weighted Tchebycheff
        let tcheby_result =
            scalarization::weighted_tchebycheff(objectives_fn, &weights, &ideal_point, &x.view());
        assert_abs_diff_eq!(tcheby_result, 2.0, epsilon = 1e-10); // max(0.5*1, 0.5*4)
    }

    #[test]
    fn test_hypervolume_2d() {
        let config = MultiObjectiveConfig::default();
        let optimizer = NSGAII::new(1, 2, Some(config));

        // Create a proper Pareto front (non-dominated points with decreasing y when x increases)
        let population = vec![
            MultiObjectiveSolution {
                variables: array![0.0],
                objectives: array![1.0, 3.0],
                constraint_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
                metadata: HashMap::new(),
            },
            MultiObjectiveSolution {
                variables: array![1.0],
                objectives: array![2.0, 2.0],
                constraint_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
                metadata: HashMap::new(),
            },
            MultiObjectiveSolution {
                variables: array![2.0],
                objectives: array![3.0, 1.0],
                constraint_violation: 0.0,
                rank: 0,
                crowding_distance: 0.0,
                metadata: HashMap::new(),
            },
        ];

        let optimizer_with_pop = NSGAII {
            population,
            ..optimizer
        };

        let reference_point = array![4.0, 4.0];
        let hypervolume = optimizer_with_pop
            .calculate_hypervolume(&reference_point)
            .unwrap();

        assert!(hypervolume > 0.0);
        assert!(hypervolume < 16.0); // Should be less than total area
    }
}
