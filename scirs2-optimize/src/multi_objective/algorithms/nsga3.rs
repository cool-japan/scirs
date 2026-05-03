//! NSGA-III (Non-dominated Sorting Genetic Algorithm III)
//!
//! Reference-point-based many-objective evolutionary algorithm.
//! NSGA-III extends NSGA-II for many-objective optimization (>3 objectives)
//! by replacing crowding distance with reference-point-based selection.
//!
//! Key features:
//! - Das-Dennis reference point generation
//! - Adaptive normalization of objective values
//! - Reference-point association using perpendicular distance
//! - Niching-based selection for population diversity
//!
//! # References
//!
//! - Deb & Jain, "An Evolutionary Many-Objective Optimization Algorithm Using
//!   Reference-Point-Based Nondominated Sorting Approach, Part I", IEEE TEC 2014

use super::{utils, MultiObjectiveConfig, MultiObjectiveOptimizer};
use crate::error::OptimizeError;
use crate::multi_objective::crossover::{CrossoverOperator, SimulatedBinaryCrossover};
use crate::multi_objective::mutation::{MutationOperator, PolynomialMutation};
use crate::multi_objective::selection::{SelectionOperator, TournamentSelection};
use crate::multi_objective::solutions::{MultiObjectiveResult, MultiObjectiveSolution, Population};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};
use std::cmp::Ordering;

/// NSGA-III optimizer for many-objective optimization
pub struct NSGAIII {
    config: MultiObjectiveConfig,
    n_objectives: usize,
    n_variables: usize,
    /// Structured reference points (Das-Dennis)
    reference_points: Vec<Array1<f64>>,
    population: Population,
    generation: usize,
    n_evaluations: usize,
    rng: StdRng,
    crossover: SimulatedBinaryCrossover,
    mutation: PolynomialMutation,
    selection: TournamentSelection,
    /// Ideal point (best value per objective)
    ideal_point: Array1<f64>,
    /// Niche count for each reference point
    niche_count: Vec<usize>,
    convergence_history: Vec<f64>,
}

impl NSGAIII {
    /// Create new NSGA-III optimizer with default reference points
    pub fn new(population_size: usize, n_objectives: usize, n_variables: usize) -> Self {
        let config = MultiObjectiveConfig {
            population_size,
            ..Default::default()
        };
        Self::with_config(config, n_objectives, n_variables, None)
    }

    /// Create NSGA-III with full configuration and optional custom reference points
    pub fn with_config(
        config: MultiObjectiveConfig,
        n_objectives: usize,
        n_variables: usize,
        custom_reference_points: Option<Vec<Array1<f64>>>,
    ) -> Self {
        let seed = config.random_seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(42)
        });

        let rng = StdRng::seed_from_u64(seed);

        let crossover =
            SimulatedBinaryCrossover::new(config.crossover_eta, config.crossover_probability);
        let mutation = PolynomialMutation::new(config.mutation_probability, config.mutation_eta);
        let selection = TournamentSelection::new(2);

        // Generate reference points
        let reference_points = custom_reference_points.unwrap_or_else(|| {
            let n_partitions = Self::auto_partitions(n_objectives, config.population_size);
            utils::generate_das_dennis_points(n_objectives, n_partitions)
        });

        let niche_count = vec![0; reference_points.len()];

        Self {
            config,
            n_objectives,
            n_variables,
            reference_points,
            population: Population::new(),
            generation: 0,
            n_evaluations: 0,
            rng,
            crossover,
            mutation,
            selection,
            ideal_point: Array1::from_elem(n_objectives, f64::INFINITY),
            niche_count,
            convergence_history: Vec::new(),
        }
    }

    /// Return a reference to the current structured reference points.
    pub fn reference_points(&self) -> &[Array1<f64>] {
        &self.reference_points
    }

    /// Automatically determine the number of partitions based on objectives and pop size
    fn auto_partitions(n_objectives: usize, pop_size: usize) -> usize {
        // Heuristic: choose partitions so Das-Dennis points ~ pop_size
        // Number of points = C(H + M - 1, M - 1) where H = partitions, M = objectives
        if n_objectives <= 2 {
            return pop_size.max(4);
        }
        // For higher dimensions, use fewer partitions
        let mut p = 1;
        loop {
            let n_points = binomial_coefficient(p + n_objectives - 1, n_objectives - 1);
            if n_points >= pop_size / 2 {
                return p;
            }
            p += 1;
            if p > 50 {
                return p;
            }
        }
    }

    /// Evaluate a single individual
    fn evaluate_individual<F>(
        &mut self,
        variables: &Array1<f64>,
        objective_function: &F,
    ) -> Result<Array1<f64>, OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        self.n_evaluations += 1;

        if let Some(max_evals) = self.config.max_evaluations {
            if self.n_evaluations > max_evals {
                return Err(OptimizeError::MaxEvaluationsReached);
            }
        }

        let objectives = objective_function(&variables.view());
        if objectives.len() != self.n_objectives {
            return Err(OptimizeError::InvalidInput(format!(
                "Expected {} objectives, got {}",
                self.n_objectives,
                objectives.len()
            )));
        }

        Ok(objectives)
    }

    /// Update the ideal point from a set of solutions
    fn update_ideal_point(&mut self, solutions: &[MultiObjectiveSolution]) {
        for sol in solutions {
            for (i, &obj) in sol.objectives.iter().enumerate() {
                if obj < self.ideal_point[i] {
                    self.ideal_point[i] = obj;
                }
            }
        }
    }

    /// Normalize objectives using ideal and extreme points (ASF-based)
    fn normalize_objectives(&self, solutions: &[MultiObjectiveSolution]) -> Vec<Array1<f64>> {
        let n = solutions.len();
        if n == 0 {
            return vec![];
        }

        // Translate by ideal point
        let translated: Vec<Array1<f64>> = solutions
            .iter()
            .map(|sol| &sol.objectives - &self.ideal_point)
            .collect();

        // Find extreme points using Achievement Scalarizing Function (ASF)
        let mut extreme_points = Vec::with_capacity(self.n_objectives);
        for j in 0..self.n_objectives {
            let mut best_asf = f64::INFINITY;
            let mut best_idx = 0;
            for (i, t) in translated.iter().enumerate() {
                // ASF: max(t_k / w_k) where w is the j-th axis weight
                let asf = (0..self.n_objectives)
                    .map(|k| {
                        if k == j {
                            t[k] // weight = 1 for target objective
                        } else {
                            t[k] * 1e6 // large weight for others
                        }
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                if asf < best_asf {
                    best_asf = asf;
                    best_idx = i;
                }
            }
            extreme_points.push(translated[best_idx].clone());
        }

        // Compute intercepts from the extreme points
        let mut intercepts = Array1::from_elem(self.n_objectives, 1.0);
        if extreme_points.len() == self.n_objectives {
            // Build the matrix of extreme points and solve for intercepts
            // Simplified: use the diagonal elements as intercepts
            for (j, ep) in extreme_points.iter().enumerate() {
                let val = ep[j];
                if val > 1e-10 {
                    intercepts[j] = val;
                }
            }
        }

        // Normalize each translated objective
        translated
            .iter()
            .map(|t| {
                let mut normalized = Array1::zeros(self.n_objectives);
                for j in 0..self.n_objectives {
                    normalized[j] = if intercepts[j] > 1e-10 {
                        t[j] / intercepts[j]
                    } else {
                        t[j]
                    };
                }
                normalized
            })
            .collect()
    }

    /// Associate each solution to its nearest reference point
    /// Returns (reference_point_index, perpendicular_distance) for each solution
    fn associate_to_reference_points(
        &self,
        normalized_objectives: &[Array1<f64>],
    ) -> Vec<(usize, f64)> {
        normalized_objectives
            .iter()
            .map(|obj| {
                let mut min_dist = f64::INFINITY;
                let mut min_idx = 0;

                for (rp_idx, rp) in self.reference_points.iter().enumerate() {
                    let dist = perpendicular_distance(obj, rp);
                    if dist < min_dist {
                        min_dist = dist;
                        min_idx = rp_idx;
                    }
                }

                (min_idx, min_dist)
            })
            .collect()
    }

    /// Niching-based selection from the last front
    fn niching_selection(
        &mut self,
        last_front_indices: &[usize],
        all_solutions: &[MultiObjectiveSolution],
        associations: &[(usize, f64)],
        selected_so_far: &[usize],
        remaining: usize,
    ) -> Vec<usize> {
        // Compute niche counts for already-selected solutions
        let mut niche_count = vec![0usize; self.reference_points.len()];
        for &idx in selected_so_far {
            let (rp_idx, _) = associations[idx];
            niche_count[rp_idx] += 1;
        }

        let mut selected = Vec::with_capacity(remaining);
        let mut available: Vec<usize> = last_front_indices.to_vec();

        for _ in 0..remaining {
            if available.is_empty() {
                break;
            }

            // Find the reference point with minimum niche count
            // among those that have at least one member in the last front
            let relevant_rps: Vec<usize> = available
                .iter()
                .map(|&idx| associations[idx].0)
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            if relevant_rps.is_empty() {
                break;
            }

            let min_niche = relevant_rps
                .iter()
                .map(|&rp| niche_count[rp])
                .min()
                .unwrap_or(0);

            // Collect reference points with minimum niche count
            let min_niche_rps: Vec<usize> = relevant_rps
                .iter()
                .filter(|&&rp| niche_count[rp] == min_niche)
                .copied()
                .collect();

            // Randomly pick one
            let chosen_rp_idx = self.rng.random_range(0..min_niche_rps.len());
            let chosen_rp = min_niche_rps[chosen_rp_idx];

            // Find members of the last front associated with this reference point
            let rp_members: Vec<usize> = available
                .iter()
                .filter(|&&idx| associations[idx].0 == chosen_rp)
                .copied()
                .collect();

            if rp_members.is_empty() {
                continue;
            }

            let chosen_member = if min_niche == 0 {
                // Pick the one with smallest perpendicular distance
                *rp_members
                    .iter()
                    .min_by(|&&a, &&b| {
                        associations[a]
                            .1
                            .partial_cmp(&associations[b].1)
                            .unwrap_or(Ordering::Equal)
                    })
                    .unwrap_or(&rp_members[0])
            } else {
                // Randomly pick
                rp_members[self.rng.random_range(0..rp_members.len())]
            };

            selected.push(chosen_member);
            niche_count[chosen_rp] += 1;
            available.retain(|&idx| idx != chosen_member);
        }

        selected
    }

    /// Create offspring through crossover and mutation
    fn create_offspring<F>(
        &mut self,
        objective_function: &F,
    ) -> Result<Vec<MultiObjectiveSolution>, OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        let mut offspring = Vec::new();
        let pop_solutions = self.population.solutions().to_vec();

        while offspring.len() < self.config.population_size {
            let selected = self.selection.select(&pop_solutions, 2);
            if selected.len() < 2 {
                break;
            }

            let p1_vars = selected[0].variables.as_slice().unwrap_or(&[]);
            let p2_vars = selected[1].variables.as_slice().unwrap_or(&[]);

            let (mut c1_vars, mut c2_vars) = self.crossover.crossover(p1_vars, p2_vars);

            let bounds: Vec<(f64, f64)> = if let Some((lower, upper)) = &self.config.bounds {
                lower
                    .iter()
                    .zip(upper.iter())
                    .map(|(&l, &u)| (l, u))
                    .collect()
            } else {
                vec![(-1.0, 1.0); self.n_variables]
            };

            self.mutation.mutate(&mut c1_vars, &bounds);
            self.mutation.mutate(&mut c2_vars, &bounds);

            let c1_arr = Array1::from_vec(c1_vars);
            let c1_obj = self.evaluate_individual(&c1_arr, objective_function)?;
            offspring.push(MultiObjectiveSolution::new(c1_arr, c1_obj));

            if offspring.len() < self.config.population_size {
                let c2_arr = Array1::from_vec(c2_vars);
                let c2_obj = self.evaluate_individual(&c2_arr, objective_function)?;
                offspring.push(MultiObjectiveSolution::new(c2_arr, c2_obj));
            }
        }

        Ok(offspring)
    }

    /// NSGA-III environmental selection using reference-point-based niching
    fn environmental_selection(
        &mut self,
        combined: Vec<MultiObjectiveSolution>,
    ) -> Vec<MultiObjectiveSolution> {
        let target_size = self.config.population_size;
        if combined.len() <= target_size {
            return combined;
        }

        // Non-dominated sorting
        let mut temp_pop = Population::from_solutions(combined.clone());
        let fronts = temp_pop.non_dominated_sort();

        // Fill until the critical front
        let mut selected_indices: Vec<usize> = Vec::new();
        let mut last_front_idx = 0;

        for (fi, front) in fronts.iter().enumerate() {
            if selected_indices.len() + front.len() <= target_size {
                selected_indices.extend(front);
            } else {
                last_front_idx = fi;
                break;
            }
        }

        if selected_indices.len() >= target_size {
            return selected_indices
                .iter()
                .take(target_size)
                .map(|&i| combined[i].clone())
                .collect();
        }

        // Need to select from the last (critical) front using niching
        let remaining = target_size - selected_indices.len();
        let last_front = &fronts[last_front_idx];

        // Update ideal point
        self.update_ideal_point(&combined);

        // Normalize objectives for ALL solutions considered
        let all_considered: Vec<usize> = selected_indices
            .iter()
            .chain(last_front.iter())
            .copied()
            .collect();
        let all_solutions: Vec<MultiObjectiveSolution> = all_considered
            .iter()
            .map(|&i| combined[i].clone())
            .collect();

        let normalized = self.normalize_objectives(&all_solutions);
        let associations = self.associate_to_reference_points(&normalized);

        // Map indices back
        let n_selected = selected_indices.len();
        let last_front_local: Vec<usize> = (n_selected..all_solutions.len()).collect();
        let selected_local: Vec<usize> = (0..n_selected).collect();

        let niching_result = self.niching_selection(
            &last_front_local,
            &all_solutions,
            &associations,
            &selected_local,
            remaining,
        );

        // Build final selection
        let mut result: Vec<MultiObjectiveSolution> = selected_indices
            .iter()
            .map(|&i| combined[i].clone())
            .collect();

        for local_idx in niching_result {
            let global_idx = all_considered[local_idx];
            result.push(combined[global_idx].clone());
        }

        result
    }

    /// Calculate metrics for the current generation
    fn calculate_metrics(&mut self) {
        if let Some(ref_point) = &self.config.reference_point {
            let pareto_front = self.population.extract_pareto_front();
            let hv = utils::calculate_hypervolume(&pareto_front, ref_point);
            self.convergence_history.push(hv);
        }
    }
}

impl MultiObjectiveOptimizer for NSGAIII {
    fn optimize<F>(&mut self, objective_function: F) -> Result<MultiObjectiveResult, OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        self.initialize_population()?;

        // Generate and evaluate initial population
        let initial_vars = utils::generate_random_population(
            self.config.population_size,
            self.n_variables,
            &self.config.bounds,
        );

        let mut initial_solutions = Vec::new();
        for vars in initial_vars {
            let objs = self.evaluate_individual(&vars, &objective_function)?;
            initial_solutions.push(MultiObjectiveSolution::new(vars, objs));
        }

        self.update_ideal_point(&initial_solutions);
        self.population = Population::from_solutions(initial_solutions);

        // Main evolution loop
        while self.generation < self.config.max_generations {
            if self.check_convergence() {
                break;
            }
            self.evolve_generation(&objective_function)?;
        }

        // Extract results
        let pareto_front = self.population.extract_pareto_front();
        let hypervolume = self
            .config
            .reference_point
            .as_ref()
            .map(|rp| utils::calculate_hypervolume(&pareto_front, rp));

        let mut result = MultiObjectiveResult::new(
            pareto_front,
            self.population.solutions().to_vec(),
            self.n_evaluations,
            self.generation,
        );
        result.hypervolume = hypervolume;
        result.metrics.convergence_history = self.convergence_history.clone();
        result.metrics.population_stats = self.population.calculate_statistics();

        Ok(result)
    }

    fn evolve_generation<F>(&mut self, objective_function: &F) -> Result<(), OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        let offspring = self.create_offspring(objective_function)?;

        let mut combined = self.population.solutions().to_vec();
        combined.extend(offspring);

        let next_pop = self.environmental_selection(combined);
        self.population = Population::from_solutions(next_pop);

        self.generation += 1;
        self.calculate_metrics();

        Ok(())
    }

    fn initialize_population(&mut self) -> Result<(), OptimizeError> {
        self.population.clear();
        self.generation = 0;
        self.n_evaluations = 0;
        self.ideal_point = Array1::from_elem(self.n_objectives, f64::INFINITY);
        self.niche_count = vec![0; self.reference_points.len()];
        self.convergence_history.clear();
        Ok(())
    }

    fn check_convergence(&self) -> bool {
        if let Some(max_evals) = self.config.max_evaluations {
            if self.n_evaluations >= max_evals {
                return true;
            }
        }

        if self.convergence_history.len() >= 10 {
            let recent = &self.convergence_history[self.convergence_history.len() - 10..];
            let max_hv = recent.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_hv = recent.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            if (max_hv - min_hv) < self.config.tolerance {
                return true;
            }
        }

        false
    }

    fn get_population(&self) -> &Population {
        &self.population
    }

    fn get_generation(&self) -> usize {
        self.generation
    }

    fn get_evaluations(&self) -> usize {
        self.n_evaluations
    }

    fn name(&self) -> &str {
        "NSGA-III"
    }
}

/// Compute perpendicular distance from a point to a reference line (through origin)
fn perpendicular_distance(point: &Array1<f64>, direction: &Array1<f64>) -> f64 {
    let dir_norm_sq = direction.dot(direction);
    if dir_norm_sq < 1e-30 {
        return point.dot(point).sqrt();
    }

    // Projection of point onto direction
    let proj_scalar = point.dot(direction) / dir_norm_sq;
    let proj = proj_scalar * direction;

    // Distance = |point - projection|
    let diff = point - &proj;
    diff.dot(&diff).sqrt()
}

/// Compute binomial coefficient C(n, k)
fn binomial_coefficient(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result.saturating_mul(n - i);
        result /= i + 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, s};

    fn zdt1(x: &ArrayView1<f64>) -> Array1<f64> {
        let f1 = x[0];
        let g = 1.0 + 9.0 * x.slice(s![1..]).sum() / (x.len() - 1) as f64;
        let f2 = g * (1.0 - (f1 / g).sqrt());
        array![f1, f2]
    }

    fn dtlz1(x: &ArrayView1<f64>) -> Array1<f64> {
        // 3-objective DTLZ1
        let n = x.len();
        let k = n - 2; // for 3 objectives, k = n - M + 1
        let g: f64 = (0..k)
            .map(|i| {
                let xi = x[n - k + i];
                (xi - 0.5).powi(2) - (20.0 * std::f64::consts::PI * (xi - 0.5)).cos()
            })
            .sum::<f64>();
        let g = 100.0 * (k as f64 + g);

        let f1 = 0.5 * x[0] * x[1] * (1.0 + g);
        let f2 = 0.5 * x[0] * (1.0 - x[1]) * (1.0 + g);
        let f3 = 0.5 * (1.0 - x[0]) * (1.0 + g);
        array![f1, f2, f3]
    }

    #[test]
    fn test_nsga3_creation() {
        let nsga3 = NSGAIII::new(100, 3, 5);
        assert_eq!(nsga3.n_objectives, 3);
        assert_eq!(nsga3.n_variables, 5);
        assert!(!nsga3.reference_points.is_empty());
        assert_eq!(nsga3.generation, 0);
    }

    #[test]
    fn test_nsga3_with_config() {
        let mut config = MultiObjectiveConfig::default();
        config.population_size = 50;
        config.max_generations = 10;
        config.random_seed = Some(42);

        let nsga3 = NSGAIII::with_config(config, 2, 3, None);
        assert_eq!(nsga3.n_objectives, 2);
        assert!(!nsga3.reference_points.is_empty());
    }

    #[test]
    fn test_nsga3_custom_reference_points() {
        let rps = vec![
            array![1.0, 0.0, 0.0],
            array![0.0, 1.0, 0.0],
            array![0.0, 0.0, 1.0],
            array![0.5, 0.5, 0.0],
            array![0.5, 0.0, 0.5],
            array![0.0, 0.5, 0.5],
            array![0.333, 0.333, 0.334],
        ];

        let config = MultiObjectiveConfig {
            population_size: 20,
            max_generations: 5,
            random_seed: Some(42),
            ..Default::default()
        };

        let nsga3 = NSGAIII::with_config(config, 3, 5, Some(rps.clone()));
        assert_eq!(nsga3.reference_points.len(), 7);
    }

    #[test]
    fn test_nsga3_optimize_zdt1() {
        let mut config = MultiObjectiveConfig::default();
        config.max_generations = 10;
        config.population_size = 20;
        config.bounds = Some((Array1::zeros(3), Array1::ones(3)));
        config.random_seed = Some(42);

        let mut nsga3 = NSGAIII::with_config(config, 2, 3, None);
        let result = nsga3.optimize(zdt1);

        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(res.success);
        assert!(!res.pareto_front.is_empty());
        assert!(res.n_evaluations > 0);
    }

    #[test]
    fn test_nsga3_optimize_dtlz1() {
        let mut config = MultiObjectiveConfig::default();
        config.max_generations = 10;
        config.population_size = 20;
        config.bounds = Some((Array1::zeros(5), Array1::ones(5)));
        config.random_seed = Some(42);

        let mut nsga3 = NSGAIII::with_config(config, 3, 5, None);
        let result = nsga3.optimize(dtlz1);

        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(res.success);
        assert!(!res.pareto_front.is_empty());
    }

    #[test]
    fn test_nsga3_max_evaluations() {
        let mut config = MultiObjectiveConfig::default();
        config.max_generations = 1000;
        config.max_evaluations = Some(50);
        config.population_size = 10;
        config.bounds = Some((Array1::zeros(3), Array1::ones(3)));
        config.random_seed = Some(42);

        let mut nsga3 = NSGAIII::with_config(config, 2, 3, None);
        let result = nsga3.optimize(zdt1);
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(res.n_evaluations <= 60); // Allow slight overshoot
    }

    #[test]
    fn test_perpendicular_distance() {
        let point = array![1.0, 1.0];
        let direction = array![1.0, 0.0];

        let dist = perpendicular_distance(&point, &direction);
        assert!(
            (dist - 1.0).abs() < 1e-10,
            "Distance should be 1.0, got {}",
            dist
        );

        // Point on the line
        let point2 = array![2.0, 0.0];
        let dist2 = perpendicular_distance(&point2, &direction);
        assert!(dist2.abs() < 1e-10, "Distance should be 0.0, got {}", dist2);
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 2), 10);
        assert_eq!(binomial_coefficient(10, 3), 120);
        assert_eq!(binomial_coefficient(4, 0), 1);
        assert_eq!(binomial_coefficient(4, 4), 1);
        assert_eq!(binomial_coefficient(0, 0), 1);
    }

    #[test]
    fn test_nsga3_name() {
        let nsga3 = NSGAIII::new(50, 2, 3);
        assert_eq!(nsga3.name(), "NSGA-III");
    }

    #[test]
    fn test_nsga3_convergence_check() {
        let config = MultiObjectiveConfig {
            tolerance: 1e-10,
            max_generations: 2,
            population_size: 10,
            ..Default::default()
        };
        let nsga3 = NSGAIII::with_config(config, 2, 2, None);
        assert!(!nsga3.check_convergence());
    }
}
