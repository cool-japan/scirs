//! # Ant Colony Optimization (ACO)
//!
//! Population-based metaheuristic for combinatorial optimization:
//! - **Ant System (AS)**: Classic ant-based algorithm with pheromone trails
//! - **Max-Min Ant System (MMAS)**: Bounded pheromone levels to avoid stagnation
//! - **TSP solver interface**: Specialized for Travelling Salesman Problems
//! - **General permutation problem interface**: Solve any permutation-based problem

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{rng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ---------------------------------------------------------------------------
// Problem Traits
// ---------------------------------------------------------------------------

/// A combinatorial optimization problem over a set of discrete elements.
///
/// The ants construct solutions by building paths through a graph where
/// each node represents a decision point and each edge has associated
/// pheromone and heuristic information.
pub trait CombinatorialProblem {
    /// Number of nodes (cities, tasks, etc.)
    fn num_nodes(&self) -> usize;

    /// Heuristic desirability of going from node `i` to node `j`.
    /// Higher is more desirable (typically 1/distance for TSP).
    fn heuristic(&self, i: usize, j: usize) -> f64;

    /// Evaluate the quality (cost) of a complete solution (permutation of node indices).
    /// Lower is better.
    fn evaluate(&self, solution: &[usize]) -> f64;

    /// Check if moving from current node to next node is feasible.
    /// Default returns true (all moves allowed).
    fn is_feasible(&self, _current: usize, _next: usize, _visited: &[bool]) -> bool {
        true
    }
}

/// TSP problem definition
#[derive(Debug, Clone)]
pub struct TspProblem {
    /// Distance matrix (n x n)
    distances: Vec<Vec<f64>>,
    n: usize,
}

impl TspProblem {
    /// Create a TSP from a distance matrix
    pub fn new(distances: Vec<Vec<f64>>) -> OptimizeResult<Self> {
        let n = distances.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Distance matrix must not be empty".to_string(),
            ));
        }
        for row in &distances {
            if row.len() != n {
                return Err(OptimizeError::InvalidInput(
                    "Distance matrix must be square".to_string(),
                ));
            }
        }
        Ok(Self { distances, n })
    }

    /// Create a TSP from 2D coordinates (Euclidean distances)
    pub fn from_coordinates(coords: &[(f64, f64)]) -> OptimizeResult<Self> {
        let n = coords.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Coordinates must not be empty".to_string(),
            ));
        }
        let mut distances = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = coords[i].0 - coords[j].0;
                let dy = coords[i].1 - coords[j].1;
                let d = (dx * dx + dy * dy).sqrt();
                distances[i][j] = d;
                distances[j][i] = d;
            }
        }
        Ok(Self { distances, n })
    }

    /// Get number of cities
    pub fn num_cities(&self) -> usize {
        self.n
    }

    /// Get distance between two cities
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        self.distances[i][j]
    }
}

impl CombinatorialProblem for TspProblem {
    fn num_nodes(&self) -> usize {
        self.n
    }

    fn heuristic(&self, i: usize, j: usize) -> f64 {
        let d = self.distances[i][j];
        if d > 0.0 {
            1.0 / d
        } else {
            1e10 // very high desirability for zero distance (same city)
        }
    }

    fn evaluate(&self, solution: &[usize]) -> f64 {
        if solution.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        for i in 0..solution.len() - 1 {
            total += self.distances[solution[i]][solution[i + 1]];
        }
        // Return to start
        total += self.distances[*solution.last().unwrap_or(&0)][solution[0]];
        total
    }
}

/// A general permutation problem
#[derive(Clone)]
pub struct PermutationProblem {
    n: usize,
    heuristic_matrix: Vec<Vec<f64>>,
    eval_fn: std::sync::Arc<dyn Fn(&[usize]) -> f64 + Send + Sync>,
}

impl std::fmt::Debug for PermutationProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PermutationProblem")
            .field("n", &self.n)
            .field("heuristic_matrix_size", &self.heuristic_matrix.len())
            .finish()
    }
}

impl PermutationProblem {
    /// Create a new permutation problem
    ///
    /// - `n`: number of elements to permute
    /// - `heuristic_matrix`: n x n matrix of heuristic values (higher = more desirable)
    /// - `eval_fn`: function that evaluates a permutation (lower = better)
    pub fn new<F>(n: usize, heuristic_matrix: Vec<Vec<f64>>, eval_fn: F) -> OptimizeResult<Self>
    where
        F: Fn(&[usize]) -> f64 + Send + Sync + 'static,
    {
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Problem size must be > 0".to_string(),
            ));
        }
        if heuristic_matrix.len() != n {
            return Err(OptimizeError::InvalidInput(
                "Heuristic matrix size must match n".to_string(),
            ));
        }
        Ok(Self {
            n,
            heuristic_matrix,
            eval_fn: std::sync::Arc::new(eval_fn),
        })
    }

    /// Create with uniform heuristic (all edges equally desirable)
    pub fn with_uniform_heuristic<F>(n: usize, eval_fn: F) -> OptimizeResult<Self>
    where
        F: Fn(&[usize]) -> f64 + Send + Sync + 'static,
    {
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Problem size must be > 0".to_string(),
            ));
        }
        let heuristic_matrix = vec![vec![1.0; n]; n];
        Ok(Self {
            n,
            heuristic_matrix,
            eval_fn: std::sync::Arc::new(eval_fn),
        })
    }
}

impl CombinatorialProblem for PermutationProblem {
    fn num_nodes(&self) -> usize {
        self.n
    }

    fn heuristic(&self, i: usize, j: usize) -> f64 {
        self.heuristic_matrix[i][j]
    }

    fn evaluate(&self, solution: &[usize]) -> f64 {
        (self.eval_fn)(solution)
    }
}

// ---------------------------------------------------------------------------
// ACO Result
// ---------------------------------------------------------------------------

/// Result from ACO optimization
#[derive(Debug, Clone)]
pub struct AcoResult {
    /// Best solution (permutation of node indices)
    pub best_solution: Vec<usize>,
    /// Cost of the best solution
    pub best_cost: f64,
    /// Number of iterations completed
    pub iterations: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// Whether the optimizer converged
    pub converged: bool,
    /// History of best cost per iteration (for convergence analysis)
    pub cost_history: Vec<f64>,
    /// Termination message
    pub message: String,
}

// ---------------------------------------------------------------------------
// Ant System (Classic AS)
// ---------------------------------------------------------------------------

/// Options for Ant System
#[derive(Debug, Clone)]
pub struct AntSystemOptions {
    /// Number of ants
    pub num_ants: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Pheromone importance (alpha)
    pub alpha: f64,
    /// Heuristic importance (beta)
    pub beta: f64,
    /// Evaporation rate (rho), in [0, 1]
    pub evaporation_rate: f64,
    /// Initial pheromone level
    pub initial_pheromone: f64,
    /// Pheromone deposit factor Q
    pub q_factor: f64,
    /// Random seed
    pub seed: Option<u64>,
    /// Convergence tolerance (stop if best cost doesn't improve by this fraction)
    pub tol: f64,
    /// Patience: number of iterations without improvement before stopping
    pub patience: usize,
}

impl Default for AntSystemOptions {
    fn default() -> Self {
        Self {
            num_ants: 20,
            max_iter: 200,
            alpha: 1.0,
            beta: 2.0,
            evaporation_rate: 0.5,
            initial_pheromone: 1.0,
            q_factor: 100.0,
            seed: None,
            tol: 1e-8,
            patience: 50,
        }
    }
}

/// Classic Ant System optimizer
pub struct AntColonyOptimizer {
    options: AntSystemOptions,
    rng: StdRng,
}

impl AntColonyOptimizer {
    /// Create a new ACO optimizer
    pub fn new(options: AntSystemOptions) -> Self {
        let seed = options.seed.unwrap_or_else(|| rng().random());
        Self {
            options,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Solve a combinatorial problem
    pub fn solve<P: CombinatorialProblem>(&mut self, problem: &P) -> OptimizeResult<AcoResult> {
        let n = problem.num_nodes();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Problem has 0 nodes".to_string(),
            ));
        }

        // Initialize pheromone matrix
        let mut pheromone = vec![vec![self.options.initial_pheromone; n]; n];
        let mut best_solution: Vec<usize> = Vec::new();
        let mut best_cost = f64::INFINITY;
        let mut nfev: usize = 0;
        let mut cost_history = Vec::with_capacity(self.options.max_iter);
        let mut no_improve_count: usize = 0;

        for iteration in 0..self.options.max_iter {
            let mut iteration_solutions = Vec::with_capacity(self.options.num_ants);
            let mut iteration_costs = Vec::with_capacity(self.options.num_ants);

            // Each ant constructs a solution
            for _ in 0..self.options.num_ants {
                let solution = self.construct_solution(problem, &pheromone, n);
                let cost = problem.evaluate(&solution);
                nfev += 1;

                if cost < best_cost {
                    best_cost = cost;
                    best_solution = solution.clone();
                    no_improve_count = 0;
                }

                iteration_solutions.push(solution);
                iteration_costs.push(cost);
            }

            cost_history.push(best_cost);

            // Evaporate pheromone
            for row in &mut pheromone {
                for val in row.iter_mut() {
                    *val *= 1.0 - self.options.evaporation_rate;
                    if *val < 1e-15 {
                        *val = 1e-15; // prevent zero pheromone
                    }
                }
            }

            // Deposit pheromone from all ants
            for (solution, cost) in iteration_solutions.iter().zip(iteration_costs.iter()) {
                if *cost > 0.0 {
                    let deposit = self.options.q_factor / cost;
                    for i in 0..solution.len() {
                        let from = solution[i];
                        let to = solution[(i + 1) % solution.len()];
                        pheromone[from][to] += deposit;
                        pheromone[to][from] += deposit;
                    }
                }
            }

            no_improve_count += 1;
            if no_improve_count >= self.options.patience {
                return Ok(AcoResult {
                    best_solution,
                    best_cost,
                    iterations: iteration + 1,
                    nfev,
                    converged: true,
                    cost_history,
                    message: format!(
                        "Converged after {} iterations (no improvement for {} iterations)",
                        iteration + 1,
                        self.options.patience
                    ),
                });
            }
        }

        Ok(AcoResult {
            best_solution,
            best_cost,
            iterations: self.options.max_iter,
            nfev,
            converged: false,
            cost_history,
            message: format!("Completed {} iterations", self.options.max_iter),
        })
    }

    /// Construct a single ant solution using probabilistic selection
    fn construct_solution<P: CombinatorialProblem>(
        &mut self,
        problem: &P,
        pheromone: &[Vec<f64>],
        n: usize,
    ) -> Vec<usize> {
        let mut visited = vec![false; n];
        let mut solution = Vec::with_capacity(n);

        // Start from a random node
        let start = self.rng.random_range(0..n);
        solution.push(start);
        visited[start] = true;

        for _ in 1..n {
            let current = *solution.last().unwrap_or(&0);
            let next = self.select_next_node(problem, pheromone, current, &visited, n);
            solution.push(next);
            visited[next] = true;
        }

        solution
    }

    /// Select the next node using roulette wheel selection
    fn select_next_node<P: CombinatorialProblem>(
        &mut self,
        problem: &P,
        pheromone: &[Vec<f64>],
        current: usize,
        visited: &[bool],
        n: usize,
    ) -> usize {
        // Compute probabilities for unvisited nodes
        let mut probabilities = Vec::with_capacity(n);
        let mut total = 0.0;

        for j in 0..n {
            if visited[j] || !problem.is_feasible(current, j, visited) {
                probabilities.push(0.0);
                continue;
            }
            let tau = pheromone[current][j].max(1e-15);
            let eta = problem.heuristic(current, j).max(1e-15);
            let p = tau.powf(self.options.alpha) * eta.powf(self.options.beta);
            probabilities.push(p);
            total += p;
        }

        if total <= 0.0 {
            // Fallback: pick first unvisited node
            for j in 0..n {
                if !visited[j] {
                    return j;
                }
            }
            return 0; // shouldn't happen if problem is well-formed
        }

        // Roulette wheel
        let threshold = self.rng.random::<f64>() * total;
        let mut cumulative = 0.0;
        for j in 0..n {
            cumulative += probabilities[j];
            if cumulative >= threshold {
                return j;
            }
        }

        // Fallback
        for j in (0..n).rev() {
            if !visited[j] {
                return j;
            }
        }
        0
    }
}

// ---------------------------------------------------------------------------
// Max-Min Ant System (MMAS)
// ---------------------------------------------------------------------------

/// Options for Max-Min Ant System
#[derive(Debug, Clone)]
pub struct MaxMinAntSystemOptions {
    /// Base AS options
    pub base: AntSystemOptions,
    /// Minimum pheromone level (tau_min)
    pub tau_min: f64,
    /// Maximum pheromone level (tau_max)
    pub tau_max: f64,
    /// Use iteration-best (true) or global-best (false) for pheromone update
    pub use_iteration_best: bool,
    /// Smooth pheromone initialization
    pub smooth_init: bool,
}

impl Default for MaxMinAntSystemOptions {
    fn default() -> Self {
        Self {
            base: AntSystemOptions::default(),
            tau_min: 0.01,
            tau_max: 10.0,
            use_iteration_best: true,
            smooth_init: true,
        }
    }
}

/// Max-Min Ant System (MMAS) optimizer
///
/// Extends Ant System with bounded pheromone levels to prevent premature convergence.
pub struct MaxMinAntSystem {
    options: MaxMinAntSystemOptions,
    rng: StdRng,
}

impl MaxMinAntSystem {
    /// Create a new MMAS optimizer
    pub fn new(options: MaxMinAntSystemOptions) -> Self {
        let seed = options.base.seed.unwrap_or_else(|| rng().random());
        Self {
            options,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Solve a combinatorial problem using MMAS
    pub fn solve<P: CombinatorialProblem>(&mut self, problem: &P) -> OptimizeResult<AcoResult> {
        let n = problem.num_nodes();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Problem has 0 nodes".to_string(),
            ));
        }

        let init_val = if self.options.smooth_init {
            self.options.tau_max
        } else {
            (self.options.tau_min + self.options.tau_max) / 2.0
        };

        let mut pheromone = vec![vec![init_val; n]; n];
        let mut best_solution: Vec<usize> = Vec::new();
        let mut best_cost = f64::INFINITY;
        let mut nfev: usize = 0;

        // Extract values from self.options to avoid borrowing self immutably
        let max_iter = self.options.base.max_iter;
        let num_ants = self.options.base.num_ants;
        let evaporation_rate = self.options.base.evaporation_rate;
        let q_factor = self.options.base.q_factor;
        let patience = self.options.base.patience;
        let tau_min = self.options.tau_min;
        let tau_max = self.options.tau_max;
        let use_iteration_best = self.options.use_iteration_best;

        let mut cost_history = Vec::with_capacity(max_iter);
        let mut no_improve_count: usize = 0;

        for iteration in 0..max_iter {
            let mut iter_best_solution: Vec<usize> = Vec::new();
            let mut iter_best_cost = f64::INFINITY;

            // Each ant constructs a solution
            for _ in 0..num_ants {
                let solution = self.construct_solution_mmas(problem, &pheromone, n);
                let cost = problem.evaluate(&solution);
                nfev += 1;

                if cost < iter_best_cost {
                    iter_best_cost = cost;
                    iter_best_solution = solution.clone();
                }

                if cost < best_cost {
                    best_cost = cost;
                    best_solution = solution;
                    no_improve_count = 0;
                }
            }

            cost_history.push(best_cost);

            // Evaporate pheromone
            for row in &mut pheromone {
                for val in row.iter_mut() {
                    *val *= 1.0 - evaporation_rate;
                }
            }

            // Deposit pheromone from best ant only
            let (update_sol, update_cost) = if use_iteration_best {
                (&iter_best_solution, iter_best_cost)
            } else {
                (&best_solution, best_cost)
            };

            if update_cost > 0.0 && !update_sol.is_empty() {
                let deposit = q_factor / update_cost;
                for i in 0..update_sol.len() {
                    let from = update_sol[i];
                    let to = update_sol[(i + 1) % update_sol.len()];
                    pheromone[from][to] += deposit;
                    pheromone[to][from] += deposit;
                }
            }

            // Clamp pheromone to [tau_min, tau_max]
            for row in &mut pheromone {
                for val in row.iter_mut() {
                    *val = val.clamp(tau_min, tau_max);
                }
            }

            no_improve_count += 1;
            if no_improve_count >= patience {
                return Ok(AcoResult {
                    best_solution,
                    best_cost,
                    iterations: iteration + 1,
                    nfev,
                    converged: true,
                    cost_history,
                    message: format!("MMAS converged after {} iterations", iteration + 1),
                });
            }
        }

        Ok(AcoResult {
            best_solution,
            best_cost,
            iterations: max_iter,
            nfev,
            converged: false,
            cost_history,
            message: format!("MMAS completed {} iterations", max_iter),
        })
    }

    /// Construct a solution for MMAS (same logic as AS but with clamped pheromone)
    fn construct_solution_mmas<P: CombinatorialProblem>(
        &mut self,
        problem: &P,
        pheromone: &[Vec<f64>],
        n: usize,
    ) -> Vec<usize> {
        let mut visited = vec![false; n];
        let mut solution = Vec::with_capacity(n);

        let start = self.rng.random_range(0..n);
        solution.push(start);
        visited[start] = true;

        for _ in 1..n {
            let current = *solution.last().unwrap_or(&0);
            let next = self.select_next_mmas(problem, pheromone, current, &visited, n);
            solution.push(next);
            visited[next] = true;
        }

        solution
    }

    /// Select next node for MMAS
    fn select_next_mmas<P: CombinatorialProblem>(
        &mut self,
        problem: &P,
        pheromone: &[Vec<f64>],
        current: usize,
        visited: &[bool],
        n: usize,
    ) -> usize {
        let alpha = self.options.base.alpha;
        let beta = self.options.base.beta;
        let mut probabilities = Vec::with_capacity(n);
        let mut total = 0.0;

        for j in 0..n {
            if visited[j] || !problem.is_feasible(current, j, visited) {
                probabilities.push(0.0);
                continue;
            }
            let tau = pheromone[current][j].max(1e-15);
            let eta = problem.heuristic(current, j).max(1e-15);
            let p = tau.powf(alpha) * eta.powf(beta);
            probabilities.push(p);
            total += p;
        }

        if total <= 0.0 {
            for j in 0..n {
                if !visited[j] {
                    return j;
                }
            }
            return 0;
        }

        let threshold = self.rng.random::<f64>() * total;
        let mut cumulative = 0.0;
        for j in 0..n {
            cumulative += probabilities[j];
            if cumulative >= threshold {
                return j;
            }
        }

        for j in (0..n).rev() {
            if !visited[j] {
                return j;
            }
        }
        0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_tsp() -> TspProblem {
        // 4-city TSP with known optimal tour
        // Cities at: (0,0), (1,0), (1,1), (0,1)  => square, optimal tour = 4.0
        TspProblem::from_coordinates(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
            .expect("valid TSP")
    }

    fn make_line_tsp() -> TspProblem {
        // 5 cities in a line: 0,1,2,3,4
        // Optimal tour: 0-1-2-3-4-0 = 1+1+1+1+4 = 8
        TspProblem::from_coordinates(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
            .expect("valid TSP")
    }

    // --- TSP problem tests ---

    #[test]
    fn test_tsp_from_coordinates() {
        let tsp = make_simple_tsp();
        assert_eq!(tsp.num_cities(), 4);
        assert!((tsp.distance(0, 1) - 1.0).abs() < 1e-10);
        let diag = (2.0_f64).sqrt();
        assert!((tsp.distance(0, 2) - diag).abs() < 1e-10);
    }

    #[test]
    fn test_tsp_evaluate() {
        let tsp = make_simple_tsp();
        // Tour 0->1->2->3->0 = perimeter of unit square = 4
        let cost = tsp.evaluate(&[0, 1, 2, 3]);
        assert!((cost - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_tsp_heuristic() {
        let tsp = make_simple_tsp();
        // heuristic = 1/distance
        let h = tsp.heuristic(0, 1);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tsp_empty_error() {
        let result = TspProblem::from_coordinates(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tsp_non_square_error() {
        let result = TspProblem::new(vec![vec![0.0, 1.0], vec![1.0]]);
        assert!(result.is_err());
    }

    // --- Ant System tests ---

    #[test]
    fn test_as_simple_tsp() {
        let tsp = make_simple_tsp();
        let opts = AntSystemOptions {
            num_ants: 10,
            max_iter: 100,
            alpha: 1.0,
            beta: 3.0,
            evaporation_rate: 0.3,
            q_factor: 100.0,
            seed: Some(42),
            patience: 50,
            ..Default::default()
        };

        let mut aco = AntColonyOptimizer::new(opts);
        let result = aco.solve(&tsp).expect("AS should solve simple TSP");

        // Optimal tour is 4.0 for unit square
        assert!(
            result.best_cost <= 4.0 + 1.0,
            "AS should find near-optimal: got {}",
            result.best_cost
        );
        assert_eq!(result.best_solution.len(), 4);
        assert!(result.nfev > 0);
    }

    #[test]
    fn test_as_line_tsp() {
        let tsp = make_line_tsp();
        let opts = AntSystemOptions {
            num_ants: 15,
            max_iter: 150,
            alpha: 1.0,
            beta: 5.0,
            evaporation_rate: 0.3,
            q_factor: 100.0,
            seed: Some(99),
            patience: 80,
            ..Default::default()
        };

        let mut aco = AntColonyOptimizer::new(opts);
        let result = aco.solve(&tsp).expect("AS should solve line TSP");

        assert_eq!(result.best_solution.len(), 5);
        // Optimal for 5 points in a line: tour = 8
        assert!(
            result.best_cost <= 9.0,
            "Line TSP best cost: {}",
            result.best_cost
        );
    }

    #[test]
    fn test_as_convergence_history() {
        let tsp = make_simple_tsp();
        let opts = AntSystemOptions {
            num_ants: 10,
            max_iter: 50,
            seed: Some(42),
            patience: 100, // don't stop early
            ..Default::default()
        };

        let mut aco = AntColonyOptimizer::new(opts);
        let result = aco.solve(&tsp).expect("AS should produce history");

        assert!(!result.cost_history.is_empty());
        // Cost history should be non-increasing (best cost tracked)
        for i in 1..result.cost_history.len() {
            assert!(
                result.cost_history[i] <= result.cost_history[i - 1] + 1e-10,
                "Cost history should be non-increasing"
            );
        }
    }

    #[test]
    fn test_as_zero_nodes_error() {
        let tsp = TspProblem::new(vec![]).unwrap_or_else(|_| {
            // Create a minimal valid TSP to pass creation, then test with 0 nodes via trait
            TspProblem::new(vec![vec![0.0]]).expect("valid")
        });
        // Actually test with an empty problem
        struct EmptyProblem;
        impl CombinatorialProblem for EmptyProblem {
            fn num_nodes(&self) -> usize {
                0
            }
            fn heuristic(&self, _i: usize, _j: usize) -> f64 {
                0.0
            }
            fn evaluate(&self, _solution: &[usize]) -> f64 {
                0.0
            }
        }

        let opts = AntSystemOptions {
            seed: Some(1),
            ..Default::default()
        };
        let mut aco = AntColonyOptimizer::new(opts);
        let result = aco.solve(&EmptyProblem);
        assert!(result.is_err());
    }

    // --- MMAS tests ---

    #[test]
    fn test_mmas_simple_tsp() {
        let tsp = make_simple_tsp();
        let opts = MaxMinAntSystemOptions {
            base: AntSystemOptions {
                num_ants: 10,
                max_iter: 100,
                alpha: 1.0,
                beta: 3.0,
                evaporation_rate: 0.3,
                q_factor: 100.0,
                seed: Some(42),
                patience: 60,
                ..Default::default()
            },
            tau_min: 0.01,
            tau_max: 10.0,
            use_iteration_best: true,
            smooth_init: true,
        };

        let mut mmas = MaxMinAntSystem::new(opts);
        let result = mmas.solve(&tsp).expect("MMAS should solve simple TSP");

        assert!(
            result.best_cost <= 4.0 + 1.0,
            "MMAS should find near-optimal: got {}",
            result.best_cost
        );
        assert_eq!(result.best_solution.len(), 4);
    }

    #[test]
    fn test_mmas_global_best_update() {
        let tsp = make_simple_tsp();
        let opts = MaxMinAntSystemOptions {
            base: AntSystemOptions {
                num_ants: 10,
                max_iter: 80,
                seed: Some(55),
                patience: 60,
                ..Default::default()
            },
            use_iteration_best: false, // global-best update
            ..Default::default()
        };

        let mut mmas = MaxMinAntSystem::new(opts);
        let result = mmas.solve(&tsp).expect("MMAS global-best should work");
        assert!(result.best_cost < f64::INFINITY);
    }

    #[test]
    fn test_mmas_non_smooth_init() {
        let tsp = make_simple_tsp();
        let opts = MaxMinAntSystemOptions {
            base: AntSystemOptions {
                num_ants: 10,
                max_iter: 60,
                seed: Some(77),
                patience: 50,
                ..Default::default()
            },
            smooth_init: false,
            ..Default::default()
        };

        let mut mmas = MaxMinAntSystem::new(opts);
        let result = mmas.solve(&tsp).expect("MMAS non-smooth init should work");
        assert!(result.best_cost < f64::INFINITY);
    }

    // --- Permutation problem tests ---

    #[test]
    fn test_permutation_problem() {
        // Simple assignment problem: minimize sum of position * value
        let eval_fn = |perm: &[usize]| -> f64 {
            perm.iter()
                .enumerate()
                .map(|(i, &v)| (i as f64) * (v as f64))
                .sum()
        };

        let problem =
            PermutationProblem::with_uniform_heuristic(5, eval_fn).expect("valid problem");

        let opts = AntSystemOptions {
            num_ants: 10,
            max_iter: 100,
            seed: Some(42),
            patience: 50,
            ..Default::default()
        };

        let mut aco = AntColonyOptimizer::new(opts);
        let result = aco
            .solve(&problem)
            .expect("AS should solve permutation problem");

        assert_eq!(result.best_solution.len(), 5);
        // Check that it's a valid permutation
        let mut sorted = result.best_solution.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_permutation_problem_with_heuristic() {
        let n = 4;
        let heuristic = vec![
            vec![0.0, 1.0, 0.5, 0.2],
            vec![1.0, 0.0, 0.8, 0.3],
            vec![0.5, 0.8, 0.0, 1.0],
            vec![0.2, 0.3, 1.0, 0.0],
        ];

        let eval_fn = |perm: &[usize]| -> f64 {
            let mut cost = 0.0;
            for i in 0..perm.len() - 1 {
                cost += (perm[i] as f64 - perm[i + 1] as f64).abs();
            }
            cost
        };

        let problem = PermutationProblem::new(n, heuristic, eval_fn).expect("valid problem");

        let opts = AntSystemOptions {
            num_ants: 10,
            max_iter: 50,
            seed: Some(42),
            ..Default::default()
        };

        let mut aco = AntColonyOptimizer::new(opts);
        let result = aco.solve(&problem).expect("should solve");
        assert_eq!(result.best_solution.len(), 4);
    }

    #[test]
    fn test_permutation_empty_error() {
        let result = PermutationProblem::with_uniform_heuristic(0, |_: &[usize]| 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_permutation_heuristic_size_mismatch() {
        let result = PermutationProblem::new(3, vec![vec![1.0; 3]; 2], |_: &[usize]| 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_mmas_patience_convergence() {
        let tsp = make_simple_tsp();
        let opts = MaxMinAntSystemOptions {
            base: AntSystemOptions {
                num_ants: 10,
                max_iter: 1000,
                seed: Some(42),
                patience: 10, // very short patience
                ..Default::default()
            },
            ..Default::default()
        };

        let mut mmas = MaxMinAntSystem::new(opts);
        let result = mmas.solve(&tsp).expect("should converge via patience");

        // Should stop well before 1000 iterations
        assert!(result.iterations < 1000);
        assert!(result.converged);
    }
}
