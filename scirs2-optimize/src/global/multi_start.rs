//! Multi-start strategies for global optimization
//!
//! These strategies run multiple optimization attempts from different
//! starting points to increase the chance of finding the global optimum.

use crate::error::OptimizeError;
use crate::unconstrained::{
    minimize, Bounds as UnconstrainedBounds, Method as UnconstrainedMethod, OptimizeResult, Options,
};
use ndarray::{Array1, ArrayView1};
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// Options for multi-start optimization
#[derive(Debug, Clone)]
pub struct MultiStartOptions {
    /// Number of starting points
    pub n_starts: usize,
    /// Local optimization method to use
    pub local_method: UnconstrainedMethod,
    /// Whether to use parallel execution
    pub parallel: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Strategy for generating starting points
    pub strategy: StartingPointStrategy,
}

impl Default for MultiStartOptions {
    fn default() -> Self {
        Self {
            n_starts: 10,
            local_method: UnconstrainedMethod::BFGS,
            parallel: true,
            seed: None,
            strategy: StartingPointStrategy::Random,
        }
    }
}

/// Strategy for generating starting points
#[derive(Debug, Clone)]
pub enum StartingPointStrategy {
    /// Random uniform sampling within bounds
    Random,
    /// Latin hypercube sampling
    LatinHypercube,
    /// Halton sequence
    Halton,
    /// Sobol sequence
    Sobol,
    /// Grid-based sampling
    Grid,
}

/// Bounds for variables
pub type Bounds = Vec<(f64, f64)>;

/// Multi-start optimization solver
pub struct MultiStart<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone + Send + Sync,
{
    func: F,
    bounds: Bounds,
    options: MultiStartOptions,
    ndim: usize,
    rng: StdRng,
}

impl<F> MultiStart<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone + Send + Sync,
{
    /// Create new multi-start solver
    pub fn new(func: F, bounds: Bounds, options: MultiStartOptions) -> Self {
        let ndim = bounds.len();
        let seed = options.seed.unwrap_or_else(rand::random);
        let rng = StdRng::seed_from_u64(seed);

        Self {
            func,
            bounds,
            options,
            ndim,
            rng,
        }
    }

    /// Generate starting points based on strategy
    fn generate_starting_points(&mut self) -> Vec<Array1<f64>> {
        match self.options.strategy {
            StartingPointStrategy::Random => self.generate_random_points(),
            StartingPointStrategy::LatinHypercube => self.generate_latin_hypercube_points(),
            StartingPointStrategy::Halton => self.generate_halton_points(),
            StartingPointStrategy::Sobol => self.generate_sobol_points(),
            StartingPointStrategy::Grid => self.generate_grid_points(),
        }
    }

    /// Generate random starting points
    fn generate_random_points(&mut self) -> Vec<Array1<f64>> {
        let mut points = Vec::with_capacity(self.options.n_starts);

        for _ in 0..self.options.n_starts {
            let mut point = Array1::zeros(self.ndim);
            for j in 0..self.ndim {
                let (lb, ub) = self.bounds[j];
                point[j] = self.rng.random_range(lb..ub);
            }
            points.push(point);
        }

        points
    }

    /// Generate Latin hypercube starting points
    fn generate_latin_hypercube_points(&mut self) -> Vec<Array1<f64>> {
        let mut points = Vec::with_capacity(self.options.n_starts);
        let n = self.options.n_starts;

        // Create segment indices for each dimension
        for i in 0..n {
            let mut point = Array1::zeros(self.ndim);

            for j in 0..self.ndim {
                let (lb, ub) = self.bounds[j];
                let segment_size = (ub - lb) / n as f64;

                // Random offset within segment
                let offset = self.rng.random::<f64>();
                point[j] = lb + (i as f64 + offset) * segment_size;
            }

            points.push(point);
        }

        // Shuffle each dimension independently
        for j in 0..self.ndim {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut self.rng);

            for (i, &idx) in indices.iter().enumerate() {
                let temp = points[i][j];
                points[i][j] = points[idx][j];
                points[idx][j] = temp;
            }
        }

        points
    }

    /// Generate Halton sequence points (simplified)
    fn generate_halton_points(&mut self) -> Vec<Array1<f64>> {
        // For now, fallback to random
        self.generate_random_points()
    }

    /// Generate Sobol sequence points (simplified)
    fn generate_sobol_points(&mut self) -> Vec<Array1<f64>> {
        // For now, fallback to random
        self.generate_random_points()
    }

    /// Generate grid-based starting points
    fn generate_grid_points(&mut self) -> Vec<Array1<f64>> {
        let points_per_dim = (self.options.n_starts as f64)
            .powf(1.0 / self.ndim as f64)
            .ceil() as usize;
        let mut points = Vec::new();

        // Generate grid points
        let mut current = vec![0usize; self.ndim];
        loop {
            let mut point = Array1::zeros(self.ndim);

            for j in 0..self.ndim {
                let (lb, ub) = self.bounds[j];
                let step = (ub - lb) / (points_per_dim - 1).max(1) as f64;
                point[j] = lb + current[j] as f64 * step;
            }

            points.push(point);

            // Increment grid position
            let mut carry = true;
            for j in current.iter_mut() {
                if carry {
                    *j += 1;
                    if *j >= points_per_dim {
                        *j = 0;
                    } else {
                        carry = false;
                    }
                }
            }

            if carry || points.len() >= self.options.n_starts {
                break;
            }
        }

        points.truncate(self.options.n_starts);
        points
    }

    /// Run optimization from a single starting point
    fn optimize_single(&self, x0: Array1<f64>) -> OptimizeResult<f64> {
        let bounds = Some(
            UnconstrainedBounds::from_vecs(
                self.bounds.iter().map(|&(lb, _)| Some(lb)).collect(),
                self.bounds.iter().map(|&(_, ub)| Some(ub)).collect(),
            )
            .unwrap(),
        );

        let options = Options {
            bounds,
            ..Default::default()
        };

        let func = self.func.clone();

        minimize(
            move |x: &ArrayView1<f64>| func(x),
            &x0.to_vec(),
            self.options.local_method,
            Some(options),
        )
        .unwrap_or_else(|_| {
            // Return a failed result if optimization fails
            OptimizeResult {
                x: x0,
                fun: f64::INFINITY,
                success: false,
                ..Default::default()
            }
        })
    }

    /// Run the multi-start optimization
    pub fn run(&mut self) -> OptimizeResult<f64> {
        let starting_points = self.generate_starting_points();

        let results = if self.options.parallel {
            // Parallel execution
            starting_points
                .into_par_iter()
                .map(|x0| self.optimize_single(x0))
                .collect::<Vec<_>>()
        } else {
            // Sequential execution
            starting_points
                .into_iter()
                .map(|x0| self.optimize_single(x0))
                .collect::<Vec<_>>()
        };

        // Find the best result
        let best_result = results
            .into_iter()
            .filter(|r| r.success)
            .min_by(|a, b| {
                a.fun
                    .partial_cmp(&b.fun)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|| OptimizeResult {
                x: Array1::zeros(self.ndim),
                fun: f64::INFINITY,
                success: false,
                message: "All optimization attempts failed".to_string(),
                ..Default::default()
            });

        OptimizeResult {
            x: best_result.x,
            fun: best_result.fun,
            nit: self.options.n_starts,
            iterations: self.options.n_starts,
            success: best_result.success,
            message: format!(
                "Multi-start optimization with {} starts",
                self.options.n_starts
            ),
            ..Default::default()
        }
    }
}

/// Perform multi-start optimization
pub fn multi_start<F>(
    func: F,
    bounds: Bounds,
    options: Option<MultiStartOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone + Send + Sync,
{
    let options = options.unwrap_or_default();
    let mut solver = MultiStart::new(func, bounds, options);
    Ok(solver.run())
}
