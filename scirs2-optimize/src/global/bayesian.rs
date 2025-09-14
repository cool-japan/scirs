//! Bayesian optimization for global optimization of expensive-to-evaluate functions
//!
//! Bayesian optimization uses a surrogate model (usually Gaussian Process) to
//! model the underlying objective function, and acquisition functions to determine
//! the next points to evaluate, balancing exploration and exploitation.
//!
//! # Example
//!
//! ```
//! use ndarray::{array, ArrayView1};
//! use scirs2_optimize::global::{bayesian_optimization, BayesianOptimizationOptions};
//!
//! // Define objective function (simple sphere)
//! fn objective(x: &ArrayView1<f64>) -> f64 {
//!     x[0].powi(2) + x[1].powi(2)
//! }
//!
//! // Define search space
//! let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];
//!
//! // Create optimizer options with more evaluations
//! let mut options = BayesianOptimizationOptions::default();
//! options.n_initial_points = 5;
//!
//! // Run optimization with more iterations
//! let result = bayesian_optimization(objective, bounds, 30, Some(options)).unwrap();
//!
//! // Check result - should find a good solution
//! assert!(result.success);
//! println!("Best value found: {}", result.fun);
//! # Ok::<(), scirs2_optimize::error::OptimizeError>(())
//! ```

use std::fmt;

use friedrich::gaussian_process::GaussianProcess;
use friedrich::kernel::SquaredExp;
use friedrich::prior::ConstantPrior;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::error::OptimizeError;
use crate::parallel::ParallelOptions;
use crate::unconstrained::{minimize, Method, OptimizeResult, Options};

/// Parameter types for search space
#[derive(Debug, Clone)]
pub enum Parameter {
    /// Continuous parameter with lower and upper bounds
    Real(f64, f64),
    /// Integer parameter with lower and upper bounds
    Integer(i64, i64),
    /// Categorical parameter with possible values
    Categorical(Vec<String>),
}

/// Search space for parameters
#[derive(Debug, Clone)]
pub struct Space {
    /// Parameters with names
    parameters: Vec<(String, Parameter)>,
    /// Dimensionality after transformation
    transformed_n_dims: usize,
    /// Bounds for parameters
    bounds: Vec<(f64, f64)>,
}

impl Space {
    /// Create a new search space
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
            transformed_n_dims: 0,
            bounds: Vec::new(),
        }
    }

    /// Add a parameter to the search space
    pub fn add<S: Into<String>>(mut self, name: S, parameter: Parameter) -> Self {
        let name = name.into();

        // Update transformed dimensionality
        self.transformed_n_dims += match &parameter {
            Parameter::Real(_, _) => 1,
            Parameter::Integer(_, _) => 1,
            Parameter::Categorical(values) => values.len(),
        };

        // Add bounds for this parameter
        let bounds = match &parameter {
            Parameter::Real(lower, upper) => (*lower, *upper),
            Parameter::Integer(lower, upper) => (*lower as f64, *upper as f64),
            Parameter::Categorical(_) => (0.0, 1.0),
        };
        self.bounds.push(bounds);

        self.parameters.push((name, parameter));
        self
    }

    /// Get number of dimensions in the original space
    pub fn n_dims(&self) -> usize {
        self.parameters.len()
    }

    /// Get number of dimensions in the transformed space
    pub fn transformed_n_dims(&self) -> usize {
        self.transformed_n_dims
    }

    /// Sample random points from the space
    pub fn sample(&self, n_samples: usize, rng: &mut StdRng) -> Vec<Array1<f64>> {
        let n_dims = self.n_dims();
        let mut samples = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let mut sample = Array1::zeros(n_dims);

            for (i, (_, param)) in self.parameters.iter().enumerate() {
                match param {
                    Parameter::Real(lower, upper) => {
                        // Use gen_range directly instead of Uniform distribution
                        sample[i] = rng.gen_range(*lower..*upper);
                    }
                    Parameter::Integer(lower, upper) => {
                        let range = rng.gen_range(*lower..=*upper);
                        sample[i] = range as f64;
                    }
                    Parameter::Categorical(values) => {
                        let index = rng.gen_range(0..values.len());
                        sample[i] = index as f64;
                    }
                }
            }

            samples.push(sample);
        }

        samples
    }

    /// Transform a point from the original space to the model space
    pub fn transform(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        let mut transformed = Array1::zeros(self.transformed_n_dims);
        let mut idx = 0;

        for (i, (_, param)) in self.parameters.iter().enumerate() {
            match param {
                Parameter::Real(lower, upper) => {
                    // Scale to [0, 1]
                    transformed[idx] = (x[i] - lower) / (upper - lower);
                    idx += 1;
                }
                Parameter::Integer(lower, upper) => {
                    // Scale to [0, 1]
                    transformed[idx] = (x[i] - *lower as f64) / (*upper as f64 - *lower as f64);
                    idx += 1;
                }
                Parameter::Categorical(values) => {
                    // One-hot encoding
                    let index = x[i] as usize;
                    for j in 0..values.len() {
                        transformed[idx + j] = if j == index { 1.0 } else { 0.0 };
                    }
                    idx += values.len();
                }
            }
        }

        transformed
    }

    /// Transform a point from the model space back to the original space
    pub fn inverse_transform(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        let mut inverse = Array1::zeros(self.n_dims());
        let mut idx = 0;

        for (i, (_, param)) in self.parameters.iter().enumerate() {
            match param {
                Parameter::Real(lower, upper) => {
                    // Scale from [0, 1] to [lower, upper]
                    inverse[i] = lower + x[idx] * (upper - lower);
                    idx += 1;
                }
                Parameter::Integer(lower, upper) => {
                    // Scale from [0, 1] to [lower, upper] and round
                    let value = lower + (x[idx] * (*upper - *lower) as f64).round() as i64;
                    inverse[i] = value as f64;
                    idx += 1;
                }
                Parameter::Categorical(values) => {
                    // Find index of maximum value in one-hot encoding
                    let mut max_idx = 0;
                    let mut max_val = x[idx];
                    for j in 1..values.len() {
                        if x[idx + j] > max_val {
                            max_val = x[idx + j];
                            max_idx = j;
                        }
                    }
                    inverse[i] = max_idx as f64;
                    idx += values.len();
                }
            }
        }

        inverse
    }

    /// Convert bounds to format used by the optimizer
    pub fn bounds_to_vec(&self) -> Vec<(f64, f64)> {
        self.parameters
            .iter()
            .map(|(_, param)| match param {
                Parameter::Real(lower, upper) => (*lower, *upper),
                Parameter::Integer(lower, upper) => (*lower as f64, *upper as f64),
                Parameter::Categorical(_) => (0.0, 1.0), // Will be handled specially
            })
            .collect()
    }
}

impl Default for Space {
    fn default() -> Self {
        Self::new()
    }
}

/// Acquisition function trait for Bayesian optimization
pub trait AcquisitionFunction: Send + Sync {
    /// Evaluate acquisition function at a point
    fn evaluate(&self, x: &ArrayView1<f64>) -> f64;

    /// Compute gradient of acquisition function (if available)
    fn gradient(&self, x: &ArrayView1<f64>) -> Option<Array1<f64>> {
        None
    }
}

/// Expected Improvement acquisition function
pub struct ExpectedImprovement {
    model: GaussianProcess<SquaredExp, ConstantPrior>,
    y_best: f64,
    xi: f64,
}

impl ExpectedImprovement {
    /// Create a new Expected Improvement acquisition function
    pub fn new(model: GaussianProcess<SquaredExp, ConstantPrior>, y_best: f64, xi: f64) -> Self {
        Self { model, y_best, xi }
    }
}

impl AcquisitionFunction for ExpectedImprovement {
    fn evaluate(&self, x: &ArrayView1<f64>) -> f64 {
        let mean = self.model.predict(&x.to_vec());
        let std = (self.model.predict_variance(&x.to_vec())).sqrt();

        if std <= 0.0 {
            return 0.0;
        }

        let z = (self.y_best - mean - self.xi) / std;
        // Use approximation for normal CDF and PDF since erf is unstable
        let norm_cdf = 0.5 * (1.0 + approx_erf(z * std::f64::consts::SQRT_2 / 2.0));
        let norm_pdf = (-0.5 * z.powi(2)).exp() / (2.0 * std::f64::consts::PI).sqrt();

        let ei = (self.y_best - mean - self.xi) * norm_cdf + std * norm_pdf;

        if ei < 0.0 {
            0.0
        } else {
            ei
        }
    }

    fn gradient(&self, x: &ArrayView1<f64>) -> Option<Array1<f64>> {
        // For now, use numerical approximation
        None
    }
}

/// Lower Confidence Bound acquisition function
pub struct LowerConfidenceBound {
    model: GaussianProcess<SquaredExp, ConstantPrior>,
    kappa: f64,
}

impl LowerConfidenceBound {
    /// Create a new Lower Confidence Bound acquisition function
    pub fn new(model: GaussianProcess<SquaredExp, ConstantPrior>, kappa: f64) -> Self {
        Self { model, kappa }
    }
}

impl AcquisitionFunction for LowerConfidenceBound {
    fn evaluate(&self, x: &ArrayView1<f64>) -> f64 {
        let mean = self.model.predict(&x.to_vec());
        let std = (self.model.predict_variance(&x.to_vec())).sqrt();

        mean - self.kappa * std
    }

    fn gradient(&self, x: &ArrayView1<f64>) -> Option<Array1<f64>> {
        // For now, use numerical approximation
        None
    }
}

/// Probability of Improvement acquisition function
pub struct ProbabilityOfImprovement {
    model: GaussianProcess<SquaredExp, ConstantPrior>,
    y_best: f64,
    xi: f64,
}

impl ProbabilityOfImprovement {
    /// Create a new Probability of Improvement acquisition function
    pub fn new(model: GaussianProcess<SquaredExp, ConstantPrior>, y_best: f64, xi: f64) -> Self {
        Self { model, y_best, xi }
    }
}

impl AcquisitionFunction for ProbabilityOfImprovement {
    fn evaluate(&self, x: &ArrayView1<f64>) -> f64 {
        let mean = self.model.predict(&x.to_vec());
        let std = (self.model.predict_variance(&x.to_vec())).sqrt();

        if std <= 0.0 {
            return 0.0;
        }

        let z = (self.y_best - mean - self.xi) / std;
        // Use approximation for normal CDF since erf is unstable
        0.5 * (1.0 + approx_erf(z * std::f64::consts::SQRT_2 / 2.0))
    }

    fn gradient(&self, x: &ArrayView1<f64>) -> Option<Array1<f64>> {
        // For now, use numerical approximation
        None
    }
}

// Approximation of the error function (erf)
// Abramowitz and Stegun formula 7.1.26
#[allow(dead_code)]
fn approx_erf(x: f64) -> f64 {
    // Constants
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    // Save the sign of x
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // A&S formula 7.1.26
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Acquisition function type enum for option selection
#[derive(Debug, Clone, Copy, Default)]
pub enum AcquisitionFunctionType {
    /// Expected Improvement (default)
    #[default]
    ExpectedImprovement,
    /// Lower Confidence Bound
    LowerConfidenceBound,
    /// Probability of Improvement
    ProbabilityOfImprovement,
}

impl fmt::Display for AcquisitionFunctionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AcquisitionFunctionType::ExpectedImprovement => write!(f, "EI"),
            AcquisitionFunctionType::LowerConfidenceBound => write!(f, "LCB"),
            AcquisitionFunctionType::ProbabilityOfImprovement => write!(f, "PI"),
        }
    }
}

/// Kernel type enum for option selection
#[derive(Debug, Clone, Copy, Default)]
pub enum KernelType {
    /// Squared Exponential (default)
    #[default]
    SquaredExponential,
    /// Matérn 5/2
    Matern52,
    /// Matérn 3/2
    Matern32,
}

impl fmt::Display for KernelType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KernelType::SquaredExponential => write!(f, "SquaredExponential"),
            KernelType::Matern52 => write!(f, "Matern52"),
            KernelType::Matern32 => write!(f, "Matern32"),
        }
    }
}

/// Initial point generator type
#[derive(Debug, Clone, Copy, Default)]
pub enum InitialPointGenerator {
    /// Random sampling (default)
    #[default]
    Random,
    /// Latin Hypercube Sampling
    LatinHypercube,
    /// Sobol sequence
    Sobol,
    /// Halton sequence
    Halton,
}

impl fmt::Display for InitialPointGenerator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InitialPointGenerator::Random => write!(f, "Random"),
            InitialPointGenerator::LatinHypercube => write!(f, "LatinHypercube"),
            InitialPointGenerator::Sobol => write!(f, "Sobol"),
            InitialPointGenerator::Halton => write!(f, "Halton"),
        }
    }
}

/// Options for Bayesian optimization
#[derive(Clone, Debug)]
pub struct BayesianOptimizationOptions {
    /// Number of initial points
    pub n_initial_points: usize,
    /// Initial point generator
    pub initial_point_generator: InitialPointGenerator,
    /// Acquisition function type
    pub acq_func: AcquisitionFunctionType,
    /// Kernel type for Gaussian Process
    pub kernel: KernelType,
    /// Exploration-exploitation trade-off for LCB
    pub kappa: f64,
    /// Exploration parameter for EI and PI
    pub xi: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Parallel computation options
    pub parallel: Option<ParallelOptions>,
    /// Number of restarts for acquisition optimization
    pub n_restarts: usize,
}

impl Default for BayesianOptimizationOptions {
    fn default() -> Self {
        Self {
            n_initial_points: 10,
            initial_point_generator: InitialPointGenerator::default(),
            acq_func: AcquisitionFunctionType::default(),
            kernel: KernelType::default(),
            kappa: 1.96,
            xi: 0.01,
            seed: None,
            parallel: None,
            n_restarts: 5,
        }
    }
}

/// Observation with input and output
#[derive(Debug, Clone)]
struct Observation {
    /// Input point
    x: Array1<f64>,
    /// Function value
    y: f64,
}

/// The Bayesian optimization algorithm
pub struct BayesianOptimizer {
    /// Search space
    space: Space,
    /// Optimization options
    options: BayesianOptimizationOptions,
    /// Observations
    observations: Vec<Observation>,
    /// Best observation so far
    best_observation: Option<Observation>,
    /// Random number generator
    rng: StdRng,
    /// Current iteration number
    iteration: usize,
}

impl BayesianOptimizer {
    /// Create a new Bayesian optimizer
    pub fn new(space: Space, options: Option<BayesianOptimizationOptions>) -> Self {
        let options = options.unwrap_or_default();
        let seed = options.seed.unwrap_or_else(|| rand::rng().random());
        let rng = StdRng::seed_from_u64(seed);

        Self {
            space,
            options,
            observations: Vec::new(),
            best_observation: None,
            rng,
            iteration: 0,
        }
    }

    /// Ask for the next point to evaluate
    pub fn ask(&mut self) -> Array1<f64> {
        // If we don't have enough points, sample randomly
        if self.observations.len() < self.options.n_initial_points {
            let samples = match self.options.initial_point_generator {
                InitialPointGenerator::Random => {
                    // Simple random sampling
                    self.space.sample(1, &mut self.rng)
                }
                InitialPointGenerator::LatinHypercube => {
                    // Implement Latin Hypercube Sampling
                    let dim = self.space.bounds.len();
                    let n_samples = 1;

                    // Create intervals for each dimension
                    let mut sample = Array1::zeros(dim);

                    for (i, (low, high)) in self.space.bounds.iter().enumerate() {
                        // For single sample, pick random position in interval
                        let interval_size = (high - low) / n_samples as f64;
                        let offset = self.rng.gen_range(0.0..1.0) * interval_size;
                        sample[i] = low + offset;
                    }

                    vec![sample]
                }
                InitialPointGenerator::Sobol => {
                    // Implement basic Sobol sequence
                    let dim = self.space.bounds.len();
                    let mut sample = Array1::zeros(dim);

                    // Use iteration count as seed for Sobol sequence
                    let seed = self.iteration as u32 + 1;

                    for (i, (low, high)) in self.space.bounds.iter().enumerate() {
                        // Simple digital scrambling based on Van der Corput sequence
                        let mut n = seed;
                        let mut denom = 1.0;
                        let mut result = 0.0;
                        let base = 2u32;

                        while n > 0 {
                            denom *= base as f64;
                            result += (n % base) as f64 / denom;
                            n /= base;
                        }

                        // Scale to bounds with dimension-specific offset
                        let offset = ((i + 1) as f64 * 0.5).fract();
                        let value = (result + offset).fract();
                        sample[i] = low + value * (high - low);
                    }

                    vec![sample]
                }
                InitialPointGenerator::Halton => {
                    // Implement Halton sequence
                    let dim = self.space.bounds.len();
                    let mut sample = Array1::zeros(dim);

                    // First few prime numbers for Halton bases
                    let primes = [2u32, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
                    let seed = self.iteration as u32 + 1;

                    for (i, (low, high)) in self.space.bounds.iter().enumerate() {
                        // Use different prime base for each dimension
                        let base = if i < primes.len() {
                            primes[i]
                        } else {
                            primes[i % primes.len()]
                        };

                        // Compute Halton value using radical inverse
                        let mut n = seed;
                        let mut denom = 1.0;
                        let mut result = 0.0;

                        while n > 0 {
                            denom *= base as f64;
                            result += (n % base) as f64 / denom;
                            n /= base;
                        }

                        // Scale to bounds
                        sample[i] = low + result * (high - low);
                    }

                    vec![sample]
                }
            };

            self.iteration += 1;
            return samples[0].clone();
        }

        // Otherwise, optimize the acquisition function
        self.iteration += 1;
        self.optimize_acquisition_function()
    }

    /// Update with an observation
    pub fn tell(&mut self, x: Array1<f64>, y: f64) {
        let observation = Observation { x, y };

        // Update best observation
        if let Some(best) = &self.best_observation {
            if y < best.y {
                self.best_observation = Some(observation.clone());
            }
        } else {
            self.best_observation = Some(observation.clone());
        }

        // Add to observations
        self.observations.push(observation);
    }

    /// Build a Gaussian Process model from observations
    fn build_model(&self) -> GaussianProcess<SquaredExp, ConstantPrior> {
        // Prepare data
        let mut x_data = Vec::with_capacity(self.observations.len());
        let mut y_data = Vec::with_capacity(self.observations.len());

        for obs in &self.observations {
            let x_transformed = self.space.transform(&obs.x.view()).to_vec();
            x_data.push(x_transformed);
            y_data.push(obs.y);
        }

        // Build model using default settings (which uses squared exponential kernel)
        // In a future implementation we could expose more kernel options
        GaussianProcess::default(x_data, y_data)
    }

    /// Create an acquisition function
    fn create_acquisition_function(&self) -> Box<dyn AcquisitionFunction> {
        let model = self.build_model();
        let y_best = self.best_observation.as_ref().unwrap().y;

        match self.options.acq_func {
            AcquisitionFunctionType::ExpectedImprovement => {
                Box::new(ExpectedImprovement::new(model, y_best, self.options.xi))
            }
            AcquisitionFunctionType::LowerConfidenceBound => {
                Box::new(LowerConfidenceBound::new(model, self.options.kappa))
            }
            AcquisitionFunctionType::ProbabilityOfImprovement => Box::new(
                ProbabilityOfImprovement::new(model, y_best, self.options.xi),
            ),
        }
    }

    /// Optimize the acquisition function
    fn optimize_acquisition_function(&mut self) -> Array1<f64> {
        let acq_func = self.create_acquisition_function();
        let bounds = self.space.bounds_to_vec();
        let n_restarts = self.options.n_restarts;

        // We want to minimize the negative acquisition function
        let f = |x: &ArrayView1<f64>| -acq_func.evaluate(x);

        // Starting points for optimization
        let mut x_starts = self.space.sample(n_restarts, &mut self.rng);

        // Add the current best point as one of the starting points
        if let Some(best) = &self.best_observation {
            if !x_starts.is_empty() {
                x_starts[0] = best.x.clone();
            } else {
                x_starts.push(best.x.clone());
            }
        }

        // Optimize from each starting point
        let mut best_x = None;
        let mut best_value = f64::INFINITY;

        for x_start in x_starts {
            let result = minimize(
                f,
                &x_start.to_vec(),
                Method::LBFGS,
                Some(Options {
                    bounds: Some(
                        crate::unconstrained::Bounds::from_vecs(
                            bounds.iter().map(|b| Some(b.0)).collect(),
                            bounds.iter().map(|b| Some(b.1)).collect(),
                        )
                        .unwrap(),
                    ),
                    ..Default::default()
                }),
            );

            if let Ok(res) = result {
                if res.fun < best_value {
                    best_value = res.fun;
                    best_x = Some(res.x);
                }
            }
        }

        // Return the best point found
        best_x.unwrap_or_else(|| {
            // If optimization fails, return a random point
            self.space.sample(1, &mut self.rng)[0].clone()
        })
    }

    /// Run the full optimization process
    pub fn optimize<F>(&mut self, func: F, n_calls: usize) -> OptimizeResult<f64>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let mut n_calls_remaining = n_calls;

        // Initial random sampling
        let n_initial = self.options.n_initial_points.min(n_calls);
        let initial_points = self.space.sample(n_initial, &mut self.rng);

        for point in initial_points {
            let value = func(&point.view());
            self.tell(point, value);
            n_calls_remaining -= 1;

            if n_calls_remaining == 0 {
                break;
            }
        }

        // Main optimization loop
        let mut iterations = 0;

        while n_calls_remaining > 0 {
            // Get next point
            let next_point = self.ask();

            // Evaluate function
            let value = func(&next_point.view());

            // Update model
            self.tell(next_point, value);

            // Update counters
            n_calls_remaining -= 1;
            iterations += 1;
        }

        // Return final result
        let best = self.best_observation.as_ref().unwrap();
        OptimizeResult {
            x: best.x.clone(),
            fun: best.y,
            nfev: self.observations.len(),
            func_evals: self.observations.len(),
            nit: iterations,
            success: true,
            message: "Optimization terminated successfully".to_string(),
            ..Default::default()
        }
    }
}

/// Perform Bayesian optimization on a function
#[allow(dead_code)]
pub fn bayesian_optimization<F>(
    func: F,
    bounds: Vec<(f64, f64)>,
    n_calls: usize,
    options: Option<BayesianOptimizationOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    // Create space from bounds
    let space = bounds
        .into_iter()
        .enumerate()
        .fold(Space::new(), |space, (i, (lower, upper))| {
            space.add(format!("x{}", i), Parameter::Real(lower, upper))
        });

    // Create optimizer
    let mut optimizer = BayesianOptimizer::new(space, options);

    // Run optimization
    let result = optimizer.optimize(func, n_calls);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_space_creation() {
        let space = Space::new()
            .add("x1", Parameter::Real(-5.0, 5.0))
            .add("x2", Parameter::Integer(0, 10))
            .add(
                "x3",
                Parameter::Categorical(vec!["a".into(), "b".into(), "c".into()]),
            );

        assert_eq!(space.n_dims(), 3);
        assert_eq!(space.transformed_n_dims(), 5); // 1 (real) + 1 (integer) + 3 (categorical)
    }

    #[test]
    fn test_space_transform() {
        let space = Space::new()
            .add("x1", Parameter::Real(-5.0, 5.0))
            .add("x2", Parameter::Integer(0, 10));

        let x = array![0.0, 5.0];
        let transformed = space.transform(&x.view());

        assert_eq!(transformed.len(), 2);
        assert!((transformed[0] - 0.5).abs() < 1e-6); // (0.0 - (-5.0)) / (5.0 - (-5.0)) = 0.5
        assert!((transformed[1] - 0.5).abs() < 1e-6); // (5.0 - 0.0) / (10.0 - 0.0) = 0.5
    }

    #[test]
    fn test_bayesian_optimization() {
        // Simple quadratic function
        let f = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let options = BayesianOptimizationOptions {
            n_initial_points: 5,
            seed: Some(42),
            ..Default::default()
        };

        let result = bayesian_optimization(f, bounds, 15, Some(options)).unwrap();

        // Should find minimum near (0, 0)
        assert!(result.fun < 0.5);
        assert!(result.x[0].abs() < 0.5);
        assert!(result.x[1].abs() < 0.5);
    }
}
