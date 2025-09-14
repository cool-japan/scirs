//! Bayesian optimization for neural architecture search
//!
//! This module implements Bayesian optimization approaches for efficiently exploring
//! the architecture search space using Gaussian processes and acquisition functions.

#![allow(dead_code)]

use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::HashMap;

use crate::error::{OptimError, Result};

/// Bayesian optimization configuration
#[derive(Debug, Clone)]
pub struct BayesianOptConfig<T: Float> {
    /// Number of initial random samples
    pub initial_samples: usize,
    
    /// Total number of optimization iterations
    pub num_iterations: usize,
    
    /// Acquisition function type
    pub acquisition_function: AcquisitionFunction,
    
    /// Gaussian process kernel type
    pub kernel_type: KernelType,
    
    /// Kernel hyperparameters
    pub kernel_params: HashMap<String, T>,
    
    /// Noise variance for observations
    pub noise_variance: T,
    
    /// Exploration-exploitation trade-off parameter
    pub exploration_weight: T,
    
    /// Maximum architecture complexity allowed
    pub max_complexity: usize,
    
    /// Convergence tolerance
    pub convergence_tolerance: T,
    
    /// Number of acquisition function optimization attempts
    pub num_acquisition_restarts: usize,
}

/// Bayesian optimizer for architecture search
#[derive(Debug)]
pub struct BayesianArchitectureOptimizer<T: Float> {
    /// Configuration
    config: BayesianOptConfig<T>,
    
    /// Gaussian process model
    gaussian_process: GaussianProcess<T>,
    
    /// Architecture encoder
    architecture_encoder: ArchitectureEncoder<T>,
    
    /// Acquisition function optimizer
    acquisition_optimizer: AcquisitionOptimizer<T>,
    
    /// Search history
    search_history: SearchHistory<T>,
    
    /// Current iteration
    current_iteration: usize,
    
    /// Best architecture found so far
    best_architecture: Option<(ArchitectureRepresentation<T>, T)>,
}

/// Gaussian process model
#[derive(Debug)]
pub struct GaussianProcess<T: Float> {
    /// Training inputs (architecture representations)
    training_inputs: Array2<T>,
    
    /// Training outputs (performance values)
    training_outputs: Array1<T>,
    
    /// Kernel function
    kernel: Kernel<T>,
    
    /// Kernel hyperparameters
    hyperparameters: HashMap<String, T>,
    
    /// Covariance matrix (K + noise*I)
    covariance_matrix: Option<Array2<T>>,
    
    /// Inverse covariance matrix
    inverse_covariance: Option<Array2<T>>,
    
    /// Number of training points
    num_points: usize,
}

/// Kernel functions for Gaussian process
#[derive(Debug)]
pub enum Kernel<T: Float> {
    /// Radial basis function (RBF) kernel
    RBF { length_scale: T },
    
    /// Matérn kernel with parameter ν=1/2
    Matern12 { length_scale: T },
    
    /// Matérn kernel with parameter ν=3/2
    Matern32 { length_scale: T },
    
    /// Matérn kernel with parameter ν=5/2
    Matern52 { length_scale: T },
    
    /// Combined kernel (sum or product of kernels)
    Combined { kernels: Vec<Kernel<T>>, operation: CombinationOp },
}

/// Kernel combination operations
#[derive(Debug, Clone, Copy)]
pub enum CombinationOp {
    Sum,
    Product,
}

/// Kernel types
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Combined,
}

/// Acquisition functions
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunction {
    /// Expected improvement
    ExpectedImprovement,
    
    /// Upper confidence bound
    UpperConfidenceBound,
    
    /// Probability of improvement
    ProbabilityOfImprovement,
    
    /// Entropy search
    EntropySearch,
    
    /// Knowledge gradient
    KnowledgeGradient,
}

/// Architecture encoder for converting architectures to vectors
#[derive(Debug)]
pub struct ArchitectureEncoder<T: Float> {
    /// Encoding dimension
    encoding_dim: usize,
    
    /// Layer type encodings
    layer_encodings: HashMap<String, Array1<T>>,
    
    /// Connection encodings
    connection_encodings: HashMap<String, Array1<T>>,
    
    /// Parameter normalization factors
    param_normalizers: HashMap<String, (T, T)>, // (mean, std)
}

/// Architecture representation as encoded vector
#[derive(Debug, Clone)]
pub struct ArchitectureRepresentation<T: Float> {
    /// Feature vector representation
    pub features: Array1<T>,
    
    /// Architecture complexity measure
    pub complexity: T,
    
    /// Metadata about the architecture
    pub metadata: ArchitectureMetadata,
}

/// Architecture metadata
#[derive(Debug, Clone)]
pub struct ArchitectureMetadata {
    /// Number of layers
    pub num_layers: usize,
    
    /// Total parameters
    pub num_parameters: usize,
    
    /// Architecture type description
    pub architecture_type: String,
    
    /// Layer types present
    pub layer_types: Vec<String>,
}

/// Acquisition function optimizer
#[derive(Debug)]
pub struct AcquisitionOptimizer<T: Float> {
    /// Optimization bounds for each dimension
    bounds: Vec<(T, T)>,
    
    /// Number of random restarts
    num_restarts: usize,
    
    /// Optimization tolerance
    tolerance: T,
    
    /// Maximum iterations per restart
    max_iterations: usize,
}

/// Search history tracking
#[derive(Debug)]
pub struct SearchHistory<T: Float> {
    /// Evaluated architectures and their performances
    evaluations: Vec<(ArchitectureRepresentation<T>, T)>,
    
    /// GP prediction accuracies over time
    prediction_errors: Vec<T>,
    
    /// Acquisition function values over time
    acquisition_values: Vec<T>,
    
    /// Best performance over time
    best_performance_history: Vec<T>,
}

impl<T: Float + Default + Clone> BayesianArchitectureOptimizer<T> {
    /// Create new Bayesian architecture optimizer
    pub fn new(config: BayesianOptConfig<T>) -> Result<Self> {
        let gaussian_process = GaussianProcess::new(config.kernel_type, &config.kernel_params)?;
        let architecture_encoder = ArchitectureEncoder::new(128)?; // 128-dimensional encoding
        let acquisition_optimizer = AcquisitionOptimizer::new(
            vec![(T::zero(), T::one()); 128], // Unit hypercube bounds
            config.num_acquisition_restarts,
        )?;
        let search_history = SearchHistory::new();

        Ok(Self {
            config,
            gaussian_process,
            architecture_encoder,
            acquisition_optimizer,
            search_history,
            current_iteration: 0,
            best_architecture: None,
        })
    }

    /// Run Bayesian optimization for architecture search
    pub fn optimize<F>(&mut self, objective_fn: F) -> Result<(ArchitectureRepresentation<T>, T)>
    where
        F: Fn(&ArchitectureRepresentation<T>) -> Result<T>,
    {
        // Phase 1: Initial random sampling
        self.initial_random_sampling(&objective_fn)?;

        // Phase 2: Bayesian optimization loop
        for iteration in 0..self.config.num_iterations {
            self.current_iteration = iteration;

            // Fit Gaussian process to current data
            self.fit_gaussian_process()?;

            // Optimize acquisition function to find next candidate
            let next_candidate = self.optimize_acquisition_function()?;

            // Evaluate the candidate
            let performance = objective_fn(&next_candidate)?;

            // Update search history
            self.update_search_history(next_candidate.clone(), performance)?;

            // Update best architecture if improved
            if let Some((_, best_perf)) = &self.best_architecture {
                if performance > *best_perf {
                    self.best_architecture = Some((next_candidate, performance));
                }
            } else {
                self.best_architecture = Some((next_candidate, performance));
            }

            // Check convergence
            if self.check_convergence()? {
                println!("Bayesian optimization converged at iteration {}", iteration);
                break;
            }

            // Print progress
            if iteration % 10 == 0 {
                println!("BO Iteration {}: Best performance = {:.4}", 
                    iteration, 
                    self.best_architecture.as_ref().unwrap().1.to_f64().unwrap_or(0.0)
                );
            }
        }

        self.best_architecture.clone().ok_or_else(|| 
            OptimError::SearchFailed("No architecture found".to_string())
        )
    }

    /// Initial random sampling phase
    fn initial_random_sampling<F>(&mut self, objective_fn: &F) -> Result<()>
    where
        F: Fn(&ArchitectureRepresentation<T>) -> Result<T>,
    {
        for _ in 0..self.config.initial_samples {
            let candidate = self.generate_random_architecture()?;
            let performance = objective_fn(&candidate)?;
            self.update_search_history(candidate, performance)?;
        }
        Ok(())
    }

    /// Generate random architecture for initial sampling
    fn generate_random_architecture(&self) -> Result<ArchitectureRepresentation<T>> {
        use rand::Rng;
        let mut rng = rand::rng();

        // Generate random feature vector
        let features = Array1::from_shape_fn(self.architecture_encoder.encoding_dim, |_| {
            T::from(rng.random::<f64>()).unwrap()
        });

        let complexity = T::from(rng.gen_range(0.1..1.0)).unwrap();

        let metadata = ArchitectureMetadata {
            num_layers: rng.gen_range(2..10),
            num_parameters: rng.gen_range(1000..100000),
            architecture_type: "random".to_string(),
            layer_types: vec!["dense".to_string(), "lstm".to_string()],
        };

        Ok(ArchitectureRepresentation {
            features,
            complexity,
            metadata,
        })
    }

    /// Fit Gaussian process to current search history
    fn fit_gaussian_process(&mut self) -> Result<()> {
        let evaluations = &self.search_history.evaluations;
        if evaluations.is_empty() {
            return Err(OptimError::InsufficientData("No evaluations to fit GP".to_string()));
        }

        let num_points = evaluations.len();
        let feature_dim = evaluations[0].0.features.len();

        // Prepare training data
        let mut training_inputs = Array2::zeros((num_points, feature_dim));
        let mut training_outputs = Array1::zeros(num_points);

        for (i, (arch, performance)) in evaluations.iter().enumerate() {
            training_inputs.row_mut(i).assign(&arch.features);
            training_outputs[i] = *performance;
        }

        // Update GP with new data
        self.gaussian_process.fit(training_inputs, training_outputs)?;

        Ok(())
    }

    /// Optimize acquisition function to find next candidate
    fn optimize_acquisition_function(&mut self) -> Result<ArchitectureRepresentation<T>> {
        let mut best_acquisition_value = T::from(f64::NEG_INFINITY).unwrap();
        let mut best_candidate = None;

        // Multiple random restarts
        for _ in 0..self.config.num_acquisition_restarts {
            let candidate = self.optimize_acquisition_single_restart()?;
            let acquisition_value = self.evaluate_acquisition_function(&candidate)?;

            if acquisition_value > best_acquisition_value {
                best_acquisition_value = acquisition_value;
                best_candidate = Some(candidate);
            }
        }

        best_candidate.ok_or_else(|| 
            OptimError::SearchFailed("Failed to optimize acquisition function".to_string())
        )
    }

    /// Single restart of acquisition function optimization
    fn optimize_acquisition_single_restart(&self) -> Result<ArchitectureRepresentation<T>> {
        use rand::Rng;
        let mut rng = rand::rng();

        // Start with random point
        let mut current_features = Array1::from_shape_fn(
            self.architecture_encoder.encoding_dim,
            |_| T::from(rng.random::<f64>()).unwrap()
        );

        let step_size = T::from(0.01).unwrap();
        let max_iterations = 100;

        // Simple gradient ascent (simplified)
        for _ in 0..max_iterations {
            let gradient = self.approximate_acquisition_gradient(&current_features)?;
            
            // Update with gradient ascent
            for i in 0..current_features.len() {
                current_features[i] = current_features[i] + step_size * gradient[i];
                // Project to bounds [0, 1]
                current_features[i] = current_features[i].max(T::zero()).min(T::one());
            }
        }

        // Convert to architecture representation
        let complexity = current_features.iter().cloned().fold(T::zero(), |acc, x| acc + x) / 
            T::from(current_features.len() as f64).unwrap();

        let metadata = ArchitectureMetadata {
            num_layers: 5, // Simplified
            num_parameters: 50000,
            architecture_type: "optimized".to_string(),
            layer_types: vec!["dense".to_string(), "attention".to_string()],
        };

        Ok(ArchitectureRepresentation {
            features: current_features,
            complexity,
            metadata,
        })
    }

    /// Approximate gradient of acquisition function using finite differences
    fn approximate_acquisition_gradient(&self, features: &Array1<T>) -> Result<Array1<T>> {
        let epsilon = T::from(1e-6).unwrap();
        let mut gradient = Array1::zeros(features.len());

        for i in 0..features.len() {
            let mut features_plus = features.clone();
            let mut features_minus = features.clone();
            
            features_plus[i] = features_plus[i] + epsilon;
            features_minus[i] = features_minus[i] - epsilon;

            let arch_plus = ArchitectureRepresentation {
                features: features_plus,
                complexity: T::from(0.5).unwrap(),
                metadata: ArchitectureMetadata {
                    num_layers: 5,
                    num_parameters: 50000,
                    architecture_type: "gradient".to_string(),
                    layer_types: vec!["dense".to_string()],
                },
            };

            let arch_minus = ArchitectureRepresentation {
                features: features_minus,
                complexity: T::from(0.5).unwrap(),
                metadata: ArchitectureMetadata {
                    num_layers: 5,
                    num_parameters: 50000,
                    architecture_type: "gradient".to_string(),
                    layer_types: vec!["dense".to_string()],
                },
            };

            let acq_plus = self.evaluate_acquisition_function(&arch_plus)?;
            let acq_minus = self.evaluate_acquisition_function(&arch_minus)?;

            gradient[i] = (acq_plus - acq_minus) / (T::from(2.0).unwrap() * epsilon);
        }

        Ok(gradient)
    }

    /// Evaluate acquisition function at given architecture
    fn evaluate_acquisition_function(&self, architecture: &ArchitectureRepresentation<T>) -> Result<T> {
        // Get GP prediction
        let (mean, variance) = self.gaussian_process.predict(&architecture.features)?;

        match self.config.acquisition_function {
            AcquisitionFunction::ExpectedImprovement => {
                self.expected_improvement(mean, variance)
            }
            AcquisitionFunction::UpperConfidenceBound => {
                self.upper_confidence_bound(mean, variance)
            }
            AcquisitionFunction::ProbabilityOfImprovement => {
                self.probability_of_improvement(mean, variance)
            }
            _ => {
                // Fallback to UCB
                self.upper_confidence_bound(mean, variance)
            }
        }
    }

    /// Expected improvement acquisition function
    fn expected_improvement(&self, mean: T, variance: T) -> Result<T> {
        if let Some((_, best_value)) = &self.best_architecture {
            let std_dev = variance.sqrt();
            if std_dev <= T::zero() {
                return Ok(T::zero());
            }

            let z = (mean - *best_value) / std_dev;
            let phi_z = self.standard_normal_pdf(z);
            let big_phi_z = self.standard_normal_cdf(z);

            let ei = (mean - *best_value) * big_phi_z + std_dev * phi_z;
            Ok(ei)
        } else {
            Ok(variance.sqrt()) // If no best value, use uncertainty
        }
    }

    /// Upper confidence bound acquisition function
    fn upper_confidence_bound(&self, mean: T, variance: T) -> Result<T> {
        let exploration_weight = self.config.exploration_weight;
        let ucb = mean + exploration_weight * variance.sqrt();
        Ok(ucb)
    }

    /// Probability of improvement acquisition function
    fn probability_of_improvement(&self, mean: T, variance: T) -> Result<T> {
        if let Some((_, best_value)) = &self.best_architecture {
            let std_dev = variance.sqrt();
            if std_dev <= T::zero() {
                return Ok(T::zero());
            }

            let z = (mean - *best_value) / std_dev;
            let poi = self.standard_normal_cdf(z);
            Ok(poi)
        } else {
            Ok(T::from(0.5).unwrap())
        }
    }

    /// Standard normal PDF (simplified approximation)
    fn standard_normal_pdf(&self, x: T) -> T {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let coefficient = T::one() / (T::from(2.0).unwrap() * pi).sqrt();
        coefficient * (-x * x / T::from(2.0).unwrap()).exp()
    }

    /// Standard normal CDF (simplified approximation)
    fn standard_normal_cdf(&self, x: T) -> T {
        // Using error function approximation
        T::from(0.5).unwrap() * (T::one() + self.erf(x / T::from(2.0_f64.sqrt()).unwrap()))
    }

    /// Error function approximation
    fn erf(&self, x: T) -> T {
        // Abramowitz and Stegun approximation
        let a1 = T::from(0.254829592).unwrap();
        let a2 = T::from(-0.284496736).unwrap();
        let a3 = T::from(1.421413741).unwrap();
        let a4 = T::from(-1.453152027).unwrap();
        let a5 = T::from(1.061405429).unwrap();
        let p = T::from(0.3275911).unwrap();

        let sign = if x >= T::zero() { T::one() } else { -T::one() };
        let x_abs = x.abs();

        let t = T::one() / (T::one() + p * x_abs);
        let y = T::one() - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp();

        sign * y
    }

    /// Update search history with new evaluation
    fn update_search_history(&mut self, architecture: ArchitectureRepresentation<T>, performance: T) -> Result<()> {
        self.search_history.evaluations.push((architecture, performance));
        
        // Update best performance history
        let current_best = if let Some((_, best_perf)) = &self.best_architecture {
            performance.max(*best_perf)
        } else {
            performance
        };
        self.search_history.best_performance_history.push(current_best);

        Ok(())
    }

    /// Check convergence criteria
    fn check_convergence(&self) -> Result<bool> {
        let window_size = 10;
        let history = &self.search_history.best_performance_history;
        
        if history.len() < window_size {
            return Ok(false);
        }

        // Check if improvement in last window is below threshold
        let recent_best = history[history.len() - 1];
        let previous_best = history[history.len() - window_size];
        
        let improvement = recent_best - previous_best;
        Ok(improvement < self.config.convergence_tolerance)
    }

    /// Get search statistics
    pub fn get_search_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("total_evaluations".to_string(), self.search_history.evaluations.len() as f64);
        stats.insert("current_iteration".to_string(), self.current_iteration as f64);
        
        if let Some((_, best_perf)) = &self.best_architecture {
            stats.insert("best_performance".to_string(), best_perf.to_f64().unwrap_or(0.0));
        }

        if !self.search_history.best_performance_history.is_empty() {
            let initial_best = self.search_history.best_performance_history[0];
            let current_best = *self.search_history.best_performance_history.last().unwrap();
            let improvement = current_best - initial_best;
            stats.insert("total_improvement".to_string(), improvement.to_f64().unwrap_or(0.0));
        }

        stats
    }
}

impl<T: Float + Default + Clone> GaussianProcess<T> {
    fn new(kernel_type: KernelType, kernel_params: &HashMap<String, T>) -> Result<Self> {
        let kernel = match kernel_type {
            KernelType::RBF => {
                let length_scale = kernel_params.get("length_scale").copied().unwrap_or(T::one());
                Kernel::RBF { length_scale }
            }
            KernelType::Matern32 => {
                let length_scale = kernel_params.get("length_scale").copied().unwrap_or(T::one());
                Kernel::Matern32 { length_scale }
            }
            _ => Kernel::RBF { length_scale: T::one() }
        };

        Ok(Self {
            training_inputs: Array2::zeros((0, 0)),
            training_outputs: Array1::zeros(0),
            kernel,
            hyperparameters: kernel_params.clone(),
            covariance_matrix: None,
            inverse_covariance: None,
            num_points: 0,
        })
    }

    fn fit(&mut self, inputs: Array2<T>, outputs: Array1<T>) -> Result<()> {
        self.training_inputs = inputs;
        self.training_outputs = outputs;
        self.num_points = outputs.len();

        // Compute covariance matrix
        self.compute_covariance_matrix()?;

        Ok(())
    }

    fn predict(&self, test_input: &Array1<T>) -> Result<(T, T)> {
        if self.num_points == 0 {
            return Ok((T::zero(), T::one()));
        }

        // Compute kernel vector between test point and training points
        let mut k_star = Array1::zeros(self.num_points);
        for i in 0..self.num_points {
            let train_point = self.training_inputs.row(i);
            k_star[i] = self.kernel_function(test_input, &train_point.to_owned())?;
        }

        // Mean prediction
        let mean = if let Some(ref inv_cov) = self.inverse_covariance {
            let alpha = inv_cov.dot(&self.training_outputs);
            k_star.dot(&alpha)
        } else {
            T::zero()
        };

        // Variance prediction
        let k_star_star = self.kernel_function(test_input, test_input)?;
        let variance = if let Some(ref inv_cov) = self.inverse_covariance {
            let v = inv_cov.dot(&k_star);
            k_star_star - k_star.dot(&v)
        } else {
            k_star_star
        };

        Ok((mean, variance.max(T::from(1e-6).unwrap())))
    }

    fn compute_covariance_matrix(&mut self) -> Result<()> {
        let n = self.num_points;
        let mut cov_matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let xi = self.training_inputs.row(i).to_owned();
                let xj = self.training_inputs.row(j).to_owned();
                cov_matrix[[i, j]] = self.kernel_function(&xi, &xj)?;
            }
        }

        // Add noise to diagonal
        let noise_var = T::from(1e-6).unwrap(); // Small noise for numerical stability
        for i in 0..n {
            cov_matrix[[i, i]] = cov_matrix[[i, i]] + noise_var;
        }

        self.covariance_matrix = Some(cov_matrix.clone());

        // Compute inverse (simplified - in practice would use Cholesky decomposition)
        self.inverse_covariance = Some(self.pseudo_inverse(&cov_matrix)?);

        Ok(())
    }

    fn kernel_function(&self, x1: &Array1<T>, x2: &Array1<T>) -> Result<T> {
        match &self.kernel {
            Kernel::RBF { length_scale } => {
                let diff = x1 - x2;
                let squared_distance = diff.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x);
                Ok((-squared_distance / (T::from(2.0).unwrap() * *length_scale * *length_scale)).exp())
            }
            Kernel::Matern32 { length_scale } => {
                let diff = x1 - x2;
                let distance = diff.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x).sqrt();
                let scaled_distance = distance * T::from(3.0_f64.sqrt()).unwrap() / *length_scale;
                Ok((T::one() + scaled_distance) * (-scaled_distance).exp())
            }
            _ => {
                // Default to RBF
                let diff = x1 - x2;
                let squared_distance = diff.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x);
                Ok((-squared_distance / T::from(2.0).unwrap()).exp())
            }
        }
    }

    fn pseudo_inverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        // Simplified pseudo-inverse (in practice would use proper linear algebra)
        let n = matrix.nrows();
        let mut result = Array2::eye(n);
        
        // Add small regularization to diagonal
        for i in 0..n {
            result[[i, i]] = T::one() / (matrix[[i, i]] + T::from(1e-6).unwrap());
        }
        
        Ok(result)
    }
}

impl<T: Float + Default + Clone> ArchitectureEncoder<T> {
    fn new(encoding_dim: usize) -> Result<Self> {
        let mut layer_encodings = HashMap::new();
        let mut connection_encodings = HashMap::new();
        let mut param_normalizers = HashMap::new();

        // Initialize some default encodings
        layer_encodings.insert("dense".to_string(), Array1::ones(encoding_dim / 4));
        layer_encodings.insert("lstm".to_string(), Array1::zeros(encoding_dim / 4));
        layer_encodings.insert("attention".to_string(), Array1::from_elem(encoding_dim / 4, T::from(0.5).unwrap()));

        connection_encodings.insert("sequential".to_string(), Array1::ones(encoding_dim / 4));
        connection_encodings.insert("skip".to_string(), Array1::zeros(encoding_dim / 4));

        param_normalizers.insert("num_parameters".to_string(), (T::from(50000.0).unwrap(), T::from(25000.0).unwrap()));

        Ok(Self {
            encoding_dim,
            layer_encodings,
            connection_encodings,
            param_normalizers,
        })
    }
}

impl<T: Float + Default + Clone> AcquisitionOptimizer<T> {
    fn new(bounds: Vec<(T, T)>, num_restarts: usize) -> Result<Self> {
        Ok(Self {
            bounds,
            num_restarts,
            tolerance: T::from(1e-6).unwrap(),
            max_iterations: 100,
        })
    }
}

impl<T: Float + Default + Clone> SearchHistory<T> {
    fn new() -> Self {
        Self {
            evaluations: Vec::new(),
            prediction_errors: Vec::new(),
            acquisition_values: Vec::new(),
            best_performance_history: Vec::new(),
        }
    }
}

impl<T: Float + Default + Clone> Default for BayesianOptConfig<T> {
    fn default() -> Self {
        let mut kernel_params = HashMap::new();
        kernel_params.insert("length_scale".to_string(), T::one());

        Self {
            initial_samples: 10,
            num_iterations: 100,
            acquisition_function: AcquisitionFunction::ExpectedImprovement,
            kernel_type: KernelType::RBF,
            kernel_params,
            noise_variance: T::from(1e-6).unwrap(),
            exploration_weight: T::from(2.0).unwrap(),
            max_complexity: 1000000,
            convergence_tolerance: T::from(1e-4).unwrap(),
            num_acquisition_restarts: 10,
        }
    }
}