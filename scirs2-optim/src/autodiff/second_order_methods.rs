//! Second-order optimization methods with automatic differentiation
//!
//! This module implements advanced second-order optimization algorithms that
//! leverage automatic differentiation for computing Hessians and curvature information.

use ndarray::{Array, Array1, Array2, Axis};
use num_traits::Float;
use std::collections::VecDeque;

use crate::autodiff::{AutodiffEngine, HigherOrderGradients, HessianMatrix, HessianComputationStrategy, JacobianComputationStrategy};
#[allow(unused_imports)]
use crate::error::Result;
use crate::optimizers::{Optimizer, OptimizerState};

/// Configuration for second-order optimization methods
#[derive(Debug, Clone)]
pub struct SecondOrderConfig {
    /// Learning rate
    pub learning_rate: f64,
    
    /// Hessian computation strategy
    pub hessian_strategy: HessianComputationStrategy,
    
    /// Damping parameter for regularization
    pub damping: f64,
    
    /// Trust region radius
    pub trust_region_radius: f64,
    
    /// Maximum number of CG iterations for Hessian inversion
    pub max_cg_iterations: usize,
    
    /// CG tolerance
    pub cg_tolerance: f64,
    
    /// Enable line search
    pub enable_line_search: bool,
    
    /// Line search parameters
    pub line_search_config: LineSearchConfig,
    
    /// Memory size for quasi-Newton methods
    pub memory_size: usize,
    
    /// Update frequency for Hessian computation
    pub hessian_update_frequency: usize,
    
    /// Enable curvature scaling
    pub enable_curvature_scaling: bool}

/// Line search configuration
#[derive(Debug, Clone)]
pub struct LineSearchConfig {
    /// Armijo condition parameter (c1)
    pub c1: f64,
    
    /// Curvature condition parameter (c2)  
    pub c2: f64,
    
    /// Maximum line search iterations
    pub max_iterations: usize,
    
    /// Initial step size
    pub initial_step: f64,
    
    /// Step size reduction factor
    pub reduction_factor: f64,
    
    /// Line search method
    pub method: LineSearchMethod}

/// Line search methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineSearchMethod {
    /// Backtracking line search
    Backtracking,
    
    /// Wolfe conditions
    Wolfe,
    
    /// Strong Wolfe conditions
    StrongWolfe,
    
    /// More-Thuente line search
    MoreThuente,
    
    /// Exact line search (for quadratic functions)
    Exact}

/// Newton's method optimizer with automatic differentiation
pub struct NewtonOptimizer<T: Float> {
    /// Configuration
    config: SecondOrderConfig,
    
    /// Higher-order gradient computer
    hog: HigherOrderGradients<T>,
    
    /// Current iteration
    iteration: usize,
    
    /// Convergence history
    convergence_history: VecDeque<ConvergenceMetrics<T>>,
    
    /// Last computed Hessian
    last_hessian: Option<HessianMatrix<T>>,
    
    /// Trust region state
    trust_region_state: TrustRegionState<T>}

/// Quasi-Newton optimizer (BFGS/L-BFGS)
pub struct QuasiNewtonOptimizer<T: Float> {
    /// Configuration
    config: SecondOrderConfig,
    
    /// BFGS memory
    bfgs_memory: BFGSMemory<T>,
    
    /// Higher-order gradient computer
    hog: HigherOrderGradients<T>,
    
    /// Current iteration
    iteration: usize,
    
    /// Convergence history
    convergence_history: VecDeque<ConvergenceMetrics<T>>}

/// K-FAC (Kronecker-Factored Approximate Curvature) optimizer
pub struct KFACOptimizer<T: Float> {
    /// Configuration
    config: SecondOrderConfig,
    
    /// KFAC state
    kfac_state: KFACState<T>,
    
    /// Higher-order gradient computer
    hog: HigherOrderGradients<T>,
    
    /// Current iteration
    iteration: usize,
    
    /// Layer information for neural networks
    layer_info: Vec<LayerInfo>}

/// Natural gradient optimizer
pub struct NaturalGradientOptimizer<T: Float> {
    /// Configuration
    config: SecondOrderConfig,
    
    /// Fisher information matrix
    fisher_matrix: Option<Array2<T>>,
    
    /// Higher-order gradient computer
    hog: HigherOrderGradients<T>,
    
    /// Current iteration
    iteration: usize,
    
    /// Running average of Fisher information
    fisher_ema_decay: T}

/// Trust region state
#[derive(Debug, Clone)]
pub struct TrustRegionState<T: Float> {
    /// Current trust region radius
    pub radius: T,
    
    /// Trust region history
    pub radius_history: VecDeque<T>,
    
    /// Reduction ratio threshold for expanding radius
    pub expand_threshold: T,
    
    /// Reduction ratio threshold for shrinking radius
    pub shrink_threshold: T,
    
    /// Expansion factor
    pub expand_factor: T,
    
    /// Shrinking factor
    pub shrink_factor: T}

/// BFGS memory for quasi-Newton methods
#[derive(Debug, Clone)]
pub struct BFGSMemory<T: Float> {
    /// Memory size
    pub memory_size: usize,
    
    /// Gradient differences (y_k = g_{k+1} - g_k)
    pub y_history: VecDeque<Array1<T>>,
    
    /// Parameter differences (s_k = x_{k+1} - x_k)
    pub s_history: VecDeque<Array1<T>>,
    
    /// Scaling factors (rho_k = 1 / (y_k^T s_k))
    pub rho_history: VecDeque<T>,
    
    /// Initial Hessian approximation scale
    pub initial_scale: T}

/// KFAC state for neural network optimization
#[derive(Debug, Clone)]
pub struct KFACState<T: Float> {
    /// A matrices (input covariances)
    pub a_matrices: Vec<Array2<T>>,
    
    /// G matrices (gradient covariances)
    pub g_matrices: Vec<Array2<T>>,
    
    /// Inverse A matrices
    pub inv_a_matrices: Vec<Array2<T>>,
    
    /// Inverse G matrices
    pub inv_g_matrices: Vec<Array2<T>>,
    
    /// Damping parameter
    pub damping: T,
    
    /// Update frequency
    pub update_frequency: usize,
    
    /// EMA decay for covariance matrices
    pub ema_decay: T}

/// Layer information for KFAC
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer name
    pub name: String,
    
    /// Input dimension
    pub input_dim: usize,
    
    /// Output dimension
    pub output_dim: usize,
    
    /// Layer type
    pub layer_type: LayerType}

/// Types of neural network layers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    /// Fully connected layer
    FullyConnected,
    
    /// Convolutional layer
    Convolutional,
    
    /// Batch normalization layer
    BatchNorm,
    
    /// Activation layer
    Activation}

/// Convergence metrics for optimization
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics<T: Float> {
    /// Iteration number
    pub iteration: usize,
    
    /// Objective function value
    pub objective_value: T,
    
    /// Gradient norm
    pub gradient_norm: T,
    
    /// Parameter change norm
    pub parameter_change_norm: T,
    
    /// Hessian condition number
    pub condition_number: Option<T>,
    
    /// Trust region radius (if applicable)
    pub trust_region_radius: Option<T>,
    
    /// Line search step size
    pub step_size: T,
    
    /// Convergence status
    pub converged: bool}

impl<T: Float + Default + Clone> NewtonOptimizer<T> {
    /// Create new Newton optimizer
    pub fn new(
        config: SecondOrderConfig,
        autodiff_engine: AutodiffEngine<T>,
    ) -> Self {
        let hog = HigherOrderGradients::new(
            autodiff_engine,
            2, // Second-order
            config.hessian_strategy,
            JacobianComputationStrategy::Reverse,
        );
        
        let trust_region_state = TrustRegionState {
            radius: num_traits::cast::cast(config.trust_region_radius).unwrap_or_else(|| T::zero()),
            radius_history: VecDeque::new(),
            expand_threshold: num_traits::cast::cast(0.75).unwrap_or_else(|| T::zero()),
            shrink_threshold: num_traits::cast::cast(0.25).unwrap_or_else(|| T::zero()),
            expand_factor: num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()),
            shrink_factor: num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero())};
        
        Self {
            config,
            hog,
            iteration: 0,
            convergence_history: VecDeque::new(),
            last_hessian: None,
            trust_region_state}
    }
    
    /// Perform one optimization step
    pub fn step(
        &mut self,
        params: &Array1<T>,
        gradients: &Array1<T>,
        objective_fn: impl Fn(&Array1<T>) -> T,
    ) -> Result<Array1<T>> {
        self.iteration += 1;
        
        // Compute Hessian if needed
        let should_compute_hessian = self.iteration % self.config.hessian_update_frequency == 0 
            || self.last_hessian.is_none();
        
        if should_compute_hessian {
            let output_id = 0; // Simplified
            let input_ids: Vec<usize> = (0..params.len()).collect();
            
            let higher_order_results = self.hog.compute_gradients(output_id, &input_ids, 2)?;
            self.last_hessian = higher_order_results.hessian;
        }
        
        // Solve Newton system: H * p = -g
        let search_direction = self.solve_newton_system(gradients)?;
        
        // Apply trust region constraint
        let constrained_direction = self.apply_trust_region_constraint(&search_direction);
        
        // Line search or trust region step
        let step_size = if self.config.enable_line_search {
            self.line_search(params, &constrained_direction, &objective_fn)?
        } else {
            T::one()
        };
        
        let new_params = params + &(constrained_direction * step_size);
        
        // Update trust region radius
        self.update_trust_region_radius(params, &new_params, &objective_fn)?;
        
        // Record convergence metrics
        self.record_convergence_metrics(params, &new_params, gradients, &objective_fn);
        
        Ok(new_params)
    }
    
    /// Solve Newton system H * p = -g
    fn solve_newton_system(&mut self, gradients: &Array1<T>) -> Result<Array1<T>> {
        if let Some(ref hessian) = self.last_hessian {
            match &hessian.full_matrix {
                Some(h_matrix) => {
                    // Add damping for numerical stability
                    let damped_hessian = h_matrix + &(Array2::eye(h_matrix.nrows()) * num_traits::cast::cast(self.config.damping).unwrap_or_else(|| T::zero()));
                    
                    // Solve using conjugate gradient method
                    self.solve_cg(&damped_hessian, &(-gradients))
                }
                None => {
                    // Use diagonal approximation or gradient descent
                    if let Some(ref diagonal) = hessian.diagonal {
                        let inv_diag = diagonal.mapv(|x| T::one() / (x + num_traits::cast::cast(self.config.damping).unwrap_or_else(|| T::zero())));
                        Ok(&(-gradients) * &inv_diag)
                    } else {
                        // Fallback to gradient descent
                        Ok(-gradients * num_traits::cast::cast(self.config.learning_rate).unwrap_or_else(|| T::zero()))
                    }
                }
            }
        } else {
            // No Hessian available, use gradient descent
            Ok(-gradients * num_traits::cast::cast(self.config.learning_rate).unwrap_or_else(|| T::zero()))
        }
    }
    
    /// Solve linear system using conjugate gradient method
    fn solve_cg(&self, matrix: &Array2<T>, rhs: &Array1<T>) -> Result<Array1<T>> {
        let n = rhs.len();
        let mut x = Array1::zeros(n);
        let mut r = rhs.clone();
        let mut p = r.clone();
        let mut rsold = r.dot(&r);
        
        for _ in 0..self.config.max_cg_iterations {
            let ap = matrix.dot(&p);
            let alpha = rsold / p.dot(&ap);
            
            x = &x + &(&p * alpha);
            r = &r - &(&ap * alpha);
            
            let rsnew = r.dot(&r);
            
            if rsnew.sqrt() < num_traits::cast::cast(self.config.cg_tolerance).unwrap_or_else(|| T::zero()) {
                break;
            }
            
            let beta = rsnew / rsold;
            p = &r + &(&p * beta);
            rsold = rsnew;
        }
        
        Ok(x)
    }
    
    /// Apply trust region constraint to search direction
    fn apply_trust_region_constraint(&self, direction: &Array1<T>) -> Array1<T> {
        let direction_norm = direction.dot(direction).sqrt();
        
        if direction_norm <= self.trust_region_state.radius {
            direction.clone()
        } else {
            direction * (self.trust_region_state.radius / direction_norm)
        }
    }
    
    /// Perform line search
    fn line_search(
        &self,
        params: &Array1<T>,
        direction: &Array1<T>,
        objective_fn: &impl Fn(&Array1<T>) -> T,
    ) -> Result<T> {
        match self.config.line_search_config.method {
            LineSearchMethod::Backtracking => {
                self.backtracking_line_search(params, direction, objective_fn)
            }
            LineSearchMethod::Wolfe => {
                self.wolfe_line_search(params, direction, objective_fn)
            }
            _ => {
                // Fallback to fixed step size
                Ok(num_traits::cast::cast(self.config.line_search_config.initial_step).unwrap_or_else(|| T::zero()))
            }
        }
    }
    
    /// Backtracking line search
    fn backtracking_line_search(
        &self,
        params: &Array1<T>,
        direction: &Array1<T>,
        objective_fn: &impl Fn(&Array1<T>) -> T,
    ) -> Result<T> {
        let mut step_size = num_traits::cast::cast(self.config.line_search_config.initial_step).unwrap_or_else(|| T::zero());
        let c1 = num_traits::cast::cast(self.config.line_search_config.c1).unwrap_or_else(|| T::zero());
        let reduction_factor = num_traits::cast::cast(self.config.line_search_config.reduction_factor).unwrap_or_else(|| T::zero());
        
        let initial_value = objective_fn(params);
        let directional_derivative = T::zero(); // Would compute actual directional derivative
        
        for _ in 0..self.config.line_search_config.max_iterations {
            let new_params = params + &(direction * step_size);
            let new_value = objective_fn(&new_params);
            
            // Armijo condition
            if new_value <= initial_value + c1 * step_size * directional_derivative {
                return Ok(step_size);
            }
            
            step_size = step_size * reduction_factor;
        }
        
        Ok(step_size)
    }
    
    /// Wolfe line search  
    fn wolfe_line_search(
        &self,
        params: &Array1<T>,
        direction: &Array1<T>,
        objective_fn: &impl Fn(&Array1<T>) -> T,
    ) -> Result<T> {
        // Simplified Wolfe line search
        // In practice, would implement full strong Wolfe conditions
        self.backtracking_line_search(params, direction, objective_fn)
    }
    
    /// Update trust region radius based on reduction ratio
    fn update_trust_region_radius(
        &mut self,
        old_params: &Array1<T>,
        new_params: &Array1<T>,
        objective_fn: &impl Fn(&Array1<T>) -> T,
    ) -> Result<()> {
        let old_value = objective_fn(old_params);
        let new_value = objective_fn(new_params);
        let actual_reduction = old_value - new_value;
        
        // Compute predicted reduction using quadratic model
        let predicted_reduction = if let Some(ref hessian) = self.last_hessian {
            // Quadratic model: m(p) = f + g^T p + 0.5 p^T H p
            let delta = new_params - old_params;
            let grad_term = T::zero(); // Would use actual gradient
            let hess_term = if let Some(ref h_matrix) = hessian.full_matrix {
                num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()) * delta.dot(&h_matrix.dot(&delta))
            } else {
                T::zero()
            };
            grad_term + hess_term
        } else {
            T::one() // Fallback
        };
        
        let reduction_ratio = if predicted_reduction != T::zero() {
            actual_reduction / predicted_reduction
        } else {
            T::zero()
        };
        
        // Update radius based on reduction ratio
        if reduction_ratio > self.trust_region_state.expand_threshold {
            self.trust_region_state.radius = self.trust_region_state.radius * self.trust_region_state.expand_factor;
        } else if reduction_ratio < self.trust_region_state.shrink_threshold {
            self.trust_region_state.radius = self.trust_region_state.radius * self.trust_region_state.shrink_factor;
        }
        
        // Record radius history
        self.trust_region_state.radius_history.push_back(self.trust_region_state.radius);
        if self.trust_region_state.radius_history.len() > 100 {
            self.trust_region_state.radius_history.pop_front();
        }
        
        Ok(())
    }
    
    /// Record convergence metrics
    fn record_convergence_metrics(
        &mut self,
        old_params: &Array1<T>,
        new_params: &Array1<T>,
        gradients: &Array1<T>,
        objective_fn: &impl Fn(&Array1<T>) -> T,
    ) {
        let objective_value = objective_fn(new_params);
        let gradient_norm = gradients.dot(gradients).sqrt();
        let parameter_change_norm = (new_params - old_params).dot(&(new_params - old_params)).sqrt();
        
        let condition_number = if let Some(ref hessian) = self.last_hessian {
            hessian.condition_number
        } else {
            None
        };
        
        let converged = gradient_norm < num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero()) 
            && parameter_change_norm < num_traits::cast::cast(1e-8).unwrap_or_else(|| T::zero());
        
        let metrics = ConvergenceMetrics {
            iteration: self.iteration,
            objective_value,
            gradient_norm,
            parameter_change_norm,
            condition_number,
            trust_region_radius: Some(self.trust_region_state.radius),
            step_size: T::one(), // Would track actual step size
            converged};
        
        self.convergence_history.push_back(metrics);
        if self.convergence_history.len() > 1000 {
            self.convergence_history.pop_front();
        }
    }
    
    /// Check convergence
    pub fn has_converged(&self) -> bool {
        self.convergence_history.back()
            .map_or(false, |metrics| metrics.converged)
    }
    
    /// Get convergence history
    pub fn convergence_history(&self) -> &VecDeque<ConvergenceMetrics<T>> {
        &self.convergence_history
    }
}

impl<T: Float + Default + Clone> QuasiNewtonOptimizer<T> {
    /// Create new quasi-Newton optimizer
    pub fn new(
        config: SecondOrderConfig,
        autodiff_engine: AutodiffEngine<T>,
    ) -> Self {
        let hog = HigherOrderGradients::new(
            autodiff_engine,
            1, // First-order only
            HessianComputationStrategy::BFGS,
            JacobianComputationStrategy::Reverse,
        );
        
        let bfgs_memory = BFGSMemory {
            memory_size: config.memory_size,
            y_history: VecDeque::new(),
            s_history: VecDeque::new(),
            rho_history: VecDeque::new(),
            initial_scale: T::one()};
        
        Self {
            config,
            bfgs_memory,
            hog,
            iteration: 0,
            convergence_history: VecDeque::new()}
    }
    
    /// Perform BFGS update step
    pub fn step(
        &mut self,
        params: &Array1<T>,
        gradients: &Array1<T>,
        objective_fn: impl Fn(&Array1<T>) -> T,
    ) -> Result<Array1<T>> {
        self.iteration += 1;
        
        // Compute search direction using L-BFGS two-loop recursion
        let search_direction = self.compute_lbfgs_direction(gradients)?;
        
        // Line search
        let step_size = if self.config.enable_line_search {
            self.line_search(params, &search_direction, &objective_fn)?
        } else {
            num_traits::cast::cast(self.config.learning_rate).unwrap_or_else(|| T::zero())
        };
        
        let new_params = params + &(search_direction * step_size);
        
        // Update BFGS memory
        if self.iteration > 1 {
            let s = &new_params - params;
            // Note: would need previous gradient for y = grad_new - grad_old
            self.update_bfgs_memory(&s, gradients)?;
        }
        
        Ok(new_params)
    }
    
    /// Compute L-BFGS search direction using two-loop recursion
    fn compute_lbfgs_direction(&self, gradients: &Array1<T>) -> Result<Array1<T>> {
        if self.bfgs_memory.s_history.is_empty() {
            // No history, use steepest descent
            return Ok(-gradients * num_traits::cast::cast(self.config.learning_rate).unwrap_or_else(|| T::zero()));
        }
        
        let mut q = gradients.clone();
        let mut alphas = Vec::new();
        
        // First loop (backward)
        for i in (0..self.bfgs_memory.s_history.len()).rev() {
            let rho_i = self.bfgs_memory.rho_history[i];
            let s_i = &self.bfgs_memory.s_history[i];
            let alpha_i = rho_i * s_i.dot(&q);
            alphas.push(alpha_i);
            q = &q - &(&self.bfgs_memory.y_history[i] * alpha_i);
        }
        
        alphas.reverse();
        
        // Scale initial Hessian approximation
        let mut r = &q * self.bfgs_memory.initial_scale;
        
        // Second loop (forward)
        for i in 0..self.bfgs_memory.s_history.len() {
            let rho_i = self.bfgs_memory.rho_history[i];
            let y_i = &self.bfgs_memory.y_history[i];
            let s_i = &self.bfgs_memory.s_history[i];
            let beta = rho_i * y_i.dot(&r);
            r = &r + &(s_i * (alphas[i] - beta));
        }
        
        Ok(-r)
    }
    
    /// Update BFGS memory with new curvature information
    fn update_bfgs_memory(&mut self, s: &Array1<T>, y: &Array1<T>) -> Result<()> {
        let sy = s.dot(y);
        
        // Skip update if curvature condition is not satisfied
        if sy <= num_traits::cast::cast(1e-10).unwrap_or_else(|| T::zero()) {
            return Ok(());
        }
        
        let rho = T::one() / sy;
        
        // Add to memory
        self.bfgs_memory.s_history.push_back(s.clone());
        self.bfgs_memory.y_history.push_back(y.clone());
        self.bfgs_memory.rho_history.push_back(rho);
        
        // Limit memory size
        if self.bfgs_memory.s_history.len() > self.bfgs_memory.memory_size {
            self.bfgs_memory.s_history.pop_front();
            self.bfgs_memory.y_history.pop_front();
            self.bfgs_memory.rho_history.pop_front();
        }
        
        // Update initial scale (using last curvature pair)
        let yy = y.dot(y);
        if yy > T::zero() {
            self.bfgs_memory.initial_scale = sy / yy;
        }
        
        Ok(())
    }
    
    /// Line search implementation
    fn line_search(
        &self,
        params: &Array1<T>,
        direction: &Array1<T>,
        objective_fn: &impl Fn(&Array1<T>) -> T,
    ) -> Result<T> {
        // Simplified line search
        let mut step_size = T::one();
        let reduction_factor = num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero());
        
        let initial_value = objective_fn(params);
        
        for _ in 0..20 {
            let new_params = params + &(direction * step_size);
            let new_value = objective_fn(&new_params);
            
            if new_value < initial_value {
                return Ok(step_size);
            }
            
            step_size = step_size * reduction_factor;
        }
        
        Ok(step_size)
    }
}

impl Default for SecondOrderConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            hessian_strategy: HessianComputationStrategy::BFGS,
            damping: 1e-4,
            trust_region_radius: 1.0,
            max_cg_iterations: 100,
            cg_tolerance: 1e-8,
            enable_line_search: true,
            line_search_config: LineSearchConfig::default(),
            memory_size: 10,
            hessian_update_frequency: 1,
            enable_curvature_scaling: true}
    }
}

impl Default for LineSearchConfig {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            max_iterations: 20,
            initial_step: 1.0,
            reduction_factor: 0.5,
            method: LineSearchMethod::Backtracking}
    }
}

/// Utility functions for second-order methods
pub mod utils {
    use super::*;
    
    /// Compute condition number of Hessian
    pub fn compute_condition_number<T: Float>(hessian: &HessianMatrix<T>) -> Option<T> {
        if let Some(ref eigenvalues) = hessian.eigenvalues {
            let max_eigenval = eigenvalues.iter().fold(T::neg_infinity(), |a, &b| a.max(b));
            let min_eigenval = eigenvalues.iter().fold(T::infinity(), |a, &b| a.min(b));
            
            if min_eigenval > T::zero() {
                Some(max_eigenval / min_eigenval)
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Check if Hessian is positive definite
    pub fn is_positive_definite<T: Float>(hessian: &HessianMatrix<T>) -> bool {
        if let Some(ref eigenvalues) = hessian.eigenvalues {
            eigenvalues.iter().all(|&val| val > T::zero())
        } else if let Some(ref diagonal) = hessian.diagonal {
            diagonal.iter().all(|&val| val > T::zero())
        } else {
            false
        }
    }
    
    /// Add damping to Hessian for numerical stability
    pub fn add_damping<T: Float>(hessian: &mut Array2<T>, damping: T) {
        for i in 0.._hessian.nrows() {
            hessian[[i, i]] = hessian[[i, i]] + damping;
        }
    }
    
    /// Compute optimal damping parameter using Levenberg-Marquardt heuristic
    pub fn compute_optimal_damping<T: Float>(
        hessian: &HessianMatrix<T>,
        gradient_norm: T,
    ) -> T {
        if let Some(ref diagonal) = hessian.diagonal {
            let max_diag = diagonal.iter().fold(T::neg_infinity(), |a, &b| a.max(b));
            let adaptive_damping = gradient_norm * num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero());
            adaptive_damping.max(max_diag * num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero()))
        } else {
            gradient_norm * num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::{AutodiffConfig, AutodiffEngine};
    
    #[test]
    fn test_second_order_config_default() {
        let config = SecondOrderConfig::default();
        assert_eq!(config.learning_rate, 1e-3);
        assert!(config.enable_line_search);
        assert_eq!(config.memory_size, 10);
    }
    
    #[test]
    fn test_line_search_config() {
        let config = LineSearchConfig::default();
        assert_eq!(config.c1, 1e-4);
        assert_eq!(config.c2, 0.9);
        assert_eq!(config.method, LineSearchMethod::Backtracking);
    }
    
    #[test]
    fn test_newton_optimizer_creation() {
        let config = SecondOrderConfig::default();
        let autodiff_config = AutodiffConfig::default();
        let engine = AutodiffEngine::<f64>::new(autodiff_config);
        
        let optimizer = NewtonOptimizer::new(config, engine);
        assert_eq!(optimizer.iteration, 0);
    }
    
    #[test]
    fn test_bfgs_memory() {
        let memory = BFGSMemory::<f64> {
            memory_size: 5,
            y_history: VecDeque::new(),
            s_history: VecDeque::new(),
            rho_history: VecDeque::new(),
            initial_scale: 1.0};
        
        assert_eq!(memory.memory_size, 5);
        assert!(memory.y_history.is_empty());
        assert!(memory.s_history.is_empty());
    }
    
    #[test]
    fn test_trust_region_state() {
        let state = TrustRegionState {
            radius: 1.0f64,
            radius_history: VecDeque::new(),
            expand_threshold: 0.75,
            shrink_threshold: 0.25,
            expand_factor: 2.0,
            shrink_factor: 0.5};
        
        assert_eq!(state.radius, 1.0);
        assert_eq!(state.expand_factor, 2.0);
        assert_eq!(state.shrink_factor, 0.5);
    }
}
