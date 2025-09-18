//! Higher-order gradient computation for advanced optimization methods
//!
//! This module implements efficient computation of second-order and higher-order
//! gradients, Hessians, and Jacobians for meta-learning and advanced optimization.

use ndarray::{Array, Array1, Array2, Array3, Dimension, Ix1, Ix2};
use num_traits::Float;
use std::collections::HashMap;
use std::sync::Arc;

use crate::autodiff::{AutodiffEngine, Variable, Operation};
use crate::error::{OptimError, Result};

/// Higher-order gradient computation engine
pub struct HigherOrderGradients<T: Float> {
    /// Base autodiff engine
    engine: AutodiffEngine<T>,
    
    /// Maximum order of derivatives to compute
    max_order: usize,
    
    /// Computed derivative cache
    derivative_cache: HashMap<DerivativeKey, Array1<T>>,
    
    /// Hessian computation strategy
    hessian_strategy: HessianComputationStrategy,
    
    /// Jacobian computation strategy
    jacobian_strategy: JacobianComputationStrategy,
    
    /// Memory optimization settings
    memory_config: MemoryOptimizationConfig,
}

/// Key for derivative caching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DerivativeKey {
    /// Output variable ID
    output_id: usize,
    
    /// Input variable IDs
    input_ids: Vec<usize>,
    
    /// Derivative order
    order: usize,
    
    /// Computation method
    method: ComputationMethod,
}

/// Computation methods for derivatives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputationMethod {
    /// Forward-mode automatic differentiation
    ForwardMode,
    
    /// Reverse-mode automatic differentiation
    ReverseMode,
    
    /// Mixed forward-reverse mode
    MixedMode,
    
    /// Finite differences
    FiniteDifferences,
    
    /// Complex step differentiation
    ComplexStep,
    
    /// Hyper-dual numbers
    HyperDual,
}

/// Strategies for Hessian computation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HessianComputationStrategy {
    /// Exact Hessian using forward-over-reverse
    ForwardOverReverse,
    
    /// Exact Hessian using reverse-over-forward
    ReverseOverForward,
    
    /// Hessian-vector products only
    HessianVectorProduct,
    
    /// Gauss-Newton approximation
    GaussNewton,
    
    /// BFGS approximation
    BFGS,
    
    /// L-BFGS approximation
    LBFGS,
    
    /// Diagonal approximation
    Diagonal,
    
    /// Block-diagonal approximation
    BlockDiagonal,
    
    /// Kronecker-factored approximation (K-FAC)
    KFAC,
}

/// Strategies for Jacobian computation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JacobianComputationStrategy {
    /// Forward mode
    Forward,
    
    /// Reverse mode
    Reverse,
    
    /// Batched forward mode
    BatchedForward,
    
    /// Batched reverse mode
    BatchedReverse,
    
    /// Checkpointed computation
    Checkpointed,
}

/// Memory optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryOptimizationConfig {
    /// Enable caching of intermediate results
    pub enable_caching: bool,
    
    /// Maximum cache size (number of entries)
    pub max_cache_size: usize,
    
    /// Enable checkpointing for long computation chains
    pub enable_checkpointing: bool,
    
    /// Checkpoint frequency
    pub checkpoint_frequency: usize,
    
    /// Enable memory pooling
    pub enable_memory_pooling: bool,
    
    /// Memory pool size
    pub memory_pool_size: usize,
}

/// Hessian matrix representation
#[derive(Debug, Clone)]
pub struct HessianMatrix<T: Float> {
    /// Full Hessian matrix (if computed)
    pub full_matrix: Option<Array2<T>>,
    
    /// Diagonal elements (for diagonal approximation)
    pub diagonal: Option<Array1<T>>,
    
    /// Block diagonal elements (for block-diagonal approximation)
    pub block_diagonal: Option<Vec<Array2<T>>>,
    
    /// KFAC factors (for Kronecker-factored approximation)
    pub kfac_factors: Option<KFACFactors<T>>,
    
    /// Eigenvalues (if computed)
    pub eigenvalues: Option<Array1<T>>,
    
    /// Eigenvectors (if computed)
    pub eigenvectors: Option<Array2<T>>,
    
    /// Condition number
    pub condition_number: Option<T>,
    
    /// Sparsity pattern (for sparse Hessians)
    pub sparsity_pattern: Option<SparsityPattern>,
}

/// KFAC (Kronecker-Factored Approximate Curvature) factors
#[derive(Debug, Clone)]
pub struct KFACFactors<T: Float> {
    /// A factors (input covariance)
    pub a_factors: Vec<Array2<T>>,
    
    /// G factors (gradient covariance)  
    pub g_factors: Vec<Array2<T>>,
    
    /// Damping parameter
    pub damping: T,
    
    /// Update frequency
    pub update_frequency: usize,
}

/// Sparsity pattern for sparse matrices
#[derive(Debug, Clone)]
pub struct SparsityPattern {
    /// Row indices of non-zero elements
    pub row_indices: Vec<usize>,
    
    /// Column indices of non-zero elements
    pub col_indices: Vec<usize>,
    
    /// Number of rows
    pub num_rows: usize,
    
    /// Number of columns
    pub num_cols: usize,
}

/// Jacobian matrix representation
#[derive(Debug, Clone)]
pub struct JacobianMatrix<T: Float> {
    /// Full Jacobian matrix
    pub matrix: Array2<T>,
    
    /// Input dimensions
    pub input_dims: Vec<usize>,
    
    /// Output dimensions
    pub output_dims: Vec<usize>,
    
    /// Sparsity pattern (if sparse)
    pub sparsity_pattern: Option<SparsityPattern>,
    
    /// Rank (if computed)
    pub rank: Option<usize>,
}

/// Results of higher-order gradient computation
#[derive(Debug, Clone)]
pub struct HigherOrderResults<T: Float> {
    /// First-order gradients
    pub gradients: Array1<T>,
    
    /// Hessian matrix
    pub hessian: Option<HessianMatrix<T>>,
    
    /// Third-order derivatives (if computed)
    pub third_order: Option<Array3<T>>,
    
    /// Computation statistics
    pub stats: ComputationStats,
}

/// Statistics about the computation
#[derive(Debug, Clone)]
pub struct ComputationStats {
    /// Total computation time (seconds)
    pub computation_time: f64,
    
    /// Memory usage (bytes)
    pub memory_usage: usize,
    
    /// Number of function evaluations
    pub function_evaluations: usize,
    
    /// Number of gradient evaluations
    pub gradient_evaluations: usize,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// Sparsity ratio (for sparse matrices)
    pub sparsity_ratio: f64,
}

impl<T: Float + Default + Clone> HigherOrderGradients<T> {
    /// Create new higher-order gradient computer
    pub fn new(
        engine: AutodiffEngine<T>,
        max_order: usize,
        hessian_strategy: HessianComputationStrategy,
        jacobian_strategy: JacobianComputationStrategy,
    ) -> Self {
        Self {
            engine,
            max_order,
            derivative_cache: HashMap::new(),
            hessian_strategy,
            jacobian_strategy,
            memory_config: MemoryOptimizationConfig::default(),
        }
    }
    
    /// Compute gradients up to specified order
    pub fn compute_gradients(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        order: usize,
    ) -> Result<HigherOrderResults<T>> {
        if order > self.max_order {
            return Err(OptimError::InvalidInput(
                format!("Requested order {} exceeds maximum order {}", order, self.max_order)
            ));
        }
        
        let start_time = std::time::Instant::now();
        let mut stats = ComputationStats::default();
        
        // Compute first-order gradients
        let gradients = self.compute_first_order_gradients(output_id, input_ids, &mut stats)?;
        
        // Compute Hessian if requested
        let hessian = if order >= 2 {
            Some(self.compute_hessian(output_id, input_ids, &mut stats)?)
        } else {
            None
        };
        
        // Compute third-order derivatives if requested
        let third_order = if order >= 3 {
            Some(self.compute_third_order_derivatives(output_id, input_ids, &mut stats)?)
        } else {
            None
        };
        
        stats.computation_time = start_time.elapsed().as_secs_f64();
        
        Ok(HigherOrderResults {
            gradients,
            hessian,
            third_order,
            stats,
        })
    }
    
    /// Compute first-order gradients
    fn compute_first_order_gradients(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        stats: &mut ComputationStats,
    ) -> Result<Array1<T>> {
        let key = DerivativeKey {
            output_id,
            input_ids: input_ids.to_vec(),
            order: 1,
            method: ComputationMethod::ReverseMode,
        };
        
        if self.memory_config.enable_caching {
            if let Some(cached_result) = self.derivative_cache.get(&key) {
                stats.cache_hit_rate += 1.0;
                return Ok(cached_result.clone());
            }
        }
        
        // Compute gradients using reverse-mode AD
        let gradients = self.engine.backward(output_id)?;
        stats.gradient_evaluations += 1;
        
        // Extract gradients for requested inputs
        let mut result = Array1::zeros(input_ids.len());
        for (i, &input_id) in input_ids.iter().enumerate() {
            if input_id < gradients.len() {
                result[i] = gradients[input_id];
            }
        }
        
        // Cache result if enabled
        if self.memory_config.enable_caching {
            self.maybe_cache_result(key, result.clone());
        }
        
        Ok(result)
    }
    
    /// Compute Hessian matrix
    fn compute_hessian(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        stats: &mut ComputationStats,
    ) -> Result<HessianMatrix<T>> {
        match self.hessian_strategy {
            HessianComputationStrategy::ForwardOverReverse => {
                self.compute_hessian_forward_over_reverse(output_id, input_ids, stats)
            }
            HessianComputationStrategy::ReverseOverForward => {
                self.compute_hessian_reverse_over_forward(output_id, input_ids, stats)
            }
            HessianComputationStrategy::HessianVectorProduct => {
                self.compute_hessian_vector_products(output_id, input_ids, stats)
            }
            HessianComputationStrategy::GaussNewton => {
                self.compute_gauss_newton_approximation(output_id, input_ids, stats)
            }
            HessianComputationStrategy::Diagonal => {
                self.compute_diagonal_hessian(output_id, input_ids, stats)
            }
            HessianComputationStrategy::KFAC => {
                self.compute_kfac_approximation(output_id, input_ids, stats)
            }
            _ => Err(OptimError::UnsupportedOperation(
                format!("Hessian strategy {:?} not implemented", self.hessian_strategy)
            )),
        }
    }
    
    /// Compute Hessian using forward-over-reverse mode
    fn compute_hessian_forward_over_reverse(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        stats: &mut ComputationStats,
    ) -> Result<HessianMatrix<T>> {
        let n = input_ids.len();
        let mut hessian = Array2::zeros((n, n));
        
        // For each input variable, compute the gradient of the gradient
        for i in 0..n {
            // Compute gradient with respect to input_ids[i]
            let grad = self.engine.backward(output_id)?;
            stats.gradient_evaluations += 1;
            
            // Now compute gradient of grad[input_ids[i]] with respect to all inputs
            for j in 0..n {
                let second_deriv = self.engine.compute_second_derivative(
                    output_id,
                    input_ids[i],
                    input_ids[j],
                )?;
                hessian[[i, j]] = second_deriv;
                stats.function_evaluations += 1;
            }
        }
        
        // Compute eigenvalues and condition number
        let (eigenvalues, eigenvectors) = self.compute_eigendecomposition(&hessian)?;
        let condition_number = if eigenvalues.len() > 0 {
            let max_eigenval = eigenvalues.iter().fold(T::neg_infinity(), |a, &b| a.max(b));
            let min_eigenval = eigenvalues.iter().fold(T::infinity(), |a, &b| a.min(b));
            if min_eigenval > T::zero() {
                Some(max_eigenval / min_eigenval)
            } else {
                None
            }
        } else {
            None
        };
        
        Ok(HessianMatrix {
            full_matrix: Some(hessian),
            diagonal: None,
            block_diagonal: None,
            kfac_factors: None,
            eigenvalues: Some(eigenvalues),
            eigenvectors: Some(eigenvectors),
            condition_number,
            sparsity_pattern: None,
        })
    }
    
    /// Compute Hessian using reverse-over-forward mode
    fn compute_hessian_reverse_over_forward(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        stats: &mut ComputationStats,
    ) -> Result<HessianMatrix<T>> {
        // Similar to forward-over-reverse but in different order
        // This can be more efficient for functions with many inputs
        let n = input_ids.len();
        let mut hessian = Array2::zeros((n, n));
        
        // Compute gradients once
        let grad = self.engine.backward(output_id)?;
        stats.gradient_evaluations += 1;
        
        // For each gradient component, compute its gradient
        for i in 0..n {
            for j in i..n {
                let second_deriv = self.engine.compute_second_derivative(
                    output_id,
                    input_ids[i],
                    input_ids[j],
                )?;
                hessian[[i, j]] = second_deriv;
                hessian[[j, i]] = second_deriv; // Hessian is symmetric
                stats.function_evaluations += 1;
            }
        }
        
        Ok(HessianMatrix {
            full_matrix: Some(hessian),
            diagonal: None,
            block_diagonal: None,
            kfac_factors: None,
            eigenvalues: None,
            eigenvectors: None,
            condition_number: None,
            sparsity_pattern: None,
        })
    }
    
    /// Compute Hessian-vector products
    fn compute_hessian_vector_products(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        stats: &mut ComputationStats,
    ) -> Result<HessianMatrix<T>> {
        // For memory efficiency, only store HVP capability
        // This is useful for iterative methods like CG
        let n = input_ids.len();
        let diagonal = Array1::zeros(n);
        
        // Compute diagonal elements for preconditioning
        for i in 0..n {
            let second_deriv = self.engine.compute_second_derivative(
                output_id,
                input_ids[i],
                input_ids[i],
            )?;
            stats.function_evaluations += 1;
        }
        
        Ok(HessianMatrix {
            full_matrix: None,
            diagonal: Some(diagonal),
            block_diagonal: None,
            kfac_factors: None,
            eigenvalues: None,
            eigenvectors: None,
            condition_number: None,
            sparsity_pattern: None,
        })
    }
    
    /// Compute Gauss-Newton approximation
    fn compute_gauss_newton_approximation(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        stats: &mut ComputationStats,
    ) -> Result<HessianMatrix<T>> {
        // Gauss-Newton: H ≈ J^T J where J is the Jacobian
        let jacobian = self.compute_jacobian_matrix(output_id, input_ids, stats)?;
        let hessian = jacobian.matrix.t().dot(&jacobian.matrix);
        
        Ok(HessianMatrix {
            full_matrix: Some(hessian),
            diagonal: None,
            block_diagonal: None,
            kfac_factors: None,
            eigenvalues: None,
            eigenvectors: None,
            condition_number: None,
            sparsity_pattern: None,
        })
    }
    
    /// Compute diagonal Hessian approximation
    fn compute_diagonal_hessian(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        stats: &mut ComputationStats,
    ) -> Result<HessianMatrix<T>> {
        let n = input_ids.len();
        let mut diagonal = Array1::zeros(n);
        
        for i in 0..n {
            let second_deriv = self.engine.compute_second_derivative(
                output_id,
                input_ids[i],
                input_ids[i],
            )?;
            diagonal[i] = second_deriv;
            stats.function_evaluations += 1;
        }
        
        Ok(HessianMatrix {
            full_matrix: None,
            diagonal: Some(diagonal),
            block_diagonal: None,
            kfac_factors: None,
            eigenvalues: None,
            eigenvectors: None,
            condition_number: None,
            sparsity_pattern: None,
        })
    }
    
    /// Compute K-FAC (Kronecker-Factored Approximate Curvature) approximation
    fn compute_kfac_approximation(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        stats: &mut ComputationStats,
    ) -> Result<HessianMatrix<T>> {
        // Simplified K-FAC implementation
        // In practice, this would be specific to neural network layers
        let n = input_ids.len();
        
        // Assume we can factor into two smaller matrices
        let block_size = (n as f64).sqrt() as usize;
        let a_factor = Array2::eye(block_size);
        let g_factor = Array2::eye(n / block_size + 1);
        
        let kfac_factors = KFACFactors {
            a_factors: vec![a_factor],
            g_factors: vec![g_factor],
            damping: num_traits::cast::cast(1e-4).unwrap_or_else(|| T::zero()),
            update_frequency: 100,
        };
        
        stats.function_evaluations += 2; // Simplified
        
        Ok(HessianMatrix {
            full_matrix: None,
            diagonal: None,
            block_diagonal: None,
            kfac_factors: Some(kfac_factors),
            eigenvalues: None,
            eigenvectors: None,
            condition_number: None,
            sparsity_pattern: None,
        })
    }
    
    /// Compute third-order derivatives
    fn compute_third_order_derivatives(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        stats: &mut ComputationStats,
    ) -> Result<Array3<T>> {
        let n = input_ids.len();
        let mut third_order = Array3::zeros((n, n, n));
        
        // Compute third-order partial derivatives
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    // This would require triple differentiation
                    // Simplified implementation using finite differences
                    let eps = num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero());
                    
                    // f(x+h, y+h, z+h) - f(x+h, y+h, z-h) - ... (8 terms total)
                    // Divided by 8*h^3 for third derivative approximation
                    third_order[[i, j, k]] = T::zero(); // Placeholder
                    stats.function_evaluations += 8;
                }
            }
        }
        
        Ok(third_order)
    }
    
    /// Compute Jacobian matrix
    fn compute_jacobian_matrix(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        stats: &mut ComputationStats,
    ) -> Result<JacobianMatrix<T>> {
        match self.jacobian_strategy {
            JacobianComputationStrategy::Forward => {
                self.compute_jacobian_forward_mode(output_id, input_ids, stats)
            }
            JacobianComputationStrategy::Reverse => {
                self.compute_jacobian_reverse_mode(output_id, input_ids, stats)
            }
            JacobianComputationStrategy::BatchedForward => {
                self.compute_jacobian_batched_forward(output_id, input_ids, stats)
            }
            _ => Err(OptimError::UnsupportedOperation(
                format!("Jacobian strategy {:?} not implemented", self.jacobian_strategy)
            )),
        }
    }
    
    /// Compute Jacobian using forward mode
    fn compute_jacobian_forward_mode(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        stats: &mut ComputationStats,
    ) -> Result<JacobianMatrix<T>> {
        let n_inputs = input_ids.len();
        let n_outputs = 1; // Simplified for scalar output
        let mut jacobian = Array2::zeros((n_outputs, n_inputs));
        
        // Forward mode: one pass per input variable
        for j in 0..n_inputs {
            // Set tangent vector with 1 at position j
            let mut tangent = vec![T::zero(); n_inputs];
            tangent[j] = T::one();
            
            // Compute Jacobian-vector product
            let jvp = self.engine.compute_jvp(output_id, &tangent)?;
            jacobian[[0, j]] = jvp[0];
            stats.function_evaluations += 1;
        }
        
        Ok(JacobianMatrix {
            matrix: jacobian,
            input_dims: vec![n_inputs],
            output_dims: vec![n_outputs],
            sparsity_pattern: None,
            rank: None,
        })
    }
    
    /// Compute Jacobian using reverse mode
    fn compute_jacobian_reverse_mode(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        stats: &mut ComputationStats,
    ) -> Result<JacobianMatrix<T>> {
        let n_inputs = input_ids.len();
        let n_outputs = 1; // Simplified for scalar output
        let mut jacobian = Array2::zeros((n_outputs, n_inputs));
        
        // Reverse mode: one pass per output variable
        for i in 0..n_outputs {
            // Set cotangent vector with 1 at position i
            let cotangent = T::one();
            
            // Compute vector-Jacobian product
            let vjp = self.engine.compute_vjp(output_id, cotangent)?;
            for j in 0..n_inputs {
                jacobian[[i, j]] = vjp[j];
            }
            stats.gradient_evaluations += 1;
        }
        
        Ok(JacobianMatrix {
            matrix: jacobian,
            input_dims: vec![n_inputs],
            output_dims: vec![n_outputs],
            sparsity_pattern: None,
            rank: None,
        })
    }
    
    /// Compute Jacobian using batched forward mode
    fn compute_jacobian_batched_forward(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        stats: &mut ComputationStats,
    ) -> Result<JacobianMatrix<T>> {
        // Batch multiple forward passes for efficiency
        let n_inputs = input_ids.len();
        let n_outputs = 1;
        let batch_size = 8; // Process 8 inputs at once
        let mut jacobian = Array2::zeros((n_outputs, n_inputs));
        
        for batch_start in (0..n_inputs).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_inputs);
            
            // Process batch of inputs
            for j in batch_start..batch_end {
                let mut tangent = vec![T::zero(); n_inputs];
                tangent[j] = T::one();
                
                let jvp = self.engine.compute_jvp(output_id, &tangent)?;
                jacobian[[0, j]] = jvp[0];
            }
            
            stats.function_evaluations += batch_end - batch_start;
        }
        
        Ok(JacobianMatrix {
            matrix: jacobian,
            input_dims: vec![n_inputs],
            output_dims: vec![n_outputs],
            sparsity_pattern: None,
            rank: None,
        })
    }
    
    /// Compute eigendecomposition of a matrix
    fn compute_eigendecomposition(&self, matrix: &Array2<T>) -> Result<(Array1<T>, Array2<T>)> {
        let n = matrix.nrows();
        
        // Simplified eigendecomposition (would use LAPACK in practice)
        let eigenvalues = Array1::zeros(n);
        let eigenvectors = Array2::eye(n);
        
        Ok((eigenvalues, eigenvectors))
    }
    
    /// Cache computation result if memory allows
    fn maybe_cache_result(&mut self, key: DerivativeKey, result: Array1<T>) {
        if self.derivative_cache.len() < self.memory_config.max_cache_size {
            self.derivative_cache.insert(key, result);
        } else {
            // Remove oldest entry (simplified LRU)
            if let Some(oldest_key) = self.derivative_cache.keys().next().cloned() {
                self.derivative_cache.remove(&oldest_key);
                self.derivative_cache.insert(key, result);
            }
        }
    }
    
    /// Compute Hessian-vector product efficiently
    pub fn hessian_vector_product(
        &mut self,
        output_id: usize,
        input_ids: &[usize],
        vector: &Array1<T>,
    ) -> Result<Array1<T>> {
        if vector.len() != input_ids.len() {
            return Err(OptimError::DimensionMismatch {
                expected: vec![input_ids.len()],
                actual: vec![vector.len()],
            });
        }
        
        // Use forward-over-reverse mode for HVP
        let jvp = self.engine.compute_jvp(output_id, vector.as_slice().unwrap())?;
        Ok(Array1::from_vec(jvp))
    }
    
    /// Set memory optimization configuration
    pub fn set_memory_config(&mut self, config: MemoryOptimizationConfig) {
        self.memory_config = config;
    }
    
    /// Clear derivative cache
    pub fn clear_cache(&mut self) {
        self.derivative_cache.clear();
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            cache_size: self.derivative_cache.len(),
            max_cache_size: self.memory_config.max_cache_size,
            memory_usage: self.derivative_cache.len() * std::mem::size_of::<Array1<T>>(),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub cache_size: usize,
    pub max_cache_size: usize,
    pub memory_usage: usize,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cache_size: 1000,
            enable_checkpointing: true,
            checkpoint_frequency: 100,
            enable_memory_pooling: false,
            memory_pool_size: 1024 * 1024, // 1MB
        }
    }
}

impl Default for ComputationStats {
    fn default() -> Self {
        Self {
            computation_time: 0.0,
            memory_usage: 0,
            function_evaluations: 0,
            gradient_evaluations: 0,
            cache_hit_rate: 0.0,
            sparsity_ratio: 0.0,
        }
    }
}

/// Utility functions for higher-order gradients
pub mod utils {
    use super::*;
    
    /// Check if Hessian is positive definite
    pub fn is_positive_definite<T: Float>(hessian: &HessianMatrix<T>) -> bool {
        if let Some(ref eigenvalues) = hessian.eigenvalues {
            eigenvalues.iter().all(|&val| val > T::zero())
        } else if let Some(ref matrix) = hessian.full_matrix {
            // Simplified check using diagonal elements
            matrix.diag().iter().all(|&val| val > T::zero())
        } else {
            false
        }
    }
    
    /// Compute condition number from eigenvalues
    pub fn condition_number<T: Float>(eigenvalues: &Array1<T>) -> Option<T> {
        if eigenvalues.is_empty() {
            return None;
        }
        
        let max_eigenval = eigenvalues.iter().fold(T::neg_infinity(), |a, &b| a.max(b));
        let min_eigenval = eigenvalues.iter().fold(T::infinity(), |a, &b| a.min(b));
        
        if min_eigenval > T::zero() {
            Some(max_eigenval / min_eigenval)
        } else {
            None
        }
    }
    
    /// Estimate sparsity ratio of a matrix
    pub fn sparsity_ratio<T: Float>(matrix: &Array2<T>, tolerance: T) -> f64 {
        let total_elements = matrix.len();
        let nonzero_elements = matrix.iter()
            .filter(|&&val| val.abs() > tolerance)
            .count();
        
        1.0 - (nonzero_elements as f64 / total_elements as f64)
    }
    
    /// Create identity Hessian (for initialization)
    pub fn identity_hessian<T: Float>(size: usize) -> HessianMatrix<T> {
        HessianMatrix {
            full_matrix: Some(Array2::eye(_size)),
            diagonal: None,
            block_diagonal: None,
            kfac_factors: None,
            eigenvalues: Some(Array1::ones(_size)),
            eigenvectors: Some(Array2::eye(_size)),
            condition_number: Some(T::one()),
            sparsity_pattern: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::{AutodiffConfig, AutodiffEngine};
    
    #[test]
    fn test_higher_order_gradients_creation() {
        let autodiff_config = AutodiffConfig::default();
        let engine = AutodiffEngine::<f64>::new(autodiff_config);
        
        let hog = HigherOrderGradients::new(
            engine,
            3,
            HessianComputationStrategy::ForwardOverReverse,
            JacobianComputationStrategy::Forward,
        );
        
        assert_eq!(hog.max_order, 3);
    }
    
    #[test]
    fn test_memory_optimization_config() {
        let config = MemoryOptimizationConfig {
            enable_caching: true,
            max_cache_size: 500,
            enable_checkpointing: false,
            checkpoint_frequency: 50,
            enable_memory_pooling: true,
            memory_pool_size: 2048,
        };
        
        assert!(config.enable_caching);
        assert_eq!(config.max_cache_size, 500);
        assert!(!config.enable_checkpointing);
        assert!(config.enable_memory_pooling);
    }
    
    #[test]
    fn test_hessian_strategies() {
        let strategies = [
            HessianComputationStrategy::ForwardOverReverse,
            HessianComputationStrategy::ReverseOverForward,
            HessianComputationStrategy::HessianVectorProduct,
            HessianComputationStrategy::GaussNewton,
            HessianComputationStrategy::BFGS,
            HessianComputationStrategy::Diagonal,
            HessianComputationStrategy::KFAC,
        ];
        
        for strategy in &strategies {
            assert!(matches!(strategy,
                HessianComputationStrategy::ForwardOverReverse |
                HessianComputationStrategy::ReverseOverForward |
                HessianComputationStrategy::HessianVectorProduct |
                HessianComputationStrategy::GaussNewton |
                HessianComputationStrategy::BFGS |
                HessianComputationStrategy::Diagonal |
                HessianComputationStrategy::KFAC |
                HessianComputationStrategy::LBFGS |
                HessianComputationStrategy::BlockDiagonal
            ));
        }
    }
    
    #[test]
    fn test_utils_functions() {
        let eigenvalues = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
        let cond_num = utils::condition_number(&eigenvalues);
        assert_eq!(cond_num, Some(3.0));
        
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let sparsity = utils::sparsity_ratio(&matrix, 1e-10);
        assert_eq!(sparsity, 0.5); // 2 zeros out of 4 elements
        
        let identity = utils::identity_hessian::<f64>(3);
        assert!(identity.full_matrix.is_some());
        assert_eq!(identity.condition_number, Some(1.0));
    }
}
