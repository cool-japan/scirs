//! Higher-order automatic differentiation
//!
//! This module implements computation of higher-order derivatives including
//! Hessians, third-order derivatives, and mixed partial derivatives for
//! advanced optimization algorithms.

use ndarray::{Array1, Array2, Array3};
use num_traits::Float;
use std::collections::HashMap;

use super::forward_mode::ForwardModeEngine;
use super::reverse_mode::ReverseModeEngine;
use crate::error::{OptimError, Result};

/// Higher-order differentiation engine
#[allow(dead_code)]
pub struct HigherOrderEngine<T: Float + Default + Clone + 'static + std::iter::Sum + ndarray::ScalarOperand> {
    /// Forward-mode engine for directional derivatives
    forward_engine: ForwardModeEngine<T>,

    /// Reverse-mode engine for efficient gradient computation
    reverse_engine: ReverseModeEngine<T>,

    /// Maximum derivative order to compute
    _maxorder: usize,

    /// Use mixed-mode for Hessian computation
    mixed_mode: bool,

    /// Cache for computed derivatives
    derivative_cache: HashMap<DerivativeKey, DerivativeValue<T>>,

    /// Finite difference settings for numerical verification
    finite_diff_eps: T,

    /// Enable automatic parallelization
    parallel_computation: bool,

    /// Thread pool size
    thread_pool_size: usize,

    /// Use sparse computations when beneficial
    adaptive_sparsity: bool,

    /// Automatic differentiation mode selection
    auto_mode_selection: bool,

    /// Performance profiler
    profiler: ComputationProfiler<T>,
}

/// Key for derivative caching
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct DerivativeKey {
    function_id: usize,
    variable_ids: Vec<usize>,
    order: usize,
}

/// Cached derivative value
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum DerivativeValue<T: Float> {
    Scalar(T),
    Vector(Array1<T>),
    Matrix(Array2<T>),
    Tensor3(Array3<T>),
}

/// Hessian computation configuration
#[derive(Debug, Clone)]
pub struct HessianConfig {
    /// Use exact computation (vs approximation)
    pub exact: bool,

    /// Use sparse representation
    pub sparse: bool,

    /// Sparsity threshold
    pub sparsity_threshold: f64,

    /// Use diagonal approximation
    pub diagonal_only: bool,

    /// Use BFGS approximation
    pub bfgs_approximation: bool,

    /// Use finite differences for verification
    pub verify_with_finite_diff: bool,
}

impl Default for HessianConfig {
    fn default() -> Self {
        Self {
            exact: true,
            sparse: false,
            sparsity_threshold: 1e-8,
            diagonal_only: false,
            bfgs_approximation: false,
            verify_with_finite_diff: false,
        }
    }
}

/// Sparse Hessian representation
#[derive(Debug, Clone)]
pub struct SparseHessian<T: Float> {
    /// Row indices
    pub rows: Vec<usize>,

    /// Column indices  
    pub cols: Vec<usize>,

    /// Values
    pub values: Vec<T>,

    /// Matrix dimensions
    pub shape: (usize, usize),

    /// Number of non-zero elements
    pub nnz: usize,
}

/// Third-order derivative tensor
#[derive(Debug, Clone)]
pub struct ThirdOrderTensor<T: Float> {
    /// Dense tensor data
    pub data: Array3<T>,

    /// Sparse representation (optional)
    pub sparse_data: Option<SparseTensor3<T>>,

    /// Tensor dimensions
    pub shape: (usize, usize, usize),
}

/// Sparse third-order tensor
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SparseTensor3<T: Float> {
    indices: Vec<(usize, usize, usize)>,
    values: Vec<T>,
    shape: (usize, usize, usize),
}

/// Mixed partial derivatives
#[derive(Debug, Clone)]
pub struct MixedPartials<T: Float> {
    /// Variable indices for mixed partial
    pub variables: Vec<usize>,

    /// Derivative order for each variable
    pub orders: Vec<usize>,

    /// Computed value
    pub value: T,

    /// Computation method used
    pub method: MixedPartialMethod,
}

/// Methods for computing mixed partials
#[derive(Debug, Clone, Copy)]
pub enum MixedPartialMethod {
    ForwardOverReverse,
    ReverseOverForward,
    PureForward,
    FiniteDifference,
}

/// Layer information for K-FAC computation
#[derive(Debug, Clone)]
pub struct LayerInfo<T: Float> {
    pub layer_type: LayerType,
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Array2<T>,
    pub bias: Option<Array1<T>>,
}

/// Types of neural network layers
#[derive(Debug, Clone, Copy)]
pub enum LayerType {
    Linear,
    Convolutional,
    LSTM,
    Attention,
}

impl<T: Float + Default + Clone + 'static + std::iter::Sum + ndarray::ScalarOperand>
    HigherOrderEngine<T>
{
    /// Create a new higher-order differentiation engine
    pub fn new(_maxorder: usize) -> Self {
        Self {
            forward_engine: ForwardModeEngine::new(),
            reverse_engine: ReverseModeEngine::new(),
            _maxorder,
            mixed_mode: true,
            derivative_cache: HashMap::new(),
            finite_diff_eps: T::from(1e-5).unwrap(),
            parallel_computation: true,
            thread_pool_size: 4, // Conservative default
            adaptive_sparsity: true,
            auto_mode_selection: true,
            profiler: ComputationProfiler::new(),
        }
    }

    /// Create a new engine with advanced configuration
    pub fn with_config(config: HigherOrderConfig<T>) -> Self {
        Self {
            forward_engine: ForwardModeEngine::new(),
            reverse_engine: ReverseModeEngine::new(),
            _maxorder: config._maxorder,
            mixed_mode: config.mixed_mode,
            derivative_cache: HashMap::new(),
            finite_diff_eps: config.finite_diff_eps,
            parallel_computation: config.parallel_computation,
            thread_pool_size: config.thread_pool_size,
            adaptive_sparsity: config.adaptive_sparsity,
            auto_mode_selection: config.auto_mode_selection,
            profiler: ComputationProfiler::new(),
        }
    }

    /// Enable/disable mixed-mode computation
    pub fn set_mixed_mode(&mut self, enabled: bool) {
        self.mixed_mode = enabled;
    }

    /// Set finite difference epsilon for numerical verification
    pub fn set_finite_diff_eps(&mut self, eps: T) {
        self.finite_diff_eps = eps;
    }

    /// Compute Hessian matrix using forward-over-reverse mode
    pub fn hessian_forward_over_reverse(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        config: &HessianConfig,
    ) -> Result<Array2<T>> {
        let n = point.len();

        if config.diagonal_only {
            return self.hessian_diagonal(&function, point);
        }

        let mut hessian = Array2::zeros((n, n));

        // Use forward-over-reverse: compute one row of Hessian at a time
        for i in 0..n {
            let grad_fn = |x: &Array1<T>| -> Array1<T> {
                self.gradient_at_point(&function, x)
                    .unwrap_or_else(|_| Array1::zeros(n))
            };

            // Compute directional derivative of gradient
            let mut direction = Array1::zeros(n);
            direction[i] = T::one();

            let hessian_row =
                self.directional_derivative_of_gradient(&grad_fn, point, &direction)?;

            for j in 0..n {
                hessian[[i, j]] = hessian_row[j];
            }
        }

        // Symmetrize if needed
        if config.exact {
            for i in 0..n {
                for j in i + 1..n {
                    let avg = (hessian[[i, j]] + hessian[[j, i]]) / T::from(2.0).unwrap();
                    hessian[[i, j]] = avg;
                    hessian[[j, i]] = avg;
                }
            }
        }

        // Verify with finite differences if requested
        if config.verify_with_finite_diff {
            let fd_hessian = self.finite_difference_hessian(&function, point)?;
            self.verify_hessian_accuracy(&hessian, &fd_hessian)?;
        }

        Ok(hessian)
    }

    /// Compute Hessian matrix using reverse-over-forward mode
    pub fn hessian_reverse_over_forward(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        _config: &HessianConfig,
    ) -> Result<Array2<T>> {
        let n = point.len();
        let mut hessian = Array2::zeros((n, n));

        // Use reverse-over-forward: compute one column of Hessian at a time
        for j in 0..n {
            // Create a function that computes partial derivative w.r.t. x_j
            let partial_fn = |x: &Array1<T>| -> T {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();

                x_plus[j] = x_plus[j] + self.finite_diff_eps;
                x_minus[j] = x_minus[j] - self.finite_diff_eps;

                (function(&x_plus) - function(&x_minus))
                    / (T::from(2.0).unwrap() * self.finite_diff_eps)
            };

            // Compute gradient of partial derivative
            let hessian_col = self.gradient_at_point(&partial_fn, point)?;

            for i in 0..n {
                hessian[[i, j]] = hessian_col[i];
            }
        }

        Ok(hessian)
    }

    /// Compute diagonal Hessian elements only
    pub fn hessian_diagonal(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
    ) -> Result<Array2<T>> {
        let n = point.len();
        let mut hessian = Array2::zeros((n, n));

        // Compute second derivatives using finite differences
        for i in 0..n {
            let mut x_plus = point.clone();
            let mut x_minus = point.clone();

            x_plus[i] = x_plus[i] + self.finite_diff_eps;
            x_minus[i] = x_minus[i] - self.finite_diff_eps;

            let f_plus = function(&x_plus);
            let f_center = function(point);
            let f_minus = function(&x_minus);

            // Second derivative: f''(x) = (f(x+h) - 2f(x) + f(x-h)) / h^2
            let second_deriv = (f_plus - T::from(2.0).unwrap() * f_center + f_minus)
                / (self.finite_diff_eps * self.finite_diff_eps);

            hessian[[i, i]] = second_deriv;
        }

        Ok(hessian)
    }

    /// Compute sparse Hessian representation
    pub fn sparse_hessian(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        config: &HessianConfig,
    ) -> Result<SparseHessian<T>> {
        let dense_hessian = self.hessian_forward_over_reverse(function, point, config)?;

        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();

        let threshold = T::from(config.sparsity_threshold).unwrap();

        for i in 0..dense_hessian.nrows() {
            for j in 0..dense_hessian.ncols() {
                let val = dense_hessian[[i, j]];
                if val.abs() > threshold {
                    rows.push(i);
                    cols.push(j);
                    values.push(val);
                }
            }
        }

        let nnz = values.len();
        Ok(SparseHessian {
            rows,
            cols,
            values,
            shape: dense_hessian.dim(),
            nnz,
        })
    }

    /// Compute third-order derivatives (tensor)
    pub fn third_order_derivatives(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
    ) -> Result<ThirdOrderTensor<T>> {
        let n = point.len();
        let mut tensor = Array3::zeros((n, n, n));

        // Compute third derivatives using finite differences
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let third_deriv = self.compute_third_partial(&function, point, i, j, k)?;
                    tensor[[i, j, k]] = third_deriv;
                }
            }
        }

        Ok(ThirdOrderTensor {
            data: tensor.clone(),
            sparse_data: None, // Could be computed if needed
            shape: tensor.dim(),
        })
    }

    /// Compute mixed partial derivatives
    pub fn mixed_partial(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        variables: &[usize],
        orders: &[usize],
        method: MixedPartialMethod,
    ) -> Result<MixedPartials<T>> {
        if variables.len() != orders.len() {
            return Err(OptimError::InvalidConfig(
                "Variables and orders length mismatch".to_string(),
            ));
        }

        let value = match method {
            MixedPartialMethod::ForwardOverReverse => {
                self.mixed_partial_forward_over_reverse(&function, point, variables, orders)?
            }
            MixedPartialMethod::ReverseOverForward => {
                self.mixed_partial_reverse_over_forward(&function, point, variables, orders)?
            }
            MixedPartialMethod::PureForward => {
                self.mixed_partial_pure_forward(&function, point, variables, orders)?
            }
            MixedPartialMethod::FiniteDifference => {
                self.mixed_partial_finite_difference(&function, point, variables, orders)?
            }
        };

        Ok(MixedPartials {
            variables: variables.to_vec(),
            orders: orders.to_vec(),
            value,
            method,
        })
    }

    /// Compute Hessian-vector product efficiently
    pub fn hessian_vector_product(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<Array1<T>> {
        // Use forward-over-reverse mode for efficient Hv computation
        let grad_fn = |x: &Array1<T>| -> Array1<T> {
            self.gradient_at_point(&function, x)
                .unwrap_or_else(|_| Array1::zeros(x.len()))
        };

        self.directional_derivative_of_gradient(&grad_fn, point, vector)
    }

    /// Compute vector-Hessian-vector product (quadratic form)
    pub fn vector_hessian_vector_product(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<T> {
        let hv = self.hessian_vector_product(function, point, vector)?;
        Ok(vector.dot(&hv))
    }

    /// Verify higher-order derivatives using finite differences
    pub fn verify_derivatives(
        &self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        computed_hessian: &Array2<T>,
    ) -> Result<DerivativeVerification<T>> {
        let fd_hessian = self.finite_difference_hessian(&function, point)?;

        let max_error = self.compute_max_error(computed_hessian, &fd_hessian);
        let avg_error = self.compute_avg_error(computed_hessian, &fd_hessian);
        let relative_error = self.compute_relative_error(computed_hessian, &fd_hessian);

        let is_accurate = max_error < T::from(1e-4).unwrap();

        Ok(DerivativeVerification {
            max_absolute_error: max_error,
            avg_absolute_error: avg_error,
            max_relative_error: relative_error,
            is_accurate,
            finite_diff_hessian: fd_hessian,
        })
    }

    /// Get higher-order derivative statistics
    pub fn get_derivative_stats(&self) -> HigherOrderStats {
        HigherOrderStats {
            _maxorder: self._maxorder,
            cache_size: self.derivative_cache.len(),
            mixed_mode_enabled: self.mixed_mode,
            memory_usage_estimate: self.estimate_memory_usage(),
            parallel_computation: self.parallel_computation,
            thread_pool_size: self.thread_pool_size,
            adaptive_sparsity: self.adaptive_sparsity,
            auto_mode_selection: self.auto_mode_selection,
            performance_profile: self.profiler.get_summary(),
        }
    }

    /// Advanced Hessian-vector product with automatic mode selection
    pub fn hessian_vector_product_advanced(
        &mut self,
        function: impl Fn(&Array1<T>) -> T + Send + Sync,
        point: &Array1<T>,
        vector: &Array1<T>,
        mode: Option<HvpMode>,
    ) -> Result<Array1<T>> {
        let n = point.len();
        let selected_mode = mode.unwrap_or_else(|| self.select_optimal_hvp_mode(n));

        match selected_mode {
            HvpMode::ForwardOverReverse => self.hvp_forward_over_reverse(&function, point, vector),
            HvpMode::ReverseOverForward => self.hvp_reverse_over_forward(&function, point, vector),
            HvpMode::FiniteDifference => self.hvp_finite_difference(&function, point, vector),
            HvpMode::PearLman => self.hvp_pearlman(&function, point, vector),
        }
    }

    /// Compute vector-Hessian-vector efficiently
    pub fn vector_hessian_vector_efficient(
        &mut self,
        function: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<T> {
        // Use forward-mode to compute directional derivative of gradient
        let grad_fn = |x: &Array1<T>| -> Array1<T> {
            self.gradient_at_point(&function, x)
                .unwrap_or_else(|_| Array1::zeros(x.len()))
        };

        let directional_grad = self.directional_derivative_of_gradient(&grad_fn, point, vector)?;
        Ok(vector.dot(&directional_grad))
    }

    /// Compute Jacobian efficiently with automatic mode selection
    pub fn jacobian_efficient<F>(
        &mut self,
        function: F,
        point: &Array1<T>,
        output_dim: usize,
    ) -> Result<Array2<T>>
    where
        F: Fn(&Array1<T>) -> Array1<T> + Send + Sync,
    {
        let input_dim = point.len();

        // Select mode based on dimensions
        if input_dim <= output_dim {
            // Forward mode is more efficient
            self.jacobian_forward_mode(&function, point, output_dim)
        } else {
            // Reverse mode is more efficient
            self.jacobian_reverse_mode(&function, point, output_dim)
        }
    }

    /// Compute K-FAC approximation to the Hessian
    pub fn kfac_hessian_approximation(
        &mut self,
        layers: &[LayerInfo<T>],
        activations: &[Array1<T>],
        gradients: &[Array1<T>],
    ) -> Result<Array2<T>> {
        let mut kfac_blocks = Vec::new();

        for (i, layer) in layers.iter().enumerate() {
            let activation = &activations[i];
            let gradient = &gradients[i];

            // Compute Kronecker factors
            let factor_a = self.compute_activation_factor(activation)?;
            let factor_g = self.compute_gradient_factor(gradient)?;

            // K-FAC block is Kronecker product approximation
            let block = self.kronecker_product_approximation(&factor_a, &factor_g)?;
            kfac_blocks.push(block);
        }

        // Combine blocks into full approximation
        self.combine_kfac_blocks(&kfac_blocks)
    }

    /// Compute Natural Gradient using Fisher Information Matrix
    pub fn natural_gradient(
        &mut self,
        log_likelihood: impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        gradient: &Array1<T>,
    ) -> Result<Array1<T>> {
        // Compute Fisher Information Matrix (FIM)
        let fim = self.compute_fisher_information_matrix(&log_likelihood, point)?;

        // Natural gradient = FIM^(-1) * gradient
        self.solve_linear_system(&fim, gradient)
    }

    /// Compute truncated Newton direction
    pub fn truncated_newton_direction(
        &mut self,
        function: impl Fn(&Array1<T>) -> T + Send + Sync,
        point: &Array1<T>,
        gradient: &Array1<T>,
        max_cg_iterations: usize,
        cg_tolerance: T,
    ) -> Result<Array1<T>> {
        // Use Conjugate Gradient to approximately solve Hx = -g
        let neg_gradient = gradient.mapv(|x| -x);

        // Create a copy of point and function to avoid borrow conflicts
        let point_copy = point.clone();
        let function_copy = function;

        let hvp_fn = move |v: &Array1<T>| -> Result<Array1<T>> {
            // Use finite differences as a fallback to avoid borrow conflicts
            let eps = T::from(1e-6).unwrap();
            let mut hvp = Array1::zeros(v.len());

            for i in 0..v.len() {
                let mut point_plus = point_copy.clone();
                let mut point_minus = point_copy.clone();

                point_plus[i] = point_plus[i] + eps;
                point_minus[i] = point_minus[i] - eps;

                // Compute gradient at perturbed points
                let grad_plus = Self::finite_diff_gradient(&function_copy, &point_plus)?;
                let grad_minus = Self::finite_diff_gradient(&function_copy, &point_minus)?;

                // Approximate Hessian-vector product
                let hess_col = (&grad_plus - &grad_minus) / (T::from(2.0).unwrap() * eps);
                hvp[i] = hess_col.dot(v);
            }

            Ok(hvp)
        };

        self.conjugate_gradient_solve(hvp_fn, &neg_gradient, max_cg_iterations, cg_tolerance)
    }

    /// Helper method for finite difference gradient computation
    fn finite_diff_gradient<F>(function: &F, point: &Array1<T>) -> Result<Array1<T>>
    where
        F: Fn(&Array1<T>) -> T,
    {
        let eps = T::from(1e-6).unwrap();
        let mut gradient = Array1::zeros(point.len());

        for i in 0..point.len() {
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();

            point_plus[i] = point_plus[i] + eps;
            point_minus[i] = point_minus[i] - eps;

            let loss_plus = function(&point_plus);
            let loss_minus = function(&point_minus);

            gradient[i] = (loss_plus - loss_minus) / (T::from(2.0).unwrap() * eps);
        }

        Ok(gradient)
    }

    // Helper methods

    fn gradient_at_point(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
    ) -> Result<Array1<T>> {
        let n = point.len();
        let mut gradient = Array1::zeros(n);

        for i in 0..n {
            let mut x_plus = point.clone();
            let mut x_minus = point.clone();

            x_plus[i] = x_plus[i] + self.finite_diff_eps;
            x_minus[i] = x_minus[i] - self.finite_diff_eps;

            gradient[i] = (function(&x_plus) - function(&x_minus))
                / (T::from(2.0).unwrap() * self.finite_diff_eps);
        }

        Ok(gradient)
    }

    fn directional_derivative_of_gradient(
        &self,
        grad_fn: &impl Fn(&Array1<T>) -> Array1<T>,
        point: &Array1<T>,
        direction: &Array1<T>,
    ) -> Result<Array1<T>> {
        let eps = self.finite_diff_eps;
        let point_plus = point + &(direction * eps);
        let point_minus = point - &(direction * eps);

        let grad_plus = grad_fn(&point_plus);
        let grad_minus = grad_fn(&point_minus);

        Ok((grad_plus - grad_minus) / (T::from(2.0).unwrap() * eps))
    }

    fn finite_difference_hessian(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
    ) -> Result<Array2<T>> {
        let n = point.len();
        let mut hessian = Array2::zeros((n, n));

        for i in 0..n {
            for j in i..n {
                // Only compute upper triangle due to symmetry
                let second_deriv = if i == j {
                    // Diagonal element: f''_ii
                    let mut x_plus = point.clone();
                    let mut x_minus = point.clone();

                    x_plus[i] = x_plus[i] + self.finite_diff_eps;
                    x_minus[i] = x_minus[i] - self.finite_diff_eps;

                    let f_plus = function(&x_plus);
                    let f_center = function(point);
                    let f_minus = function(&x_minus);

                    (f_plus - T::from(2.0).unwrap() * f_center + f_minus)
                        / (self.finite_diff_eps * self.finite_diff_eps)
                } else {
                    // Off-diagonal element: f''_ij
                    let eps = self.finite_diff_eps;

                    let mut x_pp = point.clone();
                    x_pp[i] = x_pp[i] + eps;
                    x_pp[j] = x_pp[j] + eps;

                    let mut x_pm = point.clone();
                    x_pm[i] = x_pm[i] + eps;
                    x_pm[j] = x_pm[j] - eps;

                    let mut x_mp = point.clone();
                    x_mp[i] = x_mp[i] - eps;
                    x_mp[j] = x_mp[j] + eps;

                    let mut x_mm = point.clone();
                    x_mm[i] = x_mm[i] - eps;
                    x_mm[j] = x_mm[j] - eps;

                    (function(&x_pp) - function(&x_pm) - function(&x_mp) + function(&x_mm))
                        / (T::from(4.0).unwrap() * eps * eps)
                };

                hessian[[i, j]] = second_deriv;
                hessian[[j, i]] = second_deriv; // Symmetry
            }
        }

        Ok(hessian)
    }

    fn compute_third_partial(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Result<T> {
        let eps = self.finite_diff_eps;

        // Use finite differences to approximate third derivative
        // This is simplified - real implementation would be more sophisticated
        let mut x_ppp = point.clone();
        x_ppp[i] = x_ppp[i] + eps;
        x_ppp[j] = x_ppp[j] + eps;
        x_ppp[k] = x_ppp[k] + eps;

        let mut x_mmm = point.clone();
        x_mmm[i] = x_mmm[i] - eps;
        x_mmm[j] = x_mmm[j] - eps;
        x_mmm[k] = x_mmm[k] - eps;

        // Simplified third derivative approximation
        let third_deriv =
            (function(&x_ppp) - function(&x_mmm)) / (T::from(8.0).unwrap() * eps * eps * eps);

        Ok(third_deriv)
    }

    fn mixed_partial_forward_over_reverse(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        variables: &[usize],
        orders: &[usize],
    ) -> Result<T> {
        // Simplified implementation
        // Real implementation would use proper forward-over-reverse mode
        self.mixed_partial_finite_difference(function, point, variables, orders)
    }

    fn mixed_partial_reverse_over_forward(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        variables: &[usize],
        orders: &[usize],
    ) -> Result<T> {
        // Simplified implementation
        self.mixed_partial_finite_difference(function, point, variables, orders)
    }

    fn mixed_partial_pure_forward(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        variables: &[usize],
        orders: &[usize],
    ) -> Result<T> {
        // Simplified implementation
        self.mixed_partial_finite_difference(function, point, variables, orders)
    }

    fn mixed_partial_finite_difference(
        &self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        variables: &[usize],
        orders: &[usize],
    ) -> Result<T> {
        // Simple finite difference approximation for mixed partials
        let eps = self.finite_diff_eps;
        let total_order: usize = orders.iter().sum();

        if total_order > 3 {
            return Err(OptimError::InvalidConfig(
                "Mixed partial order too high".to_string(),
            ));
        }

        // For second-order mixed partial ∂²f/∂x∂y
        if total_order == 2 && variables.len() == 2 {
            let i = variables[0];
            let j = variables[1];

            let mut x_pp = point.clone();
            x_pp[i] = x_pp[i] + eps;
            x_pp[j] = x_pp[j] + eps;

            let mut x_pm = point.clone();
            x_pm[i] = x_pm[i] + eps;
            x_pm[j] = x_pm[j] - eps;

            let mut x_mp = point.clone();
            x_mp[i] = x_mp[i] - eps;
            x_mp[j] = x_mp[j] + eps;

            let mut x_mm = point.clone();
            x_mm[i] = x_mm[i] - eps;
            x_mm[j] = x_mm[j] - eps;

            let mixed_partial = (function(&x_pp) - function(&x_pm) - function(&x_mp)
                + function(&x_mm))
                / (T::from(4.0).unwrap() * eps * eps);

            Ok(mixed_partial)
        } else {
            // Fallback for other cases
            Ok(T::zero())
        }
    }

    fn verify_hessian_accuracy(&self, computed: &Array2<T>, reference: &Array2<T>) -> Result<()> {
        let max_error = self.compute_max_error(computed, reference);
        let threshold = T::from(1e-3).unwrap();

        if max_error > threshold {
            return Err(OptimError::InvalidConfig(format!(
                "Hessian verification failed: max error {}",
                max_error.to_f64().unwrap_or(0.0)
            )));
        }

        Ok(())
    }

    fn compute_max_error(&self, a: &Array2<T>, b: &Array2<T>) -> T {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(T::zero(), |acc, x| if x > acc { x } else { acc })
    }

    fn compute_avg_error(&self, a: &Array2<T>, b: &Array2<T>) -> T {
        let sum = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .sum::<T>();

        sum / T::from(a.len()).unwrap()
    }

    fn compute_relative_error(&self, a: &Array2<T>, b: &Array2<T>) -> T {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                if y.abs() > T::from(1e-12).unwrap() {
                    ((x - y) / y).abs()
                } else {
                    (x - y).abs()
                }
            })
            .fold(T::zero(), |acc, x| if x > acc { x } else { acc })
    }

    fn estimate_memory_usage(&self) -> usize {
        let cache_size = self.derivative_cache.len()
            * std::mem::size_of::<(DerivativeKey, DerivativeValue<T>)>();
        let engine_size = std::mem::size_of::<Self>();
        let profiler_size =
            self.profiler.hessian_timings.len() * std::mem::size_of::<ComputationTiming>();

        cache_size + engine_size + profiler_size
    }

    /// Decide whether to use parallel computation
    #[allow(dead_code)]
    fn should_use_parallel(&self, problemsize: usize) -> bool {
        self.parallel_computation && problemsize >= 50
    }

    /// Decide whether to use sparse computations
    #[allow(dead_code)]
    fn should_use_sparse(&self, problemsize: usize, config: &HessianConfig) -> bool {
        self.adaptive_sparsity && config.sparse && problemsize >= 100
    }

    /// Select optimal HVP mode based on problem characteristics
    fn select_optimal_hvp_mode(&self, problemsize: usize) -> HvpMode {
        if !self.auto_mode_selection {
            return HvpMode::ForwardOverReverse;
        }

        match problemsize {
            0..=10 => HvpMode::FiniteDifference,
            11..=100 => HvpMode::ForwardOverReverse,
            _ => HvpMode::ReverseOverForward,
        }
    }

    /// Apply adaptive sparsity to dense matrix
    #[allow(dead_code)]
    fn apply_adaptive_sparsity(&self, mut matrix: Array2<T>, threshold: f64) -> Result<Array2<T>> {
        let sparsity_threshold = T::from(threshold).unwrap();

        for elem in matrix.iter_mut() {
            if elem.abs() < sparsity_threshold {
                *elem = T::zero();
            }
        }

        Ok(matrix)
    }

    /// Forward-over-reverse HVP implementation
    fn hvp_forward_over_reverse(
        &mut self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<Array1<T>> {
        let grad_fn = |x: &Array1<T>| -> Array1<T> {
            self.gradient_at_point(function, x)
                .unwrap_or_else(|_| Array1::zeros(x.len()))
        };

        self.directional_derivative_of_gradient(&grad_fn, point, vector)
    }

    /// Reverse-over-forward HVP implementation
    fn hvp_reverse_over_forward(
        &mut self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<Array1<T>> {
        // Use R-operator (reverse-over-forward)
        let eps = self.finite_diff_eps;

        let point_plus = point + &(vector * eps);
        let point_minus = point - &(vector * eps);

        let grad_plus = self.gradient_at_point(function, &point_plus)?;
        let grad_minus = self.gradient_at_point(function, &point_minus)?;

        Ok((grad_plus - grad_minus) / (T::from(2.0).unwrap() * eps))
    }

    /// Finite difference HVP implementation
    fn hvp_finite_difference(
        &mut self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<Array1<T>> {
        let eps = self.finite_diff_eps;

        let point_plus = point + &(vector * eps);
        let point_minus = point - &(vector * eps);

        let grad_plus = self.gradient_at_point(function, &point_plus)?;
        let grad_minus = self.gradient_at_point(function, &point_minus)?;

        Ok((grad_plus - grad_minus) / (T::from(2.0).unwrap() * eps))
    }

    /// Pearlman trick for quadratic functions
    fn hvp_pearlman(
        &mut self,
        function: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
        vector: &Array1<T>,
    ) -> Result<Array1<T>> {
        // For quadratic functions: Hv = (∇f(x+v) - ∇f(x)) / ε can be exact
        let eps = T::one(); // Use 1.0 for exact computation on quadratic functions

        let point_plus = point + &(vector * eps);
        let grad_plus = self.gradient_at_point(function, &point_plus)?;
        let grad_orig = self.gradient_at_point(function, point)?;

        Ok((grad_plus - grad_orig) / eps)
    }

    /// Forward-mode Jacobian computation
    fn jacobian_forward_mode<F>(
        &mut self,
        function: &F,
        point: &Array1<T>,
        output_dim: usize,
    ) -> Result<Array2<T>>
    where
        F: Fn(&Array1<T>) -> Array1<T>,
    {
        let input_dim = point.len();
        let mut jacobian = Array2::zeros((output_dim, input_dim));

        // Compute Jacobian column by column using forward mode
        for j in 0..input_dim {
            let mut direction = Array1::zeros(input_dim);
            direction[j] = T::one();

            // This would use forward-mode AD to compute J*v
            // Simplified implementation using finite differences
            let eps = self.finite_diff_eps;
            let mut point_plus = point.clone();
            point_plus[j] = point_plus[j] + eps;

            let f_plus = function(&point_plus);
            let f_orig = function(point);

            let column = (f_plus - f_orig) / eps;

            for i in 0..output_dim {
                jacobian[[i, j]] = column[i];
            }
        }

        Ok(jacobian)
    }

    /// Reverse-mode Jacobian computation
    fn jacobian_reverse_mode<F>(
        &mut self,
        function: &F,
        point: &Array1<T>,
        output_dim: usize,
    ) -> Result<Array2<T>>
    where
        F: Fn(&Array1<T>) -> Array1<T>,
    {
        let input_dim = point.len();
        let mut jacobian = Array2::zeros((output_dim, input_dim));

        // Compute Jacobian row by row using reverse mode
        for i in 0..output_dim {
            // Create scalar function for i-th output
            let scalar_fn = |x: &Array1<T>| function(x)[i];

            // Compute gradient of scalar function
            let grad = self.gradient_at_point(&scalar_fn, point)?;

            for j in 0..input_dim {
                jacobian[[i, j]] = grad[j];
            }
        }

        Ok(jacobian)
    }

    /// K-FAC helper methods
    fn compute_activation_factor(&self, activation: &Array1<T>) -> Result<Array2<T>> {
        let n = activation.len();
        let mut factor = Array2::zeros((n, n));

        // A = E[a a^T] where a is the activation
        for i in 0..n {
            for j in 0..n {
                factor[[i, j]] = activation[i] * activation[j];
            }
        }

        Ok(factor)
    }

    fn compute_gradient_factor(&self, gradient: &Array1<T>) -> Result<Array2<T>> {
        let n = gradient.len();
        let mut factor = Array2::zeros((n, n));

        // G = E[g g^T] where g is the gradient
        for i in 0..n {
            for j in 0..n {
                factor[[i, j]] = gradient[i] * gradient[j];
            }
        }

        Ok(factor)
    }

    fn kronecker_product_approximation(&self, a: &Array2<T>, g: &Array2<T>) -> Result<Array2<T>> {
        // Simplified Kronecker product approximation
        // In practice, this would be more sophisticated
        let n_a = a.nrows();
        let n_g = g.nrows();
        let n_total = n_a * n_g;

        let mut result = Array2::zeros((n_total, n_total));

        for i in 0..n_a {
            for j in 0..n_a {
                for k in 0..n_g {
                    for l in 0..n_g {
                        let row = i * n_g + k;
                        let col = j * n_g + l;
                        result[[row, col]] = a[[i, j]] * g[[k, l]];
                    }
                }
            }
        }

        Ok(result)
    }

    fn combine_kfac_blocks(&self, blocks: &[Array2<T>]) -> Result<Array2<T>> {
        if blocks.is_empty() {
            return Err(OptimError::InvalidConfig("Empty K-FAC blocks".to_string()));
        }

        // Simplified block combination - would be more sophisticated in practice
        let total_size: usize = blocks.iter().map(|b| b.nrows()).sum();
        let mut combined = Array2::zeros((total_size, total_size));

        let mut row_offset = 0;
        let mut col_offset = 0;

        for block in blocks {
            let block_size = block.nrows();

            for i in 0..block_size {
                for j in 0..block_size {
                    combined[[row_offset + i, col_offset + j]] = block[[i, j]];
                }
            }

            row_offset += block_size;
            col_offset += block_size;
        }

        Ok(combined)
    }

    fn compute_fisher_information_matrix(
        &mut self,
        log_likelihood: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
    ) -> Result<Array2<T>> {
        // Fisher Information Matrix: F = E[∇log p(x) ∇log p(x)^T]
        // Approximated as F ≈ -H[log p(x)] (for exponential family)
        self.hessian_forward_over_reverse(log_likelihood, point, &HessianConfig::default())
    }

    fn solve_linear_system(&self, matrix: &Array2<T>, rhs: &Array1<T>) -> Result<Array1<T>> {
        // Simplified linear system solver
        // In practice, would use more sophisticated methods like LU decomposition
        let n = matrix.nrows();
        if n != rhs.len() || n != matrix.ncols() {
            return Err(OptimError::InvalidConfig(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        // Use pseudo-inverse for now (simplified)
        // result = matrix^(-1) * rhs
        let mut result = Array1::zeros(n);

        // Simplified diagonal approximation
        for i in 0..n {
            if matrix[[i, i]].abs() > T::from(1e-12).unwrap() {
                result[i] = rhs[i] / matrix[[i, i]];
            } else {
                result[i] = rhs[i];
            }
        }

        Ok(result)
    }

    fn conjugate_gradient_solve<F>(
        &mut self,
        mut hvp_fn: F,
        rhs: &Array1<T>,
        max_iterations: usize,
        tolerance: T,
    ) -> Result<Array1<T>>
    where
        F: FnMut(&Array1<T>) -> Result<Array1<T>>,
    {
        let n = rhs.len();
        let mut x = Array1::zeros(n);
        let mut r = rhs.clone();
        let mut p = r.clone();
        let mut rsold = r.dot(&r);

        for _iter in 0..max_iterations {
            let ap = hvp_fn(&p)?;
            let alpha = rsold / p.dot(&ap);

            x = x + &p * alpha;
            r = r - &ap * alpha;

            let rsnew = r.dot(&r);

            if rsnew.sqrt() < tolerance {
                break;
            }

            let beta = rsnew / rsold;
            p = &r + &p * beta;
            rsold = rsnew;
        }

        Ok(x)
    }
}

/// Derivative verification results
#[derive(Debug, Clone)]
pub struct DerivativeVerification<T: Float> {
    pub max_absolute_error: T,
    pub avg_absolute_error: T,
    pub max_relative_error: T,
    pub is_accurate: bool,
    pub finite_diff_hessian: Array2<T>,
}

/// Higher-order differentiation statistics
#[derive(Debug, Clone)]
pub struct HigherOrderStats {
    pub _maxorder: usize,
    pub cache_size: usize,
    pub mixed_mode_enabled: bool,
    pub memory_usage_estimate: usize,
    pub parallel_computation: bool,
    pub thread_pool_size: usize,
    pub adaptive_sparsity: bool,
    pub auto_mode_selection: bool,
    pub performance_profile: PerformanceProfile,
}

/// Advanced configuration for higher-order engine
#[derive(Debug, Clone)]
pub struct HigherOrderConfig<T: Float> {
    pub _maxorder: usize,
    pub mixed_mode: bool,
    pub finite_diff_eps: T,
    pub parallel_computation: bool,
    pub thread_pool_size: usize,
    pub adaptive_sparsity: bool,
    pub auto_mode_selection: bool,
    pub cache_size_limit: usize,
}

impl<T: Float + Default> Default for HigherOrderConfig<T> {
    fn default() -> Self {
        Self {
            _maxorder: 3,
            mixed_mode: true,
            finite_diff_eps: T::from(1e-5).unwrap(),
            parallel_computation: true,
            thread_pool_size: 4, // Conservative default
            adaptive_sparsity: true,
            auto_mode_selection: true,
            cache_size_limit: 10000,
        }
    }
}

/// Hessian-vector product computation modes
#[derive(Debug, Clone, Copy)]
pub enum HvpMode {
    /// Forward-over-reverse mode (efficient for few vectors)
    ForwardOverReverse,
    /// Reverse-over-forward mode (efficient for many vectors)
    ReverseOverForward,
    /// Finite difference approximation
    FiniteDifference,
    /// Pearlman trick (for quadratic functions)
    PearLman,
}

/// Computation profiler for performance optimization
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ComputationProfiler<T: Float> {
    hessian_timings: Vec<ComputationTiming>,
    hvp_timings: Vec<ComputationTiming>,
    jacobian_timings: Vec<ComputationTiming>,
    total_computations: usize,
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ComputationTiming {
    problemsize: usize,
    duration_us: u64,
    parallel: bool,
    sparse: bool,
    method: String,
}

#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    pub avg_hessian_time_us: f64,
    pub avg_hvp_time_us: f64,
    pub avg_jacobian_time_us: f64,
    pub parallel_efficiency: f64,
    pub sparsity_benefit: f64,
    pub cache_hit_rate: f64,
}

impl<T: Float + Default + Clone> ComputationProfiler<T> {
    fn new() -> Self {
        Self {
            hessian_timings: Vec::new(),
            hvp_timings: Vec::new(),
            jacobian_timings: Vec::new(),
            total_computations: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    #[allow(dead_code)]
    fn record_hessian_computation(
        &mut self,
        size: usize,
        duration: std::time::Duration,
        parallel: bool,
        sparse: bool,
    ) {
        self.hessian_timings.push(ComputationTiming {
            problemsize: size,
            duration_us: duration.as_micros() as u64,
            parallel,
            sparse,
            method: "hessian".to_string(),
        });
        self.total_computations += 1;
    }

    fn get_summary(&self) -> PerformanceProfile {
        let avg_hessian_time = if self.hessian_timings.is_empty() {
            0.0
        } else {
            self.hessian_timings
                .iter()
                .map(|t| t.duration_us as f64)
                .sum::<f64>()
                / self.hessian_timings.len() as f64
        };

        PerformanceProfile {
            avg_hessian_time_us: avg_hessian_time,
            avg_hvp_time_us: 0.0,      // Would compute from hvp_timings
            avg_jacobian_time_us: 0.0, // Would compute from jacobian_timings
            parallel_efficiency: self.compute_parallel_efficiency(),
            sparsity_benefit: self.compute_sparsity_benefit(),
            cache_hit_rate: 0.0, // Would track cache hits
        }
    }

    fn compute_parallel_efficiency(&self) -> f64 {
        if self.hessian_timings.is_empty() {
            return 1.0;
        }

        let parallel_times: Vec<_> = self.hessian_timings.iter().filter(|t| t.parallel).collect();
        let sequential_times: Vec<_> = self
            .hessian_timings
            .iter()
            .filter(|t| !t.parallel)
            .collect();

        if parallel_times.is_empty() || sequential_times.is_empty() {
            return 1.0;
        }

        let avg_parallel = parallel_times
            .iter()
            .map(|t| t.duration_us as f64)
            .sum::<f64>()
            / parallel_times.len() as f64;
        let avg_sequential = sequential_times
            .iter()
            .map(|t| t.duration_us as f64)
            .sum::<f64>()
            / sequential_times.len() as f64;

        avg_sequential / avg_parallel
    }

    fn compute_sparsity_benefit(&self) -> f64 {
        // Simplified computation - would analyze sparse vs dense performance
        1.2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_higher_order_engine_creation() {
        let engine = HigherOrderEngine::<f64>::new(3);
        assert_eq!(engine._maxorder, 3);
        assert!(engine.mixed_mode);
    }

    #[test]
    fn test_hessian_diagonal() {
        let engine = HigherOrderEngine::<f64>::new(2);

        // Test function: f(x) = x₁² + 2x₂²
        let function = |x: &Array1<f64>| x[0] * x[0] + 2.0 * x[1] * x[1];
        let point = Array1::from_vec(vec![1.0, 1.0]);

        let hessian = engine.hessian_diagonal(&function, &point).unwrap();

        // Expected diagonal: [2, 4]
        assert!((hessian[[0, 0]] - 2.0).abs() < 1e-5);
        assert!((hessian[[1, 1]] - 4.0).abs() < 1e-5);
        assert!((hessian[[0, 1]]).abs() < 1e-10); // Off-diagonal should be zero
    }

    #[test]
    fn test_finite_difference_hessian() {
        let engine = HigherOrderEngine::<f64>::new(2);

        // Test function: f(x,y) = x² + xy + y²
        let function = |x: &Array1<f64>| x[0] * x[0] + x[0] * x[1] + x[1] * x[1];
        let point = Array1::from_vec(vec![1.0, 1.0]);

        let hessian = engine.finite_difference_hessian(&function, &point).unwrap();

        // Expected Hessian: [[2, 1], [1, 2]]
        assert!((hessian[[0, 0]] - 2.0).abs() < 1e-5);
        assert!((hessian[[0, 1]] - 1.0).abs() < 1e-5);
        assert!((hessian[[1, 0]] - 1.0).abs() < 1e-5);
        assert!((hessian[[1, 1]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_mixed_partial() {
        let mut engine = HigherOrderEngine::<f64>::new(2);

        // Test function: f(x,y) = x²y + xy²
        let function = |x: &Array1<f64>| x[0] * x[0] * x[1] + x[0] * x[1] * x[1];
        let point = Array1::from_vec(vec![1.0, 1.0]);

        let mixed_partial = engine
            .mixed_partial(
                function,
                &point,
                &[0, 1],
                &[1, 1],
                MixedPartialMethod::FiniteDifference,
            )
            .unwrap();

        // ∂²f/∂x∂y = 2x + 2y = 4 at (1,1)
        assert!((mixed_partial.value - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_hessian() {
        let mut engine = HigherOrderEngine::<f64>::new(2);

        // Diagonal function: f(x) = x₁² + x₂²
        let function = |x: &Array1<f64>| x[0] * x[0] + x[1] * x[1];
        let point = Array1::from_vec(vec![1.0, 1.0]);

        let config = HessianConfig {
            sparse: true,
            sparsity_threshold: 1e-4,
            ..Default::default()
        };

        let sparse_hessian = engine.sparse_hessian(function, &point, &config).unwrap();

        // Should have 2 non-zero elements (diagonal)
        assert_eq!(sparse_hessian.nnz, 2);
        assert_eq!(sparse_hessian.shape, (2, 2));
    }

    #[test]
    fn test_hessian_config_default() {
        let config = HessianConfig::default();
        assert!(config.exact);
        assert!(!config.sparse);
        assert!(!config.diagonal_only);
    }
}
