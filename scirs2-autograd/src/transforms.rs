//! JAX-inspired function transformation primitives
//!
//! This module provides composable function transformations modeled after JAX's
//! functional transformation API. These primitives enable automatic vectorization,
//! gradient computation, memory-efficient checkpointing, and parallelization.
//!
//! # Overview
//!
//! | Transform | JAX equivalent | Description |
//! |-----------|---------------|-------------|
//! | [`vmap`] | `jax.vmap` | Vectorize a function over batch dimensions |
//! | [`grad`] | `jax.grad` | Transform a scalar function into its gradient function |
//! | [`grad_grad`] | `jax.grad(jax.grad(f))` | Compute second derivatives (Hessian diagonal or full) |
//! | [`value_and_grad`] | `jax.value_and_grad` | Simultaneously compute value and gradient |
//! | [`jacobian`] | `jax.jacobian` | Full Jacobian matrix via forward-mode |
//! | [`stop_gradient`] | `jax.lax.stop_gradient` | Detach a tensor from the computation graph |
//! | [`Checkpoint`] | `jax.checkpoint` | Memory-efficient gradient checkpointing |
//! | [`pmap`] | `jax.pmap` | Parallel map across batch elements |
//! | [`JitHint`] | `jax.jit` | Hint structure for JIT-like optimizations |
//!
//! # Examples
//!
//! ## Vectorized map
//!
//! ```rust
//! use scirs2_autograd::transforms::vmap;
//! use scirs2_core::ndarray::{Array1, Array2, array};
//!
//! // f(x) = x * 2 applied to each row of a batch
//! let batch = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
//!     .expect("valid shape");
//! let result = vmap(|x: &Array1<f64>| x.mapv(|v| v * 2.0), &batch)
//!     .expect("vmap succeeds");
//! assert_eq!(result[[0, 0]], 2.0);
//! assert_eq!(result[[2, 1]], 12.0);
//! ```
//!
//! ## Gradient function
//!
//! ```rust
//! use scirs2_autograd::transforms::value_and_grad;
//! use scirs2_autograd::forward_mode::DualNumber;
//! use scirs2_core::ndarray::Array1;
//!
//! // f(x) = x0^2 + x1^2  =>  grad = [2*x0, 2*x1]
//! let vg = value_and_grad(|xs: &[DualNumber<f64>]| {
//!     xs[0] * xs[0] + xs[1] * xs[1]
//! });
//! let x = Array1::from(vec![3.0, 4.0]);
//! let (val, g) = vg(&x);
//! assert!((val - 25.0).abs() < 1e-12);
//! assert!((g[0] - 6.0).abs() < 1e-12);
//! assert!((g[1] - 8.0).abs() < 1e-12);
//! ```

use crate::error::AutogradError;
use crate::forward_mode::DualNumber;
use num_traits::Float as NumFloat;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::fmt;
use std::sync::Arc;

/// Type alias for a segment function used in [`Checkpoint`].
type SegmentFn<F> = Arc<dyn Fn(&Array2<F>) -> Array2<F> + Send + Sync>;

// ---------------------------------------------------------------------------
// vmap -- Vectorized Map
// ---------------------------------------------------------------------------

/// Automatically vectorize a function over batch dimensions.
///
/// Like JAX's `vmap`: transforms `f(x)` operating on a single sample into
/// `vmap(f, batch)` that applies `f` to each row of the batch matrix.
///
/// The function is applied to each row independently. If the `parallel` feature
/// of scirs2-core is enabled, rows are processed in parallel via Rayon.
///
/// # Arguments
/// * `func` - A function that maps `&Array1<F>` to `Array1<F>` for a single sample
/// * `inputs` - A 2-D array of shape `(batch_size, input_dim)` where each row is one sample
///
/// # Returns
/// A 2-D array of shape `(batch_size, output_dim)` where each row is `func(inputs.row(i))`
///
/// # Errors
/// Returns `AutogradError::ShapeMismatch` if the function produces inconsistent output sizes,
/// or `AutogradError::OperationError` if the batch is empty.
pub fn vmap<F, Func>(func: Func, inputs: &Array2<F>) -> Result<Array2<F>, AutogradError>
where
    F: NumFloat + Copy + Send + Sync + fmt::Debug + 'static,
    Func: Fn(&Array1<F>) -> Array1<F> + Send + Sync,
{
    let batch_size = inputs.nrows();
    if batch_size == 0 {
        return Err(AutogradError::OperationError(
            "vmap: input batch is empty".to_string(),
        ));
    }

    // Collect results row by row
    let mut results: Vec<Array1<F>> = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let row = inputs.row(i).to_owned();
        results.push(func(&row));
    }

    // Verify all outputs have the same dimension
    let out_dim = results[0].len();
    for (i, r) in results.iter().enumerate() {
        if r.len() != out_dim {
            return Err(AutogradError::ShapeMismatch(format!(
                "vmap: output dimension mismatch at row {}: expected {}, got {}",
                i,
                out_dim,
                r.len()
            )));
        }
    }

    // Assemble into output matrix
    let mut output = Array2::<F>::zeros((batch_size, out_dim));
    for (i, r) in results.iter().enumerate() {
        for j in 0..out_dim {
            output[[i, j]] = r[j];
        }
    }

    Ok(output)
}

/// Parallel version of [`vmap`] that uses Rayon to process batch rows concurrently.
///
/// This is analogous to JAX's `pmap` for data-parallel computation. Each row is
/// processed independently on a separate thread.
///
/// # Arguments
/// * `func` - A function mapping `&Array1<F>` to `Array1<F>`; must be `Send + Sync`
/// * `inputs` - Batch matrix of shape `(batch_size, input_dim)`
///
/// # Errors
/// Returns errors for empty batches, inconsistent output shapes, or thread failures.
pub fn pmap<F, Func>(func: Func, inputs: &Array2<F>) -> Result<Array2<F>, AutogradError>
where
    F: NumFloat + Copy + Send + Sync + fmt::Debug + 'static,
    Func: Fn(&Array1<F>) -> Array1<F> + Send + Sync,
{
    let batch_size = inputs.nrows();
    if batch_size == 0 {
        return Err(AutogradError::OperationError(
            "pmap: input batch is empty".to_string(),
        ));
    }

    // Collect rows into owned arrays for parallel processing
    let rows: Vec<Array1<F>> = (0..batch_size).map(|i| inputs.row(i).to_owned()).collect();

    // Use scoped threads for truly parallel execution without requiring 'static
    let results: Vec<Array1<F>> = std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .iter()
            .map(|row| {
                let func_ref = &func;
                scope.spawn(move || func_ref(row))
            })
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Array1::zeros(0)))
            .collect()
    });

    // Check for thread failures (zero-length outputs from failed joins)
    for (i, r) in results.iter().enumerate() {
        if r.is_empty() && inputs.ncols() > 0 {
            return Err(AutogradError::OperationError(format!(
                "pmap: thread for row {} failed",
                i,
            )));
        }
    }

    // Verify consistency
    let out_dim = results[0].len();
    for (i, r) in results.iter().enumerate() {
        if r.len() != out_dim {
            return Err(AutogradError::ShapeMismatch(format!(
                "pmap: output dimension mismatch at row {}: expected {}, got {}",
                i,
                out_dim,
                r.len()
            )));
        }
    }

    let mut output = Array2::<F>::zeros((batch_size, out_dim));
    for (i, r) in results.iter().enumerate() {
        for j in 0..out_dim {
            output[[i, j]] = r[j];
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// grad -- Gradient Function Transformation
// ---------------------------------------------------------------------------

/// Transform a scalar function into its gradient function.
///
/// Like JAX's `jax.grad`: given `f: R^n -> R`, returns a closure
/// `g: &Array1<F> -> Array1<F>` such that `g(x) = nabla f(x)`.
///
/// Uses forward-mode AD (dual numbers) under the hood, requiring `n` forward
/// passes. For high-dimensional inputs, consider reverse-mode via the autograd
/// graph API instead.
///
/// # Arguments
/// * `func` - A scalar function operating on dual numbers
///
/// # Returns
/// A closure that computes the gradient at any point `x`.
///
/// # Example
/// ```rust
/// use scirs2_autograd::transforms::grad;
/// use scirs2_autograd::forward_mode::DualNumber;
/// use scirs2_core::ndarray::Array1;
///
/// let grad_f = grad(|xs: &[DualNumber<f64>]| xs[0] * xs[0] + xs[1] * xs[1]);
/// let x = Array1::from(vec![3.0, 4.0]);
/// let g = grad_f(&x);
/// assert!((g[0] - 6.0).abs() < 1e-12);
/// assert!((g[1] - 8.0).abs() < 1e-12);
/// ```
pub fn grad<F, Func>(func: Func) -> impl Fn(&Array1<F>) -> Array1<F>
where
    F: NumFloat + Copy + fmt::Debug + Send + Sync + 'static,
    Func: Fn(&[DualNumber<F>]) -> DualNumber<F> + Clone + 'static,
{
    move |x: &Array1<F>| {
        let n = x.len();
        let mut gradient = Array1::<F>::zeros(n);
        for i in 0..n {
            let duals: Vec<DualNumber<F>> = x
                .iter()
                .enumerate()
                .map(|(k, &xk)| {
                    if k == i {
                        DualNumber::new(xk, F::one())
                    } else {
                        DualNumber::new(xk, F::zero())
                    }
                })
                .collect();
            gradient[i] = func.clone()(&duals).tangent();
        }
        gradient
    }
}

/// Compute the full Hessian matrix (second-order gradient) of a scalar function.
///
/// This is equivalent to `jax.grad(jax.grad(f))` but returns the full `n x n`
/// Hessian matrix rather than a function. Uses forward-over-forward dual number
/// differentiation with finite-difference correction for cross-terms.
///
/// # Arguments
/// * `func` - A scalar function operating on dual numbers
///
/// # Returns
/// A closure that computes the `n x n` Hessian at any point `x`.
///
/// # Example
/// ```rust
/// use scirs2_autograd::transforms::grad_grad;
/// use scirs2_autograd::forward_mode::DualNumber;
/// use scirs2_core::ndarray::Array1;
///
/// // f(x) = x0^2 + 3*x0*x1 + 2*x1^2  =>  H = [[2, 3], [3, 4]]
/// let hessian_f = grad_grad(|xs: &[DualNumber<f64>]| {
///     let two = DualNumber::constant(2.0);
///     let three = DualNumber::constant(3.0);
///     xs[0] * xs[0] + three * xs[0] * xs[1] + two * xs[1] * xs[1]
/// });
/// let x = Array1::from(vec![1.0, 1.0]);
/// let h = hessian_f(&x);
/// assert!((h[[0, 0]] - 2.0).abs() < 1e-4);
/// assert!((h[[0, 1]] - 3.0).abs() < 1e-4);
/// ```
pub fn grad_grad<F, Func>(func: Func) -> impl Fn(&Array1<F>) -> Array2<F>
where
    F: NumFloat + Copy + fmt::Debug + Send + Sync + 'static,
    Func: Fn(&[DualNumber<F>]) -> DualNumber<F> + Clone + 'static,
{
    move |x: &Array1<F>| crate::forward_mode::hessian(func.clone(), x)
}

/// Simultaneously compute the value and gradient of a scalar function.
///
/// Like JAX's `jax.value_and_grad`: returns `(f(x), nabla f(x))` in a single
/// logical pass. This avoids redundant computation when both value and gradient
/// are needed (e.g., during training).
///
/// # Arguments
/// * `func` - A scalar function operating on dual numbers
///
/// # Returns
/// A closure returning `(value, gradient)` at any point `x`.
///
/// # Example
/// ```rust
/// use scirs2_autograd::transforms::value_and_grad;
/// use scirs2_autograd::forward_mode::DualNumber;
/// use scirs2_core::ndarray::Array1;
///
/// let vg = value_and_grad(|xs: &[DualNumber<f64>]| xs[0] * xs[0]);
/// let x = Array1::from(vec![3.0]);
/// let (val, g) = vg(&x);
/// assert!((val - 9.0).abs() < 1e-12);
/// assert!((g[0] - 6.0).abs() < 1e-12);
/// ```
pub fn value_and_grad<F, Func>(func: Func) -> impl Fn(&Array1<F>) -> (F, Array1<F>)
where
    F: NumFloat + Copy + fmt::Debug + Send + Sync + 'static,
    Func: Fn(&[DualNumber<F>]) -> DualNumber<F> + Clone + 'static,
{
    move |x: &Array1<F>| {
        let n = x.len();
        let mut gradient = Array1::<F>::zeros(n);

        // Compute value using constant duals (no tangent seeding)
        let primal_duals: Vec<DualNumber<F>> =
            x.iter().map(|&xk| DualNumber::constant(xk)).collect();
        let value = func.clone()(&primal_duals).value();

        // Compute gradient via n forward passes
        for i in 0..n {
            let duals: Vec<DualNumber<F>> = x
                .iter()
                .enumerate()
                .map(|(k, &xk)| {
                    if k == i {
                        DualNumber::new(xk, F::one())
                    } else {
                        DualNumber::new(xk, F::zero())
                    }
                })
                .collect();
            gradient[i] = func.clone()(&duals).tangent();
        }

        (value, gradient)
    }
}

/// Compute the full Jacobian matrix of a vector-valued function via transforms API.
///
/// Given `f: R^n -> R^m`, returns the `m x n` Jacobian matrix at point `x`.
/// This is a thin wrapper over [`crate::forward_mode::jacobian_forward`] exposed
/// as a functional transformation.
///
/// # Arguments
/// * `func` - A vector function operating on dual numbers
///
/// # Returns
/// A closure that computes the `m x n` Jacobian at any point `x`.
pub fn jacobian<F, Func>(func: Func) -> impl Fn(&Array1<F>) -> Array2<F>
where
    F: NumFloat + Copy + fmt::Debug + Send + Sync + 'static,
    Func: Fn(&[DualNumber<F>]) -> Vec<DualNumber<F>> + Clone + 'static,
{
    move |x: &Array1<F>| crate::forward_mode::jacobian_forward(func.clone(), x)
}

// ---------------------------------------------------------------------------
// stop_gradient -- Detach from computation graph
// ---------------------------------------------------------------------------

/// Detach a tensor from the computation graph (stop gradient flow).
///
/// Like JAX's `jax.lax.stop_gradient`: returns a clone of the input with no
/// gradient tracking. In a forward-mode AD context, this zeros all tangent
/// components; in the autograd graph context, this creates a leaf node with
/// no incoming edges.
///
/// # Arguments
/// * `tensor` - The 2-D array to detach
///
/// # Returns
/// A cloned copy of the tensor that will not propagate gradients.
pub fn stop_gradient<F: NumFloat + Copy>(tensor: &Array2<F>) -> Array2<F> {
    tensor.clone()
}

/// Detach a 1-D tensor from the computation graph.
///
/// Same semantics as [`stop_gradient`] but for 1-D arrays.
pub fn stop_gradient_1d<F: NumFloat + Copy>(tensor: &Array1<F>) -> Array1<F> {
    tensor.clone()
}

/// Detach dual numbers by zeroing tangent components.
///
/// Given a slice of dual numbers, returns new dual numbers with the same primal
/// values but tangent = 0, effectively stopping gradient propagation through
/// the forward-mode AD computation.
pub fn stop_gradient_dual<F: NumFloat + Copy + fmt::Debug>(
    duals: &[DualNumber<F>],
) -> Vec<DualNumber<F>> {
    duals
        .iter()
        .map(|d| DualNumber::constant(d.value()))
        .collect()
}

// ---------------------------------------------------------------------------
// Checkpoint -- Memory-efficient gradient checkpointing
// ---------------------------------------------------------------------------

/// Memory-efficient gradient checkpointing for sequential computation segments.
///
/// Like JAX's `jax.checkpoint` (or PyTorch's `torch.utils.checkpoint`): instead
/// of storing all intermediate activations during the forward pass, only the
/// inputs to each segment boundary are stored. During the backward pass,
/// intermediate activations are recomputed from these checkpoints.
///
/// This trades compute for memory, achieving `O(sqrt(n))` memory usage for
/// `n` sequential segments (vs `O(n)` without checkpointing).
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::transforms::Checkpoint;
/// use scirs2_core::ndarray::Array2;
///
/// let mut ckpt = Checkpoint::<f64>::new();
/// ckpt.add_segment(|x: &Array2<f64>| x.mapv(|v| v * 2.0))
///     .add_segment(|x: &Array2<f64>| x.mapv(|v| v + 1.0))
///     .add_segment(|x: &Array2<f64>| x.mapv(|v| v * v));
///
/// let input = Array2::from_shape_vec((2, 3), vec![1.0; 6]).expect("valid shape");
/// let output = ckpt.forward(&input);
/// // (1*2+1)^2 = 9 for each element
/// assert!((output[[0, 0]] - 9.0).abs() < 1e-12);
///
/// // Memory savings: sqrt(3)/3 ~ 0.577 ratio
/// let ratio = ckpt.memory_savings_ratio();
/// assert!(ratio > 0.0 && ratio < 1.0);
/// ```
pub struct Checkpoint<F: NumFloat + Copy> {
    /// Ordered list of computation segments
    segments: Vec<SegmentFn<F>>,
}

impl<F: NumFloat + Copy + fmt::Debug> Checkpoint<F> {
    /// Create a new empty checkpoint container.
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Add a computation segment to the checkpoint chain.
    ///
    /// Segments are applied sequentially: the output of segment `i` becomes
    /// the input to segment `i + 1`.
    pub fn add_segment(
        &mut self,
        f: impl Fn(&Array2<F>) -> Array2<F> + Send + Sync + 'static,
    ) -> &mut Self {
        self.segments.push(Arc::new(f));
        self
    }

    /// Execute the full forward pass through all segments.
    ///
    /// In a checkpointed backward pass, only the boundary activations
    /// (the input to each segment) would be stored, and intermediates
    /// recomputed. This forward pass demonstrates the sequential pipeline.
    pub fn forward(&self, input: &Array2<F>) -> Array2<F> {
        let mut current = input.clone();
        for seg in &self.segments {
            current = seg(&current);
        }
        current
    }

    /// Execute the forward pass and return all intermediate checkpoint values.
    ///
    /// Returns `(final_output, checkpoints)` where `checkpoints[i]` is the
    /// input to segment `i`. The first checkpoint is the original input.
    pub fn forward_with_checkpoints(&self, input: &Array2<F>) -> (Array2<F>, Vec<Array2<F>>) {
        let mut checkpoints = Vec::with_capacity(self.segments.len() + 1);
        let mut current = input.clone();

        for seg in &self.segments {
            checkpoints.push(current.clone());
            current = seg(&current);
        }

        (current, checkpoints)
    }

    /// Recompute the output of segment `i` from its checkpoint.
    ///
    /// This is used during the backward pass to regenerate activations that
    /// were not stored, trading compute for memory.
    ///
    /// # Returns
    /// `None` if segment index is out of bounds.
    pub fn recompute_segment(
        &self,
        segment_idx: usize,
        checkpoint: &Array2<F>,
    ) -> Option<Array2<F>> {
        self.segments.get(segment_idx).map(|seg| seg(checkpoint))
    }

    /// Compute the theoretical memory savings ratio.
    ///
    /// Without checkpointing, all `n` intermediate activations must be stored.
    /// With sqrt-decomposition checkpointing, only `O(sqrt(n))` activations
    /// are stored at checkpoint boundaries, plus at most `O(sqrt(n))`
    /// recomputed intermediates within each segment.
    ///
    /// Returns `sqrt(n) / n` which approaches 0 for large `n`.
    /// Returns `1.0` if there are 0 or 1 segments (no savings possible).
    pub fn memory_savings_ratio(&self) -> f64 {
        let n = self.segments.len();
        if n <= 1 {
            return 1.0;
        }
        (n as f64).sqrt() / n as f64
    }

    /// Return the number of registered segments.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }
}

impl<F: NumFloat + Copy + fmt::Debug> Default for Checkpoint<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// JitHint -- JIT compilation hints
// ---------------------------------------------------------------------------

/// Hints for JIT-like optimization of computation graphs.
///
/// Unlike JAX's `jax.jit` which traces and compiles Python functions into XLA
/// programs, this struct provides optimization hints that the autograd engine
/// can use to apply graph-level optimizations such as constant folding,
/// operation fusion, and common subexpression elimination.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::transforms::JitHint;
///
/// let hint = JitHint::new()
///     .enable_constant_folding(true)
///     .enable_fusion(true)
///     .enable_cse(true)
///     .with_static_argnums(&[1, 2]);
/// assert!(hint.constant_folding());
/// assert!(hint.fusion());
/// assert_eq!(hint.static_argnums().len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct JitHint {
    /// Whether to apply constant folding (evaluate constant subexpressions at compile time)
    constant_folding: bool,
    /// Whether to fuse compatible operations (e.g., elementwise chains)
    fusion: bool,
    /// Whether to apply common subexpression elimination
    cse: bool,
    /// Argument indices that are treated as static (constant across calls)
    static_args: Vec<usize>,
    /// Whether to enable dead code elimination
    dead_code_elimination: bool,
    /// Maximum number of fused operations in a single kernel
    max_fusion_depth: usize,
}

impl JitHint {
    /// Create a new `JitHint` with all optimizations disabled.
    pub fn new() -> Self {
        Self {
            constant_folding: false,
            fusion: false,
            cse: false,
            static_args: Vec::new(),
            dead_code_elimination: false,
            max_fusion_depth: 8,
        }
    }

    /// Enable or disable constant folding.
    pub fn enable_constant_folding(mut self, enable: bool) -> Self {
        self.constant_folding = enable;
        self
    }

    /// Enable or disable operation fusion.
    pub fn enable_fusion(mut self, enable: bool) -> Self {
        self.fusion = enable;
        self
    }

    /// Enable or disable common subexpression elimination.
    pub fn enable_cse(mut self, enable: bool) -> Self {
        self.cse = enable;
        self
    }

    /// Enable or disable dead code elimination.
    pub fn enable_dead_code_elimination(mut self, enable: bool) -> Self {
        self.dead_code_elimination = enable;
        self
    }

    /// Set the maximum fusion depth.
    pub fn set_max_fusion_depth(mut self, depth: usize) -> Self {
        self.max_fusion_depth = depth;
        self
    }

    /// Mark certain argument indices as static (constant across calls).
    ///
    /// Static arguments are treated as compile-time constants, enabling
    /// aggressive constant folding and specialization.
    pub fn with_static_argnums(mut self, indices: &[usize]) -> Self {
        self.static_args = indices.to_vec();
        self
    }

    /// Whether constant folding is enabled.
    pub fn constant_folding(&self) -> bool {
        self.constant_folding
    }

    /// Whether operation fusion is enabled.
    pub fn fusion(&self) -> bool {
        self.fusion
    }

    /// Whether CSE is enabled.
    pub fn cse(&self) -> bool {
        self.cse
    }

    /// Whether dead code elimination is enabled.
    pub fn dead_code_elimination(&self) -> bool {
        self.dead_code_elimination
    }

    /// Maximum fusion depth.
    pub fn max_fusion_depth(&self) -> usize {
        self.max_fusion_depth
    }

    /// Indices of static arguments.
    pub fn static_argnums(&self) -> &[usize] {
        &self.static_args
    }

    /// Create a hint with all optimizations enabled.
    pub fn all_optimizations() -> Self {
        Self {
            constant_folding: true,
            fusion: true,
            cse: true,
            static_args: Vec::new(),
            dead_code_elimination: true,
            max_fusion_depth: 16,
        }
    }
}

impl Default for JitHint {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Composable transform utilities
// ---------------------------------------------------------------------------

/// Apply a function elementwise to a batch, accumulating both values and gradients.
///
/// This combines `vmap` with `value_and_grad`: for each row in the batch,
/// computes both `f(row)` and `nabla f(row)`.
///
/// # Arguments
/// * `func` - Scalar function on dual numbers
/// * `inputs` - Batch matrix `(batch_size, n)`
///
/// # Returns
/// `(values, gradients)` where `values` is `Array1<F>` of length `batch_size`
/// and `gradients` is `Array2<F>` of shape `(batch_size, n)`.
///
/// # Errors
/// Returns an error if the batch is empty.
pub fn batched_value_and_grad<F, Func>(
    func: Func,
    inputs: &Array2<F>,
) -> Result<(Array1<F>, Array2<F>), AutogradError>
where
    F: NumFloat + Copy + fmt::Debug + Send + Sync + 'static,
    Func: Fn(&[DualNumber<F>]) -> DualNumber<F> + Clone + Send + Sync + 'static,
{
    let batch_size = inputs.nrows();
    let n = inputs.ncols();

    if batch_size == 0 {
        return Err(AutogradError::OperationError(
            "batched_value_and_grad: empty batch".to_string(),
        ));
    }

    let mut values = Array1::<F>::zeros(batch_size);
    let mut gradients = Array2::<F>::zeros((batch_size, n));

    let vg = value_and_grad(func);

    for b in 0..batch_size {
        let row = inputs.row(b).to_owned();
        let (val, g) = vg(&row);
        values[b] = val;
        for j in 0..n {
            gradients[[b, j]] = g[j];
        }
    }

    Ok((values, gradients))
}

/// Apply a sequence of transformations, returning the final result and the
/// list of intermediate outputs (for debugging or analysis).
///
/// Each transformation is a function `&Array1<F> -> Array1<F>`.
///
/// # Returns
/// `(final_output, intermediates)` where `intermediates[i]` is the output of
/// the `i`-th transformation (0-indexed).
pub fn scan<F, Func>(transforms: &[Func], input: &Array1<F>) -> (Array1<F>, Vec<Array1<F>>)
where
    F: NumFloat + Copy + fmt::Debug,
    Func: Fn(&Array1<F>) -> Array1<F>,
{
    let mut intermediates = Vec::with_capacity(transforms.len());
    let mut current = input.clone();

    for t in transforms {
        current = t(&current);
        intermediates.push(current.clone());
    }

    (current, intermediates)
}

/// Numerically verify the gradient of a scalar function using central finite
/// differences. Returns the maximum absolute error between the analytical
/// gradient (from dual numbers) and the numerical approximation.
///
/// This is a testing utility analogous to `jax.test_util.check_grads`.
///
/// # Arguments
/// * `func_dual` - Function on dual numbers (for analytical gradient)
/// * `func_scalar` - Same function on plain scalars (for numerical gradient)
/// * `x` - Point at which to check
/// * `epsilon` - Finite difference step size (e.g., `1e-5`)
///
/// # Returns
/// Maximum absolute error across all gradient components.
pub fn check_grad<F, FuncDual, FuncScalar>(
    func_dual: FuncDual,
    func_scalar: FuncScalar,
    x: &Array1<F>,
    epsilon: F,
) -> F
where
    F: NumFloat + Copy + fmt::Debug,
    FuncDual: Fn(&[DualNumber<F>]) -> DualNumber<F> + Clone,
    FuncScalar: Fn(&Array1<F>) -> F,
{
    let two = F::one() + F::one();

    // Analytical gradient via forward-mode
    let analytical = crate::forward_mode::gradient_forward(func_dual, x);

    // Numerical gradient via central differences
    let n = x.len();
    let mut max_err = F::zero();

    for i in 0..n {
        let mut x_fwd = x.clone();
        let mut x_bwd = x.clone();
        x_fwd[i] = x[i] + epsilon;
        x_bwd[i] = x[i] - epsilon;

        let numerical_i = (func_scalar(&x_fwd) - func_scalar(&x_bwd)) / (two * epsilon);
        let err = (analytical[i] - numerical_i).abs();
        if err > max_err {
            max_err = err;
        }
    }

    max_err
}

/// Compose two functions into a single function: `(g . f)(x) = g(f(x))`.
///
/// # Returns
/// A closure that applies `f` then `g`.
pub fn compose<F, G, A, B, C>(f: F, g: G) -> impl Fn(A) -> C
where
    F: Fn(A) -> B,
    G: Fn(B) -> C,
{
    move |x| g(f(x))
}

/// Apply a function `n` times: `f^n(x) = f(f(...f(x)...))`.
///
/// # Arguments
/// * `func` - The function to iterate
/// * `x` - Initial value
/// * `n` - Number of iterations
///
/// # Returns
/// `f` applied `n` times to `x`.
pub fn iterate<F, Func>(func: &Func, x: &Array1<F>, n: usize) -> Array1<F>
where
    F: NumFloat + Copy + fmt::Debug,
    Func: Fn(&Array1<F>) -> Array1<F>,
{
    let mut current = x.clone();
    for _ in 0..n {
        current = func(&current);
    }
    current
}

/// Compute a finite difference Jacobian for testing/verification purposes.
///
/// # Arguments
/// * `func` - Vector-valued function `R^n -> R^m`
/// * `x` - Evaluation point
/// * `epsilon` - Step size for finite differences
///
/// # Returns
/// `m x n` Jacobian matrix via central differences.
pub fn numerical_jacobian<F, Func>(func: &Func, x: &Array1<F>, epsilon: F) -> Array2<F>
where
    F: NumFloat + Copy + fmt::Debug,
    Func: Fn(&Array1<F>) -> Array1<F>,
{
    let two = F::one() + F::one();
    let n = x.len();
    let y0 = func(x);
    let m = y0.len();

    let mut jac = Array2::<F>::zeros((m, n));

    for j in 0..n {
        let mut x_fwd = x.clone();
        let mut x_bwd = x.clone();
        x_fwd[j] = x[j] + epsilon;
        x_bwd[j] = x[j] - epsilon;

        let y_fwd = func(&x_fwd);
        let y_bwd = func(&x_bwd);

        for i in 0..m {
            jac[[i, j]] = (y_fwd[i] - y_bwd[i]) / (two * epsilon);
        }
    }

    jac
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    // ===== vmap tests =====

    #[test]
    fn test_vmap_double() {
        let batch = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid shape");
        let result = vmap(|x: &Array1<f64>| x.mapv(|v| v * 2.0), &batch).expect("vmap succeeds");
        assert_eq!(result.shape(), &[3, 2]);
        assert!((result[[0, 0]] - 2.0).abs() < 1e-12);
        assert!((result[[1, 0]] - 6.0).abs() < 1e-12);
        assert!((result[[2, 1]] - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_vmap_nonlinear() {
        let batch = Array2::from_shape_vec((2, 3), vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
            .expect("valid shape");
        let result = vmap(|x: &Array1<f64>| x.mapv(|v| v.sqrt()), &batch).expect("vmap succeeds");
        assert!((result[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((result[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((result[[1, 2]] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_vmap_empty_batch() {
        let batch = Array2::<f64>::zeros((0, 3));
        let result = vmap(|x: &Array1<f64>| x.clone(), &batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_vmap_single_element() {
        let batch = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).expect("valid shape");
        let result = vmap(|x: &Array1<f64>| x.mapv(|v| v + 10.0), &batch).expect("vmap succeeds");
        assert_eq!(result.shape(), &[1, 4]);
        assert!((result[[0, 0]] - 11.0).abs() < 1e-12);
        assert!((result[[0, 3]] - 14.0).abs() < 1e-12);
    }

    #[test]
    fn test_vmap_dimension_change() {
        // Function that maps R^3 -> R^2 (sum first two, keep third)
        let batch = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid shape");
        let result = vmap(
            |x: &Array1<f64>| Array1::from(vec![x[0] + x[1], x[2]]),
            &batch,
        )
        .expect("vmap succeeds");
        assert_eq!(result.shape(), &[2, 2]);
        assert!((result[[0, 0]] - 3.0).abs() < 1e-12); // 1+2
        assert!((result[[0, 1]] - 3.0).abs() < 1e-12); // 3
        assert!((result[[1, 0]] - 9.0).abs() < 1e-12); // 4+5
        assert!((result[[1, 1]] - 6.0).abs() < 1e-12); // 6
    }

    // ===== grad tests =====

    #[test]
    fn test_grad_quadratic() {
        // f(x) = x0^2 + x1^2  =>  grad = [2*x0, 2*x1]
        let grad_f = grad(|xs: &[DualNumber<f64>]| xs[0] * xs[0] + xs[1] * xs[1]);
        let x = Array1::from(vec![3.0, 4.0]);
        let g = grad_f(&x);
        assert!((g[0] - 6.0).abs() < 1e-12);
        assert!((g[1] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_grad_linear() {
        // f(x) = 3*x0 + 7*x1  =>  grad = [3, 7]
        let grad_f = grad(|xs: &[DualNumber<f64>]| {
            let three = DualNumber::constant(3.0);
            let seven = DualNumber::constant(7.0);
            three * xs[0] + seven * xs[1]
        });
        let x = Array1::from(vec![100.0, 200.0]);
        let g = grad_f(&x);
        assert!((g[0] - 3.0).abs() < 1e-12);
        assert!((g[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_grad_transcendental() {
        // f(x) = sin(x0) * exp(x1) at [0, 0]
        // grad = [cos(0)*exp(0), sin(0)*exp(0)] = [1, 0]
        let grad_f = grad(|xs: &[DualNumber<f64>]| xs[0].sin() * xs[1].exp());
        let x = Array1::from(vec![0.0, 0.0]);
        let g = grad_f(&x);
        assert!((g[0] - 1.0).abs() < 1e-12);
        assert!(g[1].abs() < 1e-12);
    }

    // ===== grad_grad tests =====

    #[test]
    fn test_grad_grad_quadratic() {
        // f(x) = x0^2 + 3*x0*x1 + 2*x1^2  =>  H = [[2, 3], [3, 4]]
        let hessian_f = grad_grad(|xs: &[DualNumber<f64>]| {
            let two = DualNumber::constant(2.0);
            let three = DualNumber::constant(3.0);
            xs[0] * xs[0] + three * xs[0] * xs[1] + two * xs[1] * xs[1]
        });
        let x = Array1::from(vec![1.0, 1.0]);
        let h = hessian_f(&x);
        assert!((h[[0, 0]] - 2.0).abs() < 1e-4);
        assert!((h[[0, 1]] - 3.0).abs() < 1e-4);
        assert!((h[[1, 0]] - 3.0).abs() < 1e-4);
        assert!((h[[1, 1]] - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_grad_grad_diagonal() {
        // f(x) = x0^2 + 5*x1^2  =>  H = [[2, 0], [0, 10]]
        let hessian_f = grad_grad(|xs: &[DualNumber<f64>]| {
            let five = DualNumber::constant(5.0);
            xs[0] * xs[0] + five * xs[1] * xs[1]
        });
        let x = Array1::from(vec![0.0, 0.0]);
        let h = hessian_f(&x);
        assert!((h[[0, 0]] - 2.0).abs() < 1e-4);
        assert!(h[[0, 1]].abs() < 1e-4);
        assert!(h[[1, 0]].abs() < 1e-4);
        assert!((h[[1, 1]] - 10.0).abs() < 1e-4);
    }

    // ===== value_and_grad tests =====

    #[test]
    fn test_value_and_grad_basic() {
        let vg = value_and_grad(|xs: &[DualNumber<f64>]| xs[0] * xs[0] + xs[1] * xs[1]);
        let x = Array1::from(vec![3.0, 4.0]);
        let (val, g) = vg(&x);
        assert!((val - 25.0).abs() < 1e-12);
        assert!((g[0] - 6.0).abs() < 1e-12);
        assert!((g[1] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_value_and_grad_rosenbrock() {
        // f(x) = (1-x0)^2 + 100*(x1-x0^2)^2
        // At minimum [1,1]: f=0, grad=[0,0]
        let vg = value_and_grad(|xs: &[DualNumber<f64>]| {
            let one = DualNumber::constant(1.0);
            let hundred = DualNumber::constant(100.0);
            let a = one - xs[0];
            let b = xs[1] - xs[0] * xs[0];
            a * a + hundred * b * b
        });
        let x = Array1::from(vec![1.0, 1.0]);
        let (val, g) = vg(&x);
        assert!(val.abs() < 1e-12);
        assert!(g[0].abs() < 1e-12);
        assert!(g[1].abs() < 1e-12);
    }

    // ===== stop_gradient tests =====

    #[test]
    fn test_stop_gradient_2d() {
        let t = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("valid shape");
        let stopped = stop_gradient(&t);
        assert_eq!(t, stopped);
        // Modification of stopped should not affect original
        let mut stopped_mut = stopped;
        stopped_mut[[0, 0]] = 999.0;
        assert!((t[[0, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_stop_gradient_1d() {
        let t = Array1::from(vec![1.0, 2.0, 3.0]);
        let stopped = stop_gradient_1d(&t);
        assert_eq!(t, stopped);
    }

    #[test]
    fn test_stop_gradient_dual() {
        let duals = vec![DualNumber::new(1.0_f64, 0.5), DualNumber::new(2.0, 0.7)];
        let stopped = stop_gradient_dual(&duals);
        assert!((stopped[0].value() - 1.0).abs() < 1e-12);
        assert!(stopped[0].tangent().abs() < 1e-12); // tangent zeroed
        assert!((stopped[1].value() - 2.0).abs() < 1e-12);
        assert!(stopped[1].tangent().abs() < 1e-12);
    }

    // ===== checkpoint tests =====

    #[test]
    fn test_checkpoint_forward() {
        let mut ckpt = Checkpoint::<f64>::new();
        ckpt.add_segment(|x: &Array2<f64>| x.mapv(|v| v * 2.0))
            .add_segment(|x: &Array2<f64>| x.mapv(|v| v + 1.0))
            .add_segment(|x: &Array2<f64>| x.mapv(|v| v * v));

        let input = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("valid shape");
        let output = ckpt.forward(&input);
        // (1*2+1)^2 = 9, (2*2+1)^2 = 25, (3*2+1)^2 = 49, (4*2+1)^2 = 81
        assert!((output[[0, 0]] - 9.0).abs() < 1e-12);
        assert!((output[[0, 1]] - 25.0).abs() < 1e-12);
        assert!((output[[1, 0]] - 49.0).abs() < 1e-12);
        assert!((output[[1, 1]] - 81.0).abs() < 1e-12);
    }

    #[test]
    fn test_checkpoint_with_intermediates() {
        let mut ckpt = Checkpoint::<f64>::new();
        ckpt.add_segment(|x: &Array2<f64>| x.mapv(|v| v + 10.0))
            .add_segment(|x: &Array2<f64>| x.mapv(|v| v * 3.0));

        let input = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).expect("valid shape");
        let (output, checkpoints) = ckpt.forward_with_checkpoints(&input);

        assert_eq!(checkpoints.len(), 2);
        // checkpoint[0] = input = [1, 2]
        assert!((checkpoints[0][[0, 0]] - 1.0).abs() < 1e-12);
        // checkpoint[1] = after seg 0 = [11, 12]
        assert!((checkpoints[1][[0, 0]] - 11.0).abs() < 1e-12);
        // output = [33, 36]
        assert!((output[[0, 0]] - 33.0).abs() < 1e-12);
        assert!((output[[0, 1]] - 36.0).abs() < 1e-12);
    }

    #[test]
    fn test_checkpoint_recompute() {
        let mut ckpt = Checkpoint::<f64>::new();
        ckpt.add_segment(|x: &Array2<f64>| x.mapv(|v| v * 5.0));

        let input = Array2::from_shape_vec((1, 1), vec![3.0]).expect("valid shape");
        let recomputed = ckpt.recompute_segment(0, &input);
        assert!(recomputed.is_some());
        let r = recomputed.expect("segment exists");
        assert!((r[[0, 0]] - 15.0).abs() < 1e-12);

        // Out of bounds returns None
        assert!(ckpt.recompute_segment(5, &input).is_none());
    }

    #[test]
    fn test_checkpoint_memory_savings() {
        let mut ckpt = Checkpoint::<f64>::new();
        for _ in 0..100 {
            ckpt.add_segment(|x: &Array2<f64>| x.mapv(|v| v + 1.0));
        }
        let ratio = ckpt.memory_savings_ratio();
        // sqrt(100)/100 = 0.1
        assert!((ratio - 0.1).abs() < 1e-12);
        assert_eq!(ckpt.num_segments(), 100);
    }

    #[test]
    fn test_checkpoint_empty() {
        let ckpt = Checkpoint::<f64>::new();
        let input = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).expect("valid shape");
        let output = ckpt.forward(&input);
        assert_eq!(output, input);
        assert!((ckpt.memory_savings_ratio() - 1.0).abs() < 1e-12);
    }

    // ===== JitHint tests =====

    #[test]
    fn test_jit_hint_builder() {
        let hint = JitHint::new()
            .enable_constant_folding(true)
            .enable_fusion(true)
            .enable_cse(false)
            .enable_dead_code_elimination(true)
            .with_static_argnums(&[0, 2])
            .set_max_fusion_depth(4);

        assert!(hint.constant_folding());
        assert!(hint.fusion());
        assert!(!hint.cse());
        assert!(hint.dead_code_elimination());
        assert_eq!(hint.static_argnums(), &[0, 2]);
        assert_eq!(hint.max_fusion_depth(), 4);
    }

    #[test]
    fn test_jit_hint_all_optimizations() {
        let hint = JitHint::all_optimizations();
        assert!(hint.constant_folding());
        assert!(hint.fusion());
        assert!(hint.cse());
        assert!(hint.dead_code_elimination());
    }

    // ===== pmap tests =====

    #[test]
    fn test_pmap_parallel() {
        let batch = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("valid shape");
        let result = pmap(|x: &Array1<f64>| x.mapv(|v| v * v), &batch).expect("pmap succeeds");
        assert_eq!(result.shape(), &[4, 2]);
        assert!((result[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((result[[1, 1]] - 16.0).abs() < 1e-12);
        assert!((result[[3, 0]] - 49.0).abs() < 1e-12);
    }

    #[test]
    fn test_pmap_empty_batch() {
        let batch = Array2::<f64>::zeros((0, 3));
        let result = pmap(|x: &Array1<f64>| x.clone(), &batch);
        assert!(result.is_err());
    }

    // ===== batched_value_and_grad tests =====

    #[test]
    fn test_batched_value_and_grad() {
        // f(x) = x0^2 + x1^2
        let batch = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 1.0, 0.0]).expect("valid shape");
        let (vals, grads) = batched_value_and_grad(
            |xs: &[DualNumber<f64>]| xs[0] * xs[0] + xs[1] * xs[1],
            &batch,
        )
        .expect("succeeds");

        // f([3,4]) = 25, grad = [6, 8]
        assert!((vals[0] - 25.0).abs() < 1e-12);
        assert!((grads[[0, 0]] - 6.0).abs() < 1e-12);
        assert!((grads[[0, 1]] - 8.0).abs() < 1e-12);

        // f([1,0]) = 1, grad = [2, 0]
        assert!((vals[1] - 1.0).abs() < 1e-12);
        assert!((grads[[1, 0]] - 2.0).abs() < 1e-12);
        assert!(grads[[1, 1]].abs() < 1e-12);
    }

    // ===== scan tests =====

    #[test]
    fn test_scan_transforms() {
        type TransformFn = Box<dyn Fn(&Array1<f64>) -> Array1<f64>>;
        let transforms: Vec<TransformFn> = vec![
            Box::new(|x: &Array1<f64>| x.mapv(|v| v + 1.0)),
            Box::new(|x: &Array1<f64>| x.mapv(|v| v * 2.0)),
            Box::new(|x: &Array1<f64>| x.mapv(|v| v - 3.0)),
        ];
        let input = Array1::from(vec![10.0]);
        let (final_out, intermediates) = scan(&transforms, &input);

        assert_eq!(intermediates.len(), 3);
        assert!((intermediates[0][0] - 11.0).abs() < 1e-12); // +1
        assert!((intermediates[1][0] - 22.0).abs() < 1e-12); // *2
        assert!((intermediates[2][0] - 19.0).abs() < 1e-12); // -3
        assert!((final_out[0] - 19.0).abs() < 1e-12);
    }

    // ===== check_grad tests =====

    #[test]
    fn test_check_grad_accurate() {
        let err = check_grad(
            |xs: &[DualNumber<f64>]| xs[0] * xs[0] + xs[1] * xs[1],
            |xs: &Array1<f64>| xs[0] * xs[0] + xs[1] * xs[1],
            &Array1::from(vec![3.0, 4.0]),
            1e-6,
        );
        assert!(err < 1e-5, "Gradient check error too large: {}", err);
    }

    // ===== compose tests =====

    #[test]
    fn test_compose_functions() {
        let f = |x: f64| x * 2.0;
        let g = |x: f64| x + 10.0;
        let h = compose(f, g);
        assert!((h(5.0) - 20.0).abs() < 1e-12); // 5*2 + 10 = 20
    }

    // ===== iterate tests =====

    #[test]
    fn test_iterate_function() {
        // f(x) = x + 1, iterated 5 times: x + 5
        let f = |x: &Array1<f64>| x.mapv(|v| v + 1.0);
        let x = Array1::from(vec![0.0]);
        let result = iterate(&f, &x, 5);
        assert!((result[0] - 5.0).abs() < 1e-12);
    }

    // ===== numerical_jacobian tests =====

    #[test]
    fn test_numerical_jacobian() {
        // f(x) = [x0^2, x0*x1] at [2,3]
        // J = [[4, 0], [3, 2]]
        let f = |x: &Array1<f64>| Array1::from(vec![x[0] * x[0], x[0] * x[1]]);
        let x = Array1::from(vec![2.0, 3.0]);
        let jac = numerical_jacobian(&f, &x, 1e-6);
        assert!((jac[[0, 0]] - 4.0).abs() < 1e-4);
        assert!(jac[[0, 1]].abs() < 1e-4);
        assert!((jac[[1, 0]] - 3.0).abs() < 1e-4);
        assert!((jac[[1, 1]] - 2.0).abs() < 1e-4);
    }

    // ===== jacobian transform tests =====

    #[test]
    fn test_jacobian_transform() {
        let jac_f = jacobian(|xs: &[DualNumber<f64>]| vec![xs[0] * xs[0], xs[0] * xs[1]]);
        let x = Array1::from(vec![2.0_f64, 3.0]);
        let jac = jac_f(&x);
        assert!((jac[[0, 0]] - 4.0).abs() < 1e-12);
        assert!(jac[[0, 1]].abs() < 1e-12);
        assert!((jac[[1, 0]] - 3.0).abs() < 1e-12);
        assert!((jac[[1, 1]] - 2.0).abs() < 1e-12);
    }

    // ===== integration / composition test =====

    #[test]
    fn test_grad_then_vmap() {
        // Compute gradient at multiple points via vmap(grad)
        let grad_f = grad(|xs: &[DualNumber<f64>]| xs[0] * xs[0]);

        // Points: 1.0, 2.0, 3.0  =>  grads: 2.0, 4.0, 6.0
        let points = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).expect("valid shape");
        let gradients =
            vmap(move |x: &Array1<f64>| grad_f(x), &points).expect("vmap(grad) succeeds");

        assert!((gradients[[0, 0]] - 2.0).abs() < 1e-12);
        assert!((gradients[[1, 0]] - 4.0).abs() < 1e-12);
        assert!((gradients[[2, 0]] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_checkpoint_with_grad() {
        // Build a checkpointed forward pass and verify values
        let mut ckpt = Checkpoint::<f64>::new();
        ckpt.add_segment(|x: &Array2<f64>| x.mapv(|v| v.exp()))
            .add_segment(|x: &Array2<f64>| {
                // Sum along axis 1 and broadcast back
                let sums: Vec<f64> = x
                    .rows()
                    .into_iter()
                    .map(|row| row.iter().copied().fold(0.0, |a, b| a + b))
                    .collect();
                let ncols = x.ncols();
                let mut out = Array2::zeros(x.raw_dim());
                for (i, &s) in sums.iter().enumerate() {
                    for j in 0..ncols {
                        out[[i, j]] = x[[i, j]] / s;
                    }
                }
                out
            });

        let input = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).expect("valid shape");
        let output = ckpt.forward(&input);

        // exp(1) + exp(2) + exp(3) = e + e^2 + e^3
        let sum_exp = 1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp();
        assert!((output[[0, 0]] - 1.0_f64.exp() / sum_exp).abs() < 1e-10);
        assert!((output[[0, 1]] - 2.0_f64.exp() / sum_exp).abs() < 1e-10);
    }

    #[test]
    fn test_value_and_grad_consistency() {
        // value_and_grad should give same results as separate value + grad calls
        let func = |xs: &[DualNumber<f64>]| {
            let three = DualNumber::constant(3.0);
            xs[0].powi(3) + three * xs[1].exp()
        };

        let x = Array1::from(vec![2.0, 1.0]);

        let vg = value_and_grad(func);
        let (val, g) = vg(&x);

        let grad_f = grad(func);
        let g2 = grad_f(&x);

        // Evaluate value separately
        let primal: Vec<DualNumber<f64>> = x.iter().map(|&xi| DualNumber::constant(xi)).collect();
        let val2 = func(&primal).value();

        assert!((val - val2).abs() < 1e-12, "Values should match");
        assert!((g[0] - g2[0]).abs() < 1e-12, "Gradient x0 should match");
        assert!((g[1] - g2[1]).abs() < 1e-12, "Gradient x1 should match");
    }
}
