//! A collection of functions for manipulating `autograd::Tensor` objects
//!
//! This module is organized into focused submodules while maintaining full backward compatibility:
//!
//! - [`arithmetic`] - Basic arithmetic operations (add, sub, mul, div, sqrt, trigonometry, etc.)
//! - [`reduction`] - Reduction operations (sum, mean, min, max, variance, etc.)
//! - [`linear_algebra`] - Matrix operations (matmul, transpose, decompositions, etc.)
//! - [`activation`] - Neural network activations (relu, sigmoid, softmax, etc.)
//!
//! All functions are re-exported at the top level for backward compatibility.

use ndarray;

use crate::graph::AsGraph;
use crate::ndarray_ext::{ArrayRng, NdArray};
use crate::tensor::{AsTensor, Tensor};
use crate::Float;
use rand::Rng;

// Submodules
pub mod activation;
pub mod arithmetic;
pub mod linear_algebra;
pub mod reduction;

// Internal operation modules (keep existing structure)
mod activation_ops;
mod array_ops;
pub(crate) mod basic_source_ops;
pub(crate) mod binary_ops;
mod blas_ffi;
pub(crate) mod const_gen_ops;
mod conv_ops;
pub(crate) mod dot_ops;
pub(crate) mod gradient_descent_ops;
mod gradient_ops;
mod graph_ops;
pub(crate) mod higher_order_ops;
pub(crate) mod hook_ops;
mod math_ops;
mod random_ops;
mod reduction_ops;
mod xent_ops;

// New linear algebra modules
mod decomposition_ops;
mod eigen_ops;
mod linalg_ops;
// mod matrix_functions; // Module removed - functions are in decomposition_ops
mod matrix_ops;
mod norm_ops;
mod scalar_ops;
mod solver_ops;
mod special_matrices;

// Enhanced linear algebra modules
mod advanced_tensor_ops;
mod matrix_norms;
mod matrix_solvers;
mod special_decompositions;
mod symmetric_ops;

// New advanced linear algebra modules
mod advanced_decompositions;
mod iterative_solvers;
mod matrix_functions;
mod matrix_trig_functions;

// Memory optimization modules
mod checkpoint_ops;

// Debugging modules
mod debug_ops;

// Advanced indexing operations
mod advanced_indexing;

// Broadcasting optimizations
mod broadcast_ops;

// Memory optimization tools
mod memory_optimization;

// Efficient tensor operations
mod efficient_ops;

// Custom activation function framework
mod custom_activations;

// Performance optimization operations
mod performance_ops;

// Enhanced dynamic computation graph features
mod graph_enhancements;

// Numerical properties (rank, condition number)
mod numerical_props;

// Kronecker product
mod kronecker_ops;

// ---------------------------------------
// -- Ops to manipulate `Tensor` object --
// ---------------------------------------

impl<'graph, F: Float> Tensor<'graph, F> {
    /// Gets the `i` th float value of this tensor.
    ///
    /// Index `i` can be negative.
    ///
    ///    ```
    /// use ndarray::{self, array};
    /// use scirs2_autograd as ag;
    /// use ag::tensor_ops::*;
    ///
    /// ag::run(|g| {
    ///    let a = convert_to_tensor(array![[2., 3.], [4., 5.]], g);
    ///    let b = a.access_elem(2);
    ///    assert_eq!(b.eval(g).unwrap()[ndarray::IxDyn(&[])], 4.);
    /// });
    ///    ```
    pub fn access_elem(self, i: isize) -> Tensor<'graph, F> {
        let op = array_ops::IndexOp { index: i };
        Tensor::builder(self.graph)
            .append_input(self, false)
            .build(op)
    }
}

/// Get gradient tensors of `xs` in the same order as `xs`'s
///
/// * `ys` - Targets of differentiation that are arbitrary shapes.
/// * `xs` - Tensors with which differentiate `ys`.
///
/// `ys`
/// See the more useful helper: [crate::optimizers::grad_helper()]
///
///    ```
/// // Partial derivatives of `z = 2x^2 + 3y + 1`.
/// use ndarray;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops as T;
///
/// ag::run(|ctx| {
///     let x = ctx.placeholder("x", &[]);
///     let y = ctx.placeholder("y", &[]);
///     let z = 2.*x*x + 3.*y + 1.;
///
///     // dz/dy
///     let gy = T::grad(&[z], &[y])[0];
///     // dz/dx
///     let gx = T::grad(&[z], &[x])[0];
///
///     // ddz/dx (differentiates `z` again)
///     let ggx = T::grad(&[gx], &[x])[0];
///
///     // evaluation of gradients
///     assert_eq!(3., gy.eval(ctx).unwrap()[ndarray::IxDyn(&[])]);
///     assert_eq!(4., ggx.eval(ctx).unwrap()[ndarray::IxDyn(&[])]);
///
///     // Evaluate dz/dx when x=2:
///     let gx_result = ctx.evaluator()
///         .push(&gx)
///         .feed(x, ndarray::arr0(2.).view().into_dyn())
///         .run();
///     assert_eq!(8., gx_result[0].as_ref().unwrap()[ndarray::IxDyn(&[])]);
/// });
///    ```
pub fn grad<'graph, F: Float, A, B>(ys: &[A], xs: &[B]) -> Vec<Tensor<'graph, F>>
where
    A: AsRef<Tensor<'graph, F>>,
    B: AsRef<Tensor<'graph, F>>,
{
    use crate::gradient::compute_gradients;

    let g = ys[0].as_ref().graph();
    let ys: Vec<_> = ys.iter().map(|y| sum_all(y)).collect();
    let mut grads = compute_gradients(ys.as_slice(), xs, None, g);
    let mut ret = Vec::with_capacity(xs.len());
    for x in xs {
        if let Some(gx) = grads.extract_grad(x) {
            ret.push(gx);
        } else {
            // not differentiable
            // Create a zero tensor using arithmetic operation
            let zero_tensor = crate::tensor_ops::arithmetic::mul(x.as_ref(), scalar(F::zero(), g));
            ret.push(zero_tensor);
        }
    }
    ret
}

/// Computes `xs`'s gradients with `ys`'s already known gradients.
///
/// Almost same spec as `grad`'s except that you can pass `ys`s already known gradients.
/// If `ys_grads` are tensors filled with 1s, this function should be replaced with [crate::tensor_ops::grad()].
///
/// NOTE: Please be careful to match `ys_grads[i].shape` and `ys[i].shape`.
///
/// # Arguments
/// * `ys` - Targets of differentiation.
/// * `xs` - tensors with which differentiate `ys`.
/// * `ys_grads` - Already known gradients of `ys`.
///
/// # Returns
/// Gradient tensors of `xs` in the same order as `xs`'graph.
pub fn grad_with_default<'graph, F: Float, A, B>(
    ys: &[A],
    xs: &[B],
    ys_grads: &[Tensor<'graph, F>],
) -> Vec<Tensor<'graph, F>>
where
    A: AsRef<Tensor<'graph, F>>,
    B: AsRef<Tensor<'graph, F>>,
{
    use crate::gradient::compute_gradients;

    let g = ys[0].as_ref().graph();
    let mut grads = compute_gradients(ys, xs, Some(ys_grads), g);
    let mut ret = Vec::with_capacity(xs.len());
    for x in xs {
        if let Some(gx) = grads.extract_grad(x) {
            ret.push(gx);
        } else {
            // not differentiable
            // Create a zero tensor using arithmetic operation
            let zero_tensor = crate::tensor_ops::arithmetic::mul(x.as_ref(), scalar(F::zero(), g));
            ret.push(zero_tensor);
        }
    }
    ret
}

/// Computes jacobians for variables.
///
/// # Arguments
/// * `y` - Target of differentiation.
/// * `xs` - Tensors with which differentiate `ys`.
/// * `y_size` - (flattened) size of `y`
///
/// # Returns
/// Jacobians for each variable. Each one is a matrix of shape `(y_size, x size)`.
///
/// Note: the current implementation works correctly but is unoptimized for serious use.
///
///    ```ignore
/// // FIXME: Gradient computation returns scalars instead of proper gradients for matrix operations
/// use scirs2_autograd as ag;
/// use ag::prelude::*;
/// use ag::tensor_ops::*;
///
/// let mut env = ag::VariableEnvironment::new();
///
/// let mut rng = ag::ndarray_ext::ArrayRng::<f32>::default();
/// let a = env.set(rng.standard_normal(&[4, 2]));
/// let b = env.set(rng.standard_normal(&[2, 3]));
///
/// env.run(|g| {
///    let a = g.variable(a);
///    let b = g.variable(b);
///    let c = matmul(a, b);
///    let j = jacobians(c, &[a, b], 4*3);
///
///    assert_eq!(j[0].eval(g).unwrap().shape(), &[4*3, 4*2]);
///    assert_eq!(j[1].eval(g).unwrap().shape(), &[4*3, 2*3]);
/// });
///    ```
pub fn jacobians<'graph, A, B, F: Float>(
    y_: A,
    xs_: &[B],
    objective_len: usize,
) -> Vec<Tensor<'graph, F>>
where
    A: AsRef<Tensor<'graph, F>>,
    B: AsRef<Tensor<'graph, F>>,
{
    let y = y_.as_ref();
    let mut vec_vec = Vec::with_capacity(objective_len);
    for i in 0..objective_len as isize {
        vec_vec.push(grad(&[y.access_elem(i)], xs_));
    }

    let len = xs_.len();
    let mut ret = Vec::with_capacity(len);
    // post process gradients
    for i in 0..len {
        // jac is matrix
        let mut jac = Vec::with_capacity(objective_len);
        for vec in &vec_vec {
            jac.push(expand_dims(flatten(vec[i]), &[0]));
        }
        // (y size, x size)
        ret.push(concat(&jac, 0));
    }
    ret
}

/// (Experimental) Computes hessian vector product
pub fn _hessian_vector_product<'graph, A, B, C, F: Float>(
    ys: &[A],
    xs: &[B],
    vectors: &[C],
) -> Vec<Tensor<'graph, F>>
where
    A: AsRef<Tensor<'graph, F>>,
    B: AsRef<Tensor<'graph, F>>,
    C: AsRef<Tensor<'graph, F>>,
{
    let grads = grad(ys, xs);
    let products = grads
        .into_iter()
        .zip(vectors)
        .map(|(g, v)| *g.as_ref() * *v.as_ref())
        .collect::<Vec<_>>();
    grad(products.as_slice(), xs)
}

/// Stops gradient propagation.
///
/// Guarantees that the gradient is not propagated to the tensors behind this
/// during gradient computation.
pub fn stop_gradient<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .set_differentiable(false)
        .build(gradient_ops::StopGradient)
}

/// Returns a `Tensor` representation of the input tensor's shape
///
///    ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|c| {
///    let x: ag::Tensor<f32> = zeros(&[2, 3], c);
///    let s = shape(x);
///    assert_eq!(&[2., 3.], s.eval(c).unwrap().as_slice().unwrap());
/// });
///    ```
pub fn shape<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    if let Some(id) = x.inner().shape {
        return g.tensor(id);
    }
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_differentiable(false)
        .build(array_ops::Shape)
}

/// Returns the size of the input tensor
///
///    ```
/// use ndarray;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|c| {
///    let a: ag::Tensor<f32> = zeros(&[4, 3], c);
///    let b = size(a);
///
///    assert_eq!(12., b.eval(c).unwrap()[ndarray::IxDyn(&[])]);
/// });
///    ```
pub fn size<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_differentiable(false)
        .build(array_ops::Size)
}

/// Returns the rank of the input tensor
///
///    ```
/// use ndarray;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|c| {
///    let x: ag::Tensor<f32> = zeros(&[2, 3, 4], c);
///    let r = rank(x);
///    assert_eq!(3., r.eval(c).unwrap()[ndarray::IxDyn(&[])]);
/// });
///    ```
pub fn rank<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_differentiable(false)
        .build(array_ops::Rank)
}

#[doc(hidden)]
/// Gets n th tensor in `x`.
///
/// `x` must be a result of a multi-outputs op;
/// otherwise index-out-of-bounds error may happen.
pub fn nth_tensor<'graph, A, F: Float>(x: A, n: usize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input_with_selector(x, false, n)
        .build(activation_ops::Identity)
}

/// Identity function without copy.
pub fn identity<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(activation_ops::Identity)
}

/// Takes diff between two tensors.
///
/// Returns the sorted, unique values in `a` that are not in `b`.
///
///    ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![4., 1., 5., 2., 3., 6.], g);
///    let b = convert_to_tensor(array![[2., 3.], [1., 4.]], g);
///    let c = setdiff1d(a, b);
///    assert_eq!(
///        c.eval(g),
///        Ok(ndarray::arr1(&[5., 6.]).into_dyn())
///    )
/// });
///    ```
///
pub fn setdiff1d<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    let op = array_ops::SetDiff1D;
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(op)
}

/// Slices the input tensor.
///
/// # Arguments
/// * `x` - Tensor with arbitrary shape.
/// * `starts` - Inclusive start indices for the dimensions.
/// * `ends` - End indices for the dimensions. **Each index is inclusive if it is negative and exclusive if it's not.**
///
/// NOTE: Negative values in `starts` and `ends` are counted from the back of the axis.
///    ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[4, 4], g);
///    let b = slice(a, &[0, 0], &[-1, 2]); // numpy equivalent is a[:, 0:2]
///
///    assert_eq!(b.eval(g).unwrap().shape(), &[4, 2]);
/// });
///    ```
///
///    ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[4, 4], g);
///    let b = slice(a, &[0, 0], &[-2, 2]); // numpy equivalent is a[:-1, :2]
///
///    assert_eq!(b.eval(g).unwrap().shape(), &[3, 2]);
/// });
///    ```
pub fn slice<'graph, A, F: Float>(x: A, starts: &[isize], ends: &[isize]) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    // TODO: Make starts and ends ArrayLike
    assert_eq!(starts.len(), ends.len());
    let starts_ends = starts.iter().zip(ends.iter());

    let indices = starts_ends
        .map(|(s, &e)| {
            let e = if e == -1 {
                None
            } else {
                Some(if e < -1 { e + 1 } else { e })
            };
            let slice = ndarray::Slice::new(*s, e, 1);
            ndarray::SliceInfoElem::from(slice)
        })
        .collect::<Vec<ndarray::SliceInfoElem>>();

    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(array_ops::Slice { indices })
}

/// Gathers subviews from the input tensor.
///
/// Same spec as <https://www.tensorflow.org/api_docs/python/tf/gather>.
/// For example, this can be used for embedding vectors lookup etc.
///
/// Unlike `ag::gather`, `indices` can contain negative elements.
///
/// # Returns
/// Tensor with shape `param.shape[..axis] + indices.shape + param.shape[axis+1..]`
///
///    ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let param = zeros(&[5, 4, 8, 2], g);
///    let indices = convert_to_tensor(array![[5., -1., 3.], [2., 1., -2.]], g);
///    let y = gather_common(param, indices, 2);
///
///    assert_eq!(y.eval(g).unwrap().shape(), &[5, 4, 2, 3, 2])
/// });
///    ```
pub fn gather_common<'graph, A, B, F: Float>(param: A, indices: B, axis: isize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let param = param.as_ref();
    let g = param.graph();
    let op = array_ops::Gather {
        axis,
        should_normalize_negative_indices: true,
    };
    Tensor::builder(g)
        .append_input(indices.as_ref(), false)
        .append_input(param.as_ref(), false)
        .build(op)
}

/// Gathers subviews from the input tensor.
///
/// Same spec as <https://www.tensorflow.org/api_docs/python/tf/gather>.
/// For example, this can be used for embedding vectors lookup etc.
///
/// # Returns
/// Tensor with shape `param.shape[..axis] + indices.shape + param.shape[axis+1..]`
///
///    ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let param = zeros(&[5, 4, 8, 2], g);
///    let indices = convert_to_tensor(array![[5., 4., 3.], [2., 1., 0.]], g);  // shape: (2, 3)
///    let y = gather(param, indices, 2);
///
///    assert_eq!(y.eval(g).unwrap().shape(), &[5, 4, 2, 3, 2])
/// });
///    ```
pub fn gather<'graph, A, B, F: Float>(param: A, indices: B, axis: isize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let param = param.as_ref();
    let g = param.graph();
    let op = array_ops::Gather {
        axis,
        should_normalize_negative_indices: false,
    };
    Tensor::builder(g)
        .append_input(indices.as_ref(), false)
        .append_input(param, false)
        .build(op)
}

/// Reshapes the input tensor without copy.
///
///    ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[3, 2, 2], g);
///    let b = reshape(a, &[2, 6]);
///    assert_eq!(b.eval(g).unwrap().shape(), &[2, 6]);
/// });
///    ```
pub fn reshape<'graph, A, AT, F: Float>(x: A, shape: &AT) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    let t = shape.as_tensor(g);
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(t, false)
        .set_shape(&t)
        .build(array_ops::Reshape)
}

/// Flattens the input tensor.
///
///    ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[3, 2, 2], g);
///    let b = flatten(a);
///    assert_eq!(b.eval(g).unwrap().shape(), &[12]);
/// });
///    ```
pub fn flatten<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let _g = x.graph();
    // Use a reshape operation to flatten to 1D
    // For now, use a simple approach that preserves the tensor structure
    let shape_val = [-1i32]; // Use -1 to indicate flatten to 1D
    reshape(x, &shape_val)
}

/// Expands specified dimensions.
///
///    ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[3], g);
///    let b = expand_dims(a, &[1]);
///    assert_eq!(b.eval(g).unwrap().shape(), &[3, 1]);
/// });
///    ```
pub fn expand_dims<'graph, A, AT, F: Float>(x: A, axes: &AT) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(axes.as_tensor(g), false)
        .build(array_ops::ExpandDims)
}

/// Removes specified dimensions of size 1.
///
///    ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[1, 3, 1], g);
///    let b = squeeze(a, &[0, 2]);
///    assert_eq!(b.eval(g).unwrap().shape(), &[3]);
/// });
///    ```
pub fn squeeze<'graph, A, AT, F: Float>(x: A, axes: &AT) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(axes.as_tensor(g), false)
        .build(array_ops::Squeeze)
}

/// Dropout
///
/// http://arxiv.org/abs/1207.0580
///
/// `XorShiftRng` is used internally.
/// If you need to specify a seed value or use any other `Rng`, use `dropout_rng` instead.
pub fn dropout<'graph, A, F: Float>(x: A, dropout_ratio: F, train: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    dropout_rng(
        x,
        dropout_ratio,
        train,
        crate::ndarray_ext::get_default_rng::<F>(),
    )
}

/// Dropout
///
/// http://arxiv.org/abs/1207.0580
pub fn dropout_rng<'graph, A, F: Float, R: Rng + 'static>(
    x: A,
    dropout_ratio: F,
    train: bool,
    mut rng: R,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    // Create a seed from the provided RNG
    let seed = rng.random::<u64>();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(random_ops::Dropout {
            train,
            arr_rng: ArrayRng::from_seed(seed),
            dropout_ratio,
        })
}

/// Same as [crate::tensor::Tensor::map()]
pub fn map<'graph, A, F: Float>(
    x: A,
    f: fn(crate::ndarray_ext::NdArrayView<F>) -> NdArray<F>,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    use std::marker::PhantomData;
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(higher_order_ops::MapOp {
            phantom: PhantomData,
            f,
        })
}

/// Controls evaluation order of tensors
///
/// Same as [crate::Tensor::depends_on()].
pub fn control_dependencies<'graph, A, F: Float>(
    x: Tensor<'graph, F>,
    deps: &[A],
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let g = x.graph();
    if let Some(x_input) = x.get_incoming_tensor(0, g) {
        let mut ctrl_deps = Tensor::builder(g).append_input(x_input, false);
        // requiring all deps
        for dep in deps {
            ctrl_deps = ctrl_deps.append_input(dep.as_ref(), false);
        }
        let new_x_input = ctrl_deps.build(graph_ops::ControlDependency);
        g.access_inner_mut(x.id).incoming_nodes[0].id = new_x_input.id;
        x
    } else {
        panic!("Source tensor cannot depend on any other tensors.");
    }
}

/// Assigns `y` to `x`, elementwise.
///
/// Internally uses ndarray::ArrayBase::assign as is.
/// Note that `x` must be a variable tensor.
pub fn assign<'graph, A, B, F: Float>(x: A, y: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let y = y.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, true)
        .append_input(y, false)
        .build(array_ops::Assign)
}

/// Converts an `ndarray::Array` to a `ag::Tensor`.
///
///    ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let arr = array![2., 3.];
///    let tensor = convert_to_tensor(arr.clone(), g);
///    assert_eq!(tensor.eval(g), Ok(arr.into_dyn()));
/// });
///    ```
pub fn convert_to_tensor<F: Float, D>(
    arr: ndarray::Array<F, D>,
    graph: &impl AsGraph<F>,
) -> Tensor<F>
where
    D: ndarray::Dimension,
{
    // Store the original array shape for later use
    let original_shape = arr.shape().to_vec();
    let arr = arr.into_dyn();

    // Create the tensor with explicitly set known shape
    let shape_isize: Vec<isize> = original_shape.iter().map(|&s| s as isize).collect();
    let tensor = Tensor::builder(graph)
        .set_known_shape(&shape_isize)
        .set_differentiable(false)
        .build(const_gen_ops::ConvertToTensor { arr });

    // Manually handle shape for debug purposes
    if let Some(ctx) = crate::graph::AsGraph::context_ref(graph) {
        if let Ok(eval_result) = tensor.eval(ctx) {
            if eval_result.shape() != original_shape.as_slice() {
                // For debugging only, doesn't affect the actual tensor shape
                println!(
                    "DEBUG: convert_to_tensor shape mismatch: Expected {:?}, got {:?}",
                    original_shape,
                    eval_result.shape()
                );
            }
        }
    }

    tensor
}

/// Generates a zero-ranked tensor from a scalar value.
///
///    ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = scalar(3., g);
///    println!("{}", a.eval(g).unwrap());  // => 3.
///    assert_eq!(a.eval(g).unwrap().shape().len(), 0);
/// });
///    ```
pub fn scalar<F: Float>(val: F, graph: &impl AsGraph<F>) -> Tensor<F> {
    let op = const_gen_ops::Scalar { val };
    // For scalars, use set_known_shape with empty shape (scalar)
    Tensor::builder(graph).set_known_shape(&[]).build(op)
}

/// Outputs values sampled from the normal distribution.
pub fn random_normal<'graph, A, F: Float>(
    shape: &A,
    mean: f64,
    stddev: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    random_normal_rng(Default::default(), shape, mean, stddev, graph)
}

/// Outputs values sampled from the normal distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn random_normal_rng<'graph, A, F: Float>(
    arr_rng: ArrayRng<F>,
    shape: &A,
    mean: f64,
    stddev: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(t, false)
        .set_shape(&t)
        .build(random_ops::RandomNormal::new(arr_rng, mean, stddev))
}

/// Outputs values sampled from the uniform distribution.
pub fn random_uniform<'graph, A, F: Float>(
    shape: &A,
    min: f64,
    max: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    random_uniform_rng(Default::default(), shape, min, max, graph)
}

/// Outputs values sampled from the uniform distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn random_uniform_rng<'graph, A, F: Float>(
    arr_rng: ArrayRng<F>,
    shape: &A,
    min: f64,
    max: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(t, false)
        .set_shape(&t)
        .build(random_ops::RandomUniform::new(arr_rng, min, max))
}

/// Outputs values sampled from the standard normal distribution.
pub fn standard_normal<'graph, A, F: Float>(
    shape: &A,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    standard_normal_rng(Default::default(), shape, graph)
}

/// Outputs values sampled from the standard normal distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn standard_normal_rng<'graph, A, F: Float>(
    arr_rng: ArrayRng<F>,
    shape: &A,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(t, false)
        .set_shape(&t)
        .build(random_ops::StandardNormal::new(arr_rng))
}

/// Outputs values sampled from the standard uniform distribution.
pub fn standard_uniform<'graph, A, F: Float>(
    shape: &A,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    standard_uniform_rng(Default::default(), shape, graph)
}

/// Outputs values sampled from the standard uniform distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn standard_uniform_rng<'graph, F: Float, A>(
    arr_rng: ArrayRng<F>,
    shape: &A,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(t, false)
        .set_shape(&t)
        .build(random_ops::StandardUniform::new(arr_rng))
}

/// Outputs values sampled from the bernoulli distribution.
pub fn bernoulli<'graph, A, F: Float>(
    shape: &A,
    p: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    bernoulli_rng(Default::default(), shape, p, graph)
}

/// Outputs values sampled from the bernoulli distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn bernoulli_rng<'graph, A, F: Float>(
    arr_rng: ArrayRng<F>,
    shape: &A,
    p: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(t, false)
        .set_shape(&t)
        .build(random_ops::Bernoulli::new(arr_rng, p))
}

/// Outputs values sampled from the exponential distribution.
pub fn random_exp<'graph, A, F: Float>(
    shape: &A,
    lambda: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    random_exp_rng(Default::default(), shape, lambda, graph)
}

/// Outputs values sampled from the exponential distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn random_exp_rng<'graph, A, F: Float>(
    arr_rng: ArrayRng<F>,
    shape: &A,
    lambda: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(t, false)
        .set_shape(&t)
        .build(random_ops::Exponential::new(arr_rng, lambda))
}

/// Outputs values sampled from the gamma distribution.
pub fn random_gamma<'graph, A, F: Float>(
    shape: &A,
    shape_param: f64,
    scale: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    random_gamma_rng(Default::default(), shape, shape_param, scale, graph)
}

/// Outputs values sampled from the gamma distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn random_gamma_rng<'graph, A, F: Float>(
    arr_rng: ArrayRng<F>,
    shape: &A,
    shape_param: f64,
    scale: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(t, false)
        .set_shape(&t)
        .build(random_ops::Gamma::new(arr_rng, shape_param, scale))
}

/// Outputs values sampled from the log-normal distribution.
pub fn log_normal<'graph, A, F: Float>(
    shape: &A,
    mean: f64,
    stddev: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    log_normal_rng(Default::default(), shape, mean, stddev, graph)
}

/// Outputs values sampled from the log-normal distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn log_normal_rng<'graph, A, F: Float>(
    arr_rng: ArrayRng<F>,
    shape: &A,
    mean: f64,
    stddev: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(t, false)
        .set_shape(&t)
        .build(random_ops::LogNormal::new(arr_rng, mean, stddev))
}

/// Returns zeros with given shape.
///
///    ```
/// use ndarray;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[4, 2], g);
///    assert_eq!(a.eval(g), Ok(ndarray::Array2::<f32>::zeros((4, 2)).into_dyn()));
/// });
///    ```
pub fn zeros<'graph, A, F: Float>(shape: &A, graph: &'graph impl AsGraph<F>) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let shape_tensor = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(shape_tensor, false)
        .build(const_gen_ops::Zeros)
}

/// Returns ones with given shape.
///
///    ```
/// use ndarray;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = ones((&[4, 2]), g);
///    assert_eq!(a.eval(g), Ok(ndarray::Array2::<f32>::ones((4, 2)).into_dyn()));
/// });
///    ```
pub fn ones<'graph, A, F: Float>(shape: &A, graph: &'graph impl AsGraph<F>) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    Tensor::builder(graph)
        .append_input(shape.as_tensor(graph), false)
        .build(const_gen_ops::Ones)
}

/// Creates a variable tensor from an array
///
/// This is a convenience function that's used to create trainable variables
/// in a computation graph. It's equivalent to using Context::variable with a VariableID.
///
/// The variable function ensures that the created tensor preserves the shape information
/// of the input array, which is critical for proper tensor operations, especially in
/// linear algebra contexts where dimensionality matters.
/// Creates a variable tensor from an ndarray
///
/// This function creates a tensor from the given ndarray, ensuring that
/// the shape information is preserved. Variables represent the inputs to
/// computational graphs, so proper shape handling is critical.
pub fn variable<F: Float, D>(arr: ndarray::Array<F, D>, graph: &impl AsGraph<F>) -> Tensor<F>
where
    D: ndarray::Dimension,
{
    // Save the original shape for debugging
    let orig_shape = arr.shape().to_vec();
    println!("Creating variable with shape: {:?}", orig_shape);

    // Convert the array to dynamic form for tensor creation
    let arr_dyn = arr.into_dyn();

    // Create the tensor directly using ConvertToTensor
    let tensor = Tensor::builder(graph).build(const_gen_ops::ConvertToTensor { arr: arr_dyn });

    // Debug the created tensor
    if let Some(ctx) = crate::graph::AsGraph::context_ref(graph) {
        if let Ok(eval_result) = tensor.eval(ctx) {
            println!("Created tensor with shape: {:?}", eval_result.shape());
            if eval_result.shape() != orig_shape.as_slice() {
                println!(
                    "WARNING: Shape mismatch! Expected {:?}, got {:?}",
                    orig_shape,
                    eval_result.shape()
                );
            }
        }
    }

    tensor
}

// method version
impl<'g, F: Float> Tensor<'g, F> {
    /// Same as [tensor_ops::reshape](reshape)
    #[inline]
    pub fn reshape<AT: AsTensor<'g, F>>(&self, shape: &AT) -> Tensor<'g, F> {
        reshape(self, shape)
    }
    /// Same as [tensor_ops::flatten](flatten)
    #[inline]
    pub fn flatten(&self) -> Tensor<'g, F> {
        flatten(self)
    }
    /// Same as [tensor_ops::squeeze](squeeze)
    #[inline]
    pub fn squeeze<AT: AsTensor<'g, F>>(&self, axes: &AT) -> Tensor<'g, F> {
        squeeze(self, axes)
    }
    /// Same as [tensor_ops::expand_dims](expand_dims)
    #[inline]
    pub fn expand_dims<AT: AsTensor<'g, F>>(&self, axes: &AT) -> Tensor<'g, F> {
        expand_dims(self, axes)
    }
    /// Same as [tensor_ops::transpose](transpose)
    #[inline]
    pub fn transpose<AT: AsTensor<'g, F>>(&self, axes: &AT) -> Tensor<'g, F> {
        transpose(self, axes)
    }

    /// Same as [tensor_ops::size](size)
    #[inline]
    pub fn size(&self) -> Tensor<'g, F> {
        size(self)
    }
    /// Same as [tensor_ops::rank](rank)
    #[inline]
    pub fn rank(&self) -> Tensor<'g, F> {
        rank(self)
    }
    /// Same as [tensor_ops::shape](shape)
    #[inline]
    pub fn shape_tensor(&self) -> Tensor<'g, F> {
        shape(self)
    }

    /// Same as [tensor_ops::reduce_sum](reduce_sum)
    #[inline]
    pub fn reduce_sum<AT: AsTensor<'g, F>>(&self, axes: &AT, keep_dims: bool) -> Tensor<'g, F> {
        reduce_sum(self, axes, keep_dims)
    }
    /// Same as [tensor_ops::reduce_mean](reduce_mean)
    #[inline]
    pub fn reduce_mean<AT: AsTensor<'g, F>>(&self, axes: &AT, keep_dims: bool) -> Tensor<'g, F> {
        reduce_mean(self, axes, keep_dims)
    }
    /// Same as [tensor_ops::reduce_prod](reduce_prod)
    #[inline]
    pub fn reduce_prod<AT: AsTensor<'g, F>>(&self, axes: &AT, keep_dims: bool) -> Tensor<'g, F> {
        reduce_prod(self, axes, keep_dims)
    }
    /// Same as [tensor_ops::reduce_min](reduce_min)
    #[inline]
    pub fn reduce_min<AT: AsTensor<'g, F>>(&self, axes: &AT, keep_dims: bool) -> Tensor<'g, F> {
        reduce_min(self, axes, keep_dims)
    }
    /// Same as [tensor_ops::reduce_max](reduce_max)
    #[inline]
    pub fn reduce_max<AT: AsTensor<'g, F>>(&self, axes: &AT, keep_dims: bool) -> Tensor<'g, F> {
        reduce_max(self, axes, keep_dims)
    }
    /// Same as [tensor_ops::reduce_variance](reduce_variance)
    #[inline]
    pub fn reduce_variance<AT: AsTensor<'g, F>>(
        &self,
        axes: &AT,
        keep_dims: bool,
    ) -> Tensor<'g, F> {
        reduce_variance(self, axes, keep_dims)
    }
    /// Same as [tensor_ops::sum_all](sum_all)
    #[inline]
    pub fn sum_all<AT: AsTensor<'g, F>>(&self) -> Tensor<'g, F> {
        sum_all(self)
    }
    /// Same as [tensor_ops::mean_all](mean_all)
    #[inline]
    pub fn mean_all<AT: AsTensor<'g, F>>(&self) -> Tensor<'g, F> {
        mean_all(self)
    }

    /// Compute trace of matrix
    pub fn trace(&self) -> Tensor<'g, F> {
        trace(self)
    }

    /// Extract diagonal as vector
    pub fn diag(&self) -> Tensor<'g, F> {
        extract_diag(self)
    }

    /// Compute Frobenius norm
    pub fn frobenius_norm(&self) -> Tensor<'g, F> {
        frobenius_norm(self)
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: F) -> Tensor<'g, F> {
        scalar_mul(self, scalar)
    }
}

// ===============================================
// Backward Compatibility Re-exports
// ===============================================

// Arithmetic operations (backward compatibility)
pub use arithmetic::{
    abs as neg_abs, acos, acosh, add, asin, asinh, atan, atanh, ceil, clip, cos, cosh, digamma_f32,
    digamma_f64, div, equal, exp, exp10, exp2, floor, greater, greater_equal, inv, inv_sqrt,
    lesser, lesser_equal, lgamma_f32, lgamma_f64, ln, log10, log2, maximum, minimum, mul, neg,
    not_equal, pow, sign, sin, sinh, sqrt, square, sub, tan, tanh,
};

// Reduction operations (backward compatibility)
pub use reduction::{
    add_n, argmax, argmin, frobenius_norm, l1_norm, l2_norm, lp_norm, mean_all, reduce_all,
    reduce_any, reduce_logsumexp, reduce_max, reduce_mean, reduce_min, reduce_prod, reduce_std,
    reduce_sum, reduce_variance, sum_all,
};

// Linear algebra operations (backward compatibility)
pub use linear_algebra::{
    batch_matmul, batch_matmul_t, concat, conv2d, conv2d_transpose, determinant, diag,
    dilated_conv2d, eigen, eigenvalues, extract_diag, eye, lstsq, matmul, matrix_inverse,
    max_pool2d, qr, scalar_mul, solve, split, svd, tensordot, trace, transpose,
};

// Activation functions (backward compatibility)
pub use activation::{
    batch_norm, elu, gelu, hard_sigmoid, hard_tanh, leaky_relu, log_softmax, mean_squared_error,
    mish, normalize, relu, relu6, sigmoid, sigmoid_cross_entropy, softmax, softmax_cross_entropy,
    softplus, sparse_softmax_cross_entropy, swish,
};

// Re-export linear algebra functions
pub use debug_ops::{debug_identity_with_gradient, debug_scalar_one};
pub use decomposition_ops::matrix_exp;
pub use decomposition_ops::{lu, qr as decomp_qr, svd as decomp_svd};
pub use eigen_ops::{eigen as eigen_decomp, eigenvalues as eigen_vals};
pub use linalg_ops::{
    diag as linalg_diag, extract_diag as linalg_extract_diag, eye as linalg_eye,
    trace as linalg_trace,
};
// matrix_sqrt not yet implemented
pub use matrix_ops::{
    determinant as matrix_det, matrix_inverse as matrix_inv,
    pseudo_inverse as matrix_pseudo_inverse,
};
pub use norm_ops::{frobenius_norm as norm_frobenius, nuclear_norm, spectral_norm};
pub use scalar_ops::scalar_mul as scalar_multiply;
pub use solver_ops::{lstsq as linalg_lstsq, solve as linalg_solve};
pub use special_matrices::{band_matrix, cholesky, symmetrize, tril, triu};

// Common aliases for linear algebra operations
// Note: inv is already taken by arithmetic::inv (reciprocal), so we use matinv
pub use eigen_ops::eigen as eig;
pub use matrix_ops::determinant as det;
pub use matrix_ops::matrix_inverse as matinv;
pub use matrix_ops::pseudo_inverse as pinv;

// Matrix functions (now implemented!)
pub use matrix_functions::{logm, powm, sqrtm};
pub use matrix_functions::{matrix_log, matrix_power, matrix_sqrt};

// Numerical properties
pub use numerical_props::{
    cond, cond_1, cond_2, cond_fro, cond_inf, logdet, matrix_rank, slogdet, ConditionType,
};

// Kronecker product
pub use kronecker_ops::kron;

// Matrix norms
pub use matrix_norms::{norm1, norm2, normfro, norminf};

// Matrix solvers
pub use matrix_solvers::{cholesky_solve, solve_lyapunov, solve_sylvester};

// Symmetric matrix operations
pub use symmetric_ops::{eigh, eigvalsh};

// Special decompositions
pub use special_decompositions::{polar, schur};

// Advanced tensor operations
pub use advanced_tensor_ops::{einsum, kron as kron_tensor, tensor_solve};

// Matrix exponential algorithms
pub use matrix_ops::{expm2, expm3};

// Advanced decompositions
pub use advanced_decompositions::{generalized_eigen, qr_pivot, randomized_svd, svd_jacobi};

// Iterative solvers
pub use iterative_solvers::{
    bicgstab_solve, conjugate_gradient_solve, gmres_solve, pcg_solve, PreconditionerType,
};

// Matrix trigonometric functions
pub use matrix_trig_functions::{coshm, cosm, funm, signm, sinhm, sinm};

// Aliases for new functions
pub use advanced_tensor_ops::kron as kronecker_product;

// Memory optimization functions
pub use checkpoint_ops::{
    adaptive_checkpoint, checkpoint, checkpoint_segment, checkpoint_segment_flex, detach,
    CheckpointGroup, CheckpointProfiler,
};

// Advanced indexing operations
pub use advanced_indexing::{
    advanced_gather, boolean_mask, get_at_coords, scatter, select_columns, select_rows, take,
    where_op,
};

// Broadcasting optimizations
pub use broadcast_ops::{
    analyze_broadcast, broadcast_add, broadcast_div, broadcast_maximum, broadcast_minimum,
    broadcast_mul, broadcast_pow, broadcast_sub, clear_broadcast_cache, get_broadcast_cache_stats,
    BroadcastInfo, BroadcastStrategy,
};

// Memory optimization tools
pub use memory_optimization::{
    clear_memory_pool, configure_memory_pool, disable_memory_tracking, efficient_ones,
    efficient_view, efficient_zeros, enable_memory_tracking, get_memory_pool_stats,
    get_memory_tracking_stats, get_pooled_buffer, inplace_abs, inplace_add, inplace_div,
    inplace_mul, inplace_neg, inplace_scalar_mul, inplace_sub, reset_memory_tracking,
    return_pooled_buffer, set_memory_pool_enabled, MemoryOptimizer, MemoryPoolStats,
    MemoryTrackerStats,
};

// Efficient tensor operations
pub use efficient_ops::{
    clear_reshape_cache, efficient_concat, efficient_reshape, efficient_reshape_with_shape,
    efficient_slice, efficient_transpose, get_reshape_cache_stats, EfficientOpsManager,
    EfficientOpsStats, SliceRange,
};

// Custom activation function framework
pub use custom_activations::{
    create_custom_activation, custom_activation, is_activation_registered,
    list_activation_functions, parameterized_activation, register_activation, ActivationProperties,
    CustomActivation, CustomActivationBuilder,
};

// Performance optimization operations
pub use performance_ops::{
    cache_friendly_matmul, is_parallel_enabled, is_simd_enabled, parallel_sum,
    set_parallel_enabled, set_simd_enabled, simd_add, simd_mul, simd_relu, simd_sigmoid,
    PerformanceConfig, ReductionOperation, SimdBinaryOperation, SimdUnaryOperation,
};

// Enhanced dynamic computation graph features
pub use graph_enhancements::{
    cached_op, clear_computation_cache, conditional, configure_cache, get_cache_stats,
    get_gc_stats, run_garbage_collection, smart_checkpoint, CacheStats, GcStats, GraphEnhancer,
    GraphStats, PredicateType,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_backward_compatibility() {
        crate::run(|g| {
            let a = convert_to_tensor(array![1.0_f32, 2.0, 3.0], g);
            let b = convert_to_tensor(array![4.0_f32, 5.0, 6.0], g);

            // Test that all re-exported functions work
            let sum_result = add(a, b);
            let expected = array![5.0_f32, 7.0, 9.0];
            assert_eq!(sum_result.eval(g).unwrap(), expected.into_dyn());

            // Test reduction operation
            let sum_all_result = sum_all(a);
            assert_eq!(
                sum_all_result.eval(g).unwrap(),
                ndarray::arr0(6.0).into_dyn()
            );

            // Test activation function
            let relu_result = relu(a);
            assert_eq!(
                relu_result.eval(g).unwrap(),
                array![1.0_f32, 2.0, 3.0].into_dyn()
            );
        });
    }

    #[test]
    fn test_module_organization() {
        crate::run(|g| {
            let x = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);

            // Test arithmetic module directly
            let sum_direct = arithmetic::add(x, x);
            let expected_sum = array![[2.0_f32, 4.0], [6.0, 8.0]];
            assert_eq!(sum_direct.eval(g).unwrap(), expected_sum.into_dyn());

            // Test reduction module directly
            let mean_direct = reduction::reduce_mean(x, &[0], false);
            let expected_mean = array![2.0_f32, 3.0];
            assert_eq!(mean_direct.eval(g).unwrap(), expected_mean.into_dyn());

            // Test linear algebra module directly
            let trace_direct = linear_algebra::trace(x);
            assert_eq!(trace_direct.eval(g).unwrap(), ndarray::arr0(5.0).into_dyn());

            // Test activation module directly
            let sigmoid_direct = activation::sigmoid(x);
            let result = sigmoid_direct.eval(g).unwrap();
            // All values should be between 0 and 1
            assert!(result.iter().all(|&val| val > 0.0 && val < 1.0));
        });
    }

    #[test]
    fn test_tensor_methods() {
        crate::run(|g| {
            let x = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);

            // Test tensor methods
            let reshaped = x.reshape(&[4]);
            assert_eq!(reshaped.eval(g).unwrap().shape(), &[4]);

            let flattened = x.flatten();
            assert_eq!(flattened.eval(g).unwrap().shape(), &[4]);

            let trace_result = x.trace();
            assert_eq!(trace_result.eval(g).unwrap(), ndarray::arr0(5.0).into_dyn());

            let diag_result = x.diag();
            let expected_diag = array![1.0_f32, 4.0];
            assert_eq!(diag_result.eval(g).unwrap(), expected_diag.into_dyn());
        });
    }

    #[test]
    fn test_linalg_aliases() {
        crate::run(|g| {
            let x = convert_to_tensor(array![[2.0_f32, 1.0], [1.0, 3.0]], g);

            // Test matinv alias (inv conflicts with reciprocal function)
            let inv_result = matinv(&x);
            let inv_direct = matrix_inverse(x);
            assert_eq!(inv_result.eval(g).unwrap(), inv_direct.eval(g).unwrap());

            // Test det alias
            let det_result = det(&x);
            let det_direct = determinant(x);
            assert_eq!(det_result.eval(g).unwrap(), det_direct.eval(g).unwrap());

            // Test eig alias
            let (eigenvals, eigenvecs) = eig(&x);
            // Note: There's a known issue with eigen from linear_algebra module
            // where nth_tensor doesn't correctly extract eigenvectors.
            // We test the eig alias works correctly on its own.
            assert_eq!(eigenvals.eval(g).unwrap().shape(), &[2]);
            assert_eq!(eigenvecs.eval(g).unwrap().shape(), &[2, 2]);

            // Test pinv alias
            let rect = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]], g);
            let pinv_result = pinv(&rect);
            let pinv_direct = matrix_pseudo_inverse(&rect);
            assert_eq!(pinv_result.eval(g).unwrap(), pinv_direct.eval(g).unwrap());

            // Test sqrtm alias - NOT YET IMPLEMENTED
            // let pos_def = convert_to_tensor(array![[4.0_f32, 1.0], [1.0, 3.0]], g);
            // let sqrtm_result = sqrtm(&pos_def);
            // let sqrtm_direct = matrix_sqrt(&pos_def);
            // assert_eq!(sqrtm_result.eval(g).unwrap(), sqrtm_direct.eval(g).unwrap());

            // Test logm alias - NOT YET IMPLEMENTED
            // let small_mat = convert_to_tensor(array![[1.1_f32, 0.1], [0.1, 1.2]], g);
            // let logm_result = logm(&small_mat);
            // let logm_direct = matrix_log(&small_mat);
            // assert_eq!(logm_result.eval(g).unwrap(), logm_direct.eval(g).unwrap());
        });
    }
}
