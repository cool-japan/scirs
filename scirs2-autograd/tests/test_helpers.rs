//! Helper functions for tests that require graph context
//!
//! This module provides utilities to properly set up graph contexts
//! for tests that need to create and manipulate tensors.

use scirs2_autograd::graph::{run, Context};
use scirs2_autograd::tensor::Tensor;
use scirs2_autograd::Float;

/// Run a test function within a proper graph context
///
/// This helper sets up a graph context and passes it to the test function,
/// ensuring that tensors can be properly created and manipulated.
///
/// # Example
/// ```ignore
/// use test_helpers::with_graph_context;
///
/// #[test]
/// fn my_test() {
///     with_graph_context(|ctx| {
///         let tensor = create_test_tensor_in_context(ctx, vec![2, 3]);
///         // ... test code using tensor
///     });
/// }
/// ```
#[allow(dead_code)]
pub fn with_graph_context<F, R, TestFn>(test_fn: TestFn) -> R
where
    F: Float,
    TestFn: FnOnce(&mut Context<F>) -> R,
{
    run(test_fn)
}

/// Create a test tensor within a graph context
///
/// This function creates a tensor with test data within the provided graph context.
/// The tensor values are initialized as 0.1 * index for easy verification.
#[allow(dead_code)]
pub fn create_test_tensor_in_context<'a, F>(
    ctx: &'a mut Context<F>,
    shape: Vec<usize>,
) -> Tensor<'a, F>
where
    F: Float,
{
    use ndarray::{Array, IxDyn};
    use scirs2_autograd::tensor_ops as T;

    let size: usize = shape.iter().product();
    let data: Vec<F> = (0..size)
        .map(|i| F::from(i).unwrap() * F::from(0.1).unwrap())
        .collect();

    let arr = Array::from_shape_vec(IxDyn(&shape), data).expect("Failed to create array");

    T::convert_to_tensor(arr, ctx)
}

/// Create a tensor with uncertainty for stability testing
#[allow(dead_code)]
pub fn create_uncertainty_tensor_in_context<'a, F>(
    ctx: &'a mut Context<F>,
    shape: Vec<usize>,
    magnitude: f64,
) -> Tensor<'a, F>
where
    F: Float,
{
    use ndarray::{Array, IxDyn};
    use rand::prelude::*;
    use rand::rng;
    use scirs2_autograd::tensor_ops as T;

    let size: usize = shape.iter().product();
    let mut rng = rng();

    let data: Vec<F> = (0..size)
        .map(|_| {
            let noise = rng.random_range(-magnitude..magnitude);
            F::from(noise).unwrap()
        })
        .collect();

    let arr = Array::from_shape_vec(IxDyn(&shape), data).expect("Failed to create array");

    T::convert_to_tensor(arr, ctx)
}
