//! Reduction tensor operations
//!
//! This module provides operations that reduce tensor dimensions by aggregating
//! values along specified axes:
//! - Basic reductions (sum, mean, product)
//! - Statistical reductions (min, max, variance, standard deviation)
//! - Advanced reductions (norm operations, aggregation functions)
//! - Specialized reductions (all, any, argmin, argmax)

use crate::tensor::{AsTensor, Tensor};
use crate::Float;

// Import internal operation modules
use crate::tensor_ops::{array_ops, math_ops, reduction_ops, shape};

/// Takes sum along specified axes.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Axes along which to compute the sum
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::reduction::reduce_sum;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let s1 = reduce_sum(x, &[0], false); // sum along axis 0
///    assert_eq!(s1.eval(g), Ok(array![4., 6.].into_dyn()));
///
///    let s2 = reduce_sum(x, &[1], true); // sum along axis 1, keep dims
///    assert_eq!(s2.eval(g), Ok(array![[3.], [7.]].into_dyn()));
/// });
/// ```
pub fn reduce_sum<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(axes.as_tensor(g), false)
        .build(reduction_ops::ReduceSum {
            keep_dims,
            sparse_axes: false,
        })
}

/// Takes mean along specified axes.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Axes along which to compute the mean
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::reduction::reduce_mean;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let m = reduce_mean(x, &[0], false);
///    assert_eq!(m.eval(g), Ok(array![2., 3.].into_dyn()));
/// });
/// ```
pub fn reduce_mean<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(axes.as_tensor(g), false)
        .build(reduction_ops::ReduceMean {
            keep_dims,
            sparse_axes: false,
        })
}

/// Takes product along specified axes.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Axes along which to compute the product
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::reduction::reduce_prod;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let p = reduce_prod(x, &[0], false);
///    assert_eq!(p.eval(g), Ok(array![3., 8.].into_dyn()));
/// });
/// ```
pub fn reduce_prod<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(axes.as_tensor(g), false)
        .build(reduction_ops::ReduceProd {
            keep_dims,
            sparse_axes: false,
        })
}

/// Takes min along specified axes.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Axes along which to compute the minimum
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::reduction::reduce_min;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 4.], [2., 3.]], g);
///    let min_val = reduce_min(x, &[0], false);
///    assert_eq!(min_val.eval(g), Ok(array![1., 3.].into_dyn()));
/// });
/// ```
pub fn reduce_min<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(axes.as_tensor(g), false)
        .build(reduction_ops::ReduceMin {
            keep_dims,
            sparse_axes: false,
        })
}

/// Takes max along specified axes.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Axes along which to compute the maximum
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::reduction::reduce_max;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 4.], [2., 3.]], g);
///    let max_val = reduce_max(x, &[0], false);
///    assert_eq!(max_val.eval(g), Ok(array![2., 4.].into_dyn()));
/// });
/// ```
pub fn reduce_max<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(axes.as_tensor(g), false)
        .build(reduction_ops::ReduceMax {
            keep_dims,
            sparse_axes: false,
        })
}

/// Computes variance along specified axes.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Axes along which to compute the variance
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::reduction::reduce_variance;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let var = reduce_variance(x, &[0], false);
///    // Variance of [1,3] and [2,4] along axis 0
///    assert_eq!(var.eval(g), Ok(array![1., 1.].into_dyn()));
/// });
/// ```
pub fn reduce_variance<'graph, A, AT, F: Float>(
    x: A,
    axes: &AT,
    keep_dims: bool,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(axes.as_tensor(g), false)
        .build(reduction_ops::ReduceVariance {
            keep_dims,
            sparse_axes: false,
        })
}

/// Computes standard deviation along specified axes.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Axes along which to compute the standard deviation
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
pub fn reduce_std<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let variance = reduce_variance(x, axes, keep_dims);
    crate::tensor_ops::arithmetic::sqrt(variance)
}

/// Sums all elements in the tensor.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::reduction::sum_all;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let total = sum_all(x);
///    assert_eq!(total.eval(g), Ok(ndarray::arr0(10.).into_dyn()));
/// });
/// ```
pub fn sum_all<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(reduction_ops::ReduceSumAll)
}

/// Computes the mean of all elements in the tensor.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::reduction::mean_all;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let avg = mean_all(x);
///    assert_eq!(avg.eval(g), Ok(ndarray::arr0(2.5).into_dyn()));
/// });
/// ```
pub fn mean_all<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(reduction_ops::ReduceMeanAll)
}

/// Finds the indices of the maximum values along specified axes.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axis` - Axis along which to find argmax
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::reduction::argmax;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 4.], [2., 3.]], g);
///    let indices = argmax(x, 0, false);
///    assert_eq!(indices.eval(g), Ok(array![1., 0.].into_dyn()));
/// });
/// ```
pub fn argmax<'graph, A, F: Float>(x: A, axis: isize, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(reduction_ops::ArgMax {
            axis,
            keep_dim: keep_dims,
        })
}

/// Finds the indices of the minimum values along specified axes.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axis` - Axis along which to find argmin
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::reduction::argmin;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 4.], [2., 3.]], g);
///    let indices = argmin(x, 0, false);
///    assert_eq!(indices.eval(g), Ok(array![0., 1.].into_dyn()));
/// });
/// ```
pub fn argmin<'graph, A, F: Float>(x: A, axis: isize, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(reduction_ops::ArgMin {
            axis,
            keep_dim: keep_dims,
        })
}

/// Computes `log(sum(exp(x)))` along specified axis.
///
/// `axis` can be negative.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::reduction::reduce_logsumexp;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![1., 2., 3.], g);
///    let lse = reduce_logsumexp(x, 0, false);
///    // Should be approximately log(e^1 + e^2 + e^3) ≈ 3.407
///    let result = lse.eval(g).unwrap();
///    assert!((result[ndarray::IxDyn(&[])] - 3.407_f64).abs() < 0.01);
/// });
/// ```
pub fn reduce_logsumexp<'graph, A, F: Float>(x: A, axis: isize, keep_dim: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = math_ops::LogSumExp {
        axis,
        keep_dims: keep_dim,
    };
    Tensor::builder(g).append_input(x.as_ref(), false).build(op)
}

/// Adds all input tensors, element-wise.
///
/// All the input tensors must have same shapes.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::reduction::add_n;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::ones((&[2, 2]), g);
///    let b = ag::tensor_ops::ones((&[2, 2]), g);
///    let c = ag::tensor_ops::ones((&[2, 2]), g);
///    let d = add_n(&[a, b, c]);
///
///    assert_eq!(d.eval(g).unwrap().shape(), &[2, 2]);
///    assert_eq!(d.eval(g), Ok(array![[3., 3.], [3., 3.]].into_dyn()));
/// });
/// ```
pub fn add_n<'graph, A, F: Float>(xs: &[A]) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let len = xs.len();
    assert_ne!(len, 0);
    if len == 1 {
        *xs[0].as_ref()
    } else {
        let g = xs[0].as_ref().graph();
        let mut b = Tensor::builder(g);
        for x in xs {
            b = b.append_input(x.as_ref(), false);
        }
        b.set_shape(&shape(xs[0])).build(array_ops::AddN)
    }
}

/// Computes L1 norm along specified axes.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Axes along which to compute the norm
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
pub fn l1_norm<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let abs_x = crate::tensor_ops::arithmetic::abs(x);
    reduce_sum(abs_x, axes, keep_dims)
}

/// Computes L2 norm along specified axes.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Axes along which to compute the norm
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
pub fn l2_norm<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let square_x = crate::tensor_ops::arithmetic::square(x);
    let sum_square = reduce_sum(square_x, axes, keep_dims);
    crate::tensor_ops::arithmetic::sqrt(sum_square)
}

/// Computes Lp norm along specified axes.
///
/// # Arguments
/// * `x` - Input tensor
/// * `p` - The order of the norm
/// * `axes` - Axes along which to compute the norm
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
pub fn lp_norm<'graph, A, AT, F: Float>(x: A, p: F, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let abs_x = crate::tensor_ops::arithmetic::abs(x);
    let pow_x = crate::tensor_ops::arithmetic::pow(abs_x, p);
    let sum_pow = reduce_sum(pow_x, axes, keep_dims);
    let one_over_p = F::one() / p;
    crate::tensor_ops::arithmetic::pow(sum_pow, one_over_p)
}

/// Computes the Frobenius norm (L2 norm of flattened tensor).
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::reduction::frobenius_norm;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[3., 4.], [0., 0.]], g);
///    let norm = frobenius_norm(x);
///    assert_eq!(norm.eval(g), Ok(ndarray::arr0(5.).into_dyn()));
/// });
/// ```
pub fn frobenius_norm<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let square_x = crate::tensor_ops::arithmetic::square(x);
    let sum_square = sum_all(square_x);
    crate::tensor_ops::arithmetic::sqrt(sum_square)
}

/// Tests if all elements evaluate to true.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Axes along which to test
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
pub fn reduce_all<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(axes.as_tensor(g), false)
        .build(reduction_ops::ReduceAll { keep_dims })
}

/// Tests if any element evaluates to true.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Axes along which to test
/// * `keep_dims` - If true, keeps reduced dimensions as size 1
pub fn reduce_any<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(axes.as_tensor(g), false)
        .build(reduction_ops::ReduceAny { keep_dims })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::convert_to_tensor;
    #[allow(unused_imports)]
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_basic_reductions() {
        crate::run(|g| {
            let x = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);

            // Test sum reduction
            let sum_result = reduce_sum(x, &[0], false);
            let expected_sum = array![4.0_f32, 6.0];
            assert_eq!(sum_result.eval(g).unwrap(), expected_sum.into_dyn());

            // Test mean reduction
            let mean_result = reduce_mean(x, &[0], false);
            let expected_mean = array![2.0_f32, 3.0];
            assert_eq!(mean_result.eval(g).unwrap(), expected_mean.into_dyn());

            // Test product reduction
            let prod_result = reduce_prod(x, &[0], false);
            let expected_prod = array![3.0_f32, 8.0];
            assert_eq!(prod_result.eval(g).unwrap(), expected_prod.into_dyn());
        });
    }

    #[test]
    fn test_min_max_reductions() {
        crate::run(|g| {
            let x = convert_to_tensor(array![[1.0_f32, 4.0], [2.0, 3.0]], g);

            // Test min reduction
            let min_result = reduce_min(x, &[0], false);
            let expected_min = array![1.0_f32, 3.0];
            assert_eq!(min_result.eval(g).unwrap(), expected_min.into_dyn());

            // Test max reduction
            let max_result = reduce_max(x, &[0], false);
            let expected_max = array![2.0_f32, 4.0];
            assert_eq!(max_result.eval(g).unwrap(), expected_max.into_dyn());
        });
    }

    #[test]
    fn test_statistical_reductions() {
        crate::run(|g| {
            let x = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);

            // Test variance reduction
            let var_result = reduce_variance(x, &[0], false);
            let expected_var = array![1.0_f32, 1.0]; // Variance of [1,3] and [2,4]
            assert_eq!(var_result.eval(g).unwrap(), expected_var.into_dyn());

            // Test standard deviation
            let std_result = reduce_std(x, &[0], false);
            let expected_std = array![1.0_f32, 1.0];
            assert_eq!(std_result.eval(g).unwrap(), expected_std.into_dyn());
        });
    }

    #[test]
    fn test_global_reductions() {
        crate::run(|g| {
            let x = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);

            // Test sum all
            let sum_all_result = sum_all(x);
            assert_eq!(
                sum_all_result.eval(g).unwrap(),
                ndarray::arr0(10.0).into_dyn()
            );

            // Test mean all
            let mean_all_result = mean_all(x);
            assert_eq!(
                mean_all_result.eval(g).unwrap(),
                ndarray::arr0(2.5).into_dyn()
            );
        });
    }

    #[test]
    fn test_norm_operations() {
        crate::run(|g| {
            let x = convert_to_tensor(array![[3.0_f32, 4.0], [0.0, 0.0]], g);

            // Test Frobenius norm
            let frob_norm = frobenius_norm(x);
            assert_eq!(frob_norm.eval(g).unwrap(), ndarray::arr0(5.0).into_dyn());

            // Test L1 norm
            let l1_result = l1_norm(x, &[0], false);
            let expected_l1 = array![3.0_f32, 4.0];
            assert_eq!(l1_result.eval(g).unwrap(), expected_l1.into_dyn());

            // Test L2 norm
            let l2_result = l2_norm(x, &[0], false);
            let expected_l2 = array![3.0_f32, 4.0];
            assert_eq!(l2_result.eval(g).unwrap(), expected_l2.into_dyn());
        });
    }

    #[test]
    fn test_add_n() {
        crate::run(|g| {
            let a = convert_to_tensor(array![1.0_f32, 2.0], g);
            let b = convert_to_tensor(array![3.0_f32, 4.0], g);
            let c = convert_to_tensor(array![5.0_f32, 6.0], g);

            let result = add_n(&[a, b, c]);
            let expected = array![9.0_f32, 12.0];
            assert_eq!(result.eval(g).unwrap(), expected.into_dyn());
        });
    }

    #[test]
    fn test_keep_dims() {
        crate::run(|g| {
            let x = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);

            // Test with keep_dims=true
            let sum_result = reduce_sum(x, &[0], true);
            assert_eq!(sum_result.eval(g).unwrap().shape(), &[1, 2]);

            // Test with keep_dims=false
            let sum_result = reduce_sum(x, &[0], false);
            assert_eq!(sum_result.eval(g).unwrap().shape(), &[2]);
        });
    }
}
