//! Arithmetic tensor operations
//!
//! This module provides basic arithmetic operations for tensors including:
//! - Element-wise arithmetic (add, subtract, multiply, divide)
//! - Mathematical functions (sqrt, power, logarithms, exponentials)
//! - Trigonometric functions (sin, cos, tan and their inverses)
//! - Hyperbolic functions (sinh, cosh, tanh and their inverses)
//! - Comparison operations (equal, greater, less than, etc.)
//! - Special mathematical functions (gamma, digamma)

use crate::tensor::Tensor;
use crate::{Float, Graph};

// Import internal operation modules
use crate::tensor_ops::{binary_ops, math_ops, shape};

/// Elementwise addition.
///
/// This can be replaced with `+` operation of Tensor.
#[inline]
pub fn add<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let g = a.as_ref().graph();
    Tensor::builder(g)
        .set_shape(&infer_bin_op_shape(g, shape(a), shape(b)))
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(binary_ops::AddOp)
}

/// Element-wise subtraction.
///
/// This can be replaced with `-` operation of Tensor.
#[inline]
pub fn sub<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let g = a.as_ref().graph();
    Tensor::builder(g)
        .set_shape(&infer_bin_op_shape(g, shape(a), shape(b)))
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(binary_ops::SubOp)
}

/// Elementwise multiplication.
///
/// This can be replaced with `*` operation of Tensor.
#[inline]
pub fn mul<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let g = a.as_ref().graph();
    Tensor::builder(g)
        .set_shape(&infer_bin_op_shape(g, shape(a), shape(b)))
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(binary_ops::MulOp)
}

/// Elementwise division.
///
/// This can be replaced with `/` operation of Tensor.
#[inline]
pub fn div<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let g = a.as_ref().graph();
    Tensor::builder(g)
        .set_shape(&infer_bin_op_shape(g, shape(a), shape(b)))
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(binary_ops::DivOp)
}

/// Elementwise sqrt
#[inline]
pub fn sqrt<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Sqrt)
}

/// Elementwise pow
pub fn pow<'graph, A, F: Float>(x: A, a: F) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Pow { a })
}

/// Elementwise square
pub fn square<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    mul(x, x)
}

/// Elementwise absolute value
pub fn abs<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Abs)
}

/// Elementwise negation
pub fn neg<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::NegOp)
}

/// Elementwise base e (napier) logarithm
pub fn ln<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Ln)
}

/// Elementwise base 2 logarithm
pub fn log2<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Log2)
}

/// Elementwise base 10 logarithm
pub fn log10<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Log10)
}

/// Elementwise base e (napier) exponential
pub fn exp<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Exp)
}

/// Elementwise base 2 exponential
pub fn exp2<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Exp2)
}

/// Elementwise base 10 exponential
pub fn exp10<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Exp10)
}

/// Elementwise sine
pub fn sin<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Sin)
}

/// Elementwise cosine
pub fn cos<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Cos)
}

/// Elementwise tangent
pub fn tan<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Tan)
}

/// Elementwise arcsin
pub fn asin<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Asin)
}

/// Elementwise arccos
pub fn acos<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Acos)
}

/// Elementwise arctan
pub fn atan<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Atan)
}

/// Elementwise hyperbolic sine
pub fn sinh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Sinh)
}

/// Elementwise hyperbolic cosine
pub fn cosh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Cosh)
}

/// Elementwise hyperbolic tangent
pub fn tanh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Tanh)
}

/// Elementwise hyperbolic arcsin
pub fn asinh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Asinh)
}

/// Elementwise hyperbolic arccos
pub fn acosh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Acosh)
}

/// Elementwise hyperbolic arctan
pub fn atanh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Atanh)
}

/// Elementwise lgamma function
pub fn lgamma_f32<'graph, A>(x: A) -> Tensor<'graph, f32>
where
    A: AsRef<Tensor<'graph, f32>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .build(math_ops::Lgamma)
}

/// Elementwise lgamma function
pub fn lgamma_f64<'graph, A>(x: A) -> Tensor<'graph, f64>
where
    A: AsRef<Tensor<'graph, f64>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .build(math_ops::Lgamma)
}

/// Elementwise digamma function
///
/// NOTE: derivative not implemented
pub fn digamma_f32<'graph, A>(x: A) -> Tensor<'graph, f32>
where
    A: AsRef<Tensor<'graph, f32>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .build(math_ops::Digamma)
}

/// Elementwise digamma function
///
/// NOTE: derivative not implemented
pub fn digamma_f64<'graph, A>(x: A) -> Tensor<'graph, f64>
where
    A: AsRef<Tensor<'graph, f64>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .build(math_ops::Digamma)
}

/// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
///
///    ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::arithmetic::maximum;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![1., 2., 3.], g);
///    let b = ag::tensor_ops::convert_to_tensor(array![3., 2., 1.], g);
///    let c = maximum(a, b);
///    assert_eq!(c.eval(g), Ok(array![3., 2., 3.].into_dyn()));
/// });
///    ```
pub fn maximum<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::Maximum)
}

/// Returns the min of x and y (i.e. x > y ? y : x) element-wise.
///
///    ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::arithmetic::minimum;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![1., 2., 3.], g);
///    let b = ag::tensor_ops::convert_to_tensor(array![3., 2., 1.], g);
///    let c = minimum(a, b);
///    assert_eq!(c.eval(g), Ok(array![1., 2., 1.].into_dyn()));
/// });
///    ```
pub fn minimum<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::Minimum)
}

/// Compares a couple of tensors and returns a binary tensor.
///
/// if `a[i] == b[i]` then `return-value[i]` will be 1 else 0
///
/// # Panics
/// When broadcast is impossible
///
///    ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::arithmetic::equal;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![1., 2., 3.], g);
///    let b = ag::tensor_ops::convert_to_tensor(array![3., 2., 1.], g);
///    let c = equal(a, b);
///    assert_eq!(c.eval(g), Ok(ndarray::arr1(&[0., 1., 0.]).into_dyn()));
/// });
///    ```
pub fn equal<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::Equal)
}

/// Compares a couple of tensors and returns a binary tensor.
///
/// if `a[i] != b[i]` then `return-value[i]` will be 1 else 0
///
/// # Panics
/// When broadcast is impossible
///
///    ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::arithmetic::not_equal;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![1., 2., 3.], g);
///    let b = ag::tensor_ops::convert_to_tensor(array![3., 2., 1.], g);
///    let c = not_equal(a, b);
///    assert_eq!(c.eval(g), Ok(array![1., 0., 1.].into_dyn()));
/// });
///    ```
pub fn not_equal<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::NotEqual)
}

/// Compares a couple of tensors and returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn greater<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::Greater)
}

/// Compares a couple of tensors and returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn greater_equal<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::GreaterEqual)
}

/// Compares a couple of tensors and returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn lesser<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::Lesser)
}

/// Compares a couple of tensors and returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn lesser_equal<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::LesserEqual)
}

/// Returns the floor of the input, element-wise.
///
///    ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::arithmetic::floor;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0], g);
///    let b = floor(a);
///    assert_eq!(
///        b.eval(g),
///        Ok(array![-2., -2., -1.,  0.,  1.,  1.,  2.].into_dyn())
///    );
///
/// });
///    ```
pub fn floor<'graph, A, F: Float>(a: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .set_shape(&shape(a))
        .append_input(a.as_ref(), false)
        .build(math_ops::Floor)
}

/// Returns the ceiling of the input, element-wise.
///
///    ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::arithmetic::ceil;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0], g);
///    let b = ceil(a);
///    assert_eq!(
///        b.eval(g),
///        Ok(array![-1., -1., -0.,  1.,  2.,  2.,  2.].into_dyn())
///    );
///
/// });
///    ```
pub fn ceil<'graph, A, F: Float>(a: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .set_shape(&shape(a))
        .append_input(a.as_ref(), false)
        .build(math_ops::Ceil)
}

/// Elementwise inverse square root (1/sqrt(x))
pub fn inv_sqrt<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let one = crate::tensor_ops::scalar(F::one(), x.as_ref().graph());
    div(one, sqrt(x))
}

/// Clamps values to the range [min_value, max_value]
pub fn clip<'graph, A, F: Float>(x: A, min_value: F, max_value: F) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();

    let min_tensor = crate::tensor_ops::scalar(min_value, g);
    let max_tensor = crate::tensor_ops::scalar(max_value, g);

    // clip(x, min, max) = max(min(x, max), min)
    let clipped_upper = minimum(x, max_tensor);
    maximum(clipped_upper, min_tensor)
}

/// Elementwise reciprocal (1/x)
pub fn inv<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let one = crate::tensor_ops::scalar(F::one(), x.as_ref().graph());
    div(one, x)
}

/// Elementwise sign function
pub fn sign<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Sign)
}

#[inline]
fn infer_bin_op_shape<'graph, A, B, F: Float>(
    g: &'graph Graph<F>,
    shape_a: A,
    shape_b: B,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    use crate::tensor_ops::array_ops;
    Tensor::builder(g)
        .append_input(shape_a.as_ref(), false)
        .append_input(shape_b.as_ref(), false)
        .build(array_ops::InferBinOpShape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::convert_to_tensor;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_arithmetic_operations() {
        crate::run(|g| {
            let a = convert_to_tensor(array![1.0_f32, 2.0, 3.0], g);
            let b = convert_to_tensor(array![4.0_f32, 5.0, 6.0], g);

            // Test basic arithmetic
            let sum_result = add(a, b);
            let expected_sum = array![5.0_f32, 7.0, 9.0];
            assert_eq!(sum_result.eval(g).unwrap(), expected_sum.into_dyn());

            let sub_result = sub(a, b);
            let expected_sub = array![-3.0_f32, -3.0, -3.0];
            assert_eq!(sub_result.eval(g).unwrap(), expected_sub.into_dyn());

            let mul_result = mul(a, b);
            let expected_mul = array![4.0_f32, 10.0, 18.0];
            assert_eq!(mul_result.eval(g).unwrap(), expected_mul.into_dyn());

            let div_result = div(b, a);
            let expected_div = array![4.0_f32, 2.5, 2.0];
            assert_eq!(div_result.eval(g).unwrap(), expected_div.into_dyn());
        });
    }

    #[test]
    fn test_mathematical_functions() {
        crate::run(|g| {
            let x = convert_to_tensor(array![1.0_f32, 4.0, 9.0], g);

            // Test sqrt
            let sqrt_result = sqrt(x);
            let expected_sqrt = array![1.0_f32, 2.0, 3.0];
            let actual_sqrt = sqrt_result.eval(g).unwrap();
            for (actual, expected) in actual_sqrt.iter().zip(expected_sqrt.iter()) {
                assert_relative_eq!(actual, expected, epsilon = 1e-6);
            }

            // Test square
            let square_result = square(x);
            let expected_square = array![1.0_f32, 16.0, 81.0];
            assert_eq!(square_result.eval(g).unwrap(), expected_square.into_dyn());
        });
    }

    #[test]
    fn test_trigonometric_functions() {
        crate::run(|g| {
            let x = convert_to_tensor(
                array![0.0_f32, std::f32::consts::PI / 2.0, std::f32::consts::PI],
                g,
            );

            // Test sin
            let sin_result = sin(x);
            let actual_sin = sin_result.eval(g).unwrap();
            assert_relative_eq!(actual_sin[0], 0.0, epsilon = 1e-6);
            assert_relative_eq!(actual_sin[1], 1.0, epsilon = 1e-6);
            assert_relative_eq!(actual_sin[2], 0.0, epsilon = 1e-6);

            // Test cos
            let cos_result = cos(x);
            let actual_cos = cos_result.eval(g).unwrap();
            assert_relative_eq!(actual_cos[0], 1.0, epsilon = 1e-6);
            assert_relative_eq!(actual_cos[1], 0.0, epsilon = 1e-6);
            assert_relative_eq!(actual_cos[2], -1.0, epsilon = 1e-6);
        });
    }

    #[test]
    fn test_comparison_operations() {
        crate::run(|g| {
            let a = convert_to_tensor(array![1.0_f32, 2.0, 3.0], g);
            let b = convert_to_tensor(array![3.0_f32, 2.0, 1.0], g);

            // Test equal
            let equal_result = equal(a, b);
            let expected_equal = array![0.0_f32, 1.0, 0.0];
            assert_eq!(equal_result.eval(g).unwrap(), expected_equal.into_dyn());

            // Test greater
            let greater_result = greater(a, b);
            let expected_greater = array![0.0_f32, 0.0, 1.0];
            assert_eq!(greater_result.eval(g).unwrap(), expected_greater.into_dyn());

            // Test lesser
            let lesser_result = lesser(a, b);
            let expected_lesser = array![1.0_f32, 0.0, 0.0];
            assert_eq!(lesser_result.eval(g).unwrap(), expected_lesser.into_dyn());
        });
    }

    #[test]
    fn test_max_min_operations() {
        crate::run(|g| {
            let a = convert_to_tensor(array![1.0_f32, 2.0, 3.0], g);
            let b = convert_to_tensor(array![3.0_f32, 2.0, 1.0], g);

            // Test maximum
            let max_result = maximum(a, b);
            let expected_max = array![3.0_f32, 2.0, 3.0];
            assert_eq!(max_result.eval(g).unwrap(), expected_max.into_dyn());

            // Test minimum
            let min_result = minimum(a, b);
            let expected_min = array![1.0_f32, 2.0, 1.0];
            assert_eq!(min_result.eval(g).unwrap(), expected_min.into_dyn());
        });
    }
}
