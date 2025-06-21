//! Universal Function (ufunc) system for scientific computing
//!
//! This module provides a flexible and extensible universal function (ufunc) system
//! similar to `NumPy`'s ufuncs, allowing for vectorized element-wise operations with
//! automatic broadcasting.

use ndarray::{Array, ArrayView, Ix1, Ix2};
use std::ops;

/// Common mathematical operations for numerical arrays (1D)
pub mod math {
    use super::*;
    use num_traits::{Float, FloatConst};

    /// Apply sine function element-wise to a 1D array
    pub fn sin<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.sin())
    }

    /// Apply cosine function element-wise to a 1D array
    pub fn cos<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.cos())
    }

    /// Apply tangent function element-wise to a 1D array
    pub fn tan<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.tan())
    }

    /// Apply exponential function element-wise to a 1D array
    pub fn exp<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.exp())
    }

    /// Apply natural logarithm function element-wise to a 1D array
    pub fn log<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.ln())
    }

    /// Apply square root function element-wise to a 1D array
    pub fn sqrt<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.sqrt())
    }

    /// Apply absolute value function element-wise to a 1D array
    pub fn abs<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.abs())
    }

    /// Apply hyperbolic sine function element-wise to a 1D array
    pub fn sinh<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.sinh())
    }

    /// Apply hyperbolic cosine function element-wise to a 1D array
    pub fn cosh<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.cosh())
    }

    /// Apply hyperbolic tangent function element-wise to a 1D array
    pub fn tanh<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.tanh())
    }

    /// Apply arcsine function element-wise to a 1D array
    pub fn asin<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.asin())
    }

    /// Apply arccosine function element-wise to a 1D array
    pub fn acos<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.acos())
    }

    /// Apply arctangent function element-wise to a 1D array
    pub fn atan<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.atan())
    }

    /// Apply base-10 logarithm function element-wise to a 1D array
    pub fn log10<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.log10())
    }

    /// Apply base-2 logarithm function element-wise to a 1D array
    pub fn log2<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.log2())
    }

    /// Apply ceil function element-wise to a 1D array
    pub fn ceil<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.ceil())
    }

    /// Apply floor function element-wise to a 1D array
    pub fn floor<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.floor())
    }

    /// Apply round function element-wise to a 1D array
    pub fn round<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.round())
    }

    /// Apply truncate function element-wise to a 1D array
    pub fn trunc<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.trunc())
    }

    /// Apply square function element-wise to a 1D array
    pub fn square<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float + std::ops::Mul<Output = T>,
    {
        array.mapv(|x| x * x)
    }

    /// Apply cube function element-wise to a 1D array
    pub fn cube<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float + std::ops::Mul<Output = T>,
    {
        array.mapv(|x| x * x * x)
    }

    /// Convert degrees to radians element-wise for a 1D array
    pub fn deg2rad<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float + FloatConst,
    {
        array.mapv(|x| x * T::PI() / T::from(180.0).unwrap())
    }

    /// Convert radians to degrees element-wise for a 1D array
    pub fn rad2deg<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float + FloatConst,
    {
        array.mapv(|x| x * T::from(180.0).unwrap() / T::PI())
    }

    /// Apply sign function element-wise to a 1D array
    pub fn sign<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| {
            if x.is_zero() {
                T::zero()
            } else if x < T::zero() {
                -T::one()
            } else {
                T::one()
            }
        })
    }

    /// Apply reciprocal (1/x) function element-wise to a 1D array
    pub fn reciprocal<T>(array: &ArrayView<T, Ix1>) -> Array<T, Ix1>
    where
        T: Clone + Float,
    {
        array.mapv(|x| T::one() / x)
    }
}

/// Common mathematical operations for numerical arrays (2D)
pub mod math2d {
    use super::*;
    use num_traits::{Float, FloatConst};

    /// Apply sine function element-wise to a 2D array
    pub fn sin<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.sin())
    }

    /// Apply cosine function element-wise to a 2D array
    pub fn cos<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.cos())
    }

    /// Apply tangent function element-wise to a 2D array
    pub fn tan<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.tan())
    }

    /// Apply exponential function element-wise to a 2D array
    pub fn exp<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.exp())
    }

    /// Apply natural logarithm function element-wise to a 2D array
    pub fn log<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.ln())
    }

    /// Apply square root function element-wise to a 2D array
    pub fn sqrt<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.sqrt())
    }

    /// Apply absolute value function element-wise to a 2D array
    pub fn abs<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.abs())
    }

    /// Apply hyperbolic sine function element-wise to a 2D array
    pub fn sinh<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.sinh())
    }

    /// Apply hyperbolic cosine function element-wise to a 2D array
    pub fn cosh<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.cosh())
    }

    /// Apply hyperbolic tangent function element-wise to a 2D array
    pub fn tanh<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.tanh())
    }

    /// Apply arcsine function element-wise to a 2D array
    pub fn asin<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.asin())
    }

    /// Apply arccosine function element-wise to a 2D array
    pub fn acos<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.acos())
    }

    /// Apply arctangent function element-wise to a 2D array
    pub fn atan<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.atan())
    }

    /// Apply base-10 logarithm function element-wise to a 2D array
    pub fn log10<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.log10())
    }

    /// Apply base-2 logarithm function element-wise to a 2D array
    pub fn log2<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.log2())
    }

    /// Apply ceil function element-wise to a 2D array
    pub fn ceil<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.ceil())
    }

    /// Apply floor function element-wise to a 2D array
    pub fn floor<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.floor())
    }

    /// Apply round function element-wise to a 2D array
    pub fn round<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.round())
    }

    /// Apply truncate function element-wise to a 2D array
    pub fn trunc<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| x.trunc())
    }

    /// Apply square function element-wise to a 2D array
    pub fn square<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float + std::ops::Mul<Output = T>,
    {
        array.mapv(|x| x * x)
    }

    /// Apply cube function element-wise to a 2D array
    pub fn cube<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float + std::ops::Mul<Output = T>,
    {
        array.mapv(|x| x * x * x)
    }

    /// Convert degrees to radians element-wise for a 2D array
    pub fn deg2rad<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float + FloatConst,
    {
        array.mapv(|x| x * T::PI() / T::from(180.0).unwrap())
    }

    /// Convert radians to degrees element-wise for a 2D array
    pub fn rad2deg<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float + FloatConst,
    {
        array.mapv(|x| x * T::from(180.0).unwrap() / T::PI())
    }

    /// Apply sign function element-wise to a 2D array
    pub fn sign<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| {
            if x.is_zero() {
                T::zero()
            } else if x < T::zero() {
                -T::one()
            } else {
                T::one()
            }
        })
    }

    /// Apply reciprocal (1/x) function element-wise to a 2D array
    pub fn reciprocal<T>(array: &ArrayView<T, Ix2>) -> Array<T, Ix2>
    where
        T: Clone + Float,
    {
        array.mapv(|x| T::one() / x)
    }
}

/// Binary operations for pairs of arrays (1D)
pub mod binary {
    use super::*;
    use num_traits::Float;

    /// Add two 1D arrays element-wise
    pub fn add<T>(
        a: &ArrayView<T, Ix1>,
        b: &ArrayView<T, Ix1>,
    ) -> Result<Array<T, Ix1>, &'static str>
    where
        T: Clone + ops::Add<Output = T>,
    {
        if a.shape() != b.shape() {
            return Err("Arrays must have the same shape for element-wise addition");
        }

        let result = a
            .iter()
            .zip(b.iter())
            .map(|(a_elem, b_elem)| a_elem.clone() + b_elem.clone())
            .collect::<Vec<_>>();

        Ok(Array::from_vec(result))
    }

    /// Subtract two 1D arrays element-wise (a - b)
    pub fn subtract<T>(
        a: &ArrayView<T, Ix1>,
        b: &ArrayView<T, Ix1>,
    ) -> Result<Array<T, Ix1>, &'static str>
    where
        T: Clone + ops::Sub<Output = T>,
    {
        if a.shape() != b.shape() {
            return Err("Arrays must have the same shape for element-wise subtraction");
        }

        let result = a
            .iter()
            .zip(b.iter())
            .map(|(a_elem, b_elem)| a_elem.clone() - b_elem.clone())
            .collect::<Vec<_>>();

        Ok(Array::from_vec(result))
    }

    /// Multiply two 1D arrays element-wise
    pub fn multiply<T>(
        a: &ArrayView<T, Ix1>,
        b: &ArrayView<T, Ix1>,
    ) -> Result<Array<T, Ix1>, &'static str>
    where
        T: Clone + ops::Mul<Output = T>,
    {
        if a.shape() != b.shape() {
            return Err("Arrays must have the same shape for element-wise multiplication");
        }

        let result = a
            .iter()
            .zip(b.iter())
            .map(|(a_elem, b_elem)| a_elem.clone() * b_elem.clone())
            .collect::<Vec<_>>();

        Ok(Array::from_vec(result))
    }

    /// Divide two 1D arrays element-wise (a / b)
    pub fn divide<T>(
        a: &ArrayView<T, Ix1>,
        b: &ArrayView<T, Ix1>,
    ) -> Result<Array<T, Ix1>, &'static str>
    where
        T: Clone + ops::Div<Output = T>,
    {
        if a.shape() != b.shape() {
            return Err("Arrays must have the same shape for element-wise division");
        }

        let result = a
            .iter()
            .zip(b.iter())
            .map(|(a_elem, b_elem)| a_elem.clone() / b_elem.clone())
            .collect::<Vec<_>>();

        Ok(Array::from_vec(result))
    }

    /// Raise elements of one 1D array to the power of elements in another 1D array
    pub fn power<T>(
        a: &ArrayView<T, Ix1>,
        b: &ArrayView<T, Ix1>,
    ) -> Result<Array<T, Ix1>, &'static str>
    where
        T: Clone + Float,
    {
        if a.shape() != b.shape() {
            return Err("Arrays must have the same shape for element-wise power operation");
        }

        let result = a
            .iter()
            .zip(b.iter())
            .map(|(a_elem, b_elem)| a_elem.powf(*b_elem))
            .collect::<Vec<_>>();

        Ok(Array::from_vec(result))
    }
}

/// Binary operations for pairs of arrays (2D and 1D with broadcasting)
pub mod binary2d {
    use super::*;
    use crate::ndarray_ext;
    use num_traits::Float;

    /// Add a 2D array and a 1D array with broadcasting
    pub fn add<T>(
        a: &ArrayView<T, Ix2>,
        b: &ArrayView<T, Ix1>,
    ) -> Result<Array<T, Ix2>, &'static str>
    where
        T: Clone + Default + ops::Add<Output = T>,
    {
        ndarray_ext::broadcast_apply(*a, *b, |x, y| x.clone() + y.clone())
    }

    /// Subtract a 1D array from a 2D array with broadcasting (a - b)
    pub fn subtract<T>(
        a: &ArrayView<T, Ix2>,
        b: &ArrayView<T, Ix1>,
    ) -> Result<Array<T, Ix2>, &'static str>
    where
        T: Clone + Default + ops::Sub<Output = T>,
    {
        ndarray_ext::broadcast_apply(*a, *b, |x, y| x.clone() - y.clone())
    }

    /// Multiply a 2D array and a 1D array element-wise with broadcasting
    pub fn multiply<T>(
        a: &ArrayView<T, Ix2>,
        b: &ArrayView<T, Ix1>,
    ) -> Result<Array<T, Ix2>, &'static str>
    where
        T: Clone + Default + ops::Mul<Output = T>,
    {
        ndarray_ext::broadcast_apply(*a, *b, |x, y| x.clone() * y.clone())
    }

    /// Divide a 2D array by a 1D array element-wise with broadcasting (a / b)
    pub fn divide<T>(
        a: &ArrayView<T, Ix2>,
        b: &ArrayView<T, Ix1>,
    ) -> Result<Array<T, Ix2>, &'static str>
    where
        T: Clone + Default + ops::Div<Output = T>,
    {
        ndarray_ext::broadcast_apply(*a, *b, |x, y| x.clone() / y.clone())
    }

    /// Raise elements of a 2D array to the power of elements in a 1D array with broadcasting
    pub fn power<T>(
        a: &ArrayView<T, Ix2>,
        b: &ArrayView<T, Ix1>,
    ) -> Result<Array<T, Ix2>, &'static str>
    where
        T: Clone + Default + Float,
    {
        ndarray_ext::broadcast_apply(*a, *b, |x, y| x.powf(*y))
    }
}

/// Reduction operations for arrays
pub mod reduction {
    use super::*;
    use num_traits::{Float, FromPrimitive, One, Zero};

    /// Sum of array elements
    pub fn sum<T>(array: &ArrayView<T, Ix2>, axis: Option<usize>) -> Array<T, Ix1>
    where
        T: Clone + Default + ops::Add<Output = T> + Zero,
    {
        match axis {
            Some(ax) => {
                let (rows, cols) = (array.shape()[0], array.shape()[1]);

                match ax {
                    0 => {
                        // Sum along rows (result has length cols)
                        let mut result = Array::<T, Ix1>::default(cols);

                        for j in 0..cols {
                            let mut sum = T::zero();
                            for i in 0..rows {
                                sum = sum + array[[i, j]].clone();
                            }
                            result[j] = sum;
                        }

                        result
                    }
                    1 => {
                        // Sum along columns (result has length rows)
                        let mut result = Array::<T, Ix1>::default(rows);

                        for i in 0..rows {
                            let mut sum = T::zero();
                            for j in 0..cols {
                                sum = sum + array[[i, j]].clone();
                            }
                            result[i] = sum;
                        }

                        result
                    }
                    _ => panic!("Axis must be 0 or 1 for 2D arrays"),
                }
            }
            None => {
                // Sum over all elements
                let mut sum = T::zero();

                for val in array.iter() {
                    sum = sum + val.clone();
                }

                Array::from_vec(vec![sum])
            }
        }
    }

    /// Product of array elements
    pub fn product<T>(array: &ArrayView<T, Ix2>, axis: Option<usize>) -> Array<T, Ix1>
    where
        T: Clone + Default + ops::Mul<Output = T> + One,
    {
        match axis {
            Some(ax) => {
                let (rows, cols) = (array.shape()[0], array.shape()[1]);

                match ax {
                    0 => {
                        // Product along rows (result has length cols)
                        let mut result = Array::<T, Ix1>::default(cols);

                        for j in 0..cols {
                            let mut prod = T::one();
                            for i in 0..rows {
                                prod = prod * array[[i, j]].clone();
                            }
                            result[j] = prod;
                        }

                        result
                    }
                    1 => {
                        // Product along columns (result has length rows)
                        let mut result = Array::<T, Ix1>::default(rows);

                        for i in 0..rows {
                            let mut prod = T::one();
                            for j in 0..cols {
                                prod = prod * array[[i, j]].clone();
                            }
                            result[i] = prod;
                        }

                        result
                    }
                    _ => panic!("Axis must be 0 or 1 for 2D arrays"),
                }
            }
            None => {
                // Product over all elements
                let mut prod = T::one();

                for val in array.iter() {
                    prod = prod * val.clone();
                }

                Array::from_vec(vec![prod])
            }
        }
    }

    /// Mean of array elements
    pub fn mean<T>(array: &ArrayView<T, Ix2>, axis: Option<usize>) -> Array<T, Ix1>
    where
        T: Clone + Default + ops::Add<Output = T> + ops::Div<Output = T> + Zero + FromPrimitive,
    {
        match axis {
            Some(ax) => {
                let (rows, cols) = (array.shape()[0], array.shape()[1]);

                match ax {
                    0 => {
                        // Mean along rows (result has length cols)
                        let mut result = Array::<T, Ix1>::default(cols);
                        let n = T::from_usize(rows).unwrap();

                        for j in 0..cols {
                            let mut sum = T::zero();
                            for i in 0..rows {
                                sum = sum + array[[i, j]].clone();
                            }
                            result[j] = sum / n.clone();
                        }

                        result
                    }
                    1 => {
                        // Mean along columns (result has length rows)
                        let mut result = Array::<T, Ix1>::default(rows);
                        let n = T::from_usize(cols).unwrap();

                        for i in 0..rows {
                            let mut sum = T::zero();
                            for j in 0..cols {
                                sum = sum + array[[i, j]].clone();
                            }
                            result[i] = sum / n.clone();
                        }

                        result
                    }
                    _ => panic!("Axis must be 0 or 1 for 2D arrays"),
                }
            }
            None => {
                // Mean over all elements
                let (rows, cols) = (array.shape()[0], array.shape()[1]);
                let n = T::from_usize(rows * cols).unwrap();
                let mut sum = T::zero();

                for val in array.iter() {
                    sum = sum + val.clone();
                }

                Array::from_vec(vec![sum / n])
            }
        }
    }

    /// Variance of array elements
    pub fn var<T>(array: &ArrayView<T, Ix2>, axis: Option<usize>) -> Array<T, Ix1>
    where
        T: Clone
            + Default
            + ops::Add<Output = T>
            + ops::Sub<Output = T>
            + ops::Mul<Output = T>
            + ops::Div<Output = T>
            + Zero
            + One
            + FromPrimitive,
    {
        match axis {
            Some(ax) => {
                let (rows, cols) = (array.shape()[0], array.shape()[1]);

                match ax {
                    0 => {
                        // Variance along rows (result has length cols)
                        let means = mean(array, Some(0));
                        let mut result = Array::<T, Ix1>::default(cols);
                        let n = T::from_usize(rows).unwrap();

                        for j in 0..cols {
                            let mut sum_sq_diff = T::zero();
                            for i in 0..rows {
                                let diff = array[[i, j]].clone() - means[j].clone();
                                sum_sq_diff = sum_sq_diff + (diff.clone() * diff);
                            }
                            result[j] = sum_sq_diff / n.clone();
                        }

                        result
                    }
                    1 => {
                        // Variance along columns (result has length rows)
                        let means = mean(array, Some(1));
                        let mut result = Array::<T, Ix1>::default(rows);
                        let n = T::from_usize(cols).unwrap();

                        for i in 0..rows {
                            let mut sum_sq_diff = T::zero();
                            for j in 0..cols {
                                let diff = array[[i, j]].clone() - means[i].clone();
                                sum_sq_diff = sum_sq_diff + (diff.clone() * diff);
                            }
                            result[i] = sum_sq_diff / n.clone();
                        }

                        result
                    }
                    _ => panic!("Axis must be 0 or 1 for 2D arrays"),
                }
            }
            None => {
                // Variance over all elements
                let (rows, cols) = (array.shape()[0], array.shape()[1]);
                let n = T::from_usize(rows * cols).unwrap();

                // Compute the mean
                let mean_val = mean(array, None)[0].clone();

                // Compute the sum of squared differences from the mean
                let mut sum_sq_diff = T::zero();
                for val in array.iter() {
                    let diff = val.clone() - mean_val.clone();
                    sum_sq_diff = sum_sq_diff + (diff.clone() * diff);
                }

                Array::from_vec(vec![sum_sq_diff / n])
            }
        }
    }

    /// Standard deviation of array elements
    pub fn std<T>(array: &ArrayView<T, Ix2>, axis: Option<usize>) -> Array<T, Ix1>
    where
        T: Clone
            + Default
            + ops::Add<Output = T>
            + ops::Sub<Output = T>
            + ops::Mul<Output = T>
            + ops::Div<Output = T>
            + Zero
            + One
            + FromPrimitive
            + Float,
    {
        let variances = var(array, axis);

        variances.mapv(|x| x.sqrt())
    }

    /// Minimum value in the array
    pub fn min<T>(array: &ArrayView<T, Ix2>, axis: Option<usize>) -> Array<T, Ix1>
    where
        T: Clone + Default + num_traits::Bounded + std::cmp::PartialOrd,
    {
        match axis {
            Some(ax) => {
                let (rows, cols) = (array.shape()[0], array.shape()[1]);

                match ax {
                    0 => {
                        // Min along rows (result has length cols)
                        let mut result = Array::<T, Ix1>::default(cols);

                        for j in 0..cols {
                            let mut min_val = T::max_value();
                            for i in 0..rows {
                                if array[[i, j]] < min_val {
                                    min_val = array[[i, j]].clone();
                                }
                            }
                            result[j] = min_val;
                        }

                        result
                    }
                    1 => {
                        // Min along columns (result has length rows)
                        let mut result = Array::<T, Ix1>::default(rows);

                        for i in 0..rows {
                            let mut min_val = T::max_value();
                            for j in 0..cols {
                                if array[[i, j]] < min_val {
                                    min_val = array[[i, j]].clone();
                                }
                            }
                            result[i] = min_val;
                        }

                        result
                    }
                    _ => panic!("Axis must be 0 or 1 for 2D arrays"),
                }
            }
            None => {
                // Min over all elements
                let mut min_val = T::max_value();

                for val in array.iter() {
                    if *val < min_val {
                        min_val = val.clone();
                    }
                }

                Array::from_vec(vec![min_val])
            }
        }
    }

    /// Maximum value in the array
    pub fn max<T>(array: &ArrayView<T, Ix2>, axis: Option<usize>) -> Array<T, Ix1>
    where
        T: Clone + Default + num_traits::Bounded + std::cmp::PartialOrd,
    {
        match axis {
            Some(ax) => {
                let (rows, cols) = (array.shape()[0], array.shape()[1]);

                match ax {
                    0 => {
                        // Max along rows (result has length cols)
                        let mut result = Array::<T, Ix1>::default(cols);

                        for j in 0..cols {
                            let mut max_val = T::min_value();
                            for i in 0..rows {
                                if array[[i, j]] > max_val {
                                    max_val = array[[i, j]].clone();
                                }
                            }
                            result[j] = max_val;
                        }

                        result
                    }
                    1 => {
                        // Max along columns (result has length rows)
                        let mut result = Array::<T, Ix1>::default(rows);

                        for i in 0..rows {
                            let mut max_val = T::min_value();
                            for j in 0..cols {
                                if array[[i, j]] > max_val {
                                    max_val = array[[i, j]].clone();
                                }
                            }
                            result[i] = max_val;
                        }

                        result
                    }
                    _ => panic!("Axis must be 0 or 1 for 2D arrays"),
                }
            }
            None => {
                // Max over all elements
                let mut max_val = T::min_value();

                for val in array.iter() {
                    if *val > max_val {
                        max_val = val.clone();
                    }
                }

                Array::from_vec(vec![max_val])
            }
        }
    }
}

// Export specific math operations directly
pub use math2d::{
    abs, acos, asin, atan, ceil, cos, cosh, cube, deg2rad, exp, floor, log, log10, log2, rad2deg,
    reciprocal, round, sign, sin, sinh, sqrt, square, tan, tanh, trunc,
};

// Export reduction operations directly
pub use reduction::{max, mean, min, product, std, sum, var};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f64::consts::PI;

    #[test]
    fn test_math_unary() {
        let a = array![1.0, 4.0, 9.0];
        let result = math::sqrt(&a.view());
        assert_eq!(result, array![1.0, 2.0, 9.0_f64.sqrt()]);

        let a = array![0.0, PI / 2.0, PI];
        let result = math::sin(&a.view());
        assert!((result[0]).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2]).abs() < 1e-10);
    }

    #[test]
    fn test_binary_operations() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let result = binary::add(&a.view(), &b.view()).unwrap();
        assert_eq!(result, array![5.0, 7.0, 9.0]);

        let result = binary::multiply(&a.view(), &b.view()).unwrap();
        assert_eq!(result, array![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_binary_broadcasting() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![10.0, 20.0, 30.0];

        let result = binary2d::add(&a.view(), &b.view()).unwrap();
        assert_eq!(result, array![[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]);

        let result = binary2d::multiply(&a.view(), &b.view()).unwrap();
        assert_eq!(result, array![[10.0, 40.0, 90.0], [40.0, 100.0, 180.0]]);
    }

    #[test]
    fn test_reduction_operations() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Test sum reduction
        let result = reduction::sum(&a.view(), None);
        assert_eq!(result, array![21.0]);

        // Test sum along axis 0
        let result = reduction::sum(&a.view(), Some(0));
        assert_eq!(result, array![5.0, 7.0, 9.0]);

        // Test sum along axis 1
        let result = reduction::sum(&a.view(), Some(1));
        assert_eq!(result, array![6.0, 15.0]);

        // Test mean reduction
        let result = reduction::mean(&a.view(), None);
        assert_eq!(result, array![3.5]);
    }
}
