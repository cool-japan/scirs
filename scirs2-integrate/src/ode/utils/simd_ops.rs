//! SIMD-optimized operations for ODE solvers
//!
//! This module provides SIMD-accelerated implementations of common operations
//! used in ODE solving, such as vector arithmetic, norm calculations, and
//! element-wise function evaluation. These optimizations can provide significant
//! performance improvements for large systems of ODEs.

#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::needless_range_loop)]

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use ndarray::{Array1, ArrayView1, ArrayViewMut1, Zip};

/// SIMD-optimized ODE operations
pub struct SimdOdeOps;

impl SimdOdeOps {
    /// Compute y = y + a * dy using SIMD operations
    pub fn simd_axpy<F: IntegrateFloat>(y: &mut ArrayViewMut1<F>, a: F, dy: &ArrayView1<F>) {
        #[cfg(feature = "simd")]
        {
            match std::any::TypeId::of::<F>() {
                id if id == std::any::TypeId::of::<f32>() => {
                    let y_f32 = unsafe {
                        std::mem::transmute::<
                            &mut ndarray::ArrayBase<
                                ndarray::ViewRepr<&mut F>,
                                ndarray::Dim<[usize; 1]>,
                            >,
                            &mut ndarray::ArrayBase<
                                ndarray::ViewRepr<&mut f32>,
                                ndarray::Dim<[usize; 1]>,
                            >,
                        >(y)
                    };
                    let dy_f32 = unsafe {
                        std::mem::transmute::<
                            &ndarray::ArrayBase<ndarray::ViewRepr<&F>, ndarray::Dim<[usize; 1]>>,
                            &ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 1]>>,
                        >(dy)
                    };
                    let a_f32 = unsafe { std::mem::transmute_copy(&a) };
                    simd_axpy_f32_impl(y_f32, a_f32, dy_f32);
                    return;
                }
                id if id == std::any::TypeId::of::<f64>() => {
                    let y_f64 = unsafe {
                        std::mem::transmute::<
                            &mut ndarray::ArrayBase<
                                ndarray::ViewRepr<&mut F>,
                                ndarray::Dim<[usize; 1]>,
                            >,
                            &mut ndarray::ArrayBase<
                                ndarray::ViewRepr<&mut f64>,
                                ndarray::Dim<[usize; 1]>,
                            >,
                        >(y)
                    };
                    let dy_f64 = unsafe {
                        std::mem::transmute::<
                            &ndarray::ArrayBase<ndarray::ViewRepr<&F>, ndarray::Dim<[usize; 1]>>,
                            &ndarray::ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 1]>>,
                        >(dy)
                    };
                    let a_f64 = unsafe { std::mem::transmute_copy(&a) };
                    simd_axpy_f64_impl(y_f64, a_f64, dy_f64);
                    return;
                }
                _ => {}
            }
        }

        // Fallback implementation
        Zip::from(y).and(dy).for_each(|y_val, &dy_val| {
            *y_val += a * dy_val;
        });
    }

    /// Compute linear combination: result = a*x + b*y using SIMD
    pub fn simd_linear_combination<F: IntegrateFloat>(
        x: &ArrayView1<F>,
        a: F,
        y: &ArrayView1<F>,
        b: F,
    ) -> Array1<F> {
        #[cfg(feature = "simd")]
        {
            match std::any::TypeId::of::<F>() {
                id if id == std::any::TypeId::of::<f32>() => {
                    let x_f32 = unsafe {
                        std::mem::transmute::<
                            &ndarray::ArrayBase<ndarray::ViewRepr<&F>, ndarray::Dim<[usize; 1]>>,
                            &ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 1]>>,
                        >(x)
                    };
                    let y_f32 = unsafe {
                        std::mem::transmute::<
                            &ndarray::ArrayBase<ndarray::ViewRepr<&F>, ndarray::Dim<[usize; 1]>>,
                            &ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 1]>>,
                        >(y)
                    };
                    let a_f32 = unsafe { std::mem::transmute_copy(&a) };
                    let b_f32 = unsafe { std::mem::transmute_copy(&b) };
                    let result = simd_linear_combination_f32_impl(x_f32, a_f32, y_f32, b_f32);
                    return unsafe { std::mem::transmute(result) };
                }
                id if id == std::any::TypeId::of::<f64>() => {
                    let x_f64 = unsafe {
                        std::mem::transmute::<
                            &ndarray::ArrayBase<ndarray::ViewRepr<&F>, ndarray::Dim<[usize; 1]>>,
                            &ndarray::ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 1]>>,
                        >(x)
                    };
                    let y_f64 = unsafe {
                        std::mem::transmute::<
                            &ndarray::ArrayBase<ndarray::ViewRepr<&F>, ndarray::Dim<[usize; 1]>>,
                            &ndarray::ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 1]>>,
                        >(y)
                    };
                    let a_f64 = unsafe { std::mem::transmute_copy(&a) };
                    let b_f64 = unsafe { std::mem::transmute_copy(&b) };
                    let result = simd_linear_combination_f64_impl(x_f64, a_f64, y_f64, b_f64);
                    return unsafe { std::mem::transmute(result) };
                }
                _ => {}
            }
        }

        // Fallback implementation
        let mut result = Array1::zeros(x.len());
        Zip::from(&mut result)
            .and(x)
            .and(y)
            .for_each(|r, &x_val, &y_val| {
                *r = a * x_val + b * y_val;
            });
        result
    }

    /// Compute element-wise maximum using SIMD
    pub fn simd_element_max<F: IntegrateFloat>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F> {
        let mut result = Array1::zeros(a.len());
        #[cfg(feature = "simd")]
        {
            match std::any::TypeId::of::<F>() {
                id if id == std::any::TypeId::of::<f32>() => {
                    let a_f32 = unsafe { std::mem::transmute(a) };
                    let b_f32 = unsafe { std::mem::transmute(b) };
                    let mut result_f32 = unsafe { std::mem::transmute(result.view_mut()) };
                    simd_element_max_f32_impl(a_f32, b_f32, &mut result_f32);
                    return result;
                }
                id if id == std::any::TypeId::of::<f64>() => {
                    let a_f64 = unsafe { std::mem::transmute(a) };
                    let b_f64 = unsafe { std::mem::transmute(b) };
                    let mut result_f64 = unsafe { std::mem::transmute(result.view_mut()) };
                    simd_element_max_f64_impl(a_f64, b_f64, &mut result_f64);
                    return result;
                }
                _ => {}
            }
        }

        // Fallback implementation
        Zip::from(&mut result)
            .and(a)
            .and(b)
            .for_each(|r, &a_val, &b_val| {
                *r = a_val.max(b_val);
            });
        result
    }

    /// Compute element-wise minimum using SIMD
    pub fn simd_element_min<F: IntegrateFloat>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F> {
        let mut result = Array1::zeros(a.len());
        #[cfg(feature = "simd")]
        {
            match std::any::TypeId::of::<F>() {
                id if id == std::any::TypeId::of::<f32>() => {
                    let a_f32 = unsafe { std::mem::transmute(a) };
                    let b_f32 = unsafe { std::mem::transmute(b) };
                    let mut result_f32 = unsafe { std::mem::transmute(result.view_mut()) };
                    simd_element_min_f32_impl(a_f32, b_f32, &mut result_f32);
                    return result;
                }
                id if id == std::any::TypeId::of::<f64>() => {
                    let a_f64 = unsafe { std::mem::transmute(a) };
                    let b_f64 = unsafe { std::mem::transmute(b) };
                    let mut result_f64 = unsafe { std::mem::transmute(result.view_mut()) };
                    simd_element_min_f64_impl(a_f64, b_f64, &mut result_f64);
                    return result;
                }
                _ => {}
            }
        }

        // Fallback implementation
        Zip::from(&mut result)
            .and(a)
            .and(b)
            .for_each(|r, &a_val, &b_val| {
                *r = a_val.min(b_val);
            });
        result
    }

    /// Compute L2 norm using SIMD
    pub fn simd_norm_l2<F: IntegrateFloat>(x: &ArrayView1<F>) -> F {
        #[cfg(feature = "simd")]
        {
            match std::any::TypeId::of::<F>() {
                id if id == std::any::TypeId::of::<f32>() => {
                    let x_f32 = unsafe { std::mem::transmute(x) };
                    let result = simd_norm_l2_f32_impl(x_f32);
                    return unsafe { std::mem::transmute_copy(&result) };
                }
                id if id == std::any::TypeId::of::<f64>() => {
                    let x_f64 = unsafe { std::mem::transmute(x) };
                    let result = simd_norm_l2_f64_impl(x_f64);
                    return unsafe { std::mem::transmute_copy(&result) };
                }
                _ => {}
            }
        }

        // Fallback implementation
        x.iter()
            .map(|&val| val * val)
            .fold(F::zero(), |acc, x| acc + x)
            .sqrt()
    }

    /// Compute infinity norm using SIMD
    pub fn simd_norm_inf<F: IntegrateFloat>(x: &ArrayView1<F>) -> F {
        #[cfg(feature = "simd")]
        {
            match std::any::TypeId::of::<F>() {
                id if id == std::any::TypeId::of::<f32>() => {
                    let x_f32 = unsafe { std::mem::transmute(x) };
                    let result = simd_norm_inf_f32_impl(x_f32);
                    return unsafe { std::mem::transmute_copy(&result) };
                }
                id if id == std::any::TypeId::of::<f64>() => {
                    let x_f64 = unsafe { std::mem::transmute(x) };
                    let result = simd_norm_inf_f64_impl(x_f64);
                    return unsafe { std::mem::transmute_copy(&result) };
                }
                _ => {}
            }
        }

        // Fallback implementation
        x.iter()
            .map(|&val| val.abs())
            .fold(F::zero(), |acc, x| acc.max(x))
    }

    /// Apply scalar function element-wise using SIMD when possible
    pub fn simd_map_scalar<F, Func>(x: &ArrayView1<F>, f: Func) -> Array1<F>
    where
        F: IntegrateFloat,
        Func: Fn(F) -> F + Sync,
    {
        // For now, use standard implementation as SIMD scalar functions are complex
        x.map(|&val| f(val)).to_owned()
    }
}

/// SIMD-optimized vector addition with scaling: y = y + a * x
#[cfg(feature = "simd")]
fn simd_axpy_f32_impl(y: &mut ArrayViewMut1<f32>, a: f32, x: &ArrayView1<f32>) {
    use wide::f32x8;

    let n = y.len();
    assert_eq!(n, x.len(), "Arrays must have the same length");

    if let (Some(y_slice), Some(x_slice)) = (y.as_slice_mut(), x.as_slice()) {
        let chunk_size = 8;
        let mut i = 0;

        // Process 8 elements at a time
        while i + chunk_size <= n {
            let a_vec = f32x8::splat(a);
            let x_arr = [
                x_slice[i],
                x_slice[i + 1],
                x_slice[i + 2],
                x_slice[i + 3],
                x_slice[i + 4],
                x_slice[i + 5],
                x_slice[i + 6],
                x_slice[i + 7],
            ];
            let y_arr = [
                y_slice[i],
                y_slice[i + 1],
                y_slice[i + 2],
                y_slice[i + 3],
                y_slice[i + 4],
                y_slice[i + 5],
                y_slice[i + 6],
                y_slice[i + 7],
            ];

            let x_vec = f32x8::new(x_arr);
            let y_vec = f32x8::new(y_arr);

            let result_vec = y_vec + a_vec * x_vec;
            let result_arr: [f32; 8] = result_vec.into();

            y_slice[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for j in i..n {
            y_slice[j] += a * x_slice[j];
        }
    } else {
        // Fallback for non-contiguous arrays
        Zip::from(y).and(x).for_each(|y_val, &x_val| {
            *y_val += a * x_val;
        });
    }
}

/// SIMD-optimized vector addition with scaling: y = y + a * x (f64 version)
#[cfg(feature = "simd")]
fn simd_axpy_f64_impl(y: &mut ArrayViewMut1<f64>, a: f64, x: &ArrayView1<f64>) {
    use wide::f64x4;

    let n = y.len();
    assert_eq!(n, x.len(), "Arrays must have the same length");

    if let (Some(y_slice), Some(x_slice)) = (y.as_slice_mut(), x.as_slice()) {
        let chunk_size = 4;
        let mut i = 0;

        // Process 4 elements at a time
        while i + chunk_size <= n {
            let a_vec = f64x4::splat(a);
            let x_arr = [x_slice[i], x_slice[i + 1], x_slice[i + 2], x_slice[i + 3]];
            let y_arr = [y_slice[i], y_slice[i + 1], y_slice[i + 2], y_slice[i + 3]];

            let x_vec = f64x4::new(x_arr);
            let y_vec = f64x4::new(y_arr);

            let result_vec = y_vec + a_vec * x_vec;
            let result_arr: [f64; 4] = result_vec.into();

            y_slice[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for j in i..n {
            y_slice[j] += a * x_slice[j];
        }
    } else {
        // Fallback for non-contiguous arrays
        Zip::from(y).and(x).for_each(|y_val, &x_val| {
            *y_val += a * x_val;
        });
    }
}

/// SIMD-optimized linear combination: result = a*x + b*y
#[cfg(feature = "simd")]
fn simd_linear_combination_f32_impl(
    x: &ArrayView1<f32>,
    a: f32,
    y: &ArrayView1<f32>,
    b: f32,
) -> Array1<f32> {
    use wide::f32x8;

    let n = x.len();
    assert_eq!(n, y.len(), "Arrays must have the same length");
    let mut result = Array1::zeros(n);

    if let (Some(x_slice), Some(y_slice), Some(result_slice)) =
        (x.as_slice(), y.as_slice(), result.as_slice_mut())
    {
        let chunk_size = 8;
        let mut i = 0;

        while i + chunk_size <= n {
            let a_vec = f32x8::splat(a);
            let b_vec = f32x8::splat(b);

            let x_arr = [
                x_slice[i],
                x_slice[i + 1],
                x_slice[i + 2],
                x_slice[i + 3],
                x_slice[i + 4],
                x_slice[i + 5],
                x_slice[i + 6],
                x_slice[i + 7],
            ];
            let y_arr = [
                y_slice[i],
                y_slice[i + 1],
                y_slice[i + 2],
                y_slice[i + 3],
                y_slice[i + 4],
                y_slice[i + 5],
                y_slice[i + 6],
                y_slice[i + 7],
            ];

            let x_vec = f32x8::new(x_arr);
            let y_vec = f32x8::new(y_arr);

            let result_vec = a_vec * x_vec + b_vec * y_vec;
            let result_arr: [f32; 8] = result_vec.into();

            result_slice[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for j in i..n {
            result_slice[j] = a * x_slice[j] + b * y_slice[j];
        }
    } else {
        // Fallback for non-contiguous arrays
        Zip::from(&mut result)
            .and(x)
            .and(y)
            .for_each(|res, &x_val, &y_val| {
                *res = a * x_val + b * y_val;
            });
    }

    result
}

/// SIMD-optimized linear combination: result = a*x + b*y (f64 version)
#[cfg(feature = "simd")]
fn simd_linear_combination_f64_impl(
    x: &ArrayView1<f64>,
    a: f64,
    y: &ArrayView1<f64>,
    b: f64,
) -> Array1<f64> {
    use wide::f64x4;

    let n = x.len();
    assert_eq!(n, y.len(), "Arrays must have the same length");
    let mut result = Array1::zeros(n);

    if let (Some(x_slice), Some(y_slice), Some(result_slice)) =
        (x.as_slice(), y.as_slice(), result.as_slice_mut())
    {
        let chunk_size = 4;
        let mut i = 0;

        while i + chunk_size <= n {
            let a_vec = f64x4::splat(a);
            let b_vec = f64x4::splat(b);

            let x_arr = [x_slice[i], x_slice[i + 1], x_slice[i + 2], x_slice[i + 3]];
            let y_arr = [y_slice[i], y_slice[i + 1], y_slice[i + 2], y_slice[i + 3]];

            let x_vec = f64x4::new(x_arr);
            let y_vec = f64x4::new(y_arr);

            let result_vec = a_vec * x_vec + b_vec * y_vec;
            let result_arr: [f64; 4] = result_vec.into();

            result_slice[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for j in i..n {
            result_slice[j] = a * x_slice[j] + b * y_slice[j];
        }
    } else {
        // Fallback for non-contiguous arrays
        Zip::from(&mut result)
            .and(x)
            .and(y)
            .for_each(|res, &x_val, &y_val| {
                *res = a * x_val + b * y_val;
            });
    }

    result
}

/// SIMD-optimized L2 norm computation
#[cfg(feature = "simd")]
fn simd_norm_l2_f32_impl(x: &ArrayView1<f32>) -> f32 {
    use wide::f32x8;

    let n = x.len();
    if let Some(x_slice) = x.as_slice() {
        let chunk_size = 8;
        let mut i = 0;
        let mut sum_vec = f32x8::splat(0.0);

        while i + chunk_size <= n {
            let x_arr = [
                x_slice[i],
                x_slice[i + 1],
                x_slice[i + 2],
                x_slice[i + 3],
                x_slice[i + 4],
                x_slice[i + 5],
                x_slice[i + 6],
                x_slice[i + 7],
            ];
            let x_vec = f32x8::new(x_arr);
            sum_vec += x_vec * x_vec;
            i += chunk_size;
        }

        // Sum all elements of the SIMD vector
        let sum_arr: [f32; 8] = sum_vec.into();
        let mut total_sum = sum_arr.iter().sum::<f32>();

        // Process remaining elements
        for j in i..n {
            total_sum += x_slice[j] * x_slice[j];
        }

        total_sum.sqrt()
    } else {
        // Fallback for non-contiguous arrays
        x.iter().map(|&val| val * val).sum::<f32>().sqrt()
    }
}

/// SIMD-optimized L2 norm computation (f64 version)
#[cfg(feature = "simd")]
fn simd_norm_l2_f64_impl(x: &ArrayView1<f64>) -> f64 {
    use wide::f64x4;

    let n = x.len();
    if let Some(x_slice) = x.as_slice() {
        let chunk_size = 4;
        let mut i = 0;
        let mut sum_vec = f64x4::splat(0.0);

        while i + chunk_size <= n {
            let x_arr = [x_slice[i], x_slice[i + 1], x_slice[i + 2], x_slice[i + 3]];
            let x_vec = f64x4::new(x_arr);
            sum_vec += x_vec * x_vec;
            i += chunk_size;
        }

        // Sum all elements of the SIMD vector
        let sum_arr: [f64; 4] = sum_vec.into();
        let mut total_sum = sum_arr.iter().sum::<f64>();

        // Process remaining elements
        for j in i..n {
            total_sum += x_slice[j] * x_slice[j];
        }

        total_sum.sqrt()
    } else {
        // Fallback for non-contiguous arrays
        x.iter().map(|&val| val * val).sum::<f64>().sqrt()
    }
}

/// SIMD-optimized infinity norm computation
#[cfg(feature = "simd")]
fn simd_norm_inf_f32_impl(x: &ArrayView1<f32>) -> f32 {
    use wide::f32x8;

    let n = x.len();
    if let Some(x_slice) = x.as_slice() {
        let chunk_size = 8;
        let mut i = 0;
        let mut max_vec = f32x8::splat(0.0);

        while i + chunk_size <= n {
            let x_arr = [
                x_slice[i],
                x_slice[i + 1],
                x_slice[i + 2],
                x_slice[i + 3],
                x_slice[i + 4],
                x_slice[i + 5],
                x_slice[i + 6],
                x_slice[i + 7],
            ];
            let x_vec = f32x8::new(x_arr);
            let abs_vec = x_vec.abs();
            max_vec = max_vec.max(abs_vec);
            i += chunk_size;
        }

        // Find maximum element of the SIMD vector
        let max_arr: [f32; 8] = max_vec.into();
        let mut max_val = max_arr.iter().fold(0.0f32, |a, &b| a.max(b));

        // Process remaining elements
        for j in i..n {
            max_val = max_val.max(x_slice[j].abs());
        }

        max_val
    } else {
        // Fallback for non-contiguous arrays
        x.iter().map(|&val| val.abs()).fold(0.0f32, |a, b| a.max(b))
    }
}

/// SIMD-optimized infinity norm computation (f64 version)
#[cfg(feature = "simd")]
fn simd_norm_inf_f64_impl(x: &ArrayView1<f64>) -> f64 {
    use wide::f64x4;

    let n = x.len();
    if let Some(x_slice) = x.as_slice() {
        let chunk_size = 4;
        let mut i = 0;
        let mut max_vec = f64x4::splat(0.0);

        while i + chunk_size <= n {
            let x_arr = [x_slice[i], x_slice[i + 1], x_slice[i + 2], x_slice[i + 3]];
            let x_vec = f64x4::new(x_arr);
            let abs_vec = x_vec.abs();
            max_vec = max_vec.max(abs_vec);
            i += chunk_size;
        }

        // Find maximum element of the SIMD vector
        let max_arr: [f64; 4] = max_vec.into();
        let mut max_val = max_arr.iter().fold(0.0f64, |a, &b| a.max(b));

        // Process remaining elements
        for j in i..n {
            max_val = max_val.max(x_slice[j].abs());
        }

        max_val
    } else {
        // Fallback for non-contiguous arrays
        x.iter().map(|&val| val.abs()).fold(0.0f64, |a, b| a.max(b))
    }
}

/// SIMD-optimized element-wise maximum (f32 version)
#[cfg(feature = "simd")]
fn simd_element_max_f32_impl(
    a: &ArrayView1<f32>,
    b: &ArrayView1<f32>,
    result: &mut ArrayViewMut1<f32>,
) {
    use wide::f32x8;

    let n = a.len();
    assert_eq!(n, b.len(), "Arrays must have the same length");
    assert_eq!(n, result.len(), "Result array must have the same length");

    if let (Some(a_slice), Some(b_slice), Some(result_slice)) =
        (a.as_slice(), b.as_slice(), result.as_slice_mut())
    {
        let chunk_size = 8;
        let mut i = 0;

        while i + chunk_size <= n {
            let a_arr = [
                a_slice[i],
                a_slice[i + 1],
                a_slice[i + 2],
                a_slice[i + 3],
                a_slice[i + 4],
                a_slice[i + 5],
                a_slice[i + 6],
                a_slice[i + 7],
            ];
            let b_arr = [
                b_slice[i],
                b_slice[i + 1],
                b_slice[i + 2],
                b_slice[i + 3],
                b_slice[i + 4],
                b_slice[i + 5],
                b_slice[i + 6],
                b_slice[i + 7],
            ];

            let a_vec = f32x8::new(a_arr);
            let b_vec = f32x8::new(b_arr);

            let result_vec = a_vec.max(b_vec);
            let result_arr: [f32; 8] = result_vec.into();

            result_slice[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for j in i..n {
            result_slice[j] = a_slice[j].max(b_slice[j]);
        }
    } else {
        // Fallback for non-contiguous arrays
        Zip::from(result)
            .and(a)
            .and(b)
            .for_each(|r, &a_val, &b_val| {
                *r = a_val.max(b_val);
            });
    }
}

/// SIMD-optimized element-wise maximum (f64 version)
#[cfg(feature = "simd")]
fn simd_element_max_f64_impl(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    result: &mut ArrayViewMut1<f64>,
) {
    use wide::f64x4;

    let n = a.len();
    assert_eq!(n, b.len(), "Arrays must have the same length");
    assert_eq!(n, result.len(), "Result array must have the same length");

    if let (Some(a_slice), Some(b_slice), Some(result_slice)) =
        (a.as_slice(), b.as_slice(), result.as_slice_mut())
    {
        let chunk_size = 4;
        let mut i = 0;

        while i + chunk_size <= n {
            let a_arr = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
            let b_arr = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];

            let a_vec = f64x4::new(a_arr);
            let b_vec = f64x4::new(b_arr);

            let result_vec = a_vec.max(b_vec);
            let result_arr: [f64; 4] = result_vec.into();

            result_slice[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for j in i..n {
            result_slice[j] = a_slice[j].max(b_slice[j]);
        }
    } else {
        // Fallback for non-contiguous arrays
        Zip::from(result)
            .and(a)
            .and(b)
            .for_each(|r, &a_val, &b_val| {
                *r = a_val.max(b_val);
            });
    }
}

/// SIMD-optimized element-wise minimum (f32 version)
#[cfg(feature = "simd")]
fn simd_element_min_f32_impl(
    a: &ArrayView1<f32>,
    b: &ArrayView1<f32>,
    result: &mut ArrayViewMut1<f32>,
) {
    use wide::f32x8;

    let n = a.len();
    assert_eq!(n, b.len(), "Arrays must have the same length");
    assert_eq!(n, result.len(), "Result array must have the same length");

    if let (Some(a_slice), Some(b_slice), Some(result_slice)) =
        (a.as_slice(), b.as_slice(), result.as_slice_mut())
    {
        let chunk_size = 8;
        let mut i = 0;

        while i + chunk_size <= n {
            let a_arr = [
                a_slice[i],
                a_slice[i + 1],
                a_slice[i + 2],
                a_slice[i + 3],
                a_slice[i + 4],
                a_slice[i + 5],
                a_slice[i + 6],
                a_slice[i + 7],
            ];
            let b_arr = [
                b_slice[i],
                b_slice[i + 1],
                b_slice[i + 2],
                b_slice[i + 3],
                b_slice[i + 4],
                b_slice[i + 5],
                b_slice[i + 6],
                b_slice[i + 7],
            ];

            let a_vec = f32x8::new(a_arr);
            let b_vec = f32x8::new(b_arr);

            let result_vec = a_vec.min(b_vec);
            let result_arr: [f32; 8] = result_vec.into();

            result_slice[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for j in i..n {
            result_slice[j] = a_slice[j].min(b_slice[j]);
        }
    } else {
        // Fallback for non-contiguous arrays
        Zip::from(result)
            .and(a)
            .and(b)
            .for_each(|r, &a_val, &b_val| {
                *r = a_val.min(b_val);
            });
    }
}

/// SIMD-optimized element-wise minimum (f64 version)
#[cfg(feature = "simd")]
fn simd_element_min_f64_impl(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    result: &mut ArrayViewMut1<f64>,
) {
    use wide::f64x4;

    let n = a.len();
    assert_eq!(n, b.len(), "Arrays must have the same length");
    assert_eq!(n, result.len(), "Result array must have the same length");

    if let (Some(a_slice), Some(b_slice), Some(result_slice)) =
        (a.as_slice(), b.as_slice(), result.as_slice_mut())
    {
        let chunk_size = 4;
        let mut i = 0;

        while i + chunk_size <= n {
            let a_arr = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
            let b_arr = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];

            let a_vec = f64x4::new(a_arr);
            let b_vec = f64x4::new(b_arr);

            let result_vec = a_vec.min(b_vec);
            let result_arr: [f64; 4] = result_vec.into();

            result_slice[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for j in i..n {
            result_slice[j] = a_slice[j].min(b_slice[j]);
        }
    } else {
        // Fallback for non-contiguous arrays
        Zip::from(result)
            .and(a)
            .and(b)
            .for_each(|r, &a_val, &b_val| {
                *r = a_val.min(b_val);
            });
    }
}

/// SIMD-optimized Runge-Kutta step computation
///
/// Performs the computation: y_new = y + h * (c1*k1 + c2*k2 + c3*k3 + c4*k4)
/// where k1, k2, k3, k4 are the RK stage derivatives and c1, c2, c3, c4 are coefficients
pub fn simd_rk_step<F: IntegrateFloat>(
    y: &ArrayView1<F>,
    h: F,
    k1: &ArrayView1<F>,
    k2: &ArrayView1<F>,
    k3: &ArrayView1<F>,
    k4: &ArrayView1<F>,
    c1: F,
    c2: F,
    c3: F,
    c4: F,
) -> Array1<F> {
    let n = y.len();
    let mut y_new = Array1::zeros(n);

    // y_new = y + h * (c1*k1 + c2*k2 + c3*k3 + c4*k4)
    Zip::from(&mut y_new)
        .and(y)
        .and(k1)
        .and(k2)
        .and(k3)
        .and(k4)
        .for_each(
            |y_new_elem, &y_elem, &k1_elem, &k2_elem, &k3_elem, &k4_elem| {
                *y_new_elem =
                    y_elem + h * (c1 * k1_elem + c2 * k2_elem + c3 * k3_elem + c4 * k4_elem);
            },
        );

    y_new
}

/// Evaluate ODE function with SIMD-optimized operations where possible
///
/// This function provides a framework for utilizing SIMD operations in ODE function evaluation.
/// For systems where the ODE function can be decomposed into SIMD-friendly operations,
/// this can provide significant performance improvements.
pub fn simd_ode_function_eval<F, Func>(
    f: &Func,
    t: F,
    y: &ArrayView1<F>,
    use_simd_postprocess: bool,
) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Call the user's ODE function
    let mut dy_dt = f(t, *y);

    // Optional SIMD post-processing (e.g., for clipping, normalization, etc.)
    if use_simd_postprocess {
        // Example: clip values to prevent overflow
        let max_val = F::from_f64(1e10).unwrap_or(F::infinity());
        let min_val = -max_val;

        // This would use SIMD operations if available
        dy_dt.map_inplace(|val| {
            if *val > max_val {
                *val = max_val;
            } else if *val < min_val {
                *val = min_val;
            }
        });
    }

    Ok(dy_dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_simd_rk_step() {
        let y = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let k1 = arr1(&[0.1, 0.2, 0.3, 0.4]);
        let k2 = arr1(&[0.2, 0.3, 0.4, 0.5]);
        let k3 = arr1(&[0.3, 0.4, 0.5, 0.6]);
        let k4 = arr1(&[0.4, 0.5, 0.6, 0.7]);

        let h = 0.1;
        let (c1, c2, c3, c4) = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0); // RK4 coefficients

        let result = simd_rk_step(
            &y.view(),
            h,
            &k1.view(),
            &k2.view(),
            &k3.view(),
            &k4.view(),
            c1,
            c2,
            c3,
            c4,
        );

        // Expected: y + h * (c1*k1 + c2*k2 + c3*k3 + c4*k4)
        for i in 0..y.len() {
            let expected = y[i] + h * (c1 * k1[i] + c2 * k2[i] + c3 * k3[i] + c4 * k4[i]);
            assert_relative_eq!(result[i], expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_simd_ode_function_eval() {
        let ode_func = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
            -y.to_owned() // Simple exponential decay
        };

        let y = arr1(&[1.0, 2.0, 3.0]);
        let result = simd_ode_function_eval(&ode_func, 0.0, &y.view(), false).unwrap();

        let expected = arr1(&[-1.0, -2.0, -3.0]);
        for i in 0..result.len() {
            assert_relative_eq!(result[i], expected[i], epsilon = 1e-12);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_axpy() {
        let mut y = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let x = arr1(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let a = 2.0f32;

        let expected = y.clone() + &x * a;

        SimdOdeOps::simd_axpy(&mut y.view_mut(), a, &x.view());

        for i in 0..y.len() {
            assert_relative_eq!(y[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_linear_combination() {
        let x = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let y = arr1(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let (a, b) = (2.0f32, 3.0f32);

        let result = SimdOdeOps::simd_linear_combination(&x.view(), a, &y.view(), b);
        let expected = &x * a + &y * b;

        for i in 0..result.len() {
            assert_relative_eq!(result[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_norms() {
        let x = arr1(&[3.0f32, 4.0, 0.0, -5.0, 12.0, 0.0, 0.0, 0.0]);

        let l2_norm = SimdOdeOps::simd_norm_l2(&x.view());
        let inf_norm = SimdOdeOps::simd_norm_inf(&x.view());

        let expected_l2 = (3.0f32 * 3.0 + 4.0 * 4.0 + 5.0 * 5.0 + 12.0 * 12.0).sqrt();
        let expected_inf = 12.0f32;

        assert_relative_eq!(l2_norm, expected_l2, epsilon = 1e-6);
        assert_relative_eq!(inf_norm, expected_inf, epsilon = 1e-6);
    }
}
