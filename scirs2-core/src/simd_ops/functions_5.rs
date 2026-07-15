//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use ::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};

use super::functions::SimdUnifiedOps;
use super::functions_2::{
    digamma_f32, digamma_f64, erf_f32, erf_f64, erfc_f32, erfc_f64, erfinv_f32, lanczos_gamma_f32,
    lanczos_gamma_f64, ln_gamma_f32, ln_gamma_f64, trigamma_f32, trigamma_f64,
};
use super::functions_3::{
    elu_f32, elu_f64, erfcinv_f32, erfcinv_f64, erfinv_f64, gelu_f32, gelu_f64, hardsigmoid_f32,
    hardsigmoid_f64, hardswish_f32, hardswish_f64, mish_f32, mish_f64, selu_f32, selu_f64,
    sigmoid_f32, sigmoid_f64, sinc_f32, sinc_f64, softplus_f32, softplus_f64, swish_f32, swish_f64,
};
#[cfg(feature = "simd")]
use crate::simd_ops_polynomial;

impl SimdUnifiedOps for f64 {
    #[cfg(feature = "simd")]
    fn simd_add(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_add_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_add(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        (a + b).to_owned()
    }
    #[cfg(feature = "simd")]
    fn simd_sub(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_sub_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_sub(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        (a - b).to_owned()
    }
    #[cfg(feature = "simd")]
    fn simd_mul(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_mul_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_mul(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        (a * b).to_owned()
    }
    #[cfg(feature = "simd")]
    fn simd_div(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_div_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_div(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        (a / b).to_owned()
    }
    #[cfg(feature = "simd")]
    fn simd_dot(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_dot_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_dot(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.dot(b)
    }
    fn simd_gemv(a: &ArrayView2<Self>, x: &ArrayView1<Self>, beta: Self, y: &mut Array1<Self>) {
        let m = a.nrows();
        let n = a.ncols();
        assert_eq!(n, x.len());
        assert_eq!(m, y.len());
        if beta == 0.0 {
            y.fill(0.0);
        } else if beta != 1.0 {
            y.mapv_inplace(|v| v * beta);
        }
        for i in 0..m {
            let row = a.row(i);
            y[i] += Self::simd_dot(&row, x);
        }
    }
    fn simd_gemm(
        alpha: Self,
        a: &ArrayView2<Self>,
        b: &ArrayView2<Self>,
        beta: Self,
        c: &mut Array2<Self>,
    ) {
        let m = a.nrows();
        let k = a.ncols();
        let n = b.ncols();
        assert_eq!(k, b.nrows());
        assert_eq!((m, n), c.dim());
        if beta == 0.0 {
            c.fill(0.0);
        } else if beta != 1.0 {
            c.mapv_inplace(|v| v * beta);
        }
        const GEMM_TRANSPOSE_THRESHOLD: usize = 4096;
        if n * k > GEMM_TRANSPOSE_THRESHOLD {
            let b_t = Self::simd_transpose_blocked(b);
            for i in 0..m {
                let a_row = a.row(i);
                for j in 0..n {
                    let b_row = b_t.row(j);
                    c[[i, j]] += alpha * Self::simd_dot(&a_row, &b_row);
                }
            }
        } else {
            for i in 0..m {
                let a_row = a.row(i);
                for j in 0..n {
                    let b_col = b.column(j);
                    c[[i, j]] += alpha * Self::simd_dot(&a_row, &b_col);
                }
            }
        }
    }
    #[cfg(feature = "simd")]
    fn simd_norm(a: &ArrayView1<Self>) -> Self {
        crate::simd::norms::simd_norm_l2_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_norm(a: &ArrayView1<Self>) -> Self {
        a.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
    #[cfg(feature = "simd")]
    fn simd_max(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_maximum_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_max(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for i in 0..a.len() {
            result[i] = a[i].max(b[i]);
        }
        result
    }
    #[cfg(feature = "simd")]
    fn simd_min(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_minimum_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_min(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for i in 0..a.len() {
            result[i] = a[i].min(b[i]);
        }
        result
    }
    #[cfg(feature = "simd")]
    fn simd_scalar_mul(a: &ArrayView1<Self>, scalar: Self) -> Array1<Self> {
        crate::simd::simd_scalar_mul_f64(a, scalar)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_scalar_mul(a: &ArrayView1<Self>, scalar: Self) -> Array1<Self> {
        a.mapv(|x| x * scalar)
    }
    #[cfg(feature = "simd")]
    fn simd_sum(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_sum_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_sum(a: &ArrayView1<Self>) -> Self {
        a.sum()
    }
    fn simd_mean(a: &ArrayView1<Self>) -> Self {
        if a.is_empty() {
            0.0
        } else {
            Self::simd_sum(a) / (a.len() as f64)
        }
    }
    #[cfg(feature = "simd")]
    fn simd_max_element(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_max_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_max_element(a: &ArrayView1<Self>) -> Self {
        a.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
    }
    #[cfg(feature = "simd")]
    fn simd_min_element(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_min_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_min_element(a: &ArrayView1<Self>) -> Self {
        a.fold(f64::INFINITY, |acc, &x| acc.min(x))
    }
    #[cfg(feature = "simd")]
    fn simd_fma(a: &ArrayView1<Self>, b: &ArrayView1<Self>, c: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_fused_multiply_add_f64(a, b, c)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_fma(a: &ArrayView1<Self>, b: &ArrayView1<Self>, c: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for i in 0..a.len() {
            result[i] = a[i] * b[i] + c[i];
        }
        result
    }
    #[cfg(feature = "simd")]
    fn simd_add_cache_optimized(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_add_cache_optimized_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_add_cache_optimized(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a + b
    }
    #[cfg(feature = "simd")]
    fn simd_fma_advanced_optimized(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        c: &ArrayView1<Self>,
    ) -> Array1<Self> {
        crate::simd::simd_fma_advanced_optimized_f64(a, b, c)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_fma_advanced_optimized(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        c: &ArrayView1<Self>,
    ) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for _i in 0..a.len() {
            result[0] = a[0] * b[0] + c[0];
        }
        result
    }
    #[cfg(feature = "simd")]
    fn simd_add_adaptive(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_adaptive_add_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_add_adaptive(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a + b
    }
    fn simd_transpose(a: &ArrayView2<Self>) -> Array2<Self> {
        a.t().to_owned()
    }
    fn simd_transpose_blocked(a: &ArrayView2<Self>) -> Array2<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_transpose_blocked_f64(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.t().to_owned()
        }
    }
    fn simd_sum_squares(a: &ArrayView1<Self>) -> Self {
        a.iter().map(|&x| x * x).sum()
    }
    fn simd_multiply(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        Self::simd_mul(a, b)
    }
    #[cfg(feature = "simd")]
    fn simd_available() -> bool {
        true
    }
    #[cfg(not(feature = "simd"))]
    fn simd_available() -> bool {
        false
    }
    fn simd_sub_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let sub_result = Self::simd_sub(a, b);
        result.assign(&sub_result);
    }
    fn simd_mul_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let mul_result = Self::simd_mul(a, b);
        result.assign(&mul_result);
    }
    fn simd_sum_cubes(a: &ArrayView1<Self>) -> Self {
        a.iter().map(|&x| x * x * x).sum()
    }
    fn simd_div_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let div_result = Self::simd_div(a, b);
        result.assign(&div_result);
    }
    fn simd_sin_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>) {
        let sin_result = a.mapv(|x| x.sin());
        result.assign(&sin_result);
    }
    fn simd_add_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let add_result = Self::simd_add(a, b);
        result.assign(&add_result);
    }
    fn simd_fma_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        c: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let fma_result = Self::simd_fma(a, b, c);
        result.assign(&fma_result);
    }
    fn simd_pow_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let pow_result = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x.powf(y))
            .collect::<Vec<_>>();
        result.assign(&Array1::from_vec(pow_result));
    }
    fn simd_exp_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>) {
        let exp_result = a.mapv(|x| x.exp());
        result.assign(&exp_result);
    }
    fn simd_cos_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>) {
        let cos_result = a.mapv(|x| x.cos());
        result.assign(&cos_result);
    }
    fn simd_dot_f32_ultra(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        Self::simd_dot(a, b)
    }
    #[cfg(feature = "simd")]
    fn simd_variance(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_variance_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_variance(a: &ArrayView1<Self>) -> Self {
        let mean = Self::simd_mean(a);
        let n = a.len() as f64;
        if n < 2.0 {
            return f64::NAN;
        }
        a.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    }
    #[cfg(feature = "simd")]
    fn simd_std(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_std_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_std(a: &ArrayView1<Self>) -> Self {
        Self::simd_variance(a).sqrt()
    }
    #[cfg(feature = "simd")]
    fn simd_norm_l1(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_norm_l1_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_norm_l1(a: &ArrayView1<Self>) -> Self {
        a.iter().map(|&x| x.abs()).sum()
    }
    #[cfg(feature = "simd")]
    fn simd_norm_linf(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_norm_linf_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_norm_linf(a: &ArrayView1<Self>) -> Self {
        a.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()))
    }
    #[cfg(feature = "simd")]
    fn simd_cosine_similarity(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_cosine_similarity_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_cosine_similarity(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        let dot = Self::simd_dot(a, b);
        let norm_a = Self::simd_norm(a);
        let norm_b = Self::simd_norm(b);
        dot / (norm_a * norm_b)
    }
    #[cfg(feature = "simd")]
    fn simd_distance_euclidean(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_distance_euclidean_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_distance_euclidean(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
    #[cfg(feature = "simd")]
    fn simd_distance_manhattan(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_distance_manhattan_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_distance_manhattan(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
    }
    #[cfg(feature = "simd")]
    fn simd_distance_chebyshev(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_distance_chebyshev_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_distance_chebyshev(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.iter()
            .zip(b.iter())
            .fold(0.0f64, |acc, (&x, &y)| acc.max((x - y).abs()))
    }
    #[cfg(feature = "simd")]
    fn simd_distance_cosine(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_distance_cosine_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_distance_cosine(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        1.0 - Self::simd_cosine_similarity(a, b)
    }
    #[cfg(feature = "simd")]
    fn simd_distance_squared_euclidean(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_distance_squared_euclidean_f64(a, b)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_distance_squared_euclidean(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
    }
    #[cfg(feature = "simd")]
    fn simd_weighted_sum(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self {
        crate::simd::simd_weighted_sum_f64(values, weights)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_weighted_sum(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self {
        values
            .iter()
            .zip(weights.iter())
            .map(|(&v, &w)| v * w)
            .sum()
    }
    #[cfg(feature = "simd")]
    fn simd_weighted_mean(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self {
        crate::simd::simd_weighted_mean_f64(values, weights)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_weighted_mean(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self {
        let weighted_sum = Self::simd_weighted_sum(values, weights);
        let weight_sum: f64 = weights.iter().sum();
        weighted_sum / weight_sum
    }
    #[cfg(feature = "simd")]
    fn simd_argmin(a: &ArrayView1<Self>) -> Option<usize> {
        crate::simd::simd_argmin_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_argmin(a: &ArrayView1<Self>) -> Option<usize> {
        if a.is_empty() {
            return None;
        }
        let mut min_idx = 0;
        let mut min_val = a[0];
        for (i, &v) in a.iter().enumerate().skip(1) {
            if v < min_val {
                min_val = v;
                min_idx = i;
            }
        }
        Some(min_idx)
    }
    #[cfg(feature = "simd")]
    fn simd_argmax(a: &ArrayView1<Self>) -> Option<usize> {
        crate::simd::simd_argmax_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_argmax(a: &ArrayView1<Self>) -> Option<usize> {
        if a.is_empty() {
            return None;
        }
        let mut max_idx = 0;
        let mut max_val = a[0];
        for (i, &v) in a.iter().enumerate().skip(1) {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        Some(max_idx)
    }
    #[cfg(feature = "simd")]
    fn simd_clip(a: &ArrayView1<Self>, min_val: Self, max_val: Self) -> Array1<Self> {
        crate::simd::simd_clip_f64(a, min_val, max_val)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_clip(a: &ArrayView1<Self>, min_val: Self, max_val: Self) -> Array1<Self> {
        a.mapv(|v| v.max(min_val).min(max_val))
    }
    #[cfg(feature = "simd")]
    fn simd_log_sum_exp(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_log_sum_exp_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_log_sum_exp(a: &ArrayView1<Self>) -> Self {
        if a.is_empty() {
            return f64::NEG_INFINITY;
        }
        let max_val = a.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let sum_exp: f64 = a.iter().map(|&x| (x - max_val).exp()).sum();
        max_val + sum_exp.ln()
    }
    #[cfg(feature = "simd")]
    fn simd_softmax(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_softmax_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_softmax(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        let lse = Self::simd_log_sum_exp(a);
        a.mapv(|x| (x - lse).exp())
    }
    #[cfg(feature = "simd")]
    fn simd_cumsum(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_cumsum_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_cumsum(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        let mut cumsum = 0.0f64;
        a.mapv(|x| {
            cumsum += x;
            cumsum
        })
    }
    #[cfg(feature = "simd")]
    fn simd_cumprod(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_cumprod_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_cumprod(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        let mut cumprod = 1.0f64;
        a.mapv(|x| {
            cumprod *= x;
            cumprod
        })
    }
    #[cfg(feature = "simd")]
    fn simd_diff(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_diff_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_diff(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.len() <= 1 {
            return Array1::zeros(0);
        }
        Array1::from_iter((1..a.len()).map(|i| a[i] - a[i - 1]))
    }
    #[cfg(feature = "simd")]
    fn simd_sign(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_sign_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_sign(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
    }
    #[cfg(feature = "simd")]
    fn simd_relu(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_relu_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_relu(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.max(0.0))
    }
    #[cfg(feature = "simd")]
    fn simd_leaky_relu(a: &ArrayView1<Self>, alpha: Self) -> Array1<Self> {
        crate::simd::simd_leaky_relu_f64(a, alpha)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_leaky_relu(a: &ArrayView1<Self>, alpha: Self) -> Array1<Self> {
        a.mapv(|x| if x > 0.0 { x } else { alpha * x })
    }
    #[cfg(feature = "simd")]
    fn simd_normalize(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_normalize_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_normalize(a: &ArrayView1<Self>) -> Array1<Self> {
        let norm: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return a.to_owned();
        }
        a.mapv(|x| x / norm)
    }
    #[cfg(feature = "simd")]
    fn simd_standardize(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_standardize_f64(a)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_standardize(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.len() <= 1 {
            return Array1::zeros(a.len());
        }
        let mean: f64 = a.iter().sum::<f64>() / a.len() as f64;
        let variance: f64 =
            a.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (a.len() - 1) as f64;
        let std = variance.sqrt();
        if std == 0.0 {
            return Array1::zeros(a.len());
        }
        a.mapv(|x| (x - mean) / std)
    }
    fn simd_abs(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.abs())
    }
    fn simd_sqrt(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.sqrt())
    }
    fn simd_exp(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.exp())
    }
    fn simd_ln(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.ln())
    }
    fn simd_sin(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.sin())
    }
    fn simd_cos(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.cos())
    }
    fn simd_tan(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.tan())
    }
    fn simd_sinh(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            simd_ops_polynomial::simd_sinh_f64_poly(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.sinh())
        }
    }
    fn simd_cosh(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            simd_ops_polynomial::simd_cosh_f64_poly(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.cosh())
        }
    }
    fn simd_tanh(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            simd_ops_polynomial::simd_tanh_f64_poly(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.tanh())
        }
    }
    fn simd_floor(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_floor_f64(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.floor())
        }
    }
    fn simd_ceil(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_ceil_f64(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.ceil())
        }
    }
    fn simd_round(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_round_f64(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.round())
        }
    }
    fn simd_atan(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.atan())
    }
    fn simd_asin(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.asin())
    }
    fn simd_acos(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.acos())
    }
    fn simd_atan2(y: &ArrayView1<Self>, x: &ArrayView1<Self>) -> Array1<Self> {
        y.iter()
            .zip(x.iter())
            .map(|(&y_val, &x_val)| y_val.atan2(x_val))
            .collect::<Vec<_>>()
            .into()
    }
    fn simd_log10(a: &ArrayView1<Self>) -> Array1<Self> {
        const LOG10_E: f64 = std::f64::consts::LOG10_E;
        let ln_a = Self::simd_ln(a);
        Self::simd_scalar_mul(&ln_a.view(), LOG10_E)
    }
    fn simd_log2(a: &ArrayView1<Self>) -> Array1<Self> {
        const LOG2_E: f64 = std::f64::consts::LOG2_E;
        let ln_a = Self::simd_ln(a);
        Self::simd_scalar_mul(&ln_a.view(), LOG2_E)
    }
    #[cfg(feature = "simd")]
    fn simd_clamp(a: &ArrayView1<Self>, min: Self, max: Self) -> Array1<Self> {
        crate::simd::simd_clip_f64(a, min, max)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_clamp(a: &ArrayView1<Self>, min: Self, max: Self) -> Array1<Self> {
        a.mapv(|x| x.clamp(min, max))
    }
    fn simd_fract(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            let truncated = crate::simd::simd_trunc_f64(a);
            Self::simd_sub(a, &truncated.view())
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.fract())
        }
    }
    fn simd_trunc(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_trunc_f64(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.trunc())
        }
    }
    fn simd_recip(a: &ArrayView1<Self>) -> Array1<Self> {
        let ones = Array1::from_elem(a.len(), 1.0f64);
        Self::simd_div(&ones.view(), a)
    }
    fn simd_powf(base: &ArrayView1<Self>, exp: Self) -> Array1<Self> {
        let ln_base = Self::simd_ln(base);
        let scaled = Self::simd_scalar_mul(&ln_base.view(), exp);
        Self::simd_exp(&scaled.view())
    }
    fn simd_pow(base: &ArrayView1<Self>, exp: &ArrayView1<Self>) -> Array1<Self> {
        let ln_base = Self::simd_ln(base);
        let scaled = Self::simd_mul(&ln_base.view(), exp);
        Self::simd_exp(&scaled.view())
    }
    #[cfg(feature = "simd")]
    fn simd_powi(base: &ArrayView1<Self>, n: i32) -> Array1<Self> {
        crate::simd::unary_powi::simd_powi_f64(base, n)
    }
    #[cfg(not(feature = "simd"))]
    fn simd_powi(base: &ArrayView1<Self>, n: i32) -> Array1<Self> {
        base.mapv(|x| x.powi(n))
    }
    fn simd_gamma(x: &ArrayView1<Self>) -> Array1<Self> {
        x.mapv(lanczos_gamma_f64)
    }
    fn simd_exp2(a: &ArrayView1<Self>) -> Array1<Self> {
        const LN2: f64 = std::f64::consts::LN_2;
        let scaled = Self::simd_scalar_mul(a, LN2);
        Self::simd_exp(&scaled.view())
    }
    fn simd_cbrt(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.cbrt())
    }
    fn simd_ln_1p(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.ln_1p())
    }
    fn simd_exp_m1(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.exp_m1())
    }
    fn simd_to_radians(a: &ArrayView1<Self>) -> Array1<Self> {
        const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
        Self::simd_scalar_mul(a, DEG_TO_RAD)
    }
    fn simd_to_degrees(a: &ArrayView1<Self>) -> Array1<Self> {
        const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;
        Self::simd_scalar_mul(a, RAD_TO_DEG)
    }
    fn simd_digamma(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(digamma_f64)
    }
    fn simd_trigamma(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(trigamma_f64)
    }
    fn simd_ln_gamma(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(ln_gamma_f64)
    }
    fn simd_erf(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(erf_f64)
    }
    fn simd_erfc(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(erfc_f64)
    }
    fn simd_erfinv(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(erfinv_f64)
    }
    fn simd_erfcinv(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(erfcinv_f64)
    }
    fn simd_sigmoid(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(sigmoid_f64)
    }
    fn simd_gelu(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(gelu_f64)
    }
    fn simd_swish(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(swish_f64)
    }
    fn simd_softplus(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(softplus_f64)
    }
    fn simd_mish(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(mish_f64)
    }
    fn simd_elu(a: &ArrayView1<Self>, alpha: Self) -> Array1<Self> {
        a.mapv(|x| elu_f64(x, alpha))
    }
    fn simd_selu(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(selu_f64)
    }
    fn simd_hardsigmoid(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(hardsigmoid_f64)
    }
    fn simd_hardswish(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(hardswish_f64)
    }
    fn simd_sinc(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(sinc_f64)
    }
    fn simd_log_softmax(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        let lse = Self::simd_log_sum_exp(a);
        a.mapv(|x| x - lse)
    }
    fn simd_asinh(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.asinh())
    }
    fn simd_acosh(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.acosh())
    }
    fn simd_atanh(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.atanh())
    }
    fn simd_ln_beta(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        let ln_gamma_a = Self::simd_ln_gamma(a);
        let ln_gamma_b = Self::simd_ln_gamma(b);
        let a_plus_b = Self::simd_add(a, b);
        let ln_gamma_ab = Self::simd_ln_gamma(&a_plus_b.view());
        Self::simd_sub(
            &Self::simd_add(&ln_gamma_a.view(), &ln_gamma_b.view()).view(),
            &ln_gamma_ab.view(),
        )
    }
    fn simd_beta(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        let ln_beta = Self::simd_ln_beta(a, b);
        Self::simd_exp(&ln_beta.view())
    }
    fn simd_lerp(a: &ArrayView1<Self>, b: &ArrayView1<Self>, t: Self) -> Array1<Self> {
        if a.is_empty() || b.is_empty() {
            return Array1::zeros(0);
        }
        let diff = Self::simd_sub(b, a);
        let scaled = Self::simd_scalar_mul(&diff.view(), t);
        Self::simd_add(a, &scaled.view())
    }
    fn simd_smoothstep(edge0: Self, edge1: Self, x: &ArrayView1<Self>) -> Array1<Self> {
        if x.is_empty() {
            return Array1::zeros(0);
        }
        let range = edge1 - edge0;
        if range.abs() < Self::EPSILON {
            return x.mapv(|xi| if xi < edge0 { 0.0 } else { 1.0 });
        }
        x.mapv(|xi| {
            let t = ((xi - edge0) / range).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        })
    }
    fn simd_hypot(x: &ArrayView1<Self>, y: &ArrayView1<Self>) -> Array1<Self> {
        if x.is_empty() || y.is_empty() {
            return Array1::zeros(0);
        }
        let len = x.len().min(y.len());
        Array1::from_iter((0..len).map(|i| x[i].hypot(y[i])))
    }
    fn simd_copysign(x: &ArrayView1<Self>, y: &ArrayView1<Self>) -> Array1<Self> {
        if x.is_empty() || y.is_empty() {
            return Array1::zeros(0);
        }
        let len = x.len().min(y.len());
        Array1::from_iter((0..len).map(|i| x[i].copysign(y[i])))
    }
    fn simd_smootherstep(edge0: Self, edge1: Self, x: &ArrayView1<Self>) -> Array1<Self> {
        if x.is_empty() {
            return Array1::zeros(0);
        }
        let range = edge1 - edge0;
        if range.abs() < Self::EPSILON {
            return x.mapv(|xi| if xi < edge0 { 0.0 } else { 1.0 });
        }
        x.mapv(|xi| {
            let t = ((xi - edge0) / range).clamp(0.0, 1.0);
            let t3 = t * t * t;
            t3 * (t * (t * 6.0 - 15.0) + 10.0)
        })
    }
    fn simd_logaddexp(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() || b.is_empty() {
            return Array1::zeros(0);
        }
        let len = a.len().min(b.len());
        Array1::from_iter((0..len).map(|i| {
            let ai = a[i];
            let bi = b[i];
            let max_val = ai.max(bi);
            let diff = (ai - bi).abs();
            if diff > 50.0 {
                max_val
            } else {
                max_val + (1.0 + (-diff).exp()).ln()
            }
        }))
    }
    fn simd_logit(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|p| {
            if p <= 0.0 {
                Self::NEG_INFINITY
            } else if p >= 1.0 {
                Self::INFINITY
            } else {
                (p / (1.0 - p)).ln()
            }
        })
    }
    fn simd_square(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|x| x * x)
    }
    fn simd_rsqrt(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|x| {
            if x <= 0.0 {
                if x == 0.0 {
                    Self::INFINITY
                } else {
                    Self::NAN
                }
            } else {
                1.0 / x.sqrt()
            }
        })
    }
    fn simd_sincos(a: &ArrayView1<Self>) -> (Array1<Self>, Array1<Self>) {
        if a.is_empty() {
            return (Array1::zeros(0), Array1::zeros(0));
        }
        let sin_result = a.mapv(|x| x.sin());
        let cos_result = a.mapv(|x| x.cos());
        (sin_result, cos_result)
    }
    fn simd_expm1(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|x| x.exp_m1())
    }
    fn simd_log1p(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|x| x.ln_1p())
    }

    // ============================================================================
    // ZERO-ALLOCATION SIMD OPERATIONS (Phase 1: ToRSh SIMD Performance Fix)
    // ============================================================================

    #[cfg(feature = "simd")]
    fn simd_add_into(a: &[Self], b: &[Self], output: &mut [Self]) {
        assert_eq!(a.len(), b.len(), "Input arrays must have same length");
        assert_eq!(
            a.len(),
            output.len(),
            "Output buffer must match input length"
        );

        let len = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;

            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let mut i = 0;
                    // Process 4 f64s at a time with AVX2
                    while i + 4 <= len {
                        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
                        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
                        let result_vec = _mm256_add_pd(a_vec, b_vec);
                        _mm256_storeu_pd(output.as_mut_ptr().add(i), result_vec);
                        i += 4;
                    }
                    while i < len {
                        *output.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
                        i += 1;
                    }
                }
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;

            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    let mut i = 0;
                    // Process 2 f64s at a time with NEON
                    while i + 2 <= len {
                        let a_vec = vld1q_f64(a.as_ptr().add(i));
                        let b_vec = vld1q_f64(b.as_ptr().add(i));
                        let result_vec = vaddq_f64(a_vec, b_vec);
                        vst1q_f64(output.as_mut_ptr().add(i), result_vec);
                        i += 2;
                    }
                    while i < len {
                        *output.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
                        i += 1;
                    }
                }
                return;
            }
        }

        for i in 0..len {
            output[i] = a[i] + b[i];
        }
    }

    #[cfg(not(feature = "simd"))]
    fn simd_add_into(a: &[Self], b: &[Self], output: &mut [Self]) {
        assert_eq!(a.len(), b.len(), "Input arrays must have same length");
        assert_eq!(
            a.len(),
            output.len(),
            "Output buffer must match input length"
        );
        for i in 0..a.len() {
            output[i] = a[i] + b[i];
        }
    }

    fn simd_sub_into(a: &[Self], b: &[Self], output: &mut [Self]) {
        assert_eq!(a.len(), b.len(), "Input arrays must have same length");
        assert_eq!(
            a.len(),
            output.len(),
            "Output buffer must match input length"
        );
        for i in 0..a.len() {
            output[i] = a[i] - b[i];
        }
    }

    #[cfg(feature = "simd")]
    fn simd_mul_into(a: &[Self], b: &[Self], output: &mut [Self]) {
        assert_eq!(a.len(), b.len(), "Input arrays must have same length");
        assert_eq!(
            a.len(),
            output.len(),
            "Output buffer must match input length"
        );

        let len = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;

            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let mut i = 0;
                    while i + 4 <= len {
                        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
                        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
                        let result_vec = _mm256_mul_pd(a_vec, b_vec);
                        _mm256_storeu_pd(output.as_mut_ptr().add(i), result_vec);
                        i += 4;
                    }
                    while i < len {
                        *output.get_unchecked_mut(i) = *a.get_unchecked(i) * *b.get_unchecked(i);
                        i += 1;
                    }
                }
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;

            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    let mut i = 0;
                    while i + 2 <= len {
                        let a_vec = vld1q_f64(a.as_ptr().add(i));
                        let b_vec = vld1q_f64(b.as_ptr().add(i));
                        let result_vec = vmulq_f64(a_vec, b_vec);
                        vst1q_f64(output.as_mut_ptr().add(i), result_vec);
                        i += 2;
                    }
                    while i < len {
                        *output.get_unchecked_mut(i) = *a.get_unchecked(i) * *b.get_unchecked(i);
                        i += 1;
                    }
                }
                return;
            }
        }

        for i in 0..len {
            output[i] = a[i] * b[i];
        }
    }

    #[cfg(not(feature = "simd"))]
    fn simd_mul_into(a: &[Self], b: &[Self], output: &mut [Self]) {
        assert_eq!(a.len(), b.len(), "Input arrays must have same length");
        assert_eq!(
            a.len(),
            output.len(),
            "Output buffer must match input length"
        );
        for i in 0..a.len() {
            output[i] = a[i] * b[i];
        }
    }

    fn simd_div_into(a: &[Self], b: &[Self], output: &mut [Self]) {
        assert_eq!(a.len(), b.len(), "Input arrays must have same length");
        assert_eq!(
            a.len(),
            output.len(),
            "Output buffer must match input length"
        );
        for i in 0..a.len() {
            output[i] = a[i] / b[i];
        }
    }

    fn simd_add_inplace(a: &mut [Self], b: &[Self]) {
        assert_eq!(a.len(), b.len(), "Arrays must have same length");
        for i in 0..a.len() {
            a[i] += b[i];
        }
    }

    fn simd_sub_inplace(a: &mut [Self], b: &[Self]) {
        assert_eq!(a.len(), b.len(), "Arrays must have same length");
        for i in 0..a.len() {
            a[i] -= b[i];
        }
    }

    fn simd_mul_inplace(a: &mut [Self], b: &[Self]) {
        assert_eq!(a.len(), b.len(), "Arrays must have same length");
        for i in 0..a.len() {
            a[i] *= b[i];
        }
    }

    fn simd_div_inplace(a: &mut [Self], b: &[Self]) {
        assert_eq!(a.len(), b.len(), "Arrays must have same length");
        for i in 0..a.len() {
            a[i] /= b[i];
        }
    }

    fn simd_add_scalar_inplace(a: &mut [Self], scalar: Self) {
        for x in a.iter_mut() {
            *x += scalar;
        }
    }

    fn simd_mul_scalar_inplace(a: &mut [Self], scalar: Self) {
        for x in a.iter_mut() {
            *x *= scalar;
        }
    }

    fn simd_fma_into(a: &[Self], b: &[Self], c: &[Self], output: &mut [Self]) {
        assert_eq!(a.len(), b.len(), "Input arrays must have same length");
        assert_eq!(a.len(), c.len(), "Input arrays must have same length");
        assert_eq!(
            a.len(),
            output.len(),
            "Output buffer must match input length"
        );
        for i in 0..a.len() {
            output[i] = a[i].mul_add(b[i], c[i]);
        }
    }
}
