//! SIMD-optimized operations for ODE solvers
//!
//! This module provides SIMD-accelerated implementations of common operations
//! used in ODE solving, such as vector arithmetic, norm calculations, and
//! element-wise function evaluation. These optimizations can provide significant
//! performance improvements for large systems of ODEs.
//!
//! All SIMD operations are delegated to scirs2-core's unified SIMD abstraction layer
//! in compliance with the project-wide SIMD policy.

#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::needless_range_loop)]

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayViewMut1, Zip};
use scirs2_core::simd_ops::SimdUnifiedOps;

/// SIMD-optimized ODE operations
pub struct SimdOdeOps;

impl SimdOdeOps {
    /// Compute y = y + a * dy using SIMD operations
    pub fn simd_axpy<F: IntegrateFloat + SimdUnifiedOps>(
        y: &mut ArrayViewMut1<F>,
        a: F,
        dy: &ArrayView1<F>,
    ) {
        // Use core SIMD operations: y = y + a * dy
        #[cfg(feature = "simd")]
        if F::simd_available() {
            // Compute a * dy
            let scaled_dy = F::simd_scalar_mul(dy, a);
            // Add to y
            let y_view = ArrayView1::from(&*y);
            let result = F::simd_add(&y_view, &scaled_dy.view());
            // Copy result back to y
            y.assign(&result);
            return;
        }

        // Fallback implementation
        Zip::from(y).and(dy).for_each(|y_val, &dy_val| {
            *y_val += a * dy_val;
        });
    }

    /// Compute linear combination: result = a*x + b*y using SIMD
    pub fn simd_linear_combination<F: IntegrateFloat + SimdUnifiedOps>(
        x: &ArrayView1<F>,
        a: F,
        y: &ArrayView1<F>,
        b: F,
    ) -> Array1<F> {
        #[cfg(feature = "simd")]
        if F::simd_available() {
            // Compute a*x and b*y, then add them
            let ax = F::simd_scalar_mul(x, a);
            let by = F::simd_scalar_mul(y, b);
            return F::simd_add(&ax.view(), &by.view());
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
    pub fn simd_element_max<F: IntegrateFloat + SimdUnifiedOps>(
        a: &ArrayView1<F>,
        b: &ArrayView1<F>,
    ) -> Array1<F> {
        #[cfg(feature = "simd")]
        if F::simd_available() {
            return F::simd_max(a, b);
        }

        // Fallback implementation
        let mut result = Array1::zeros(a.len());
        Zip::from(&mut result)
            .and(a)
            .and(b)
            .for_each(|r, &a_val, &b_val| {
                *r = a_val.max(b_val);
            });
        result
    }

    /// Compute element-wise minimum using SIMD
    pub fn simd_element_min<F: IntegrateFloat + SimdUnifiedOps>(
        a: &ArrayView1<F>,
        b: &ArrayView1<F>,
    ) -> Array1<F> {
        #[cfg(feature = "simd")]
        if F::simd_available() {
            return F::simd_min(a, b);
        }

        // Fallback implementation
        let mut result = Array1::zeros(a.len());
        Zip::from(&mut result)
            .and(a)
            .and(b)
            .for_each(|r, &a_val, &b_val| {
                *r = a_val.min(b_val);
            });
        result
    }

    /// Compute L2 norm using SIMD
    pub fn simd_norm_l2<F: IntegrateFloat + SimdUnifiedOps>(x: &ArrayView1<F>) -> F {
        #[cfg(feature = "simd")]
        if F::simd_available() {
            return F::simd_norm(x);
        }

        // Fallback implementation
        let mut sum = F::zero();
        for &val in x.iter() {
            sum += val * val;
        }
        sum.sqrt()
    }

    /// Compute infinity norm using SIMD
    pub fn simd_norm_inf<F: IntegrateFloat + SimdUnifiedOps>(x: &ArrayView1<F>) -> F {
        #[cfg(feature = "simd")]
        if F::simd_available() {
            // Use SIMD to compute absolute values and find maximum
            let abs_x = F::simd_abs(x);
            return F::simd_max_element(&abs_x.view());
        }

        // Fallback implementation
        let mut max_val = F::zero();
        for &val in x.iter() {
            let abs_val = val.abs();
            if abs_val > max_val {
                max_val = abs_val;
            }
        }
        max_val
    }

    /// Apply scalar function element-wise using SIMD where possible
    pub fn simd_map_scalar<F, Func>(x: &ArrayView1<F>, f: Func) -> Array1<F>
    where
        F: IntegrateFloat + SimdUnifiedOps,
        Func: Fn(F) -> F,
    {
        // Note: Generic scalar functions cannot be vectorized directly
        // This is kept for API compatibility but doesn't use SIMD
        let mut result = Array1::zeros(x.len());
        Zip::from(&mut result).and(x).for_each(|r, &x_val| {
            *r = f(x_val);
        });
        result
    }
}

/// SIMD-optimized dense update for ODE solvers
///
/// Computes: y = a0 * y0 + a1 * y1 + a2 * y2 + ... + an * yn
///
/// This is a common operation in multistage ODE methods like Runge-Kutta.
#[allow(dead_code)]
pub fn simd_dense_update<F: IntegrateFloat + SimdUnifiedOps>(
    coefficients: &[F],
    states: &[ArrayView1<F>],
) -> IntegrateResult<Array1<F>> {
    if coefficients.is_empty() || states.is_empty() {
        return Err(crate::error::IntegrateError::ValueError(
            "Empty coefficients or states".to_string(),
        ));
    }

    if coefficients.len() != states.len() {
        return Err(crate::error::IntegrateError::ValueError(
            "Coefficients and states must have the same length".to_string(),
        ));
    }

    let n = states[0].len();
    for state in states.iter() {
        if state.len() != n {
            return Err(crate::error::IntegrateError::ValueError(
                "All states must have the same length".to_string(),
            ));
        }
    }

    // Start with the first term
    let mut result = F::simd_scalar_mul(&states[0], coefficients[0]);

    // Add remaining terms using SIMD FMA when available
    for (coeff, state) in coefficients[1..].iter().zip(&states[1..]) {
        let term = F::simd_scalar_mul(state, *coeff);
        result = F::simd_add(&result.view(), &term.view());
    }

    Ok(result)
}

/// SIMD-optimized Runge-Kutta step evaluation
///
/// Evaluates: k_new = f(t + c*dt, y + sum(a_ij * k_j * dt))
#[allow(dead_code)]
pub fn simd_rk_step<F: IntegrateFloat + SimdUnifiedOps>(
    y: &ArrayView1<F>,
    k_stages: &[Array1<F>],
    coefficients: &[F],
    dt: F,
) -> IntegrateResult<Array1<F>> {
    if coefficients.is_empty() || k_stages.is_empty() {
        return Ok(y.to_owned());
    }

    if coefficients.len() != k_stages.len() {
        return Err(crate::error::IntegrateError::ValueError(
            "Coefficients and k_stages must have the same length".to_string(),
        ));
    }

    // Compute y + sum(a_ij * k_j * dt) using SIMD operations
    let mut temp_state = y.to_owned();

    for (coeff, k) in coefficients.iter().zip(k_stages.iter()) {
        let scaled_k = F::simd_scalar_mul(&k.view(), *coeff * dt);
        temp_state = F::simd_add(&temp_state.view(), &scaled_k.view());
    }

    Ok(temp_state)
}

/// SIMD-optimized function evaluation for systems of ODEs
///
/// Evaluates multiple ODE functions in parallel when possible.
#[allow(dead_code)]
pub fn simd_ode_function_eval<F, Func>(
    t: F,
    y: &ArrayView1<F>,
    f: &Func,
) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat + SimdUnifiedOps,
    Func: Fn(F, &ArrayView1<F>) -> IntegrateResult<Array1<F>>,
{
    // Direct function evaluation - SIMD optimizations would be within the function itself
    f(t, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_simd_axpy() {
        let mut y = array![1.0, 2.0, 3.0, 4.0];
        let dy = array![0.1, 0.2, 0.3, 0.4];
        let a = 2.0;

        SimdOdeOps::simd_axpy(&mut y.view_mut(), a, &dy.view());

        assert_eq!(y, array![1.2, 2.4, 3.6, 4.8]);
    }

    #[test]
    fn test_simd_linear_combination() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![0.1, 0.2, 0.3, 0.4];
        let a = 2.0;
        let b = 3.0;

        let result = SimdOdeOps::simd_linear_combination(&x.view(), a, &y.view(), b);

        assert_eq!(result, array![2.3, 4.6, 6.9, 9.2]);
    }

    #[test]
    fn test_simd_element_max() {
        let a = array![1.0, 5.0, 3.0, 7.0];
        let b = array![2.0, 4.0, 6.0, 1.0];

        let result = SimdOdeOps::simd_element_max(&a.view(), &b.view());

        assert_eq!(result, array![2.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_simd_norm_l2() {
        let x = array![3.0, 4.0];
        let norm = SimdOdeOps::simd_norm_l2(&x.view());
        assert_eq!(norm, 5.0);
    }

    #[test]
    fn test_simd_norm_inf() {
        let x = array![-3.0, 4.0, -5.0, 2.0];
        let norm = SimdOdeOps::simd_norm_inf(&x.view());
        assert_eq!(norm, 5.0);
    }
}
