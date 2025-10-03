//! Common types and utilities for time series decomposition

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Result of time series decomposition
#[derive(Debug, Clone)]
pub struct DecompositionResult<F> {
    /// Trend component
    pub trend: Array1<F>,
    /// Seasonal component
    pub seasonal: Array1<F>,
    /// Residual component
    pub residual: Array1<F>,
    /// Original time series
    pub original: Array1<F>,
}

/// Decomposition model type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompositionModel {
    /// Additive model: Y = T + S + R
    Additive,
    /// Multiplicative model: Y = T * S * R
    Multiplicative,
}

/// Helper function to perform Box-Cox transformation for variance stabilization
pub(crate) fn box_cox_transform<F>(ts: &Array1<F>, lambda: f64) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let lambda_f = F::from_f64(lambda).ok_or_else(|| TimeSeriesError::InvalidParameter {
        name: "lambda".to_string(),
        message: "Failed to convert lambda to type F".to_string(),
    })?;
    let zero = F::zero();
    let one = F::one();

    if (lambda - 0.0).abs() < 1e-10 {
        // Log transform when lambda ~= 0
        let result = ts.mapv(|x| {
            if x <= zero {
                return zero; // Handle non-positive values
            }
            x.ln()
        });
        Ok(result)
    } else {
        // General Box-Cox transform
        let result = ts.mapv(|x| {
            if x <= zero {
                return zero; // Handle non-positive values
            }
            (x.powf(lambda_f) - one) / lambda_f
        });
        Ok(result)
    }
}

/// Helper function to apply inverse Box-Cox transformation
#[allow(dead_code)]
pub(crate) fn inverse_box_cox<F>(ts: &Array1<F>, lambda: f64) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let lambda_f = F::from_f64(lambda).ok_or_else(|| TimeSeriesError::InvalidParameter {
        name: "lambda".to_string(),
        message: "Failed to convert lambda to type F".to_string(),
    })?;
    let one = F::one();

    if (lambda - 0.0).abs() < 1e-10 {
        // Exponential for log transform
        let result = ts.mapv(|x| x.exp());
        Ok(result)
    } else {
        // General inverse Box-Cox
        let result = ts.mapv(|x| (x * lambda_f + one).powf(one / lambda_f));
        Ok(result)
    }
}

/// Helper function to get minimum value between two Ord objects
pub(crate) fn min<T: Ord>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}
