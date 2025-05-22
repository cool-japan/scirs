//! Regression metrics module
//!
//! This module provides functions for evaluating regression models, including
//! error metrics, correlation metrics, residual analysis, and robust metrics.

mod correlation;
mod error;
mod residual;
mod robust;

// Re-export all public items from submodules
pub use self::correlation::*;
pub use self::error::*;
pub use self::residual::*;
pub use self::robust::*;

// Common utility functions that might be used across multiple submodules
use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{Float, FromPrimitive, NumCast};

/// Check if two arrays have the same shape
pub(crate) fn check_same_shape<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> crate::error::Result<()>
where
    F: num_traits::Float,
    S1: ndarray::Data<Elem = F>,
    S2: ndarray::Data<Elem = F>,
    D1: ndarray::Dimension,
    D2: ndarray::Dimension,
{
    if y_true.shape() != y_pred.shape() {
        return Err(crate::error::MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(crate::error::MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    Ok(())
}

/// Check if all values in arrays are non-negative
pub(crate) fn check_non_negative<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> crate::error::Result<()>
where
    F: num_traits::Float + std::fmt::Debug,
    S1: ndarray::Data<Elem = F>,
    S2: ndarray::Data<Elem = F>,
    D1: ndarray::Dimension,
    D2: ndarray::Dimension,
{
    for val in y_true.iter() {
        if *val < F::zero() {
            return Err(crate::error::MetricsError::InvalidInput(
                "y_true contains negative values".to_string(),
            ));
        }
    }

    for val in y_pred.iter() {
        if *val < F::zero() {
            return Err(crate::error::MetricsError::InvalidInput(
                "y_pred contains negative values".to_string(),
            ));
        }
    }

    Ok(())
}

/// Check if all values in arrays are strictly positive
pub(crate) fn check_positive<F, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> crate::error::Result<()>
where
    F: num_traits::Float + std::fmt::Debug,
    S1: ndarray::Data<Elem = F>,
    S2: ndarray::Data<Elem = F>,
    D1: ndarray::Dimension,
    D2: ndarray::Dimension,
{
    for val in y_true.iter() {
        if *val <= F::zero() {
            return Err(crate::error::MetricsError::InvalidInput(
                "y_true contains non-positive values".to_string(),
            ));
        }
    }

    for val in y_pred.iter() {
        if *val <= F::zero() {
            return Err(crate::error::MetricsError::InvalidInput(
                "y_pred contains non-positive values".to_string(),
            ));
        }
    }

    Ok(())
}

/// Calculate the mean of an array
pub(crate) fn mean<F, S, D>(arr: &ArrayBase<S, D>) -> F
where
    F: num_traits::Float,
    S: ndarray::Data<Elem = F>,
    D: ndarray::Dimension,
{
    let sum = arr.iter().fold(F::zero(), |acc, &x| acc + x);
    sum / num_traits::NumCast::from(arr.len()).unwrap()
}
