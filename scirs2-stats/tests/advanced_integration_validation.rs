//! Basic integration validation tests
//!
//! This module contains basic integration tests for core statistical functionality.

use scirs2_core::ndarray::Array1;
use scirs2_stats::{mean, std, var};

#[test]
fn test_basic_statistics_integration() {
    let data: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // Test mean
    let mean_result = mean(&data.view()).unwrap();
    assert!((mean_result - 3.0).abs() < 1e-10);

    // Test variance
    let var_result = var(&data.view(), 1, None).unwrap();
    assert!(var_result > 0.0);

    // Test standard deviation
    let std_result = std(&data.view(), 1, None).unwrap();
    assert!(std_result > 0.0);
    assert!((std_result.powi(2) - var_result).abs() < 1e-10);
}

#[test]
fn test_empty_data_handling() {
    let empty_data: Array1<f64> = Array1::from_vec(vec![]);

    // These should return errors for empty data
    assert!(mean(&empty_data.view()).is_err());
    assert!(var(&empty_data.view(), 1, None).is_err());
    assert!(std(&empty_data.view(), 1, None).is_err());
}

#[test]
fn test_single_elementdata() {
    let singledata: Array1<f64> = Array1::from_vec(vec![42.0]);

    // Mean should work with single element
    let mean_result = mean(&singledata.view()).unwrap();
    assert!((mean_result - 42.0).abs() < 1e-10);

    // Variance should return an error for single element (needs at least 2 elements)
    assert!(var(&singledata.view(), 1, None).is_err());
}
