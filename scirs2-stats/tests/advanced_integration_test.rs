//! Basic integration tests
//!
//! This module contains basic integration tests for statistical functionality.

use scirs2_core::ndarray::Array1;
use scirs2_stats::{mean, std, var};

#[test]
fn test_basic_integration() {
    let data: Array1<f64> =
        Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    // Test mean
    let mean_result = mean(&data.view()).unwrap();
    assert!((mean_result - 5.5).abs() < 1e-10);

    // Test variance
    let var_result = var(&data.view(), 1, None).unwrap();
    assert!(var_result > 0.0);

    // Test standard deviation
    let std_result = std(&data.view(), 1, None).unwrap();
    assert!(std_result > 0.0);
}

#[test]
fn test_largedata_integration() {
    let largedata: Array1<f64> = Array1::from_iter((1..=1000).map(|x| x as f64));

    // Test that operations complete successfully on larger datasets
    let mean_result = mean(&largedata.view()).unwrap();
    assert!((mean_result - 500.5).abs() < 1e-10);

    let var_result = var(&largedata.view(), 1, None).unwrap();
    assert!(var_result > 0.0);

    let std_result = std(&largedata.view(), 1, None).unwrap();
    assert!(std_result > 0.0);
}

#[test]
fn test_consistency() {
    let data: Array1<f64> = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    // Test that variance and std are consistent
    let var_result = var(&data.view(), 1, None).unwrap();
    let std_result = std(&data.view(), 1, None).unwrap();

    assert!((std_result.powi(2) - var_result).abs() < 1e-10);
}
