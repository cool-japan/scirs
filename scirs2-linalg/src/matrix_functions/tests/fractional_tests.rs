//! Tests for fractional matrix functions

use super::super::fractional::*;
use ndarray::array;

#[test]
fn test_fractional_power_half() {
    let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
    let result = fractionalmatrix_power(&a.view(), 0.5, "eigen").unwrap();

    assert!((result[[0, 0]] - 2.0).abs() < 1e-10);
    assert!((result[[1, 1]] - 3.0).abs() < 1e-10);
}

#[test]
fn test_spd_matrix_function_sqrt() {
    let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
    let result = spdmatrix_function(&a.view(), |x| x.sqrt(), true).unwrap();

    assert!((result[[0, 0]] - 2.0).abs() < 1e-10);
    assert!((result[[1, 1]] - 3.0).abs() < 1e-10);
}
