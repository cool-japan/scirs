//! Tests for exponential matrix functions

use super::super::exponential::*;
use ndarray::array;

#[test]
fn test_expm_identity() {
    let identity = array![[1.0_f64, 0.0], [0.0, 1.0]];
    let zero = array![[0.0_f64, 0.0], [0.0, 0.0]];
    let result = expm(&zero.view(), None).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            assert!((result[[i, j]] - identity[[i, j]]).abs() < 1e-10);
        }
    }
}

#[test]
fn test_logm_identity() {
    let identity = array![[1.0_f64, 0.0], [0.0, 1.0]];
    let result = logm(&identity.view()).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            assert!(result[[i, j]].abs() < 1e-10);
        }
    }
}

#[test]
fn test_sqrtm_diagonal() {
    let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
    let result = sqrtm(&a.view(), 20, 1e-10).unwrap();

    assert!((result[[0, 0]] - 2.0).abs() < 1e-10);
    assert!((result[[0, 1]] - 0.0).abs() < 1e-10);
    assert!((result[[1, 0]] - 0.0).abs() < 1e-10);
    assert!((result[[1, 1]] - 3.0).abs() < 1e-10);
}

#[test]
fn test_matrix_power_integer() {
    let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
    let result = matrix_power(&a.view(), 2.0).unwrap();

    assert!((result[[0, 0]] - 4.0).abs() < 1e-10);
    assert!((result[[0, 1]] - 0.0).abs() < 1e-10);
    assert!((result[[1, 0]] - 0.0).abs() < 1e-10);
    assert!((result[[1, 1]] - 9.0).abs() < 1e-10);
}
