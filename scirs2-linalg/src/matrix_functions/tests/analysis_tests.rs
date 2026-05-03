//! Tests for analysis matrix functions

use super::super::analysis::*;
use scirs2_core::ndarray::array;

#[test]
fn test_spectral_radius_diagonal() {
    let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
    let result = spectral_radius(&a.view(), None).expect("Test: operation failed");

    assert!((result - 3.0).abs() < 1e-10);
}

#[test]
fn test_nuclear_norm_diagonal() {
    let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
    let result = nuclear_norm(&a.view(), None).expect("Test: operation failed");

    assert!((result - 5.0).abs() < 1e-10);
}

#[test]
fn test_tikhonov_regularization() {
    let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
    let result = tikhonov_regularization(&a.view(), 0.1, true).expect("Test: operation failed");

    assert!((result[[0, 0]] - 1.1).abs() < 1e-10);
    assert!((result[[1, 1]] - 1.1).abs() < 1e-10);
}

#[test]
fn test_nuclear_norm_nonsquare() {
    // For a 2x3 matrix with known singular values, the nuclear norm is their sum.
    // A = [[3, 0, 0], [0, 2, 0]] has singular values 3 and 2, nuclear norm = 5.
    let a = array![[3.0_f64, 0.0, 0.0], [0.0, 2.0, 0.0]];
    let result = nuclear_norm(&a.view(), None).expect("nuclear_norm non-square");
    assert!((result - 5.0).abs() < 1e-8, "expected 5.0, got {}", result);
}

#[test]
fn test_nuclear_norm_rectangular_general() {
    // 3x2 matrix with orthonormal columns: singular values are sqrt(sums of squared column norms)
    // A = [[1, 0], [0, 1], [0, 0]]. Singular values = [1, 1], nuclear norm = 2.
    let a = array![[1.0_f64, 0.0], [0.0, 1.0], [0.0, 0.0]];
    let result = nuclear_norm(&a.view(), None).expect("nuclear_norm 3x2");
    assert!((result - 2.0).abs() < 1e-8, "expected 2.0, got {}", result);
}
