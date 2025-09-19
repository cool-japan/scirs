//! Tests for analysis matrix functions

use super::super::analysis::*;
use ndarray::array;

#[test]
fn test_spectral_radius_diagonal() {
    let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
    let result = spectral_radius(&a.view(), None).unwrap();

    assert!((result - 3.0).abs() < 1e-10);
}

#[test]
fn test_nuclear_norm_diagonal() {
    let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
    let result = nuclear_norm(&a.view(), None).unwrap();

    assert!((result - 5.0).abs() < 1e-10);
}

#[test]
fn test_tikhonov_regularization() {
    let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
    let result = tikhonov_regularization(&a.view(), 0.1, true).unwrap();

    assert!((result[[0, 0]] - 1.1).abs() < 1e-10);
    assert!((result[[1, 1]] - 1.1).abs() < 1e-10);
}
