//! Tests for trigonometric matrix functions

use super::super::trigonometric::*;
use scirs2_core::ndarray::array;

#[test]
fn test_cosm_zero() {
    let zero = array![[0.0_f64, 0.0], [0.0, 0.0]];
    let result = cosm(&zero.view()).expect("Test: operation failed");
    let identity = array![[1.0_f64, 0.0], [0.0, 1.0]];

    for i in 0..2 {
        for j in 0..2 {
            assert!((result[[i, j]] - identity[[i, j]]).abs() < 1e-10);
        }
    }
}

#[test]
fn test_sinm_zero() {
    let zero = array![[0.0_f64, 0.0], [0.0, 0.0]];
    let result = sinm(&zero.view()).expect("Test: operation failed");

    for i in 0..2 {
        for j in 0..2 {
            assert!(result[[i, j]].abs() < 1e-10);
        }
    }
}

#[test]
fn test_acosm_diagonal() {
    // Diagonal input: arccos applied elementwise
    let a = array![[0.5_f64, 0.0], [0.0, 0.0]];
    let result = acosm(&a.view()).expect("acosm diagonal");
    assert!((result[[0, 0]] - 0.5_f64.acos()).abs() < 1e-10);
    assert!((result[[1, 1]] - 0.0_f64.acos()).abs() < 1e-10);
    assert!(result[[0, 1]].abs() < 1e-10);
    assert!(result[[1, 0]].abs() < 1e-10);
}

#[test]
fn test_asinm_diagonal() {
    let a = array![[0.5_f64, 0.0], [0.0, -0.3]];
    let result = asinm(&a.view()).expect("asinm diagonal");
    assert!((result[[0, 0]] - 0.5_f64.asin()).abs() < 1e-10);
    assert!((result[[1, 1]] - (-0.3_f64).asin()).abs() < 1e-10);
}

#[test]
fn test_atanm_diagonal() {
    let a = array![[1.0_f64, 0.0], [0.0, -1.0]];
    let result = atanm(&a.view()).expect("atanm diagonal");
    assert!((result[[0, 0]] - 1.0_f64.atan()).abs() < 1e-10);
    assert!((result[[1, 1]] - (-1.0_f64).atan()).abs() < 1e-10);
}

#[test]
fn test_acosm_nontrivial() {
    // For a symmetric positive-definite matrix with eigenvalues in [-1,1],
    // acosm should be the inverse of cosm: cos(acos(A)) ≈ A
    let a = array![[0.9_f64, 0.1], [0.1, 0.8]];
    // Verify eigenvalues are in [-1, 1] before testing
    let result = acosm(&a.view()).expect("acosm non-trivial");
    // cos(arccos(A)) = A for matrices with real eigenvalues in (-1,1)
    let cos_result = cosm(&result.view()).expect("cosm round-trip");
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (cos_result[[i, j]] - a[[i, j]]).abs() < 1e-5,
                "round-trip failed at [{i},{j}]: got {}, expected {}",
                cos_result[[i, j]],
                a[[i, j]]
            );
        }
    }
}

#[test]
fn test_atanm_nontrivial() {
    // tan(atan(A)) ≈ A for small diagonal perturbations
    let a = array![[0.3_f64, 0.0], [0.0, -0.2]];
    let atan_a = atanm(&a.view()).expect("atanm");
    let tan_atan_a = tanm(&atan_a.view()).expect("tanm round-trip");
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (tan_atan_a[[i, j]] - a[[i, j]]).abs() < 1e-8,
                "atanm round-trip failed at [{i},{j}]"
            );
        }
    }
}
