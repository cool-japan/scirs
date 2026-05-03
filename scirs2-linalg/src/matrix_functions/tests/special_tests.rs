//! Tests for special matrix functions

use super::super::special::*;
use scirs2_core::ndarray::array;

#[test]
fn test_sigmoid_basic() {
    let a = array![[0.0_f64, 1.0], [-1.0, 2.0]];
    let result = sigmoid(&a.view()).expect("Test: operation failed");

    // sigmoid(0) should be 0.5
    assert!((result[[0, 0]] - 0.5).abs() < 1e-10);
    // sigmoid should be between 0 and 1
    for i in 0..2 {
        for j in 0..2 {
            assert!(result[[i, j]] > 0.0 && result[[i, j]] < 1.0);
        }
    }
}

#[test]
fn test_softmax_basic() {
    let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let result = softmax(&a.view(), Some(1)).expect("Test: operation failed");

    // Each row should sum to 1
    for i in 0..2 {
        let row_sum: f64 = (0..2).map(|j| result[[i, j]]).sum();
        assert!((row_sum - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_signm_diagonal() {
    // Diagonal matrix: sign function applied elementwise
    let a = array![[2.0_f64, 0.0], [0.0, -3.0]];
    let s = signm(&a.view()).expect("signm diagonal");
    assert!((s[[0, 0]] - 1.0).abs() < 1e-10);
    assert!((s[[1, 1]] + 1.0).abs() < 1e-10);
    assert!(s[[0, 1]].abs() < 1e-10);
    assert!(s[[1, 0]].abs() < 1e-10);
}

#[test]
fn test_signm_general_2x2() {
    // A = [[3, 1], [0, 2]] has eigenvalues 3 and 2, both positive → sign(A) = I
    let a = array![[3.0_f64, 1.0], [0.0, 2.0]];
    let s = signm(&a.view()).expect("signm general");
    // For a matrix with all positive eigenvalues, sign(A) = I
    assert!((s[[0, 0]] - 1.0).abs() < 1e-6, "sign_00={}", s[[0, 0]]);
    assert!((s[[1, 1]] - 1.0).abs() < 1e-6, "sign_11={}", s[[1, 1]]);
    assert!(s[[1, 0]].abs() < 1e-6, "sign_10={}", s[[1, 0]]);
}

#[test]
fn test_signm_converges_to_identity_squared() {
    // Property: sign(A)^2 = I for matrices with no purely imaginary eigenvalues
    let a = array![[2.0_f64, 1.0], [-1.0, 3.0]];
    let s = signm(&a.view()).expect("signm");
    let s2 = s.dot(&s);
    assert!((s2[[0, 0]] - 1.0).abs() < 1e-5, "s2_00={}", s2[[0, 0]]);
    assert!((s2[[1, 1]] - 1.0).abs() < 1e-5, "s2_11={}", s2[[1, 1]]);
    assert!(s2[[0, 1]].abs() < 1e-5, "s2_01={}", s2[[0, 1]]);
    assert!(s2[[1, 0]].abs() < 1e-5, "s2_10={}", s2[[1, 0]]);
}
