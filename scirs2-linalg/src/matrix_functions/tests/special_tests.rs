//! Tests for special matrix functions

use super::super::special::*;
use scirs2_core::ndarray::array;

#[test]
fn test_sigmoid_basic() {
    let a = array![[0.0_f64, 1.0], [-1.0, 2.0]];
    let result = sigmoid(&a.view()).unwrap();

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
    let result = softmax(&a.view(), Some(1)).unwrap();

    // Each row should sum to 1
    for i in 0..2 {
        let row_sum: f64 = (0..2).map(|j| result[[i, j]]).sum();
        assert!((row_sum - 1.0).abs() < 1e-10);
    }
}
