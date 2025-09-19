//! Tests for trigonometric matrix functions

use super::super::trigonometric::*;
use ndarray::array;

#[test]
fn test_cosm_zero() {
    let zero = array![[0.0_f64, 0.0], [0.0, 0.0]];
    let result = cosm(&zero.view()).unwrap();
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
    let result = sinm(&zero.view()).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            assert!(result[[i, j]].abs() < 1e-10);
        }
    }
}
