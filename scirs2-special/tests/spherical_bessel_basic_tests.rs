//! Basic tests for spherical Bessel functions
//!
//! These tests verify the basic functionality and series expansion accuracy
//! of the spherical Bessel functions implementation.

use approx::assert_relative_eq;
use scirs2_special::bessel::spherical::spherical_jn;
use std::f64;

#[test]
fn test_small_arguments_series() {
    // Test series expansions for small x
    let small_x: f64 = 0.01;

    // For j₀(x) ≈ 1 - x²/6 + x⁴/120
    let j0_series: f64 = 1.0 - small_x * small_x / 6.0 + small_x.powi(4) / 120.0;
    let j0: f64 = spherical_jn(0, small_x);
    assert_relative_eq!(j0, j0_series, epsilon = 1e-12);

    // For j₁(x) ≈ x/3 - x³/30
    let j1_series: f64 = small_x / 3.0 - small_x.powi(3) / 30.0;
    let j1: f64 = spherical_jn(1, small_x);
    assert_relative_eq!(j1, j1_series, epsilon = 1e-12);

    // For j₂(x) ≈ x²/15
    let j2_series: f64 = small_x.powi(2) / 15.0;
    let j2: f64 = spherical_jn(2, small_x);
    assert_relative_eq!(j2, j2_series, epsilon = 1e-10);
}
