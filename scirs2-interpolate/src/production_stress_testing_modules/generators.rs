//! Test data generators for production stress testing
//!
//! This module contains functions to generate various types of test data including
//! pathological cases, edge cases, and stress test scenarios.

use crate::error::InterpolateResult;
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::Array1;

/// Generate large test data with sine wave pattern
pub fn create_large_test_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(T::zero(), T::from_f64(10.0).unwrap(), size);
    let y =
        x.mapv(|xi| xi.sin() + T::from_f64(0.1).unwrap() * (xi * T::from_f64(10.0).unwrap()).cos());
    Ok((x, y))
}

/// Create constant data (pathological case)
pub fn create_constant_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(T::zero(), T::from_f64(10.0).unwrap(), size);
    let y = Array1::from_elem(size, T::from_f64(5.0).unwrap());
    Ok((x, y))
}

/// Create data with duplicate x values (pathological case)
pub fn create_duplicate_x_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let mut x = Array1::linspace(T::zero(), T::from_f64(10.0).unwrap(), size);
    let y = Array1::ones(size);

    // Introduce duplicates
    if size > 10 {
        x[size / 2] = x[size / 2 - 1];
        x[size / 4] = x[size / 4 - 1];
    }

    Ok((x, y))
}

/// Create data with extreme y values
pub fn create_extreme_y_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(T::zero(), T::from_f64(10.0).unwrap(), size);
    let mut y = Array1::zeros(size);

    for (i, &_xi) in x.iter().enumerate() {
        y[i] = if i % 2 == 0 {
            T::from_f64(1e10).unwrap()
        } else {
            T::from_f64(-1e10).unwrap()
        };
    }

    Ok((x, y))
}

/// Create data with NaN and infinity values
pub fn create_nan_inf_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(T::zero(), T::from_f64(10.0).unwrap(), size);
    let mut y = x.mapv(|xi| xi.sin());

    // Introduce problematic values
    if size > 10 {
        y[size / 4] = T::infinity();
        y[size / 2] = T::neg_infinity();
    }

    Ok((x, y))
}

/// Create sparse data
pub fn create_sparse_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let sparse_size = (size / 100).max(3);
    let x = Array1::linspace(T::zero(), T::from_f64(1000.0).unwrap(), sparse_size);
    let y = x.mapv(|xi| xi.sin());
    Ok((x, y))
}

/// Create highly oscillatory data
pub fn create_oscillatory_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(T::zero(), T::from_f64(10.0).unwrap(), size);
    let y = x.mapv(|xi| (xi * T::from_f64(100.0).unwrap()).sin());
    Ok((x, y))
}

/// Create monotonic extreme data
pub fn create_monotonic_extreme_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(T::zero(), T::from_f64(10.0).unwrap(), size);
    let y = x.mapv(|xi| xi.powi(10));
    Ok((x, y))
}

/// Create edge case data with specific value range
pub fn create_edge_case_data<T: InterpolationFloat>(
    size: usize,
    min_val: f64,
    max_val: f64,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(
        T::from_f64(min_val).unwrap(),
        T::from_f64(max_val).unwrap(),
        size,
    );
    let y = x.mapv(|xi| xi * T::from_f64(1.1).unwrap());
    Ok((x, y))
}

/// Create linear data for baseline testing
pub fn create_linear_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(T::zero(), T::from_f64(10.0).unwrap(), size);
    let y = x.mapv(|xi| xi * T::from_f64(2.0).unwrap() + T::from_f64(3.0).unwrap());
    Ok((x, y))
}

/// Create quadratic data for testing
pub fn create_quadratic_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(T::from_f64(-5.0).unwrap(), T::from_f64(5.0).unwrap(), size);
    let y = x.mapv(|xi| xi * xi + T::from_f64(2.0).unwrap() * xi + T::from_f64(1.0).unwrap());
    Ok((x, y))
}

/// Create noisy data for testing
pub fn create_noisy_data<T: InterpolationFloat>(
    size: usize,
    noise_level: f64,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(T::zero(), T::from_f64(10.0).unwrap(), size);
    let mut y = x.mapv(|xi| xi.sin());

    // Add noise (simplified - real implementation would use proper random number generation)
    for (i, yi) in y.iter_mut().enumerate() {
        let noise = T::from_f64(noise_level * ((i as f64 * 12345.0).sin() * 0.1)).unwrap();
        *yi += noise;
    }

    Ok((x, y))
}

/// Create exponential data for testing
pub fn create_exponential_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(T::zero(), T::from_f64(5.0).unwrap(), size);
    let y = x.mapv(|xi| (xi * T::from_f64(0.5).unwrap()).exp());
    Ok((x, y))
}

/// Create step function data
pub fn create_step_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(T::zero(), T::from_f64(10.0).unwrap(), size);
    let y = x.mapv(|xi| {
        if xi < T::from_f64(5.0).unwrap() {
            T::zero()
        } else {
            T::one()
        }
    });
    Ok((x, y))
}

/// Create random-like data (deterministic for testing)
pub fn create_pseudo_random_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(T::zero(), T::from_f64(10.0).unwrap(), size);
    let y = x.mapv(|xi| {
        // Deterministic pseudo-random using sine functions
        let val = xi.to_f64().unwrap();
        T::from_f64((val * 12.34).sin() + (val * 56.78).cos() * 0.3).unwrap()
    });
    Ok((x, y))
}

/// Create data with rapid changes
pub fn create_rapid_change_data<T: InterpolationFloat>(
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    let x = Array1::linspace(T::zero(), T::from_f64(10.0).unwrap(), size);
    let y = x.mapv(|xi| {
        let val = xi.to_f64().unwrap();
        if val > 4.9 && val < 5.1 {
            T::from_f64(100.0).unwrap() // Rapid change around x=5
        } else {
            xi
        }
    });
    Ok((x, y))
}

/// Create empty data for error testing
pub fn create_empty_data<T: InterpolationFloat>() -> InterpolateResult<(Array1<T>, Array1<T>)> {
    Ok((Array1::<T>::zeros(0), Array1::<T>::zeros(0)))
}

/// Create single point data for error testing
pub fn create_single_point_data<T: InterpolationFloat>() -> InterpolateResult<(Array1<T>, Array1<T>)>
{
    Ok((
        Array1::from_vec(vec![T::zero()]),
        Array1::from_vec(vec![T::one()]),
    ))
}

/// Create mismatched length data for error testing
pub fn create_mismatched_data<T: InterpolationFloat>() -> InterpolateResult<(Array1<T>, Array1<T>)>
{
    Ok((Array1::<T>::zeros(10), Array1::<T>::zeros(5)))
}

/// Create unsorted x data for error testing
pub fn create_unsorted_x_data<T: InterpolationFloat>() -> InterpolateResult<(Array1<T>, Array1<T>)>
{
    let x = Array1::from_vec(vec![
        T::from_f64(1.0).unwrap(),
        T::from_f64(3.0).unwrap(),
        T::from_f64(2.0).unwrap(),
        T::from_f64(4.0).unwrap(),
    ]);
    let y = Array1::ones(4);
    Ok((x, y))
}

/// Generate test data by name
pub fn generate_test_data<T: InterpolationFloat>(
    data_type: &str,
    size: usize,
) -> InterpolateResult<(Array1<T>, Array1<T>)> {
    match data_type {
        "large" => create_large_test_data(size),
        "constant" => create_constant_data(size),
        "duplicate_x" => create_duplicate_x_data(size),
        "extreme_y" => create_extreme_y_data(size),
        "nan_inf" => create_nan_inf_data(size),
        "sparse" => create_sparse_data(size),
        "oscillatory" => create_oscillatory_data(size),
        "monotonic_extreme" => create_monotonic_extreme_data(size),
        "linear" => create_linear_data(size),
        "quadratic" => create_quadratic_data(size),
        "exponential" => create_exponential_data(size),
        "step" => create_step_data(size),
        "pseudo_random" => create_pseudo_random_data(size),
        "rapid_change" => create_rapid_change_data(size),
        "empty" => create_empty_data(),
        "single_point" => create_single_point_data(),
        "mismatched" => create_mismatched_data(),
        "unsorted_x" => create_unsorted_x_data(),
        "noisy_low" => create_noisy_data(size, 0.01),
        "noisy_medium" => create_noisy_data(size, 0.1),
        "noisy_high" => create_noisy_data(size, 0.5),
        _ => create_large_test_data(size), // Default fallback
    }
}

/// Get all available pathological data types
pub fn get_pathological_data_types() -> Vec<&'static str> {
    vec![
        "constant",
        "duplicate_x",
        "extreme_y",
        "nan_inf",
        "sparse",
        "oscillatory",
        "monotonic_extreme",
    ]
}

/// Get all available error test data types
pub fn get_error_test_data_types() -> Vec<&'static str> {
    vec!["empty", "single_point", "mismatched", "unsorted_x"]
}

/// Get all available general test data types
pub fn get_general_test_data_types() -> Vec<&'static str> {
    vec![
        "large",
        "linear",
        "quadratic",
        "exponential",
        "step",
        "pseudo_random",
        "rapid_change",
        "noisy_low",
        "noisy_medium",
        "noisy_high",
    ]
}
