//! Test to demonstrate the resolution of method conflicts between ndarray's std and our StatsExt::standard_deviation

use crate::utils::scaling::StatsExt;
use scirs2_core::ndarray::{Array1, ArrayView1};

#[allow(dead_code)]
pub fn test_method_resolution() {
    let data: Array1<f64> = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let view: ArrayView1<f64> = data.view();

    println!("Array with data: {data:?}");

    // This calls ndarray's built-in std method
    println!("Calling view.std(0.0) - this calls ndarray's built-in method:");
    let builtin_std = view.std(0.0);
    println!("Built-in std result: {builtin_std:?}");

    // This calls our trait method - no longer conflicts!
    println!("Calling view.standard_deviation(0.0) - this calls our trait method:");
    let trait_std = view.standard_deviation(0.0);
    println!("Trait standard_deviation result: {trait_std:?}");

    println!("Conflict resolved: Both methods are now accessible without ambiguity!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confirm_method_conflict() {
        test_method_resolution();

        // Verify the resolution - both methods work without conflict
        let data: Array1<f64> = Array1::from(vec![]);
        let view = data.view();

        // ndarray's std returns NaN for empty arrays
        let ndarray_result = view.std(0.0);
        assert!(ndarray_result.is_nan());

        // Our trait method returns 0.0 for empty arrays - no longer conflicts!
        let our_result = view.standard_deviation(0.0);
        assert_eq!(our_result, 0.0);

        println!("Method resolution conflict resolved!");
        println!("ndarray std result: {ndarray_result}");
        println!("Our StatsExt standard_deviation result: {our_result}");

        // Test with actual data to show both methods work
        let data_with_values: Array1<f64> = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let view_with_values = data_with_values.view();

        let ndarray_std = view_with_values.std(0.0);
        let our_std = view_with_values.standard_deviation(0.0);

        // Both should give the same result (approximately)
        assert!((ndarray_std - our_std).abs() < 1e-10);
        println!("Both methods give consistent results: {ndarray_std} ≈ {our_std}");
    }
}
