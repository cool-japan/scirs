//! Test to understand method resolution conflict between ndarray's std and our StatsExt::std

use crate::utils::scaling::StatsExt;
use ndarray::{Array1, ArrayView1};

#[allow(dead_code)]
pub fn test_method_resolution() {
    let data: Array1<f64> = Array1::from(vec![]);
    let view: ArrayView1<f64> = data.view();

    println!("Empty array view.len(): {}", view.len());

    // This will call ndarray's built-in std method
    println!("Calling view.std(0.0) - this calls ndarray's built-in method:");
    let builtin_std = view.std(0.0);
    println!("Built-in std result: {:?}", builtin_std);

    // This will call our trait method explicitly
    println!("Calling StatsExt::std explicitly:");
    let trait_std = StatsExt::std(&view, 0.0);
    println!("Trait std result: {:?}", trait_std);

    println!("Conflict confirmed: ndarray's std() takes precedence over our trait method!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confirm_method_conflict() {
        test_method_resolution();

        // Verify the conflict
        let data: Array1<f64> = Array1::from(vec![]);
        let view = data.view();

        // ndarray's std returns NaN for empty arrays
        let ndarray_result = view.std(0.0);
        assert!(ndarray_result.is_nan());

        // Our trait method returns 0.0 for empty arrays
        let our_result = StatsExt::std(&view, 0.0);
        assert_eq!(our_result, 0.0);

        println!("Method resolution conflict confirmed!");
        println!("ndarray std result: {}", ndarray_result);
        println!("Our StatsExt std result: {}", our_result);
    }
}
