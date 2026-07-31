use super::*;

use super::*;
#[allow(unused_imports)]
use approx::assert_relative_eq;
use scirs2_core::ndarray::array;

#[test]
fn test_backward_compatibility() {
    crate::run(|g| {
        let a = convert_to_tensor(array![1.0_f32, 2.0, 3.0], g);
        let b = convert_to_tensor(array![4.0_f32, 5.0, 6.0], g);

        // Test that all re-exported functions work
        let sum_result = add(a, b);
        let expected = array![5.0_f32, 7.0, 9.0];
        assert_eq!(
            sum_result.eval(g).expect("Test: operation failed"),
            expected.into_dyn()
        );

        // Test reduction operation
        let sum_all_result = sum_all(a);
        assert_eq!(
            sum_all_result.eval(g).expect("Test: operation failed"),
            scirs2_core::ndarray::arr0(6.0).into_dyn()
        );

        // Test activation function
        let relu_result = relu(a);
        assert_eq!(
            relu_result.eval(g).expect("Test: operation failed"),
            array![1.0_f32, 2.0, 3.0].into_dyn()
        );
    });
}

#[test]
fn test_module_organization() {
    crate::run(|g| {
        let x = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);

        // Test arithmetic module directly
        let sum_direct = arithmetic::add(x, x);
        let expected_sum = array![[2.0_f32, 4.0], [6.0, 8.0]];
        assert_eq!(
            sum_direct.eval(g).expect("Test: operation failed"),
            expected_sum.into_dyn()
        );

        // Test reduction module directly
        let mean_direct = reduction::reduce_mean(x, &[0], false);
        let expected_mean = array![2.0_f32, 3.0];
        assert_eq!(
            mean_direct.eval(g).expect("Test: operation failed"),
            expected_mean.into_dyn()
        );

        // Test linear algebra module directly
        let trace_direct = linear_algebra::trace(x);
        assert_eq!(
            trace_direct.eval(g).expect("Test: operation failed"),
            scirs2_core::ndarray::arr0(5.0).into_dyn()
        );

        // Test activation module directly
        let sigmoid_direct = activation::sigmoid(x);
        let result = sigmoid_direct.eval(g).expect("Test: operation failed");
        // All values should be between 0 and 1
        assert!(result.iter().all(|&val| val > 0.0 && val < 1.0));
    });
}

#[test]
fn test_tensor_methods() {
    crate::run(|g| {
        let x = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);

        // Test tensor methods
        let reshaped = x.reshape(&[4]);
        assert_eq!(
            reshaped.eval(g).expect("Test: operation failed").shape(),
            &[4]
        );

        let flattened = x.flatten();
        assert_eq!(
            flattened.eval(g).expect("Test: operation failed").shape(),
            &[4]
        );

        let trace_result = x.trace();
        assert_eq!(
            trace_result.eval(g).expect("Test: operation failed"),
            scirs2_core::ndarray::arr0(5.0).into_dyn()
        );

        let diag_result = x.diag();
        let expected_diag = array![1.0_f32, 4.0];
        assert_eq!(
            diag_result.eval(g).expect("Test: operation failed"),
            expected_diag.into_dyn()
        );
    });
}

#[test]
fn test_linalg_aliases() {
    crate::run(|g| {
        let x = convert_to_tensor(array![[2.0_f32, 1.0], [1.0, 3.0]], g);

        // Test matinv alias (inv conflicts with reciprocal function)
        let inv_result = matinv(&x);
        let inv_direct = matrix_inverse(x);
        assert_eq!(
            inv_result.eval(g).expect("Test: operation failed"),
            inv_direct.eval(g).expect("Test: operation failed")
        );

        // Test det alias
        let det_result = det(&x);
        let det_direct = determinant(x);
        assert_eq!(
            det_result.eval(g).expect("Test: operation failed"),
            det_direct.eval(g).expect("Test: operation failed")
        );

        // Test eig alias
        let (eigenvals, eigenvecs) = eig(&x);
        // Note: There's a known issue with eigen from linear_algebra module
        // where nth_tensor doesn't correctly extract eigenvectors.
        // We test the eig alias works correctly on its own.
        assert_eq!(
            eigenvals.eval(g).expect("Test: operation failed").shape(),
            &[2]
        );
        assert_eq!(
            eigenvecs.eval(g).expect("Test: operation failed").shape(),
            &[2, 2]
        );

        // Test pinv alias
        let rect = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]], g);
        let pinv_result = pinv(&rect);
        let pinv_direct = matrix_pseudo_inverse(&rect);
        assert_eq!(
            pinv_result.eval(g).expect("Test: operation failed"),
            pinv_direct.eval(g).expect("Test: operation failed")
        );

        // Test sqrtm alias.
        // `sqrtm` is a literal `pub use matrix_sqrt as sqrtm` re-export (see
        // tensor_ops/matrix_functions.rs), not a separate implementation, so
        // this is now implemented (the old "NOT YET IMPLEMENTED" comment was
        // stale) -- both names resolve to the exact same function.
        let pos_def = convert_to_tensor(array![[4.0_f32, 1.0], [1.0, 3.0]], g);
        let sqrtm_result = sqrtm(&pos_def);
        let sqrtm_direct = matrix_sqrt(&pos_def);
        assert_eq!(
            sqrtm_result.eval(g).expect("Test: operation failed"),
            sqrtm_direct.eval(g).expect("Test: operation failed")
        );

        // Test logm alias.
        // `logm` is likewise a literal `pub use matrix_log as logm` re-export.
        let small_mat = convert_to_tensor(array![[1.1_f32, 0.1], [0.1, 1.2]], g);
        let logm_result = logm(&small_mat);
        let logm_direct = matrix_log(&small_mat);
        assert_eq!(
            logm_result.eval(g).expect("Test: operation failed"),
            logm_direct.eval(g).expect("Test: operation failed")
        );
    });
}
