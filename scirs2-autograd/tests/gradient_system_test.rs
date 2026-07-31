#[cfg(test)]
mod tests {
    use ag::tensor_ops as T;
    use scirs2_autograd as ag;

    #[test]
    fn test_proper_gradient_computation() {
        // Create a simple computation: y = x^2
        // The gradient should be dy/dx = 2x
        ag::run(|g| {
            // Create a tensor for x with value 3.0
            let x = T::scalar(3.0_f64, g);

            // Compute y = x^2
            let y = T::pow(x, 2.0);

            // Compute gradient dy/dx
            let gradients = T::grad(&[y], &[x]);

            // The gradient should now be computed properly
            assert!(gradients[0].eval(g).is_ok(), "Gradient should be evaluable");

            // With the improved gradient system, we can get the actual gradient value
            // which should be 2x = 2*3 = 6.0
            let result = gradients[0].eval(g).expect("Test: operation failed");
            assert!(
                (result[[]] - 6.0).abs() < 1e-5,
                "Gradient should be 6.0, but got {}",
                result[[]]
            );
        });
    }

    #[test]
    fn test_matrix_gradient() {
        // Create a simple matrix and test its gradient
        ag::run(|g| {
            // Create a matrix [[1, 2], [3, 4]]
            //
            // NOTE: must use `variable()`, not `convert_to_tensor()`, here.
            // `convert_to_tensor` explicitly calls `.set_differentiable(false)`
            // (it's meant for embedding non-differentiable constants), so a
            // tensor created that way is never on the backprop path -- T::grad
            // would silently fall back to a zero-filled "not differentiable"
            // gradient regardless of what operations are applied to it.
            // `variable()` is the dedicated constructor for a tensor you
            // actually want to differentiate with respect to.
            let data = scirs2_core::ndarray::array![[1.0_f64, 2.0], [3.0, 4.0]].into_dyn();
            let x = T::variable(data.clone(), g);

            // Compute sum of elements
            let y = T::sum_all(x);

            // Compute gradient
            let gradients = T::grad(&[y], &[x]);

            // The gradient should be evaluable
            assert!(gradients[0].eval(g).is_ok(), "Gradient should be evaluable");

            // With our improved system, operations like sum_all should eventually
            // return a gradient of ones with the same shape as the input x
            let result = gradients[0].eval(g).expect("Test: operation failed");
            for (idx, _) in data.indexed_iter() {
                let grad_value = result[idx.clone()];
                assert!(
                    (grad_value - 1.0).abs() < 1e-5,
                    "Gradient at {:?} should be 1.0, but got {}",
                    idx,
                    grad_value
                );
            }
        });
    }
}
