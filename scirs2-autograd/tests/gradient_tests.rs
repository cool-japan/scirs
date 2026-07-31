#[cfg(test)]
mod tests {
    use ag::tensor_ops as T;
    use scirs2_autograd as ag;

    #[test]
    fn test_scalar_gradient() {
        // Create a simple computation: y = x + 1
        // This is a very basic test given our temporary gradient implementation
        ag::run(|g| {
            // Create a tensor with a scalar value
            let x = T::scalar(3.0, g);

            // Compute y = x + 1
            let y = x + T::scalar(1.0, g);

            // Compute gradient dy/dx
            let gradients = T::grad(&[y], &[x]);

            // Just check that the gradient exists and can be evaluated
            assert!(!gradients.is_empty(), "Should have at least one gradient");
            let result = gradients[0].eval(g);
            assert!(result.is_ok(), "Gradient should be evaluable");
        });
    }

    #[test]
    fn test_basic_gradientshape() {
        // Test that the gradient has both the expected shape AND the expected
        // values.
        ag::run(|g| {
            // Create a tensor with shape [2, 2]. This is differentiated below
            // via T::grad, so it must be `T::variable`: `T::convert_to_tensor`
            // marks the node non-differentiable, so the gradient below would
            // silently be the exact-zero fallback regardless of whether
            // `add`'s backward pass works at all.
            let data = scirs2_core::ndarray::array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
            let x = T::variable(data.clone(), g);

            // Compute y = add(x, scalar)
            let y = x + T::scalar(1.0, g);

            // Compute gradient: d(x+1)/dx = 1 everywhere.
            let gradients = T::grad(&[y], &[x]);

            assert_eq!(gradients.len(), 1, "Should have exactly one gradient");
            let result = gradients[0].eval(g).expect("Gradient should be evaluable");

            assert_eq!(
                result.shape(),
                [2, 2],
                "Gradient shape should match input shape"
            );
            for &v in result.iter() {
                assert_eq!(v, 1.0, "d(x+1)/dx must be 1 everywhere, got {v}");
            }
        });
    }
}
