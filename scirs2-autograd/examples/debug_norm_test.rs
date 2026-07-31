use ag::tensor_ops as T;
use scirs2_autograd as ag;
use scirs2_core::ndarray::array;

#[allow(dead_code)]
fn main() {
    ag::run(|ctx| {
        println!("=== Debug Norm Gradient Test ===");

        // Test with a simple 2x2 matrix. Differentiated below via T::grad, so
        // it must be `T::variable`: `T::convert_to_tensor` marks the node
        // non-differentiable, so the gradient computed and printed below
        // would silently be all-zero instead of `input/norm` -- exactly the
        // footgun this debug script exists to catch.
        let a = T::variable(array![[3.0, 4.0], [5.0, 12.0]], ctx);
        println!(
            "Input matrix A: {:?}",
            a.eval(ctx).expect("Operation failed")
        );

        // Test Frobenius norm computation
        let norm = T::frobenius_norm(a);
        let norm_result = norm.eval(ctx).expect("Operation failed");
        println!("Frobenius norm: {}", norm_result[[]]);

        // Expected: sqrt(3^2 + 4^2 + 5^2 + 12^2) = sqrt(194) ≈ 13.928
        let expected_norm = (194.0_f64).sqrt();
        println!("Expected norm: {}", expected_norm);
        assert!(
            (norm_result[[]] - expected_norm).abs() < 1e-9,
            "Frobenius norm mismatch: got {}, expected {}",
            norm_result[[]],
            expected_norm
        );

        // Test if the tensor is properly connected in the graph
        println!("Norm tensor id: {}", norm.id());
        println!("Input tensor id: {}", a.id());

        // Check the shape of norm (should be scalar)
        let normshape = norm.shape();
        println!("Norm shape: {:?}", normshape);

        // Try gradient computation
        println!("Computing gradient...");
        let grad_tensors = T::grad(&[norm], &[&a]);
        let grad = grad_tensors[0];

        println!("Gradient tensor id: {}", grad.id());
        // Skip checking private graph field

        // Evaluate gradient
        println!("Evaluating gradient...");
        let grad_result = grad.eval(ctx).expect("Operation failed");
        println!("Gradient result: {:?}", grad_result);

        // Expected gradient: input / norm
        let input_array = a.eval(ctx).expect("Operation failed");
        let expected_grad = input_array.mapv(|x| x / expected_norm);
        println!("Expected gradient: {:?}", expected_grad);

        // This is the actual point of the script: verify the printed
        // gradient really is `input / norm`, not the all-zero fallback that a
        // non-differentiable input would silently produce.
        for (got, exp) in grad_result.iter().zip(expected_grad.iter()) {
            assert!(
                (got - exp).abs() < 1e-9,
                "gradient element mismatch: got {}, expected {}",
                got,
                exp
            );
        }
        println!("Gradient matches input/norm as expected.");
    });
}
