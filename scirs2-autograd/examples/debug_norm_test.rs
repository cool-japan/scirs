use ag::tensor_ops as T;
use ndarray::array;
use scirs2_autograd as ag;

fn main() {
    ag::run::<f64, _, _>(|ctx| {
        println!("=== Debug Norm Gradient Test ===");

        // Test with a simple 2x2 matrix
        let a = T::convert_to_tensor(array![[3.0, 4.0], [5.0, 12.0]], ctx);
        println!("Input matrix A: {:?}", a.eval(ctx).unwrap());

        // Test Frobenius norm computation
        let norm = T::frobenius_norm(a);
        let norm_result = norm.eval(ctx).unwrap();
        println!("Frobenius norm: {}", norm_result[[]]);

        // Expected: sqrt(3^2 + 4^2 + 5^2 + 12^2) = sqrt(194) ≈ 13.928
        let expected_norm = (194.0_f64).sqrt();
        println!("Expected norm: {}", expected_norm);

        // Test if the tensor is properly connected in the graph
        println!("Norm tensor id: {}", norm.id());
        println!("Input tensor id: {}", a.id());

        // Check the shape of norm (should be scalar)
        let norm_shape = norm.shape();
        println!("Norm shape: {:?}", norm_shape);

        // Try gradient computation
        println!("Computing gradient...");
        let grad_tensors = T::grad(&[norm], &[&a]);
        let grad = grad_tensors[0];

        println!("Gradient tensor id: {}", grad.id());
        // Skip checking private graph field

        // Evaluate gradient
        println!("Evaluating gradient...");
        let grad_result = grad.eval(ctx).unwrap();
        println!("Gradient result: {:?}", grad_result);

        // Expected gradient: input / norm
        let input_array = a.eval(ctx).unwrap();
        let expected_grad = input_array.mapv(|x| x / expected_norm);
        println!("Expected gradient: {:?}", expected_grad);
    });
}
