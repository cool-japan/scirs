use ag::tensor_ops as T;
use ndarray::array;
use scirs2_autograd as ag;

#[test]
fn debug_gradient_system() {
    ag::run::<f64, _, _>(|ctx| {
        println!("=== Debug Gradient System Test ===");

        // Test with a simple 2x2 matrix
        let a = T::convert_to_tensor(array![[3.0, 4.0], [5.0, 12.0]], ctx);
        println!("Input matrix A: {:?}", a.eval(ctx).unwrap());

        // First try our new debug operators that should definitely work
        let debug_op = T::debug_identity_with_gradient(&a);
        let debug_result = debug_op.eval(ctx).unwrap();
        println!(
            "Debug op result (should be same as input): {:?}",
            debug_result
        );

        println!("Computing gradient of debug op...");
        let debug_grad = T::grad(&[debug_op], &[&a])[0];
        let debug_grad_result = debug_grad.eval(ctx).unwrap();
        println!("Debug op gradient result: {:?}", debug_grad_result);

        // Check if the debug gradient works
        let debug_is_all_zeros = debug_grad_result.iter().all(|&x| x == 0.0);
        println!("Is debug gradient all zeros? {}", debug_is_all_zeros);

        if debug_is_all_zeros {
            println!("ERROR: Debug gradient is also zeros - the gradient system itself is not working properly");
        } else {
            println!("SUCCESS: Debug gradient is not zeros - the debug op's gradient system works");
        }

        // Now test with the scalar one op - should return a scalar 1 and gradient of all 1s
        let scalar_one = T::debug_scalar_one(&a);
        let scalar_one_result = scalar_one.eval(ctx).unwrap();
        println!(
            "Scalar one op result (should be 1.0): {:?}",
            scalar_one_result
        );

        println!("Computing gradient of scalar one op...");
        let scalar_one_grad = T::grad(&[scalar_one], &[&a])[0];
        let scalar_one_grad_result = scalar_one_grad.eval(ctx).unwrap();
        println!(
            "Scalar one gradient result (should be all 1s): {:?}",
            scalar_one_grad_result
        );

        // Check the scalar one gradient
        let scalar_one_is_all_zeros = scalar_one_grad_result.iter().all(|&x| x == 0.0);
        println!(
            "Is scalar one gradient all zeros? {}",
            scalar_one_is_all_zeros
        );
        let scalar_one_is_all_ones = scalar_one_grad_result.iter().all(|&x| x == 1.0);
        println!(
            "Is scalar one gradient all ones? {}",
            scalar_one_is_all_ones
        );

        // Finally try the norm gradient
        println!("Testing norm gradient...");
        let norm = T::frobenius_norm(&a);
        let norm_result = norm.eval(ctx).unwrap();
        println!("Frobenius norm: {}", norm_result[[]]);

        println!("Computing norm gradient...");
        let norm_grad = T::grad(&[norm], &[&a])[0];
        let norm_grad_result = norm_grad.eval(ctx).unwrap();
        println!("Norm gradient result: {:?}", norm_grad_result);

        // Check the norm gradient
        let norm_is_all_zeros = norm_grad_result.iter().all(|&x| x == 0.0);
        println!("Is norm gradient all zeros? {}", norm_is_all_zeros);

        if norm_is_all_zeros && !debug_is_all_zeros {
            println!("CONCLUSION: The gradient system works but there's a specific issue with the norm op's gradient");
        } else if norm_is_all_zeros && debug_is_all_zeros {
            println!("CONCLUSION: The entire gradient system is not working properly");
        } else {
            println!("CONCLUSION: The norm gradient is actually working!");
        }
    });
}
