use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

fn main() {
    println!("Testing matrix norm gradient fixes...");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Test Frobenius norm gradient
        println!("\n=== Testing Frobenius Norm Gradient ===");
        let matrix = convert_to_tensor(array![[3.0, 4.0], [0.0, 0.0]].into_dyn(), ctx);
        let norm = frobenius_norm(matrix);

        // Expected norm: sqrt(3^2 + 4^2) = 5.0
        let norm_result = norm.eval(ctx).unwrap();
        println!("Frobenius norm result: {}", norm_result[[]]);
        assert!(
            (norm_result[[]] - 5.0).abs() < 1e-6,
            "Frobenius norm calculation failed"
        );

        // Test gradient
        let grads = grad(&[norm], &[matrix]);
        let grad_tensor = &grads[0];
        let grad_result = grad_tensor.eval(ctx).unwrap();
        // Expected gradient: [3/5, 4/5; 0, 0] = [0.6, 0.8; 0, 0]
        println!("Gradient: {:.6}", grad_result);
        assert!(
            (grad_result[[0, 0]] - 0.6).abs() < 1e-6,
            "Frobenius gradient [0,0] failed"
        );
        assert!(
            (grad_result[[0, 1]] - 0.8).abs() < 1e-6,
            "Frobenius gradient [0,1] failed"
        );
        assert!(
            (grad_result[[1, 0]] - 0.0).abs() < 1e-6,
            "Frobenius gradient [1,0] failed"
        );
        assert!(
            (grad_result[[1, 1]] - 0.0).abs() < 1e-6,
            "Frobenius gradient [1,1] failed"
        );
        println!("✓ Frobenius norm gradient test passed!");

        // Test spectral norm on diagonal matrix
        println!("\n=== Testing Spectral Norm (Diagonal Matrix) ===");
        let diag_matrix = convert_to_tensor(array![[5.0, 0.0], [0.0, 3.0]].into_dyn(), ctx);
        let spec_norm = spectral_norm(&diag_matrix);

        // Expected spectral norm: max(5, 3) = 5.0
        let spec_result = spec_norm.eval(ctx).unwrap();
        println!("Spectral norm result: {}", spec_result[[]]);
        assert!(
            (spec_result[[]] - 5.0).abs() < 1e-3,
            "Spectral norm calculation failed"
        );

        // Test gradient
        let spec_grads = grad(&[spec_norm], &[diag_matrix]);
        let grad_tensor = &spec_grads[0];
        let grad_result = grad_tensor.eval(ctx).unwrap();
        // Expected gradient: [1, 0; 0, 0] (derivative w.r.t. largest element)
        println!("Spectral gradient: {:.6}", grad_result);
        assert!(
            (grad_result[[0, 0]] - 1.0).abs() < 1e-6,
            "Spectral gradient [0,0] failed"
        );
        assert!(
            (grad_result[[0, 1]] - 0.0).abs() < 1e-6,
            "Spectral gradient [0,1] failed"
        );
        assert!(
            (grad_result[[1, 0]] - 0.0).abs() < 1e-6,
            "Spectral gradient [1,0] failed"
        );
        assert!(
            (grad_result[[1, 1]] - 0.0).abs() < 1e-6,
            "Spectral gradient [1,1] failed"
        );
        println!("✓ Spectral norm gradient test passed!");

        // Test nuclear norm on diagonal matrix
        println!("\n=== Testing Nuclear Norm (Diagonal Matrix) ===");
        let diag_matrix2 = convert_to_tensor(array![[2.0, 0.0], [0.0, -3.0]].into_dyn(), ctx);
        let nuc_norm = nuclear_norm(&diag_matrix2);

        // Expected nuclear norm: |2| + |-3| = 5.0
        let nuc_result = nuc_norm.eval(ctx).unwrap();
        println!("Nuclear norm result: {}", nuc_result[[]]);
        assert!(
            (nuc_result[[]] - 5.0).abs() < 1e-6,
            "Nuclear norm calculation failed"
        );

        // Test gradient
        let nuc_grads = grad(&[nuc_norm], &[diag_matrix2]);
        let grad_tensor = &nuc_grads[0];
        let grad_result = grad_tensor.eval(ctx).unwrap();
        // Expected gradient: sign(diag) = [1, 0; 0, -1]
        println!("Nuclear gradient: {:.6}", grad_result);
        assert!(
            (grad_result[[0, 0]] - 1.0).abs() < 1e-6,
            "Nuclear gradient [0,0] failed"
        );
        assert!(
            (grad_result[[0, 1]] - 0.0).abs() < 1e-6,
            "Nuclear gradient [0,1] failed"
        );
        assert!(
            (grad_result[[1, 0]] - 0.0).abs() < 1e-6,
            "Nuclear gradient [1,0] failed"
        );
        assert!(
            (grad_result[[1, 1]] - (-1.0)).abs() < 1e-6,
            "Nuclear gradient [1,1] failed"
        );
        println!("✓ Nuclear norm gradient test passed!");

        println!("\n🎉 All matrix norm gradient tests passed! Issue #42 appears to be fixed.");
    });
}
