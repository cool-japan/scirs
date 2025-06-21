#[cfg(feature = "tensor_contraction")]
mod tensor_train_example {
    use ndarray::{array, ArrayD};
    use scirs2_linalg::tensor_contraction::tensor_train::tensor_train_decomposition;

    pub fn run() {
        println!("Tensor Train Decomposition Examples");
        println!("==================================\n");

        // Example 1: Basic tensor train decomposition
        println!("Example 1: Basic Tensor Train Decomposition");

        // Create a 2x3x2 tensor
        let tensor = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];

        println!("Original tensor shape: {:?}", tensor.shape());

        // Decompose into tensor train format
        let tt = tensor_train_decomposition(&tensor.view(), None, None).unwrap();

        println!("Tensor Train ranks: {:?}", tt.ranks);
        println!("Tensor Train cores:");
        for (i, core) in tt.cores.iter().enumerate() {
            println!("  Core {}: shape {:?}", i, core.shape());
        }

        // Reconstruct the tensor
        let reconstructed = tt.to_full().unwrap();

        println!("Reconstructed tensor shape: {:?}", reconstructed.shape());
        println!(
            "Reconstruction error: {:.2e}",
            compute_relative_error(&tensor.clone().into_dyn(), &reconstructed)
        );
        println!();

        // Example 2: Tensor train decomposition with rank truncation
        println!("Example 2: Tensor Train Decomposition with Rank Truncation");

        // Create a 4x3x2x2 tensor with outer product structure
        let mut tensor4d = ArrayD::<f64>::zeros(ndarray::IxDyn(&[4, 3, 2, 2]));

        // Fill with values (outer product-like pattern)
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..2 {
                    for l in 0..2 {
                        tensor4d[[i, j, k, l]] =
                            (i + 1) as f64 * (j + 1) as f64 * (k + 1) as f64 * (l + 1) as f64;
                    }
                }
            }
        }

        println!("Original 4D tensor shape: {:?}", tensor4d.shape());

        // Decompose with maximum rank 2
        let tt_truncated = tensor_train_decomposition(&tensor4d.view(), Some(2), None).unwrap();

        println!("Truncated Tensor Train ranks: {:?}", tt_truncated.ranks);
        println!("Truncated Tensor Train cores:");
        for (i, core) in tt_truncated.cores.iter().enumerate() {
            println!("  Core {}: shape {:?}", i, core.shape());
        }

        // Reconstruct the tensor
        let reconstructed4d = tt_truncated.to_full().unwrap();

        println!("Reconstructed tensor shape: {:?}", reconstructed4d.shape());
        println!(
            "Reconstruction error: {:.2e}",
            compute_relative_error(&tensor4d, &reconstructed4d)
        );
        println!();

        // Example 3: Element-wise evaluation
        println!("Example 3: Element-wise Tensor Train Evaluation");

        // Get specific elements from the tensor train
        let indices = [1, 2, 1];
        let value = tt.get(&indices).unwrap();
        let original_value = tensor[[1, 2, 1]];

        println!("Original tensor at {:?}: {}", indices, original_value);
        println!("Tensor Train evaluation at {:?}: {}", indices, value);
        println!("Absolute error: {:.2e}", (value - original_value).abs());
        println!();

        // Example 4: Tensor rounding
        println!("Example 4: Tensor Train Rounding (Compression)");

        // Round the tensor train with different tolerances
        for &eps in &[1e-8, 1e-4, 1e-2] {
            let rounded_tt = tt.round(eps).unwrap();

            println!("Rounded Tensor Train (epsilon = {:.0e}):", eps);
            println!("  Ranks: {:?}", rounded_tt.ranks);

            // Reconstruct and measure error
            let reconstructed = rounded_tt.to_full().unwrap();
            println!(
                "  Reconstruction error: {:.2e}",
                compute_relative_error(&tensor.clone().into_dyn(), &reconstructed)
            );
        }
    }

    // Helper function to compute relative error
    fn compute_relative_error(original: &ArrayD<f64>, reconstructed: &ArrayD<f64>) -> f64 {
        let mut diff_sum = 0.0;
        let mut orig_sum = 0.0;

        // Iterate through the original tensor
        for (idx, &val) in original.indexed_iter() {
            let rec_val = reconstructed[idx.clone()];
            diff_sum += (val - rec_val).powi(2);
            orig_sum += val.powi(2);
        }

        (diff_sum / orig_sum).sqrt()
    }
}

fn main() {
    #[cfg(feature = "tensor_contraction")]
    {
        tensor_train_example::run();
    }

    #[cfg(not(feature = "tensor_contraction"))]
    {
        println!("This example requires tensor_contraction feature.");
        println!(
            "Run with: cargo run --example tensor_train_example --features tensor_contraction"
        );
    }
}
