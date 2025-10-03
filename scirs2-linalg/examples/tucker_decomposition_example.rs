//! This example demonstrates the Tucker decomposition functionality.
//!
//! To run this example with the tensor_contraction feature:
//! ```bash
//! cargo run --example tucker_decomposition_example --features tensor_contraction
//! ```

#[cfg(feature = "tensor_contraction")]
#[allow(dead_code)]
fn main() -> scirs2_linalg::error::LinalgResult<()> {
    use scirs2_core::ndarray::array;
    use scirs2_linalg::tensor_contraction::tucker::{tucker_als, tucker_decomposition};

    println!("Tucker Decomposition Example");
    println!("===========================\n");

    // Create a 3D tensor (2x3x2)
    let tensor = array![
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    ];

    println!("Original tensor shape: {:?}", tensor.shape());
    println!("Tensor values:\n{:?}\n", tensor);

    // --- HOSVD-based Tucker Decomposition (Full Rank) ---

    println!("1. Tucker Decomposition with HOSVD (Full Rank)");
    println!("---------------------------------------------");

    // Decompose with full rank
    let tucker_full = tucker_decomposition(&tensor.view(), &[2, 3, 2])?;

    println!("Core tensor shape: {:?}", tucker_full.core.shape());
    println!("Factor matrices shapes:");
    for (i, factor) in tucker_full.factors.iter().enumerate() {
        println!("  Mode {}: {:?}", i, factor.shape());
    }

    // Reconstruct and check error
    let reconstructed = tucker_full.to_full()?;
    let error = tucker_full.reconstruction_error(&tensor.view())?;

    println!("Reconstruction error: {:.10e}", error);
    println!("Reconstructed tensor:\n{:?}\n", reconstructed);

    // --- HOSVD-based Tucker Decomposition (Truncated) ---

    println!("2. Tucker Decomposition with HOSVD (Truncated Rank)");
    println!("-------------------------------------------------");

    // Decompose with truncated rank
    let tucker_trunc = tucker_decomposition(&tensor.view(), &[2, 2, 2])?;

    println!("Core tensor shape: {:?}", tucker_trunc.core.shape());
    println!("Factor matrices shapes:");
    for (i, factor) in tucker_trunc.factors.iter().enumerate() {
        println!("  Mode {}: {:?}", i, factor.shape());
    }

    // Reconstruct and check error
    let reconstructed_trunc = tucker_trunc.to_full()?;
    let error_trunc = tucker_trunc.reconstruction_error(&tensor.view())?;

    println!("Reconstruction error: {:.10e}", error_trunc);
    println!("Reconstructed tensor:\n{:?}\n", reconstructed_trunc);

    // --- Tucker-ALS Decomposition ---

    println!("3. Tucker Decomposition with ALS");
    println!("------------------------------");

    // Decompose with ALS
    let tucker_als = tucker_als(&tensor.view(), &[2, 2, 2], 10, 1e-4)?;

    println!("Core tensor shape: {:?}", tucker_als.core.shape());
    println!("Factor matrices shapes:");
    for (i, factor) in tucker_als.factors.iter().enumerate() {
        println!("  Mode {}: {:?}", i, factor.shape());
    }

    // Reconstruct and check error
    let reconstructed_als = tucker_als.to_full()?;
    let error_als = tucker_als.reconstruction_error(&tensor.view())?;

    println!("Reconstruction error: {:.10e}", error_als);
    println!("Reconstructed tensor:\n{:?}\n", reconstructed_als);

    // --- Compression with epsilon ---

    println!("4. Compression with epsilon");
    println!("--------------------------");

    // Compress the full-rank decomposition
    let compressed = tucker_full.compress(None, Some(0.1))?;

    println!(
        "Compressed core tensor shape: {:?}",
        compressed.core.shape()
    );
    println!("Compressed factor matrices shapes:");
    for (i, factor) in compressed.factors.iter().enumerate() {
        println!("  Mode {}: {:?}", i, factor.shape());
    }

    // Reconstruct and check error
    let _reconstructed_comp = compressed.to_full()?;
    let error_comp = compressed.reconstruction_error(&tensor.view())?;

    println!("Reconstruction error: {:.10e}", error_comp);

    // --- Compression with specified ranks ---

    println!("5. Compression with specified ranks");
    println!("--------------------------------");

    // Compress with specified ranks
    let compressed_rank = tucker_full.compress(Some(vec![1, 2, 1]), None)?;

    println!(
        "Compressed core tensor shape: {:?}",
        compressed_rank.core.shape()
    );
    println!("Compressed factor matrices shapes:");
    for (i, factor) in compressed_rank.factors.iter().enumerate() {
        println!("  Mode {}: {:?}", i, factor.shape());
    }

    // Reconstruct and check error
    let error_comp_rank = compressed_rank.reconstruction_error(&tensor.view())?;
    println!("Reconstruction error: {:.10e}", error_comp_rank);

    // --- Compare errors ---

    println!("\nComparison of Reconstruction Errors:");
    println!("----------------------------------");
    println!("Full-rank HOSVD:           {:.10e}", error);
    println!("Truncated-rank HOSVD:      {:.10e}", error_trunc);
    println!("ALS (same truncated rank): {:.10e}", error_als);
    println!("Epsilon compression:       {:.10e}", error_comp);
    println!("Rank compression:          {:.10e}", error_comp_rank);

    Ok(())
}

#[cfg(not(feature = "tensor_contraction"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the 'tensor_contraction' feature.");
    println!("Please run with: cargo run --example tucker_decomposition_example --features tensor_contraction");
}
