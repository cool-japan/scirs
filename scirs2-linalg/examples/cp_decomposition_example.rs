//! This example demonstrates the Canonical Polyadic (CP) decomposition functionality.
//!
//! To run this example with the tensor_contraction feature:
//! ```bash
//! cargo run --example cp_decomposition_example --features tensor_contraction
//! ```

#[cfg(feature = "tensor_contraction")]
#[allow(dead_code)]
fn main() -> scirs2_linalg::error::LinalgResult<()> {
    use scirs2_core::ndarray::array;
    use scirs2_linalg::tensor_contraction::cp::{cp_als, CanonicalPolyadic};

    println!("Canonical Polyadic (CP) Decomposition Example");
    println!("============================================\n");

    // Create a 3D tensor (2x3x2)
    let tensor = array![
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    ];

    println!("Original tensor shape: {:?}", tensor.shape());
    println!("Tensor values:\n{:?}\n", tensor);

    // --- CP Decomposition (Rank = 2) ---

    println!("1. CP Decomposition with rank 2");
    println!("------------------------------");

    // Decompose with rank 2, 50 iterations, tolerance 1e-4, and normalize factors
    let cp_rank2 = cp_als(&tensor.view(), 2, 50, 1e-4, true)?;

    println!("Factor matrices shapes:");
    for (i, factor) in cp_rank2.factors.iter().enumerate() {
        println!("  Mode {}: {:?}", i, factor.shape());
    }

    println!("Component weights:");
    if let Some(weights) = &cp_rank2.weights {
        println!("  {:?}", weights);
    } else {
        println!("  None (uniform weighting)");
    }

    // Reconstruct and check error
    let reconstructed = cp_rank2.to_full()?;
    let error = cp_rank2.reconstruction_error(&tensor.view())?;

    println!("Reconstruction error: {:.10e}", error);
    println!("Reconstructed tensor:\n{:?}\n", reconstructed);

    // --- CP Decomposition (Rank = 3) ---

    println!("2. CP Decomposition with rank 3");
    println!("------------------------------");

    // Decompose with rank 3, 50 iterations, tolerance 1e-4, and normalize factors
    let cp_rank3 = cp_als(&tensor.view(), 3, 50, 1e-4, true)?;

    println!("Factor matrices shapes:");
    for (i, factor) in cp_rank3.factors.iter().enumerate() {
        println!("  Mode {}: {:?}", i, factor.shape());
    }

    println!("Component weights:");
    if let Some(weights) = &cp_rank3.weights {
        println!("  {:?}", weights);
    } else {
        println!("  None (uniform weighting)");
    }

    // Reconstruct and check error
    let reconstructed_rank3 = cp_rank3.to_full()?;
    let error_rank3 = cp_rank3.reconstruction_error(&tensor.view())?;

    println!("Reconstruction error: {:.10e}", error_rank3);
    println!("Reconstructed tensor:\n{:?}\n", reconstructed_rank3);

    // --- CP Decomposition (Non-normalized) ---

    println!("3. CP Decomposition without normalization");
    println!("---------------------------------------");

    // Decompose with rank 3, 50 iterations, tolerance 1e-4, without normalizing factors
    let cp_nonorm = cp_als(&tensor.view(), 3, 50, 1e-4, false)?;

    println!("Factor matrices shapes:");
    for (i, factor) in cp_nonorm.factors.iter().enumerate() {
        println!("  Mode {}: {:?}", i, factor.shape());
    }

    println!("Component weights:");
    if let Some(weights) = &cp_nonorm.weights {
        println!("  {:?}", weights);
    } else {
        println!("  None (uniform weighting)");
    }

    // Reconstruct and check error
    let error_nonorm = cp_nonorm.reconstruction_error(&tensor.view())?;
    println!("Reconstruction error: {:.10e}", error_nonorm);

    // --- Compression Example ---

    println!("4. Compression from rank 3 to rank 2");
    println!("----------------------------------");

    // Compress the rank 3 decomposition to rank 2
    let compressed = cp_rank3.compress(2)?;

    println!("Compressed factor matrices shapes:");
    for (i, factor) in compressed.factors.iter().enumerate() {
        println!("  Mode {}: {:?}", i, factor.shape());
    }

    println!("Compressed component weights:");
    if let Some(weights) = &compressed.weights {
        println!("  {:?}", weights);
    } else {
        println!("  None (uniform weighting)");
    }

    // Reconstruct and check error
    let error_compressed = compressed.reconstruction_error(&tensor.view())?;

    println!("Reconstruction error: {:.10e}", error_compressed);

    // --- Manual Construction Example ---

    println!("5. Manual Construction of a CP Decomposition");
    println!("-----------------------------------------");

    // Create factor matrices manually
    let a = array![[1.0, 2.0], [3.0, 4.0]]; // First mode, rank 2
    let b = array![[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]; // Second mode, rank 2
    let c = array![[11.0, 12.0], [13.0, 14.0]]; // Third mode, rank 2

    let factors = vec![a, b, c];
    let weights = Some(array![1.5, 0.5]); // Custom weights

    // Create CP decomposition
    let manual_cp = CanonicalPolyadic::new(factors, weights, Some(vec![2, 3, 2]))?;

    println!("Manually constructed CP decomposition with 2 components");
    println!("Reconstruct and check components...");

    // Reconstruct and check error
    let reconstructed_manual = manual_cp.to_full()?;
    let error_manual = manual_cp.reconstruction_error(&tensor.view())?;

    println!("Reconstruction error: {:.10e}", error_manual);
    println!("Reconstructed tensor:\n{:?}\n", reconstructed_manual);

    // --- Compare errors ---

    println!("\nComparison of Reconstruction Errors:");
    println!("----------------------------------");
    println!("CP with rank 2:           {:.10e}", error);
    println!("CP with rank 3:           {:.10e}", error_rank3);
    println!("CP without normalization: {:.10e}", error_nonorm);
    println!("Compressed from 3 to 2:   {:.10e}", error_compressed);
    println!("Manual construction:      {:.10e}", error_manual);

    Ok(())
}

#[cfg(not(feature = "tensor_contraction"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the 'tensor_contraction' feature.");
    println!("Please run with: cargo run --example cp_decomposition_example --features tensor_contraction");
}
