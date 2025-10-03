//! Tensor-Train Decomposition for High-Dimensional Problems Example
//!
//! This example demonstrates the revolutionary Tensor-Train (TT) decomposition technique
//! for handling high-dimensional tensors with exponential compression ratios. TT decomposition
//! enables efficient representation and computation on tensors that would otherwise be
//! impossible to store or manipulate due to the curse of dimensionality.
//!
//! Key Features Demonstrated:
//! - TT decomposition of high-dimensional tensors
//! - Exponential compression vs. full tensor storage
//! - Efficient TT arithmetic operations (addition, Hadamard product)
//! - Adaptive rank truncation and rounding
//! - Applications in quantum many-body systems, PDEs, and machine learning
//!
//! TT decomposition transforms the exponential scaling O(n^d) into linear scaling O(d·n·R²),
//! making high-dimensional computations tractable for real-world applications.

use scirs2_core::ndarray::{Array, Array3, IxDyn};
use scirs2_linalg::tensor_train::{tt_add, tt_decomposition, tt_hadamard, TTTensor};
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 TENSOR-TRAIN DECOMPOSITION - Advanced DEMONSTRATION");
    println!("========================================================");
    println!("Revolutionary High-Dimensional Tensor Compression and Computation");
    println!("========================================================");

    // Test 1: Basic TT Tensor Construction and Properties
    println!("\n1. BASIC TT TENSOR CONSTRUCTION");
    println!("-------------------------------");

    // Create a simple 3D tensor in TT format manually
    let core1 = Array3::from_shape_fn((1, 3, 2), |(_, i, r)| {
        // First core: maps mode 1 to rank-2
        (i + r + 1) as f64 * 0.5
    });

    let core2 = Array3::from_shape_fn((2, 4, 3), |(r1, i, r2)| {
        // Second core: rank-2 to rank-3
        (r1 + i + r2 + 1) as f64 * 0.3
    });

    let core3 = Array3::from_shape_fn((3, 2, 1), |(r1, i_, _)| {
        // Third core: rank-3 to rank-1 (boundary)
        (r1 + i_ + 1) as f64 * 0.2
    });

    let tt_tensor = TTTensor::new(vec![core1, core2, core3])?;

    println!("   ✅ TT Tensor Properties:");
    println!("      - Dimensions: {:?}", tt_tensor.shape());
    println!("      - TT ranks: {:?}", tt_tensor.ranks);
    println!("      - Max rank: {}", tt_tensor.max_rank());
    println!("      - Storage size: {} elements", tt_tensor.storagesize());

    let fullsize: usize = tt_tensor.shape().iter().product();
    println!("      - Full tensor size: {} elements", fullsize);
    println!(
        "      - Compression ratio: {:.1}x",
        tt_tensor.compression_ratio()
    );

    // Test element access
    let element = tt_tensor.get_element(&[1, 2, 0])?;
    println!("      - Element [1,2,0]: {:.6}", element);

    // Test 2: TT Decomposition of Dense Tensors
    println!("\n2. TT DECOMPOSITION OF DENSE TENSORS");
    println!("------------------------------------");

    // Create a structured 4D tensor (separable function)
    let shape = vec![4, 4, 4, 4];
    let tensor_data: Vec<f64> = (0..4_usize.pow(4))
        .map(|flat_idx| {
            // Convert flat index to multi-index
            let mut idx = [0; 4];
            let mut remaining = flat_idx;
            for k in (0..4).rev() {
                idx[k] = remaining % 4;
                remaining /= 4;
            }

            // Create a separable function: f(i,j,k,l) = (i+1) * (j+1) * (k+1) * (l+1)
            (idx[0] + 1) as f64 * (idx[1] + 1) as f64 * (idx[2] + 1) as f64 * (idx[3] + 1) as f64
        })
        .collect();

    let dense_tensor = Array::from_shape_vec(IxDyn(&shape), tensor_data)?;

    println!("   📊 Dense Tensor Analysis:");
    println!("      - Original shape: {:?}", dense_tensor.shape());
    println!("      - Original size: {} elements", dense_tensor.len());
    println!(
        "      - Memory usage: {:.1} KB",
        dense_tensor.len() * 8 / 1024
    );

    // Perform TT decomposition
    let start_time = Instant::now();
    let tt_result = tt_decomposition(&dense_tensor.view(), 1e-12, Some(10))?;
    let decomp_time = start_time.elapsed();

    println!("\n   ✅ TT Decomposition Results:");
    println!("      - Decomposition time: {:?}", decomp_time);
    println!("      - TT ranks: {:?}", tt_result.ranks);
    println!("      - TT storage: {} elements", tt_result.storagesize());
    println!(
        "      - Compression ratio: {:.1}x",
        tt_result.compression_ratio()
    );
    println!(
        "      - Memory savings: {:.1}%",
        (1.0 - tt_result.storagesize() as f64 / dense_tensor.len() as f64) * 100.0
    );

    // Verify decomposition accuracy
    let mut max_error: f64 = 0.0;
    let mut total_error = 0.0;
    let num_samples = 50; // Sample some elements for verification

    for sample in 0..num_samples {
        let idx = vec![
            sample % 4,
            (sample / 4) % 4,
            (sample / 16) % 4,
            (sample / 64) % 4,
        ];
        let original = dense_tensor[IxDyn(&idx)];
        let reconstructed = tt_result.get_element(&idx)?;
        let error = (original - reconstructed).abs();

        max_error = max_error.max(error);
        total_error += error;
    }

    println!("      - Max reconstruction error: {:.2e}", max_error);
    println!(
        "      - Avg reconstruction error: {:.2e}",
        total_error / num_samples as f64
    );

    // Test 3: High-Dimensional Tensor Applications
    println!("\n3. HIGH-DIMENSIONAL APPLICATIONS");
    println!("--------------------------------");

    // Demonstrate exponential scaling challenge
    let dimensions = vec![6, 7, 8, 9, 10];

    for &d in &dimensions {
        let modesize: usize = 3;
        let fullsize = modesize.pow(d as u32);
        let estimated_ttsize = d * modesize * 4 * 4; // Assuming max rank ~4
        let compression = fullsize as f64 / estimated_ttsize as f64;

        println!("   🎯 {}-Dimensional Tensor ({}^{}):", d, modesize, d);
        println!(
            "      - Full size: {} elements ({:.1} GB)",
            fullsize,
            fullsize as f64 * 8.0 / 1e9
        );
        println!(
            "      - TT size estimate: {} elements ({:.1} MB)",
            estimated_ttsize,
            estimated_ttsize as f64 * 8.0 / 1e6
        );
        println!("      - Compression ratio: {:.0e}x", compression);

        if d >= 10 {
            println!(
                "      - Full tensor would require {:.1} TB!",
                fullsize as f64 * 8.0 / 1e12
            );
        }
    }

    // Test 4: TT Arithmetic Operations
    println!("\n4. TT ARITHMETIC OPERATIONS");
    println!("---------------------------");

    // Create two simple TT tensors for arithmetic
    let core_a = Array3::from_shape_fn((1, 3, 1), |(_, i_, _)| (i_ + 1) as f64);
    let tt_a = TTTensor::new(vec![core_a])?;

    let core_b = Array3::from_shape_fn((1, 3, 1), |(_, i_, _)| (i_ + 2) as f64);
    let tt_b = TTTensor::new(vec![core_b])?;

    println!(
        "   📐 TT Tensor A: [{:.0}, {:.0}, {:.0}]",
        tt_a.get_element(&[0])?,
        tt_a.get_element(&[1])?,
        tt_a.get_element(&[2])?
    );
    println!(
        "   📐 TT Tensor B: [{:.0}, {:.0}, {:.0}]",
        tt_b.get_element(&[0])?,
        tt_b.get_element(&[1])?,
        tt_b.get_element(&[2])?
    );

    // TT Addition
    let start_time = Instant::now();
    let tt_sum = tt_add(&tt_a, &tt_b)?;
    let add_time = start_time.elapsed();

    println!("\n   ➕ TT Addition (A + B):");
    println!(
        "      - Result: [{:.0}, {:.0}, {:.0}]",
        tt_sum.get_element(&[0])?,
        tt_sum.get_element(&[1])?,
        tt_sum.get_element(&[2])?
    );
    println!("      - Computation time: {:?}", add_time);
    println!("      - Result ranks: {:?}", tt_sum.ranks);

    // TT Hadamard Product
    let start_time = Instant::now();
    let tt_product = tt_hadamard(&tt_a, &tt_b)?;
    let hadamard_time = start_time.elapsed();

    println!("\n   ⊙ TT Hadamard Product (A ⊙ B):");
    println!(
        "      - Result: [{:.0}, {:.0}, {:.0}]",
        tt_product.get_element(&[0])?,
        tt_product.get_element(&[1])?,
        tt_product.get_element(&[2])?
    );
    println!("      - Computation time: {:?}", hadamard_time);
    println!("      - Result ranks: {:?}", tt_product.ranks);

    // Test 5: TT Rounding and Compression
    println!("\n5. TT ROUNDING AND COMPRESSION");
    println!("------------------------------");

    // Create a higher-rank tensor and demonstrate rounding
    let high_rank_tensor = tt_sum; // Use sum which has higher ranks

    println!("   📊 Original TT Tensor:");
    println!("      - Ranks: {:?}", high_rank_tensor.ranks);
    println!(
        "      - Storage: {} elements",
        high_rank_tensor.storagesize()
    );

    let tolerances = vec![1e-1, 1e-2, 1e-3, 1e-6];

    for &tol in &tolerances {
        let start_time = Instant::now();
        let rounded = high_rank_tensor.round(tol, Some(5))?;
        let round_time = start_time.elapsed();

        // Check approximation quality
        let error_0: f64 = (high_rank_tensor.get_element(&[0])? - rounded.get_element(&[0])?).abs();
        let error_1: f64 = (high_rank_tensor.get_element(&[1])? - rounded.get_element(&[1])?).abs();
        let error_2: f64 = (high_rank_tensor.get_element(&[2])? - rounded.get_element(&[2])?).abs();
        let max_error = error_0.max(error_1).max(error_2);

        println!("\n   🔄 Rounding (tolerance: {:.0e}):", tol);
        println!("      - New ranks: {:?}", rounded.ranks);
        println!("      - New storage: {} elements", rounded.storagesize());
        println!(
            "      - Compression: {:.1}x",
            high_rank_tensor.storagesize() as f64 / rounded.storagesize() as f64
        );
        println!("      - Max error: {:.2e}", max_error);
        println!("      - Rounding time: {:?}", round_time);
    }

    // Test 6: Real-World Application Scenarios
    println!("\n6. REAL-WORLD APPLICATIONS");
    println!("--------------------------");

    println!("   ✅ QUANTUM MANY-BODY SYSTEMS:");
    println!("      - Efficient representation of quantum states");
    println!("      - Ground state search algorithms (DMRG)");
    println!("      - Quantum circuit simulation");
    println!("      - Many-body localization studies");

    println!("   ✅ MACHINE LEARNING:");
    println!("      - Neural network weight compression");
    println!("      - High-dimensional feature spaces");
    println!("      - Tensor regression and classification");
    println!("      - Reinforcement learning value functions");

    println!("   ✅ NUMERICAL PDEs:");
    println!("      - High-dimensional Schrödinger equations");
    println!("      - Stochastic partial differential equations");
    println!("      - Monte Carlo methods in high dimensions");
    println!("      - Financial derivative pricing");

    println!("   ✅ DATA ANALYSIS:");
    println!("      - High-order tensor factorization");
    println!("      - Multilinear algebra computations");
    println!("      - Signal processing in multiple dimensions");
    println!("      - Computer vision and image analysis");

    // Test 7: Performance and Scalability Analysis
    println!("\n7. PERFORMANCE ANALYSIS");
    println!("----------------------");

    // Demonstrate scalability of TT operations
    let tensorsizes = vec![(3, 3), (4, 4), (5, 5)];

    for &(d, n) in &tensorsizes {
        // Create a random-like structured tensor
        let shape: Vec<usize> = vec![n; d];
        let total_elements: usize = shape.iter().product();

        // Generate data (structured to have low TT rank)
        let tensor_data: Vec<f64> = (0..total_elements)
            .map(|flat_idx| {
                let mut idx = vec![0; d];
                let mut remaining = flat_idx;
                for k in (0..d).rev() {
                    idx[k] = remaining % n;
                    remaining /= n;
                }

                // Sum of coordinates (separable, rank-1)
                idx.iter().map(|&i| (i + 1) as f64).sum::<f64>()
            })
            .collect();

        let dense_tensor = Array::from_shape_vec(IxDyn(&shape), tensor_data)?;

        let start_time = Instant::now();
        let tt_tensor = tt_decomposition(&dense_tensor.view(), 1e-10, Some(8))?;
        let decomp_time = start_time.elapsed();

        println!("   📈 {}D Tensor ({}^{}):", d, n, d);
        println!("      - Original size: {} elements", dense_tensor.len());
        println!("      - TT storage: {} elements", tt_tensor.storagesize());
        println!("      - Compression: {:.1}x", tt_tensor.compression_ratio());
        println!("      - TT ranks: {:?}", tt_tensor.ranks);
        println!("      - Decomposition time: {:?}", decomp_time);

        // Compute Frobenius norm efficiently
        let start_time = Instant::now();
        let tt_norm = tt_tensor.frobenius_norm()?;
        let norm_time = start_time.elapsed();

        println!("      - Frobenius norm: {:.6}", tt_norm);
        println!("      - Norm computation: {:?}", norm_time);
    }

    println!("\n========================================================");
    println!("🎯 Advanced ACHIEVEMENT: TENSOR-TRAIN DECOMPOSITION");
    println!("========================================================");
    println!("✅ Revolutionary high-dimensional tensor representation");
    println!("✅ Exponential compression: O(d·n·R²) vs O(n^d) storage");
    println!("✅ Efficient TT arithmetic with rank-aware algorithms");
    println!("✅ Adaptive rank control and SVD-based truncation");
    println!("✅ Applications in quantum physics, ML, and numerical PDEs");
    println!("✅ Scalable algorithms for curse of dimensionality problems");
    println!("========================================================");

    Ok(())
}
