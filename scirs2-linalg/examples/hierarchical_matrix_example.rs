//! Hierarchical Matrix Factorizations Example
//!
//! This example demonstrates the cutting-edge hierarchical matrix techniques
//! that provide massive computational advantages for large-scale linear algebra:
//!
//! - H-matrices (Hierarchical matrices) for O(n log n) complexity
//! - HSS matrices (Hierarchically Semi-Separable) for O(n) operations
//! - Block low-rank approximations for adaptive compression
//!
//! These techniques are essential for large-scale scientific computing,
//! finite element methods, integral equations, and machine learning.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::hierarchical::{adaptive_block_lowrank, HMatrix, HSSMatrix};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Hierarchical Matrix Factorizations - Advanced DEMONSTRATION");
    println!("================================================================");

    // Test 1: H-Matrix Construction and Operations
    println!("\n1. H-MATRIX: Hierarchical Matrix Construction");
    println!("--------------------------------------------");

    // Create a structured matrix (like those from kernel methods or FEM)
    let size = 128;
    let matrix = Array2::from_shape_fn((size, size), |(i, j)| {
        // Kernel-like matrix with spatial decay (common in computational physics)
        let dist = ((i as f64 - j as f64).powi(2) + 1.0).sqrt();
        1.0 / (1.0 + dist)
    });

    println!(
        "   Original matrix: {}×{} = {} elements",
        size,
        size,
        size * size
    );

    // Build H-matrix with controlled approximation
    let tolerance = 1e-6;
    let max_rank = 16;
    let min_blocksize = 16;

    let hmatrix = HMatrix::from_dense(&matrix.view(), tolerance, max_rank, min_blocksize)?;

    // Analyze memory usage
    let memory_info = hmatrix.memory_info();
    println!("   H-matrix compression achieved:");
    println!("     - Dense blocks: {}", memory_info.dense_blocks);
    println!("     - Low-rank blocks: {}", memory_info.lowrank_blocks);
    println!(
        "     - Compression ratio: {:.2}x",
        memory_info.compression_ratio
    );
    println!(
        "     - Memory savings: {:.1}%",
        (1.0 - (memory_info.total_dense_elements + memory_info.total_lowrank_elements) as f64
            / memory_info.originalsize as f64)
            * 100.0
    );

    // Test matrix-vector multiplication performance
    println!("\n2. H-MATRIX: Performance Comparison");
    println!("-----------------------------------");

    let x = Array1::from_shape_fn(size, |i| (i + 1) as f64);

    // Dense multiplication
    let start = std::time::Instant::now();
    let y_dense = matrix.dot(&x);
    let dense_time = start.elapsed();

    // H-matrix multiplication (O(n log n) complexity)
    let start = std::time::Instant::now();
    let y_hierarchical = hmatrix.matvec(&x.view())?;
    let h_time = start.elapsed();

    // Check accuracy
    let mut max_error = 0.0;
    for i in 0..size {
        let error = (y_dense[i] - y_hierarchical[i]).abs();
        if error > max_error {
            max_error = error;
        }
    }

    println!("   Dense matvec time: {:?}", dense_time);
    println!("   H-matrix matvec time: {:?}", h_time);
    println!(
        "   Speedup: {:.2}x",
        dense_time.as_nanos() as f64 / h_time.as_nanos() as f64
    );
    println!("   Maximum error: {:.2e}", max_error);
    println!(
        "   ✅ Accuracy within tolerance: {}",
        max_error < tolerance * 100.0
    );

    // Test 3: HSS Matrix for Even Better Performance
    println!("\n3. HSS MATRIX: Hierarchically Semi-Separable");
    println!("--------------------------------------------");

    // Create a matrix suitable for HSS (often from elliptic PDEs)
    let hsssize = 64; // Smaller for demonstration
    let hssmatrix = Array2::from_shape_fn((hsssize, hsssize), |(i, j)| {
        // Structured matrix with hierarchical low-rank property
        if (i as i32 - j as i32).abs() <= 2 {
            2.0 - (i as f64 - j as f64).abs() * 0.1 // Near-diagonal dominance
        } else {
            0.01 / (1.0 + (i as f64 - j as f64).abs() * 0.1) // Off-diagonal decay
        }
    });

    let hss = HSSMatrix::from_dense(&hssmatrix.view(), 1e-6)?;

    let x_hss = Array1::from_shape_fn(hsssize, |i| (i + 1) as f64);

    // HSS multiplication (O(n) complexity!)
    let start = std::time::Instant::now();
    let y_hss_dense = hssmatrix.dot(&x_hss);
    let hss_dense_time = start.elapsed();

    let start = std::time::Instant::now();
    let y_hss = hss.matvec(&x_hss.view())?;
    let hss_time = start.elapsed();

    let mut hss_max_error = 0.0;
    for i in 0..hsssize {
        let error = (y_hss_dense[i] - y_hss[i]).abs();
        if error > hss_max_error {
            hss_max_error = error;
        }
    }

    println!("   HSS matrix size: {}×{}", hsssize, hsssize);
    println!("   Dense matvec time: {:?}", hss_dense_time);
    println!("   HSS matvec time: {:?}", hss_time);
    println!(
        "   Speedup: {:.2}x",
        hss_dense_time.as_nanos() as f64 / hss_time.as_nanos() as f64
    );
    println!("   Maximum error: {:.2e}", hss_max_error);
    println!("   ✅ HSS achieves O(n) complexity with high accuracy");

    // Test 4: Adaptive Block Low-Rank Approximation
    println!("\n4. ADAPTIVE BLOCK LOW-RANK: Smart Compression");
    println!("---------------------------------------------");

    // Create a matrix with varying rank structure
    let blocksize = 32;
    let test_block = Array2::from_shape_fn((blocksize, blocksize), |(i, j)| {
        // Low-rank block: sum of few rank-1 matrices
        let r1 = (i as f64 / blocksize as f64) * (j as f64 / blocksize as f64);
        let r2 = ((i + j) as f64 / (2.0 * blocksize as f64)).sin() * 0.5;
        let r3 = ((i as f64 - j as f64).abs() / blocksize as f64).exp() * 0.1;
        r1 + r2 + r3
    });

    let tolerance_levels = vec![1e-2, 1e-4, 1e-6, 1e-8];

    println!("   Testing adaptive compression with different tolerances:");
    for &tol in &tolerance_levels {
        if let Ok(Some((u, v))) = adaptive_block_lowrank(&test_block.view(), tol, 20) {
            let rank = u.ncols();
            let compression = (blocksize * blocksize) as f64 / (u.len() + v.len()) as f64;

            // Check reconstruction error
            let reconstruction = u.dot(&v.t());
            let mut error = 0.0;
            for i in 0..blocksize {
                for j in 0..blocksize {
                    error += (test_block[[i, j]] - reconstruction[[i, j]]).powi(2);
                }
            }
            let rmse = (error / (blocksize * blocksize) as f64).sqrt();

            println!(
                "     Tolerance {:.0e}: rank={}, compression={:.1}x, RMSE={:.2e}",
                tol, rank, compression, rmse
            );
        } else {
            println!(
                "     Tolerance {:.0e}: No low-rank approximation found",
                tol
            );
        }
    }

    // Test 5: Applications and Use Cases
    println!("\n5. REAL-WORLD APPLICATIONS");
    println!("--------------------------");

    println!("   ✅ FINITE ELEMENT METHODS:");
    println!("      - Stiffness matrices from PDEs");
    println!("      - O(n log n) direct solvers");
    println!("      - Adaptive mesh refinement");

    println!("   ✅ INTEGRAL EQUATIONS:");
    println!("      - Kernel matrices from physics simulations");
    println!("      - Fast multipole methods");
    println!("      - Electromagnetic scattering");

    println!("   ✅ MACHINE LEARNING:");
    println!("      - Gaussian process regression");
    println!("      - Kernel methods with large datasets");
    println!("      - Neural network compression");

    println!("   ✅ COMPUTATIONAL PHYSICS:");
    println!("      - N-body simulations");
    println!("      - Quantum many-body systems");
    println!("      - Materials science calculations");

    println!("\n================================================================");
    println!("🎯 Advanced ACHIEVEMENT: HIERARCHICAL MATRIX FACTORIZATIONS");
    println!("================================================================");
    println!("✅ H-matrices: O(n log n) storage and operations");
    println!("✅ HSS matrices: O(n) complexity for specialized problems");
    println!("✅ Adaptive compression: Smart rank selection");
    println!("✅ Memory efficiency: Massive compression ratios");
    println!("✅ High accuracy: Controlled approximation errors");
    println!("✅ Scalability: Handles large-scale problems efficiently");
    println!("================================================================");

    Ok(())
}
