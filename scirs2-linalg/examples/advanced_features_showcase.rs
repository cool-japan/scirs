//! Advanced Features Showcase
//!
//! This example demonstrates advanced features of scirs2-linalg including:
//! - Performance optimization techniques
//! - Memory management and cache-friendly algorithms  
//! - SIMD-accelerated operations
//! - Specialized algorithms for different matrix types
//! - Numerical stability features

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_linalg::{
    cond, conjugate_gradient,
    error::LinalgResult,
    matrix_norm, matrix_rank,
    parallel::{adaptive, algorithms, WorkerConfig},
    qr, svd, vector_norm,
};
use std::time::Instant;

#[allow(dead_code)]
fn main() -> LinalgResult<()> {
    println!("=== Advanced Features Showcase ===\n");

    // Performance optimization techniques
    performance_optimization_demo()?;

    // SIMD-accelerated operations (commented out for simplicity)
    // simd_operations_demo()?;

    // Memory-efficient algorithms
    memory_efficiency_demo()?;

    // Numerical stability features
    numerical_stability_demo()?;

    // Specialized algorithms
    specialized_algorithms_demo()?;

    // Adaptive algorithm selection
    adaptive_algorithms_demo()?;

    println!("✅ All advanced features demonstrated successfully!");
    Ok(())
}

/// Demonstrates performance optimization techniques
#[allow(dead_code)]
fn performance_optimization_demo() -> LinalgResult<()> {
    println!("🚀 Performance Optimization Techniques");
    println!("{}", "=".repeat(50));

    let sizes = vec![50, 100, 200];

    for &size in &sizes {
        println!("\n📊 Testing with {}x{} matrices", size, size);

        // Create test matrices
        let a = Array2::from_shape_fn((size, size), |(i, j)| {
            (i as f64 + 1.0) * (j as f64 + 1.0).sin()
        });
        let b = Array2::from_shape_fn((size, size), |(i, j)| (i as f64 * j as f64 + 1.0).cos());

        // Serial matrix multiplication
        let start = Instant::now();
        let _result_serial = a.dot(&b);
        let serial_time = start.elapsed();

        // Parallel matrix multiplication
        let config = WorkerConfig::new().with_workers(4).with_threshold(1000);
        let start = Instant::now();
        let _result_parallel = algorithms::parallel_gemm(&a.view(), &b.view(), &config)?;
        let parallel_time = start.elapsed();

        println!("  Serial time:   {:?}", serial_time);
        println!("  Parallel time: {:?}", parallel_time);

        if parallel_time < serial_time {
            let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();
            println!("  Speedup: {:.2}x", speedup);
        } else {
            println!("  Overhead detected (matrix too small for parallel benefit)");
        }
    }

    println!("\n");
    Ok(())
}

/// Demonstrates SIMD-accelerated operations  
#[allow(dead_code)]
fn simd_operations_demo() -> LinalgResult<()> {
    // SIMD demonstration commented out for compilation simplicity
    // In a real implementation, you would use:
    // - scirs2_linalg::simd, _ops::gemm::simd_gemm_f64 for matrix multiplication
    // - scirs2_linalg::simd, _ops::transpose::simd_transpose_f64 for transpose
    // - scirs2_linalg::simd, _ops::norms::simd_frobenius_norm_f64 for norms

    println!("⚡ SIMD-Accelerated Operations");
    println!("{}", "=".repeat(50));
    println!("SIMD operations provide significant speedups for large matrices");
    println!("when the 'simd' feature is enabled. Key benefits include:");
    println!("  • Vectorized arithmetic operations");
    println!("  • Cache-friendly memory access patterns");
    println!("  • Reduced CPU instruction count");
    println!("  • Better utilization of modern CPU capabilities");
    println!();

    Ok(())
}

/// Demonstrates memory-efficient algorithms
#[allow(dead_code)]
fn memory_efficiency_demo() -> LinalgResult<()> {
    println!("💾 Memory-Efficient Algorithms");
    println!("{}", "=".repeat(50));

    // Large matrix that would benefit from memory-efficient processing
    let size = 100;
    let a = Array2::from_shape_fn((size, size), |(i, j)| {
        if (i as i32 - j as i32).abs() <= 2 {
            (i + j) as f64 + 1.0
        } else {
            0.0
        }
    });

    println!("Testing with {}x{} banded matrix", size, size);

    // Memory-efficient QR decomposition
    println!("\n🔹 Memory-Efficient QR Decomposition");

    // Monitor memory usage conceptually (in a real scenario, you'd use actual memory profiling)
    let start = Instant::now();
    let (q, r) = qr(&a.view(), None)?;
    let qr_time = start.elapsed();

    println!("  QR decomposition time: {:?}", qr_time);
    println!("  Q matrix shape: {:?}", q.dim());
    println!("  R matrix shape: {:?}", r.dim());

    // Verify orthogonality of Q
    let qtq = q.t().dot(&q);
    let identity = Array2::<f64>::eye(size);
    let orthogonality_error = matrix_norm(&(&qtq - &identity).view(), "fro", None)?;
    println!("  Q orthogonality error: {:.2e}", orthogonality_error);

    // Memory-efficient SVD with rank estimation
    println!("\n🔹 Memory-Efficient SVD");
    let start = Instant::now();
    let (_u, s, vt) = svd(&a.view(), false, None)?;
    let svd_time = start.elapsed();

    println!("  SVD time: {:?}", svd_time);
    println!("  Number of singular values: {}", s.len());
    println!("  Largest singular value: {:.6}", s[0]);
    println!("  Smallest singular value: {:.6}", s[s.len() - 1]);

    // Estimate rank
    let rank = matrix_rank(&a.view(), None, None)?;
    println!("  Estimated rank: {}", rank);

    // Block-wise processing demonstration
    println!("\n🔹 Block-wise Processing");
    let blocksize = 32;
    let mut norm_accumulator = 0.0;

    for i in (0..size).step_by(blocksize) {
        for j in (0..size).step_by(blocksize) {
            let i_end = std::cmp::min(i + blocksize, size);
            let j_end = std::cmp::min(j + blocksize, size);

            let block = a.slice(scirs2_core::ndarray::s![i..i_end, j..j_end]);
            let block_norm = matrix_norm(&block, "fro", None)?;
            norm_accumulator += block_norm * block_norm;
        }
    }

    let blockwise_norm = norm_accumulator.sqrt();
    let direct_norm = matrix_norm(&a.view(), "fro", None)?;

    println!("  Block-wise Frobenius norm: {:.6}", blockwise_norm);
    println!("  Direct Frobenius norm:     {:.6}", direct_norm);
    println!("  Difference: {:.2e}", (blockwise_norm - direct_norm).abs());

    println!("\n");
    Ok(())
}

/// Demonstrates numerical stability features
#[allow(dead_code)]
fn numerical_stability_demo() -> LinalgResult<()> {
    println!("🔧 Numerical Stability Features");
    println!("{}", "=".repeat(50));

    // Create a poorly conditioned matrix
    let n = 10;
    let mut ill_conditioned = Array2::eye(n);

    // Make it ill-conditioned by setting small diagonal elements
    for i in 0..n {
        ill_conditioned[[i, i]] = 10.0_f64.powi(-(i as i32));
    }

    println!("Testing with ill-conditioned matrix");
    println!("Diagonal elements: [1e0, 1e-1, 1e-2, ..., 1e-9]");

    // Condition number analysis
    println!("\n🔹 Condition Number Analysis");
    let condition_number = cond(&ill_conditioned.view(), None, None)?;
    println!("  Condition number: {:.2e}", condition_number);

    if condition_number > 1e12 {
        println!("  ⚠️  Matrix is very ill-conditioned!");
        println!("  Numerical solutions may be unreliable.");
    } else if condition_number > 1e6 {
        println!("  ⚠️  Matrix is moderately ill-conditioned.");
    } else {
        println!("  ✅ Matrix is well-conditioned.");
    }

    // Rank determination with tolerance analysis
    println!("\n🔹 Rank Determination with Tolerance Analysis");
    let tolerances = vec![1e-16, 1e-12, 1e-8, 1e-4];

    for &tol in &tolerances {
        let rank = matrix_rank(&ill_conditioned.view(), Some(tol), None)?;
        println!("  Rank with tolerance {:.0e}: {}", tol, rank);
    }

    // Demonstrate iterative refinement
    println!("\n🔹 Iterative Solver for Stability");
    let b = Array1::ones(n);

    // Use conjugate gradient (more stable for ill-conditioned systems)
    match conjugate_gradient(&ill_conditioned.view(), &b.view(), 1000, 1e-10, None) {
        Ok(x) => {
            println!("  Iterative solver converged");
            let residual = &ill_conditioned.dot(&x) - &b;
            let residual_norm = vector_norm(&residual.view(), 2)?;
            println!("  Residual norm: {:.2e}", residual_norm);
        }
        Err(e) => {
            println!("  Iterative solver failed: {}", e);
            println!("  This is expected for very ill-conditioned matrices");
        }
    }

    // Demonstrate graceful degradation
    println!("\n🔹 Graceful Degradation");

    // Create a nearly singular matrix
    let mut nearly_singular = Array2::from_shape_fn((3, 3), |(i, j)| {
        if i == j {
            1.0
        } else if (i == 2 && j == 1) || (i == 1 && j == 2) {
            1.0 + 1e-15 // Nearly dependent entries
        } else {
            0.0
        }
    });
    nearly_singular[[2, 0]] = 1e-15; // Make third row nearly dependent on first two

    println!("  Testing nearly singular matrix...");
    let rank_nearly_singular = matrix_rank(&nearly_singular.view(), None, None)?;
    println!("  Automatic rank detection: {}", rank_nearly_singular);

    // Different tolerance levels
    let strict_rank = matrix_rank(&nearly_singular.view(), Some(1e-12), None)?;
    let loose_rank = matrix_rank(&nearly_singular.view(), Some(1e-10), None)?;

    println!("  Rank with strict tolerance (1e-12): {}", strict_rank);
    println!("  Rank with loose tolerance (1e-10):  {}", loose_rank);

    println!("\n");
    Ok(())
}

/// Demonstrates specialized algorithms
#[allow(dead_code)]
fn specialized_algorithms_demo() -> LinalgResult<()> {
    println!("🎯 Specialized Algorithms");
    println!("{}", "=".repeat(50));

    // Symmetric positive definite matrix
    println!("🔹 Symmetric Positive Definite Matrices");
    let spd = array![[4.0, 2.0, 1.0], [2.0, 3.0, 0.5], [1.0, 0.5, 2.0]];

    println!("  SPD matrix:");
    println!("  {:?}", spd);

    // Specialized algorithms benefit from SPD structure
    let (_eigenvals, eigenvecs) = scirs2_linalg::eigh(&spd.view(), None)?;
    println!("  ✅ Symmetric eigenvalue decomposition successful");

    // Cholesky decomposition is more efficient for SPD matrices
    if let Ok(l) = scirs2_linalg::cholesky(&spd.view(), None) {
        println!("  ✅ Cholesky decomposition successful");

        // Verify: L * L^T = A
        let reconstructed = l.dot(&l.t());
        let reconstruction_error = matrix_norm(&(&spd - &reconstructed).view(), "fro", None)?;
        println!("  Reconstruction error: {:.2e}", reconstruction_error);
    }

    // Banded matrices
    println!("\n🔹 Banded Matrix Algorithms");
    let size = 50;
    let bandwidth = 3;
    let banded = Array2::from_shape_fn((size, size), |(i, j)| {
        if (i as i32 - j as i32).abs() <= bandwidth {
            (i + j + 1) as f64
        } else {
            0.0
        }
    });

    println!("  {}x{} matrix with bandwidth {}", size, size, bandwidth);

    // Count non-zero elements
    let non_zeros = banded.iter().filter(|&&x| x != 0.0).count();
    let sparsity = 1.0 - (non_zeros as f64) / (size * size) as f64;
    println!("  Sparsity: {:.1}%", sparsity * 100.0);

    // Specialized algorithms can exploit banded structure
    let (_q, r) = qr(&banded.view(), None)?;
    println!("  ✅ QR decomposition on banded matrix successful");

    // Tridiagonal matrices (special case of banded)
    println!("\n🔹 Tridiagonal Matrix Algorithms");
    let tridiag = Array2::from_shape_fn((10, 10), |(i, j)| {
        if i == j {
            2.0 // Main diagonal
        } else if (i as i32 - j as i32).abs() == 1 {
            -1.0 // Super- and sub-diagonals
        } else {
            0.0
        }
    });

    println!("  Symmetric tridiagonal matrix (discrete Laplacian)");

    // Eigenvalues of symmetric tridiagonal matrices can be computed very efficiently
    let (tridiag_eigenvals, eigenvecs) = scirs2_linalg::eigh(&tridiag.view(), None)?;
    println!(
        "  Eigenvalue range: [{:.6}, {:.6}]",
        tridiag_eigenvals[0],
        tridiag_eigenvals[tridiag_eigenvals.len() - 1]
    );

    println!("\n");
    Ok(())
}

/// Demonstrates adaptive algorithm selection
#[allow(dead_code)]
fn adaptive_algorithms_demo() -> LinalgResult<()> {
    println!("🤖 Adaptive Algorithm Selection");
    println!("{}", "=".repeat(50));

    let sizes_and_thresholds = vec![
        (10, 100),  // Small matrix, below threshold
        (50, 100),  // Medium matrix, above threshold
        (100, 100), // Large matrix, above threshold
    ];

    for &(size, threshold) in &sizes_and_thresholds {
        println!(
            "\n📊 Testing {}x{} matrix (threshold: {})",
            size, size, threshold
        );

        let matrix = Array2::from_shape_fn((size, size), |(i, j)| ((i + 1) * (j + 1)) as f64);
        let vector = Array1::from_shape_fn(size, |i| (i + 1) as f64);

        let config = WorkerConfig::new()
            .with_threshold(threshold)
            .with_workers(4);

        let datasize = size * size;
        let strategy = adaptive::choose_strategy(datasize, &config);

        println!("  Data size: {} elements", datasize);
        println!("  Recommended strategy: {:?}", strategy);

        // Demonstrate adaptive matrix-vector multiplication
        let should_parallelize = adaptive::should_use_parallel(datasize, &config);
        println!("  Use parallel processing: {}", should_parallelize);

        let start = Instant::now();
        let _result = if should_parallelize {
            algorithms::parallel_matvec(&matrix.view(), &vector.view(), &config)?
        } else {
            matrix.dot(&vector)
        };
        let duration = start.elapsed();

        println!("  Execution time: {:?}", duration);

        // Show the adaptive decision making in action
        if should_parallelize {
            println!("  ✅ Used parallel algorithm (data size above threshold)");
        } else {
            println!("  ✅ Used serial algorithm (data size below threshold)");
        }
    }

    // Demonstrate work-stealing effectiveness
    println!("\n🔹 Work-Stealing Effectiveness Demo");

    // Create an unbalanced workload
    let unbalancedmatrix = Array2::from_shape_fn((100, 100), |(i, j)| {
        if i < 50 {
            // Light computation for first half
            (i + j) as f64
        } else {
            // Heavy computation for second half (simulated)
            let mut result = (i + j) as f64;
            for _ in 0..10 {
                result = result.sin().cos();
            }
            result
        }
    });

    let config_balanced = WorkerConfig::new()
        .with_workers(4)
        .with_threshold(1000)
        .with_chunksize(25); // Balanced chunks

    let config_imbalanced = WorkerConfig::new()
        .with_workers(4)
        .with_threshold(1000)
        .with_chunksize(50); // Larger chunks that may cause imbalance

    let vector = Array1::ones(100);

    println!("  Testing load balancing with unbalanced computation...");

    let start = Instant::now();
    let _result1 =
        algorithms::parallel_matvec(&unbalancedmatrix.view(), &vector.view(), &config_balanced)?;
    let balanced_time = start.elapsed();

    let start = Instant::now();
    let _result2 =
        algorithms::parallel_matvec(&unbalancedmatrix.view(), &vector.view(), &config_imbalanced)?;
    let imbalanced_time = start.elapsed();

    println!("  Balanced chunks time:   {:?}", balanced_time);
    println!("  Imbalanced chunks time: {:?}", imbalanced_time);

    if balanced_time < imbalanced_time {
        println!("  ✅ Better load balancing with smaller chunks");
    } else {
        println!("  ⚠️  Chunk size effect not visible (may need larger problem)");
    }

    println!("\n");
    Ok(())
}
