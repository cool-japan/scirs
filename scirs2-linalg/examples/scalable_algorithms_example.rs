//! Scalable Algorithms for Tall-and-Skinny and Short-and-Fat Matrices Example
//!
//! This example demonstrates the comprehensive scalable algorithm capabilities
//! that provide cutting-edge solutions for extreme aspect ratio matrices in
//! modern machine learning, data science, and scientific computing:
//!
//! - Tall-and-Skinny QR (TSQR) for communication-optimal decomposition
//! - LQ decomposition for short-and-fat matrices
//! - Adaptive algorithm selection based on aspect ratio
//! - Blocked matrix operations for memory efficiency
//! - Randomized sketching for dimensionality reduction
//! - Performance analytics and optimization recommendations
//!
//! These algorithms are crucial for big data applications where matrices
//! have millions of rows or columns but extreme aspect ratios.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::scalable::{
    adaptive_decomposition, blocked_matmul, classify_aspect_ratio, lq_decomposition,
    randomized_svd, tsqr, AspectRatio, ScalableConfig,
};
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 SCALABLE ALGORITHMS - Advanced DEMONSTRATION");
    println!("=================================================");

    // Test 1: Aspect Ratio Classification
    println!("\n1. ASPECT RATIO CLASSIFICATION");
    println!("------------------------------");

    let aspect_examples = vec![
        (1000, 50, "Machine Learning Features"),
        (20, 800, "Compressed Sensing"),
        (200, 180, "Nearly Square Matrix"),
        (5000, 10, "Time Series Data"),
        (5, 2000, "Genomics Data"),
    ];

    for (m, n, description) in aspect_examples {
        let dummymatrix = Array2::<f64>::zeros((m, n));
        let aspect = classify_aspect_ratio(&dummymatrix.view(), 4.0);
        println!("   {} ({}×{}): {:?}", description, m, n, aspect);
    }

    // Test 2: Tall-and-Skinny QR (TSQR)
    println!("\n2. TALL-AND-SKINNY QR (TSQR): Communication-Optimal Decomposition");
    println!("------------------------------------------------------------------");

    // Create a tall-and-skinny matrix representing feature vectors
    let m_tall = 2000;
    let n_tall = 25;
    let tallmatrix = Array2::from_shape_fn((m_tall, n_tall), |(i, j)| {
        // Simulate feature matrix with some structure
        let freq1 = 2.0 * std::f64::consts::PI * (i as f64) / 100.0;
        let freq2 = 2.0 * std::f64::consts::PI * (j as f64) / 10.0;
        freq1.sin() + 0.5 * freq2.cos() + 0.1 * (i + j) as f64 / 1000.0
    });

    println!(
        "   Matrix dimensions: {}×{} (aspect ratio: {:.1})",
        m_tall,
        n_tall,
        m_tall as f64 / n_tall as f64
    );

    let config = ScalableConfig::default().with_blocksize(512);

    let start_time = Instant::now();
    let (q, r) = tsqr(&tallmatrix.view(), &config)?;
    let tsqr_time = start_time.elapsed();

    println!(
        "   TSQR computation time: {:.2}ms",
        tsqr_time.as_nanos() as f64 / 1_000_000.0
    );
    println!("   Q dimensions: {}×{}", q.nrows(), q.ncols());
    println!("   R dimensions: {}×{}", r.nrows(), r.ncols());

    // Verify orthogonality of Q
    let qtq = q.t().dot(&q);
    let identity = Array2::<f64>::eye(n_tall);
    let mut max_orthogonality_error = 0.0;
    for i in 0..n_tall {
        for j in 0..n_tall {
            let error = (qtq[[i, j]] - identity[[i, j]]).abs();
            if error > max_orthogonality_error {
                max_orthogonality_error = error;
            }
        }
    }
    println!("   Orthogonality error: {:.2e}", max_orthogonality_error);

    // Verify reconstruction: A ≈ Q*R
    let reconstructed = q.dot(&r);
    let mut max_reconstruction_error = 0.0;
    for i in 0..m_tall.min(100) {
        // Check first 100 rows for efficiency
        for j in 0..n_tall {
            let error = (tallmatrix[[i, j]] - reconstructed[[i, j]]).abs();
            if error > max_reconstruction_error {
                max_reconstruction_error = error;
            }
        }
    }
    println!("   Reconstruction error: {:.2e}", max_reconstruction_error);
    println!("   ✅ TSQR provides communication-optimal O(n²) complexity");

    // Test 3: LQ Decomposition for Short-and-Fat Matrices
    println!("\n3. LQ DECOMPOSITION: Optimal for Short-and-Fat Matrices");
    println!("-------------------------------------------------------");

    let m_short = 20;
    let n_short = 1000;
    let shortmatrix = Array2::from_shape_fn((m_short, n_short), |(i, j)| {
        // Simulate compressed sensing or genomics data
        let signal = (j as f64 / 50.0).sin() * (i as f64 + 1.0);
        signal + 0.01 * (i * j) as f64 / 10000.0
    });

    println!(
        "   Matrix dimensions: {}×{} (aspect ratio: {:.1})",
        m_short,
        n_short,
        m_short as f64 / n_short as f64
    );

    let start_time = Instant::now();
    let (l, q) = lq_decomposition(&shortmatrix.view(), &config)?;
    let lq_time = start_time.elapsed();

    println!(
        "   LQ computation time: {:.2}ms",
        lq_time.as_nanos() as f64 / 1_000_000.0
    );
    println!("   L dimensions: {}×{}", l.nrows(), l.ncols());
    println!("   Q dimensions: {}×{}", q.nrows(), q.ncols());

    // Verify L is lower triangular
    let mut max_upper_triangular = 0.0;
    for i in 0..m_short {
        for j in (i + 1)..m_short {
            let val = l[[i, j]].abs();
            if val > max_upper_triangular {
                max_upper_triangular = val;
            }
        }
    }
    println!("   Lower triangular error: {:.2e}", max_upper_triangular);

    // Verify reconstruction: A ≈ L*Q
    let lq_reconstructed = l.dot(&q);
    let mut max_lq_error = 0.0;
    for i in 0..m_short {
        for j in 0..n_short.min(100) {
            // Check first 100 columns for efficiency
            let error = (shortmatrix[[i, j]] - lq_reconstructed[[i, j]]).abs();
            if error > max_lq_error {
                max_lq_error = error;
            }
        }
    }
    println!("   Reconstruction error: {:.2e}", max_lq_error);
    println!("   ✅ LQ decomposition optimal for overdetermined systems");

    // Test 4: Adaptive Algorithm Selection
    println!("\n4. ADAPTIVE ALGORITHM SELECTION: Smart Optimization");
    println!("---------------------------------------------------");

    let test_matrices = vec![
        (
            "Tall Feature Matrix",
            Array2::from_shape_fn((800, 15), |(i, j)| (i + j + 1) as f64),
        ),
        (
            "Short Genomics Matrix",
            Array2::from_shape_fn((12, 600), |(i, j)| (i * j + 1) as f64),
        ),
        (
            "Square-ish Matrix",
            Array2::from_shape_fn((120, 100), |(i, j)| (i + j + 1) as f64),
        ),
    ];

    for (description, matrix) in test_matrices {
        println!("\n   Testing: {}", description);
        let (m, n) = matrix.dim();

        let result = adaptive_decomposition(&matrix.view(), &config)?;

        println!("     Detected aspect ratio: {:?}", result.aspect_ratio);
        println!("     Selected algorithm: {}", result.algorithm_used);
        println!(
            "     Complexity estimate: {} FLOPs",
            result.complexity_estimate
        );
        println!(
            "     Memory estimate: {:.1} KB",
            result.memory_estimate as f64 / 1024.0
        );

        let metrics = &result.performance_metrics;
        println!(
            "     Communication volume: {} elements",
            metrics.communication_volume
        );
        println!(
            "     Cache efficiency: {:.1}%",
            metrics.cache_efficiency * 100.0
        );
        println!(
            "     Memory bandwidth: {:.1} MB/s",
            metrics.memory_bandwidth
        );

        // Verify decomposition quality
        let reconstruction = result.factor1.dot(&result.factor2);
        let mut max_error = 0.0;
        for i in 0..m.min(50) {
            for j in 0..n.min(50) {
                let error = (matrix[[i, j]] - reconstruction[[i, j]]).abs();
                if error > max_error {
                    max_error = error;
                }
            }
        }
        println!("     Reconstruction error: {:.2e}", max_error);
    }

    // Test 5: Blocked Matrix Multiplication
    println!("\n5. BLOCKED MATRIX MULTIPLICATION: Memory-Efficient Operations");
    println!("------------------------------------------------------------");

    let asize = (400, 200);
    let bsize = (200, 300);

    let matrix_a = Array2::from_shape_fn(asize, |(i, j)| ((i + j + 1) as f64).sin() / 100.0);
    let matrix_b = Array2::from_shape_fn(bsize, |(i, j)| ((i * j + 1) as f64).cos() / 100.0);

    println!(
        "   Matrix A: {}×{}, Matrix B: {}×{}",
        asize.0, asize.1, bsize.0, bsize.1
    );

    // Compare blocked vs standard multiplication
    let config_small_blocks = config.clone().with_blocksize(64);

    let start_time = Instant::now();
    let result_blocked = blocked_matmul(&matrix_a.view(), &matrix_b.view(), &config_small_blocks)?;
    let blocked_time = start_time.elapsed();

    let start_time = Instant::now();
    let result_standard = matrix_a.dot(&matrix_b);
    let standard_time = start_time.elapsed();

    println!(
        "   Blocked multiplication time: {:.2}ms",
        blocked_time.as_nanos() as f64 / 1_000_000.0
    );
    println!(
        "   Standard multiplication time: {:.2}ms",
        standard_time.as_nanos() as f64 / 1_000_000.0
    );

    // Verify results are identical
    let mut max_diff = 0.0;
    for i in 0..asize.0 {
        for j in 0..bsize.1 {
            let diff = (result_blocked[[i, j]] - result_standard[[i, j]]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    println!("   Maximum difference: {:.2e}", max_diff);
    println!("   ✅ Blocked algorithm provides same results with better cache efficiency");

    // Test 6: Randomized SVD for Low-Rank Approximation
    println!("\n6. RANDOMIZED SVD: Efficient Low-Rank Approximation");
    println!("---------------------------------------------------");

    // Create a low-rank matrix for demonstration
    let rank_true = 10;
    let u_true =
        Array2::from_shape_fn((300, rank_true), |(i, j)| ((i + j + 1) as f64).sin() / 10.0);
    let s_true = Array1::from_shape_fn(rank_true, |i| {
        10.0 * (-(i as f64) / 2.0).exp() // Exponentially decaying singular values
    });
    let vt_true =
        Array2::from_shape_fn((rank_true, 200), |(i, j)| ((i * j + 1) as f64).cos() / 10.0);

    // Construct low-rank matrix: A = U * S * V^T
    let smatrix = Array2::from_diag(&s_true);
    let us = u_true.dot(&smatrix);
    let low_rankmatrix = us.dot(&vt_true);

    println!(
        "   Original matrix: {}×{} (true rank: {})",
        low_rankmatrix.nrows(),
        low_rankmatrix.ncols(),
        rank_true
    );

    let target_rank = 8;
    let config_randomized = config.clone().with_oversampling(4);

    let start_time = Instant::now();
    let (u_approx, s_approx, vt_approx) =
        randomized_svd(&low_rankmatrix.view(), target_rank, &config_randomized)?;
    let randomized_time = start_time.elapsed();

    println!(
        "   Randomized SVD time: {:.2}ms",
        randomized_time.as_nanos() as f64 / 1_000_000.0
    );
    println!("   Approximation rank: {}", target_rank);
    println!(
        "   Computed singular values: {:?}",
        s_approx
            .iter()
            .take(5)
            .map(|&x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );

    // Compute approximation error
    let s_diag = Array2::from_diag(&s_approx);
    let approximation = u_approx.dot(&s_diag).dot(&vt_approx);

    let mut approximation_error = 0.0;
    for i in 0..low_rankmatrix.nrows() {
        for j in 0..low_rankmatrix.ncols() {
            let error = (low_rankmatrix[[i, j]] - approximation[[i, j]]).powi(2);
            approximation_error += error;
        }
    }
    approximation_error = approximation_error.sqrt();

    let matrix_norm = low_rankmatrix.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let relative_error = approximation_error / matrix_norm;

    println!("   Relative approximation error: {:.2e}", relative_error);
    println!("   ✅ Randomized SVD provides efficient low-rank approximation");

    // Test 7: Performance Scaling Analysis
    println!("\n7. PERFORMANCE SCALING ANALYSIS");
    println!("-------------------------------");

    let testsizes = vec![
        (500, 10, "Small tall"),
        (1000, 20, "Medium tall"),
        (2000, 40, "Large tall"),
        (10, 500, "Small fat"),
        (20, 1000, "Medium fat"),
        (40, 2000, "Large fat"),
    ];

    println!("   Scaling analysis for different matrix sizes:");
    println!("   Size        Aspect     Algorithm      Time (ms)  Complexity");

    for (m, n, _description) in testsizes {
        let testmatrix =
            Array2::from_shape_fn((m, n), |(i, j)| (i + j + 1) as f64 / (m + n) as f64);

        let start_time = Instant::now();
        let result = adaptive_decomposition(&testmatrix.view(), &config)?;
        let elapsed = start_time.elapsed();

        let aspect_str = match result.aspect_ratio {
            AspectRatio::TallSkinny => "Tall",
            AspectRatio::ShortFat => "Fat",
            AspectRatio::Square => "Square",
        };

        let algorithm_short = if result.algorithm_used.contains("TSQR") {
            "TSQR"
        } else if result.algorithm_used.contains("LQ") {
            "LQ"
        } else {
            "QR"
        };

        println!(
            "   {}×{:<4} {:<10} {:<14} {:>8.1}  {:>10}",
            m,
            n,
            aspect_str,
            algorithm_short,
            elapsed.as_nanos() as f64 / 1_000_000.0,
            result.complexity_estimate
        );
    }

    // Test 8: Big Data Applications Summary
    println!("\n8. BIG DATA APPLICATIONS");
    println!("------------------------");

    println!("   ✅ MACHINE LEARNING:");
    println!("      - Feature matrices in deep learning (tall-and-skinny)");
    println!("      - Mini-batch processing with TSQR");
    println!("      - Dimensionality reduction with randomized SVD");
    println!("      - Gradient computation in large-scale optimization");

    println!("   ✅ DATA SCIENCE:");
    println!("      - Time series analysis with tall feature vectors");
    println!("      - Principal component analysis on high-dimensional data");
    println!("      - Least squares regression with many samples");
    println!("      - Matrix completion and collaborative filtering");

    println!("   ✅ SCIENTIFIC COMPUTING:");
    println!("      - Compressed sensing and sparse recovery");
    println!("      - Genomics and bioinformatics (wide matrices)");
    println!("      - Climate modeling with spatial-temporal data");
    println!("      - Quantum chemistry and molecular dynamics");

    println!("   ✅ ENGINEERING:");
    println!("      - Signal processing and communications");
    println!("      - Computer vision and image processing");
    println!("      - Control systems and state estimation");
    println!("      - Finite element analysis and simulation");

    // Test 9: Algorithm Selection Guidelines
    println!("\n9. ALGORITHM SELECTION GUIDELINES");
    println!("---------------------------------");

    println!("   📊 ASPECT RATIO THRESHOLDS:");
    println!("      - Tall-and-skinny: height/width ≥ 4.0 → Use TSQR");
    println!("      - Short-and-fat: height/width ≤ 0.25 → Use LQ decomposition");
    println!("      - Square-ish: 0.25 < height/width < 4.0 → Use standard QR");

    println!("   🚀 PERFORMANCE OPTIMIZATIONS:");
    println!("      - Block size: 256-512 for cache efficiency");
    println!("      - TSQR: Optimal for m >> n, reduces communication O(n²)");
    println!("      - LQ: Natural for m << n, efficient least-norm solutions");
    println!("      - Randomized SVD: Oversampling 5-15 for numerical stability");

    println!("   💾 MEMORY CONSIDERATIONS:");
    println!("      - TSQR: ~3mn memory for tall matrices");
    println!("      - LQ: ~2mn memory for fat matrices");
    println!("      - Blocked operations: Configurable memory footprint");
    println!("      - Randomized methods: Reduced memory for low-rank approximation");

    println!("\n========================================================");
    println!("🎯 Advanced ACHIEVEMENT: SCALABLE ALGORITHMS COMPLETE");
    println!("========================================================");
    println!("✅ Tall-and-Skinny QR (TSQR): Communication-optimal O(n²) complexity");
    println!("✅ LQ decomposition: Optimal for short-and-fat matrices");
    println!("✅ Adaptive selection: Smart algorithm choice based on aspect ratio");
    println!("✅ Blocked operations: Memory-efficient for massive matrices");
    println!("✅ Randomized SVD: Probabilistic low-rank approximation");
    println!("✅ Performance analytics: Comprehensive optimization metrics");
    println!("✅ Big data ready: Designed for millions of rows/columns");
    println!("========================================================");

    Ok(())
}
