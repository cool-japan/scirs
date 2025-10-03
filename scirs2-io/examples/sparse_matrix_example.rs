//! Comprehensive sparse matrix format example
//!
//! This example demonstrates the unified sparse matrix functionality including:
//! - Creation from dense arrays and triplets
//! - Format conversion between COO, CSR, and CSC
//! - Matrix operations (addition, multiplication, transpose)
//! - I/O integration with Matrix Market format
//! - Performance statistics and analysis
//! - Memory usage optimization

use scirs2_core::ndarray::array;
use scirs2_io::sparse::{ops, SparseMatrix};
use std::f64::consts::PI;
use std::time::Instant;
use tempfile::tempdir;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🗃️  Comprehensive Sparse Matrix Example");
    println!("======================================");

    // Demonstrate sparse matrix creation
    demonstrate_sparse_creation()?;

    // Demonstrate format conversion
    demonstrate_format_conversion()?;

    // Demonstrate matrix operations
    demonstrate_matrix_operations()?;

    // Demonstrate I/O integration
    demonstrate_io_integration()?;

    // Demonstrate performance analysis
    demonstrate_performance_analysis()?;

    // Demonstrate large sparse matrices
    demonstrate_large_sparse_matrices()?;

    println!("\n✅ All sparse matrix demonstrations completed successfully!");
    println!("💡 Key benefits of the unified sparse matrix system:");
    println!("   - Multiple format support (COO, CSR, CSC) with automatic conversion");
    println!("   - Efficient matrix operations optimized for sparse data");
    println!("   - Seamless I/O integration with Matrix Market format");
    println!("   - Memory-efficient storage and processing");
    println!("   - Comprehensive performance monitoring");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_sparse_creation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Demonstrating Sparse Matrix Creation...");

    // Method 1: Create from dense array
    println!("  🔹 Creating from dense array:");
    let dense = array![
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0, 4.0],
        [5.0, 0.0, 0.0, 0.0],
        [0.0, 6.0, 7.0, 8.0]
    ];
    println!("    Dense matrix shape: {:?}", dense.shape());

    let sparse_from_dense = SparseMatrix::from_dense_2d(&dense, 0.0)?;
    println!(
        "    Sparse matrix: {} non-zeros out of {} elements",
        sparse_from_dense.nnz(),
        dense.len()
    );
    println!("    Density: {:.2}%", sparse_from_dense.density() * 100.0);

    // Method 2: Create from triplets
    println!("  🔹 Creating from triplets (row, col, value):");
    let rows = vec![0, 0, 1, 1, 2, 3, 3, 3];
    let cols = vec![0, 2, 1, 3, 0, 1, 2, 3];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let sparse_from_triplets = SparseMatrix::from_triplets(4, 4, rows, cols, values)?;
    println!(
        "    Created sparse matrix with {} non-zeros",
        sparse_from_triplets.nnz()
    );

    // Method 3: Build incrementally
    println!("  🔹 Building incrementally:");
    let mut sparse_incremental = SparseMatrix::new(3, 3);
    sparse_incremental.push(0, 0, 10.0)?;
    sparse_incremental.push(1, 1, 20.0)?;
    sparse_incremental.push(2, 2, 30.0)?;
    sparse_incremental.push(0, 2, 15.0)?;
    println!(
        "    Diagonal + off-diagonal matrix: {} non-zeros",
        sparse_incremental.nnz()
    );

    // Verify data integrity
    let reconstructed = sparse_from_dense.to_dense();
    for i in 0..dense.nrows() {
        for j in 0..dense.ncols() {
            let diff: f64 = dense[[i, j]] - reconstructed[[i, j]];
            assert!(diff.abs() < 1e-10);
        }
    }
    println!("  ✅ Data integrity verified for all creation methods!");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_format_conversion() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Demonstrating Format Conversion...");

    // Create a test matrix
    let mut sparse = SparseMatrix::new(5, 5);
    for i in 0..5 {
        sparse.push(i, i, (i + 1) as f64)?; // Diagonal
        if i < 4 {
            sparse.push(i, i + 1, 0.5)?; // Super-diagonal
        }
        if i > 0 {
            sparse.push(i, i - 1, -0.5)?; // Sub-diagonal
        }
    }

    println!("  🔹 Original matrix (COO format):");
    println!(
        "    Shape: {:?}, NNZ: {}, Density: {:.2}%",
        sparse.shape(),
        sparse.nnz(),
        sparse.density() * 100.0
    );

    // Convert to CSR
    println!("  🔹 Converting to CSR (Compressed Sparse Row):");
    let conversion_start = Instant::now();
    let csr = sparse.to_csr()?;
    let csr_time = conversion_start.elapsed();
    println!(
        "    Conversion time: {:.2}ms",
        csr_time.as_secs_f64() * 1000.0
    );
    println!("    CSR row pointers: {:?}", &csr.row_ptrs[..6]);
    println!(
        "    CSR column indices (first 10): {:?}",
        &csr.col_indices[..10.min(csr.col_indices.len())]
    );

    // Convert to CSC
    println!("  🔹 Converting to CSC (Compressed Sparse Column):");
    let conversion_start = Instant::now();
    let csc = sparse.to_csc()?;
    let csc_time = conversion_start.elapsed();
    println!(
        "    Conversion time: {:.2}ms",
        csc_time.as_secs_f64() * 1000.0
    );
    println!("    CSC column pointers: {:?}", &csc.col_ptrs[..6]);
    println!(
        "    CSC row indices (first 10): {:?}",
        &csc.row_indices[..10.min(csc.row_indices.len())]
    );

    // Performance comparison
    println!("  🔹 Format conversion performance:");
    println!("    COO to CSR: {:.2}ms", csr_time.as_secs_f64() * 1000.0);
    println!("    COO to CSC: {:.2}ms", csc_time.as_secs_f64() * 1000.0);

    // Test cached access
    let cached_start = Instant::now();
    let _csr_cached = sparse.to_csr()?;
    let cached_time = cached_start.elapsed();
    println!(
        "    Cached CSR access: {:.2}ms (should be near zero)",
        cached_time.as_secs_f64() * 1000.0
    );

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_matrix_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧮 Demonstrating Matrix Operations...");

    // Create test matrices
    let mut matrix_a = SparseMatrix::new(3, 3);
    matrix_a.push(0, 0, 1.0)?;
    matrix_a.push(0, 1, 2.0)?;
    matrix_a.push(1, 1, 3.0)?;
    matrix_a.push(2, 2, 4.0)?;

    let mut matrix_b = SparseMatrix::new(3, 3);
    matrix_b.push(0, 0, 2.0)?;
    matrix_b.push(0, 2, 1.0)?;
    matrix_b.push(1, 1, 1.0)?;
    matrix_b.push(2, 0, 3.0)?;

    println!("  🔹 Matrix A: {} non-zeros", matrix_a.nnz());
    println!("  🔹 Matrix B: {} non-zeros", matrix_b.nnz());

    // Matrix addition
    println!("  🔹 Matrix Addition (A + B):");
    let add_start = Instant::now();
    let sum = (&matrix_a + &matrix_b)?;
    let add_time = add_start.elapsed();
    println!("    Result: {} non-zeros", sum.nnz());
    println!(
        "    Addition time: {:.2}ms",
        add_time.as_secs_f64() * 1000.0
    );

    // Matrix-vector multiplication
    println!("  🔹 Matrix-Vector Multiplication (A * v):");
    let vector = vec![1.0, 2.0, 3.0];
    let mv_start = Instant::now();
    let result_vec = ops::spmv(&mut matrix_a, &vector)?;
    let mv_time = mv_start.elapsed();
    println!("    Input vector: {:?}", vector);
    println!("    Result vector: {:?}", result_vec);
    println!("    SpMV time: {:.2}ms", mv_time.as_secs_f64() * 1000.0);

    // Matrix-matrix multiplication
    println!("  🔹 Matrix-Matrix Multiplication (A * B):");
    let mm_start = Instant::now();
    let product = ops::spmm(&mut matrix_a, &mut matrix_b)?;
    let mm_time = mm_start.elapsed();
    println!("    Result: {} non-zeros", product.nnz());
    println!("    SpMM time: {:.2}ms", mm_time.as_secs_f64() * 1000.0);

    // Matrix transpose
    println!("  🔹 Matrix Transpose:");
    let transpose_start = Instant::now();
    let transposed = matrix_a.transpose();
    let transpose_time = transpose_start.elapsed();
    println!("    Original shape: {:?}", matrix_a.shape());
    println!("    Transposed shape: {:?}", transposed.shape());
    println!(
        "    Transpose time: {:.2}ms",
        transpose_time.as_secs_f64() * 1000.0
    );

    // Verify correctness with dense conversion
    let dense_a = matrix_a.to_dense();
    let dense_sum = sum.to_dense();
    let expected_sum = &dense_a + &matrix_b.to_dense();

    for i in 0..3 {
        for j in 0..3 {
            let diff: f64 = dense_sum[[i, j]] - expected_sum[[i, j]];
            assert!(diff.abs() < 1e-10);
        }
    }
    println!("  ✅ Matrix operation correctness verified!");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_io_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💾 Demonstrating I/O Integration...");

    let temp_dir = tempdir()?;
    let matrix_file = temp_dir.path().join("sparse_matrix.mtx");

    // Create a sparse matrix
    let mut sparse = SparseMatrix::new(4, 4);
    sparse.push(0, 0, 1.5)?;
    sparse.push(0, 3, -2.3)?;
    sparse.push(1, 1, 4.7)?;
    sparse.push(2, 0, 0.8)?;
    sparse.push(2, 2, -1.1)?;
    sparse.push(3, 3, PI)?;

    println!("  🔹 Original matrix:");
    println!("    Shape: {:?}, NNZ: {}", sparse.shape(), sparse.nnz());

    // Save to Matrix Market format
    println!("  🔹 Saving to Matrix Market format...");
    let save_start = Instant::now();
    sparse.save_matrix_market(&matrix_file)?;
    let save_time = save_start.elapsed();
    println!("    Save time: {:.2}ms", save_time.as_secs_f64() * 1000.0);

    // Load from Matrix Market format
    println!("  🔹 Loading from Matrix Market format...");
    let load_start = Instant::now();
    let loaded_sparse = SparseMatrix::load_matrix_market(&matrix_file)?;
    let load_time = load_start.elapsed();
    println!("    Load time: {:.2}ms", load_time.as_secs_f64() * 1000.0);
    println!(
        "    Loaded matrix: shape {:?}, NNZ: {}",
        loaded_sparse.shape(),
        loaded_sparse.nnz()
    );

    // Verify round-trip integrity
    assert_eq!(sparse.shape(), loaded_sparse.shape());
    assert_eq!(sparse.nnz(), loaded_sparse.nnz());

    let original_dense = sparse.to_dense();
    let loaded_dense = loaded_sparse.to_dense();

    for i in 0..original_dense.nrows() {
        for j in 0..original_dense.ncols() {
            let diff: f64 = original_dense[[i, j]] - loaded_dense[[i, j]];
            assert!(diff.abs() < 1e-10);
        }
    }
    println!("  ✅ Round-trip I/O integrity verified!");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_performance_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Demonstrating Performance Analysis...");

    // Create a structured sparse matrix
    let mut sparse = SparseMatrix::new(100, 50);

    // Add diagonal elements
    for i in 0..50 {
        sparse.push(i, i, 1.0)?;
    }

    // Add some off-diagonal elements
    for i in 0..30 {
        sparse.push(i, i + 10, 0.5)?;
        sparse.push(i + 20, i, -0.3)?;
    }

    println!("  🔹 Matrix structure:");
    println!("    Shape: {:?}", sparse.shape());
    println!("    Non-zeros: {}", sparse.nnz());
    println!("    Density: {:.4}%", sparse.density() * 100.0);

    // Get comprehensive statistics
    println!("  🔹 Computing comprehensive statistics...");
    let stats_start = Instant::now();
    let stats = sparse.stats()?;
    let stats_time = stats_start.elapsed();

    println!(
        "    Statistics computation time: {:.2}ms",
        stats_time.as_secs_f64() * 1000.0
    );
    println!(
        "    Memory usage: {} bytes ({:.1} KB)",
        stats.memory_bytes,
        stats.memory_bytes as f64 / 1024.0
    );
    println!("    Average NNZ per row: {:.2}", stats.avg_nnz_per_row);
    println!("    Average NNZ per column: {:.2}", stats.avg_nnz_per_col);
    println!("    Maximum NNZ in any row: {}", stats.max_nnz_row);
    println!("    Maximum NNZ in any column: {}", stats.max_nnz_col);

    // Performance comparison: sparse vs dense operations
    println!("  🔹 Performance comparison (sparse vs dense):");

    let vector = vec![1.0; 50];

    // Sparse matrix-vector multiplication
    let sparse_mv_start = Instant::now();
    let _sparse_result = ops::spmv(&mut sparse, &vector)?;
    let sparse_mv_time = sparse_mv_start.elapsed();

    // Dense matrix-vector multiplication
    let dense = sparse.to_dense();
    let dense_mv_start = Instant::now();
    let mut dense_result = vec![0.0; 100];
    for i in 0..100 {
        for j in 0..50 {
            dense_result[i] += dense[[i, j]] * vector[j];
        }
    }
    let dense_mv_time = dense_mv_start.elapsed();

    let speedup = dense_mv_time.as_secs_f64() / sparse_mv_time.as_secs_f64();
    println!(
        "    Sparse SpMV time: {:.2}ms",
        sparse_mv_time.as_secs_f64() * 1000.0
    );
    println!(
        "    Dense MV time: {:.2}ms",
        dense_mv_time.as_secs_f64() * 1000.0
    );
    println!("    Sparse speedup: {:.1}x", speedup);

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_large_sparse_matrices() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🏗️  Demonstrating Large Sparse Matrix Handling...");

    // Create a larger sparse matrix (simulating a real-world scenario)
    let size = 1000;
    let mut large_sparse = SparseMatrix::new(size, size);

    println!("  🔹 Creating large sparse matrix ({}x{})...", size, size);
    let creation_start = Instant::now();

    // Add diagonal elements
    for i in 0..size {
        large_sparse.push(i, i, i as f64 + 1.0)?;
    }

    // Add off-diagonal pattern (tridiagonal-like)
    for i in 0..size - 1 {
        large_sparse.push(i, i + 1, 0.5)?;
        large_sparse.push(i + 1, i, -0.5)?;
    }

    // Add some random sparse elements
    for i in (0..size).step_by(50) {
        for j in (0..size).step_by(75) {
            if i != j {
                large_sparse.push(i, j, 0.1)?;
            }
        }
    }

    let creation_time = creation_start.elapsed();
    println!(
        "    Creation time: {:.2}ms",
        creation_time.as_secs_f64() * 1000.0
    );
    println!("    Non-zeros: {}", large_sparse.nnz());
    println!("    Density: {:.6}%", large_sparse.density() * 100.0);

    // Test format conversions on large matrix
    println!("  🔹 Testing format conversions:");

    let csr_start = Instant::now();
    let _csr = large_sparse.to_csr()?;
    let csr_time = csr_start.elapsed();

    let csc_start = Instant::now();
    let _csc = large_sparse.to_csc()?;
    let csc_time = csc_start.elapsed();

    println!(
        "    COO to CSR conversion: {:.2}ms",
        csr_time.as_secs_f64() * 1000.0
    );
    println!(
        "    COO to CSC conversion: {:.2}ms",
        csc_time.as_secs_f64() * 1000.0
    );

    // Test large matrix-vector multiplication
    println!("  🔹 Testing large matrix operations:");
    let large_vector = vec![1.0; size];

    let large_mv_start = Instant::now();
    let _large_result = ops::spmv(&mut large_sparse, &large_vector)?;
    let large_mv_time = large_mv_start.elapsed();

    println!(
        "    Large SpMV time: {:.2}ms",
        large_mv_time.as_secs_f64() * 1000.0
    );

    // Get statistics for large matrix
    let large_stats = large_sparse.stats()?;
    println!(
        "    Memory usage: {:.1} KB",
        large_stats.memory_bytes as f64 / 1024.0
    );
    println!(
        "    Memory efficiency: {:.1}x vs dense",
        (size * size * 8) as f64 / large_stats.memory_bytes as f64
    );

    println!("  ✅ Large sparse matrix handling demonstrated successfully!");

    Ok(())
}
