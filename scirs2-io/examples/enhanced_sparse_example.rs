//! Enhanced sparse matrix serialization example
//!
//! This example demonstrates the improved sparse matrix capabilities:
//! - Multiple sparse matrix formats (COO, CSR, CSC)
//! - Format conversion and caching
//! - Integration with Matrix Market format
//! - Sparse matrix operations (addition, multiplication, transpose)
//! - Memory efficiency analysis
//! - Performance comparison between formats

use scirs2_io::error::Result;
use scirs2_io::serialize::{
    deserialize_enhanced_sparse_matrix, serialize_enhanced_sparse_matrix, sparse_ops,
    SerializationFormat, SparseMatrix,
};
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== Enhanced Sparse Matrix Serialization Example ===");

    // Example 1: Create and work with different sparse matrix formats
    demonstrate_sparse_formats()?;

    // Example 2: Format conversion and performance
    demonstrate_format_conversion()?;

    // Example 3: Sparse matrix operations
    demonstrate_sparse_operations()?;

    // Example 4: Serialization and compression
    demonstrate_serialization()?;

    // Example 5: Memory efficiency analysis
    demonstrate_memory_analysis()?;

    println!("Enhanced sparse matrix example completed successfully!");
    Ok(())
}

fn demonstrate_sparse_formats() -> Result<()> {
    println!("\n1. Demonstrating different sparse matrix formats...");

    // Create a sparse matrix with a pattern (tridiagonal matrix)
    let size = 1000;
    let mut sparse = SparseMatrix::new(size, size);

    println!("  Creating tridiagonal matrix ({}x{})...", size, size);

    // Add main diagonal
    for i in 0..size {
        sparse.insert(i, i, 2.0_f64);
    }

    // Add super-diagonal
    for i in 0..size - 1 {
        sparse.insert(i, i + 1, -1.0_f64);
    }

    // Add sub-diagonal
    for i in 1..size {
        sparse.insert(i, i - 1, -1.0_f64);
    }

    println!("    Matrix properties:");
    println!(
        "      Dimensions: {}x{}",
        sparse.shape().0,
        sparse.shape().1
    );
    println!("      Non-zeros: {}", sparse.nnz());
    println!("      Sparsity: {:.4}%", sparse.sparsity() * 100.0);
    println!("      Memory usage: {} bytes", sparse.memory_usage());

    // Convert to different formats and measure time
    println!("  Converting to different formats...");

    let start = Instant::now();
    let _csr = sparse.to_csr()?;
    let csr_time = start.elapsed();
    println!("    COO -> CSR conversion: {:?}", csr_time);

    let start = Instant::now();
    let _csc = sparse.to_csc()?;
    let csc_time = start.elapsed();
    println!("    COO -> CSC conversion: {:?}", csc_time);

    // Memory usage after conversion
    println!(
        "    Memory usage after conversions: {} bytes",
        sparse.memory_usage()
    );

    Ok(())
}

fn demonstrate_format_conversion() -> Result<()> {
    println!("\n2. Demonstrating format conversion performance...");

    // Create different sized matrices for performance testing
    let sizes = vec![100, 500, 1000];

    for size in sizes {
        println!("  Testing {}x{} matrix:", size, size);

        let mut sparse = create_sample_sparse_matrix(size);

        // Time COO -> CSR conversion
        let start = Instant::now();
        let csr = sparse.to_csr()?;
        let csr_time = start.elapsed();
        let csr_nnz = csr.nnz();

        // Test row access in CSR
        let start = Instant::now();
        for i in 0..std::cmp::min(100, size) {
            if let Some((cols, _vals)) = csr.row(i) {
                let _row_nnz = cols.len();
            }
        }
        let row_access_time = start.elapsed();

        // Drop CSR reference before getting CSC
        let _ = csr;

        // Time COO -> CSC conversion
        let start = Instant::now();
        let csc = sparse.to_csc()?;
        let csc_time = start.elapsed();
        let csc_nnz = csc.nnz();

        // Test column access in CSC
        let start = Instant::now();
        for j in 0..std::cmp::min(100, size) {
            if let Some((rows, _vals)) = csc.column(j) {
                let _col_nnz = rows.len();
            }
        }
        let col_access_time = start.elapsed();

        println!("    COO -> CSR: {:?} ({} nnz)", csr_time, csr_nnz);
        println!("    COO -> CSC: {:?} ({} nnz)", csc_time, csc_nnz);
        println!("    CSR row access (100 rows): {:?}", row_access_time);
        println!("    CSC column access (100 cols): {:?}", col_access_time);
        println!();
    }

    Ok(())
}

fn demonstrate_sparse_operations() -> Result<()> {
    println!("3. Demonstrating sparse matrix operations...");

    // Create two small sparse matrices for operations
    let size = 100;
    let mut matrix_a = create_sample_sparse_matrix(size);
    let mut matrix_b = create_sample_sparse_matrix(size);

    // Modify matrix B to make it different
    for i in 0..size / 2 {
        matrix_b.insert(i, i, 0.5);
    }

    println!(
        "  Matrix A: {}x{} with {} non-zeros",
        matrix_a.shape().0,
        matrix_a.shape().1,
        matrix_a.nnz()
    );
    println!(
        "  Matrix B: {}x{} with {} non-zeros",
        matrix_b.shape().0,
        matrix_b.shape().1,
        matrix_b.nnz()
    );

    // Test matrix addition
    println!("  Testing sparse matrix addition...");
    let start = Instant::now();
    let sum_result = sparse_ops::add_coo(&matrix_a.coo_data, &matrix_b.coo_data)?;
    let add_time = start.elapsed();
    println!("    Addition completed in {:?}", add_time);
    println!(
        "    Result: {}x{} with {} non-zeros",
        sum_result.rows,
        sum_result.cols,
        sum_result.nnz()
    );

    // Test matrix transpose
    println!("  Testing matrix transpose...");
    let start = Instant::now();
    let transpose_result = sparse_ops::transpose_coo(&matrix_a.coo_data);
    let transpose_time = start.elapsed();
    println!("    Transpose completed in {:?}", transpose_time);
    println!(
        "    Original: {}x{}, Transposed: {}x{}",
        matrix_a.shape().0,
        matrix_a.shape().1,
        transpose_result.rows,
        transpose_result.cols
    );

    // Test matrix-vector multiplication
    println!("  Testing matrix-vector multiplication...");
    let vector = vec![1.0_f64; size];

    let csr = matrix_a.to_csr()?;
    let start = Instant::now();
    let matvec_result = sparse_ops::csr_matvec(csr, &vector)?;
    let matvec_time = start.elapsed();

    println!(
        "    Matrix-vector multiplication completed in {:?}",
        matvec_time
    );
    println!("    Result vector length: {}", matvec_result.len());

    // Show some results
    if matvec_result.len() >= 5 {
        println!("    First 5 elements: {:?}", &matvec_result[0..5]);
    }

    Ok(())
}

fn demonstrate_serialization() -> Result<()> {
    println!("\n4. Demonstrating sparse matrix serialization...");

    let size = 500;
    let mut sparse = create_sample_sparse_matrix(size);

    // Add metadata
    sparse.metadata.insert(
        "description".to_string(),
        "Test tridiagonal matrix".to_string(),
    );
    sparse.metadata.insert(
        "created_by".to_string(),
        "Enhanced sparse example".to_string(),
    );
    sparse
        .metadata
        .insert("matrix_type".to_string(), "tridiagonal".to_string());

    println!(
        "  Original matrix: {}x{} with {} non-zeros",
        sparse.shape().0,
        sparse.shape().1,
        sparse.nnz()
    );

    // Test different serialization formats
    let formats = vec![
        (SerializationFormat::JSON, "sparse_enhanced.json"),
        (SerializationFormat::Binary, "sparse_enhanced.bin"),
        (SerializationFormat::MessagePack, "sparse_enhanced.msgpack"),
    ];

    for (format, filename) in formats {
        println!("  Testing {:?} format...", format);

        // Serialize
        let start = Instant::now();
        serialize_enhanced_sparse_matrix(filename, &sparse, format)?;
        let serialize_time = start.elapsed();

        // Get file size
        let file_size = std::fs::metadata(filename)
            .map(|metadata| metadata.len())
            .unwrap_or(0);

        // Deserialize
        let start = Instant::now();
        let loaded: SparseMatrix<f64> = deserialize_enhanced_sparse_matrix(filename, format)?;
        let deserialize_time = start.elapsed();

        println!(
            "    Serialize: {:?}, Deserialize: {:?}",
            serialize_time, deserialize_time
        );
        println!("    File size: {} bytes", file_size);
        println!(
            "    Loaded matrix: {}x{} with {} non-zeros",
            loaded.shape().0,
            loaded.shape().1,
            loaded.nnz()
        );

        // Verify metadata
        if let Some(description) = loaded.metadata.get("description") {
            println!("    Metadata preserved: description = '{}'", description);
        }

        println!();
    }

    Ok(())
}

fn demonstrate_memory_analysis() -> Result<()> {
    println!("5. Demonstrating memory efficiency analysis...");

    let sizes = vec![100, 500, 1000, 2000];

    println!("  Matrix Size | Dense Memory | Sparse Memory | Memory Ratio | Sparsity");
    println!("  -----------|--------------|---------------|--------------|----------");

    for size in sizes {
        let sparse = create_sample_sparse_matrix(size);

        // Calculate dense matrix memory usage
        let dense_memory = size * size * std::mem::size_of::<f64>();

        // Calculate sparse matrix memory usage
        let sparse_memory = sparse.memory_usage();

        // Calculate memory ratio
        let memory_ratio = dense_memory as f64 / sparse_memory as f64;

        println!(
            "  {:^10} | {:^11} | {:^12} | {:^11.2} | {:^7.2}%",
            format!("{}x{}", size, size),
            format_bytes(dense_memory),
            format_bytes(sparse_memory),
            memory_ratio,
            sparse.sparsity() * 100.0
        );
    }

    // Analyze format efficiency
    println!("\n  Format efficiency comparison:");
    let size = 1000;
    let mut sparse = create_sample_sparse_matrix(size);

    let coo_memory = sparse.coo_data.values.len()
        * (std::mem::size_of::<f64>() + 2 * std::mem::size_of::<usize>());

    let csr = sparse.to_csr()?;
    let csr_memory = csr.values.len() * std::mem::size_of::<f64>()
        + csr.col_indices.len() * std::mem::size_of::<usize>()
        + csr.row_ptrs.len() * std::mem::size_of::<usize>();

    let csc = sparse.to_csc()?;
    let csc_memory = csc.values.len() * std::mem::size_of::<f64>()
        + csc.row_indices.len() * std::mem::size_of::<usize>()
        + csc.col_ptrs.len() * std::mem::size_of::<usize>();

    println!("    COO format: {} bytes", format_bytes(coo_memory));
    println!("    CSR format: {} bytes", format_bytes(csr_memory));
    println!("    CSC format: {} bytes", format_bytes(csc_memory));

    Ok(())
}

/// Create a sample sparse matrix (tridiagonal pattern)
fn create_sample_sparse_matrix(size: usize) -> SparseMatrix<f64> {
    let mut sparse = SparseMatrix::new(size, size);

    // Main diagonal
    for i in 0..size {
        sparse.insert(i, i, 2.0);
    }

    // Super-diagonal
    for i in 0..size - 1 {
        sparse.insert(i, i + 1, -1.0);
    }

    // Sub-diagonal
    for i in 1..size {
        sparse.insert(i, i - 1, -1.0);
    }

    sparse
}

/// Format bytes in human-readable format
fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Demonstration of Matrix Market integration (if feature is enabled)
#[allow(dead_code)]
fn demonstrate_matrix_market_integration() -> Result<()> {
    println!("6. Demonstrating Matrix Market integration...");

    // This would require the matrix_market feature to be enabled
    // let sparse = create_sample_sparse_matrix(100);
    // let mm_matrix = to_matrix_market(&sparse);
    // println!("  Converted to Matrix Market format: {}x{} with {} entries",
    //          mm_matrix.rows, mm_matrix.cols, mm_matrix.nnz);

    Ok(())
}

/// Performance benchmarking utilities
#[allow(dead_code)]
mod benchmarks {
    use super::*;

    pub fn benchmark_format_conversions(size: usize, iterations: usize) -> Result<()> {
        println!(
            "Benchmarking format conversions for {}x{} matrix ({} iterations):",
            size, size, iterations
        );

        let mut total_csr_time = std::time::Duration::new(0, 0);
        let mut total_csc_time = std::time::Duration::new(0, 0);

        for _ in 0..iterations {
            let mut sparse = create_sample_sparse_matrix(size);

            let start = Instant::now();
            let _csr = sparse.to_csr()?;
            total_csr_time += start.elapsed();

            let start = Instant::now();
            let _csc = sparse.to_csc()?;
            total_csc_time += start.elapsed();
        }

        println!(
            "  Average COO -> CSR: {:?}",
            total_csr_time / iterations as u32
        );
        println!(
            "  Average COO -> CSC: {:?}",
            total_csc_time / iterations as u32
        );

        Ok(())
    }
}
