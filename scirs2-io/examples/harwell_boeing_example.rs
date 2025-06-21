//! Harwell-Boeing sparse matrix format example
//!
//! This example demonstrates how to work with Harwell-Boeing sparse matrix files,
//! including reading, writing, and converting between different formats.

use ndarray::Array1;
use scirs2_io::harwell_boeing::{self, ccs_to_hb, hb_to_ccs, HBMatrixType, HBSparseMatrix};
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔢 Harwell-Boeing Sparse Matrix Format Example");
    println!("==============================================");

    // Create temporary directory for test files
    let temp_dir = tempdir()?;
    println!("📁 Using temporary directory: {:?}", temp_dir.path());

    // Demonstrate creating and working with Harwell-Boeing matrices
    create_and_write_matrix(&temp_dir)?;
    read_and_analyze_matrix(&temp_dir)?;
    demonstrate_format_conversion(&temp_dir)?;
    demonstrate_different_matrix_types(&temp_dir)?;

    println!("\n✅ All Harwell-Boeing examples completed successfully!");
    println!("💡 The Harwell-Boeing format is ideal for storing large sparse matrices in scientific computing");

    Ok(())
}

fn create_and_write_matrix(temp_dir: &tempfile::TempDir) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📝 Creating and Writing Harwell-Boeing Matrix...");

    // Create a simple sparse matrix in CCS format
    // Matrix:
    // [1.0  0   3.0]
    // [2.0  4.0  0 ]
    // [0    5.0  6.0]

    let colptr = Array1::from(vec![0, 2, 4, 6]); // Column pointers
    let rowind = Array1::from(vec![0, 1, 1, 2, 0, 2]); // Row indices
    let values = Array1::from(vec![1.0, 2.0, 4.0, 5.0, 3.0, 6.0]); // Values

    println!("  🔢 Created 3x3 sparse matrix with 6 non-zero entries");
    println!("     Matrix structure:");
    println!("     [1.0  0   3.0]");
    println!("     [2.0  4.0  0 ]");
    println!("     [0    5.0  6.0]");

    // Convert to Harwell-Boeing format
    let hb_matrix = ccs_to_hb(
        &colptr,
        &rowind,
        &values,
        (3, 3),
        "Example 3x3 sparse matrix".to_string(),
        "EX3X3".to_string(),
        HBMatrixType::RealUnsymmetric,
    );

    println!("  📋 Matrix metadata:");
    println!("     Title: {}", hb_matrix.header.title);
    println!("     Key: {}", hb_matrix.header.key);
    println!("     Type: {}", hb_matrix.header.mxtype);
    println!(
        "     Dimensions: {}x{}",
        hb_matrix.header.nrow, hb_matrix.header.ncol
    );
    println!("     Non-zeros: {}", hb_matrix.header.nnzero);

    // Write to file
    let hb_file = temp_dir.path().join("example_matrix.hb");
    harwell_boeing::write_harwell_boeing(&hb_file, &hb_matrix)?;

    println!(
        "  ✅ Matrix written to Harwell-Boeing file: {:?}",
        hb_file.file_name().unwrap()
    );

    Ok(())
}

fn read_and_analyze_matrix(temp_dir: &tempfile::TempDir) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📖 Reading and Analyzing Harwell-Boeing Matrix...");

    let hb_file = temp_dir.path().join("example_matrix.hb");

    // Read the matrix back
    let matrix = harwell_boeing::read_harwell_boeing(&hb_file)?;

    println!("  📊 Matrix analysis:");
    println!("     Title: {}", matrix.header.title);
    println!("     Matrix type: {}", matrix.header.mxtype);
    println!(
        "     Dimensions: {}x{}",
        matrix.header.nrow, matrix.header.ncol
    );
    println!("     Non-zero entries: {}", matrix.header.nnzero);
    println!("     Storage: {} KB", estimate_storage_size(&matrix));

    // Analyze sparsity
    let total_elements = matrix.header.nrow * matrix.header.ncol;
    let sparsity = (total_elements - matrix.header.nnzero) as f64 / total_elements as f64;
    println!(
        "     Sparsity: {:.1}% ({}% of entries are zero)",
        sparsity * 100.0,
        (sparsity * 100.0) as i32
    );

    // Display column pointer structure
    println!("  🗂️  Column structure:");
    for (col, &start) in matrix.colptr.iter().enumerate().take(matrix.header.ncol) {
        let end = if col + 1 < matrix.colptr.len() {
            matrix.colptr[col + 1]
        } else {
            matrix.header.nnzero
        };
        let nnz_in_col = end - start;
        println!("     Column {}: {} non-zeros", col, nnz_in_col);
    }

    // Verify matrix structure
    verify_matrix_structure(&matrix)?;

    Ok(())
}

fn demonstrate_format_conversion(
    temp_dir: &tempfile::TempDir,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Demonstrating Format Conversion...");

    let hb_file = temp_dir.path().join("example_matrix.hb");

    // Read Harwell-Boeing matrix
    let hb_matrix = harwell_boeing::read_harwell_boeing(&hb_file)?;

    // Convert to CCS format
    let (colptr, rowind, values) = hb_to_ccs(&hb_matrix);

    println!("  📊 CCS Format representation:");
    println!("     Column pointers: {:?}", colptr.as_slice().unwrap());
    println!("     Row indices: {:?}", rowind.as_slice().unwrap());
    println!("     Values: {:?}", values.as_slice().unwrap());

    // Convert back to Harwell-Boeing
    let reconverted = ccs_to_hb(
        &colptr,
        &rowind,
        &values,
        (hb_matrix.header.nrow, hb_matrix.header.ncol),
        "Reconverted matrix".to_string(),
        "RECONV".to_string(),
        hb_matrix.header.mxtype,
    );

    // Verify round-trip conversion
    assert_eq!(
        reconverted.colptr, hb_matrix.colptr,
        "Column pointers don't match after round-trip"
    );
    assert_eq!(
        reconverted.rowind, hb_matrix.rowind,
        "Row indices don't match after round-trip"
    );
    if let (Some(ref orig_vals), Some(ref new_vals)) = (&hb_matrix.values, &reconverted.values) {
        for (orig, new) in orig_vals.iter().zip(new_vals.iter()) {
            assert!(
                (orig - new).abs() < 1e-14,
                "Values don't match after round-trip"
            );
        }
    }

    println!("  ✅ Round-trip conversion successful - data integrity preserved");

    // Save reconverted matrix
    let reconverted_file = temp_dir.path().join("reconverted_matrix.hb");
    harwell_boeing::write_harwell_boeing(&reconverted_file, &reconverted)?;

    println!(
        "  💾 Reconverted matrix saved as: {:?}",
        reconverted_file.file_name().unwrap()
    );

    Ok(())
}

fn demonstrate_different_matrix_types(
    temp_dir: &tempfile::TempDir,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎭 Demonstrating Different Matrix Types...");

    // Create different types of matrices

    // 1. Symmetric matrix
    let sym_colptr = Array1::from(vec![0, 2, 3, 4]);
    let sym_rowind = Array1::from(vec![0, 1, 1, 2]);
    let sym_values = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);

    let symmetric_matrix = ccs_to_hb(
        &sym_colptr,
        &sym_rowind,
        &sym_values,
        (3, 3),
        "Symmetric test matrix".to_string(),
        "SYM3X3".to_string(),
        HBMatrixType::RealSymmetric,
    );

    let sym_file = temp_dir.path().join("symmetric_matrix.hb");
    harwell_boeing::write_harwell_boeing(&sym_file, &symmetric_matrix)?;
    println!(
        "  ✅ Symmetric matrix saved: {:?}",
        sym_file.file_name().unwrap()
    );

    // 2. Pattern matrix (no values, structure only)
    let pattern_matrix = ccs_to_hb(
        &sym_colptr,
        &sym_rowind,
        &sym_values, // Values will be ignored for pattern matrix
        (3, 3),
        "Pattern test matrix".to_string(),
        "PAT3X3".to_string(),
        HBMatrixType::Pattern,
    );

    let pattern_file = temp_dir.path().join("pattern_matrix.hb");
    harwell_boeing::write_harwell_boeing(&pattern_file, &pattern_matrix)?;
    println!(
        "  ✅ Pattern matrix saved: {:?}",
        pattern_file.file_name().unwrap()
    );

    // 3. Read back and verify
    let read_symmetric = harwell_boeing::read_harwell_boeing(&sym_file)?;
    let read_pattern = harwell_boeing::read_harwell_boeing(&pattern_file)?;

    println!("  📊 Matrix type verification:");
    println!(
        "     Symmetric matrix type: {}",
        read_symmetric.header.mxtype
    );
    println!(
        "     Symmetric has values: {}",
        read_symmetric.values.is_some()
    );
    println!("     Pattern matrix type: {}", read_pattern.header.mxtype);
    println!("     Pattern has values: {}", read_pattern.values.is_some());

    assert_eq!(read_symmetric.header.mxtype, HBMatrixType::RealSymmetric);
    assert_eq!(read_pattern.header.mxtype, HBMatrixType::Pattern);
    assert!(read_symmetric.values.is_some());
    assert!(read_pattern.values.is_none());

    println!("  ✅ All matrix types handled correctly");

    Ok(())
}

fn verify_matrix_structure(matrix: &HBSparseMatrix<f64>) -> Result<(), Box<dyn std::error::Error>> {
    // Basic structural checks
    assert_eq!(
        matrix.colptr.len(),
        matrix.header.ncol + 1,
        "Column pointer array size mismatch"
    );
    assert_eq!(
        matrix.rowind.len(),
        matrix.header.nnzero,
        "Row index array size mismatch"
    );

    if let Some(ref values) = matrix.values {
        assert_eq!(
            values.len(),
            matrix.header.nnzero,
            "Values array size mismatch"
        );
    }

    // Check column pointer monotonicity
    for i in 1..matrix.colptr.len() {
        assert!(
            matrix.colptr[i] >= matrix.colptr[i - 1],
            "Column pointers are not monotonic at position {}",
            i
        );
    }

    // Check row indices are within bounds
    for &row_idx in &matrix.rowind {
        assert!(
            row_idx < matrix.header.nrow,
            "Row index {} out of bounds (nrow={})",
            row_idx,
            matrix.header.nrow
        );
    }

    // Check first and last column pointers
    assert_eq!(matrix.colptr[0], 0, "First column pointer should be 0");
    assert_eq!(
        matrix.colptr[matrix.header.ncol], matrix.header.nnzero,
        "Last column pointer should equal nnzero"
    );

    println!("  ✅ Matrix structure verification passed");
    Ok(())
}

fn estimate_storage_size(matrix: &HBSparseMatrix<f64>) -> usize {
    let ptr_size = matrix.colptr.len() * std::mem::size_of::<usize>();
    let idx_size = matrix.rowind.len() * std::mem::size_of::<usize>();
    let val_size = if let Some(ref values) = matrix.values {
        values.len() * std::mem::size_of::<f64>()
    } else {
        0
    };

    (ptr_size + idx_size + val_size) / 1024 // Convert to KB
}
