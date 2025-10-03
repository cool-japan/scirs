// Example demonstrating optimized operations with symmetric sparse matrices
//
// This example shows how to use the specialized symmetric sparse formats
// and the optimized operations that take advantage of symmetry.

use scirs2_core::ndarray::Array1;
use scirs2_sparse::{
    sparray::SparseArray,
    sym_coo::SymCooArray,
    sym_coo::SymCooMatrix,
    sym_csr::SymCsrArray,
    sym_csr::SymCsrMatrix,
    sym_ops::{sym_csr_matvec, sym_csr_quadratic_form, sym_csr_rank1_update, sym_csr_trace},
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# Symmetric Sparse Matrix Operations");
    println!("Demonstrating optimized operations on symmetric sparse matrices\n");

    // Create a symmetric matrix:
    // [ 2  1  0 ]
    // [ 1  2  3 ]
    // [ 0  3  1 ]

    // For storage efficiency, we only store the lower triangular part:
    // [ 2  0  0 ]
    // [ 1  2  0 ]
    // [ 0  3  1 ]

    let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
    let indices = vec![0, 0, 1, 1, 2];
    let indptr = vec![0, 1, 3, 5];

    // Create SymCsrMatrix - the base representation
    let sym_csr_matrix = SymCsrMatrix::new(data.clone(), indptr, indices, (3, 3))?;
    println!("Created a 3x3 symmetric matrix in CSR format");
    println!(
        "  Storage: only the lower triangular part ({} elements)",
        sym_csr_matrix.nnz_stored()
    );
    println!("  Full matrix: {} non-zero elements", sym_csr_matrix.nnz());

    // Create a vector for testing
    let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    println!("\nTest vector: {:?}", x);

    // 1. Direct optimized matrix-vector product
    println!("\n1. Optimized matrix-vector product:");
    let y1 = sym_csr_matvec(&sym_csr_matrix, &x.view())?;
    println!("  Result = {:?}", y1);

    // 2. Compute the matrix trace
    println!("\n2. Matrix trace:");
    let trace = sym_csr_trace(&sym_csr_matrix);
    println!("  Trace = {:.1}", trace);

    // 3. Compute quadratic form
    println!("\n3. Quadratic form x^T*A*x:");
    let quad_form = sym_csr_quadratic_form(&sym_csr_matrix, &x.view())?;
    println!("  x^T*A*x = {:.1}", quad_form);

    // 4. Rank-1 update
    println!("\n4. Rank-1 update A = A + alpha*x*x^T:");
    let alpha = 2.0;
    let update_vector = Array1::from_vec(vec![1.0, 0.0, 0.0]);

    // Create a copy of the original matrix for updating
    let mut updated_matrix = sym_csr_matrix.clone();
    sym_csr_rank1_update(&mut updated_matrix, &update_vector.view(), alpha)?;

    println!(
        "  Original matrix diagonal: [{:.1}, {:.1}, {:.1}]",
        sym_csr_matrix.get(0, 0),
        sym_csr_matrix.get(1, 1),
        sym_csr_matrix.get(2, 2)
    );
    println!(
        "  Updated matrix diagonal:  [{:.1}, {:.1}, {:.1}]",
        updated_matrix.get(0, 0),
        updated_matrix.get(1, 1),
        updated_matrix.get(2, 2)
    );

    // 5. Using symmetric matrices through the SparseArray trait
    println!("\n5. Using symmetric matrices via SparseArray trait:");

    // Create SymCsrArray
    let sym_csr_array = SymCsrArray::new(sym_csr_matrix.clone());
    // Using SparseArray trait method, which calls our optimized implementation
    let y2 = sym_csr_array.dot_vector(&x.view())?;
    println!("  SymCsrArray.dot_vector result = {:?}", y2);

    // 6. Convert between symmetric formats
    println!("\n6. Symmetric COO format operations:");

    // Convert to symmetric COO format
    let rows = vec![0, 1, 1, 2, 2];
    let cols = vec![0, 0, 1, 1, 2];
    let sym_coo_matrix = SymCooMatrix::new(data, rows, cols, (3, 3))?;
    let sym_coo_array = SymCooArray::new(sym_coo_matrix);

    // Matrix-vector product using COO format
    let y3 = sym_coo_array.dot_vector(&x.view())?;
    println!("  SymCooArray.dot_vector result = {:?}", y3);

    // 7. Compare performance vs. standard CSR (this would be more meaningful with large matrices)
    println!("\n7. Memory efficiency comparison:");
    // Convert to standard CSR format (stores both triangular parts)
    let standard_csr = sym_csr_matrix.to_csr()?;

    println!(
        "  Symmetric CSR: {} stored elements",
        sym_csr_matrix.nnz_stored()
    );
    println!("  Standard CSR:  {} stored elements", standard_csr.nnz());
    println!(
        "  Memory savings: {:.1}%",
        (1.0 - (sym_csr_matrix.nnz_stored() as f64 / standard_csr.nnz() as f64)) * 100.0
    );

    println!("\nSymmetric formats are particularly useful for large, symmetric matrices");
    println!("like those encountered in finite element methods, graph Laplacians,");
    println!("and numerical solution of PDEs.");

    Ok(())
}
