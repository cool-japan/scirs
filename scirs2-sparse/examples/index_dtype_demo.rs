// Example demonstrating index dtype handling in sparse arrays
//
// This example shows how to use the index dtype utilities to determine
// the appropriate dtype for sparse array indices and safely cast index arrays.

use scirs2_core::ndarray::Array1;
use scirs2_sparse::{
    can_cast_safely, get_index_dtype, safely_cast_index_arrays, CooArray, SparseArray,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# Index dtype handling in sparse arrays\n");

    // Example 1: Determine appropriate dtype for small array
    let smallshape = (100, 100);
    let small_dtype = get_index_dtype(smallshape, &[]);
    println!(
        "For a small array with shape {:?}, recommended dtype: {}",
        smallshape, small_dtype
    );

    // Example 2: Determine appropriate dtype for medium array
    let mediumshape = (50_000, 50_000);
    let medium_dtype = get_index_dtype(mediumshape, &[]);
    println!(
        "For a medium array with shape {:?}, recommended dtype: {}",
        mediumshape, medium_dtype
    );

    // Example 3: Determine appropriate dtype for large array
    let largeshape = (1_000_000_000, 1_000_000_000);
    let large_dtype = get_index_dtype(largeshape, &[]);
    println!(
        "For a large array with shape {:?}, recommended dtype: {}",
        largeshape, large_dtype
    );

    // Example 4: Consider existing indices when determining dtype
    let indices1 = Array1::from_vec(vec![0, 10, 20, 30]);
    let indices2 = Array1::from_vec(vec![5, 15, 25, 1000]);

    let shape = (100, 100);
    let dtype = get_index_dtype(shape, &[indices1.view(), indices2.view()]);
    println!(
        "For shape {:?} with existing indices (max value 1000), recommended dtype: {}",
        shape, dtype
    );

    // Example 5: Now with larger values in indices
    let large_indices = Array1::from_vec(vec![0, i32::MAX as usize + 1]);
    let dtype_with_large = get_index_dtype(shape, &[large_indices.view()]);
    println!(
        "For shape {:?} with large indices (max value > i32::MAX), recommended dtype: {}",
        shape, dtype_with_large
    );

    // Example 6: Safely cast arrays when possible
    let indices = Array1::from_vec(vec![0, 5, 10, 100]);
    println!("\nSafely casting index arrays:");

    // This should succeed
    match safely_cast_index_arrays::<i32>(&[indices.view()]) {
        Ok(arrays) => {
            println!("  Successfully cast to i32: {:?}", arrays[0]);
        }
        Err(e) => {
            println!("  Failed to cast to i32: {}", e);
        }
    }

    // This should fail (i8 can't hold 100)
    match safely_cast_index_arrays::<i8>(&[indices.view()]) {
        Ok(arrays) => {
            println!("  Successfully cast to i8: {:?}", arrays[0]);
        }
        Err(e) => {
            println!("  Failed to cast to i8: {}", e);
        }
    }

    // Example 7: Check if arrays can be safely cast
    println!("\nChecking if arrays can be safely cast:");
    let small_values = Array1::from_vec(vec![0, 5, 10, 20]);
    let large_values = Array1::from_vec(vec![0, 100, 200]);

    println!(
        "  Array with values {:?} can be cast to i8: {}",
        small_values,
        can_cast_safely::<i8>(small_values.view())
    );

    println!(
        "  Array with values {:?} can be cast to i8: {}",
        large_values,
        can_cast_safely::<i8>(large_values.view())
    );

    // Example 8: Creating a sparse array and using the utilities in practice
    println!("\nPractical example with sparse arrays:");

    // Create a small sparse array
    let rows = vec![0, 0, 1, 2, 2];
    let cols = vec![0, 2, 2, 0, 1];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let shape = (3, 3);

    // Determine the best index dtype
    let rowsarray = Array1::from_vec(rows.clone());
    let cols_array = Array1::from_vec(cols.clone());
    let dtype = get_index_dtype(shape, &[rowsarray.view(), cols_array.view()]);

    println!(
        "  For a sparse array with shape {:?}, recommended index dtype: {}",
        shape, dtype
    );

    // Create sparse matrix
    let sparse_array = CooArray::from_triplets(&rows, &cols, &data, shape, false)?;
    println!(
        "  Created sparse array with {} non-zero elements",
        sparse_array.nnz()
    );

    // Convert to CSR format (which often uses index arrays internally)
    let _csr_array = sparse_array.to_csr()?;
    println!("  Converted to CSR format successfully");

    println!("\nIndex dtype handling is particularly important for large sparse arrays");
    println!("where indices might exceed 32-bit integer ranges.");

    Ok(())
}
