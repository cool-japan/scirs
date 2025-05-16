// Copyright (c) 2025, SciRS2 Team
//
// Licensed under either of
//
// * Apache License, Version 2.0
//   (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
// * MIT license
//   (LICENSE-MIT or http://opensource.org/licenses/MIT)
//
// at your option.
//

//! Example demonstrating how to implement the array protocol for a third-party array library.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

use ndarray::Array2;
use scirs2_core::array_protocol::{
    self, matmul, sum, transpose, ArrayFunction, ArrayProtocol, NdarrayWrapper, NotImplemented,
};

/// A custom sparse array implementation.
struct SparseArray {
    indices: Vec<(usize, usize)>,
    values: Vec<f64>,
    shape: (usize, usize),
}

impl SparseArray {
    /// Create a new sparse array.
    #[allow(dead_code)]
    fn new(indices: Vec<(usize, usize)>, values: Vec<f64>, shape: (usize, usize)) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "Indices and values must have the same length"
        );
        Self {
            indices,
            values,
            shape,
        }
    }

    /// Create a sparse array from a dense array by keeping only non-zero elements.
    fn from_dense(array: &Array2<f64>) -> Self {
        let shape = array.dim();
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for ((i, j), &val) in array.indexed_iter() {
            if val != 0.0 {
                indices.push((i, j));
                values.push(val);
            }
        }

        Self {
            indices,
            values,
            shape,
        }
    }

    /// Convert the sparse array to a dense array.
    fn to_dense(&self) -> Array2<f64> {
        let mut result = Array2::<f64>::zeros(self.shape);

        for (i, &val) in self.indices.iter().zip(self.values.iter()) {
            result[[i.0, i.1]] = val;
        }

        result
    }

    /// Get the number of non-zero elements.
    fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get the sparsity ratio (number of non-zeros / total elements).
    fn sparsity(&self) -> f64 {
        let total = self.shape.0 * self.shape.1;
        self.nnz() as f64 / total as f64
    }
}

impl fmt::Debug for SparseArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseArray")
            .field("shape", &self.shape)
            .field("nnz", &self.nnz())
            .field("sparsity", &self.sparsity())
            .finish()
    }
}

/// Implement the ArrayProtocol trait for SparseArray.
impl ArrayProtocol for SparseArray {
    fn box_clone(&self) -> Box<dyn ArrayProtocol> {
        Box::new(SparseArray {
            indices: self.indices.clone(),
            values: self.values.clone(),
            shape: self.shape,
        })
    }

    fn array_function(
        &self,
        func: &ArrayFunction,
        _types: &[TypeId],
        args: &[Box<dyn Any>],
        kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        match func.name {
            "scirs2::array_protocol::operations::matmul" => {
                // Matrix multiplication for sparse arrays
                if args.len() != 2 {
                    return Err(NotImplemented);
                }

                // Extract the second argument (it should be a SparseArray or something we can handle)
                let other = if let Some(sparse) = args[1].downcast_ref::<&dyn ArrayProtocol>() {
                    if let Some(sparse_array) = sparse.as_any().downcast_ref::<SparseArray>() {
                        sparse_array
                    } else if let Some(ndarray_wrapper) =
                        sparse.as_any().downcast_ref::<NdarrayWrapper<f64, _>>()
                    {
                        // Convert ndarray to sparse array (simplified for example)
                        return Ok(Box::new(SparseArray::from_dense(
                            ndarray_wrapper.as_array(),
                        )));
                    } else {
                        return Err(NotImplemented);
                    }
                } else {
                    return Err(NotImplemented);
                };

                // Check if shapes are compatible
                if self.shape.1 != other.shape.0 {
                    return Err(NotImplemented);
                }

                // For the example, we'll convert to dense, multiply, and convert back to sparse
                let a_dense = self.to_dense();
                let b_dense = other.to_dense();
                let result_dense = a_dense.dot(&b_dense);
                let result = SparseArray::from_dense(&result_dense);

                Ok(Box::new(result))
            }
            "scirs2::array_protocol::operations::add" => {
                // Element-wise addition for sparse arrays
                if args.len() != 2 {
                    return Err(NotImplemented);
                }

                // Extract the second argument
                let other = if let Some(sparse) = args[1].downcast_ref::<&dyn ArrayProtocol>() {
                    if let Some(sparse_array) = sparse.as_any().downcast_ref::<SparseArray>() {
                        sparse_array
                    } else {
                        return Err(NotImplemented);
                    }
                } else {
                    return Err(NotImplemented);
                };

                // Check if shapes are compatible
                if self.shape != other.shape {
                    return Err(NotImplemented);
                }

                // Convert to dense, add, and convert back to sparse (simplified)
                let a_dense = self.to_dense();
                let b_dense = other.to_dense();
                let result_dense = &a_dense + &b_dense;
                let result = SparseArray::from_dense(&result_dense);

                Ok(Box::new(result))
            }
            "scirs2::array_protocol::operations::sum" => {
                // Sum operation for sparse arrays
                // We can compute this directly from the values without converting to dense
                let result: f64 = self.values.iter().sum();

                // Check if summing along an axis
                if let Some(axis_box) = kwargs.get("axis") {
                    if let Some(axis) = axis_box.downcast_ref::<usize>() {
                        // For the example, we'll just convert to dense and use ndarray's sum
                        let dense = self.to_dense();
                        let result = dense.sum_axis(ndarray::Axis(*axis));
                        let sparse_result =
                            SparseArray::from_dense(&result.into_dimensionality().unwrap());
                        return Ok(Box::new(sparse_result));
                    }
                }

                Ok(Box::new(result))
            }
            "scirs2::array_protocol::operations::transpose" => {
                // Transpose operation for sparse arrays
                // We can compute this directly by swapping the indices
                let mut new_indices = Vec::with_capacity(self.indices.len());
                for &(i, j) in &self.indices {
                    new_indices.push((j, i));
                }

                let result = SparseArray {
                    indices: new_indices,
                    values: self.values.clone(),
                    shape: (self.shape.1, self.shape.0),
                };

                Ok(Box::new(result))
            }
            _ => Err(NotImplemented),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn main() {
    // Initialize the array protocol system
    array_protocol::init();

    println!("Custom Array Protocol Example");
    println!("============================");

    // Create a dense array with some sparsity
    let mut dense = Array2::<f64>::zeros((5, 5));
    dense[[0, 0]] = 1.0;
    dense[[1, 2]] = 2.0;
    dense[[2, 1]] = 3.0;
    dense[[3, 3]] = 4.0;
    dense[[4, 4]] = 5.0;

    // Create a sparse array from the dense array
    let sparse = SparseArray::from_dense(&dense);

    println!("\nOriginal sparse array:");
    println!("{:?}", sparse);
    println!("Non-zero elements: {}", sparse.nnz());
    println!("Sparsity: {:.2}%", sparse.sparsity() * 100.0);

    // Create a dense array wrapped for the array protocol
    let wrapped_dense = NdarrayWrapper::new(dense.clone());

    // Test operations with sparse arrays

    // 1. Matrix multiplication
    println!("\n1. Matrix multiplication:");

    // Sparse * Sparse
    match matmul(&sparse, &sparse) {
        Ok(result) => {
            if let Some(sparse_result) = result.as_any().downcast_ref::<SparseArray>() {
                println!("Sparse * Sparse result:");
                println!("{:?}", sparse_result);
            } else {
                println!("Result is not a SparseArray type");
            }
        }
        Err(e) => println!("Error in Sparse * Sparse: {}", e),
    }

    // Sparse * Dense
    match matmul(&sparse, &wrapped_dense) {
        Ok(result) => {
            if let Some(sparse_result) = result.as_any().downcast_ref::<SparseArray>() {
                println!("Sparse * Dense result:");
                println!("{:?}", sparse_result);
            } else {
                println!("Result is not a SparseArray type");
            }
        }
        Err(e) => println!("Error in Sparse * Dense: {}", e),
    }

    // 2. Sum operation
    println!("\n2. Sum operation:");

    match sum(&sparse, None) {
        Ok(result) => {
            if let Some(sum_value) = result.downcast_ref::<f64>() {
                println!("Sum of sparse array: {}", sum_value);
            } else {
                println!("Result is not a f64 type");
            }
        }
        Err(e) => println!("Error in Sum operation: {}", e),
    }

    // 3. Transpose operation
    println!("\n3. Transpose operation:");

    match transpose(&sparse) {
        Ok(result) => {
            if let Some(sparse_result) = result.as_any().downcast_ref::<SparseArray>() {
                println!("Transpose of sparse array:");
                println!("{:?}", sparse_result);
            } else {
                println!("Result is not a SparseArray type");
            }
        }
        Err(e) => println!("Error in Transpose operation: {}", e),
    }

    // Verify correctness by comparing with dense operations
    println!("\nVerification with dense operations:");

    // Matrix multiplication
    let dense_result = dense.dot(&dense);
    match matmul(&sparse, &sparse) {
        Ok(sparse_result) => {
            if let Some(sparse_array) = sparse_result.as_any().downcast_ref::<SparseArray>() {
                let sparse_dense = sparse_array.to_dense();
                // Manual approximate equality check
                let is_approx_equal = dense_result
                    .iter()
                    .zip(sparse_dense.iter())
                    .all(|(&a, &b)| (a - b).abs() < 1e-10);
                println!("Matrix multiplication matches dense: {}", is_approx_equal);
            } else {
                println!("Matrix multiplication result is not a SparseArray type");
            }
        }
        Err(e) => println!("Error in matrix multiplication verification: {}", e),
    }

    // Sum
    let dense_sum = dense.sum();
    match sum(&sparse, None) {
        Ok(result) => {
            if let Some(sparse_sum) = result.downcast_ref::<f64>() {
                println!(
                    "Sum matches dense: {}",
                    (dense_sum - sparse_sum).abs() < 1e-10
                );
            } else {
                println!("Sum result is not a f64 type");
            }
        }
        Err(e) => println!("Error in sum verification: {}", e),
    }

    // Transpose
    let dense_transpose = dense.t().to_owned();
    match transpose(&sparse) {
        Ok(sparse_transpose) => {
            if let Some(sparse_array) = sparse_transpose.as_any().downcast_ref::<SparseArray>() {
                let sparse_dense_transpose = sparse_array.to_dense();
                // Manual approximate equality check
                let is_transpose_equal = dense_transpose
                    .iter()
                    .zip(sparse_dense_transpose.iter())
                    .all(|(&a, &b)| (a - b).abs() < 1e-10);
                println!("Transpose matches dense: {}", is_transpose_equal);
            } else {
                println!("Transpose result is not a SparseArray type");
            }
        }
        Err(e) => println!("Error in transpose verification: {}", e),
    }

    println!("\nAll operations completed successfully!");
}
