//! Example demonstrating NumPy-style broadcasting for higher-dimensional arrays

use ndarray::array;
use scirs2_linalg::prelude::*;

fn main() {
    println!("NumPy-style Broadcasting Example\n");

    // Example 1: 3D arrays (batch of matrices)
    broadcast_3d_example();

    // Example 2: Dynamic dimensional arrays
    broadcast_dynamic_example();

    // Example 3: Broadcasting with different batch sizes
    broadcast_different_batch_example();
}

fn broadcast_3d_example() {
    println!("=== 3D Array Broadcasting ===");

    // Create batch of 2x2 matrices
    let a = array![[[1.0_f64, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let b = array![
        [[1.0, 0.0], [0.0, 1.0]], // Identity matrix
        [[2.0, 0.0], [0.0, 2.0]]  // 2 * Identity
    ];

    // Batch matrix multiplication
    let c = broadcast_matmul_3d(&a, &b).unwrap();

    println!("First batch (A * I):");
    println!("{:?}", c.index_axis(ndarray::Axis(0), 0));

    println!("\nSecond batch (A * 2I):");
    println!("{:?}", c.index_axis(ndarray::Axis(0), 1));

    // Batch matrix-vector multiplication (using dynamic arrays)
    let x = array![[1.0, 1.0], [2.0, 1.0]].into_dyn();
    let a_dyn = a.into_dyn();
    let y = broadcast_matvec(&a_dyn, &x).unwrap();

    println!("\nBatch matrix-vector multiplication:");
    println!("First batch result: {:?}", &y.as_slice().unwrap()[0..2]);
    println!("Second batch result: {:?}", &y.as_slice().unwrap()[2..4]);
}

fn broadcast_dynamic_example() {
    println!("\n=== Dynamic Array Broadcasting ===");

    // Create 4D array: (2, 3, 2, 2) - 2 groups of 3 batches of 2x2 matrices
    let a = ndarray::Array4::<f64>::ones((2, 3, 2, 2)).into_dyn();
    let b = ndarray::Array4::<f64>::from_elem((2, 3, 2, 2), 2.0).into_dyn();

    // Multiply all matrices by 2
    let c = broadcast_matmul(&a, &b).unwrap();

    println!("Shape of result: {:?}", c.shape());
    println!("Sample value: {}", c[[0, 0, 0, 0]]);
}

fn broadcast_different_batch_example() {
    println!("\n=== Broadcasting with Different Batch Sizes ===");

    // Batch of 2 matrices
    let a = array![[[1.0_f64, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];

    // Single matrix (will be broadcast to both)
    let b = array![[[2.0, 0.0], [0.0, 2.0]]]; // 2 * Identity

    // Broadcasting multiplication
    let c = broadcast_matmul_3d(&a, &b).unwrap();

    println!("First batch (multiplied by 2*I):");
    println!("{:?}", c.index_axis(ndarray::Axis(0), 0));

    println!("\nSecond batch (multiplied by 2*I):");
    println!("{:?}", c.index_axis(ndarray::Axis(0), 1));

    // Check broadcast compatibility
    let incompatible = array![[[1.0, 2.0, 3.0]]];
    println!("\nBroadcast compatibility check:");
    println!("a and b compatible: {}", a.broadcast_compatible(&b));
    println!(
        "a and incompatible compatible: {}",
        a.broadcast_compatible(&incompatible)
    );
}
