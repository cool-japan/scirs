//! Example demonstrating accelerated neural operations
//!
//! This example shows how to use the enhanced neural operations framework
//! that provides a foundation for GPU acceleration while currently using
//! optimized CPU implementations.

use scirs2_core::ndarray::array;
use scirs2_neural::error::Result;
// use scirs2_neural::gpu::{create_neural_ops, create_neural_ops_with_backend}; // GPU module disabled

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("=== Accelerated Neural Operations Demo ===\n");
    println!("Note: GPU acceleration is not available in the minimal version of scirs2-neural.");
    println!("This example is a placeholder for future GPU functionality.");

    // Simple matrix multiplication demo using ndarray
    println!("\n1. Basic Matrix Multiplication (using ndarray):");
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
    println!("Matrix A (2x3):\n{a:?}");
    println!("Matrix B (3x2):\n{b:?}");
    let result = a.dot(&b);
    println!("Result A * B (2x2):\n{result:?}\n");

    println!("To enable GPU acceleration, add the appropriate features to Cargo.toml:");
    println!("scirs2-neural = {{ features = [\"gpu\"] }}");

    println!("\n=== Demo Complete ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_runs() {
        // Test that the example runs without panicking
        assert!(main().is_ok());
    }
}
