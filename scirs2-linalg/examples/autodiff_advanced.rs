//! Example demonstrating advanced linear algebra operations with automatic differentiation
//!
//! This example shows how to use matrix inverse, SVD, eigendecomposition, and matrix
//! exponential with gradient tracking.

#[cfg(feature = "autograd")]
fn main() {
    println!("Advanced matrix operations with automatic differentiation");
    println!("=======================================================");
    println!();
    println!("This example is currently disabled as the advanced autograd API");
    println!("is under development. The var_* functions used in this example");
    println!("are not yet implemented in the current API.");
    println!();
    println!("For working autograd examples, see:");
    println!("  cargo run --example autograd_simple_example --features autograd");
    println!();
    println!("The current autograd integration uses the scirs2-autograd crate directly.");
}

#[cfg(not(feature = "autograd"))]
fn main() {
    println!("This example requires the 'autograd' feature.");
    println!("Run with: cargo run --example autodiff_advanced --features autograd");
}
