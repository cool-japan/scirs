//! Automatic differentiation example for linear algebra operations
//!
//! This example demonstrates the use of automatic differentiation
//! with various linear algebra operations in scirs2-linalg.
//!
//! NOTE: This example is currently disabled due to API incompatibilities
//! in the scirs2-autograd integration. The API is under active development.

#[cfg(feature = "autograd")]
fn main() {
    println!("Automatic Differentiation in Linear Algebra");
    println!("===========================================\n");
    println!("This example is currently disabled due to API incompatibilities");
    println!("in the scirs2-autograd integration. Several functions used in this");
    println!("example have changed signatures or are not yet implemented.");
    println!();
    println!("For working autograd examples, see:");
    println!("  cargo run --example autograd_simple_example --features autograd");
    println!();
    println!("The autograd integration is being actively developed and the API");
    println!("will be stabilized in future releases.");
}

#[cfg(not(feature = "autograd"))]
fn main() {
    println!("This example requires the 'autograd' feature. Run with:");
    println!("cargo run --example autograd_example --features=autograd");
}
