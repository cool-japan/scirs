//! Comprehensive automatic differentiation example
//!
//! This example demonstrates various complex automatic differentiation
//! operations with advanced linear algebra computations.
//!
//! NOTE: This example is currently disabled due to API incompatibilities
//! in the scirs2-autograd integration. The API is under active development.

#[cfg(feature = "autograd")]
fn main() {
    println!("Comprehensive Automatic Differentiation Example");
    println!("===============================================\n");
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
    println!("cargo run --example autodiff_comprehensive --features=autograd");
}
