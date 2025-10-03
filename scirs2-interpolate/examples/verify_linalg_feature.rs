use scirs2_core::ndarray::{Array1, Array2};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing scirs2-interpolate linalg feature configuration");

    // Create some test arrays
    let a = Array2::<f64>::eye(3);
    let b = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);

    // Check if linalg feature is enabled
    #[cfg(feature = "linalg")]
    {
        println!("✅ linalg feature is ENABLED");
        println!("This build has linalg feature enabled (currently provides compatibility)");

        // Since ndarray-linalg dependency is removed, demonstrate fallback approach
        println!("Using scirs2-linalg module for linear algebra operations");

        // Example of what would be done with the linalg feature
        println!("Would solve A*x = b where A = {:?} and b = {:?}", a, b);
        println!("Using internal linear algebra implementations from scirs2-linalg");
    }

    #[cfg(not(feature = "linalg"))]
    {
        println!("ℹ️ linalg feature is NOT enabled");
        println!("This build uses fallback implementations that don't require OpenBLAS");
        println!("Enable with: cargo run --example verify_linalg_feature --features linalg");

        // In the real code, we'd use a fallback implementation here
        // For this example, we'll just output what would happen
        println!("Using fallback methods for linear algebra operations");

        // Use the variables to avoid warnings
        println!(
            "Would solve A*x = b where A shape: {:?} and b shape: {:?}",
            a.shape(),
            b.shape()
        );
    }

    // Always runs regardless of feature
    println!("\nTest completed successfully!");

    Ok(())
}
