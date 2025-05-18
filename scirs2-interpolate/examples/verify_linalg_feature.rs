use ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing scirs2-interpolate linalg feature configuration");

    // Create some test arrays
    let a = Array2::<f64>::eye(3);
    let b = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);

    // Check if linalg feature is enabled
    #[cfg(feature = "linalg")]
    {
        println!("✅ linalg feature is ENABLED");
        println!("This build can use OpenBLAS for advanced linear algebra operations");

        // Import the Solve trait only when linalg feature is enabled
        use ndarray_linalg::Solve;

        // Demonstrate solving a system using ndarray-linalg
        match a.solve(&b) {
            Ok(x) => println!("Solved system: x = {:?}", x),
            Err(e) => println!("Error solving system: {:?}", e),
        }
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
