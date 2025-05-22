//! Spherical Bessel Functions Visualization Example
//!
//! This example generates a CSV file with values of spherical Bessel functions
//! for visualization purposes. It demonstrates the stability of the implementation
//! across different ranges of arguments and orders.
//!
//! The CSV format is:
//! x, j₀(x), j₁(x), j₂(x), j₃(x), y₀(x), y₁(x), j₀_scaled(x), j₁_scaled(x)

use scirs2_special::bessel::spherical::{spherical_jn, spherical_jn_scaled, spherical_yn};
use std::fs::File;
use std::io::{self, Write};

fn main() -> io::Result<()> {
    // Create output file
    let mut file = File::create("spherical_bessel_functions.csv")?;

    // Write CSV header
    writeln!(
        file,
        "x,j₀(x),j₁(x),j₂(x),j₃(x),y₀(x),y₁(x),j₀_scaled(x),j₁_scaled(x)"
    )?;

    // Small values (dense sampling)
    let mut x = 0.01; // Start at small positive value since y_n diverges at 0
    while x <= 1.0 {
        write_bessel_values(&mut file, x)?;
        x += 0.01;
    }

    // Medium values
    while x <= 10.0 {
        write_bessel_values(&mut file, x)?;
        x += 0.1;
    }

    // Large values (to show stability in oscillatory region)
    while x <= 100.0 {
        write_bessel_values(&mut file, x)?;
        x += 1.0;
    }

    // Very large values (to demonstrate stability at large arguments)
    for x in [150.0, 200.0, 250.0, 300.0, 500.0, 1000.0].iter() {
        write_bessel_values(&mut file, *x)?;
    }

    println!("Generated spherical_bessel_functions.csv for visualization");
    println!("You can plot this data using any plotting tool (e.g., Python with matplotlib)");

    Ok(())
}

fn write_bessel_values(file: &mut File, x: f64) -> io::Result<()> {
    // Calculate regular spherical Bessel functions
    let j0 = spherical_jn(0, x);
    let j1 = spherical_jn(1, x);
    let j2 = spherical_jn(2, x);
    let j3 = spherical_jn(3, x);
    let y0 = spherical_yn(0, x);
    let y1 = spherical_yn(1, x);

    // Calculate scaled versions (which are more stable for large arguments)
    let j0_scaled = spherical_jn_scaled(0, x);
    let j1_scaled = spherical_jn_scaled(1, x);

    // Write row to CSV
    writeln!(
        file,
        "{:.6},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10}",
        x, j0, j1, j2, j3, y0, y1, j0_scaled, j1_scaled
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format() {
        // Check that a few sample points produce output without errors
        let test_points = [0.1, 1.0, 10.0, 100.0];

        for &x in &test_points {
            // This would fail if the function crashed
            let j0: f64 = spherical_jn(0, x);
            let j1: f64 = spherical_jn(1, x);

            // Verify values are finite
            assert!(j0.is_finite());
            assert!(j1.is_finite());
        }
    }
}
