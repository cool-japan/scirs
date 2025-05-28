//! Bessel Function Visualization Example
//!
//! This example generates a CSV file with values of various Bessel functions
//! that can be used for visualization with plotting tools.
//!
//! The CSV format is:
//! x, J0(x), J1(x), J2(x), J3(x), Y0(x), Y1(x), I0(x), I1(x), K0(x), K1(x)

use scirs2_special::bessel::{i0, i1, j0, j1, jn, k0, k1, y0, y1};
use std::fs::File;
use std::io::{self, Write};

fn main() -> io::Result<()> {
    // Create output file
    let mut file = File::create("bessel_functions.csv")?;

    // Write CSV header
    writeln!(
        file,
        "x,J0(x),J1(x),J2(x),J3(x),Y0(x),Y1(x),I0(x),I1(x),K0(x),K1(x)"
    )?;

    // Generate values from 0.1 to 20.0
    let mut x = 0.1;
    while x <= 20.0 {
        // Evaluate Bessel functions
        let j0_val = j0(x);
        let j1_val = j1(x);
        let j2_val = jn(2, x);
        let j3_val = jn(3, x);
        let y0_val = y0(x);
        let y1_val = y1(x);
        let i0_val = i0(x);
        let i1_val = i1(x);
        let k0_val = k0(x);
        let k1_val = k1(x);

        // Write row to CSV
        writeln!(
            file,
            "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            x, j0_val, j1_val, j2_val, j3_val, y0_val, y1_val, i0_val, i1_val, k0_val, k1_val
        )?;

        // Increase x with smaller steps for lower values to better capture oscillations
        if x < 1.0 {
            x += 0.05;
        } else if x < 5.0 {
            x += 0.1;
        } else {
            x += 0.2;
        }
    }

    println!("Generated bessel_functions.csv for visualization");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_values() {
        // Test that all Bessel functions produce valid results for a range of inputs
        for x in [0.5, 1.0, 5.0, 10.0].iter() {
            // Test that regular Bessel functions produce finite values
            let j0_val: f64 = j0(*x);
            let j1_val: f64 = j1(*x);
            let j2_val: f64 = jn(2, *x);

            // Check that values are finite (not NaN or infinite)
            assert!(
                j0_val.is_finite(),
                "J0({}) = {} should be finite",
                x,
                j0_val
            );
            assert!(
                j1_val.is_finite(),
                "J1({}) = {} should be finite",
                x,
                j1_val
            );
            assert!(
                j2_val.is_finite(),
                "J2({}) = {} should be finite",
                x,
                j2_val
            );

            // Modified functions should be positive for positive inputs
            assert!(i0(*x) > 0.0);
            assert!(i1(*x) > 0.0);
            assert!(k0(*x) > 0.0);
            assert!(k1(*x) > 0.0);
        }
    }
}
