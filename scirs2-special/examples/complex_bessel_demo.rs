//! Complex Bessel Functions Demonstration
//!
//! This example showcases the complex number support for Bessel functions
//! in scirs2-special, including J₀, J₁, Jₙ, Jᵥ, I₀, and K₀ functions.

use num_complex::Complex64;
use scirs2_special::{i0_complex, j0_complex, j1_complex, jn_complex, jv_complex, k0_complex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SCIRS2-SPECIAL Complex Bessel Functions Demo ===\n");

    // 1. Bessel Functions of the First Kind
    demo_first_kind_bessel()?;

    // 2. Modified Bessel Functions
    demo_modified_bessel()?;

    // 3. Half-Integer Orders
    demo_half_integer_bessel()?;

    // 4. Mathematical Properties
    demo_mathematical_properties()?;

    // 5. Complex Arguments
    demo_complex_arguments()?;

    println!("=== Complex Bessel functions demo completed successfully! ===");
    Ok(())
}

fn demo_first_kind_bessel() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Bessel Functions of the First Kind");
    println!("====================================");

    // J₀(z) function
    println!("J₀(z) - First kind, order 0:");
    let test_values = [
        Complex64::new(0.0, 0.0),    // J₀(0) = 1
        Complex64::new(1.0, 0.0),    // Real value
        Complex64::new(2.4048, 0.0), // First zero of J₀
        Complex64::new(1.0, 1.0),    // Complex value
        Complex64::new(0.0, 1.0),    // Pure imaginary
    ];

    for &z in &test_values {
        let result = j0_complex(z);
        println!(
            "  J₀({:.4} + {:.4}i) = {:.6} + {:.6}i",
            z.re, z.im, result.re, result.im
        );
    }

    println!();

    // J₁(z) function
    println!("J₁(z) - First kind, order 1:");
    for &z in &test_values {
        let result = j1_complex(z);
        println!(
            "  J₁({:.4} + {:.4}i) = {:.6} + {:.6}i",
            z.re, z.im, result.re, result.im
        );
    }

    println!();

    // Jₙ(z) function for various orders
    println!("Jₙ(z) - First kind, integer orders:");
    let z = Complex64::new(2.0, 0.0);
    for n in 0..=5 {
        let result = jn_complex(n, z);
        println!(
            "  J{}({:.1}) = {:.6} + {:.6}i",
            n, z.re, result.re, result.im
        );
    }

    println!();
    Ok(())
}

fn demo_modified_bessel() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Modified Bessel Functions");
    println!("============================");

    // I₀(z) function
    println!("I₀(z) - Modified first kind, order 0:");
    let test_values = [
        Complex64::new(0.0, 0.0), // I₀(0) = 1
        Complex64::new(1.0, 0.0), // Real value
        Complex64::new(2.0, 0.0), // Real value
        Complex64::new(1.0, 1.0), // Complex value
        Complex64::new(0.0, 1.0), // Pure imaginary
    ];

    for &z in &test_values {
        let result = i0_complex(z);
        println!(
            "  I₀({:.1} + {:.1}i) = {:.6} + {:.6}i",
            z.re, z.im, result.re, result.im
        );
    }

    println!();

    // K₀(z) function
    println!("K₀(z) - Modified second kind, order 0:");
    let test_values = [
        Complex64::new(0.5, 0.0), // Real small value
        Complex64::new(1.0, 0.0), // Real value
        Complex64::new(2.0, 0.0), // Real value
        Complex64::new(1.0, 1.0), // Complex value
        Complex64::new(0.5, 0.5), // Complex value
    ];

    for &z in &test_values {
        let result = k0_complex(z);
        println!(
            "  K₀({:.1} + {:.1}i) = {:.6} + {:.6}i",
            z.re, z.im, result.re, result.im
        );
    }

    println!();
    Ok(())
}

fn demo_half_integer_bessel() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Half-Integer Order Bessel Functions");
    println!("======================================");

    // Half-integer orders have simple expressions in terms of elementary functions
    let test_values = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(pi(), 0.0),
        Complex64::new(1.0, 0.5),
    ];

    let orders = [0.5, 1.5, 2.5, -0.5];

    for &v in &orders {
        println!("Jᵥ(z) with ν = {}:", v);
        for &z in &test_values {
            let result = jv_complex(v, z);
            println!(
                "  J{:.1}({:.1} + {:.1}i) = {:.6} + {:.6}i",
                v, z.re, z.im, result.re, result.im
            );
        }
        println!();
    }

    Ok(())
}

fn demo_mathematical_properties() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Mathematical Properties Verification");
    println!("======================================");

    // Test recurrence relation: J_{n-1}(z) + J_{n+1}(z) = (2n/z)J_n(z)
    println!("Recurrence relation verification:");
    println!("J_{{n-1}}(z) + J_{{n+1}}(z) = (2n/z)J_n(z)");

    let test_cases = [
        (Complex64::new(2.0, 0.0), 2),
        (Complex64::new(1.0, 1.0), 1),
        (Complex64::new(3.0, 0.5), 3),
    ];

    for &(z, n) in &test_cases {
        let j_nm1 = jn_complex(n - 1, z);
        let j_n = jn_complex(n, z);
        let j_np1 = jn_complex(n + 1, z);

        let lhs = j_nm1 + j_np1;
        let rhs = Complex64::new(2.0 * n as f64, 0.0) / z * j_n;
        let error = (lhs - rhs).norm();

        println!(
            "  z = {:.1} + {:.1}i, n = {}: error = {:.2e}",
            z.re, z.im, n, error
        );
    }

    println!();

    // Test Wronskian: J_n(z)Y_{n+1}(z) - J_{n+1}(z)Y_n(z) = -2/(πz)
    println!("Wronskian properties:");
    println!("For modified Bessel functions: W{{I_0, K_0}} = -1/z");

    let test_values = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.5),
        Complex64::new(0.5, 1.0),
    ];

    for &z in &test_values {
        let i0_z = i0_complex(z);
        let k0_z = k0_complex(z);

        // The Wronskian for I₀ and K₀ is -1/z
        let _expected_wronskian = -Complex64::new(1.0, 0.0) / z;

        // For verification, we'd need derivatives, so we'll just show the values
        println!(
            "  z = {:.1} + {:.1}i: I₀(z) = {:.4} + {:.4}i, K₀(z) = {:.4} + {:.4}i",
            z.re, z.im, i0_z.re, i0_z.im, k0_z.re, k0_z.im
        );
    }

    println!();
    Ok(())
}

fn demo_complex_arguments() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Complex Arguments and Special Cases");
    println!("=====================================");

    // Pure imaginary arguments
    println!("Pure imaginary arguments:");
    let imaginary_values = [
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 2.0),
        Complex64::new(0.0, 5.0),
    ];

    for &z in &imaginary_values {
        let j0_result = j0_complex(z);
        let i0_result = i0_complex(z);

        println!("  z = {:.1}i:", z.im);
        println!(
            "    J₀({:.1}i) = {:.6} + {:.6}i",
            z.im, j0_result.re, j0_result.im
        );
        println!(
            "    I₀({:.1}i) = {:.6} + {:.6}i",
            z.im, i0_result.re, i0_result.im
        );
    }

    println!();

    // Large arguments
    println!("Large arguments (asymptotic behavior):");
    let large_values = [
        Complex64::new(10.0, 0.0),
        Complex64::new(20.0, 0.0),
        Complex64::new(10.0, 10.0),
    ];

    for &z in &large_values {
        let j0_result = j0_complex(z);
        let j1_result = j1_complex(z);

        println!("  z = {:.1} + {:.1}i:", z.re, z.im);
        println!("    J₀(z) = {:.6} + {:.6}i", j0_result.re, j0_result.im);
        println!("    J₁(z) = {:.6} + {:.6}i", j1_result.re, j1_result.im);
    }

    println!();

    // Negative orders
    println!("Negative orders:");
    let z = Complex64::new(2.0, 1.0);
    let negative_orders = [-1, -2, -3];

    for &n in &negative_orders {
        let result = jn_complex(n, z);
        println!(
            "  J{}({:.1} + {:.1}i) = {:.6} + {:.6}i",
            n, z.re, z.im, result.re, result.im
        );
    }

    println!();
    Ok(())
}

// Helper function for π
fn pi() -> f64 {
    std::f64::consts::PI
}
