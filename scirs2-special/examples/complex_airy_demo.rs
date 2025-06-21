//! Complex Airy Functions Demonstration
//!
//! This example showcases the complex number support for Airy functions
//! in scirs2-special, including Ai, Ai', Bi, and Bi' functions.

use num_complex::Complex64;
use scirs2_special::{ai_complex, aip_complex, bi_complex, bip_complex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SCIRS2-SPECIAL Complex Airy Functions Demo ===\n");

    // 1. Basic Airy Functions
    demo_basic_airy_functions()?;

    // 2. Airy Function Derivatives
    demo_airy_derivatives()?;

    // 3. Mathematical Properties
    demo_mathematical_properties()?;

    // 4. Complex Arguments
    demo_complex_arguments()?;

    println!("=== Complex Airy functions demo completed successfully! ===");
    Ok(())
}

fn demo_basic_airy_functions() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Complex Airy Functions of the First Kind - Ai(z)");
    println!("==================================================");

    let test_values = [
        Complex64::new(0.0, 0.0),  // Ai(0) ≈ 0.3550
        Complex64::new(1.0, 0.0),  // Real value
        Complex64::new(2.0, 0.0),  // Real value
        Complex64::new(-1.0, 0.0), // Negative real value
        Complex64::new(0.0, 1.0),  // Pure imaginary
        Complex64::new(1.0, 1.0),  // Complex value
    ];

    for &z in &test_values {
        let ai_result = ai_complex(z);
        println!(
            "  Ai({:.1} + {:.1}i) = {:.6} + {:.6}i",
            z.re, z.im, ai_result.re, ai_result.im
        );
    }

    println!();
    Ok(())
}

fn demo_airy_derivatives() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Complex Airy Function Derivatives");
    println!("===================================");

    println!("Ai'(z) - Derivative of Airy function of the first kind:");
    let test_values = [
        Complex64::new(0.0, 0.0), // Ai'(0) ≈ -0.2588
        Complex64::new(1.0, 0.0), // Real value
        Complex64::new(0.0, 1.0), // Pure imaginary
        Complex64::new(1.0, 1.0), // Complex value
    ];

    for &z in &test_values {
        let aip_result = aip_complex(z);
        println!(
            "  Ai'({:.1} + {:.1}i) = {:.6} + {:.6}i",
            z.re, z.im, aip_result.re, aip_result.im
        );
    }

    println!("\nBi(z) and Bi'(z) - Airy functions of the second kind:");
    for &z in &test_values {
        let bi_result = bi_complex(z);
        let bip_result = bip_complex(z);
        println!(
            "  Bi({:.1} + {:.1}i) = {:.6} + {:.6}i",
            z.re, z.im, bi_result.re, bi_result.im
        );
        println!(
            "  Bi'({:.1} + {:.1}i) = {:.6} + {:.6}i",
            z.re, z.im, bip_result.re, bip_result.im
        );
    }

    println!();
    Ok(())
}

fn demo_mathematical_properties() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Mathematical Properties Verification");
    println!("======================================");

    // Test Wronskian identity: Ai(z)*Bi'(z) - Ai'(z)*Bi(z) = 1/π
    println!("Wronskian Identity: Ai(z)*Bi'(z) - Ai'(z)*Bi(z) = 1/π");
    let test_values = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.5, 0.0),
    ];

    for &z in &test_values {
        let ai_z = ai_complex(z);
        let aip_z = aip_complex(z);
        let bi_z = bi_complex(z);
        let bip_z = bip_complex(z);

        let wronskian = ai_z * bip_z - aip_z * bi_z;
        let expected = 1.0 / std::f64::consts::PI;

        println!(
            "  z = {:.1} + {:.1}i: Wronskian = {:.6} + {:.6}i (expected: {:.6})",
            z.re, z.im, wronskian.re, wronskian.im, expected
        );
    }

    println!();

    // Test real value consistency
    println!("Real Value Consistency: Complex functions match real functions for real arguments");
    let real_values = [0.0, 1.0, 2.0, -1.0];

    for &x in &real_values {
        let z = Complex64::new(x, 0.0);
        let ai_complex_result = ai_complex(z);
        let ai_real_result = scirs2_special::ai(x);

        let error = (ai_complex_result.re - ai_real_result).abs();
        println!(
            "  Ai({:.1}): Complex = {:.6}, Real = {:.6}, Error = {:.2e}",
            x, ai_complex_result.re, ai_real_result, error
        );
    }

    println!();
    Ok(())
}

fn demo_complex_arguments() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Complex Arguments and Special Cases");
    println!("=====================================");

    // Pure imaginary arguments
    println!("Pure imaginary arguments:");
    let imaginary_values = [
        Complex64::new(0.0, 0.5),
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 2.0),
    ];

    for &z in &imaginary_values {
        let ai_result = ai_complex(z);
        let bi_result = bi_complex(z);

        println!("  z = {:.1}i:", z.im);
        println!(
            "    Ai({:.1}i) = {:.6} + {:.6}i",
            z.im, ai_result.re, ai_result.im
        );
        println!(
            "    Bi({:.1}i) = {:.6} + {:.6}i",
            z.im, bi_result.re, bi_result.im
        );
    }

    println!();

    // Large arguments - showing asymptotic behavior
    println!("Large arguments (asymptotic behavior):");
    let large_values = [
        Complex64::new(5.0, 0.0),
        Complex64::new(0.0, 5.0),
        Complex64::new(3.0, 3.0),
    ];

    for &z in &large_values {
        let ai_result = ai_complex(z);
        let bi_result = bi_complex(z);

        println!("  z = {:.1} + {:.1}i:", z.re, z.im);
        println!("    Ai(z) = {:.2e} + {:.2e}i", ai_result.re, ai_result.im);
        println!("    Bi(z) = {:.2e} + {:.2e}i", bi_result.re, bi_result.im);
    }

    println!();

    // Application: Wave functions in quantum mechanics
    println!("Application: Quantum mechanical wave functions");
    println!("The Airy function appears as solutions to the Schrödinger equation");
    println!("with a linear potential (quantum tunneling problems).");

    let turning_point = Complex64::new(0.0, 0.0);
    let classical_region = Complex64::new(-2.0, 0.0);
    let quantum_region = Complex64::new(2.0, 0.0);

    println!(
        "  At turning point z=0: Ai(0) = {:.6}",
        ai_complex(turning_point).re
    );
    println!(
        "  In classical region z=-2: Ai(-2) = {:.6}",
        ai_complex(classical_region).re
    );
    println!(
        "  In quantum region z=2: Ai(2) = {:.6}",
        ai_complex(quantum_region).re
    );

    println!();
    Ok(())
}
