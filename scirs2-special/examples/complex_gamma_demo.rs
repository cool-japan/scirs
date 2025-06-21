//! Complex Gamma Functions Demonstration
//!
//! This example showcases the complex number support for gamma functions
//! in scirs2-special, including gamma, loggamma, digamma, and beta functions.

use num_complex::Complex64;
use scirs2_special::{beta_complex, digamma_complex, gamma_complex, loggamma_complex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SCIRS2-SPECIAL Complex Gamma Functions Demo ===\n");

    // 1. Basic Complex Gamma Function
    demo_complex_gamma()?;

    // 2. Complex Log Gamma Function
    demo_complex_loggamma()?;

    // 3. Complex Digamma Function
    demo_complex_digamma()?;

    // 4. Complex Beta Function
    demo_complex_beta()?;

    // 5. Mathematical Properties
    demo_mathematical_properties()?;

    println!("=== Complex gamma functions demo completed successfully! ===");
    Ok(())
}

fn demo_complex_gamma() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Complex Gamma Function Γ(z)");
    println!("==============================");

    let test_values = [
        Complex64::new(1.0, 0.0), // Real: Γ(1) = 1
        Complex64::new(2.0, 0.0), // Real: Γ(2) = 1
        Complex64::new(5.0, 0.0), // Real: Γ(5) = 24
        Complex64::new(0.5, 0.0), // Real: Γ(0.5) = √π
        Complex64::new(1.0, 1.0), // Complex
        Complex64::new(2.0, 1.0), // Complex
        Complex64::new(0.5, 0.5), // Complex
    ];

    for &z in &test_values {
        let result = gamma_complex(z);
        println!(
            "  Γ({:.1} + {:.1}i) = {:.6} + {:.6}i",
            z.re, z.im, result.re, result.im
        );
    }

    println!();
    Ok(())
}

fn demo_complex_loggamma() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Complex Log Gamma Function log(Γ(z))");
    println!("=======================================");

    let test_values = [
        Complex64::new(1.0, 0.0),  // log(Γ(1)) = log(1) = 0
        Complex64::new(2.0, 0.0),  // log(Γ(2)) = log(1) = 0
        Complex64::new(5.0, 0.0),  // log(Γ(5)) = log(24) ≈ 3.178
        Complex64::new(10.0, 0.0), // Large real value
        Complex64::new(1.0, 1.0),  // Complex
        Complex64::new(2.0, 1.0),  // Complex
    ];

    for &z in &test_values {
        let result = loggamma_complex(z);
        println!(
            "  log(Γ({:.1} + {:.1}i)) = {:.6} + {:.6}i",
            z.re, z.im, result.re, result.im
        );
    }

    println!();
    Ok(())
}

fn demo_complex_digamma() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Complex Digamma Function ψ(z)");
    println!("=================================");

    let test_values = [
        Complex64::new(1.0, 0.0),  // ψ(1) = -γ ≈ -0.5772
        Complex64::new(2.0, 0.0),  // ψ(2) = 1 - γ ≈ 0.4228
        Complex64::new(10.0, 0.0), // Large real value
        Complex64::new(1.0, 1.0),  // Complex
        Complex64::new(2.0, 1.0),  // Complex
    ];

    for &z in &test_values {
        let result = digamma_complex(z);
        println!(
            "  ψ({:.1} + {:.1}i) = {:.6} + {:.6}i",
            z.re, z.im, result.re, result.im
        );
    }

    println!();
    Ok(())
}

fn demo_complex_beta() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Complex Beta Function B(a,b)");
    println!("===============================");

    let test_pairs = [
        (Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)), // B(1,1) = 1
        (Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)), // B(2,3) = 1/12
        (Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.0)), // B(0.5,0.5) = π
        (Complex64::new(1.0, 0.5), Complex64::new(1.0, -0.5)), // Complex pair
        (Complex64::new(2.0, 1.0), Complex64::new(3.0, 1.0)), // Complex pair
    ];

    for &(a, b) in &test_pairs {
        let result = beta_complex(a, b);
        println!(
            "  B({:.1} + {:.1}i, {:.1} + {:.1}i) = {:.6} + {:.6}i",
            a.re, a.im, b.re, b.im, result.re, result.im
        );
    }

    println!();
    Ok(())
}

fn demo_mathematical_properties() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Mathematical Properties Verification");
    println!("======================================");

    // Test gamma recurrence relation: Γ(z+1) = z * Γ(z)
    println!("Gamma recurrence relation: Γ(z+1) = z * Γ(z)");

    let test_values = [
        Complex64::new(1.5, 0.5),
        Complex64::new(2.0, 1.0),
        Complex64::new(0.8, -0.3),
    ];

    for &z in &test_values {
        let gamma_z = gamma_complex(z);
        let gamma_z_plus_1 = gamma_complex(z + Complex64::new(1.0, 0.0));
        let expected = z * gamma_z;
        let error = (gamma_z_plus_1 - expected).norm();

        println!("  z = {:.1} + {:.1}i, error = {:.2e}", z.re, z.im, error);
    }

    println!();

    // Test beta function symmetry: B(a,b) = B(b,a)
    println!("Beta function symmetry: B(a,b) = B(b,a)");

    let test_pairs = [
        (Complex64::new(1.5, 0.5), Complex64::new(2.0, 0.3)),
        (Complex64::new(2.0, 1.0), Complex64::new(1.0, -0.5)),
    ];

    for &(a, b) in &test_pairs {
        let beta_ab = beta_complex(a, b);
        let beta_ba = beta_complex(b, a);
        let error = (beta_ab - beta_ba).norm();

        println!(
            "  a = {:.1} + {:.1}i, b = {:.1} + {:.1}i, error = {:.2e}",
            a.re, a.im, b.re, b.im, error
        );
    }

    println!();

    // Test consistency: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    println!("Beta-Gamma relationship: B(a,b) = Γ(a)Γ(b)/Γ(a+b)");

    for &(a, b) in &test_pairs {
        let beta_result = beta_complex(a, b);
        let gamma_a = gamma_complex(a);
        let gamma_b = gamma_complex(b);
        let gamma_ab = gamma_complex(a + b);
        let gamma_ratio = gamma_a * gamma_b / gamma_ab;
        let error = (beta_result - gamma_ratio).norm();

        println!(
            "  a = {:.1} + {:.1}i, b = {:.1} + {:.1}i, error = {:.2e}",
            a.re, a.im, b.re, b.im, error
        );
    }

    println!();
    Ok(())
}
