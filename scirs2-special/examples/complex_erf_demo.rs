//! Complex Error Functions Demonstration
//!
//! This example showcases the complex number support for error functions
//! in scirs2-special, including erf, erfc, erfcx, and Faddeeva functions.

use num_complex::Complex64;
use scirs2_special::{erf_complex, erfc_complex, erfcx_complex, faddeeva_complex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SCIRS2-SPECIAL Complex Error Functions Demo ===\n");

    // 1. Basic Error Functions
    demo_basic_error_functions()?;

    // 2. Complementary Error Function
    demo_complementary_error_function()?;

    // 3. Scaled Complementary Error Function
    demo_scaled_error_function()?;

    // 4. Faddeeva Function
    demo_faddeeva_function()?;

    // 5. Mathematical Properties
    demo_mathematical_properties()?;

    // 6. Special Cases and Applications
    demo_special_cases()?;

    println!("=== Complex error functions demo completed successfully! ===");
    Ok(())
}

fn demo_basic_error_functions() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Complex Error Function erf(z)");
    println!("=================================");

    let test_values = [
        Complex64::new(0.0, 0.0),  // erf(0) = 0
        Complex64::new(1.0, 0.0),  // Real value: erf(1) ≈ 0.8427
        Complex64::new(2.0, 0.0),  // Real value: erf(2) ≈ 0.9953
        Complex64::new(0.0, 1.0),  // Pure imaginary
        Complex64::new(1.0, 1.0),  // Complex value
        Complex64::new(-1.0, 0.5), // Negative real part
        Complex64::new(0.5, -0.5), // Negative imaginary part
    ];

    for &z in &test_values {
        let result = erf_complex(z);
        println!(
            "  erf({:.1} + {:.1}i) = {:.6} + {:.6}i",
            z.re, z.im, result.re, result.im
        );
    }

    println!();
    Ok(())
}

fn demo_complementary_error_function() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Complex Complementary Error Function erfc(z)");
    println!("===============================================");

    let test_values = [
        Complex64::new(0.0, 0.0),  // erfc(0) = 1
        Complex64::new(1.0, 0.0),  // Real value: erfc(1) ≈ 0.1573
        Complex64::new(2.0, 0.0),  // Real value: erfc(2) ≈ 0.0047
        Complex64::new(0.0, 1.0),  // Pure imaginary
        Complex64::new(1.0, 1.0),  // Complex value
        Complex64::new(-1.0, 0.0), // Negative real
        Complex64::new(0.5, 0.5),  // Complex value
    ];

    for &z in &test_values {
        let result = erfc_complex(z);
        println!(
            "  erfc({:.1} + {:.1}i) = {:.6} + {:.6}i",
            z.re, z.im, result.re, result.im
        );
    }

    println!();
    Ok(())
}

fn demo_scaled_error_function() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Complex Scaled Complementary Error Function erfcx(z)");
    println!("=======================================================");

    println!("erfcx(z) = e^(z²) * erfc(z) - useful for large |z| to avoid overflow");

    let test_values = [
        Complex64::new(1.0, 0.0), // Real value
        Complex64::new(2.0, 0.0), // Large real value
        Complex64::new(5.0, 0.0), // Very large real value
        Complex64::new(1.0, 1.0), // Complex value
        Complex64::new(2.0, 1.0), // Complex value with large real part
        Complex64::new(0.5, 2.0), // Complex value with large imaginary part
    ];

    for &z in &test_values {
        let result = erfcx_complex(z);
        println!(
            "  erfcx({:.1} + {:.1}i) = {:.6} + {:.6}i",
            z.re, z.im, result.re, result.im
        );

        // For large real z, erfcx(z) ≈ 1/(√π * z)
        if z.im.abs() < 1e-10 && z.re > 1.0 {
            let asymptotic = 1.0 / (std::f64::consts::PI.sqrt() * z.re);
            println!("    Asymptotic approximation: {:.6}", asymptotic);
        }
    }

    println!();
    Ok(())
}

fn demo_faddeeva_function() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Complex Faddeeva Function w(z)");
    println!("==================================");

    println!("w(z) = e^(-z²) * erfc(-iz) - appears in plasma physics and spectroscopy");

    let test_values = [
        Complex64::new(0.0, 0.0),  // w(0) = 1
        Complex64::new(1.0, 0.0),  // Real value
        Complex64::new(0.0, 1.0),  // Pure imaginary
        Complex64::new(1.0, 1.0),  // Complex value
        Complex64::new(2.0, 0.5),  // Complex value
        Complex64::new(0.5, 2.0),  // Complex value
        Complex64::new(-1.0, 1.0), // Negative real part
    ];

    for &z in &test_values {
        let result = faddeeva_complex(z);
        println!(
            "  w({:.1} + {:.1}i) = {:.6} + {:.6}i",
            z.re, z.im, result.re, result.im
        );
    }

    println!();
    Ok(())
}

fn demo_mathematical_properties() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Mathematical Properties Verification");
    println!("======================================");

    // Test erf(z) + erfc(z) = 1
    println!("Property: erf(z) + erfc(z) = 1");
    let test_values = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(-0.5, 2.0),
    ];

    for &z in &test_values {
        let erf_z = erf_complex(z);
        let erfc_z = erfc_complex(z);
        let sum = erf_z + erfc_z;
        let error = (sum - Complex64::new(1.0, 0.0)).norm();

        println!("  z = {:.1} + {:.1}i: error = {:.2e}", z.re, z.im, error);
    }

    println!();

    // Test erf(-z) = -erf(z) (odd function)
    println!("Property: erf(-z) = -erf(z) (odd function)");
    for &z in &test_values {
        let erf_z = erf_complex(z);
        let erf_minus_z = erf_complex(-z);
        let error = (erf_minus_z + erf_z).norm();

        println!("  z = {:.1} + {:.1}i: error = {:.2e}", z.re, z.im, error);
    }

    println!();

    // Test erfcx(z) = e^(z²) * erfc(z)
    println!("Property: erfcx(z) = e^(z²) * erfc(z)");
    let small_values = [
        Complex64::new(0.5, 0.0),
        Complex64::new(1.0, 0.5),
        Complex64::new(0.5, 1.0),
    ];

    for &z in &small_values {
        let erfcx_z = erfcx_complex(z);
        let erfc_z = erfc_complex(z);
        let exp_z2 = (z * z).exp();
        let expected = exp_z2 * erfc_z;
        let error = (erfcx_z - expected).norm();

        println!("  z = {:.1} + {:.1}i: error = {:.2e}", z.re, z.im, error);
    }

    println!();
    Ok(())
}

fn demo_special_cases() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Special Cases and Applications");
    println!("================================");

    // Pure imaginary arguments
    println!("Pure imaginary arguments:");
    let imaginary_values = [
        Complex64::new(0.0, 0.5),
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 2.0),
    ];

    for &z in &imaginary_values {
        let erf_result = erf_complex(z);
        let erfc_result = erfc_complex(z);

        println!("  z = {:.1}i:", z.im);
        println!(
            "    erf({:.1}i) = {:.6} + {:.6}i",
            z.im, erf_result.re, erf_result.im
        );
        println!(
            "    erfc({:.1}i) = {:.6} + {:.6}i",
            z.im, erfc_result.re, erfc_result.im
        );

        // For pure imaginary z = iy, erf(iy) should be pure imaginary
        println!(
            "    Real part of erf(iy): {:.2e} (should be ~0)",
            erf_result.re.abs()
        );
    }

    println!();

    // Large arguments - showing asymptotic behavior
    println!("Large arguments (asymptotic behavior):");
    let large_values = [
        Complex64::new(5.0, 0.0),
        Complex64::new(10.0, 0.0),
        Complex64::new(3.0, 3.0),
    ];

    for &z in &large_values {
        let erf_result = erf_complex(z);
        let erfc_result = erfc_complex(z);
        let erfcx_result = erfcx_complex(z);

        println!("  z = {:.1} + {:.1}i:", z.re, z.im);
        println!("    erf(z) = {:.6} + {:.6}i", erf_result.re, erf_result.im);
        println!(
            "    erfc(z) = {:.2e} + {:.2e}i",
            erfc_result.re, erfc_result.im
        );
        println!(
            "    erfcx(z) = {:.6} + {:.6}i",
            erfcx_result.re, erfcx_result.im
        );
    }

    println!();

    // Application: Plasma dispersion function
    println!("Application: Plasma Dispersion Function Z(ζ)");
    println!("Z(ζ) = i√π * w(ζ) where w is the Faddeeva function");

    let plasma_values = [
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.1),
        Complex64::new(0.5, 0.5),
    ];

    for &zeta in &plasma_values {
        let w_zeta = faddeeva_complex(zeta);
        let z_zeta = Complex64::new(0.0, std::f64::consts::PI.sqrt()) * w_zeta;

        println!(
            "  ζ = {:.1} + {:.1}i: Z(ζ) = {:.6} + {:.6}i",
            zeta.re, zeta.im, z_zeta.re, z_zeta.im
        );
    }

    println!();

    // Dawson integral F(x) = e^(-x²) ∫₀ˣ e^(t²) dt
    println!("Related function: Dawson integral F(x)");
    println!("F(x) = (√π/2) * Im[w(x)] for real x");

    let dawson_values = [0.5, 1.0, 2.0, 5.0];

    for &x in &dawson_values {
        let z = Complex64::new(x, 0.0);
        let w_x = faddeeva_complex(z);
        let dawson_f = std::f64::consts::PI.sqrt() / 2.0 * w_x.im;

        println!("  F({:.1}) = {:.6}", x, dawson_f);
    }

    println!();
    Ok(())
}
