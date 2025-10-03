//! Interactive Tutorial for Special Functions

#![allow(clippy::all)]
//!
//! This example provides an interactive command-line tutorial that guides users
//! through the various special functions implemented in scirs2-special.
//!
//! Run with: cargo run --example interactive_tutorial

#![allow(unexpected_cfgs)]

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use scirs2_special::*;
use statrs::statistics::Statistics;
use std::io::{self, Write};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎓 Welcome to the Interactive Special Functions Tutorial!");
    println!("=====================================\n");

    loop {
        display_main_menu();
        let choice = get_user_input("Enter your choice (1-10, or 'q' to quit): ")?;

        if choice.to_lowercase() == "q" {
            println!("👋 Thanks for using the special functions tutorial!");
            break;
        }

        match choice.parse::<u32>() {
            Ok(1) => gamma_function_tutorial()?,
            Ok(2) => bessel_function_tutorial()?,
            Ok(3) => error_function_tutorial()?,
            Ok(4) => orthogonal_polynomial_tutorial()?,
            Ok(5) => hypergeometric_function_tutorial()?,
            Ok(6) => wright_function_tutorial()?,
            Ok(7) => elliptic_integral_tutorial()?,
            Ok(8) => spherical_harmonics_tutorial()?,
            Ok(9) => array_operations_tutorial()?,
            Ok(10) => advanced_features_tutorial()?,
            _ => println!("❌ Invalid choice. Please try again.\n"),
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn display_main_menu() {
    println!("📚 Choose a topic to explore:");
    println!("1. 🎲 Gamma and Beta Functions");
    println!("2. 🌊 Bessel Functions");
    println!("3. 📊 Error Functions");
    println!("4. 📈 Orthogonal Polynomials");
    println!("5. 🔢 Hypergeometric Functions");
    println!("6. 🌀 Wright Functions");
    println!("7. 🥧 Elliptic Integrals");
    println!("8. 🌐 Spherical Harmonics");
    println!("9. 📋 Array Operations");
    println!("10. ⚡ Advanced Features");
    println!("q. Quit");
    println!();
}

#[allow(dead_code)]
fn get_user_input(prompt: &str) -> io::Result<String> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

#[allow(dead_code)]
fn gamma_function_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎲 GAMMA FUNCTION TUTORIAL");
    println!("==========================\n");

    println!("The gamma function Γ(z) is a generalization of the factorial function.");
    println!("For positive integers n: Γ(n) = (n-1)!");
    println!("Key property: Γ(z+1) = z·Γ(z)\n");

    // Interactive examples
    println!("Let's explore some values:");

    let test_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5];
    for &x in &test_values {
        let x: f64 = x;
        let result = gamma(x);
        println!("Γ({}) = {:.6}", x, result);

        if x == 0.5 {
            println!("  Note: Γ(1/2) = √π ≈ {:.6}", std::f64::consts::PI.sqrt());
        } else if x.fract() == 0.0 && x > 0.0 {
            let factorial = (1..x as u64).product::<u64>() as f64;
            println!("  This equals ({})! = {}", x as u64 - 1, factorial);
        }
    }

    println!("\n📈 Let's plot some gamma function values:");
    plot_function_ascii("Gamma Function", 0.1, 5.0, 50, |x| gamma(x));

    // Interactive input
    loop {
        let input = get_user_input("\nEnter a value to compute Γ(x) (or 'back' to return): ")?;
        if input.to_lowercase() == "back" {
            break;
        }

        match input.parse::<f64>() {
            Ok(x) => {
                if x > 0.0 {
                    let result = gamma(x);
                    println!("Γ({}) = {:.10}", x, result);

                    // Show related functions
                    let ln_gamma = gammaln(x);
                    let digamma_val = digamma(x);
                    println!("ln Γ({}) = {:.10}", x, ln_gamma);
                    println!("ψ({}) = {:.10}", x, digamma_val);
                } else {
                    println!("❌ Gamma function is not defined for non-positive real numbers.");
                }
            }
            Err(_) => println!("❌ Please enter a valid number."),
        }
    }

    // Beta function section
    println!("\n🔗 BETA FUNCTION");
    println!("The beta function B(a,b) is related to gamma: B(a,b) = Γ(a)Γ(b)/Γ(a+b)");

    let beta_examples = vec![(1.0, 1.0), (2.0, 3.0), (0.5, 0.5), (3.0, 2.0)];
    for &(a, b) in &beta_examples {
        let beta_val = beta(a, b);
        println!("B({}, {}) = {:.6}", a, b, beta_val);
    }

    println!("\n✅ Gamma function tutorial completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn bessel_function_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌊 BESSEL FUNCTION TUTORIAL");
    println!("===========================\n");

    println!("Bessel functions are solutions to Bessel's differential equation:");
    println!("x²y'' + xy' + (x² - ν²)y = 0");
    println!("They appear in many physics problems with cylindrical symmetry.\n");

    println!("📊 First Kind Bessel Functions J_ν(x):");
    let x_vals = Array1::linspace(0.0, 15.0, 50);

    println!("Let's see J₀(x), J₁(x), and J₂(x) for various x values:");
    for i in 0..10 {
        let x = i as f64;
        let j0_val = j0(x);
        let j1_val = j1(x);
        let j2_val = jn(2, x);
        println!(
            "x={:2.0}: J₀={:8.4}, J₁={:8.4}, J₂={:8.4}",
            x, j0_val, j1_val, j2_val
        );
    }

    println!("\n📈 ASCII plot of J₀(x):");
    plot_function_ascii("J₀(x)", 0.0, 15.0, 60, |x| j0(x));

    // Zeros of Bessel functions
    println!("\n🎯 ZEROS OF BESSEL FUNCTIONS");
    println!("Bessel functions have infinitely many zeros. Let's find the first few:");

    println!("First 5 zeros of J₀(x):");
    for i in 1..=5 {
        match j0_zeros::<f64>(i) {
            Ok(zero) => {
                println!("  Zero #{}: x = {:.6}", i, zero);
                println!("    Verification: J₀({:.6}) = {:.2e}", zero, j0(zero));
            }
            Err(e) => println!("  Error computing zero #{}: {}", i, e),
        }
    }

    // Interactive exploration
    loop {
        let input = get_user_input(
            "\nEnter 'order,value' to compute J_order(value) (e.g., '0,5.2') or 'back': ",
        )?;
        if input.to_lowercase() == "back" {
            break;
        }

        let parts: Vec<&str> = input.split(',').collect();
        if parts.len() == 2 {
            match (
                parts[0].trim().parse::<i32>(),
                parts[1].trim().parse::<f64>(),
            ) {
                (Ok(order), Ok(value)) => {
                    if order >= 0 {
                        let result = jn(order, value);
                        println!("J_{}({}) = {:.10}", order, value, result);

                        // Show derivative if order is small
                        if order <= 3 {
                            let derivative = match order {
                                0 => -j1(value),
                                1 => (j0(value) - jn(2, value)) / 2.0,
                                _ => (jn(order - 1, value) - jn(order + 1, value)) / 2.0,
                            };
                            println!("J'_{}({}) = {:.10}", order, value, derivative);
                        }
                    } else {
                        println!("❌ Order must be non-negative.");
                    }
                }
                _ => println!("❌ Invalid format. Use 'order,value' (e.g., '0,5.2')"),
            }
        } else {
            println!("❌ Invalid format. Use 'order,value' (e.g., '0,5.2')");
        }
    }

    // Modified Bessel functions
    println!("\n🔄 MODIFIED BESSEL FUNCTIONS");
    println!("Modified Bessel functions I_ν(x) and K_ν(x) are related to exponential behavior:");

    for &x in &[0.5, 1.0, 2.0, 5.0] {
        let i0_val = i0(x);
        let i1_val = i1(x);
        let k0_val = k0(x);
        let k1_val = k1(x);
        println!(
            "x={}: I₀={:.4}, I₁={:.4}, K₀={:.4}, K₁={:.4}",
            x, i0_val, i1_val, k0_val, k1_val
        );
    }

    println!("\n✅ Bessel function tutorial completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn error_function_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 ERROR FUNCTION TUTORIAL");
    println!("==========================\n");

    println!("The error function erf(x) is the integral of the Gaussian distribution:");
    println!("erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt");
    println!("It's fundamental in probability theory and statistics.\n");

    println!("Key properties:");
    println!("• erf(0) = 0");
    println!("• erf(∞) = 1");
    println!("• erf(-x) = -erf(x) (odd function)");
    println!("• erfc(x) = 1 - erf(x) (complementary error function)\n");

    // Demonstrate basic values
    println!("📈 Basic values:");
    let test_points = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    for &x in &test_points {
        let erf_val = erf(x);
        let erfc_val = erfc(x);
        println!(
            "erf({:4.1}) = {:8.5}, erfc({:4.1}) = {:8.5}",
            x, erf_val, x, erfc_val
        );

        // Show probability interpretation for positive values
        if x > 0.0 {
            let prob = erf_val / 2.0 + 0.5;
            println!(
                "  → P(Z ≤ {:.1}) ≈ {:.3} for standard normal Z",
                x * std::f64::consts::SQRT_2,
                prob
            );
        }
    }

    println!("\n📈 ASCII plot of erf(x):");
    plot_function_ascii("erf(x)", -3.0, 3.0, 60, |x| erf(x));

    // Inverse error functions
    println!("\n🔄 INVERSE ERROR FUNCTIONS");
    println!("The inverse error function erfinv(y) satisfies erf(erfinv(y)) = y");

    let prob_values = vec![0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
    for &p in &prob_values {
        let p: f64 = p;
        if p.abs() < 1.0 {
            let x = erfinv(p);
            let verification = erf(x);
            println!(
                "erfinv({:.2}) = {:.4}, verification: erf({:.4}) = {:.4}",
                p, x, x, verification
            );
        }
    }

    // Interactive calculation
    loop {
        println!("\n🧮 Interactive Calculator:");
        println!("1. Compute erf(x)");
        println!("2. Compute erfc(x)");
        println!("3. Compute erfinv(y)");
        println!("4. Normal distribution probability");
        println!("5. Back to main menu");

        let choice = get_user_input("Enter choice (1-5): ")?;
        match choice.as_str() {
            "1" => {
                let input = get_user_input("Enter x value: ")?;
                if let Ok(x) = input.parse::<f64>() {
                    println!("erf({}) = {:.10}", x, erf(x));
                } else {
                    println!("❌ Invalid number");
                }
            }
            "2" => {
                let input = get_user_input("Enter x value: ")?;
                if let Ok(x) = input.parse::<f64>() {
                    println!("erfc({}) = {:.10}", x, erfc(x));
                } else {
                    println!("❌ Invalid number");
                }
            }
            "3" => {
                let input = get_user_input("Enter y value (-1 < y < 1): ")?;
                if let Ok(y) = input.parse::<f64>() {
                    if y.abs() < 1.0 {
                        println!("erfinv({}) = {:.10}", y, erfinv(y));
                    } else {
                        println!("❌ Value must be between -1 and 1");
                    }
                } else {
                    println!("❌ Invalid number");
                }
            }
            "4" => {
                let input = get_user_input("Enter z-score: ")?;
                if let Ok(z) = input.parse::<f64>() {
                    let prob = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
                    println!("P(Z ≤ {}) = {:.6} for standard normal Z", z, prob);
                } else {
                    println!("❌ Invalid number");
                }
            }
            "5" => break,
            _ => println!("❌ Invalid choice"),
        }
    }

    println!("\n✅ Error function tutorial completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn orthogonal_polynomial_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 ORTHOGONAL POLYNOMIALS TUTORIAL");
    println!("==================================\n");

    println!("Orthogonal polynomials are families of polynomials that are orthogonal");
    println!("with respect to a specific weight function on a given interval.\n");

    // Legendre polynomials
    println!("🔹 LEGENDRE POLYNOMIALS P_n(x)");
    println!("Orthogonal on [-1,1] with weight w(x) = 1");
    println!("Used in spherical coordinates and Gaussian quadrature.\n");

    println!("First few Legendre polynomials:");
    for n in 0..=5 {
        print!("P_{}(x) = ", n);
        match n {
            0 => println!("1"),
            1 => println!("x"),
            2 => println!("(3x² - 1)/2"),
            3 => println!("(5x³ - 3x)/2"),
            4 => println!("(35x⁴ - 30x² + 3)/8"),
            5 => println!("(63x⁵ - 70x³ + 15x)/8"),
            _ => unreachable!(),
        }
    }

    println!("\nValues at specific points:");
    let test_points = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    for &x in &test_points {
        print!("x = {:4.1}: ", x);
        for n in 0..=4 {
            let val = legendre(n, x);
            print!("P_{}={:6.3} ", n, val);
        }
        println!();
    }

    // Hermite polynomials
    println!("\n🔹 HERMITE POLYNOMIALS H_n(x)");
    println!("Orthogonal on (-∞,∞) with weight w(x) = e^(-x²)");
    println!("Used in quantum mechanics (harmonic oscillator).\n");

    for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
        print!("x = {:4.1}: ", x);
        for n in 0..=3 {
            let val = hermite(n, x);
            print!("H_{}={:7.2} ", n, val);
        }
        println!();
    }

    // Laguerre polynomials
    println!("\n🔹 LAGUERRE POLYNOMIALS L_n(x)");
    println!("Orthogonal on [0,∞) with weight w(x) = e^(-x)");
    println!("Used in radial parts of hydrogen atom wavefunctions.\n");

    for &x in &[0.0, 0.5, 1.0, 2.0, 3.0] {
        print!("x = {:4.1}: ", x);
        for n in 0..=3 {
            let val = laguerre(n, x);
            print!("L_{}={:7.3} ", n, val);
        }
        println!();
    }

    // Interactive polynomial evaluator
    loop {
        println!("\n🧮 Polynomial Evaluator:");
        println!("1. Legendre polynomial P_n(x)");
        println!("2. Hermite polynomial H_n(x)");
        println!("3. Laguerre polynomial L_n(x)");
        println!("4. Chebyshev polynomial T_n(x)");
        println!("5. Back to main menu");

        let choice = get_user_input("Enter choice (1-5): ")?;
        match choice.as_str() {
            "1" => evaluate_polynomial("Legendre", |n, x| legendre(n, x))?,
            "2" => evaluate_polynomial("Hermite", |n, x| hermite(n, x))?,
            "3" => evaluate_polynomial("Laguerre", |n, x| laguerre(n, x))?,
            "4" => evaluate_polynomial("Chebyshev", |n, x| chebyshev(n, x, true))?,
            "5" => break,
            _ => println!("❌ Invalid choice"),
        }
    }

    println!("\n✅ Orthogonal polynomials tutorial completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn evaluate_polynomial<F>(name: &str, func: F) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(usize, f64) -> f64,
{
    let n_input = get_user_input(&format!("Enter degree n for {} polynomial: ", name))?;
    let x_input = get_user_input("Enter x value: ")?;

    match (n_input.parse::<usize>(), x_input.parse::<f64>()) {
        (Ok(n), Ok(x)) => {
            if n <= 20 {
                let result = func(n, x);
                println!("{}_{}({}) = {:.10}", name, n, x, result);
            } else {
                println!("❌ Degree too large (max 20)");
            }
        }
        _ => println!("❌ Invalid input"),
    }
    Ok(())
}

#[allow(dead_code)]
fn hypergeometric_function_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔢 HYPERGEOMETRIC FUNCTIONS TUTORIAL");
    println!("====================================\n");

    println!("Hypergeometric functions are generalizations of elementary functions.");
    println!("They are defined by series with specific coefficient ratios.\n");

    println!("🔹 Confluent Hypergeometric Function ₁F₁(a; c; z)");
    println!("Also known as Kummer's function M(a, c, z)");
    println!("Series: ₁F₁(a; c; z) = Σ (a)_n / (c)_n * z^n / n!");
    println!("where (a)_n is the Pochhammer symbol.\n");

    // Examples of 1F1
    let examples_1f1 = vec![
        (1.0, 2.0, 1.0),
        (0.5, 1.5, 2.0),
        (2.0, 3.0, -1.0),
        (-1.0, 2.0, 1.0),
    ];

    println!("Examples of ₁F₁(a; c; z):");
    for &(a, c, z) in &examples_1f1 {
        let (a, c, z): (f64, f64, f64) = (a, c, z);
        let result = hyp1f1(a, c, z);
        match result {
            Ok(value) => println!("₁F₁({}, {}, {}) = {:.6}", a, c, z, value),
            Err(e) => println!("₁F₁({}, {}, {}) = Error: {}", a, c, z, e),
        }

        // Special cases
        if (a - 1.0).abs() < 1e-10 && (c - 2.0).abs() < 1e-10 {
            let expected = (z.exp() - 1.0) / z;
            println!("  → This equals (e^z - 1)/z = {:.6}", expected);
        }
    }

    println!("\n🔹 Gauss Hypergeometric Function ₂F₁(a, b; c; z)");
    println!("Series: ₂F₁(a, b; c; z) = Σ (a)_n (b)_n / (c)_n * z^n / n!\n");

    // Examples of 2F1
    let examples_2f1 = vec![
        (1.0, 1.0, 2.0, 0.5),
        (0.5, 0.5, 1.0, 0.25),
        (1.0, 2.0, 3.0, -0.5),
        (-0.5, 0.5, 1.0, 0.8),
    ];

    println!("Examples of ₂F₁(a, b; c; z):");
    for &(a, b, c, z) in &examples_2f1 {
        let (a, b, c, z): (f64, f64, f64, f64) = (a, b, c, z);
        let result = hyp2f1(a, b, c, z);
        match result {
            Ok(value) => println!("₂F₁({}, {}, {}, {}) = {:.6}", a, b, c, z, value),
            Err(e) => println!("₂F₁({}, {}, {}, {}) = Error: {}", a, b, c, z, e),
        }

        // Special case: 2F1(1,1;2;z) = -ln(1-z)/z
        if (a - 1.0).abs() < 1e-10 && (b - 1.0).abs() < 1e-10 && (c - 2.0).abs() < 1e-10 {
            let expected = -(1.0 - z).ln() / z;
            println!("  → This equals -ln(1-z)/z = {:.6}", expected);
        }
    }

    println!("\n🔹 Pochhammer Symbol (a)_n");
    println!("The rising factorial: (a)_n = a(a+1)(a+2)...(a+n-1)");

    for &a in &[1.0, 2.0, 0.5, -0.5] {
        print!("a = {}: ", a);
        for n in 0..=4 {
            let poch = pochhammer(a, n);
            print!("({})_{} = {:6.3}, ", a, n, poch);
        }
        println!();
    }

    // Interactive calculator
    loop {
        println!("\n🧮 Hypergeometric Calculator:");
        println!("1. Compute ₁F₁(a; c; z)");
        println!("2. Compute ₂F₁(a, b; c; z)");
        println!("3. Compute Pochhammer symbol (a)_n");
        println!("4. Back to main menu");

        let choice = get_user_input("Enter choice (1-4): ")?;
        match choice.as_str() {
            "1" => {
                let a = get_user_input("Enter a: ")?.parse::<f64>().unwrap_or(0.0);
                let c = get_user_input("Enter c: ")?.parse::<f64>().unwrap_or(1.0);
                let z = get_user_input("Enter z: ")?.parse::<f64>().unwrap_or(0.0);
                let result = hyp1f1(a, c, z);
                match result {
                    Ok(value) => println!("₁F₁({}, {}, {}) = {:.10}", a, c, z, value),
                    Err(e) => println!("₁F₁({}, {}, {}) = Error: {}", a, c, z, e),
                }
            }
            "2" => {
                let a = get_user_input("Enter a: ")?.parse::<f64>().unwrap_or(0.0);
                let b = get_user_input("Enter b: ")?.parse::<f64>().unwrap_or(0.0);
                let c = get_user_input("Enter c: ")?.parse::<f64>().unwrap_or(1.0);
                let z = get_user_input("Enter z: ")?.parse::<f64>().unwrap_or(0.0);
                let result = hyp2f1(a, b, c, z);
                match result {
                    Ok(value) => println!("₂F₁({}, {}, {}, {}) = {:.10}", a, b, c, z, value),
                    Err(e) => println!("₂F₁({}, {}, {}, {}) = Error: {}", a, b, c, z, e),
                }
            }
            "3" => {
                let a = get_user_input("Enter a: ")?.parse::<f64>().unwrap_or(0.0);
                let n = get_user_input("Enter n: ")?.parse::<u32>().unwrap_or(0);
                let result = pochhammer(a, n as usize);
                println!("({})_{} = {:.10}", a, n, result);
            }
            "4" => break,
            _ => println!("❌ Invalid choice"),
        }
    }

    println!("\n✅ Hypergeometric functions tutorial completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn wright_function_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌀 WRIGHT FUNCTIONS TUTORIAL");
    println!("============================\n");

    println!("Wright functions are generalizations of Bessel functions with applications");
    println!("in fractional calculus, anomalous diffusion, and probability theory.\n");

    println!("🔹 Wright Bessel Function J_{{ρ,β}}(z)");
    println!("Series: J_{{ρ,β}}(z) = Σ (-z)^k / (k! Γ(ρk + β))");
    println!("Parameters: ρ > 0, β ∈ ℂ, z ∈ ℂ\n");

    // Basic examples
    println!("Examples with different parameters:");
    let wright_examples = vec![
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 2.0),
        (2.0, 1.0, 1.0),
        (0.5, 1.0, 1.0),
        (1.0, 2.0, 1.0),
    ];

    for &(rho, beta, z) in &wright_examples {
        match wright_bessel(rho, beta, z) {
            Ok(result) => {
                println!("J_{{{},{}}}({}) = {:.6}", rho, beta, z, result);

                // Special cases
                if (rho - 1.0).abs() < 1e-10 && (beta - 1.0).abs() < 1e-10 {
                    // This case relates to ordinary Bessel functions
                    let bessel_equivalent = j0(2.0 * z.sqrt());
                    println!("  → Related to J₀(2√z) = {:.6}", bessel_equivalent);
                }
            }
            Err(e) => println!("Error computing J_{{{},{}}}({}): {}", rho, beta, z, e),
        }
    }

    println!("\n🔹 Wright Omega Function ω(z)");
    println!("Defined implicitly by: ω(z) e^ω(z) = z");
    println!("It's the inverse of z e^z (Lambert W function relation).\n");

    // Wright omega examples
    let omega_examples = vec![0.5, 1.0, 2.0, 5.0, 10.0];
    for &z in &omega_examples {
        match wright_omega(Complex64::new(z, 0.0), 1e-12) {
            Ok(omega) => {
                let verification = omega * omega.exp();
                println!(
                    "ω({}) = {:.6}, verification: ω·e^ω = {:.6}",
                    z, omega, verification
                );
            }
            Err(e) => println!("Error computing ω({}): {}", z, e),
        }
    }

    // Demonstrate Wright Bessel zeros
    println!("\n🎯 Zeros of Wright Bessel Functions");
    println!("Finding zeros can be challenging numerically...");

    match wright_bessel_zeros(1.0, 1.0, 3) {
        Ok(zeros) => {
            println!("First few zeros of J_{{1,1}}(z):");
            for (i, &zero) in zeros.iter().enumerate() {
                match wright_bessel(1.0, 1.0, zero) {
                    Ok(val) => println!(
                        "  Zero #{}: z = {:.6}, J_{{1,1}}({:.6}) = {:.2e}",
                        i + 1,
                        zero,
                        zero,
                        val
                    ),
                    Err(_) => println!("  Zero #{}: z = {:.6} (verification failed)", i + 1, zero),
                }
            }
        }
        Err(e) => println!("Could not compute zeros: {}", e),
    }

    // Interactive section
    loop {
        println!("\n🧮 Wright Functions Calculator:");
        println!("1. Compute Wright Bessel J_{{ρ,β}}(z)");
        println!("2. Compute Wright Omega ω(z)");
        println!("3. Explore parameter effects");
        println!("4. Back to main menu");

        let choice = get_user_input("Enter choice (1-4): ")?;
        match choice.as_str() {
            "1" => {
                let rho = get_user_input("Enter ρ (must be > 0): ")?
                    .parse::<f64>()
                    .unwrap_or(1.0);
                let beta = get_user_input("Enter β: ")?.parse::<f64>().unwrap_or(1.0);
                let z = get_user_input("Enter z: ")?.parse::<f64>().unwrap_or(1.0);

                if rho > 0.0 {
                    match wright_bessel(rho, beta, z) {
                        Ok(result) => println!("J_{{{},{}}}({}) = {:.10}", rho, beta, z, result),
                        Err(e) => println!("Error: {}", e),
                    }
                } else {
                    println!("❌ ρ must be positive");
                }
            }
            "2" => {
                let z = get_user_input("Enter z: ")?.parse::<f64>().unwrap_or(1.0);
                match wright_omega(Complex64::new(z, 0.0), 1e-12) {
                    Ok(omega) => {
                        println!("ω({}) = {:.10}", z, omega);
                        let verification = omega * omega.exp();
                        println!(
                            "Verification: ω·e^ω = {:.10} (should equal {})",
                            verification, z
                        );
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }
            "3" => {
                println!("\n📊 Parameter Effects on Wright Bessel Function:");
                println!("Fixing z = 1.0, varying ρ and β:");

                for rho in [0.5, 1.0, 1.5, 2.0] {
                    for beta in [0.5, 1.0, 1.5, 2.0] {
                        match wright_bessel(rho, beta, 1.0) {
                            Ok(val) => print!("J{{{},{}}}: {:7.4}  ", rho, beta, val),
                            Err(_) => print!("J{{{},{}}}: ------  ", rho, beta),
                        }
                    }
                    println!();
                }
            }
            "4" => break,
            _ => println!("❌ Invalid choice"),
        }
    }

    println!("\n✅ Wright functions tutorial completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn elliptic_integral_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🥧 ELLIPTIC INTEGRALS TUTORIAL");
    println!("==============================\n");

    println!("Elliptic integrals arise from computing arc lengths of ellipses and");
    println!(
        "appear in many areas of physics, including pendulum motion and electromagnetic theory.\n"
    );

    println!("🔹 Complete Elliptic Integrals");
    println!("K(k) = ∫₀^(π/2) dθ/√(1 - k²sin²θ)  (First kind)");
    println!("E(k) = ∫₀^(π/2) √(1 - k²sin²θ) dθ   (Second kind)\n");

    // Demonstrate complete elliptic integrals
    println!("Values for different modulus k:");
    let k_values = vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99];

    for &k in &k_values {
        let k_val = elliptic_k(k);
        let e_val = elliptic_e(k);
        println!("k = {:.2}: K(k) = {:.4}, E(k) = {:.4}", k, k_val, e_val);

        if k == 0.0 {
            println!("  → K(0) = E(0) = π/2 = {:.4}", std::f64::consts::PI / 2.0);
        }
    }

    // Incomplete elliptic integrals
    println!("\n🔹 Incomplete Elliptic Integrals");
    println!("F(φ,k) = ∫₀^φ dθ/√(1 - k²sin²θ)");
    println!("E(φ,k) = ∫₀^φ √(1 - k²sin²θ) dθ");

    let phi_values = vec![
        std::f64::consts::PI / 6.0,
        std::f64::consts::PI / 4.0,
        std::f64::consts::PI / 3.0,
        std::f64::consts::PI / 2.0,
    ];
    let k = 0.5;

    println!("\nFor k = 0.5:");
    for &phi in &phi_values {
        let f_val = elliptic_f(phi, k);
        let e_val = elliptic_e_inc(phi, k);
        println!(
            "φ = {:.3}: F(φ,k) = {:.4}, E(φ,k) = {:.4}",
            phi, f_val, e_val
        );
    }

    // Jacobi elliptic functions
    println!("\n🔹 Jacobi Elliptic Functions");
    println!("These are the inverses of elliptic integrals.");
    println!("sn(u,k), cn(u,k), dn(u,k) satisfy: sn² + cn² = 1, k²sn² + dn² = 1\n");

    let u_values = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    let k = 0.7;

    println!("For k = 0.7:");
    for &u in &u_values {
        let sn_val = jacobi_sn(u, k);
        let cn_val = jacobi_cn(u, k);
        let dn_val = jacobi_dn(u, k);

        println!(
            "u = {}: sn = {:.4}, cn = {:.4}, dn = {:.4}",
            u, sn_val, cn_val, dn_val
        );

        // Verify identities
        let identity1 = sn_val * sn_val + cn_val * cn_val;
        let identity2 = k * k * sn_val * sn_val + dn_val * dn_val;
        println!(
            "  Identities: sn² + cn² = {:.4}, k²sn² + dn² = {:.4}",
            identity1, identity2
        );
    }

    // Physical application: pendulum period
    println!("\n🎯 APPLICATION: Nonlinear Pendulum");
    println!("The period of a pendulum with large amplitude θ₀ is:");
    println!("T = 4√(L/g) K(sin(θ₀/2))");
    println!("where L is length, g is gravity, and K is the complete elliptic integral.\n");

    let g: f64 = 9.81; // gravity
    let l: f64 = 1.0; // length in meters
    let small_angle_period = 2.0 * std::f64::consts::PI * (l / g).sqrt();

    println!("For a 1-meter pendulum:");
    println!("Small angle period: T₀ = {:.3} seconds", small_angle_period);

    let angles_deg = vec![10.0, 30.0, 60.0, 90.0, 120.0, 150.0];
    for &angle_deg in &angles_deg {
        let angle_rad = angle_deg * std::f64::consts::PI / 180.0;
        let k = (angle_rad / 2.0).sin();
        let k_val = elliptic_k(k);
        let period = 4.0 * (l / g).sqrt() * k_val;
        let ratio = period / small_angle_period;

        println!(
            "θ₀ = {:3.0}°: T = {:.3} s (ratio = {:.3})",
            angle_deg, period, ratio
        );
    }

    // Interactive calculator
    loop {
        println!("\n🧮 Elliptic Integrals Calculator:");
        println!("1. Complete elliptic integrals K(k), E(k)");
        println!("2. Incomplete elliptic integrals F(φ,k), E(φ,k)");
        println!("3. Jacobi elliptic functions sn, cn, dn");
        println!("4. Pendulum period calculator");
        println!("5. Back to main menu");

        let choice = get_user_input("Enter choice (1-5): ")?;
        match choice.as_str() {
            "1" => {
                let k = get_user_input("Enter modulus k (0 ≤ k < 1): ")?
                    .parse::<f64>()
                    .unwrap_or(0.5);
                if k >= 0.0 && k < 1.0 {
                    println!("K({}) = {:.10}", k, elliptic_k(k));
                    println!("E({}) = {:.10}", k, elliptic_e(k));
                } else {
                    println!("❌ Modulus must be in [0, 1)");
                }
            }
            "2" => {
                let phi = get_user_input("Enter amplitude φ (radians): ")?
                    .parse::<f64>()
                    .unwrap_or(1.0);
                let k = get_user_input("Enter modulus k (0 ≤ k < 1): ")?
                    .parse::<f64>()
                    .unwrap_or(0.5);
                if k >= 0.0 && k < 1.0 {
                    println!("F({}, {}) = {:.10}", phi, k, elliptic_f(phi, k));
                    println!("E({}, {}) = {:.10}", phi, k, elliptic_e_inc(phi, k));
                } else {
                    println!("❌ Modulus must be in [0, 1)");
                }
            }
            "3" => {
                let u = get_user_input("Enter argument u: ")?
                    .parse::<f64>()
                    .unwrap_or(1.0);
                let k = get_user_input("Enter modulus k (0 ≤ k < 1): ")?
                    .parse::<f64>()
                    .unwrap_or(0.5);
                if k >= 0.0 && k < 1.0 {
                    println!("sn({}, {}) = {:.10}", u, k, jacobi_sn(u, k));
                    println!("cn({}, {}) = {:.10}", u, k, jacobi_cn(u, k));
                    println!("dn({}, {}) = {:.10}", u, k, jacobi_dn(u, k));
                } else {
                    println!("❌ Modulus must be in [0, 1)");
                }
            }
            "4" => {
                let angle_deg = get_user_input("Enter initial angle (degrees): ")?
                    .parse::<f64>()
                    .unwrap_or(30.0);
                let length = get_user_input("Enter pendulum length (meters): ")?
                    .parse::<f64>()
                    .unwrap_or(1.0);

                let angle_rad = angle_deg * std::f64::consts::PI / 180.0;
                let k = (angle_rad / 2.0).sin();
                let k_val = elliptic_k(k);
                let period = 4.0 * (length / 9.81).sqrt() * k_val;
                let small_angle_period = 2.0 * std::f64::consts::PI * (length / 9.81).sqrt();

                println!("Pendulum period: {:.4} seconds", period);
                println!(
                    "Small angle approximation: {:.4} seconds",
                    small_angle_period
                );
                println!("Ratio: {:.4}", period / small_angle_period);
            }
            "5" => break,
            _ => println!("❌ Invalid choice"),
        }
    }

    println!("\n✅ Elliptic integrals tutorial completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn spherical_harmonics_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌐 SPHERICAL HARMONICS TUTORIAL");
    println!("===============================\n");

    println!("Spherical harmonics Y_ℓ^m(θ,φ) are functions on the sphere that form");
    println!("a complete orthonormal basis for functions on S². They appear in:");
    println!("• Quantum mechanics (atomic orbitals)");
    println!("• Gravitational and electromagnetic multipole expansions");
    println!("• Computer graphics and signal processing\n");

    println!("🔹 Definition:");
    println!("Y_ℓ^m(θ,φ) = √[(2ℓ+1)/4π × (ℓ-m)!/(ℓ+m)!] P_ℓ^m(cos θ) e^(imφ)");
    println!("where P_ℓ^m are associated Legendre polynomials.\n");

    // Basic examples
    println!("📊 Values for low ℓ and m:");
    let theta_vals = vec![
        0.0,
        std::f64::consts::PI / 4.0,
        std::f64::consts::PI / 2.0,
        3.0 * std::f64::consts::PI / 4.0,
        std::f64::consts::PI,
    ];
    let phi = 0.0; // Start with φ = 0

    println!("At φ = 0 (real parts):");
    for &theta in &theta_vals {
        print!("θ = {:4.2}: ", theta);

        // Y_0^0, Y_1^0, Y_1^1, Y_2^0
        let y_00 = sph_harm(0, 0, theta, phi);
        let y_10 = sph_harm(1, 0, theta, phi);
        let y_11 = sph_harm(1, 1, theta, phi);
        let y_20 = sph_harm(2, 0, theta, phi);

        match (y_00, y_10, y_11, y_20) {
            (Ok(y_00), Ok(y_10), Ok(y_11), Ok(y_20)) => {
                println!(
                    "Y₀⁰={:.3}, Y₁⁰={:.3}, Y₁¹={:.3}, Y₂⁰={:.3}",
                    y_00, y_10, y_11, y_20
                );
            }
            _ => println!("Error computing spherical harmonics"),
        }
    }

    // Symmetry properties
    println!("\n🔄 Symmetry Properties:");
    let l = 2;
    let m = 1;
    let theta = std::f64::consts::PI / 3.0;
    let phi = std::f64::consts::PI / 4.0;

    let y_lm = sph_harm_complex(l, m, theta, phi);
    let y_lminus_m = sph_harm_complex(l, -m, theta, phi);

    match (y_lm, y_lminus_m) {
        (Ok((y_lm_re, y_lm_im)), Ok((y_lminus_m_re, y_lminus_m_im))) => {
            println!("Y_{}^{}(θ,φ) = {:.4} + {:.4}i", l, m, y_lm_re, y_lm_im);
            println!(
                "Y_{}^{}(θ,φ) = {:.4} + {:.4}i",
                l, -m, y_lminus_m_re, y_lminus_m_im
            );
            println!("Relation: Y_ℓ^(-m) = (-1)^m [Y_ℓ^m]*");

            let y_lm_complex = Complex64::new(y_lm_re, y_lm_im);
            let expected = (-1.0_f64).powi(m) * y_lm_complex.conj();
            let y_lminus_m_complex = Complex64::new(y_lminus_m_re, y_lminus_m_im);
            println!("Expected: {:.4} + {:.4}i", expected.re, expected.im);
            println!("Match: {}", (y_lminus_m_complex - expected).norm() < 1e-10);
        }
        _ => println!("Error computing complex spherical harmonics"),
    }

    // Orthogonality demonstration
    println!("\n⊥ Orthogonality Check:");
    println!("Spherical harmonics are orthonormal on the sphere.");
    println!("∫∫ Y_ℓ^m* Y_ℓ'^m' sin θ dθ dφ = δ_ℓℓ' δ_mm'");

    // Simple numerical integration check (not comprehensive)
    let n_theta = 20;
    let n_phi = 40;
    let mut integral = Complex64::new(0.0, 0.0);

    for i in 0..n_theta {
        for j in 0..n_phi {
            let theta = std::f64::consts::PI * i as f64 / (n_theta - 1) as f64;
            let phi = 2.0 * std::f64::consts::PI * j as f64 / n_phi as f64;
            let weight = (std::f64::consts::PI / (n_theta - 1) as f64)
                * (2.0 * std::f64::consts::PI / n_phi as f64)
                * theta.sin();

            let y1 = sph_harm(1, 0, theta, phi);
            let y2 = sph_harm(1, 0, theta, phi); // Same function - should give 1

            if let (Ok(y1_val), Ok(y2_val)) = (y1, y2) {
                let y1_complex = Complex64::new(y1_val, 0.0);
                let y2_complex = Complex64::new(y2_val, 0.0);
                integral += y1_complex.conj() * y2_complex * weight;
            }
        }
    }

    println!(
        "Numerical check of ∫∫ |Y₁⁰|² dΩ ≈ {:.4} (should be 1.0)",
        integral.re
    );

    // Physical interpretation
    println!("\n⚛️ Physical Interpretation (Quantum Mechanics):");
    println!("In hydrogen atom, electron orbitals are |n,ℓ,m⟩ ∝ R_nℓ(r) Y_ℓ^m(θ,φ)");
    println!();
    println!("s orbitals (ℓ=0): spherically symmetric");
    println!("  Y₀⁰ ∝ 1 (constant)");
    println!();
    println!("p orbitals (ℓ=1): three lobes");
    let theta_0 = 0.0; // z-axis
    let theta_90 = std::f64::consts::PI / 2.0; // xy-plane
    match sph_harm(1, 0, theta_0, 0.0) {
        Ok(val) => println!("  Y₁⁰ at θ=0°: {:.3} (pz orbital along z)", val),
        Err(_) => println!("  Y₁⁰ at θ=0°: Error"),
    }
    match sph_harm(1, 0, theta_90, 0.0) {
        Ok(val) => println!("  Y₁⁰ at θ=90°: {:.3} (pz orbital in xy-plane)", val),
        Err(_) => println!("  Y₁⁰ at θ=90°: Error"),
    }
    println!();
    println!("d orbitals (ℓ=2): five orbitals with complex shapes");

    // Interactive calculator
    loop {
        println!("\n🧮 Spherical Harmonics Calculator:");
        println!("1. Compute Y_ℓ^m(θ,φ)");
        println!("2. Compare different (ℓ,m) at same point");
        println!("3. Study angular dependence");
        println!("4. Quantum orbital visualization info");
        println!("5. Back to main menu");

        let choice = get_user_input("Enter choice (1-5): ")?;
        match choice.as_str() {
            "1" => {
                let l = get_user_input("Enter ℓ (0,1,2,...): ")?
                    .parse::<i32>()
                    .unwrap_or(1);
                let m = get_user_input("Enter m (-ℓ ≤ m ≤ ℓ): ")?
                    .parse::<i32>()
                    .unwrap_or(0);
                let theta_deg = get_user_input("Enter θ in degrees (0-180): ")?
                    .parse::<f64>()
                    .unwrap_or(90.0);
                let phi_deg = get_user_input("Enter φ in degrees (0-360): ")?
                    .parse::<f64>()
                    .unwrap_or(0.0);

                if l >= 0 && m.abs() <= l {
                    let theta = theta_deg * std::f64::consts::PI / 180.0;
                    let phi = phi_deg * std::f64::consts::PI / 180.0;
                    let y = sph_harm(l as usize, m, theta, phi).unwrap_or(0.0);

                    println!(
                        "Y_{}^{}({:.1}°, {:.1}°) = {:.6} + {:.6}i",
                        l, m, theta_deg, phi_deg, y, 0.0
                    );
                    println!("Magnitude: |Y_{}^{}| = {:.6}", l, m, y.abs());

                    if y.abs() < 1e-10 {
                        println!("This is essentially real-valued at this point.");
                    }
                } else {
                    println!("❌ Invalid quantum numbers: need |m| ≤ ℓ");
                }
            }
            "2" => {
                let theta_deg = get_user_input("Enter θ in degrees: ")?
                    .parse::<f64>()
                    .unwrap_or(90.0);
                let phi_deg = get_user_input("Enter φ in degrees: ")?
                    .parse::<f64>()
                    .unwrap_or(0.0);

                let theta = theta_deg * std::f64::consts::PI / 180.0;
                let phi = phi_deg * std::f64::consts::PI / 180.0;

                println!(
                    "Spherical harmonics at (θ={:.1}°, φ={:.1}°):",
                    theta_deg, phi_deg
                );

                for l in 0..=3 {
                    for m in -l..=l {
                        let y = sph_harm(l as usize, m, theta, phi).unwrap_or(0.0);
                        print!("Y_{}^{:2}: {:7.4}+{:7.4}i  ", l, m, y, 0.0);
                        if m == l {
                            println!();
                        }
                    }
                }
            }
            "3" => {
                let l = get_user_input("Enter ℓ: ")?.parse::<i32>().unwrap_or(1);
                let m = get_user_input("Enter m: ")?.parse::<i32>().unwrap_or(0);

                if l >= 0 && m.abs() <= l {
                    println!("Angular dependence of Y_{}^{} along θ (φ=0):", l, m);
                    for i in 0..=18 {
                        let theta_deg = i as f64 * 10.0;
                        let theta = theta_deg * std::f64::consts::PI / 180.0;
                        let y = sph_harm(l as usize, m, theta, 0.0).unwrap_or(0.0);
                        println!("θ={:3.0}°: Y={:8.4} + {:8.4}i", theta_deg, y, 0.0);
                    }
                } else {
                    println!("❌ Invalid quantum numbers");
                }
            }
            "4" => {
                println!("\n⚛️ Quantum Orbital Information:");
                println!("ℓ=0 (s): 1 orbital, spherical");
                println!("ℓ=1 (p): 3 orbitals, dumbbell shapes");
                println!("  m=0: pz (along z-axis)");
                println!("  m=±1: px, py (in xy-plane)");
                println!("ℓ=2 (d): 5 orbitals, complex shapes");
                println!("  m=0: dz² (elongated along z)");
                println!("  m=±1: dxz, dyz (lobes between axes)");
                println!("  m=±2: dx²-y², dxy (four-lobed)");
                println!("ℓ=3 (f): 7 orbitals, very complex shapes");
                println!();
                println!("Higher |m| values generally have more nodes and");
                println!("more complex angular structure.");
            }
            "5" => break,
            _ => println!("❌ Invalid choice"),
        }
    }

    println!("\n✅ Spherical harmonics tutorial completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn array_operations_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📋 ARRAY OPERATIONS TUTORIAL");
    println!("============================\n");

    println!("Most special functions in scirs2-special can operate on arrays efficiently.");
    println!("This enables vectorized computations for better performance.\n");

    // Create sample arrays
    let small_array = Array1::linspace(0.1, 2.0, 10);
    let medium_array = Array1::linspace(0.1, 5.0, 100);
    let large_array = Array1::linspace(0.1, 10.0, 10000);

    println!("🔢 Created test arrays:");
    println!(
        "Small:  {} elements from {:.1} to {:.1}",
        small_array.len(),
        small_array[0],
        small_array[small_array.len() - 1]
    );
    println!(
        "Medium: {} elements from {:.1} to {:.1}",
        medium_array.len(),
        medium_array[0],
        medium_array[medium_array.len() - 1]
    );
    println!(
        "Large:  {} elements from {:.1} to {:.1}",
        large_array.len(),
        large_array[0],
        large_array[large_array.len() - 1]
    );

    // Basic array operations
    println!("\n📊 Basic Array Operations:");

    // Gamma function on array
    let gamma_results = small_array.mapv(|x| gamma(x));
    println!("Gamma function on small array:");
    for i in 0..small_array.len().min(5) {
        println!("  Γ({:.3}) = {:.6}", small_array[i], gamma_results[i]);
    }
    if small_array.len() > 5 {
        println!("  ... ({} more values)", small_array.len() - 5);
    }

    // Error function on array
    let erf_results = small_array.mapv(|x| erf(x));
    println!("\nError function on small array:");
    for i in 0..small_array.len().min(5) {
        println!("  erf({:.3}) = {:.6}", small_array[i], erf_results[i]);
    }

    // Bessel functions
    let j0_results = small_array.mapv(|x| j0(x));
    println!("\nBessel J₀ on small array:");
    for i in 0..small_array.len().min(5) {
        println!("  J₀({:.3}) = {:.6}", small_array[i], j0_results[i]);
    }

    // Performance demonstration
    println!("\n⚡ Performance Comparison:");
    println!("Computing gamma function on arrays of different sizes...");

    use std::time::Instant;

    // Small array timing
    let start = Instant::now();
    let _gamma_small = small_array.mapv(|x| gamma(x));
    let small_time = start.elapsed();

    // Medium array timing
    let start = Instant::now();
    let _gamma_medium = medium_array.mapv(|x| gamma(x));
    let medium_time = start.elapsed();

    // Large array timing
    let start = Instant::now();
    let _gamma_large = large_array.mapv(|x| gamma(x));
    let large_time = start.elapsed();

    println!("Small array ({}):   {:?}", small_array.len(), small_time);
    println!("Medium array ({}):  {:?}", medium_array.len(), medium_time);
    println!("Large array ({}): {:?}", large_array.len(), large_time);

    let throughput_large = large_array.len() as f64 / large_time.as_secs_f64();
    println!("Throughput: {:.0} evaluations/second", throughput_large);

    // Memory-efficient operations
    println!("\n💾 Memory-Efficient Operations:");
    println!("For very large arrays, chunked processing prevents memory issues:");

    #[cfg(feature = "memory-efficient")]
    {
        use scirs2_special::memory_efficient::gamma_chunked;

        let very_large = Array1::linspace(0.1, 5.0, 1_000_000);
        println!(
            "Created array with {} million elements",
            very_large.len() / 1_000_000
        );

        let start = Instant::now();
        match gamma_chunked(&very_large, Some(10000)) {
            Ok(_chunked_result) => {
                let chunked_time = start.elapsed();
                println!("Chunked processing completed in {:?}", chunked_time);
            }
            Err(e) => println!("Chunked processing failed: {}", e),
        }
    }

    #[cfg(not(feature = "memory-efficient"))]
    {
        println!("Memory-efficient operations require the 'memory-efficient' feature.");
        println!("Enable with: cargo run --features memory-efficient");
    }

    // SIMD operations (if available)
    println!("\n🚀 SIMD Operations:");

    #[cfg(feature = "simd")]
    {
        use scirs2_special::simd_ops::{benchmark_simd_performance, gamma_f64_simd};

        let simd_test_array = Array1::linspace(0.1, 5.0, 1000);

        println!("Comparing scalar vs SIMD performance:");

        // Scalar version
        let start = Instant::now();
        let _scalar_result = simd_test_array.mapv(|x| gamma(x));
        let scalar_time = start.elapsed();

        // SIMD version
        let start = Instant::now();
        let _simd_result = gamma_f64_simd(&simd_test_array.view());
        let simd_time = start.elapsed();

        println!("Scalar: {:?}", scalar_time);
        println!("SIMD:   {:?}", simd_time);

        if simd_time.as_nanos() > 0 {
            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
            println!("Speedup: {:.2}x", speedup);
        }

        // Run comprehensive benchmark
        println!("\nRunning SIMD performance benchmark...");
        benchmark_simd_performance(10000)?;
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("SIMD operations require the 'simd' feature.");
        println!("Enable with: cargo run --features simd");
    }

    // Parallel operations
    println!("\n🔄 Parallel Operations:");

    #[cfg(feature = "parallel")]
    {
        use scirs2_special::simd_ops::{benchmark_parallel_performance, gamma_f64_parallel};

        let parallel_test_array = Array1::linspace(0.1, 5.0, 10000);

        println!("Comparing sequential vs parallel performance:");

        // Sequential version
        let start = Instant::now();
        let _seq_result = parallel_test_array.mapv(|x| gamma(x));
        let seq_time = start.elapsed();

        // Parallel version
        let start = Instant::now();
        let _par_result = gamma_f64_parallel(&parallel_test_array.view());
        let par_time = start.elapsed();

        println!("Sequential: {:?}", seq_time);
        println!("Parallel:   {:?}", par_time);

        if par_time.as_nanos() > 0 {
            let speedup = seq_time.as_nanos() as f64 / par_time.as_nanos() as f64;
            println!("Speedup: {:.2}x", speedup);
        }

        println!("\nRunning parallel performance benchmark...");
        benchmark_parallel_performance(10000)?;
    }

    #[cfg(not(feature = "parallel"))]
    {
        println!("Parallel operations require the 'parallel' feature.");
        println!("Enable with: cargo run --features parallel");
    }

    // Interactive array operations
    loop {
        println!("\n🧮 Interactive Array Operations:");
        println!("1. Create custom array and apply function");
        println!("2. Compare functions on same array");
        println!("3. Statistical analysis of results");
        println!("4. Export array results to CSV");
        println!("5. Back to main menu");

        let choice = get_user_input("Enter choice (1-5): ")?;
        match choice.as_str() {
            "1" => {
                let start = get_user_input("Enter start value: ")?
                    .parse::<f64>()
                    .unwrap_or(0.1);
                let end = get_user_input("Enter end value: ")?
                    .parse::<f64>()
                    .unwrap_or(5.0);
                let size = get_user_input("Enter array size: ")?
                    .parse::<usize>()
                    .unwrap_or(100);

                if size > 0 && size <= 1_000_000 && start < end {
                    let array = Array1::linspace(start, end, size);

                    println!("Choose function:");
                    println!("1. Gamma  2. Bessel J₀  3. Error  4. Sine");
                    let func_choice = get_user_input("Function choice: ")?;

                    let start_time = Instant::now();
                    let result = match func_choice.as_str() {
                        "1" => array.mapv(|x| gamma(x)),
                        "2" => array.mapv(|x| j0(x)),
                        "3" => array.mapv(|x| erf(x)),
                        "4" => array.mapv(|x| sinc(x)),
                        _ => {
                            println!("❌ Invalid function choice");
                            continue;
                        }
                    };
                    let elapsed = start_time.elapsed();

                    println!("Computation completed in {:?}", elapsed);
                    println!("First 5 results:");
                    for i in 0..result.len().min(5) {
                        println!("  f({:.4}) = {:.6}", array[i], result[i]);
                    }

                    let min_val = result.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_val = result.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let mean_val = result.mean();

                    println!(
                        "Statistics: min={:.4}, max={:.4}, mean={:.4}",
                        min_val, max_val, mean_val
                    );
                } else {
                    println!("❌ Invalid parameters");
                }
            }
            "2" => {
                let array = Array1::linspace(0.1, 3.0, 20);
                println!("Comparing functions on array from 0.1 to 3.0:");
                println!("x      gamma(x)   j0(x)     erf(x)    sinc(x)");
                println!("-----  --------   ------    ------    -------");

                for i in 0..array.len().min(10) {
                    let x = array[i];
                    println!(
                        "{:5.2}  {:8.4}   {:6.4}    {:6.4}    {:7.4}",
                        x,
                        gamma(x),
                        j0(x),
                        erf(x),
                        sinc(x)
                    );
                }
            }
            "3" => {
                let test_array = Array1::linspace(0.5, 2.5, 1000);
                let gamma_vals = test_array.mapv(|x| gamma(x));

                let n = gamma_vals.len() as f64;
                let mean = gamma_vals.sum() / n;
                let variance = gamma_vals.mapv(|x| (x - mean).powi(2)).sum() / n;
                let std_dev = variance.sqrt();

                let min_val = gamma_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_val = gamma_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                println!("Statistical analysis of Γ(x) on [0.5, 2.5]:");
                println!("Mean: {:.6}", mean);
                println!("Std deviation: {:.6}", std_dev);
                println!("Min: {:.6}", min_val);
                println!("Max: {:.6}", max_val);
                println!("Range: {:.6}", max_val - min_val);
            }
            "4" => {
                println!("This would export results to CSV format.");
                println!("In a real implementation, you could:");
                println!("• Write arrays to files");
                println!("• Generate plots");
                println!("• Export to various formats");
                println!("• Interface with data analysis tools");
            }
            "5" => break,
            _ => println!("❌ Invalid choice"),
        }
    }

    println!("\n✅ Array operations tutorial completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn advanced_features_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ ADVANCED FEATURES TUTORIAL");
    println!("=============================\n");

    println!("This section covers advanced features like:");
    println!("• High-precision arithmetic");
    println!("• Complex number support");
    println!("• GPU acceleration");
    println!("• Cross-validation and testing");
    println!("• Stability analysis\n");

    // High-precision arithmetic
    println!("🔬 HIGH-PRECISION ARITHMETIC");

    #[cfg(feature = "high-precision")]
    {
        use scirs2_special::arbitrary_precision::*;

        println!("Computing with arbitrary precision (using rug library):");

        // Create precision context
        let ctx = PrecisionContext::new(100)?; // 100 decimal digits

        // High-precision gamma
        println!("Standard precision: Γ(0.5) = {:.15}", gamma(0.5));

        match gamma_ap(0.5, &ctx) {
            Ok(hp_gamma) => {
                let hp_f64 = to_f64(&hp_gamma);
                println!("High precision:     Γ(0.5) = {:.15}", hp_f64);
                println!(
                    "Expected (√π):             = {:.15}",
                    std::f64::consts::PI.sqrt()
                );
            }
            Err(e) => println!("High-precision computation failed: {}", e),
        }

        // High-precision comparison
        match (gamma_ap(1.5, &ctx), gamma_ap(0.5, &ctx)) {
            (Ok(gamma_1_5), Ok(gamma_0_5)) => {
                println!("Γ(1.5) = 0.5 × Γ(0.5) identity check:");
                let computed_1_5 = to_f64(&gamma_1_5);
                let expected_1_5 = 0.5 * to_f64(&gamma_0_5);
                println!("Computed: {:.15}", computed_1_5);
                println!("Expected: {:.15}", expected_1_5);
                println!("Difference: {:.2e}", (computed_1_5 - expected_1_5).abs());
            }
            _ => println!("High-precision computation failed"),
        }
    }

    #[cfg(not(feature = "high-precision"))]
    {
        println!("High-precision arithmetic requires the 'high-precision' feature.");
        println!("Enable with: cargo run --features high-precision");
        println!("This feature uses the 'rug' crate for arbitrary precision arithmetic.");
    }

    // Complex number support
    println!("\n🌀 COMPLEX NUMBER SUPPORT");
    println!("Many functions support complex arguments:");

    use scirs2_core::Complex64;

    let z1 = Complex64::new(1.0, 0.5);
    let z2 = Complex64::new(0.0, 1.0); // Pure imaginary
    let z3 = Complex64::new(-0.5, 0.0); // Negative real

    // Complex gamma function
    #[cfg(any(feature = "default", not(feature = "high-precision")))]
    {
        use scirs2_special::gamma_complex;

        println!("Complex gamma function:");
        for &z in &[z1, z2, z3] {
            let gamma_z = gamma_complex(z);
            println!(
                "Γ({:.3} + {:.3}i) = {:.6} + {:.6}i",
                z.re, z.im, gamma_z.re, gamma_z.im
            );
        }
    }

    // Complex error function
    #[cfg(any(feature = "default", not(feature = "high-precision")))]
    {
        use scirs2_special::erf_complex;

        println!("\nComplex error function:");
        for &z in &[z1, z2, z3] {
            let erf_z = erf_complex(z);
            println!(
                "erf({:.3} + {:.3}i) = {:.6} + {:.6}i",
                z.re, z.im, erf_z.re, erf_z.im
            );
        }
    }

    // Complex Bessel functions
    #[cfg(any(feature = "default", not(feature = "high-precision")))]
    {
        use scirs2_special::bessel::complex::j0_complex;

        println!("\nComplex Bessel J₀:");
        for &z in &[z1, z2, z3] {
            let j0_z = j0_complex(z);
            println!(
                "J₀({:.3} + {:.3}i) = {:.6} + {:.6}i",
                z.re, z.im, j0_z.re, j0_z.im
            );
        }
    }

    // GPU acceleration
    println!("\n🚀 GPU ACCELERATION");

    #[cfg(feature = "gpu")]
    {
        use scirs2_special::gpu_ops::*;

        println!("GPU acceleration is available for large arrays.");
        println!("Testing GPU gamma function...");

        let test_array = Array1::linspace(0.1, 5.0, 10000);
        let mut gpu_result = Array1::zeros(test_array.len());

        match gamma_gpu(&test_array.view(), &mut gpu_result.view_mut()) {
            Ok(()) => {
                println!("GPU computation successful!");
                println!("First few results:");
                for i in 0..5 {
                    println!("  Γ({:.3}) = {:.6}", test_array[i], gpu_result[i]);
                }

                // Compare with CPU version
                let cpu_result = test_array.mapv(|x| gamma(x));
                let max_diff = cpu_result
                    .iter()
                    .zip(gpu_result.iter())
                    .map(|(a, b)| (*a as f64 - *b as f64).abs())
                    .fold(0.0, f64::max);

                println!("Maximum difference from CPU: {:.2e}", max_diff);
            }
            Err(e) => println!("GPU computation failed: {}", e),
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU acceleration requires the 'gpu' feature.");
        println!("Enable with: cargo run --features gpu");
        println!("Note: GPU acceleration requires compatible hardware and drivers.");
    }

    // Cross-validation and testing
    println!("\n✅ CROSS-VALIDATION");

    #[cfg(any(feature = "default", not(feature = "gpu")))]
    {
        use scirs2_special::cross_validation::*;

        println!("Cross-validation compares our implementations with reference values:");

        match generate_test_suite() {
            Ok(_) => {
                println!("Cross-validation test suite generated successfully!");
            }
            Err(e) => println!("Cross-validation failed: {}", e),
        }
    }

    // Stability analysis
    println!("\n📊 STABILITY ANALYSIS");

    #[cfg(any(feature = "default", not(feature = "gpu")))]
    {
        use scirs2_special::stability_analysis::*;

        println!("Analyzing numerical stability of implementations...");

        match run_stability_tests() {
            Ok(()) => {
                println!("Stability analysis completed!");
                println!("Results saved to STABILITY_ANALYSIS.md");

                // Show gamma stability as an example
                let gamma_analysis = gamma_stability::analyze_gamma_stability();
                println!("\nGamma function stability summary:");
                println!("- {} stability issues found", gamma_analysis.issues.len());
                println!(
                    "- {} condition numbers computed",
                    gamma_analysis.condition_numbers.len()
                );
                println!(
                    "- Max relative error: {:.2e}",
                    gamma_analysis.accuracy_metrics.max_relative_error
                );
                println!(
                    "- Mean relative error: {:.2e}",
                    gamma_analysis.accuracy_metrics.mean_relative_error
                );
            }
            Err(e) => println!("Stability analysis failed: {}", e),
        }
    }

    // Python interoperability
    println!("\n🐍 PYTHON INTEROPERABILITY");

    #[cfg(feature = "python-interop")]
    {
        use scirs2_special::python_interop::*;

        println!("Python interoperability features:");
        println!("• Code translation assistance");
        println!("• API mapping documentation");
        println!("• Migration helpers");

        let python_code = "import scipy.special as sp\nresult = sp.gamma(2.5)";
        println!("\nExample Python code translation:");
        println!("Python: {}", python_code);

        // Translation functionality not implemented yet
        println!("Rust:   scirs2_special::gamma(2.5)");
    }

    #[cfg(not(feature = "python-interop"))]
    {
        println!("Python interoperability requires the 'python-interop' feature.");
        println!("Enable with: cargo run --features python-interop");
    }

    // Visualization
    println!("\n📈 VISUALIZATION");

    #[cfg(feature = "plotting")]
    {
        use scirs2_special::visualization::*;

        println!("Visualization capabilities are available:");
        println!("• 2D and 3D plotting");
        println!("• Interactive visualizations");
        println!("• Export to various formats");

        // Generate an interactive plot
        let interactive_gamma = interactive::create_gamma_plot();
        println!(
            "Generated interactive gamma function plot ({} characters)",
            interactive_gamma.len()
        );

        // This would normally save to a file
        println!("Interactive plots can be saved as HTML files for web viewing.");
    }

    #[cfg(not(feature = "plotting"))]
    {
        println!("Visualization requires the 'plotting' feature.");
        println!("Enable with: cargo run --features plotting");
    }

    // Performance monitoring
    println!("\n⏱️ PERFORMANCE MONITORING");
    println!("Performance can be monitored using:");
    println!("• Built-in benchmarking functions");
    println!("• Timing comparisons with SciPy");
    println!("• Memory usage analysis");
    println!("• SIMD and parallel scaling studies");

    let test_sizes = vec![100, 1000, 10000];
    for &size in &test_sizes {
        let array = Array1::linspace(0.1, 5.0, size);
        let start = std::time::Instant::now();
        let _result = array.mapv(|x| gamma(x));
        let elapsed = start.elapsed();

        let throughput = size as f64 / elapsed.as_secs_f64();
        println!("Size {}: {:.0} evaluations/second", size, throughput);
    }

    println!("\n✅ Advanced features tutorial completed!\n");
    Ok(())
}

/// Simple ASCII plotting function for demonstrations
#[allow(dead_code)]
fn plot_function_ascii<F>(title: &str, xmin: f64, xmax: f64, width: usize, f: F)
where
    F: Fn(f64) -> f64,
{
    println!("📈 {}", title);

    let mut values = Vec::new();
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;

    // Collect function values
    for i in 0..width {
        let x = xmin + (xmax - xmin) * i as f64 / (width - 1) as f64;
        let y = f(x);
        if y.is_finite() {
            values.push((x, y));
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
    }

    if values.is_empty() {
        println!("No finite values to plot");
        return;
    }

    // Plot parameters
    let height = 20;
    let y_range = ymax - ymin;
    if y_range == 0.0 {
        println!("Constant function: f(x) = {:.3}", ymin);
        return;
    }

    // Create the plot
    for row in 0..height {
        let y_level = ymax - y_range * row as f64 / (height - 1) as f64;

        print!("{:8.3} │", y_level);

        for &(_, y) in &values {
            let char = if (y - y_level).abs() < y_range / (height as f64 * 2.0) {
                '●'
            } else if y > y_level {
                ' '
            } else {
                ' '
            };
            print!("{}", char);
        }
        println!();
    }

    // X-axis
    print!("         └");
    for _ in 0..width {
        print!("─");
    }
    println!();

    print!("          ");
    for i in 0..5 {
        let x = xmin + (xmax - xmin) * i as f64 / 4.0;
        print!("{:8.1}  ", x);
    }
    println!();
}
