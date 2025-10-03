//! Advanced usage patterns for scirs2-special
//!
//! This example demonstrates advanced patterns and use cases for the
//! scirs2-special crate, including numerical analysis applications,
//! optimization techniques, and integration with scientific workflows.

use scirs2_core::ndarray::Array1;
use scirs2_special::*;
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Usage Patterns ===\n");

    // 1. Numerical Analysis Applications
    numerical_analysis_demo()?;

    // 2. Statistical Computing
    statistical_computing_demo()?;

    // 3. Physics and Engineering Applications
    physics_applications_demo()?;

    // 4. Optimization and Root Finding
    optimization_demo()?;

    println!("=== Advanced usage demo completed! ===");
    Ok(())
}

#[allow(dead_code)]
fn numerical_analysis_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Numerical Analysis Applications");
    println!("==================================");

    // Approximating functions with orthogonal polynomials
    println!("Orthogonal Polynomial Approximation:");
    let x = 0.5;
    let n_terms = 5;

    // Approximate exp(x) using Laguerre polynomials
    let mut exp_approx = 0.0;
    for n in 0..n_terms {
        let coeff = (-1.0_f64).powi(n as i32) / factorial(n as u32)? as f64;
        exp_approx += coeff * laguerre(n, x);
    }
    println!("exp({}) ≈ {:.6} (Laguerre series)", x, exp_approx);
    println!("exp({}) = {:.6} (exact)", x, x.exp());

    // Error function series validation
    println!("\nError Function Series Validation:");
    let x: f64 = 1.0;
    let mut erf_series: f64 = 0.0;
    let two_over_sqrt_pi: f64 = 2.0 / PI.sqrt();

    for n in 0..20 {
        let term =
            ((-1.0_f64).powi(n) * x.powi(2 * n + 1)) / (factorial(n as u32)? * (2 * n + 1) as f64);
        erf_series += term;
    }
    erf_series *= two_over_sqrt_pi;

    println!("erf({}) ≈ {:.10} (series)", x, erf_series);
    println!("erf({}) = {:.10} (library)", x, erf(x));
    println!("Error: {:.2e}", (erf_series - erf(x)).abs());

    println!();
    Ok(())
}

#[allow(dead_code)]
fn statistical_computing_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Statistical Computing");
    println!("========================");

    // Beta distribution density function
    fn beta_pdf(x: f64, alpha: f64, betaparam: f64) -> Result<f64, Box<dyn std::error::Error>> {
        if !(0.0..=1.0).contains(&x) || alpha <= 0.0 || betaparam <= 0.0 {
            return Ok(0.0);
        }

        let numerator = x.powf(alpha - 1.0) * (1.0 - x).powf(betaparam - 1.0);
        let denominator = beta(alpha, betaparam);
        Ok(numerator / denominator)
    }

    // Sample some values from Beta(2,3) density
    let alpha = 2.0;
    let beta_param = 3.0;
    let x_values = [0.1, 0.3, 0.5, 0.7, 0.9];

    println!("Beta({},{}) PDF values:", alpha, beta_param);
    for &x in &x_values {
        let pdf_val = beta_pdf(x, alpha, beta_param)?;
        println!("  f({:.1}) = {:.6}", x, pdf_val);
    }

    // Multinomial coefficient calculation using combinatorics
    println!("\nMultinomial Coefficients:");
    let n = 10;
    let categories = [3, 3, 4]; // n = sum of categories

    let mut multinomial = factorial(n)?;
    for &k in &categories {
        multinomial /= factorial(k)?;
    }
    println!(
        "Multinomial({};{},{},{}) = {:.0}",
        n, categories[0], categories[1], categories[2], multinomial
    );

    // Softmax temperature scaling
    println!("\nSoftmax Temperature Scaling:");
    let logits = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);

    for temperature in [0.5, 1.0, 2.0] {
        let scaled_logits = &logits / temperature;
        let probabilities = softmax(scaled_logits.view())?;
        println!("T={:.1}: {:?}", temperature, probabilities);
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn physics_applications_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Physics and Engineering Applications");
    println!("=======================================");

    // Quantum harmonic oscillator wave functions using Hermite polynomials
    println!("Quantum Harmonic Oscillator Wave Functions:");
    let x_values = [-2.0, -1.0, 0.0, 1.0, 2.0];

    for n in 0..=3 {
        println!("ψ_{}(x) values:", n);
        for &x in &x_values {
            // Normalized wave function: ψ_n(x) = (1/√(2^n n! √π)) * H_n(x) * exp(-x²/2)
            let normalization =
                1.0 / (2.0_f64.powi(n as i32) * factorial(n as u32)? * PI.sqrt()).sqrt();
            let hermite_val = hermite(n, x);
            let wave_function = normalization * hermite_val * (-x * x / 2.0).exp();
            print!("  x={:4.1}: {:8.5}", x, wave_function);
        }
        println!();
    }

    // Bessel functions in cylindrical coordinates (electromagnetic waves)
    println!("\nBessel Functions (Cylindrical Waveguide Modes):");
    let a = 1.0; // waveguide radius
    let r_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

    // TM₀₁ mode (first zero of J₀)
    let beta_01 = 2.405; // First zero of J₀
    println!("TM₀₁ mode field distribution:");
    for &r in &r_values {
        let field = j0(beta_01 * r / a);
        println!("  r/a={:.1}: E_z = {:7.4}", r, field);
    }

    // Airy functions in quantum mechanics (particle in triangular potential)
    println!("\nAiry Functions (Triangular Potential):");
    let x_values = [-3.0, -2.0, -1.0, 0.0, 1.0];
    for &x in &x_values {
        let ai_val = ai(x);
        let bi_val = bi(x);
        println!("  x={:4.1}: Ai(x)={:8.5}, Bi(x)={:8.5}", x, ai_val, bi_val);
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn optimization_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Optimization and Root Finding");
    println!("===============================");

    // Using special functions in optimization
    // Example: Find maximum of a function involving special functions

    // Newton's method for finding zeros using derivatives
    println!("Root Finding with Special Functions:");

    // Find where J₀(x) = 0.5 using Newton's method
    let target: f64 = 0.5;
    let mut x: f64 = 1.0; // Initial guess

    println!("Finding x where J₀(x) = {}", target);
    for iteration in 0..10 {
        let fx: f64 = j0(x) - target;
        let fpx: f64 = -j1(x); // J₀'(x) = -J₁(x)

        if fpx.abs() < 1e-15 {
            break;
        }

        let x_new: f64 = x - fx / fpx;
        println!(
            "  Iteration {}: x = {:.8}, J₀(x) = {:.8}",
            iteration,
            x,
            j0(x)
        );

        if (x_new - x).abs() < 1e-12 {
            x = x_new;
            break;
        }
        x = x_new;
    }
    println!("  Solution: x = {:.8}, J₀(x) = {:.8}", x, j0(x));

    // Optimization using gradient information
    println!("\nGradient-Based Optimization:");

    // Minimize f(x) = x² - 2*erf(x) + 1
    // f'(x) = 2x - 2*erf'(x) = 2x - 4/√π * exp(-x²)
    let mut x: f64 = 1.0;
    let learning_rate: f64 = 0.1;

    println!("Minimizing f(x) = x² - 2*erf(x) + 1:");
    for iteration in 0..10 {
        let f_val: f64 = x * x - 2.0 * erf(x) + 1.0;
        let gradient: f64 = 2.0 * x - 4.0 / PI.sqrt() * (-x * x).exp();

        println!(
            "  Iteration {}: x = {:.6}, f(x) = {:.6}",
            iteration, x, f_val
        );

        if gradient.abs() < 1e-8 {
            break;
        }

        x -= learning_rate * gradient;
    }

    // Performance benchmarking with different implementations
    println!("\nPerformance Optimization Example:");

    use std::time::Instant;

    // Compare standard vs optimized gamma function
    let test_values: Vec<f64> = (0..1000).map(|i| 0.5 + i as f64 * 0.01).collect();

    let start = Instant::now();
    let mut sum1 = 0.0;
    for &x in &test_values {
        sum1 += gamma(x);
    }
    let time1 = start.elapsed();

    let start = Instant::now();
    let mut sum2 = 0.0;
    for &x in &test_values {
        sum2 += scirs2_special::optimizations::adaptive::gamma_adaptive(x);
    }
    let time2 = start.elapsed();

    println!("Standard gamma: {:.2?}, sum = {:.6}", time1, sum1);
    println!("Adaptive gamma: {:.2?}, sum = {:.6}", time2, sum2);
    println!(
        "Speedup: {:.2}x",
        time1.as_nanos() as f64 / time2.as_nanos() as f64
    );

    println!();
    Ok(())
}
