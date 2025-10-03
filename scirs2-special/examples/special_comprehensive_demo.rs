//! Comprehensive demonstration of scirs2-special functions
//!
//! This example showcases the wide range of special functions available
//! in the scirs2-special crate, including mathematical properties,
//! numerical precision, and performance optimizations.

use scirs2_core::ndarray::Array1;
use scirs2_special::*;
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SCIRS2-SPECIAL Comprehensive Demo ===\n");

    // 1. Basic Special Functions
    demo_basic_functions()?;

    // 2. Combinatorial Functions
    demo_combinatorial_functions()?;

    // 3. Statistical Functions
    demo_statistical_functions()?;

    // 4. Mathematical Properties and Precision
    demo_mathematical_properties()?;

    // 5. Performance Optimizations
    demo_performance_features()?;

    // 6. Edge Cases and Numerical Stability
    demo_edge_cases()?;

    println!("=== Demo completed successfully! ===");
    Ok(())
}

#[allow(dead_code)]
fn demo_basic_functions() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Special Functions");
    println!("=========================");

    // Gamma function
    println!("Γ(5) = {:.10}", gamma(5.0));
    println!("Γ(0.5) = √π = {:.10}", gamma(0.5));

    // Beta function
    println!("B(2,3) = {:.10}", beta(2.0, 3.0));

    // Error functions
    println!("erf(1) = {:.10}", erf(1.0));
    println!("erfc(1) = {:.10}", erfc(1.0));

    // Bessel functions
    println!("J₀(2.4) = {:.10}", j0(2.4));
    println!("J₁(1.0) = {:.10}", j1(1.0));

    // Airy functions
    println!("Ai(1.0) = {:.10}", ai(1.0));
    println!("Bi(1.0) = {:.10}", bi(1.0));

    // Lambert W function
    use scirs2_core::Complex64;
    let w = lambert_w(Complex64::new(1.0, 0.0), 0, 1e-12)?;
    println!("W(1) = {:.10}", w.re);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_combinatorial_functions() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Combinatorial Functions");
    println!("==========================");

    // Basic combinatorial functions
    println!("10! = {:.0}", factorial(10)?);
    println!("7!! = {:.0}", double_factorial(7)?);
    println!("C(10,3) = {:.0}", binomial(10, 3)?);
    println!("P(10,3) = {:.0}", permutations(10, 3)?);

    // Advanced combinatorial numbers
    println!("Stirling 1st kind s(5,3) = {:.0}", stirling_first(5, 3)?);
    println!("Stirling 2nd kind S(5,3) = {:.0}", stirling_second(5, 3)?);
    println!("Bell number B(5) = {:.0}", bell_number(5)?);

    // Special number sequences
    println!("Bernoulli B₂ = {:.10}", bernoulli_number(2)?);
    println!("Euler E₄ = {:.0}", euler_number(4)?);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_statistical_functions() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Statistical Functions");
    println!("========================");

    // Logistic function
    let x = 2.0;
    println!("σ({}) = {:.10}", x, logistic(x));
    println!("σ'({}) = {:.10}", x, logistic_derivative(x));

    // Softmax function
    let values = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
    let softmax_result = softmax(values.view())?;
    println!("softmax([1,2,3,4]) = {:?}", softmax_result);
    println!("Sum = {:.10}", softmax_result.sum());

    // Log-softmax for numerical stability
    let log_softmax_result = log_softmax(values.view())?;
    println!("log_softmax([1,2,3,4]) = {:?}", log_softmax_result);

    // LogSumExp
    println!("LogSumExp([1,2,3,4]) = {:.10}", logsumexp(values.view())?);

    // Sinc function
    println!("sinc(π) = {:.10}", sinc(PI));
    println!("sinc(0) = {:.10}", sinc(0.0));

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_mathematical_properties() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Mathematical Properties and Precision");
    println!("========================================");

    // Gamma function recurrence relation: Γ(x+1) = x·Γ(x)
    let x: f64 = 2.5;
    let gamma_x: f64 = gamma(x);
    let gamma_x_plus_1: f64 = gamma(x + 1.0);
    let recurrence_check: f64 = x * gamma_x;
    println!("Γ({}) = {:.10}", x, gamma_x);
    println!("Γ({}) = {:.10}", x + 1.0, gamma_x_plus_1);
    println!("{}·Γ({}) = {:.10}", x, x, recurrence_check);
    println!(
        "Recurrence error: {:.2e}",
        (gamma_x_plus_1 - recurrence_check).abs() as f64
    );

    // Error function symmetry: erf(-x) = -erf(x)
    let x: f64 = 1.5;
    let erf_x: f64 = erf(x);
    let erf_neg_x: f64 = erf(-x);
    println!("erf({}) = {:.10}", x, erf_x);
    println!("erf({}) = {:.10}", -x, erf_neg_x);
    println!("Symmetry error: {:.2e}", (erf_x + erf_neg_x).abs() as f64);

    // Error function complement: erf(x) + erfc(x) = 1
    let complement_sum: f64 = erf_x + erfc(x);
    println!("erf({}) + erfc({}) = {:.10}", x, x, complement_sum);
    println!(
        "Complement error: {:.2e}",
        (complement_sum - 1.0).abs() as f64
    );

    // High-precision constants
    use scirs2_special::precision::constants::*;
    println!("π (extended) = {:.15}", PI_EXTENDED);
    println!("e (extended) = {:.15}", E_EXTENDED);
    println!("γ (extended) = {:.15}", GAMMA_EXTENDED);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_performance_features() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Performance Features");
    println!("=======================");

    // Lookup table optimization for factorials
    use scirs2_special::optimizations::lookup_tables;
    println!(
        "Factorial lookup 10! = {:.0}",
        lookup_tables::factorial_lookup(10).unwrap_or(0.0)
    );
    println!(
        "Gamma lookup Γ(0.5) = {:.10}",
        lookup_tables::gamma_lookup(0.5).unwrap_or(0.0)
    );

    // SIMD operations for arrays
    use scirs2_special::optimizations::simd;
    let values = vec![0.0, 1.0, 2.0, 3.0];
    let exp_results = simd::exp_simd(&values);
    println!("SIMD exp([0,1,2,3]) = {:?}", exp_results);

    // Adaptive algorithms
    use scirs2_special::optimizations::adaptive;
    println!("Adaptive γ(5) = {:.10}", adaptive::gamma_adaptive(5.0));
    println!("Adaptive exp(2) = {:.10}", adaptive::exp_adaptive(2.0));

    // Vectorized operations
    use scirs2_special::optimizations::vectorized;
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; 4];
    vectorized::exp_vectorized(&input, &mut output)?;
    println!("Vectorized exp([1,2,3,4]) = {:?}", output);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Edge Cases and Numerical Stability");
    println!("=====================================");

    // Safe operations with overflow protection
    use scirs2_special::precision::safe_ops;
    println!("Safe exp(1000) = {}", safe_ops::safe_exp(1000.0));
    println!("Safe exp(-1000) = {}", safe_ops::safe_exp(-1000.0));

    // Extreme values
    println!("Γ(very small) = {:.2e}", gamma(1e-10));
    println!("erf(very large) = {:.10}", erf(10.0));
    println!("J₀(0) = {:.10}", j0(0.0));

    // Extended precision computations
    use scirs2_special::precision::extended;
    println!(
        "Extended sin(π/6) = {:.15}",
        extended::sin_extended(PI / 6.0)
    );
    println!(
        "Extended cos(π/3) = {:.15}",
        extended::cos_extended(PI / 3.0)
    );

    // Error analysis
    use scirs2_special::precision::error_analysis;
    let computed = std::f64::consts::PI;
    let exact = PI;
    let digits = error_analysis::accurate_digits(computed, exact);
    println!("π approximation accuracy: {} digits", digits);

    println!();
    Ok(())
}
