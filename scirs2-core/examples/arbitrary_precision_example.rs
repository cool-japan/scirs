//! Example demonstrating arbitrary precision numerical computation support
//!
//! This example shows how to use the arbitrary precision types for
//! high-precision scientific calculations.

#[cfg(feature = "arbitrary-precision")]
use scirs2_core::numeric::arbitrary_precision::{
    utils, ArbitraryComplex, ArbitraryFloat, ArbitraryInt, ArbitraryPrecisionBuilder,
    ArbitraryRational, RoundingMode,
};
#[cfg(feature = "arbitrary-precision")]
use scirs2_core::CoreResult;

#[cfg(not(feature = "arbitrary-precision"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the 'arbitrary-precision' feature.");
    println!(
        "Run with: cargo run --example arbitrary_precision_example --features arbitrary-precision"
    );
}

#[cfg(feature = "arbitrary-precision")]
#[allow(dead_code)]
fn main() -> CoreResult<()> {
    println!("=== Arbitrary Precision Numerical Computation Example ===\n");

    // Example 1: Basic arbitrary precision integers
    println!("1. Arbitrary Precision Integers:");
    demo_arbitrary_integers()?;

    // Example 2: High-precision floating point
    println!("\n2. High-Precision Floating Point:");
    demo_arbitrary_floats()?;

    // Example 3: Exact rational arithmetic
    println!("\n3. Exact Rational Arithmetic:");
    demo_rational_arithmetic()?;

    // Example 4: Complex numbers with arbitrary precision
    println!("\n4. Arbitrary Precision Complex Numbers:");
    demo_complex_numbers()?;

    // Example 5: Mathematical constants to high precision
    println!("\n5. Mathematical Constants:");
    demo_constants()?;

    // Example 6: Precision builder pattern
    println!("\n6. Precision Builder Pattern:");
    demo_precision_builder()?;

    // Example 7: Numerical analysis applications
    println!("\n7. Numerical Analysis Applications:");
    demo_numerical_analysis()?;

    // Example 8: Large number computations
    println!("\n8. Large Number Computations:");
    demo_large_numbers()?;

    Ok(())
}

#[cfg(feature = "arbitrary-precision")]
#[allow(dead_code)]
fn demo_arbitrary_integers() -> CoreResult<()> {
    // Basic arithmetic with large integers
    let a = ArbitraryInt::from_str_radix("12345678901234567890", 10)?;
    let b = ArbitraryInt::from_str_radix("98765432109876543210", 10)?;

    println!("a = {}", a);
    println!("b = {}", b);
    println!("a + b = {}", a.clone() + b.clone());
    println!("a * b = {}", a.clone() * b.clone());

    // Factorial of large numbers
    let n = 100;
    let factorial_100 = ArbitraryInt::factorial(n);
    println!("\n{}! = {}", n, factorial_100);
    println!("Number of digits: {}", factorial_100.to_string().len());

    // Binomial coefficients
    let binom = ArbitraryInt::binomial(100, 50);
    println!("\nC(100, 50) = {}", binom);

    // GCD and LCM
    let x = ArbitraryInt::from_i64(123456789);
    let y = ArbitraryInt::from_i64(987654321);
    println!("\nGCD({}, {}) = {}", x, y, x.gcd(&y));
    println!("LCM({}, {}) = {}", x, y, x.lcm(&y));

    // Prime testing
    let prime_candidate =
        ArbitraryInt::from_str_radix("170141183460469231731687303715884105727", 10)?;
    println!(
        "\nIs {} prime? {}",
        prime_candidate,
        prime_candidate.is_probably_prime(20)
    );

    Ok(())
}

#[cfg(feature = "arbitrary-precision")]
#[allow(dead_code)]
fn demo_arbitrary_floats() -> CoreResult<()> {
    // Set precision to 256 bits (about 77 decimal digits)
    let prec = 256;

    // Basic arithmetic with high precision
    let one_third = ArbitraryFloat::from_f64_withprecision(1.0, prec)?
        / ArbitraryFloat::from_f64_withprecision(3.0, prec)?;
    println!("1/3 with {} bits precision:", prec);
    println!("{}", one_third);
    println!("Decimal precision: {} digits", one_third.decimalprecision());

    // Verify precision by multiplying back
    let three = ArbitraryFloat::from_f64_withprecision(3.0, prec)?;
    let result = one_third * three;
    println!("\n(1/3) * 3 = {}", result);

    // Transcendental functions
    let x = ArbitraryFloat::from_f64_withprecision(0.5, prec)?;
    println!("\nx = {}", x);
    println!("sin(x) = {}", x.sin());
    println!("cos(x) = {}", x.cos());
    println!("exp(x) = {}", x.exp());
    println!("ln(exp(x)) = {}", x.exp().ln()?);

    // Demonstrate precision loss in subtraction
    let a = ArbitraryFloat::from_f64_withprecision(1.0, prec)?;
    let b = ArbitraryFloat::from_f64_withprecision(1.0 - 1e-50, prec)?;
    let diff = a - b;
    println!("\nCatastrophic cancellation example:");
    println!("1.0 - (1.0 - 1e-50) = {}", diff);

    Ok(())
}

#[cfg(feature = "arbitrary-precision")]
#[allow(dead_code)]
fn demo_rational_arithmetic() -> CoreResult<()> {
    // Exact rational arithmetic
    let r1 = ArbitraryRational::num(22, 7)?;
    let r2 = ArbitraryRational::num(355, 113)?;

    println!("r1 = {} ≈ {:.10}", r1, r1.to_f64());
    println!("r2 = {} ≈ {:.10}", r2, r2.to_f64());

    // Rational arithmetic is exact
    let sum = r1.clone() + r2.clone();
    println!("\nr1 + r2 = {}", sum);

    let product = r1.clone() * r2.clone();
    println!("r1 * r2 = {}", product);

    // Working with fractions
    let half = ArbitraryRational::num(1, 2)?;
    let third = ArbitraryRational::num(1, 3)?;
    let sixth = ArbitraryRational::num(1, 6)?;

    let result = half + third.clone() - sixth;
    println!("\n1/2 + 1/3 - 1/6 = {}", result);

    // Convert to high precision float
    let pi_approx = ArbitraryRational::num(355, 113)?;
    let pi_float = pi_approx.to_arbitrary_float(256)?;
    println!("\n355/113 as 256-bit float: {}", pi_float);

    Ok(())
}

#[cfg(feature = "arbitrary-precision")]
#[allow(dead_code)]
fn demo_complex_numbers() -> CoreResult<()> {
    let prec = 128;

    // Create complex numbers
    let z1 = ArbitraryComplex::re_2(3.0, 4.0);
    let z2 = ArbitraryComplex::re_2(1.0, -2.0);

    println!("z1 = {}", z1);
    println!("z2 = {}", z2);

    // Complex arithmetic
    println!("\nz1 + z2 = {}", z1.clone() + z2.clone());
    println!("z1 * z2 = {}", z1.clone() * z2.clone());
    println!("z1 / z2 = {}", z1.clone() / z2.clone());

    // Complex functions
    println!("\n|z1| = {}", z1.abs());
    println!("arg(z1) = {} radians", z1.arg());
    println!("conj(z1) = {}", z1.conj());

    // Euler's identity: e^(iπ) + 1 = 0
    let pi = utils::pi(prec)?;
    let i_pi = ArbitraryComplex::re(&ArbitraryFloat::from_f64_withprecision(0.0, prec)?, &pi);
    let euler = i_pi.exp() + ArbitraryComplex::re_2(1.0, 0.0);
    println!("\nEuler's identity: e^(iπ) + 1 = {}", euler);
    println!("Magnitude: {}", euler.abs());

    Ok(())
}

#[cfg(feature = "arbitrary-precision")]
#[allow(dead_code)]
fn demo_constants() -> CoreResult<()> {
    // Compute mathematical constants to high precision
    let prec = 512; // About 154 decimal digits

    println!(
        "Computing constants to {} bits ({} decimal digits):",
        prec,
        (prec as f64 / 3.32) as u32
    );

    let pi = utils::pi(prec)?;
    println!("\nπ = {}", pi);

    let e = utils::e(prec)?;
    println!("\ne = {}", e);

    let ln2 = utils::ln2(prec)?;
    println!("\nln(2) = {}", ln2);

    let sqrt2 = utils::sqrt2(prec)?;
    println!("\n√2 = {}", sqrt2);

    let phi = utils::golden_ratio(prec)?;
    println!("\nφ (golden ratio) = {}", phi);

    // Verify: φ = (1 + √5) / 2
    let one = ArbitraryFloat::from_f64_withprecision(1.0, prec)?;
    let five = ArbitraryFloat::from_f64_withprecision(5.0, prec)?;
    let two = ArbitraryFloat::from_f64_withprecision(2.0, prec)?;
    let phi_check = (one + five.sqrt()?) / two;
    println!("Verification: (1 + √5) / 2 = {}", phi_check);

    Ok(())
}

#[cfg(feature = "arbitrary-precision")]
#[allow(dead_code)]
fn demo_precision_builder() -> CoreResult<()> {
    // Use the builder pattern for custom precision settings
    let calc = ArbitraryPrecisionBuilder::new()
        .decimalprecision(100)  // 100 decimal digits
        .rounding(RoundingMode::Nearest)
        .trackprecision(true)
        .build_float();

    println!("Built float with {} bits precision", calc.precision());

    // Perform calculations with specific precision context
    let result: ArbitraryFloat = ArbitraryPrecisionBuilder::new()
        .precision(384)  // 384 bits
        .calculate(|ctx| -> CoreResult<ArbitraryFloat> {
            println!("Calculating with {} bits precision", ctx.floatprecision);

            // Compute π² / 6 (Basel problem)
            let pi = utils::pi(ctx.floatprecision)?;
            let six = ArbitraryFloat::from_f64_withprecision(6.0, ctx.floatprecision)?;
            let pi_squared = pi.clone() * pi;
            Ok(pi_squared / six)
        })?;

    println!("\nπ²/6 = {}", result);

    // The Basel problem: π²/6 = 1 + 1/4 + 1/9 + 1/16 + ...
    // Let's verify by computing the sum
    let mut sum = ArbitraryFloat::from_f64_withprecision(0.0, 384)?;
    for n in 1..=1000 {
        let n_float = ArbitraryFloat::from_f64_withprecision(n as f64, 384)?;
        let term = ArbitraryFloat::from_f64_withprecision(1.0, 384)? / (n_float.clone() * n_float);
        sum = sum + term;
    }
    println!("Sum of 1/n² for n=1..1000 = {}", sum);

    Ok(())
}

#[cfg(feature = "arbitrary-precision")]
#[allow(dead_code)]
fn demo_numerical_analysis() -> CoreResult<()> {
    let prec = 256;

    // Newton's method for square root
    println!("Newton's method for √2:");
    let two = ArbitraryFloat::from_f64_withprecision(2.0, prec)?;
    let mut x = ArbitraryFloat::from_f64_withprecision(1.5, prec)?; // Initial guess

    for i in 0..5 {
        let x_new = (x.clone() + two.clone() / x.clone())
            / ArbitraryFloat::from_f64_withprecision(2.0, prec)?;
        println!("Iteration {}: {}", i + 1, x_new);
        x = x_new;
    }

    let actual_sqrt2 = two.sqrt()?;
    println!("Actual √2:     {}", actual_sqrt2);

    // Computing derivatives numerically with high precision
    println!("\nNumerical differentiation of sin(x) at x = π/4:");
    let pi = utils::pi(prec)?;
    let four = ArbitraryFloat::from_f64_withprecision(4.0, prec)?;
    let x = pi / four;

    // Use very small h for numerical derivative
    let h = ArbitraryFloat::from_f64_withprecision(1e-50, prec)?;
    let f_x = x.sin();
    let f_x_plus_h = (x.clone() + h.clone()).sin();
    let derivative = (f_x_plus_h - f_x) / h;

    println!("f'(π/4) ≈ {}", derivative);
    println!("cos(π/4) = {}", x.cos());

    Ok(())
}

#[cfg(feature = "arbitrary-precision")]
#[allow(dead_code)]
fn demo_large_numbers() -> CoreResult<()> {
    // Mersenne primes
    println!("Testing Mersenne numbers:");

    // 2^n - 1 for various n
    let two = ArbitraryInt::from_i64(2);
    for n in [13, 17, 19, 31, 61] {
        let exp = ArbitraryInt::from_i64(n);
        let one = ArbitraryInt::from_i64(1);

        // Use modular arithmetic for efficiency
        let mersenne =
            two.mod_pow(&exp, &ArbitraryInt::from_str_radix(&"1".repeat(1000), 10)?)? - one;

        // For demonstration, we'll create the actual number for smaller values
        if n <= 31 {
            let mut mersenne = ArbitraryInt::from_i64(1);
            for _ in 0..n {
                mersenne = mersenne * ArbitraryInt::from_i64(2);
            }
            mersenne = mersenne - ArbitraryInt::from_i64(1);

            println!("\nM{} = 2^{} - 1 = {}", n, n, mersenne);
            println!("Is probably prime? {}", mersenne.is_probably_prime(20));
        }
    }

    // Fibonacci numbers with arbitrary precision
    println!("\n\nLarge Fibonacci numbers:");
    let mut a = ArbitraryInt::from_i64(0);
    let mut b = ArbitraryInt::from_i64(1);

    for n in 0..=100 {
        if n % 25 == 0 {
            println!("F({}) = {}", n, a);
            println!("  Digits: {}", a.to_string().len());
        }
        let next = a.clone() + b.clone();
        a = b;
        b = next;
    }

    Ok(())
}
