use num_complex::Complex64;
use std::f64::consts::PI;
// Removed legacy numeric constant imports

// Simple implementation of Wright Omega function for demonstration purposes
fn wright_omega(z: Complex64, tol: f64) -> Complex64 {
    // Handle NaN inputs
    if z.re.is_nan() || z.im.is_nan() {
        return Complex64::new(f64::NAN, f64::NAN);
    }

    // Handle infinities
    if z.re.is_infinite() || z.im.is_infinite() {
        if z.re == f64::INFINITY {
            return z; // ω(∞ + yi) = ∞ + yi
        } else if z.re == f64::NEG_INFINITY {
            // Special cases for -∞ + yi based on the angle
            let angle = z.im;
            if angle.abs() <= PI / 2.0 {
                let zero = if angle >= 0.0 { 0.0 } else { -0.0 };
                return Complex64::new(0.0, zero);
            } else {
                let zero = if angle >= 0.0 { -0.0 } else { 0.0 };
                return Complex64::new(zero, 0.0);
            }
        }
        return z; // Other infinite cases map to themselves
    }

    // Handle singular points at z = -1 ± πi
    if (z.re + 1.0).abs() < tol && (z.im.abs() - PI).abs() < tol {
        return Complex64::new(-1.0, 0.0);
    }

    // For real z with large positive values, use an asymptotic approximation
    if z.im.abs() < tol && z.re > 1e20 {
        return Complex64::new(z.re, 0.0);
    }

    // For real z with large negative values, use exponential approximation
    if z.im.abs() < tol && z.re < -50.0 {
        return Complex64::new((-z.re).exp(), 0.0);
    }

    // Simple iterative solution using Halley's method
    // Initial guess
    let mut w = if z.norm() < 1.0 {
        // For small |z|, use a simple approximation
        z
    } else {
        // For larger |z|, use log(z) as initial guess
        z.ln()
    };

    // Halley's iteration
    let max_iterations = 100;
    for _ in 0..max_iterations {
        let w_exp_w = w * w.exp();
        let f = w_exp_w - z;

        // Check if we've converged
        if f.norm() < tol {
            break;
        }

        // Compute derivatives
        let f_prime = w.exp() * (w + Complex64::new(1.0, 0.0));
        let f_double_prime = w.exp() * (w + Complex64::new(2.0, 0.0));

        // Halley's formula (simplified)
        if (Complex64::new(2.0, 0.0) * f_prime * f_prime - f * f_double_prime).norm() < 1e-10 {
            // Use a dampened Newton step when denominator is small
            w -= f / f_prime * Complex64::new(0.5, 0.0);
        } else {
            // Full Newton step otherwise
            w -= f / f_prime;
        }
    }

    w
}

// Simple implementation for real-valued Wright Omega
fn wright_omega_real(x: f64, tol: f64) -> f64 {
    // Handle NaN input
    if x.is_nan() {
        return f64::NAN;
    }

    // Handle infinities
    if x == f64::INFINITY {
        return f64::INFINITY;
    } else if x == f64::NEG_INFINITY {
        return 0.0;
    }

    // For large positive values, use an asymptotic approximation
    if x > 1e20 {
        return x;
    }

    // For large negative values, use exponential approximation
    if x < -50.0 {
        return (-x).exp();
    }

    // For x < -1, the result could be complex, but we only handle real results
    // so we need to check if the complex result has a negligible imaginary part
    if x < -1.0 {
        let complex_result = wright_omega(Complex64::new(x, 0.0), tol);
        if complex_result.im.abs() < tol {
            return complex_result.re;
        } else {
            // Not a real result
            return f64::NAN;
        }
    }

    // Simple iterative solution for regular values
    // Initial guess
    let mut w = if x > -1.0 && x < 1.0 {
        // For small x, use a simple approximation
        x
    } else {
        // For larger x, use log(x) as initial guess
        x.ln().max(-100.0) // Avoid very negative values
    };

    // Newton's method
    let max_iterations = 50;
    for _ in 0..max_iterations {
        let f = w + w.ln() - x;

        // Check if we've converged
        if f.abs() < tol {
            break;
        }

        // Compute derivative: f'(w) = 1 + 1/w
        let f_prime = 1.0 + 1.0 / w;

        // Newton step
        w -= f / f_prime;
    }

    w
}

fn main() {
    println!("Wright Omega Function Example\n");

    // Real values
    println!("Wright Omega Function for Real Values:");
    println!("-------------------------------------");
    let real_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];
    for &x in &real_values {
        let omega = wright_omega_real(x, 1e-8);
        let check = omega + omega.ln();
        println!("ω({:.2}) = {:.10}", x, omega);
        println!(
            "  Verification: ω + ln(ω) = {:.10} (should be {:.2})",
            check, x
        );
        println!();
    }

    // Complex values
    println!("\nWright Omega Function for Complex Values:");
    println!("----------------------------------------");
    let complex_values = [
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(-1.0, 2.0),
        Complex64::new(2.0, -3.0),
    ];

    for &z in &complex_values {
        let omega = wright_omega(z, 1e-8);
        let check = omega + omega.ln();
        println!(
            "ω({:.2}{:+.2}i) = {:.6}{:+.6}i",
            z.re, z.im, omega.re, omega.im
        );
        println!(
            "  Verification: ω + ln(ω) = {:.6}{:+.6}i (should be {:.2}{:+.2}i)",
            check.re, check.im, z.re, z.im
        );
        println!();
    }

    // Relationship with Lambert W function
    println!("\nSpecial Properties of the Wright Omega Function:");
    println!("---------------------------------------------");
    let z = Complex64::new(0.5, 3.0);
    let omega = wright_omega(z, 1e-8);
    println!("For z = {:.2}{:+.2}i:", z.re, z.im);
    println!("ω(z) = {:.6}{:+.6}i", omega.re, omega.im);
    println!(
        "Verification: ω + ln(ω) = {:.6}{:+.6}i",
        (omega + omega.ln()).re,
        (omega + omega.ln()).im
    );

    // Special values
    println!("\nSpecial Values:");
    println!("--------------");
    println!("ω(-∞) = {}", wright_omega_real(f64::NEG_INFINITY, 1e-8));
    println!("ω(+∞) = {}", wright_omega_real(f64::INFINITY, 1e-8));

    // Numerical properties
    println!("\nNumerical Properties:");
    println!("--------------------");
    let large_positive = 1e21;
    let large_negative = -100.0;
    println!("For large positive x = {:.1e}:", large_positive);
    println!("ω(x) ≈ {:.6e}", wright_omega_real(large_positive, 1e-8));
    println!("For large negative x = {:.1}:", large_negative);
    println!("ω(x) ≈ {:.6e}", wright_omega_real(large_negative, 1e-8));
    println!(
        "  (Should be approximately e^({:.1}) = {:.6e})",
        large_negative,
        (-large_negative).exp()
    );
}
