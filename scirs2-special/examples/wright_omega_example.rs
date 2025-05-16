use num_complex::Complex64;
use scirs2_special::{wright_omega, wright_omega_real};

fn main() {
    println!("Wright Omega Function Example\n");

    // Real values
    println!("Wright Omega Function for Real Values:");
    println!("-------------------------------------");
    let real_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];
    for &x in &real_values {
        let omega = wright_omega_real(x, 1e-8).unwrap();
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
        let omega = wright_omega(z, 1e-8).unwrap();
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
    println!("\nRelationship with Lambert W Function:");
    println!("----------------------------------");
    let z = Complex64::new(0.5, 3.0);
    let omega = wright_omega(z, 1e-8).unwrap();
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
    println!(
        "ω(-∞) = {}",
        wright_omega_real(f64::NEG_INFINITY, 1e-8).unwrap()
    );
    println!(
        "ω(+∞) = {}",
        wright_omega_real(f64::INFINITY, 1e-8).unwrap()
    );

    // Numerical properties
    println!("\nNumerical Properties:");
    println!("--------------------");
    let large_positive = 1e21;
    let large_negative = -100.0;
    println!("For large positive x = {:.1e}:", large_positive);
    println!(
        "ω(x) ≈ {:.6e}",
        wright_omega_real(large_positive, 1e-8).unwrap()
    );
    println!("For large negative x = {:.1}:", large_negative);
    println!(
        "ω(x) ≈ {:.6e}",
        wright_omega_real(large_negative, 1e-8).unwrap()
    );
    println!(
        "  (Should be approximately e^({:.1}) = {:.6e})",
        large_negative,
        (-large_negative).exp()
    );
}
