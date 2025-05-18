use num_complex::Complex64;
use scirs2_special::{lambert_w, lambert_w_real};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Lambert W Function Example");
    println!("=========================\n");

    // Principal branch examples
    println!("Principal Branch (k=0):");
    let z = 1.0;
    let w = lambert_w_real(z, 1e-12)?;
    println!("W_0({}) = {}", z, w);
    println!("Verification: {} * e^{} = {}", w, w, w * w.exp());

    // Negative input
    let z_neg = -0.1;
    match lambert_w_real(z_neg, 1e-12) {
        Ok(w) => println!("W_0({}) = {}", z_neg, w),
        Err(_) => {
            // Use the complex version
            let w = lambert_w(Complex64::new(z_neg, 0.0), 0, 1e-12)?;
            println!("W_0({}) = {} + {}i", z_neg, w.re, w.im);

            let w_exp_w = w * w.exp();
            println!(
                "Verification: ({} + {}i) * e^({} + {}i) = {} + {}i",
                w.re, w.im, w.re, w.im, w_exp_w.re, w_exp_w.im
            );
        }
    }

    println!("\nNon-principal Branches:");

    // Branch k = -1
    let z = 0.1;
    let w_minus1 = lambert_w(Complex64::new(z, 0.0), -1, 1e-12)?;
    println!("W_-1({}) = {} + {}i", z, w_minus1.re, w_minus1.im);

    let w_exp_w = w_minus1 * w_minus1.exp();
    println!(
        "Verification: ({} + {}i) * e^({} + {}i) = {} + {}i",
        w_minus1.re, w_minus1.im, w_minus1.re, w_minus1.im, w_exp_w.re, w_exp_w.im
    );

    // Branch k = 1
    let w_1 = lambert_w(Complex64::new(z, 0.0), 1, 1e-12)?;
    println!("W_1({}) = {} + {}i", z, w_1.re, w_1.im);

    let w_exp_w = w_1 * w_1.exp();
    println!(
        "Verification: ({} + {}i) * e^({} + {}i) = {} + {}i",
        w_1.re, w_1.im, w_1.re, w_1.im, w_exp_w.re, w_exp_w.im
    );

    // Solution to equation x = a + b * exp(c * x)
    println!("\nSolving Equation x = a + b * exp(c * x):");
    let a = 3.0;
    let b = 2.0;
    let c = -0.5;

    // Solution is x = a - W(-b*c*e^(a*c))/c
    let z = -b * c * f64::exp(a * c);
    let w = lambert_w(Complex64::new(z, 0.0), 0, 1e-12)?;
    let x = a - w.re / c;

    println!("For a = {}, b = {}, c = {}", a, b, c);
    println!("Solution: x = {}", x);

    // Verify
    let right_side = a + b * (c * x).exp();
    println!("Verification: {} = {}", x, right_side);

    // Infinite power tower example
    println!("\nInfinite Power Tower z^(z^(z^...)):");
    let z: f64 = 0.5;
    // The value of z^(z^(z^...)) is -W(-ln(z))/ln(z)
    let w = lambert_w(Complex64::new(-z.ln(), 0.0), 0, 1e-12)?;
    let tower_value = -w / z.ln();

    println!("For z = {}", z);
    println!("z^(z^(z^...)) = {}", tower_value.re);

    // Verify by evaluating tower to a large depth
    fn evaluate_tower(z: f64, depth: i32) -> f64 {
        if depth == 0 {
            return z;
        }
        z.powf(evaluate_tower(z, depth - 1))
    }

    let approx_tower = evaluate_tower(z, 20);
    println!("Approximation (depth 20): {}", approx_tower);
    println!("Difference: {}", (tower_value.re - approx_tower).abs());

    Ok(())
}
