use scirs2_special::{hurwitz_zeta, zeta, zetac};
use std::f64::consts::PI;

#[allow(unused_variables)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Zeta Functions Example");
    println!("=====================\n");

    // Example 1: Riemann zeta function for positive even integers
    println!("Riemann Zeta Function - Positive Even Integers:");
    println!("----------------------------------------------");

    println!("ζ(2) = {:.15}", zeta(2.0)?);
    println!("π²/6 = {:.15}", PI * PI / 6.0);
    println!();

    println!("ζ(4) = {:.15}", zeta(4.0)?);
    println!("π⁴/90 = {:.15}", PI.powi(4) / 90.0);
    println!();

    println!("ζ(6) = {:.15}", zeta(6.0)?);
    println!("π⁶/945 = {:.15}", PI.powi(6) / 945.0);
    println!();

    // Example 2: Riemann zeta function for negative integers
    println!("Riemann Zeta Function - Negative Integers:");
    println!("----------------------------------------");

    println!("ζ(0) = {:.15}", zeta(0.0)?);
    println!("Expected: -1/2 = -0.5");
    println!();

    println!("ζ(-1) = {:.15}", zeta(-1.0)?);
    println!("Expected: -1/12 ≈ -0.083333");
    println!();

    println!("ζ(-2) = {:.15}", zeta(-2.0)?);
    println!("Expected: 0");
    println!();

    println!("ζ(-3) = {:.15}", zeta(-3.0)?);
    println!("Expected: 1/120 ≈ 0.008333");
    println!();

    // Example 3: Riemann zeta function minus 1 (zetac)
    println!("Riemann Zeta Function Minus 1 (zetac):");
    println!("------------------------------------");

    println!("ζ(2) - 1 = {:.15}", zetac(2.0)?);
    println!("π²/6 - 1 = {:.15}", PI * PI / 6.0 - 1.0);
    println!();

    println!("ζ(4) - 1 = {:.15}", zetac(4.0)?);
    println!("π⁴/90 - 1 = {:.15}", PI.powi(4) / 90.0 - 1.0);
    println!();

    // For large s, ζ(s) approaches 1, so ζ(s) - 1 approaches 0
    let s_large = 50.0;
    println!("ζ({}) - 1 = {:.15e}", s_large, zetac(s_large)?);
    println!("Expected: Very close to 0");
    println!();

    // Example 4: Hurwitz zeta function
    println!("Hurwitz Zeta Function:");
    println!("---------------------");

    println!("ζ(2, 1) = {:.15}", hurwitz_zeta(2.0, 1.0)?);
    println!("Expected: ζ(2) = π²/6 ≈ 1.644934...");
    println!();

    println!("ζ(2, 2) = {:.15}", hurwitz_zeta(2.0, 2.0)?);
    println!("Expected: ζ(2) - 1 = π²/6 - 1 ≈ 0.644934...");
    println!();

    println!("ζ(2, 0.5) = {:.15}", hurwitz_zeta(2.0, 0.5)?);
    println!("Expected: 2π²/3 ≈ 6.579736...");
    println!();

    // Example 5: Relationship between Hurwitz zeta and polygamma functions
    println!("Relationship to Polygamma Functions:");
    println!("----------------------------------");
    println!("The polygamma function ψ^(m)(x) is related to the Hurwitz zeta by:");
    println!("ψ^(m)(x) = (-1)^(m+1) * m! * ζ(m+1, x)");
    println!();

    // Calculate a few values of polygamma functions using Hurwitz zeta
    let m = 2; // Third derivative of digamma function (m=0 is digamma)
    let x = 1.5;
    let factorial = match m {
        0 => 1.0,
        1 => 1.0,
        2 => 2.0,
        3 => 6.0,
        4 => 24.0,
        _ => (1..=m).product::<i32>() as f64,
    };
    let sign = if m % 2 == 0 { 1.0 } else { -1.0 };

    println!(
        "ψ^({})({}) = {:.15}",
        m,
        x,
        sign * factorial * hurwitz_zeta(m as f64 + 1.0, x)?
    );
    println!();

    // Example 6: Convergence of zeta function
    println!("Convergence of Zeta Function as s increases:");
    println!("------------------------------------------");
    println!("As s approaches infinity, ζ(s) approaches 1");
    println!();

    for s in [5.0, 10.0, 20.0, 50.0, 100.0] {
        println!("ζ({}) = {:.15}", s, zeta(s)?);
    }
    println!();

    // Example 7: Critical strip (0 < s < 1)
    println!("Zeta Function in the Critical Strip (0 < s < 1):");
    println!("---------------------------------------------");
    println!("The Riemann Hypothesis concerns zeros in this region");
    println!();

    for s in [0.25, 0.5, 0.75] {
        println!("ζ({}) = {:.15}", s, zeta(s)?);
    }

    Ok(())
}
