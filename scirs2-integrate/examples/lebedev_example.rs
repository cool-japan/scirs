//! Example demonstrating the use of Lebedev quadrature for spherical integration
//!
//! This example shows various use cases for Lebedev quadrature:
//! - Basic integration over a unit sphere
//! - Integration of spherical harmonics
//! - Computation of moments of distributions on the sphere
//! - Application to physics problems (electrostatic potential)

use scirs2_integrate::lebedev::{lebedev_integrate, lebedev_rule, LebedevOrder};
use std::f64::consts::PI;

fn main() {
    println!("Lebedev Quadrature Examples");
    println!("==========================\n");

    // Example 1: Basic Lebedev rule information
    println!("Example 1: Basic Lebedev Rule Information\n");
    println!("Available Lebedev quadrature orders:");

    let orders = [
        LebedevOrder::Order6,
        LebedevOrder::Order14,
        LebedevOrder::Order26,
        LebedevOrder::Order38,
        LebedevOrder::Order50,
        LebedevOrder::Order74,
        LebedevOrder::Order86,
        LebedevOrder::Order110,
    ];

    for &order in &orders {
        print!("  {:?}: {} points", order, order.num_points());

        // Try to generate the rule to see if it's implemented
        match lebedev_rule::<f64>(order) {
            Ok(rule) => println!(" (implemented, degree {})", rule.degree),
            Err(_) => println!(" (not yet implemented)"),
        }
    }

    // Example 2: Integrating functions over the unit sphere
    println!("\nExample 2: Integrating Functions Over the Unit Sphere\n");

    // Test with constant function f(x,y,z) = 1
    // The integral over the unit sphere should be 4π (the surface area)
    let constant_result: f64 = lebedev_integrate(|_, _, _| 1.0, LebedevOrder::Order14).unwrap();
    println!("Integrating f(x,y,z) = 1:");
    println!("  Result: {:.10}", constant_result);
    println!("  Expected: {:.10} (4π)", 4.0 * PI);
    println!("  Error: {:.10e}", (constant_result - 4.0 * PI).abs());

    // Test with a function dependent on coordinates
    // Here we use f(x,y,z) = x^2, which should integrate to 4π/3
    let x2_result: f64 = lebedev_integrate(|x, _, _| x * x, LebedevOrder::Order14).unwrap();
    println!("\nIntegrating f(x,y,z) = x²:");
    println!("  Result: {:.10}", x2_result);
    println!("  Expected: {:.10} (4π/3)", 4.0 * PI / 3.0);
    println!("  Error: {:.10e}", (x2_result - 4.0 * PI / 3.0).abs());

    // Test with a more complex function
    // f(x,y,z) = x^2 * y^2 * z^2, which should integrate to 4π/15
    let xyz_result: f64 =
        lebedev_integrate(|x, y, z| x * x * y * y * z * z, LebedevOrder::Order26).unwrap();
    println!("\nIntegrating f(x,y,z) = x² * y² * z²:");
    println!("  Result: {:.10}", xyz_result);
    println!("  Expected: {:.10} (4π/15)", 4.0 * PI / 15.0);
    println!("  Error: {:.10e}", (xyz_result - 4.0 * PI / 15.0).abs());

    // Example 3: Spherical harmonics integration
    println!("\nExample 3: Spherical Harmonics Integration\n");

    // Spherical harmonics should integrate to zero over the sphere
    // due to orthogonality conditions

    // Y₁₀ ∝ z
    let y10_result: f64 = lebedev_integrate(|_, _, z| z, LebedevOrder::Order14).unwrap();
    println!("Integrating Y₁₀ ∝ z:");
    println!("  Result: {:.10e}", y10_result);
    println!("  Expected: 0 (by orthogonality)");

    // Y₂₀ ∝ (3z² - 1)
    let y20_result: f64 =
        lebedev_integrate(|_, _, z| 3.0 * z * z - 1.0, LebedevOrder::Order14).unwrap();
    println!("\nIntegrating Y₂₀ ∝ (3z² - 1):");
    println!("  Result: {:.10e}", y20_result);
    println!("  Expected: 0 (by orthogonality)");

    // Y₂₂ ∝ (x² - y²)
    let y22_result: f64 =
        lebedev_integrate(|x, y, _| x * x - y * y, LebedevOrder::Order14).unwrap();
    println!("\nIntegrating Y₂₂ ∝ (x² - y²):");
    println!("  Result: {:.10e}", y22_result);
    println!("  Expected: 0 (by orthogonality)");

    // Example 4: Comparison of different Lebedev orders
    println!("\nExample 4: Comparing Different Lebedev Orders\n");

    // Define a function with known integral (x^4 + y^4 + z^4 = 3/5 * (x² + y² + z²)² on the sphere)
    // The integral of (x² + y² + z²)² = 1² = 1 for points on the unit sphere
    // So the expected result is 3/5 * 4π
    let test_func = |x: f64, y: f64, z: f64| x.powi(4) + y.powi(4) + z.powi(4);
    let expected = 3.0 / 5.0 * 4.0 * PI;

    println!("Integrating f(x,y,z) = x⁴ + y⁴ + z⁴:");
    println!("Expected result: {:.10} (3/5 * 4π)", expected);

    for &order in &[
        LebedevOrder::Order6,
        LebedevOrder::Order14,
        LebedevOrder::Order26,
        LebedevOrder::Order38,
        LebedevOrder::Order50,
    ] {
        // Skip orders that aren't implemented
        if let Ok(result) = lebedev_integrate::<f64, _>(test_func, order) {
            let error = (result - expected).abs();
            println!(
                "  {:?} ({} points): {:.10} (error: {:.10e})",
                order,
                order.num_points(),
                result,
                error
            );
        }
    }

    // Example 5: Physical application - Electrostatic potential
    println!("\nExample 5: Physical Application - Electrostatic Potential\n");

    println!("Calculating the electrostatic potential from a spherical charge distribution");

    // Simulate a point outside a uniformly charged sphere
    // The potential outside the sphere is equivalent to a point charge at the center
    // We'll place our observation point at (0, 0, 2) - twice the sphere radius away

    let observation_point = [0.0, 0.0, 2.0];

    // Define our charge distribution - uniform over the sphere
    let charge_density = 1.0 / (4.0 * PI); // Normalized to total charge of 1

    // Calculate potential using Lebedev quadrature
    // V(x') = ∫[ρ(x) / |x - x'|] dΩ
    let potential: f64 = lebedev_integrate(
        |x, y, z| {
            let dx: f64 = x - observation_point[0];
            let dy: f64 = y - observation_point[1];
            let dz: f64 = z - observation_point[2];
            let distance: f64 = (dx * dx + dy * dy + dz * dz).sqrt();

            charge_density / distance
        },
        LebedevOrder::Order26,
    )
    .unwrap();

    // For a unit sphere with unit total charge, the expected potential at distance 2
    // is simply 1/2 (from Coulomb's law: V = q/r)
    println!(
        "  Observation point: ({}, {}, {})",
        observation_point[0], observation_point[1], observation_point[2]
    );
    println!("  Calculated potential: {:.10}", potential);
    println!("  Expected potential: {:.10} (1/2)", 0.5);
    println!(
        "  Relative error: {:.10e}",
        ((potential - 0.5).abs() / 0.5) as f64
    );
}
