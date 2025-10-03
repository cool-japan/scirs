//! Ellipsoidal Harmonics Functions Demo
//!
//! This example demonstrates the usage of ellipsoidal harmonic functions
//! which are used in geodesy and geophysics for modeling gravitational fields
//! of rotating bodies like planets.

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use scirs2_special::{
    ellip_harm, ellip_harm_2, ellip_harm_array, ellip_harm_coefficients, ellip_harm_complex,
    ellip_normal,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌍 Ellipsoidal Harmonics Functions Demo");
    println!("==========================================");

    // Example 1: Basic ellipsoidal harmonic functions
    println!("\n📊 Example 1: Basic Ellipsoidal Harmonics");
    println!("-----------------------------------------");

    let h2 = 0.1; // Parameter related to ellipsoid's eccentricity
    let k2 = 0.05; // Parameter related to coordinate system
    let n = 2; // Degree parameter
    let p = 1; // Order parameter
    let s = 1.5; // Coordinate parameter

    let e_harm = ellip_harm(h2, k2, n, p, s)?;
    println!("E_{}^{}({:.1}; {:.1}) = {:.6}", n, p, s, h2, e_harm);

    // Example 2: Second-kind ellipsoidal harmonics
    println!("\n📊 Example 2: Second-Kind Ellipsoidal Harmonics");
    println!("-----------------------------------------------");

    let f_harm = ellip_harm_2(h2, k2, n, p, s)?;
    println!("F_{}^{}({:.1}; {:.1}) = {:.6}", n, p, s, h2, f_harm);

    // Example 3: Normalization constants
    println!("\n📊 Example 3: Normalization Constants");
    println!("------------------------------------");

    let norm = ellip_normal(h2, k2, n, p)?;
    println!("N_{}^{}({:.2}, {:.2}) = {:.6}", n, p, h2, k2, norm);

    // Example 4: Array operations
    println!("\n📊 Example 4: Array Operations");
    println!("-----------------------------");

    let s_values = Array1::linspace(1.0, 3.0, 5);
    let harmonics = ellip_harm_array(h2, k2, n, p, &s_values.view())?;

    println!("Coordinate values: {:?}", s_values.to_vec());
    println!(
        "Harmonic values:   {:?}",
        harmonics
            .iter()
            .map(|&x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    // Example 5: Expansion coefficients
    println!("\n📊 Example 5: Expansion Coefficients");
    println!("-----------------------------------");

    let max_degree = 3;
    let max_order = 2;
    let coeffs = ellip_harm_coefficients(h2, k2, max_degree, max_order)?;

    println!(
        "Expansion coefficients ({}x{} matrix):",
        max_degree + 1,
        max_order + 1
    );
    for (i, row) in coeffs.axis_iter(scirs2_core::ndarray::Axis(0)).enumerate() {
        print!("n={}: ", i);
        for &coeff in row.iter() {
            print!("{:8.4} ", coeff);
        }
        println!();
    }

    // Example 6: Complex ellipsoidal harmonics
    println!("\n📊 Example 6: Complex Ellipsoidal Harmonics");
    println!("------------------------------------------");

    let h2_complex = Complex64::new(0.1, 0.02);
    let k2_complex = Complex64::new(0.05, 0.01);
    let z_complex = Complex64::new(1.5, 0.5);

    let complex_harm = ellip_harm_complex(h2_complex, k2_complex, n, p, z_complex)?;
    println!(
        "E_{}^{}({:.1}+{:.1}i; {:.2}+{:.2}i) = {:.4}+{:.4}i",
        n,
        p,
        z_complex.re,
        z_complex.im,
        h2_complex.re,
        h2_complex.im,
        complex_harm.re,
        complex_harm.im
    );

    // Example 7: Comparison with spherical harmonics limit
    println!("\n📊 Example 7: Spherical Harmonics Limit");
    println!("--------------------------------------");

    // When h² = k² = 0, ellipsoidal harmonics should reduce to spherical harmonics
    let h2_spherical = 0.0;
    let k2_spherical = 0.0;
    let s_spherical = 1.0; // This corresponds to cos(θ) = 1, or θ = 0

    let ellip_spherical = ellip_harm(h2_spherical, k2_spherical, n, 0, s_spherical)?;
    println!(
        "Ellipsoidal harmonic (spherical limit): E_{}^0({:.1}) = {:.6}",
        n, s_spherical, ellip_spherical
    );

    // Example 8: Physical interpretation
    println!("\n🌍 Example 8: Physical Interpretation");
    println!("------------------------------------");

    println!("Ellipsoidal harmonics are used in geodesy to model:");
    println!("• Earth's gravitational field");
    println!("• Geoid determination");
    println!("• Satellite orbit perturbations");
    println!("• Tidal effects");

    println!("\nParameters:");
    println!("• h² = {:.2} - Related to ellipsoid's eccentricity", h2);
    println!("• k² = {:.2} - Related to coordinate system", k2);
    println!("• n = {} - Degree (controls spatial frequency)", n);
    println!("• p = {} - Order (controls azimuthal dependence)", p);

    // Example 9: Geodetic applications
    println!("\n🛰️ Example 9: Geodetic Applications");
    println!("-----------------------------------");

    println!("Computing harmonics for different degrees and orders:");
    for degree in 0..=3 {
        for order in 0..=degree {
            let harm_val = ellip_harm(0.1, 0.05, degree, order, 2.0)?;
            println!("E_{}^{} = {:8.4}", degree, order, harm_val);
        }
    }

    // Example 10: Performance considerations
    println!("\n⚡ Example 10: Performance Notes");
    println!("------------------------------");

    println!("For large-scale computations:");
    println!("• Use array operations (ellip_harm_array) for multiple points");
    println!("• Precompute coefficients for series expansions");
    println!("• Consider numerical stability for extreme parameter values");
    println!("• Complex functions available for analytical continuations");

    Ok(())
}
