use scirs2_special::{coulomb_f, coulomb_g, coulomb_h_minus, coulomb_h_plus, coulomb_phase_shift};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Coulomb Wave Functions Example");
    println!("=============================\n");

    // Compute Coulomb phase shift
    println!("Coulomb Phase Shift σ_L(η):");
    println!("-------------------------");

    let l_values = [0.0, 1.0, 2.0, 3.0];
    let eta_values = [0.0, 0.1, 0.5, 1.0, 2.0];

    for &l in &l_values {
        println!("L = {}", l);
        for &eta in &eta_values {
            match coulomb_phase_shift(l, eta) {
                Ok(sigma) => println!("  σ_{{{}}}({}) = {:.6}", l, eta, sigma),
                Err(e) => println!("  σ_{{{}}}({}) = Not implemented: {}", l, eta, e),
            }
        }
        println!();
    }

    // Compute regular Coulomb wave function F_L(η,ρ)
    println!("Regular Coulomb Wave Function F_L(η,ρ):");
    println!("------------------------------------");

    let l = 0.0;
    let eta = 0.0;
    let rho_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];

    // For η=0, F_L(0,ρ) = ρ j_L(ρ) where j_L is the spherical Bessel function
    println!("For L = {}, η = {} (free-particle case):", l, eta);
    for &rho in &rho_values {
        match coulomb_f(l, eta, rho) {
            Ok(f) => println!("  F_{{{}}}({}, {}) = {:.6}", l, eta, rho, f),
            Err(e) => println!("  F_{{{}}}({}, {}) = Not implemented: {}", l, eta, rho, e),
        }
    }
    println!();

    // Compute with Coulomb potential
    let eta = 1.0;
    println!("For L = {}, η = {} (with Coulomb potential):", l, eta);
    for &rho in &rho_values {
        match coulomb_f(l, eta, rho) {
            Ok(f) => println!("  F_{{{}}}({}, {}) = {:.6}", l, eta, rho, f),
            Err(e) => println!("  F_{{{}}}({}, {}) = Not implemented: {}", l, eta, rho, e),
        }
    }
    println!();

    // Compute irregular Coulomb wave function G_L(η,ρ)
    println!("Irregular Coulomb Wave Function G_L(η,ρ):");
    println!("--------------------------------------");

    let l = 0.0;
    let eta = 0.0;

    // For η=0, G_L(0,ρ) = -ρ y_L(ρ) where y_L is the spherical Neumann function
    println!("For L = {}, η = {} (free-particle case):", l, eta);
    for &rho in &rho_values {
        match coulomb_g(l, eta, rho) {
            Ok(g) => println!("  G_{{{}}}({}, {}) = {:.6}", l, eta, rho, g),
            Err(e) => println!("  G_{{{}}}({}, {}) = Not implemented: {}", l, eta, rho, e),
        }
    }
    println!();

    // Compute complex Coulomb wave functions H⁺_L(η,ρ) and H⁻_L(η,ρ)
    println!("Complex Coulomb Wave Functions:");
    println!("----------------------------");

    println!("Outgoing Coulomb Wave Function H⁺_L(η,ρ):");
    for &rho in &rho_values {
        match coulomb_h_plus(l, eta, rho) {
            Ok(h) => println!(
                "  H⁺_{{{}}}({}, {}) = {:.6} + {:.6}i",
                l, eta, rho, h.re, h.im
            ),
            Err(e) => println!("  H⁺_{{{}}}({}, {}) = Not implemented: {}", l, eta, rho, e),
        }
    }
    println!();

    println!("Incoming Coulomb Wave Function H⁻_L(η,ρ):");
    for &rho in &rho_values {
        match coulomb_h_minus(l, eta, rho) {
            Ok(h) => println!(
                "  H⁻_{{{}}}({}, {}) = {:.6} + {:.6}i",
                l, eta, rho, h.re, h.im
            ),
            Err(e) => println!("  H⁻_{{{}}}({}, {}) = Not implemented: {}", l, eta, rho, e),
        }
    }

    Ok(())
}
