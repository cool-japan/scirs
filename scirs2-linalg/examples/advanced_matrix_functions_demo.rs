//! Advanced Matrix Functions Demonstration
//!
//! This example demonstrates the new advanced matrix functions added to scirs2-linalg,
//! including spectral analysis, polar decomposition, geometric means, and regularization.

use scirs2_core::ndarray::array;
use scirs2_linalg::matrix_functions::{
    geometric_mean_spd, nuclear_norm, polar_decomposition, spectral_condition_number,
    spectral_radius, tikhonov_regularization,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Matrix Functions Demonstration");
    println!("{}", "=".repeat(50));

    // Example 1: Spectral Analysis
    println!("\n📊 Example 1: Spectral Analysis");
    let matrix_a = array![[3.0_f64, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 1.0]];

    let spectral_rad = spectral_radius(&matrix_a.view(), None)?;
    let condition_num = spectral_condition_number(&matrix_a.view(), None)?;

    println!("Matrix A:");
    println!("{:8.3}", matrix_a);
    println!("Spectral radius ρ(A): {:.6}", spectral_rad);
    println!("Condition number κ₂(A): {:.2e}", condition_num);

    // Example 2: Polar Decomposition
    println!("\n🔄 Example 2: Polar Decomposition A = UH");
    let matrix_b = array![[2.0_f64, 1.0], [1.0, 2.0]];
    let (u, h) = polar_decomposition(&matrix_b.view(), "right")?;

    println!("Matrix B:");
    println!("{:8.3}", matrix_b);
    println!("Orthogonal part U:");
    println!("{:8.6}", u);
    println!("Positive definite part H:");
    println!("{:8.6}", h);

    // Verify decomposition: A = U * H
    let reconstructed = u.dot(&h);
    let error = (&reconstructed - &matrix_b)
        .iter()
        .map(|&x| x.abs())
        .fold(0.0, f64::max);
    println!("Reconstruction error: {:.2e}", error);

    // Example 3: Geometric Mean of SPD Matrices
    println!("\n📐 Example 3: Geometric Mean of Positive Definite Matrices");
    let spd_a = array![[4.0_f64, 0.0], [0.0, 1.0]];
    let spd_b = array![[1.0_f64, 0.0], [0.0, 4.0]];

    let geom_mean = geometric_mean_spd(&spd_a.view(), &spd_b.view(), 0.5)?;

    println!("SPD Matrix A:");
    println!("{:8.3}", spd_a);
    println!("SPD Matrix B:");
    println!("{:8.3}", spd_b);
    println!("Geometric Mean G(A,B):");
    println!("{:8.6}", geom_mean);

    // For diagonal matrices, the geometric mean should be the element-wise geometric mean
    println!("Expected (for diagonal case): [[2.0, 0.0], [0.0, 2.0]]");

    // Example 4: Nuclear Norm
    println!("\n🎯 Example 4: Nuclear Norm (Trace Norm)");
    let matrix_c = array![[3.0_f64, 1.0], [1.0, 2.0]];
    let nuclear = nuclear_norm(&matrix_c.view(), None)?;

    println!("Matrix C:");
    println!("{:8.3}", matrix_c);
    println!("Nuclear norm ||C||*: {:.6}", nuclear);

    // Example 5: Tikhonov Regularization
    println!("\n🛡️  Example 5: Tikhonov Regularization");
    let ill_conditioned = array![[1.0_f64, 0.0], [0.0, 1e-12]];

    // Check condition number before regularization
    let cond_before = spectral_condition_number(&ill_conditioned.view(), None)?;
    println!("Ill-conditioned matrix:");
    println!("{:12.2e}", ill_conditioned);
    println!("Condition number before: {:.2e}", cond_before);

    // Apply regularization
    let regularized = tikhonov_regularization(&ill_conditioned.view(), 1e-6, false)?;
    let cond_after = spectral_condition_number(&regularized.view(), None)?;

    println!("After Tikhonov regularization (λ = 1e-6):");
    println!("{:12.6e}", regularized);
    println!("Condition number after: {:.2e}", cond_after);
    println!("Improvement factor: {:.2e}", cond_before / cond_after);

    // Example 6: Adaptive Regularization
    println!("\n🧠 Example 6: Adaptive Regularization");
    let adaptive_reg = tikhonov_regularization(&ill_conditioned.view(), 1e-8, true)?;
    let cond_adaptive = spectral_condition_number(&adaptive_reg.view(), None)?;

    println!("Adaptive regularization (base λ = 1e-8):");
    println!("{:12.6e}", adaptive_reg);
    println!("Condition number: {:.2e}", cond_adaptive);

    println!("\n✨ Advanced matrix functions demonstration completed!");
    println!("These functions are useful for:");
    println!("  • Spectral analysis and convergence studies");
    println!("  • Matrix manifold optimization (Riemannian geometry)");
    println!("  • Regularization of ill-conditioned problems");
    println!("  • Low-rank matrix optimization and compressed sensing");
    println!("  • Signal processing and machine learning applications");

    Ok(())
}
