use ndarray::s;
// Validation utilities for 2D Discrete Wavelet Transform
//
// This module provides comprehensive validation for 2D DWT implementations,
// including perfect reconstruction, energy conservation, and comparison
// with reference implementations.

use crate::dwt::Wavelet;
use crate::dwt2d::{dwt2d_decompose, dwt2d_reconstruct};
use crate::error::{SignalError, SignalResult};
use ndarray::Array2;
use rand::Rng;
use statrs::statistics::Statistics;
use std::f64;
use std::time::Instant;

#[allow(unused_imports)]
use crate::dwt2d_enhanced::{
    enhanced_dwt2d_decompose, wavedec2_enhanced, BoundaryMode, Dwt2dConfig, EnhancedDwt2dResult,
};
/// 2D DWT validation result
#[derive(Debug, Clone)]
pub struct Dwt2dValidationResult {
    /// Perfect reconstruction error
    pub reconstruction_error: ReconstructionMetrics,
    /// Energy conservation metrics
    pub energy_conservation: EnergyMetrics,
    /// Boundary handling validation
    pub boundary_validation: BoundaryValidation,
    /// Performance comparison
    pub performance: PerformanceComparison,
    /// Overall validation score
    pub overall_score: f64,
    /// Issues found
    pub issues: Vec<String>,
}

/// Reconstruction error metrics
#[derive(Debug, Clone)]
pub struct ReconstructionMetrics {
    /// Maximum absolute error
    pub max_error: f64,
    /// Mean absolute error
    pub mean_error: f64,
    /// Root mean square error
    pub rmse: f64,
    /// Peak signal-to-noise ratio (dB)
    pub psnr: f64,
    /// Structural similarity index
    pub ssim: f64,
}

/// Energy conservation metrics
#[derive(Debug, Clone)]
pub struct EnergyMetrics {
    /// Input energy
    pub input_energy: f64,
    /// Output energy (sum of all subbands)
    pub output_energy: f64,
    /// Energy ratio (should be 1.0)
    pub energy_ratio: f64,
    /// Energy distribution across subbands
    pub subband_energies: SubbandEnergies,
}

/// Energy distribution in subbands
#[derive(Debug, Clone)]
pub struct SubbandEnergies {
    /// Approximation energy percentage
    pub approx_percent: f64,
    /// Horizontal detail energy percentage
    pub detail_h_percent: f64,
    /// Vertical detail energy percentage
    pub detail_v_percent: f64,
    /// Diagonal detail energy percentage
    pub detail_d_percent: f64,
}

/// Boundary handling validation
#[derive(Debug, Clone)]
pub struct BoundaryValidation {
    /// Edge artifact measure
    pub edge_artifacts: f64,
    /// Boundary continuity score
    pub continuity_score: f64,
    /// Symmetric extension accuracy
    pub symmetry_accuracy: f64,
    /// All boundary modes passed
    pub all_modes_valid: bool,
}

/// Performance comparison
#[derive(Debug, Clone)]
pub struct PerformanceComparison {
    /// Standard implementation time (ms)
    pub standard_time_ms: f64,
    /// Enhanced implementation time (ms)
    pub enhanced_time_ms: f64,
    /// Speedup factor
    pub speedup: f64,
    /// Memory usage ratio
    pub memory_ratio: f64,
}

/// Validate 2D DWT implementation
#[allow(dead_code)]
pub fn validate_dwt2d(
    test_image: &Array2<f64>,
    wavelet: Wavelet,
    tolerance: f64,
) -> SignalResult<Dwt2dValidationResult> {
    let mut issues: Vec<String> = Vec::new();

    // Check input
    if test_image.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Test _image contains non-finite values".to_string(),
        ));
    }

    // Test perfect reconstruction
    let reconstruction_error = test_perfect_reconstruction(test_image, wavelet, tolerance)?;

    if reconstruction_error.max_error > tolerance {
        issues.push(format!(
            "Reconstruction error ({:.2e}) exceeds tolerance ({:.2e})",
            reconstruction_error.max_error, tolerance
        ));
    }

    // Test energy conservation
    let energy_conservation = test_energy_conservation(test_image, wavelet)?;

    if ((energy_conservation.energy_ratio - 1.0) as f64).abs() > tolerance * 10.0 {
        issues.push(format!(
            "Energy not conserved: ratio = {:.6}",
            energy_conservation.energy_ratio
        ));
    }

    // Test boundary handling
    let boundary_validation = test_boundary_handling(test_image, wavelet)?;

    if !boundary_validation.all_modes_valid {
        issues.push("Some boundary modes failed validation".to_string());
    }

    // Performance comparison
    let performance = compare_performance(test_image, wavelet)?;

    // Calculate overall score
    let mut score = 0.0;

    // Reconstruction quality (40%)
    score += 40.0 * (1.0 - reconstruction_error.mean_error.min(1.0));

    // Energy conservation (30%)
    score += 30.0
        * ((1.0 - (energy_conservation.energy_ratio - 1.0) as f64)
            .abs()
            .min(1.0));

    // Boundary handling (20%)
    score += 20.0 * boundary_validation.continuity_score;

    // Performance (10%)
    if performance.speedup > 1.0 {
        score += 10.0;
    } else {
        score += 10.0 * performance.speedup;
    }

    Ok(Dwt2dValidationResult {
        reconstruction_error,
        energy_conservation,
        boundary_validation,
        performance,
        overall_score: score,
        issues,
    })
}

/// Test perfect reconstruction
#[allow(dead_code)]
fn test_perfect_reconstruction(
    image: &Array2<f64>,
    wavelet: Wavelet,
    tolerance: f64,
) -> SignalResult<ReconstructionMetrics> {
    // Standard decomposition and reconstruction
    let decomposition = dwt2d_decompose(image, wavelet, None)?;
    let reconstructed = dwt2d_reconstruct(&decomposition, wavelet, None)?;

    // Compute errors
    let (rows, cols) = image.dim();
    let (rec_rows, rec_cols) = reconstructed.dim();

    // Handle size mismatch due to boundary effects
    let min_rows = rows.min(rec_rows);
    let min_cols = cols.min(rec_cols);

    let mut errors = Vec::with_capacity(min_rows * min_cols);
    let mut sum_sq_error = 0.0;

    for i in 0..min_rows {
        for j in 0..min_cols {
            let error = (image[[i, j]] - reconstructed[[i, j]]).abs();
            errors.push(error);
            sum_sq_error += error * error;
        }
    }

    let max_error = errors.iter().cloned().fold(0.0, f64::max);
    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
    let rmse = (sum_sq_error / errors.len() as f64).sqrt();

    // PSNR calculation
    let max_pixel_value = image.iter().cloned().fold(0.0, f64::max);
    let psnr = if rmse > 1e-10 {
        20.0 * max_pixel_value.log10() - 10.0 * rmse.log10()
    } else {
        f64::INFINITY
    };

    // SSIM calculation (simplified)
    let ssim = compute_ssim(image, &reconstructed, min_rows, min_cols)?;

    Ok(ReconstructionMetrics {
        max_error,
        mean_error,
        rmse,
        psnr,
        ssim,
    })
}

/// Test energy conservation
#[allow(dead_code)]
fn test_energy_conservation(image: &Array2<f64>, wavelet: Wavelet) -> SignalResult<EnergyMetrics> {
    // Input energy
    let input_energy = compute_energy(_image);

    // Decompose
    let decomposition = dwt2d_decompose(_image, wavelet, None)?;

    // Subband energies
    let approx_energy = compute_energy(&decomposition.approx);
    let h_energy = compute_energy(&decomposition.detail_h);
    let v_energy = compute_energy(&decomposition.detail_v);
    let d_energy = compute_energy(&decomposition.detail_d);

    let output_energy = approx_energy + h_energy + v_energy + d_energy;
    let energy_ratio = output_energy / input_energy;

    // Energy distribution
    let subband_energies = SubbandEnergies {
        approx_percent: 100.0 * approx_energy / output_energy,
        detail_h_percent: 100.0 * h_energy / output_energy,
        detail_v_percent: 100.0 * v_energy / output_energy,
        detail_d_percent: 100.0 * d_energy / output_energy,
    };

    Ok(EnergyMetrics {
        input_energy,
        output_energy,
        energy_ratio,
        subband_energies,
    })
}

/// Test boundary handling
#[allow(dead_code)]
fn test_boundary_handling(
    image: &Array2<f64>,
    wavelet: Wavelet,
) -> SignalResult<BoundaryValidation> {
    let config_default = Dwt2dConfig::default();
    let mut all_valid = true;
    let mut total_artifacts = 0.0;
    let mut total_continuity = 0.0;
    let mut symmetry_scores = Vec::new();

    // Test different boundary modes
    let boundary_modes = [
        BoundaryMode::Zero,
        BoundaryMode::Symmetric,
        BoundaryMode::Periodic,
        BoundaryMode::Constant(image[[0, 0]]),
    ];

    for mode in &boundary_modes {
        let config = Dwt2dConfig {
            boundary_mode: *mode,
            ..config_default.clone()
        };

        let result = enhanced_dwt2d_decompose(image, wavelet, &config)?;

        // Measure edge artifacts
        let artifacts = measure_edge_artifacts(&result)?;
        total_artifacts += artifacts;

        // Measure boundary continuity
        let continuity = measure_boundary_continuity(&result)?;
        total_continuity += continuity;

        // For symmetric mode, check symmetry preservation
        if let BoundaryMode::Symmetric = mode {
            let symmetry = check_symmetry_preservation(image, &result)?;
            symmetry_scores.push(symmetry);
        }

        // Check if reconstruction is valid
        let dwt_result = crate::dwt2d::Dwt2dResult {
            approx: result.approx.clone(),
            detail_h: result.detail_h.clone(),
            detail_v: result.detail_v.clone(),
            detail_d: result.detail_d.clone(),
        };
        let reconstructed = dwt2d_reconstruct(&dwt_result, wavelet, Some("symmetric"))?;

        if !is_valid_reconstruction(&reconstructed) {
            all_valid = false;
        }
    }

    let n_modes = boundary_modes.len() as f64;
    let edge_artifacts = total_artifacts / n_modes;
    let continuity_score = total_continuity / n_modes;
    let symmetry_accuracy = if symmetry_scores.is_empty() {
        1.0
    } else {
        symmetry_scores.iter().sum::<f64>() / symmetry_scores.len() as f64
    };

    Ok(BoundaryValidation {
        edge_artifacts,
        continuity_score,
        symmetry_accuracy,
        all_modes_valid: all_valid,
    })
}

/// Compare performance between implementations
#[allow(dead_code)]
fn compare_performance(
    image: &Array2<f64>,
    wavelet: Wavelet,
) -> SignalResult<PerformanceComparison> {
    let n_runs = 10;

    // Time standard implementation
    let start = Instant::now();
    for _ in 0..n_runs {
        let _ = dwt2d_decompose(image, wavelet, None)?;
    }
    let standard_time = start.elapsed().as_micros() as f64 / (n_runs as f64 * 1000.0);

    // Time enhanced implementation
    let config = Dwt2dConfig::default();
    let start = Instant::now();
    for _ in 0..n_runs {
        let _ = enhanced_dwt2d_decompose(image, wavelet, &config)?;
    }
    let enhanced_time = start.elapsed().as_micros() as f64 / (n_runs as f64 * 1000.0);

    let speedup = standard_time / enhanced_time.max(0.001);

    // Memory usage estimate (simplified)
    let memory_ratio = 1.0; // Would need actual memory profiling

    Ok(PerformanceComparison {
        standard_time_ms: standard_time,
        enhanced_time_ms: enhanced_time,
        speedup,
        memory_ratio,
    })
}

/// Compute energy of 2D array
#[allow(dead_code)]
fn compute_energy(array: &Array2<f64>) -> f64 {
    array.iter().map(|&x| x * x).sum()
}

/// Compute structural similarity index (simplified)
#[allow(dead_code)]
fn compute_ssim(
    image1: &Array2<f64>,
    image2: &Array2<f64>,
    rows: usize,
    cols: usize,
) -> SignalResult<f64> {
    let c1 = 0.01_f64.powi(2);
    let c2 = 0.03_f64.powi(2);

    let mut sum_ssim = 0.0;
    let window_size = 8;
    let mut n_windows = 0;

    for i in (0..rows - window_size).step_by(4) {
        for j in (0..cols - window_size).step_by(4) {
            // Extract windows
            let window1 = image1.slice(s![i..i + window_size, j..j + window_size]);
            let window2 = image2.slice(s![i..i + window_size, j..j + window_size]);

            // Compute means
            let mu1 = window1.mean();
            let mu2 = window2.mean();

            // Compute variances and covariance
            let var1 = window1.iter().map(|&x| (x - mu1).powi(2)).sum::<f64>()
                / (window_size * window_size) as f64;
            let var2 = window2.iter().map(|&x| (x - mu2).powi(2)).sum::<f64>()
                / (window_size * window_size) as f64;

            let mut cov = 0.0;
            for wi in 0..window_size {
                for wj in 0..window_size {
                    cov += (window1[[wi, wj]] - mu1) * (window2[[wi, wj]] - mu2);
                }
            }
            cov /= (window_size * window_size) as f64;

            // SSIM formula
            let ssim = (2.0 * mu1 * mu2 + c1) * (2.0 * cov + c2)
                / ((mu1.powi(2) + mu2.powi(2) + c1) * (var1 + var2 + c2));

            sum_ssim += ssim;
            n_windows += 1;
        }
    }

    Ok(sum_ssim / n_windows as f64)
}

/// Measure edge artifacts
#[allow(dead_code)]
fn measure_edge_artifacts(result: &EnhancedDwt2dResult) -> SignalResult<f64> {
    // Check for discontinuities at subband edges
    let mut total_artifacts = 0.0;

    // Check approximation edges
    let (rows, cols) = result.approx.dim();
    for i in 0..rows {
        total_artifacts += (_result.approx[[i, 0]] - result.approx[[i, 1]]).abs();
        total_artifacts += (_result.approx[[i, cols - 1]] - result.approx[[i, cols - 2]]).abs();
    }

    for j in 0..cols {
        total_artifacts += (_result.approx[[0, j]] - result.approx[[1, j]]).abs();
        total_artifacts += (_result.approx[[rows - 1, j]] - result.approx[[rows - 2, j]]).abs();
    }

    // Normalize by perimeter
    let perimeter = 2.0 * (rows + cols) as f64;
    Ok(total_artifacts / perimeter)
}

/// Measure boundary continuity
#[allow(dead_code)]
fn measure_boundary_continuity(result: &EnhancedDwt2dResult) -> SignalResult<f64> {
    // Measure smoothness of transitions
    let mut continuity_score = 0.0;
    let mut n_measurements = 0;

    // Check horizontal continuity
    let (rows, _) = result.approx.dim();
    for i in 1..rows - 1 {
        let left_diff = (_result.approx[[i, 0]] - result.approx[[i, 1]]).abs();
        let right_diff = (_result.approx[[i, 1]] - result.approx[[i, 2]]).abs();
        continuity_score += 1.0 / ((1.0 + (left_diff - right_diff) as f64).abs());
        n_measurements += 1;
    }

    Ok(continuity_score / n_measurements as f64)
}

/// Check symmetry preservation
#[allow(dead_code)]
fn check_symmetry_preservation(
    original: &Array2<f64>,
    result: &EnhancedDwt2dResult,
) -> SignalResult<f64> {
    // For symmetric signals, check if symmetry is preserved in decomposition
    // This is a simplified check
    let (rows, cols) = original.dim();

    // Check if original has approximate symmetry
    let mut h_symmetry = 0.0;
    let mut v_symmetry = 0.0;

    for i in 0..rows / 2 {
        for j in 0..cols {
            h_symmetry += (original[[i, j]] - original[[rows - 1 - i, j]]).abs();
        }
    }

    for i in 0..rows {
        for j in 0..cols / 2 {
            v_symmetry += (original[[i, j]] - original[[i, cols - 1 - j]]).abs();
        }
    }

    // If original is symmetric, check if decomposition preserves it
    let symmetry_threshold = 0.1 * compute_energy(original);

    if h_symmetry < symmetry_threshold || v_symmetry < symmetry_threshold {
        // Check symmetry in approximation
        let (a_rows, a_cols) = result.approx.dim();
        let mut approx_symmetry = 0.0;

        for i in 0..a_rows / 2 {
            for j in 0..a_cols {
                approx_symmetry +=
                    (result.approx[[i, j]] - result.approx[[a_rows - 1 - i, j]]).abs();
            }
        }

        let preservation_score =
            1.0 - (approx_symmetry / (h_symmetry + v_symmetry + 1e-10)).min(1.0);
        Ok(preservation_score)
    } else {
        Ok(1.0) // Not symmetric, so preservation is not applicable
    }
}

/// Check if reconstruction is valid
#[allow(dead_code)]
fn is_valid_reconstruction(reconstructed: &Array2<f64>) -> bool {
    reconstructed.iter().all(|&x: &f64| x.is_finite())
}

/// Validate multilevel decomposition
#[allow(dead_code)]
pub fn validate_multilevel_dwt2d(
    image: &Array2<f64>,
    wavelet: Wavelet,
    levels: usize,
    tolerance: f64,
) -> SignalResult<bool> {
    let config = Dwt2dConfig::default();

    // Perform multilevel decomposition
    let decomp = wavedec2_enhanced(image, wavelet, levels, &config)?;

    // Check energy conservation across all levels
    let input_energy = compute_energy(image);
    let mut total_energy = compute_energy(&decomp.approx);

    for (h, v, d) in &decomp.details {
        total_energy += compute_energy(h);
        total_energy += compute_energy(v);
        total_energy += compute_energy(d);
    }

    let energy_ratio = total_energy / input_energy;

    if ((energy_ratio - 1.0) as f64).abs() > tolerance * 10.0 {
        return Ok(false);
    }

    // Verify level consistency
    let mut current_size = image.dim();
    for level in 0..decomp.details.len() {
        let expected_size = ((current_size.0 + 1) / 2, (current_size.1 + 1) / 2);
        let (h, v, d) = &decomp.details[decomp.details.len() - 1 - level];

        if h.dim() != expected_size || v.dim() != expected_size || d.dim() != expected_size {
            return Ok(false);
        }

        current_size = expected_size;
    }

    Ok(true)
}

/// Generate test images for validation
#[allow(dead_code)]
pub fn generate_test_images() -> Vec<(&'static str, Array2<f64>)> {
    let mut test_images = Vec::new();

    // 1. Constant image
    let constant = Array2::from_elem((64, 64), 1.0);
    test_images.push(("constant", constant));

    // 2. Linear gradient
    let mut gradient = Array2::zeros((64, 64));
    for i in 0..64 {
        for j in 0..64 {
            gradient[[i, j]] = (i + j) as f64 / 128.0;
        }
    }
    test_images.push(("gradient", gradient));

    // 3. Checkerboard
    let mut checkerboard = Array2::zeros((64, 64));
    for i in 0..64 {
        for j in 0..64 {
            checkerboard[[i, j]] = if (i / 8 + j / 8) % 2 == 0 { 1.0 } else { 0.0 };
        }
    }
    test_images.push(("checkerboard", checkerboard));

    // 4. Gaussian blob
    let mut gaussian = Array2::zeros((64, 64));
    let center = 32.0;
    let sigma = 10.0;
    for i in 0..64 {
        for j in 0..64 {
            let dx = i as f64 - center;
            let dy = j as f64 - center;
            gaussian[[i, j]] = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
        }
    }
    test_images.push(("gaussian", gaussian));

    // 5. Random noise
    let mut noise = Array2::zeros((64, 64));
    let mut rng = rand::rng();
    for i in 0..64 {
        for j in 0..64 {
            noise[[i, j]] = rng.gen_range(0.0..1.0);
        }
    }
    test_images.push(("noise", noise));

    test_images
}

/// Run comprehensive validation suite
#[allow(dead_code)]
pub fn run_comprehensive_validation(wavelet: Wavelet) -> SignalResult<()> {
    println!("Running comprehensive 2D DWT validation for {:?}", wavelet);
    println!("{}", "=".repeat(60));

    let test_images = generate_test_images();
    let tolerance = 1e-10;

    for (name, image) in test_images {
        println!("\nTesting with {} image:", name);

        let result = validate_dwt2d(&image, wavelet, tolerance)?;

        println!(
            "  Reconstruction error: {:.2e}",
            result.reconstruction_error.max_error
        );
        println!(
            "  Energy ratio: {:.6}",
            result.energy_conservation.energy_ratio
        );
        println!("  PSNR: {:.2} dB", result.reconstruction_error.psnr);
        println!("  SSIM: {:.4}", result.reconstruction_error.ssim);
        println!("  Speedup: {:.2}x", result.performance.speedup);
        println!("  Overall score: {:.1}/100", result.overall_score);

        if !result.issues.is_empty() {
            println!("  Issues:");
            for issue in &result.issues {
                println!("    - {}", issue);
            }
        }

        // Test multilevel
        if validate_multilevel_dwt2d(&image, wavelet, 3, tolerance)? {
            println!("  ✓ Multilevel decomposition valid");
        } else {
            println!("  ✗ Multilevel decomposition failed");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_validation_basic() {
        let image = Array2::eye(32);
        let result = validate_dwt2d(&image, Wavelet::Haar, 1e-10).unwrap();

        assert!(result.overall_score > 80.0);
        assert!(result.reconstruction_error.max_error < 1e-10);
    }

    #[test]
    fn test_energy_conservation() {
        let image = Array2::from_elem((64, 64), 1.0);
        let metrics = test_energy_conservation(&image, Wavelet::DB(4)).unwrap();

        assert!(((metrics.energy_ratio - 1.0) as f64).abs() < 1e-10);
    }

    #[test]
    fn test_multilevel_validation() {
        let image = Array2::from_shape_fn((128, 128), |(i, j)| ((i + j) as f64).sin());

        assert!(validate_multilevel_dwt2d(&image, Wavelet::Sym(8), 4, 1e-10).unwrap());
    }
}
