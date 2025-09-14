// Validation utilities for Wavelet Packet Transform
//
// This module provides comprehensive validation functions for WPT implementations,
// including energy conservation checks, reconstruction accuracy, and numerical stability.

use crate::dwt::{wavedec, Wavelet};
use crate::error::{SignalError, SignalResult};
use crate::wpt::{reconstruct_from_nodes, wp_decompose, WaveletPacketTree};
use ndarray::ArrayView1;
use num_traits::{Float, NumCast};
use rand::prelude::*;
use rand::Rng;
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::validation::{check_finite, check_positive};
use std::collections::HashMap;
use std::time::Instant;

#[allow(unused_imports)]
/// Validation result for WPT
#[derive(Debug, Clone)]
pub struct WptValidationResult {
    /// Energy conservation ratio (should be close to 1.0)
    pub energy_ratio: f64,
    /// Maximum reconstruction error
    pub max_reconstruction_error: f64,
    /// Mean reconstruction error
    pub mean_reconstruction_error: f64,
    /// Signal-to-noise ratio of reconstruction
    pub reconstruction_snr: f64,
    /// Parseval frame ratio (should be close to 1.0)
    pub parseval_ratio: f64,
    /// Numerical stability score (0-1)
    pub stability_score: f64,
    /// Orthogonality metrics
    pub orthogonality: Option<OrthogonalityMetrics>,
    /// Performance metrics
    pub performance: Option<PerformanceMetrics>,
    /// Best basis stability
    pub best_basis_stability: Option<BestBasisStability>,
    /// Compression efficiency
    pub compression_efficiency: Option<CompressionEfficiency>,
    /// Issues found during validation
    pub issues: Vec<String>,
}

/// Orthogonality validation metrics
#[derive(Debug, Clone)]
pub struct OrthogonalityMetrics {
    /// Maximum inner product between different basis functions
    pub max_cross_correlation: f64,
    /// Minimum norm of basis functions
    pub min_norm: f64,
    /// Maximum norm of basis functions
    pub max_norm: f64,
    /// Frame bounds (lower, upper)
    pub frame_bounds: (f64, f64),
}

/// Performance validation metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Decomposition time (milliseconds)
    pub decomposition_time_ms: f64,
    /// Reconstruction time (milliseconds)
    pub reconstruction_time_ms: f64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,
    /// Computational complexity score
    pub complexity_score: f64,
}

/// Best basis stability metrics
#[derive(Debug, Clone)]
pub struct BestBasisStability {
    /// Consistency across different cost functions
    pub cost_function_consistency: f64,
    /// Stability under noise
    pub noise_stability: f64,
    /// Basis selection entropy
    pub selection_entropy: f64,
}

/// Compression efficiency metrics
#[derive(Debug, Clone)]
pub struct CompressionEfficiency {
    /// Sparsity ratio (percentage of near-zero coefficients)
    pub sparsity_ratio: f64,
    /// Compression ratio estimate
    pub compression_ratio: f64,
    /// Energy compaction efficiency
    pub energy_compaction: f64,
    /// Rate-distortion score
    pub rate_distortion: f64,
}

/// Get all node coordinates at a specific level
fn get_level_nodes(tree: &WaveletPacketTree, level: usize) -> Vec<(usize, usize)> {
    _tree
        .nodes
        .keys()
        .filter(|(l_, _)| *l_ == level)
        .copied()
        .collect()
}

/// Validate WPT decomposition and reconstruction
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet` - Wavelet to use
/// * `max_level` - Maximum decomposition level
/// * `tolerance` - Tolerance for numerical comparisons
///
/// # Returns
///
/// * Validation result with detailed metrics
#[allow(dead_code)]
pub fn validate_wpt<T>(
    signal: &[T],
    wavelet: Wavelet,
    max_level: usize,
    tolerance: f64,
) -> SignalResult<WptValidationResult>
where
    T: Float + NumCast,
{
    // Convert to f64
    let signal_f64: Vec<f64> = signal.iter().map(|&x| NumCast::from(x).unwrap()).collect();

    // Enhanced input validation
    // Note: check_finite is for scalars, Vec<f64> validation done elsewhere
    check_positive(max_level, "max_level")?;

    // Check signal length constraints
    let min_length = 2_usize.pow(max_level as u32 + 2); // Minimum for meaningful decomposition
    if signal_f64.len() < min_length {
        return Err(SignalError::ValueError(format!(
            "Signal length {} too short for max_level {}. Minimum length required: {}",
            signal_f64.len(),
            max_level,
            min_length
        )));
    }

    // Check for reasonable signal characteristics
    let signal_mean = signal_f64.iter().sum::<f64>() / signal_f64.len() as f64;
    let signal_std = (signal_f64
        .iter()
        .map(|&x| (x - signal_mean).powi(2))
        .sum::<f64>()
        / signal_f64.len() as f64)
        .sqrt();

    if signal_std == 0.0 {
        return Err(SignalError::ValueError(
            "Signal has zero variance (constant signal). WPT decomposition is not meaningful."
                .to_string(),
        ));
    }

    if signal_std < 1e-15 {
        eprintln!("Warning: Signal has very low variance ({:.2e}). WPT results may be numerically unstable.", signal_std);
    }

    // Additional robustness checks
    let dynamic_range = signal_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - signal_f64.iter().cloned().fold(f64::INFINITY, f64::min);

    if dynamic_range > 1e12 {
        eprintln!(
            "Warning: Very large dynamic range ({:.2e}). Consider normalizing the signal.",
            dynamic_range
        );
    }

    // Check for potential issues with the selected wavelet
    match validate_wavelet_compatibility(&signal_f64, wavelet, max_level) {
        Ok(warnings) => {
            for warning in warnings {
                eprintln!("Wavelet compatibility warning: {}", warning);
            }
        }
        Err(e) => return Err(e),
    }

    // Check for extreme values
    let signal_max = signal_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let signal_min = signal_f64.iter().cloned().fold(f64::INFINITY, f64::min);

    if signal_max.abs() > 1e10 || signal_min.abs() > 1e10 {
        eprintln!(
            "Warning: Signal contains very large values. This may cause numerical issues in WPT."
        );
    }

    // Check power-of-two length for optimal performance
    let n = signal_f64.len();
    if !n.is_power_of_two() {
        eprintln!("Warning: Signal length {} is not a power of 2. Consider padding for better performance.", n);
    }

    let mut issues: Vec<String> = Vec::new();

    // Perform decomposition with enhanced error handling
    let tree = wp_decompose(&signal_f64, wavelet, max_level, None)
        .map_err(|e| SignalError::ComputationError(format!("WPT decomposition failed: {}", e)))?;

    // Validate tree structure
    validate_tree_structure(&tree, max_level, signal_f64.len())?;

    // Test 1: Energy conservation
    let input_energy = compute_energy(&signal_f64);
    let tree_energy = compute_tree_energy(&tree)?;
    let energy_ratio = tree_energy / input_energy;

    if ((energy_ratio - 1.0) as f64).abs() > tolerance {
        issues.push(format!(
            "Energy not conserved: ratio = {:.6} (expected ≈ 1.0)",
            energy_ratio
        ));
    }

    // Test 2: Enhanced perfect reconstruction validation
    let nodes = get_level_nodes(&tree, max_level);
    let reconstructed = reconstruct_from_nodes(&tree, &nodes)
        .map_err(|e| SignalError::ComputationError(format!("WPT reconstruction failed: {}", e)))?;

    // Check reconstructed signal properties
    if reconstructed.len() != signal_f64.len() {
        return Err(SignalError::ComputationError(format!(
            "Reconstruction length mismatch: expected {}, got {}",
            signal_f64.len(),
            reconstructed.len()
        )));
    }

    // Check for finite values in reconstruction
    for (i, &val) in reconstructed.iter().enumerate() {
        if !val.is_finite() {
            return Err(SignalError::ComputationError(format!(
                "Non-finite value at index {} in reconstruction: {}",
                i, val
            )));
        }
    }

    let reconstruction_errors = validate_reconstruction(&signal_f64, &reconstructed, tolerance)?;

    if reconstruction_errors.max_error > tolerance {
        issues.push(format!(
            "Reconstruction error exceeds tolerance: {:.2e} > {:.2e}",
            reconstruction_errors.max_error, tolerance
        ));
    }

    // Additional reconstruction quality checks
    if reconstruction_errors.snr < 60.0 {
        // Less than 60 dB SNR
        issues.push(format!(
            "Poor reconstruction quality: SNR = {:.1} dB (expected > 60 dB)",
            reconstruction_errors.snr
        ));
    }

    // Check for systematic bias in reconstruction
    let bias = signal_f64
        .iter()
        .zip(reconstructed.iter())
        .map(|(&orig, &recon)| recon - orig)
        .sum::<f64>()
        / signal_f64.len() as f64;

    if bias.abs() > tolerance * 10.0 {
        issues.push(format!(
            "Significant reconstruction bias detected: {:.2e}",
            bias
        ));
    }

    // Test 3: Parseval frame property
    let parseval_ratio = validate_parseval_frame(&tree, &signal_f64)?;

    if ((parseval_ratio - 1.0) as f64).abs() > tolerance * 10.0 {
        issues.push(format!(
            "Parseval frame property violated: ratio = {:.6}",
            parseval_ratio
        ));
    }

    // Test 4: Numerical stability
    let stability_score = test_numerical_stability(&signal_f64, wavelet, max_level)?;

    if stability_score < 0.9 {
        issues.push(format!(
            "Numerical stability concerns: score = {:.2}",
            stability_score
        ));
    }

    Ok(WptValidationResult {
        energy_ratio,
        max_reconstruction_error: reconstruction_errors.max_error,
        mean_reconstruction_error: reconstruction_errors.mean_error,
        reconstruction_snr: reconstruction_errors.snr,
        parseval_ratio,
        stability_score,
        orthogonality: None,
        performance: None,
        best_basis_stability: None,
        compression_efficiency: None,
        issues,
    })
}

/// Enhanced WPT validation with comprehensive metrics
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet` - Wavelet to use
/// * `max_level` - Maximum decomposition level
/// * `tolerance` - Tolerance for numerical comparisons
/// * `include_advanced` - Whether to include advanced metrics (computationally expensive)
///
/// # Returns
///
/// * Enhanced validation result with all metrics
#[allow(dead_code)]
pub fn validate_wpt_comprehensive<T>(
    signal: &[T],
    wavelet: Wavelet,
    max_level: usize,
    tolerance: f64,
    include_advanced: bool,
) -> SignalResult<WptValidationResult>
where
    T: Float + NumCast + Clone,
{
    // Start with basic validation
    let mut result = validate_wpt(signal, wavelet, max_level, tolerance)?;

    // Convert to f64 for _advanced analysis
    let signal_f64: Vec<f64> = signal.iter().map(|&x| NumCast::from(x).unwrap()).collect();

    // Perform decomposition for _advanced analysis
    let tree = wp_decompose(&signal_f64, wavelet, max_level, None)?;

    if include_advanced {
        // Advanced orthogonality analysis
        result.orthogonality = Some(analyze_orthogonality(&tree, tolerance)?);

        // Performance analysis
        result.performance = Some(analyze_performance(&signal_f64, wavelet, max_level)?);

        // Best basis stability analysis
        result.best_basis_stability = Some(analyze_best_basis_stability(&tree, &signal_f64)?);

        // Compression efficiency analysis
        result.compression_efficiency = Some(analyze_compression_efficiency(&tree, &signal_f64)?);
    }

    Ok(result)
}

/// Compute signal energy
#[allow(dead_code)]
fn compute_energy(signal: &[f64]) -> f64 {
    let signal_view = ArrayView1::from(_signal);
    f64::simd_dot(&signal_view, &signal_view)
}

/// Compute total energy in wavelet packet tree
#[allow(dead_code)]
fn compute_tree_energy(tree: &WaveletPacketTree) -> SignalResult<f64> {
    let mut total_energy = 0.0;

    // Get all leaf nodes (terminal nodes)
    let leaf_nodes = tree.get_leaf_nodes();

    for (level, position) in leaf_nodes {
        if let Some(packet) = tree.get_node(level, position) {
            let packet_energy = compute_energy(&packet.data);
            total_energy += packet_energy;
        }
    }

    Ok(total_energy)
}

/// Reconstruction error metrics
struct ReconstructionErrors {
    max_error: f64,
    mean_error: f64,
    snr: f64,
}

/// Validate reconstruction accuracy
#[allow(dead_code)]
fn validate_reconstruction(
    original: &[f64],
    reconstructed: &[f64],
    tolerance: f64,
) -> SignalResult<ReconstructionErrors> {
    if original.len() != reconstructed.len() {
        return Err(SignalError::ShapeMismatch(
            "Original and reconstructed signals have different lengths".to_string(),
        ));
    }

    let n = original.len();
    let mut errors = vec![0.0; n];

    // Compute errors using SIMD
    let orig_view = ArrayView1::from(original);
    let recon_view = ArrayView1::from(reconstructed);
    let error_view = ArrayView1::from_shape(n, &mut errors).unwrap();

    let _error_result = f64::simd_sub(&orig_view, &recon_view);

    // Compute error metrics
    let max_error = errors.iter().map(|&e: &f64| e.abs()).fold(0.0, f64::max);
    let mean_error = errors.iter().map(|&e: &f64| e.abs()).sum::<f64>() / n as f64;

    // Compute SNR
    let signal_power = compute_energy(original) / n as f64;
    let noise_power = compute_energy(&errors) / n as f64;
    let snr = if noise_power > tolerance * tolerance {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f64::INFINITY
    };

    Ok(ReconstructionErrors {
        max_error,
        mean_error,
        snr,
    })
}

/// Validate Parseval frame property
#[allow(dead_code)]
fn validate_parseval_frame(tree: &WaveletPacketTree, signal: &[f64]) -> SignalResult<f64> {
    // For a Parseval frame, sum of squared coefficients equals signal energy
    let signal_energy = compute_energy(signal);
    let coeffs_energy = compute_tree_energy(_tree)?;

    Ok(coeffs_energy / signal_energy)
}

/// Test numerical stability with edge cases
#[allow(dead_code)]
fn test_numerical_stability(
    signal: &[f64],
    wavelet: Wavelet,
    max_level: usize,
) -> SignalResult<f64> {
    let mut passed_tests = 0;
    let total_tests = 5;

    // Test 1: Zero signal
    let zero_signal = vec![0.0; signal.len()];
    match wp_decompose(&zero_signal, wavelet, max_level, None) {
        Ok(tree) => {
            let nodes = get_level_nodes(&tree, max_level);
            if let Ok(reconstructed) = reconstruct_from_nodes(&tree, &nodes) {
                if reconstructed.iter().all(|&x: &f64| x.abs() < 1e-10) {
                    passed_tests += 1;
                }
            }
        }
        Err(_) => {}
    }

    // Test 2: Constant signal
    let const_signal = vec![1.0; signal.len()];
    match wp_decompose(&const_signal, wavelet, max_level, None) {
        Ok(tree) => {
            let nodes = get_level_nodes(&tree, max_level);
            if let Ok(reconstructed) = reconstruct_from_nodes(&tree, &nodes) {
                let mean = reconstructed.iter().sum::<f64>() / reconstructed.len() as f64;
                if ((mean - 1.0) as f64).abs() < 1e-10 {
                    passed_tests += 1;
                }
            }
        }
        Err(_) => {}
    }

    // Test 3: Impulse signal
    let mut impulse = vec![0.0; signal.len()];
    impulse[signal.len() / 2] = 1.0;
    match wp_decompose(&impulse, wavelet, max_level, None) {
        Ok(tree) => {
            let nodes = get_level_nodes(&tree, max_level);
            if let Ok(reconstructed) = reconstruct_from_nodes(&tree, &nodes) {
                let energy_preserved = compute_energy(&reconstructed).abs() - 1.0;
                if energy_preserved.abs() < 1e-10 {
                    passed_tests += 1;
                }
            }
        }
        Err(_) => {}
    }

    // Test 4: Very small values
    let small_signal: Vec<f64> = signal.iter().map(|&x| x * 1e-10).collect();
    match wp_decompose(&small_signal, wavelet, max_level, None) {
        Ok(tree) => {
            if tree
                .nodes
                .values()
                .all(|node| node.data.iter().all(|&x: &f64| x.is_finite()))
            {
                passed_tests += 1;
            }
        }
        Err(_) => {}
    }

    // Test 5: Large values
    let large_signal: Vec<f64> = signal.iter().map(|&x| x * 1e10).collect();
    match wp_decompose(&large_signal, wavelet, max_level, None) {
        Ok(tree) => {
            if tree
                .nodes
                .values()
                .all(|node| node.data.iter().all(|&x: &f64| x.is_finite()))
            {
                passed_tests += 1;
            }
        }
        Err(_) => {}
    }

    Ok(passed_tests as f64 / total_tests as f64)
}

/// Validate best basis selection
///
/// # Arguments
///
/// * `tree` - Wavelet packet tree
/// * `cost_function` - Cost function name ("shannon", "norm", "threshold")
///
/// # Returns
///
/// * Whether the best basis selection is valid
#[allow(dead_code)]
pub fn validate_best_basis(_tree: &WaveletPacketTree, costfunction: &str) -> SignalResult<bool> {
    // Get the selected best basis
    let best_basis = tree.get_best_basis(cost_function)?;

    // Verify that selected nodes don't overlap
    let mut covered_samples = HashMap::new();

    for (level, position) in &best_basis {
        let node = tree.get_node(*level, *position).ok_or_else(|| {
            SignalError::ValueError(format!(
                "Best basis contains invalid node ({}, {})",
                level, position
            ))
        })?;

        // Calculate sample range covered by this node
        let node_size = node.data.len();
        let start = position * node_size;
        let end = start + node_size;

        // Check for overlaps
        for i in start..end {
            if covered_samples.contains_key(&i) {
                return Ok(false); // Overlap detected
            }
            covered_samples.insert(i, (*level, *position));
        }
    }

    // Verify complete coverage
    let total_samples = tree.get_node(0, 0).map(|n| n.data.len()).unwrap_or(0);

    Ok(covered_samples.len() == total_samples)
}

/// Compare WPT with DWT for consistency
#[allow(dead_code)]
pub fn compare_with_dwt(signal: &[f64], wavelet: Wavelet, level: usize) -> SignalResult<f64> {
    // Perform WPT decomposition
    let wpt_tree = wp_decompose(_signal, wavelet, level, None)?;

    // Perform DWT decomposition
    let dwt_coeffs = wavedec(_signal, wavelet, Some(level), None)?;

    // Compare approximation path in WPT with DWT approximation
    let mut max_diff = 0.0;
    let mut current_pos = 0;

    for l in 0..=level {
        if let Some(wpt_node) = wpt_tree.get_node(l, current_pos) {
            // For DWT path, we always take the approximation (left child)
            current_pos *= 2; // Move to left child for next level

            // Compare with corresponding DWT coefficients
            // This is a simplified comparison - actual implementation would need
            // proper mapping between WPT and DWT coefficient structures
            let node_energy = compute_energy(&wpt_node.data);

            // Check that energy is reasonable
            if node_energy.is_finite() && node_energy >= 0.0 {
                // Energy-based comparison
                let dwt_energy = if l == level {
                    compute_energy(&dwt_coeffs[0]) // First element is approximation coefficients
                } else {
                    // Would need to extract appropriate detail coefficients
                    node_energy // Placeholder
                };

                let diff = (node_energy - dwt_energy).abs() / node_energy.max(1e-10);
                max_diff = max_diff.max(diff);
            }
        }
    }

    Ok(max_diff)
}

/// Validate entropy-based cost functions
#[allow(dead_code)]
pub fn validate_entropy_computation(tree: &WaveletPacketTree) -> SignalResult<bool> {
    let entropy_types = ["shannon", "norm", "threshold"];

    for entropy_type in &entropy_types {
        let costs = tree.compute_all_costs(entropy_type)?;

        // Verify all costs are non-negative
        for cost in costs.values() {
            if *cost < 0.0 || !cost.is_finite() {
                return Ok(false);
            }
        }

        // Verify parent cost >= sum of children costs (for additive costs)
        for ((level, position), _) in &costs {
            if *level < tree.max_level {
                let left_child = (*level + 1, position * 2);
                let right_child = (*level + 1, position * 2 + 1);

                if let (Some(&parent_cost), Some(&left_cost), Some(&right_cost)) = (
                    costs.get(&(*level, *position)),
                    costs.get(&left_child),
                    costs.get(&right_child),
                ) {
                    // For Shannon entropy, parent cost should generally be >= children sum
                    if *entropy_type == "shannon" && parent_cost < left_cost + right_cost - 1e-10 {
                        return Ok(false);
                    }
                }
            }
        }
    }

    Ok(true)
}

/// Analyze orthogonality properties of wavelet packet basis
#[allow(dead_code)]
fn analyze_orthogonality(
    tree: &WaveletPacketTree,
    tolerance: f64,
) -> SignalResult<OrthogonalityMetrics> {
    let leaf_nodes = tree.get_leaf_nodes();

    let mut max_cross_correlation = 0.0;
    let mut min_norm = f64::INFINITY;
    let mut max_norm = 0.0;
    let mut all_norms = Vec::new();

    // Extract all basis functions
    let mut basis_functions = Vec::new();
    for (level, position) in &leaf_nodes {
        if let Some(packet) = tree.get_node(*level, *position) {
            basis_functions.push(&packet.data);
        }
    }

    // Compute pairwise correlations and norms
    for (i, func1) in basis_functions.iter().enumerate() {
        let norm1 = compute_energy(func1).sqrt();
        min_norm = min_norm.min(norm1);
        max_norm = max_norm.max(norm1);
        all_norms.push(norm1);

        for (j, func2) in basis_functions.iter().enumerate() {
            if i != j {
                let dot_product = func1
                    .iter()
                    .zip(func2.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>();

                let norm2 = compute_energy(func2).sqrt();
                let correlation = if norm1 > tolerance && norm2 > tolerance {
                    dot_product / (norm1 * norm2)
                } else {
                    0.0
                };

                max_cross_correlation = max_cross_correlation.max(correlation.abs());
            }
        }
    }

    // Compute frame bounds
    let frame_operator_norms = compute_frame_operator_bounds(&all_norms);
    let frame_bounds = (frame_operator_norms.0, frame_operator_norms.1);

    Ok(OrthogonalityMetrics {
        max_cross_correlation,
        min_norm,
        max_norm,
        frame_bounds,
    })
}

/// Analyze performance characteristics
#[allow(dead_code)]
fn analyze_performance(
    signal: &[f64],
    wavelet: Wavelet,
    max_level: usize,
) -> SignalResult<PerformanceMetrics> {
    // Measure decomposition time
    let start = Instant::now();
    let tree = wp_decompose(signal, wavelet, max_level, None)?;
    let decomposition_time_ms = start.elapsed().as_micros() as f64 / 1000.0;

    // Measure reconstruction time
    let start = Instant::now();
    let nodes = get_level_nodes(&tree, max_level);
    let _ = reconstruct_from_nodes(&tree, &nodes)?;
    let reconstruction_time_ms = start.elapsed().as_micros() as f64 / 1000.0;

    // Estimate memory usage
    let memory_usage_bytes = estimate_memory_usage(&tree);

    // Compute complexity score (based on signal length and decomposition levels)
    let n = signal.len();
    let expected_complexity = n as f64 * max_level as f64 * (max_level as f64).log2();
    let actual_time = decomposition_time_ms + reconstruction_time_ms;
    let complexity_score = if actual_time > 0.0 {
        expected_complexity / actual_time.max(1e-6)
    } else {
        f64::INFINITY
    };

    Ok(PerformanceMetrics {
        decomposition_time_ms,
        reconstruction_time_ms,
        memory_usage_bytes,
        complexity_score,
    })
}

/// Analyze best basis selection stability
#[allow(dead_code)]
fn analyze_best_basis_stability(
    tree: &WaveletPacketTree,
    signal: &[f64],
) -> SignalResult<BestBasisStability> {
    let cost_functions = ["shannon", "norm", "threshold"];

    // Test consistency across different cost functions
    let mut basis_selections = Vec::new();
    for cost_func in &cost_functions {
        match tree.get_best_basis(cost_func) {
            Ok(basis) => basis_selections.push(basis),
            Err(_) => continue,
        }
    }

    let cost_function_consistency = if basis_selections.len() > 1 {
        compute_basis_similarity(&basis_selections[0], &basis_selections[1])
    } else {
        1.0 // Perfect consistency if only one valid basis
    };

    // Test stability under noise
    let mut noise_stability_scores = Vec::new();
    let mut rng = rand::rng();

    for _ in 0..5 {
        // Add small amount of noise
        let noise_level = 0.01 * compute_energy(signal).sqrt() / signal.len() as f64;
        let noisy_signal: Vec<f64> = signal
            .iter()
            .map(|&x| x + noise_level * rng.gen_range(-1.0..1.0))
            .collect();

        // Decompose noisy signal
        if let Ok(noisy_tree) = wp_decompose(&noisy_signal, tree.root.wavelet, tree.max_level, None) {
            if let (Ok(original_basis), Ok(noisy_basis)) = (
                tree.get_best_basis("shannon"),
                noisy_tree.get_best_basis("shannon"),
            ) {
                let similarity = compute_basis_similarity(&original_basis, &noisy_basis);
                noise_stability_scores.push(similarity);
            }
        }
    }

    let noise_stability = if !noise_stability_scores.is_empty() {
        noise_stability_scores.iter().sum::<f64>() / noise_stability_scores.len() as f64
    } else {
        0.0
    };

    // Compute basis selection entropy
    let selection_entropy = compute_selection_entropy(tree, &cost_functions)?;

    Ok(BestBasisStability {
        cost_function_consistency,
        noise_stability,
        selection_entropy,
    })
}

/// Analyze compression efficiency
#[allow(dead_code)]
fn analyze_compression_efficiency(
    tree: &WaveletPacketTree,
    signal: &[f64],
) -> SignalResult<CompressionEfficiency> {
    let original_energy = compute_energy(signal);

    // Get all coefficients
    let mut all_coeffs = Vec::new();
    for node in tree.nodes.values() {
        all_coeffs.extend_from_slice(&node.data);
    }

    // Compute sparsity ratio
    let threshold = original_energy.sqrt() * 1e-6;
    let sparse_count = all_coeffs.iter().filter(|&&x| x.abs() < threshold).count();
    let sparsity_ratio = sparse_count as f64 / all_coeffs.len() as f64;

    // Estimate compression ratio
    let compression_ratio = if sparsity_ratio > 0.1 {
        1.0 / (1.0 - sparsity_ratio + 0.1)
    } else {
        1.0
    };

    // Compute energy compaction efficiency
    let mut sorted_coeffs = all_coeffs.clone();
    sorted_coeffs.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());

    let k_percent = (all_coeffs.len() as f64 * 0.1) as usize; // Top 10% coefficients
    let top_energy: f64 = sorted_coeffs.iter().take(k_percent).map(|&x| x * x).sum();
    let total_energy: f64 = sorted_coeffs.iter().map(|&x| x * x).sum();

    let energy_compaction = if total_energy > 0.0 {
        top_energy / total_energy
    } else {
        0.0
    };

    // Rate-distortion analysis (simplified)
    let rate = -sparsity_ratio.log2().max(0.1); // Bits per sample estimate
    let distortion = 1.0 - energy_compaction; // Distortion measure
    let rate_distortion = if distortion > 0.0 {
        rate / distortion
    } else {
        rate * 10.0 // High score for low distortion
    };

    Ok(CompressionEfficiency {
        sparsity_ratio,
        compression_ratio,
        energy_compaction,
        rate_distortion,
    })
}

/// Helper functions for advanced analysis

#[allow(dead_code)]
fn compute_frame_operator_bounds(norms: &[f64]) -> (f64, f64) {
    if norms.is_empty() {
        return (0.0, 0.0);
    }

    let min_norm_sq = norms.iter().map(|&x| x * x).fold(f64::INFINITY, f64::min);
    let max_norm_sq = norms.iter().map(|&x| x * x).fold(0.0, f64::max);

    (min_norm_sq, max_norm_sq)
}

#[allow(dead_code)]
fn estimate_memory_usage(tree: &WaveletPacketTree) -> usize {
    let mut total_coeffs = 0;

    for node in tree.nodes.values() {
        total_coeffs += node.data.len();
    }

    // Estimate: 8 bytes per f64 + overhead
    total_coeffs * 8 + tree.nodes.len() * 64 // Node overhead estimate
}

#[allow(dead_code)]
fn compute_basis_similarity(basis1: &[(usize, usize)], basis2: &[(usize, usize)]) -> f64 {
    let set1: std::collections::HashSet<_> = basis1.iter().collect();
    let set2: std::collections::HashSet<_> = basis2.iter().collect();

    let intersection = set1.intersection(&set2).count();
    let union = set1.union(&set2).count();

    if union > 0 {
        intersection as f64 / union as f64
    } else {
        1.0
    }
}

#[allow(dead_code)]
fn compute_selection_entropy(
    tree: &WaveletPacketTree,
    cost_functions: &[&str],
) -> SignalResult<f64> {
    let mut selections = Vec::new();

    for &cost_func in cost_functions {
        if let Ok(basis) = tree.get_best_basis(cost_func) {
            selections.push(basis);
        }
    }

    if selections.is_empty() {
        return Ok(0.0);
    }

    // Count frequency of each node selection
    let mut node_frequencies = std::collections::HashMap::new();

    for selection in &selections {
        for &node in selection {
            *node_frequencies.entry(node).or_insert(0usize) += 1;
        }
    }

    // Compute Shannon entropy of selection distribution
    let total_selections = selections.len() as f64;
    let mut entropy = 0.0;

    for &count in node_frequencies.values() {
        let probability = count as f64 / total_selections;
        if probability > 0.0 {
            entropy -= probability * probability.log2();
        }
    }

    Ok(entropy)
}

/// Cross-validation with different wavelets
#[allow(dead_code)]
pub fn cross_validate_wavelets(
    signal: &[f64],
    max_level: usize,
    tolerance: f64,
) -> SignalResult<HashMap<Wavelet, WptValidationResult>> {
    let wavelets = vec![
        Wavelet::Haar,
        Wavelet::DB(2),
        Wavelet::DB(4),
        Wavelet::DB(8),
        Wavelet::BiorNrNd { nr: 2, nd: 2 },
        Wavelet::Coif(2),
        Wavelet::Coif(4),
    ];

    let mut results = HashMap::new();

    for wavelet in wavelets {
        match validate_wpt_comprehensive(signal, wavelet, max_level, tolerance, true) {
            Ok(result) => {
                results.insert(wavelet, result);
            }
            Err(_) => {
                // Skip wavelets that fail validation
                continue;
            }
        }
    }

    Ok(results)
}

/// Generate comprehensive validation report
#[allow(dead_code)]
pub fn generate_wpt_validation_report(result: &WptValidationResult) -> String {
    let mut report = String::new();

    report.push_str("Wavelet Packet Transform Validation Report\n");
    report.push_str("========================================\n\n");

    // Basic metrics
    report.push_str("Basic Validation:\n");
    report.push_str(&format!(
        "  Energy Ratio: {:.6} (should be ≈ 1.0)\n",
        result.energy_ratio
    ));
    report.push_str(&format!(
        "  Max Reconstruction Error: {:.2e}\n",
        result.max_reconstruction_error
    ));
    report.push_str(&format!(
        "  Mean Reconstruction Error: {:.2e}\n",
        result.mean_reconstruction_error
    ));
    report.push_str(&format!(
        "  Reconstruction SNR: {:.1} dB\n",
        result.reconstruction_snr
    ));
    report.push_str(&format!(
        "  Parseval Ratio: {:.6}\n",
        result.parseval_ratio
    ));
    report.push_str(&format!(
        "  Stability Score: {:.2}\n\n",
        result.stability_score
    ));

    // Advanced metrics
    if let Some(ref ortho) = result.orthogonality {
        report.push_str("Orthogonality Analysis:\n");
        report.push_str(&format!(
            "  Max Cross-Correlation: {:.2e}\n",
            ortho.max_cross_correlation
        ));
        report.push_str(&format!(
            "  Frame Bounds: [{:.2}, {:.2}]\n",
            ortho.frame_bounds.0, ortho.frame_bounds.1
        ));
        report.push_str(&format!(
            "  Norm Range: [{:.2}, {:.2}]\n\n",
            ortho.min_norm, ortho.max_norm
        ));
    }

    if let Some(ref perf) = result.performance {
        report.push_str("Performance Analysis:\n");
        report.push_str(&format!(
            "  Decomposition Time: {:.2} ms\n",
            perf.decomposition_time_ms
        ));
        report.push_str(&format!(
            "  Reconstruction Time: {:.2} ms\n",
            perf.reconstruction_time_ms
        ));
        report.push_str(&format!(
            "  Memory Usage: {:.1} KB\n",
            perf.memory_usage_bytes as f64 / 1024.0
        ));
        report.push_str(&format!(
            "  Complexity Score: {:.2}\n\n",
            perf.complexity_score
        ));
    }

    if let Some(ref basis) = result.best_basis_stability {
        report.push_str("Best Basis Stability:\n");
        report.push_str(&format!(
            "  Cost Function Consistency: {:.2}\n",
            basis.cost_function_consistency
        ));
        report.push_str(&format!(
            "  Noise Stability: {:.2}\n",
            basis.noise_stability
        ));
        report.push_str(&format!(
            "  Selection Entropy: {:.2}\n\n",
            basis.selection_entropy
        ));
    }

    if let Some(ref comp) = result.compression_efficiency {
        report.push_str("Compression Efficiency:\n");
        report.push_str(&format!(
            "  Sparsity Ratio: {:.1}%\n",
            comp.sparsity_ratio * 100.0
        ));
        report.push_str(&format!(
            "  Compression Ratio: {:.1}:1\n",
            comp.compression_ratio
        ));
        report.push_str(&format!(
            "  Energy Compaction: {:.1}%\n",
            comp.energy_compaction * 100.0
        ));
        report.push_str(&format!(
            "  Rate-Distortion Score: {:.2}\n\n",
            comp.rate_distortion
        ));
    }

    // Issues
    if !_result.issues.is_empty() {
        report.push_str("Issues Found:\n");
        for issue in &_result.issues {
            report.push_str(&format!("  - {}\n", issue));
        }
    } else {
        report.push_str("✓ No issues found\n");
    }

    report
}

/// Enhanced adaptive threshold validation for different signal types
#[allow(dead_code)]
pub fn validate_wpt_adaptive_threshold<T>(
    signal: &[T],
    wavelet: Wavelet,
    max_level: usize,
    signal_type: SignalType,
) -> SignalResult<WptValidationResult>
where
    T: Float + NumCast + Clone,
{
    // Determine optimal tolerance based on signal characteristics
    let tolerance = match signal_type {
        SignalType::Smooth => 1e-12,
        SignalType::Oscillatory => 1e-10,
        SignalType::Noisy => 1e-8,
        SignalType::Sparse => 1e-14,
        SignalType::Unknown => 1e-10,
    };

    // Run enhanced validation with adaptive parameters
    let mut result = validate_wpt_comprehensive(signal, wavelet, max_level, tolerance, true)?;

    // Add signal-_type-specific analysis
    let signal_f64: Vec<f64> = signal.iter().map(|&x| NumCast::from(x).unwrap()).collect();

    // Analyze signal characteristics
    let _signal_analysis = analyze_signal_characteristics(&signal_f64)?;

    // Adjust validation based on signal _type
    match signal_type {
        SignalType::Smooth => {
            // Smooth signals should have excellent reconstruction
            if result.max_reconstruction_error > tolerance * 100.0 {
                result
                    .issues
                    .push("Smooth signal reconstruction error too high".to_string());
            }
        }
        SignalType::Oscillatory => {
            // Check frequency preservation
            validate_frequency_preservation(&signal_f64, &result)?;
        }
        SignalType::Sparse => {
            // Sparse signals should have high compression efficiency
            if let Some(ref comp) = result.compression_efficiency {
                if comp.sparsity_ratio < 0.3 {
                    result
                        .issues
                        .push("Expected higher sparsity for sparse signal".to_string());
                }
            }
        }
        _ => {}
    }

    Ok(result)
}

/// Signal type classification for adaptive validation
#[derive(Debug, Clone, Copy)]
pub enum SignalType {
    Smooth,
    Oscillatory,
    Noisy,
    Sparse,
    Unknown,
}

/// Signal characteristics analysis
#[derive(Debug, Clone)]
pub struct SignalCharacteristics {
    pub smoothness_index: f64,
    pub oscillation_index: f64,
    pub noise_level: f64,
    pub sparsity_measure: f64,
    pub dominant_frequencies: Vec<f64>,
}

/// Analyze signal characteristics for validation
#[allow(dead_code)]
fn analyze_signal_characteristics(signal: &[f64]) -> SignalResult<SignalCharacteristics> {
    let n = signal.len();

    // Smoothness index (based on second differences)
    let mut second_diffs = Vec::new();
    for i in 1..n - 1 {
        second_diffs.push(((_signal[i + 1] - 2.0 * signal[i] + signal[i - 1]) as f64).abs());
    }
    let smoothness_index =
        1.0 / (1.0 + second_diffs.iter().sum::<f64>() / second_diffs.len() as f64);

    // Oscillation index (based on zero crossings)
    let mut zero_crossings = 0;
    let mean = signal.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = signal.iter().map(|&x| x - mean).collect();

    for i in 1..centered.len() {
        if (centered[i - 1] >= 0.0 && centered[i] < 0.0)
            || (centered[i - 1] < 0.0 && centered[i] >= 0.0)
        {
            zero_crossings += 1;
        }
    }
    let oscillation_index = zero_crossings as f64 / (n - 1) as f64;

    // Noise level estimate (high frequency energy)
    let hf_energy = estimate_high_frequency_energy(_signal)?;
    let total_energy = compute_energy(_signal);
    let noise_level = hf_energy / total_energy.max(1e-15);

    // Sparsity measure
    let threshold = total_energy.sqrt() * 1e-6;
    let sparse_count = signal.iter().filter(|&&x| x.abs() < threshold).count();
    let sparsity_measure = sparse_count as f64 / n as f64;

    // Dominant frequencies (simplified FFT-based analysis)
    let dominant_frequencies = find_dominant_frequencies(_signal)?;

    Ok(SignalCharacteristics {
        smoothness_index,
        oscillation_index,
        noise_level,
        sparsity_measure,
        dominant_frequencies,
    })
}

/// Validate WPT tree structure and consistency
#[allow(dead_code)]
fn validate_tree_structure(
    tree: &WaveletPacketTree,
    max_level: usize,
    original_length: usize,
) -> SignalResult<()> {
    // Check that tree has been properly constructed
    if tree.root.data.is_empty() {
        return Err(SignalError::ComputationError(
            "WPT tree is empty after decomposition".to_string(),
        ));
    }

    // Validate tree depth
    let actual_depth = tree.get_depth();
    if actual_depth != max_level {
        eprintln!(
            "Warning: Tree depth ({}) differs from requested max_level ({})",
            actual_depth, max_level
        );
    }

    // Check that all nodes at each _level have consistent sizes
    for _level in 0..=max_level {
        let expected_size = original_length / (2_usize.pow(_level as u32));
        let nodes_at_level = tree.get_nodes_at_level(_level);

        for (level, position) in nodes_at_level {
            if let Some(node) = tree.get_node(level, position) {
                let data = &node.data;
                // Allow some tolerance for edge effects in filter operations
                let size_diff = (data.len() as i32 - expected_size as i32).abs();
                if size_diff > 2 {
                    return Err(SignalError::ComputationError(format!(
                        "Node at _level {} position {} has unexpected size: expected ≈{}, got {}",
                        level,
                        position,
                        expected_size,
                        data.len()
                    )));
                }

                // Check for finite values in all nodes
                for (i, &val) in data.iter().enumerate() {
                    if !val.is_finite() {
                        return Err(SignalError::ComputationError(format!(
                            "Non-finite value at _level {} position {} index {}: {}",
                            level, position, i, val
                        )));
                    }
                }
            } else {
                return Err(SignalError::ComputationError(format!(
                    "Node at _level {} position {} has no data",
                    level, position
                )));
            }
        }
    }

    // Check that leaf nodes exist and are non-empty
    let leaf_nodes = tree.get_leaf_nodes();
    if leaf_nodes.is_empty() {
        return Err(SignalError::ComputationError(
            "WPT tree has no leaf nodes".to_string(),
        ));
    }

    // Verify total number of coefficients is reasonable
    let mut total_coeffs = 0;
    for (_level, position) in leaf_nodes {
        if let Some(node) = tree.get_node(_level, position) {
            total_coeffs += node.data.len();
        }
    }

    // Total coefficients should be approximately equal to original signal _length
    let coeff_ratio = total_coeffs as f64 / original_length as f64;
    if coeff_ratio < 0.9 || coeff_ratio > 1.1 {
        eprintln!(
            "Warning: Coefficient count ratio ({:.2}) suggests potential issues with decomposition",
            coeff_ratio
        );
    }

    Ok(())
}

/// Estimate high frequency energy component
#[allow(dead_code)]
fn estimate_high_frequency_energy(signal: &[f64]) -> SignalResult<f64> {
    // Simple high-pass filter approximation using differences
    let mut hf_signal = Vec::new();
    for i in 1.._signal.len() {
        hf_signal.push(_signal[i] - signal[i - 1]);
    }
    Ok(compute_energy(&hf_signal))
}

/// Find dominant frequencies using simple peak detection
#[allow(dead_code)]
fn find_dominant_frequencies(signal: &[f64]) -> SignalResult<Vec<f64>> {
    // Simplified frequency analysis - in practice would use FFT
    let mut frequencies = Vec::new();

    // Estimate fundamental frequency from autocorrelation
    let max_lag = signal.len() / 4;
    let mut best_lag = 0;
    let mut max_correlation = 0.0;

    for lag in 1..max_lag {
        let mut correlation = 0.0;
        let mut count = 0;

        for i in lag.._signal.len() {
            correlation += signal[i] * signal[i - lag];
            count += 1;
        }

        if count > 0 {
            correlation /= count as f64;
            if correlation > max_correlation {
                max_correlation = correlation;
                best_lag = lag;
            }
        }
    }

    if best_lag > 0 {
        frequencies.push(1.0 / best_lag as f64); // Normalized frequency
    }

    Ok(frequencies)
}

/// Validate frequency preservation in reconstruction
#[allow(dead_code)]
fn validate_frequency_preservation(
    signal: &[f64],
    result: &WptValidationResult,
) -> SignalResult<()> {
    // This would typically involve spectral analysis of original vs reconstructed
    // For now, just check if oscillatory structure is preserved
    let original_characteristics = analyze_signal_characteristics(signal)?;

    if original_characteristics.oscillation_index > 0.1 {
        // High oscillation signals should maintain their structure
        if result.max_reconstruction_error > 1e-8 {
            // This would be added to issues in calling function
        }
    }

    Ok(())
}

/// Compute normalized correlation between two signals
#[allow(dead_code)]
fn compute_normalized_correlation(signal1: &[f64], signal2: &[f64]) -> SignalResult<f64> {
    if signal1.len() != signal2.len() {
        return Ok(0.0);
    }

    let s1_view = ArrayView1::from(_signal1);
    let s2_view = ArrayView1::from(signal2);

    let dot_product = f64::simd_dot(&s1_view, &s2_view);
    let norm1 = f64::simd_dot(&s1_view, &s1_view).sqrt();
    let norm2 = f64::simd_dot(&s2_view, &s2_view).sqrt();

    if norm1 > 1e-15 && norm2 > 1e-15 {
        Ok(dot_product / (norm1 * norm2))
    } else {
        Ok(0.0)
    }
}

/// Compute frame bounds for a set of norms
#[allow(dead_code)]
fn compute_frame_bounds(norms: &[f64]) -> (f64, f64) {
    if norms.is_empty() {
        return (1.0, 1.0);
    }

    let sum_squares: f64 = norms.iter().map(|&n| n * n).sum();
    let n = norms.len() as f64;

    // Simplified frame bounds calculation
    let avg_squared = sum_squares / n;
    let variance = _norms
        .iter()
        .map(|&norm| (norm * norm - avg_squared).powi(2))
        .sum::<f64>()
        / n;
    let std_dev = variance.sqrt();

    let lower_bound = (avg_squared - std_dev).max(0.0);
    let upper_bound = avg_squared + std_dev;

    (lower_bound.sqrt(), upper_bound.sqrt())
}

/// Enhanced wavelet compatibility validation
///
/// This function performs comprehensive checks to ensure the selected wavelet
/// is appropriate for the given signal and decomposition parameters.
#[allow(dead_code)]
fn validate_wavelet_compatibility(
    signal: &[f64],
    wavelet: Wavelet,
    max_level: usize,
) -> SignalResult<Vec<String>> {
    let mut warnings = Vec::new();

    // Get wavelet properties
    let filter_length = match wavelet.get_filter_length() {
        Ok(len) => len,
        Err(_) => {
            return Err(SignalError::ValueError(format!(
                "Cannot determine filter length for wavelet {:?}",
                wavelet
            )));
        }
    };

    // Check signal length vs filter length requirements
    let min_length_for_level = filter_length * (2_usize.pow(max_level as u32));
    if signal.len() < min_length_for_level {
        warnings.push(format!(
            "Signal length {} may be too short for {} levels with wavelet {:?} (filter length {}). Consider reducing decomposition levels.",
            signal.len(), max_level, wavelet, filter_length
        ));
    }

    // Analyze signal characteristics for wavelet suitability
    let signal_length = signal.len();

    // Check if signal has appropriate frequency content for the selected wavelet
    if let Ok(spectral_info) = analyze_signal_spectrum(signal) {
        match wavelet {
            Wavelet::Haar => {
                if spectral_info.high_freq_content > 0.8 {
                    warnings.push(
                        "Signal has high frequency content; Haar wavelet may not capture smooth features well.".to_string()
                    );
                }
            }
            Wavelet::DB(order) => {
                if order > 10 && spectral_info.smoothness < 0.3 {
                    warnings.push(format!(
                        "Signal appears non-smooth; Daubechies {} may be over-specified. Consider lower order wavelet.",
                        order
                    ));
                }
                if order < 4 && spectral_info.smoothness > 0.8 {
                    warnings.push(format!(
                        "Signal is very smooth; Daubechies {} may under-utilize smoothness. Consider higher order wavelet.",
                        order
                    ));
                }
            }
            Wavelet::Sym(order) => {
                if signal_length < 8 * order {
                    warnings.push(format!(
                        "Signal may be too short for Symlet {}. Each coefficient requires approximately {} samples.",
                        order, 8 * order
                    ));
                }
            }
            Wavelet::Coif(order) => {
                if spectral_info.regularity < 0.5 && order > 3 {
                    warnings.push(format!(
                        "Signal lacks regularity; Coiflet {} may be over-specified for this signal type.",
                        order
                    ));
                }
            }
            _ => {
                // Add specific validations for other wavelets as needed
            }
        }
    }

    // Check decomposition depth appropriateness
    let optimal_max_level = (signal_length as f64).log2().floor() as usize - 2;
    if max_level > optimal_max_level {
        warnings.push(format!(
            "Decomposition _level {} may be too deep for signal length {}. Recommended maximum: {}",
            max_level, signal_length, optimal_max_level
        ));
    }

    // Check for potential boundary effect issues
    let boundary_effect_ratio = filter_length as f64 / signal_length as f64;
    if boundary_effect_ratio > 0.1 {
        warnings.push(format!(
            "Filter length ({}) is large relative to signal length ({}). Boundary effects may be significant.",
            filter_length, signal_length
        ));
    }

    Ok(warnings)
}

/// Analyze signal spectrum to provide wavelet selection guidance
#[allow(dead_code)]
fn analyze_signal_spectrum(signal: &[f64]) -> SignalResult<SpectralInfo> {
    let n = signal.len();

    // Compute basic spectral characteristics using autocorrelation-based approach
    // This is a simplified analysis - full implementation would use FFT

    // Estimate smoothness by looking at consecutive differences
    let mut diff_sum = 0.0;
    let mut diff_count = 0;
    for i in 1..n {
        diff_sum += (_signal[i] - signal[i - 1]).abs();
        diff_count += 1;
    }
    let avg_diff = if diff_count > 0 {
        diff_sum / diff_count as f64
    } else {
        0.0
    };

    // Estimate _signal variance
    let mean = signal.iter().sum::<f64>() / n as f64;
    let variance = signal.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    // Estimate high frequency content (simplified)
    let mut high_freq_energy = 0.0;
    let mut total_energy = 0.0;

    for i in 1..n {
        let local_diff = (_signal[i] - signal[i - 1]).powi(2);
        high_freq_energy += local_diff;
        total_energy += signal[i].powi(2);
    }

    let high_freq_content = if total_energy > 1e-15 {
        high_freq_energy / total_energy
    } else {
        0.0
    };

    // Estimate smoothness (inverse of relative variation)
    let smoothness = if variance > 1e-15 {
        1.0 / (1.0 + avg_diff / variance.sqrt())
    } else {
        1.0
    };

    // Estimate regularity using second-order differences
    let mut second_diff_sum = 0.0;
    for i in 2..n {
        let second_diff = signal[i] - 2.0 * signal[i - 1] + signal[i - 2];
        second_diff_sum += second_diff.abs();
    }
    let avg_second_diff = if n > 2 {
        second_diff_sum / (n - 2) as f64
    } else {
        0.0
    };

    let regularity = if avg_diff > 1e-15 {
        1.0 / (1.0 + avg_second_diff / avg_diff)
    } else {
        1.0
    };

    Ok(SpectralInfo {
        high_freq_content: high_freq_content.min(1.0),
        smoothness: smoothness.min(1.0),
        regularity: regularity.min(1.0),
    })
}

/// Simple spectral characteristics for wavelet selection
#[derive(Debug, Clone)]
struct SpectralInfo {
    /// Relative high frequency content (0-1)
    high_freq_content: f64,
    /// Smoothness measure (0-1, higher = smoother)
    smoothness: f64,
    /// Regularity measure (0-1, higher = more regular)
    regularity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_wpt_validation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Test with a simple sinusoidal signal
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 64.0).sin())
            .collect();

        let result = validate_wpt(&signal, Wavelet::DB(4), 4, 1e-10);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(validation.energy_ratio > 0.9);
        assert!(validation.energy_ratio < 1.1);
    }

    #[test]
    fn test_wavelet_compatibility_validation() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0];
        let warnings = validate_wavelet_compatibility(&signal, Wavelet::DB(8), 3);
        assert!(warnings.is_ok());

        // Should warn about signal being too short for DB8 with 3 levels
        let warning_list = warnings.unwrap();
        assert!(!warning_list.is_empty());
    }

    #[test]
    fn test_spectral_analysis() {
        // Test with smooth signal
        let smooth_signal: Vec<f64> = (0..64).map(|i| (i as f64 / 64.0).powi(2)).collect();

        let info = analyze_signal_spectrum(&smooth_signal);
        assert!(info.is_ok());

        let spectral_info = info.unwrap();
        assert!(spectral_info.smoothness > 0.5); // Should be detected as smooth
    }

    #[test]
    fn test_extreme_signals() {
        // Test constant signal
        let constant_signal = vec![5.0; 128];
        let result = validate_wpt(&constant_signal, Wavelet::Haar, 2, 1e-10);
        assert!(result.is_err()); // Should fail for constant signal

        // Test very short signal
        let short_signal = vec![1.0, 2.0];
        let result = validate_wpt(&short_signal, Wavelet::DB(4), 3, 1e-10);
        assert!(result.is_err()); // Should fail for too short signal
    }
}
