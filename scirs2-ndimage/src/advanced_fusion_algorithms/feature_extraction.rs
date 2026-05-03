//! # Multi-Dimensional Feature Extraction
//!
//! This module handles sophisticated feature extraction across multiple dimensions
//! for advanced image processing algorithms. It provides comprehensive feature
//! extraction capabilities including:
//!
//! - **Spatial Features**: Traditional spatial domain features like gradients, textures, and local statistics
//! - **Temporal Features**: Time-based features for processing sequences and temporal patterns
//! - **Frequency Features**: Spectral domain features using Gabor-like filters and frequency analysis
//! - **Quantum Features**: Quantum-inspired features for advanced processing paradigms
//! - **Consciousness Features**: Bio-inspired consciousness-level processing features
//! - **Causal Features**: Causal relationship and dependency features
//! - **Advanced Dimensional Features**: Multi-dimensional feature fusion and combination
//!
//! The module supports processing in multiple advanced dimensions beyond traditional
//! spatial coordinates, enabling sophisticated analysis of complex image data.

use scirs2_core::ndarray::s;
use scirs2_core::ndarray::{Array2, Array3, Array4, Array5, ArrayView2};
use scirs2_core::numeric::Complex;
use scirs2_core::numeric::{Float, FromPrimitive};
use statrs::statistics::Statistics;
use std::collections::{BTreeMap, VecDeque};
use std::f64::consts::PI;

use super::config::*;
use crate::error::NdimageResult;

/// Advanced-Dimensional Feature Extraction
///
/// Extracts features in multiple dimensions beyond traditional spatial dimensions,
/// including temporal, frequency, quantum, and consciousness dimensions.
#[allow(dead_code)]
pub fn extract_advanced_dimensionalfeatures<T>(
    image: &ArrayView2<T>,
    advancedstate: &mut AdvancedState,
    config: &AdvancedConfig,
) -> NdimageResult<Array5<f64>>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width) = image.dim();
    let mut advancedfeatures = Array5::zeros((
        height,
        width,
        config.advanced_dimensions,
        config.temporal_window,
        config.consciousness_depth,
    ));

    // Extract features across all advanced-dimensions
    for y in 0..height {
        for x in 0..width {
            let pixel_value = image[(y, x)].to_f64().unwrap_or(0.0);

            // Spatial dimension features
            let spatialfeatures = extract_spatialfeatures(pixel_value, (y, x), image, config)?;

            // Temporal dimension features
            let temporalfeatures =
                extract_temporalfeatures(pixel_value, &advancedstate.temporal_memory, config)?;

            // Frequency dimension features
            let frequencyfeatures = extract_frequencyfeatures(pixel_value, (y, x), image, config)?;

            // Quantum dimension features
            let quantumfeatures = extract_quantumfeatures(
                pixel_value,
                &advancedstate.consciousness_amplitudes,
                config,
            )?;

            // Consciousness dimension features
            let consciousnessfeatures =
                extract_consciousnessfeatures(pixel_value, advancedstate, config)?;

            // Causal dimension features
            let causalfeatures =
                extract_causalfeatures(pixel_value, &advancedstate.causal_graph, config)?;

            // Store in advanced-dimensional array
            for d in 0..config.advanced_dimensions {
                for t in 0..config.temporal_window {
                    for c in 0..config.consciousness_depth {
                        let feature_value = combine_dimensionalfeatures(
                            &spatialfeatures,
                            &temporalfeatures,
                            &frequencyfeatures,
                            &quantumfeatures,
                            &consciousnessfeatures,
                            &causalfeatures,
                            d,
                            t,
                            c,
                            config,
                        )?;

                        advancedfeatures[(y, x, d, t, c)] = feature_value;
                    }
                }
            }
        }
    }

    // Update advanced-dimensional feature state
    advancedstate.advancedfeatures = advancedfeatures.clone();

    Ok(advancedfeatures)
}

/// Extract Spatial Features
///
/// Extracts spatial domain features from pixel values and their local neighborhoods.
/// Includes normalized positions, gradients, local statistics, edge orientations,
/// and complexity measures.
#[allow(dead_code)]
fn extract_spatialfeatures<T>(
    pixel_value: f64,
    position: (usize, usize),
    image: &ArrayView2<T>,
    _config: &AdvancedConfig,
) -> NdimageResult<Vec<f64>>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width) = image.dim();
    let (y, x) = position;
    let mut features = Vec::with_capacity(8);

    // Feature 1: Normalized pixel intensity
    features.push(pixel_value);

    // Feature 2: Normalized position (x-coordinate)
    features.push(x as f64 / width.max(1) as f64);

    // Feature 3: Normalized position (y-coordinate)
    features.push(y as f64 / height.max(1) as f64);

    // Feature 4: Distance from center
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;
    let distance_from_center =
        ((x as f64 - center_x).powi(2) + (y as f64 - center_y).powi(2)).sqrt();
    let max_distance = (center_x.powi(2) + center_y.powi(2)).sqrt();
    features.push(distance_from_center / max_distance.max(1.0));

    // Feature 5: Local gradient magnitude (approximation)
    let gradient_x = if x > 0 && x < width - 1 {
        let left = image[(y, x - 1)].to_f64().unwrap_or(0.0);
        let right = image[(y, x + 1)].to_f64().unwrap_or(0.0);
        (right - left) / 2.0
    } else {
        0.0
    };

    let gradient_y = if y > 0 && y < height - 1 {
        let top = image[(y - 1, x)].to_f64().unwrap_or(0.0);
        let bottom = image[(y + 1, x)].to_f64().unwrap_or(0.0);
        (bottom - top) / 2.0
    } else {
        0.0
    };

    let gradient_magnitude = (gradient_x.powi(2) + gradient_y.powi(2)).sqrt();
    features.push(gradient_magnitude);

    // Feature 6: Local variance (3x3 neighborhood)
    let mut neighborhood_values = Vec::new();
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let ny = y as i32 + dy;
            let nx = x as i32 + dx;
            if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                neighborhood_values.push(image[(ny as usize, nx as usize)].to_f64().unwrap_or(0.0));
            }
        }
    }

    let mean = neighborhood_values.iter().sum::<f64>() / neighborhood_values.len().max(1) as f64;
    let variance = neighborhood_values
        .iter()
        .map(|&v| (v - mean).powi(2))
        .sum::<f64>()
        / neighborhood_values.len().max(1) as f64;
    features.push(variance.sqrt()); // Standard deviation

    // Feature 7: Edge orientation (approximation)
    let edge_orientation = if gradient_magnitude > 1e-10 {
        gradient_y.atan2(gradient_x)
    } else {
        0.0
    };
    features.push(edge_orientation / PI); // Normalized to [-1, 1]

    // Feature 8: Advanced-dimensional complexity measure
    let complexity = pixel_value * variance.sqrt() * (1.0 + gradient_magnitude);
    features.push(complexity.tanh()); // Bounded complexity measure

    Ok(features)
}

/// Extract Temporal Features
///
/// Extracts temporal features from pixel values over time using temporal memory.
/// Includes temporal gradients, acceleration, variance, periodicity, entropy,
/// momentum, and coherence measures.
#[allow(dead_code)]
fn extract_temporalfeatures(
    pixel_value: f64,
    temporal_memory: &VecDeque<Array3<f64>>,
    config: &AdvancedConfig,
) -> NdimageResult<Vec<f64>> {
    let mut features = Vec::with_capacity(8);

    if temporal_memory.is_empty() {
        return Ok(vec![0.0; 8]);
    }

    // Feature 1: Current intensity
    features.push(pixel_value);

    // Feature 2: Temporal gradient (rate of change)
    let temporal_gradient = if temporal_memory.len() >= 2 {
        let current = pixel_value;
        let previous = temporal_memory.back().expect("Operation failed")[(0, 0, 0)];
        current - previous
    } else {
        0.0
    };
    features.push(temporal_gradient.tanh()); // Bounded gradient

    // Feature 3: Temporal acceleration (second derivative)
    let temporal_acceleration = if temporal_memory.len() >= 3 {
        let current = pixel_value;
        let prev1 = temporal_memory[temporal_memory.len() - 1][(0, 0, 0)];
        let prev2 = temporal_memory[temporal_memory.len() - 2][(0, 0, 0)];
        (current - prev1) - (prev1 - prev2)
    } else {
        0.0
    };
    features.push(temporal_acceleration.tanh());

    // Feature 4: Temporal variance over window
    let temporal_values: Vec<f64> = temporal_memory
        .iter()
        .map(|arr| arr[(0, 0, 0)])
        .chain(std::iter::once(pixel_value))
        .collect();

    let temporal_mean = temporal_values.iter().sum::<f64>() / temporal_values.len() as f64;
    let temporal_variance = temporal_values
        .iter()
        .map(|&v| (v - temporal_mean).powi(2))
        .sum::<f64>()
        / temporal_values.len() as f64;
    features.push(temporal_variance.sqrt());

    // Feature 5: Temporal periodicity (simple autocorrelation measure)
    let autocorr = if temporal_values.len() >= 4 {
        let half_len = temporal_values.len() / 2;
        let first_half = &temporal_values[0..half_len];
        let second_half = &temporal_values[half_len..half_len * 2];

        let correlation = first_half
            .iter()
            .zip(second_half.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>()
            / half_len as f64;
        correlation.tanh()
    } else {
        0.0
    };
    features.push(autocorr);

    // Feature 6: Temporal entropy (approximate)
    let entropy = if temporal_values.len() > 1 {
        let mut hist = [0u32; 10];
        for &val in &temporal_values {
            let bin = ((val.clamp(0.0, 1.0) * 9.0) as usize).min(9);
            hist[bin] += 1;
        }

        let total = temporal_values.len() as f64;
        hist.iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.ln()
            })
            .sum::<f64>()
    } else {
        0.0
    };
    features.push(entropy / 10.0.ln()); // Normalized entropy

    // Feature 7: Temporal momentum (weighted recent changes)
    let momentum = temporal_values
        .windows(2)
        .enumerate()
        .map(|(i, window)| {
            let weight = (i + 1) as f64 / temporal_values.len() as f64;
            weight * (window[1] - window[0])
        })
        .sum::<f64>();
    features.push(momentum.tanh());

    // Feature 8: Temporal coherence measure
    let coherence = if temporal_values.len() >= config.temporal_window / 4 {
        let smoothed: Vec<f64> = temporal_values
            .windows(3)
            .map(|window| window.iter().sum::<f64>() / 3.0)
            .collect();

        let original_var = temporal_variance;
        let smoothed_mean = smoothed.iter().sum::<f64>() / smoothed.len() as f64;
        let smoothed_var = smoothed
            .iter()
            .map(|&v| (v - smoothed_mean).powi(2))
            .sum::<f64>()
            / smoothed.len() as f64;

        1.0 - (smoothed_var / original_var.max(1e-10))
    } else {
        0.0
    };
    features.push(coherence.clamp(0.0, 1.0));

    Ok(features)
}

/// Extract Frequency Features
///
/// Extracts frequency domain features using local spectral analysis.
/// Includes DC components, high frequency energy, Gabor-like responses,
/// orientation strength, and spectral characteristics.
#[allow(dead_code)]
fn extract_frequencyfeatures<T>(
    pixel_value: f64,
    position: (usize, usize),
    image: &ArrayView2<T>,
    config: &AdvancedConfig,
) -> NdimageResult<Vec<f64>>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width) = image.dim();
    let (y, x) = position;
    let mut features = Vec::with_capacity(8);

    // Define window size for local frequency analysis
    let window_size = 7; // 7x7 window for local analysis
    let half_window = window_size / 2;

    // Extract local window around the pixel
    let mut local_window = Vec::new();
    for dy in -(half_window as i32)..=(half_window as i32) {
        for dx in -(half_window as i32)..=(half_window as i32) {
            let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
            let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
            local_window.push(image[(ny, nx)].to_f64().unwrap_or(0.0));
        }
    }

    // Feature 1: Local DC component (mean)
    let dc_component = local_window.iter().sum::<f64>() / local_window.len() as f64;
    features.push(dc_component);

    // Feature 2: High frequency energy (local Laplacian response)
    let mut high_freq_energy = 0.0;
    if y > 0 && y < height - 1 && x > 0 && x < width - 1 {
        let laplacian = -4.0 * pixel_value
            + image[(y - 1, x)].to_f64().unwrap_or(0.0)
            + image[(y + 1, x)].to_f64().unwrap_or(0.0)
            + image[(y, x - 1)].to_f64().unwrap_or(0.0)
            + image[(y, x + 1)].to_f64().unwrap_or(0.0);
        high_freq_energy = laplacian.abs();
    }
    features.push(high_freq_energy.tanh()); // Normalized high frequency energy

    // Feature 3 & 4: Gabor-like responses (horizontal and vertical)
    let mut gabor_horizontal = 0.0;
    let mut gabor_vertical = 0.0;

    for i in 0..window_size {
        for j in 0..window_size {
            let val = local_window[i * window_size + j];
            let rel_y = i as f64 - half_window as f64;
            let rel_x = j as f64 - half_window as f64;

            // Simplified Gabor filter responses
            let gaussian = (-0.5 * (rel_x * rel_x + rel_y * rel_y) / 2.0).exp();
            let horizontal_freq = (2.0 * PI * rel_x / 3.0).cos();
            let vertical_freq = (2.0 * PI * rel_y / 3.0).cos();

            gabor_horizontal += val * gaussian * horizontal_freq;
            gabor_vertical += val * gaussian * vertical_freq;
        }
    }

    features.push(gabor_horizontal.tanh());
    features.push(gabor_vertical.tanh());

    // Feature 5: Local frequency variance (energy spread)
    let window_mean = dc_component;
    let frequency_variance = local_window
        .iter()
        .map(|&val| (val - window_mean).powi(2))
        .sum::<f64>()
        / local_window.len() as f64;
    features.push(frequency_variance.sqrt().tanh());

    // Feature 6: Dominant orientation strength
    let mut gradient_x_total = 0.0;
    let mut gradient_y_total = 0.0;

    for i in 1..window_size - 1 {
        for j in 1..window_size - 1 {
            let _idx = i * window_size + j;
            let left_idx = i * window_size + (j - 1);
            let right_idx = i * window_size + (j + 1);
            let top_idx = (i - 1) * window_size + j;
            let bottom_idx = (i + 1) * window_size + j;

            let gx = (local_window[right_idx] - local_window[left_idx]) / 2.0;
            let gy = (local_window[bottom_idx] - local_window[top_idx]) / 2.0;

            gradient_x_total += gx;
            gradient_y_total += gy;
        }
    }

    let orientation_strength =
        (gradient_x_total * gradient_x_total + gradient_y_total * gradient_y_total).sqrt();
    features.push(orientation_strength.tanh());

    // Feature 7: Local spectral centroid (center of frequency mass)
    let mut weighted_sum = 0.0;
    let mut total_energy = 0.0;

    for (i, &val) in local_window.iter().enumerate() {
        let weight = (i as f64 + 1.0) / local_window.len() as f64; // Simple frequency weighting
        weighted_sum += val.abs() * weight;
        total_energy += val.abs();
    }

    let spectral_centroid = if total_energy > 1e-10 {
        weighted_sum / total_energy
    } else {
        0.5
    };
    features.push(spectral_centroid);

    // Feature 8: Advanced-dimensional frequency complexity
    let complexity_factor = config.advanced_dimensions as f64;
    let temporal_factor = config.temporal_window as f64;

    let advanced_frequency = (high_freq_energy * orientation_strength * frequency_variance)
        .powf(1.0 / 3.0) // Geometric mean
        * (1.0 + (complexity_factor / 100.0).tanh())
        * (1.0 + (temporal_factor / 1000.0).tanh());

    features.push(advanced_frequency.tanh());

    Ok(features)
}

/// Extract Quantum Features
///
/// Extracts 8 statistical moments of |ψ|² from the quantum walk probability distribution:
/// `[mean, std, skewness, kurtosis, entropy, max_prob, min_prob_nonzero, spatial_spread]`
#[allow(dead_code)]
fn extract_quantumfeatures(
    pixel_value: f64,
    consciousness_amplitudes: &Array4<Complex<f64>>,
    _config: &AdvancedConfig,
) -> NdimageResult<Vec<f64>> {
    let probs: Vec<f64> = consciousness_amplitudes
        .iter()
        .map(|c| c.norm_sqr())
        .collect();

    let total: f64 = probs.iter().sum();
    if total < 1e-12 {
        // No quantum state: return pixel-based fallback features
        return Ok(vec![pixel_value, 0.0, 0.0, 0.0, 0.0, pixel_value, 0.0, 0.0]);
    }

    let n = probs.len() as f64;
    let norm_probs: Vec<f64> = probs.iter().map(|p| p / total).collect();

    // Mean
    let mean = norm_probs.iter().sum::<f64>() / n;

    // Standard deviation
    let variance = norm_probs.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    // Skewness
    let skewness = if std_dev > 1e-12 {
        norm_probs
            .iter()
            .map(|p| ((p - mean) / std_dev).powi(3))
            .sum::<f64>()
            / n
    } else {
        0.0
    };

    // Kurtosis
    let kurtosis = if std_dev > 1e-12 {
        norm_probs
            .iter()
            .map(|p| ((p - mean) / std_dev).powi(4))
            .sum::<f64>()
            / n
            - 3.0
    } else {
        0.0
    };

    // Shannon entropy of probability distribution
    let entropy: f64 = -norm_probs
        .iter()
        .filter(|&&p| p > 1e-12)
        .map(|&p| p * p.ln())
        .sum::<f64>();

    // Max probability
    let max_prob = norm_probs.iter().cloned().fold(0.0_f64, f64::max);

    // Min non-zero probability
    let min_prob_nonzero = norm_probs
        .iter()
        .cloned()
        .filter(|&p| p > 1e-12)
        .fold(f64::INFINITY, f64::min);
    let min_prob_nonzero = if min_prob_nonzero.is_infinite() {
        0.0
    } else {
        min_prob_nonzero
    };

    // Spatial spread: weighted std of index position
    let indices: Vec<f64> = (0..norm_probs.len()).map(|i| i as f64 / n).collect();
    let weighted_mean_idx: f64 = indices
        .iter()
        .zip(norm_probs.iter())
        .map(|(i, p)| i * p)
        .sum();
    let spatial_spread = indices
        .iter()
        .zip(norm_probs.iter())
        .map(|(i, p)| p * (i - weighted_mean_idx).powi(2))
        .sum::<f64>()
        .sqrt();

    Ok(vec![
        mean,
        std_dev,
        skewness,
        kurtosis,
        entropy,
        max_prob,
        min_prob_nonzero,
        spatial_spread,
    ])
}

/// Extract Consciousness Features
///
/// Extracts 8 features from the effective information distribution:
/// `[mean_phi, std_phi, max_phi, min_phi, phi_entropy, phi_skewness, phi_range, phi_autocorr]`
#[allow(dead_code)]
fn extract_consciousnessfeatures(
    pixel_value: f64,
    advancedstate: &AdvancedState,
    _config: &AdvancedConfig,
) -> NdimageResult<Vec<f64>> {
    // Compute per-element phi approximation: |ψ|² normalized
    let phi_vals: Vec<f64> = advancedstate
        .consciousness_amplitudes
        .iter()
        .map(|c| c.norm_sqr())
        .collect();

    let total: f64 = phi_vals.iter().sum();
    if total < 1e-12 || phi_vals.is_empty() {
        return Ok(vec![pixel_value, 0.0, pixel_value, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    let n = phi_vals.len() as f64;
    let norm: Vec<f64> = phi_vals.iter().map(|p| p / total).collect();

    let mean_phi = norm.iter().sum::<f64>() / n;
    let variance = norm.iter().map(|p| (p - mean_phi).powi(2)).sum::<f64>() / n;
    let std_phi = variance.sqrt();
    let max_phi = norm.iter().cloned().fold(0.0_f64, f64::max);
    let min_phi = norm.iter().cloned().fold(f64::INFINITY, f64::min);
    let min_phi = if min_phi.is_infinite() { 0.0 } else { min_phi };

    // Phi entropy
    let phi_entropy: f64 = -norm
        .iter()
        .filter(|&&p| p > 1e-12)
        .map(|&p| p * p.ln())
        .sum::<f64>();

    // Phi skewness
    let phi_skewness = if std_phi > 1e-12 {
        norm.iter()
            .map(|p| ((p - mean_phi) / std_phi).powi(3))
            .sum::<f64>()
            / n
    } else {
        0.0
    };

    // Phi range
    let phi_range = max_phi - min_phi;

    // Phi autocorrelation (lag-1)
    let phi_autocorr = if norm.len() >= 2 {
        let pairs: f64 = norm.windows(2).map(|w| w[0] * w[1]).sum();
        pairs / (n - 1.0).max(1.0)
    } else {
        0.0
    };

    Ok(vec![
        mean_phi,
        std_phi,
        max_phi,
        min_phi,
        phi_entropy,
        phi_skewness,
        phi_range,
        phi_autocorr,
    ])
}

/// Extract Causal Features
///
/// Extracts 8 features from Granger causality statistics across causal relations:
/// `[mean_strength, max_strength, fraction_significant, mean_confidence, std_confidence,
///   mean_delay, causal_density, weighted_influence]`
#[allow(dead_code)]
fn extract_causalfeatures(
    pixel_value: f64,
    causal_graph: &BTreeMap<usize, Vec<CausalRelation>>,
    _config: &AdvancedConfig,
) -> NdimageResult<Vec<f64>> {
    if causal_graph.is_empty() {
        return Ok(vec![pixel_value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    let all_relations: Vec<&CausalRelation> = causal_graph.values().flatten().collect();

    if all_relations.is_empty() {
        return Ok(vec![pixel_value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    let n = all_relations.len() as f64;

    let mean_strength = all_relations.iter().map(|r| r.strength).sum::<f64>() / n;
    let max_strength = all_relations
        .iter()
        .map(|r| r.strength)
        .fold(0.0_f64, f64::max);

    let significant_threshold = 0.3;
    let fraction_significant = all_relations
        .iter()
        .filter(|r| r.strength > significant_threshold)
        .count() as f64
        / n;

    let mean_confidence = all_relations.iter().map(|r| r.confidence).sum::<f64>() / n;
    let var_confidence = all_relations
        .iter()
        .map(|r| (r.confidence - mean_confidence).powi(2))
        .sum::<f64>()
        / n;
    let std_confidence = var_confidence.sqrt();

    let mean_delay = all_relations.iter().map(|r| r.delay as f64).sum::<f64>() / n;

    // Causal density: number of relations relative to graph size
    let causal_density = if !causal_graph.is_empty() {
        n / causal_graph.len() as f64
    } else {
        0.0
    };

    // Weighted influence: sum(strength * confidence)
    let weighted_influence = all_relations
        .iter()
        .map(|r| r.strength * r.confidence)
        .sum::<f64>()
        / n;

    Ok(vec![
        mean_strength,
        max_strength,
        fraction_significant,
        mean_confidence,
        std_confidence,
        mean_delay,
        causal_density,
        weighted_influence,
    ])
}

/// Combine Dimensional Features
///
/// Combines features from multiple dimensions into a single feature value using
/// weighted L2-norm combination across feature groups, selected by dimension/temporal/consciousness indices.
#[allow(dead_code)]
fn combine_dimensionalfeatures(
    spatial: &[f64],
    temporal: &[f64],
    frequency: &[f64],
    quantum: &[f64],
    consciousness: &[f64],
    causal: &[f64],
    d: usize,
    t: usize,
    c: usize,
    _config: &AdvancedConfig,
) -> NdimageResult<f64> {
    // Group weights: spatial, temporal, frequency, quantum, consciousness, causal
    let group_weights = [0.25_f64, 0.15, 0.20, 0.15, 0.15, 0.10];

    let groups: [&[f64]; 6] = [spatial, temporal, frequency, quantum, consciousness, causal];

    let mut combined = 0.0_f64;

    for (group_idx, (group, &weight)) in groups.iter().zip(group_weights.iter()).enumerate() {
        if group.is_empty() {
            continue;
        }

        // Select element from group using the dimension/temporal/consciousness index
        let selector = match group_idx {
            0 => d,     // spatial: indexed by dimension
            1 => t,     // temporal: indexed by temporal window
            2 => d,     // frequency: indexed by dimension
            3 => c,     // quantum: indexed by consciousness depth
            4 => c,     // consciousness: indexed by consciousness depth
            _ => d % 2, // causal: alternating
        };

        let element_idx = selector % group.len();
        let val = group[element_idx];

        combined += weight * val * val;
    }

    // Return sqrt of weighted sum of squares (weighted L2 norm)
    Ok(combined.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array2, Array4};
    use scirs2_core::numeric::Complex;
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::{Arc, RwLock};

    fn make_test_state() -> AdvancedState {
        use scirs2_core::ndarray::{Array1, Array5};

        let mut amps = Array4::zeros((4, 4, 2, 2));
        for (i, v) in amps.iter_mut().enumerate() {
            *v = Complex::new((i as f64 * 0.3).sin(), (i as f64 * 0.3).cos());
        }

        AdvancedState {
            consciousness_amplitudes: amps,
            meta_parameters: Array2::zeros((4, 4)),
            network_topology: Arc::new(RwLock::new(NetworkTopology {
                connections: std::collections::HashMap::new(),
                nodes: Vec::new(),
                global_properties: NetworkProperties {
                    coherence: 0.5,
                    self_organization_index: 0.3,
                    consciousness_emergence: 0.2,
                    efficiency: 0.8,
                },
            })),
            temporal_memory: VecDeque::new(),
            causal_graph: BTreeMap::new(),
            advancedfeatures: Array5::zeros((1, 1, 1, 1, 1)),
            resource_allocation: ResourceState {
                cpu_allocation: vec![0.5],
                memory_allocation: 0.5,
                gpu_allocation: None,
                quantum_allocation: None,
                allocationhistory: VecDeque::new(),
            },
            efficiencymetrics: EfficiencyMetrics {
                ops_per_second: 1000.0,
                memory_efficiency: 0.8,
                energy_efficiency: 0.6,
                quality_efficiency: 0.75,
                temporal_efficiency: 0.9,
            },
            processing_cycles: 0,
        }
    }

    #[test]
    fn test_extract_quantumfeatures_length() {
        let amps = Array4::from_elem((4, 4, 2, 2), Complex::new(0.5, 0.3));
        let config = AdvancedConfig::default();
        let result = extract_quantumfeatures(0.5, &amps, &config)
            .expect("extract_quantumfeatures should not fail");
        assert_eq!(
            result.len(),
            8,
            "quantum features must have exactly 8 elements, got {}",
            result.len()
        );
    }

    #[test]
    fn test_extract_consciousnessfeatures_length() {
        let state = make_test_state();
        let config = AdvancedConfig::default();
        let result = extract_consciousnessfeatures(0.5, &state, &config)
            .expect("extract_consciousnessfeatures should not fail");
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_extract_causalfeatures_length() {
        let causal_graph = BTreeMap::new();
        let config = AdvancedConfig::default();
        let result = extract_causalfeatures(0.5, &causal_graph, &config)
            .expect("extract_causalfeatures should not fail");
        assert_eq!(result.len(), 8);
    }
}
