//! Wavelet transform features for time series analysis
//!
//! This module provides comprehensive wavelet-based feature extraction including
//! Discrete Wavelet Transform (DWT), Continuous Wavelet Transform (CWT),
//! multi-resolution analysis, time-frequency analysis, and wavelet-based denoising.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::config::{DenoisingMethod, WaveletConfig, WaveletFamily};
use crate::error::{Result, TimeSeriesError};

/// Wavelet-based features for time series analysis
///
/// This struct contains comprehensive wavelet transform features including
/// energy distribution across scales, entropy measures, regularity indices,
/// and time-frequency analysis results.
#[derive(Debug, Clone)]
pub struct WaveletFeatures<F> {
    /// Energy at different frequency bands from DWT decomposition
    pub energy_bands: Vec<F>,
    /// Relative wavelet energy (normalized energy distribution)
    pub relative_energy: Vec<F>,
    /// Wavelet entropy (Shannon entropy of wavelet coefficients)
    pub wavelet_entropy: F,
    /// Wavelet variance (measure of signal variability)
    pub wavelet_variance: F,
    /// Regularity measure based on wavelet coefficients
    pub regularity_index: F,
    /// Dominant scale from wavelet decomposition
    pub dominant_scale: usize,
    /// Multi-resolution analysis features
    pub mra_features: MultiResolutionFeatures<F>,
    /// Time-frequency analysis features
    pub time_frequency_features: TimeFrequencyFeatures<F>,
    /// Wavelet coefficient statistics
    pub coefficient_stats: WaveletCoefficientStats<F>,
}

impl<F> Default for WaveletFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            energy_bands: Vec::new(),
            relative_energy: Vec::new(),
            wavelet_entropy: F::zero(),
            wavelet_variance: F::zero(),
            regularity_index: F::zero(),
            dominant_scale: 0,
            mra_features: MultiResolutionFeatures::default(),
            time_frequency_features: TimeFrequencyFeatures::default(),
            coefficient_stats: WaveletCoefficientStats::default(),
        }
    }
}

/// Multi-resolution analysis features from wavelet decomposition
#[derive(Debug, Clone)]
pub struct MultiResolutionFeatures<F> {
    /// Energy per resolution level
    pub level_energies: Vec<F>,
    /// Relative energy per level
    pub level_relative_energies: Vec<F>,
    /// Energy distribution entropy across levels
    pub level_entropy: F,
    /// Dominant resolution level
    pub dominant_level: usize,
    /// Coefficient of variation across levels
    pub level_cv: F,
}

impl<F> Default for MultiResolutionFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            level_energies: Vec::new(),
            level_relative_energies: Vec::new(),
            level_entropy: F::zero(),
            dominant_level: 0,
            level_cv: F::zero(),
        }
    }
}

/// Time-frequency analysis features from continuous wavelet transform
#[derive(Debug, Clone)]
pub struct TimeFrequencyFeatures<F> {
    /// Instantaneous frequency estimates
    pub instantaneous_frequencies: Vec<F>,
    /// Time-localized energy concentrations
    pub energy_concentrations: Vec<F>,
    /// Frequency content stability over time
    pub frequency_stability: F,
    /// Scalogram entropy (time-frequency entropy)
    pub scalogram_entropy: F,
    /// Peak frequency evolution over time
    pub frequency_evolution: Vec<F>,
}

impl<F> Default for TimeFrequencyFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            instantaneous_frequencies: Vec::new(),
            energy_concentrations: Vec::new(),
            frequency_stability: F::zero(),
            scalogram_entropy: F::zero(),
            frequency_evolution: Vec::new(),
        }
    }
}

/// Statistical features of wavelet coefficients
#[derive(Debug, Clone)]
pub struct WaveletCoefficientStats<F> {
    /// Mean of coefficients per level
    pub level_means: Vec<F>,
    /// Standard deviation of coefficients per level
    pub level_stds: Vec<F>,
    /// Skewness of coefficients per level
    pub level_skewness: Vec<F>,
    /// Kurtosis of coefficients per level
    pub level_kurtosis: Vec<F>,
    /// Maximum coefficient magnitude per level
    pub level_max_magnitudes: Vec<F>,
    /// Zero-crossing rate per level
    pub level_zero_crossings: Vec<usize>,
}

impl<F> Default for WaveletCoefficientStats<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            level_means: Vec::new(),
            level_stds: Vec::new(),
            level_skewness: Vec::new(),
            level_kurtosis: Vec::new(),
            level_max_magnitudes: Vec::new(),
            level_zero_crossings: Vec::new(),
        }
    }
}

/// Wavelet denoising features
#[derive(Debug, Clone)]
pub struct WaveletDenoisingFeatures<F> {
    /// Signal-to-noise ratio improvement
    pub snr_improvement: F,
    /// Energy preserved after denoising
    pub energy_preserved: F,
    /// Number of coefficients thresholded
    pub coefficients_thresholded: usize,
    /// Optimal threshold value used
    pub optimal_threshold: F,
    /// Mean squared error reduction
    pub mse_reduction: F,
}

impl<F> Default for WaveletDenoisingFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            snr_improvement: F::zero(),
            energy_preserved: F::zero(),
            coefficients_thresholded: 0,
            optimal_threshold: F::zero(),
            mse_reduction: F::zero(),
        }
    }
}

// =============================================================================
// Main Calculation Functions
// =============================================================================

/// Calculate comprehensive wavelet-based features
///
/// This function performs wavelet decomposition and extracts various features
/// including energy distribution, entropy measures, regularity indices,
/// and time-frequency characteristics.
///
/// # Mathematical Background
///
/// The Discrete Wavelet Transform (DWT) decomposes a signal into different
/// frequency bands (scales). For a signal x(t), the DWT coefficients are:
///
/// ```text
/// W(j,k) = ∑ x(n) ψ*_{j,k}(n)
/// ```
///
/// where ψ_{j,k} are the wavelet basis functions at scale j and position k.
///
/// The Continuous Wavelet Transform (CWT) provides time-frequency analysis:
///
/// ```text
/// CWT(a,b) = (1/√a) ∫ x(t) ψ*((t-b)/a) dt
/// ```
///
/// where a is the scale parameter and b is the translation parameter.
///
/// # Arguments
///
/// * `ts` - Input time series data
/// * `config` - Wavelet analysis configuration
///
/// # Returns
///
/// Comprehensive wavelet features including energy distribution,
/// entropy measures, and time-frequency characteristics.
pub fn calculate_wavelet_features<F>(
    ts: &Array1<F>,
    config: &WaveletConfig,
) -> Result<WaveletFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone + ndarray::ScalarOperand,
{
    let n = ts.len();
    if n < 8 {
        return Ok(WaveletFeatures::default());
    }

    // Perform Discrete Wavelet Transform
    let dwt_result = discrete_wavelet_transform(ts, config)?;

    // Calculate energy-based features
    let energy_bands = calculate_wavelet_energy_bands(&dwt_result.coefficients)?;
    let relative_energy = calculate_relative_wavelet_energy(&energy_bands)?;

    // Calculate wavelet entropy
    let wavelet_entropy = calculate_wavelet_entropy(&dwt_result.coefficients)?;

    // Calculate wavelet variance
    let wavelet_variance = calculate_wavelet_variance(&dwt_result.coefficients)?;

    // Calculate regularity index
    let regularity_index = calculate_regularity_index(&dwt_result.coefficients)?;

    // Find dominant scale
    let dominant_scale = find_dominant_wavelet_scale(&energy_bands);

    // Calculate multi-resolution analysis features
    let mra_features = calculate_mra_features(&dwt_result)?;

    // Calculate time-frequency features (CWT-based)
    let time_frequency_features = if config.calculate_cwt {
        calculate_time_frequency_features(ts, config)?
    } else {
        TimeFrequencyFeatures::default()
    };

    // Calculate coefficient statistics
    let coefficient_stats = calculate_coefficient_statistics(&dwt_result.coefficients)?;

    Ok(WaveletFeatures {
        energy_bands,
        relative_energy,
        wavelet_entropy,
        wavelet_variance,
        regularity_index,
        dominant_scale,
        mra_features,
        time_frequency_features,
        coefficient_stats,
    })
}

// =============================================================================
// DWT Implementation
// =============================================================================

/// Result of Discrete Wavelet Transform
#[derive(Debug, Clone)]
struct DWTResult<F> {
    /// Wavelet coefficients organized by decomposition level
    /// coefficients[0] = approximation coefficients (lowest frequency)
    /// coefficients[1..n] = detail coefficients from level 1 to n
    coefficients: Vec<Array1<F>>,
    /// Number of decomposition levels
    #[allow(dead_code)]
    levels: usize,
    /// Original signal length
    #[allow(dead_code)]
    original_length: usize,
}

/// Perform Discrete Wavelet Transform
///
/// Implements a simplified DWT using Haar wavelets or Daubechies wavelets.
/// This is a basic implementation for demonstration purposes.
/// In production, you would typically use a specialized wavelet library.
fn discrete_wavelet_transform<F>(signal: &Array1<F>, config: &WaveletConfig) -> Result<DWTResult<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = signal.len();
    let max_levels = (n as f64).log2().floor() as usize - 1;
    let levels = config.levels.min(max_levels).max(1);

    let mut coefficients = Vec::with_capacity(levels + 1);
    let mut current_signal = signal.clone();

    // Get wavelet filter coefficients
    let (h, g) = get_wavelet_filters(&config.family)?;

    // Perform multilevel decomposition
    for _level in 0..levels {
        let (approx, detail) = wavelet_decompose_level(&current_signal, &h, &g)?;

        // Store detail coefficients for this level
        coefficients.push(detail);

        // Use approximation for next level
        current_signal = approx;

        // Stop if signal becomes too short
        if current_signal.len() < 4 {
            break;
        }
    }

    // Store final approximation coefficients
    coefficients.insert(0, current_signal);

    Ok(DWTResult {
        coefficients,
        levels,
        original_length: n,
    })
}

/// Get wavelet filter coefficients for different wavelet families
fn get_wavelet_filters<F>(family: &WaveletFamily) -> Result<(Array1<F>, Array1<F>)>
where
    F: Float + FromPrimitive,
{
    match family {
        WaveletFamily::Haar => {
            // Haar wavelet filters
            let sqrt_2_inv = F::from(std::f64::consts::FRAC_1_SQRT_2).unwrap();
            let h = Array1::from_vec(vec![sqrt_2_inv, sqrt_2_inv]);
            let g = Array1::from_vec(vec![-sqrt_2_inv, sqrt_2_inv]);
            Ok((h, g))
        }
        WaveletFamily::Daubechies(n) => {
            match n {
                2 => {
                    // db2 (same as Haar)
                    let sqrt_2_inv = F::from(std::f64::consts::FRAC_1_SQRT_2).unwrap();
                    let h = Array1::from_vec(vec![sqrt_2_inv, sqrt_2_inv]);
                    let g = Array1::from_vec(vec![-sqrt_2_inv, sqrt_2_inv]);
                    Ok((h, g))
                }
                4 => {
                    // db4 Daubechies-4 coefficients
                    let h = Array1::from_vec(vec![
                        F::from(0.48296291314469025).unwrap(),
                        F::from(0.8365163037378079).unwrap(),
                        F::from(0.22414386804185735).unwrap(),
                        F::from(-0.12940952255092145).unwrap(),
                    ]);
                    let g = Array1::from_vec(vec![
                        F::from(-0.12940952255092145).unwrap(),
                        F::from(-0.22414386804185735).unwrap(),
                        F::from(0.8365163037378079).unwrap(),
                        F::from(-0.48296291314469025).unwrap(),
                    ]);
                    Ok((h, g))
                }
                6 => {
                    // db6 Daubechies-6 coefficients
                    let h = Array1::from_vec(vec![
                        F::from(0.3326705529509569).unwrap(),
                        F::from(0.8068915093133388).unwrap(),
                        F::from(0.4598775021193313).unwrap(),
                        F::from(-0.13501102001039084).unwrap(),
                        F::from(-0.08544127388224149).unwrap(),
                        F::from(0.035226291882100656).unwrap(),
                    ]);
                    let g = Array1::from_vec(vec![
                        F::from(0.035226291882100656).unwrap(),
                        F::from(0.08544127388224149).unwrap(),
                        F::from(-0.13501102001039084).unwrap(),
                        F::from(-0.4598775021193313).unwrap(),
                        F::from(0.8068915093133388).unwrap(),
                        F::from(-0.3326705529509569).unwrap(),
                    ]);
                    Ok((h, g))
                }
                _ => {
                    // Default to db4 for unsupported orders
                    let h = Array1::from_vec(vec![
                        F::from(0.48296291314469025).unwrap(),
                        F::from(0.8365163037378079).unwrap(),
                        F::from(0.22414386804185735).unwrap(),
                        F::from(-0.12940952255092145).unwrap(),
                    ]);
                    let g = Array1::from_vec(vec![
                        F::from(-0.12940952255092145).unwrap(),
                        F::from(-0.22414386804185735).unwrap(),
                        F::from(0.8365163037378079).unwrap(),
                        F::from(-0.48296291314469025).unwrap(),
                    ]);
                    Ok((h, g))
                }
            }
        }
        _ => {
            // Default to Haar for unsupported families
            let h = Array1::from_vec(vec![
                F::from(std::f64::consts::FRAC_1_SQRT_2).unwrap(),
                F::from(std::f64::consts::FRAC_1_SQRT_2).unwrap(),
            ]);
            let g = Array1::from_vec(vec![
                F::from(-std::f64::consts::FRAC_1_SQRT_2).unwrap(),
                F::from(std::f64::consts::FRAC_1_SQRT_2).unwrap(),
            ]);
            Ok((h, g))
        }
    }
}

/// Perform one level of wavelet decomposition
fn wavelet_decompose_level<F>(
    signal: &Array1<F>,
    h: &Array1<F>, // Low-pass filter
    g: &Array1<F>, // High-pass filter
) -> Result<(Array1<F>, Array1<F>)>
where
    F: Float + FromPrimitive + Clone,
{
    let n = signal.len();
    let filter_len = h.len();

    if n < filter_len {
        return Err(TimeSeriesError::InsufficientData {
            message: "Signal too short for wavelet decomposition".to_string(),
            required: filter_len,
            actual: n,
        });
    }

    // Convolve with filters and downsample
    let approx_len = (n + filter_len - 1) / 2;
    let detail_len = approx_len;

    let mut approx = Array1::zeros(approx_len);
    let mut detail = Array1::zeros(detail_len);

    let mut approx_idx = 0;
    let mut detail_idx = 0;

    // Convolution with downsampling by 2
    for i in (0..n).step_by(2) {
        let mut approx_val = F::zero();
        let mut detail_val = F::zero();

        for j in 0..filter_len {
            let signal_idx = if i + j < n { i + j } else { n - 1 };

            approx_val = approx_val + h[j] * signal[signal_idx];
            detail_val = detail_val + g[j] * signal[signal_idx];
        }

        if approx_idx < approx_len {
            approx[approx_idx] = approx_val;
            approx_idx += 1;
        }

        if detail_idx < detail_len {
            detail[detail_idx] = detail_val;
            detail_idx += 1;
        }
    }

    Ok((approx, detail))
}

// =============================================================================
// Energy and Entropy Analysis
// =============================================================================

/// Calculate energy in each wavelet frequency band
fn calculate_wavelet_energy_bands<F>(coefficients: &[Array1<F>]) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    let mut energy_bands = Vec::with_capacity(coefficients.len());

    for coeff_level in coefficients {
        let energy = coeff_level.mapv(|x| x * x).sum();
        energy_bands.push(energy);
    }

    Ok(energy_bands)
}

/// Calculate relative wavelet energy (normalized energy distribution)
fn calculate_relative_wavelet_energy<F>(energy_bands: &[F]) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    let total_energy: F = energy_bands.iter().fold(F::zero(), |acc, &x| acc + x);

    if total_energy <= F::zero() {
        return Ok(vec![F::zero(); energy_bands.len()]);
    }

    let relative_energy = energy_bands
        .iter()
        .map(|&energy| energy / total_energy)
        .collect();

    Ok(relative_energy)
}

/// Calculate wavelet entropy based on energy distribution
///
/// Wavelet entropy measures the disorder in the wavelet coefficient
/// energy distribution across different scales.
///
/// ```text
/// WE = -∑ p_j * log(p_j)
/// ```
///
/// where p_j is the relative energy at scale j.
fn calculate_wavelet_entropy<F>(coefficients: &[Array1<F>]) -> Result<F>
where
    F: Float + FromPrimitive,
{
    let energy_bands = calculate_wavelet_energy_bands(coefficients)?;
    let relative_energy = calculate_relative_wavelet_energy(&energy_bands)?;

    let mut entropy = F::zero();
    for &p in &relative_energy {
        if p > F::zero() {
            entropy = entropy - p * p.ln();
        }
    }

    Ok(entropy)
}

/// Calculate wavelet variance as a measure of signal variability
fn calculate_wavelet_variance<F>(coefficients: &[Array1<F>]) -> Result<F>
where
    F: Float + FromPrimitive,
{
    let mut total_variance = F::zero();
    let mut total_count = 0;

    // Skip the first level (approximation coefficients) and only use detail coefficients
    for coeff_level in coefficients.iter().skip(1) {
        if coeff_level.len() > 1 {
            let mean = coeff_level.sum() / F::from(coeff_level.len()).unwrap();
            let variance = coeff_level.mapv(|x| (x - mean) * (x - mean)).sum()
                / F::from(coeff_level.len() - 1).unwrap();

            total_variance = total_variance + variance;
            total_count += 1;
        }
    }

    if total_count > 0 {
        Ok(total_variance / F::from(total_count).unwrap())
    } else {
        Ok(F::zero())
    }
}

/// Calculate regularity index based on wavelet coefficients
///
/// The regularity index measures the smoothness/regularity of the signal
/// based on the decay of wavelet coefficients across scales.
fn calculate_regularity_index<F>(coefficients: &[Array1<F>]) -> Result<F>
where
    F: Float + FromPrimitive,
{
    if coefficients.len() < 2 {
        return Ok(F::zero());
    }

    let mut scale_energies = Vec::new();

    // Calculate log of average energy per scale
    for (scale, coeff_level) in coefficients.iter().enumerate().skip(1) {
        if !coeff_level.is_empty() {
            let avg_energy =
                coeff_level.mapv(|x| x * x).sum() / F::from(coeff_level.len()).unwrap();

            if avg_energy > F::zero() {
                let log_energy = avg_energy.ln();
                let log_scale = F::from(scale).unwrap().ln();
                scale_energies.push((log_scale, log_energy));
            }
        }
    }

    if scale_energies.len() < 2 {
        return Ok(F::zero());
    }

    // Linear regression to estimate slope (regularity)
    let n = F::from(scale_energies.len()).unwrap();
    let sum_x: F = scale_energies
        .iter()
        .map(|(x, _)| *x)
        .fold(F::zero(), |acc, x| acc + x);
    let sum_y: F = scale_energies
        .iter()
        .map(|(_, y)| *y)
        .fold(F::zero(), |acc, y| acc + y);
    let sum_xy: F = scale_energies
        .iter()
        .map(|(x, y)| *x * *y)
        .fold(F::zero(), |acc, xy| acc + xy);
    let sum_xx: F = scale_energies
        .iter()
        .map(|(x, _)| *x * *x)
        .fold(F::zero(), |acc, xx| acc + xx);

    let denominator = n * sum_xx - sum_x * sum_x;
    if denominator.abs() < F::from(1e-10).unwrap() {
        return Ok(F::zero());
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;

    // Regularity index is related to the negative slope
    Ok(-slope)
}

/// Find the dominant scale (frequency band) based on energy distribution
fn find_dominant_wavelet_scale<F>(energy_bands: &[F]) -> usize
where
    F: Float + PartialOrd,
{
    energy_bands
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

// =============================================================================
// Multi-Resolution Analysis
// =============================================================================

/// Calculate multi-resolution analysis features
fn calculate_mra_features<F>(dwt_result: &DWTResult<F>) -> Result<MultiResolutionFeatures<F>>
where
    F: Float + FromPrimitive,
{
    let level_energies = calculate_wavelet_energy_bands(&dwt_result.coefficients)?;
    let level_relative_energies = calculate_relative_wavelet_energy(&level_energies)?;

    // Calculate entropy across levels
    let mut level_entropy = F::zero();
    for &p in &level_relative_energies {
        if p > F::zero() {
            level_entropy = level_entropy - p * p.ln();
        }
    }

    // Find dominant level
    let dominant_level = find_dominant_wavelet_scale(&level_energies);

    // Calculate coefficient of variation across levels
    let mean_energy = level_energies.iter().fold(F::zero(), |acc, &x| acc + x)
        / F::from(level_energies.len()).unwrap();

    let variance_energy = level_energies.iter().fold(F::zero(), |acc, &x| {
        acc + (x - mean_energy) * (x - mean_energy)
    }) / F::from(level_energies.len()).unwrap();

    let level_cv = if mean_energy > F::zero() {
        variance_energy.sqrt() / mean_energy
    } else {
        F::zero()
    };

    Ok(MultiResolutionFeatures {
        level_energies,
        level_relative_energies,
        level_entropy,
        dominant_level,
        level_cv,
    })
}

// =============================================================================
// Continuous Wavelet Transform (CWT)
// =============================================================================

/// Calculate time-frequency features using simplified CWT
fn calculate_time_frequency_features<F>(
    signal: &Array1<F>,
    config: &WaveletConfig,
) -> Result<TimeFrequencyFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = signal.len();
    if n < 16 {
        return Ok(TimeFrequencyFeatures::default());
    }

    // Simplified CWT using Morlet wavelet
    let scales = generate_cwt_scales(config);
    let cwt_matrix = compute_simplified_cwt(signal, &scales)?;

    // Calculate instantaneous frequencies (simplified)
    let instantaneous_frequencies = estimate_instantaneous_frequencies(&cwt_matrix, &scales)?;

    // Calculate energy concentrations
    let energy_concentrations = calculate_energy_concentrations(&cwt_matrix)?;

    // Calculate frequency stability
    let frequency_stability = calculate_frequency_stability(&instantaneous_frequencies)?;

    // Calculate scalogram entropy
    let scalogram_entropy = calculate_scalogram_entropy(&cwt_matrix)?;

    // Calculate frequency evolution
    let frequency_evolution = calculate_frequency_evolution(&cwt_matrix, &scales)?;

    Ok(TimeFrequencyFeatures {
        instantaneous_frequencies,
        energy_concentrations,
        frequency_stability,
        scalogram_entropy,
        frequency_evolution,
    })
}

/// Generate scales for CWT analysis
fn generate_cwt_scales(config: &WaveletConfig) -> Vec<f64> {
    let (min_scale, max_scale) = config.cwt_scales.unwrap_or((1.0, 32.0));
    let count = config.cwt_scale_count;

    let log_min = min_scale.ln();
    let log_max = max_scale.ln();
    let step = (log_max - log_min) / (count - 1) as f64;

    (0..count)
        .map(|i| (log_min + i as f64 * step).exp())
        .collect()
}

/// Compute simplified CWT using Morlet-like wavelet
fn compute_simplified_cwt<F>(signal: &Array1<F>, scales: &[f64]) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Clone,
{
    let n = signal.len();
    let n_scales = scales.len();
    let mut cwt_matrix = Array2::zeros((n_scales, n));

    for (scale_idx, &scale) in scales.iter().enumerate() {
        // Simple wavelet: modulated Gaussian
        let omega0 = 6.0; // Central frequency
        let wavelet_support = (8.0 * scale) as usize;

        for t in 0..n {
            let mut cwt_value = F::zero();
            let mut norm = F::zero();

            for tau in 0..wavelet_support {
                let t_shifted = t as isize - tau as isize;
                if t_shifted >= 0 && (t_shifted as usize) < n {
                    let signal_idx = t_shifted as usize;

                    // Simplified Morlet wavelet
                    let t_norm = (tau as f64) / scale;
                    let envelope = (-0.5 * t_norm * t_norm).exp();
                    let oscillation = (omega0 * t_norm).cos();
                    let wavelet_val = F::from(envelope * oscillation).unwrap();

                    cwt_value = cwt_value + signal[signal_idx] * wavelet_val;
                    norm = norm + wavelet_val * wavelet_val;
                }
            }

            // Normalize
            if norm > F::zero() {
                cwt_matrix[[scale_idx, t]] = cwt_value / norm.sqrt();
            }
        }
    }

    Ok(cwt_matrix)
}

/// Estimate instantaneous frequencies from CWT
fn estimate_instantaneous_frequencies<F>(cwt_matrix: &Array2<F>, scales: &[f64]) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + PartialOrd,
{
    let (_, n_time) = cwt_matrix.dim();
    let mut inst_freqs = Vec::with_capacity(n_time);

    for t in 0..n_time {
        let time_slice = cwt_matrix.column(t);

        // Find scale with maximum magnitude
        let max_scale_idx = time_slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Convert scale to frequency (simplified)
        let scale = scales[max_scale_idx];
        let freq = 1.0 / scale; // Simplified frequency estimation
        inst_freqs.push(F::from(freq).unwrap());
    }

    Ok(inst_freqs)
}

/// Calculate energy concentrations from CWT
fn calculate_energy_concentrations<F>(cwt_matrix: &Array2<F>) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    let (_, n_time) = cwt_matrix.dim();
    let mut concentrations = Vec::with_capacity(n_time);

    for t in 0..n_time {
        let time_slice = cwt_matrix.column(t);
        let energy = time_slice.mapv(|x| x * x).sum();
        concentrations.push(energy);
    }

    Ok(concentrations)
}

/// Calculate frequency stability over time
fn calculate_frequency_stability<F>(instantaneous_frequencies: &[F]) -> Result<F>
where
    F: Float + FromPrimitive,
{
    if instantaneous_frequencies.len() < 2 {
        return Ok(F::zero());
    }

    let n = instantaneous_frequencies.len();
    let mean = instantaneous_frequencies
        .iter()
        .fold(F::zero(), |acc, &x| acc + x)
        / F::from(n).unwrap();

    let variance = instantaneous_frequencies
        .iter()
        .fold(F::zero(), |acc, &x| acc + (x - mean) * (x - mean))
        / F::from(n - 1).unwrap();

    // Stability is inverse of coefficient of variation
    if mean > F::zero() {
        let cv = variance.sqrt() / mean;
        Ok(F::one() / (F::one() + cv))
    } else {
        Ok(F::zero())
    }
}

/// Calculate scalogram entropy
fn calculate_scalogram_entropy<F>(cwt_matrix: &Array2<F>) -> Result<F>
where
    F: Float + FromPrimitive,
{
    let total_energy = cwt_matrix.mapv(|x| x * x).sum();

    if total_energy <= F::zero() {
        return Ok(F::zero());
    }

    let mut entropy = F::zero();
    for &coeff in cwt_matrix.iter() {
        let energy = coeff * coeff;
        if energy > F::zero() {
            let p = energy / total_energy;
            entropy = entropy - p * p.ln();
        }
    }

    Ok(entropy)
}

/// Calculate frequency evolution over time
fn calculate_frequency_evolution<F>(cwt_matrix: &Array2<F>, scales: &[f64]) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + PartialOrd,
{
    let (_, n_time) = cwt_matrix.dim();
    let mut evolution = Vec::with_capacity(n_time);

    for t in 0..n_time {
        let time_slice = cwt_matrix.column(t);

        // Calculate weighted average frequency
        let mut weighted_freq = F::zero();
        let mut total_weight = F::zero();

        for (scale_idx, &scale) in scales.iter().enumerate() {
            let weight = time_slice[scale_idx] * time_slice[scale_idx];
            let freq = F::from(1.0 / scale).unwrap();

            weighted_freq = weighted_freq + weight * freq;
            total_weight = total_weight + weight;
        }

        if total_weight > F::zero() {
            evolution.push(weighted_freq / total_weight);
        } else {
            evolution.push(F::zero());
        }
    }

    Ok(evolution)
}

// =============================================================================
// Coefficient Statistics
// =============================================================================

/// Calculate statistical features of wavelet coefficients
fn calculate_coefficient_statistics<F>(
    coefficients: &[Array1<F>],
) -> Result<WaveletCoefficientStats<F>>
where
    F: Float + FromPrimitive + PartialOrd,
{
    let mut level_means = Vec::new();
    let mut level_stds = Vec::new();
    let mut level_skewness = Vec::new();
    let mut level_kurtosis = Vec::new();
    let mut level_max_magnitudes = Vec::new();
    let mut level_zero_crossings = Vec::new();

    for coeff_level in coefficients {
        if coeff_level.is_empty() {
            level_means.push(F::zero());
            level_stds.push(F::zero());
            level_skewness.push(F::zero());
            level_kurtosis.push(F::zero());
            level_max_magnitudes.push(F::zero());
            level_zero_crossings.push(0);
            continue;
        }

        let n = coeff_level.len();
        let n_f = F::from(n).unwrap();

        // Mean
        let mean = coeff_level.sum() / n_f;
        level_means.push(mean);

        // Standard deviation
        let variance = coeff_level.mapv(|x| (x - mean) * (x - mean)).sum() / n_f;
        let std_dev = variance.sqrt();
        level_stds.push(std_dev);

        // Skewness and kurtosis
        if std_dev > F::zero() {
            let mut sum_cube = F::zero();
            let mut sum_fourth = F::zero();

            for &x in coeff_level.iter() {
                let norm_dev = (x - mean) / std_dev;
                let norm_dev_sq = norm_dev * norm_dev;
                sum_cube = sum_cube + norm_dev * norm_dev_sq;
                sum_fourth = sum_fourth + norm_dev_sq * norm_dev_sq;
            }

            let skewness = sum_cube / n_f;
            let kurtosis = sum_fourth / n_f - F::from(3.0).unwrap();

            level_skewness.push(skewness);
            level_kurtosis.push(kurtosis);
        } else {
            level_skewness.push(F::zero());
            level_kurtosis.push(F::zero());
        }

        // Maximum magnitude
        let max_magnitude = coeff_level
            .iter()
            .map(|&x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(F::zero());
        level_max_magnitudes.push(max_magnitude);

        // Zero crossings
        let mut zero_crossings = 0;
        for i in 1..coeff_level.len() {
            if (coeff_level[i - 1] >= F::zero()) != (coeff_level[i] >= F::zero()) {
                zero_crossings += 1;
            }
        }
        level_zero_crossings.push(zero_crossings);
    }

    Ok(WaveletCoefficientStats {
        level_means,
        level_stds,
        level_skewness,
        level_kurtosis,
        level_max_magnitudes,
        level_zero_crossings,
    })
}

// =============================================================================
// Wavelet Denoising
// =============================================================================

/// Perform wavelet denoising and extract denoising-related features
///
/// # Arguments
///
/// * `signal` - Input noisy signal
/// * `config` - Wavelet configuration including denoising method
///
/// # Returns
///
/// Tuple of (denoised_signal, denoising_features)
pub fn wavelet_denoise<F>(
    signal: &Array1<F>,
    config: &WaveletConfig,
) -> Result<(Array1<F>, WaveletDenoisingFeatures<F>)>
where
    F: Float + FromPrimitive + Debug + Clone + PartialOrd,
{
    // Perform DWT
    let dwt_result = discrete_wavelet_transform(signal, config)?;

    // Calculate optimal threshold
    let threshold =
        calculate_optimal_threshold(&dwt_result.coefficients, &config.denoising_method)?;

    // Apply thresholding
    let (thresholded_coeffs, coefficients_thresholded) = apply_thresholding(
        &dwt_result.coefficients,
        threshold,
        &config.denoising_method,
    )?;

    // Reconstruct signal (simplified - in practice would use inverse DWT)
    let denoised_signal = reconstruct_signal_simplified(&thresholded_coeffs)?;

    // Calculate denoising features
    let original_energy = signal.mapv(|x| x * x).sum();
    let denoised_energy = denoised_signal.mapv(|x| x * x).sum();
    let energy_preserved = if original_energy > F::zero() {
        denoised_energy / original_energy
    } else {
        F::zero()
    };

    // Calculate SNR improvement (simplified)
    let snr_improvement = calculate_snr_improvement(signal, &denoised_signal)?;

    // Calculate MSE reduction (simplified)
    let mse_reduction = calculate_mse_reduction(signal, &denoised_signal)?;

    let features = WaveletDenoisingFeatures {
        snr_improvement,
        energy_preserved,
        coefficients_thresholded,
        optimal_threshold: threshold,
        mse_reduction,
    };

    Ok((denoised_signal, features))
}

/// Calculate optimal threshold for denoising
fn calculate_optimal_threshold<F>(coefficients: &[Array1<F>], method: &DenoisingMethod) -> Result<F>
where
    F: Float + FromPrimitive + PartialOrd,
{
    // Calculate noise level estimate using MAD of finest detail coefficients
    let finest_detail = &coefficients[coefficients.len() - 1];
    if finest_detail.is_empty() {
        return Ok(F::zero());
    }

    let mut sorted_coeffs: Vec<F> = finest_detail.iter().map(|&x| x.abs()).collect();
    sorted_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_idx = sorted_coeffs.len() / 2;
    let mad = if sorted_coeffs.len() % 2 == 0 {
        (sorted_coeffs[median_idx - 1] + sorted_coeffs[median_idx]) / F::from(2.0).unwrap()
    } else {
        sorted_coeffs[median_idx]
    };

    let sigma = mad / F::from(0.6745).unwrap(); // MAD to standard deviation conversion

    match method {
        DenoisingMethod::Hard | DenoisingMethod::Soft => {
            // Universal threshold
            let n = F::from(finest_detail.len()).unwrap();
            Ok(sigma * (F::from(2.0).unwrap() * n.ln()).sqrt())
        }
        DenoisingMethod::Sure => {
            // SURE threshold (simplified)
            Ok(sigma * F::from(1.5).unwrap())
        }
        DenoisingMethod::Minimax => {
            // Minimax threshold (simplified)
            Ok(sigma * F::from(0.8).unwrap())
        }
    }
}

/// Apply thresholding to wavelet coefficients
fn apply_thresholding<F>(
    coefficients: &[Array1<F>],
    threshold: F,
    method: &DenoisingMethod,
) -> Result<(Vec<Array1<F>>, usize)>
where
    F: Float + FromPrimitive + PartialOrd + Clone,
{
    let mut thresholded_coeffs = Vec::new();
    let mut total_thresholded = 0;

    for (level, coeff_level) in coefficients.iter().enumerate() {
        if level == 0 {
            // Don't threshold approximation coefficients
            thresholded_coeffs.push(coeff_level.clone());
            continue;
        }

        let mut thresholded_level = Array1::zeros(coeff_level.len());
        let mut _level_thresholded = 0;

        for (i, &coeff) in coeff_level.iter().enumerate() {
            let abs_coeff = coeff.abs();

            if abs_coeff <= threshold {
                _level_thresholded += 1;
                total_thresholded += 1;
                // Coefficient is set to zero (already initialized)
            } else {
                thresholded_level[i] = match method {
                    DenoisingMethod::Hard => coeff,
                    DenoisingMethod::Soft => {
                        let sign = if coeff >= F::zero() {
                            F::one()
                        } else {
                            -F::one()
                        };
                        sign * (abs_coeff - threshold)
                    }
                    DenoisingMethod::Sure | DenoisingMethod::Minimax => {
                        // Use soft thresholding for these methods
                        let sign = if coeff >= F::zero() {
                            F::one()
                        } else {
                            -F::one()
                        };
                        sign * (abs_coeff - threshold)
                    }
                };
            }
        }

        thresholded_coeffs.push(thresholded_level);
    }

    Ok((thresholded_coeffs, total_thresholded))
}

/// Simplified signal reconstruction from thresholded coefficients
fn reconstruct_signal_simplified<F>(coefficients: &[Array1<F>]) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Clone,
{
    if coefficients.is_empty() {
        return Ok(Array1::zeros(0));
    }

    // Simplified reconstruction: use approximation coefficients scaled by levels
    let approx_coeffs = &coefficients[0];
    let mut reconstructed = approx_coeffs.clone();

    // Add scaled detail coefficients (simplified approach)
    for (level, detail_coeffs) in coefficients.iter().enumerate().skip(1) {
        let scale_factor = F::from(2.0_f64.powi(level as i32)).unwrap();

        // Upsample and add details (very simplified)
        for (i, &detail) in detail_coeffs.iter().enumerate() {
            let target_idx = i.min(reconstructed.len() - 1);
            reconstructed[target_idx] = reconstructed[target_idx] + detail / scale_factor;
        }
    }

    Ok(reconstructed)
}

/// Calculate SNR improvement after denoising
fn calculate_snr_improvement<F>(original: &Array1<F>, denoised: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive,
{
    let signal_power = original.mapv(|x| x * x).sum();
    let noise_power = original
        .iter()
        .zip(denoised.iter())
        .fold(F::zero(), |acc, (&orig, &den)| {
            let diff = orig - den;
            acc + diff * diff
        });

    if noise_power > F::zero() && signal_power > F::zero() {
        let snr = (signal_power / noise_power).ln() / F::from(10.0).unwrap().ln()
            * F::from(10.0).unwrap();
        Ok(snr)
    } else {
        Ok(F::zero())
    }
}

/// Calculate MSE reduction after denoising
fn calculate_mse_reduction<F>(original: &Array1<F>, denoised: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive,
{
    let n = F::from(original.len()).unwrap();
    let mse = original
        .iter()
        .zip(denoised.iter())
        .fold(F::zero(), |acc, (&orig, &den)| {
            let diff = orig - den;
            acc + diff * diff
        })
        / n;

    // Return normalized MSE reduction
    let signal_variance = original.mapv(|x| x * x).sum() / n;
    if signal_variance > F::zero() {
        Ok(F::one() - (mse / signal_variance))
    } else {
        Ok(F::zero())
    }
}
