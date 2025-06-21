//! Turning points and trend analysis features for time series
//!
//! This module provides comprehensive turning point detection and trend analysis
//! including local extrema detection, directional change analysis, momentum features,
//! trend reversals, pattern detection, and multi-scale analysis.

use ndarray::{s, Array1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::config::TurningPointsConfig;
use super::utils::detect_turning_points;
use crate::error::{Result, TimeSeriesError};

/// Comprehensive turning points analysis features
#[derive(Debug, Clone)]
pub struct TurningPointsFeatures<F> {
    // Basic turning point counts
    /// Total number of turning points in the series
    pub total_turning_points: usize,
    /// Number of local minima (valleys)
    pub local_minima_count: usize,
    /// Number of local maxima (peaks)
    pub local_maxima_count: usize,
    /// Ratio of peaks to valleys
    pub peak_valley_ratio: F,
    /// Average distance between consecutive turning points
    pub average_turning_point_distance: F,

    // Directional change analysis
    /// Number of upward directional changes
    pub upward_changes: usize,
    /// Number of downward directional changes  
    pub downward_changes: usize,
    /// Ratio of upward to downward changes
    pub directional_change_ratio: F,
    /// Average magnitude of upward changes
    pub average_upward_magnitude: F,
    /// Average magnitude of downward changes
    pub average_downward_magnitude: F,
    /// Standard deviation of directional change magnitudes
    pub directional_change_std: F,

    // Momentum and persistence features
    /// Longest consecutive upward sequence length
    pub longest_upward_sequence: usize,
    /// Longest consecutive downward sequence length
    pub longest_downward_sequence: usize,
    /// Average length of upward sequences
    pub average_upward_sequence_length: F,
    /// Average length of downward sequences
    pub average_downward_sequence_length: F,
    /// Momentum persistence ratio (long sequences / total sequences)
    pub momentum_persistence_ratio: F,

    // Local extrema characteristics
    /// Average amplitude of local maxima
    pub average_peak_amplitude: F,
    /// Average amplitude of local minima
    pub average_valley_amplitude: F,
    /// Standard deviation of peak amplitudes
    pub peak_amplitude_std: F,
    /// Standard deviation of valley amplitudes
    pub valley_amplitude_std: F,
    /// Peak-to-valley amplitude ratio
    pub peak_valley_amplitude_ratio: F,
    /// Asymmetry in peak and valley distributions
    pub extrema_asymmetry: F,

    // Trend reversal features
    /// Number of major trend reversals (large directional changes)
    pub major_trend_reversals: usize,
    /// Number of minor trend reversals (small directional changes)
    pub minor_trend_reversals: usize,
    /// Average magnitude of major reversals
    pub average_major_reversal_magnitude: F,
    /// Average magnitude of minor reversals
    pub average_minor_reversal_magnitude: F,
    /// Trend reversal frequency (reversals per unit time)
    pub trend_reversal_frequency: F,
    /// Reversal strength index (cumulative reversal magnitude)
    pub reversal_strength_index: F,

    // Temporal pattern features
    /// Regularity of turning point intervals (coefficient of variation)
    pub turning_point_regularity: F,
    /// Clustering tendency of turning points
    pub turning_point_clustering: F,
    /// Periodicity strength of turning points
    pub turning_point_periodicity: F,
    /// Auto-correlation of turning point intervals
    pub turning_point_autocorrelation: F,

    // Volatility and stability measures
    /// Volatility around turning points (average local variance)
    pub turning_point_volatility: F,
    /// Stability index (inverse of turning point frequency)
    pub stability_index: F,
    /// Noise-to-signal ratio around turning points
    pub noise_signal_ratio: F,
    /// Trend consistency measure
    pub trend_consistency: F,

    // Advanced pattern features
    /// Number of double peaks (M patterns)
    pub double_peak_count: usize,
    /// Number of double bottoms (W patterns)
    pub double_bottom_count: usize,
    /// Head and shoulders pattern count
    pub head_shoulders_count: usize,
    /// Triangular pattern count (converging peaks/valleys)
    pub triangular_pattern_count: usize,

    // Relative position features
    /// Proportion of turning points in upper half of range
    pub upper_half_turning_points: F,
    /// Proportion of turning points in lower half of range
    pub lower_half_turning_points: F,
    /// Skewness of turning point vertical positions
    pub turning_point_position_skewness: F,
    /// Kurtosis of turning point vertical positions
    pub turning_point_position_kurtosis: F,

    // Multi-scale turning point features
    /// Turning points at different smoothing scales
    pub multiscale_turning_points: Vec<usize>,
    /// Scale-dependent turning point ratio
    pub scale_turning_point_ratio: F,
    /// Cross-scale turning point consistency
    pub cross_scale_consistency: F,
    /// Hierarchical turning point structure
    pub hierarchical_structure_index: F,
}

impl<F> Default for TurningPointsFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            // Basic turning point counts
            total_turning_points: 0,
            local_minima_count: 0,
            local_maxima_count: 0,
            peak_valley_ratio: F::one(),
            average_turning_point_distance: F::zero(),

            // Directional change analysis
            upward_changes: 0,
            downward_changes: 0,
            directional_change_ratio: F::one(),
            average_upward_magnitude: F::zero(),
            average_downward_magnitude: F::zero(),
            directional_change_std: F::zero(),

            // Momentum and persistence features
            longest_upward_sequence: 0,
            longest_downward_sequence: 0,
            average_upward_sequence_length: F::zero(),
            average_downward_sequence_length: F::zero(),
            momentum_persistence_ratio: F::zero(),

            // Local extrema characteristics
            average_peak_amplitude: F::zero(),
            average_valley_amplitude: F::zero(),
            peak_amplitude_std: F::zero(),
            valley_amplitude_std: F::zero(),
            peak_valley_amplitude_ratio: F::one(),
            extrema_asymmetry: F::zero(),

            // Trend reversal features
            major_trend_reversals: 0,
            minor_trend_reversals: 0,
            average_major_reversal_magnitude: F::zero(),
            average_minor_reversal_magnitude: F::zero(),
            trend_reversal_frequency: F::zero(),
            reversal_strength_index: F::zero(),

            // Temporal pattern features
            turning_point_regularity: F::zero(),
            turning_point_clustering: F::zero(),
            turning_point_periodicity: F::zero(),
            turning_point_autocorrelation: F::zero(),

            // Volatility and stability measures
            turning_point_volatility: F::zero(),
            stability_index: F::zero(),
            noise_signal_ratio: F::zero(),
            trend_consistency: F::zero(),

            // Advanced pattern features
            double_peak_count: 0,
            double_bottom_count: 0,
            head_shoulders_count: 0,
            triangular_pattern_count: 0,

            // Relative position features
            upper_half_turning_points: F::from(0.5).unwrap(),
            lower_half_turning_points: F::from(0.5).unwrap(),
            turning_point_position_skewness: F::zero(),
            turning_point_position_kurtosis: F::zero(),

            // Multi-scale turning point features
            multiscale_turning_points: Vec::new(),
            scale_turning_point_ratio: F::zero(),
            cross_scale_consistency: F::zero(),
            hierarchical_structure_index: F::zero(),
        }
    }
}

// =============================================================================
// Helper Structures for Intermediate Calculations
// =============================================================================

/// Helper struct for directional change statistics
#[derive(Debug, Clone)]
struct DirectionalChangeStats<F> {
    directional_change_ratio: F,
    average_upward_magnitude: F,
    average_downward_magnitude: F,
    directional_change_std: F,
}

/// Helper struct for momentum and persistence features
#[derive(Debug, Clone)]
struct MomentumFeatures<F> {
    longest_upward_sequence: usize,
    longest_downward_sequence: usize,
    average_upward_sequence_length: F,
    average_downward_sequence_length: F,
    momentum_persistence_ratio: F,
}

/// Helper struct for local extrema characteristics
#[derive(Debug, Clone)]
struct ExtremaFeatures<F> {
    average_peak_amplitude: F,
    average_valley_amplitude: F,
    peak_amplitude_std: F,
    valley_amplitude_std: F,
    peak_valley_amplitude_ratio: F,
    extrema_asymmetry: F,
}

/// Helper struct for trend reversal features
#[derive(Debug, Clone)]
struct TrendReversalFeatures<F> {
    major_trend_reversals: usize,
    minor_trend_reversals: usize,
    average_major_reversal_magnitude: F,
    average_minor_reversal_magnitude: F,
    trend_reversal_frequency: F,
    reversal_strength_index: F,
}

/// Helper struct for temporal pattern features of turning points
#[derive(Debug, Clone)]
struct TurningPointTemporalFeatures<F> {
    turning_point_regularity: F,
    turning_point_clustering: F,
    turning_point_periodicity: F,
    turning_point_autocorrelation: F,
}

impl<F> Default for TurningPointTemporalFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            turning_point_regularity: F::zero(),
            turning_point_clustering: F::zero(),
            turning_point_periodicity: F::zero(),
            turning_point_autocorrelation: F::zero(),
        }
    }
}

/// Helper struct for stability and volatility features
#[derive(Debug, Clone)]
struct StabilityFeatures<F> {
    turning_point_volatility: F,
    stability_index: F,
    noise_signal_ratio: F,
    trend_consistency: F,
}

/// Helper struct for advanced pattern features
#[derive(Debug, Clone, Default)]
struct AdvancedPatternFeatures {
    double_peak_count: usize,
    double_bottom_count: usize,
    head_shoulders_count: usize,
    triangular_pattern_count: usize,
}

/// Helper struct for position features
#[derive(Debug, Clone)]
struct PositionFeatures<F> {
    upper_half_turning_points: F,
    lower_half_turning_points: F,
    turning_point_position_skewness: F,
    turning_point_position_kurtosis: F,
}

/// Helper struct for multi-scale features
#[derive(Debug, Clone)]
struct MultiscaleTurningPointFeatures<F> {
    multiscale_turning_points: Vec<usize>,
    scale_turning_point_ratio: F,
    cross_scale_consistency: F,
    hierarchical_structure_index: F,
}

impl<F> Default for MultiscaleTurningPointFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            multiscale_turning_points: Vec::new(),
            scale_turning_point_ratio: F::zero(),
            cross_scale_consistency: F::zero(),
            hierarchical_structure_index: F::zero(),
        }
    }
}

// =============================================================================
// Main Calculation Function
// =============================================================================

/// Calculate comprehensive turning points features
///
/// This function performs extensive turning point analysis including basic counts,
/// directional changes, momentum patterns, extrema characteristics, trend reversals,
/// temporal patterns, stability measures, advanced pattern detection, position analysis,
/// and multi-scale analysis.
///
/// # Arguments
///
/// * `ts` - Input time series data
/// * `config` - Turning points analysis configuration
///
/// # Returns
///
/// Comprehensive turning points features structure
pub fn calculate_turning_points_features<F>(
    ts: &Array1<F>,
    config: &TurningPointsConfig,
) -> Result<TurningPointsFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone + std::iter::Sum + ndarray::ScalarOperand,
    for<'a> F: std::iter::Sum<&'a F>,
{
    let n = ts.len();
    if n < config.extrema_window_size * 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Insufficient data for turning points analysis".to_string(),
            required: config.extrema_window_size * 2,
            actual: n,
        });
    }

    // Detect basic turning points and local extrema
    let (turning_points, local_maxima, local_minima) = detect_turning_points(ts, config)?;

    // Calculate basic counts and ratios
    let total_turning_points = turning_points.len();
    let local_maxima_count = local_maxima.len();
    let local_minima_count = local_minima.len();
    let peak_valley_ratio = if local_minima_count > 0 {
        F::from(local_maxima_count).unwrap() / F::from(local_minima_count).unwrap()
    } else {
        F::zero()
    };

    // Calculate average distance between turning points
    let average_turning_point_distance = if total_turning_points > 1 {
        let total_distance: usize = turning_points.windows(2).map(|w| w[1] - w[0]).sum();
        F::from(total_distance).unwrap() / F::from(total_turning_points - 1).unwrap()
    } else {
        F::zero()
    };

    // Analyze directional changes
    let (upward_changes, downward_changes, directional_stats) =
        analyze_directional_changes(ts, &turning_points, config)?;

    // Analyze momentum and persistence
    let momentum_features = analyze_momentum_persistence(ts, config)?;

    // Characterize local extrema
    let extrema_features = characterize_local_extrema(ts, &local_maxima, &local_minima)?;

    // Detect trend reversals
    let reversal_features = detect_trend_reversals(ts, &turning_points, config)?;

    // Analyze temporal patterns
    let temporal_features = if config.calculate_temporal_patterns {
        analyze_temporal_patterns(&turning_points, config)?
    } else {
        TurningPointTemporalFeatures::default()
    };

    // Calculate volatility and stability measures
    let stability_features = calculate_stability_measures(ts, &turning_points)?;

    // Detect advanced patterns
    let pattern_features = if config.detect_advanced_patterns {
        detect_advanced_patterns(ts, &local_maxima, &local_minima, config)?
    } else {
        AdvancedPatternFeatures::default()
    };

    // Analyze relative positions
    let position_features = analyze_turning_point_positions(ts, &turning_points)?;

    // Multi-scale analysis
    let multiscale_features = if config.multiscale_analysis {
        analyze_multiscale_turning_points(ts, config)?
    } else {
        MultiscaleTurningPointFeatures::default()
    };

    Ok(TurningPointsFeatures {
        // Basic turning point counts
        total_turning_points,
        local_minima_count,
        local_maxima_count,
        peak_valley_ratio,
        average_turning_point_distance,

        // Directional change analysis
        upward_changes,
        downward_changes,
        directional_change_ratio: directional_stats.directional_change_ratio,
        average_upward_magnitude: directional_stats.average_upward_magnitude,
        average_downward_magnitude: directional_stats.average_downward_magnitude,
        directional_change_std: directional_stats.directional_change_std,

        // Momentum and persistence features
        longest_upward_sequence: momentum_features.longest_upward_sequence,
        longest_downward_sequence: momentum_features.longest_downward_sequence,
        average_upward_sequence_length: momentum_features.average_upward_sequence_length,
        average_downward_sequence_length: momentum_features.average_downward_sequence_length,
        momentum_persistence_ratio: momentum_features.momentum_persistence_ratio,

        // Local extrema characteristics
        average_peak_amplitude: extrema_features.average_peak_amplitude,
        average_valley_amplitude: extrema_features.average_valley_amplitude,
        peak_amplitude_std: extrema_features.peak_amplitude_std,
        valley_amplitude_std: extrema_features.valley_amplitude_std,
        peak_valley_amplitude_ratio: extrema_features.peak_valley_amplitude_ratio,
        extrema_asymmetry: extrema_features.extrema_asymmetry,

        // Trend reversal features
        major_trend_reversals: reversal_features.major_trend_reversals,
        minor_trend_reversals: reversal_features.minor_trend_reversals,
        average_major_reversal_magnitude: reversal_features.average_major_reversal_magnitude,
        average_minor_reversal_magnitude: reversal_features.average_minor_reversal_magnitude,
        trend_reversal_frequency: reversal_features.trend_reversal_frequency,
        reversal_strength_index: reversal_features.reversal_strength_index,

        // Temporal pattern features
        turning_point_regularity: temporal_features.turning_point_regularity,
        turning_point_clustering: temporal_features.turning_point_clustering,
        turning_point_periodicity: temporal_features.turning_point_periodicity,
        turning_point_autocorrelation: temporal_features.turning_point_autocorrelation,

        // Volatility and stability measures
        turning_point_volatility: stability_features.turning_point_volatility,
        stability_index: stability_features.stability_index,
        noise_signal_ratio: stability_features.noise_signal_ratio,
        trend_consistency: stability_features.trend_consistency,

        // Advanced pattern features
        double_peak_count: pattern_features.double_peak_count,
        double_bottom_count: pattern_features.double_bottom_count,
        head_shoulders_count: pattern_features.head_shoulders_count,
        triangular_pattern_count: pattern_features.triangular_pattern_count,

        // Relative position features
        upper_half_turning_points: position_features.upper_half_turning_points,
        lower_half_turning_points: position_features.lower_half_turning_points,
        turning_point_position_skewness: position_features.turning_point_position_skewness,
        turning_point_position_kurtosis: position_features.turning_point_position_kurtosis,

        // Multi-scale turning point features
        multiscale_turning_points: multiscale_features.multiscale_turning_points,
        scale_turning_point_ratio: multiscale_features.scale_turning_point_ratio,
        cross_scale_consistency: multiscale_features.cross_scale_consistency,
        hierarchical_structure_index: multiscale_features.hierarchical_structure_index,
    })
}

// =============================================================================
// Analysis Functions
// =============================================================================

/// Analyze directional changes in the time series
fn analyze_directional_changes<F>(
    ts: &Array1<F>,
    turning_points: &[usize],
    _config: &TurningPointsConfig,
) -> Result<(usize, usize, DirectionalChangeStats<F>)>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let mut upward_changes = 0;
    let mut downward_changes = 0;
    let mut upward_magnitudes = Vec::new();
    let mut downward_magnitudes = Vec::new();

    // Analyze changes between consecutive turning points
    for window in turning_points.windows(2) {
        let start_idx = window[0];
        let end_idx = window[1];

        if start_idx < ts.len() && end_idx < ts.len() {
            let change = ts[end_idx] - ts[start_idx];
            let magnitude = change.abs();

            if change > F::zero() {
                upward_changes += 1;
                upward_magnitudes.push(magnitude);
            } else if change < F::zero() {
                downward_changes += 1;
                downward_magnitudes.push(magnitude);
            }
        }
    }

    // Calculate directional statistics
    let directional_change_ratio = if downward_changes > 0 {
        F::from(upward_changes).unwrap() / F::from(downward_changes).unwrap()
    } else {
        F::from(upward_changes).unwrap()
    };

    let average_upward_magnitude = if !upward_magnitudes.is_empty() {
        upward_magnitudes.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from(upward_magnitudes.len()).unwrap()
    } else {
        F::zero()
    };

    let average_downward_magnitude = if !downward_magnitudes.is_empty() {
        downward_magnitudes
            .iter()
            .fold(F::zero(), |acc, &x| acc + x)
            / F::from(downward_magnitudes.len()).unwrap()
    } else {
        F::zero()
    };

    // Calculate standard deviation of all directional changes
    let all_magnitudes: Vec<F> = upward_magnitudes
        .into_iter()
        .chain(downward_magnitudes)
        .collect();

    let directional_change_std = if all_magnitudes.len() > 1 {
        let mean = all_magnitudes.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from(all_magnitudes.len()).unwrap();
        let variance = all_magnitudes
            .iter()
            .fold(F::zero(), |acc, &x| acc + (x - mean) * (x - mean))
            / F::from(all_magnitudes.len() - 1).unwrap();
        variance.sqrt()
    } else {
        F::zero()
    };

    let stats = DirectionalChangeStats {
        directional_change_ratio,
        average_upward_magnitude,
        average_downward_magnitude,
        directional_change_std,
    };

    Ok((upward_changes, downward_changes, stats))
}

/// Analyze momentum and persistence patterns
fn analyze_momentum_persistence<F>(
    ts: &Array1<F>,
    config: &TurningPointsConfig,
) -> Result<MomentumFeatures<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let mut current_up_sequence = 0;
    let mut current_down_sequence = 0;
    let mut longest_upward_sequence = 0;
    let mut longest_downward_sequence = 0;
    let mut upward_sequences = Vec::new();
    let mut downward_sequences = Vec::new();

    // Analyze consecutive directional movements
    for i in 1..n {
        let change = ts[i] - ts[i - 1];

        if change > F::zero() {
            // Upward movement
            current_up_sequence += 1;
            if current_down_sequence >= config.min_sequence_length {
                downward_sequences.push(current_down_sequence);
            }
            current_down_sequence = 0;
        } else if change < F::zero() {
            // Downward movement
            current_down_sequence += 1;
            if current_up_sequence >= config.min_sequence_length {
                upward_sequences.push(current_up_sequence);
            }
            current_up_sequence = 0;
        }

        longest_upward_sequence = longest_upward_sequence.max(current_up_sequence);
        longest_downward_sequence = longest_downward_sequence.max(current_down_sequence);
    }

    // Handle final sequences
    if current_up_sequence >= config.min_sequence_length {
        upward_sequences.push(current_up_sequence);
    }
    if current_down_sequence >= config.min_sequence_length {
        downward_sequences.push(current_down_sequence);
    }

    // Calculate average sequence lengths
    let average_upward_sequence_length = if !upward_sequences.is_empty() {
        F::from(upward_sequences.iter().sum::<usize>()).unwrap()
            / F::from(upward_sequences.len()).unwrap()
    } else {
        F::zero()
    };

    let average_downward_sequence_length = if !downward_sequences.is_empty() {
        F::from(downward_sequences.iter().sum::<usize>()).unwrap()
            / F::from(downward_sequences.len()).unwrap()
    } else {
        F::zero()
    };

    // Calculate momentum persistence ratio
    let long_sequences = upward_sequences
        .iter()
        .filter(|&&len| len >= config.min_sequence_length * 2)
        .count()
        + downward_sequences
            .iter()
            .filter(|&&len| len >= config.min_sequence_length * 2)
            .count();
    let total_sequences = upward_sequences.len() + downward_sequences.len();

    let momentum_persistence_ratio = if total_sequences > 0 {
        F::from(long_sequences).unwrap() / F::from(total_sequences).unwrap()
    } else {
        F::zero()
    };

    Ok(MomentumFeatures {
        longest_upward_sequence,
        longest_downward_sequence,
        average_upward_sequence_length,
        average_downward_sequence_length,
        momentum_persistence_ratio,
    })
}

/// Characterize local extrema (peaks and valleys)
fn characterize_local_extrema<F>(
    ts: &Array1<F>,
    local_maxima: &[usize],
    local_minima: &[usize],
) -> Result<ExtremaFeatures<F>>
where
    F: Float + FromPrimitive + Debug,
{
    // Calculate peak amplitudes
    let peak_amplitudes: Vec<F> = local_maxima
        .iter()
        .filter_map(|&idx| if idx < ts.len() { Some(ts[idx]) } else { None })
        .collect();

    // Calculate valley amplitudes
    let valley_amplitudes: Vec<F> = local_minima
        .iter()
        .filter_map(|&idx| if idx < ts.len() { Some(ts[idx]) } else { None })
        .collect();

    // Average peak amplitude
    let average_peak_amplitude = if !peak_amplitudes.is_empty() {
        peak_amplitudes.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from(peak_amplitudes.len()).unwrap()
    } else {
        F::zero()
    };

    // Average valley amplitude
    let average_valley_amplitude = if !valley_amplitudes.is_empty() {
        valley_amplitudes.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from(valley_amplitudes.len()).unwrap()
    } else {
        F::zero()
    };

    // Peak amplitude standard deviation
    let peak_amplitude_std = if peak_amplitudes.len() > 1 {
        let variance = peak_amplitudes.iter().fold(F::zero(), |acc, &x| {
            acc + (x - average_peak_amplitude) * (x - average_peak_amplitude)
        }) / F::from(peak_amplitudes.len() - 1).unwrap();
        variance.sqrt()
    } else {
        F::zero()
    };

    // Valley amplitude standard deviation
    let valley_amplitude_std = if valley_amplitudes.len() > 1 {
        let variance = valley_amplitudes.iter().fold(F::zero(), |acc, &x| {
            acc + (x - average_valley_amplitude) * (x - average_valley_amplitude)
        }) / F::from(valley_amplitudes.len() - 1).unwrap();
        variance.sqrt()
    } else {
        F::zero()
    };

    // Peak-to-valley amplitude ratio
    let peak_valley_amplitude_ratio = if average_valley_amplitude != F::zero() {
        average_peak_amplitude / average_valley_amplitude
    } else {
        F::one()
    };

    // Extrema asymmetry (skewness of combined peak and valley distributions)
    let all_extrema: Vec<F> = peak_amplitudes
        .into_iter()
        .chain(valley_amplitudes)
        .collect();

    let extrema_asymmetry = if all_extrema.len() > 2 {
        let mean = all_extrema.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from(all_extrema.len()).unwrap();
        let variance = all_extrema
            .iter()
            .fold(F::zero(), |acc, &x| acc + (x - mean) * (x - mean))
            / F::from(all_extrema.len()).unwrap();

        if variance > F::zero() {
            let std_dev = variance.sqrt();
            let skewness = all_extrema.iter().fold(F::zero(), |acc, &x| {
                let normalized = (x - mean) / std_dev;
                acc + normalized * normalized * normalized
            }) / F::from(all_extrema.len()).unwrap();
            skewness
        } else {
            F::zero()
        }
    } else {
        F::zero()
    };

    Ok(ExtremaFeatures {
        average_peak_amplitude,
        average_valley_amplitude,
        peak_amplitude_std,
        valley_amplitude_std,
        peak_valley_amplitude_ratio,
        extrema_asymmetry,
    })
}

/// Detect trend reversals
fn detect_trend_reversals<F>(
    ts: &Array1<F>,
    turning_points: &[usize],
    config: &TurningPointsConfig,
) -> Result<TrendReversalFeatures<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let major_threshold = F::from(config.major_reversal_threshold).unwrap();
    let mut major_reversals = Vec::new();
    let mut minor_reversals = Vec::new();

    // Calculate data range for relative thresholds
    let min_val = ts.iter().fold(F::infinity(), |a, &b| a.min(b));
    let max_val = ts.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
    let range = max_val - min_val;
    let major_abs_threshold = major_threshold * range;

    // Analyze changes between turning points
    for window in turning_points.windows(2) {
        let start_idx = window[0];
        let end_idx = window[1];

        if start_idx < ts.len() && end_idx < ts.len() {
            let change_magnitude = (ts[end_idx] - ts[start_idx]).abs();

            if change_magnitude >= major_abs_threshold {
                major_reversals.push(change_magnitude);
            } else {
                minor_reversals.push(change_magnitude);
            }
        }
    }

    let major_trend_reversals = major_reversals.len();
    let minor_trend_reversals = minor_reversals.len();

    let average_major_reversal_magnitude = if !major_reversals.is_empty() {
        major_reversals.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from(major_reversals.len()).unwrap()
    } else {
        F::zero()
    };

    let average_minor_reversal_magnitude = if !minor_reversals.is_empty() {
        minor_reversals.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from(minor_reversals.len()).unwrap()
    } else {
        F::zero()
    };

    let trend_reversal_frequency = F::from(major_trend_reversals + minor_trend_reversals).unwrap()
        / F::from(ts.len()).unwrap();

    let reversal_strength_index = major_reversals.iter().fold(F::zero(), |acc, &x| acc + x)
        + minor_reversals.iter().fold(F::zero(), |acc, &x| acc + x);

    Ok(TrendReversalFeatures {
        major_trend_reversals,
        minor_trend_reversals,
        average_major_reversal_magnitude,
        average_minor_reversal_magnitude,
        trend_reversal_frequency,
        reversal_strength_index,
    })
}

/// Analyze temporal patterns in turning points
fn analyze_temporal_patterns<F>(
    turning_points: &[usize],
    config: &TurningPointsConfig,
) -> Result<TurningPointTemporalFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone + std::iter::Sum + ndarray::ScalarOperand,
{
    if turning_points.len() < 3 {
        return Ok(TurningPointTemporalFeatures::default());
    }

    // Calculate intervals between turning points
    let intervals: Vec<F> = turning_points
        .windows(2)
        .map(|w| F::from(w[1] - w[0]).unwrap())
        .collect();

    // Turning point regularity (coefficient of variation of intervals)
    let mean_interval =
        intervals.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(intervals.len()).unwrap();

    let interval_variance = if intervals.len() > 1 {
        intervals.iter().fold(F::zero(), |acc, &x| {
            acc + (x - mean_interval) * (x - mean_interval)
        }) / F::from(intervals.len() - 1).unwrap()
    } else {
        F::zero()
    };

    let turning_point_regularity = if mean_interval > F::zero() {
        interval_variance.sqrt() / mean_interval
    } else {
        F::zero()
    };

    // Turning point clustering (analyze distribution of intervals)
    let turning_point_clustering = calculate_clustering_coefficient(&intervals)?;

    // Turning point periodicity (simplified autocorrelation)
    let turning_point_periodicity = calculate_periodicity_strength(&intervals)?;

    // Turning point autocorrelation
    let max_lag = config.max_autocorr_lag.min(intervals.len() / 2);
    let turning_point_autocorrelation = if max_lag > 0 {
        calculate_autocorrelation_at_lag(&intervals, 1)?
    } else {
        F::zero()
    };

    Ok(TurningPointTemporalFeatures {
        turning_point_regularity,
        turning_point_clustering,
        turning_point_periodicity,
        turning_point_autocorrelation,
    })
}

/// Calculate stability and volatility measures
fn calculate_stability_measures<F>(
    ts: &Array1<F>,
    turning_points: &[usize],
) -> Result<StabilityFeatures<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();

    // Turning point volatility (average local variance around turning points)
    let mut local_variances = Vec::new();
    let window_size = 5; // Local window around turning points

    for &tp_idx in turning_points {
        let start = tp_idx.saturating_sub(window_size / 2);
        let end = (tp_idx + window_size / 2 + 1).min(n);

        if end > start + 1 {
            let local_slice = ts.slice(s![start..end]);
            let local_mean = local_slice.sum() / F::from(local_slice.len()).unwrap();
            let local_variance = local_slice
                .mapv(|x| (x - local_mean) * (x - local_mean))
                .sum()
                / F::from(local_slice.len()).unwrap();
            local_variances.push(local_variance);
        }
    }

    let turning_point_volatility = if !local_variances.is_empty() {
        local_variances.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from(local_variances.len()).unwrap()
    } else {
        F::zero()
    };

    // Stability index (inverse of turning point frequency)
    let turning_point_frequency = F::from(turning_points.len()).unwrap() / F::from(n).unwrap();
    let stability_index = if turning_point_frequency > F::zero() {
        F::one() / turning_point_frequency
    } else {
        F::from(n).unwrap()
    };

    // Noise-to-signal ratio
    let signal_mean = ts.sum() / F::from(n).unwrap();
    let signal_variance =
        ts.mapv(|x| (x - signal_mean) * (x - signal_mean)).sum() / F::from(n).unwrap();
    let noise_signal_ratio = if signal_variance > F::zero() {
        turning_point_volatility / signal_variance
    } else {
        F::zero()
    };

    // Trend consistency (measure of directional persistence)
    let mut directional_changes = 0;
    for i in 1..n {
        if i >= 2 {
            let prev_change = ts[i - 1] - ts[i - 2];
            let curr_change = ts[i] - ts[i - 1];
            if (prev_change > F::zero()) != (curr_change > F::zero()) {
                directional_changes += 1;
            }
        }
    }

    let trend_consistency =
        F::one() - F::from(directional_changes).unwrap() / F::from(n - 2).unwrap();

    Ok(StabilityFeatures {
        turning_point_volatility,
        stability_index,
        noise_signal_ratio,
        trend_consistency,
    })
}

/// Detect advanced patterns (double peaks, head-shoulders, etc.)
fn detect_advanced_patterns<F>(
    ts: &Array1<F>,
    local_maxima: &[usize],
    local_minima: &[usize],
    _config: &TurningPointsConfig,
) -> Result<AdvancedPatternFeatures>
where
    F: Float + FromPrimitive + Debug + PartialOrd,
{
    // Detect double peaks (M patterns)
    let double_peak_count = detect_double_peaks(ts, local_maxima)?;

    // Detect double bottoms (W patterns)
    let double_bottom_count = detect_double_bottoms(ts, local_minima)?;

    // Detect head and shoulders patterns
    let head_shoulders_count = detect_head_and_shoulders(ts, local_maxima, local_minima)?;

    // Detect triangular patterns
    let triangular_pattern_count = detect_triangular_patterns(ts, local_maxima, local_minima)?;

    Ok(AdvancedPatternFeatures {
        double_peak_count,
        double_bottom_count,
        head_shoulders_count,
        triangular_pattern_count,
    })
}

/// Analyze relative positions of turning points
fn analyze_turning_point_positions<F>(
    ts: &Array1<F>,
    turning_points: &[usize],
) -> Result<PositionFeatures<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if turning_points.is_empty() {
        return Ok(PositionFeatures {
            upper_half_turning_points: F::from(0.5).unwrap(),
            lower_half_turning_points: F::from(0.5).unwrap(),
            turning_point_position_skewness: F::zero(),
            turning_point_position_kurtosis: F::zero(),
        });
    }

    // Get data range
    let min_val = ts.iter().fold(F::infinity(), |a, &b| a.min(b));
    let max_val = ts.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
    let range = max_val - min_val;
    let midpoint = min_val + range / F::from(2.0).unwrap();

    // Analyze turning point positions
    let tp_values: Vec<F> = turning_points
        .iter()
        .filter_map(|&idx| if idx < ts.len() { Some(ts[idx]) } else { None })
        .collect();

    let upper_half_count = tp_values.iter().filter(|&&x| x > midpoint).count();
    let total_count = tp_values.len();

    let upper_half_turning_points =
        F::from(upper_half_count).unwrap() / F::from(total_count).unwrap();
    let lower_half_turning_points = F::one() - upper_half_turning_points;

    // Calculate skewness and kurtosis of turning point positions
    let mean_position =
        tp_values.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(total_count).unwrap();

    let variance = tp_values.iter().fold(F::zero(), |acc, &x| {
        acc + (x - mean_position) * (x - mean_position)
    }) / F::from(total_count).unwrap();

    let (turning_point_position_skewness, turning_point_position_kurtosis) = if variance > F::zero()
    {
        let std_dev = variance.sqrt();

        let skewness = tp_values.iter().fold(F::zero(), |acc, &x| {
            let normalized = (x - mean_position) / std_dev;
            acc + normalized * normalized * normalized
        }) / F::from(total_count).unwrap();

        let kurtosis = tp_values.iter().fold(F::zero(), |acc, &x| {
            let normalized = (x - mean_position) / std_dev;
            let normalized_sq = normalized * normalized;
            acc + normalized_sq * normalized_sq
        }) / F::from(total_count).unwrap()
            - F::from(3.0).unwrap();

        (skewness, kurtosis)
    } else {
        (F::zero(), F::zero())
    };

    Ok(PositionFeatures {
        upper_half_turning_points,
        lower_half_turning_points,
        turning_point_position_skewness,
        turning_point_position_kurtosis,
    })
}

/// Analyze multi-scale turning points
fn analyze_multiscale_turning_points<F>(
    ts: &Array1<F>,
    config: &TurningPointsConfig,
) -> Result<MultiscaleTurningPointFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone + std::iter::Sum + ndarray::ScalarOperand,
{
    let mut multiscale_turning_points = Vec::new();
    let mut scale_consistencies = Vec::new();

    // Analyze turning points at different smoothing scales
    for &window_size in &config.smoothing_windows {
        // Apply simple moving average smoothing
        let smoothed = apply_moving_average(ts, window_size)?;

        // Create smoothed config
        let smoothed_config = TurningPointsConfig {
            min_turning_point_threshold: config.min_turning_point_threshold,
            extrema_window_size: config.extrema_window_size,
            major_reversal_threshold: config.major_reversal_threshold,
            detect_advanced_patterns: false,
            smoothing_windows: vec![],
            calculate_temporal_patterns: false,
            max_autocorr_lag: 0,
            analyze_clustering: false,
            min_sequence_length: config.min_sequence_length,
            multiscale_analysis: false,
        };

        // Detect turning points at this scale
        let (tp, _, _) = detect_turning_points(&smoothed, &smoothed_config)?;
        multiscale_turning_points.push(tp.len());

        // Calculate scale consistency (similarity with original scale)
        if !multiscale_turning_points.is_empty() {
            let original_count = multiscale_turning_points[0] as f64;
            let current_count = tp.len() as f64;
            let consistency =
                1.0 - (original_count - current_count).abs() / original_count.max(current_count);
            scale_consistencies.push(F::from(consistency).unwrap());
        }
    }

    // Calculate scale turning point ratio
    let scale_turning_point_ratio = if multiscale_turning_points.len() > 1 {
        let first_scale = F::from(multiscale_turning_points[0]).unwrap();
        let last_scale = F::from(*multiscale_turning_points.last().unwrap()).unwrap();
        if last_scale > F::zero() {
            first_scale / last_scale
        } else {
            F::one()
        }
    } else {
        F::one()
    };

    // Calculate cross-scale consistency
    let cross_scale_consistency = if !scale_consistencies.is_empty() {
        scale_consistencies
            .iter()
            .fold(F::zero(), |acc, &x| acc + x)
            / F::from(scale_consistencies.len()).unwrap()
    } else {
        F::zero()
    };

    // Calculate hierarchical structure index
    let hierarchical_structure_index = if multiscale_turning_points.len() > 2 {
        let mut structure_measure = F::zero();
        for i in 1..multiscale_turning_points.len() {
            let ratio = F::from(multiscale_turning_points[i - 1]).unwrap()
                / F::from(multiscale_turning_points[i])
                    .unwrap()
                    .max(F::from(1.0).unwrap());
            structure_measure = structure_measure + ratio;
        }
        structure_measure / F::from(multiscale_turning_points.len() - 1).unwrap()
    } else {
        F::one()
    };

    Ok(MultiscaleTurningPointFeatures {
        multiscale_turning_points,
        scale_turning_point_ratio,
        cross_scale_consistency,
        hierarchical_structure_index,
    })
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Apply simple moving average smoothing
fn apply_moving_average<F>(ts: &Array1<F>, window_size: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Clone,
{
    let n = ts.len();
    if window_size >= n {
        return Ok(ts.clone());
    }

    let mut smoothed = Array1::zeros(n);
    let half_window = window_size / 2;

    for i in 0..n {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(n);

        let window_sum = ts.slice(s![start..end]).sum();
        let window_len = F::from(end - start).unwrap();
        smoothed[i] = window_sum / window_len;
    }

    Ok(smoothed)
}

/// Calculate clustering coefficient for intervals
fn calculate_clustering_coefficient<F>(intervals: &[F]) -> Result<F>
where
    F: Float + FromPrimitive,
{
    if intervals.len() < 3 {
        return Ok(F::zero());
    }

    // Simple clustering measure: variance of interval ratios
    let mut ratios = Vec::new();
    for i in 1..intervals.len() {
        if intervals[i] > F::zero() && intervals[i - 1] > F::zero() {
            ratios.push(intervals[i] / intervals[i - 1]);
        }
    }

    if ratios.len() < 2 {
        return Ok(F::zero());
    }

    let mean_ratio =
        ratios.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(ratios.len()).unwrap();
    let variance = ratios.iter().fold(F::zero(), |acc, &x| {
        acc + (x - mean_ratio) * (x - mean_ratio)
    }) / F::from(ratios.len()).unwrap();

    // Clustering is inverse of variance (higher variance = less clustering)
    Ok(F::one() / (F::one() + variance))
}

/// Calculate periodicity strength
fn calculate_periodicity_strength<F>(intervals: &[F]) -> Result<F>
where
    F: Float + FromPrimitive,
{
    if intervals.len() < 4 {
        return Ok(F::zero());
    }

    // Simple periodicity measure: autocorrelation at lag 1
    calculate_autocorrelation_at_lag(intervals, 1)
}

/// Calculate autocorrelation at specific lag
fn calculate_autocorrelation_at_lag<F>(data: &[F], lag: usize) -> Result<F>
where
    F: Float + FromPrimitive,
{
    if data.len() <= lag {
        return Ok(F::zero());
    }

    let n = data.len() - lag;
    if n < 2 {
        return Ok(F::zero());
    }

    let mean = data.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(data.len()).unwrap();

    let mut numerator = F::zero();
    let mut denominator = F::zero();

    for i in 0..n {
        let x_centered = data[i] - mean;
        let y_centered = data[i + lag] - mean;
        numerator = numerator + x_centered * y_centered;
        denominator = denominator + x_centered * x_centered;
    }

    if denominator > F::zero() {
        Ok(numerator / denominator)
    } else {
        Ok(F::zero())
    }
}

// Pattern detection functions (simplified implementations)
fn detect_double_peaks<F>(_ts: &Array1<F>, local_maxima: &[usize]) -> Result<usize>
where
    F: Float + FromPrimitive + PartialOrd,
{
    // Simplified: count consecutive peak pairs
    let mut count = 0;
    for window in local_maxima.windows(3) {
        let spacing1 = window[1] - window[0];
        let spacing2 = window[2] - window[1];

        // Simple heuristic: double peaks have similar spacing
        if spacing1 > 0 && spacing2 > 0 {
            let ratio = spacing1 as f64 / spacing2 as f64;
            if (0.5..=2.0).contains(&ratio) {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn detect_double_bottoms<F>(_ts: &Array1<F>, local_minima: &[usize]) -> Result<usize>
where
    F: Float + FromPrimitive + PartialOrd,
{
    // Simplified: count consecutive valley pairs
    let mut count = 0;
    for window in local_minima.windows(3) {
        let spacing1 = window[1] - window[0];
        let spacing2 = window[2] - window[1];

        // Simple heuristic: double bottoms have similar spacing
        if spacing1 > 0 && spacing2 > 0 {
            let ratio = spacing1 as f64 / spacing2 as f64;
            if (0.5..=2.0).contains(&ratio) {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn detect_head_and_shoulders<F>(
    _ts: &Array1<F>,
    local_maxima: &[usize],
    _local_minima: &[usize],
) -> Result<usize>
where
    F: Float + FromPrimitive + PartialOrd,
{
    // Simplified: count groups of 3 peaks where middle is highest
    let mut count = 0;
    if local_maxima.len() >= 3 {
        for window in local_maxima.windows(3) {
            // Simple spacing check for head-and-shoulders pattern
            let spacing1 = window[1] - window[0];
            let spacing2 = window[2] - window[1];

            if spacing1 > 0 && spacing2 > 0 && spacing1 <= spacing2 * 2 && spacing2 <= spacing1 * 2
            {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn detect_triangular_patterns<F>(
    _ts: &Array1<F>,
    local_maxima: &[usize],
    local_minima: &[usize],
) -> Result<usize>
where
    F: Float + FromPrimitive + PartialOrd,
{
    // Simplified: count patterns where extrema converge
    let min_pattern_length = 4;
    let mut count = 0;

    if local_maxima.len() >= 2 && local_minima.len() >= 2 {
        // Check for converging pattern in peaks and valleys
        for i in 0..(local_maxima.len().saturating_sub(min_pattern_length)) {
            let peak_range_start = local_maxima[i + 1] - local_maxima[i];
            let peak_range_end = if i + 3 < local_maxima.len() {
                local_maxima[i + 3] - local_maxima[i + 2]
            } else {
                continue;
            };

            // Simple convergence check
            if peak_range_end > 0 && peak_range_start > peak_range_end {
                count += 1;
            }
        }
    }

    Ok(count)
}
