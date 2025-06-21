//! Time series correlation and relationship analysis
//!
//! This module provides various methods for analyzing correlations and relationships between time series:
//! - Cross-correlation functions (CCF)
//! - Dynamic time warping (DTW) with various constraints
//! - Time-frequency analysis using wavelets and spectrograms
//! - Coherence analysis for frequency domain relationships

use crate::error::TimeSeriesError;
use ndarray::{s, Array1, Array2};
use scirs2_core::validation::check_array_finite;
use std::f64::consts::PI;

/// Result type for correlation analysis
pub type CorrelationResult<T> = Result<T, TimeSeriesError>;

/// Cross-correlation function result
#[derive(Debug, Clone)]
pub struct CrossCorrelationResult {
    /// Cross-correlation values
    pub correlations: Array1<f64>,
    /// Lag values corresponding to correlations
    pub lags: Array1<i32>,
    /// Maximum correlation value
    pub max_correlation: f64,
    /// Lag at maximum correlation
    pub lag_at_max: i32,
    /// Confidence intervals (if computed)
    pub confidence_lower: Option<Array1<f64>>,
    /// Confidence intervals (if computed)
    pub confidence_upper: Option<Array1<f64>>,
}

/// Dynamic time warping result
#[derive(Debug, Clone)]
pub struct DTWResult {
    /// DTW distance between series
    pub distance: f64,
    /// Optimal warping path (indices)
    pub warping_path: Vec<(usize, usize)>,
    /// Cost matrix
    pub cost_matrix: Array2<f64>,
    /// Normalized distance (if requested)
    pub normalized_distance: Option<f64>,
    /// Local cost function used
    pub cost_function: DTWCostFunction,
    /// Constraint type used
    pub constraint: DTWConstraint,
}

/// Time-frequency analysis result
#[derive(Debug, Clone)]
pub struct TimeFrequencyResult {
    /// Time-frequency representation (spectrogram)
    pub spectrogram: Array2<f64>,
    /// Time vector
    pub times: Array1<f64>,
    /// Frequency vector
    pub frequencies: Array1<f64>,
    /// Analysis method used
    pub method: TimeFrequencyMethod,
    /// Window parameters (if applicable)
    pub window_info: Option<WindowInfo>,
}

/// Coherence analysis result
#[derive(Debug, Clone)]
pub struct CoherenceResult {
    /// Coherence values
    pub coherence: Array1<f64>,
    /// Phase difference
    pub phase: Array1<f64>,
    /// Frequency vector
    pub frequencies: Array1<f64>,
    /// Cross-power spectral density
    pub cross_psd: Array1<f64>,
    /// Power spectral densities for each series
    pub psd_x: Array1<f64>,
    /// Power spectral densities for each series
    pub psd_y: Array1<f64>,
    /// Confidence level (if computed)
    pub confidence_level: Option<f64>,
    /// Confidence threshold for coherence
    pub confidence_threshold: Option<f64>,
}

/// DTW cost functions
#[derive(Debug, Clone, Copy)]
pub enum DTWCostFunction {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Squared Euclidean distance
    SquaredEuclidean,
    /// Cosine distance
    Cosine,
}

/// DTW constraint types
#[derive(Debug, Clone, Copy)]
pub enum DTWConstraint {
    /// No constraint (full DTW)
    None,
    /// Sakoe-Chiba band constraint
    SakoeChiba(usize),
    /// Itakura parallelogram constraint
    Itakura,
}

/// Time-frequency analysis methods
#[derive(Debug, Clone, Copy)]
pub enum TimeFrequencyMethod {
    /// Short-time Fourier transform
    STFT,
    /// Continuous wavelet transform
    CWT,
    /// Morlet wavelet
    Morlet,
    /// Gabor transform
    Gabor,
}

/// Window information for time-frequency analysis
#[derive(Debug, Clone)]
pub struct WindowInfo {
    /// Window type
    pub window_type: WindowType,
    /// Window size
    pub window_size: usize,
    /// Overlap between windows
    pub overlap: usize,
}

/// Window types for analysis
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// Hamming window
    Hamming,
    /// Hanning window
    Hanning,
    /// Blackman window
    Blackman,
    /// Gaussian window
    Gaussian,
    /// Rectangular window
    Rectangular,
}

/// Configuration for cross-correlation analysis
#[derive(Debug, Clone)]
pub struct CrossCorrelationConfig {
    /// Maximum lag to compute
    pub max_lag: usize,
    /// Whether to normalize the series
    pub normalize: bool,
    /// Confidence level for intervals
    pub confidence_level: Option<f64>,
    /// Method for correlation calculation
    pub method: CorrelationMethod,
}

/// Correlation calculation methods
#[derive(Debug, Clone, Copy)]
pub enum CorrelationMethod {
    /// Pearson correlation
    Pearson,
    /// Spearman rank correlation
    Spearman,
    /// Kendall's tau
    Kendall,
}

impl Default for CrossCorrelationConfig {
    fn default() -> Self {
        Self {
            max_lag: 20,
            normalize: true,
            confidence_level: Some(0.95),
            method: CorrelationMethod::Pearson,
        }
    }
}

/// Configuration for DTW analysis
#[derive(Debug, Clone)]
pub struct DTWConfig {
    /// Cost function to use
    pub cost_function: DTWCostFunction,
    /// Constraint type
    pub constraint: DTWConstraint,
    /// Whether to normalize the distance
    pub normalize: bool,
    /// Step pattern (for advanced DTW variants)
    pub step_pattern: StepPattern,
}

/// DTW step patterns
#[derive(Debug, Clone, Copy)]
pub enum StepPattern {
    /// Symmetric step pattern
    Symmetric,
    /// Asymmetric step pattern
    Asymmetric,
    /// Quasi-symmetric step pattern
    QuasiSymmetric,
}

impl Default for DTWConfig {
    fn default() -> Self {
        Self {
            cost_function: DTWCostFunction::Euclidean,
            constraint: DTWConstraint::None,
            normalize: true,
            step_pattern: StepPattern::Symmetric,
        }
    }
}

/// Configuration for time-frequency analysis
#[derive(Debug, Clone)]
pub struct TimeFrequencyConfig {
    /// Analysis method
    pub method: TimeFrequencyMethod,
    /// Window configuration
    pub window: WindowInfo,
    /// Sampling frequency
    pub sampling_freq: f64,
    /// Frequency range (optional)
    pub freq_range: Option<(f64, f64)>,
    /// Number of frequency bins
    pub n_freq_bins: Option<usize>,
}

impl Default for TimeFrequencyConfig {
    fn default() -> Self {
        Self {
            method: TimeFrequencyMethod::STFT,
            window: WindowInfo {
                window_type: WindowType::Hanning,
                window_size: 256,
                overlap: 128,
            },
            sampling_freq: 1.0,
            freq_range: None,
            n_freq_bins: None,
        }
    }
}

/// Configuration for coherence analysis
#[derive(Debug, Clone)]
pub struct CoherenceConfig {
    /// Window size for spectral estimation
    pub window_size: usize,
    /// Overlap between windows
    pub overlap: usize,
    /// Window type
    pub window_type: WindowType,
    /// Sampling frequency
    pub sampling_freq: f64,
    /// Confidence level for significance testing
    pub confidence_level: Option<f64>,
    /// Detrending method
    pub detrend: DetrendMethod,
}

/// Detrending methods
#[derive(Debug, Clone, Copy)]
pub enum DetrendMethod {
    /// No detrending
    None,
    /// Linear detrending
    Linear,
    /// Mean removal
    Mean,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            window_size: 256,
            overlap: 128,
            window_type: WindowType::Hanning,
            sampling_freq: 1.0,
            confidence_level: Some(0.95),
            detrend: DetrendMethod::Linear,
        }
    }
}

/// Main struct for correlation analysis
pub struct CorrelationAnalyzer {
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl CorrelationAnalyzer {
    /// Create a new correlation analyzer
    pub fn new() -> Self {
        Self { random_seed: None }
    }

    /// Create a new analyzer with random seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            random_seed: Some(seed),
        }
    }

    /// Compute cross-correlation function between two time series
    ///
    /// # Arguments
    ///
    /// * `x` - First time series
    /// * `y` - Second time series
    /// * `config` - Configuration for cross-correlation
    ///
    /// # Returns
    ///
    /// Result containing cross-correlation function and statistics
    pub fn cross_correlation(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        config: &CrossCorrelationConfig,
    ) -> CorrelationResult<CrossCorrelationResult> {
        check_array_finite(x, "x")?;
        check_array_finite(y, "y")?;

        if x.len() != y.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Time series must have the same length".to_string(),
            ));
        }

        let n = x.len();
        if n < 2 * config.max_lag + 1 {
            return Err(TimeSeriesError::InvalidInput(
                "Time series too short for the specified maximum lag".to_string(),
            ));
        }

        // Normalize series if requested
        let x_proc = if config.normalize {
            self.normalize_series(x)?
        } else {
            x.clone()
        };

        let y_proc = if config.normalize {
            self.normalize_series(y)?
        } else {
            y.clone()
        };

        // Compute cross-correlations for different lags
        let mut correlations = Array1::zeros(2 * config.max_lag + 1);
        let mut lags = Array1::zeros(2 * config.max_lag + 1);

        for i in 0..correlations.len() {
            let lag = i as i32 - config.max_lag as i32;
            lags[i] = lag;

            correlations[i] = match config.method {
                CorrelationMethod::Pearson => {
                    self.compute_lagged_correlation(&x_proc, &y_proc, lag)?
                }
                CorrelationMethod::Spearman => {
                    self.compute_lagged_spearman(&x_proc, &y_proc, lag)?
                }
                CorrelationMethod::Kendall => self.compute_lagged_kendall(&x_proc, &y_proc, lag)?,
            };
        }

        // Find maximum correlation and its lag
        let max_idx = correlations
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let max_correlation = correlations[max_idx];
        let lag_at_max = lags[max_idx];

        // Compute confidence intervals if requested
        let (confidence_lower, confidence_upper) = if let Some(conf_level) = config.confidence_level
        {
            let (lower, upper) =
                self.compute_correlation_confidence_intervals(&correlations, n, conf_level)?;
            (Some(lower), Some(upper))
        } else {
            (None, None)
        };

        Ok(CrossCorrelationResult {
            correlations,
            lags,
            max_correlation,
            lag_at_max,
            confidence_lower,
            confidence_upper,
        })
    }

    /// Compute dynamic time warping distance between two time series
    ///
    /// # Arguments
    ///
    /// * `x` - First time series
    /// * `y` - Second time series
    /// * `config` - Configuration for DTW
    ///
    /// # Returns
    ///
    /// Result containing DTW distance and warping path
    pub fn dynamic_time_warping(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        config: &DTWConfig,
    ) -> CorrelationResult<DTWResult> {
        check_array_finite(x, "x")?;
        check_array_finite(y, "y")?;

        let n = x.len();
        let m = y.len();

        if n == 0 || m == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Time series cannot be empty".to_string(),
            ));
        }

        // Initialize cost matrix
        let mut cost_matrix = Array2::from_elem((n + 1, m + 1), f64::INFINITY);
        cost_matrix[[0, 0]] = 0.0;

        // Fill cost matrix according to constraint
        match config.constraint {
            DTWConstraint::None => {
                for i in 1..=n {
                    for j in 1..=m {
                        let local_cost =
                            self.compute_local_cost(x[i - 1], y[j - 1], config.cost_function);
                        cost_matrix[[i, j]] = local_cost
                            + self.min_predecessor(&cost_matrix, i, j, config.step_pattern);
                    }
                }
            }
            DTWConstraint::SakoeChiba(radius) => {
                for i in 1..=n {
                    let j_start =
                        ((i as f64 * m as f64 / n as f64) - radius as f64).max(1.0) as usize;
                    let j_end =
                        ((i as f64 * m as f64 / n as f64) + radius as f64).min(m as f64) as usize;

                    for j in j_start..=j_end {
                        let local_cost =
                            self.compute_local_cost(x[i - 1], y[j - 1], config.cost_function);
                        cost_matrix[[i, j]] = local_cost
                            + self.min_predecessor(&cost_matrix, i, j, config.step_pattern);
                    }
                }
            }
            DTWConstraint::Itakura => {
                // Simplified Itakura parallelogram constraint
                for i in 1..=n {
                    let slope_constraint = 2.0;
                    let j_start = ((i as f64 / slope_constraint).max(1.0)) as usize;
                    let j_end = ((i as f64 * slope_constraint).min(m as f64)) as usize;

                    for j in j_start..=j_end {
                        let local_cost =
                            self.compute_local_cost(x[i - 1], y[j - 1], config.cost_function);
                        cost_matrix[[i, j]] = local_cost
                            + self.min_predecessor(&cost_matrix, i, j, config.step_pattern);
                    }
                }
            }
        }

        let distance = cost_matrix[[n, m]];

        if !distance.is_finite() {
            return Err(TimeSeriesError::ComputationError(
                "DTW computation resulted in infinite distance".to_string(),
            ));
        }

        // Backtrack to find optimal warping path
        let warping_path = self.backtrack_warping_path(&cost_matrix, n, m, config.step_pattern)?;

        // Normalize distance if requested
        let normalized_distance = if config.normalize {
            Some(distance / warping_path.len() as f64)
        } else {
            None
        };

        Ok(DTWResult {
            distance,
            warping_path,
            cost_matrix: cost_matrix.slice(s![1.., 1..]).to_owned(),
            normalized_distance,
            cost_function: config.cost_function,
            constraint: config.constraint,
        })
    }

    /// Perform time-frequency analysis on a time series
    ///
    /// # Arguments
    ///
    /// * `x` - Input time series
    /// * `config` - Configuration for time-frequency analysis
    ///
    /// # Returns
    ///
    /// Result containing time-frequency representation
    pub fn time_frequency_analysis(
        &self,
        x: &Array1<f64>,
        config: &TimeFrequencyConfig,
    ) -> CorrelationResult<TimeFrequencyResult> {
        check_array_finite(x, "x")?;

        if x.len() < config.window.window_size {
            return Err(TimeSeriesError::InvalidInput(
                "Time series too short for the specified window size".to_string(),
            ));
        }

        match config.method {
            TimeFrequencyMethod::STFT => self.compute_stft(x, config),
            TimeFrequencyMethod::CWT => self.compute_cwt(x, config),
            TimeFrequencyMethod::Morlet => self.compute_morlet_wavelet(x, config),
            TimeFrequencyMethod::Gabor => self.compute_gabor_transform(x, config),
        }
    }

    /// Compute coherence between two time series
    ///
    /// # Arguments
    ///
    /// * `x` - First time series
    /// * `y` - Second time series
    /// * `config` - Configuration for coherence analysis
    ///
    /// # Returns
    ///
    /// Result containing coherence and phase information
    pub fn coherence_analysis(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        config: &CoherenceConfig,
    ) -> CorrelationResult<CoherenceResult> {
        check_array_finite(x, "x")?;
        check_array_finite(y, "y")?;

        if x.len() != y.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Time series must have the same length".to_string(),
            ));
        }

        if x.len() < config.window_size {
            return Err(TimeSeriesError::InvalidInput(
                "Time series too short for the specified window size".to_string(),
            ));
        }

        // Preprocess series
        let x_proc = self.detrend_series(x, config.detrend)?;
        let y_proc = self.detrend_series(y, config.detrend)?;

        // Compute windowed segments
        let hop_size = config.window_size - config.overlap;
        let n_windows = (x.len() - config.overlap) / hop_size;

        if n_windows < 2 {
            return Err(TimeSeriesError::InvalidInput(
                "Not enough data for reliable coherence estimation".to_string(),
            ));
        }

        // Generate window function
        let window = self.generate_window(config.window_type, config.window_size)?;

        // Compute spectral estimates
        let freq_bins = config.window_size / 2 + 1;
        let mut cross_psd = Array1::zeros(freq_bins);
        let mut psd_x = Array1::zeros(freq_bins);
        let mut psd_y = Array1::zeros(freq_bins);

        for i in 0..n_windows {
            let start_idx = i * hop_size;
            let end_idx = start_idx + config.window_size;

            if end_idx > x_proc.len() {
                break;
            }

            let x_segment = x_proc.slice(s![start_idx..end_idx]).to_owned();
            let y_segment = y_proc.slice(s![start_idx..end_idx]).to_owned();

            // Apply window
            let x_windowed = &x_segment * &window;
            let y_windowed = &y_segment * &window;

            // Compute FFTs
            let x_fft = self.compute_fft(&x_windowed)?;
            let y_fft = self.compute_fft(&y_windowed)?;

            // Accumulate spectral estimates
            for k in 0..freq_bins {
                let x_complex = x_fft[k];
                let y_complex = y_fft[k];

                cross_psd[k] += x_complex.re * y_complex.re + x_complex.im * y_complex.im;
                psd_x[k] += x_complex.re * x_complex.re + x_complex.im * x_complex.im;
                psd_y[k] += y_complex.re * y_complex.re + y_complex.im * y_complex.im;
            }
        }

        // Normalize by number of windows
        cross_psd /= n_windows as f64;
        psd_x /= n_windows as f64;
        psd_y /= n_windows as f64;

        // Compute coherence
        let mut coherence = Array1::zeros(freq_bins);
        let mut phase = Array1::zeros(freq_bins);

        for k in 0..freq_bins {
            let denominator = (psd_x[k] * psd_y[k]).sqrt();
            if denominator > f64::EPSILON {
                coherence[k] = (cross_psd[k] / denominator).abs();
                phase[k] = cross_psd[k].atan2(0.0); // Simplified phase calculation
            }
        }

        // Generate frequency vector
        let frequencies = Array1::from_iter(
            (0..freq_bins).map(|k| k as f64 * config.sampling_freq / (2.0 * freq_bins as f64)),
        );

        // Compute confidence threshold if requested
        let confidence_threshold = config
            .confidence_level
            .map(|conf_level| self.coherence_confidence_threshold(conf_level, n_windows));

        Ok(CoherenceResult {
            coherence,
            phase,
            frequencies,
            cross_psd,
            psd_x,
            psd_y,
            confidence_level: config.confidence_level,
            confidence_threshold,
        })
    }

    // Helper methods

    fn normalize_series(&self, x: &Array1<f64>) -> CorrelationResult<Array1<f64>> {
        let mean = x.mean().unwrap_or(0.0);
        let std = (x.mapv(|xi| (xi - mean).powi(2)).sum() / x.len() as f64).sqrt();

        if std < f64::EPSILON {
            return Ok(x - mean); // Only remove mean if std is zero
        }

        Ok((x - mean) / std)
    }

    fn compute_lagged_correlation(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        lag: i32,
    ) -> CorrelationResult<f64> {
        let _n = x.len() as i32;

        let (x_slice, y_slice) = if lag >= 0 {
            let lag = lag as usize;
            if lag >= x.len() {
                return Ok(0.0);
            }
            (
                x.slice(s![lag..]).to_owned(),
                y.slice(s![..x.len() - lag]).to_owned(),
            )
        } else {
            let lag = (-lag) as usize;
            if lag >= y.len() {
                return Ok(0.0);
            }
            (
                x.slice(s![..x.len() - lag]).to_owned(),
                y.slice(s![lag..]).to_owned(),
            )
        };

        if x_slice.is_empty() || y_slice.is_empty() {
            return Ok(0.0);
        }

        self.pearson_correlation(&x_slice, &y_slice)
    }

    fn compute_lagged_spearman(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        lag: i32,
    ) -> CorrelationResult<f64> {
        let (x_slice, y_slice) = if lag >= 0 {
            let lag = lag as usize;
            if lag >= x.len() {
                return Ok(0.0);
            }
            (
                x.slice(s![lag..]).to_owned(),
                y.slice(s![..x.len() - lag]).to_owned(),
            )
        } else {
            let lag = (-lag) as usize;
            if lag >= y.len() {
                return Ok(0.0);
            }
            (
                x.slice(s![..x.len() - lag]).to_owned(),
                y.slice(s![lag..]).to_owned(),
            )
        };

        if x_slice.is_empty() || y_slice.is_empty() {
            return Ok(0.0);
        }

        // Convert to ranks and compute Pearson correlation
        let x_ranks = self.compute_ranks(&x_slice);
        let y_ranks = self.compute_ranks(&y_slice);

        self.pearson_correlation(&x_ranks, &y_ranks)
    }

    fn compute_lagged_kendall(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        lag: i32,
    ) -> CorrelationResult<f64> {
        let (x_slice, y_slice) = if lag >= 0 {
            let lag = lag as usize;
            if lag >= x.len() {
                return Ok(0.0);
            }
            (
                x.slice(s![lag..]).to_owned(),
                y.slice(s![..x.len() - lag]).to_owned(),
            )
        } else {
            let lag = (-lag) as usize;
            if lag >= y.len() {
                return Ok(0.0);
            }
            (
                x.slice(s![..x.len() - lag]).to_owned(),
                y.slice(s![lag..]).to_owned(),
            )
        };

        if x_slice.is_empty() || y_slice.is_empty() {
            return Ok(0.0);
        }

        self.kendall_tau(&x_slice, &y_slice)
    }

    fn pearson_correlation(&self, x: &Array1<f64>, y: &Array1<f64>) -> CorrelationResult<f64> {
        if x.len() != y.len() || x.is_empty() {
            return Ok(0.0);
        }

        let n = x.len() as f64;
        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let diff_x = xi - mean_x;
            let diff_y = yi - mean_y;
            numerator += diff_x * diff_y;
            sum_sq_x += diff_x * diff_x;
            sum_sq_y += diff_y * diff_y;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator < f64::EPSILON {
            return Ok(0.0);
        }

        Ok(numerator / denominator)
    }

    fn compute_ranks(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut indexed_values: Vec<(usize, f64)> =
            x.iter().enumerate().map(|(i, &val)| (i, val)).collect();
        indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut ranks = Array1::zeros(x.len());
        for (rank, &(original_index, _)) in indexed_values.iter().enumerate() {
            ranks[original_index] = rank as f64 + 1.0;
        }

        ranks
    }

    fn kendall_tau(&self, x: &Array1<f64>, y: &Array1<f64>) -> CorrelationResult<f64> {
        let n = x.len();
        if n < 2 {
            return Ok(0.0);
        }

        let mut concordant = 0;
        let mut discordant = 0;

        for i in 0..n {
            for j in i + 1..n {
                let x_diff = x[i] - x[j];
                let y_diff = y[i] - y[j];

                if x_diff * y_diff > 0.0 {
                    concordant += 1;
                } else if x_diff * y_diff < 0.0 {
                    discordant += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        if total_pairs == 0 {
            return Ok(0.0);
        }

        Ok((concordant - discordant) as f64 / total_pairs as f64)
    }

    fn compute_correlation_confidence_intervals(
        &self,
        correlations: &Array1<f64>,
        n: usize,
        confidence_level: f64,
    ) -> CorrelationResult<(Array1<f64>, Array1<f64>)> {
        let alpha = 1.0 - confidence_level;
        let z_score = self.normal_quantile(1.0 - alpha / 2.0);
        let std_error = 1.0 / (n as f64).sqrt();

        let lower = correlations.mapv(|r| r - z_score * std_error);
        let upper = correlations.mapv(|r| r + z_score * std_error);

        Ok((lower, upper))
    }

    fn normal_quantile(&self, p: f64) -> f64 {
        // Beasley-Springer-Moro algorithm approximation
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        if (p - 0.5).abs() < f64::EPSILON {
            return 0.0;
        }

        let a = [
            -3.969_683_028_665_376e1,
            2.209_460_984_245_205e2,
            -2.759_285_104_469_687e2,
            1.383_577_518_672_69e2,
            -3.066_479_806_614_716e1,
            2.506_628_277_459_239,
        ];

        let b = [
            -5.447_609_879_822_406e1,
            1.615_858_368_580_409e2,
            -1.556_989_798_598_866e2,
            6.680_131_188_771_972e1,
            -1.328_068_155_288_572e1,
        ];

        if !(0.02425..=0.97575).contains(&p) {
            let q = if p < 0.5 { p } else { 1.0 - p };
            let u = (-2.0 * q.ln()).sqrt();
            let sign = if p < 0.5 { -1.0 } else { 1.0 };
            sign * (a[0] + u * (a[1] + u * (a[2] + u * (a[3] + u * (a[4] + u * a[5])))))
                / (1.0 + u * (b[0] + u * (b[1] + u * (b[2] + u * (b[3] + u * b[4])))))
        } else {
            let q = p - 0.5;
            let r = q * q;
            q * (a[0] + r * (a[1] + r * (a[2] + r * (a[3] + r * (a[4] + r * a[5])))))
                / (1.0 + r * (b[0] + r * (b[1] + r * (b[2] + r * (b[3] + r * b[4])))))
        }
    }

    fn compute_local_cost(&self, x: f64, y: f64, cost_function: DTWCostFunction) -> f64 {
        match cost_function {
            DTWCostFunction::Euclidean => (x - y).abs(),
            DTWCostFunction::Manhattan => (x - y).abs(),
            DTWCostFunction::SquaredEuclidean => (x - y).powi(2),
            DTWCostFunction::Cosine => {
                let dot_product = x * y;
                let magnitude = (x * x + y * y).sqrt();
                if magnitude < f64::EPSILON {
                    0.0
                } else {
                    1.0 - dot_product / magnitude
                }
            }
        }
    }

    fn min_predecessor(
        &self,
        cost_matrix: &Array2<f64>,
        i: usize,
        j: usize,
        step_pattern: StepPattern,
    ) -> f64 {
        match step_pattern {
            StepPattern::Symmetric => {
                let candidates = [
                    cost_matrix[[i - 1, j]],     // Vertical
                    cost_matrix[[i, j - 1]],     // Horizontal
                    cost_matrix[[i - 1, j - 1]], // Diagonal
                ];
                candidates.iter().cloned().fold(f64::INFINITY, f64::min)
            }
            StepPattern::Asymmetric => {
                let candidates = [
                    cost_matrix[[i - 1, j]] + cost_matrix[[i - 1, j - 1]],
                    cost_matrix[[i, j - 1]],
                    cost_matrix[[i - 1, j - 1]],
                ];
                candidates.iter().cloned().fold(f64::INFINITY, f64::min)
            }
            StepPattern::QuasiSymmetric => {
                let candidates = [
                    cost_matrix[[i - 1, j]],
                    cost_matrix[[i, j - 1]],
                    2.0 * cost_matrix[[i - 1, j - 1]],
                ];
                candidates.iter().cloned().fold(f64::INFINITY, f64::min)
            }
        }
    }

    fn backtrack_warping_path(
        &self,
        cost_matrix: &Array2<f64>,
        n: usize,
        m: usize,
        step_pattern: StepPattern,
    ) -> CorrelationResult<Vec<(usize, usize)>> {
        let mut path = Vec::new();
        let mut i = n;
        let mut j = m;

        path.push((i - 1, j - 1)); // Convert to 0-indexed

        while i > 1 || j > 1 {
            let candidates = match step_pattern {
                StepPattern::Symmetric => vec![
                    (
                        i.saturating_sub(1),
                        j,
                        cost_matrix[[i.saturating_sub(1), j]],
                    ),
                    (
                        i,
                        j.saturating_sub(1),
                        cost_matrix[[i, j.saturating_sub(1)]],
                    ),
                    (
                        i.saturating_sub(1),
                        j.saturating_sub(1),
                        cost_matrix[[i.saturating_sub(1), j.saturating_sub(1)]],
                    ),
                ],
                _ => vec![
                    // Simplified for other patterns
                    (
                        i.saturating_sub(1),
                        j,
                        cost_matrix[[i.saturating_sub(1), j]],
                    ),
                    (
                        i,
                        j.saturating_sub(1),
                        cost_matrix[[i, j.saturating_sub(1)]],
                    ),
                    (
                        i.saturating_sub(1),
                        j.saturating_sub(1),
                        cost_matrix[[i.saturating_sub(1), j.saturating_sub(1)]],
                    ),
                ],
            };

            let (next_i, next_j, _) = candidates
                .into_iter()
                .filter(|(ni, nj, _)| *ni > 0 && *nj > 0)
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                .unwrap_or((1, 1, 0.0));

            i = next_i;
            j = next_j;
            path.push((i - 1, j - 1)); // Convert to 0-indexed
        }

        path.reverse();
        Ok(path)
    }

    fn compute_stft(
        &self,
        x: &Array1<f64>,
        config: &TimeFrequencyConfig,
    ) -> CorrelationResult<TimeFrequencyResult> {
        let window = self.generate_window(config.window.window_type, config.window.window_size)?;
        let hop_size = config.window.window_size - config.window.overlap;
        let n_windows = (x.len() - config.window.overlap) / hop_size;
        let freq_bins = config.window.window_size / 2 + 1;

        let mut spectrogram = Array2::zeros((freq_bins, n_windows));
        let mut times = Array1::zeros(n_windows);

        for i in 0..n_windows {
            let start_idx = i * hop_size;
            let end_idx = start_idx + config.window.window_size;

            if end_idx > x.len() {
                break;
            }

            times[i] = start_idx as f64 / config.sampling_freq;

            let segment = x.slice(s![start_idx..end_idx]).to_owned();
            let windowed = &segment * &window;

            let fft_result = self.compute_fft(&windowed)?;

            for k in 0..freq_bins {
                let magnitude = (fft_result[k].re.powi(2) + fft_result[k].im.powi(2)).sqrt();
                spectrogram[[k, i]] = magnitude;
            }
        }

        let frequencies = Array1::from_iter(
            (0..freq_bins)
                .map(|k| k as f64 * config.sampling_freq / config.window.window_size as f64),
        );

        Ok(TimeFrequencyResult {
            spectrogram,
            times,
            frequencies,
            method: config.method,
            window_info: Some(config.window.clone()),
        })
    }

    fn compute_cwt(
        &self,
        x: &Array1<f64>,
        config: &TimeFrequencyConfig,
    ) -> CorrelationResult<TimeFrequencyResult> {
        // Simplified continuous wavelet transform implementation
        let n_scales = config.n_freq_bins.unwrap_or(50);
        let n_times = x.len();

        let mut spectrogram = Array2::zeros((n_scales, n_times));
        let times = Array1::from_iter((0..n_times).map(|i| i as f64 / config.sampling_freq));
        let mut frequencies = Array1::zeros(n_scales);

        // Generate scales logarithmically
        let min_scale = 1.0;
        let max_scale = n_times as f64 / 4.0;
        let scale_factor = (max_scale / min_scale).powf(1.0 / (n_scales - 1) as f64);

        for (scale_idx, _scale) in (0..n_scales).enumerate() {
            let current_scale = min_scale * scale_factor.powi(scale_idx as i32);
            frequencies[scale_idx] = config.sampling_freq / (2.0 * PI * current_scale);

            // Convolve with Morlet wavelet at current scale
            for t in 0..n_times {
                let mut convolution_result = 0.0;

                for tau in 0..n_times {
                    let t_normalized = (tau as f64 - t as f64) / current_scale;
                    let wavelet_value = self.morlet_wavelet(t_normalized);
                    convolution_result += x[tau] * wavelet_value;
                }

                spectrogram[[scale_idx, t]] = convolution_result.abs();
            }
        }

        Ok(TimeFrequencyResult {
            spectrogram,
            times,
            frequencies,
            method: config.method,
            window_info: None,
        })
    }

    fn compute_morlet_wavelet(
        &self,
        x: &Array1<f64>,
        config: &TimeFrequencyConfig,
    ) -> CorrelationResult<TimeFrequencyResult> {
        // Similar to CWT but specifically for Morlet wavelets
        self.compute_cwt(x, config)
    }

    fn compute_gabor_transform(
        &self,
        x: &Array1<f64>,
        config: &TimeFrequencyConfig,
    ) -> CorrelationResult<TimeFrequencyResult> {
        // Gabor transform is essentially a windowed Fourier transform
        self.compute_stft(x, config)
    }

    fn morlet_wavelet(&self, t: f64) -> f64 {
        let sigma = 1.0;
        let omega = 6.0;
        let gaussian = (-t * t / (2.0 * sigma * sigma)).exp();
        let oscillation = (omega * t).cos();
        gaussian * oscillation / (PI.powf(0.25) * sigma.sqrt())
    }

    fn generate_window(
        &self,
        window_type: WindowType,
        size: usize,
    ) -> CorrelationResult<Array1<f64>> {
        let mut window = Array1::zeros(size);

        match window_type {
            WindowType::Hamming => {
                for i in 0..size {
                    window[i] = 0.54 - 0.46 * (2.0 * PI * i as f64 / (size - 1) as f64).cos();
                }
            }
            WindowType::Hanning => {
                for i in 0..size {
                    window[i] = 0.5 * (1.0 - (2.0 * PI * i as f64 / (size - 1) as f64).cos());
                }
            }
            WindowType::Blackman => {
                for i in 0..size {
                    let factor = 2.0 * PI * i as f64 / (size - 1) as f64;
                    window[i] = 0.42 - 0.5 * factor.cos() + 0.08 * (2.0 * factor).cos();
                }
            }
            WindowType::Gaussian => {
                let sigma = (size as f64) / 6.0;
                let center = (size - 1) as f64 / 2.0;
                for i in 0..size {
                    let x = (i as f64 - center) / sigma;
                    window[i] = (-0.5 * x * x).exp();
                }
            }
            WindowType::Rectangular => {
                window.fill(1.0);
            }
        }

        Ok(window)
    }

    fn detrend_series(
        &self,
        x: &Array1<f64>,
        method: DetrendMethod,
    ) -> CorrelationResult<Array1<f64>> {
        match method {
            DetrendMethod::None => Ok(x.clone()),
            DetrendMethod::Mean => {
                let mean = x.mean().unwrap_or(0.0);
                Ok(x - mean)
            }
            DetrendMethod::Linear => {
                let n = x.len() as f64;
                let t: Array1<f64> = Array1::from_iter((0..x.len()).map(|i| i as f64));

                // Linear regression: y = a + b*t
                let sum_t = t.sum();
                let sum_y = x.sum();
                let sum_tt = t.mapv(|ti| ti * ti).sum();
                let sum_ty = t.iter().zip(x.iter()).map(|(ti, yi)| ti * yi).sum::<f64>();

                let b = (n * sum_ty - sum_t * sum_y) / (n * sum_tt - sum_t * sum_t);
                let a = (sum_y - b * sum_t) / n;

                let trend = t.mapv(|ti| a + b * ti);
                Ok(x - &trend)
            }
        }
    }

    fn compute_fft(&self, x: &Array1<f64>) -> CorrelationResult<Vec<Complex>> {
        // Simplified FFT implementation using DFT
        let n = x.len();
        let mut result = vec![Complex { re: 0.0, im: 0.0 }; n];

        #[allow(clippy::needless_range_loop)]
        for k in 0..n {
            let mut sum = Complex { re: 0.0, im: 0.0 };
            for t in 0..n {
                let angle = -2.0 * PI * k as f64 * t as f64 / n as f64;
                sum.re += x[t] * angle.cos();
                sum.im += x[t] * angle.sin();
            }
            result[k] = sum;
        }

        Ok(result)
    }

    fn coherence_confidence_threshold(&self, confidence_level: f64, n_segments: usize) -> f64 {
        // Approximation for coherence confidence threshold
        let alpha = 1.0 - confidence_level;
        let dof = 2 * n_segments;

        // For large DOF, use chi-squared approximation
        if dof > 30 {
            let z = self.normal_quantile(1.0 - alpha);
            1.0 - (z * z / dof as f64).exp()
        } else {
            // Conservative threshold for small samples
            0.5
        }
    }
}

impl Default for CorrelationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Complex number representation for FFT
#[derive(Debug, Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_cross_correlation() {
        let n = 50;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..n).map(|i| ((i as f64 + 2.0) * 0.1).sin()).collect());

        let analyzer = CorrelationAnalyzer::new();
        let config = CrossCorrelationConfig::default();
        let result = analyzer.cross_correlation(&x, &y, &config).unwrap();

        assert_eq!(result.correlations.len(), 2 * config.max_lag + 1);
        assert!(result.max_correlation.abs() <= 1.0 + f64::EPSILON * 10.0);
        assert!(result.lag_at_max.abs() <= config.max_lag as i32);
    }

    #[test]
    fn test_dynamic_time_warping() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0]);

        let analyzer = CorrelationAnalyzer::new();
        let config = DTWConfig::default();
        let result = analyzer.dynamic_time_warping(&x, &y, &config).unwrap();

        assert!(result.distance >= 0.0);
        assert!(!result.warping_path.is_empty());
        assert_eq!(result.cost_matrix.nrows(), x.len());
        assert_eq!(result.cost_matrix.ncols(), y.len());
    }

    #[test]
    fn test_coherence_analysis() {
        let n = 512;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec(
            (0..n)
                .map(|i| (i as f64 * 0.1).sin() + 0.1 * rand::random::<f64>())
                .collect(),
        );

        let analyzer = CorrelationAnalyzer::new();
        let config = CoherenceConfig::default();
        let result = analyzer.coherence_analysis(&x, &y, &config).unwrap();

        assert!(!result.coherence.is_empty());
        assert!(!result.frequencies.is_empty());
        assert_eq!(result.coherence.len(), result.frequencies.len());

        // Coherence values should be between 0 and 1
        for &coh in result.coherence.iter() {
            assert!((0.0..=1.0).contains(&coh));
        }
    }
}
