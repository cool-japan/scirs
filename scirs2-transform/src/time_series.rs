//! Time series feature extraction
//!
//! This module provides utilities for extracting features from time series data,
//! including Fourier features, wavelet features, and lag features.

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use scirs2_core::numeric::Complex;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_fft::fft;

use crate::error::{Result, TransformError};

/// Fourier feature extractor for time series
///
/// Extracts frequency domain features using Fast Fourier Transform (FFT).
/// Useful for capturing periodic patterns in time series data.
#[derive(Debug, Clone)]
pub struct FourierFeatures {
    /// Number of Fourier components to extract
    n_components: usize,
    /// Whether to include phase information
    include_phase: bool,
    /// Whether to normalize by series length
    normalize: bool,
    /// Sampling frequency (if known)
    sampling_freq: Option<f64>,
}

impl FourierFeatures {
    /// Create a new FourierFeatures extractor
    ///
    /// # Arguments
    /// * `n_components` - Number of frequency components to extract
    pub fn new(ncomponents: usize) -> Self {
        FourierFeatures {
            n_components: ncomponents,
            include_phase: false,
            normalize: true,
            sampling_freq: None,
        }
    }

    /// Include phase information in features
    pub fn with_phase(mut self) -> Self {
        self.include_phase = true;
        self
    }

    /// Set sampling frequency
    pub fn with_sampling_freq(mut self, freq: f64) -> Self {
        self.sampling_freq = Some(freq);
        self
    }

    /// Extract Fourier features from a single time series
    fn extract_features_1d<S>(&self, x: &ArrayBase<S, Ix1>) -> Result<Array1<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = x.len();
        if n == 0 {
            return Err(TransformError::InvalidInput(
                "Empty time series".to_string(),
            ));
        }

        // Convert to f64 for FFT
        let real_data: Vec<f64> = x
            .iter()
            .map(|&val| NumCast::from(val).unwrap_or(0.0))
            .collect();

        // Compute FFT
        let fft_result = fft(&real_data, None)?;

        // Extract features (only positive frequencies due to symmetry)
        let n_freq = (n / 2).min(self.n_components);
        let mut features = if self.include_phase {
            Array1::zeros(n_freq * 2)
        } else {
            Array1::zeros(n_freq)
        };

        let norm_factor = if self.normalize { 1.0 / n as f64 } else { 1.0 };

        for i in 0..n_freq {
            let magnitude =
                (fft_result[i].re * fft_result[i].re + fft_result[i].im * fft_result[i].im).sqrt()
                    * norm_factor;

            features[i] = magnitude;

            if self.include_phase && magnitude > 1e-10 {
                let phase = fft_result[i].im.atan2(fft_result[i].re);
                features[n_freq + i] = phase;
            }
        }

        Ok(features)
    }

    /// Transform time series data to Fourier features
    ///
    /// # Arguments
    /// * `x` - Time series data, shape (n_samples, n_timesteps)
    ///
    /// # Returns
    /// * Fourier features, shape (n_samples, n_features)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let n_features = if self.include_phase {
            self.n_components * 2
        } else {
            self.n_components
        };

        let mut result = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            let features = self.extract_features_1d(&x.row(i))?;
            let feat_len = features.len().min(n_features);
            result
                .slice_mut(scirs2_core::ndarray::s![i, ..feat_len])
                .assign(&features.slice(scirs2_core::ndarray::s![..feat_len]));
        }

        Ok(result)
    }
}

/// Lag feature extractor for time series
///
/// Creates lagged versions of time series as features. Useful for
/// autoregressive modeling and capturing temporal dependencies.
#[derive(Debug, Clone)]
pub struct LagFeatures {
    /// List of lags to include
    lags: Vec<usize>,
    /// Whether to drop NaN values resulting from lagging
    drop_na: bool,
}

impl LagFeatures {
    /// Create a new LagFeatures extractor
    ///
    /// # Arguments
    /// * `lags` - List of lag values (e.g., vec![1, 2, 3] for lags 1, 2, and 3)
    pub fn new(lags: Vec<usize>) -> Self {
        LagFeatures {
            lags,
            drop_na: true,
        }
    }

    /// Create with a range of lags
    pub fn with_range(start: usize, end: usize) -> Self {
        let lags = (start..=end).collect();
        LagFeatures {
            lags,
            drop_na: true,
        }
    }

    /// Set whether to drop NaN values
    pub fn with_drop_na(mut self, dropna: bool) -> Self {
        self.drop_na = dropna;
        self
    }

    /// Transform time series data to lag features
    ///
    /// # Arguments
    /// * `x` - Time series data, shape (n_timesteps,) or (n_samples, n_timesteps)
    ///
    /// # Returns
    /// * Lag features
    pub fn transform_1d<S>(&self, x: &ArrayBase<S, Ix1>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = x.len();
        let max_lag = *self.lags.iter().max().unwrap_or(&0);

        if max_lag >= n {
            return Err(TransformError::InvalidInput(format!(
                "Maximum lag {max_lag} must be less than series length {n}"
            )));
        }

        let start_idx = if self.drop_na { max_lag } else { 0 };
        let n_samples = n - start_idx;
        let n_features = self.lags.len() + 1; // Original + lags

        let mut result = Array2::zeros((n_samples, n_features));

        // Original series
        for i in 0..n_samples {
            result[[i, 0]] = NumCast::from(x[start_idx + i]).unwrap_or(0.0);
        }

        // Lagged features
        for (lag_idx, &lag) in self.lags.iter().enumerate() {
            for i in 0..n_samples {
                let idx = start_idx + i;
                if idx >= lag {
                    result[[i, lag_idx + 1]] = NumCast::from(x[idx - lag]).unwrap_or(0.0);
                } else if !self.drop_na {
                    result[[i, lag_idx + 1]] = f64::NAN;
                }
            }
        }

        Ok(result)
    }

    /// Transform multiple time series
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Vec<Array2<f64>>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_series = x.shape()[0];
        let mut results = Vec::new();

        for i in 0..n_series {
            let series = x.row(i);
            let lag_features = self.transform_1d(&series)?;
            results.push(lag_features);
        }

        Ok(results)
    }
}

/// Wavelet feature extractor for time series
///
/// Extracts features using wavelet decomposition. Useful for
/// multi-resolution analysis of time series.
#[derive(Debug, Clone)]
pub struct WaveletFeatures {
    /// Wavelet type: 'db1' (Haar), 'db2', 'db4', 'db8', 'bior2.2', 'bior4.4'
    wavelet: String,
    /// Decomposition level
    level: usize,
    /// Whether to include approximation coefficients
    include_approx: bool,
}

impl WaveletFeatures {
    /// Create a new WaveletFeatures extractor
    ///
    /// # Arguments
    /// * `wavelet` - Wavelet type (e.g., "db1" for Haar wavelet)
    /// * `level` - Decomposition level
    pub fn new(wavelet: &str, level: usize) -> Self {
        WaveletFeatures {
            wavelet: wavelet.to_string(),
            level,
            include_approx: true,
        }
    }

    /// Set whether to include approximation coefficients
    pub fn with_include_approx(mut self, include: bool) -> Self {
        self.include_approx = include;
        self
    }

    /// Haar wavelet transform (simplified)
    #[allow(dead_code)]
    fn haar_transform(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = x.len();
        let mut approx = Vec::with_capacity(n / 2);
        let mut detail = Vec::with_capacity(n / 2);

        for i in (0..n).step_by(2) {
            if i + 1 < n {
                approx.push((x[i] + x[i + 1]) / 2.0_f64.sqrt());
                detail.push((x[i] - x[i + 1]) / 2.0_f64.sqrt());
            } else {
                // Handle odd length
                approx.push(x[i]);
            }
        }

        (approx, detail)
    }

    /// Get wavelet filter coefficients
    fn get_wavelet_coeffs(&self) -> Result<(Vec<f64>, Vec<f64>)> {
        match self.wavelet.as_str() {
            "db1" | "haar" => {
                // Haar wavelet
                let h = vec![1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];
                let g = vec![1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt()];
                Ok((h, g))
            }
            "db2" => {
                // Daubechies-2 wavelet
                let sqrt3 = 3.0_f64.sqrt();
                let denom = 4.0 * 2.0_f64.sqrt();
                let h = vec![
                    (1.0 + sqrt3) / denom,
                    (3.0 + sqrt3) / denom,
                    (3.0 - sqrt3) / denom,
                    (1.0 - sqrt3) / denom,
                ];
                let g = vec![h[3], -h[2], h[1], -h[0]];
                Ok((h, g))
            }
            "db4" => {
                // Daubechies-4 wavelet
                let h = vec![
                    -0.010597401784997,
                    0.032883011666983,
                    0.030841381835987,
                    -0.187034811718881,
                    -0.027983769416984,
                    0.630880767929590,
                    0.714846570552542,
                    0.230377813308855,
                ];
                let mut g = Vec::with_capacity(h.len());
                for (i, &coeff) in h.iter().enumerate() {
                    g.push(if i % 2 == 0 { coeff } else { -coeff });
                }
                g.reverse();
                Ok((h, g))
            }
            "db8" => {
                // Daubechies-8 wavelet
                let h = vec![
                    -0.00011747678400228,
                    0.0006754494059985,
                    -0.0003917403729959,
                    -0.00487035299301066,
                    0.008746094047015655,
                    0.013981027917015516,
                    -0.04408825393106472,
                    -0.01736930100202211,
                    0.128747426620186,
                    0.00047248457399797254,
                    -0.2840155429624281,
                    -0.015829105256023893,
                    0.5853546836548691,
                    0.6756307362980128,
                    0.3182301045617746,
                    0.05441584224308161,
                ];
                let mut g = Vec::with_capacity(h.len());
                for (i, &coeff) in h.iter().enumerate() {
                    g.push(if i % 2 == 0 { coeff } else { -coeff });
                }
                g.reverse();
                Ok((h, g))
            }
            "bior2.2" => {
                // Biorthogonal 2.2 wavelet
                let h = vec![
                    0.0,
                    -0.17677669529663687,
                    0.35355339059327373,
                    1.0606601717798214,
                    0.35355339059327373,
                    -0.17677669529663687,
                    0.0,
                ];
                let g = vec![
                    0.0,
                    0.35355339059327373,
                    -std::f64::consts::FRAC_1_SQRT_2,
                    0.35355339059327373,
                    0.0,
                ];
                Ok((h, g))
            }
            "bior4.4" => {
                // Biorthogonal 4.4 wavelet
                let h = vec![
                    0.0,
                    0.03314563036811941,
                    -0.06629126073623884,
                    -0.17677669529663687,
                    0.4198446513295126,
                    0.9943689110435825,
                    0.4198446513295126,
                    -0.17677669529663687,
                    -0.06629126073623884,
                    0.03314563036811941,
                    0.0,
                ];
                let g = vec![
                    0.0,
                    -0.06453888262893856,
                    0.04068941760955867,
                    0.41809227322161724,
                    -0.7884856164056651,
                    0.41809227322161724,
                    0.04068941760955867,
                    -0.06453888262893856,
                    0.0,
                ];
                Ok((h, g))
            }
            _ => Err(TransformError::InvalidInput(format!(
                "Unsupported wavelet type: {}",
                self.wavelet
            ))),
        }
    }

    /// Generic wavelet transform using filter bank
    fn wavelet_transform(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let (h, g) = self.get_wavelet_coeffs()?;

        if x.len() < h.len() {
            return Err(TransformError::InvalidInput(
                "Input signal too short for selected wavelet".to_string(),
            ));
        }

        let n = x.len();
        let mut approx = Vec::with_capacity(n / 2);
        let mut detail = Vec::with_capacity(n / 2);

        // Convolution with downsampling
        for i in (0..n).step_by(2) {
            let mut h_sum = 0.0;
            let mut g_sum = 0.0;

            for (j, (&h_coeff, &g_coeff)) in h.iter().zip(g.iter()).enumerate() {
                let idx = (i + j) % n; // Periodic boundary condition
                h_sum += h_coeff * x[idx];
                g_sum += g_coeff * x[idx];
            }

            approx.push(h_sum);
            detail.push(g_sum);
        }

        Ok((approx, detail))
    }

    /// Multi-level wavelet decomposition
    fn wavelet_decompose(&self, x: &[f64]) -> Result<Vec<Vec<f64>>> {
        let mut coefficients = Vec::new();
        let mut current = x.to_vec();

        for _ in 0..self.level {
            let (approx, detail) = self.wavelet_transform(&current)?;
            coefficients.push(detail);
            current = approx;

            if current.len() < 2 {
                break;
            }
        }

        if self.include_approx {
            coefficients.push(current);
        }

        Ok(coefficients)
    }

    /// Extract wavelet features from a single time series
    fn extract_features_1d<S>(&self, x: &ArrayBase<S, Ix1>) -> Result<Array1<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_vec: Vec<f64> = x.iter().map(|&v| NumCast::from(v).unwrap_or(0.0)).collect();

        let coefficients = self.wavelet_decompose(&x_vec)?;

        // Calculate total number of features
        let n_features: usize = coefficients.iter().map(|c| c.len()).sum();
        let mut features = Array1::zeros(n_features);

        let mut idx = 0;
        for coeff_level in coefficients {
            for &coeff in &coeff_level {
                features[idx] = coeff;
                idx += 1;
            }
        }

        Ok(features)
    }

    /// Transform time series data to wavelet features
    ///
    /// # Arguments
    /// * `x` - Time series data, shape (n_samples, n_timesteps)
    ///
    /// # Returns
    /// * Wavelet features
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Vec<Array1<f64>>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let mut results = Vec::new();

        for i in 0..n_samples {
            let features = self.extract_features_1d(&x.row(i))?;
            results.push(features);
        }

        Ok(results)
    }
}

/// Combined time series feature extractor
///
/// Combines multiple feature extraction methods for comprehensive
/// time series feature engineering.
#[derive(Debug, Clone)]
pub struct TimeSeriesFeatures {
    /// Whether to include Fourier features
    use_fourier: bool,
    /// Whether to include lag features
    use_lags: bool,
    /// Whether to include wavelet features
    use_wavelets: bool,
    /// Fourier feature configuration
    fourier_config: Option<FourierFeatures>,
    /// Lag feature configuration
    lag_config: Option<LagFeatures>,
    /// Wavelet feature configuration
    wavelet_config: Option<WaveletFeatures>,
}

impl TimeSeriesFeatures {
    /// Create a new combined feature extractor
    pub fn new() -> Self {
        TimeSeriesFeatures {
            use_fourier: true,
            use_lags: true,
            use_wavelets: false,
            fourier_config: Some(FourierFeatures::new(10)),
            lag_config: Some(LagFeatures::with_range(1, 5)),
            wavelet_config: None,
        }
    }

    /// Configure Fourier features
    pub fn with_fourier(mut self, n_components: usize, includephase: bool) -> Self {
        self.use_fourier = true;
        let mut fourier = FourierFeatures::new(n_components);
        if includephase {
            fourier = fourier.with_phase();
        }
        self.fourier_config = Some(fourier);
        self
    }

    /// Configure lag features
    pub fn with_lags(mut self, lags: Vec<usize>) -> Self {
        self.use_lags = true;
        self.lag_config = Some(LagFeatures::new(lags));
        self
    }

    /// Configure wavelet features
    pub fn with_wavelets(mut self, wavelet: &str, level: usize) -> Self {
        self.use_wavelets = true;
        self.wavelet_config = Some(WaveletFeatures::new(wavelet, level));
        self
    }
}

impl Default for TimeSeriesFeatures {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_fourier_features() {
        // Create a simple sinusoidal signal
        let n = 100;
        let mut signal = Vec::new();
        for i in 0..n {
            let t = i as f64 / n as f64 * 4.0 * std::f64::consts::PI;
            signal.push((t).sin() + 0.5 * (2.0 * t).sin());
        }
        let x = Array::from_shape_vec((1, n), signal).unwrap();

        let fourier = FourierFeatures::new(10);
        let features = fourier.transform(&x).unwrap();

        assert_eq!(features.shape(), &[1, 10]);

        // DC component should be near zero for this signal
        assert!(features[[0, 0]].abs() < 1e-10);

        // The fundamental frequency components should have significant magnitude
        // sin(t) component should be at index 1
        // sin(2t) component should be at index 2
        assert!(features[[0, 1]] > 0.1); // Fundamental frequency
        assert!(features[[0, 2]] > 0.04); // Second harmonic (0.5 amplitude)
    }

    #[test]
    fn test_lag_features() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let lag_extractor = LagFeatures::new(vec![1, 2]);
        let features = lag_extractor.transform_1d(&x.view()).unwrap();

        // Should have 4 samples (6 - max_lag(2)) and 3 features (original + 2 lags)
        assert_eq!(features.shape(), &[4, 3]);

        // Check first row: x[2]=3, x[1]=2, x[0]=1
        assert_eq!(features[[0, 0]], 3.0);
        assert_eq!(features[[0, 1]], 2.0);
        assert_eq!(features[[0, 2]], 1.0);
    }

    #[test]
    fn test_wavelet_features() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let x_2d = x.clone().into_shape_with_order((1, 8)).unwrap();

        let wavelet = WaveletFeatures::new("db1", 2);
        let features = wavelet.transform(&x_2d).unwrap();

        assert!(!features.is_empty());
        assert!(features[0].len() > 0);
    }
}
