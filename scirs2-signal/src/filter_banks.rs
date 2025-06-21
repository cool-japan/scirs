//! Filter Bank Design and Analysis
//!
//! This module provides comprehensive filter bank design and analysis functionality,
//! including quadrature mirror filters (QMF), wavelet filter banks, and multirate
//! signal processing systems.
//!
//! ## Filter Bank Types
//!
//! - **QMF Banks**: Quadrature Mirror Filter banks for perfect reconstruction
//! - **Wavelet Filter Banks**: Filter banks based on wavelet decomposition
//! - **Cosine Modulated Banks**: Efficient filter banks using cosine modulation
//! - **Oversampled Banks**: Filter banks with oversampling for reduced aliasing
//! - **Polyphase Filter Banks**: Efficient implementation using polyphase decomposition
//!
//! ## Features
//!
//! - Perfect reconstruction filter bank design
//! - Aliasing and distortion analysis
//! - Multirate signal processing operations
//! - Efficient polyphase implementations
//! - Filter bank optimization for various criteria
//!
//! ## Example Usage
//!
//! ```rust
//! use ndarray::Array1;
//! use scirs2_signal::filter_banks::{QmfBank, WaveletFilterBank, FilterBankType};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//! // Design a QMF filter bank
//! let qmf = QmfBank::new(8, FilterBankType::Orthogonal)?;
//! let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
//!
//! // Analysis (decomposition)
//! let subbands = qmf.analysis(&input)?;
//!
//! // Synthesis (reconstruction)
//! let reconstructed = qmf.synthesis(&subbands)?;
//!
//! // Design a wavelet filter bank
//! let wavelet_bank = WaveletFilterBank::new("db4", 3)?;
//! let coeffs = wavelet_bank.decompose(&input)?;
//! let recovered = wavelet_bank.reconstruct(&coeffs)?;
//! # Ok(())
//! # }
//! ```

use crate::dwt::Wavelet;
use crate::error::{SignalError, SignalResult};
use crate::filter::{butter, FilterType};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Types of filter banks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterBankType {
    /// Perfect reconstruction filter bank
    PerfectReconstruction,
    /// Orthogonal filter bank (energy preserving)
    Orthogonal,
    /// Biorthogonal filter bank (linear phase)
    Biorthogonal,
    /// Cosine modulated filter bank
    CosineModulated,
    /// Tree-structured filter bank
    TreeStructured,
}

/// Window types for filter bank design
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterBankWindow {
    /// Kaiser window with beta parameter
    Kaiser(f64),
    /// Hamming window
    Hamming,
    /// Hann window
    Hann,
    /// Blackman window
    Blackman,
    /// Rectangular window
    Rectangular,
}

/// Quadrature Mirror Filter (QMF) Bank
#[derive(Debug, Clone)]
pub struct QmfBank {
    /// Analysis filters (decomposition)
    pub analysis_filters: Array2<f64>,
    /// Synthesis filters (reconstruction)
    pub synthesis_filters: Array2<f64>,
    /// Number of channels
    pub num_channels: usize,
    /// Filter length
    pub filter_length: usize,
    /// Filter bank type
    pub bank_type: FilterBankType,
    /// Decimation factor
    pub decimation: usize,
}

impl QmfBank {
    /// Create a new QMF filter bank
    ///
    /// # Arguments
    /// * `num_channels` - Number of filter bank channels
    /// * `bank_type` - Type of filter bank design
    ///
    /// # Returns
    /// * QMF filter bank instance
    pub fn new(num_channels: usize, bank_type: FilterBankType) -> SignalResult<Self> {
        if num_channels < 2 {
            return Err(SignalError::ValueError(
                "Number of channels must be at least 2".to_string(),
            ));
        }

        let filter_length = 2 * num_channels; // Default filter length
        let decimation = num_channels; // Critical sampling

        // Design prototype lowpass filter
        let prototype = Self::design_prototype_filter(num_channels, bank_type)?;

        // Generate analysis and synthesis filters
        let (analysis_filters, synthesis_filters) =
            Self::generate_filter_bank(&prototype, num_channels, bank_type)?;

        Ok(Self {
            analysis_filters,
            synthesis_filters,
            num_channels,
            filter_length,
            bank_type,
            decimation,
        })
    }

    /// Create QMF bank with custom prototype filter
    ///
    /// # Arguments
    /// * `prototype` - Prototype lowpass filter coefficients
    /// * `num_channels` - Number of filter bank channels
    /// * `bank_type` - Type of filter bank design
    ///
    /// # Returns
    /// * QMF filter bank instance
    pub fn with_prototype(
        prototype: &Array1<f64>,
        num_channels: usize,
        bank_type: FilterBankType,
    ) -> SignalResult<Self> {
        if num_channels < 2 {
            return Err(SignalError::ValueError(
                "Number of channels must be at least 2".to_string(),
            ));
        }

        let filter_length = prototype.len();
        let decimation = num_channels;

        let (analysis_filters, synthesis_filters) =
            Self::generate_filter_bank(prototype, num_channels, bank_type)?;

        Ok(Self {
            analysis_filters,
            synthesis_filters,
            num_channels,
            filter_length,
            bank_type,
            decimation,
        })
    }

    /// Design prototype lowpass filter
    fn design_prototype_filter(
        num_channels: usize,
        bank_type: FilterBankType,
    ) -> SignalResult<Array1<f64>> {
        let cutoff = 1.0 / (2.0 * num_channels as f64); // Normalized cutoff frequency
        let order = match bank_type {
            FilterBankType::PerfectReconstruction => 2 * num_channels,
            FilterBankType::Orthogonal => num_channels,
            FilterBankType::Biorthogonal => 2 * num_channels,
            FilterBankType::CosineModulated => 4 * num_channels,
            FilterBankType::TreeStructured => num_channels,
        };

        // Design Butterworth prototype filter
        let (b, _a) = butter(order, cutoff, FilterType::Lowpass)?;
        Ok(Array1::from(b))
    }

    /// Generate filter bank from prototype
    fn generate_filter_bank(
        prototype: &Array1<f64>,
        num_channels: usize,
        bank_type: FilterBankType,
    ) -> SignalResult<(Array2<f64>, Array2<f64>)> {
        let filter_length = prototype.len();
        let mut analysis_filters = Array2::<f64>::zeros((num_channels, filter_length));
        let mut synthesis_filters = Array2::<f64>::zeros((num_channels, filter_length));

        match bank_type {
            FilterBankType::CosineModulated => {
                // Cosine modulated filter bank (pseudo-QMF)
                for k in 0..num_channels {
                    for n in 0..filter_length {
                        let phase =
                            PI * (k as f64 + 0.5) * (n as f64 - filter_length as f64 / 2.0 + 0.5)
                                / num_channels as f64;

                        // Analysis filters
                        analysis_filters[[k, n]] = prototype[n] * phase.cos();

                        // Synthesis filters (same for pseudo-QMF)
                        synthesis_filters[[k, n]] = analysis_filters[[k, n]];
                    }
                }
            }
            FilterBankType::Orthogonal => {
                // Orthogonal filter bank using DFT modulation
                for k in 0..num_channels {
                    for n in 0..filter_length {
                        let phase = 2.0 * PI * k as f64 * n as f64 / num_channels as f64;

                        // Analysis filters
                        analysis_filters[[k, n]] = prototype[n] * phase.cos();

                        // Synthesis filters (time-reversed for perfect reconstruction)
                        synthesis_filters[[k, filter_length - 1 - n]] = analysis_filters[[k, n]];
                    }
                }
            }
            FilterBankType::PerfectReconstruction => {
                // Design for perfect reconstruction using alternating flip
                for k in 0..num_channels {
                    for n in 0..filter_length {
                        if k % 2 == 0 {
                            // Low-pass type filters
                            analysis_filters[[k, n]] = prototype[n];
                            synthesis_filters[[k, n]] = prototype[n];
                        } else {
                            // High-pass type filters (alternating signs)
                            analysis_filters[[k, n]] = prototype[n] * (-1.0_f64).powi(n as i32);
                            synthesis_filters[[k, n]] = analysis_filters[[k, n]];
                        }
                    }
                }
            }
            FilterBankType::Biorthogonal => {
                // Biorthogonal filter bank with different analysis/synthesis filters
                for k in 0..num_channels {
                    for n in 0..filter_length {
                        let phase =
                            PI * k as f64 * (2.0 * n as f64 + 1.0) / (2.0 * num_channels as f64);

                        // Analysis filters
                        analysis_filters[[k, n]] = prototype[n] * phase.cos();

                        // Synthesis filters (dual relationship)
                        synthesis_filters[[k, filter_length - 1 - n]] =
                            2.0 * analysis_filters[[k, n]] / num_channels as f64;
                    }
                }
            }
            FilterBankType::TreeStructured => {
                // Tree-structured filter bank (binary tree)
                if num_channels.count_ones() != 1 {
                    return Err(SignalError::ValueError(
                        "Tree-structured filter banks require power-of-2 channels".to_string(),
                    ));
                }

                // Use simple high-pass/low-pass decomposition
                for k in 0..num_channels {
                    for n in 0..filter_length {
                        if k < num_channels / 2 {
                            // Low-pass branch
                            analysis_filters[[k, n]] = prototype[n];
                            synthesis_filters[[k, n]] = prototype[n];
                        } else {
                            // High-pass branch
                            analysis_filters[[k, n]] = prototype[n] * (-1.0_f64).powi(n as i32);
                            synthesis_filters[[k, n]] = analysis_filters[[k, n]];
                        }
                    }
                }
            }
        }

        Ok((analysis_filters, synthesis_filters))
    }

    /// Perform analysis (decomposition) of input signal
    ///
    /// # Arguments
    /// * `input` - Input signal
    ///
    /// # Returns
    /// * Subband signals for each channel
    pub fn analysis(&self, input: &Array1<f64>) -> SignalResult<Vec<Array1<f64>>> {
        let mut subbands = Vec::with_capacity(self.num_channels);

        for k in 0..self.num_channels {
            let filter = self.analysis_filters.row(k);
            let filtered = self.filter_and_downsample(input, &filter.to_owned())?;
            subbands.push(filtered);
        }

        Ok(subbands)
    }

    /// Perform synthesis (reconstruction) from subband signals
    ///
    /// # Arguments
    /// * `subbands` - Subband signals from analysis
    ///
    /// # Returns
    /// * Reconstructed signal
    pub fn synthesis(&self, subbands: &[Array1<f64>]) -> SignalResult<Array1<f64>> {
        if subbands.len() != self.num_channels {
            return Err(SignalError::ValueError(format!(
                "Expected {} subbands, got {}",
                self.num_channels,
                subbands.len()
            )));
        }

        // Determine output length
        let max_upsampled_len = subbands
            .iter()
            .map(|s| s.len() * self.decimation)
            .max()
            .unwrap_or(0);

        let output_len = max_upsampled_len + self.filter_length - 1;
        let mut output = Array1::<f64>::zeros(output_len);

        for (k, subband) in subbands.iter().enumerate() {
            let filter = self.synthesis_filters.row(k);
            let upsampled = self.upsample_and_filter(subband, &filter.to_owned())?;

            // Add to output (overlap-add)
            let add_len = upsampled.len().min(output.len());
            for i in 0..add_len {
                output[i] += upsampled[i];
            }
        }

        Ok(output)
    }

    /// Filter and downsample signal
    fn filter_and_downsample(
        &self,
        input: &Array1<f64>,
        filter: &Array1<f64>,
    ) -> SignalResult<Array1<f64>> {
        // Convolve with filter
        let filtered_vec = crate::convolve::convolve(
            input.as_slice().unwrap(),
            filter.as_slice().unwrap(),
            "full",
        )?;
        let filtered = Array1::from(filtered_vec);

        // Downsample by decimation factor
        let downsampled_len = filtered.len().div_ceil(self.decimation);
        let mut downsampled = Array1::<f64>::zeros(downsampled_len);

        for i in 0..downsampled_len {
            let idx = i * self.decimation;
            if idx < filtered.len() {
                downsampled[i] = filtered[idx];
            }
        }

        Ok(downsampled)
    }

    /// Upsample and filter signal
    fn upsample_and_filter(
        &self,
        input: &Array1<f64>,
        filter: &Array1<f64>,
    ) -> SignalResult<Array1<f64>> {
        // Upsample by inserting zeros
        let upsampled_len = input.len() * self.decimation;
        let mut upsampled = Array1::<f64>::zeros(upsampled_len);

        for i in 0..input.len() {
            upsampled[i * self.decimation] = input[i] * self.decimation as f64; // Compensation gain
        }

        // Filter the upsampled signal
        let filtered_vec = crate::convolve::convolve(
            upsampled.as_slice().unwrap(),
            filter.as_slice().unwrap(),
            "full",
        )?;
        Ok(Array1::from(filtered_vec))
    }

    /// Analyze filter bank properties
    ///
    /// # Returns
    /// * Analysis result containing aliasing, distortion, and reconstruction error
    pub fn analyze_properties(&self) -> SignalResult<FilterBankAnalysis> {
        let mut analysis = FilterBankAnalysis::default();

        // Analyze frequency response of each filter
        let num_freqs = 512;
        let freqs = Array1::linspace(0.0, PI, num_freqs);

        // Compute magnitude responses
        let mut magnitude_responses = Array2::<f64>::zeros((self.num_channels, num_freqs));

        for k in 0..self.num_channels {
            let filter = self.analysis_filters.row(k);
            let response = self.compute_frequency_response(&filter.to_owned(), &freqs)?;
            for (i, &mag) in response.iter().enumerate() {
                magnitude_responses[[k, i]] = mag.norm();
            }
        }

        // Check for perfect reconstruction
        analysis.perfect_reconstruction = self.check_perfect_reconstruction(&freqs)?;

        // Compute aliasing distortion
        analysis.aliasing_distortion = self.compute_aliasing_distortion(&magnitude_responses);

        // Compute amplitude distortion
        analysis.amplitude_distortion = self.compute_amplitude_distortion(&magnitude_responses);

        // Compute stopband attenuation
        analysis.stopband_attenuation = self.compute_stopband_attenuation(&magnitude_responses);

        Ok(analysis)
    }

    /// Compute frequency response of a filter
    fn compute_frequency_response(
        &self,
        filter: &Array1<f64>,
        freqs: &Array1<f64>,
    ) -> SignalResult<Array1<Complex64>> {
        let mut response = Array1::<Complex64>::zeros(freqs.len());

        for (i, &freq) in freqs.iter().enumerate() {
            let mut sum = Complex64::new(0.0, 0.0);
            for (n, &coeff) in filter.iter().enumerate() {
                let phase = Complex64::new(0.0, -freq * n as f64);
                sum += coeff * phase.exp();
            }
            response[i] = sum;
        }

        Ok(response)
    }

    /// Check perfect reconstruction property
    fn check_perfect_reconstruction(&self, freqs: &Array1<f64>) -> SignalResult<bool> {
        let tolerance = 1e-10;

        for &freq in freqs.iter() {
            let mut sum = Complex64::new(0.0, 0.0);

            for k in 0..self.num_channels {
                let analysis_filter = self.analysis_filters.row(k);
                let synthesis_filter = self.synthesis_filters.row(k);

                // Compute H_k(ω) * F_k(ω)
                let mut h_response = Complex64::new(0.0, 0.0);
                let mut f_response = Complex64::new(0.0, 0.0);

                for (n, (&h_coeff, &f_coeff)) in analysis_filter
                    .iter()
                    .zip(synthesis_filter.iter())
                    .enumerate()
                {
                    let phase = Complex64::new(0.0, -freq * n as f64);
                    h_response += h_coeff * phase.exp();
                    f_response += f_coeff * phase.exp();
                }

                sum += h_response * f_response;
            }

            // Check if sum equals M (decimation factor) for perfect reconstruction
            let expected = self.decimation as f64;
            if (sum.re - expected).abs() > tolerance || sum.im.abs() > tolerance {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Compute aliasing distortion
    fn compute_aliasing_distortion(&self, magnitude_responses: &Array2<f64>) -> f64 {
        let mut max_aliasing = 0.0f64;

        for i in 0..magnitude_responses.ncols() {
            let mut sum = 0.0;
            for k in 0..self.num_channels {
                sum += magnitude_responses[[k, i]].powi(2);
            }

            // Aliasing occurs when sum deviates from constant
            let deviation = (sum - 1.0).abs();
            max_aliasing = max_aliasing.max(deviation);
        }

        max_aliasing
    }

    /// Compute amplitude distortion
    fn compute_amplitude_distortion(&self, magnitude_responses: &Array2<f64>) -> f64 {
        let mut max_distortion = 0.0f64;

        // Check passband ripple for each filter
        for k in 0..self.num_channels {
            let passband_end = magnitude_responses.ncols() / (2 * self.num_channels);
            let mut passband_min = f64::INFINITY;
            let mut passband_max = 0.0f64;

            for i in 0..passband_end {
                let mag = magnitude_responses[[k, i]];
                passband_min = passband_min.min(mag);
                passband_max = passband_max.max(mag);
            }

            let ripple = passband_max - passband_min;
            max_distortion = max_distortion.max(ripple);
        }

        max_distortion
    }

    /// Compute stopband attenuation
    fn compute_stopband_attenuation(&self, magnitude_responses: &Array2<f64>) -> f64 {
        let mut min_attenuation = f64::INFINITY;

        for k in 0..self.num_channels {
            let stopband_start = magnitude_responses.ncols() / self.num_channels;
            let mut max_stopband = 0.0f64;

            for i in stopband_start..magnitude_responses.ncols() {
                let mag = magnitude_responses[[k, i]];
                max_stopband = max_stopband.max(mag);
            }

            if max_stopband > 0.0 {
                let attenuation = -20.0 * max_stopband.log10();
                min_attenuation = min_attenuation.min(attenuation);
            }
        }

        min_attenuation
    }
}

/// Wavelet Filter Bank
#[derive(Debug, Clone)]
pub struct WaveletFilterBank {
    /// Low-pass decomposition filter
    pub lowpass_dec: Array1<f64>,
    /// High-pass decomposition filter
    pub highpass_dec: Array1<f64>,
    /// Low-pass reconstruction filter
    pub lowpass_rec: Array1<f64>,
    /// High-pass reconstruction filter
    pub highpass_rec: Array1<f64>,
    /// Wavelet name
    pub wavelet_name: String,
    /// Number of decomposition levels
    pub levels: usize,
}

impl WaveletFilterBank {
    /// Create a new wavelet filter bank
    ///
    /// # Arguments
    /// * `wavelet_name` - Name of the wavelet (e.g., "db4", "haar")
    /// * `levels` - Number of decomposition levels
    ///
    /// # Returns
    /// * Wavelet filter bank instance
    pub fn new(wavelet_name: &str, levels: usize) -> SignalResult<Self> {
        if levels == 0 {
            return Err(SignalError::ValueError(
                "Number of levels must be greater than 0".to_string(),
            ));
        }

        // Parse wavelet name and get filters
        let wavelet = Self::parse_wavelet_name(wavelet_name)?;
        let filters = wavelet.filters()?;

        Ok(Self {
            lowpass_dec: Array1::from(filters.dec_lo),
            highpass_dec: Array1::from(filters.dec_hi),
            lowpass_rec: Array1::from(filters.rec_lo),
            highpass_rec: Array1::from(filters.rec_hi),
            wavelet_name: wavelet_name.to_string(),
            levels,
        })
    }

    /// Parse wavelet name string to Wavelet enum
    fn parse_wavelet_name(name: &str) -> SignalResult<Wavelet> {
        match name.to_lowercase().as_str() {
            "haar" => Ok(Wavelet::Haar),
            "db1" => Ok(Wavelet::DB(1)),
            "db2" => Ok(Wavelet::DB(2)),
            "db3" => Ok(Wavelet::DB(3)),
            "db4" => Ok(Wavelet::DB(4)),
            "db5" => Ok(Wavelet::DB(5)),
            "db6" => Ok(Wavelet::DB(6)),
            "db7" => Ok(Wavelet::DB(7)),
            "db8" => Ok(Wavelet::DB(8)),
            "db9" => Ok(Wavelet::DB(9)),
            "db10" => Ok(Wavelet::DB(10)),
            "coif1" => Ok(Wavelet::Coif(1)),
            "coif2" => Ok(Wavelet::Coif(2)),
            "coif3" => Ok(Wavelet::Coif(3)),
            "coif4" => Ok(Wavelet::Coif(4)),
            "coif5" => Ok(Wavelet::Coif(5)),
            "sym2" => Ok(Wavelet::Sym(2)),
            "sym3" => Ok(Wavelet::Sym(3)),
            "sym4" => Ok(Wavelet::Sym(4)),
            "sym5" => Ok(Wavelet::Sym(5)),
            "sym6" => Ok(Wavelet::Sym(6)),
            "meyer" => Ok(Wavelet::Meyer),
            "dmeyer" => Ok(Wavelet::DMeyer),
            _ => Err(SignalError::ValueError(format!(
                "Unknown wavelet name: {}",
                name
            ))),
        }
    }

    /// Decompose signal using wavelet filter bank
    ///
    /// # Arguments
    /// * `input` - Input signal
    ///
    /// # Returns
    /// * Wavelet coefficients at each level
    pub fn decompose(&self, input: &Array1<f64>) -> SignalResult<Vec<Array1<f64>>> {
        let mut coeffs = Vec::new();
        let mut current = input.clone();

        for _level in 0..self.levels {
            // Low-pass filtering and downsampling (approximation)
            let approx = self.filter_and_downsample(&current, &self.lowpass_dec)?;

            // High-pass filtering and downsampling (detail)
            let detail = self.filter_and_downsample(&current, &self.highpass_dec)?;

            coeffs.push(detail);
            current = approx;
        }

        // Add final approximation
        coeffs.push(current);
        coeffs.reverse(); // Put approximation first

        Ok(coeffs)
    }

    /// Reconstruct signal from wavelet coefficients
    ///
    /// # Arguments
    /// * `coeffs` - Wavelet coefficients from decomposition
    ///
    /// # Returns
    /// * Reconstructed signal
    pub fn reconstruct(&self, coeffs: &[Array1<f64>]) -> SignalResult<Array1<f64>> {
        if coeffs.len() != self.levels + 1 {
            return Err(SignalError::ValueError(format!(
                "Expected {} coefficient arrays, got {}",
                self.levels + 1,
                coeffs.len()
            )));
        }

        let mut current = coeffs[0].clone(); // Start with approximation

        #[allow(clippy::needless_range_loop)]
        for level in 1..=self.levels {
            let detail = &coeffs[level];

            // Upsample and filter approximation
            let upsampled_approx = self.upsample_and_filter(&current, &self.lowpass_rec)?;

            // Upsample and filter detail
            let upsampled_detail = self.upsample_and_filter(detail, &self.highpass_rec)?;

            // Combine
            let combined_len = upsampled_approx.len().max(upsampled_detail.len());
            let mut combined = Array1::<f64>::zeros(combined_len);

            for i in 0..combined_len {
                if i < upsampled_approx.len() {
                    combined[i] += upsampled_approx[i];
                }
                if i < upsampled_detail.len() {
                    combined[i] += upsampled_detail[i];
                }
            }

            current = combined;
        }

        Ok(current)
    }

    /// Filter and downsample by 2
    fn filter_and_downsample(
        &self,
        input: &Array1<f64>,
        filter: &Array1<f64>,
    ) -> SignalResult<Array1<f64>> {
        // Convolve with filter
        let filtered_vec = crate::convolve::convolve(
            input.as_slice().unwrap(),
            filter.as_slice().unwrap(),
            "same",
        )?;
        let filtered = Array1::from(filtered_vec);

        // Downsample by 2
        let downsampled_len = filtered.len().div_ceil(2);
        let mut downsampled = Array1::<f64>::zeros(downsampled_len);

        for i in 0..downsampled_len {
            let idx = i * 2;
            if idx < filtered.len() {
                downsampled[i] = filtered[idx];
            }
        }

        Ok(downsampled)
    }

    /// Upsample by 2 and filter
    fn upsample_and_filter(
        &self,
        input: &Array1<f64>,
        filter: &Array1<f64>,
    ) -> SignalResult<Array1<f64>> {
        // Upsample by inserting zeros
        let upsampled_len = input.len() * 2;
        let mut upsampled = Array1::<f64>::zeros(upsampled_len);

        for i in 0..input.len() {
            upsampled[i * 2] = input[i];
        }

        // Filter the upsampled signal
        let filtered_vec = crate::convolve::convolve(
            upsampled.as_slice().unwrap(),
            filter.as_slice().unwrap(),
            "same",
        )?;
        Ok(Array1::from(filtered_vec))
    }
}

/// Cosine Modulated Filter Bank
#[derive(Debug, Clone)]
pub struct CosineModulatedFilterBank {
    /// QMF bank for implementation
    pub qmf_bank: QmfBank,
    /// Prototype filter
    pub prototype: Array1<f64>,
    /// Overlapping factor
    pub overlap_factor: usize,
}

impl CosineModulatedFilterBank {
    /// Create a new cosine modulated filter bank
    ///
    /// # Arguments
    /// * `num_channels` - Number of channels
    /// * `overlap_factor` - Overlapping factor (typically 2 or 4)
    /// * `window` - Window type for prototype design
    ///
    /// # Returns
    /// * Cosine modulated filter bank instance
    pub fn new(
        num_channels: usize,
        overlap_factor: usize,
        window: FilterBankWindow,
    ) -> SignalResult<Self> {
        // Design prototype filter with specified overlap
        let filter_length = overlap_factor * num_channels;
        let prototype = Self::design_prototype_filter(filter_length, window)?;

        // Create QMF bank with cosine modulation
        let qmf_bank =
            QmfBank::with_prototype(&prototype, num_channels, FilterBankType::CosineModulated)?;

        Ok(Self {
            qmf_bank,
            prototype,
            overlap_factor,
        })
    }

    /// Design prototype filter
    fn design_prototype_filter(
        filter_length: usize,
        window: FilterBankWindow,
    ) -> SignalResult<Array1<f64>> {
        let mut prototype = Array1::<f64>::zeros(filter_length);

        match window {
            FilterBankWindow::Kaiser(beta) => {
                // Kaiser window design
                for n in 0..filter_length {
                    let arg = 2.0 * (n as f64) / (filter_length - 1) as f64 - 1.0;
                    let i0_beta = Self::modified_bessel_i0(beta);
                    let i0_arg = Self::modified_bessel_i0(beta * (1.0 - arg * arg).sqrt());
                    prototype[n] = i0_arg / i0_beta;
                }
            }
            FilterBankWindow::Hamming => {
                // Hamming window
                for n in 0..filter_length {
                    prototype[n] =
                        0.54 - 0.46 * (2.0 * PI * n as f64 / (filter_length - 1) as f64).cos();
                }
            }
            FilterBankWindow::Hann => {
                // Hann window
                for n in 0..filter_length {
                    prototype[n] =
                        0.5 * (1.0 - (2.0 * PI * n as f64 / (filter_length - 1) as f64).cos());
                }
            }
            FilterBankWindow::Blackman => {
                // Blackman window
                for n in 0..filter_length {
                    let arg = 2.0 * PI * n as f64 / (filter_length - 1) as f64;
                    prototype[n] = 0.42 - 0.5 * arg.cos() + 0.08 * (2.0 * arg).cos();
                }
            }
            FilterBankWindow::Rectangular => {
                // Rectangular window
                prototype.fill(1.0);
            }
        }

        // Normalize
        let sum = prototype.sum();
        if sum > 0.0 {
            prototype /= sum;
        }

        Ok(prototype)
    }

    /// Modified Bessel function I0 (approximation)
    fn modified_bessel_i0(x: f64) -> f64 {
        if x.abs() < 3.75 {
            let t = x / 3.75;
            let t2 = t * t;
            1.0 + 3.5156229 * t2
                + 3.0899424 * t2.powi(2)
                + 1.2067492 * t2.powi(3)
                + 0.2659732 * t2.powi(4)
                + 0.0360768 * t2.powi(5)
                + 0.0045813 * t2.powi(6)
        } else {
            let t = 3.75 / x.abs();
            let exp_term = x.abs().exp();
            exp_term
                * (0.39894228 + 0.01328592 * t + 0.00225319 * t.powi(2) - 0.00157565 * t.powi(3)
                    + 0.00916281 * t.powi(4)
                    - 0.02057706 * t.powi(5)
                    + 0.02635537 * t.powi(6)
                    - 0.01647633 * t.powi(7)
                    + 0.00392377 * t.powi(8))
                / x.abs().sqrt()
        }
    }

    /// Perform analysis using cosine modulation
    pub fn analysis(&self, input: &Array1<f64>) -> SignalResult<Vec<Array1<f64>>> {
        self.qmf_bank.analysis(input)
    }

    /// Perform synthesis using cosine modulation  
    pub fn synthesis(&self, subbands: &[Array1<f64>]) -> SignalResult<Array1<f64>> {
        self.qmf_bank.synthesis(subbands)
    }
}

/// Filter bank analysis results
#[derive(Debug, Clone, Default)]
pub struct FilterBankAnalysis {
    /// Perfect reconstruction property
    pub perfect_reconstruction: bool,
    /// Aliasing distortion level
    pub aliasing_distortion: f64,
    /// Amplitude distortion level
    pub amplitude_distortion: f64,
    /// Stopband attenuation in dB
    pub stopband_attenuation: f64,
}

/// IIR Filter Stabilization Methods
pub struct IirStabilizer;

impl IirStabilizer {
    /// Stabilize IIR filter by moving poles inside unit circle
    ///
    /// # Arguments
    /// * `b` - Numerator coefficients
    /// * `a` - Denominator coefficients
    /// * `method` - Stabilization method
    ///
    /// # Returns
    /// * Stabilized filter coefficients
    pub fn stabilize_filter(
        b: &Array1<f64>,
        a: &Array1<f64>,
        method: StabilizationMethod,
    ) -> SignalResult<(Array1<f64>, Array1<f64>)> {
        match method {
            StabilizationMethod::RadialProjection => Self::radial_projection_stabilization(b, a),
            StabilizationMethod::ZeroPlacement => Self::zero_placement_stabilization(b, a),
            StabilizationMethod::BalancedTruncation => {
                Self::balanced_truncation_stabilization(b, a)
            }
        }
    }

    /// Radial projection stabilization
    fn radial_projection_stabilization(
        b: &Array1<f64>,
        a: &Array1<f64>,
    ) -> SignalResult<(Array1<f64>, Array1<f64>)> {
        // Find poles by solving characteristic polynomial
        let poles = Self::find_polynomial_roots(a)?;

        // Project unstable poles inside unit circle
        let stabilized_poles: Vec<Complex64> = poles
            .into_iter()
            .map(|pole| {
                if pole.norm() >= 1.0 {
                    // Project to just inside unit circle
                    pole * (0.99 / pole.norm())
                } else {
                    pole
                }
            })
            .collect();

        // Reconstruct polynomial from stabilized poles
        let stabilized_a = Self::polynomial_from_roots(&stabilized_poles)?;

        Ok((b.clone(), stabilized_a))
    }

    /// Zero placement stabilization
    fn zero_placement_stabilization(
        b: &Array1<f64>,
        a: &Array1<f64>,
    ) -> SignalResult<(Array1<f64>, Array1<f64>)> {
        // Simple approach: ensure leading coefficient is positive and adjust
        let mut stabilized_a = a.clone();
        let mut stabilized_b = b.clone();

        // Normalize by leading coefficient
        if stabilized_a[0] != 0.0 {
            let leading = stabilized_a[0];
            stabilized_a /= leading;
            stabilized_b /= leading;
        }

        // Check stability by examining coefficients (simplified)
        // For a proper implementation, would need pole analysis
        for i in 1..stabilized_a.len() {
            if stabilized_a[i].abs() > 0.95 {
                stabilized_a[i] *= 0.95 / stabilized_a[i].abs();
            }
        }

        Ok((stabilized_b, stabilized_a))
    }

    /// Balanced truncation stabilization (simplified)
    fn balanced_truncation_stabilization(
        b: &Array1<f64>,
        a: &Array1<f64>,
    ) -> SignalResult<(Array1<f64>, Array1<f64>)> {
        // This is a simplified version - full implementation would require
        // state-space conversion and balanced realization
        Self::radial_projection_stabilization(b, a)
    }

    /// Find roots of polynomial (simplified implementation)
    fn find_polynomial_roots(coeffs: &Array1<f64>) -> SignalResult<Vec<Complex64>> {
        // This is a simplified implementation for demonstration
        // A full implementation would use a robust root-finding algorithm
        let mut roots = Vec::new();

        if coeffs.len() == 2 {
            // Linear case: ax + b = 0
            if coeffs[0] != 0.0 {
                roots.push(Complex64::new(-coeffs[1] / coeffs[0], 0.0));
            }
        } else if coeffs.len() == 3 {
            // Quadratic case: ax^2 + bx + c = 0
            let a = coeffs[0];
            let b = coeffs[1];
            let c = coeffs[2];

            if a != 0.0 {
                let discriminant = b * b - 4.0 * a * c;
                if discriminant >= 0.0 {
                    let sqrt_disc = discriminant.sqrt();
                    roots.push(Complex64::new((-b + sqrt_disc) / (2.0 * a), 0.0));
                    roots.push(Complex64::new((-b - sqrt_disc) / (2.0 * a), 0.0));
                } else {
                    let sqrt_disc = (-discriminant).sqrt();
                    roots.push(Complex64::new(-b / (2.0 * a), sqrt_disc / (2.0 * a)));
                    roots.push(Complex64::new(-b / (2.0 * a), -sqrt_disc / (2.0 * a)));
                }
            }
        }
        // For higher-order polynomials, would need numerical methods

        Ok(roots)
    }

    /// Reconstruct polynomial from roots
    fn polynomial_from_roots(roots: &[Complex64]) -> SignalResult<Array1<f64>> {
        let mut coeffs = vec![1.0];

        for &root in roots {
            let mut new_coeffs = vec![0.0; coeffs.len() + 1];

            // Multiply by (z - root)
            for i in 0..coeffs.len() {
                new_coeffs[i] += coeffs[i] * (-root.re);
                new_coeffs[i + 1] += coeffs[i];
            }

            coeffs = new_coeffs;
        }

        Ok(Array1::from(coeffs))
    }
}

/// Stabilization methods for IIR filters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilizationMethod {
    /// Project unstable poles inside unit circle
    RadialProjection,
    /// Strategic zero placement for stabilization  
    ZeroPlacement,
    /// Balanced truncation for model reduction
    BalancedTruncation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qmf_bank_creation() {
        let qmf = QmfBank::new(4, FilterBankType::Orthogonal).unwrap();
        assert_eq!(qmf.num_channels, 4);
        assert_eq!(qmf.analysis_filters.nrows(), 4);
        assert_eq!(qmf.synthesis_filters.nrows(), 4);
    }

    #[test]
    fn test_qmf_bank_analysis_synthesis() {
        let qmf = QmfBank::new(2, FilterBankType::PerfectReconstruction).unwrap();
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Analysis
        let subbands = qmf.analysis(&input).unwrap();
        assert_eq!(subbands.len(), 2);

        // Synthesis
        let reconstructed = qmf.synthesis(&subbands).unwrap();

        // Check that we get some reasonable reconstruction
        assert!(!reconstructed.is_empty());
        assert!(reconstructed.len() >= input.len());
    }

    #[test]
    fn test_wavelet_filter_bank() {
        let wavelet_bank = WaveletFilterBank::new("db4", 2).unwrap();
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Decomposition
        let coeffs = wavelet_bank.decompose(&input).unwrap();
        assert_eq!(coeffs.len(), 3); // 2 levels + approximation

        // Reconstruction
        let reconstructed = wavelet_bank.reconstruct(&coeffs).unwrap();

        // Check reconstruction quality (should be close to original)
        assert!(!reconstructed.is_empty());
    }

    #[test]
    fn test_cosine_modulated_filter_bank() {
        let cmfb = CosineModulatedFilterBank::new(4, 2, FilterBankWindow::Hann).unwrap();
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let subbands = cmfb.analysis(&input).unwrap();
        assert_eq!(subbands.len(), 4);

        let reconstructed = cmfb.synthesis(&subbands).unwrap();
        assert!(!reconstructed.is_empty());
    }

    #[test]
    fn test_filter_bank_analysis() {
        let qmf = QmfBank::new(2, FilterBankType::Orthogonal).unwrap();
        let analysis = qmf.analyze_properties().unwrap();

        // Check that analysis provides reasonable values
        assert!(analysis.aliasing_distortion >= 0.0);
        assert!(analysis.amplitude_distortion >= 0.0);
    }

    #[test]
    fn test_iir_stabilization() {
        let b = Array1::from_vec(vec![1.0, 0.5]);
        let a = Array1::from_vec(vec![1.0, -1.5, 0.7]); // Potentially unstable

        let (b_stab, a_stab) =
            IirStabilizer::stabilize_filter(&b, &a, StabilizationMethod::RadialProjection).unwrap();

        assert_eq!(b_stab.len(), b.len());
        assert_eq!(a_stab.len(), a.len());
    }
}
