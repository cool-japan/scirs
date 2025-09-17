//! Configuration types and enums for STFT operations
//!
//! This module contains all the configuration structures, enums, and
//! related types used throughout the STFT implementation.

use crate::error::SignalError;

/// Configuration options for STFT
#[derive(Debug, Clone, Default)]
pub struct StftConfig {
    /// FFT mode (e.g., "real", "complex")
    pub fft_mode: Option<String>,
    /// FFT size override
    pub mfft: Option<usize>,
    /// Optional dual window for analysis/synthesis
    pub dual_win: Option<Vec<f64>>,
    /// Scaling option
    pub scale_to: Option<String>,
    /// Phase shift
    pub phase_shift: Option<isize>,
}

/// FFT mode options for STFT
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FftMode {
    /// Two-sided spectrum with negative frequencies preceding positive frequencies
    TwoSided,
    /// Two-sided spectrum with negative frequencies following positive frequencies
    Centered,
    /// One-sided spectrum (positive frequencies only)
    #[default]
    OneSided,
    /// One-sided spectrum with doubled amplitudes for energy conservation
    OneSided2X,
}

impl std::str::FromStr for FftMode {
    type Err = SignalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "twosided" => Ok(FftMode::TwoSided),
            "centered" => Ok(FftMode::Centered),
            "onesided" => Ok(FftMode::OneSided),
            "onesided2x" => Ok(FftMode::OneSided2X),
            _ => Err(SignalError::ValueError(format!(
                "Invalid FFT mode: '{}'. Valid options are: 'twosided', 'centered', 'onesided', 'onesided2x'",
                s
            ))),
        }
    }
}

/// Scaling options for STFT
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalingMode {
    /// No scaling (raw FFT values)
    #[default]
    None,
    /// Scale for magnitude spectrum
    Magnitude,
    /// Scale for power spectral density
    Psd,
}

impl std::str::FromStr for ScalingMode {
    type Err = SignalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(ScalingMode::None),
            "magnitude" => Ok(ScalingMode::Magnitude),
            "psd" => Ok(ScalingMode::Psd),
            _ => Err(SignalError::ValueError(format!(
                "Invalid scaling mode: '{}'. Valid options are: 'none', 'magnitude', 'psd'",
                s
            ))),
        }
    }
}

/// Configuration for memory-efficient STFT processing
#[derive(Debug, Clone)]
pub struct MemoryEfficientStftConfig {
    /// Maximum memory limit in MB for STFT processing
    pub memory_limit: usize,
    /// Chunk size for processing large signals
    pub chunk_size: Option<usize>,
    /// Overlap between chunks for seamless processing
    pub chunk_overlap: usize,
    /// Store only magnitude (not complex values)
    pub magnitude_only: bool,
    /// Use parallel processing
    pub parallel: bool,
}