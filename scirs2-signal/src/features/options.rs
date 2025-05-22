use std::fmt::Debug;

/// Time series feature extraction options
#[derive(Debug, Clone)]
pub struct FeatureOptions {
    /// Statistical features to extract
    pub statistical: bool,
    /// Spectral features to extract
    pub spectral: bool,
    /// Entropy features to extract
    pub entropy: bool,
    /// Linear trend features to extract
    pub trend: bool,
    /// Zero-crossing-based features to extract
    pub zero_crossings: bool,
    /// Peak-based features to extract
    pub peaks: bool,
    /// Sample rate (needed for some features)
    pub sample_rate: Option<f64>,
    /// Fast calculation mode (may reduce accuracy)
    pub fast_mode: bool,
}

impl Default for FeatureOptions {
    fn default() -> Self {
        Self {
            statistical: true,
            spectral: true,
            entropy: true,
            trend: true,
            zero_crossings: true,
            peaks: true,
            sample_rate: None,
            fast_mode: false,
        }
    }
}
