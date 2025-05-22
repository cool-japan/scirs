//! Type definitions for image feature extraction

/// Image feature extraction options
#[derive(Debug, Clone)]
pub struct ImageFeatureOptions {
    /// Extract histogram features
    pub histogram: bool,
    /// Extract texture features
    pub texture: bool,
    /// Extract edge features
    pub edges: bool,
    /// Extract moment-based features
    pub moments: bool,
    /// Extract Haralick texture features
    pub haralick: bool,
    /// Extract local binary pattern features
    pub lbp: bool,
    /// Number of bins for histogram features
    pub histogram_bins: usize,
    /// Normalize histograms to sum to 1.0
    pub normalize_histogram: bool,
    /// Co-occurrence matrix distance for texture features
    pub cooccurrence_distance: usize,
    /// Fast calculation mode (may reduce accuracy)
    pub fast_mode: bool,
}

impl Default for ImageFeatureOptions {
    fn default() -> Self {
        Self {
            histogram: true,
            texture: true,
            edges: true,
            moments: true,
            haralick: true,
            lbp: true,
            histogram_bins: 256,
            normalize_histogram: true,
            cooccurrence_distance: 1,
            fast_mode: false,
        }
    }
}
