//! Registration quality metrics

use crate::error::Result;
use image::GrayImage;

/// Registration quality metrics
#[derive(Debug, Clone)]
pub struct RegistrationMetrics {
    /// Mean squared error between images
    pub mean_squared_error: f64,
    /// Peak signal-to-noise ratio in dB
    pub peak_signal_to_noise_ratio: f64,
    /// Structural similarity index (SSIM)
    pub structural_similarity_index: f64,
    /// Normalized cross-correlation coefficient
    pub normalized_cross_correlation: f64,
}

/// Compute registration quality metrics
pub fn compute_registration_metrics(
    _reference: &GrayImage,
    _registered: &GrayImage,
) -> Result<RegistrationMetrics> {
    todo!("Registration metrics not yet implemented")
}

/// Compute Mean Squared Error
pub fn compute_mse_metric(_image1: &GrayImage, _image2: &GrayImage) -> Result<f64> {
    todo!("MSE metric not yet implemented")
}

/// Compute Peak Signal-to-Noise Ratio
pub fn compute_psnr_metric(_image1: &GrayImage, _image2: &GrayImage) -> Result<f64> {
    todo!("PSNR metric not yet implemented")
}
