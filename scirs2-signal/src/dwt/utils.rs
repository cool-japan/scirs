//! Utility functions for wavelet transforms
//!
//! This module provides various utilities for working with wavelets,
//! such as calculating filter norms, checking properties, and other
//! common operations used across the wavelet transform modules.

use super::filters::WaveletFilters;
use crate::error::SignalResult;

/// Calculate the squared norm (energy) of a filter
///
/// # Arguments
///
/// * `filter` - The filter coefficients
///
/// # Returns
///
/// The squared norm (sum of squares) of the filter
///
/// # Examples
///
/// ```
/// use scirs2_signal::dwt::utils::filter_energy;
///
/// let filter = vec![0.7071067811865475, 0.7071067811865475];
/// let energy = filter_energy(&filter);
/// assert!((energy - 1.0).abs() < 1e-10);
/// ```
pub fn filter_energy(filter: &[f64]) -> f64 {
    filter.iter().map(|&x| x * x).sum()
}

/// Check if a set of wavelet filters satisfies the perfect reconstruction condition
///
/// For perfect reconstruction, the filters must satisfy several conditions, including:
/// 1. The sum of the lowpass filter coefficients equals sqrt(2)
/// 2. The sum of the highpass filter coefficients equals 0
/// 3. The filters satisfy the orthogonality condition
///
/// # Arguments
///
/// * `filters` - The wavelet filters to check
/// * `tol` - Tolerance for numerical comparisons (default: 1e-10)
///
/// # Returns
///
/// A Result containing a boolean indicating whether the filters satisfy the conditions
///
/// # Examples
///
/// ```
/// use scirs2_signal::dwt::{Wavelet, utils::check_perfect_reconstruction};
///
/// let wavelet = Wavelet::Haar;
/// let filters = wavelet.filters().unwrap();
/// let is_perfect = check_perfect_reconstruction(&filters, Some(1e-10)).unwrap();
/// assert!(is_perfect);
/// ```
pub fn check_perfect_reconstruction(
    filters: &WaveletFilters,
    tol: Option<f64>,
) -> SignalResult<bool> {
    let tolerance = tol.unwrap_or(1e-10);
    
    // Check filter lengths
    if filters.dec_lo.len() != filters.dec_hi.len() ||
       filters.rec_lo.len() != filters.rec_hi.len() ||
       filters.dec_lo.len() != filters.rec_lo.len() {
        return Ok(false);
    }
    
    // Orthogonal wavelets: Sum of lowpass filter should be sqrt(2)
    // This is the DC gain of the filter
    let sum_lo = filters.dec_lo.iter().sum::<f64>();
    if (sum_lo - 2.0_f64.sqrt()).abs() > tolerance {
        return Ok(false);
    }
    
    // Sum of highpass filter should be approximately 0
    // This ensures the highpass filter has zero DC gain
    let sum_hi = filters.dec_hi.iter().sum::<f64>();
    if sum_hi.abs() > tolerance {
        return Ok(false);
    }
    
    // Check filter energies (should be 1 for normalized filters)
    let energy_dec_lo = filter_energy(&filters.dec_lo);
    let energy_dec_hi = filter_energy(&filters.dec_hi);
    
    if (energy_dec_lo - 1.0).abs() > tolerance || (energy_dec_hi - 1.0).abs() > tolerance {
        return Ok(false);
    }
    
    // Check orthogonality condition between lowpass and highpass
    let mut ortho_sum = 0.0;
    for i in 0..filters.dec_lo.len() {
        for j in 0..filters.dec_hi.len() {
            // Only consider overlapping indices
            if i + 2 * j < filters.dec_lo.len() && i + 2 * j >= 0 {
                ortho_sum += filters.dec_lo[i] * filters.dec_hi[j];
            }
        }
    }
    
    if ortho_sum.abs() > tolerance {
        return Ok(false);
    }
    
    // All checks passed
    Ok(true)
}

/// Calculate the center frequency of a wavelet filter
///
/// This is useful for understanding the frequency response of a wavelet
///
/// # Arguments
///
/// * `filter` - The filter coefficients
///
/// # Returns
///
/// The center frequency of the filter (in normalized frequency, 0.0 to 0.5)
///
/// # Examples
///
/// ```
/// use scirs2_signal::dwt::{Wavelet, utils::center_frequency};
///
/// let wavelet = Wavelet::Haar;
/// let filters = wavelet.filters().unwrap();
/// let center_freq = center_frequency(&filters.dec_hi);
/// // Haar wavelet highpass filter has center frequency close to 0.25
/// assert!((center_freq - 0.25).abs() < 0.1);
/// ```
pub fn center_frequency(filter: &[f64]) -> f64 {
    let pi = std::f64::consts::PI;
    let n = filter.len();
    
    // Calculate the first moment of the squared magnitude response
    let mut num = 0.0;
    let mut den = 0.0;
    
    let points = 1024;
    for k in 0..points {
        let omega = pi * k as f64 / points as f64;
        
        // Calculate frequency response at this frequency
        let mut resp_re = 0.0;
        let mut resp_im = 0.0;
        for i in 0..n {
            resp_re += filter[i] * (omega * i as f64).cos();
            resp_im -= filter[i] * (omega * i as f64).sin();
        }
        
        // Squared magnitude response
        let magnitude_squared = resp_re * resp_re + resp_im * resp_im;
        
        // Accumulate weighted by frequency
        num += (k as f64 / points as f64 / 2.0) * magnitude_squared;
        den += magnitude_squared;
    }
    
    // Return normalized center frequency
    if den.abs() < 1e-10 {
        0.0 // Avoid division by zero
    } else {
        num / den
    }
}

/// Calculate the number of vanishing moments of a wavelet filter
///
/// The number of vanishing moments is the number of moments of the wavelet
/// function that are zero. A wavelet with n vanishing moments can represent
/// polynomials of degree n-1 exactly.
///
/// # Arguments
///
/// * `highpass_filter` - The highpass filter coefficients
/// * `tol` - Tolerance for numerical comparisons (default: 1e-10)
///
/// # Returns
///
/// The estimated number of vanishing moments
///
/// # Examples
///
/// ```
/// use scirs2_signal::dwt::{Wavelet, utils::estimate_vanishing_moments};
///
/// let wavelet = Wavelet::DB(4);
/// let filters = wavelet.filters().unwrap();
/// let moments = estimate_vanishing_moments(&filters.dec_hi, None);
/// assert_eq!(moments, 4); // DB4 has 4 vanishing moments
/// ```
pub fn estimate_vanishing_moments(highpass_filter: &[f64], tol: Option<f64>) -> usize {
    let tolerance = tol.unwrap_or(1e-10);
    let mut n_moments = 0;
    
    // Calculate moments until one is non-zero
    for k in 0..highpass_filter.len() {
        let mut moment = 0.0;
        
        // Calculate the k-th moment of the highpass filter
        for (i, &coef) in highpass_filter.iter().enumerate() {
            moment += coef * (i as f64).powi(k as i32);
        }
        
        if moment.abs() > tolerance {
            break;
        }
        
        n_moments += 1;
    }
    
    n_moments
}

/// Calculate the effective filter length (removing leading/trailing zeros)
///
/// # Arguments
///
/// * `filter` - The filter coefficients
/// * `tol` - Tolerance for numerical comparisons (default: 1e-10)
///
/// # Returns
///
/// The effective length of the filter
///
/// # Examples
///
/// ```
/// use scirs2_signal::dwt::utils::effective_filter_length;
///
/// let filter = vec![0.0, 0.0, 0.2, 0.6, 0.2, 0.0];
/// let length = effective_filter_length(&filter, None);
/// assert_eq!(length, 3); // Effective length is 3, not 6
/// ```
pub fn effective_filter_length(filter: &[f64], tol: Option<f64>) -> usize {
    let tolerance = tol.unwrap_or(1e-10);
    let n = filter.len();
    
    // Find first non-zero coefficient
    let mut start = 0;
    while start < n && filter[start].abs() <= tolerance {
        start += 1;
    }
    
    // Find last non-zero coefficient
    let mut end = n;
    while end > start && filter[end - 1].abs() <= tolerance {
        end -= 1;
    }
    
    // Return effective length
    end - start
}