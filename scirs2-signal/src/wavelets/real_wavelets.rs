// Real-valued wavelets (e.g., Ricker)

use crate::error::{SignalError, SignalResult};

#[allow(unused_imports)]
/// Generate a Ricker (Mexican hat) wavelet
///
/// # Arguments
///
/// * `points` - Number of points in the wavelet
/// * `a` - Width parameter
///
/// # Returns
///
/// * The Ricker wavelet as a vector of length `points`
///
/// # Examples
///
/// ```
/// use scirs2_signal::wavelets::ricker;
///
/// // Generate a Ricker wavelet with 100 points and width parameter 4.0
/// let wavelet = ricker(100, 4.0).unwrap();
/// ```
#[allow(dead_code)]
pub fn ricker(points: usize, a: f64) -> SignalResult<Vec<f64>> {
    if points == 0 {
        return Err(SignalError::ValueError(
            "points must be greater than 0".to_string(),
        ));
    }

    if a <= 0.0 {
        return Err(SignalError::ValueError(
            "width parameter 'a' must be positive".to_string(),
        ));
    }

    // Calculate amplitude factor
    let amplitude = 2.0 / (std::f64::consts::PI.powf(0.25) * (3.0 * a).sqrt());
    let wsq = a * a;

    // Generate position vector
    let mid_point = (points - 1) as f64 / 2.0;
    let mut wavelet = Vec::with_capacity(points);

    for i in 0..points {
        let x = i as f64 - mid_point;
        let xsq = x * x;
        let mod_term = 1.0 - xsq / wsq;
        let gauss = (-xsq / (2.0 * wsq)).exp();
        let value = amplitude * mod_term * gauss;
        wavelet.push(value);
    }

    Ok(wavelet)
}
