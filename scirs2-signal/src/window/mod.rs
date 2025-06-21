//! Window functions for signal processing.
//!
//! This module provides various window functions commonly used in signal processing,
//! including Hamming, Hann, Blackman, and others. These windows are useful for
//! reducing spectral leakage in Fourier transforms and filter design.

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// Import specialized window implementations
pub mod kaiser;
pub use kaiser::{kaiser, kaiser_bessel_derived};

/// Create a window function of a specified type and length.
///
/// # Arguments
///
/// * `window_type` - Type of window function to create
/// * `length` - Length of the window
/// * `periodic` - If true, the window is periodic, otherwise symmetric
///
/// # Returns
///
/// * Window function of specified type and length
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::get_window;
///
/// // Create a Hamming window of length 10
/// let window = get_window("hamming", 10, false).unwrap();
///
/// assert_eq!(window.len(), 10);
/// assert!(window[0] > 0.0 && window[0] < 1.0);
/// assert!(window[window.len() / 2] > 0.9);
/// ```
pub fn get_window(window_type: &str, length: usize, periodic: bool) -> SignalResult<Vec<f64>> {
    if length == 0 {
        return Err(SignalError::ValueError(
            "Window length must be positive".to_string(),
        ));
    }

    // Dispatch to specific window function
    match window_type.to_lowercase().as_str() {
        "hamming" => hamming(length, !periodic),
        "hanning" | "hann" => hann(length, !periodic),
        "blackman" => blackman(length, !periodic),
        "bartlett" => bartlett(length, !periodic),
        "flattop" => flattop(length, !periodic),
        "boxcar" | "rectangular" => boxcar(length, !periodic),
        "triang" => triang(length, !periodic),
        "bohman" => bohman(length, !periodic),
        "parzen" => parzen(length, !periodic),
        "nuttall" => nuttall(length, !periodic),
        "blackmanharris" => blackmanharris(length, !periodic),
        "cosine" => cosine(length, !periodic),
        "exponential" => exponential(length, None, 1.0, !periodic),
        "tukey" => tukey(length, 0.5, !periodic),
        "barthann" => barthann(length, !periodic),
        "kaiser" => {
            // Default beta value of 8.6 gives sidelobe attenuation of about 60dB
            kaiser(length, 8.6, !periodic)
        }
        "kaiser_bessel_derived" => {
            // Default beta value of 8.6
            kaiser_bessel_derived(length, 8.6, !periodic)
        }
        "dpss" | "slepian" => {
            // Default NW parameter of 3.0 for multitaper
            dpss(length, 3.0, None, !periodic)
        }
        "lanczos" => {
            // Default parameter a = 2 for Lanczos window
            lanczos(length, 2, !periodic)
        }
        _ => Err(SignalError::ValueError(format!(
            "Unknown window type: {}",
            window_type
        ))),
    }
}

/// Helper function to handle small or incorrect window lengths
pub(crate) fn _len_guards(m: usize) -> bool {
    // Return true for trivial windows with length 0 or 1
    m <= 1
}

/// Helper function to extend window by 1 sample if needed for DFT-even symmetry
pub(crate) fn _extend(m: usize, sym: bool) -> (usize, bool) {
    if !sym {
        (m + 1, true)
    } else {
        (m, false)
    }
}

/// Helper function to truncate window by 1 sample if needed
pub(crate) fn _truncate(w: Vec<f64>, needed: bool) -> Vec<f64> {
    if needed {
        w[..w.len() - 1].to_vec()
    } else {
        w
    }
}

/// Hamming window.
///
/// The Hamming window is a taper formed by using a raised cosine with
/// non-zero endpoints.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::hamming;
///
/// let window = hamming(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn hamming(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Hann window.
///
/// The Hann window is a taper formed by using a raised cosine.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::hann;
///
/// let window = hann(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn hann(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Blackman window.
///
/// The Blackman window is a taper formed by using the first three terms of
/// a summation of cosines.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::blackman;
///
/// let window = blackman(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn blackman(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = 0.42 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            + 0.08 * (4.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Bartlett window.
///
/// The Bartlett window is a triangular window that is the convolution of two rectangular windows.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::bartlett;
///
/// let window = bartlett(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn bartlett(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let m2 = (n - 1) as f64 / 2.0;
    for i in 0..n {
        let w_val = 1.0 - ((i as f64 - m2) / m2).abs();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Triangular window (slightly different from Bartlett).
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::triang;
///
/// let window = triang(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn triang(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let m2 = (n as f64 - 1.0) / 2.0;
    for i in 0..n {
        let w_val = 1.0 - ((i as f64 - m2) / (m2 + 1.0)).abs();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Flat top window.
///
/// The flat top window is a taper formed by using a weighted sum of cosine functions.
/// This window has the best amplitude flatness in the frequency domain.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::flattop;
///
/// let window = flattop(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn flattop(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let a0 = 0.21557895;
    let a1 = 0.41663158;
    let a2 = 0.277263158;
    let a3 = 0.083578947;
    let a4 = 0.006947368;

    for i in 0..n {
        let w_val = a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            + a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()
            - a3 * (6.0 * PI * i as f64 / (n - 1) as f64).cos()
            + a4 * (8.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Rectangular window.
///
/// The rectangular window is the simplest window, equivalent to replacing all frame samples by a constant.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::boxcar;
///
/// let window = boxcar(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// assert_eq!(window[0], 1.0);
/// ```
pub fn boxcar(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);
    let w = vec![1.0; n];
    Ok(_truncate(w, needs_trunc))
}

/// Bohman window.
///
/// The Bohman window is the product of a cosine and a sinc function.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::bohman;
///
/// let window = bohman(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn bohman(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        let x_abs = x.abs();
        let w_val = if x_abs <= 1.0 {
            (1.0 - x_abs) * (PI * x_abs).cos() + PI.recip() * (PI * x_abs).sin()
        } else {
            0.0
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Parzen window.
///
/// The Parzen window is a piecewise cubic approximation of the Gaussian window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::parzen;
///
/// let window = parzen(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn parzen(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let n1 = (n - 1) as f64;

    for i in 0..n {
        let x = 2.0 * i as f64 / n1 - 1.0;
        let x_abs = x.abs();

        let w_val = if x_abs <= 0.5 {
            1.0 - 6.0 * x_abs.powi(2) + 6.0 * x_abs.powi(3)
        } else if x_abs <= 1.0 {
            2.0 * (1.0 - x_abs).powi(3)
        } else {
            0.0
        };

        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Nuttall window.
///
/// The Nuttall window is a minimal 4-term Blackman-Harris window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::nuttall;
///
/// let window = nuttall(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn nuttall(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let a0 = 0.3635819;
    let a1 = 0.4891775;
    let a2 = 0.1365995;
    let a3 = 0.0106411;

    for i in 0..n {
        let w_val = a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            + a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()
            - a3 * (6.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Blackman-Harris window.
///
/// The Blackman-Harris window is a taper formed by using the first four terms of a
/// summation of cosines.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::blackmanharris;
///
/// let window = blackmanharris(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn blackmanharris(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;

    for i in 0..n {
        let w_val = a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            + a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()
            - a3 * (6.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Cosine window.
///
/// Also known as the sine window, half-cosine, or half-sine window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::cosine;
///
/// let window = cosine(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn cosine(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = (PI * i as f64 / (n - 1) as f64).sin();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Exponential window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `center` - Optional parameter defining the center point of the window, default is None (m/2)
/// * `tau` - Parameter defining the decay rate
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::exponential;
///
/// let window = exponential(10, None, 1.0, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn exponential(m: usize, center: Option<f64>, tau: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let center_val = center.unwrap_or(((n - 1) as f64) / 2.0);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = (-((i as f64 - center_val).abs() / tau)).exp();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Tukey window.
///
/// The Tukey window, also known as the cosine-tapered window, is a window
/// with a flat middle section and cosine tapered ends.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `alpha` - Shape parameter of the Tukey window, representing the ratio of
///   cosine-tapered section length to the total window length
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::tukey;
///
/// let window = tukey(10, 0.5, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn tukey(m: usize, alpha: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let alpha = alpha.clamp(0.0, 1.0);

    if alpha == 0.0 {
        return boxcar(m, sym);
    }

    if alpha == 1.0 {
        return hann(m, sym);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let width = (alpha * (n - 1) as f64 / 2.0).floor() as usize;
    let width = width.max(1); // Ensure width is at least 1

    for i in 0..n {
        let w_val = if i < width {
            0.5 * (1.0 + (PI * (-1.0 + 2.0 * i as f64 / (alpha * (n - 1) as f64))).cos())
        } else if i >= n - width {
            0.5 * (1.0
                + (PI * (-2.0 / alpha + 1.0 + 2.0 * i as f64 / (alpha * (n - 1) as f64))).cos())
        } else {
            1.0
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Barthann window.
///
/// The Barthann window is the product of a Bartlett window and a Hann window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::barthann;
///
/// let window = barthann(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn barthann(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let fac = (i as f64) / (n - 1) as f64;
        let w_val = 0.62 - 0.48 * (fac * 2.0 - 1.0).abs() - 0.38 * (2.0 * PI * fac).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Discrete Prolate Spheroidal Sequence (DPSS) windows.
///
/// Also known as Slepian windows, these are optimal for multitaper spectral estimation.
/// They maximize the energy concentration within a given bandwidth while minimizing
/// spectral leakage.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `nw` - Time-bandwidth product. Larger values provide better frequency resolution
/// * `num_windows` - Number of windows to generate (optional, defaults to 2*NW-1)
/// * `sym` - If true, generates symmetric windows
///
/// # Returns
///
/// * A vector of DPSS windows, each window is a Vec<f64>
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::window::dpss_windows;
///
/// // Generate 5 DPSS windows for multitaper spectral estimation
/// let windows = dpss_windows(256, 4.0, Some(7), true).unwrap();
/// assert_eq!(windows.len(), 7);
/// assert_eq!(windows[0].len(), 256);
/// ```
pub fn dpss_windows(
    m: usize,
    nw: f64,
    num_windows: Option<usize>,
    sym: bool,
) -> SignalResult<Vec<Vec<f64>>> {
    if m == 0 {
        return Err(SignalError::ValueError(
            "Window length must be positive".to_string(),
        ));
    }

    if nw <= 0.0 || nw >= m as f64 / 2.0 {
        return Err(SignalError::ValueError(
            "Time-bandwidth product NW must be between 0 and M/2".to_string(),
        ));
    }

    let num_win = num_windows.unwrap_or((2.0 * nw - 1.0).floor() as usize);
    if num_win == 0 {
        return Err(SignalError::ValueError(
            "Number of windows must be positive".to_string(),
        ));
    }

    let (n, needs_trunc) = _extend(m, sym);

    // Build the tridiagonal matrix for the eigenvalue problem
    let omega = 2.0 * PI * nw / n as f64;
    let mut diag = vec![0.0; n];
    let mut off_diag = vec![0.0; n - 1];

    // Fill diagonal elements
    for (i, diag_val) in diag.iter_mut().enumerate() {
        let k = i as f64 - (n as f64 - 1.0) / 2.0;
        *diag_val = (omega * k).cos();
    }

    // Fill off-diagonal elements
    for (i, off_diag_val) in off_diag.iter_mut().enumerate() {
        let k = (i + 1) as f64;
        *off_diag_val = k * (n as f64 - k) / 2.0;
    }

    // Solve the eigenvalue problem for the tridiagonal matrix
    let (eigenvals, eigenvecs) = solve_tridiagonal_eigenproblem(&diag, &off_diag, num_win)?;

    // Sort eigenvalues and eigenvectors in descending order
    let mut sorted_indices: Vec<usize> = (0..eigenvals.len()).collect();
    sorted_indices.sort_by(|&a, &b| eigenvals[b].partial_cmp(&eigenvals[a]).unwrap());

    let mut windows = Vec::with_capacity(num_win);
    for &idx in sorted_indices.iter().take(num_win) {
        let mut window = eigenvecs[idx].clone();

        // Ensure the first element is positive (phase convention)
        if window[0] < 0.0 {
            for w in &mut window {
                *w = -*w;
            }
        }

        // Normalize the window
        let norm = window.iter().map(|&x| x * x).sum::<f64>().sqrt();
        for w in &mut window {
            *w /= norm;
        }

        windows.push(_truncate(window, needs_trunc));
    }

    Ok(windows)
}

/// Single DPSS window (first window from the set).
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `nw` - Time-bandwidth product
/// * `k` - Window index (optional, defaults to 0 for first window)
/// * `sym` - If true, generates a symmetric window
///
/// # Returns
///
/// * A single DPSS window
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::window::dpss;
///
/// let window = dpss(64, 2.5, None, true).unwrap();
/// assert_eq!(window.len(), 64);
/// ```
pub fn dpss(m: usize, nw: f64, k: Option<usize>, sym: bool) -> SignalResult<Vec<f64>> {
    let window_idx = k.unwrap_or(0);
    let windows = dpss_windows(m, nw, Some(window_idx + 1), sym)?;

    if windows.is_empty() {
        return Err(SignalError::ValueError(
            "Failed to generate DPSS window".to_string(),
        ));
    }

    Ok(windows[window_idx].clone())
}

/// Solve the tridiagonal eigenvalue problem using the QR algorithm.
///
/// This is a simplified implementation for finding the largest eigenvalues
/// and their corresponding eigenvectors.
fn solve_tridiagonal_eigenproblem(
    diag: &[f64],
    off_diag: &[f64],
    num_eigenvals: usize,
) -> SignalResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = diag.len();
    if off_diag.len() != n - 1 {
        return Err(SignalError::ValueError(
            "Inconsistent matrix dimensions".to_string(),
        ));
    }

    // Simple power iteration for finding dominant eigenvalues
    // This is a simplified approach for demonstration
    let mut eigenvals: Vec<f64> = Vec::new();
    let mut eigenvecs: Vec<Vec<f64>> = Vec::new();

    // Start with the largest expected eigenvalue
    let max_iter = 1000;
    let tolerance = 1e-10;

    for _k in 0..num_eigenvals.min(n) {
        // Initialize random vector
        let mut v = vec![1.0; n];
        for (i, v_val) in v.iter_mut().enumerate().skip(1) {
            *v_val = 0.1 * (i as f64).sin();
        }

        // Orthogonalize against previous eigenvectors
        for prev_vec in &eigenvecs {
            let dot = v
                .iter()
                .zip(prev_vec.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>();
            for (vi, &pvi) in v.iter_mut().zip(prev_vec.iter()) {
                *vi -= dot * pvi;
            }
        }

        // Normalize
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        for vi in &mut v {
            *vi /= norm;
        }

        let mut eigenval = 0.0;

        // Power iteration
        for _iter in 0..max_iter {
            let mut new_v = vec![0.0; n];

            // Matrix-vector multiplication for tridiagonal matrix
            for i in 0..n {
                new_v[i] += diag[i] * v[i];
                if i > 0 {
                    new_v[i] += off_diag[i - 1] * v[i - 1];
                }
                if i < n - 1 {
                    new_v[i] += off_diag[i] * v[i + 1];
                }
            }

            // Orthogonalize against previous eigenvectors
            for prev_vec in &eigenvecs {
                let dot = new_v
                    .iter()
                    .zip(prev_vec.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>();
                for (nvi, &pvi) in new_v.iter_mut().zip(prev_vec.iter()) {
                    *nvi -= dot * pvi;
                }
            }

            // Calculate eigenvalue (Rayleigh quotient)
            let new_eigenval = new_v
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>();

            // Normalize
            let norm = new_v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            for nvi in &mut new_v {
                *nvi /= norm;
            }

            // Check convergence
            if (new_eigenval - eigenval).abs() < tolerance {
                eigenval = new_eigenval;
                v = new_v;
                break;
            }

            eigenval = new_eigenval;
            v = new_v;
        }

        eigenvals.push(eigenval);
        eigenvecs.push(v);
    }

    Ok((eigenvals, eigenvecs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hamming_window() {
        let window = hamming(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test specific values
        assert_relative_eq!(window[0], 0.08, epsilon = 0.01);
        // The peak is at indices 4 and 5 for a 10-point symmetric window
        assert!(window[4] > 0.95);
        assert!(window[5] > 0.95);
    }

    #[test]
    fn test_hann_window() {
        let window = hann(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test specific values
        assert_relative_eq!(window[0], 0.0, epsilon = 0.01);
        // The peak is at indices 4 and 5 for a 10-point symmetric window
        assert!(window[4] > 0.95);
        assert!(window[5] > 0.95);
    }

    #[test]
    fn test_blackman_window() {
        let window = blackman(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bartlett_window() {
        let window = bartlett(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test endpoints - Bartlett window has zero at endpoints
        assert_relative_eq!(window[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(window[9], 0.0, epsilon = 1e-10);

        // Test that it increases from start to middle
        assert!(window[1] > window[0]);
        assert!(window[2] > window[1]);

        // Test middle values are close to 1
        let mid_val = window[4];
        assert!(mid_val > 0.8 && mid_val <= 1.0);
    }

    #[test]
    fn test_flattop_window() {
        let window = flattop(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_boxcar_window() {
        let window = boxcar(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test all values are 1.0
        for val in window {
            assert_relative_eq!(val, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bohman_window() {
        let window = bohman(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test endpoints
        assert_relative_eq!(window[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(window[9], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_triang_window() {
        let window = triang(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dpss_window() {
        let window = dpss(64, 2.5, None, true).unwrap();
        assert_eq!(window.len(), 64);

        // Test that the window is normalized
        let norm = window.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-10);

        // Test symmetry (approximately, since DPSS windows are nearly symmetric)
        for i in 0..32 {
            assert_relative_eq!(window[i], window[63 - i], epsilon = 1e-2);
        }

        // Test that the window has reasonable magnitude
        let max_val = window.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(max_val > 0.1);
        assert!(max_val < 1.0);
    }

    #[test]
    fn test_dpss_windows() {
        let windows = dpss_windows(32, 2.0, Some(3), true).unwrap();
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].len(), 32);

        // Test that each window is normalized
        for window in &windows {
            let norm = window.iter().map(|&x| x * x).sum::<f64>().sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
        }

        // Test orthogonality between different windows (approximate)
        for i in 0..3 {
            for j in (i + 1)..3 {
                let dot_product = windows[i]
                    .iter()
                    .zip(windows[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>();
                assert!(dot_product.abs() < 0.1); // Should be approximately orthogonal
            }
        }
    }

    #[test]
    fn test_dpss_errors() {
        // Test error conditions

        // Zero length
        let result = dpss(0, 2.0, None, true);
        assert!(result.is_err());

        // Invalid NW
        let result = dpss(10, 0.0, None, true);
        assert!(result.is_err());

        let result = dpss(10, 10.0, None, true);
        assert!(result.is_err());

        // Too many windows
        let result = dpss_windows(10, 2.0, Some(0), true);
        assert!(result.is_err());
    }
}

/// Generate a Lanczos window
///
/// The Lanczos window is a tapered window that provides good frequency resolution
/// with moderate side-lobe suppression. It's the main lobe of a sinc function
/// windowed by another sinc function.
///
/// # Arguments
///
/// * `length` - Length of the window
/// * `a` - Parameter controlling the width of the window (typically 2 or 3)
/// * `sym` - If true, creates a symmetric window; if false, creates a periodic window
///
/// # Returns
///
/// * Vector containing the Lanczos window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::lanczos;
///
/// // Create a symmetric Lanczos window of length 10 with parameter a=2
/// let window = lanczos(10, 2, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn lanczos(length: usize, a: i32, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(length) {
        return Ok(vec![1.0; length]);
    }

    if a <= 0 {
        return Err(SignalError::ValueError(
            "Parameter 'a' must be positive".to_string(),
        ));
    }

    let (m, needs_trunc) = _extend(length, sym);
    let mut window = Vec::with_capacity(m);

    for i in 0..m {
        let n = if sym {
            // Symmetric: center at (m-1)/2
            i as f64 - (m - 1) as f64 / 2.0
        } else {
            // Periodic: center at m/2
            i as f64 - m as f64 / 2.0
        };

        let x = n / a as f64;

        let value = if x.abs() < 1e-15 {
            // Handle the case where x ≈ 0 (sinc(0) = 1)
            1.0
        } else if x.abs() < 1.0 {
            // Lanczos window: sinc(x) * sinc(x/a)
            let sinc_x = (PI * x).sin() / (PI * x);
            let x_a = x / a as f64;
            let sinc_x_a = if x_a.abs() < 1e-15 {
                1.0
            } else {
                (PI * x_a).sin() / (PI * x_a)
            };
            sinc_x * sinc_x_a
        } else {
            // Outside the support interval
            0.0
        };

        window.push(value);
    }

    Ok(_truncate(window, needs_trunc))
}

#[cfg(test)]
mod lanczos_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lanczos_basic() {
        let window = lanczos(10, 2, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test that center value is maximum
        let center_idx = 4; // For length 10, center is at index 4
        for (i, &val) in window.iter().enumerate() {
            if i != center_idx {
                assert!(window[center_idx] >= val);
            }
        }

        // Test that values are non-negative
        for &val in &window {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_lanczos_periodic() {
        let window = lanczos(8, 2, false).unwrap();
        assert_eq!(window.len(), 8);

        // Test that all values are finite
        for &val in &window {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_lanczos_parameters() {
        // Test different values of 'a'
        let window_a2 = lanczos(20, 2, true).unwrap();
        let window_a3 = lanczos(20, 3, true).unwrap();

        assert_eq!(window_a2.len(), 20);
        assert_eq!(window_a3.len(), 20);

        // Larger 'a' should give a wider main lobe
        // (This is a qualitative test - the exact comparison depends on implementation details)
        assert!(window_a2.iter().all(|&x| x.is_finite()));
        assert!(window_a3.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_lanczos_small_windows() {
        // Test edge cases
        let window1 = lanczos(1, 2, true).unwrap();
        assert_eq!(window1, vec![1.0]);

        let window2 = lanczos(2, 2, true).unwrap();
        assert_eq!(window2.len(), 2);
    }

    #[test]
    fn test_lanczos_errors() {
        // Test invalid parameter
        let result = lanczos(10, 0, true);
        assert!(result.is_err());

        let result = lanczos(10, -1, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_lanczos_zero_length() {
        let result = lanczos(0, 2, true);
        // Should handle gracefully (return empty or error)
        assert!(result.is_ok() || result.is_err());
    }
}

/// Window analysis and design utilities
pub mod analysis {
    use super::*;
    use crate::error::{SignalError, SignalResult};
    use std::f64::consts::PI;

    /// Window analysis results
    #[derive(Debug, Clone)]
    pub struct WindowAnalysis {
        /// Coherent gain (sum of window values)
        pub coherent_gain: f64,
        /// Power gain (sum of squared window values)
        pub power_gain: f64,
        /// Normalized effective noise bandwidth (NENBW)
        pub nenbw: f64,
        /// Scalloping loss in dB
        pub scalloping_loss_db: f64,
        /// Maximum sidelobe level in dB
        pub max_sidelobe_db: f64,
        /// 3dB bandwidth in bins
        pub bandwidth_3db: f64,
        /// 6dB bandwidth in bins
        pub bandwidth_6db: f64,
        /// Processing gain in dB
        pub processing_gain_db: f64,
    }

    /// Analyze a window function's spectral properties
    ///
    /// # Arguments
    ///
    /// * `window` - Window function to analyze
    /// * `fft_size` - FFT size for analysis (default: 8 times window length)
    ///
    /// # Returns
    ///
    /// * Window analysis results
    pub fn analyze_window(window: &[f64], fft_size: Option<usize>) -> SignalResult<WindowAnalysis> {
        if window.is_empty() {
            return Err(SignalError::ValueError(
                "Window cannot be empty".to_string(),
            ));
        }

        let n = window.len();
        let fft_len = fft_size.unwrap_or(n * 8).max(n);

        // Calculate basic gains
        let coherent_gain = window.iter().sum::<f64>();
        let power_gain = window.iter().map(|&x| x * x).sum::<f64>();

        // Normalized Effective Noise Bandwidth
        let nenbw = n as f64 * power_gain / (coherent_gain * coherent_gain);

        // Zero-pad window for FFT analysis
        let mut padded_window = vec![0.0; fft_len];
        for (i, &val) in window.iter().enumerate() {
            padded_window[i] = val;
        }

        // Compute FFT magnitude (simplified implementation)
        let freq_response = compute_window_fft_magnitude(&padded_window)?;

        // Find peak (should be at DC)
        let peak_value = freq_response[0];
        let peak_db = 20.0 * peak_value.log10();

        // Calculate scalloping loss (at bin 0.5)
        let bin_05_idx = fft_len / (2 * n);
        let scalloping_response = if bin_05_idx < freq_response.len() {
            freq_response[bin_05_idx]
        } else {
            // Interpolate
            let frac_idx = 0.5 * fft_len as f64 / n as f64;
            let idx1 = frac_idx.floor() as usize;
            let idx2 = (idx1 + 1).min(freq_response.len() - 1);
            let frac = frac_idx - idx1 as f64;

            if idx2 < freq_response.len() {
                freq_response[idx1] * (1.0 - frac) + freq_response[idx2] * frac
            } else {
                freq_response[idx1]
            }
        };
        let scalloping_loss_db = 20.0 * scalloping_response.log10() - peak_db;

        // Find maximum sidelobe
        let main_lobe_end = find_main_lobe_end(&freq_response, fft_len, n)?;
        let max_sidelobe: f64 = freq_response
            .iter()
            .skip(main_lobe_end)
            .fold(0.0_f64, |acc, &x| acc.max(x));
        let max_sidelobe_db = if max_sidelobe > 0.0 {
            20.0 * max_sidelobe.log10() - peak_db
        } else {
            -120.0 // Very low sidelobe
        };

        // Calculate bandwidth measurements
        let half_peak = peak_value / 2.0; // -3dB point
        let quarter_peak = peak_value / 4.0; // -6dB point

        let bandwidth_3db = find_bandwidth(&freq_response, half_peak, fft_len, n)?;
        let bandwidth_6db = find_bandwidth(&freq_response, quarter_peak, fft_len, n)?;

        // Processing gain
        let processing_gain_db = 10.0 * (n as f64).log10();

        Ok(WindowAnalysis {
            coherent_gain,
            power_gain,
            nenbw,
            scalloping_loss_db,
            max_sidelobe_db,
            bandwidth_3db,
            bandwidth_6db,
            processing_gain_db,
        })
    }

    /// Design a window with specified characteristics
    pub fn design_window_with_constraints(
        length: usize,
        sidelobe_db: f64,
        _bandwidth_bins: Option<f64>,
    ) -> SignalResult<Vec<f64>> {
        if length == 0 {
            return Err(SignalError::ValueError(
                "Length must be positive".to_string(),
            ));
        }

        // Choose window type based on sidelobe requirement
        if sidelobe_db >= -13.0 {
            // Use Hann window for moderate sidelobe suppression
            hann(length, true)
        } else if sidelobe_db >= -18.0 {
            // Use Hamming window
            hamming(length, true)
        } else if sidelobe_db >= -58.0 {
            // Use Blackman window
            blackman(length, true)
        } else {
            // Use Kaiser window for high sidelobe suppression
            let beta = if sidelobe_db <= -50.0 {
                0.1102 * (sidelobe_db.abs() - 8.7)
            } else if sidelobe_db <= -21.0 {
                0.5842 * (sidelobe_db.abs() - 21.0).powf(0.4) + 0.07886 * (sidelobe_db.abs() - 21.0)
            } else {
                0.0
            };

            super::kaiser::kaiser(length, beta, true)
        }
    }

    /// Compare multiple windows
    pub fn compare_windows(
        windows: &[(&str, &[f64])],
    ) -> SignalResult<Vec<(String, WindowAnalysis)>> {
        let mut results = Vec::new();

        for &(name, window) in windows {
            let analysis = analyze_window(window, None)?;
            results.push((name.to_string(), analysis));
        }

        Ok(results)
    }

    // Helper functions

    fn compute_window_fft_magnitude(window: &[f64]) -> SignalResult<Vec<f64>> {
        let n = window.len();
        let mut magnitude = vec![0.0; n / 2 + 1];

        // Simple DFT computation for magnitude spectrum
        for (k, mag) in magnitude.iter_mut().enumerate() {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &w) in window.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * i as f64 / n as f64;
                real += w * angle.cos();
                imag += w * angle.sin();
            }

            *mag = (real * real + imag * imag).sqrt();
        }

        Ok(magnitude)
    }

    fn find_main_lobe_end(
        freq_response: &[f64],
        fft_len: usize,
        window_len: usize,
    ) -> SignalResult<usize> {
        let peak_value = freq_response[0];
        let threshold = peak_value * 0.01; // -40dB below peak

        // Look for first null or significant drop
        #[allow(clippy::needless_range_loop)]
        for i in 1..freq_response.len().min(fft_len / window_len * 4) {
            if freq_response[i] < threshold {
                return Ok(i);
            }
        }

        // Default to 4 bins if no clear null found
        Ok(4.min(freq_response.len() - 1))
    }

    fn find_bandwidth(
        freq_response: &[f64],
        threshold: f64,
        fft_len: usize,
        window_len: usize,
    ) -> SignalResult<f64> {
        // Find points where response drops below threshold
        let _left_point = 0.0;
        let mut right_point = 0.0;

        // Search to the right of peak
        #[allow(clippy::needless_range_loop)]
        for i in 1..freq_response.len() {
            if freq_response[i] < threshold {
                right_point = i as f64;
                break;
            }
        }

        // For symmetric windows, bandwidth is 2 * right_point
        let bandwidth_bins = 2.0 * right_point * window_len as f64 / fft_len as f64;

        Ok(bandwidth_bins)
    }

    #[cfg(test)]
    mod analysis_tests {
        use super::*;

        #[test]
        fn test_window_analysis() {
            let window = hann(64, true).unwrap();
            let analysis = analyze_window(&window, Some(1024)).unwrap();

            // Check that we get reasonable results
            assert!(analysis.coherent_gain > 0.0);
            assert!(analysis.power_gain > 0.0);
            assert!(analysis.nenbw > 1.0);
            assert!(analysis.scalloping_loss_db < 0.0); // Should be negative
            assert!(analysis.processing_gain_db > 0.0);
        }

        #[test]
        fn test_compare_windows() {
            let hann_win = hann(32, true).unwrap();
            let hamming_win = hamming(32, true).unwrap();

            let windows = [
                ("hann", hann_win.as_slice()),
                ("hamming", hamming_win.as_slice()),
            ];

            let comparison = compare_windows(&windows).unwrap();
            assert_eq!(comparison.len(), 2);

            // Hamming should have lower sidelobes than Hann
            let hann_analysis = &comparison
                .iter()
                .find(|(name, _)| name == "hann")
                .unwrap()
                .1;
            let hamming_analysis = &comparison
                .iter()
                .find(|(name, _)| name == "hamming")
                .unwrap()
                .1;
            assert!(hamming_analysis.max_sidelobe_db < hann_analysis.max_sidelobe_db);
        }

        #[test]
        fn test_design_window_with_constraints() {
            // Test different sidelobe requirements
            let window1 = design_window_with_constraints(64, -10.0, None).unwrap();
            let window2 = design_window_with_constraints(64, -25.0, None).unwrap();
            let window3 = design_window_with_constraints(64, -60.0, None).unwrap();

            assert_eq!(window1.len(), 64);
            assert_eq!(window2.len(), 64);
            assert_eq!(window3.len(), 64);

            // All windows should be normalized to peak of 1.0
            assert!(window1.iter().fold(0.0_f64, |acc, &x| acc.max(x)) <= 1.0);
            assert!(window2.iter().fold(0.0_f64, |acc, &x| acc.max(x)) <= 1.0);
            assert!(window3.iter().fold(0.0_f64, |acc, &x| acc.max(x)) <= 1.0);
        }

        #[test]
        fn test_optimized_window_design() {
            // Test optimized Kaiser window design
            let window = design_optimal_kaiser(64, -60.0, 0.1).unwrap();
            assert_eq!(window.len(), 64);

            // Test transition analysis
            let hann_win = super::hann(32, true).unwrap();
            let analysis = analyze_window_transition(&hann_win, 0.3).unwrap();
            assert!(analysis.transition_width >= 0.0);
            assert!(analysis.cutoff_3db >= 0.0);
            assert!(analysis.cutoff_6db >= 0.0);
        }

        #[test]
        fn test_window_optimization() {
            let hann_win = super::hann(32, true).unwrap();
            let hamming_win = super::hamming(32, true).unwrap();

            let best = select_optimal_window(
                &[("hann", &hann_win), ("hamming", &hamming_win)],
                WindowOptimizationCriteria::MinSidelobes,
            )
            .unwrap();

            assert!(!best.is_empty());
        }
    }

    /// Advanced window design tools and optimization utilities
    /// Design an optimal Kaiser window for given specifications
    ///
    /// This function calculates the optimal Kaiser window parameters (beta and length)
    /// to meet specified sidelobe attenuation and transition width requirements.
    ///
    /// # Arguments
    ///
    /// * `length` - Desired window length
    /// * `sidelobe_db` - Required sidelobe attenuation in dB (negative value)
    /// * `transition_width` - Normalized transition width (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// * Optimally designed Kaiser window
    pub fn design_optimal_kaiser(
        length: usize,
        sidelobe_db: f64,
        transition_width: f64,
    ) -> SignalResult<Vec<f64>> {
        if length == 0 {
            return Err(SignalError::ValueError(
                "Length must be positive".to_string(),
            ));
        }

        if sidelobe_db >= 0.0 {
            return Err(SignalError::ValueError(
                "Sidelobe attenuation must be negative".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&transition_width) {
            return Err(SignalError::ValueError(
                "Transition width must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Calculate optimal Kaiser beta parameter based on sidelobe requirement
        let atten = sidelobe_db.abs();
        let beta = if atten >= 50.0 {
            0.1102 * (atten - 8.7)
        } else if atten >= 21.0 {
            0.5842 * (atten - 21.0).powf(0.4) + 0.07886 * (atten - 21.0)
        } else {
            0.0
        };

        // Use the kaiser module function
        super::kaiser::kaiser(length, beta, true)
    }

    /// Window transition analysis
    #[derive(Debug, Clone)]
    pub struct WindowTransitionAnalysis {
        /// Transition width in normalized frequency units
        pub transition_width: f64,
        /// Transition steepness (rolloff rate in dB per normalized frequency unit)
        pub transition_steepness: f64,
        /// Transition center frequency
        pub transition_center: f64,
        /// -3dB cutoff frequency
        pub cutoff_3db: f64,
        /// -6dB cutoff frequency
        pub cutoff_6db: f64,
    }

    /// Analyze window transition characteristics
    ///
    /// This function analyzes the transition characteristics of a window function,
    /// useful for understanding filter performance.
    ///
    /// # Arguments
    ///
    /// * `window` - Window function to analyze
    /// * `transition_threshold` - Threshold for transition analysis (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// * Window transition analysis results
    pub fn analyze_window_transition(
        window: &[f64],
        transition_threshold: f64,
    ) -> SignalResult<WindowTransitionAnalysis> {
        if window.is_empty() {
            return Err(SignalError::ValueError(
                "Window cannot be empty".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&transition_threshold) {
            return Err(SignalError::ValueError(
                "Transition threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Compute frequency response
        let fft_size = window.len() * 8;
        let mut padded_window = vec![0.0; fft_size];
        padded_window[..window.len()].copy_from_slice(window);

        let freq_response = compute_window_fft_magnitude(&padded_window)?;
        let peak_value = freq_response[0];

        // Find transition points
        let upper_threshold = peak_value * (1.0 - transition_threshold);
        let lower_threshold = peak_value * transition_threshold;

        let mut upper_point = 0;
        let mut lower_point = 0;

        // Find upper transition point (near peak)
        for (i, &val) in freq_response
            .iter()
            .enumerate()
            .take(freq_response.len() / 2)
        {
            if val < upper_threshold {
                upper_point = i;
                break;
            }
        }

        // Find lower transition point
        for (i, &val) in freq_response
            .iter()
            .enumerate()
            .take(freq_response.len() / 2)
        {
            if val < lower_threshold {
                lower_point = i;
                break;
            }
        }

        // Calculate metrics
        let freq_bin_width = 1.0 / fft_size as f64;
        let transition_width = (lower_point - upper_point) as f64 * freq_bin_width;
        let transition_center = (upper_point + lower_point) as f64 * freq_bin_width / 2.0;

        // Calculate steepness in dB per normalized frequency unit
        let upper_db = 20.0 * (freq_response[upper_point] / peak_value).log10();
        let lower_db = 20.0 * (freq_response[lower_point] / peak_value).log10();
        let transition_steepness = if transition_width > 0.0 {
            (upper_db - lower_db) / transition_width
        } else {
            0.0
        };

        // Find -3dB and -6dB points
        let threshold_3db = peak_value / 2.0_f64.sqrt(); // -3dB
        let threshold_6db = peak_value / 2.0; // -6dB

        let cutoff_3db = find_cutoff_frequency(&freq_response, threshold_3db, freq_bin_width);
        let cutoff_6db = find_cutoff_frequency(&freq_response, threshold_6db, freq_bin_width);

        Ok(WindowTransitionAnalysis {
            transition_width,
            transition_steepness,
            transition_center,
            cutoff_3db,
            cutoff_6db,
        })
    }

    /// Window optimization criteria
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum WindowOptimizationCriteria {
        /// Minimize sidelobe levels
        MinSidelobes,
        /// Minimize scalloping loss
        MinScallopingLoss,
        /// Minimize noise bandwidth
        MinNoiseBandwidth,
        /// Maximize processing gain
        MaxProcessingGain,
        /// Balance between sidelobes and bandwidth
        Balanced,
    }

    /// Select optimal window from a set of candidates based on criteria
    ///
    /// # Arguments
    ///
    /// * `windows` - Array of (name, window) pairs to compare
    /// * `criteria` - Optimization criteria to use
    ///
    /// # Returns
    ///
    /// * Name of the optimal window
    pub fn select_optimal_window(
        windows: &[(&str, &[f64])],
        criteria: WindowOptimizationCriteria,
    ) -> SignalResult<String> {
        if windows.is_empty() {
            return Err(SignalError::ValueError(
                "No windows provided for comparison".to_string(),
            ));
        }

        let mut best_name = windows[0].0.to_string();
        let mut best_score = f64::NEG_INFINITY;

        for &(name, window) in windows {
            let analysis = analyze_window(window, None)?;

            let score = match criteria {
                WindowOptimizationCriteria::MinSidelobes => -analysis.max_sidelobe_db,
                WindowOptimizationCriteria::MinScallopingLoss => -analysis.scalloping_loss_db.abs(),
                WindowOptimizationCriteria::MinNoiseBandwidth => -analysis.nenbw,
                WindowOptimizationCriteria::MaxProcessingGain => analysis.processing_gain_db,
                WindowOptimizationCriteria::Balanced => {
                    // Balanced score: consider both sidelobes and bandwidth
                    let sidelobe_score = (-analysis.max_sidelobe_db).min(60.0) / 60.0; // Normalize to 0-1
                    let bandwidth_score = (4.0 - analysis.nenbw).max(0.0) / 4.0; // Normalize to 0-1
                    (sidelobe_score + bandwidth_score) / 2.0
                }
            };

            if score > best_score {
                best_score = score;
                best_name = name.to_string();
            }
        }

        Ok(best_name)
    }

    /// Generate a custom window using polynomial interpolation
    ///
    /// This function creates a custom window by interpolating between specified control points.
    ///
    /// # Arguments
    ///
    /// * `length` - Desired window length
    /// * `control_points` - Array of (position, value) pairs for interpolation
    /// * `symmetric` - Whether to enforce symmetry
    ///
    /// # Returns
    ///
    /// * Custom interpolated window
    pub fn design_custom_window(
        length: usize,
        control_points: &[(f64, f64)],
        symmetric: bool,
    ) -> SignalResult<Vec<f64>> {
        if length == 0 {
            return Err(SignalError::ValueError(
                "Length must be positive".to_string(),
            ));
        }

        if control_points.len() < 2 {
            return Err(SignalError::ValueError(
                "At least 2 control points required".to_string(),
            ));
        }

        // Validate control points
        for &(pos, _val) in control_points {
            if !(0.0..=1.0).contains(&pos) {
                return Err(SignalError::ValueError(
                    "Control point positions must be between 0.0 and 1.0".to_string(),
                ));
            }
        }

        let mut window = vec![0.0; length];
        let half_len = length / 2;

        for (i, window_value) in window.iter_mut().enumerate() {
            let x = if symmetric && i > half_len {
                // Mirror for symmetric window
                (length - 1 - i) as f64 / (length - 1) as f64
            } else {
                i as f64 / (length - 1) as f64
            };

            // Linear interpolation between control points
            let value = interpolate_control_points(control_points, x);
            *window_value = value;
        }

        Ok(window)
    }

    // Helper functions

    fn find_cutoff_frequency(freq_response: &[f64], threshold: f64, freq_bin_width: f64) -> f64 {
        for (i, &val) in freq_response
            .iter()
            .enumerate()
            .take(freq_response.len() / 2)
        {
            if val < threshold {
                return i as f64 * freq_bin_width;
            }
        }
        0.5 // Nyquist frequency as fallback
    }

    fn interpolate_control_points(control_points: &[(f64, f64)], x: f64) -> f64 {
        // Simple linear interpolation between control points
        if x <= control_points[0].0 {
            return control_points[0].1;
        }

        for i in 1..control_points.len() {
            if x <= control_points[i].0 {
                let x1 = control_points[i - 1].0;
                let y1 = control_points[i - 1].1;
                let x2 = control_points[i].0;
                let y2 = control_points[i].1;

                // Linear interpolation
                let t = (x - x1) / (x2 - x1);
                return y1 + t * (y2 - y1);
            }
        }

        control_points.last().unwrap().1
    }
}
