//! Extended window functions and analysis tools
//!
//! This module provides additional window functions and analysis capabilities
//! to complement the basic window module, matching SciPy's comprehensive catalog.

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

use crate::error::{FFTError, FFTResult};
use crate::fft::fft;
use crate::helper::fftfreq;

/// Extended window types not covered in the basic window module
#[derive(Debug, Clone, PartialEq)]
pub enum ExtendedWindow {
    /// Chebyshev window (equiripple)
    Chebyshev { attenuation_db: f64 },
    /// Slepian (DPSS) window
    Slepian { width: f64 },
    /// Lanczos window (sinc window)
    Lanczos,
    /// Planck-taper window
    PlanckTaper { epsilon: f64 },
    /// Dolph-Chebyshev window
    DolphChebyshev { attenuation_db: f64 },
    /// Poisson window
    Poisson { alpha: f64 },
    /// Hann-Poisson window
    HannPoisson { alpha: f64 },
    /// Cauchy window
    Cauchy { alpha: f64 },
    /// Ultraspherical window
    Ultraspherical { mu: f64, x0: f64 },
    /// Taylor window
    Taylor {
        n_sidelobes: usize,
        sidelobe_level_db: f64,
    },
}

/// Generate an extended window function
pub fn get_extended_window(window: ExtendedWindow, n: usize) -> FFTResult<Array1<f64>> {
    let mut w = Array1::zeros(n);

    match window {
        ExtendedWindow::Chebyshev { attenuation_db } => {
            generate_chebyshev_window(&mut w, attenuation_db)?;
        }
        ExtendedWindow::Slepian { width } => {
            generate_slepian_window(&mut w, width)?;
        }
        ExtendedWindow::Lanczos => {
            generate_lanczos_window(&mut w);
        }
        ExtendedWindow::PlanckTaper { epsilon } => {
            generate_planck_taper_window(&mut w, epsilon)?;
        }
        ExtendedWindow::DolphChebyshev { attenuation_db } => {
            generate_dolph_chebyshev_window(&mut w, attenuation_db)?;
        }
        ExtendedWindow::Poisson { alpha } => {
            generate_poisson_window(&mut w, alpha);
        }
        ExtendedWindow::HannPoisson { alpha } => {
            generate_hann_poisson_window(&mut w, alpha);
        }
        ExtendedWindow::Cauchy { alpha } => {
            generate_cauchy_window(&mut w, alpha);
        }
        ExtendedWindow::Ultraspherical { mu, x0 } => {
            generate_ultraspherical_window(&mut w, mu, x0)?;
        }
        ExtendedWindow::Taylor {
            n_sidelobes,
            sidelobe_level_db,
        } => {
            generate_taylor_window(&mut w, n_sidelobes, sidelobe_level_db)?;
        }
    }

    Ok(w)
}

/// Generate Chebyshev window
fn generate_chebyshev_window(w: &mut Array1<f64>, attenuation_db: f64) -> FFTResult<()> {
    let n = w.len();
    if attenuation_db <= 0.0 {
        return Err(FFTError::ValueError(
            "Attenuation must be positive".to_string(),
        ));
    }

    // Simplified Chebyshev window implementation
    let r = 10.0_f64.powf(attenuation_db / 20.0);
    let beta = (r + (r * r - 1.0).sqrt()).ln() / n as f64;

    for i in 0..n {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        w[i] = ((n as f64 - 1.0) * beta * (1.0 - x * x).sqrt().acos()).cosh() / r.cosh();
    }

    Ok(())
}

/// Generate Slepian (DPSS) window
fn generate_slepian_window(w: &mut Array1<f64>, width: f64) -> FFTResult<()> {
    let n = w.len();
    if width <= 0.0 || width >= 0.5 {
        return Err(FFTError::ValueError(
            "Width must be between 0 and 0.5".to_string(),
        ));
    }

    // Simplified DPSS window (this is an approximation)
    for i in 0..n {
        let t = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        w[i] = (PI * width * n as f64 * t).sin() / (PI * t);
        if t == 0.0 {
            w[i] = width * n as f64;
        }
    }

    // Normalize
    let sum: f64 = w.sum();
    w.mapv_inplace(|x| x / sum * n as f64);

    Ok(())
}

/// Generate Lanczos window
fn generate_lanczos_window(w: &mut Array1<f64>) {
    let n = w.len();
    for i in 0..n {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        w[i] = if x == 0.0 {
            1.0
        } else {
            (PI * x).sin() / (PI * x)
        };
    }
}

/// Generate Planck-taper window
fn generate_planck_taper_window(w: &mut Array1<f64>, epsilon: f64) -> FFTResult<()> {
    let n = w.len();
    if epsilon <= 0.0 || epsilon >= 0.5 {
        return Err(FFTError::ValueError(
            "Epsilon must be between 0 and 0.5".to_string(),
        ));
    }

    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        w[i] = if t < epsilon {
            1.0 / ((-epsilon / (t * (epsilon - t))).exp() + 1.0)
        } else if t > 1.0 - epsilon {
            1.0 / ((-epsilon / ((1.0 - t) * (t - 1.0 + epsilon))).exp() + 1.0)
        } else {
            1.0
        };
    }

    Ok(())
}

/// Generate Dolph-Chebyshev window
fn generate_dolph_chebyshev_window(w: &mut Array1<f64>, attenuation_db: f64) -> FFTResult<()> {
    // This is similar to Chebyshev but with different normalization
    generate_chebyshev_window(w, attenuation_db)?;

    // Normalize to unit sum
    let sum: f64 = w.sum();
    let n = w.len() as f64;
    w.mapv_inplace(|x| x / sum * n);

    Ok(())
}

/// Generate Poisson window
fn generate_poisson_window(w: &mut Array1<f64>, alpha: f64) {
    let n = w.len();
    let half_n = n as f64 / 2.0;

    for i in 0..n {
        let t = (i as f64 - half_n).abs() / half_n;
        w[i] = (-alpha * t).exp();
    }
}

/// Generate Hann-Poisson window
fn generate_hann_poisson_window(w: &mut Array1<f64>, alpha: f64) {
    let n = w.len();

    for i in 0..n {
        let hann_part = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
        let poisson_part = (-alpha * (n as f64 / 2.0 - i as f64).abs() / (n as f64 / 2.0)).exp();
        w[i] = hann_part * poisson_part;
    }
}

/// Generate Cauchy window
fn generate_cauchy_window(w: &mut Array1<f64>, alpha: f64) {
    let n = w.len();
    let center = (n - 1) as f64 / 2.0;

    for i in 0..n {
        let t = (i as f64 - center) / center;
        w[i] = 1.0 / (1.0 + (alpha * t).powi(2));
    }
}

/// Generate Ultraspherical window
fn generate_ultraspherical_window(w: &mut Array1<f64>, mu: f64, x0: f64) -> FFTResult<()> {
    let n = w.len();
    if x0 <= 0.0 || x0 >= 1.0 {
        return Err(FFTError::ValueError(
            "x0 must be between 0 and 1".to_string(),
        ));
    }

    // Simplified ultraspherical window
    for i in 0..n {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        if x.abs() < x0 {
            w[i] = (1.0 - (x / x0).powi(2)).powf(mu);
        } else {
            w[i] = 0.0;
        }
    }

    Ok(())
}

/// Generate Taylor window
fn generate_taylor_window(
    w: &mut Array1<f64>,
    n_sidelobes: usize,
    sidelobe_level_db: f64,
) -> FFTResult<()> {
    let n = w.len();
    if n_sidelobes == 0 {
        return Err(FFTError::ValueError(
            "Number of sidelobes must be positive".to_string(),
        ));
    }

    // Simplified Taylor window implementation
    let a = sidelobe_level_db.abs() / 20.0;
    let r = 10.0_f64.powf(a);

    for i in 0..n {
        let mut sum = 0.0;
        for k in 1..=n_sidelobes {
            let x = 2.0 * PI * k as f64 * (i as f64 / (n - 1) as f64 - 0.5);
            sum += (-1.0_f64).powi(k as i32 + 1) * x.cos() / k as f64;
        }
        w[i] = 1.0 + 2.0 * r * sum;
    }

    // Normalize
    let max_val = w.iter().cloned().fold(0.0, f64::max);
    w.mapv_inplace(|x| x / max_val);

    Ok(())
}

/// Analyze window properties
#[derive(Debug, Clone)]
pub struct WindowProperties {
    /// Main lobe width (in bins)
    pub main_lobe_width: f64,
    /// Sidelobe level (in dB)
    pub sidelobe_level_db: f64,
    /// Coherent gain
    pub coherent_gain: f64,
    /// Processing gain
    pub processing_gain: f64,
    /// Equivalent noise bandwidth
    pub enbw: f64,
    /// Scalloping loss (in dB)
    pub scalloping_loss_db: f64,
    /// Worst-case processing loss (in dB)
    pub worst_case_loss_db: f64,
}

/// Analyze properties of a window function
pub fn analyze_window(
    window: &Array1<f64>,
    sample_rate: Option<f64>,
) -> FFTResult<WindowProperties> {
    let n = window.len();
    let fs = sample_rate.unwrap_or(1.0);

    // Coherent gain (DC gain)
    let coherent_gain = window.sum() / n as f64;

    // Processing gain
    let sum_squared: f64 = window.mapv(|x| x * x).sum();
    let processing_gain = window.sum().powi(2) / (n as f64 * sum_squared);

    // ENBW (Equivalent Noise Bandwidth)
    let enbw = n as f64 * sum_squared / window.sum().powi(2);

    // Compute window spectrum via FFT
    let n_fft = 8 * n; // Use larger FFT for better resolution
    let mut padded = Array1::zeros(n_fft);
    for i in 0..n {
        padded[i] = window[i];
    }

    let fft_result = fft(
        &padded
            .mapv(|x| num_complex::Complex64::new(x, 0.0))
            .to_vec(),
        None,
    )?;
    let magnitude: Vec<f64> = fft_result.iter().map(|c| c.norm()).collect();

    // Convert to dB
    let max_mag = magnitude.iter().cloned().fold(0.0, f64::max);
    let mag_db: Vec<f64> = magnitude
        .iter()
        .map(|&m| 20.0 * (m / max_mag).log10())
        .collect();

    // Find main lobe width (3dB down from peak)
    let mut main_lobe_width = 0.0;
    for (i, &val) in mag_db.iter().enumerate().take(n_fft / 2).skip(1) {
        if val < -3.0 {
            main_lobe_width = 2.0 * i as f64 * fs / n_fft as f64;
            break;
        }
    }

    // Find sidelobe level
    let mut sidelobe_level_db = -1000.0;
    let main_lobe_end = (main_lobe_width * n_fft as f64 / fs).ceil() as usize;
    for &val in mag_db.iter().take(n_fft / 2).skip(main_lobe_end) {
        if val > sidelobe_level_db {
            sidelobe_level_db = val;
        }
    }

    // Scalloping loss (loss at bin edge)
    let bin_edge_response = magnitude[n_fft / (2 * n)];
    let scalloping_loss_db = -20.0 * (bin_edge_response / max_mag).log10();

    // Worst-case processing loss
    let worst_case_loss_db = scalloping_loss_db + 10.0 * processing_gain.log10();

    Ok(WindowProperties {
        main_lobe_width,
        sidelobe_level_db,
        coherent_gain,
        processing_gain,
        enbw,
        scalloping_loss_db,
        worst_case_loss_db,
    })
}

/// Create a window visualization plot data
pub fn visualize_window(
    window: &Array1<f64>,
) -> FFTResult<(Array1<f64>, Array1<f64>, Array2<f64>)> {
    let n = window.len();

    // Time domain
    let time = Array1::range(0.0, n as f64, 1.0);

    // Frequency response
    let n_fft = 8 * n;
    let mut padded = vec![num_complex::Complex64::new(0.0, 0.0); n_fft];
    for i in 0..n {
        padded[i] = num_complex::Complex64::new(window[i], 0.0);
    }

    let fft_result = fft(&padded, None).unwrap_or(padded);
    let freq = fftfreq(n_fft, 1.0)?;
    let magnitude: Vec<f64> = fft_result.iter().map(|c| c.norm()).collect();

    // Convert to dB
    let max_mag = magnitude.iter().cloned().fold(0.0, f64::max);
    let mag_db: Vec<f64> = magnitude
        .iter()
        .map(|&m| 20.0 * (m / max_mag).max(1e-100).log10())
        .collect();

    // Create frequency response matrix (frequency, magnitude_db)
    let mut freq_response = Array2::zeros((n_fft, 2));
    for i in 0..n_fft {
        freq_response[[i, 0]] = freq[i];
        freq_response[[i, 1]] = mag_db[i];
    }

    Ok((time, window.clone(), freq_response))
}

/// Compare multiple windows
pub fn compare_windows(
    windows: &[(String, Array1<f64>)],
) -> FFTResult<Vec<(String, WindowProperties)>> {
    let mut results = Vec::new();

    for (name, window) in windows {
        let props = analyze_window(window, None)?;
        results.push((name.clone(), props));
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Window;

    #[test]
    fn test_extended_windows() {
        let n = 64;

        // Test Chebyshev window
        let cheby = get_extended_window(
            ExtendedWindow::Chebyshev {
                attenuation_db: 60.0,
            },
            n,
        )
        .unwrap();
        assert_eq!(cheby.len(), n);

        // Test Lanczos window
        let lanczos = get_extended_window(ExtendedWindow::Lanczos, n).unwrap();
        assert_eq!(lanczos.len(), n);

        // Test Poisson window
        let poisson = get_extended_window(ExtendedWindow::Poisson { alpha: 2.0 }, n).unwrap();
        assert_eq!(poisson.len(), n);
    }

    #[test]
    fn test_window_analysis() {
        let n = 64;
        let window = crate::window::get_window(Window::Hann, n, true).unwrap();

        let props = analyze_window(&window, Some(1000.0)).unwrap();

        // Check basic properties
        assert!(props.coherent_gain > 0.0);
        assert!(props.processing_gain > 0.0);
        assert!(props.enbw > 0.0);
        assert!(props.sidelobe_level_db < 0.0);
    }

    #[test]
    fn test_window_comparison() {
        let n = 64;
        let windows = vec![
            (
                "Hann".to_string(),
                crate::window::get_window(Window::Hann, n, true).unwrap(),
            ),
            (
                "Hamming".to_string(),
                crate::window::get_window(Window::Hamming, n, true).unwrap(),
            ),
        ];

        let comparison = compare_windows(&windows).unwrap();
        assert_eq!(comparison.len(), 2);
    }
}
