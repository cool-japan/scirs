// Signal deconvolution module
//
// This module implements various deconvolution techniques for signal processing,
// including Wiener deconvolution, Richardson-Lucy deconvolution, regularized deconvolution,
// and blind deconvolution.

use crate::convolve;
use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2, Array3};
use rustfft::{num_complex::Complex, FftPlanner};

/// Deconvolution configuration
#[derive(Debug, Clone)]
pub struct DeconvolutionConfig {
    /// Regularization parameter for Wiener and Tikhonov deconvolution
    pub reg_param: f64,
    /// Maximum number of iterations for iterative methods
    pub max_iterations: usize,
    /// Convergence threshold for iterative methods
    pub convergence_threshold: f64,
    /// Whether to apply positivity constraint
    pub positivity_constraint: bool,
    /// Whether to use FFT-based method (true) or direct method (false)
    pub use_fft: bool,
    /// Whether to pad the signal to avoid edge effects
    pub pad_signal: bool,
    /// Whether to optimize the regularization parameter automatically
    pub auto_regularization: bool,
    /// Whether to apply Gaussian smoothing before deconvolution
    pub prefilter: bool,
    /// Sigma parameter for Gaussian prefilter
    pub prefilter_sigma: f64,
    /// Whether to enforce boundary conditions for image deconvolution
    pub enforce_boundary: bool,
}

impl Default for DeconvolutionConfig {
    fn default() -> Self {
        Self {
            reg_param: 0.1,
            max_iterations: 50,
            convergence_threshold: 1e-6,
            positivity_constraint: true,
            use_fft: true,
            pad_signal: true,
            auto_regularization: false,
            prefilter: false,
            prefilter_sigma: 0.5,
            enforce_boundary: true,
        }
    }
}

/// Apply Wiener deconvolution to a 1D signal
///
/// Wiener deconvolution is a frequency-domain approach that incorporates
/// prior knowledge about the noise level.
///
/// # Arguments
///
/// * `signal` - The observed signal
/// * `psf` - The point spread function (impulse response)
/// * `noise_level` - The noise-to-signal power ratio
/// * `config` - Deconvolution configuration
///
/// # Returns
///
/// * The deconvolved signal
pub fn wiener_deconvolution_1d(
    signal: &Array1<f64>,
    psf: &Array1<f64>,
    noise_level: f64,
    config: &DeconvolutionConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check inputs
    if psf.len() > n {
        return Err(SignalError::ValueError(
            "PSF length cannot be greater than signal length".to_string(),
        ));
    }

    // Pad signals if requested
    let (padded_signal, padded_psf, pad_amount) = if config.pad_signal {
        let pad_len = n + psf.len() - 1;
        let mut padded_signal = Array1::<f64>::zeros(pad_len);
        let mut padded_psf = Array1::<f64>::zeros(pad_len);

        padded_signal.slice_mut(s![..n]).assign(signal);
        padded_psf.slice_mut(s![..psf.len()]).assign(psf);

        (padded_signal, padded_psf, (pad_len - n) / 2)
    } else {
        let mut padded_signal = Array1::<f64>::zeros(n);
        let mut padded_psf = Array1::<f64>::zeros(n);

        padded_signal.assign(signal);

        // Center the PSF
        let start = (n - psf.len()) / 2;
        padded_psf
            .slice_mut(s![start..start + psf.len()])
            .assign(psf);

        (padded_signal, padded_psf, 0)
    };

    let pad_len = padded_signal.len();

    // Apply Gaussian prefilter if requested
    let filtered_signal = if config.prefilter {
        gaussian_filter_1d(&padded_signal, config.prefilter_sigma)
    } else {
        padded_signal.clone()
    };

    // Convert to complex arrays for FFT
    let mut signal_complex = vec![Complex::new(0.0, 0.0); pad_len];
    let mut psf_complex = vec![Complex::new(0.0, 0.0); pad_len];

    for i in 0..pad_len {
        signal_complex[i] = Complex::new(filtered_signal[i], 0.0);
        psf_complex[i] = Complex::new(padded_psf[i], 0.0);
    }

    // Create FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(pad_len);
    let ifft = planner.plan_fft_inverse(pad_len);

    // Forward FFT of signal and PSF
    fft.process(&mut signal_complex);
    fft.process(&mut psf_complex);

    // Apply Wiener deconvolution in frequency domain
    let mut result_complex = vec![Complex::new(0.0, 0.0); pad_len];

    for i in 0..pad_len {
        let h = psf_complex[i];
        let h_conj = h.conj();
        let h_abs_sq = h.norm_sqr();

        // Wiener filter
        let denom = h_abs_sq + noise_level;
        if denom > 1e-10 {
            result_complex[i] = signal_complex[i] * h_conj / denom;
        } else {
            result_complex[i] = Complex::new(0.0, 0.0);
        }
    }

    // Inverse FFT to get the deconvolved signal
    ifft.process(&mut result_complex);

    // Scale and convert back to real
    let scale = 1.0 / (pad_len as f64);
    let mut deconvolved = Array1::<f64>::zeros(n);

    if config.pad_signal {
        for i in 0..n {
            let val = result_complex[i + pad_amount].re * scale;
            deconvolved[i] = if config.positivity_constraint {
                val.max(0.0)
            } else {
                val
            };
        }
    } else {
        for i in 0..n {
            let val = result_complex[i].re * scale;
            deconvolved[i] = if config.positivity_constraint {
                val.max(0.0)
            } else {
                val
            };
        }
    }

    Ok(deconvolved)
}

/// Apply Tikhonov regularized deconvolution to a 1D signal
///
/// Tikhonov regularization adds a smoothness constraint to stabilize
/// the deconvolution process.
///
/// # Arguments
///
/// * `signal` - The observed signal
/// * `psf` - The point spread function (impulse response)
/// * `alpha` - Regularization parameter (0 for no regularization)
/// * `config` - Deconvolution configuration
///
/// # Returns
///
/// * The deconvolved signal
pub fn tikhonov_deconvolution_1d(
    signal: &Array1<f64>,
    psf: &Array1<f64>,
    alpha: Option<f64>,
    config: &DeconvolutionConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check inputs
    if psf.len() > n {
        return Err(SignalError::ValueError(
            "PSF length cannot be greater than signal length".to_string(),
        ));
    }

    // Use provided alpha or config value
    let reg_param = alpha.unwrap_or(config.reg_param);

    // Pad signals if requested
    let (padded_signal, padded_psf, pad_amount) = if config.pad_signal {
        let pad_len = n + psf.len() - 1;
        let mut padded_signal = Array1::<f64>::zeros(pad_len);
        let mut padded_psf = Array1::<f64>::zeros(pad_len);

        padded_signal.slice_mut(s![..n]).assign(signal);
        padded_psf.slice_mut(s![..psf.len()]).assign(psf);

        (padded_signal, padded_psf, (pad_len - n) / 2)
    } else {
        let mut padded_signal = Array1::<f64>::zeros(n);
        let mut padded_psf = Array1::<f64>::zeros(n);

        padded_signal.assign(signal);

        // Center the PSF
        let start = (n - psf.len()) / 2;
        padded_psf
            .slice_mut(s![start..start + psf.len()])
            .assign(psf);

        (padded_signal, padded_psf, 0)
    };

    let pad_len = padded_signal.len();

    // Apply Gaussian prefilter if requested
    let filtered_signal = if config.prefilter {
        gaussian_filter_1d(&padded_signal, config.prefilter_sigma)
    } else {
        padded_signal.clone()
    };

    // Convert to complex arrays for FFT
    let mut signal_complex = vec![Complex::new(0.0, 0.0); pad_len];
    let mut psf_complex = vec![Complex::new(0.0, 0.0); pad_len];
    let mut l_complex = vec![Complex::new(0.0, 0.0); pad_len];

    for i in 0..pad_len {
        signal_complex[i] = Complex::new(filtered_signal[i], 0.0);
        psf_complex[i] = Complex::new(padded_psf[i], 0.0);
    }

    // Create second derivative operator for regularization
    // This is a simple approximation to the Laplacian
    l_complex[0] = Complex::new(1.0, 0.0);
    l_complex[1] = Complex::new(-2.0, 0.0);
    l_complex[2] = Complex::new(1.0, 0.0);

    // Create FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(pad_len);
    let ifft = planner.plan_fft_inverse(pad_len);

    // Forward FFT of signal, PSF, and regularization operator
    fft.process(&mut signal_complex);
    fft.process(&mut psf_complex);
    fft.process(&mut l_complex);

    // Apply Tikhonov regularized deconvolution in frequency domain
    let mut result_complex = vec![Complex::new(0.0, 0.0); pad_len];

    for i in 0..pad_len {
        let h = psf_complex[i];
        let h_conj = h.conj();
        let h_abs_sq = h.norm_sqr();
        let l_abs_sq = l_complex[i].norm_sqr();

        // Tikhonov filter
        let denom = h_abs_sq + reg_param * l_abs_sq;
        if denom > 1e-10 {
            result_complex[i] = signal_complex[i] * h_conj / denom;
        } else {
            result_complex[i] = Complex::new(0.0, 0.0);
        }
    }

    // Inverse FFT to get the deconvolved signal
    ifft.process(&mut result_complex);

    // Scale and convert back to real
    let scale = 1.0 / (pad_len as f64);
    let mut deconvolved = Array1::<f64>::zeros(n);

    if config.pad_signal {
        for i in 0..n {
            let val = result_complex[i + pad_amount].re * scale;
            deconvolved[i] = if config.positivity_constraint {
                val.max(0.0)
            } else {
                val
            };
        }
    } else {
        for i in 0..n {
            let val = result_complex[i].re * scale;
            deconvolved[i] = if config.positivity_constraint {
                val.max(0.0)
            } else {
                val
            };
        }
    }

    Ok(deconvolved)
}

/// Apply Richardson-Lucy deconvolution to a 1D signal
///
/// Richardson-Lucy deconvolution is an iterative method derived from
/// Bayesian probability theory. It enforces positivity and is well-suited
/// for Poisson noise.
///
/// # Arguments
///
/// * `signal` - The observed signal
/// * `psf` - The point spread function (impulse response)
/// * `iterations` - Number of iterations (or None to use config default)
/// * `config` - Deconvolution configuration
///
/// # Returns
///
/// * The deconvolved signal
pub fn richardson_lucy_deconvolution_1d(
    signal: &Array1<f64>,
    psf: &Array1<f64>,
    iterations: Option<usize>,
    config: &DeconvolutionConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check inputs
    if psf.len() > n {
        return Err(SignalError::ValueError(
            "PSF length cannot be greater than signal length".to_string(),
        ));
    }

    // Use provided iterations or config value
    let max_iter = iterations.unwrap_or(config.max_iterations);

    // Pad signals if requested
    let (padded_signal, padded_psf, pad_amount) = if config.pad_signal {
        let pad_len = n + psf.len() - 1;
        let mut padded_signal = Array1::<f64>::zeros(pad_len);
        let mut padded_psf = Array1::<f64>::zeros(pad_len);

        padded_signal.slice_mut(s![..n]).assign(signal);
        padded_psf.slice_mut(s![..psf.len()]).assign(psf);

        (padded_signal, padded_psf, (pad_len - n) / 2)
    } else {
        let mut padded_signal = Array1::<f64>::zeros(n);
        let mut padded_psf = Array1::<f64>::zeros(n);

        padded_signal.assign(signal);

        // Center the PSF
        let start = (n - psf.len()) / 2;
        padded_psf
            .slice_mut(s![start..start + psf.len()])
            .assign(psf);

        (padded_signal, padded_psf, 0)
    };

    let pad_len = padded_signal.len();

    // Flip the PSF for convolution (correlation)
    let mut flipped_psf = Array1::<f64>::zeros(pad_len);
    for i in 0..psf.len() {
        flipped_psf[pad_len - 1 - i] = padded_psf[i];
    }

    // Initialize estimate with uniform positive values
    let signal_mean = padded_signal.sum() / (pad_len as f64);
    let mut estimate = Array1::<f64>::zeros(pad_len);
    estimate.fill(signal_mean.max(1e-6));

    // Normalize the PSF
    let psf_sum = padded_psf.sum();
    let mut normalized_psf = padded_psf.clone();
    if psf_sum > 0.0 {
        normalized_psf /= psf_sum;
    }

    let mut prev_estimate = Array1::<f64>::zeros(pad_len);

    // Iterative Richardson-Lucy algorithm
    for _iter in 0..max_iter {
        // Save previous estimate for convergence check
        prev_estimate.assign(&estimate);

        // Compute the predicted signal
        let predicted = convolve::convolve(
            estimate.as_slice().unwrap(),
            normalized_psf.as_slice().unwrap(),
            "same",
        )?;

        // Compute the correction factor
        let mut correction = Array1::<f64>::zeros(pad_len);
        for i in 0..pad_len {
            let pred_val = predicted[i].max(1e-10);
            correction[i] = padded_signal[i] / pred_val;
        }

        // Apply the correction
        let correction_blurred = convolve::convolve(
            correction.as_slice().unwrap(),
            flipped_psf.as_slice().unwrap(),
            "same",
        )?;
        for i in 0..pad_len {
            estimate[i] *= correction_blurred[i];
        }

        // Check for convergence
        if config.convergence_threshold > 0.0 {
            let diff = (&estimate - &prev_estimate).mapv(|x| x.abs()).sum()
                / prev_estimate.mapv(|x| x.abs()).sum();
            if diff < config.convergence_threshold {
                break;
            }
        }
    }

    // Extract result
    let mut deconvolved = Array1::<f64>::zeros(n);
    if config.pad_signal {
        for i in 0..n {
            deconvolved[i] = estimate[i + pad_amount];
        }
    } else {
        deconvolved.assign(&estimate.slice(s![..n]));
    }

    Ok(deconvolved)
}

/// Apply CLEAN deconvolution to a 1D signal
///
/// CLEAN is an iterative deconvolution algorithm commonly used in radio astronomy.
/// It works by successively subtracting the PSF from the brightest components.
///
/// # Arguments
///
/// * `signal` - The observed signal
/// * `psf` - The point spread function (impulse response)
/// * `gain` - Loop gain factor (0 < gain ≤ 1)
/// * `threshold` - Stopping threshold (fraction of peak)
/// * `config` - Deconvolution configuration
///
/// # Returns
///
/// * The deconvolved signal
pub fn clean_deconvolution_1d(
    signal: &Array1<f64>,
    psf: &Array1<f64>,
    gain: f64,
    threshold: f64,
    config: &DeconvolutionConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check inputs
    if psf.len() > n {
        return Err(SignalError::ValueError(
            "PSF length cannot be greater than signal length".to_string(),
        ));
    }

    if gain <= 0.0 || gain > 1.0 {
        return Err(SignalError::ValueError(
            "Gain must be between 0 and 1".to_string(),
        ));
    }

    if threshold <= 0.0 || threshold >= 1.0 {
        return Err(SignalError::ValueError(
            "Threshold must be between 0 and 1".to_string(),
        ));
    }

    // Pad signals if requested
    let (padded_signal, padded_psf, pad_amount) = if config.pad_signal {
        let pad_len = n + psf.len() - 1;
        let mut padded_signal = Array1::<f64>::zeros(pad_len);
        let mut padded_psf = Array1::<f64>::zeros(pad_len);

        padded_signal.slice_mut(s![..n]).assign(signal);
        padded_psf.slice_mut(s![..psf.len()]).assign(psf);

        (padded_signal, padded_psf, (pad_len - n) / 2)
    } else {
        let mut padded_signal = Array1::<f64>::zeros(n);
        let mut padded_psf = Array1::<f64>::zeros(n);

        padded_signal.assign(signal);

        // Center the PSF
        let start = (n - psf.len()) / 2;
        padded_psf
            .slice_mut(s![start..start + psf.len()])
            .assign(psf);

        (padded_signal, padded_psf, 0)
    };

    let pad_len = padded_signal.len();

    // Initialize model (clean components) and residual
    let mut model = Array1::<f64>::zeros(pad_len);
    let mut residual = padded_signal.clone();

    // Normalize the PSF
    let psf_max = padded_psf.iter().fold(0.0, |m, &x| f64::max(m, x.abs()));
    let mut normalized_psf = padded_psf.clone();
    if psf_max > 0.0 {
        normalized_psf /= psf_max;
    }

    // Calculate initial peak value in the signal
    let signal_peak = padded_signal.iter().fold(0.0, |m, &x| f64::max(m, x.abs()));
    let stop_threshold = signal_peak * threshold;

    // CLEAN iterative algorithm
    for _ in 0..config.max_iterations {
        // Find the peak in the residual
        let (peak_idx, peak_val) = {
            let (idx, val) = residual
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .unwrap();
            (idx, *val)
        };

        // Check if peak is below threshold
        if peak_val.abs() < stop_threshold {
            break;
        }

        // Update the model
        model[peak_idx] += gain * peak_val;

        // Subtract the scaled PSF from the residual
        for i in 0..pad_len {
            let psf_idx =
                (i as isize - peak_idx as isize + pad_len as isize / 2) as usize % pad_len;
            residual[i] -= gain * peak_val * normalized_psf[psf_idx];
        }
    }

    // Convolve model with a restoring beam (Gaussian)
    let restoring_beam = create_gaussian_kernel((psf.len() / 2).max(3));
    let restored_vec = convolve::convolve(
        model.as_slice().unwrap(),
        restoring_beam.as_slice().unwrap(),
        "same",
    )?;
    let restored = Array1::from_vec(restored_vec);

    // Add residual back to the model
    let result = restored + residual;

    // Extract final result
    let mut deconvolved = Array1::<f64>::zeros(n);
    if config.pad_signal {
        for i in 0..n {
            deconvolved[i] = result[i + pad_amount];
        }
    } else {
        deconvolved.assign(&result.slice(s![..n]));
    }

    Ok(deconvolved)
}

/// Apply Maximum Entropy Method (MEM) deconvolution to a 1D signal
///
/// MEM is a Bayesian approach that selects the most probable solution
/// consistent with the data while maximizing entropy.
///
/// # Arguments
///
/// * `signal` - The observed signal
/// * `psf` - The point spread function (impulse response)
/// * `noise_level` - Estimate of the noise level in the signal
/// * `config` - Deconvolution configuration
///
/// # Returns
///
/// * The deconvolved signal
pub fn mem_deconvolution_1d(
    signal: &Array1<f64>,
    psf: &Array1<f64>,
    noise_level: f64,
    config: &DeconvolutionConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check inputs
    if psf.len() > n {
        return Err(SignalError::ValueError(
            "PSF length cannot be greater than signal length".to_string(),
        ));
    }

    // Pad signals if requested
    let (padded_signal, padded_psf, pad_amount) = if config.pad_signal {
        let pad_len = n + psf.len() - 1;
        let mut padded_signal = Array1::<f64>::zeros(pad_len);
        let mut padded_psf = Array1::<f64>::zeros(pad_len);

        padded_signal.slice_mut(s![..n]).assign(signal);
        padded_psf.slice_mut(s![..psf.len()]).assign(psf);

        (padded_signal, padded_psf, (pad_len - n) / 2)
    } else {
        let mut padded_signal = Array1::<f64>::zeros(n);
        let mut padded_psf = Array1::<f64>::zeros(n);

        padded_signal.assign(signal);

        // Center the PSF
        let start = (n - psf.len()) / 2;
        padded_psf
            .slice_mut(s![start..start + psf.len()])
            .assign(psf);

        (padded_signal, padded_psf, 0)
    };

    let pad_len = padded_signal.len();

    // Normalize the PSF
    let psf_sum = padded_psf.sum();
    let mut normalized_psf = padded_psf.clone();
    if psf_sum > 0.0 {
        normalized_psf /= psf_sum;
    }

    // Initialize with flat positive spectrum
    let total_flux = padded_signal.sum().max(1e-6);
    let mut model = Array1::<f64>::zeros(pad_len);
    model.fill(total_flux / (pad_len as f64));

    let mut prev_model = model.clone();
    let mut lagrange_multiplier = 0.0;

    // MEM iterative algorithm
    for _iter in 0..config.max_iterations {
        // Save previous model for convergence check
        prev_model.assign(&model);

        // Calculate model response
        let response = convolve::convolve(
            model.as_slice().unwrap(),
            normalized_psf.as_slice().unwrap(),
            "same",
        )?;

        // Calculate chi-squared
        let mut chi_squared = 0.0;
        for i in 0..pad_len {
            let diff = response[i] - padded_signal[i];
            chi_squared += (diff * diff) / (noise_level * noise_level);
        }

        // Adjust Lagrange multiplier to meet the chi-squared constraint
        let target_chi_squared = pad_len as f64;
        if chi_squared < target_chi_squared {
            lagrange_multiplier *= 0.9;
        } else {
            lagrange_multiplier = f64::max(lagrange_multiplier * 1.1, 1e-6);
        }

        // Update the model using gradient descent on the entropy
        for i in 0..pad_len {
            // Calculate the gradient of chi-squared term
            let mut chi_grad = 0.0;
            for j in 0..pad_len {
                let k = (j as isize - i as isize + pad_len as isize) as usize % pad_len;
                let diff = response[j] - padded_signal[j];
                chi_grad += diff * normalized_psf[k] / (noise_level * noise_level);
            }

            // Entropy gradient (assuming uniform default model)
            let entropy_grad = -1.0 - model[i].ln();

            // Update model
            let update = entropy_grad - lagrange_multiplier * chi_grad;
            model[i] *= (1.0 + update * 0.1).max(0.1);
        }

        // Normalize the model to conserve total flux
        let model_sum = model.sum();
        if model_sum > 0.0 {
            model *= total_flux / model_sum;
        }

        // Check for convergence
        if config.convergence_threshold > 0.0 {
            let diff =
                (&model - &prev_model).mapv(|x| x.abs()).sum() / prev_model.mapv(|x| x.abs()).sum();
            if diff < config.convergence_threshold {
                break;
            }
        }
    }

    // Extract final result
    let mut deconvolved = Array1::<f64>::zeros(n);
    if config.pad_signal {
        for i in 0..n {
            deconvolved[i] = model[i + pad_amount];
        }
    } else {
        deconvolved.assign(&model.slice(s![..n]));
    }

    Ok(deconvolved)
}

/// Apply blind deconvolution to a 1D signal
///
/// Blind deconvolution attempts to recover both the original signal and the PSF
/// from the observed signal.
///
/// # Arguments
///
/// * `signal` - The observed signal
/// * `psf_size` - Estimated size of the PSF
/// * `config` - Deconvolution configuration
///
/// # Returns
///
/// * Tuple containing (deconvolved signal, estimated PSF)
pub fn blind_deconvolution_1d(
    signal: &Array1<f64>,
    psf_size: usize,
    config: &DeconvolutionConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    let n = signal.len();

    // Check inputs
    if psf_size >= n {
        return Err(SignalError::ValueError(
            "PSF size must be smaller than signal length".to_string(),
        ));
    }

    // Pad signal if requested
    let (padded_signal, pad_amount) = if config.pad_signal {
        let pad_len = n + psf_size - 1;
        let mut padded_signal = Array1::<f64>::zeros(pad_len);

        padded_signal.slice_mut(s![..n]).assign(signal);

        (padded_signal, (pad_len - n) / 2)
    } else {
        (signal.clone(), 0)
    };

    let pad_len = padded_signal.len();

    // Initialize with a Gaussian PSF
    let mut estimated_psf = create_gaussian_kernel(psf_size);

    // Normalize the PSF
    let psf_sum = estimated_psf.sum();
    if psf_sum > 0.0 {
        estimated_psf /= psf_sum;
    }

    // Initialize model with a deconvolution using the initial PSF
    let mut estimated_signal =
        wiener_deconvolution_1d(&padded_signal, &estimated_psf, 0.1, config)?;

    // Alternating minimization
    for _iter in 0..config.max_iterations {
        // Save previous estimates for convergence check
        let prev_signal = estimated_signal.clone();
        let prev_psf = estimated_psf.clone();

        // Update signal estimate (keeping PSF fixed)
        estimated_signal = richardson_lucy_deconvolution_1d(
            &padded_signal,
            &estimated_psf,
            Some(5), // Just a few iterations per outer iteration
            config,
        )?;

        // Apply constraints to the signal
        if config.positivity_constraint {
            estimated_signal = estimated_signal.mapv(|x| x.max(0.0));
        }

        // Update PSF estimate (keeping signal fixed)
        // This is essentially solving the same deconvolution with roles reversed
        let temp_psf =
            richardson_lucy_deconvolution_1d(&padded_signal, &estimated_signal, Some(5), config)?;

        // Constrain PSF to its expected size
        let _centered_psf = Array1::<f64>::zeros(pad_len);
        let start = (pad_len - psf_size) / 2;
        let mut psf_cropped = temp_psf.slice(s![start..start + psf_size]).to_owned();

        // Apply constraints to the PSF
        if config.positivity_constraint {
            psf_cropped = psf_cropped.mapv(|x| x.max(0.0));
        }

        // Normalize PSF
        let psf_cropped_sum = psf_cropped.sum();
        if psf_cropped_sum > 0.0 {
            psf_cropped /= psf_cropped_sum;
        }

        estimated_psf = psf_cropped;

        // Check for convergence
        if config.convergence_threshold > 0.0 {
            let signal_diff = (&estimated_signal - &prev_signal).mapv(|x| x.abs()).sum()
                / prev_signal.mapv(|x| x.abs()).sum();

            let psf_diff = (&estimated_psf - &prev_psf).mapv(|x| x.abs()).sum()
                / prev_psf.mapv(|x| x.abs()).sum();

            if signal_diff < config.convergence_threshold && psf_diff < config.convergence_threshold
            {
                break;
            }
        }
    }

    // Extract final signal result
    let mut deconvolved = Array1::<f64>::zeros(n);
    if config.pad_signal {
        for i in 0..n {
            deconvolved[i] = estimated_signal[i + pad_amount];
        }
    } else {
        deconvolved.assign(&estimated_signal.slice(s![..n]));
    }

    Ok((deconvolved, estimated_psf))
}

/// Apply Wiener deconvolution to a 2D image
///
/// Wiener deconvolution is a frequency-domain approach that incorporates
/// prior knowledge about the noise level.
///
/// # Arguments
///
/// * `image` - The observed image
/// * `psf` - The point spread function
/// * `noise_level` - The noise-to-signal power ratio
/// * `config` - Deconvolution configuration
///
/// # Returns
///
/// * The deconvolved image
pub fn wiener_deconvolution_2d(
    image: &Array2<f64>,
    psf: &Array2<f64>,
    noise_level: f64,
    config: &DeconvolutionConfig,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();
    let (psf_h, psf_w) = psf.dim();

    // Check inputs
    if psf_h > height || psf_w > width {
        return Err(SignalError::ValueError(
            "PSF dimensions cannot be greater than image dimensions".to_string(),
        ));
    }

    // Pad image and PSF if requested
    let (padded_image, padded_psf, pad_h, pad_w) = if config.pad_signal {
        let pad_h = psf_h - 1;
        let pad_w = psf_w - 1;
        let pad_height = height + pad_h;
        let pad_width = width + pad_w;

        let mut padded_image = Array2::<f64>::zeros((pad_height, pad_width));
        let mut padded_psf = Array2::<f64>::zeros((pad_height, pad_width));

        // Copy original image
        padded_image.slice_mut(s![..height, ..width]).assign(image);

        // Copy PSF to the center (centered at top-left corner)
        padded_psf.slice_mut(s![..psf_h, ..psf_w]).assign(psf);

        (padded_image, padded_psf, pad_h / 2, pad_w / 2)
    } else {
        let padded_image = image.clone();
        let mut padded_psf = Array2::<f64>::zeros((height, width));

        // Center the PSF
        let start_h = (height - psf_h) / 2;
        let start_w = (width - psf_w) / 2;
        padded_psf
            .slice_mut(s![start_h..start_h + psf_h, start_w..start_w + psf_w])
            .assign(psf);

        (padded_image, padded_psf, 0, 0)
    };

    let (pad_height, pad_width) = padded_image.dim();

    // Apply prefilter if requested
    let filtered_image = if config.prefilter {
        gaussian_filter_2d(&padded_image, config.prefilter_sigma)
    } else {
        padded_image.clone()
    };

    // We'll use the 2D FFT by reshaping data into 1D arrays
    // Convert to complex arrays for FFT
    let mut image_complex = vec![Complex::new(0.0, 0.0); pad_height * pad_width];
    let mut psf_complex = vec![Complex::new(0.0, 0.0); pad_height * pad_width];

    for i in 0..pad_height {
        for j in 0..pad_width {
            let idx = i * pad_width + j;
            image_complex[idx] = Complex::new(filtered_image[[i, j]], 0.0);
            psf_complex[idx] = Complex::new(padded_psf[[i, j]], 0.0);
        }
    }

    // Create FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(pad_height * pad_width);
    let ifft = planner.plan_fft_inverse(pad_height * pad_width);

    // Forward FFT of image and PSF
    fft.process(&mut image_complex);
    fft.process(&mut psf_complex);

    // Apply Wiener deconvolution in frequency domain
    let mut result_complex = vec![Complex::new(0.0, 0.0); pad_height * pad_width];

    for i in 0..pad_height * pad_width {
        let h = psf_complex[i];
        let h_conj = h.conj();
        let h_abs_sq = h.norm_sqr();

        // Wiener filter
        let denom = h_abs_sq + noise_level;
        if denom > 1e-10 {
            result_complex[i] = image_complex[i] * h_conj / denom;
        } else {
            result_complex[i] = Complex::new(0.0, 0.0);
        }
    }

    // Inverse FFT to get the deconvolved image
    ifft.process(&mut result_complex);

    // Scale and convert back to real
    let scale = 1.0 / (pad_height * pad_width) as f64;
    let mut deconvolved = Array2::<f64>::zeros((height, width));

    for i in 0..height {
        for j in 0..width {
            let idx = (i + pad_h) * pad_width + (j + pad_w);
            let val = result_complex[idx].re * scale;
            deconvolved[[i, j]] = if config.positivity_constraint {
                val.max(0.0)
            } else {
                val
            };
        }
    }

    Ok(deconvolved)
}

/// Apply Richardson-Lucy deconvolution to a 2D image
///
/// Richardson-Lucy deconvolution is an iterative method derived from
/// Bayesian probability theory. It enforces positivity and is well-suited
/// for Poisson noise.
///
/// # Arguments
///
/// * `image` - The observed image
/// * `psf` - The point spread function
/// * `iterations` - Number of iterations (or None to use config default)
/// * `config` - Deconvolution configuration
///
/// # Returns
///
/// * The deconvolved image
pub fn richardson_lucy_deconvolution_2d(
    image: &Array2<f64>,
    psf: &Array2<f64>,
    iterations: Option<usize>,
    config: &DeconvolutionConfig,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();
    let (psf_h, psf_w) = psf.dim();

    // Check inputs
    if psf_h > height || psf_w > width {
        return Err(SignalError::ValueError(
            "PSF dimensions cannot be greater than image dimensions".to_string(),
        ));
    }

    // Use provided iterations or config value
    let max_iter = iterations.unwrap_or(config.max_iterations);

    // Pad image and PSF if requested
    let (padded_image, padded_psf, pad_h, pad_w) = if config.pad_signal {
        let pad_h = psf_h - 1;
        let pad_w = psf_w - 1;
        let pad_height = height + pad_h;
        let pad_width = width + pad_w;

        let mut padded_image = Array2::<f64>::zeros((pad_height, pad_width));
        let mut padded_psf = Array2::<f64>::zeros((pad_height, pad_width));

        // Copy original image
        padded_image.slice_mut(s![..height, ..width]).assign(image);

        // Copy PSF to the center
        padded_psf.slice_mut(s![..psf_h, ..psf_w]).assign(psf);

        (padded_image, padded_psf, pad_h / 2, pad_w / 2)
    } else {
        let padded_image = image.clone();
        let mut padded_psf = Array2::<f64>::zeros((height, width));

        // Center the PSF
        let start_h = (height - psf_h) / 2;
        let start_w = (width - psf_w) / 2;
        padded_psf
            .slice_mut(s![start_h..start_h + psf_h, start_w..start_w + psf_w])
            .assign(psf);

        (padded_image, padded_psf, 0, 0)
    };

    let (pad_height, pad_width) = padded_image.dim();

    // Flip the PSF for convolution (correlation)
    let mut flipped_psf = Array2::<f64>::zeros((pad_height, pad_width));
    for i in 0..psf_h {
        for j in 0..psf_w {
            flipped_psf[[pad_height - 1 - i, pad_width - 1 - j]] = padded_psf[[i, j]];
        }
    }

    // Initialize estimate with uniform positive values
    let image_mean = padded_image.sum() / (pad_height * pad_width) as f64;
    let mut estimate = Array2::<f64>::zeros((pad_height, pad_width));
    estimate.fill(image_mean.max(1e-6));

    // Normalize the PSF
    let psf_sum = padded_psf.sum();
    let mut normalized_psf = padded_psf.clone();
    if psf_sum > 0.0 {
        normalized_psf /= psf_sum;
    }

    let mut prev_estimate = Array2::<f64>::zeros((pad_height, pad_width));

    // Iterative Richardson-Lucy algorithm
    for _iter in 0..max_iter {
        // Save previous estimate for convergence check
        prev_estimate.assign(&estimate);

        // Compute the predicted image
        let predicted = convolve::convolve2d(&estimate, &normalized_psf, "same")?;

        // Compute the correction factor
        let mut correction = Array2::<f64>::zeros((pad_height, pad_width));
        for i in 0..pad_height {
            for j in 0..pad_width {
                let pred_val = predicted[[i, j]].max(1e-10);
                correction[[i, j]] = padded_image[[i, j]] / pred_val;
            }
        }

        // Apply the correction
        let correction_blurred = convolve::convolve2d(&correction, &flipped_psf, "same")?;
        for i in 0..pad_height {
            for j in 0..pad_width {
                estimate[[i, j]] *= correction_blurred[[i, j]];
            }
        }

        // Check for convergence
        if config.convergence_threshold > 0.0 {
            let diff = (&estimate - &prev_estimate).mapv(|x| x.abs()).sum()
                / prev_estimate.mapv(|x| x.abs()).sum();
            if diff < config.convergence_threshold {
                break;
            }
        }
    }

    // Extract result
    let mut deconvolved = Array2::<f64>::zeros((height, width));
    for i in 0..height {
        for j in 0..width {
            deconvolved[[i, j]] = estimate[[i + pad_h, j + pad_w]];
        }
    }

    Ok(deconvolved)
}

/// Apply total variation regularized deconvolution to a 2D image
///
/// Total variation regularization preserves edges while suppressing noise.
///
/// # Arguments
///
/// * `image` - The observed image
/// * `psf` - The point spread function
/// * `reg_param` - Regularization parameter
/// * `config` - Deconvolution configuration
///
/// # Returns
///
/// * The deconvolved image
pub fn tv_deconvolution_2d(
    image: &Array2<f64>,
    psf: &Array2<f64>,
    reg_param: f64,
    config: &DeconvolutionConfig,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();
    let (psf_h, psf_w) = psf.dim();

    // Check inputs
    if psf_h > height || psf_w > width {
        return Err(SignalError::ValueError(
            "PSF dimensions cannot be greater than image dimensions".to_string(),
        ));
    }

    // Pad image and PSF if requested
    let (padded_image, padded_psf, pad_h, pad_w) = if config.pad_signal {
        let pad_h = psf_h - 1;
        let pad_w = psf_w - 1;
        let pad_height = height + pad_h;
        let pad_width = width + pad_w;

        let mut padded_image = Array2::<f64>::zeros((pad_height, pad_width));
        let mut padded_psf = Array2::<f64>::zeros((pad_height, pad_width));

        // Copy original image
        padded_image.slice_mut(s![..height, ..width]).assign(image);

        // Copy PSF to the center
        padded_psf.slice_mut(s![..psf_h, ..psf_w]).assign(psf);

        (padded_image, padded_psf, pad_h / 2, pad_w / 2)
    } else {
        let padded_image = image.clone();
        let mut padded_psf = Array2::<f64>::zeros((height, width));

        // Center the PSF
        let start_h = (height - psf_h) / 2;
        let start_w = (width - psf_w) / 2;
        padded_psf
            .slice_mut(s![start_h..start_h + psf_h, start_w..start_w + psf_w])
            .assign(psf);

        (padded_image, padded_psf, 0, 0)
    };

    let (pad_height, pad_width) = padded_image.dim();

    // Flip the PSF for convolution (correlation)
    let mut flipped_psf = Array2::<f64>::zeros((pad_height, pad_width));
    for i in 0..psf_h {
        for j in 0..psf_w {
            flipped_psf[[pad_height - 1 - i, pad_width - 1 - j]] = padded_psf[[i, j]];
        }
    }

    // Initialize with a standard non-regularized deconvolution
    let mut estimate = wiener_deconvolution_2d(&padded_image, &padded_psf, 0.01, config)?;

    // Normalize the PSF
    let psf_sum = padded_psf.sum();
    let mut normalized_psf = padded_psf.clone();
    if psf_sum > 0.0 {
        normalized_psf /= psf_sum;
    }

    let mut prev_estimate = Array2::<f64>::zeros((pad_height, pad_width));

    // Small constant to avoid division by zero
    let eps = 1e-6;

    // Iterative total variation minimization
    for _iter in 0..config.max_iterations {
        // Save previous estimate for convergence check
        prev_estimate.assign(&estimate);

        // Compute the predicted image
        let predicted = convolve::convolve2d(&estimate, &normalized_psf, "same")?;

        // Compute the data fidelity gradient
        let diff = &predicted - &padded_image;
        let fidelity_grad = convolve::convolve2d(&diff, &flipped_psf, "same")?;

        // Compute the total variation gradient
        let mut tv_grad = Array2::<f64>::zeros((pad_height, pad_width));

        for i in 1..pad_height - 1 {
            for j in 1..pad_width - 1 {
                // Compute horizontal and vertical derivatives
                let dx = estimate[[i, j + 1]] - estimate[[i, j]];
                let dy = estimate[[i + 1, j]] - estimate[[i, j]];

                // Compute the gradient magnitude
                let grad_mag = (dx * dx + dy * dy).sqrt() + eps;

                // Compute the divergence of the normalized gradient
                let div_x = (estimate[[i, j]] - estimate[[i, j - 1]])
                    / ((estimate[[i, j]] - estimate[[i, j - 1]]).powi(2)
                        + (estimate[[i, j]] - estimate[[i - 1, j]]).powi(2))
                    .sqrt()
                    - dx / grad_mag;

                let div_y = (estimate[[i, j]] - estimate[[i - 1, j]])
                    / ((estimate[[i, j]] - estimate[[i, j - 1]]).powi(2)
                        + (estimate[[i, j]] - estimate[[i - 1, j]]).powi(2))
                    .sqrt()
                    - dy / grad_mag;

                tv_grad[[i, j]] = div_x + div_y;
            }
        }

        // Update estimate with combined gradients
        let step_size = 0.1;
        for i in 0..pad_height {
            for j in 0..pad_width {
                estimate[[i, j]] -=
                    step_size * (fidelity_grad[[i, j]] - reg_param * tv_grad[[i, j]]);

                // Apply positivity constraint if requested
                if config.positivity_constraint {
                    estimate[[i, j]] = estimate[[i, j]].max(0.0);
                }
            }
        }

        // Check for convergence
        if config.convergence_threshold > 0.0 {
            let diff = (&estimate - &prev_estimate).mapv(|x| x.abs()).sum()
                / prev_estimate.mapv(|x| x.abs()).sum();
            if diff < config.convergence_threshold {
                break;
            }
        }
    }

    // Extract result
    let mut deconvolved = Array2::<f64>::zeros((height, width));
    for i in 0..height {
        for j in 0..width {
            deconvolved[[i, j]] = estimate[[i + pad_h, j + pad_w]];
        }
    }

    Ok(deconvolved)
}

/// Apply blind deconvolution to a 2D image
///
/// Blind deconvolution attempts to recover both the original image and the PSF
/// from the observed image.
///
/// # Arguments
///
/// * `image` - The observed image
/// * `psf_size_h` - Estimated height of the PSF
/// * `psf_size_w` - Estimated width of the PSF
/// * `config` - Deconvolution configuration
///
/// # Returns
///
/// * Tuple containing (deconvolved image, estimated PSF)
pub fn blind_deconvolution_2d(
    image: &Array2<f64>,
    psf_size_h: usize,
    psf_size_w: usize,
    config: &DeconvolutionConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (height, width) = image.dim();

    // Check inputs
    if psf_size_h >= height || psf_size_w >= width {
        return Err(SignalError::ValueError(
            "PSF dimensions must be smaller than image dimensions".to_string(),
        ));
    }

    // Pad image if requested
    let (padded_image, pad_h, pad_w) = if config.pad_signal {
        let pad_h = psf_size_h - 1;
        let pad_w = psf_size_w - 1;
        let pad_height = height + pad_h;
        let pad_width = width + pad_w;

        let mut padded_image = Array2::<f64>::zeros((pad_height, pad_width));

        // Copy original image
        padded_image.slice_mut(s![..height, ..width]).assign(image);

        (padded_image, pad_h / 2, pad_w / 2)
    } else {
        (image.clone(), 0, 0)
    };

    let (pad_height, pad_width) = padded_image.dim();

    // Initialize with a Gaussian PSF
    let mut estimated_psf = create_gaussian_kernel_2d(psf_size_h, psf_size_w);

    // Normalize the PSF
    let psf_sum = estimated_psf.sum();
    if psf_sum > 0.0 {
        estimated_psf /= psf_sum;
    }

    // Initialize padded PSF
    let mut padded_psf = Array2::<f64>::zeros((pad_height, pad_width));
    padded_psf
        .slice_mut(s![..psf_size_h, ..psf_size_w])
        .assign(&estimated_psf);

    // Initialize image with a deconvolution using the initial PSF
    let mut estimated_image = wiener_deconvolution_2d(&padded_image, &padded_psf, 0.1, config)?;

    // Alternating minimization
    for _iter in 0..config.max_iterations {
        // Save previous estimates for convergence check
        let prev_image = estimated_image.clone();
        let prev_psf = estimated_psf.clone();

        // Update image estimate (keeping PSF fixed)
        estimated_image = richardson_lucy_deconvolution_2d(
            &padded_image,
            &padded_psf,
            Some(5), // Just a few iterations per outer iteration
            config,
        )?;

        // Apply constraints to the image
        if config.positivity_constraint {
            estimated_image = estimated_image.mapv(|x| x.max(0.0));
        }

        // Update PSF estimate (keeping image fixed)
        // This is essentially solving the same deconvolution with roles reversed
        let temp_psf =
            richardson_lucy_deconvolution_2d(&padded_image, &estimated_image, Some(5), config)?;

        // Constrain PSF to its expected size
        let start_h = (pad_height - psf_size_h) / 2;
        let start_w = (pad_width - psf_size_w) / 2;
        let mut psf_cropped = temp_psf
            .slice(s![
                start_h..start_h + psf_size_h,
                start_w..start_w + psf_size_w
            ])
            .to_owned();

        // Apply constraints to the PSF
        if config.positivity_constraint {
            psf_cropped = psf_cropped.mapv(|x| x.max(0.0));
        }

        // Normalize PSF
        let psf_cropped_sum = psf_cropped.sum();
        if psf_cropped_sum > 0.0 {
            psf_cropped /= psf_cropped_sum;
        }

        estimated_psf = psf_cropped;

        // Update padded PSF for next iteration
        padded_psf.fill(0.0);
        padded_psf
            .slice_mut(s![..psf_size_h, ..psf_size_w])
            .assign(&estimated_psf);

        // Check for convergence
        if config.convergence_threshold > 0.0 {
            let image_diff = (&estimated_image - &prev_image).mapv(|x| x.abs()).sum()
                / prev_image.mapv(|x| x.abs()).sum();

            let psf_diff = (&estimated_psf - &prev_psf).mapv(|x| x.abs()).sum()
                / prev_psf.mapv(|x| x.abs()).sum();

            if image_diff < config.convergence_threshold && psf_diff < config.convergence_threshold
            {
                break;
            }
        }
    }

    // Extract final image result
    let mut deconvolved = Array2::<f64>::zeros((height, width));
    for i in 0..height {
        for j in 0..width {
            deconvolved[[i, j]] = estimated_image[[i + pad_h, j + pad_w]];
        }
    }

    Ok((deconvolved, estimated_psf))
}

/// Helper function to create a Gaussian kernel for a PSF
fn create_gaussian_kernel(size: usize) -> Array1<f64> {
    let half_size = size as isize / 2;
    let mut kernel = Array1::<f64>::zeros(size);

    let sigma = size as f64 / 6.0; // Standard deviation
    let two_sigma_sq = 2.0 * sigma * sigma;

    for i in 0..size {
        let x = (i as isize - half_size) as f64;
        kernel[i] = (-x * x / two_sigma_sq).exp();
    }

    // Normalize
    let sum = kernel.sum();
    if sum > 0.0 {
        kernel /= sum;
    }

    kernel
}

/// Helper function to create a 2D Gaussian kernel for a PSF
fn create_gaussian_kernel_2d(height: usize, width: usize) -> Array2<f64> {
    let half_h = height as isize / 2;
    let half_w = width as isize / 2;
    let mut kernel = Array2::<f64>::zeros((height, width));

    let sigma_h = height as f64 / 6.0; // Standard deviation for height
    let sigma_w = width as f64 / 6.0; // Standard deviation for width
    let two_sigma_h_sq = 2.0 * sigma_h * sigma_h;
    let two_sigma_w_sq = 2.0 * sigma_w * sigma_w;

    for i in 0..height {
        let y = (i as isize - half_h) as f64;
        for j in 0..width {
            let x = (j as isize - half_w) as f64;
            kernel[[i, j]] = (-x * x / two_sigma_w_sq - y * y / two_sigma_h_sq).exp();
        }
    }

    // Normalize
    let sum = kernel.sum();
    if sum > 0.0 {
        kernel /= sum;
    }

    kernel
}

/// Helper function to apply Gaussian filtering to a 1D signal
fn gaussian_filter_1d(signal: &Array1<f64>, sigma: f64) -> Array1<f64> {
    let _n = signal.len();
    let kernel_size = (6.0 * sigma).ceil() as usize;
    let kernel_size = kernel_size + (1 - kernel_size % 2); // Ensure odd size

    let kernel = create_gaussian_kernel(kernel_size);

    match convolve::convolve(
        signal.as_slice().unwrap(),
        kernel.as_slice().unwrap(),
        "same",
    ) {
        Ok(filtered) => Array1::from_vec(filtered),
        Err(_) => signal.clone(),
    }
}

/// Helper function to apply Gaussian filtering to a 2D image
fn gaussian_filter_2d(image: &Array2<f64>, sigma: f64) -> Array2<f64> {
    let (_height, _width) = image.dim();
    let kernel_size = (6.0 * sigma).ceil() as usize;
    let kernel_size = kernel_size + (1 - kernel_size % 2); // Ensure odd size

    let kernel = create_gaussian_kernel_2d(kernel_size, kernel_size);

    match convolve::convolve2d(image, &kernel, "same") {
        Ok(filtered) => filtered,
        Err(_) => image.clone(),
    }
}

/// Apply Wiener deconvolution to a 3D color image
///
/// # Arguments
///
/// * `image` - The observed color image (3D array with last axis being color channels)
/// * `psf` - The point spread function (2D array)
/// * `noise_level` - The noise-to-signal power ratio
/// * `config` - Deconvolution configuration
///
/// # Returns
///
/// * The deconvolved color image
pub fn wiener_deconvolution_color(
    image: &Array3<f64>,
    psf: &Array2<f64>,
    noise_level: f64,
    config: &DeconvolutionConfig,
) -> SignalResult<Array3<f64>> {
    let (height, width, channels) = image.dim();

    // Process each channel independently
    let mut deconvolved = Array3::zeros((height, width, channels));

    for c in 0..channels {
        let channel = image.slice(s![.., .., c]).to_owned();
        let deconvolved_channel = wiener_deconvolution_2d(&channel, psf, noise_level, config)?;

        // Copy result to output
        for i in 0..height {
            for j in 0..width {
                deconvolved[[i, j, c]] = deconvolved_channel[[i, j]];
            }
        }
    }

    Ok(deconvolved)
}

/// Apply Richardson-Lucy deconvolution to a 3D color image
///
/// # Arguments
///
/// * `image` - The observed color image (3D array with last axis being color channels)
/// * `psf` - The point spread function (2D array)
/// * `iterations` - Number of iterations (or None to use config default)
/// * `config` - Deconvolution configuration
///
/// # Returns
///
/// * The deconvolved color image
pub fn richardson_lucy_deconvolution_color(
    image: &Array3<f64>,
    psf: &Array2<f64>,
    iterations: Option<usize>,
    config: &DeconvolutionConfig,
) -> SignalResult<Array3<f64>> {
    let (height, width, channels) = image.dim();

    // Process each channel independently
    let mut deconvolved = Array3::zeros((height, width, channels));

    for c in 0..channels {
        let channel = image.slice(s![.., .., c]).to_owned();
        let deconvolved_channel =
            richardson_lucy_deconvolution_2d(&channel, psf, iterations, config)?;

        // Copy result to output
        for i in 0..height {
            for j in 0..width {
                deconvolved[[i, j, c]] = deconvolved_channel[[i, j]];
            }
        }
    }

    Ok(deconvolved)
}

/// Automatically estimate the regularization parameter for deconvolution
///
/// # Arguments
///
/// * `signal` - The observed signal
/// * `psf` - The point spread function
/// * `min_param` - Minimum regularization parameter to try
/// * `max_param` - Maximum regularization parameter to try
/// * `num_values` - Number of parameter values to evaluate
///
/// # Returns
///
/// * The optimal regularization parameter based on GCV
pub fn estimate_regularization_param(
    signal: &Array1<f64>,
    psf: &Array1<f64>,
    min_param: f64,
    max_param: f64,
    num_values: usize,
) -> SignalResult<f64> {
    let n = signal.len();

    // Generate logarithmically spaced parameter values
    let log_min = min_param.ln();
    let log_max = max_param.ln();
    let step = (log_max - log_min) / (num_values - 1) as f64;

    let mut param_values = Vec::with_capacity(num_values);
    for i in 0..num_values {
        param_values.push((log_min + i as f64 * step).exp());
    }

    // Set up for FFT-based deconvolution
    let padded_signal = signal.clone();
    let mut padded_psf = Array1::<f64>::zeros(n);

    // Center the PSF
    let start = (n - psf.len()) / 2;
    padded_psf
        .slice_mut(s![start..start + psf.len()])
        .assign(psf);

    // Convert to complex arrays for FFT
    let mut signal_complex = vec![Complex::new(0.0, 0.0); n];
    let mut psf_complex = vec![Complex::new(0.0, 0.0); n];

    for i in 0..n {
        signal_complex[i] = Complex::new(padded_signal[i], 0.0);
        psf_complex[i] = Complex::new(padded_psf[i], 0.0);
    }

    // Create FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // Forward FFT of signal and PSF
    fft.process(&mut signal_complex);
    fft.process(&mut psf_complex);

    // Evaluate GCV function for each parameter value
    let mut best_param = min_param;
    let mut min_gcv = f64::INFINITY;

    for &param in &param_values {
        let mut result_complex = vec![Complex::new(0.0, 0.0); n];
        let mut filter_diag = vec![0.0; n];

        // Apply Tikhonov regularized filter
        for i in 0..n {
            let h = psf_complex[i];
            let h_conj = h.conj();
            let h_abs_sq = h.norm_sqr();

            // Filter diagonal element (for effective degrees of freedom)
            let filter_elem = h_abs_sq / (h_abs_sq + param);
            filter_diag[i] = filter_elem;

            // Apply filter to signal
            let denom = h_abs_sq + param;
            if denom > 1e-10 {
                result_complex[i] = signal_complex[i] * h_conj / denom;
            } else {
                result_complex[i] = Complex::new(0.0, 0.0);
            }
        }

        // Inverse FFT to get the solution
        ifft.process(&mut result_complex);

        // Scale and convert back to real
        let scale = 1.0 / (n as f64);
        let mut solution = Array1::<f64>::zeros(n);

        for i in 0..n {
            solution[i] = result_complex[i].re * scale;
        }

        // Compute residual sum of squares (RSS)
        let predicted = convolve::convolve(
            solution.as_slice().unwrap(),
            padded_psf.as_slice().unwrap(),
            "same",
        )?;
        let mut rss = 0.0;

        for i in 0..n {
            let diff = predicted[i] - padded_signal[i];
            rss += diff * diff;
        }

        // Compute effective degrees of freedom
        let mut df = 0.0;
        for &f in &filter_diag {
            df += f;
        }

        // Compute GCV score
        let gcv = n as f64 * rss / (n as f64 - df).powi(2);

        if gcv < min_gcv {
            min_gcv = gcv;
            best_param = param;
        }
    }

    Ok(best_param)
}

/// Apply deconvolution to a 1D signal with optimal parameter selection
///
/// # Arguments
///
/// * `signal` - The observed signal
/// * `psf` - The point spread function
/// * `method` - Deconvolution method to use
/// * `config` - Deconvolution configuration
///
/// # Returns
///
/// * The deconvolved signal
pub fn optimal_deconvolution_1d(
    signal: &Array1<f64>,
    psf: &Array1<f64>,
    method: DeconvolutionMethod,
    config: &DeconvolutionConfig,
) -> SignalResult<Array1<f64>> {
    // Estimate optimal regularization parameter if requested
    let reg_param = if config.auto_regularization {
        estimate_regularization_param(signal, psf, 1e-6, 1.0, 20)?
    } else {
        config.reg_param
    };

    // Apply the selected deconvolution method
    match method {
        DeconvolutionMethod::Wiener => wiener_deconvolution_1d(signal, psf, reg_param, config),
        DeconvolutionMethod::Tikhonov => {
            tikhonov_deconvolution_1d(signal, psf, Some(reg_param), config)
        }
        DeconvolutionMethod::RichardsonLucy => {
            richardson_lucy_deconvolution_1d(signal, psf, None, config)
        }
        DeconvolutionMethod::CLEAN => clean_deconvolution_1d(signal, psf, 0.1, 0.01, config),
        DeconvolutionMethod::MaximumEntropy => mem_deconvolution_1d(signal, psf, reg_param, config),
        DeconvolutionMethod::Blind => {
            let (deconvolved, _) = blind_deconvolution_1d(signal, psf.len(), config)?;
            Ok(deconvolved)
        }
    }
}

/// Deconvolution methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeconvolutionMethod {
    /// Wiener deconvolution (frequency domain)
    Wiener,
    /// Tikhonov regularized deconvolution
    Tikhonov,
    /// Richardson-Lucy iterative deconvolution
    RichardsonLucy,
    /// CLEAN algorithm (iterative component subtraction)
    CLEAN,
    /// Maximum Entropy Method
    MaximumEntropy,
    /// Blind deconvolution (estimates PSF)
    Blind,
}
