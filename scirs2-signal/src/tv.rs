// Total Variation denoising module
//
// This module implements Total Variation (TV) denoising techniques for signal and image
// processing. TV denoising preserves sharp edges while removing noise by minimizing
// the total variation of the signal or image.
//
// The implementation includes:
// - 1D Total Variation denoising (Rudin-Osher-Fatemi model)
// - 2D Total Variation denoising for images
// - Anisotropic and isotropic TV variants
// - Accelerated optimization algorithms
//
// # Example
// ```
// use ndarray::Array1;
// use scirs2_signal::tv::{tv_denoise_1d, TvConfig};
// use rand::prelude::*;
//
// // Create a test signal with noise
// let n = 500;
// let mut clean_signal = Array1::zeros(n);
// for i in 100..400 {
//     clean_signal[i] = 1.0;
// }
//
// // Add noise
// let mut rng = rand::rng();
// let mut noisy_signal = clean_signal.clone();
// for i in 0..n {
//     noisy_signal[i] += 0.2 * rng.gen_range(-1.0..1.0);
// }
//
// // Apply Total Variation denoising
// let config = TvConfig::default();
// let denoised = tv_denoise_1d(&noisy_signal, 0.5, &config).unwrap();
// ```

use crate::error::{SignalError, SignalResult};
use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;

#[allow(unused_imports)]
/// Variant of Total Variation regularization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TvVariant {
    /// Anisotropic TV (uses L1 norm of gradients)
    Anisotropic,

    /// Isotropic TV (uses L2 norm of gradients)
    Isotropic,
}

/// Configuration for Total Variation denoising
#[derive(Debug, Clone)]
pub struct TvConfig {
    /// TV variant (anisotropic or isotropic)
    pub variant: TvVariant,

    /// Maximum number of iterations
    pub max_iterations: usize,

    /// Convergence tolerance
    pub tol: f64,

    /// Whether to use accelerated algorithms
    pub accelerated: bool,

    /// Initial step size for gradient methods
    pub initial_step: f64,

    /// Whether to adapt step size
    pub adaptive_step: bool,
}

impl Default for TvConfig {
    fn default() -> Self {
        Self {
            variant: TvVariant::Isotropic,
            max_iterations: 200,
            tol: 1e-4,
            accelerated: true,
            initial_step: 1.0,
            adaptive_step: true,
        }
    }
}

/// Applies Total Variation denoising to a 1D signal.
///
/// TV denoising preserves sharp edges while removing noise by minimizing
/// the total variation (sum of absolute differences between adjacent elements)
/// of the signal.
///
/// # Arguments
/// * `signal` - Noisy input signal
/// * `weight` - Regularization weight (higher values = more smoothing)
/// * `config` - TV configuration parameters
///
/// # Returns
/// * The denoised signal
///
/// # Example
/// ```
/// use ndarray::Array1;
/// use scirs2_signal::tv::{tv_denoise_1d, TvConfig};
///
/// let signal = Array1::from_vec(vec![1.2, 2.3, 3.1, 2.2, 1.3, 0.2, -0.3, -1.1]);
/// let config = TvConfig::default();
/// let denoised = tv_denoise_1d(&signal, 0.5, &config).unwrap();
/// ```
#[allow(dead_code)]
pub fn tv_denoise_1d(
    signal: &Array1<f64>,
    weight: f64,
    config: &TvConfig,
) -> SignalResult<Array1<f64>> {
    // Validate parameters
    if weight <= 0.0 {
        return Err(SignalError::ValueError(
            "Regularization weight must be positive".to_string(),
        ));
    }

    // Choose algorithm based on configuration
    if config.accelerated {
        tv_denoise_1d_fista(signal, weight, config)
    } else {
        tv_denoise_1d_chambolle(signal, weight, config)
    }
}

/// Chambolle's algorithm for 1D Total Variation denoising
#[allow(dead_code)]
fn tv_denoise_1d_chambolle(
    signal: &Array1<f64>,
    weight: f64,
    config: &TvConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Dual variable (related to gradient)
    let mut p = Array1::<f64>::zeros(n - 1);

    // Result initialization
    let mut denoised = signal.clone();

    // Temporary arrays
    let mut div_p = Array1::zeros(n);
    let mut grad_div_p = Array1::zeros(n - 1);

    // Step size
    let mut step = config.initial_step;

    // Main iteration loop
    for iter in 0..config.max_iterations {
        // Store previous result for convergence check
        let prev_denoised = denoised.clone();

        // Compute divergence of p (adjoint of gradient)
        // div_p[0] = -p[0]
        div_p[0] = -p[0];

        // div_p[i] = p[i-1] - p[i] for i=1..n-1
        for i in 1..n - 1 {
            div_p[i] = p[i - 1] - p[i];
        }

        // div_p[n-1] = p[n-2]
        div_p[n - 1] = p[n - 2];

        // Compute denoised signal
        // denoised = signal + weight * div_p
        for i in 0..n {
            denoised[i] = signal[i] + weight * div_p[i];
        }

        // Compute gradient of div_p
        for i in 0..n - 1 {
            grad_div_p[i] = denoised[i + 1] - denoised[i];
        }

        // Update dual variable p
        for i in 0..n - 1 {
            p[i] = (p[i] + step * grad_div_p[i]) / (1.0 + step * grad_div_p[i].abs());
        }

        // Check convergence
        let mut change = 0.0;
        let mut norm = 0.0;
        for i in 0..n {
            let diff = denoised[i] - prev_denoised[i];
            change += diff * diff;
            norm += prev_denoised[i] * prev_denoised[i];
        }

        // Relative change in solution
        let relative_change = (change / norm.max(1e-10)).sqrt();
        if relative_change < config.tol {
            return Ok(denoised);
        }

        // Adapt step size if needed
        if config.adaptive_step && iter % 10 == 0 {
            step *= 1.2;
        }
    }

    // Return result after max iterations
    Ok(denoised)
}

/// Fast Iterative Shrinkage-Thresholding Algorithm for 1D Total Variation denoising
#[allow(dead_code)]
fn tv_denoise_1d_fista(
    signal: &Array1<f64>,
    weight: f64,
    config: &TvConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Initialize variables
    let mut x = signal.clone(); // Current solution
    let mut y = x.clone(); // Momentum variable
    let mut x_prev = x.clone(); // Previous solution

    // Compute Lipschitz constant for gradient
    // For 1D TV, it's approximately 8
    let l = 8.0;
    let step_size = 1.0 / l;

    // Initialize momentum term
    let mut t = 1.0;

    // Main iteration loop
    for _iter in 0..config.max_iterations {
        // Store previous solution
        x_prev.assign(&x);

        // Forward-backward step on y
        let grad = compute_tv_gradient_1d(&y, signal, weight);

        // Gradient step
        for i in 0..n {
            x[i] = y[i] - step_size * grad[i];
        }

        // Compute new momentum factor
        let t_prev = t;
        t = (1.0 + f64::sqrt(1.0 + 4.0 * t_prev * t_prev)) / 2.0;

        // Update momentum variable
        let momentum = (t_prev - 1.0) / t;
        for i in 0..n {
            y[i] = x[i] + momentum * (x[i] - x_prev[i]);
        }

        // Check convergence
        let mut change = 0.0;
        let mut norm = 0.0;
        for i in 0..n {
            let diff = x[i] - x_prev[i];
            change += diff * diff;
            norm += x_prev[i] * x_prev[i];
        }

        // Relative change in solution
        let relative_change = (change / norm.max(1e-10)).sqrt();
        if relative_change < config.tol {
            return Ok(x);
        }
    }

    // Return result after max iterations
    Ok(x)
}

/// Computes the gradient of the Total Variation energy for 1D signals
#[allow(dead_code)]
fn compute_tv_gradient_1d(
    signal: &Array1<f64>,
    original: &Array1<f64>,
    weight: f64,
) -> Array1<f64> {
    let n = signal.len();
    let mut gradient = Array1::zeros(n);

    // Data fidelity term gradient: 2 * (signal - original)
    for i in 0..n {
        gradient[i] = 2.0 * (signal[i] - original[i]);
    }

    // TV regularization gradient
    // For anisotropic TV, it's the divergence of sign(gradient)
    // For isotropic TV in 1D, it's the same as anisotropic

    // Forward differences at the first point
    if n > 1 {
        let diff = signal[1] - signal[0];
        if diff.abs() > 1e-10 {
            gradient[0] += weight * diff.signum();
            gradient[1] -= weight * diff.signum();
        }
    }

    // Central differences for interior points
    for i in 1..n - 1 {
        let diff1 = signal[i] - signal[i - 1];
        let diff2 = signal[i + 1] - signal[i];

        if diff1.abs() > 1e-10 {
            gradient[i] += weight * diff1.signum();
            gradient[i - 1] -= weight * diff1.signum();
        }

        if diff2.abs() > 1e-10 {
            gradient[i] += weight * diff2.signum();
            gradient[i + 1] -= weight * diff2.signum();
        }
    }

    // Backward differences at the last point
    if n > 1 {
        let diff = signal[n - 1] - signal[n - 2];
        if diff.abs() > 1e-10 {
            gradient[n - 1] += weight * diff.signum();
            gradient[n - 2] -= weight * diff.signum();
        }
    }

    gradient
}

/// Applies Total Variation denoising to a 2D image.
///
/// TV denoising preserves sharp edges and boundaries while removing noise.
/// The algorithm minimizes the total variation of the image, which is the
/// L1 norm (anisotropic) or L2 norm (isotropic) of the gradient.
///
/// # Arguments
/// * `image` - Noisy input image
/// * `weight` - Regularization weight (higher values = more smoothing)
/// * `config` - TV configuration parameters
///
/// # Returns
/// * The denoised image
///
/// # Example
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::tv::{tv_denoise_2d, TvConfig};
///
/// let image = Array2::from_shape_fn((10, 10), |(i, j)| (i + j) as f64 / 20.0);
/// let config = TvConfig::default();
/// let denoised = tv_denoise_2d(&image, 0.5, &config).unwrap();
/// ```
#[allow(dead_code)]
pub fn tv_denoise_2d(
    image: &Array2<f64>,
    weight: f64,
    config: &TvConfig,
) -> SignalResult<Array2<f64>> {
    // Validate parameters
    if weight <= 0.0 {
        return Err(SignalError::ValueError(
            "Regularization weight must be positive".to_string(),
        ));
    }

    // Choose algorithm based on configuration
    if config.accelerated {
        tv_denoise_2d_fista(image, weight, config)
    } else {
        tv_denoise_2d_chambolle(image, weight, config)
    }
}

/// Chambolle's algorithm for 2D Total Variation denoising
#[allow(dead_code)]
fn tv_denoise_2d_chambolle(
    image: &Array2<f64>,
    weight: f64,
    config: &TvConfig,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();

    // Dual variables (related to gradients)
    let mut p1 = Array2::<f64>::zeros((height, width)); // Horizontal component
    let mut p2 = Array2::<f64>::zeros((height, width)); // Vertical component

    // Result initialization
    let mut denoised = image.clone();

    // Step size
    let mut step = config.initial_step;

    // Main iteration loop
    for iter in 0..config.max_iterations {
        // Store previous result for convergence check
        let prev_denoised = denoised.clone();

        // Compute divergence of p (adjoint of gradient)
        let mut div_p = Array2::<f64>::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                // Compute x-component of divergence
                if j < width - 1 {
                    div_p[[i, j]] -= p1[[i, j]];
                }
                if j > 0 {
                    div_p[[i, j]] += p1[[i, j - 1]];
                }

                // Compute y-component of divergence
                if i < height - 1 {
                    div_p[[i, j]] -= p2[[i, j]];
                }
                if i > 0 {
                    div_p[[i, j]] += p2[[i - 1, j]];
                }
            }
        }

        // Update denoised image: u = f + lambda * div(p)
        for i in 0..height {
            for j in 0..width {
                denoised[[i, j]] = image[[i, j]] + (weight * div_p[[i, j]]);
            }
        }

        // Compute gradient of divergence
        let mut grad_div_p1 = Array2::<f64>::zeros((height, width));
        let mut grad_div_p2 = Array2::<f64>::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                // X-gradient
                if j < width - 1 {
                    grad_div_p1[[i, j]] = denoised[[i, j + 1]] - denoised[[i, j]];
                }

                // Y-gradient
                if i < height - 1 {
                    grad_div_p2[[i, j]] = denoised[[i + 1, j]] - denoised[[i, j]];
                }
            }
        }

        // Update dual variables based on TV variant
        match config.variant {
            TvVariant::Anisotropic => {
                // Anisotropic TV (L1 norm of gradients)
                for i in 0..height {
                    for j in 0..width {
                        // Update p1 (x-component)
                        p1[[i, j]] = (p1[[i, j]] + step * grad_div_p1[[i, j]])
                            / (1.0 + step * grad_div_p1[[i, j]].abs());

                        // Update p2 (y-component)
                        p2[[i, j]] = (p2[[i, j]] + step * grad_div_p2[[i, j]])
                            / (1.0 + step * grad_div_p2[[i, j]].abs());
                    }
                }
            }
            TvVariant::Isotropic => {
                // Isotropic TV (L2 norm of gradients)
                for i in 0..height {
                    for j in 0..width {
                        // Update p1 and p2 simultaneously
                        let dp1 = step * grad_div_p1[[i, j]];
                        let dp2 = step * grad_div_p2[[i, j]];

                        let new_p1 = p1[[i, j]] + dp1;
                        let new_p2 = p2[[i, j]] + dp2;

                        let sum: f64 = new_p1 * new_p1 + new_p2 * new_p2;
                        let norm = f64::max(sum.sqrt(), 1.0);

                        p1[[i, j]] = new_p1 / norm;
                        p2[[i, j]] = new_p2 / norm;
                    }
                }
            }
        }

        // Check convergence
        let mut change = 0.0;
        let mut norm = 0.0;
        for i in 0..height {
            for j in 0..width {
                let diff = denoised[[i, j]] - prev_denoised[[i, j]];
                change += diff * diff;
                norm += prev_denoised[[i, j]] * prev_denoised[[i, j]];
            }
        }

        // Relative change in solution
        let relative_change = (change / norm.max(1e-10)).sqrt();
        if relative_change < config.tol {
            return Ok(denoised);
        }

        // Adapt step size if needed
        if config.adaptive_step && iter % 10 == 0 {
            step *= 1.2;
        }
    }

    // Return result after max iterations
    Ok(denoised)
}

/// Fast Iterative Shrinkage-Thresholding Algorithm for 2D Total Variation denoising
#[allow(dead_code)]
fn tv_denoise_2d_fista(
    image: &Array2<f64>,
    weight: f64,
    config: &TvConfig,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();

    // Initialize variables
    let mut x = image.clone(); // Current solution
    let mut y = x.clone(); // Momentum variable
    let mut x_prev = x.clone(); // Previous solution

    // Compute Lipschitz constant for gradient
    // For 2D TV, it's approximately 8
    let l = 8.0;
    let step_size = 1.0 / l;

    // Initialize momentum term
    let mut t = 1.0;

    // Main iteration loop
    for _iter in 0..config.max_iterations {
        // Store previous solution
        x_prev.assign(&x);

        // Forward-backward step on y
        let grad = compute_tv_gradient_2d(&y, image, weight, config.variant);

        // Gradient step
        for i in 0..height {
            for j in 0..width {
                x[[i, j]] = y[[i, j]] - step_size * grad[[i, j]];
            }
        }

        // Compute new momentum factor
        let t_prev = t;
        t = (1.0 + f64::sqrt(1.0 + 4.0 * t_prev * t_prev)) / 2.0;

        // Update momentum variable
        let momentum = (t_prev - 1.0) / t;
        for i in 0..height {
            for j in 0..width {
                y[[i, j]] = x[[i, j]] + momentum * (x[[i, j]] - x_prev[[i, j]]);
            }
        }

        // Check convergence
        let mut change = 0.0;
        let mut norm = 0.0;
        for i in 0..height {
            for j in 0..width {
                let diff = x[[i, j]] - x_prev[[i, j]];
                change += diff * diff;
                norm += x_prev[[i, j]] * x_prev[[i, j]];
            }
        }

        // Relative change in solution
        let relative_change = (change / norm.max(1e-10)).sqrt();
        if relative_change < config.tol {
            return Ok(x);
        }
    }

    // Return result after max iterations
    Ok(x)
}

/// Computes the gradient of the Total Variation energy for 2D images
#[allow(dead_code)]
fn compute_tv_gradient_2d(
    image: &Array2<f64>,
    original: &Array2<f64>,
    weight: f64,
    variant: TvVariant,
) -> Array2<f64> {
    let (height, width) = image.dim();
    let mut gradient = Array2::<f64>::zeros((height, width));

    // Data fidelity term gradient: 2 * (image - original)
    for i in 0..height {
        for j in 0..width {
            gradient[[i, j]] = 2.0 * (image[[i, j]] - original[[i, j]]);
        }
    }

    // TV regularization gradient
    // For each pixel, compute the contribution from its neighbors
    for i in 0..height {
        for j in 0..width {
            // Compute horizontal and vertical differences
            let mut dx_forward = 0.0;
            let mut dy_forward = 0.0;
            let mut dx_backward = 0.0;
            let mut dy_backward = 0.0;

            // Forward differences
            if j < width - 1 {
                dx_forward = image[[i, j + 1]] - image[[i, j]];
            }
            if i < height - 1 {
                dy_forward = image[[i + 1, j]] - image[[i, j]];
            }

            // Backward differences
            if j > 0 {
                dx_backward = image[[i, j]] - image[[i, j - 1]];
            }
            if i > 0 {
                dy_backward = image[[i, j]] - image[[i - 1, j]];
            }

            // Apply the appropriate TV gradient based on variant
            match variant {
                TvVariant::Anisotropic => {
                    // Anisotropic TV (L1 norm of gradients)
                    if dx_forward.abs() > 1e-10 {
                        gradient[[i, j]] += weight * dx_forward.signum();
                    }
                    if dy_forward.abs() > 1e-10 {
                        gradient[[i, j]] += weight * dy_forward.signum();
                    }
                    if dx_backward.abs() > 1e-10 {
                        gradient[[i, j]] -= weight * dx_backward.signum();
                    }
                    if dy_backward.abs() > 1e-10 {
                        gradient[[i, j]] -= weight * dy_backward.signum();
                    }
                }
                TvVariant::Isotropic => {
                    // Isotropic TV (L2 norm of gradients)
                    // Forward gradient norm
                    let forward_norm = (dx_forward * dx_forward + dy_forward * dy_forward)
                        .sqrt()
                        .max(1e-10);

                    // Backward gradient norm
                    let backward_norm = (dx_backward * dx_backward + dy_backward * dy_backward)
                        .sqrt()
                        .max(1e-10);

                    // Add weighted contributions
                    gradient[[i, j]] += weight
                        * (dx_forward / forward_norm + dy_forward / forward_norm
                            - dx_backward / backward_norm
                            - dy_backward / backward_norm);
                }
            }
        }
    }

    gradient
}

/// Applies Total Variation denoising to a color image.
///
/// This function handles color images (3D arrays with the last axis representing
/// color channels) by either applying TV denoising independently to each channel
/// or using a vectorial TV approach that preserves color coherence.
///
/// # Arguments
/// * `image` - Noisy color image (3D array with last axis being color channels)
/// * `weight` - Regularization weight
/// * `config` - TV configuration parameters
/// * `vectorial` - Whether to use vectorial TV (preserves color edges better)
///
/// # Returns
/// * The denoised color image
#[allow(dead_code)]
pub fn tv_denoise_color(
    image: &Array3<f64>,
    weight: f64,
    config: &TvConfig,
    vectorial: bool,
) -> SignalResult<Array3<f64>> {
    let (height, width, channels) = image.dim();

    if vectorial {
        // Vectorial TV denoising (processes all channels together)
        tv_denoise_color_vectorial(image, weight, config)
    } else {
        // Channel-by-channel TV denoising
        let mut result = Array3::zeros((height, width, channels));

        for c in 0..channels {
            // Extract channel
            let channel = image.index_axis(Axis(2), c);
            let channel_owned = channel.to_owned();

            // Apply TV denoising to the channel
            let denoised_channel = tv_denoise_2d(&channel_owned, weight, config)?;

            // Store result
            for i in 0..height {
                for j in 0..width {
                    result[[i, j, c]] = denoised_channel[[i, j]];
                }
            }
        }

        Ok(result)
    }
}

/// Applies vectorial Total Variation denoising to a color image.
///
/// Vectorial TV treats gradients as vectors across all channels, preserving
/// color edge coherence better than channel-by-channel processing.
///
/// # Arguments
/// * `image` - Noisy color image
/// * `weight` - Regularization weight
/// * `config` - TV configuration parameters
///
/// # Returns
/// * The denoised color image
#[allow(dead_code)]
fn tv_denoise_color_vectorial(
    image: &Array3<f64>,
    weight: f64,
    config: &TvConfig,
) -> SignalResult<Array3<f64>> {
    let (height, width, channels) = image.dim();

    // Initialize result array
    let mut result = image.clone();

    // Previous solution for convergence check
    let mut prev_result = result.clone();

    // Dual variables
    let mut p = Array3::<f64>::zeros((height, width, 2 * channels));

    // Step size
    let mut step = config.initial_step;

    // Main iteration loop
    for iter in 0..config.max_iterations {
        // Store previous result
        prev_result.assign(&result);

        // Compute divergence of p
        let mut div_p = Array3::<f64>::zeros((height, width, channels));

        for c in 0..channels {
            // X-component index in the dual variable
            let px_idx = c;

            // Y-component index in the dual variable
            let py_idx = c + channels;

            for i in 0..height {
                for j in 0..width {
                    // X-component divergence
                    if j < width - 1 {
                        div_p[[i, j, c]] -= p[[i, j, px_idx]];
                    }
                    if j > 0 {
                        div_p[[i, j, c]] += p[[i, j - 1, px_idx]];
                    }

                    // Y-component divergence
                    if i < height - 1 {
                        div_p[[i, j, c]] -= p[[i, j, py_idx]];
                    }
                    if i > 0 {
                        div_p[[i, j, c]] += p[[i - 1, j, py_idx]];
                    }
                }
            }
        }

        // Update result
        for i in 0..height {
            for j in 0..width {
                for c in 0..channels {
                    result[[i, j, c]] = image[[i, j, c]] + weight * div_p[[i, j, c]];
                }
            }
        }

        // Compute gradient of u + div_p
        let mut grad = Array3::zeros((height, width, 2 * channels));

        for c in 0..channels {
            // X-component index
            let px_idx = c;

            // Y-component index
            let py_idx = c + channels;

            for i in 0..height {
                for j in 0..width {
                    // X-gradient
                    if j < width - 1 {
                        grad[[i, j, px_idx]] = result[[i, j + 1, c]] - result[[i, j, c]];
                    }

                    // Y-gradient
                    if i < height - 1 {
                        grad[[i, j, py_idx]] = result[[i + 1, j, c]] - result[[i, j, c]];
                    }
                }
            }
        }

        // Update dual variables based on TV variant
        for i in 0..height {
            for j in 0..width {
                match config.variant {
                    TvVariant::Anisotropic => {
                        // Update each component independently
                        for pc in 0..(2 * channels) {
                            p[[i, j, pc]] = (p[[i, j, pc]] + step * grad[[i, j, pc]])
                                / (1.0 + step * grad[[i, j, pc]].abs());
                        }
                    }
                    TvVariant::Isotropic => {
                        // Compute the vectorial norm
                        let mut norm_squared = 0.0;
                        for pc in 0..(2 * channels) {
                            let new_p = p[[i, j, pc]] + step * grad[[i, j, pc]];
                            norm_squared += new_p * new_p;
                        }

                        let norm = f64::max(f64::sqrt(norm_squared), 1.0);

                        // Update with normalization
                        for pc in 0..(2 * channels) {
                            p[[i, j, pc]] = (p[[i, j, pc]] + step * grad[[i, j, pc]]) / norm;
                        }
                    }
                }
            }
        }

        // Check convergence
        let mut change = 0.0;
        let mut norm = 0.0;
        for i in 0..height {
            for j in 0..width {
                for c in 0..channels {
                    let diff = result[[i, j, c]] - prev_result[[i, j, c]];
                    change += diff * diff;
                    norm += prev_result[[i, j, c]] * prev_result[[i, j, c]];
                }
            }
        }

        // Relative change in solution
        let relative_change = (change / norm.max(1e-10)).sqrt();
        if relative_change < config.tol {
            return Ok(result);
        }

        // Adapt step size if needed
        if config.adaptive_step && iter % 10 == 0 {
            step *= 1.2;
        }
    }

    // Return result after max iterations
    Ok(result)
}

/// Total Variation denoising with Bregman iterations for improved detail preservation.
///
/// Bregman iterations help recover lost details in TV denoising by adding back
/// the residual signal in controlled steps.
///
/// # Arguments
/// * `signal` - Noisy input signal
/// * `weight` - Initial regularization weight
/// * `num_iter` - Number of Bregman iterations
/// * `config` - TV configuration parameters
///
/// # Returns
/// * The denoised signal
#[allow(dead_code)]
pub fn tv_bregman_1d(
    signal: &Array1<f64>,
    weight: f64,
    num_iter: usize,
    config: &TvConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Initialize variables
    let u = signal.clone();
    let mut v = Array1::zeros(n);

    // Main iteration loop
    for _ in 0..num_iter {
        // Solve TV subproblem with current residual
        v = tv_denoise_1d(&(u.clone() + &v), weight, config)?;

        // Update residual
        for i in 0..n {
            v[i] = u[i] - v[i] + v[i];
        }
    }

    Ok(v)
}

/// Total Variation denoising with Bregman iterations for 2D images.
///
/// # Arguments
/// * `image` - Noisy input image
/// * `weight` - Initial regularization weight
/// * `num_iter` - Number of Bregman iterations
/// * `config` - TV configuration parameters
///
/// # Returns
/// * The denoised image
#[allow(dead_code)]
pub fn tv_bregman_2d(
    image: &Array2<f64>,
    weight: f64,
    num_iter: usize,
    config: &TvConfig,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();

    // Initialize variables
    let u = image.clone();
    let mut v = Array2::<f64>::zeros((height, width));

    // Main iteration loop
    for _ in 0..num_iter {
        // Solve TV subproblem with current residual
        v = tv_denoise_2d(&(u.clone() + &v), weight, config)?;

        // Update residual
        for i in 0..height {
            for j in 0..width {
                v[[i, j]] = u[[i, j]] - v[[i, j]] + v[[i, j]];
            }
        }
    }

    Ok(v)
}

/// Implements sparse regularization with Total Variation for inpainting.
///
/// This function can restore missing pixels in an image by combining
/// data fidelity on known pixels with TV regularization.
///
/// # Arguments
/// * `image` - Image with missing data (NaN values indicate missing pixels)
/// * `weight` - Regularization weight
/// * `config` - TV configuration parameters
///
/// # Returns
/// * The inpainted image
#[allow(dead_code)]
pub fn tv_inpaint(
    image: &Array2<f64>,
    weight: f64,
    config: &TvConfig,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();

    // Create a mask of known pixels (1 = known, 0 = unknown)
    let mut mask = Array2::<f64>::zeros((height, width));
    for i in 0..height {
        for j in 0..width {
            if !image[[i, j]].is_nan() {
                mask[[i, j]] = 1.0;
            }
        }
    }

    // Create a copy of the image with zeros for missing values
    let mut clean_image = image.clone();
    for i in 0..height {
        for j in 0..width {
            if clean_image[[i, j]].is_nan() {
                clean_image[[i, j]] = 0.0;
            }
        }
    }

    // Initialize solution with known pixels and simple interpolation for unknown
    let mut result = clean_image.clone();

    // Initial filling of missing values with local average
    for _ in 0..3 {
        // A few iterations of simple diffusion
        let prev = result.clone();
        for i in 0..height {
            for j in 0..width {
                if mask[[i, j]] < 0.5 {
                    // Unknown pixel
                    let mut sum = 0.0;
                    let mut count = 0;

                    // Check neighbors
                    for di in -1..=1 {
                        for dj in -1..=1 {
                            let ni = i as isize + di;
                            let nj = j as isize + dj;

                            if ni >= 0 && ni < height as isize && nj >= 0 && nj < width as isize {
                                sum += prev[[ni as usize, nj as usize]];
                                count += 1;
                            }
                        }
                    }

                    if count > 0 {
                        result[[i, j]] = sum / count as f64;
                    }
                }
            }
        }
    }

    // Apply TV optimization with mask
    let mut prev_result = result.clone();

    // Dual variables
    let mut p1 = Array2::<f64>::zeros((height, width));
    let mut p2 = Array2::<f64>::zeros((height, width));

    // Step size
    let mut step = config.initial_step;

    // Main iteration loop
    for iter in 0..config.max_iterations {
        // Store previous result
        prev_result.assign(&result);

        // Compute divergence of p
        let mut div_p = Array2::<f64>::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                // X-component of divergence
                if j < width - 1 {
                    div_p[[i, j]] -= p1[[i, j]];
                }
                if j > 0 {
                    div_p[[i, j]] += p1[[i, j - 1]];
                }

                // Y-component of divergence
                if i < height - 1 {
                    div_p[[i, j]] -= p2[[i, j]];
                }
                if i > 0 {
                    div_p[[i, j]] += p2[[i - 1, j]];
                }
            }
        }

        // Update result with masked data fidelity
        for i in 0..height {
            for j in 0..width {
                if mask[[i, j]] > 0.5 {
                    // Known pixel: strong data fidelity
                    result[[i, j]] = clean_image[[i, j]];
                } else {
                    // Unknown pixel: only TV regularization
                    result[[i, j]] += weight * div_p[[i, j]];
                }
            }
        }

        // Compute gradient
        let mut grad1 = Array2::<f64>::zeros((height, width));
        let mut grad2 = Array2::<f64>::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                // X-gradient
                if j < width - 1 {
                    grad1[[i, j]] = result[[i, j + 1]] - result[[i, j]];
                }

                // Y-gradient
                if i < height - 1 {
                    grad2[[i, j]] = result[[i + 1, j]] - result[[i, j]];
                }
            }
        }

        // Update dual variables based on TV variant
        match config.variant {
            TvVariant::Anisotropic => {
                // Anisotropic TV
                for i in 0..height {
                    for j in 0..width {
                        p1[[i, j]] = (p1[[i, j]] + step * grad1[[i, j]])
                            / (1.0 + step * grad1[[i, j]].abs());
                        p2[[i, j]] = (p2[[i, j]] + step * grad2[[i, j]])
                            / (1.0 + step * grad2[[i, j]].abs());
                    }
                }
            }
            TvVariant::Isotropic => {
                // Isotropic TV
                for i in 0..height {
                    for j in 0..width {
                        let new_p1 = p1[[i, j]] + step * grad1[[i, j]];
                        let new_p2 = p2[[i, j]] + step * grad2[[i, j]];

                        let sum: f64 = new_p1 * new_p1 + new_p2 * new_p2;
                        let norm = f64::max(sum.sqrt(), 1.0);

                        p1[[i, j]] = new_p1 / norm;
                        p2[[i, j]] = new_p2 / norm;
                    }
                }
            }
        }

        // Check convergence
        let mut change = 0.0;
        let mut norm = 0.0;

        for i in 0..height {
            for j in 0..width {
                if mask[[i, j]] < 0.5 {
                    // Only check convergence on unknown pixels
                    let diff = result[[i, j]] - prev_result[[i, j]];
                    change += diff * diff;
                    norm += prev_result[[i, j]] * prev_result[[i, j]].max(1e-10);
                }
            }
        }

        // Relative change in solution
        let relative_change = (change / norm.max(1e-10)).sqrt();
        if relative_change < config.tol {
            return Ok(result);
        }

        // Adapt step size if needed
        if config.adaptive_step && iter % 10 == 0 {
            step *= 1.2;
        }
    }

    // Return result after max iterations
    Ok(result)
}
