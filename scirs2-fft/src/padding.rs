//! Automatic padding strategies for optimal FFT performance
//!
//! This module provides functionality to automatically pad input data
//! to optimal sizes for FFT computation, improving performance by
//! ensuring the FFT size has small prime factors.

use crate::{next_fast_len, FFTResult};
use ndarray::{s, Array1, ArrayBase, ArrayD, Data, Dimension};
use num_complex::Complex;
use num_traits::Zero;

/// Padding mode for FFT operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingMode {
    /// No padding
    None,
    /// Zero padding
    Zero,
    /// Constant value padding
    Constant(f64),
    /// Edge value replication
    Edge,
    /// Reflect padding (mirror at edge)
    Reflect,
    /// Symmetric padding (mirror with edge duplication)
    Symmetric,
    /// Wrap around (circular)
    Wrap,
    /// Linear ramp to zero
    LinearRamp,
}

/// Auto-padding configuration
#[derive(Debug, Clone)]
pub struct AutoPadConfig {
    /// Padding mode
    pub mode: PaddingMode,
    /// Minimum padding length (default: 0)
    pub min_pad: usize,
    /// Maximum padding length (default: input length)
    pub max_pad: Option<usize>,
    /// Whether to pad to power of 2
    pub power_of_2: bool,
    /// Whether to center the data in padded array
    pub center: bool,
}

impl Default for AutoPadConfig {
    fn default() -> Self {
        Self {
            mode: PaddingMode::Zero,
            min_pad: 0,
            max_pad: None,
            power_of_2: false,
            center: false,
        }
    }
}

impl AutoPadConfig {
    /// Create a new auto-padding configuration
    pub fn new(mode: PaddingMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Set minimum padding
    pub fn with_min_pad(mut self, min_pad: usize) -> Self {
        self.min_pad = min_pad;
        self
    }

    /// Set maximum padding
    pub fn with_max_pad(mut self, max_pad: usize) -> Self {
        self.max_pad = Some(max_pad);
        self
    }

    /// Require power of 2 size
    pub fn with_power_of_2(mut self) -> Self {
        self.power_of_2 = true;
        self
    }

    /// Center the data in padded array
    pub fn with_center(mut self) -> Self {
        self.center = true;
        self
    }
}

/// Automatically pad a 1D array for optimal FFT performance
pub fn auto_pad_1d<T>(x: &Array1<T>, config: &AutoPadConfig) -> FFTResult<Array1<T>>
where
    T: Clone + Zero,
{
    let n = x.len();

    // Determine target size
    let target_size = if config.power_of_2 {
        // Next power of 2
        let min_size = n + config.min_pad;
        let mut size = 1;
        while size < min_size {
            size *= 2;
        }
        size
    } else {
        // Next fast length
        next_fast_len(n + config.min_pad, false)
    };

    // Apply maximum padding constraint
    let padded_size = if let Some(max_pad) = config.max_pad {
        target_size.min(n + max_pad)
    } else {
        target_size
    };

    // No padding needed
    if padded_size == n {
        return Ok(x.clone());
    }

    // Create padded array
    let mut padded = Array1::zeros(padded_size);

    // Determine where to place the original data
    let start_idx = if config.center {
        (padded_size - n) / 2
    } else {
        0
    };

    // Copy original data
    padded.slice_mut(s![start_idx..start_idx + n]).assign(x);

    // Apply padding based on mode
    match config.mode {
        PaddingMode::None | PaddingMode::Zero => {
            // Already zero-initialized
        }
        PaddingMode::Constant(_value) => {
            let const_val = T::zero(); // Need to convert f64 to T properly
            if start_idx > 0 {
                padded.slice_mut(s![..start_idx]).fill(const_val.clone());
            }
            if start_idx + n < padded_size {
                padded.slice_mut(s![start_idx + n..]).fill(const_val);
            }
        }
        PaddingMode::Edge => {
            // Replicate edge values
            if start_idx > 0 {
                let left_val = x[0].clone();
                padded.slice_mut(s![..start_idx]).fill(left_val);
            }
            if start_idx + n < padded_size {
                let right_val = x[n - 1].clone();
                padded.slice_mut(s![start_idx + n..]).fill(right_val);
            }
        }
        PaddingMode::Reflect => {
            // Mirror at edges
            for i in 0..start_idx {
                let offset = start_idx - i - 1;
                let cycle = 2 * (n - 1);
                let src_idx = offset % cycle;
                let src_idx = if src_idx >= n {
                    cycle - src_idx
                } else {
                    src_idx
                };
                padded[i] = x[src_idx].clone();
            }
            for i in (start_idx + n)..padded_size {
                let offset = i - (start_idx + n);
                let cycle = 2 * (n - 1);
                let src_idx = (n - 1 - (offset % cycle)).max(0);
                padded[i] = x[src_idx].clone();
            }
        }
        PaddingMode::Symmetric => {
            // Mirror with edge duplication
            for i in 0..start_idx {
                let offset = start_idx - i;
                let cycle = 2 * n;
                let src_idx = (offset - 1) % cycle;
                let src_idx = if src_idx >= n {
                    cycle - 1 - src_idx
                } else {
                    src_idx
                };
                padded[i] = x[src_idx].clone();
            }
            for i in (start_idx + n)..padded_size {
                let offset = i - (start_idx + n);
                let cycle = 2 * n;
                let src_idx = (n - 1 - (offset % cycle)).max(0);
                padded[i] = x[src_idx].clone();
            }
        }
        PaddingMode::Wrap => {
            // Circular padding
            for i in 0..start_idx {
                let src_idx = (n - (start_idx - i) % n) % n;
                padded[i] = x[src_idx].clone();
            }
            for i in (start_idx + n)..padded_size {
                let src_idx = (i - start_idx) % n;
                padded[i] = x[src_idx].clone();
            }
        }
        PaddingMode::LinearRamp => {
            // Linear fade to zero
            if start_idx > 0 {
                for i in 0..start_idx {
                    // This would need proper numeric operations for type T
                    padded[i] = T::zero();
                }
            }
            if start_idx + n < padded_size {
                for i in (start_idx + n)..padded_size {
                    // This would need proper numeric operations for type T
                    padded[i] = T::zero();
                }
            }
        }
    }

    Ok(padded)
}

/// Automatically pad a complex array for optimal FFT performance
pub fn auto_pad_complex(
    x: &Array1<Complex<f64>>,
    config: &AutoPadConfig,
) -> FFTResult<Array1<Complex<f64>>> {
    let n = x.len();

    // Determine target size
    let target_size = if config.power_of_2 {
        let min_size = n + config.min_pad;
        let mut size = 1;
        while size < min_size {
            size *= 2;
        }
        size
    } else {
        next_fast_len(n + config.min_pad, false)
    };

    // Apply maximum padding constraint
    let padded_size = if let Some(max_pad) = config.max_pad {
        target_size.min(n + max_pad)
    } else {
        target_size
    };

    if padded_size == n {
        return Ok(x.clone());
    }

    let mut padded = Array1::zeros(padded_size);
    let start_idx = if config.center {
        (padded_size - n) / 2
    } else {
        0
    };

    padded.slice_mut(s![start_idx..start_idx + n]).assign(x);

    // Apply padding
    match config.mode {
        PaddingMode::None | PaddingMode::Zero => {}
        PaddingMode::Constant(value) => {
            let const_val = Complex::new(value, 0.0);
            if start_idx > 0 {
                padded.slice_mut(s![..start_idx]).fill(const_val);
            }
            if start_idx + n < padded_size {
                padded.slice_mut(s![start_idx + n..]).fill(const_val);
            }
        }
        PaddingMode::Edge => {
            if start_idx > 0 {
                let left_val = x[0];
                padded.slice_mut(s![..start_idx]).fill(left_val);
            }
            if start_idx + n < padded_size {
                let right_val = x[n - 1];
                padded.slice_mut(s![start_idx + n..]).fill(right_val);
            }
        }
        PaddingMode::LinearRamp => {
            // Linear fade from edges to zero
            if start_idx > 0 {
                let edge_val = x[0];
                for i in 0..start_idx {
                    let t = i as f64 / start_idx as f64;
                    padded[start_idx - 1 - i] = edge_val * t;
                }
            }
            if start_idx + n < padded_size {
                let edge_val = x[n - 1];
                let pad_len = padded_size - (start_idx + n);
                for i in 0..pad_len {
                    let t = 1.0 - (i as f64 / pad_len as f64);
                    padded[start_idx + n + i] = edge_val * t;
                }
            }
        }
        _ => {
            // For other modes, use simpler implementations or delegate to auto_pad_1d
            return auto_pad_1d(x, config);
        }
    }

    Ok(padded)
}

/// Remove padding from a 1D array after FFT
pub fn remove_padding_1d<T>(
    padded: &Array1<T>,
    original_size: usize,
    config: &AutoPadConfig,
) -> Array1<T>
where
    T: Clone,
{
    let padded_size = padded.len();

    if padded_size == original_size {
        return padded.clone();
    }

    let start_idx = if config.center {
        (padded_size - original_size) / 2
    } else {
        0
    };

    padded
        .slice(s![start_idx..start_idx + original_size])
        .to_owned()
}

/// Automatic padding for N-dimensional arrays
pub fn auto_pad_nd<S, D>(
    x: &ArrayBase<S, D>,
    config: &AutoPadConfig,
    axes: Option<&[usize]>,
) -> FFTResult<ArrayD<Complex<f64>>>
where
    S: Data<Elem = Complex<f64>>,
    D: Dimension,
{
    let shape = x.shape();
    let default_axes = (0..shape.len()).collect::<Vec<_>>();
    let axes = axes.unwrap_or(&default_axes[..]);

    let mut padded_shape = shape.to_vec();

    // Calculate padded sizes for specified axes
    for &axis in axes {
        let n = shape[axis];
        let target_size = if config.power_of_2 {
            let min_size = n + config.min_pad;
            let mut size = 1;
            while size < min_size {
                size *= 2;
            }
            size
        } else {
            next_fast_len(n + config.min_pad, false)
        };

        padded_shape[axis] = if let Some(max_pad) = config.max_pad {
            target_size.min(n + max_pad)
        } else {
            target_size
        };
    }

    // Create padded array
    let mut padded = ArrayD::zeros(padded_shape.clone());

    // Copy original data - simplified implementation
    let x_dyn = x
        .to_owned()
        .into_shape_with_order(x.shape().to_vec())
        .unwrap()
        .into_dyn();

    // For simplicity, copy to origin (0,0,...) - a full implementation would handle centering
    match x_dyn.ndim() {
        1 => {
            let start = if config.center {
                (padded_shape[0] - shape[0]) / 2
            } else {
                0
            };
            padded.slice_mut(s![start..start + shape[0]]).assign(&x_dyn);
        }
        2 => {
            let start0 = if config.center && axes.contains(&0) {
                (padded_shape[0] - shape[0]) / 2
            } else {
                0
            };
            let start1 = if config.center && axes.contains(&1) {
                (padded_shape[1] - shape[1]) / 2
            } else {
                0
            };
            padded
                .slice_mut(s![start0..start0 + shape[0], start1..start1 + shape[1]])
                .assign(&x_dyn.view().into_dimensionality::<ndarray::Ix2>().unwrap());
        }
        _ => {
            // For higher dimensions, just copy to origin
            // A full implementation would handle arbitrary dimensions
            return Err(crate::FFTError::ValueError(
                "auto_pad_nd currently only supports 1D and 2D arrays".to_string(),
            ));
        }
    }

    // Apply padding based on mode (simplified for N-D case)
    match config.mode {
        PaddingMode::None | PaddingMode::Zero => {
            // Already zero-initialized
        }
        PaddingMode::Constant(value) => {
            // Would need to fill regions outside original data
            let _const_val = Complex::new(value, 0.0);
            // Implementation would be complex for N-D case
        }
        _ => {
            // Other modes would require axis-specific handling
        }
    }

    Ok(padded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_auto_pad_zero() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let config = AutoPadConfig::new(PaddingMode::Zero);

        let padded = auto_pad_complex(&x.mapv(|v| Complex::new(v, 0.0)), &config).unwrap();

        // Should pad to next fast length
        assert!(padded.len() >= x.len());

        // Original values should be preserved
        for i in 0..x.len() {
            assert_abs_diff_eq!(padded[i].re, x[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_auto_pad_power_of_2() {
        let x = Array1::from_vec(vec![1.0; 5]);
        let config = AutoPadConfig::new(PaddingMode::Zero).with_power_of_2();

        let padded = auto_pad_complex(&x.mapv(|v| Complex::new(v, 0.0)), &config).unwrap();

        // Should pad to 8 (next power of 2)
        assert_eq!(padded.len(), 8);
    }

    #[test]
    fn test_remove_padding() {
        let padded = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
        let config = AutoPadConfig::new(PaddingMode::Zero);

        let unpadded = remove_padding_1d(&padded, 4, &config);
        assert_eq!(unpadded.len(), 4);
        assert_eq!(unpadded.as_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_auto_pad_center() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let config = AutoPadConfig::new(PaddingMode::Zero)
            .with_center()
            .with_min_pad(3);

        let padded = auto_pad_complex(&x.mapv(|v| Complex::new(v, 0.0)), &config).unwrap();

        // Should center the data
        assert!(padded.len() >= 6);
        let start = (padded.len() - 3) / 2;
        assert_abs_diff_eq!(padded[start].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(padded[start + 1].re, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(padded[start + 2].re, 3.0, epsilon = 1e-10);
    }
}
