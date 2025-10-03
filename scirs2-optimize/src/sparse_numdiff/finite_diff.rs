//! Finite difference methods for sparse numerical differentiation
//!
//! This module provides utilities and options for finite difference
//! methods used in sparse numerical differentiation.

use crate::parallel::ParallelOptions;
use scirs2_core::ndarray::ArrayView1;

/// Options for sparse numerical differentiation
#[derive(Debug, Clone)]
pub struct SparseFiniteDiffOptions {
    /// Method to use for finite differences ('2-point', '3-point', 'cs')
    pub method: String,
    /// Relative step size (if None, determined automatically)
    pub rel_step: Option<f64>,
    /// Absolute step size (if None, determined automatically)
    pub abs_step: Option<f64>,
    /// Bounds on the variables
    pub bounds: Option<Vec<(f64, f64)>>,
    /// Parallel computation options
    pub parallel: Option<ParallelOptions>,
    /// Random seed for coloring algorithm
    pub seed: Option<u64>,
    /// Maximum number of columns to group together
    pub max_group_size: usize,
}

impl Default for SparseFiniteDiffOptions {
    fn default() -> Self {
        Self {
            method: "2-point".to_string(),
            rel_step: None,
            abs_step: None,
            bounds: None,
            parallel: None,
            seed: None,
            max_group_size: 100,
        }
    }
}

/// Calculates appropriate step sizes for finite differences
///
/// # Arguments
///
/// * `x` - Point at which to compute derivatives
/// * `options` - Finite difference options
///
/// # Returns
///
/// Vector of step sizes for each dimension
#[allow(dead_code)]
pub fn compute_step_sizes(x: &ArrayView1<f64>, options: &SparseFiniteDiffOptions) -> Vec<f64> {
    let n = x.len();
    let mut h = vec![0.0; n];

    // Typical values for different methods
    let typical_rel_step = match options.method.as_str() {
        "2-point" => 1e-8,
        "3-point" => 1e-5,
        "cs" => 1e-14, // Complex step
        _ => 1e-8,     // Default to 2-point
    };

    let rel_step = options.rel_step.unwrap_or(typical_rel_step);
    let abs_step = options.abs_step.unwrap_or(1e-8);

    // Calculate step size for each dimension
    for i in 0..n {
        let xi = x[i];

        // Base step size on relative and absolute components
        let mut hi = abs_step.max(rel_step * xi.abs());

        // Ensure step moves in the correct direction for zero elements
        if xi == 0.0 {
            hi = abs_step;
        }

        // Adjust for bounds if provided
        if let Some(ref bounds) = options.bounds {
            if i < bounds.len() {
                let (lower, upper) = bounds[i];
                if xi + hi > upper {
                    hi = -hi; // Reverse direction
                }
                if xi + hi < lower {
                    hi = abs_step.min((upper - xi) / 2.0); // Reduce size
                }
            }
        }

        h[i] = hi;
    }

    h
}
