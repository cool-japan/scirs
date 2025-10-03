//! Chan-Vese segmentation (Active Contours Without Edges)
//!
//! This module implements the Chan-Vese model for image segmentation,
//! which is based on the Mumford-Shah functional and uses level sets.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};
use crate::filters::{gaussian_filter, BorderMode};
use statrs::statistics::Statistics;

/// Parameters for Chan-Vese segmentation
#[derive(Clone, Debug)]
pub struct ChanVeseParams {
    /// Weight for the length term
    pub mu: f64,
    /// Weight for the area term
    pub nu: f64,
    /// Weight for fitting term inside the contour
    pub lambda1: f64,
    /// Weight for fitting term outside the contour
    pub lambda2: f64,
    /// Time step
    pub dt: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Reinitialization frequency (0 means no reinitialization)
    pub reinit_frequency: usize,
}

impl Default for ChanVeseParams {
    fn default() -> Self {
        Self {
            mu: 0.25,
            nu: 0.0,
            lambda1: 1.0,
            lambda2: 1.0,
            dt: 0.5,
            max_iterations: 500,
            tolerance: 1e-3,
            reinit_frequency: 20,
        }
    }
}

/// Heaviside function (smoothed step function)
#[allow(dead_code)]
fn heaviside(x: f64, epsilon: f64) -> f64 {
    0.5 * (1.0 + (2.0 / std::f64::consts::PI) * (x / epsilon).atan())
}

/// Derivative of Heaviside function (Dirac delta)
#[allow(dead_code)]
fn dirac(x: f64, epsilon: f64) -> f64 {
    epsilon / (std::f64::consts::PI * (epsilon * epsilon + x * x))
}

/// Compute curvature of level set function
#[allow(dead_code)]
fn compute_curvature(phi: &ArrayView2<f64>) -> Array2<f64> {
    let (height, width) = phi.dim();
    let mut curvature = Array2::zeros((height, width));

    for i in 1..height - 1 {
        for j in 1..width - 1 {
            // Central differences
            let phi_x = (phi[[i, j + 1]] - phi[[i, j - 1]]) / 2.0;
            let phi_y = (phi[[i + 1, j]] - phi[[i - 1, j]]) / 2.0;

            let phi_xx = phi[[i, j + 1]] - 2.0 * phi[[i, j]] + phi[[i, j - 1]];
            let phi_yy = phi[[i + 1, j]] - 2.0 * phi[[i, j]] + phi[[i - 1, j]];
            let phi_xy = (phi[[i + 1, j + 1]] - phi[[i + 1, j - 1]] - phi[[i - 1, j + 1]]
                + phi[[i - 1, j - 1]])
                / 4.0;

            let denominator = (phi_x * phi_x + phi_y * phi_y).powf(1.5) + 1e-10;

            curvature[[i, j]] = (phi_xx * phi_y * phi_y - 2.0 * phi_x * phi_y * phi_xy
                + phi_yy * phi_x * phi_x)
                / denominator;
        }
    }

    curvature
}

/// Reinitialize level set function to signed distance function
#[allow(dead_code)]
fn reinitialize_level_set(phi: &mut Array2<f64>, iterations: usize) {
    let (height, width) = phi.dim();
    let dt = 0.5;

    for _ in 0..iterations {
        let mut phi_new = phi.clone();

        for i in 1..height - 1 {
            for j in 1..width - 1 {
                // Upwind scheme for reinitialization
                let a = phi[[i, j]] - phi[[i - 1, j]];
                let b = phi[[i + 1, j]] - phi[[i, j]];
                let c = phi[[i, j]] - phi[[i, j - 1]];
                let d = phi[[i, j + 1]] - phi[[i, j]];

                let a_plus = a.max(0.0);
                let b_minus = b.min(0.0);
                let c_plus = c.max(0.0);
                let d_minus = d.min(0.0);

                let sign_phi = phi[[i, j]] / (phi[[i, j]].abs() + 1e-10);

                let grad_plus = ((a_plus * a_plus).max(b_minus * b_minus)
                    + (c_plus * c_plus).max(d_minus * d_minus))
                .sqrt();

                let a_minus = a.min(0.0);
                let b_plus = b.max(0.0);
                let c_minus = c.min(0.0);
                let d_plus = d.max(0.0);

                let grad_minus = ((a_minus * a_minus).max(b_plus * b_plus)
                    + (c_minus * c_minus).max(d_plus * d_plus))
                .sqrt();

                let grad = if sign_phi > 0.0 {
                    grad_plus
                } else {
                    grad_minus
                };

                phi_new[[i, j]] = phi[[i, j]] - dt * sign_phi * (grad - 1.0);
            }
        }

        *phi = phi_new;
    }
}

/// Chan-Vese segmentation using level sets
///
/// # Arguments
/// * `image` - Input image
/// * `initial_level_set` - Initial level set function (optional)
/// * `params` - Segmentation parameters
///
/// # Returns
/// Binary segmentation mask where true indicates inside the contour
#[allow(dead_code)]
pub fn chan_vese<T>(
    image: &ArrayView2<T>,
    initial_level_set: Option<&ArrayView2<f64>>,
    params: Option<ChanVeseParams>,
) -> NdimageResult<Array2<bool>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
{
    let params = params.unwrap_or_default();
    let (height, width) = image.dim();

    // Convert image to f64
    let img = image.mapv(|x| x.to_f64().unwrap_or(0.0));

    // Initialize level _set function
    let mut phi = if let Some(init) = initial_level_set {
        if init.dim() != image.dim() {
            return Err(NdimageError::DimensionError(
                "Initial level _set must have same dimensions as image".into(),
            ));
        }
        init.to_owned()
    } else {
        // Default initialization: circle at center
        let center_y = height as f64 / 2.0;
        let center_x = width as f64 / 2.0;
        let radius = (height.min(width) as f64) / 4.0;

        Array2::from_shape_fn((height, width), |(i, j)| {
            let dy = i as f64 - center_y;
            let dx = j as f64 - center_x;
            radius - (dy * dy + dx * dx).sqrt()
        })
    };

    let epsilon = 1.0; // Width of Heaviside and Dirac

    // Evolution loop
    for iteration in 0..params.max_iterations {
        // Compute region averages
        let mut c1 = 0.0; // Average inside
        let mut c2 = 0.0; // Average outside
        let mut area1 = 0.0;
        let mut area2 = 0.0;

        for i in 0..height {
            for j in 0..width {
                let h = heaviside(phi[[i, j]], epsilon);
                c1 += img[[i, j]] * h;
                area1 += h;
                c2 += img[[i, j]] * (1.0 - h);
                area2 += 1.0 - h;
            }
        }

        c1 /= area1.max(1.0);
        c2 /= area2.max(1.0);

        // Update level _set
        let phi_old = phi.clone();
        let curvature = compute_curvature(&phi.view());

        for i in 1..height - 1 {
            for j in 1..width - 1 {
                let delta = dirac(phi[[i, j]], epsilon);

                // Data fitting terms
                let f1 = (img[[i, j]] - c1).powi(2);
                let f2 = (img[[i, j]] - c2).powi(2);

                // Update equation
                phi[[i, j]] += params.dt
                    * delta
                    * (params.mu * curvature[[i, j]] - params.nu - params.lambda1 * f1
                        + params.lambda2 * f2);
            }
        }

        // Reinitialize if needed
        if params.reinit_frequency > 0 && (iteration + 1) % params.reinit_frequency == 0 {
            reinitialize_level_set(&mut phi, 10);
        }

        // Check convergence
        let change = ((phi.clone() - phi_old).mapv(|x| x.abs())).mean();
        if change < params.tolerance {
            break;
        }
    }

    // Convert level _set to binary mask
    let mask = phi.mapv(|x| x >= 0.0);

    Ok(mask)
}

/// Multi-phase Chan-Vese segmentation for multiple regions
///
/// # Arguments
/// * `image` - Input image
/// * `num_phases` - Number of level set functions (regions = 2^num_phases)
/// * `params` - Segmentation parameters
///
/// # Returns
/// Label array where each pixel is assigned to a region (0 to 2^num_phases - 1)
#[allow(dead_code)]
pub fn chan_vese_multiphase<T>(
    image: &ArrayView2<T>,
    num_phases: usize,
    params: Option<ChanVeseParams>,
) -> NdimageResult<Array2<usize>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
{
    let params = params.unwrap_or_default();
    let (height, width) = image.dim();

    if num_phases == 0 || num_phases > 3 {
        return Err(NdimageError::InvalidInput(
            "Number of _phases must be between 1 and 3".into(),
        ));
    }

    // Convert image to f64
    let img = image.mapv(|x| x.to_f64().unwrap_or(0.0));

    // Initialize level set functions
    let mut phi_list = Vec::new();

    for k in 0..num_phases {
        // Initialize with different patterns
        let phi =
            match k {
                0 => {
                    // Vertical division
                    Array2::from_shape_fn(
                        (height, width),
                        |(_, j)| {
                            if j < width / 2 {
                                10.0
                            } else {
                                -10.0
                            }
                        },
                    )
                }
                1 => {
                    // Horizontal division
                    Array2::from_shape_fn((height, width), |(i, _)| {
                        if i < height / 2 {
                            10.0
                        } else {
                            -10.0
                        }
                    })
                }
                2 => {
                    // Diagonal division
                    Array2::from_shape_fn((height, width), |(i, j)| {
                        if i + j < (height + width) / 2 {
                            10.0
                        } else {
                            -10.0
                        }
                    })
                }
                _ => unreachable!(),
            };

        phi_list.push(phi);
    }

    let epsilon = 1.0;
    let num_regions = 1 << num_phases; // 2^num_phases

    // Evolution loop
    for iteration in 0..params.max_iterations {
        // Compute region averages
        let mut c = vec![0.0; num_regions];
        let mut area = vec![0.0; num_regions];

        for i in 0..height {
            for j in 0..width {
                // Determine region membership
                let mut region = 0;
                for (k, phi) in phi_list.iter().enumerate() {
                    if phi[[i, j]] >= 0.0 {
                        region |= 1 << k;
                    }
                }

                // Compute membership function
                let mut membership = 1.0;
                for (k, phi) in phi_list.iter().enumerate() {
                    if (region >> k) & 1 == 1 {
                        membership *= heaviside(phi[[i, j]], epsilon);
                    } else {
                        membership *= 1.0 - heaviside(phi[[i, j]], epsilon);
                    }
                }

                c[region] += img[[i, j]] * membership;
                area[region] += membership;
            }
        }

        // Normalize averages
        for k in 0..num_regions {
            if area[k] > 0.0 {
                c[k] /= area[k];
            }
        }

        // Update each level set
        let mut converged = true;

        // Clone phi_list for read access during mutable iteration
        let phi_list_snapshot: Vec<_> = phi_list.iter().map(|phi| phi.clone()).collect();

        for (phase_idx, phi) in phi_list.iter_mut().enumerate() {
            let phi_old = phi.clone();
            let curvature = compute_curvature(&phi.view());

            for i in 1..height - 1 {
                for j in 1..width - 1 {
                    let delta = dirac(phi[[i, j]], epsilon);

                    // Compute data fitting term
                    let mut data_term = 0.0;

                    for region in 0..num_regions {
                        let mut weight = params.lambda1;

                        // Check if this phase should be positive or negative for this region
                        if (region >> phase_idx) & 1 == 0 {
                            weight = -params.lambda2;
                        }

                        // Compute membership for other _phases
                        let mut other_membership = 1.0;
                        for (k, other_phi) in phi_list_snapshot.iter().enumerate() {
                            if k != phase_idx {
                                if (region >> k) & 1 == 1 {
                                    other_membership *= heaviside(other_phi[[i, j]], epsilon);
                                } else {
                                    other_membership *= 1.0 - heaviside(other_phi[[i, j]], epsilon);
                                }
                            }
                        }

                        data_term += weight * (img[[i, j]] - c[region]).powi(2) * other_membership;
                    }

                    // Update equation
                    phi[[i, j]] +=
                        params.dt * delta * (params.mu * curvature[[i, j]] - params.nu - data_term);
                }
            }

            // Check convergence
            let change = ((phi.clone() - phi_old).mapv(|x| x.abs())).mean();
            if change >= params.tolerance {
                converged = false;
            }
        }

        // Reinitialize if needed
        if params.reinit_frequency > 0 && (iteration + 1) % params.reinit_frequency == 0 {
            for phi in &mut phi_list {
                reinitialize_level_set(phi, 10);
            }
        }

        if converged {
            break;
        }
    }

    // Convert level sets to label array
    let mut labels = Array2::zeros((height, width));

    for i in 0..height {
        for j in 0..width {
            let mut region = 0;
            for (k, phi) in phi_list.iter().enumerate() {
                if phi[[i, j]] >= 0.0 {
                    region |= 1 << k;
                }
            }
            labels[[i, j]] = region;
        }
    }

    Ok(labels)
}

/// Initialize level set from binary mask
#[allow(dead_code)]
pub fn mask_to_level_set(
    mask: &ArrayView2<bool>,
    smoothing: Option<f64>,
) -> NdimageResult<Array2<f64>> {
    let (height, width) = mask.dim();

    // Convert mask to signed distance function
    let mut phi = Array2::zeros((height, width));

    // Simple approximation: positive inside, negative outside
    for i in 0..height {
        for j in 0..width {
            if mask[[i, j]] {
                // Find distance to nearest boundary
                let mut min_dist = f64::INFINITY;

                for di in -3..=3 {
                    for dj in -3..=3 {
                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;

                        if ni >= 0 && ni < height as i32 && nj >= 0 && nj < width as i32 {
                            if !mask[[ni as usize, nj as usize]] {
                                let dist = ((di * di + dj * dj) as f64).sqrt();
                                min_dist = min_dist.min(dist);
                            }
                        }
                    }
                }

                phi[[i, j]] = min_dist;
            } else {
                // Find distance to nearest interior point
                let mut min_dist = f64::INFINITY;

                for di in -3..=3 {
                    for dj in -3..=3 {
                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;

                        if ni >= 0 && ni < height as i32 && nj >= 0 && nj < width as i32 {
                            if mask[[ni as usize, nj as usize]] {
                                let dist = ((di * di + dj * dj) as f64).sqrt();
                                min_dist = min_dist.min(dist);
                            }
                        }
                    }
                }

                phi[[i, j]] = -min_dist;
            }
        }
    }

    // Apply smoothing if requested
    if let Some(sigma) = smoothing {
        phi = gaussian_filter(&phi, sigma, Some(BorderMode::Reflect), None)?;
    }

    Ok(phi)
}

/// Create checkerboard initialization for multi-phase segmentation
#[allow(dead_code)]
pub fn checkerboard_level_set(shape: (usize, usize), square_size: usize) -> Array2<f64> {
    let (height, width) = shape;

    Array2::from_shape_fn((height, width), |(i, j)| {
        let row_even = (i / square_size) % 2 == 0;
        let col_even = (j / square_size) % 2 == 0;

        if row_even == col_even {
            10.0
        } else {
            -10.0
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heaviside_dirac() {
        let epsilon = 1.0;

        // Test Heaviside
        assert!((heaviside(10.0, epsilon) - 1.0).abs() < 0.1);
        assert!((heaviside(-10.0, epsilon) - 0.0).abs() < 0.1);
        assert!((heaviside(0.0, epsilon) - 0.5).abs() < 0.01);

        // Test Dirac
        assert!(dirac(0.0, epsilon) > dirac(1.0, epsilon));
        assert!(dirac(1.0, epsilon) > dirac(10.0, epsilon));
    }

    #[test]
    fn test_chan_vese_simple() {
        // Create a simple test image with two regions
        let mut image = Array2::zeros((50, 50));

        // Create a bright square in the center
        for i in 15..35 {
            for j in 15..35 {
                image[[i, j]] = 1.0;
            }
        }

        // Run Chan-Vese
        let params = ChanVeseParams {
            max_iterations: 50,
            tolerance: 1e-2,
            ..Default::default()
        };

        let result = chan_vese(&image.view(), None, Some(params)).unwrap();

        // Check that center region is segmented
        assert!(result[[25, 25]]);
        // Check that corners are background
        assert!(!result[[5, 5]]);
        assert!(!result[[45, 45]]);
    }

    #[test]
    fn test_mask_to_level_set() {
        let mut mask = Array2::default((10, 10));

        // Create a small square
        for i in 3..7 {
            for j in 3..7 {
                mask[[i, j]] = true;
            }
        }

        let phi = mask_to_level_set(&mask.view(), None).unwrap();

        // Check signs
        assert!(phi[[5, 5]] > 0.0); // Inside
        assert!(phi[[1, 1]] < 0.0); // Outside
    }

    #[test]
    fn test_checkerboard_level_set() {
        let phi = checkerboard_level_set((20, 20), 5);

        assert_eq!(phi.dim(), (20, 20));

        // Check pattern
        assert!(phi[[0, 0]] > 0.0);
        assert!(phi[[5, 0]] < 0.0);
        assert!(phi[[0, 5]] < 0.0);
        assert!(phi[[5, 5]] > 0.0);
    }
}
