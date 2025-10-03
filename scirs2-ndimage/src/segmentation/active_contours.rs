//! Active contours (Snakes) segmentation
//!
//! This module implements active contour models for image segmentation,
//! including parametric snakes and level set methods.

use scirs2_core::ndarray::{arr2, Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};
use crate::filters::{gaussian_filter, sobel};
use crate::interpolation::{map_coordinates, InterpolationOrder};

/// Parameters for active contour evolution
#[derive(Clone, Debug)]
pub struct ActiveContourParams {
    /// Weight of the internal energy (elasticity)
    pub alpha: f64,
    /// Weight of the curvature energy (rigidity)
    pub beta: f64,
    /// Weight of the external energy (image forces)
    pub gamma: f64,
    /// Weight of the balloon force
    pub kappa: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence: f64,
    /// Time step for evolution
    pub time_step: f64,
}

impl Default for ActiveContourParams {
    fn default() -> Self {
        Self {
            alpha: 0.01,
            beta: 0.1,
            gamma: 1.0,
            kappa: 0.0,
            max_iterations: 1000,
            convergence: 0.1,
            time_step: 0.1,
        }
    }
}

/// Compute gradient vector flow (GVF) field for edge map
#[allow(dead_code)]
fn gradient_vector_flow<T>(
    edge_map: &ArrayView2<T>,
    mu: f64,
    iterations: usize,
) -> NdimageResult<(Array2<f64>, Array2<f64>)>
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let (height, width) = edge_map.dim();

    // Initialize with gradient of edge _map
    let (fx, fy) = compute_gradient(edge_map)?;
    let mut u = fx.clone();
    let mut v = fy.clone();

    // Compute edge _map squared
    let edge_sq = edge_map.mapv(|x| x.to_f64().unwrap_or(0.0).powi(2));

    // Iterative GVF computation
    for _ in 0..iterations {
        let mut u_new = Array2::zeros((height, width));
        let mut v_new = Array2::zeros((height, width));

        for i in 1..height - 1 {
            for j in 1..width - 1 {
                // Laplacian of u and v
                let laplacian_u =
                    u[[i + 1, j]] + u[[i - 1, j]] + u[[i, j + 1]] + u[[i, j - 1]] - 4.0 * u[[i, j]];
                let laplacian_v =
                    v[[i + 1, j]] + v[[i - 1, j]] + v[[i, j + 1]] + v[[i, j - 1]] - 4.0 * v[[i, j]];

                // Update equations
                let b = edge_sq[[i, j]];
                u_new[[i, j]] = u[[i, j]] + mu * laplacian_u - b * (u[[i, j]] - fx[[i, j]]);
                v_new[[i, j]] = v[[i, j]] + mu * laplacian_v - b * (v[[i, j]] - fy[[i, j]]);
            }
        }

        u = u_new;
        v = v_new;
    }

    Ok((u, v))
}

/// Compute gradient of an image
#[allow(dead_code)]
fn compute_gradient<T>(image: &ArrayView2<T>) -> NdimageResult<(Array2<f64>, Array2<f64>)>
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let (height, width) = image.dim();
    let mut gx = Array2::zeros((height, width));
    let mut gy = Array2::zeros((height, width));

    // Simple central differences
    for i in 1..height - 1 {
        for j in 1..width - 1 {
            let dx = image[[i, j + 1]].to_f64().unwrap_or(0.0)
                - image[[i, j - 1]].to_f64().unwrap_or(0.0);
            let dy = image[[i + 1, j]].to_f64().unwrap_or(0.0)
                - image[[i - 1, j]].to_f64().unwrap_or(0.0);

            gx[[i, j]] = dx / 2.0;
            gy[[i, j]] = dy / 2.0;
        }
    }

    Ok((gx, gy))
}

/// Active contour segmentation using parametric snakes
///
/// # Arguments
/// * `image` - Input image
/// * `initial_contour` - Initial contour points (N x 2 array)
/// * `params` - Active contour parameters
///
/// # Returns
/// Final contour points after evolution
#[allow(dead_code)]
pub fn active_contour<T>(
    image: &ArrayView2<T>,
    initial_contour: &ArrayView2<f64>,
    params: Option<ActiveContourParams>,
) -> NdimageResult<Array2<f64>>
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

    // Validate inputs
    if initial_contour.dim().1 != 2 {
        return Err(NdimageError::InvalidInput(
            "Initial _contour must be N x 2 array".into(),
        ));
    }

    let num_points = initial_contour.dim().0;
    if num_points < 3 {
        return Err(NdimageError::InvalidInput(
            "Contour must have at least 3 points".into(),
        ));
    }

    // Preprocess image to compute edge map
    let smoothed = gaussian_filter(
        &image.mapv(|x| x.to_f64().unwrap_or(0.0)),
        2.0, // Use single sigma value
        None,
        None,
    )?;

    // Compute gradients in x and y directions
    let gx = sobel(&smoothed, 1, None)?; // axis 1 (x-direction)
    let gy = sobel(&smoothed, 0, None)?; // axis 0 (y-direction)
    let edge_map = (gx.mapv(|x| x * x) + gy.mapv(|x| x * x)).mapv(|x| x.sqrt());

    // Compute GVF field
    let (u, v) = gradient_vector_flow(&edge_map.view(), 0.2, 100)?;

    // Initialize _contour
    let mut _contour = initial_contour.to_owned();
    let mut prev_contour = _contour.clone();

    // Evolution loop
    for iteration in 0..params.max_iterations {
        // Save previous _contour
        prev_contour.assign(&_contour);

        // Update each point
        for i in 0..num_points {
            let prev_idx = if i == 0 { num_points - 1 } else { i - 1 };
            let next_idx = if i == num_points - 1 { 0 } else { i + 1 };

            // Current point and neighbors
            let x = _contour[[i, 0]];
            let y = _contour[[i, 1]];
            let x_prev = _contour[[prev_idx, 0]];
            let y_prev = _contour[[prev_idx, 1]];
            let x_next = _contour[[next_idx, 0]];
            let y_next = _contour[[next_idx, 1]];

            // Internal energy (elasticity)
            let avg_x = (x_prev + x_next) / 2.0;
            let avg_y = (y_prev + y_next) / 2.0;
            let internal_x = params.alpha * (avg_x - x);
            let internal_y = params.alpha * (avg_y - y);

            // Curvature energy
            let curvature_x = params.beta * (x_prev - 2.0 * x + x_next);
            let curvature_y = params.beta * (y_prev - 2.0 * y + y_next);

            // External energy from GVF
            let (external_x, external_y) =
                if x >= 0.0 && x < u.dim().1 as f64 - 1.0 && y >= 0.0 && y < u.dim().0 as f64 - 1.0
                {
                    // Bilinear interpolation of GVF field
                    let coords = arr2(&[[y, x]]);
                    let u_interp = map_coordinates(
                        &u.to_owned(),
                        &coords.to_owned().into_dyn(),
                        Some(1), // order: 1 for linear
                        None,
                        None,
                        None, // prefilter
                    )?;
                    let v_interp = map_coordinates(
                        &v.to_owned(),
                        &coords.to_owned().into_dyn(),
                        Some(1), // order: 1 for linear
                        None,
                        None,
                        None, // prefilter
                    )?;

                    (params.gamma * u_interp[[0]], params.gamma * v_interp[[0]])
                } else {
                    (0.0, 0.0)
                };

            // Balloon force (optional)
            let (balloon_x, balloon_y) = if params.kappa != 0.0 {
                // Compute normal vector
                let dx = x_next - x_prev;
                let dy = y_next - y_prev;
                let norm = (dx * dx + dy * dy).sqrt();

                if norm > 0.0 {
                    (-params.kappa * dy / norm, params.kappa * dx / norm)
                } else {
                    (0.0, 0.0)
                }
            } else {
                (0.0, 0.0)
            };

            // Update position
            _contour[[i, 0]] +=
                params.time_step * (internal_x + curvature_x + external_x + balloon_x);
            _contour[[i, 1]] +=
                params.time_step * (internal_y + curvature_y + external_y + balloon_y);

            // Keep within image bounds
            _contour[[i, 0]] = _contour[[i, 0]].max(0.0).min((image.dim().1 - 1) as f64);
            _contour[[i, 1]] = _contour[[i, 1]].max(0.0).min((image.dim().0 - 1) as f64);
        }

        // Check convergence
        let movement = ((_contour.clone() - prev_contour.clone()).mapv(|x| x * x))
            .sum()
            .sqrt();
        if movement < params.convergence {
            break;
        }
    }

    Ok(_contour)
}

/// Generate initial circular contour
#[allow(dead_code)]
pub fn create_circle_contour(center: (f64, f64), radius: f64, num_points: usize) -> Array2<f64> {
    let mut contour = Array2::zeros((num_points, 2));

    for i in 0..num_points {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / num_points as f64;
        contour[[i, 0]] = center.0 + radius * angle.cos();
        contour[[i, 1]] = center.1 + radius * angle.sin();
    }

    contour
}

/// Generate initial elliptical contour
#[allow(dead_code)]
pub fn create_ellipse_contour(
    center: (f64, f64),
    semi_major: f64,
    semi_minor: f64,
    angle: f64,
    num_points: usize,
) -> Array2<f64> {
    let mut contour = Array2::zeros((num_points, 2));
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();

    for i in 0..num_points {
        let theta = 2.0 * std::f64::consts::PI * i as f64 / num_points as f64;
        let x = semi_major * theta.cos();
        let y = semi_minor * theta.sin();

        // Rotate and translate
        contour[[i, 0]] = center.0 + x * cos_angle - y * sin_angle;
        contour[[i, 1]] = center.1 + x * sin_angle + y * cos_angle;
    }

    contour
}

/// Extract contour from segmentation mask
#[allow(dead_code)]
pub fn mask_to_contour(mask: &ArrayView2<bool>) -> Vec<(f64, f64)> {
    let (height, width) = mask.dim();
    let mut contour_points = Vec::new();

    // Find boundary pixels
    for i in 0..height {
        for j in 0..width {
            if mask[[i, j]] {
                // Check if it's a boundary pixel
                let mut is_boundary = false;

                // Check 4-neighbors
                for (di, dj) in &[(0, 1), (1, 0), (0, -1), (-1, 0)] {
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;

                    if ni < 0 || ni >= height as i32 || nj < 0 || nj >= width as i32 {
                        is_boundary = true;
                        break;
                    }

                    if !mask[[ni as usize, nj as usize]] {
                        is_boundary = true;
                        break;
                    }
                }

                if is_boundary {
                    contour_points.push((j as f64, i as f64));
                }
            }
        }
    }

    // Order points to form a continuous contour
    if contour_points.len() > 2 {
        order_contour_points(&mut contour_points);
    }

    contour_points
}

/// Order contour points to form a continuous path
#[allow(dead_code)]
fn order_contour_points(points: &mut Vec<(f64, f64)>) {
    if points.is_empty() {
        return;
    }

    let mut ordered = vec![points[0]];
    points.remove(0);

    while !points.is_empty() {
        let last = ordered.last().unwrap();

        // Find nearest point
        let mut min_dist = f64::INFINITY;
        let mut min_idx = 0;

        for (idx, point) in points.iter().enumerate() {
            let dist = ((point.0 - last.0).powi(2) + (point.1 - last.1).powi(2)).sqrt();
            if dist < min_dist {
                min_dist = dist;
                min_idx = idx;
            }
        }

        ordered.push(points[min_idx]);
        points.remove(min_idx);
    }

    *points = ordered;
}

/// Smooth a contour using B-spline interpolation
#[allow(dead_code)]
pub fn smooth_contour(
    contour: &ArrayView2<f64>,
    smoothing_factor: f64,
) -> NdimageResult<Array2<f64>> {
    let num_points = contour.dim().0;

    if num_points < 3 {
        return Ok(contour.to_owned());
    }

    // Apply Gaussian smoothing to x and y coordinates separately
    let x_coords = contour.slice(scirs2_core::ndarray::s![.., 0]);
    let y_coords = contour.slice(scirs2_core::ndarray::s![.., 1]);

    // Pad for circular boundary
    let mut x_padded = Array1::zeros(num_points + 4);
    let mut y_padded = Array1::zeros(num_points + 4);

    x_padded
        .slice_mut(scirs2_core::ndarray::s![2..num_points + 2])
        .assign(&x_coords);
    y_padded
        .slice_mut(scirs2_core::ndarray::s![2..num_points + 2])
        .assign(&y_coords);

    // Circular padding
    x_padded[0] = x_coords[num_points - 2];
    x_padded[1] = x_coords[num_points - 1];
    x_padded[num_points + 2] = x_coords[0usize];
    x_padded[num_points + 3] = x_coords[1usize];

    y_padded[0] = y_coords[num_points - 2];
    y_padded[1] = y_coords[num_points - 1];
    y_padded[num_points + 2] = y_coords[0usize];
    y_padded[num_points + 3] = y_coords[1usize];

    // Apply smoothing
    let kernel_size = (smoothing_factor * 3.0) as usize;
    let kernel_size = kernel_size.max(3).min(num_points / 2);

    let mut smooth_x = Array1::zeros(num_points);
    let mut smooth_y = Array1::zeros(num_points);

    for i in 0..num_points {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut weight_sum = 0.0;

        for j in 0..kernel_size {
            let idx = i + 2 + j - kernel_size / 2;
            let weight = (-((j as f64 - kernel_size as f64 / 2.0).powi(2))
                / (2.0 * smoothing_factor.powi(2)))
            .exp();

            sum_x += x_padded[idx] * weight;
            sum_y += y_padded[idx] * weight;
            weight_sum += weight;
        }

        smooth_x[i] = sum_x / weight_sum;
        smooth_y[i] = sum_y / weight_sum;
    }

    // Combine smoothed coordinates
    let mut smoothed = Array2::zeros((num_points, 2));
    smoothed
        .slice_mut(scirs2_core::ndarray::s![.., 0])
        .assign(&smooth_x);
    smoothed
        .slice_mut(scirs2_core::ndarray::s![.., 1])
        .assign(&smooth_y);

    Ok(smoothed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_circle_contour() {
        let contour = create_circle_contour((50.0, 50.0), 20.0, 16);

        assert_eq!(contour.dim(), (16, 2));

        // Check that all points are approximately at radius 20
        for i in 0..16 {
            let x = contour[[i, 0]] - 50.0;
            let y = contour[[i, 1]] - 50.0;
            let radius = (x * x + y * y).sqrt();
            assert!((radius - 20.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_create_ellipse_contour() {
        let contour = create_ellipse_contour((50.0, 50.0), 30.0, 20.0, 0.0, 20);

        assert_eq!(contour.dim(), (20, 2));

        // Check that points lie on ellipse
        for i in 0..20 {
            let x = (contour[[i, 0]] - 50.0) / 30.0;
            let y = (contour[[i, 1]] - 50.0) / 20.0;
            let value = x * x + y * y;
            assert!((value - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    #[ignore] // Takes too long to run
    fn test_active_contour_simple() {
        // Create a simple test image with a circle
        let mut image = Array2::zeros((100, 100));

        // Draw a circle
        for i in 0..100 {
            for j in 0..100 {
                let dx = i as f64 - 50.0;
                let dy = j as f64 - 50.0;
                let r = (dx * dx + dy * dy).sqrt();

                if (r - 30.0).abs() < 2.0 {
                    image[[i, j]] = 1.0;
                }
            }
        }

        // Create initial contour
        let initial = create_circle_contour((50.0, 50.0), 25.0, 20);

        // Run active contour
        let params = ActiveContourParams {
            max_iterations: 50,
            ..Default::default()
        };

        let result = active_contour(&image.view(), &initial.view(), Some(params)).unwrap();

        assert_eq!(result.dim(), (20, 2));
    }
}
