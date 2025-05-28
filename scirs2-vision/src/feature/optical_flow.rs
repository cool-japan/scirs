//! Optical flow computation for motion analysis
//!
//! This module provides algorithms for computing optical flow between
//! consecutive frames, useful for motion analysis and tracking.

use crate::error::Result;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use ndarray::{s, Array2};

/// Optical flow vector at a point
#[derive(Debug, Clone, Copy)]
pub struct FlowVector {
    /// Horizontal displacement
    pub u: f32,
    /// Vertical displacement
    pub v: f32,
}

/// Parameters for Lucas-Kanade optical flow
#[derive(Debug, Clone)]
pub struct LucasKanadeParams {
    /// Window size for local computation
    pub window_size: usize,
    /// Maximum iterations for iterative refinement
    pub max_iterations: usize,
    /// Convergence threshold
    pub epsilon: f32,
    /// Number of pyramid levels (0 for no pyramid)
    pub pyramid_levels: usize,
}

impl Default for LucasKanadeParams {
    fn default() -> Self {
        Self {
            window_size: 15,
            max_iterations: 20,
            epsilon: 0.01,
            pyramid_levels: 3,
        }
    }
}

/// Compute optical flow using Lucas-Kanade method
///
/// # Arguments
///
/// * `img1` - First frame
/// * `img2` - Second frame
/// * `points` - Points to track (if None, computes dense flow)
/// * `params` - Algorithm parameters
///
/// # Returns
///
/// * Flow field as 2D array of flow vectors
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::{lucas_kanade_flow, LucasKanadeParams};
/// use image::{DynamicImage, RgbImage};
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// // Create simple test images
/// let frame1 = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
/// let frame2 = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
/// let flow = lucas_kanade_flow(&frame1, &frame2, None, &LucasKanadeParams::default())?;
/// # Ok(())
/// # }
/// ```
pub fn lucas_kanade_flow(
    img1: &DynamicImage,
    img2: &DynamicImage,
    points: Option<&[(f32, f32)]>,
    params: &LucasKanadeParams,
) -> Result<Array2<FlowVector>> {
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();

    if params.pyramid_levels > 0 {
        pyramidal_lucas_kanade(&gray1, &gray2, points, params)
    } else {
        simple_lucas_kanade(&gray1, &gray2, points, params)
    }
}

/// Simple Lucas-Kanade without pyramid
fn simple_lucas_kanade(
    img1: &GrayImage,
    img2: &GrayImage,
    points: Option<&[(f32, f32)]>,
    params: &LucasKanadeParams,
) -> Result<Array2<FlowVector>> {
    let (width, height) = img1.dimensions();

    // Convert images to float arrays
    let i1 = image_to_float_array(img1);
    let i2 = image_to_float_array(img2);

    // Compute image gradients
    let (ix, iy) = compute_gradients(&i1);

    let half_window = params.window_size / 2;

    // Determine points to compute flow for
    let track_points: Vec<(f32, f32)> = if let Some(pts) = points {
        pts.to_vec()
    } else {
        // Dense flow - compute for all pixels with sufficient margin
        let mut pts = Vec::new();
        for y in half_window..height as usize - half_window {
            for x in half_window..width as usize - half_window {
                pts.push((x as f32, y as f32));
            }
        }
        pts
    };

    // Initialize flow field
    let mut flow = Array2::from_elem(
        (height as usize, width as usize),
        FlowVector { u: 0.0, v: 0.0 },
    );

    // Compute flow for each point
    for &(px, py) in &track_points {
        let x = px as usize;
        let y = py as usize;

        // Skip boundary points
        if x < half_window
            || x >= width as usize - half_window
            || y < half_window
            || y >= height as usize - half_window
        {
            continue;
        }

        // Extract window around point
        let window_ix = ix.slice(s![
            y - half_window..=y + half_window,
            x - half_window..=x + half_window
        ]);
        let window_iy = iy.slice(s![
            y - half_window..=y + half_window,
            x - half_window..=x + half_window
        ]);
        let window_i1 = i1.slice(s![
            y - half_window..=y + half_window,
            x - half_window..=x + half_window
        ]);

        // Build system matrix A^T A
        let mut a11 = 0.0f32;
        let mut a12 = 0.0f32;
        let mut a22 = 0.0f32;

        for ((ix_val, iy_val), _) in window_ix.iter().zip(window_iy.iter()).zip(window_i1.iter()) {
            a11 += ix_val * ix_val;
            a12 += ix_val * iy_val;
            a22 += iy_val * iy_val;
        }

        let det = a11 * a22 - a12 * a12;
        if det.abs() < 1e-6 {
            continue; // Singular matrix, skip this point
        }

        // Iterative refinement
        let mut u = 0.0f32;
        let mut v = 0.0f32;

        for _ in 0..params.max_iterations {
            // Get warped window from second image
            let warped_x = (x as f32 + u) as usize;
            let warped_y = (y as f32 + v) as usize;

            if warped_x < half_window
                || warped_x >= width as usize - half_window
                || warped_y < half_window
                || warped_y >= height as usize - half_window
            {
                break;
            }

            let window_i2 = i2.slice(s![
                warped_y - half_window..=warped_y + half_window,
                warped_x - half_window..=warped_x + half_window
            ]);

            // Compute temporal derivative and error
            let mut b1 = 0.0f32;
            let mut b2 = 0.0f32;

            for ((&ix_val, &iy_val), (&i1_val, &i2_val)) in window_ix
                .iter()
                .zip(window_iy.iter())
                .zip(window_i1.iter().zip(window_i2.iter()))
            {
                let it = i2_val - i1_val;
                b1 -= ix_val * it;
                b2 -= iy_val * it;
            }

            // Solve for flow update
            let inv_det = 1.0 / det;
            let du = inv_det * (a22 * b1 - a12 * b2);
            let dv = inv_det * (-a12 * b1 + a11 * b2);

            u += du;
            v += dv;

            if du.abs() < params.epsilon && dv.abs() < params.epsilon {
                break;
            }
        }

        flow[[y, x]] = FlowVector { u, v };
    }

    Ok(flow)
}

/// Pyramidal Lucas-Kanade
fn pyramidal_lucas_kanade(
    img1: &GrayImage,
    img2: &GrayImage,
    points: Option<&[(f32, f32)]>,
    params: &LucasKanadeParams,
) -> Result<Array2<FlowVector>> {
    let (width, height) = img1.dimensions();

    // Build image pyramids
    let pyramid1 = build_pyramid(img1, params.pyramid_levels);
    let pyramid2 = build_pyramid(img2, params.pyramid_levels);

    // Initialize flow
    let mut flow = Array2::from_elem(
        (height as usize, width as usize),
        FlowVector { u: 0.0, v: 0.0 },
    );

    // Process from coarse to fine
    for level in (0..params.pyramid_levels).rev() {
        let scale = 2.0_f32.powi(level as i32);

        // Scale points for this level
        let scaled_points: Option<Vec<(f32, f32)>> =
            points.map(|pts| pts.iter().map(|&(x, y)| (x / scale, y / scale)).collect());

        // Compute flow at this level
        let level_params = LucasKanadeParams {
            pyramid_levels: 0, // No recursion
            ..params.clone()
        };

        let level_flow = simple_lucas_kanade(
            &pyramid1[level],
            &pyramid2[level],
            scaled_points.as_deref(),
            &level_params,
        )?;

        // Propagate flow to finer level
        if level > 0 {
            let (level_width, level_height) = pyramid1[level].dimensions();
            for y in 0..level_height as usize {
                for x in 0..level_width as usize {
                    let fine_x = (x * 2).min(width as usize - 1);
                    let fine_y = (y * 2).min(height as usize - 1);

                    flow[[fine_y, fine_x]].u = level_flow[[y, x]].u * 2.0;
                    flow[[fine_y, fine_x]].v = level_flow[[y, x]].v * 2.0;

                    // Fill neighboring pixels
                    if fine_x + 1 < width as usize {
                        flow[[fine_y, fine_x + 1]] = flow[[fine_y, fine_x]];
                    }
                    if fine_y + 1 < height as usize {
                        flow[[fine_y + 1, fine_x]] = flow[[fine_y, fine_x]];
                        if fine_x + 1 < width as usize {
                            flow[[fine_y + 1, fine_x + 1]] = flow[[fine_y, fine_x]];
                        }
                    }
                }
            }
        } else {
            flow = level_flow;
        }
    }

    Ok(flow)
}

/// Build image pyramid
fn build_pyramid(img: &GrayImage, levels: usize) -> Vec<GrayImage> {
    let mut pyramid = vec![img.clone()];

    for _ in 1..levels {
        let prev = &pyramid[pyramid.len() - 1];
        let (width, height) = prev.dimensions();
        let new_width = width / 2;
        let new_height = height / 2;

        let mut downsampled = ImageBuffer::new(new_width, new_height);

        for y in 0..new_height {
            for x in 0..new_width {
                // Simple 2x2 average
                let x2 = x * 2;
                let y2 = y * 2;

                let sum = prev.get_pixel(x2, y2)[0] as u32
                    + prev.get_pixel(x2 + 1, y2)[0] as u32
                    + prev.get_pixel(x2, y2 + 1)[0] as u32
                    + prev.get_pixel(x2 + 1, y2 + 1)[0] as u32;

                downsampled.put_pixel(x, y, Luma([(sum / 4) as u8]));
            }
        }

        pyramid.push(downsampled);
    }

    pyramid
}

/// Convert image to float array
fn image_to_float_array(img: &GrayImage) -> Array2<f32> {
    let (width, height) = img.dimensions();
    let mut array = Array2::zeros((height as usize, width as usize));

    for y in 0..height {
        for x in 0..width {
            array[[y as usize, x as usize]] = img.get_pixel(x, y)[0] as f32 / 255.0;
        }
    }

    array
}

/// Compute image gradients using Scharr operator
fn compute_gradients(img: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let (height, width) = img.dim();
    let mut ix = Array2::zeros((height, width));
    let mut iy = Array2::zeros((height, width));

    // Scharr kernels
    let scharr_x = [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]];
    let scharr_y = [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx = 0.0;
            let mut gy = 0.0;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    let pixel = img[[(y as i32 + dy) as usize, (x as i32 + dx) as usize]];
                    gx += pixel * scharr_x[(dy + 1) as usize][(dx + 1) as usize] / 32.0;
                    gy += pixel * scharr_y[(dy + 1) as usize][(dx + 1) as usize] / 32.0;
                }
            }

            ix[[y, x]] = gx;
            iy[[y, x]] = gy;
        }
    }

    (ix, iy)
}

/// Visualize optical flow as color image
///
/// # Arguments
///
/// * `flow` - Flow field
/// * `max_flow` - Maximum flow magnitude for scaling (None for auto)
///
/// # Returns
///
/// * RGB image with flow visualization
pub fn visualize_flow(flow: &Array2<FlowVector>, max_flow: Option<f32>) -> RgbImage {
    let (height, width) = flow.dim();
    let mut result = RgbImage::new(width as u32, height as u32);

    // Find maximum flow if not provided
    let max_magnitude = if let Some(max) = max_flow {
        max
    } else {
        let mut max = 0.0f32;
        for flow_vec in flow.iter() {
            let magnitude = (flow_vec.u.powi(2) + flow_vec.v.powi(2)).sqrt();
            if magnitude > max {
                max = magnitude;
            }
        }
        max.max(1.0) // Avoid division by zero
    };

    for y in 0..height {
        for x in 0..width {
            let flow_vec = &flow[[y, x]];
            let magnitude = (flow_vec.u.powi(2) + flow_vec.v.powi(2)).sqrt();
            let angle = flow_vec.v.atan2(flow_vec.u);

            // Convert to HSV color
            let hue = (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
            let saturation = (magnitude / max_magnitude).min(1.0);
            let value = saturation; // Or use 1.0 for constant brightness

            // Convert HSV to RGB
            let (r, g, b) = hsv_to_rgb(hue, saturation, value);
            result.put_pixel(
                x as u32,
                y as u32,
                Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]),
            );
        }
    }

    result
}

/// Convert HSV to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h * 6.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (r + m, g + m, b + m)
}

/// Dense optical flow using Farneback method (simplified version)
pub fn farneback_flow(
    img1: &DynamicImage,
    img2: &DynamicImage,
    _pyr_scale: f32,
    _levels: usize,
    winsize: usize,
    _iterations: usize,
) -> Result<Array2<FlowVector>> {
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();
    let (width, height) = gray1.dimensions();

    // Initialize flow
    let mut flow = Array2::from_elem(
        (height as usize, width as usize),
        FlowVector { u: 0.0, v: 0.0 },
    );

    // Simplified dense flow computation
    let i1 = image_to_float_array(&gray1);
    let i2 = image_to_float_array(&gray2);
    let (ix, iy) = compute_gradients(&i1);

    let half_win = winsize / 2;

    for y in half_win..height as usize - half_win {
        for x in half_win..width as usize - half_win {
            // Extract windows
            let win_ix = ix.slice(s![y - half_win..=y + half_win, x - half_win..=x + half_win]);
            let win_iy = iy.slice(s![y - half_win..=y + half_win, x - half_win..=x + half_win]);

            // Compute structure tensor
            let mut ixx = 0.0;
            let mut ixy = 0.0;
            let mut iyy = 0.0;

            for (&ix_val, &iy_val) in win_ix.iter().zip(win_iy.iter()) {
                ixx += ix_val * ix_val;
                ixy += ix_val * iy_val;
                iyy += iy_val * iy_val;
            }

            let det = ixx * iyy - ixy * ixy;
            if det > 1e-6 {
                // Simplified flow computation
                let win_i1 = i1.slice(s![y - half_win..=y + half_win, x - half_win..=x + half_win]);
                let win_i2 = i2.slice(s![y - half_win..=y + half_win, x - half_win..=x + half_win]);

                let mut bx = 0.0;
                let mut by = 0.0;

                for ((&i1_val, &i2_val), (&ix_val, &iy_val)) in win_i1
                    .iter()
                    .zip(win_i2.iter())
                    .zip(win_ix.iter().zip(win_iy.iter()))
                {
                    let it = i2_val - i1_val;
                    bx -= ix_val * it;
                    by -= iy_val * it;
                }

                let inv_det = 1.0 / det;
                flow[[y, x]] = FlowVector {
                    u: inv_det * (iyy * bx - ixy * by),
                    v: inv_det * (-ixy * bx + ixx * by),
                };
            }
        }
    }

    Ok(flow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lucas_kanade_basic() {
        let img1 = DynamicImage::new_luma8(50, 50);
        let img2 = img1.clone();

        let flow = lucas_kanade_flow(&img1, &img2, None, &LucasKanadeParams::default()).unwrap();
        assert_eq!(flow.dim(), (50, 50));

        // Flow should be zero for identical images
        for flow_vec in flow.iter() {
            assert!(flow_vec.u.abs() < 0.1);
            assert!(flow_vec.v.abs() < 0.1);
        }
    }

    #[test]
    fn test_pyramid_building() {
        let img = GrayImage::new(64, 64);
        let pyramid = build_pyramid(&img, 3);

        assert_eq!(pyramid.len(), 3);
        assert_eq!(pyramid[0].dimensions(), (64, 64));
        assert_eq!(pyramid[1].dimensions(), (32, 32));
        assert_eq!(pyramid[2].dimensions(), (16, 16));
    }

    #[test]
    fn test_flow_visualization() {
        let mut flow = Array2::from_elem((10, 10), FlowVector { u: 0.0, v: 0.0 });
        flow[[5, 5]] = FlowVector { u: 1.0, v: 0.0 };

        let vis = visualize_flow(&flow, Some(1.0));
        assert_eq!(vis.dimensions(), (10, 10));
    }
}
