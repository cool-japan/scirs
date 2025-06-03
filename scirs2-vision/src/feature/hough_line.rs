//! Hough line detection
//!
//! This module provides functionality for detecting lines in images using
//! the Hough transform.

use crate::error::Result;
use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage};
use ndarray::Array2;
use std::collections::HashMap;
use std::f32::consts::PI;

/// A detected line in Hough space
#[derive(Debug, Clone, Copy)]
pub struct HoughLine {
    /// Distance from origin to the line
    pub rho: f32,
    /// Angle of the line normal (in radians)
    pub theta: f32,
    /// Accumulator value (strength of the line)
    pub votes: u32,
}

/// Parameters for Hough line detection
#[derive(Debug, Clone)]
pub struct HoughParams {
    /// Resolution of rho in pixels
    pub rho_resolution: f32,
    /// Resolution of theta in radians
    pub theta_resolution: f32,
    /// Minimum number of votes to consider a line
    pub threshold: u32,
    /// Minimum line length (for probabilistic Hough)
    pub min_line_length: f32,
    /// Maximum gap between line segments
    pub max_line_gap: f32,
}

impl Default for HoughParams {
    fn default() -> Self {
        Self {
            rho_resolution: 1.0,
            theta_resolution: PI / 180.0, // 1 degree
            threshold: 100,
            min_line_length: 50.0,
            max_line_gap: 10.0,
        }
    }
}

/// Detect lines using the standard Hough transform
///
/// # Arguments
///
/// * `edges` - Binary edge image (non-zero pixels are edges)
/// * `params` - Hough transform parameters
///
/// # Returns
///
/// * Result containing detected lines
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::{hough_lines, canny, HoughParams, PreprocessMode};
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("examples/input/input.jpg").unwrap();
/// let edges = canny(&img, 1.0, Some(50.0), Some(100.0), None, false, PreprocessMode::Constant(0.0))?;
/// let lines = hough_lines(&edges, &HoughParams::default())?;
/// # Ok(())
/// # }
/// ```
pub fn hough_lines(edges: &GrayImage, params: &HoughParams) -> Result<Vec<HoughLine>> {
    let (width, height) = edges.dimensions();

    // Calculate rho range
    let max_rho = ((width * width + height * height) as f32).sqrt();
    let n_rho = (2.0 * max_rho / params.rho_resolution).ceil() as usize;
    let n_theta = (PI / params.theta_resolution).ceil() as usize;

    // Create accumulator
    let mut accumulator = Array2::zeros((n_rho, n_theta));

    // Vote in Hough space
    for y in 0..height {
        for x in 0..width {
            if edges.get_pixel(x, y)[0] > 0 {
                // For each edge point, vote for all possible lines through it
                for theta_idx in 0..n_theta {
                    let theta = theta_idx as f32 * params.theta_resolution;
                    let rho = x as f32 * theta.cos() + y as f32 * theta.sin();

                    // Convert rho to accumulator index
                    let rho_idx = ((rho + max_rho) / params.rho_resolution) as usize;
                    if rho_idx < n_rho {
                        accumulator[[rho_idx, theta_idx]] += 1;
                    }
                }
            }
        }
    }

    // Find peaks in accumulator
    let mut lines = Vec::new();

    for rho_idx in 1..n_rho - 1 {
        for theta_idx in 1..n_theta - 1 {
            let votes = accumulator[[rho_idx, theta_idx]];

            if votes >= params.threshold {
                // Check if it's a local maximum
                let mut is_peak = true;
                for dr in -1..=1 {
                    for dt in -1..=1 {
                        if dr == 0 && dt == 0 {
                            continue;
                        }
                        let r = (rho_idx as i32 + dr) as usize;
                        let t = (theta_idx as i32 + dt) as usize;
                        if r < n_rho && t < n_theta && accumulator[[r, t]] > votes {
                            is_peak = false;
                            break;
                        }
                    }
                    if !is_peak {
                        break;
                    }
                }

                if is_peak {
                    let rho = (rho_idx as f32 * params.rho_resolution) - max_rho;
                    let theta = theta_idx as f32 * params.theta_resolution;
                    lines.push(HoughLine { rho, theta, votes });
                }
            }
        }
    }

    // Sort by votes (strongest lines first)
    lines.sort_by(|a, b| b.votes.cmp(&a.votes));

    Ok(lines)
}

/// Line segment with endpoints
#[derive(Debug, Clone, Copy)]
pub struct LineSegment {
    /// X coordinate of start point
    pub x1: f32,
    /// Y coordinate of start point
    pub y1: f32,
    /// X coordinate of end point
    pub x2: f32,
    /// Y coordinate of end point
    pub y2: f32,
    /// Line strength
    pub strength: f32,
}

/// Detect line segments using probabilistic Hough transform
///
/// # Arguments
///
/// * `edges` - Binary edge image
/// * `params` - Hough transform parameters
///
/// # Returns
///
/// * Result containing detected line segments
pub fn hough_lines_p(edges: &GrayImage, params: &HoughParams) -> Result<Vec<LineSegment>> {
    let (width, height) = edges.dimensions();
    let mut segments = Vec::new();

    // Create a copy of edges to mark visited pixels
    let mut edge_map = edges.clone();

    // Collect all edge points
    let mut edge_points = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if edge_map.get_pixel(x, y)[0] > 0 {
                edge_points.push((x, y));
            }
        }
    }

    // Process edge points randomly
    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    edge_points.shuffle(&mut rng);

    for &(x, y) in &edge_points {
        if edge_map.get_pixel(x, y)[0] == 0 {
            continue; // Already processed
        }

        // Try to find a line through this point
        let max_rho = ((width * width + height * height) as f32).sqrt();
        let n_theta = (PI / params.theta_resolution).ceil() as usize;

        // Vote for lines through this point
        let mut line_votes: HashMap<(i32, usize), u32> = HashMap::new();

        // Check neighborhood for supporting points
        let search_radius = 10;
        for dy in -search_radius..=search_radius {
            for dx in -search_radius..=search_radius {
                let nx = (x as i32 + dx).max(0).min(width as i32 - 1) as u32;
                let ny = (y as i32 + dy).max(0).min(height as i32 - 1) as u32;

                if edge_map.get_pixel(nx, ny)[0] > 0 {
                    // Vote for lines through both points
                    for theta_idx in 0..n_theta {
                        let theta = theta_idx as f32 * params.theta_resolution;
                        let rho = nx as f32 * theta.cos() + ny as f32 * theta.sin();
                        let rho_idx = ((rho + max_rho) / params.rho_resolution) as i32;

                        *line_votes.entry((rho_idx, theta_idx)).or_insert(0) += 1;
                    }
                }
            }
        }

        // Find the best line
        if let Some((&(rho_idx, theta_idx), &votes)) = line_votes.iter().max_by_key(|&(_, &v)| v) {
            if votes >= params.threshold / 10 {
                let theta = theta_idx as f32 * params.theta_resolution;
                let rho = (rho_idx as f32 * params.rho_resolution) - max_rho;

                // Extract line segment
                if let Some(segment) = extract_line_segment(&mut edge_map, rho, theta, params) {
                    if segment_length(&segment) >= params.min_line_length {
                        segments.push(segment);
                    }
                }
            }
        }
    }

    Ok(segments)
}

/// Extract a line segment from edge map
fn extract_line_segment(
    edge_map: &mut GrayImage,
    rho: f32,
    theta: f32,
    params: &HoughParams,
) -> Option<LineSegment> {
    let (width, height) = edge_map.dimensions();
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();

    let mut points = Vec::new();

    // Find all points on the line
    if sin_theta.abs() > cos_theta.abs() {
        // More vertical line
        for x in 0..width {
            let y = ((rho - x as f32 * cos_theta) / sin_theta).round() as i32;
            if y >= 0 && y < height as i32 && edge_map.get_pixel(x, y as u32)[0] > 0 {
                points.push((x as f32, y as f32));
                edge_map.put_pixel(x, y as u32, Luma([0])); // Mark as visited
            }
        }
    } else {
        // More horizontal line
        for y in 0..height {
            let x = ((rho - y as f32 * sin_theta) / cos_theta).round() as i32;
            if x >= 0 && x < width as i32 && edge_map.get_pixel(x as u32, y)[0] > 0 {
                points.push((x as f32, y as f32));
                edge_map.put_pixel(x as u32, y, Luma([0])); // Mark as visited
            }
        }
    }

    if points.len() < 2 {
        return None;
    }

    // Find connected segments
    points.sort_by(|a, b| {
        let dist_a = a.0 * a.0 + a.1 * a.1;
        let dist_b = b.0 * b.0 + b.1 * b.1;
        dist_a.partial_cmp(&dist_b).unwrap()
    });

    let mut segments = Vec::new();
    let mut start = points[0];
    let mut end = points[0];

    for &point in points.iter().skip(1) {
        let dist = ((point.0 - end.0).powi(2) + (point.1 - end.1).powi(2)).sqrt();

        if dist <= params.max_line_gap {
            end = point;
        } else {
            // Gap too large, save current segment
            let length = ((end.0 - start.0).powi(2) + (end.1 - start.1).powi(2)).sqrt();
            if length >= params.min_line_length {
                segments.push(LineSegment {
                    x1: start.0,
                    y1: start.1,
                    x2: end.0,
                    y2: end.1,
                    strength: length,
                });
            }
            start = point;
            end = point;
        }
    }

    // Don't forget the last segment
    let length = ((end.0 - start.0).powi(2) + (end.1 - start.1).powi(2)).sqrt();
    if length >= params.min_line_length {
        segments.push(LineSegment {
            x1: start.0,
            y1: start.1,
            x2: end.0,
            y2: end.1,
            strength: length,
        });
    }

    // Return the longest segment
    segments
        .into_iter()
        .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap())
}

/// Calculate segment length
fn segment_length(segment: &LineSegment) -> f32 {
    ((segment.x2 - segment.x1).powi(2) + (segment.y2 - segment.y1).powi(2)).sqrt()
}

/// Draw detected lines on an image
///
/// # Arguments
///
/// * `img` - Input image
/// * `lines` - Detected lines
/// * `color` - Line color
///
/// # Returns
///
/// * Image with lines drawn
pub fn draw_lines(img: &DynamicImage, lines: &[HoughLine], color: [u8; 3]) -> RgbImage {
    let mut result = img.to_rgb8();
    let (width, height) = result.dimensions();

    for line in lines {
        draw_hough_line(&mut result, line.rho, line.theta, width, height, color);
    }

    result
}

/// Draw detected line segments on an image
pub fn draw_line_segments(
    img: &DynamicImage,
    segments: &[LineSegment],
    color: [u8; 3],
) -> RgbImage {
    let mut result = img.to_rgb8();

    for segment in segments {
        draw_line_segment(&mut result, segment, color);
    }

    result
}

/// Draw a single line in Hough space representation
fn draw_hough_line(
    img: &mut RgbImage,
    rho: f32,
    theta: f32,
    width: u32,
    height: u32,
    color: [u8; 3],
) {
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let rgb_color = Rgb(color);

    if sin_theta.abs() > 0.001 {
        // Draw for each x
        for x in 0..width {
            let y = ((rho - x as f32 * cos_theta) / sin_theta).round() as i32;
            if y >= 0 && y < height as i32 {
                img.put_pixel(x, y as u32, rgb_color);
            }
        }
    }

    if cos_theta.abs() > 0.001 {
        // Draw for each y
        for y in 0..height {
            let x = ((rho - y as f32 * sin_theta) / cos_theta).round() as i32;
            if x >= 0 && x < width as i32 {
                img.put_pixel(x as u32, y, rgb_color);
            }
        }
    }
}

/// Draw a line segment
fn draw_line_segment(img: &mut RgbImage, segment: &LineSegment, color: [u8; 3]) {
    let rgb_color = Rgb(color);

    // Bresenham's line algorithm
    let mut x0 = segment.x1.round() as i32;
    let mut y0 = segment.y1.round() as i32;
    let x1 = segment.x2.round() as i32;
    let y1 = segment.y2.round() as i32;

    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;

    let (width, height) = img.dimensions();

    loop {
        if x0 >= 0 && x0 < width as i32 && y0 >= 0 && y0 < height as i32 {
            img.put_pixel(x0 as u32, y0 as u32, rgb_color);
        }

        if x0 == x1 && y0 == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x0 += sx;
        }
        if e2 < dx {
            err += dx;
            y0 += sy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hough_lines() {
        // Create a simple edge image with a diagonal line
        let mut edges = GrayImage::new(50, 50);
        for i in 0..50 {
            edges.put_pixel(i, i, Luma([255]));
        }

        let params = HoughParams {
            threshold: 30,
            ..Default::default()
        };

        let lines = hough_lines(&edges, &params).unwrap();
        assert!(!lines.is_empty());

        // Should detect a line at approximately 45 degrees
        let best_line = &lines[0];
        // The diagonal line from (0,0) to (49,49) could be detected at different angles
        // depending on the coordinate system orientation
        let angle_diff_1 = (best_line.theta - PI / 4.0).abs();
        let angle_diff_2 = (best_line.theta - 3.0 * PI / 4.0).abs();
        assert!(
            angle_diff_1 < 0.1 || angle_diff_2 < 0.1,
            "Expected angle near π/4 or 3π/4, got {}",
            best_line.theta
        );
    }

    #[test]
    fn test_line_segments() {
        let segment = LineSegment {
            x1: 0.0,
            y1: 0.0,
            x2: 3.0,
            y2: 4.0,
            strength: 1.0,
        };

        assert_eq!(segment_length(&segment), 5.0);
    }
}
