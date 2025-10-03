//! Hough Circle Transform
//!
//! Detects circles in images using the Hough transform approach.

use crate::error::Result;
use image::{DynamicImage, GrayImage, Luma};
use scirs2_core::ndarray::{Array2, Array3};

/// Configuration for Hough Circle detection
#[derive(Clone, Debug)]
pub struct HoughCircleConfig {
    /// Minimum circle radius
    pub min_radius: usize,
    /// Maximum circle radius
    pub max_radius: usize,
    /// Accumulator threshold
    pub threshold: f32,
    /// Minimum distance between circle centers
    pub mindistance: usize,
    /// Edge magnitude threshold
    pub edge_threshold: f32,
    /// Maximum number of circles to return
    pub max_circles: Option<usize>,
}

impl Default for HoughCircleConfig {
    fn default() -> Self {
        Self {
            min_radius: 10,
            max_radius: 100,
            threshold: 0.3,
            mindistance: 20,
            edge_threshold: 0.1,
            max_circles: None,
        }
    }
}

/// Represents a detected circle
#[derive(Clone, Debug)]
pub struct Circle {
    /// X coordinate of the circle center
    pub center_x: usize,
    /// Y coordinate of the circle center
    pub center_y: usize,
    /// Radius of the circle
    pub radius: usize,
    /// Confidence score of the detection
    pub confidence: f32,
}

/// Detect circles using Hough Transform
#[allow(dead_code)]
pub fn hough_circles(img: &DynamicImage, config: HoughCircleConfig) -> Result<Vec<Circle>> {
    let gray = img.to_luma8();
    let (width, height) = (gray.width() as usize, gray.height() as usize);

    // Compute edge map and gradients
    let (edges, grad_x, grad_y) = compute_edges(&gray)?;

    // Create accumulator array
    let num_radii = config.max_radius - config.min_radius + 1;
    let mut accumulator = Array3::<f32>::zeros((height, width, num_radii));

    // Vote in parameter space
    for y in 0..height {
        for x in 0..width {
            if edges[[y, x]] < config.edge_threshold {
                continue;
            }

            // Calculate gradient direction
            let gx = grad_x[[y, x]];
            let gy = grad_y[[y, x]];
            let magnitude = (gx * gx + gy * gy).sqrt();

            if magnitude < 1e-6 {
                continue;
            }

            let dx = gx / magnitude;
            let dy = gy / magnitude;

            // Vote for circles along gradient direction
            for r in config.min_radius..=config.max_radius {
                let r_idx = r - config.min_radius;

                // Vote for positive direction
                let cx1 = x as f32 + r as f32 * dx;
                let cy1 = y as f32 + r as f32 * dy;

                if cx1 >= 0.0 && cx1 < width as f32 && cy1 >= 0.0 && cy1 < height as f32 {
                    let cx1 = cx1 as usize;
                    let cy1 = cy1 as usize;
                    accumulator[[cy1, cx1, r_idx]] += edges[[y, x]];
                }

                // Vote for negative direction
                let cx2 = x as f32 - r as f32 * dx;
                let cy2 = y as f32 - r as f32 * dy;

                if cx2 >= 0.0 && cx2 < width as f32 && cy2 >= 0.0 && cy2 < height as f32 {
                    let cx2 = cx2 as usize;
                    let cy2 = cy2 as usize;
                    accumulator[[cy2, cx2, r_idx]] += edges[[y, x]];
                }
            }
        }
    }

    // Find local maxima in accumulator
    let mut circles = find_circles(&accumulator, config.threshold)?;

    // Adjust indices for minimum radius offset
    for circle in &mut circles {
        circle.radius += config.min_radius;
    }

    // Apply non-maximum suppression
    circles = non_max_suppression(circles, config.mindistance);

    // Limit number of circles if specified
    if let Some(max_circles) = config.max_circles {
        circles.truncate(max_circles);
    }

    Ok(circles)
}

/// Draw circles on an image
#[allow(dead_code)]
pub fn draw_circles(img: &mut GrayImage, circles: &[Circle], intensity: u8) {
    for circle in circles {
        draw_circle(
            img,
            circle.center_x as i32,
            circle.center_y as i32,
            circle.radius as i32,
            intensity,
        );
    }
}

// Helper functions

#[allow(dead_code)]
fn compute_edges(img: &GrayImage) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
    let (height, width) = (img.height() as usize, img.width() as usize);
    let mut edges = Array2::<f32>::zeros((height, width));
    let mut grad_x = Array2::<f32>::zeros((height, width));
    let mut grad_y = Array2::<f32>::zeros((height, width));

    // Apply Sobel operator
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let (x_u32, y_u32) = (x as u32, y as u32);

            // Sobel X
            let gx = -(img.get_pixel(x_u32 - 1, y_u32 - 1).0[0] as f32)
                + 0.0 * img.get_pixel(x_u32, y_u32 - 1).0[0] as f32
                + img.get_pixel(x_u32 + 1, y_u32 - 1).0[0] as f32
                + -2.0 * img.get_pixel(x_u32 - 1, y_u32).0[0] as f32
                + 0.0 * img.get_pixel(x_u32, y_u32).0[0] as f32
                + 2.0 * img.get_pixel(x_u32 + 1, y_u32).0[0] as f32
                + -(img.get_pixel(x_u32 - 1, y_u32 + 1).0[0] as f32)
                + 0.0 * img.get_pixel(x_u32, y_u32 + 1).0[0] as f32
                + img.get_pixel(x_u32 + 1, y_u32 + 1).0[0] as f32;

            // Sobel Y
            let gy = -(img.get_pixel(x_u32 - 1, y_u32 - 1).0[0] as f32)
                + -2.0 * img.get_pixel(x_u32, y_u32 - 1).0[0] as f32
                + -(img.get_pixel(x_u32 + 1, y_u32 - 1).0[0] as f32)
                + 0.0 * img.get_pixel(x_u32 - 1, y_u32).0[0] as f32
                + 0.0 * img.get_pixel(x_u32, y_u32).0[0] as f32
                + 0.0 * img.get_pixel(x_u32 + 1, y_u32).0[0] as f32
                + 1.0 * img.get_pixel(x_u32 - 1, y_u32 + 1).0[0] as f32
                + 2.0 * img.get_pixel(x_u32, y_u32 + 1).0[0] as f32
                + 1.0 * img.get_pixel(x_u32 + 1, y_u32 + 1).0[0] as f32;

            grad_x[[y, x]] = gx / 255.0;
            grad_y[[y, x]] = gy / 255.0;
            edges[[y, x]] = (gx * gx + gy * gy).sqrt() / 255.0;
        }
    }

    Ok((edges, grad_x, grad_y))
}

#[allow(dead_code)]
fn find_circles(accumulator: &Array3<f32>, threshold: f32) -> Result<Vec<Circle>> {
    let (height, width, num_radii) = accumulator.dim();
    let mut circles = Vec::new();

    // Find local maxima
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            for r in 1..num_radii - 1 {
                let val = accumulator[[y, x, r]];

                if val < threshold {
                    continue;
                }

                // Check if it's a local maximum
                let mut is_max = true;

                for dy in -1..=1 {
                    for dx in -1..=1 {
                        for dr in -1..=1 {
                            if dy == 0 && dx == 0 && dr == 0 {
                                continue;
                            }

                            let ny = (y as i32 + dy) as usize;
                            let nx = (x as i32 + dx) as usize;
                            let nr = (r as i32 + dr) as usize;

                            if accumulator[[ny, nx, nr]] >= val {
                                is_max = false;
                                break;
                            }
                        }
                        if !is_max {
                            break;
                        }
                    }
                    if !is_max {
                        break;
                    }
                }

                if is_max {
                    circles.push(Circle {
                        center_x: x,
                        center_y: y,
                        radius: r,
                        confidence: val,
                    });
                }
            }
        }
    }

    Ok(circles)
}

#[allow(dead_code)]
fn non_max_suppression(mut circles: Vec<Circle>, mindistance: usize) -> Vec<Circle> {
    // Sort by confidence
    circles.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept = Vec::new();
    let mut suppressed = vec![false; circles.len()];

    for i in 0..circles.len() {
        if suppressed[i] {
            continue;
        }

        kept.push(circles[i].clone());

        // Suppress nearby circles
        for j in i + 1..circles.len() {
            if suppressed[j] {
                continue;
            }

            let dx = circles[i].center_x as f32 - circles[j].center_x as f32;
            let dy = circles[i].center_y as f32 - circles[j].center_y as f32;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < mindistance as f32 {
                suppressed[j] = true;
            }
        }
    }

    kept
}

#[allow(dead_code)]
fn draw_circle(img: &mut GrayImage, cx: i32, cy: i32, radius: i32, intensity: u8) {
    let (width, height) = (img.width() as i32, img.height() as i32);

    // Draw circle using Bresenham's algorithm
    let mut x = 0;
    let mut y = radius;
    let mut d = 3 - 2 * radius;

    while x <= y {
        // Draw 8 octants
        set_pixel(img, cx + x, cy + y, intensity, width, height);
        set_pixel(img, cx - x, cy + y, intensity, width, height);
        set_pixel(img, cx + x, cy - y, intensity, width, height);
        set_pixel(img, cx - x, cy - y, intensity, width, height);
        set_pixel(img, cx + y, cy + x, intensity, width, height);
        set_pixel(img, cx - y, cy + x, intensity, width, height);
        set_pixel(img, cx + y, cy - x, intensity, width, height);
        set_pixel(img, cx - y, cy - x, intensity, width, height);

        if d < 0 {
            d += 4 * x + 6;
        } else {
            d += 4 * (x - y) + 10;
            y -= 1;
        }
        x += 1;
    }
}

#[allow(dead_code)]
fn set_pixel(img: &mut GrayImage, x: i32, y: i32, intensity: u8, width: i32, height: i32) {
    if x >= 0 && x < width && y >= 0 && y < height {
        img.put_pixel(x as u32, y as u32, Luma([intensity]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hough_circle_config_default() {
        let config = HoughCircleConfig::default();
        assert_eq!(config.min_radius, 10);
        assert_eq!(config.max_radius, 100);
        assert_eq!(config.threshold, 0.3);
        assert_eq!(config.mindistance, 20);
        assert_eq!(config.edge_threshold, 0.1);
        assert!(config.max_circles.is_none());
    }

    #[test]
    fn test_circle_creation() {
        let circle = Circle {
            center_x: 50,
            center_y: 75,
            radius: 20,
            confidence: 0.85,
        };

        assert_eq!(circle.center_x, 50);
        assert_eq!(circle.center_y, 75);
        assert_eq!(circle.radius, 20);
        assert_eq!(circle.confidence, 0.85);
    }

    #[test]
    fn test_non_max_suppression_basic() {
        let circles = vec![
            Circle {
                center_x: 10,
                center_y: 10,
                radius: 5,
                confidence: 0.9,
            },
            Circle {
                center_x: 12,
                center_y: 12,
                radius: 5,
                confidence: 0.8,
            },
            Circle {
                center_x: 50,
                center_y: 50,
                radius: 10,
                confidence: 0.7,
            },
        ];

        let result = non_max_suppression(circles, 10);

        // Should keep the first (highest confidence) and third (far away)
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].center_x, 10);
        assert_eq!(result[1].center_x, 50);
    }
}
