//! Region growing segmentation algorithm
//!
//! Region growing is a simple region-based image segmentation method that
//! groups pixels or subregions into larger regions based on predefined criteria.

use crate::error::Result;
use image::{DynamicImage, GrayImage, Rgb, RgbImage};
use ndarray::Array2;
use std::collections::{HashSet, VecDeque};

/// Parameters for region growing
#[derive(Debug, Clone)]
pub struct RegionGrowingParams {
    /// Similarity threshold for adding pixels to region
    pub threshold: f32,
    /// Minimum region size to keep
    pub min_region_size: usize,
    /// Connectivity (4 or 8)
    pub connectivity: u8,
}

impl Default for RegionGrowingParams {
    fn default() -> Self {
        Self {
            threshold: 10.0,
            min_region_size: 50,
            connectivity: 8,
        }
    }
}

/// Seed point for region growing
#[derive(Debug, Clone, Copy)]
pub struct SeedPoint {
    /// X coordinate
    pub x: usize,
    /// Y coordinate  
    pub y: usize,
    /// Optional label for this seed
    pub label: Option<u32>,
}

/// Perform region growing segmentation
///
/// # Arguments
///
/// * `img` - Input image
/// * `seeds` - Initial seed points
/// * `params` - Region growing parameters
///
/// # Returns
///
/// * Result containing segmented label map
///
/// # Example
///
/// ```rust
/// use scirs2_vision::segmentation::{region_growing, RegionGrowingParams, SeedPoint};
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("examples/input/input.jpg").unwrap();
/// let seeds = vec![
///     SeedPoint { x: 100, y: 100, label: Some(1) },
///     SeedPoint { x: 200, y: 200, label: Some(2) },
/// ];
/// let labels = region_growing(&img, &seeds, &RegionGrowingParams::default())?;
/// # Ok(())
/// # }
/// ```
pub fn region_growing(
    img: &DynamicImage,
    seeds: &[SeedPoint],
    params: &RegionGrowingParams,
) -> Result<Array2<u32>> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Initialize label map
    let mut labels = Array2::zeros((height as usize, width as usize));
    let mut visited = Array2::from_elem((height as usize, width as usize), false);

    // Process each seed point
    for (seed_idx, seed) in seeds.iter().enumerate() {
        if seed.x >= width as usize || seed.y >= height as usize {
            continue;
        }

        if visited[[seed.y, seed.x]] {
            continue;
        }

        let label = seed.label.unwrap_or((seed_idx + 1) as u32);
        grow_region(
            &gray,
            &mut labels,
            &mut visited,
            seed.x,
            seed.y,
            label,
            params,
        );
    }

    // Auto-generate seeds if none provided
    if seeds.is_empty() {
        auto_seeded_region_growing(&gray, &mut labels, &mut visited, params)?;
    }

    // Remove small regions
    remove_small_regions(&mut labels, params.min_region_size);

    Ok(labels)
}

/// Grow a single region from a seed point
fn grow_region(
    img: &GrayImage,
    labels: &mut Array2<u32>,
    visited: &mut Array2<bool>,
    seed_x: usize,
    seed_y: usize,
    label: u32,
    params: &RegionGrowingParams,
) {
    let (width, height) = img.dimensions();
    let seed_value = img.get_pixel(seed_x as u32, seed_y as u32)[0] as f32;

    let mut queue = VecDeque::new();
    queue.push_back((seed_x, seed_y));
    visited[[seed_y, seed_x]] = true;
    labels[[seed_y, seed_x]] = label;

    let neighbors = get_neighbors(params.connectivity);

    while let Some((x, y)) = queue.pop_front() {
        for &(dx, dy) in &neighbors {
            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;

            if nx < width as usize && ny < height as usize && !visited[[ny, nx]] {
                let pixel_value = img.get_pixel(nx as u32, ny as u32)[0] as f32;

                if (pixel_value - seed_value).abs() <= params.threshold {
                    visited[[ny, nx]] = true;
                    labels[[ny, nx]] = label;
                    queue.push_back((nx, ny));
                }
            }
        }
    }
}

/// Auto-generate seeds and perform region growing
fn auto_seeded_region_growing(
    img: &GrayImage,
    labels: &mut Array2<u32>,
    visited: &mut Array2<bool>,
    params: &RegionGrowingParams,
) -> Result<()> {
    let (width, height) = img.dimensions();
    let mut current_label = 1u32;

    // Grid-based seed generation
    let grid_size = ((width * height) as f32 / 100.0).sqrt() as usize;
    let grid_size = grid_size.max(10);

    for y in (0..height as usize).step_by(grid_size) {
        for x in (0..width as usize).step_by(grid_size) {
            if !visited[[y, x]] {
                grow_region(img, labels, visited, x, y, current_label, params);
                current_label += 1;
            }
        }
    }

    Ok(())
}

/// Get neighbor offsets based on connectivity
fn get_neighbors(connectivity: u8) -> Vec<(i32, i32)> {
    match connectivity {
        4 => vec![(-1, 0), (1, 0), (0, -1), (0, 1)],
        8 => vec![
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ],
        _ => vec![(-1, 0), (1, 0), (0, -1), (0, 1)], // Default to 4-connectivity
    }
}

/// Remove regions smaller than minimum size
fn remove_small_regions(labels: &mut Array2<u32>, min_size: usize) {
    let (height, width) = labels.dim();

    // Count region sizes
    let mut region_sizes = std::collections::HashMap::new();
    for &label in labels.iter() {
        if label > 0 {
            *region_sizes.entry(label).or_insert(0) += 1;
        }
    }

    // Find small regions
    let small_regions: HashSet<_> = region_sizes
        .iter()
        .filter(|(_, &size)| size < min_size)
        .map(|(&label, _)| label)
        .collect();

    // Remove small regions
    for y in 0..height {
        for x in 0..width {
            if small_regions.contains(&labels[[y, x]]) {
                labels[[y, x]] = 0;
            }
        }
    }
}

/// Region growing with adaptive threshold
///
/// The threshold is adapted based on local statistics
pub fn adaptive_region_growing(
    img: &DynamicImage,
    seeds: &[SeedPoint],
    base_threshold: f32,
    window_size: usize,
) -> Result<Array2<u32>> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Compute local statistics
    let (local_mean, local_std) = compute_local_stats(&gray, window_size)?;

    let mut labels = Array2::zeros((height as usize, width as usize));
    let mut visited = Array2::from_elem((height as usize, width as usize), false);

    for (seed_idx, seed) in seeds.iter().enumerate() {
        if seed.x >= width as usize || seed.y >= height as usize {
            continue;
        }

        if visited[[seed.y, seed.x]] {
            continue;
        }

        let label = seed.label.unwrap_or((seed_idx + 1) as u32);

        // Adaptive threshold based on local statistics
        let local_threshold = base_threshold * (1.0 + local_std[[seed.y, seed.x]] / 128.0);

        grow_region_adaptive(
            &gray,
            &mut labels,
            &mut visited,
            seed.x,
            seed.y,
            label,
            local_threshold,
            &local_mean,
            &local_std,
        );
    }

    Ok(labels)
}

/// Grow region with adaptive threshold
fn grow_region_adaptive(
    img: &GrayImage,
    labels: &mut Array2<u32>,
    visited: &mut Array2<bool>,
    seed_x: usize,
    seed_y: usize,
    label: u32,
    base_threshold: f32,
    _local_mean: &Array2<f32>,
    local_std: &Array2<f32>,
) {
    let (width, height) = img.dimensions();
    let mut queue = VecDeque::new();
    queue.push_back((seed_x, seed_y));
    visited[[seed_y, seed_x]] = true;
    labels[[seed_y, seed_x]] = label;

    // Running mean of region
    let mut region_sum = img.get_pixel(seed_x as u32, seed_y as u32)[0] as f32;
    let mut region_count = 1;

    while let Some((x, y)) = queue.pop_front() {
        let region_mean = region_sum / region_count as f32;

        for &(dx, dy) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;

            if nx < width as usize && ny < height as usize && !visited[[ny, nx]] {
                let pixel_value = img.get_pixel(nx as u32, ny as u32)[0] as f32;

                // Adaptive threshold based on local statistics
                let threshold = base_threshold * (1.0 + local_std[[ny, nx]] / 128.0);

                if (pixel_value - region_mean).abs() <= threshold {
                    visited[[ny, nx]] = true;
                    labels[[ny, nx]] = label;
                    queue.push_back((nx, ny));

                    // Update running mean
                    region_sum += pixel_value;
                    region_count += 1;
                }
            }
        }
    }
}

/// Compute local mean and standard deviation
fn compute_local_stats(img: &GrayImage, window_size: usize) -> Result<(Array2<f32>, Array2<f32>)> {
    let (width, height) = img.dimensions();
    let mut local_mean = Array2::zeros((height as usize, width as usize));
    let mut local_std = Array2::zeros((height as usize, width as usize));

    let half_window = window_size / 2;

    for y in 0..height as usize {
        for x in 0..width as usize {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let mut count = 0;

            for wy in y.saturating_sub(half_window)..=(y + half_window).min(height as usize - 1) {
                for wx in x.saturating_sub(half_window)..=(x + half_window).min(width as usize - 1)
                {
                    let val = img.get_pixel(wx as u32, wy as u32)[0] as f32;
                    sum += val;
                    sum_sq += val * val;
                    count += 1;
                }
            }

            let mean = sum / count as f32;
            let variance = (sum_sq / count as f32) - mean * mean;

            local_mean[[y, x]] = mean;
            local_std[[y, x]] = variance.sqrt();
        }
    }

    Ok((local_mean, local_std))
}

/// Convert labels to color image for visualization
pub fn region_labels_to_color(labels: &Array2<u32>) -> RgbImage {
    let (height, width) = labels.dim();
    let mut result = RgbImage::new(width as u32, height as u32);

    // Find unique labels
    let mut unique_labels = HashSet::new();
    for &label in labels.iter() {
        if label > 0 {
            unique_labels.insert(label);
        }
    }

    // Generate colors
    let mut label_colors = std::collections::HashMap::new();
    let golden_ratio = 0.618_034;
    let mut hue = 0.0;

    for &label in &unique_labels {
        hue += golden_ratio;
        hue %= 1.0;

        let (r, g, b) = hsv_to_rgb(hue, 0.7, 0.9);
        label_colors.insert(
            label,
            [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8],
        );
    }

    // Apply colors
    for y in 0..height {
        for x in 0..width {
            let label = labels[[y, x]];
            let color = if label == 0 {
                [0, 0, 0]
            } else {
                label_colors.get(&label).copied().unwrap_or([0, 0, 0])
            };
            result.put_pixel(x as u32, y as u32, Rgb(color));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_growing_basic() {
        // Create an image with two distinct regions
        let mut img_buf = image::GrayImage::new(50, 50);

        // Create two regions with different intensities
        for y in 0..50 {
            for x in 0..50 {
                if x < 25 {
                    img_buf.put_pixel(x, y, image::Luma([50u8]));
                } else {
                    img_buf.put_pixel(x, y, image::Luma([200u8]));
                }
            }
        }

        let img = DynamicImage::ImageLuma8(img_buf);

        let seeds = vec![
            SeedPoint {
                x: 10,
                y: 10,
                label: Some(1),
            },
            SeedPoint {
                x: 30,
                y: 30,
                label: Some(2),
            },
        ];

        let result = region_growing(&img, &seeds, &RegionGrowingParams::default());
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert_eq!(labels.dim(), (50, 50));

        // Seeds should be labeled (note: array indexing is [y, x])
        assert_eq!(labels[[10, 10]], 1);
        assert_eq!(labels[[30, 30]], 2);

        // Check that regions are properly segmented
        assert_eq!(labels[[10, 5]], 1); // Left region
        assert_eq!(labels[[30, 40]], 2); // Right region
    }

    #[test]
    fn test_auto_seeded() {
        let img = DynamicImage::new_luma8(20, 20);
        let result = region_growing(&img, &[], &RegionGrowingParams::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_neighbors() {
        let neighbors_4 = get_neighbors(4);
        assert_eq!(neighbors_4.len(), 4);

        let neighbors_8 = get_neighbors(8);
        assert_eq!(neighbors_8.len(), 8);
    }

    #[test]
    fn test_adaptive_region_growing() {
        let img = DynamicImage::new_luma8(30, 30);
        let seeds = vec![SeedPoint {
            x: 15,
            y: 15,
            label: None,
        }];

        let result = adaptive_region_growing(&img, &seeds, 10.0, 5);
        assert!(result.is_ok());
    }
}
