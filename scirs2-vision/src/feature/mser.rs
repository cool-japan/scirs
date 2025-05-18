//! Maximally Stable Extremal Regions (MSER) detector
//!
//! MSER detects regions that remain stable across multiple thresholds.

use crate::error::Result;
use image::{DynamicImage, GrayImage, Luma};
use std::collections::{HashMap, HashSet};

/// Configuration for MSER detection
#[derive(Clone, Debug)]
pub struct MserConfig {
    /// Size difference threshold
    pub delta: u8,
    /// Minimum region area
    pub min_area: usize,
    /// Maximum region area
    pub max_area: usize,
    /// Maximum variation of region area
    pub max_variation: f32,
    /// Minimum diversity between regions
    pub min_diversity: f32,
}

impl Default for MserConfig {
    fn default() -> Self {
        Self {
            delta: 5,
            min_area: 60,
            max_area: 14400,
            max_variation: 0.25,
            min_diversity: 0.2,
        }
    }
}

/// Represents a detected MSER region
#[derive(Clone, Debug)]
pub struct MserRegion {
    /// List of pixel coordinates belonging to this region
    pub pixels: Vec<(usize, usize)>,
    /// Gray level at which this region was detected
    pub level: u8,
    /// Area of the region in pixels
    pub area: usize,
    /// Stability score of the region
    pub stability: f32,
}

/// Component in the component tree
#[derive(Clone, Debug)]
struct Component {
    pixels: Vec<(usize, usize)>,
    level: u8,
    parent: Option<usize>,
    children: Vec<usize>,
    area: usize,
    stability: f32,
}

/// Detect MSER regions
pub fn mser_detect(img: &DynamicImage, config: MserConfig) -> Result<Vec<MserRegion>> {
    let gray = img.to_luma8();
    let (width, height) = (gray.width() as usize, gray.height() as usize);

    // Sort pixels by gray level
    let mut pixel_levels: Vec<(u8, usize, usize)> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let level = gray.get_pixel(x as u32, y as u32).0[0];
            pixel_levels.push((level, x, y));
        }
    }
    pixel_levels.sort_by_key(|&(level, _, _)| level);

    // Build component tree
    let mut components = Vec::new();
    let mut pixel_component: HashMap<(usize, usize), usize> = HashMap::new();
    let mut active_components: HashMap<u8, Vec<usize>> = HashMap::new();

    // Process pixels in order of increasing gray level
    for (level, x, y) in pixel_levels {
        // Create new component for this pixel
        let comp_idx = components.len();
        components.push(Component {
            pixels: vec![(x, y)],
            level,
            parent: None,
            children: Vec::new(),
            area: 1,
            stability: 0.0,
        });
        pixel_component.insert((x, y), comp_idx);

        // Check 4-connected neighbors
        let neighbors = [
            (x as i32 - 1, y as i32),
            (x as i32 + 1, y as i32),
            (x as i32, y as i32 - 1),
            (x as i32, y as i32 + 1),
        ];

        let mut merged_components = Vec::new();

        for &(nx, ny) in &neighbors {
            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                let nx = nx as usize;
                let ny = ny as usize;

                if let Some(&neighbor_comp) = pixel_component.get(&(nx, ny)) {
                    let neighbor_level = components[neighbor_comp].level;
                    if neighbor_level <= level {
                        merged_components.push(neighbor_comp);
                    }
                }
            }
        }

        // Merge components
        if !merged_components.is_empty() {
            // Find root components
            let mut roots = HashSet::new();
            for &comp in &merged_components {
                roots.insert(find_root(&components, comp));
            }

            match roots.len().cmp(&1) {
                std::cmp::Ordering::Greater => {
                    // Multiple components to merge
                    let mut root_vec: Vec<_> = roots.into_iter().collect();
                    root_vec.sort();

                    // Keep the first as parent, merge others into it
                    let parent_root = root_vec[0];

                    for &child_root in &root_vec[1..] {
                        merge_components(&mut components, parent_root, child_root);
                    }

                    // Add current pixel to parent
                    components[parent_root].pixels.push((x, y));
                    components[parent_root].area += 1;
                    pixel_component.insert((x, y), parent_root);
                }
                std::cmp::Ordering::Equal => {
                    // Single component
                    let root = roots.into_iter().next().unwrap();
                    components[root].pixels.push((x, y));
                    components[root].area += 1;
                    pixel_component.insert((x, y), root);
                }
                std::cmp::Ordering::Less => {
                    // No components (should not happen)
                }
            }
        }

        // Update active components for this level
        active_components.entry(level).or_default().push(comp_idx);
    }

    // Calculate stability for each component
    calculate_stability(&mut components, config.delta);

    // Extract stable regions
    let mut regions = Vec::new();

    for (idx, component) in components.iter().enumerate() {
        if component.area >= config.min_area
            && component.area <= config.max_area
            && component.stability > 0.0
            && component.stability < config.max_variation
        {
            // Check if this is a maximally stable region
            let is_maximal = is_maximally_stable(&components, idx, config.min_diversity);

            if is_maximal {
                regions.push(MserRegion {
                    pixels: component.pixels.clone(),
                    level: component.level,
                    area: component.area,
                    stability: component.stability,
                });
            }
        }
    }

    Ok(regions)
}

/// Convert MSER regions to image
pub fn mser_to_image(regions: &[MserRegion], width: u32, height: u32) -> Result<GrayImage> {
    let mut img = GrayImage::new(width, height);

    // Draw each region with a different intensity
    for (idx, region) in regions.iter().enumerate() {
        let intensity = ((idx * 255) / regions.len().max(1)) as u8;

        for &(x, y) in &region.pixels {
            img.put_pixel(x as u32, y as u32, Luma([intensity]));
        }
    }

    Ok(img)
}

// Helper functions

fn find_root(components: &[Component], mut idx: usize) -> usize {
    while let Some(parent) = components[idx].parent {
        idx = parent;
    }
    idx
}

fn merge_components(components: &mut [Component], parent: usize, child: usize) {
    // Merge child into parent
    let child_pixels = components[child].pixels.clone();
    components[parent].pixels.extend(child_pixels);
    components[parent].area += components[child].area;

    // Update parent-child relationships
    components[child].parent = Some(parent);
    components[parent].children.push(child);
}

fn calculate_stability(components: &mut [Component], delta: u8) {
    for i in 0..components.len() {
        let level = components[i].level;
        let area = components[i].area;

        // Find parent at level + delta
        let mut parent_idx = i;
        let target_level = level.saturating_add(delta);
        while let Some(parent) = components[parent_idx].parent {
            if components[parent].level >= target_level {
                break;
            }
            parent_idx = parent;
        }

        if parent_idx != i {
            let parent_area = components[parent_idx].area;
            let stability = (parent_area as f32 - area as f32) / area as f32;
            components[i].stability = stability;
        }
    }
}

fn is_maximally_stable(components: &[Component], idx: usize, min_diversity: f32) -> bool {
    let component = &components[idx];

    // Check parent
    if let Some(parent_idx) = component.parent {
        let parent = &components[parent_idx];
        if parent.stability <= component.stability {
            return false;
        }

        // Check diversity
        let diversity = (parent.area as f32 - component.area as f32) / component.area as f32;
        if diversity < min_diversity {
            return false;
        }
    }

    // Check children
    for &child_idx in &component.children {
        let child = &components[child_idx];
        if child.stability >= component.stability {
            return false;
        }

        // Check diversity
        let diversity = (component.area as f32 - child.area as f32) / child.area as f32;
        if diversity < min_diversity {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mser_config_default() {
        let config = MserConfig::default();
        assert_eq!(config.delta, 5);
        assert_eq!(config.min_area, 60);
        assert_eq!(config.max_area, 14400);
        assert_eq!(config.max_variation, 0.25);
        assert_eq!(config.min_diversity, 0.2);
    }

    #[test]
    fn test_component_creation() {
        let component = Component {
            pixels: vec![(10, 20), (11, 20)],
            level: 128,
            parent: None,
            children: vec![1, 2],
            area: 2,
            stability: 0.1,
        };

        assert_eq!(component.pixels.len(), 2);
        assert_eq!(component.level, 128);
        assert!(component.parent.is_none());
        assert_eq!(component.children.len(), 2);
        assert_eq!(component.area, 2);
        assert_eq!(component.stability, 0.1);
    }

    #[test]
    fn test_find_root_simple() {
        let components = vec![
            Component {
                pixels: vec![(0, 0)],
                level: 0,
                parent: None,
                children: vec![1],
                area: 1,
                stability: 0.0,
            },
            Component {
                pixels: vec![(1, 0)],
                level: 1,
                parent: Some(0),
                children: vec![],
                area: 1,
                stability: 0.0,
            },
        ];

        assert_eq!(find_root(&components, 0), 0);
        assert_eq!(find_root(&components, 1), 0);
    }
}
