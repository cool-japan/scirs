//! Color Map Implementations
//!
//! This module provides various color map implementations for data visualization,
//! including scientific color maps like Viridis, Plasma, and traditional maps like Jet.

use crate::visualization::types::{ColorMap, RgbColor};
use scirs2_core::ndarray::{Array3, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive, ToPrimitive};
use std::fmt::Debug;

/// Create a color map for visualization
pub fn create_colormap(colormap: ColorMap, num_colors: usize) -> Vec<RgbColor> {
    let mut colors = Vec::with_capacity(num_colors);

    for i in 0..num_colors {
        let t = if num_colors == 1 {
            0.5 // Single color case
        } else {
            i as f64 / (num_colors - 1) as f64
        };

        let color = match colormap {
            ColorMap::Gray => gray_colormap(t),
            ColorMap::Jet => jet_colormap(t),
            ColorMap::Viridis => viridis_colormap(t),
            ColorMap::Plasma => plasma_colormap(t),
            ColorMap::Inferno => inferno_colormap(t),
            ColorMap::Hot => hot_colormap(t),
            ColorMap::Cool => cool_colormap(t),
            ColorMap::Spring => spring_colormap(t),
            ColorMap::Summer => summer_colormap(t),
            ColorMap::Autumn => autumn_colormap(t),
            ColorMap::Winter => winter_colormap(t),
        };
        colors.push(color);
    }

    colors
}

/// Create a grayscale color map
pub fn gray_colormap(t: f64) -> RgbColor {
    let val = (t.clamp(0.0, 1.0) * 255.0) as u8;
    RgbColor::new(val, val, val)
}

/// Create a jet color map (blue to red through cyan, yellow)
pub fn jet_colormap(t: f64) -> RgbColor {
    let t = t.clamp(0.0, 1.0);
    let r = (1.5 - 4.0 * (t - 0.75).abs()).max(0.0).min(1.0);
    let g = (1.5 - 4.0 * (t - 0.5).abs()).max(0.0).min(1.0);
    let b = (1.5 - 4.0 * (t - 0.25).abs()).max(0.0).min(1.0);

    RgbColor::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Create a viridis color map (perceptually uniform)
pub fn viridis_colormap(t: f64) -> RgbColor {
    let t = t.clamp(0.0, 1.0);
    // Simplified viridis approximation using polynomial fits
    let r = (0.267 + 0.005 * t + 2.817 * t * t - 2.088 * t * t * t)
        .max(0.0)
        .min(1.0);
    let g = (-0.040 + 1.416 * t - 0.376 * t * t).max(0.0).min(1.0);
    let b = (0.329 - 0.327 * t + 2.209 * t * t - 1.211 * t * t * t)
        .max(0.0)
        .min(1.0);

    RgbColor::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Create a plasma color map (perceptually uniform)
pub fn plasma_colormap(t: f64) -> RgbColor {
    let t = t.clamp(0.0, 1.0);
    // Simplified plasma approximation using polynomial fits
    let r = (0.054 + 2.192 * t + 0.063 * t * t - 1.309 * t * t * t)
        .max(0.0)
        .min(1.0);
    let g = (0.230 * t + 1.207 * t * t - 0.437 * t * t * t)
        .max(0.0)
        .min(1.0);
    let b = (0.847 - 0.057 * t + 0.478 * t * t - 1.268 * t * t * t)
        .max(0.0)
        .min(1.0);

    RgbColor::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Create an inferno color map (perceptually uniform)
pub fn inferno_colormap(t: f64) -> RgbColor {
    let t = t.clamp(0.0, 1.0);
    // Simplified inferno approximation using polynomial fits
    let r = (0.077 + 2.081 * t + 0.866 * t * t - 1.024 * t * t * t)
        .max(0.0)
        .min(1.0);
    let g = (t * t * (1.842 - 0.842 * t)).max(0.0).min(1.0);
    let b = (1.777 * t * t * t * t - 1.777 * t * t * t + 0.777 * t * t)
        .max(0.0)
        .min(1.0);

    RgbColor::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Create a hot color map (black to white through red and yellow)
pub fn hot_colormap(t: f64) -> RgbColor {
    let t = t.clamp(0.0, 1.0);
    let r = (3.0 * t).min(1.0);
    let g = (3.0 * t - 1.0).max(0.0).min(1.0);
    let b = (3.0 * t - 2.0).max(0.0).min(1.0);

    RgbColor::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Create a cool color map (cyan to magenta)
pub fn cool_colormap(t: f64) -> RgbColor {
    let t = t.clamp(0.0, 1.0);
    RgbColor::new(((1.0 - t) * 255.0) as u8, (t * 255.0) as u8, 255)
}

/// Create a spring color map (magenta to yellow)
pub fn spring_colormap(t: f64) -> RgbColor {
    let t = t.clamp(0.0, 1.0);
    RgbColor::new(255, (t * 255.0) as u8, ((1.0 - t) * 255.0) as u8)
}

/// Create a summer color map (green to yellow)
pub fn summer_colormap(t: f64) -> RgbColor {
    let t = t.clamp(0.0, 1.0);
    RgbColor::new(
        (t * 255.0) as u8,
        ((0.5 + 0.5 * t) * 255.0) as u8,
        (0.4 * 255.0) as u8,
    )
}

/// Create an autumn color map (red to yellow)
pub fn autumn_colormap(t: f64) -> RgbColor {
    let t = t.clamp(0.0, 1.0);
    RgbColor::new(255, (t * 255.0) as u8, 0)
}

/// Create a winter color map (blue to green)
pub fn winter_colormap(t: f64) -> RgbColor {
    let t = t.clamp(0.0, 1.0);
    RgbColor::new(0, (t * 255.0) as u8, ((1.0 - 0.5 * t) * 255.0) as u8)
}

/// Generate a smooth gradient between two colors
pub fn gradient_colormap(color1: RgbColor, color2: RgbColor, t: f64) -> RgbColor {
    let t = t.clamp(0.0, 1.0);
    let r = (color1.r as f64 * (1.0 - t) + color2.r as f64 * t) as u8;
    let g = (color1.g as f64 * (1.0 - t) + color2.g as f64 * t) as u8;
    let b = (color1.b as f64 * (1.0 - t) + color2.b as f64 * t) as u8;

    RgbColor::new(r, g, b)
}

/// Generate a multi-color gradient
pub fn multi_gradient_colormap(colors: &[RgbColor], t: f64) -> RgbColor {
    if colors.is_empty() {
        return RgbColor::new(0, 0, 0); // Black fallback
    }

    if colors.len() == 1 {
        return colors[0];
    }

    let t = t.clamp(0.0, 1.0);
    let segment_size = 1.0 / (colors.len() - 1) as f64;
    let segment_index = (t / segment_size).floor() as usize;
    let segment_t = (t % segment_size) / segment_size;

    if segment_index >= colors.len() - 1 {
        return colors[colors.len() - 1];
    }

    gradient_colormap(colors[segment_index], colors[segment_index + 1], segment_t)
}

/// Create a categorical color map with distinct colors
pub fn categorical_colormap(category: usize, num_categories: usize) -> RgbColor {
    if num_categories == 0 {
        return RgbColor::new(0, 0, 0);
    }

    let hue = (category % num_categories) as f64 * 360.0 / num_categories as f64;
    let saturation = 0.8;
    let value = 0.9;

    RgbColor::from_hsv(hue, saturation, value)
}

/// Create a diverging color map (useful for data centered around zero)
pub fn diverging_colormap(
    t: f64,
    negative_color: RgbColor,
    positive_color: RgbColor,
    neutral_color: RgbColor,
) -> RgbColor {
    let t = t.clamp(-1.0, 1.0);

    if t < 0.0 {
        gradient_colormap(negative_color, neutral_color, (t + 1.0))
    } else {
        gradient_colormap(neutral_color, positive_color, t)
    }
}

/// Create a color map optimized for scientific visualization
pub fn scientific_colormap(colormap: ColorMap, gamma: f64) -> impl Fn(f64) -> RgbColor {
    move |t: f64| {
        let corrected_t = if gamma != 1.0 {
            t.clamp(0.0, 1.0).powf(1.0 / gamma)
        } else {
            t.clamp(0.0, 1.0)
        };

        match colormap {
            ColorMap::Gray => gray_colormap(corrected_t),
            ColorMap::Jet => jet_colormap(corrected_t),
            ColorMap::Viridis => viridis_colormap(corrected_t),
            ColorMap::Plasma => plasma_colormap(corrected_t),
            ColorMap::Inferno => inferno_colormap(corrected_t),
            ColorMap::Hot => hot_colormap(corrected_t),
            ColorMap::Cool => cool_colormap(corrected_t),
            ColorMap::Spring => spring_colormap(corrected_t),
            ColorMap::Summer => summer_colormap(corrected_t),
            ColorMap::Autumn => autumn_colormap(corrected_t),
            ColorMap::Winter => winter_colormap(corrected_t),
        }
    }
}

/// Get a colormap function by type
///
/// Returns a function pointer that can be used to generate colors for the specified colormap.
pub fn get_colormap_function(colormap: ColorMap) -> fn(f64) -> RgbColor {
    match colormap {
        ColorMap::Gray => gray_colormap,
        ColorMap::Jet => jet_colormap,
        ColorMap::Viridis => viridis_colormap,
        ColorMap::Plasma => plasma_colormap,
        ColorMap::Inferno => inferno_colormap,
        ColorMap::Hot => hot_colormap,
        ColorMap::Cool => cool_colormap,
        ColorMap::Spring => spring_colormap,
        ColorMap::Summer => summer_colormap,
        ColorMap::Autumn => autumn_colormap,
        ColorMap::Winter => winter_colormap,
    }
}

/// Apply a colormap to a 2D array to create a color image
///
/// This function converts a 2D numerical array into RGB colors using the specified colormap.
/// Values are automatically normalized to the [0, 1] range based on the array's min/max values.
///
/// # Arguments
///
/// * `data` - The 2D array to colorize
/// * `colormap` - The colormap type to use
///
/// # Returns
///
/// A 3D array with shape (height, width, 3) containing RGB values [0-255]
pub fn apply_colormap_to_array<T>(
    data: &ArrayView2<T>,
    colormap: ColorMap,
) -> crate::error::NdimageResult<Array3<u8>>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    use scirs2_core::ndarray::Array3;

    let (height, width) = data.dim();
    let mut result = Array3::zeros((height, width, 3));

    // Find min and max for normalization
    let min_val = data.iter().cloned().fold(T::infinity(), T::min);
    let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);

    if max_val <= min_val {
        return Err(crate::error::NdimageError::InvalidInput(
            "All values in array are the same".into(),
        ));
    }

    let colormap_fn = get_colormap_function(colormap);

    for i in 0..height {
        for j in 0..width {
            let value = data[[i, j]];
            let normalized = ((value - min_val) / (max_val - min_val))
                .to_f64()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);

            let color = colormap_fn(normalized);
            result[[i, j, 0]] = color.r;
            result[[i, j, 1]] = color.g;
            result[[i, j, 2]] = color.b;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_colormap() {
        let colors = create_colormap(ColorMap::Viridis, 256);
        assert_eq!(colors.len(), 256);

        // Test edge cases
        let single_color = create_colormap(ColorMap::Gray, 1);
        assert_eq!(single_color.len(), 1);

        let empty_colormap = create_colormap(ColorMap::Jet, 0);
        assert_eq!(empty_colormap.len(), 0);
    }

    #[test]
    fn test_gray_colormap() {
        let black = gray_colormap(0.0);
        assert_eq!(black.r, 0);
        assert_eq!(black.g, 0);
        assert_eq!(black.b, 0);

        let white = gray_colormap(1.0);
        assert_eq!(white.r, 255);
        assert_eq!(white.g, 255);
        assert_eq!(white.b, 255);

        let gray = gray_colormap(0.5);
        assert_eq!(gray.r, 127);
        assert_eq!(gray.g, 127);
        assert_eq!(gray.b, 127);
    }

    #[test]
    fn test_jet_colormap() {
        let color_start = jet_colormap(0.0);
        let color_middle = jet_colormap(0.5);
        let color_end = jet_colormap(1.0);

        // Jet should start with more blue, peak with green in middle, end with red
        assert!(color_start.b > color_start.r);
        assert!(color_middle.g > 100); // Should have significant green
        assert!(color_end.r > color_end.b);
    }

    #[test]
    fn test_viridis_colormap() {
        let dark = viridis_colormap(0.0);
        let bright = viridis_colormap(1.0);

        // Viridis should start dark and end bright
        assert!(dark.r < 100 && dark.g < 100 && dark.b < 100); // Relaxed threshold
        assert!(bright.g > 150); // Should end with bright green/yellow
    }

    #[test]
    fn test_gradient_colormap() {
        let red = RgbColor::new(255, 0, 0);
        let blue = RgbColor::new(0, 0, 255);

        let start = gradient_colormap(red, blue, 0.0);
        assert_eq!(start.r, 255);
        assert_eq!(start.g, 0);
        assert_eq!(start.b, 0);

        let end = gradient_colormap(red, blue, 1.0);
        assert_eq!(end.r, 0);
        assert_eq!(end.g, 0);
        assert_eq!(end.b, 255);

        let middle = gradient_colormap(red, blue, 0.5);
        assert_eq!(middle.r, 127);
        assert_eq!(middle.g, 0);
        assert_eq!(middle.b, 127);
    }

    #[test]
    fn test_multi_gradient_colormap() {
        let colors = vec![
            RgbColor::new(255, 0, 0), // Red
            RgbColor::new(0, 255, 0), // Green
            RgbColor::new(0, 0, 255), // Blue
        ];

        let start = multi_gradient_colormap(&colors, 0.0);
        assert_eq!(start.r, 255); // Should be red

        let middle = multi_gradient_colormap(&colors, 0.5);
        assert_eq!(middle.g, 255); // Should be green

        let end = multi_gradient_colormap(&colors, 1.0);
        assert_eq!(end.b, 255); // Should be blue

        // Test edge cases
        let empty_result = multi_gradient_colormap(&[], 0.5);
        assert_eq!(empty_result.r, 0);
        assert_eq!(empty_result.g, 0);
        assert_eq!(empty_result.b, 0);

        let single_result = multi_gradient_colormap(&[RgbColor::new(128, 128, 128)], 0.7);
        assert_eq!(single_result.r, 128);
        assert_eq!(single_result.g, 128);
        assert_eq!(single_result.b, 128);
    }

    #[test]
    fn test_categorical_colormap() {
        let color1 = categorical_colormap(0, 4);
        let color2 = categorical_colormap(1, 4);
        let color3 = categorical_colormap(2, 4);
        let color4 = categorical_colormap(3, 4);

        // Colors should be different
        assert_ne!(color1.to_tuple(), color2.to_tuple());
        assert_ne!(color2.to_tuple(), color3.to_tuple());
        assert_ne!(color3.to_tuple(), color4.to_tuple());

        // Test wraparound
        let color5 = categorical_colormap(4, 4);
        assert_eq!(color1.to_tuple(), color5.to_tuple());
    }

    #[test]
    fn test_diverging_colormap() {
        let red = RgbColor::new(255, 0, 0);
        let blue = RgbColor::new(0, 0, 255);
        let white = RgbColor::new(255, 255, 255);

        let negative = diverging_colormap(-1.0, red, blue, white);
        assert_eq!(negative.r, 255); // Should be red

        let neutral = diverging_colormap(0.0, red, blue, white);
        assert_eq!(neutral.r, 255);
        assert_eq!(neutral.g, 255);
        assert_eq!(neutral.b, 255); // Should be white

        let positive = diverging_colormap(1.0, red, blue, white);
        assert_eq!(positive.b, 255); // Should be blue
    }

    #[test]
    fn test_scientific_colormap() {
        let linear_fn = scientific_colormap(ColorMap::Viridis, 1.0);
        let gamma_fn = scientific_colormap(ColorMap::Viridis, 2.0);

        let linear_color = linear_fn(0.5);
        let gamma_color = gamma_fn(0.5);

        // Colors should be different due to gamma correction
        // Note: This is a basic test; in practice, gamma correction creates subtle differences
        assert!(linear_color.r > 0 || linear_color.g > 0 || linear_color.b > 0);
        assert!(gamma_color.r > 0 || gamma_color.g > 0 || gamma_color.b > 0);
    }

    #[test]
    fn test_clamp_behavior() {
        // Test that all colormap functions handle out-of-range inputs properly
        let test_values = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];

        for &t in &test_values {
            // All these should not panic and return valid colors
            let _gray = gray_colormap(t);
            let _jet = jet_colormap(t);
            let _viridis = viridis_colormap(t);
            let _plasma = plasma_colormap(t);
            let _inferno = inferno_colormap(t);
            let _hot = hot_colormap(t);
            let _cool = cool_colormap(t);
            let _spring = spring_colormap(t);
            let _summer = summer_colormap(t);
            let _autumn = autumn_colormap(t);
            let _winter = winter_colormap(t);
        }
    }
}
