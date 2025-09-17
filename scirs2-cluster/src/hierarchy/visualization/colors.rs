//! Color palette utilities for dendrogram visualization
//!
//! This module provides various color schemes and palette generation functions
//! for enhancing dendrogram visualizations.

use super::types::ColorScheme;

/// Get a color palette based on the specified color scheme
///
/// This function returns a vector of color strings (hex format) for the given
/// color scheme and number of colors requested.
///
/// # Arguments
/// * `scheme` - The color scheme to use
/// * `ncolors` - Number of colors to generate
///
/// # Returns
/// * `Vec<String>` - Vector of hex color strings (e.g., "#1f77b4")
///
/// # Example
/// ```rust
/// use scirs2_cluster::hierarchy::visualization::{get_color_palette, ColorScheme};
///
/// let colors = get_color_palette(ColorScheme::Viridis, 5);
/// assert_eq!(colors.len(), 5);
/// assert!(colors[0].starts_with("#"));
/// ```
pub fn get_color_palette(scheme: ColorScheme, ncolors: usize) -> Vec<String> {
    match scheme {
        ColorScheme::Default => get_default_colors(ncolors),
        ColorScheme::HighContrast => get_high_contrast_colors(ncolors),
        ColorScheme::Viridis => get_viridis_colors(ncolors),
        ColorScheme::Plasma => get_plasma_colors(ncolors),
        ColorScheme::Grayscale => get_grayscale_colors(ncolors),
    }
}

/// Default color palette using matplotlib-style colors
///
/// This palette provides a good balance of distinguishability and aesthetics
/// for general-purpose clustering visualization.
///
/// # Arguments
/// * `ncolors` - Number of colors to generate
///
/// # Returns
/// * `Vec<String>` - Vector of hex color strings
fn get_default_colors(ncolors: usize) -> Vec<String> {
    let base_colors = vec![
        "#1f77b4", // Blue
        "#ff7f0e", // Orange
        "#2ca02c", // Green
        "#d62728", // Red
        "#9467bd", // Purple
        "#8c564b", // Brown
        "#e377c2", // Pink
        "#7f7f7f", // Gray
        "#bcbd22", // Olive
        "#17becf", // Cyan
    ];

    base_colors
        .into_iter()
        .cycle()
        .take(ncolors)
        .map(|s| s.to_string())
        .collect()
}

/// High contrast color palette for accessibility
///
/// This palette uses maximally contrasting colors, making it suitable
/// for users with color vision deficiencies or black and white printing.
///
/// # Arguments
/// * `ncolors` - Number of colors to generate
///
/// # Returns
/// * `Vec<String>` - Vector of hex color strings
fn get_high_contrast_colors(ncolors: usize) -> Vec<String> {
    let base_colors = vec![
        "#000000", // Black
        "#FFFFFF", // White
        "#FF0000", // Red
        "#00FF00", // Green
        "#0000FF", // Blue
        "#FFFF00", // Yellow
        "#FF00FF", // Magenta
        "#00FFFF", // Cyan
        "#800000", // Maroon
        "#008000", // Dark Green
        "#000080", // Navy
        "#808000", // Olive
        "#800080", // Purple
        "#008080", // Teal
        "#C0C0C0", // Silver
        "#808080", // Gray
    ];

    base_colors
        .into_iter()
        .cycle()
        .take(ncolors)
        .map(|s| s.to_string())
        .collect()
}

/// Viridis color palette (perceptually uniform)
///
/// The Viridis palette is designed to be perceptually uniform, meaning
/// that equal steps in data correspond to equal steps in perceived color change.
/// It's also colorblind-friendly.
///
/// # Arguments
/// * `ncolors` - Number of colors to generate
///
/// # Returns
/// * `Vec<String>` - Vector of hex color strings
fn get_viridis_colors(ncolors: usize) -> Vec<String> {
    let base_colors = vec![
        "#440154", // Dark purple
        "#482777", // Purple
        "#3f4a8a", // Dark blue
        "#31678e", // Blue
        "#26838f", // Teal
        "#1f9d8a", // Green-teal
        "#6cce5a", // Green
        "#b6de2b", // Yellow-green
        "#fee825", // Yellow
        "#fff200", // Bright yellow
    ];

    if ncolors <= base_colors.len() {
        // For small numbers, pick evenly spaced colors
        let step = base_colors.len() / ncolors.max(1);
        base_colors
            .into_iter()
            .step_by(step.max(1))
            .take(ncolors)
            .map(|s| s.to_string())
            .collect()
    } else {
        // For large numbers, cycle through the base colors
        base_colors
            .into_iter()
            .cycle()
            .take(ncolors)
            .map(|s| s.to_string())
            .collect()
    }
}

/// Plasma color palette (perceptually uniform)
///
/// The Plasma palette is another perceptually uniform color scheme that
/// goes from purple through magenta and orange to yellow. It's suitable
/// for scientific visualization.
///
/// # Arguments
/// * `ncolors` - Number of colors to generate
///
/// # Returns
/// * `Vec<String>` - Vector of hex color strings
fn get_plasma_colors(ncolors: usize) -> Vec<String> {
    let base_colors = vec![
        "#0c0887", // Dark purple
        "#5c01a6", // Purple
        "#900da4", // Magenta-purple
        "#bf3984", // Magenta
        "#e16462", // Red-orange
        "#f89441", // Orange
        "#fdc328", // Yellow-orange
        "#f0f921", // Yellow
        "#fcffa4", // Light yellow
        "#ffffff", // White
    ];

    if ncolors <= base_colors.len() {
        // For small numbers, pick evenly spaced colors
        let step = base_colors.len() / ncolors.max(1);
        base_colors
            .into_iter()
            .step_by(step.max(1))
            .take(ncolors)
            .map(|s| s.to_string())
            .collect()
    } else {
        // For large numbers, cycle through the base colors
        base_colors
            .into_iter()
            .cycle()
            .take(ncolors)
            .map(|s| s.to_string())
            .collect()
    }
}

/// Grayscale color palette for black and white publications
///
/// This palette generates colors from black to white, suitable for
/// publications that don't support color or when color is not desired.
///
/// # Arguments
/// * `ncolors` - Number of colors to generate
///
/// # Returns
/// * `Vec<String>` - Vector of hex color strings
fn get_grayscale_colors(ncolors: usize) -> Vec<String> {
    if ncolors == 0 {
        return Vec::new();
    }

    (0..ncolors)
        .map(|i| {
            let intensity = if ncolors == 1 {
                128 // Middle gray for single color
            } else {
                255 * i / (ncolors - 1) // Evenly spaced from black to white
            };
            format!("#{:02x}{:02x}{:02x}", intensity, intensity, intensity)
        })
        .collect()
}

/// Generate interpolated colors between two colors
///
/// This function creates a gradient between two hex colors, useful for
/// creating smooth color transitions.
///
/// # Arguments
/// * `start_color` - Starting color in hex format (e.g., "#ff0000")
/// * `end_color` - Ending color in hex format (e.g., "#0000ff")
/// * `ncolors` - Number of colors to generate in the gradient
///
/// # Returns
/// * `Result<Vec<String>, String>` - Vector of interpolated hex color strings
pub fn interpolate_colors(
    start_color: &str,
    end_color: &str,
    ncolors: usize,
) -> Result<Vec<String>, String> {
    if ncolors == 0 {
        return Ok(Vec::new());
    }

    // Parse hex colors
    let start_rgb = parse_hex_color(start_color)?;
    let end_rgb = parse_hex_color(end_color)?;

    let mut colors = Vec::with_capacity(ncolors);

    for i in 0..ncolors {
        let t = if ncolors == 1 {
            0.5 // Middle color for single color
        } else {
            i as f64 / (ncolors - 1) as f64
        };

        let r = (start_rgb.0 as f64 + t * (end_rgb.0 as f64 - start_rgb.0 as f64)) as u8;
        let g = (start_rgb.1 as f64 + t * (end_rgb.1 as f64 - start_rgb.1 as f64)) as u8;
        let b = (start_rgb.2 as f64 + t * (end_rgb.2 as f64 - start_rgb.2 as f64)) as u8;

        colors.push(format!("#{:02x}{:02x}{:02x}", r, g, b));
    }

    Ok(colors)
}

/// Parse a hex color string to RGB values
///
/// # Arguments
/// * `hex_color` - Hex color string (with or without #)
///
/// # Returns
/// * `Result<(u8, u8, u8), String>` - RGB values as tuple
fn parse_hex_color(hex_color: &str) -> Result<(u8, u8, u8), String> {
    let hex = hex_color.trim_start_matches('#');

    if hex.len() != 6 {
        return Err(format!("Invalid hex color format: {}", hex_color));
    }

    let r = u8::from_str_radix(&hex[0..2], 16)
        .map_err(|_| format!("Invalid red component: {}", &hex[0..2]))?;
    let g = u8::from_str_radix(&hex[2..4], 16)
        .map_err(|_| format!("Invalid green component: {}", &hex[2..4]))?;
    let b = u8::from_str_radix(&hex[4..6], 16)
        .map_err(|_| format!("Invalid blue component: {}", &hex[4..6]))?;

    Ok((r, g, b))
}

/// Convert RGB values to hex color string
///
/// # Arguments
/// * `r` - Red component (0-255)
/// * `g` - Green component (0-255)
/// * `b` - Blue component (0-255)
///
/// # Returns
/// * `String` - Hex color string (e.g., "#ff0000")
pub fn rgb_to_hex(r: u8, g: u8, b: u8) -> String {
    format!("#{:02x}{:02x}{:02x}", r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_color_palette_default() {
        let colors = get_color_palette(ColorScheme::Default, 5);
        assert_eq!(colors.len(), 5);
        assert!(colors.iter().all(|c| c.starts_with("#")));
        assert_eq!(colors[0], "#1f77b4");
    }

    #[test]
    fn test_get_color_palette_high_contrast() {
        let colors = get_color_palette(ColorScheme::HighContrast, 3);
        assert_eq!(colors.len(), 3);
        assert_eq!(colors[0], "#000000");
        assert_eq!(colors[1], "#FFFFFF");
        assert_eq!(colors[2], "#FF0000");
    }

    #[test]
    fn test_get_color_palette_viridis() {
        let colors = get_color_palette(ColorScheme::Viridis, 4);
        assert_eq!(colors.len(), 4);
        assert!(colors.iter().all(|c| c.starts_with("#")));
    }

    #[test]
    fn test_get_color_palette_plasma() {
        let colors = get_color_palette(ColorScheme::Plasma, 3);
        assert_eq!(colors.len(), 3);
        assert!(colors.iter().all(|c| c.starts_with("#")));
    }

    #[test]
    fn test_get_grayscale_colors() {
        let colors = get_grayscale_colors(5);
        assert_eq!(colors.len(), 5);
        assert_eq!(colors[0], "#000000"); // Black
        assert_eq!(colors[4], "#ffffff"); // White
    }

    #[test]
    fn test_get_grayscale_colors_single() {
        let colors = get_grayscale_colors(1);
        assert_eq!(colors.len(), 1);
        assert_eq!(colors[0], "#808080"); // Middle gray
    }

    #[test]
    fn test_interpolate_colors() {
        let colors = interpolate_colors("#ff0000", "#0000ff", 3).unwrap();
        assert_eq!(colors.len(), 3);
        assert_eq!(colors[0], "#ff0000"); // Red
        assert_eq!(colors[2], "#0000ff"); // Blue
    }

    #[test]
    fn test_parse_hex_color() {
        let (r, g, b) = parse_hex_color("#ff0000").unwrap();
        assert_eq!((r, g, b), (255, 0, 0));

        let (r, g, b) = parse_hex_color("00ff00").unwrap();
        assert_eq!((r, g, b), (0, 255, 0));
    }

    #[test]
    fn test_rgb_to_hex() {
        assert_eq!(rgb_to_hex(255, 0, 0), "#ff0000");
        assert_eq!(rgb_to_hex(0, 255, 0), "#00ff00");
        assert_eq!(rgb_to_hex(0, 0, 255), "#0000ff");
        assert_eq!(rgb_to_hex(128, 128, 128), "#808080");
    }

    #[test]
    fn test_get_color_palette_empty() {
        let colors = get_color_palette(ColorScheme::Default, 0);
        assert_eq!(colors.len(), 0);
    }

    #[test]
    fn test_interpolate_colors_empty() {
        let colors = interpolate_colors("#ff0000", "#0000ff", 0).unwrap();
        assert_eq!(colors.len(), 0);
    }
}
