//! Utility functions for visualization tasks
//!
//! This module provides helper functions for common visualization operations
//! including colormap generation and statistical analysis.

use super::types::{ColorScheme, PlotStatistics};

/// Generate color map values
pub fn generate_colormap(values: &[f64], scheme: ColorScheme) -> Vec<(u8, u8, u8)> {
    let n = values.len();
    let mut colors = Vec::with_capacity(n);

    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    for &val in values {
        let normalized = if range > 0.0 {
            (val - min_val) / range
        } else {
            0.5
        };

        let color = match scheme {
            ColorScheme::Viridis => viridis_color(normalized),
            ColorScheme::Plasma => plasma_color(normalized),
            ColorScheme::Inferno => inferno_color(normalized),
            ColorScheme::Grayscale => {
                let gray = (normalized * 255.0) as u8;
                (gray, gray, gray)
            }
        };
        colors.push(color);
    }

    colors
}

/// Viridis colormap
fn viridis_color(t: f64) -> (u8, u8, u8) {
    let t = t.max(0.0).min(1.0);
    let r = (0.267004 + t * (0.127568 + t * (0.019234 - t * 0.012814))) * 255.0;
    let g = (0.004874 + t * (0.950141 + t * (-0.334896 + t * 0.158789))) * 255.0;
    let b = (0.329415 + t * (0.234092 + t * (1.384085 - t * 1.388488))) * 255.0;
    (r as u8, g as u8, b as u8)
}

/// Plasma colormap
fn plasma_color(t: f64) -> (u8, u8, u8) {
    let t = t.max(0.0).min(1.0);
    let r = (0.050383 + t * (1.075483 + t * (-0.346066 + t * 0.220971))) * 255.0;
    let g = (0.029803 + t * (0.089467 + t * (1.234884 - t * 1.281864))) * 255.0;
    let b = (0.527975 + t * (0.670134 + t * (-1.397127 + t * 1.149498))) * 255.0;
    (r as u8, g as u8, b as u8)
}

/// Inferno colormap
fn inferno_color(t: f64) -> (u8, u8, u8) {
    let t = t.max(0.0).min(1.0);
    let r = (0.001462 + t * (0.998260 + t * (-0.149678 + t * 0.150124))) * 255.0;
    let g = (0.000466 + t * (0.188724 + t * (1.203007 - t * 1.391543))) * 255.0;
    let b = (0.013866 + t * (0.160930 + t * (0.690929 - t * 0.865624))) * 255.0;
    (r as u8, g as u8, b as u8)
}

/// Calculate optimal grid resolution for vector field plots
pub fn optimal_grid_resolution(_domainsize: (f64, f64), target_density: f64) -> (usize, usize) {
    let (width, height) = _domainsize;
    let area = width * height;
    let total_points = (area * target_density) as usize;

    let aspect_ratio = width / height;
    let ny = (total_points as f64 / aspect_ratio).sqrt() as usize;
    let nx = (ny as f64 * aspect_ratio) as usize;

    (nx.max(10), ny.max(10))
}

/// Create summary statistics for plot data
pub fn plot_statistics(data: &[f64]) -> PlotStatistics {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = sorted_data[0];
    let max = sorted_data[sorted_data.len() - 1];
    let median = if sorted_data.len().is_multiple_of(2) {
        (sorted_data[sorted_data.len() / 2 - 1] + sorted_data[sorted_data.len() / 2]) / 2.0
    } else {
        sorted_data[sorted_data.len() / 2]
    };

    PlotStatistics {
        count: data.len(),
        mean,
        std_dev,
        min,
        max,
        median,
    }
}
