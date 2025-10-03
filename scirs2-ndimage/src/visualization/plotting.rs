//! Plotting and Visualization Functions
//!
//! This module provides comprehensive plotting functionality for creating
//! various types of visualizations including histograms, line plots, surface plots,
//! contour plots, and gradient vector field visualizations.

use crate::error::{NdimageError, NdimageResult};
use crate::utils::{safe_f64_to_float, safe_usize_to_float};
use crate::visualization::colormap::create_colormap;
use crate::visualization::types::{PlotConfig, ReportFormat};
use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive, ToPrimitive, Zero};
use std::fmt::{Debug, Write};

/// Generate a histogram plot representation
pub fn plot_histogram<T>(data: &ArrayView1<T>, config: &PlotConfig) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if data.is_empty() {
        return Err(NdimageError::InvalidInput("Data array is empty".into()));
    }

    // Find min and max values
    let min_val = data.iter().cloned().fold(T::infinity(), T::min);
    let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);

    if max_val <= min_val {
        return Err(NdimageError::InvalidInput(
            "All data values are the same".into(),
        ));
    }

    // Create histogram bins
    let mut histogram = vec![0usize; config.num_bins];
    let range = max_val - min_val;
    let bin_size = range / safe_usize_to_float::<T>(config.num_bins)?;

    for &value in data.iter() {
        let normalized = (value - min_val) / bin_size;
        let bin_idx = normalized.to_usize().unwrap_or(0).min(config.num_bins - 1);
        histogram[bin_idx] += 1;
    }

    // Generate plot representation
    let max_count = *histogram.iter().max().unwrap_or(&1);
    let mut plot = String::new();

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='histogram-plot'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(&mut plot, "<div class='histogram-bars'>")?;

            for (i, &count) in histogram.iter().enumerate() {
                let height_percent = (count as f64 / max_count as f64) * 100.0;
                let bin_start = min_val + safe_usize_to_float::<T>(i)? * bin_size;
                let bin_end = bin_start + bin_size;

                writeln!(
                    &mut plot,
                    "<div class='bar' style='height: {:.1}%' title='[{:.3}, {:.3}): {}'></div>",
                    height_percent,
                    bin_start.to_f64().unwrap_or(0.0),
                    bin_end.to_f64().unwrap_or(0.0),
                    count
                )?;
            }

            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "<div class='axis-labels'>")?;
            writeln!(&mut plot, "<span class='xlabel'>{}</span>", config.xlabel)?;
            writeln!(&mut plot, "<span class='ylabel'>{}</span>", config.ylabel)?;
            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {}", config.title)?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "```")?;

            for (i, &count) in histogram.iter().enumerate() {
                let bar_length = (count as f64 / max_count as f64 * 50.0) as usize;
                let bin_center = min_val
                    + (safe_usize_to_float::<T>(i)? + safe_f64_to_float::<T>(0.5)?) * bin_size;

                writeln!(
                    &mut plot,
                    "{:8.3} |{:<50} {}",
                    bin_center.to_f64().unwrap_or(0.0),
                    "*".repeat(bar_length),
                    count
                )?;
            }

            writeln!(&mut plot, "```")?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "**{}** vs **{}**", config.xlabel, config.ylabel)?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{}", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len()))?;
            writeln!(&mut plot)?;

            for (i, &count) in histogram.iter().enumerate() {
                let bar_length = (count as f64 / max_count as f64 * 50.0) as usize;
                let bin_center = min_val
                    + (safe_usize_to_float::<T>(i)? + safe_f64_to_float::<T>(0.5)?) * bin_size;

                writeln!(
                    &mut plot,
                    "{:8.3} |{:<50} {}",
                    bin_center.to_f64().unwrap_or(0.0),
                    "*".repeat(bar_length),
                    count
                )?;
            }

            writeln!(&mut plot)?;
            writeln!(&mut plot, "X-axis: {}", config.xlabel)?;
            writeln!(&mut plot, "Y-axis: {}", config.ylabel)?;
        }
    }

    Ok(plot)
}

/// Generate a profile plot (line plot) representation
pub fn plot_profile<T>(
    x_data: &ArrayView1<T>,
    y_data: &ArrayView1<T>,
    config: &PlotConfig,
) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if x_data.len() != y_data.len() {
        return Err(NdimageError::InvalidInput(
            "X and Y data must have the same length".into(),
        ));
    }

    if x_data.is_empty() {
        return Err(NdimageError::InvalidInput("Data arrays are empty".into()));
    }

    let mut plot = String::new();

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='profile-plot'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(
                &mut plot,
                "<svg width='{}' height='{}'>",
                config.width, config.height
            )?;

            // Plot data points and lines
            let x_min = x_data.iter().cloned().fold(T::infinity(), T::min);
            let x_max = x_data.iter().cloned().fold(T::neg_infinity(), T::max);
            let y_min = y_data.iter().cloned().fold(T::infinity(), T::min);
            let y_max = y_data.iter().cloned().fold(T::neg_infinity(), T::max);

            let x_range = x_max - x_min;
            let y_range = y_max - y_min;

            if x_range > T::zero() && y_range > T::zero() {
                let mut path_data = String::new();

                for (i, (&x, &y)) in x_data.iter().zip(y_data.iter()).enumerate() {
                    let px = ((x - x_min) / x_range * safe_usize_to_float(config.width - 100)?
                        + safe_f64_to_float::<T>(50.0)?)
                    .to_f64()
                    .unwrap_or(0.0);
                    let py = (config.height as f64 - 50.0)
                        - ((y - y_min) / y_range * safe_usize_to_float(config.height - 100)?)
                            .to_f64()
                            .unwrap_or(0.0);

                    if i == 0 {
                        write!(&mut path_data, "M {} {}", px, py)?;
                    } else {
                        write!(&mut path_data, " L {} {}", px, py)?;
                    }
                }

                writeln!(
                    &mut plot,
                    "<path d='{}' stroke='blue' stroke-width='2' fill='none'/>",
                    path_data
                )?;

                // Add grid if requested
                if config.show_grid {
                    add_svg_grid(&mut plot, config.width, config.height)?;
                }
            }

            writeln!(&mut plot, "</svg>")?;
            writeln!(&mut plot, "<div class='axis-labels'>")?;
            writeln!(&mut plot, "<span class='xlabel'>{}</span>", config.xlabel)?;
            writeln!(&mut plot, "<span class='ylabel'>{}</span>", config.ylabel)?;
            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {}", config.title)?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "```")?;

            for (&x, &y) in x_data.iter().zip(y_data.iter()) {
                writeln!(
                    &mut plot,
                    "{:10.4} {:10.4}",
                    x.to_f64().unwrap_or(0.0),
                    y.to_f64().unwrap_or(0.0)
                )?;
            }

            writeln!(&mut plot, "```")?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "**{}** vs **{}**", config.xlabel, config.ylabel)?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{}", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len()))?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "{:>10} {:>10}", config.xlabel, config.ylabel)?;
            writeln!(&mut plot, "{}", "-".repeat(22))?;

            for (&x, &y) in x_data.iter().zip(y_data.iter()) {
                writeln!(
                    &mut plot,
                    "{:10.4} {:10.4}",
                    x.to_f64().unwrap_or(0.0),
                    y.to_f64().unwrap_or(0.0)
                )?;
            }
        }
    }

    Ok(plot)
}

/// Generate a 3D surface plot representation of a 2D array
pub fn plot_surface<T>(data: &ArrayView2<T>, config: &PlotConfig) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    let (height, width) = data.dim();
    if height == 0 || width == 0 {
        return Err(NdimageError::InvalidInput("Data array is empty".into()));
    }

    let mut plot = String::new();

    // Find min and max values for scaling
    let min_val = data.iter().cloned().fold(T::infinity(), T::min);
    let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);

    if max_val <= min_val {
        return Err(NdimageError::InvalidInput(
            "All data values are the same".into(),
        ));
    }

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='surface-plot'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(&mut plot, "<div class='surface-container'>")?;

            // Create a simplified 3D representation using CSS transforms
            let step_x = width.max(1) / (config.width / 20).max(1);
            let step_y = height.max(1) / (config.height / 20).max(1);

            for i in (0..height).step_by(step_y) {
                for j in (0..width).step_by(step_x) {
                    let value = data[[i, j]];
                    let normalized = ((value - min_val) / (max_val - min_val))
                        .to_f64()
                        .unwrap_or(0.0);
                    let z_height = normalized * 100.0; // Scale to percentage

                    let x_pos = (j as f64 / width as f64) * config.width as f64;
                    let y_pos = (i as f64 / height as f64) * config.height as f64;

                    // Color based on height
                    let colormap = create_colormap(config.colormap, 256);
                    let color_idx = (normalized * 255.0) as usize;
                    let color = colormap.get(color_idx).unwrap_or(&colormap[0]);

                    writeln!(
                        &mut plot,
                        "<div class='surface-point' style='left: {:.1}px; top: {:.1}px; height: {:.1}%; background-color: {};'></div>",
                        x_pos, y_pos, z_height, color.to_hex()
                    )?;
                }
            }

            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "<div class='surface-info'>")?;
            writeln!(
                &mut plot,
                "<p>Value range: [{:.3}, {:.3}]</p>",
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {} (3D Surface)", config.title)?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "```")?;
            writeln!(&mut plot, "3D Surface Plot of {}×{} data", height, width)?;
            writeln!(
                &mut plot,
                "Value range: [{:.3}, {:.3}]",
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot)?;

            // Simple ASCII art representation
            let ascii_height = 20;
            let ascii_width = 60;
            for i in 0..ascii_height {
                for j in 0..ascii_width {
                    let data_i = (i * height) / ascii_height;
                    let data_j = (j * width) / ascii_width;
                    let value = data[[data_i, data_j]];
                    let normalized = ((value - min_val) / (max_val - min_val))
                        .to_f64()
                        .unwrap_or(0.0);

                    let char = match (normalized * 10.0) as u32 {
                        0..=1 => ' ',
                        2..=3 => '.',
                        4..=5 => ':',
                        6..=7 => '+',
                        8..=9 => '*',
                        _ => '#',
                    };
                    write!(&mut plot, "{}", char)?;
                }
                writeln!(&mut plot)?;
            }

            writeln!(&mut plot, "```")?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{} (3D Surface)", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len() + 13))?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Data dimensions: {}×{}", height, width)?;
            writeln!(
                &mut plot,
                "Value range: [{:.3}, {:.3}]",
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot)?;

            // Add ASCII surface representation
            add_ascii_surface(&mut plot, data, 20, 60)?;
        }
    }

    Ok(plot)
}

/// Generate a contour plot representation of a 2D array
pub fn plot_contour<T>(
    data: &ArrayView2<T>,
    num_levels: usize,
    config: &PlotConfig,
) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    let (height, width) = data.dim();
    if height == 0 || width == 0 {
        return Err(NdimageError::InvalidInput("Data array is empty".into()));
    }

    let mut plot = String::new();

    // Find min and max values for level calculation
    let min_val = data.iter().cloned().fold(T::infinity(), T::min);
    let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);

    if max_val <= min_val {
        return Err(NdimageError::InvalidInput(
            "All data values are the same".into(),
        ));
    }

    // Calculate contour levels
    let mut levels = Vec::new();
    for i in 0..num_levels {
        let t = i as f64 / (num_levels - 1).max(1) as f64;
        let level = min_val + (max_val - min_val) * safe_f64_to_float::<T>(t)?;
        levels.push(level);
    }

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='contour-plot'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(
                &mut plot,
                "<svg width='{}' height='{}'>",
                config.width, config.height
            )?;

            // Simple contour approximation by drawing level sets
            for (level_idx, &level) in levels.iter().enumerate() {
                let color_intensity = (level_idx as f64 / num_levels as f64 * 255.0) as u8;
                let color = format!(
                    "rgb({}, {}, {})",
                    color_intensity,
                    100,
                    255 - color_intensity
                );

                // Find points close to this level
                for i in 0..height.saturating_sub(1) {
                    for j in 0..width.saturating_sub(1) {
                        let val = data[[i, j]];
                        let threshold = (max_val - min_val) * safe_f64_to_float::<T>(0.02)?; // 2% tolerance

                        if (val - level).abs() < threshold {
                            let x = (j as f64 / width as f64) * config.width as f64;
                            let y = (i as f64 / height as f64) * config.height as f64;

                            writeln!(
                                &mut plot,
                                "<circle cx='{:.1}' cy='{:.1}' r='1' fill='{}' opacity='0.7'/>",
                                x, y, color
                            )?;
                        }
                    }
                }
            }

            writeln!(&mut plot, "</svg>")?;
            writeln!(&mut plot, "<div class='contour-legend'>")?;
            writeln!(&mut plot, "<h4>Contour Levels:</h4>")?;
            for (i, &level) in levels.iter().enumerate() {
                writeln!(
                    &mut plot,
                    "<span style='color: rgb({}, 100, {})'>Level {}: {:.3}</span><br/>",
                    (i as f64 / num_levels as f64 * 255.0) as u8,
                    255 - (i as f64 / num_levels as f64 * 255.0) as u8,
                    i + 1,
                    level.to_f64().unwrap_or(0.0)
                )?;
            }
            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {} (Contour)", config.title)?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Contour levels:")?;
            for (i, &level) in levels.iter().enumerate() {
                writeln!(
                    &mut plot,
                    "- Level {}: {:.3}",
                    i + 1,
                    level.to_f64().unwrap_or(0.0)
                )?;
            }
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{} (Contour)", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len() + 10))?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Contour levels:")?;
            for (i, &level) in levels.iter().enumerate() {
                writeln!(
                    &mut plot,
                    "  Level {}: {:.3}",
                    i + 1,
                    level.to_f64().unwrap_or(0.0)
                )?;
            }
        }
    }

    Ok(plot)
}

/// Visualize gradient information as a vector field
pub fn visualize_gradient<T>(
    gradient_x: &ArrayView2<T>,
    gradient_y: &ArrayView2<T>,
    config: &PlotConfig,
) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if gradient_x.dim() != gradient_y.dim() {
        return Err(NdimageError::DimensionError(
            "Gradient components must have the same dimensions".into(),
        ));
    }

    let (height, width) = gradient_x.dim();
    let mut plot = String::new();

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='gradient-plot'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(
                &mut plot,
                "<svg width='{}' height='{}'>",
                config.width, config.height
            )?;

            // Sample gradient vectors at regular intervals
            let step_x = width.max(1) / (config.width / 20).max(1);
            let step_y = height.max(1) / (config.height / 20).max(1);

            for i in (0..height).step_by(step_y) {
                for j in (0..width).step_by(step_x) {
                    let gx = gradient_x[[i, j]].to_f64().unwrap_or(0.0);
                    let gy = gradient_y[[i, j]].to_f64().unwrap_or(0.0);

                    let magnitude = (gx * gx + gy * gy).sqrt();
                    if magnitude > 1e-6 {
                        let scale = 10.0 / magnitude.max(1e-6);
                        let start_x = j as f64 * config.width as f64 / width as f64;
                        let start_y = i as f64 * config.height as f64 / height as f64;
                        let end_x = start_x + gx * scale;
                        let end_y = start_y + gy * scale;

                        writeln!(
                            &mut plot,
                            "<line x1='{:.1}' y1='{:.1}' x2='{:.1}' y2='{:.1}' stroke='red' stroke-width='1'/>",
                            start_x, start_y, end_x, end_y
                        )?;

                        // Add arrowhead
                        add_svg_arrowhead(&mut plot, start_x, start_y, end_x, end_y)?;
                    }
                }
            }

            writeln!(&mut plot, "</svg>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {}", config.title)?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Gradient vector field visualization")?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "- Image dimensions: {}×{}", width, height)?;

            // Compute some statistics
            let magnitude_sum: f64 = gradient_x
                .iter()
                .zip(gradient_y.iter())
                .map(|(&gx, &gy)| {
                    let gx_f = gx.to_f64().unwrap_or(0.0);
                    let gy_f = gy.to_f64().unwrap_or(0.0);
                    (gx_f * gx_f + gy_f * gy_f).sqrt()
                })
                .sum();

            let avg_magnitude = magnitude_sum / (width * height) as f64;
            writeln!(
                &mut plot,
                "- Average gradient magnitude: {:.4}",
                avg_magnitude
            )?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{}", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len()))?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Gradient Vector Field")?;
            writeln!(&mut plot, "Image dimensions: {}×{}", width, height)?;

            // Show a text-based representation
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Sample gradient vectors:")?;
            writeln!(
                &mut plot,
                "{:>5} {:>5} {:>10} {:>10} {:>10}",
                "Row", "Col", "Grad_X", "Grad_Y", "Magnitude"
            )?;
            writeln!(&mut plot, "{}", "-".repeat(50))?;

            let step = height.max(width) / 10;
            for i in (0..height).step_by(step.max(1)) {
                for j in (0..width).step_by(step.max(1)) {
                    let gx = gradient_x[[i, j]].to_f64().unwrap_or(0.0);
                    let gy = gradient_y[[i, j]].to_f64().unwrap_or(0.0);
                    let magnitude = (gx * gx + gy * gy).sqrt();

                    writeln!(
                        &mut plot,
                        "{:5} {:5} {:10.4} {:10.4} {:10.4}",
                        i, j, gx, gy, magnitude
                    )?;
                }
            }
        }
    }

    Ok(plot)
}

/// Create a scatter plot of two data arrays
pub fn plot_scatter<T>(
    x_data: &ArrayView1<T>,
    y_data: &ArrayView1<T>,
    config: &PlotConfig,
) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if x_data.len() != y_data.len() {
        return Err(NdimageError::InvalidInput(
            "X and Y data must have the same length".into(),
        ));
    }

    if x_data.is_empty() {
        return Err(NdimageError::InvalidInput("Data arrays are empty".into()));
    }

    let mut plot = String::new();

    // Find data ranges
    let x_min = x_data.iter().cloned().fold(T::infinity(), T::min);
    let x_max = x_data.iter().cloned().fold(T::neg_infinity(), T::max);
    let y_min = y_data.iter().cloned().fold(T::infinity(), T::min);
    let y_max = y_data.iter().cloned().fold(T::neg_infinity(), T::max);

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='scatter-plot'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(
                &mut plot,
                "<svg width='{}' height='{}'>",
                config.width, config.height
            )?;

            if config.show_grid {
                add_svg_grid(&mut plot, config.width, config.height)?;
            }

            let x_range = x_max - x_min;
            let y_range = y_max - y_min;

            if x_range > T::zero() && y_range > T::zero() {
                for (&x, &y) in x_data.iter().zip(y_data.iter()) {
                    let px = ((x - x_min) / x_range * safe_usize_to_float(config.width - 100)?
                        + safe_f64_to_float::<T>(50.0)?)
                    .to_f64()
                    .unwrap_or(0.0);
                    let py = (config.height as f64 - 50.0)
                        - ((y - y_min) / y_range * safe_usize_to_float(config.height - 100)?)
                            .to_f64()
                            .unwrap_or(0.0);

                    writeln!(
                        &mut plot,
                        "<circle cx='{:.1}' cy='{:.1}' r='3' fill='blue' opacity='0.7'/>",
                        px, py
                    )?;
                }
            }

            writeln!(&mut plot, "</svg>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown | ReportFormat::Text => {
            // Use the same logic as profile plot for text formats
            return plot_profile(x_data, y_data, config);
        }
    }

    Ok(plot)
}

/// Helper function to add SVG grid
fn add_svg_grid(plot: &mut String, width: usize, height: usize) -> std::fmt::Result {
    let grid_lines = 10;
    let x_step = width as f64 / grid_lines as f64;
    let y_step = height as f64 / grid_lines as f64;

    // Vertical grid lines
    for i in 0..=grid_lines {
        let x = i as f64 * x_step;
        writeln!(
            plot,
            "<line x1='{}' y1='0' x2='{}' y2='{}' stroke='#ddd' stroke-width='1'/>",
            x, x, height
        )?;
    }

    // Horizontal grid lines
    for i in 0..=grid_lines {
        let y = i as f64 * y_step;
        writeln!(
            plot,
            "<line x1='0' y1='{}' x2='{}' y2='{}' stroke='#ddd' stroke-width='1'/>",
            y, width, y
        )?;
    }

    Ok(())
}

/// Helper function to add SVG arrowhead
fn add_svg_arrowhead(
    plot: &mut String,
    start_x: f64,
    start_y: f64,
    end_x: f64,
    end_y: f64,
) -> std::fmt::Result {
    let arrow_len = 3.0;
    let dx = end_x - start_x;
    let dy = end_y - start_y;
    let angle = dy.atan2(dx);

    let arrow1_x = end_x - arrow_len * (angle - 0.5).cos();
    let arrow1_y = end_y - arrow_len * (angle - 0.5).sin();
    let arrow2_x = end_x - arrow_len * (angle + 0.5).cos();
    let arrow2_y = end_y - arrow_len * (angle + 0.5).sin();

    writeln!(
        plot,
        "<polygon points='{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}' fill='red'/>",
        end_x, end_y, arrow1_x, arrow1_y, arrow2_x, arrow2_y
    )
}

/// Helper function to add ASCII surface representation
fn add_ascii_surface<T>(
    plot: &mut String,
    data: &ArrayView2<T>,
    ascii_height: usize,
    ascii_width: usize,
) -> std::fmt::Result
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    let (height, width) = data.dim();
    let min_val = data.iter().cloned().fold(T::infinity(), T::min);
    let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);

    for i in 0..ascii_height {
        for j in 0..ascii_width {
            let data_i = (i * height) / ascii_height.max(1);
            let data_j = (j * width) / ascii_width.max(1);
            let value = data[[data_i, data_j]];
            let normalized = if max_val > min_val {
                ((value - min_val) / (max_val - min_val))
                    .to_f64()
                    .unwrap_or(0.0)
            } else {
                0.5
            };

            let char = match (normalized * 10.0) as u32 {
                0..=1 => ' ',
                2..=3 => '.',
                4..=5 => ':',
                6..=7 => '+',
                8..=9 => '*',
                _ => '#',
            };
            write!(plot, "{}", char)?;
        }
        writeln!(plot)?;
    }
    Ok(())
}

/// Generate a heatmap visualization of a 2D array
pub fn plot_heatmap<T>(data: &ArrayView2<T>, config: &PlotConfig) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if data.is_empty() {
        return Err(NdimageError::InvalidInput("Data array is empty".into()));
    }

    let (height, width) = data.dim();
    let mut plot = String::new();

    // Find min and max values for scaling
    let min_val = data.iter().cloned().fold(T::infinity(), T::min);
    let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);

    if max_val <= min_val {
        return Err(NdimageError::InvalidInput(
            "All values in array are the same".into(),
        ));
    }

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='heatmap-plot'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;

            // Create a simple HTML table representation
            writeln!(&mut plot, "<table style='border-collapse: collapse;'>")?;
            let display_height = height.min(20);
            let display_width = width.min(20);

            for i in 0..display_height {
                writeln!(&mut plot, "<tr>")?;
                for j in 0..display_width {
                    let data_i = (i * height) / display_height;
                    let data_j = (j * width) / display_width;
                    let value = data[[data_i, data_j]];
                    let normalized = ((value - min_val) / (max_val - min_val))
                        .to_f64()
                        .unwrap_or(0.0);

                    let intensity = (normalized * 255.0) as u8;
                    let color = format!("rgb({}, {}, {})", intensity, intensity, intensity);

                    writeln!(
                        &mut plot,
                        "<td style='width: 15px; height: 15px; background-color: {}; border: 1px solid #ccc;'></td>",
                        color
                    )?;
                }
                writeln!(&mut plot, "</tr>")?;
            }
            writeln!(&mut plot, "</table>")?;

            writeln!(&mut plot, "<p>Data dimensions: {}×{}</p>", height, width)?;
            writeln!(
                &mut plot,
                "<p>Value range: [{:.3}, {:.3}]</p>",
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {} (Heatmap)", config.title)?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "```")?;
            writeln!(&mut plot, "Data dimensions: {}×{}", height, width)?;
            writeln!(
                &mut plot,
                "Value range: [{:.3}, {:.3}]",
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot)?;

            // Simple ASCII art heatmap
            let display_height = height.min(30);
            let display_width = width.min(60);

            for i in 0..display_height {
                for j in 0..display_width {
                    let data_i = (i * height) / display_height;
                    let data_j = (j * width) / display_width;
                    let value = data[[data_i, data_j]];
                    let normalized = ((value - min_val) / (max_val - min_val))
                        .to_f64()
                        .unwrap_or(0.0);

                    let char = match (normalized * 9.0) as u32 {
                        0 => ' ',
                        1 => '.',
                        2 => ':',
                        3 => '-',
                        4 => '=',
                        5 => '+',
                        6 => '*',
                        7 => '#',
                        8 => '@',
                        _ => '█',
                    };
                    write!(&mut plot, "{}", char)?;
                }
                writeln!(&mut plot)?;
            }

            writeln!(&mut plot, "```")?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{} (Heatmap)", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len() + 10))?;
            writeln!(&mut plot)?;
            writeln!(&mut plot, "Data dimensions: {}×{}", height, width)?;
            writeln!(
                &mut plot,
                "Value range: [{:.3}, {:.3}]",
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot)?;

            // Simple ASCII representation for text mode
            let display_height = height.min(20);
            let display_width = width.min(40);

            for i in 0..display_height {
                for j in 0..display_width {
                    let data_i = (i * height) / display_height;
                    let data_j = (j * width) / display_width;
                    let value = data[[data_i, data_j]];
                    let normalized = ((value - min_val) / (max_val - min_val))
                        .to_f64()
                        .unwrap_or(0.0);

                    let char = match (normalized * 4.0) as u32 {
                        0 => ' ',
                        1 => '.',
                        2 => 'o',
                        3 => 'O',
                        _ => '#',
                    };
                    write!(&mut plot, "{}", char)?;
                }
                writeln!(&mut plot)?;
            }
        }
    }

    Ok(plot)
}

/// Alias for plot_gradient for backward compatibility
pub fn plot_gradient<T>(data: &ArrayView2<T>, config: &PlotConfig) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    // For now, this is an alias to the existing gradient plotting functionality
    // that was extracted from the original code
    plot_heatmap(data, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualization::types::{ColorMap, PlotConfig, ReportFormat};
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_plot_histogram() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = PlotConfig {
            title: "Test Histogram".to_string(),
            format: ReportFormat::Text,
            num_bins: 5,
            ..Default::default()
        };

        let result = plot_histogram(&data.view(), &config);
        assert!(result.is_ok());

        let plot_str = result.unwrap();
        assert!(plot_str.contains("Test Histogram"));
    }

    #[test]
    fn test_plot_profile() {
        let x_data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_data = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let config = PlotConfig {
            title: "Test Profile".to_string(),
            format: ReportFormat::Text,
            ..Default::default()
        };

        let result = plot_profile(&x_data.view(), &y_data.view(), &config);
        assert!(result.is_ok());

        let plot_str = result.unwrap();
        assert!(plot_str.contains("Test Profile"));
    }

    #[test]
    fn test_plot_surface() {
        let data = scirs2_core::ndarray::Array2::from_shape_fn((10, 10), |(i, j)| (i + j) as f64);
        let config = PlotConfig {
            title: "Test Surface".to_string(),
            format: ReportFormat::Text,
            ..Default::default()
        };

        let result = plot_surface(&data.view(), &config);
        assert!(result.is_ok());

        let plot_str = result.unwrap();
        assert!(plot_str.contains("Test Surface"));
        assert!(plot_str.contains("Data dimensions"));
    }

    #[test]
    fn test_plot_scatter() {
        let x_data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_data = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let config = PlotConfig {
            title: "Test Scatter".to_string(),
            format: ReportFormat::Html,
            ..Default::default()
        };

        let result = plot_scatter(&x_data.view(), &y_data.view(), &config);
        assert!(result.is_ok());

        let plot_str = result.unwrap();
        assert!(plot_str.contains("Test Scatter"));
        assert!(plot_str.contains("<svg"));
    }

    #[test]
    fn test_empty_data_error() {
        let empty_data = Array1::<f64>::from_vec(vec![]);
        let config = PlotConfig::default();

        let result = plot_histogram(&empty_data.view(), &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_data_error() {
        let x_data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y_data = Array1::from_vec(vec![1.0, 2.0]);
        let config = PlotConfig::default();

        let result = plot_profile(&x_data.view(), &y_data.view(), &config);
        assert!(result.is_err());
    }
}
