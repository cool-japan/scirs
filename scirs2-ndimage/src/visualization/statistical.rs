//! Statistical Visualization Functions
//!
//! This module provides specialized visualization functions for statistical analysis,
//! comparative studies, and multi-dataset visualization. These functions are designed
//! to support statistical research and data analysis workflows.

use ndarray::{ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use std::fmt::{Debug, Write};

use crate::error::{NdimageError, NdimageResult};
use crate::visualization::types::{PlotConfig, ReportFormat};
use statrs::statistics::Statistics;

/// Create an image montage/grid from multiple 2D arrays
///
/// This function arranges multiple images in a grid layout for comparison and analysis.
/// It automatically scales all images using global min/max values for consistent visualization.
///
/// # Arguments
///
/// * `images` - Slice of 2D array views representing the images to arrange
/// * `grid_cols` - Number of columns in the grid layout
/// * `config` - Plot configuration specifying format and styling
///
/// # Returns
///
/// A formatted string representation of the image grid in the specified format
///
/// # Examples
///
/// ```rust,ignore
/// use ndarray::Array2;
/// use scirs2_ndimage::visualization::{PlotConfig, ReportFormat, create_image_montage};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let img1 = Array2::zeros((10, 10));
/// let img2 = Array2::ones((10, 10));
/// let images = vec![img1.view(), img2.view()];
///
/// let config = PlotConfig::new()
///     .with_format(ReportFormat::Text)
///     .with_title("Image Comparison");
///
/// let montage = create_image_montage(&images, 2, &config)?;
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn create_image_montage<T>(
    images: &[ArrayView2<T>],
    grid_cols: usize,
    config: &PlotConfig,
) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if images.is_empty() {
        return Err(NdimageError::InvalidInput("No images provided".into()));
    }

    if grid_cols == 0 {
        return Err(NdimageError::InvalidInput(
            "Grid columns must be positive".into(),
        ));
    }

    let mut plot = String::new();
    let grid_rows = (images.len() + grid_cols - 1) / grid_cols;

    // Find global min/max for consistent scaling
    let mut global_min = T::infinity();
    let mut global_max = T::neg_infinity();

    for image in images {
        let min_val = image.iter().cloned().fold(T::infinity(), T::min);
        let max_val = image.iter().cloned().fold(T::neg_infinity(), T::max);
        global_min = global_min.min(min_val);
        global_max = global_max.max(max_val);
    }

    if global_max <= global_min {
        return Err(NdimageError::InvalidInput(
            "All image values are the same".into(),
        ));
    }

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='image-montage'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(&mut plot, "<div class='montage-grid' style='display: grid; grid-template-columns: repeat({}, 1fr); gap: 10px;'>", grid_cols)?;

            for (idx, image) in images.iter().enumerate() {
                let (height, width) = image.dim();
                writeln!(&mut plot, "<div class='montage-cell'>")?;
                writeln!(&mut plot, "<h4>Image {}</h4>", idx + 1)?;
                writeln!(
                    &mut plot,
                    "<div class='image-data' data-width='{}' data-height='{}'>",
                    width, height
                )?;

                // Simple representation - would need actual image rendering in practice
                writeln!(&mut plot, "<p>{}×{} array</p>", height, width)?;
                writeln!(
                    &mut plot,
                    "<p>Range: [{:.3}, {:.3}]</p>",
                    image
                        .iter()
                        .cloned()
                        .fold(T::infinity(), T::min)
                        .to_f64()
                        .unwrap_or(0.0),
                    image
                        .iter()
                        .cloned()
                        .fold(T::neg_infinity(), T::max)
                        .to_f64()
                        .unwrap_or(0.0)
                )?;

                writeln!(&mut plot, "</div>")?;
                writeln!(&mut plot, "</div>")?;
            }

            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "<div class='montage-info'>")?;
            writeln!(
                &mut plot,
                "<p>Global range: [{:.3}, {:.3}]</p>",
                global_min.to_f64().unwrap_or(0.0),
                global_max.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                &mut plot,
                "<p>Grid: {} rows × {} columns</p>",
                grid_rows, grid_cols
            )?;
            writeln!(&mut plot, "</div>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {} (Image Montage)", config.title)?;
            writeln!(&mut plot)?;
            writeln!(
                &mut plot,
                "Grid layout: {} rows × {} columns",
                grid_rows, grid_cols
            )?;
            writeln!(
                &mut plot,
                "Global value range: [{:.3}, {:.3}]",
                global_min.to_f64().unwrap_or(0.0),
                global_max.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot)?;

            for (idx, image) in images.iter().enumerate() {
                let (height, width) = image.dim();
                let min_val = image.iter().cloned().fold(T::infinity(), T::min);
                let max_val = image.iter().cloned().fold(T::neg_infinity(), T::max);

                writeln!(&mut plot, "### Image {}", idx + 1)?;
                writeln!(&mut plot, "- Dimensions: {}×{}", height, width)?;
                writeln!(
                    &mut plot,
                    "- Value range: [{:.3}, {:.3}]",
                    min_val.to_f64().unwrap_or(0.0),
                    max_val.to_f64().unwrap_or(0.0)
                )?;
                writeln!(&mut plot)?;
            }
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{} (Image Montage)", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len() + 16))?;
            writeln!(&mut plot)?;
            writeln!(
                &mut plot,
                "Grid layout: {} rows × {} columns",
                grid_rows, grid_cols
            )?;
            writeln!(
                &mut plot,
                "Global value range: [{:.3}, {:.3}]",
                global_min.to_f64().unwrap_or(0.0),
                global_max.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut plot)?;

            for (idx, image) in images.iter().enumerate() {
                let (height, width) = image.dim();
                let min_val = image.iter().cloned().fold(T::infinity(), T::min);
                let max_val = image.iter().cloned().fold(T::neg_infinity(), T::max);

                writeln!(
                    &mut plot,
                    "Image {}: {}×{}, range [{:.3}, {:.3}]",
                    idx + 1,
                    height,
                    width,
                    min_val.to_f64().unwrap_or(0.0),
                    max_val.to_f64().unwrap_or(0.0)
                )?;
            }
        }
    }

    Ok(plot)
}

/// Generate a comparative statistical plot for multiple datasets
///
/// This function creates a comprehensive statistical comparison table showing
/// key statistics (count, mean, standard deviation, min, max) for multiple datasets.
/// Useful for comparing different experimental conditions or processing results.
///
/// # Arguments
///
/// * `datasets` - Slice of tuples containing dataset names and their 1D data arrays
/// * `config` - Plot configuration specifying format and styling
///
/// # Returns
///
/// A formatted statistical comparison table in the specified format
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_ndimage::visualization::{PlotConfig, ReportFormat, plot_statistical_comparison};
///
/// let control = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let treatment = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);
///
/// let datasets = vec![
///     ("Control", control.view()),
///     ("Treatment", treatment.view()),
/// ];
///
/// let config = PlotConfig::new()
///     .with_format(ReportFormat::Markdown)
///     .with_title("Statistical Comparison");
///
/// let comparison = plot_statistical_comparison(&datasets, &config)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(dead_code)]
pub fn plot_statistical_comparison<T>(
    datasets: &[(&str, ArrayView1<T>)],
    config: &PlotConfig,
) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if datasets.is_empty() {
        return Err(NdimageError::InvalidInput("No datasets provided".into()));
    }

    let mut plot = String::new();

    // Compute statistics for each dataset
    let mut stats = Vec::new();
    for (name, data) in datasets {
        if data.is_empty() {
            continue;
        }

        let mean = data.mean().unwrap_or(T::zero());
        let min_val = data.iter().cloned().fold(T::infinity(), T::min);
        let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);
        let variance = data
            .mapv(|x| (x - mean) * (x - mean))
            .mean()
            .unwrap_or(T::zero());
        let std_dev = variance.sqrt();

        stats.push((name, mean, std_dev, min_val, max_val, data.len()));
    }

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='statistical-comparison'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(&mut plot, "<table class='stats-table'>")?;
            writeln!(&mut plot, "<tr><th>Dataset</th><th>Count</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr>")?;

            for (name, mean, std_dev, min_val, max_val, count) in &stats {
                writeln!(
                    &mut plot,
                    "<tr><td>{}</td><td>{}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td></tr>",
                    name, count,
                    mean.to_f64().unwrap_or(0.0),
                    std_dev.to_f64().unwrap_or(0.0),
                    min_val.to_f64().unwrap_or(0.0),
                    max_val.to_f64().unwrap_or(0.0)
                )?;
            }

            writeln!(&mut plot, "</table>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {} (Statistical Comparison)", config.title)?;
            writeln!(&mut plot)?;
            writeln!(
                &mut plot,
                "| Dataset | Count | Mean | Std Dev | Min | Max |"
            )?;
            writeln!(
                &mut plot,
                "|---------|-------|------|---------|-----|-----|"
            )?;

            for (name, mean, std_dev, min_val, max_val, count) in &stats {
                writeln!(
                    &mut plot,
                    "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} |",
                    name,
                    count,
                    mean.to_f64().unwrap_or(0.0),
                    std_dev.to_f64().unwrap_or(0.0),
                    min_val.to_f64().unwrap_or(0.0),
                    max_val.to_f64().unwrap_or(0.0)
                )?;
            }
            writeln!(&mut plot)?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{} (Statistical Comparison)", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len() + 25))?;
            writeln!(&mut plot)?;
            writeln!(
                &mut plot,
                "{:<15} {:>8} {:>10} {:>10} {:>10} {:>10}",
                "Dataset", "Count", "Mean", "Std Dev", "Min", "Max"
            )?;
            writeln!(&mut plot, "{}", "-".repeat(75))?;

            for (name, mean, std_dev, min_val, max_val, count) in &stats {
                writeln!(
                    &mut plot,
                    "{:<15} {:>8} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                    name,
                    count,
                    mean.to_f64().unwrap_or(0.0),
                    std_dev.to_f64().unwrap_or(0.0),
                    min_val.to_f64().unwrap_or(0.0),
                    max_val.to_f64().unwrap_or(0.0)
                )?;
            }
        }
    }

    Ok(plot)
}

/// Calculate statistical summary for a dataset
///
/// Helper function that computes comprehensive statistics for a single dataset.
/// Used internally by other statistical visualization functions.
///
/// # Arguments
///
/// * `data` - 1D array view of the dataset
///
/// # Returns
///
/// Tuple containing (mean, std_dev, min, max, count)
pub fn calculate_dataset_statistics<T>(data: &ArrayView1<T>) -> (T, T, T, T, usize)
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if data.is_empty() {
        return (T::zero(), T::zero(), T::zero(), T::zero(), 0);
    }

    let mean = data.mean().unwrap_or(T::zero());
    let min_val = data.iter().cloned().fold(T::infinity(), T::min);
    let max_val = data.iter().cloned().fold(T::neg_infinity(), T::max);
    let variance = data
        .mapv(|x| (x - mean) * (x - mean))
        .mean()
        .unwrap_or(T::zero());
    let std_dev = variance.sqrt();

    (mean, std_dev, min_val, max_val, data.len())
}

/// Generate correlation matrix visualization for multiple datasets
///
/// Creates a text-based correlation matrix showing relationships between datasets.
/// Useful for understanding data dependencies and relationships.
///
/// # Arguments
///
/// * `datasets` - Slice of tuples containing dataset names and their 1D data arrays
/// * `config` - Plot configuration specifying format and styling
///
/// # Returns
///
/// A formatted correlation matrix in the specified format
#[allow(dead_code)]
pub fn plot_correlation_matrix<T>(
    datasets: &[(&str, ArrayView1<T>)],
    config: &PlotConfig,
) -> NdimageResult<String>
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if datasets.len() < 2 {
        return Err(NdimageError::InvalidInput(
            "Need at least 2 datasets for correlation".into(),
        ));
    }

    let mut plot = String::new();
    let n = datasets.len();

    // Calculate correlation matrix
    let mut correlations = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                correlations[i][j] = 1.0;
            } else {
                let corr = calculate_correlation(&datasets[i].1, &datasets[j].1);
                correlations[i][j] = corr;
            }
        }
    }

    match config.format {
        ReportFormat::Html => {
            writeln!(&mut plot, "<div class='correlation-matrix'>")?;
            writeln!(&mut plot, "<h3>{}</h3>", config.title)?;
            writeln!(&mut plot, "<table class='correlation-table'>")?;

            // Header row
            write!(&mut plot, "<tr><th></th>")?;
            for (name, _) in datasets {
                write!(&mut plot, "<th>{}</th>", name)?;
            }
            writeln!(&mut plot, "</tr>")?;

            // Data rows
            for i in 0..n {
                write!(&mut plot, "<tr><th>{}</th>", datasets[i].0)?;
                for j in 0..n {
                    let corr = correlations[i][j];
                    let color_class = if corr.abs() > 0.7 {
                        "strong-corr"
                    } else {
                        "weak-corr"
                    };
                    write!(&mut plot, "<td class='{}'>{:.3}</td>", color_class, corr)?;
                }
                writeln!(&mut plot, "</tr>")?;
            }

            writeln!(&mut plot, "</table>")?;
            writeln!(&mut plot, "</div>")?;
        }
        ReportFormat::Markdown => {
            writeln!(&mut plot, "## {} (Correlation Matrix)", config.title)?;
            writeln!(&mut plot)?;

            // Header row
            write!(&mut plot, "|")?;
            for (name, _) in datasets {
                write!(&mut plot, " {} |", name)?;
            }
            writeln!(&mut plot)?;

            // Separator row
            write!(&mut plot, "|")?;
            for _ in 0..n {
                write!(&mut plot, "------|")?;
            }
            writeln!(&mut plot)?;

            // Data rows
            for i in 0..n {
                write!(&mut plot, "| **{}** |", datasets[i].0)?;
                for j in 0..n {
                    write!(&mut plot, " {:.3} |", correlations[i][j])?;
                }
                writeln!(&mut plot)?;
            }
            writeln!(&mut plot)?;
        }
        ReportFormat::Text => {
            writeln!(&mut plot, "{} (Correlation Matrix)", config.title)?;
            writeln!(&mut plot, "{}", "=".repeat(config.title.len() + 20))?;
            writeln!(&mut plot)?;

            // Header row
            write!(&mut plot, "{:>12}", "")?;
            for (name, _) in datasets {
                write!(&mut plot, " {:>8}", &name[..name.len().min(8)])?;
            }
            writeln!(&mut plot)?;

            // Data rows
            for i in 0..n {
                write!(
                    &mut plot,
                    "{:>12}",
                    &datasets[i].0[..datasets[i].0.len().min(12)]
                )?;
                for j in 0..n {
                    write!(&mut plot, " {:>8.3}", correlations[i][j])?;
                }
                writeln!(&mut plot)?;
            }
        }
    }

    Ok(plot)
}

/// Calculate Pearson correlation coefficient between two datasets
fn calculate_correlation<T>(data1: &ArrayView1<T>, data2: &ArrayView1<T>) -> f64
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
{
    if data1.len() != data2.len() || data1.len() < 2 {
        return 0.0;
    }

    let mean1 = data1.mean().unwrap_or(T::zero()).to_f64().unwrap_or(0.0);
    let mean2 = data2.mean().unwrap_or(T::zero()).to_f64().unwrap_or(0.0);

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..data1.len() {
        let x = data1[i].to_f64().unwrap_or(0.0) - mean1;
        let y = data2[i].to_f64().unwrap_or(0.0) - mean2;

        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }

    let denominator = (sum_x2 * sum_y2).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        sum_xy / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_create_image_montage() {
        let img1 = Array2::zeros((5, 5));
        let img2 = Array2::ones((5, 5));
        let img3 = Array2::from_elem((5, 5), 2.0);

        let images = vec![img1.view(), img2.view(), img3.view()];

        let config = PlotConfig::new()
            .with_format(ReportFormat::Text)
            .with_title("Test Montage");

        let result = create_image_montage(&images, 2, &config);
        assert!(result.is_ok());

        let montage = result.unwrap();
        assert!(montage.contains("Test Montage"));
        assert!(montage.contains("Grid layout: 2 rows × 2 columns"));
        assert!(montage.contains("Image 1: 5×5"));
        assert!(montage.contains("Image 2: 5×5"));
        assert!(montage.contains("Image 3: 5×5"));
    }

    #[test]
    fn test_plot_statistical_comparison() {
        let data1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let data2 = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let datasets = vec![("Dataset A", data1.view()), ("Dataset B", data2.view())];

        let config = PlotConfig::new()
            .with_format(ReportFormat::Markdown)
            .with_title("Statistical Test");

        let result = plot_statistical_comparison(&datasets, &config);
        assert!(result.is_ok());

        let comparison = result.unwrap();
        assert!(comparison.contains("Statistical Test"));
        assert!(comparison.contains("Dataset A"));
        assert!(comparison.contains("Dataset B"));
        assert!(comparison.contains("| Dataset | Count | Mean"));
    }

    #[test]
    fn test_calculate_dataset_statistics() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (mean, std_dev, min_val, max_val, count) = calculate_dataset_statistics(&data.view());

        assert!((mean - 3.0).abs() < 1e-6);
        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 5.0);
        assert_eq!(count, 5);
        assert!(std_dev > 0.0);
    }

    #[test]
    fn test_calculate_correlation() {
        let data1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let data2 = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]); // Perfect positive correlation

        let corr = calculate_correlation(&data1.view(), &data2.view());
        assert!((corr - 1.0).abs() < 1e-10); // Should be very close to 1.0
    }

    #[test]
    fn test_plot_correlation_matrix() {
        let data1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let data2 = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let data3 = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);

        let datasets = vec![
            ("Data A", data1.view()),
            ("Data B", data2.view()),
            ("Data C", data3.view()),
        ];

        let config = PlotConfig::new()
            .with_format(ReportFormat::Text)
            .with_title("Correlation Test");

        let result = plot_correlation_matrix(&datasets, &config);
        assert!(result.is_ok());

        let matrix = result.unwrap();
        assert!(matrix.contains("Correlation Test"));
        assert!(matrix.contains("Data A"));
        assert!(matrix.contains("Data B"));
        assert!(matrix.contains("Data C"));
    }

    #[test]
    fn test_empty_image_montage() {
        let images: Vec<ndarray::ArrayView2<f64>> = vec![];
        let config = PlotConfig::new();

        let result = create_image_montage(&images, 2, &config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No images provided"));
    }

    #[test]
    fn test_zero_grid_cols() {
        let img = Array2::<f64>::zeros((5, 5));
        let images = vec![img.view()];
        let config = PlotConfig::new();

        let result = create_image_montage(&images, 0, &config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Grid columns must be positive"));
    }
}
