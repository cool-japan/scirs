//! Backend adapters for visualization
//!
//! This module provides adapters for different plotting backends like plotters, plotly, etc.
//! It allows metrics visualizations to be rendered using different plotting libraries.

use std::error::Error;
use std::path::Path;

use crate::visualization::{VisualizationData, VisualizationMetadata, VisualizationOptions};

#[cfg(feature = "plotly_backend")]
mod plotly;
#[cfg(feature = "plotly_backend")]
mod plotly_interactive;
#[cfg(feature = "plotters_backend")]
mod plotters;

#[cfg(feature = "plotly_backend")]
pub use self::plotly::PlotlyBackend;
#[cfg(feature = "plotly_backend")]
pub use self::plotly_interactive::{PlotlyInteractiveBackend, PlotlyInteractiveBackendInterface};
#[cfg(feature = "plotters_backend")]
pub use self::plotters::PlottersBackend;

/// A trait for plotting backends
///
/// This trait provides a common interface for rendering visualizations using different
/// plotting libraries. It allows metrics visualizations to be rendered using the most
/// appropriate backend for a given application.
pub trait PlottingBackend {
    /// Save a visualization to a file
    ///
    /// # Arguments
    ///
    /// * `data` - The visualization data to render
    /// * `metadata` - The visualization metadata (title, labels, etc.)
    /// * `options` - Options for the visualization (size, dpi, etc.)
    /// * `path` - The output file path
    ///
    /// # Returns
    ///
    /// * `Result<(), Box<dyn Error>>` - Ok if the visualization was successfully saved,
    ///   or an error if something went wrong
    fn save_to_file(
        &self,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<dyn Error>>;

    /// Render a visualization to a byte array as SVG
    ///
    /// # Arguments
    ///
    /// * `data` - The visualization data to render
    /// * `metadata` - The visualization metadata (title, labels, etc.)
    /// * `options` - Options for the visualization (size, dpi, etc.)
    ///
    /// # Returns
    ///
    /// * `Result<Vec<u8>, Box<dyn Error>>` - A byte array containing the SVG representation
    ///   of the visualization
    fn render_svg(
        &self,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<Vec<u8>, Box<dyn Error>>;

    /// Render a visualization to a byte array as PNG
    ///
    /// # Arguments
    ///
    /// * `data` - The visualization data to render
    /// * `metadata` - The visualization metadata (title, labels, etc.)
    /// * `options` - Options for the visualization (size, dpi, etc.)
    ///
    /// # Returns
    ///
    /// * `Result<Vec<u8>, Box<dyn Error>>` - A byte array containing the PNG representation
    ///   of the visualization
    fn render_png(
        &self,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<Vec<u8>, Box<dyn Error>>;
}

/// Create the default plotting backend
///
/// This function returns the default plotting backend for the current configuration.
/// The default backend is determined by the available feature flags.
///
/// # Example
///
/// ```
/// use scirs2_metrics::visualization::backends;
///
/// let backend = backends::default_backend();
/// ```
#[allow(dead_code)]
pub fn default_backend() -> impl PlottingBackend {
    #[cfg(feature = "plotly_backend")]
    {
        PlotlyBackend::new()
    }
    #[cfg(not(feature = "plotly_backend"))]
    #[cfg(feature = "plotters_backend")]
    {
        PlottersBackend::new()
    }
    #[cfg(not(feature = "plotly_backend"))]
    #[cfg(not(feature = "plotters_backend"))]
    {
        // Fallback implementation that does nothing
        struct NoopBackend;

        impl PlottingBackend for NoopBackend {
            fn save_to_file(
                &self,
                _data: &VisualizationData,
                _metadata: &VisualizationMetadata,
                _options: &VisualizationOptions,
                _path: impl AsRef<Path>,
            ) -> Result<(), Box<dyn Error>> {
                Err("No visualization backend available. Enable either 'plotly_backend' or 'plotters_backend' feature.".into())
            }

            fn render_svg(
                &self,
                _data: &VisualizationData,
                _metadata: &VisualizationMetadata,
                _options: &VisualizationOptions,
            ) -> Result<Vec<u8>, Box<dyn Error>> {
                Err("No visualization backend available. Enable either 'plotly_backend' or 'plotters_backend' feature.".into())
            }

            fn render_png(
                &self,
                _data: &VisualizationData,
                _metadata: &VisualizationMetadata,
                _options: &VisualizationOptions,
            ) -> Result<Vec<u8>, Box<dyn Error>> {
                Err("No visualization backend available. Enable either 'plotly_backend' or 'plotters_backend' feature.".into())
            }
        }

        NoopBackend
    }
}

/// Create the default interactive plotting backend
///
/// This function returns the default interactive plotting backend.
/// Currently, only Plotly is supported for interactive visualizations.
///
/// # Example
///
/// ```
/// use scirs2_metrics::visualization::backends;
///
/// let backend = backends::default_interactive_backend();
/// ```
#[cfg(feature = "plotly_backend")]
#[allow(dead_code)]
pub fn default_interactive_backend() -> PlotlyInteractiveBackend {
    PlotlyInteractiveBackend::new()
}

/// Enhance a visualization data structure with additional data
///
/// This function adds additional data to a visualization data structure,
/// such as computed averages, confidence intervals, etc.
///
/// # Arguments
///
/// * `data` - The visualization data to enhance
/// * `metadata` - The visualization metadata
///
/// # Returns
///
/// * `VisualizationData` - The enhanced visualization data
#[allow(dead_code)]
pub fn enhance_visualization(
    data: &VisualizationData,
    metadata: &VisualizationMetadata,
) -> VisualizationData {
    // Create a copy of the original data
    let mut enhanced = data.clone();

    // Enhance based on plot type
    match metadata.plot_type {
        crate::visualization::PlotType::Line => {
            // Add a trend line for line plots
            if data.x.len() > 5 && data.y.len() > 5 {
                // Simple linear regression
                let n = data.x.len() as f64;
                let sum_x: f64 = data.x.iter().sum();
                let sum_y: f64 = data.y.iter().sum();
                let sum_xy: f64 = data.x.iter().zip(data.y.iter()).map(|(&x, &y)| x * y).sum();
                let sum_xx: f64 = data.x.iter().map(|&x| x * x).sum();

                // Calculate slope and intercept
                let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
                let intercept = (sum_y - slope * sum_x) / n;

                // Add trend line data
                let trend_line: Vec<f64> = data.x.iter().map(|&x| slope * x + intercept).collect();

                // Store trend line in the enhanced data
                enhanced
                    .auxiliary_data
                    .insert("trend_line".to_string(), trend_line);
                enhanced
                    .auxiliary_metadata
                    .insert("trend_slope".to_string(), slope.to_string());
                enhanced
                    .auxiliary_metadata
                    .insert("trend_intercept".to_string(), intercept.to_string());
            }
        }
        crate::visualization::PlotType::Scatter => {
            // Add a center of mass for scatter plots
            if data.x.len() > 0 && data.y.len() > 0 {
                let center_x = data.x.iter().sum::<f64>() / data.x.len() as f64;
                let center_y = data.y.iter().sum::<f64>() / data.y.len() as f64;

                enhanced
                    .auxiliary_data
                    .insert("center_x".to_string(), vec![center_x]);
                enhanced
                    .auxiliary_data
                    .insert("center_y".to_string(), vec![center_y]);
            }
        }
        _ => {
            // No enhancement for other plot types yet
        }
    }

    enhanced
}
