//! Visualization and Reporting Utilities for Image Processing Results
//!
//! This module provides comprehensive tools for creating visual representations of
//! image processing results, statistical plots, and comprehensive analysis reports.
//! Designed for scientific documentation and presentation of image analysis workflows.
//!
//! The module is organized into focused sub-modules for better maintainability:
//!
//! - [`types`] - Core data structures, enums, and configuration types
//! - [`colormap`] - Color map implementations for scientific visualization
//! - [`plotting`] - Basic plotting functions (histograms, profiles, heatmaps, etc.)
//! - [`reports`] - Comprehensive report generation system
//! - [`statistical`] - Statistical visualization and comparison functions
//!
//! # Examples
//!
//! ## Basic Plotting
//!
//! ```rust
//! use scirs2_core::ndarray::Array2;
//! use scirs2_ndimage::visualization::{PlotConfig, ColorMap, ReportFormat, plot_histogram};
//!
//! let data = Array2::from_shape_fn((100, 100), |(i, j)| {
//!     ((i as f64).sin() * (j as f64).cos()).abs()
//! });
//!
//! let config = PlotConfig::new()
//!     .with_colormap(ColorMap::Viridis)
//!     .with_format(ReportFormat::Html)
//!     .with_title("Sample Heatmap");
//!
//! let histogram = plot_histogram(&data.view().into_shape(10000)?.view(), &config)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Report Generation
//!
//! ```rust
//! use scirs2_core::ndarray::Array2;
//! use scirs2_ndimage::visualization::{ReportConfig, ReportFormat, generate_report};
//!
//! let image = Array2::from_shape_fn((50, 50), |(i, j)| {
//!     (i + j) as f64 / 100.0
//! });
//!
//! let config = ReportConfig::new()
//!     .with_format(ReportFormat::Markdown)
//!     .with_header("Analysis Report", "SciRS2 NDImage")
//!     .with_all_sections();
//!
//! let report = generate_report(&image.view(), None, None, &config)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Statistical Comparison
//!
//! ```rust
//! use scirs2_core::ndarray::Array1;
//! use scirs2_ndimage::visualization::{PlotConfig, ReportFormat, plot_statistical_comparison};
//!
//! let control = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
//! let treatment = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! let datasets = vec![
//!     ("Control Group", control.view()),
//!     ("Treatment Group", treatment.view()),
//! ];
//!
//! let config = PlotConfig::new()
//!     .with_format(ReportFormat::Markdown)
//!     .with_title("Group Comparison");
//!
//! let comparison = plot_statistical_comparison(&datasets, &config)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Sub-module declarations
pub mod colormap;
pub mod plotting;
pub mod reports;
pub mod statistical;
pub mod types;

// Re-export all core types for backward compatibility
pub use types::{ColorMap, ColorSchemes, PlotConfig, ReportConfig, ReportFormat, RgbColor};

// Re-export all colormap functions
pub use colormap::{
    apply_colormap_to_array, autumn_colormap, cool_colormap, create_colormap,
    get_colormap_function, gray_colormap, hot_colormap, inferno_colormap, jet_colormap,
    plasma_colormap, spring_colormap, summer_colormap, viridis_colormap, winter_colormap,
};

// Re-export all plotting functions
pub use plotting::{
    plot_contour, plot_gradient, plot_heatmap, plot_histogram, plot_profile, plot_surface,
    visualize_gradient,
};

// Re-export all report generation functions
pub use reports::{
    add_basic_statistics, add_image_info, add_quality_metrics, add_texture_metrics, generate_report,
};

// Re-export all statistical visualization functions
pub use statistical::{
    calculate_dataset_statistics, create_image_montage, plot_correlation_matrix,
    plot_statistical_comparison,
};

// Legacy aliases for backward compatibility
/// Legacy alias for `create_image_montage`
pub use statistical::create_image_montage as createimage_montage;

/// Legacy alias for `add_image_info`
pub use reports::add_image_info as addimage_info;

/// Legacy alias for `add_quality_metrics`
pub use reports::add_quality_metrics as add_qualitymetrics;

/// Legacy alias for `add_texture_metrics`
pub use reports::add_texture_metrics as addtexturemetrics;

// Export utilities module inline for compatibility
/// Export utilities for saving visualization output to files
pub mod export {
    use super::reports::generate_report;
    use super::types::{ReportConfig, ReportFormat};
    use crate::analysis::{ImageQualityMetrics, TextureMetrics};
    use crate::error::{NdimageError, NdimageResult};
    use scirs2_core::ndarray::ArrayView2;
    use scirs2_core::numeric::{Float, FromPrimitive, ToPrimitive};
    use std::fmt::Debug;
    use std::fs;
    use std::path::Path;

    /// Export configuration for file output
    #[derive(Debug, Clone)]
    pub struct ExportConfig {
        /// Output file path
        pub output_path: String,
        /// Image quality (for compressed formats)
        pub quality: Option<u8>,
        /// DPI for vector formats
        pub dpi: Option<u32>,
        /// Whether to include metadata
        pub include_metadata: bool,
    }

    impl Default for ExportConfig {
        fn default() -> Self {
            Self {
                output_path: "output.html".to_string(),
                quality: Some(95),
                dpi: Some(300),
                include_metadata: true,
            }
        }
    }

    /// Save a generated plot to file
    pub fn save_plot(content: &str, config: &ExportConfig) -> NdimageResult<()> {
        let path = Path::new(&config.output_path);

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                NdimageError::ComputationError(format!("Failed to create directory: {}", e))
            })?;
        }

        // Add metadata if requested
        let mut output_content = content.to_string();
        if config.include_metadata {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let metadata = format!(
                "\n<!-- Generated by scirs2-ndimage visualization module at {} -->\n",
                timestamp
            );

            if content.contains("</html>") {
                output_content = content.replace("</html>", &format!("{}</html>", metadata));
            } else if content.contains("# ") {
                output_content = format!(
                    "{}\n{}",
                    content,
                    metadata.replace("<!--", "").replace("-->", "")
                );
            } else {
                output_content = format!(
                    "{}\n{}",
                    content,
                    metadata.replace("<!--", "").replace("-->", "")
                );
            }
        }

        fs::write(path, output_content)
            .map_err(|e| NdimageError::ComputationError(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Generate and save a comprehensive analysis report
    pub fn export_analysis_report<T>(
        image: &ArrayView2<T>,
        qualitymetrics: Option<&ImageQualityMetrics<T>>,
        texturemetrics: Option<&TextureMetrics<T>>,
        output_path: &str,
        format: ReportFormat,
    ) -> NdimageResult<()>
    where
        T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
    {
        let config = ReportConfig {
            title: "Comprehensive Image Analysis Report".to_string(),
            author: "SciRS2 NDImage".to_string(),
            format,
            ..ReportConfig::default()
        };

        let report = generate_report(image, qualitymetrics, texturemetrics, &config)?;

        let export_config = ExportConfig {
            output_path: output_path.to_string(),
            ..ExportConfig::default()
        };

        save_plot(&report, &export_config)?;
        Ok(())
    }
}

// Advanced visualization utilities module inline for compatibility
/// Advanced visualization utilities
pub mod advanced {
    use super::plotting::plot_heatmap;
    use super::types::{ColorMap, PlotConfig, ReportFormat};
    use crate::error::{NdimageError, NdimageResult};
    use scirs2_core::ndarray::ArrayView2;
    use scirs2_core::numeric::{Float, FromPrimitive, ToPrimitive, Zero};
    use std::fmt::{Debug, Write};

    /// Create an interactive HTML visualization with multiple views
    pub fn create_interactive_visualization<T>(
        image: &ArrayView2<T>,
        title: &str,
    ) -> NdimageResult<String>
    where
        T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
    {
        let mut html = String::new();

        writeln!(&mut html, "<!DOCTYPE html>")?;
        writeln!(&mut html, "<html><head>")?;
        writeln!(&mut html, "<title>{}</title>", title)?;
        writeln!(&mut html, "<style>")?;
        writeln!(
            &mut html,
            r#"
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .visualization-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .plot-panel {{ border: 1px solid #ccc; padding: 15px; border-radius: 5px; min-width: 300px; }}
            .plot-title {{ font-weight: bold; margin-bottom: 10px; color: #333; }}
            .controls {{ margin-bottom: 15px; }}
            .control-group {{ margin-bottom: 10px; }}
            label {{ display: inline-block; width: 100px; }}
            select, input {{ margin-left: 10px; }}
            .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
            .stat-item {{ background: #f5f5f5; padding: 8px; border-radius: 3px; }}
            .heatmap-container {{ position: relative; width: 400px; height: 300px; }}
            .colorbar {{ width: 20px; height: 300px; background: linear-gradient(to top, blue, cyan, yellow, red); }}
        "#
        )?;
        writeln!(&mut html, "</style>")?;
        writeln!(&mut html, "<script>")?;
        writeln!(
            &mut html,
            r#"
            function updateVisualization() {{
                const colormap = document.getElementById('colormap').value;
                const plotType = document.getElementById('plotType').value;
                // Update visualization based on controls
                console.log('Updating visualization:', colormap, plotType);
            }}

            function exportView() {{
                const content = document.getElementById('main-content').innerHTML;
                const blob = new Blob([content], {{ type: 'text/html' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'visualization.html';
                a.click();
                URL.revokeObjectURL(url);
            }}
        "#
        )?;
        writeln!(&mut html, "</script>")?;
        writeln!(&mut html, "</head><body>")?;

        writeln!(&mut html, "<h1>{}</h1>", title)?;

        // Controls panel
        writeln!(&mut html, "<div class='controls'>")?;
        writeln!(&mut html, "<div class='control-group'>")?;
        writeln!(&mut html, "<label>Color Map:</label>")?;
        writeln!(
            &mut html,
            r#"<select id="colormap" onchange="updateVisualization()">"#
        )?;
        writeln!(&mut html, "<option value='viridis'>Viridis</option>")?;
        writeln!(&mut html, "<option value='plasma'>Plasma</option>")?;
        writeln!(&mut html, "<option value='jet'>Jet</option>")?;
        writeln!(&mut html, "<option value='hot'>Hot</option>")?;
        writeln!(&mut html, "</select>")?;
        writeln!(&mut html, "</div>")?;

        writeln!(&mut html, "<div class='control-group'>")?;
        writeln!(&mut html, "<label>Plot Type:</label>")?;
        writeln!(
            &mut html,
            r#"<select id="plotType" onchange="updateVisualization()">"#
        )?;
        writeln!(&mut html, "<option value='heatmap'>Heatmap</option>")?;
        writeln!(&mut html, "<option value='contour'>Contour</option>")?;
        writeln!(&mut html, "<option value='surface'>3D Surface</option>")?;
        writeln!(&mut html, "</select>")?;
        writeln!(&mut html, "</div>")?;

        writeln!(
            &mut html,
            r#"<button onclick="exportView()">Export View</button>"#
        )?;
        writeln!(&mut html, "</div>")?;

        writeln!(
            &mut html,
            "<div id='main-content' class='visualization-container'>"
        )?;

        // Statistics panel
        let (height, width) = image.dim();
        let mean = image.mean().unwrap_or(T::zero());
        let min_val = image.iter().cloned().fold(T::infinity(), T::min);
        let max_val = image.iter().cloned().fold(T::neg_infinity(), T::max);

        writeln!(&mut html, "<div class='plot-panel'>")?;
        writeln!(&mut html, "<div class='plot-title'>Image Statistics</div>")?;
        writeln!(&mut html, "<div class='stats-grid'>")?;
        writeln!(&mut html, "<div class='stat-item'>Width: {}</div>", width)?;
        writeln!(&mut html, "<div class='stat-item'>Height: {}</div>", height)?;
        writeln!(
            &mut html,
            "<div class='stat-item'>Mean: {:.4}</div>",
            mean.to_f64().unwrap_or(0.0)
        )?;
        writeln!(
            &mut html,
            "<div class='stat-item'>Min: {:.4}</div>",
            min_val.to_f64().unwrap_or(0.0)
        )?;
        writeln!(
            &mut html,
            "<div class='stat-item'>Max: {:.4}</div>",
            max_val.to_f64().unwrap_or(0.0)
        )?;
        writeln!(
            &mut html,
            "<div class='stat-item'>Pixels: {}</div>",
            width * height
        )?;
        writeln!(&mut html, "</div>")?;
        writeln!(&mut html, "</div>")?;

        // Heatmap panel
        writeln!(&mut html, "<div class='plot-panel'>")?;
        writeln!(&mut html, "<div class='plot-title'>Heatmap View</div>")?;
        writeln!(&mut html, "<div class='heatmap-container'>")?;

        // Generate a simplified heatmap representation
        let config = PlotConfig {
            format: ReportFormat::Html,
            colormap: ColorMap::Viridis,
            title: "Interactive Heatmap".to_string(),
            ..PlotConfig::default()
        };

        let heatmap = plot_heatmap(image, &config)?;
        writeln!(&mut html, "{}", heatmap)?;

        writeln!(&mut html, "</div>")?;
        writeln!(&mut html, "</div>")?;

        writeln!(&mut html, "</div>")?;
        writeln!(&mut html, "</body></html>")?;

        Ok(html)
    }

    /// Create a comparison visualization between multiple images
    pub fn create_comparison_view<T>(
        images: &[(&str, ArrayView2<T>)],
        title: &str,
    ) -> NdimageResult<String>
    where
        T: Float + FromPrimitive + ToPrimitive + Debug + Clone,
    {
        if images.is_empty() {
            return Err(NdimageError::InvalidInput(
                "No images provided for comparison".into(),
            ));
        }

        let mut html = String::new();

        writeln!(&mut html, "<!DOCTYPE html>")?;
        writeln!(&mut html, "<html><head>")?;
        writeln!(&mut html, "<title>{}</title>", title)?;
        writeln!(&mut html, "<style>")?;
        writeln!(
            &mut html,
            r#"
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .comparison-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .image-panel {{ border: 1px solid #ccc; padding: 15px; border-radius: 5px; }}
            .image-title {{ font-weight: bold; margin-bottom: 10px; color: #333; text-align: center; }}
            .image-stats {{ background: #f9f9f9; padding: 10px; margin-top: 10px; border-radius: 3px; }}
            .stat-row {{ display: flex; justify-content: space-between; margin-bottom: 5px; }}
            .difference-highlight {{ background: #ffe6e6; }}
        "#
        )?;
        writeln!(&mut html, "</style>")?;
        writeln!(&mut html, "</head><body>")?;

        writeln!(&mut html, "<h1>{}</h1>", title)?;
        writeln!(&mut html, "<div class='comparison-grid'>")?;

        for (name, image) in images {
            writeln!(&mut html, "<div class='image-panel'>")?;
            writeln!(&mut html, "<div class='image-title'>{}</div>", name)?;

            // Generate heatmap for this image
            let config = PlotConfig {
                format: ReportFormat::Html,
                colormap: ColorMap::Viridis,
                title: name.to_string(),
                width: 250,
                height: 200,
                ..PlotConfig::default()
            };

            let heatmap = plot_heatmap(image, &config)?;
            writeln!(&mut html, "{}", heatmap)?;

            // Add statistics
            let (height, width) = image.dim();
            let mean = image.mean().unwrap_or(T::zero());
            let min_val = image.iter().cloned().fold(T::infinity(), T::min);
            let max_val = image.iter().cloned().fold(T::neg_infinity(), T::max);

            writeln!(&mut html, "<div class='image-stats'>")?;
            writeln!(
                &mut html,
                "<div class='stat-row'><span>Dimensions:</span><span>{}×{}</span></div>",
                height, width
            )?;
            writeln!(
                &mut html,
                "<div class='stat-row'><span>Mean:</span><span>{:.4}</span></div>",
                mean.to_f64().unwrap_or(0.0)
            )?;
            writeln!(
                &mut html,
                "<div class='stat-row'><span>Range:</span><span>[{:.3}, {:.3}]</span></div>",
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0)
            )?;
            writeln!(&mut html, "</div>")?;

            writeln!(&mut html, "</div>")?;
        }

        writeln!(&mut html, "</div>")?;
        writeln!(&mut html, "</body></html>")?;

        Ok(html)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_module_exports() {
        // Test that all re-exports are available
        let _config = PlotConfig::new();
        let _colormap = ColorMap::Viridis;
        let _format = ReportFormat::Html;
        let _color = RgbColor::new(255, 0, 0);
    }

    #[test]
    fn test_backward_compatibility_aliases() {
        let img1 = Array2::<f64>::zeros((5, 5));
        let img2 = Array2::<f64>::ones((5, 5));
        let images = vec![img1.view(), img2.view()];
        let config = PlotConfig::new().with_format(ReportFormat::Text);

        // Test legacy alias
        let result = createimage_montage(&images, 2, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_statistical_functions() {
        let data1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let data2 = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let datasets = vec![("Test A", data1.view()), ("Test B", data2.view())];

        let config = PlotConfig::new()
            .with_format(ReportFormat::Text)
            .with_title("Statistical Test");

        let result = plot_statistical_comparison(&datasets, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_plotting_functions() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = PlotConfig::new()
            .with_format(ReportFormat::Text)
            .with_title("Histogram Test");

        let result = plot_histogram(&data.view(), &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_colormap_functions() {
        let colors = create_colormap(ColorMap::Viridis, 10);
        assert_eq!(colors.len(), 10);

        let color = viridis_colormap(0.5);
        assert!(color.r > 0 || color.g > 0 || color.b > 0);
    }

    #[test]
    fn test_report_generation() {
        let image = Array2::<f64>::ones((10, 10));
        let config = ReportConfig::new()
            .with_format(ReportFormat::Text)
            .minimal();

        let result = generate_report(&image.view(), None, None, &config);
        assert!(result.is_ok());
    }
}
