//! Native Plotting for Advanced Clustering - Advanced Visualization Engine
//!
//! This module provides comprehensive native plotting capabilities for Advanced clustering,
//! including interactive dendrogram visualization, 3D cluster plots, real-time animation,
//! and advanced quantum state visualizations without external dependencies.

mod plotter;
mod svg;
mod types;

// Re-export everything publicly
pub use plotter::AdvancedNativePlotter;
pub use types::{
    AnimationEngine, AnimationFrame, Camera3D, DendrogramNode, DendrogramTree, DirectionalLight,
    ExecutionSummary, ExportQuality, InteractiveController, InteractiveFeature,
    InteractivePerformanceDashboard, Lighting3D, MetricTimelinePoint, Native3DClusterPlot,
    NativeClusterPlot, NativeDendrogramPlot, NativePlotConfig, NativeVisualizationOutput,
    NeuromorphicActivityPlot, PlotColorScheme, PointLight, QuantumCoherenceAnimation,
    QuantumCoherenceFrame, QuantumField3D, SvgCanvas, SvgElement, Transformation,
};

use crate::advanced_clustering::AdvancedClusteringResult;
use crate::error::{ClusteringError, Result};
use scirs2_core::ndarray::ArrayView2;

/// Convenience function to create native Advanced visualization
#[allow(dead_code)]
pub fn create_native_advanced_plot(
    data: &ArrayView2<f64>,
    result: &AdvancedClusteringResult,
    config: Option<NativePlotConfig>,
) -> Result<NativeVisualizationOutput> {
    let config = config.unwrap_or_default();
    let mut plotter = AdvancedNativePlotter::new(config);
    plotter.create_comprehensive_plot(data, result)
}

/// Export native visualization to file
#[allow(dead_code)]
pub fn export_native_visualization(
    output: &NativeVisualizationOutput,
    filename: &str,
    format: &str,
) -> Result<()> {
    match format.to_lowercase().as_str() {
        "svg" => {
            use std::fs::File;
            use std::io::Write;

            let mut file = File::create(format!("{}.svg", filename)).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to create SVG file: {}", e))
            })?;

            file.write_all(output.svg_content.as_bytes()).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to write SVG file: {}", e))
            })?;

            println!("Exported native Advanced visualization to {filename}.svg");
        }
        "html" => {
            use std::fs::File;
            use std::io::Write;

            let html_content = format!(
                r#"<!DOCTYPE html>
<html>
<head>
    <title>Advanced Native Visualization</title>
    <style>
        body {{ margin: 0; padding: 20px; background: #1a1a2e; }}
        .selected {{ stroke: #FFD700 !important; stroke-width: 3px !important; }}
    </style>
</head>
<body>
    {}
    <script>{}</script>
</body>
</html>"#,
                output.svg_content, output.interactive_script
            );

            let mut file = File::create(format!("{}.html", filename)).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to create HTML file: {}", e))
            })?;

            file.write_all(html_content.as_bytes()).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to write HTML file: {}", e))
            })?;

            println!("Exported interactive Advanced visualization to {filename}.html");
        }
        _ => {
            return Err(ClusteringError::InvalidInput(format!(
                "Unsupported export format: {}",
                format
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advanced_clustering::{AdvancedClusteringResult, AdvancedPerformanceMetrics};
    use scirs2_core::ndarray::{array, Array1, Array2};

    /// Build a minimal AdvancedClusteringResult from small 4-point 2-D data.
    /// Two clusters: points 0,1 in cluster 0; points 2,3 in cluster 1.
    fn make_result() -> AdvancedClusteringResult {
        let centroids = array![[0.5, 0.5], [5.5, 5.5]];
        let clusters = Array1::from_vec(vec![0, 0, 1, 1]);
        AdvancedClusteringResult {
            clusters,
            centroids,
            ai_speedup: 1.0,
            quantum_advantage: 1.0,
            neuromorphic_benefit: 1.0,
            meta_learning_improvement: 0.0,
            selected_algorithm: "test".to_string(),
            confidence: 0.9,
            performance: AdvancedPerformanceMetrics {
                silhouette_score: 0.8,
                execution_time: 0.001,
                memory_usage: 1.0,
                quantum_coherence: 0.7,
                neural_adaptation_rate: 0.5,
                ai_iterations: 5,
                energy_efficiency: 0.9,
            },
        }
    }

    /// Small 4-point 2-D data matching make_result().
    fn make_data() -> Array2<f64> {
        array![[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]]
    }

    // -----------------------------------------------------------------------
    // build_dendrogram_tree
    // -----------------------------------------------------------------------
    #[test]
    fn test_build_dendrogram_tree_basic() {
        let plotter = AdvancedNativePlotter::new(NativePlotConfig::default());
        let data = make_data();
        let result = make_result();
        let tree = plotter
            .build_dendrogram_tree(&data.view(), &result)
            .expect("build_dendrogram_tree failed");

        assert_eq!(tree.leaf_count, 4, "leaf_count should equal n_samples");
        assert!(
            tree.root.height >= 0.0,
            "root height must be non-negative, got {}",
            tree.root.height
        );
    }

    // -----------------------------------------------------------------------
    // calculate_dendrogram_layout
    // -----------------------------------------------------------------------
    #[test]
    fn test_calculate_dendrogram_layout_all_nodes_have_positions() {
        let plotter = AdvancedNativePlotter::new(NativePlotConfig::default());
        let data = make_data();
        let result = make_result();
        let tree = plotter
            .build_dendrogram_tree(&data.view(), &result)
            .expect("build_dendrogram_tree failed");
        let layout = plotter
            .calculate_dendrogram_layout(&tree)
            .expect("calculate_dendrogram_layout failed");

        // We should have at least one entry (the root).
        assert!(
            !layout.is_empty(),
            "layout must contain at least one position"
        );
        for (id, (x, y)) in &layout {
            assert!(
                x.is_finite() && y.is_finite(),
                "position for node '{id}' is not finite: ({x}, {y})"
            );
        }
    }

    // -----------------------------------------------------------------------
    // calculate_quantum_branch_lengths
    // -----------------------------------------------------------------------
    #[test]
    fn test_calculate_quantum_branch_lengths_all_nodes() {
        let plotter = AdvancedNativePlotter::new(NativePlotConfig::default());
        let data = make_data();
        let result = make_result();
        let tree = plotter
            .build_dendrogram_tree(&data.view(), &result)
            .expect("build_dendrogram_tree failed");
        let lengths = plotter
            .calculate_quantum_branch_lengths(&tree, &result)
            .expect("calculate_quantum_branch_lengths failed");

        assert!(!lengths.is_empty(), "branch lengths map must not be empty");
        for (id, length) in &lengths {
            assert!(
                length.is_finite(),
                "branch length for '{id}' is not finite: {length}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // calculate_dendrogram_quantum_enhancements
    // -----------------------------------------------------------------------
    #[test]
    fn test_calculate_dendrogram_quantum_enhancements_in_range() {
        let plotter = AdvancedNativePlotter::new(NativePlotConfig::default());
        let data = make_data();
        let result = make_result();
        let tree = plotter
            .build_dendrogram_tree(&data.view(), &result)
            .expect("build_dendrogram_tree failed");
        let enhancements = plotter
            .calculate_dendrogram_quantum_enhancements(&tree, &result)
            .expect("calculate_dendrogram_quantum_enhancements failed");

        assert!(
            !enhancements.is_empty(),
            "enhancements map must not be empty"
        );
        for (id, enh) in &enhancements {
            assert!(
                *enh >= 0.0 && *enh <= 1.0,
                "enhancement for '{id}' is out of [0, 1]: {enh}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // render_dendrogram_to_svg
    // -----------------------------------------------------------------------
    #[test]
    fn test_render_dendrogram_to_svg_adds_elements() {
        let mut plotter = AdvancedNativePlotter::new(NativePlotConfig::default());
        let data = make_data();
        let result = make_result();
        let tree = plotter
            .build_dendrogram_tree(&data.view(), &result)
            .expect("build_dendrogram_tree failed");
        let positions = plotter
            .calculate_dendrogram_layout(&tree)
            .expect("calculate_dendrogram_layout failed");
        let branch_lengths = plotter
            .calculate_quantum_branch_lengths(&tree, &result)
            .expect("calculate_quantum_branch_lengths failed");
        let enhancements = plotter
            .calculate_dendrogram_quantum_enhancements(&tree, &result)
            .expect("calculate_dendrogram_quantum_enhancements failed");

        plotter
            .render_dendrogram_to_svg(&tree, &positions, &branch_lengths, &enhancements)
            .expect("render_dendrogram_to_svg failed");

        // The SVG canvas should now contain content.
        let svg = plotter.svg_canvas.to_svg();
        assert!(
            !svg.is_empty(),
            "SVG canvas must be non-empty after rendering"
        );
    }

    // -----------------------------------------------------------------------
    // create_quantum_field_3d
    // -----------------------------------------------------------------------
    #[test]
    fn test_create_quantum_field_3d_grid_dimensions() {
        let plotter = AdvancedNativePlotter::new(NativePlotConfig::default());
        let data = make_data();
        let result = make_result();
        // Need 3-D data for create_quantum_field_3d.
        let data_3d: Array2<f64> = array![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0]
        ];
        let field = plotter
            .create_quantum_field_3d(&data_3d, &result)
            .expect("create_quantum_field_3d failed");

        // Method uses a 10×10 grid.
        let shape = field.field_strength.shape();
        assert_eq!(shape[0], 10, "field grid rows should be 10");
        assert_eq!(shape[1], 10, "field grid cols should be 10");
        // All values should be finite and >= 0.
        for v in field.field_strength.iter() {
            assert!(
                v.is_finite() && *v >= 0.0,
                "field value should be finite and >= 0, got {v}"
            );
        }
        // Unused variable suppression — we built result but only use data_3d.
        let _ = data;
    }

    // -----------------------------------------------------------------------
    // create_quantum_coherence_frame
    // -----------------------------------------------------------------------
    #[test]
    fn test_create_quantum_coherence_frame_shape_and_timestamp() {
        let plotter = AdvancedNativePlotter::new(NativePlotConfig::default());
        let result = make_result();
        let frame = plotter
            .create_quantum_coherence_frame(&result, 0.0)
            .expect("create_quantum_coherence_frame failed");

        assert_eq!(
            frame.timestamp, 0.0,
            "timestamp should match argument t=0.0"
        );
        let shape = frame.field_strength.shape();
        assert_eq!(shape[0], 8, "field_strength rows should be 8");
        assert_eq!(shape[1], 8, "field_strength cols should be 8");
        // The result has 2 centroids, so there should be 2 SVG elements.
        assert_eq!(
            frame.elements.len(),
            2,
            "should have one element per centroid"
        );
    }

    // -----------------------------------------------------------------------
    // create_comprehensive_plot (exercises multiple paths via create_native_advanced_plot)
    // -----------------------------------------------------------------------
    #[test]
    fn test_create_comprehensive_plot_runs_ok() {
        let data = make_data();
        let result = make_result();
        let output = create_native_advanced_plot(&data.view(), &result, None)
            .expect("create_native_advanced_plot failed");

        assert!(
            !output.svg_content.is_empty(),
            "svg_content must be non-empty"
        );
        // For 2-D data, plot_3d should be None (ncols <= 2).
        assert!(
            output.plot_3d.is_none(),
            "plot_3d should be None for 2-D data"
        );
    }

    // -----------------------------------------------------------------------
    // create_comprehensive_plot with hierarchical algorithm triggers dendrogram path
    // -----------------------------------------------------------------------
    #[test]
    fn test_comprehensive_plot_hierarchical_triggers_dendrogram() {
        let data = make_data();
        let mut result = make_result();
        result.selected_algorithm = "hierarchical".to_string();

        let output = create_native_advanced_plot(&data.view(), &result, None)
            .expect("create_native_advanced_plot (hierarchical) failed");

        assert!(
            output.dendrogram.is_some(),
            "dendrogram should be Some when selected_algorithm contains 'hierarchical'"
        );
    }
}
