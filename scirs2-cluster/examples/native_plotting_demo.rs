//! Demonstration of native plotting capabilities for clustering results
//!
//! This example shows how to use the new native plotting features in scirs2-cluster
//! to create static plots with plotters and interactive visualizations with egui.

use ndarray::Array2;
use scirs2_cluster::preprocess::standardize;
use scirs2_cluster::vq::{kmeans, kmeans2, vq};
use scirs2_cluster::{BoundaryType, ColorScheme, DimensionalityReduction};

#[cfg(feature = "plotters")]
use scirs2_cluster::{save_clustering_plot, PlotFormat, PlotOutput};

#[cfg(feature = "egui")]
use scirs2_cluster::launch_interactive_visualization;

use scirs2_cluster::VisualizationConfig;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Native Plotting Demo for scirs2-cluster");
    println!("======================================");

    // Create sample clustering data with 3 well-separated clusters
    let data = Array2::from_shape_vec((300, 2), generate_sample_data())?;

    println!("Generated {} data points with 2 features", data.nrows());

    // Standardize the data
    let standardized = standardize(data.view(), true)?;

    // Perform K-means clustering
    let k = 3;
    let (centroids, _distortion) = kmeans(
        standardized.view(),
        k,
        Some(100),  // max_iter
        Some(1e-4), // tolerance
        Some(true), // check_finite
        Some(42),   // random_seed
    )?;
    // Generate labels using vq
    let (labels_usize, _) = vq(standardized.view(), centroids.view())?;
    // Convert labels from usize to i32 for plotting functions
    let labels = labels_usize.mapv(|x| x as i32);

    println!("K-means clustering completed with {} clusters", k);

    // Static plotting with plotters (if feature is enabled)
    #[cfg(feature = "plotters")]
    {
        println!("\nCreating static plots with plotters...");

        // Create visualization configuration
        let vis_config = VisualizationConfig {
            color_scheme: ColorScheme::ColorblindFriendly,
            point_size: 4.0,
            point_opacity: 0.8,
            show_centroids: true,
            show_boundaries: false,
            boundary_type: BoundaryType::ConvexHull,
            interactive: false,
            animation: None,
            dimensionality_reduction: DimensionalityReduction::None,
        };

        // PNG output configuration
        let png_output = PlotOutput {
            format: PlotFormat::PNG,
            dimensions: (1000, 800),
            dpi: 300,
            background_color: "#FFFFFF".to_string(),
            show_grid: true,
            show_axes: true,
            title: Some("K-means Clustering Results".to_string()),
            axis_labels: (
                Some("Standardized Feature 1".to_string()),
                Some("Standardized Feature 2".to_string()),
                None,
            ),
        };

        // Save as PNG
        save_clustering_plot(
            standardized.view(),
            &labels,
            Some(&centroids),
            "clustering_result.png",
            Some(&vis_config),
            Some(&png_output),
        )?;
        println!("✓ Saved PNG plot: clustering_result.png");

        // SVG output configuration
        let svg_output = PlotOutput {
            format: PlotFormat::SVG,
            dimensions: (1000, 800),
            title: Some("K-means Clustering Results (Vector Graphics)".to_string()),
            ..png_output
        };

        // Save as SVG
        save_clustering_plot(
            standardized.view(),
            &labels,
            Some(&centroids),
            "clustering_result.svg",
            Some(&vis_config),
            Some(&svg_output),
        )?;
        println!("✓ Saved SVG plot: clustering_result.svg");
    }

    #[cfg(not(feature = "plotters"))]
    {
        println!("\nStatic plotting with plotters is not available.");
        println!("Enable with: cargo run --example native_plotting_demo --features plotters");
    }

    // Interactive visualization with egui (if feature is enabled)
    #[cfg(feature = "egui")]
    {
        println!("\nLaunching interactive visualization with egui...");
        println!("Use mouse to pan and zoom, click clusters in the legend to highlight them.");

        let vis_config = VisualizationConfig {
            color_scheme: ColorScheme::Viridis,
            point_size: 6.0,
            point_opacity: 0.9,
            show_centroids: true,
            show_boundaries: false,
            boundary_type: BoundaryType::Ellipse,
            interactive: true,
            animation: None,
            dimensionality_reduction: DimensionalityReduction::None,
        };

        // Launch interactive visualization
        launch_interactive_visualization(
            standardized.view(),
            &labels,
            Some(&centroids),
            Some(&vis_config),
        )?;
    }

    #[cfg(not(feature = "egui"))]
    {
        println!("\nInteractive visualization with egui is not available.");
        println!("Enable with: cargo run --example native_plotting_demo --features egui");
    }

    println!("\nDemo completed!");

    Ok(())
}

/// Generate sample data with 3 well-separated clusters
#[allow(dead_code)]
fn generate_sample_data() -> Vec<f64> {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Vec::with_capacity(600); // 300 points * 2 features

    // Cluster 1: centered at (0, 0)
    let normal1 = Normal::new(0.0, 0.5).unwrap();
    for _ in 0..100 {
        data.push(rng.sample(normal1)); // x
        data.push(rng.sample(normal1)); // y
    }

    // Cluster 2: centered at (4, 0)
    let normal2_x = Normal::new(4.0, 0.5).unwrap();
    let normal2_y = Normal::new(0.0, 0.5).unwrap();
    for _ in 0..100 {
        data.push(rng.sample(normal2_x)); // x
        data.push(rng.sample(normal2_y)); // y
    }

    // Cluster 3: centered at (2, 3)
    let normal3_x = Normal::new(2.0, 0.5).unwrap();
    let normal3_y = Normal::new(3.0, 0.5).unwrap();
    for _ in 0..100 {
        data.push(rng.sample(normal3_x)); // x
        data.push(rng.sample(normal3_y)); // y
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_data_generation() {
        let data = generate_sample_data();
        assert_eq!(data.len(), 600); // 300 points * 2 features
    }

    #[test]
    fn test_clustering_pipeline() {
        let data = Array2::from_shape_vec((300, 2), generate_sample_data()).unwrap();
        let standardized = standardize(data.view(), true).unwrap();
        let (centroids, labels) = kmeans2(
            standardized.view(),
            3,
            Some(10), // fewer iterations for test
            Some(1e-4),
            None,     // minit method
            None,     // missing method
            None,     // check_finite
            Some(42), // random seed
        )
        .unwrap();

        assert_eq!(centroids.nrows(), 3);
        assert_eq!(centroids.ncols(), 2);
        assert_eq!(labels.len(), 300);

        // Check that all labels are valid
        for &label in labels.iter() {
            assert!(label < 3);
        }
    }
}
