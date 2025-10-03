//! Demonstration of dendrogram plotting capabilities
//!
//! This example shows how to use the dendrogram plotting features in scirs2-cluster
//! to create publication-ready dendrograms from hierarchical clustering results.

use scirs2_cluster::hierarchy::{linkage, LinkageMethod, Metric};
use scirs2_core::ndarray::Array2;

#[cfg(feature = "plotters")]
use scirs2_cluster::{save_dendrogram_plot, PlotFormat, PlotOutput};

use scirs2_cluster::hierarchy::visualization::{
    ColorScheme, ColorThreshold, DendrogramConfig, DendrogramOrientation,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Dendrogram Plotting Demo for scirs2-cluster");
    println!("==========================================");

    // Create sample hierarchical clustering data
    let data = create_sample_data();
    println!(
        "Generated {} data points with {} features",
        data.nrows(),
        data.ncols()
    );

    // Perform hierarchical clustering with different linkage methods
    let linkage_methods = vec![
        (LinkageMethod::Ward, "ward"),
        (LinkageMethod::Complete, "complete"),
        (LinkageMethod::Average, "average"),
        (LinkageMethod::Single, "single"),
    ];

    for (method, method_name) in linkage_methods {
        println!(
            "\nPerforming hierarchical clustering with {} linkage...",
            method_name
        );

        let linkage_matrix = linkage(data.view(), method, Metric::Euclidean)?;
        println!("Linkage matrix shape: {:?}", linkage_matrix.shape());

        // Create custom sample labels
        let labels: Vec<String> = (0..data.nrows()).map(|i| format!("Sample_{}", i)).collect();

        #[cfg(feature = "plotters")]
        {
            // Create various dendrogram configurations
            create_dendrograms(&linkage_matrix.view(), &labels, method_name)?;
        }

        #[cfg(not(feature = "plotters"))]
        {
            println!("Static plotting with plotters is not available.");
            println!(
                "Enable with: cargo run --example dendrogram_plotting_demo --features plotters"
            );
        }
    }

    println!("\nDemo completed!");

    Ok(())
}

#[cfg(feature = "plotters")]
#[allow(dead_code)]
fn create_dendrograms(
    linkage_matrix: &scirs2_core::ndarray::ArrayView2<f64>,
    labels: &[String],
    method_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Basic dendrogram with default settings
    {
        let config = DendrogramConfig::default();
        let output = PlotOutput {
            format: PlotFormat::PNG,
            dimensions: (1200, 800),
            title: Some(format!(
                "Dendrogram - {} Linkage",
                method_name.to_uppercase()
            )),
            axis_labels: (
                Some("Sample Index".to_string()),
                Some("Distance".to_string()),
                None,
            ),
            ..Default::default()
        };

        save_dendrogram_plot(
            *linkage_matrix,
            Some(labels),
            format!("dendrogram_{}_basic.png", method_name),
            Some(&config),
            Some(&output),
        )?;
        println!(
            "✓ Saved basic dendrogram: dendrogram_{}_basic.png",
            method_name
        );
    }

    // 2. High contrast dendrogram with custom colors
    {
        let config = DendrogramConfig {
            color_scheme: ColorScheme::HighContrast,
            color_threshold: ColorThreshold {
                threshold: 0.0,
                above_color: "#d62728".to_string(), // Red
                below_color: "#2ca02c".to_string(), // Green
                auto_threshold: true,
                target_clusters: Some(3),
            },
            show_labels: true,
            show_distances: true,
            orientation: DendrogramOrientation::Top,
            line_width: 3.0,
            font_size: 14.0,
            truncate_mode: None,
            styling: scirs2_cluster::hierarchy::visualization::DendrogramStyling::default(),
        };

        let output = PlotOutput {
            format: PlotFormat::SVG,
            dimensions: (1200, 800),
            title: Some(format!(
                "High Contrast Dendrogram - {} Linkage",
                method_name.to_uppercase()
            )),
            background_color: "#f8f8f8".to_string(),
            show_grid: true,
            axis_labels: (
                Some("Samples".to_string()),
                Some("Linkage Distance".to_string()),
                None,
            ),
            ..Default::default()
        };

        save_dendrogram_plot(
            *linkage_matrix,
            Some(labels),
            format!("dendrogram_{}_high_contrast.svg", method_name),
            Some(&config),
            Some(&output),
        )?;
        println!(
            "✓ Saved high contrast dendrogram: dendrogram_{}_high_contrast.svg",
            method_name
        );
    }

    // 3. Viridis color scheme with different orientation
    {
        let config = DendrogramConfig {
            color_scheme: ColorScheme::Viridis,
            orientation: DendrogramOrientation::Left,
            show_labels: true,
            show_distances: false,
            line_width: 2.5,
            font_size: 12.0,
            ..Default::default()
        };

        let output = PlotOutput {
            format: PlotFormat::PNG,
            dimensions: (800, 1200), // Switched for left orientation
            title: Some(format!(
                "Viridis Dendrogram (Left) - {} Linkage",
                method_name.to_uppercase()
            )),
            axis_labels: (
                Some("Distance".to_string()),
                Some("Sample Index".to_string()),
                None,
            ),
            ..Default::default()
        };

        save_dendrogram_plot(
            *linkage_matrix,
            Some(labels),
            format!("dendrogram_{}_viridis_left.png", method_name),
            Some(&config),
            Some(&output),
        )?;
        println!(
            "✓ Saved Viridis dendrogram: dendrogram_{}_viridis_left.png",
            method_name
        );
    }

    // 4. Grayscale for publication
    {
        let config = DendrogramConfig {
            color_scheme: ColorScheme::Grayscale,
            color_threshold: ColorThreshold {
                auto_threshold: true,
                target_clusters: Some(4),
                ..Default::default()
            },
            show_labels: true,
            show_distances: true,
            line_width: 2.0,
            font_size: 10.0,
            ..Default::default()
        };

        let output = PlotOutput {
            format: PlotFormat::SVG,
            dimensions: (1000, 600),
            title: Some(format!(
                "Publication Ready - {} Linkage",
                method_name.to_uppercase()
            )),
            background_color: "#ffffff".to_string(),
            show_grid: false,
            dpi: 600, // High DPI for publication
            axis_labels: (
                Some("Sample".to_string()),
                Some("Distance".to_string()),
                None,
            ),
            ..Default::default()
        };

        save_dendrogram_plot(
            *linkage_matrix,
            Some(labels),
            format!("dendrogram_{}_publication.svg", method_name),
            Some(&config),
            Some(&output),
        )?;
        println!(
            "✓ Saved publication dendrogram: dendrogram_{}_publication.svg",
            method_name
        );
    }

    Ok(())
}

/// Create sample hierarchical data with distinct clusters
#[allow(dead_code)]
fn create_sample_data() -> Array2<f64> {
    use scirs2_core::essentials::Normal;
    use scirs2_core::random::prelude::*;

    let mut rng = StdRng::seed_from_u64(12345);
    let mut data = Vec::with_capacity(60); // 12 points * 5 features

    // Cluster 1: Low values
    let normal1 = Normal::new(1.0, 0.3).unwrap();
    for _ in 0..4 {
        for _ in 0..5 {
            data.push(rng.sample(normal1));
        }
    }

    // Cluster 2: Medium values
    let normal2 = Normal::new(3.0, 0.4).unwrap();
    for _ in 0..4 {
        for _ in 0..5 {
            data.push(rng.sample(normal2));
        }
    }

    // Cluster 3: High values
    let normal3 = Normal::new(6.0, 0.5).unwrap();
    for _ in 0..4 {
        for _ in 0..5 {
            data.push(rng.sample(normal3));
        }
    }

    Array2::from_shape_vec((12, 5), data).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_data_creation() {
        let data = create_sample_data();
        assert_eq!(data.shape(), &[12, 5]);
    }

    #[test]
    fn test_hierarchical_clustering() {
        let data = create_sample_data();
        let linkage_matrix = linkage(data.view(), LinkageMethod::Ward, Metric::Euclidean).unwrap();

        // Should have n-1 rows for n samples
        assert_eq!(linkage_matrix.nrows(), data.nrows() - 1);
        assert_eq!(linkage_matrix.ncols(), 4); // [left, right, distance, count]

        // Distances should be non-negative and increasing
        let mut prev_distance = 0.0;
        for row in linkage_matrix.rows() {
            let distance = row[2];
            assert!(distance >= 0.0);
            assert!(distance >= prev_distance);
            prev_distance = distance;
        }
    }

    #[test]
    fn test_different_linkage_methods() {
        let data = create_sample_data();
        let methods = vec![
            LinkageMethod::Ward,
            LinkageMethod::Complete,
            LinkageMethod::Average,
            LinkageMethod::Single,
        ];

        for method in methods {
            let linkage_matrix = linkage(data.view(), method, Metric::Euclidean).unwrap();
            assert_eq!(linkage_matrix.nrows(), data.nrows() - 1);
            assert_eq!(linkage_matrix.ncols(), 4);
        }
    }
}
