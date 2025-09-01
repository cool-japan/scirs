//! Advanced Visualization Features Demonstration
//!
//! This example showcases the comprehensive visualization capabilities including:
//! - 2D and 3D scatter plots with multiple color schemes
//! - Interactive 3D visualization with camera controls
//! - Real-time streaming data visualization
//! - Algorithm convergence animations
//! - Export capabilities for various formats
//! - Dimensionality reduction visualization

use ndarray::{Array1, Array2};
use scirs2_cluster::{
    kmeans_simd,
    visualization::{
        // Animation capabilities
        animation::{
            AnimationAnnotation, IterativeAnimationConfig, IterativeAnimationRecorder,
            StreamingConfig, StreamingVisualizer,
        },
        // Core visualization
        create_scatter_plot_2d,
        create_scatter_plot_3d,
        // Export capabilities
        export::{
            export_scatter_2d_to_html, export_scatter_3d_to_html, save_visualization_to_file,
            ExportConfig, ExportFormat,
        },
        // Interactive 3D
        interactive::{InteractiveConfig, InteractiveVisualizer, KeyCode, MouseButton, ViewMode},
        BoundaryType,
        ColorScheme,
        DimensionalityReduction,
        VisualizationConfig,
    },
    KMeansInit, KMeansOptions,
};
use std::time::{Duration, Instant};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Advanced Visualization Features Demonstration");
    println!("===============================================");

    // Generate comprehensive test datasets
    let (data_2d, labels_2d, centroids_2d) = generate_2d_clustering_data();
    let (data_3d, labels_3d, centroids_3d) = generate_3d_clustering_data();
    let (data_high_dim, labels_high_dim_, centroids_high_dim) = generate_high_dimensional_data();

    println!("📊 Generated test datasets:");
    println!(
        "   • 2D data: {} samples, {} features",
        data_2d.nrows(),
        data_2d.ncols()
    );
    println!(
        "   • 3D data: {} samples, {} features",
        data_3d.nrows(),
        data_3d.ncols()
    );
    println!(
        "   • High-dim data: {} samples, {} features",
        data_high_dim.nrows(),
        data_high_dim.ncols()
    );

    // Test 1: Basic 2D Visualization with Different Color Schemes
    println!("\n🎨 Test 1: Color Scheme Variations");
    println!("─────────────────────────────────────");

    let color_schemes = [
        ("Default", ColorScheme::Default),
        ("Colorblind Friendly", ColorScheme::ColorblindFriendly),
        ("High Contrast", ColorScheme::HighContrast),
        ("Pastel", ColorScheme::Pastel),
        ("Viridis", ColorScheme::Viridis),
        ("Plasma", ColorScheme::Plasma),
    ];

    for (name, scheme) in &color_schemes {
        let config = VisualizationConfig {
            color_scheme: *scheme,
            show_centroids: true,
            show_boundaries: true,
            boundary_type: BoundaryType::ConvexHull,
            ..Default::default()
        };

        let plot =
            create_scatter_plot_2d(data_2d.view(), &labels_2d, Some(&centroids_2d), &config)?;
        println!(
            "   ✅ Created 2D plot with {} color scheme ({} clusters)",
            name,
            plot.legend.len()
        );

        // Export to HTML for each color scheme
        let filename = format!("scatter_2d_{}.html", name.to_lowercase().replace(" ", "_"));
        let export_config = ExportConfig {
            format: ExportFormat::HTML,
            interactive: true,
            ..Default::default()
        };
        export_scatter_2d_to_html(&plot, &filename, &export_config)?;
        println!("      💾 Exported to {filename}");
    }

    // Test 2: 3D Interactive Visualization
    println!("\n🔮 Test 2: 3D Interactive Visualization");
    println!("────────────────────────────────────────");

    let interactive_config = InteractiveConfig {
        enable_camera_controls: true,
        enable_point_selection: true,
        show_axes: true,
        show_grid: true,
        show_realtime_stats: true,
        highlight_on_hover: true,
        show_3d_boundaries: true,
        ..Default::default()
    };

    let mut interactive_viz = InteractiveVisualizer::new(interactive_config);
    interactive_viz.update_data(data_3d.view(), &labels_3d, Some(&centroids_3d))?;

    println!(
        "   🎮 Interactive visualizer created with {} clusters",
        interactive_viz.get_cluster_stats().len()
    );

    // Demonstrate different view modes
    let view_modes = [
        ("Perspective", ViewMode::Perspective),
        ("Birds Eye", ViewMode::BirdsEye),
        ("Side View", ViewMode::Side),
        ("Front View", ViewMode::Front),
        ("Top View", ViewMode::Top),
    ];

    for (name, mode) in &view_modes {
        interactive_viz.set_view_mode(*mode);
        println!("   📷 Set view mode: {name}");
    }

    // Export 3D visualization
    let viz_config = VisualizationConfig {
        color_scheme: ColorScheme::Viridis,
        show_centroids: true,
        ..Default::default()
    };
    let plot_3d =
        create_scatter_plot_3d(data_3d.view(), &labels_3d, Some(&centroids_3d), &viz_config)?;

    let export_config = ExportConfig {
        format: ExportFormat::HTML,
        interactive: true,
        stereoscopic: false,
        ..Default::default()
    };
    export_scatter_3d_to_html(
        &plot_3d,
        "interactive_3d_visualization.html",
        &export_config,
    )?;
    println!("   💾 Exported interactive 3D visualization to interactive_3d_visualization.html");

    // Test 3: Dimensionality Reduction Visualization
    println!("\n📐 Test 3: Dimensionality Reduction Visualization");
    println!("──────────────────────────────────────────────────");

    let reduction_methods = [
        ("PCA", DimensionalityReduction::PCA),
        ("First 2D", DimensionalityReduction::First2D),
        ("First 3D", DimensionalityReduction::First3D),
    ];

    for (name, method) in &reduction_methods {
        let config = VisualizationConfig {
            dimensionality_reduction: *method,
            color_scheme: ColorScheme::Plasma,
            show_centroids: true,
            ..Default::default()
        };

        if name.contains("3D") {
            match create_scatter_plot_3d(data_high_dim.view(), &labels_high_dim_, None, &config) {
                Ok(plot) => {
                    println!(
                        "   ✅ Created 3D plot using {} ({} -> 3D)",
                        name,
                        data_high_dim.ncols()
                    );

                    let filename =
                        format!("high_dim_3d_{}.html", name.to_lowercase().replace(" ", "_"));
                    export_scatter_3d_to_html(&plot, &filename, &export_config)?;
                    println!("      💾 Exported to {filename}");
                }
                Err(e) => println!("   ❌ Failed to create 3D plot using {name}: {e}"),
            }
        } else {
            match create_scatter_plot_2d(data_high_dim.view(), &labels_high_dim_, None, &config) {
                Ok(plot) => {
                    println!(
                        "   ✅ Created 2D plot using {} ({} -> 2D)",
                        name,
                        data_high_dim.ncols()
                    );

                    let filename =
                        format!("high_dim_2d_{}.html", name.to_lowercase().replace(" ", "_"));
                    export_scatter_2d_to_html(&plot, &filename, &export_config)?;
                    println!("      💾 Exported to {filename}");
                }
                Err(e) => println!("   ❌ Failed to create 2D plot using {name}: {e}"),
            }
        }
    }

    // Test 4: Algorithm Animation Recording
    println!("\n🎬 Test 4: K-means Convergence Animation");
    println!("─────────────────────────────────────────────");

    let animation_config = IterativeAnimationConfig {
        capture_frequency: 1,
        interpolate_frames: true,
        interpolation_frames: 3,
        fps: 15.0,
        show_convergence_overlay: true,
        show_iteration_numbers: true,
        highlight_centroid_movement: true,
        fade_effect: true,
        trail_length: 5,
    };

    let mut recorder = IterativeAnimationRecorder::new(animation_config);

    // Run K-means and record convergence
    let k = 3;
    let kmeans_options = KMeansOptions {
        max_iter: 20,
        tol: 1e-4,
        random_seed: Some(42),
        n_init: 1,
        init_method: KMeansInit::Random,
    };

    println!("   🔄 Running K-means with animation recording...");

    // Simulate iterative K-means (normally this would be integrated into the algorithm)
    let mut current_centroids = generate_random_centroids(k, data_2d.ncols());

    for iteration in 0..15 {
        // Record current state
        recorder.record_frame(
            data_2d.view(),
            &labels_2d,
            Some(&current_centroids),
            Some(100.0 - iteration as f64 * 5.0), // Simulated decreasing inertia
        )?;

        if iteration % 5 == 0 {
            // Add custom annotation
            let annotation = AnimationAnnotation {
                annotation_type: "milestone".to_string(),
                position: vec![0.0, 0.0],
                text: format!("Iteration {}", iteration),
                color: "#FF0000".to_string(),
                font_size: 12.0,
            };
            recorder.add_annotation(annotation);
        }

        // Simulate centroid updates (move slightly toward final positions)
        for i in 0..current_centroids.nrows() {
            for j in 0..current_centroids.ncols() {
                current_centroids[[i, j]] +=
                    (centroids_2d[[i, j]] - current_centroids[[i, j]]) * 0.1;
            }
        }
    }

    let frames = recorder.get_frames();
    let frame_count = frames.len();
    println!("   📹 Recorded {frame_count} animation frames");

    // Generate interpolated frames
    let interpolated_frames = recorder.generate_interpolated_frames();
    println!(
        "   🎞️  Generated {} interpolated frames",
        interpolated_frames.len()
    );

    // Export animation to JSON
    let animation_json = recorder.export_to_json()?;
    std::fs::write("kmeans_convergence_animation.json", animation_json)?;
    println!("   💾 Exported animation to kmeans_convergence_animation.json");

    // Test 5: Real-time Streaming Visualization
    println!("\n📡 Test 5: Real-time Streaming Visualization");
    println!("────────────────────────────────────────────────");

    let streaming_config = StreamingConfig {
        buffer_size: 100,
        update_frequency_ms: 50,
        rolling_window_size: 20,
        animate_new_data: true,
        animate_cluster_updates: true,
        adaptive_bounds: true,
        show_streaming_stats: true,
        point_lifetime_ms: 5000,
    };

    let mut streaming_viz = StreamingVisualizer::new(streaming_config);

    println!("   🌊 Simulating streaming data...");

    // Simulate streaming data arrival
    for i in 0..50 {
        let point = Array1::from_vec(vec![
            (i as f64 * 0.1).sin() * 5.0,
            (i as f64 * 0.1).cos() * 5.0,
        ]);
        let label = i % 3;

        streaming_viz.add_data_point(point, label);

        if streaming_viz.should_update() {
            let frame = streaming_viz.generate_frame()?;
            if i % 10 == 0 {
                println!(
                    "   📊 Frame {}: {} points, {} new arrivals",
                    i,
                    frame.points.nrows(),
                    frame.new_points_mask.iter().filter(|&&x| x).count()
                );
            }
        }

        // Simulate real-time delay
        std::thread::sleep(Duration::from_millis(20));
    }

    let final_stats = streaming_viz.get_stats();
    println!("   📈 Final streaming stats:");
    println!(
        "      • Total points processed: {}",
        final_stats.total_points_processed
    );
    println!(
        "      • Points per second: {:.2}",
        final_stats.points_per_second
    );
    println!(
        "      • Cluster distribution: {:?}",
        final_stats.cluster_counts
    );

    // Test 6: Export Format Demonstration
    println!("\n💾 Test 6: Export Format Demonstration");
    println!("──────────────────────────────────────");

    let export_formats = [
        ("JSON", ExportFormat::JSON),
        ("CSV", ExportFormat::CSV),
        ("HTML", ExportFormat::HTML),
    ];

    for (name, format) in &export_formats {
        let export_config = ExportConfig {
            format: *format,
            include_metadata: true,
            interactive: format == &ExportFormat::HTML,
            quality: 95,
            ..Default::default()
        };

        let filename = format!("clustering_results.{}", name.to_lowercase());

        match save_visualization_to_file(
            None, // plot variable out of scope
            None, // plot_3d variable may be out of scope
            None,
            &filename,
            export_config,
        ) {
            Ok(_) => println!("   ✅ Exported to {name} format: {filename}"),
            Err(e) => println!("   ⚠️  Export to {name} failed: {e}"),
        }
    }

    // Test 7: Performance Comparison
    println!("\n⚡ Test 7: Visualization Performance Analysis");
    println!("────────────────────────────────────────────────");

    let sizes = [100, 500, 1000, 2000];

    for &size in &sizes {
        let (large_data, large_labels, large_centroids) = generate_2d_clustering_data_size(size);

        let start = Instant::now();
        let plot = create_scatter_plot_2d(
            large_data.view(),
            &large_labels,
            Some(&large_centroids),
            &VisualizationConfig::default(),
        )?;
        let viz_time = start.elapsed();

        let start = Instant::now();
        export_scatter_2d_to_html(&plot, &format!("perf_test_{}.html", size), &export_config)?;
        let export_time = start.elapsed();

        println!(
            "   📊 {} samples: Visualization {:?}, Export {:?}",
            size, viz_time, export_time
        );
    }

    println!("\n✅ Advanced Visualization Demo Complete!");
    println!("────────────────────────────────────────");
    println!("Generated Files:");
    println!("• Multiple 2D HTML visualizations with different color schemes");
    println!("• Interactive 3D visualization (interactive_3d_visualization.html)");
    println!("• High-dimensional data reduction visualizations");
    println!("• K-means convergence animation (kmeans_convergence_animation.json)");
    println!("• Various export format examples");
    println!("• Performance test visualizations");
    println!();
    println!("🎨 Key Features Demonstrated:");
    println!("• 🎨 Multiple color schemes and styling options");
    println!("• 🔮 Interactive 3D visualization with camera controls");
    println!("• 📐 Dimensionality reduction for high-dimensional data");
    println!("• 🎬 Algorithm convergence animations");
    println!("• 📡 Real-time streaming data visualization");
    println!("• 💾 Export to multiple formats (HTML, JSON, CSV)");
    println!("• ⚡ Performance analysis across different data sizes");

    Ok(())
}

/// Generate 2D clustering test data
#[allow(dead_code)]
fn generate_2d_clustering_data() -> (Array2<f64>, Array1<i32>, Array2<f64>) {
    let data = Array2::from_shape_vec(
        (60, 2),
        vec![
            // Cluster 0
            1.0, 1.0, 1.1, 1.2, 0.9, 0.8, 1.2, 1.1, 0.8, 1.0, 1.3, 0.9, 1.0, 1.3, 0.7, 1.1, 1.1,
            0.9, 1.0, 0.8, // Cluster 1
            4.0, 4.0, 4.1, 4.2, 3.9, 3.8, 4.2, 4.1, 3.8, 4.0, 4.3, 3.9, 4.0, 4.3, 3.7, 4.1, 4.1,
            3.9, 4.0, 3.8, // Cluster 2
            7.0, 1.0, 7.1, 1.2, 6.9, 0.8, 7.2, 1.1, 6.8, 1.0, 7.3, 0.9, 7.0, 1.3, 6.7, 1.1, 7.1,
            0.9, 7.0, 0.8, // Additional points for each cluster
            1.4, 1.4, 1.5, 1.1, 0.6, 1.2, 1.3, 0.7, 0.9, 1.4, 4.4, 4.4, 4.5, 4.1, 3.6, 4.2, 4.3,
            3.7, 3.9, 4.4, 7.4, 1.4, 7.5, 1.1, 6.6, 1.2, 7.3, 0.7, 6.9, 1.4, 1.2, 0.6, 0.8, 1.5,
            1.6, 1.0, 0.5, 1.3, 1.1, 1.7, 4.2, 3.6, 3.8, 4.5, 4.6, 4.0, 3.5, 4.3, 4.1, 4.7, 7.2,
            0.6, 6.8, 1.5, 7.6, 1.0, 6.5, 1.3, 7.1, 1.7,
        ],
    )
    .unwrap();

    let labels = Array1::from_vec((0..60).map(|i| (i / 20) as i32).collect());
    let centroids = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 4.0, 4.0, 7.0, 1.0]).unwrap();

    (data, labels, centroids)
}

/// Generate 2D clustering test data with specified size
#[allow(dead_code)]
fn generate_2d_clustering_data_size(size: usize) -> (Array2<f64>, Array1<i32>, Array2<f64>) {
    use rand::Rng;

    let mut rng = rand::rng();
    let n_clusters = 3;
    let points_per_cluster = size / n_clusters;

    let mut data_vec = Vec::with_capacity(size * 2);
    let mut labels_vec = Vec::with_capacity(size);

    let cluster_centers = [(1.0, 1.0), (4.0, 4.0), (7.0, 1.0)];

    for (cluster_id, &(cx, cy)) in cluster_centers.iter().enumerate() {
        let end_idx = if cluster_id == n_clusters - 1 {
            size
        } else {
            (cluster_id + 1) * points_per_cluster
        };
        let start_idx = cluster_id * points_per_cluster;

        for _ in start_idx..end_idx {
            data_vec.push(cx + rng.random_range(-0.5..0.5));
            data_vec.push(cy + rng.random_range(-0.5..0.5));
            labels_vec.push(cluster_id as i32);
        }
    }

    let data = Array2::from_shape_vec((size, 2), data_vec).unwrap();
    let labels = Array1::from_vec(labels_vec);
    let centroids = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 4.0, 4.0, 7.0, 1.0]).unwrap();

    (data, labels, centroids)
}

/// Generate 3D clustering test data
#[allow(dead_code)]
fn generate_3d_clustering_data() -> (Array2<f64>, Array1<i32>, Array2<f64>) {
    let data = Array2::from_shape_vec(
        (30, 3),
        vec![
            // Cluster 0
            1.0, 1.0, 1.0, 1.1, 1.2, 0.9, 0.9, 0.8, 1.1, 1.2, 1.1, 0.8, 1.3, 0.9, 1.2, 1.0, 1.3,
            0.7, 0.8, 1.1, 1.0, 0.9, 0.8, 1.2, 1.1, 0.7, 1.3, 0.7, 1.4, 0.9, // Cluster 1
            4.0, 4.0, 4.0, 4.1, 4.2, 3.9, 3.9, 3.8, 4.1, 4.2, 4.1, 3.8, 4.3, 3.9, 4.2, 4.0, 4.3,
            3.7, 3.8, 4.1, 4.0, 3.9, 3.8, 4.2, 4.1, 3.7, 4.3, 3.7, 4.4, 3.9, // Cluster 2
            7.0, 1.0, 7.0, 7.1, 1.2, 6.9, 6.9, 0.8, 7.1, 7.2, 1.1, 6.8, 7.3, 0.9, 7.2, 7.0, 1.3,
            6.7, 6.8, 1.1, 7.0, 6.9, 0.8, 7.2, 7.1, 0.7, 7.3, 6.7, 1.4, 6.9,
        ],
    )
    .unwrap();

    let labels = Array1::from_vec((0..30).map(|i| (i / 10) as i32).collect());
    let centroids =
        Array2::from_shape_vec((3, 3), vec![1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 7.0, 1.0, 7.0]).unwrap();

    (data, labels, centroids)
}

/// Generate high-dimensional test data for dimensionality reduction
#[allow(dead_code)]
fn generate_high_dimensional_data() -> (Array2<f64>, Array1<i32>, Array2<f64>) {
    use rand::Rng;

    let mut rng = rand::rng();
    let n_samples = 150;
    let n_features = 10;
    let n_clusters = 3;

    let mut data_vec = Vec::with_capacity(n_samples * n_features);
    let mut labels_vec = Vec::with_capacity(n_samples);

    for cluster_id in 0..n_clusters {
        let cluster_center: Vec<f64> = (0..n_features)
            .map(|i| (cluster_id as f64 + 1.0) * (i as f64 + 1.0))
            .collect();

        for _ in 0..(n_samples / n_clusters) {
            for j in 0..n_features {
                data_vec.push(cluster_center[j] + rng.random_range(-1.0..1.0));
            }
            labels_vec.push(cluster_id as i32);
        }
    }

    let data = Array2::from_shape_vec((n_samples, n_features), data_vec).unwrap();
    let labels = Array1::from_vec(labels_vec);
    let centroids = Array2::zeros((n_clusters, n_features)); // Simplified for demo

    (data, labels, centroids)
}

/// Generate random centroids for animation demo
#[allow(dead_code)]
fn generate_random_centroids(k: usize, nfeatures: usize) -> Array2<f64> {
    use rand::Rng;

    let mut rng = rand::rng();
    let mut centroids = Array2::zeros((k, nfeatures));

    for i in 0..k {
        for j in 0..nfeatures {
            centroids[[i, j]] = rng.random_range(-2.0..8.0);
        }
    }

    centroids
}
