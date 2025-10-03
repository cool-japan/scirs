//! Demonstration of animated K-means clustering visualization
//!
//! This example shows how to create animated visualizations of K-means clustering
//! that capture the iterative convergence process with rich visual feedback.

use scirs2_cluster::preprocess::standardize;
use scirs2_cluster::visualization::animation::{
    AnimationFrame, ConvergenceInfo, IterativeAnimationConfig, IterativeAnimationRecorder,
};
use scirs2_cluster::vq::kmeans2;
use scirs2_cluster::VisualizationConfig;
use scirs2_core::ndarray::Array2;
use std::path::Path;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Animated K-means Clustering Demo");
    println!("===============================");

    // Create sample data with multiple distinct clusters
    let data = generate_multi_cluster_data();
    println!(
        "Generated {} data points with {} features",
        data.nrows(),
        data.ncols()
    );

    // Standardize the data
    let standardized = standardize(data.view(), true)?;

    // Create animation configuration
    let animation_config = IterativeAnimationConfig {
        capture_frequency: 1,
        interpolate_frames: true,
        interpolation_frames: 5,
        fps: 10.0,
        show_convergence_overlay: true,
        show_iteration_numbers: true,
        highlight_centroid_movement: true,
        fade_effect: true,
        trail_length: 8,
    };

    // Create visualization configuration
    let vis_config = VisualizationConfig {
        color_scheme: scirs2_cluster::visualization::ColorScheme::ColorblindFriendly,
        point_size: 6.0,
        point_opacity: 0.8,
        show_centroids: true,
        show_boundaries: false,
        boundary_type: scirs2_cluster::visualization::BoundaryType::None,
        interactive: false,
        animation: Some(scirs2_cluster::visualization::AnimationConfig {
            duration_ms: 3000,
            frames: 60,
            easing: scirs2_cluster::visualization::EasingFunction::EaseInOut,
            loop_animation: false,
        }),
        dimensionality_reduction: scirs2_cluster::visualization::DimensionalityReduction::None,
    };

    // Demonstrate animated K-means with different values of K
    for k in [2, 3, 4, 5] {
        println!("\nRunning animated K-means with k={}", k);

        let output_path = format!("animated_kmeans_k{}.html", k);
        run_animated_kmeans(
            standardized.view(),
            k,
            &animation_config,
            &vis_config,
            &output_path,
        )?;

        println!("Animation saved to: {}", output_path);
    }

    // Demonstrate streaming visualization
    println!("\nDemonstrating streaming visualization...");
    demonstrate_streaming_visualization(&standardized)?;

    // Demonstrate 3D animation
    println!("\nDemonstrating 3D animation...");
    demonstrate_3d_animation()?;

    println!("\nAnimation demos completed!");
    println!("Open the generated HTML files in a web browser to view the animations.");

    Ok(())
}

/// Run animated K-means clustering with full convergence tracking
#[allow(dead_code)]
fn run_animated_kmeans(
    data: scirs2_core::ndarray::ArrayView2<f64>,
    k: usize,
    animation_config: &IterativeAnimationConfig,
    vis_config: &VisualizationConfig,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut recorder = IterativeAnimationRecorder::new(animation_config.clone());

    // Manual K-means implementation with animation capture
    let n_samples = data.nrows();
    let n_features = data.ncols();

    // Initialize centroids randomly
    let mut centroids = Array2::zeros((k, n_features));
    for i in 0..k {
        for j in 0..n_features {
            centroids[[i, j]] = data[[i * n_samples / k, j]];
        }
    }

    let mut labels = scirs2_core::ndarray::Array1::zeros(n_samples);
    let mut prev_inertia = f64::INFINITY;
    let max_iterations = 50;

    // Capture initial state
    recorder.record_frame(
        data,
        &labels.mapv(|x| x as i32),
        Some(&centroids),
        Some(prev_inertia),
    )?;

    for iteration in 1..=max_iterations {
        let mut points_changed = 0;
        let mut total_distance = 0.0;

        // Assignment step
        for i in 0..n_samples {
            let sample = data.row(i);
            let mut min_distance = f64::INFINITY;
            let mut best_cluster = 0;

            for j in 0..k {
                let centroid = centroids.row(j);
                let distance: f64 = sample
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                total_distance += distance;

                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = j;
                }
            }

            if labels[i] != best_cluster {
                points_changed += 1;
            }
            labels[i] = best_cluster;
        }

        // Update step
        let old_centroids = centroids.clone();
        let mut max_centroid_movement: f64 = 0.0;

        for j in 0..k {
            let cluster_points: Vec<usize> = (0..n_samples).filter(|&i| labels[i] == j).collect();

            if !cluster_points.is_empty() {
                for feature in 0..n_features {
                    let mean = cluster_points
                        .iter()
                        .map(|&i| data[[i, feature]])
                        .sum::<f64>()
                        / cluster_points.len() as f64;
                    centroids[[j, feature]] = mean;
                }

                // Calculate centroid movement
                let movement: f64 = (0..n_features)
                    .map(|f| (centroids[[j, f]] - old_centroids[[j, f]]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                max_centroid_movement = max_centroid_movement.max(movement);
            }
        }

        // Calculate inertia
        let mut inertia = 0.0;
        for i in 0..n_samples {
            let sample = data.row(i);
            let cluster = labels[i];
            let centroid = centroids.row(cluster);
            let distance_sq: f64 = sample
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            inertia += distance_sq;
        }

        let inertia_change = prev_inertia - inertia;
        let converged = inertia_change < 1e-4 && points_changed == 0;

        // Capture frame
        recorder.record_frame(
            data,
            &labels.mapv(|x| x as i32),
            Some(&centroids),
            Some(inertia),
        )?;

        println!(
            "  Iteration {}: inertia={:.4}, change={:.6}, points_changed={}, converged={}",
            iteration, inertia, inertia_change, points_changed, converged
        );

        if converged {
            println!("  Converged after {} iterations", iteration);
            break;
        }

        prev_inertia = inertia;
    }

    // Export animation to JSON
    let json_content = recorder.export_to_json()?;
    std::fs::write(output_path, json_content)?;

    Ok(())
}

/// Demonstrate streaming visualization capabilities
#[allow(dead_code)]
fn demonstrate_streaming_visualization(
    data: &Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_cluster::visualization::animation::{StreamingConfig, StreamingVisualizer};

    let streaming_config = StreamingConfig {
        buffer_size: 500,
        update_frequency_ms: 100,
        rolling_window_size: 100,
        animate_new_data: true,
        animate_cluster_updates: true,
        adaptive_bounds: true,
        show_streaming_stats: true,
        point_lifetime_ms: 5000,
    };

    let mut visualizer = StreamingVisualizer::new(streaming_config);

    println!("Simulating streaming data...");

    // Simulate streaming data by adding points one by one
    for (i, point) in data.rows().into_iter().enumerate() {
        let label = i % 3; // Simulate clustering labels
        visualizer.add_data_point(point.to_owned(), label as i32);
        let should_update = true; // Simplified for demo

        if should_update {
            println!("  Updated visualization at point {}", i + 1);
        }

        // Simulate real-time delay
        std::thread::sleep(std::time::Duration::from_millis(10));

        if i > 100 {
            break;
        } // Limit demo
    }

    // Export streaming data (simplified)
    println!("Streaming visualization completed");

    Ok(())
}

/// Demonstrate 3D animation capabilities
#[allow(dead_code)]
fn demonstrate_3d_animation() -> Result<(), Box<dyn std::error::Error>> {
    // Generate 3D data
    let data_3d = generate_3d_cluster_data();
    println!("Generated 3D data with {} points", data_3d.nrows());

    // Create 3D visualization
    let vis_config = VisualizationConfig {
        dimensionality_reduction: scirs2_cluster::visualization::DimensionalityReduction::None,
        ..Default::default()
    };

    // Run K-means on 3D data
    let (centroids, labels) = kmeans2(
        data_3d.view(),
        3,
        Some(30),
        Some(1e-4),
        None,
        None,
        None,
        Some(42),
    )?;

    // Create 3D scatter plot
    let scatter_plot_3d = scirs2_cluster::visualization::create_scatter_plot_3d(
        data_3d.view(),
        &labels.mapv(|x| x as i32),
        Some(&centroids),
        &vis_config,
    )?;

    // Export 3D visualization to HTML
    scirs2_cluster::visualization::export::export_scatter_3d_to_html(
        &scatter_plot_3d,
        "3d_clustering_visualization.html",
        &Default::default(),
    )?;
    println!("3D visualization saved to: 3d_clustering_visualization.html");

    Ok(())
}

/// Generate multi-cluster 2D data for demonstration
#[allow(dead_code)]
fn generate_multi_cluster_data() -> Array2<f64> {
    let mut data = Vec::new();

    // Cluster 1: centered at (2, 2)
    for _ in 0..50 {
        data.push(2.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.5);
        data.push(2.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.5);
    }

    // Cluster 2: centered at (-2, 2)
    for _ in 0..50 {
        data.push(-2.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.5);
        data.push(2.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.5);
    }

    // Cluster 3: centered at (0, -2)
    for _ in 0..50 {
        data.push(0.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.5);
        data.push(-2.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.5);
    }

    // Cluster 4: centered at (4, -1)
    for _ in 0..50 {
        data.push(4.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.5);
        data.push(-1.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.5);
    }

    Array2::from_shape_vec((200, 2), data).unwrap()
}

/// Generate 3D cluster data for demonstration
#[allow(dead_code)]
fn generate_3d_cluster_data() -> Array2<f64> {
    let mut data = Vec::new();

    // Generate 3 distinct 3D clusters
    let centers = [(2.0, 2.0, 2.0), (-2.0, 2.0, -2.0), (0.0, -2.0, 0.0)];

    for center in &centers {
        for _ in 0..40 {
            data.push(center.0 + (scirs2_core::random::random::<f64>() - 0.5) * 2.0);
            data.push(center.1 + (scirs2_core::random::random::<f64>() - 0.5) * 2.0);
            data.push(center.2 + (scirs2_core::random::random::<f64>() - 0.5) * 2.0);
        }
    }

    Array2::from_shape_vec((120, 3), data).unwrap()
}
