//! Advanced Clustering Visualization Demo
//!
//! This example demonstrates the advanced visualization capabilities for Advanced clustering,
//! including quantum state visualization, neuromorphic adaptation plots, and AI algorithm
//! selection insights.

use scirs2_cluster::advanced_clustering::AdvancedClusterer;
use scirs2_cluster::advanced_visualization::{
    create_advanced_visualization_report, visualize_advanced_results, AdvancedVisualizationConfig,
    AdvancedVisualizer, QuantumColorScheme, VisualizationExportFormat,
};
use scirs2_core::ndarray::Array2;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Clustering Visualization Demo");
    println!("==========================================");

    // Create sample data with multiple clusters
    let data = create_sample_data();
    println!(
        "📊 Created sample data with {} points in {} dimensions",
        data.nrows(),
        data.ncols()
    );

    // Perform Advanced clustering with all features enabled
    println!("\n🧠 Running Advanced clustering with full AI capabilities...");
    let mut clusterer = AdvancedClusterer::new()
        .with_ai_algorithm_selection(true)
        .with_quantum_neuromorphic_fusion(true)
        .with_meta_learning(true)
        .with_continual_adaptation(true)
        .with_multi_objective_optimization(true);

    let result = clusterer.cluster(&data.view())?;

    println!("✅ Clustering completed!");
    println!("   Selected algorithm: {}", result.selected_algorithm);
    println!("   AI speedup: {:.2}x", result.ai_speedup);
    println!("   Quantum advantage: {:.2}x", result.quantum_advantage);
    println!(
        "   Neuromorphic benefit: {:.2}x",
        result.neuromorphic_benefit
    );
    println!(
        "   Meta-learning improvement: {:.2}x",
        result.meta_learning_improvement
    );
    println!("   Confidence: {:.1}%", result.confidence * 100.0);

    // Create comprehensive visualization
    println!("\n🎨 Creating comprehensive Advanced visualization...");

    // Configuration for visualization
    let vis_config = AdvancedVisualizationConfig {
        show_quantum_coherence: true,
        show_neuromorphic_adaptation: true,
        show_ai_selection: true,
        quantum_color_scheme: QuantumColorScheme::QuantumRainbow,
        animation_speed: 1.0,
        export_format: VisualizationExportFormat::InteractiveHTML,
    };

    // Create visualizer and generate plots
    let mut visualizer = AdvancedVisualizer::new(vis_config);
    let visualization_output = visualizer.visualize_results(&data.view(), &result)?;

    println!("✅ Visualization created successfully!");

    // Display visualization insights
    display_visualization_insights(&visualization_output);

    // Export visualization in multiple formats
    println!("\n💾 Exporting visualizations...");
    visualizer.export_visualization(&visualization_output, "advanced_clustering_full")?;

    // Create a quick visualization report
    create_advanced_visualization_report(&data.view(), &result, "advanced_clustering_report")?;

    println!("✅ Visualizations exported successfully!");

    // Demonstrate different visualization configurations
    demonstrate_visualization_configurations(&data, &result)?;

    println!("\n🎯 Demo completed! Check the generated visualization files.");

    Ok(())
}

#[allow(dead_code)]
fn create_sample_data() -> Array2<f64> {
    // Create a multi-cluster dataset with interesting properties
    let mut data_vec = Vec::new();

    // Cluster 1: Dense circular cluster
    for i in 0..20 {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / 20.0;
        let radius = 1.0 + 0.2 * (i as f64 / 5.0).sin();
        data_vec.push(2.0 + radius * angle.cos());
        data_vec.push(2.0 + radius * angle.sin());
        data_vec.push(1.0 + 0.1 * angle.sin()); // Third dimension
    }

    // Cluster 2: Elongated cluster
    for i in 0..15 {
        let t = i as f64 / 14.0;
        data_vec.push(8.0 + 3.0 * t);
        data_vec.push(1.0 + 0.5 * t + 0.1 * (t * 10.0).sin());
        data_vec.push(2.0 + 0.2 * (t * 5.0).cos());
    }

    // Cluster 3: Spiral cluster
    for i in 0..18 {
        let angle = 4.0 * std::f64::consts::PI * i as f64 / 18.0;
        let radius = 0.5 + angle / 8.0;
        data_vec.push(15.0 + radius * angle.cos());
        data_vec.push(8.0 + radius * angle.sin());
        data_vec.push(3.0 + 0.3 * angle);
    }

    // Cluster 4: Random high-dimensional component
    for i in 0..12 {
        let base_x = 5.0;
        let base_y = 15.0;
        let noise_scale = 0.8;
        data_vec.push(base_x + noise_scale * ((i * 7) as f64 % 31.0) / 31.0);
        data_vec.push(base_y + noise_scale * ((i * 11) as f64 % 37.0) / 37.0);
        data_vec.push(4.0 + 0.5 * ((i * 13) as f64 % 23.0) / 23.0);
    }

    Array2::from_shape_vec((65, 3), data_vec).unwrap()
}

#[allow(dead_code)]
fn display_visualization_insights(
    output: &scirs2_cluster::advanced_visualization::AdvancedVisualizationOutput,
) {
    println!("\n📈 Visualization Insights:");
    println!("  📊 Cluster Plot:");
    println!("     - Data points: {}", output.cluster_plot.data.nrows());
    println!("     - Dimensions: {}", output.cluster_plot.data.ncols());
    println!(
        "     - Quantum enhancement: {:.2}x",
        output.cluster_plot.quantum_enhancement
    );
    println!(
        "     - Color scheme: {} unique colors",
        output.cluster_plot.colors.len()
    );

    if let Some(ref quantum_plot) = output.quantum_plot {
        println!("  ⚛️  Quantum Coherence Plot:");
        println!("     - Time points: {}", quantum_plot.time_points.len());
        println!("     - Coherence evolution tracked");
        println!(
            "     - Entanglement network: {} connections",
            quantum_plot.entanglement_network.len()
        );
    }

    if let Some(ref neuro_plot) = output.neuromorphic_plot {
        println!("  🧠 Neuromorphic Adaptation Plot:");
        println!(
            "     - Neuron activity: {}x{} matrix",
            neuro_plot.neuron_activity.nrows(),
            neuro_plot.neuron_activity.ncols()
        );
        println!(
            "     - Weight evolution: {} time steps",
            neuro_plot.weight_evolution.len()
        );
        println!("     - Learning curve tracked");
    }

    if let Some(ref ai_plot) = output.ai_selection_plot {
        println!("  🤖 AI Algorithm Selection Plot:");
        println!(
            "     - Algorithms evaluated: {}",
            ai_plot.algorithm_scores.len()
        );
        println!("     - Selected algorithm: {}", ai_plot.selected_algorithm);
        println!(
            "     - Selection timeline: {} steps",
            ai_plot.selection_timeline.len()
        );
        println!(
            "     - Meta-learning progress: {} steps",
            ai_plot.meta_learning_timeline.len()
        );
    }

    println!("  📊 Performance Dashboard:");
    println!(
        "     - Advanced metrics: {}",
        output.performance_dashboard.advanced_metrics.len()
    );
    println!(
        "     - Classical baseline comparisons: {}",
        output.performance_dashboard.classical_baseline.len()
    );
    println!(
        "     - Execution time: {:.3}s",
        output.performance_dashboard.execution_time
    );
    println!(
        "     - Memory usage: {:.1} MB",
        output.performance_dashboard.memory_usage
    );
    println!(
        "     - AI optimization iterations: {}",
        output.performance_dashboard.ai_iterations
    );
    println!(
        "     - Improvement factors: {}",
        output.performance_dashboard.improvement_factors.len()
    );
}

#[allow(dead_code)]
fn demonstrate_visualization_configurations(
    data: &Array2<f64>,
    result: &scirs2_cluster::advanced_clustering::AdvancedClusteringResult,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎨 Demonstrating different visualization configurations...");

    // Configuration 1: Quantum-focused visualization
    println!("  ⚛️  Quantum-focused visualization...");
    let quantum_config = AdvancedVisualizationConfig {
        show_quantum_coherence: true,
        show_neuromorphic_adaptation: false,
        show_ai_selection: false,
        quantum_color_scheme: QuantumColorScheme::PhaseWheel,
        animation_speed: 2.0,
        export_format: VisualizationExportFormat::JSONData,
    };

    let quantum_vis = visualize_advanced_results(&data.view(), result, Some(quantum_config))?;
    println!("     ✅ Quantum visualization created");

    // Configuration 2: Neuromorphic-focused visualization
    println!("  🧠 Neuromorphic-focused visualization...");
    let neuro_config = AdvancedVisualizationConfig {
        show_quantum_coherence: false,
        show_neuromorphic_adaptation: true,
        show_ai_selection: false,
        quantum_color_scheme: QuantumColorScheme::CoherenceScale,
        animation_speed: 0.5,
        export_format: VisualizationExportFormat::VectorSVG,
    };

    let neuro_vis = visualize_advanced_results(&data.view(), result, Some(neuro_config))?;
    println!("     ✅ Neuromorphic visualization created");

    // Configuration 3: AI selection analysis
    println!("  🤖 AI selection analysis visualization...");
    let ai_config = AdvancedVisualizationConfig {
        show_quantum_coherence: false,
        show_neuromorphic_adaptation: false,
        show_ai_selection: true,
        quantum_color_scheme: QuantumColorScheme::QuantumRainbow,
        animation_speed: 1.5,
        export_format: VisualizationExportFormat::InteractiveHTML,
    };

    let ai_vis = visualize_advanced_results(&data.view(), result, Some(ai_config))?;
    println!("     ✅ AI selection visualization created");

    // Configuration 4: Custom color scheme
    println!("  🌈 Custom color scheme visualization...");
    let custom_colors = vec![
        [1.0, 0.0, 0.0], // Red
        [0.0, 1.0, 0.0], // Green
        [0.0, 0.0, 1.0], // Blue
        [1.0, 1.0, 0.0], // Yellow
        [1.0, 0.0, 1.0], // Magenta
    ];

    let custom_config = AdvancedVisualizationConfig {
        show_quantum_coherence: true,
        show_neuromorphic_adaptation: true,
        show_ai_selection: true,
        quantum_color_scheme: QuantumColorScheme::Custom(custom_colors),
        animation_speed: 0.8,
        export_format: VisualizationExportFormat::AnimatedGIF,
    };

    let custom_vis = visualize_advanced_results(&data.view(), result, Some(custom_config))?;
    println!("     ✅ Custom color visualization created");

    println!("  🎯 All visualization configurations completed!");

    Ok(())
}
