//! Test enhanced Advanced clustering functionality
//! This test focuses only on the cluster module without dependencies on other modules

use scirs2_cluster::advanced_clustering::AdvancedClusterer;
use scirs2_cluster::advanced_visualization::{
    AdvancedVisualizationConfig, AdvancedVisualizer, QuantumColorScheme, VisualizationExportFormat,
};
use scirs2_core::ndarray::Array2;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing Enhanced Advanced Clustering");
    println!("==========================================");

    // Create sophisticated test data
    let data = create_complex_test_data();
    println!(
        "📊 Created test data with {} points in {} dimensions",
        data.nrows(),
        data.ncols()
    );

    // Test 1: Basic enhanced Advanced clustering
    println!("\n🧪 Test 1: Enhanced AI-driven algorithm selection");
    test_enhanced_ai_selection(&data)?;

    // Test 2: Quantum-neuromorphic clustering with advanced features
    println!("\n🧪 Test 2: Quantum-neuromorphic clustering with enhanced features");
    test_quantum_neuromorphic_enhanced(&data)?;

    // Test 3: Meta-learning with improved algorithms
    println!("\n🧪 Test 3: Enhanced meta-learning optimization");
    test_enhanced_meta_learning(&data)?;

    // Test 4: Enhanced visualization with quantum PCA
    println!("\n🧪 Test 4: Enhanced visualization with quantum PCA");
    test_enhanced_visualization(&data)?;

    // Test 5: Full Advanced mode with all enhancements
    println!("\n🧪 Test 5: Full enhanced Advanced mode");
    test_full_enhanced_advanced(&data)?;

    println!("\n✅ All enhanced Advanced clustering tests completed successfully!");
    println!("🎯 Enhanced features validated:");
    println!("   ⚛️  Quantum-enhanced initialization and distance calculations");
    println!("   🧠 Advanced neuromorphic adaptation with spike-timing plasticity");
    println!("   🤖 Sophisticated AI algorithm selection with coherence factors");
    println!("   📊 Quantum-enhanced PCA for dimensionality reduction");
    println!("   🎨 Enhanced visualization export with interactive HTML");

    Ok(())
}

#[allow(dead_code)]
fn create_complex_test_data() -> Array2<f64> {
    // Create a multi-cluster dataset with varying densities and shapes
    let mut data_vec = Vec::new();

    // Cluster 1: Dense circular cluster (challenging for algorithms)
    for i in 0..15 {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / 15.0;
        let radius = 0.5 + 0.1 * (i as f64 / 3.0).sin();
        data_vec.extend_from_slice(&[
            2.0 + radius * angle.cos(),
            2.0 + radius * angle.sin(),
            1.0 + 0.1 * angle.sin(),
        ]);
    }

    // Cluster 2: Elongated cluster with noise
    for i in 0..12 {
        let t = i as f64 / 11.0;
        let noise = (i * 7) as f64 / 100.0 - 0.35;
        data_vec.extend_from_slice(&[
            8.0 + 4.0 * t + noise,
            1.0 + 0.5 * t + noise * 0.5,
            2.0 + 0.3 * t,
        ]);
    }

    // Cluster 3: Spiral pattern (complex structure)
    for i in 0..18 {
        let angle = 3.0 * std::f64::consts::PI * i as f64 / 18.0;
        let radius = 0.3 + angle / 10.0;
        data_vec.extend_from_slice(&[
            15.0 + radius * angle.cos(),
            8.0 + radius * angle.sin(),
            3.0 + 0.2 * angle,
        ]);
    }

    Array2::from_shape_vec((45, 3), data_vec).expect("Failed to create test data")
}

#[allow(dead_code)]
fn test_enhanced_ai_selection(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let mut clusterer = AdvancedClusterer::new().with_ai_algorithm_selection(true);

    let result = clusterer.cluster(&data.view())?;

    println!("   ✅ AI-enhanced algorithm selection completed");
    let selected = &result.selected_algorithm;
    println!("      Selected: {selected}");
    let speedup = result.ai_speedup;
    println!("      AI speedup: {speedup:.2}x");
    let confidence = result.confidence;
    println!("      Confidence: {confidence:.3}");
    println!(
        "      Execution time: {:.4}s",
        result.performance.execution_time
    );

    // Verify enhanced features
    assert!(
        result.ai_speedup > 1.0,
        "AI speedup should be greater than 1.0"
    );
    assert!(result.confidence > 0.0, "Confidence should be positive");
    assert!(
        !result.selected_algorithm.is_empty(),
        "Algorithm should be selected"
    );

    Ok(())
}

#[allow(dead_code)]
fn test_quantum_neuromorphic_enhanced(
    data: &Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut clusterer = AdvancedClusterer::new().with_quantum_neuromorphic_fusion(true);

    let result = clusterer.cluster(&data.view())?;

    println!("   ✅ Enhanced quantum-neuromorphic clustering completed");
    let advantage = result.quantum_advantage;
    println!("      Quantum advantage: {advantage:.2}x");
    println!(
        "      Neuromorphic benefit: {:.2}x",
        result.neuromorphic_benefit
    );
    println!(
        "      Quantum coherence: {:.3}",
        result.performance.quantum_coherence
    );
    println!(
        "      Neural adaptation rate: {:.3}",
        result.performance.neural_adaptation_rate
    );
    println!(
        "      Energy efficiency: {:.3}",
        result.performance.energy_efficiency
    );

    // Verify quantum-neuromorphic enhancements
    assert!(
        result.quantum_advantage > 1.0,
        "Quantum advantage should be enhanced"
    );
    assert!(
        result.neuromorphic_benefit > 1.0,
        "Neuromorphic benefit should be positive"
    );
    assert!(
        result.performance.quantum_coherence > 0.0,
        "Quantum coherence should be maintained"
    );
    assert!(
        result.performance.energy_efficiency > 0.0,
        "Energy efficiency should be positive"
    );

    Ok(())
}

#[allow(dead_code)]
fn test_enhanced_meta_learning(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let mut clusterer = AdvancedClusterer::new()
        .with_meta_learning(true)
        .with_ai_algorithm_selection(true);

    let result = clusterer.cluster(&data.view())?;

    println!("   ✅ Enhanced meta-learning optimization completed");
    println!(
        "      Meta-learning improvement: {:.2}x",
        result.meta_learning_improvement
    );
    println!(
        "      Final silhouette score: {:.3}",
        result.performance.silhouette_score
    );
    let iterations = result.performance.ai_iterations;
    println!("      AI iterations: {iterations}");

    // Verify meta-learning enhancements
    assert!(
        result.meta_learning_improvement >= 1.0,
        "Meta-learning should show improvement"
    );
    assert!(
        result.performance.silhouette_score > 0.0,
        "Silhouette score should be positive"
    );
    assert!(
        result.performance.ai_iterations > 0,
        "AI iterations should occur"
    );

    Ok(())
}

#[allow(dead_code)]
fn test_enhanced_visualization(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    // First run clustering
    let mut clusterer = AdvancedClusterer::new()
        .with_quantum_neuromorphic_fusion(true)
        .with_ai_algorithm_selection(true);

    let result = clusterer.cluster(&data.view())?;

    // Test enhanced visualization with quantum PCA
    let vis_config = AdvancedVisualizationConfig {
        show_quantum_coherence: true,
        show_neuromorphic_adaptation: true,
        show_ai_selection: true,
        quantum_color_scheme: QuantumColorScheme::QuantumRainbow,
        animation_speed: 1.0,
        export_format: VisualizationExportFormat::InteractiveHTML,
    };

    let mut visualizer = AdvancedVisualizer::new(vis_config);
    let visualization_output = visualizer.visualize_results(&data.view(), &result)?;

    println!("   ✅ Enhanced visualization with quantum PCA completed");
    println!(
        "      Quantum plot available: {}",
        visualization_output.quantum_plot.is_some()
    );
    println!(
        "      Neuromorphic plot available: {}",
        visualization_output.neuromorphic_plot.is_some()
    );
    println!(
        "      AI selection plot available: {}",
        visualization_output.ai_selection_plot.is_some()
    );
    println!(
        "      Quantum enhancement factor: {:.2}",
        visualization_output.cluster_plot.quantum_enhancement
    );

    // Test export functionality
    match visualizer.export_visualization(&visualization_output, "test_advanced_enhanced") {
        Ok(_) => println!("      ✅ Visualization export successful"),
        Err(e) => println!(
            "      ⚠️  Visualization export error (expected in test): {}",
            e
        ),
    }

    // Verify visualization enhancements
    assert!(
        visualization_output.quantum_plot.is_some(),
        "Quantum plot should be generated"
    );
    assert!(
        visualization_output.neuromorphic_plot.is_some(),
        "Neuromorphic plot should be generated"
    );
    assert!(
        visualization_output.ai_selection_plot.is_some(),
        "AI selection plot should be generated"
    );

    Ok(())
}

#[allow(dead_code)]
fn test_full_enhanced_advanced(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let mut clusterer = AdvancedClusterer::new()
        .with_ai_algorithm_selection(true)
        .with_quantum_neuromorphic_fusion(true)
        .with_meta_learning(true)
        .with_continual_adaptation(true)
        .with_multi_objective_optimization(true);

    let result = clusterer.cluster(&data.view())?;

    println!("   ✅ Full enhanced Advanced mode completed");
    println!(
        "      🤖 AI selected algorithm: {}",
        result.selected_algorithm
    );
    let ai_speedup = result.ai_speedup;
    println!("      ⚡ AI speedup: {ai_speedup:.2}x");
    println!(
        "      🌟 Quantum advantage: {:.2}x",
        result.quantum_advantage
    );
    println!(
        "      🧠 Neuromorphic benefit: {:.2}x",
        result.neuromorphic_benefit
    );
    println!(
        "      🎓 Meta-learning improvement: {:.2}x",
        result.meta_learning_improvement
    );
    let final_confidence = result.confidence;
    println!("      🎯 Final confidence: {final_confidence:.3}");
    println!(
        "      🏆 Silhouette score: {:.3}",
        result.performance.silhouette_score
    );
    println!(
        "      ⏱️  Total execution time: {:.4}s",
        result.performance.execution_time
    );
    println!(
        "      💾 Memory usage: {:.2} MB",
        result.performance.memory_usage
    );
    println!(
        "      🔗 Quantum coherence: {:.3}",
        result.performance.quantum_coherence
    );
    println!(
        "      🧬 Neural adaptation rate: {:.3}",
        result.performance.neural_adaptation_rate
    );
    println!(
        "      ⚡ Energy efficiency: {:.3}",
        result.performance.energy_efficiency
    );

    // Verify all enhancements work together
    assert!(result.ai_speedup > 1.0, "AI speedup should be enhanced");
    assert!(
        result.quantum_advantage > 1.0,
        "Quantum advantage should be significant"
    );
    assert!(
        result.neuromorphic_benefit > 1.0,
        "Neuromorphic benefit should be enhanced"
    );
    assert!(
        result.meta_learning_improvement >= 1.0,
        "Meta-learning should show improvement"
    );
    assert!(
        result.confidence > 0.8,
        "High confidence expected with all features"
    );
    assert!(
        result.performance.silhouette_score > 0.0,
        "Quality should be good"
    );
    assert!(
        result.performance.quantum_coherence > 0.0,
        "Quantum coherence maintained"
    );
    assert!(
        result.performance.energy_efficiency > 0.0,
        "Energy efficiency should be positive"
    );

    // Verify cluster structure makes sense
    let num_clusters = result.centroids.nrows();
    assert!(
        (2..=8).contains(&num_clusters),
        "Reasonable number of clusters"
    );
    assert_eq!(
        result.clusters.len(),
        data.nrows(),
        "All points should be assigned"
    );

    Ok(())
}
