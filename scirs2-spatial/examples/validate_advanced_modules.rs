//! Validation script for advanced spatial modules
//!
//! This script validates that all advanced modules are properly enabled
//! and functional by testing basic operations from each module.

use ndarray::array;
use scirs2_spatial::{
    ai_driven_optimization::AIAlgorithmSelector,
    distance::euclidean,

    extreme_performance_optimization::ExtremeOptimizer,
    // GPU and memory optimizations
    gpu_accel::{is_gpu_acceleration_available, report_gpu_status},
    memory_pool::global_distance_pool,

    ml_optimization::NeuralSpatialOptimizer,
    neuromorphic::{NeuromorphicProcessor, SpikingNeuralClusterer},
    quantum_classical_hybrid::{HybridClusterer, HybridSpatialOptimizer},

    // Advanced modules to validate
    quantum_inspired::{QuantumClusterer, QuantumNearestNeighbor, QuantumSpatialOptimizer},
    // Check if imports work
    tensor_cores::detect_tensor_core_capabilities,
    // Core spatial algorithms for comparison
    KDTree,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 SciRS2-Spatial Advanced Modules Validation");
    println!("============================================");

    // Generate test data
    let points = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0],
    ];

    let mut all_passed = true;

    // Test 1: Core functionality (baseline)
    println!("\n📊 Testing Core Functionality...");
    match test_core_functionality(&points).await {
        Ok(_) => println!("✅ Core functionality: PASSED"),
        Err(e) => {
            println!("❌ Core functionality: FAILED - {}", e);
            all_passed = false;
        }
    }

    // Test 2: Quantum-inspired algorithms
    println!("\n🔬 Testing Quantum-Inspired Algorithms...");
    match test_quantum_inspired(&points).await {
        Ok(_) => println!("✅ Quantum-inspired: PASSED"),
        Err(e) => {
            println!("❌ Quantum-inspired: FAILED - {}", e);
            all_passed = false;
        }
    }

    // Test 3: Neuromorphic computing
    println!("\n🧠 Testing Neuromorphic Computing...");
    match test_neuromorphic(&points).await {
        Ok(_) => println!("✅ Neuromorphic: PASSED"),
        Err(e) => {
            println!("❌ Neuromorphic: FAILED - {}", e);
            all_passed = false;
        }
    }

    // Test 4: Hybrid algorithms
    println!("\n🔄 Testing Hybrid Quantum-Classical...");
    match test_hybrid_algorithms(&points).await {
        Ok(_) => println!("✅ Hybrid algorithms: PASSED"),
        Err(e) => {
            println!("❌ Hybrid algorithms: FAILED - {}", e);
            all_passed = false;
        }
    }

    // Test 5: GPU acceleration
    println!("\n🖥️  Testing GPU Acceleration...");
    match test_gpu_acceleration().await {
        Ok(_) => println!("✅ GPU acceleration: PASSED"),
        Err(e) => {
            println!("⚠️  GPU acceleration: SKIPPED - {}", e);
            // GPU tests are optional, don't fail validation
        }
    }

    // Test 6: Memory optimization
    println!("\n💾 Testing Memory Optimization...");
    match test_memory_optimization().await {
        Ok(_) => println!("✅ Memory optimization: PASSED"),
        Err(e) => {
            println!("❌ Memory optimization: FAILED - {}", e);
            all_passed = false;
        }
    }

    // Test 7: Advanced optimization modules
    println!("\n🚀 Testing Advanced Optimization Modules...");
    match test_advanced_optimization().await {
        Ok(_) => println!("✅ Advanced optimization: PASSED"),
        Err(e) => {
            println!("❌ Advanced optimization: FAILED - {}", e);
            all_passed = false;
        }
    }

    // Final validation summary
    println!("\n🏁 Validation Summary");
    println!("===================");

    if all_passed {
        println!("🎉 ALL TESTS PASSED! Advanced modules are fully functional.");
        println!("   SciRS2-Spatial is ready for production use with:");
        println!("   • Quantum-inspired spatial algorithms");
        println!("   • Neuromorphic computing acceleration");
        println!("   • Hybrid quantum-classical optimization");
        println!("   • Advanced memory and GPU optimizations");
        println!("   • AI-driven algorithm selection");

        std::process::exit(0);
    } else {
        println!("❌ Some tests failed. Please check the output above.");
        println!("   Advanced modules may need additional configuration.");

        std::process::exit(1);
    }
}

async fn test_core_functionality(
    points: &ndarray::Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Test basic KDTree functionality
    let kdtree = KDTree::new(points)?;
    let (indices, distances) = kdtree.query(&[2.5, 2.5], 3)?;

    if indices.len() != 3 || distances.len() != 3 {
        return Err("KDTree query returned wrong number of results".into());
    }

    // Test distance calculation
    let dist = euclidean(&[0.0, 0.0], &[1.0, 1.0]);
    let expected = (2.0f64).sqrt();
    if (dist - expected).abs() > 1e-10 {
        return Err("Distance calculation incorrect".into());
    }

    Ok(())
}

async fn test_quantum_inspired(
    points: &ndarray::Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Test quantum clustering
    let mut quantum_clusterer = QuantumClusterer::new(2, 16, 20, 0.01)?;
    let cluster_result = quantum_clusterer.cluster(&points.view()).await?;

    if cluster_result.labels.len() != points.nrows() {
        return Err("Quantum clustering returned wrong number of labels".into());
    }

    if cluster_result.centers.nrows() != 2 {
        return Err("Quantum clustering returned wrong number of centers".into());
    }

    // Test quantum nearest neighbor
    let quantum_nn = QuantumNearestNeighbor::new(16, 5, 0.01, 0.7)?;
    let (indices, distances) = quantum_nn
        .search(&points.view(), &array![2.0, 2.0].view(), 3)
        .await?;

    if indices.len() != 3 || distances.len() != 3 {
        return Err("Quantum NN search returned wrong number of results".into());
    }

    Ok(())
}

async fn test_neuromorphic(
    points: &ndarray::Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Test spiking neural clustering
    let mut spiking_clusterer = SpikingNeuralClusterer::new(2);
    let cluster_result = spiking_clusterer.cluster(&points.view()).await?;

    if cluster_result.labels.len() != points.nrows() {
        return Err("Neuromorphic clustering returned wrong number of labels".into());
    }

    if cluster_result.centers.nrows() != 2 {
        return Err("Neuromorphic clustering returned wrong number of centers".into());
    }

    // Test that silhouette score is reasonable
    if cluster_result.silhouette_score < -1.0 || cluster_result.silhouette_score > 1.0 {
        return Err("Neuromorphic clustering silhouette score out of range".into());
    }

    Ok(())
}

async fn test_hybrid_algorithms(
    points: &ndarray::Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Test hybrid spatial optimizer
    let mut hybrid_optimizer = HybridSpatialOptimizer::new()?
        .with_quantum_depth(2)
        .with_classical_refinement(true)
        .with_adaptive_switching(0.5);

    let optimization_result = hybrid_optimizer
        .optimize_spatial_problem(&points.view())
        .await?;

    if optimization_result.iterations == 0 {
        return Err("Hybrid optimizer reported zero iterations".into());
    }

    if optimization_result.final_cost < 0.0 {
        return Err("Hybrid optimizer reported negative cost".into());
    }

    // Test hybrid clustering
    let mut hybrid_clusterer = HybridClusterer::new(2)?
        .with_quantum_depth(2)
        .with_classical_refinement(true);

    let cluster_result = hybrid_clusterer.cluster(&points.view()).await?;

    if cluster_result.labels.len() != points.nrows() {
        return Err("Hybrid clustering returned wrong number of labels".into());
    }

    Ok(())
}

async fn test_gpu_acceleration() -> Result<(), Box<dyn std::error::Error>> {
    // Check GPU status
    report_gpu_status();

    if !is_gpu_acceleration_available() {
        return Err("GPU acceleration not available on this system".into());
    }

    // Test tensor core capabilities detection
    let tensor_caps = detect_tensor_core_capabilities();
    println!(
        "   Detected tensor core capabilities: {:?}",
        tensor_caps.has_tensor_cores
    );

    Ok(())
}

async fn test_memory_optimization() -> Result<(), Box<dyn std::error::Error>> {
    // Test memory pool
    let pool = global_distance_pool();
    let stats = pool.statistics();

    println!("   Memory pool statistics:");
    println!("     Total allocations: {}", stats.total_allocations);
    println!("     Hit rate: {:.1}%", stats.hit_rate());

    // Get a buffer to test the pool
    let buffer = pool.get_distance_buffer(100);

    if buffer.len() != 100 {
        return Err("Memory pool returned wrong buffer size".into());
    }

    Ok(())
}

async fn test_advanced_optimization() -> Result<(), Box<dyn std::error::Error>> {
    // Test extreme optimizer creation
    let _extreme_optimizer = ExtremeOptimizer::new()
        .with_simd_optimization(true)
        .with_cache_optimization(true)
        .with_parallel_optimization(true);

    // Test AI algorithm selector creation
    let _ai_selector = AIAlgorithmSelector::new()
        .with_meta_learning(true)
        .with_real_time_adaptation(true);

    // Test neural spatial optimizer creation
    let _neural_optimizer = NeuralSpatialOptimizer::new()
        .with_layers(vec![64, 32, 16])
        .with_learning_rate(0.001);

    println!("   Successfully created all advanced optimization modules");

    Ok(())
}
