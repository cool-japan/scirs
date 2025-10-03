//! Demonstration of advanced features in scirs2-transform
//!
//! This example showcases the newly implemented Version 1.0.0 features:
//! - GPU acceleration
//! - Distributed processing  
//! - Automated feature engineering
//! - Production monitoring

use scirs2_core::ndarray::Array2;
use scirs2_transform::{auto_feature_engineering::AutoFeatureEngineer, Result};

#[cfg(feature = "gpu")]
use scirs2_transform::gpu::GpuPCA;

#[cfg(feature = "distributed")]
use scirs2_transform::distributed::{
    DistributedConfig, DistributedPCA, NodeInfo, PartitioningStrategy,
};

#[cfg(feature = "monitoring")]
use scirs2_transform::monitoring::{
    AlertConfig, DriftMethod, PerformanceMetrics, TransformationMonitor,
};

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("🚀 SciRS2 Transform Advanced Features Demo");
    println!("==========================================");

    // Generate sample data
    let data = generate_sample_data(1000, 50)?;
    println!(
        "✅ Generated sample data: {} x {}",
        data.nrows(),
        data.ncols()
    );

    // Demo 1: Automated Feature Engineering
    demo_automated_feature_engineering(&data)?;

    // Demo 2: GPU Acceleration (if available)
    #[cfg(feature = "gpu")]
    demo_gpu_acceleration(&data)?;

    // Demo 3: Distributed Processing (if available)
    #[cfg(feature = "distributed")]
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(demo_distributed_processing(&data))?;

    // Demo 4: Production Monitoring (if available)
    #[cfg(feature = "monitoring")]
    demo_production_monitoring(&data)?;

    println!("\n🎉 All advanced features demonstrated successfully!");
    Ok(())
}

#[allow(dead_code)]
fn demo_automated_feature_engineering(data: &Array2<f64>) -> Result<()> {
    println!("\n📊 Automated Feature Engineering Demo");
    println!("=====================================");

    let auto_engineer = AutoFeatureEngineer::new()?;

    // Extract meta-features from the dataset
    let meta_features = auto_engineer.extract_meta_features(&data.view())?;
    println!("📈 Dataset meta-features:");
    println!("   - Samples: {}", meta_features.n_samples);
    println!("   - Features: {}", meta_features.n_features);
    println!("   - Sparsity: {:.3}", meta_features.sparsity);
    println!(
        "   - Mean correlation: {:.3}",
        meta_features.mean_correlation
    );
    println!("   - Outlier ratio: {:.3}", meta_features.outlier_ratio);

    // Get transformation recommendations
    let recommendations = auto_engineer.recommend_transformations(&data.view())?;
    println!("\n🎯 Recommended transformations:");
    for (i, config) in recommendations.iter().enumerate() {
        println!(
            "   {}. {:?} (score: {:.3})",
            i + 1,
            config.transformation_type,
            config.expected_performance
        );
    }

    Ok(())
}

#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn demo_gpu_acceleration(data: &Array2<f64>) -> Result<()> {
    println!("\n🎮 GPU Acceleration Demo");
    println!("========================");

    match GpuPCA::new(10) {
        Ok(mut gpu_pca) => {
            println!("✅ GPU PCA initialized successfully");

            // Fit and transform would be called here in real usage
            println!("   Ready for GPU-accelerated PCA with 10 components");
            println!(
                "   Features: Fast matrix operations, eigendecomposition, and memory management"
            );
        }
        Err(e) => {
            println!("⚠️  GPU not available: {}", e);
            println!("   Install CUDA toolkit and enable 'gpu' feature for GPU acceleration");
        }
    }

    Ok(())
}

#[cfg(feature = "distributed")]
async fn demo_distributed_processing(data: &Array2<f64>) -> Result<()> {
    println!("\n🌐 Distributed Processing Demo");
    println!("==============================");

    // Create distributed configuration
    let config = DistributedConfig {
        nodes: vec![
            NodeInfo {
                id: "node1".to_string(),
                address: "localhost".to_string(),
                port: 8001,
                memory_gb: 8.0,
                cpu_cores: 4,
                has_gpu: false,
            },
            NodeInfo {
                id: "node2".to_string(),
                address: "localhost".to_string(),
                port: 8002,
                memory_gb: 16.0,
                cpu_cores: 8,
                has_gpu: true,
            },
        ],
        max_concurrent_tasks: 4,
        timeout_seconds: 300,
        partitioning_strategy: PartitioningStrategy::RowWise,
    };

    println!("⚙️  Distributed configuration:");
    println!("   - Nodes: {}", config.nodes.len());
    println!("   - Max concurrent tasks: {}", config.max_concurrent_tasks);
    println!("   - Partitioning: {:?}", config.partitioning_strategy);

    match DistributedPCA::new(10, config).await {
        Ok(_distributed_pca) => {
            println!("✅ Distributed PCA initialized successfully");
            println!("   Ready for multi-node processing");
        }
        Err(e) => {
            println!("⚠️  Distributed processing setup failed: {}", e);
            println!("   Enable 'distributed' feature for multi-node capabilities");
        }
    }

    Ok(())
}

#[cfg(feature = "monitoring")]
#[allow(dead_code)]
fn demo_production_monitoring(data: &Array2<f64>) -> Result<()> {
    println!("\n📊 Production Monitoring Demo");
    println!("=============================");

    let mut monitor = TransformationMonitor::new()?;

    // Set reference _data for drift detection
    let reference_data = data.slice(scirs2_core::ndarray::s![..500, ..]).to_owned();
    monitor.set_reference_data(reference_data, None)?;
    println!("✅ Reference _data set for drift detection");

    // Configure drift detection methods
    monitor.set_drift_method("feature_0", DriftMethod::KolmogorovSmirnov)?;
    monitor.set_drift_method("feature_1", DriftMethod::PopulationStabilityIndex)?;
    println!("⚙️  Drift detection methods configured");

    // Simulate new _data for drift detection
    let new_data = data.slice(scirs2_core::ndarray::s![500.., ..]);
    let drift_results = monitor.detect_drift(&new_data)?;

    println!("📈 Drift detection results:");
    for result in drift_results.iter().take(3) {
        println!(
            "   - {}: {} detected (severity: {:.3})",
            result.feature_name,
            if result.is_drift_detected {
                "DRIFT"
            } else {
                "no drift"
            },
            result.severity
        );
    }

    // Record performance metrics
    let metrics = PerformanceMetrics {
        processing_time_ms: 125.0,
        memory_usage_mb: 256.0,
        error_rate: 0.001,
        throughput: 800.0,
        data_quality_score: 0.95,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };

    let alerts = monitor.record_metrics(metrics)?;
    println!("🔔 Active alerts: {}", alerts.len());

    Ok(())
}

#[allow(dead_code)]
fn generate_sample_data(_n_samples: usize, nfeatures: usize) -> Result<Array2<f64>> {
    use scirs2_core::random::{thread_rng, Distribution, Normal};

    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let data = Array2::from_shape_fn((_n_samples, nfeatures), |_| dist.sample(&mut rng));
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_data_generation() {
        let data = generate_sample_data(100, 10).unwrap();
        assert_eq!(data.dim(), (100, 10));
    }

    #[test]
    fn test_automated_feature_engineering() {
        let data = generate_sample_data(50, 5).unwrap();
        demo_automated_feature_engineering(&data).unwrap();
    }

    #[cfg(feature = "monitoring")]
    #[test]
    fn test_monitoring_demo() {
        let data = generate_sample_data(100, 5).unwrap();
        demo_production_monitoring(&data).unwrap();
    }
}
