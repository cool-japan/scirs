//! GPU acceleration and advanced ML integration demonstration
//!
//! This example shows how to use GPU acceleration for large-scale time series processing
//! and demonstrates advanced machine learning integration capabilities.

use scirs2_core::ndarray::Array1;
use scirs2_series::{
    forecasting::neural::{LSTMConfig, LSTMForecaster, NeuralForecaster},
    gpu_acceleration::{
        algorithms::{
            FeatureConfig, ForecastMethod, GpuFeatureExtractor, GpuTimeSeriesProcessor,
            WindowStatistic,
        },
        utils, GpuAccelerated, GpuArray, GpuConfig, GpuDeviceManager, GraphOptimizationLevel,
        MemoryStrategy,
    },
};
use statrs::statistics::Statistics;
use std::time::Instant;

#[allow(dead_code)]
fn main() {
    println!("=== GPU Acceleration and Advanced ML Demo ===\n");

    // Check GPU availability
    println!("1. GPU System Detection");
    check_gpu_capabilities();

    // Demo GPU device management
    println!("\n2. GPU Device Management");
    gpu_device_management_demo();

    // Demo GPU-accelerated batch processing
    println!("\n3. GPU-accelerated Batch Processing");
    gpu_batch_processing_demo();

    // Demo GPU feature extraction
    println!("\n4. GPU Feature Extraction");
    gpu_feature_extraction_demo();

    // Demo GPU memory optimization
    println!("\n5. GPU Memory Optimization");
    gpu_memory_optimization_demo();

    // Demo Advanced ML integration
    println!("\n6. Advanced ML Integration");
    advanced_ml_demo();

    // Performance comparison
    println!("\n7. Performance Comparison: CPU vs GPU");
    performance_comparison();

    println!("\n=== GPU and ML Demo Complete ===");
}

#[allow(dead_code)]
fn check_gpu_capabilities() {
    println!("  Checking GPU acceleration support...");

    if utils::is_gpu_supported() {
        println!("  ✓ GPU acceleration is available");
    } else {
        println!("  ⚠ GPU acceleration not available - using CPU fallback");
    }

    // Check system memory
    let recommended_batch = utils::get_recommended_batch_size(100_000, 1024 * 1024 * 1024);
    println!(
        "  Recommended batch size for 100k series: {}",
        recommended_batch
    );

    let memory_usage = utils::estimate_memory_usage(50_000, 0.5);
    println!(
        "  Estimated memory usage for 50k elements: {:.1} MB",
        memory_usage as f64 / (1024.0 * 1024.0)
    );
}

#[allow(dead_code)]
fn gpu_device_management_demo() {
    match GpuDeviceManager::new() {
        Ok(mut manager) => {
            let devices = manager.get_devices();
            println!("  Found {} compute device(s):", devices.len());

            for (i, device) in devices.iter().enumerate() {
                println!("    Device {}: {:?}", i, device.backend);
                println!(
                    "      Memory: {:.1} GB",
                    device.memory as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                println!("      Multiprocessors: {}", device.multiprocessors);
                println!("      FP16 support: {}", device.supports_fp16);
                println!("      Tensor cores: {}", device.supports_tensor_cores);

                if let Some((major, minor)) = device.compute_capability {
                    println!("      Compute capability: {}.{}", major, minor);
                }
            }

            // Try to set a device
            if !devices.is_empty() && manager.set_device(0).is_ok() {
                println!("  Successfully set device 0 as active");

                if let Some(caps) = manager.current_device_capabilities() {
                    println!(
                        "  Active device supports GPU: {}",
                        !matches!(
                            caps.backend,
                            scirs2_series::gpu_acceleration::GpuBackend::CpuFallback
                        )
                    );
                }
            }
        }
        Err(e) => {
            println!("  Failed to initialize device manager: {}", e);
        }
    }
}

#[allow(dead_code)]
fn gpu_batch_processing_demo() {
    println!("  Setting up GPU-accelerated batch processing...");

    let config = GpuConfig {
        device_id: 0,
        memory_pool_size: Some(512 * 1024 * 1024), // 512MB
        enable_memory_optimization: true,
        batch_size: 128,
        use_half_precision: false,
        enable_async: true,
        tensor_cores: Default::default(),
        memory_strategy: MemoryStrategy::OnDemand,
        dynamic_batching: true,
        graph_optimization: GraphOptimizationLevel::Extended,
    };

    match GpuTimeSeriesProcessor::<f64>::new(config.clone()) {
        Ok(processor) => {
            // Generate batch of time series
            let batch_size = 50;
            let series_length = 100;
            let mut series_batch = Vec::new();

            for i in 0..batch_size {
                let mut series = Array1::zeros(series_length);
                for j in 0..series_length {
                    let t = j as f64;
                    let trend = 0.01 * i as f64 * t;
                    let seasonal = 5.0 * (2.0 * std::f64::consts::PI * t / 12.0).sin();
                    let noise = (i + j) as f64 * 0.001; // Deterministic "noise"
                    series[j] = 100.0 + trend + seasonal + noise;
                }
                series_batch.push(series);
            }

            println!(
                "  Generated {} time series of length {}",
                batch_size, series_length
            );

            // Test different forecasting methods
            let forecast_methods = vec![
                (
                    "Exponential Smoothing",
                    ForecastMethod::ExponentialSmoothing { alpha: 0.3 },
                ),
                ("Linear Trend", ForecastMethod::LinearTrend),
                (
                    "Moving Average",
                    ForecastMethod::MovingAverage { window: 10 },
                ),
                (
                    "Autoregressive",
                    ForecastMethod::AutoRegressive { order: 3 },
                ),
            ];

            for (name, method) in forecast_methods {
                let start_time = Instant::now();

                match processor.batch_forecast(&series_batch, 5, method) {
                    Ok(forecasts) => {
                        let duration = start_time.elapsed();
                        println!(
                            "    {}: {} forecasts in {:.2}ms",
                            name,
                            forecasts.len(),
                            duration.as_millis()
                        );

                        // Validate forecast shapes
                        if !forecasts.is_empty() {
                            println!("      Sample forecast length: {}", forecasts[0].len());
                            println!(
                                "      Sample forecast values: {:?}",
                                forecasts[0]
                                    .iter()
                                    .take(3)
                                    .map(|x| format!("{:.1}", x))
                                    .collect::<Vec<_>>()
                            );
                        }
                    }
                    Err(e) => {
                        println!("    {}: Failed - {}", name, e);
                    }
                }
            }

            // Test correlation matrix computation
            let start_time = Instant::now();
            match processor.batch_correlation_matrix(&series_batch[..10.min(series_batch.len())]) {
                Ok(corr_matrix) => {
                    let duration = start_time.elapsed();
                    println!(
                        "  Correlation matrix ({}x{}) computed in {:.2}ms",
                        corr_matrix.nrows(),
                        corr_matrix.ncols(),
                        duration.as_millis()
                    );

                    // Show sample correlations
                    if corr_matrix.nrows() >= 3 && corr_matrix.ncols() >= 3 {
                        println!("    Sample correlations:");
                        for i in 0..3 {
                            for j in 0..3 {
                                print!("    {:.3}", corr_matrix[[i, j]]);
                            }
                            println!();
                        }
                    }
                }
                Err(e) => {
                    println!("  Correlation matrix computation failed: {}", e);
                }
            }

            // Test sliding window statistics
            if !series_batch.is_empty() {
                let statistics = vec![
                    WindowStatistic::Mean,
                    WindowStatistic::Variance,
                    WindowStatistic::Min,
                    WindowStatistic::Max,
                    WindowStatistic::Range,
                ];

                let start_time = Instant::now();
                match processor.sliding_window_statistics(&series_batch[0], 10, &statistics) {
                    Ok(window_stats) => {
                        let duration = start_time.elapsed();
                        println!(
                            "  Sliding window statistics computed in {:.2}ms",
                            duration.as_millis()
                        );
                        println!("    Number of statistics: {}", window_stats.len());
                        println!("    Window count: {}", window_stats[0].len());
                    }
                    Err(e) => {
                        println!("  Sliding window statistics failed: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("  Failed to create GPU processor: {}", e);
        }
    }
}

#[allow(dead_code)]
fn gpu_feature_extraction_demo() {
    println!("  GPU-accelerated feature extraction...");

    let config = GpuConfig::default();
    let feature_config = FeatureConfig {
        extract_statistical: true,
        extract_frequency: true,
        extract_complexity: true,
        window_sizes: vec![5, 10, 20],
    };

    match GpuFeatureExtractor::<f64>::new(config, feature_config) {
        Ok(extractor) => {
            // Generate diverse time series for feature extraction
            let mut series_batch = Vec::new();
            let patterns = [
                "linear",
                "sinusoidal",
                "exponential",
                "random_walk",
                "seasonal",
            ];

            for (i, &pattern) in patterns.iter().enumerate() {
                let series = generate_pattern_series(pattern, 100, i as f64);
                series_batch.push(series);
            }

            println!(
                "  Generated {} series with different patterns",
                series_batch.len()
            );

            let start_time = Instant::now();
            match extractor.batch_extract_features(&series_batch) {
                Ok(feature_matrix) => {
                    let duration = start_time.elapsed();
                    println!(
                        "  Feature extraction completed in {:.2}ms",
                        duration.as_millis()
                    );
                    println!(
                        "    Feature matrix shape: {}x{}",
                        feature_matrix.nrows(),
                        feature_matrix.ncols()
                    );

                    // Analyze extracted features
                    for (i, pattern) in patterns.iter().enumerate() {
                        if i < feature_matrix.nrows() {
                            let features = feature_matrix.row(i);
                            println!(
                                "    {} pattern features (first 5): {:?}",
                                pattern,
                                features
                                    .iter()
                                    .take(5)
                                    .map(|x| format!("{:.2}", x))
                                    .collect::<Vec<_>>()
                            );
                        }
                    }

                    // Compute feature statistics
                    for j in 0..feature_matrix.ncols().min(5) {
                        let column = feature_matrix.column(j);
                        let mean = column.mean();
                        let std = column.variance().sqrt();
                        println!("    Feature {}: mean={:.3}, std={:.3}", j, mean, std);
                    }
                }
                Err(e) => {
                    println!("  Feature extraction failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("  Failed to create feature extractor: {}", e);
        }
    }
}

#[allow(dead_code)]
fn generate_pattern_series(pattern: &str, length: usize, seed: f64) -> Array1<f64> {
    let mut series = Array1::zeros(length);

    for i in 0..length {
        let t = i as f64;
        let value = match pattern {
            "linear" => 10.0 + 0.5 * t + seed,
            "sinusoidal" => 50.0 + 20.0 * (0.2 * t + seed).sin(),
            "exponential" => 10.0 * (0.02 * t).exp() + seed,
            "random_walk" => {
                let prev = if i > 0 { series[i - 1] } else { 50.0 + seed };
                prev + (t * 0.1 + seed).sin() * 2.0
            }
            "seasonal" => {
                let trend = 0.1 * t;
                let seasonal = 15.0 * (2.0 * std::f64::consts::PI * t / 24.0 + seed).sin();
                let noise = (t + seed).cos();
                100.0 + trend + seasonal + noise
            }
            _ => 50.0 + seed + (t * 0.1).sin(),
        };
        series[i] = value;
    }

    series
}

#[allow(dead_code)]
fn gpu_memory_optimization_demo() {
    println!("  GPU memory optimization strategies...");

    // Test different configurations
    let configs = vec![
        (
            "Standard",
            GpuConfig {
                batch_size: 64,
                use_half_precision: false,
                enable_memory_optimization: false,
                ..Default::default()
            },
        ),
        (
            "Optimized",
            GpuConfig {
                batch_size: 128,
                use_half_precision: true,
                enable_memory_optimization: true,
                memory_pool_size: Some(256 * 1024 * 1024),
                ..Default::default()
            },
        ),
        (
            "Large Batch",
            GpuConfig {
                batch_size: 256,
                use_half_precision: true,
                enable_memory_optimization: true,
                memory_pool_size: Some(512 * 1024 * 1024),
                ..Default::default()
            },
        ),
    ];

    for (name, config) in configs {
        println!("  Testing {} configuration:", name);

        // Test GPU array operations
        let data = Array1::linspace(0.0, 100.0, 1000);
        let gpu_array = GpuArray::from_cpu(data.clone(), config.clone());

        println!("    Original array length: {}", gpu_array.len());
        println!(
            "    GPU memory usage: {} bytes",
            gpu_array.gpu_memory_usage()
        );

        // Test GPU transfer
        match gpu_array.to_gpu(&config) {
            Ok(gpu_transferred) => {
                println!("    GPU transfer successful");
                println!("    On GPU: {}", gpu_transferred.is_on_gpu());
                println!(
                    "    GPU memory usage: {} bytes",
                    gpu_transferred.gpu_memory_usage()
                );

                // Test CPU transfer back
                match gpu_transferred.to_cpu() {
                    Ok(cpu_transferred) => {
                        println!("    CPU transfer back successful");
                        println!("    On GPU: {}", cpu_transferred.is_on_gpu());
                    }
                    Err(e) => {
                        println!("    CPU transfer failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("    GPU transfer failed: {}", e);
            }
        }

        // Memory optimization recommendations
        let data_size = 10_000;
        let memory_limit = 1024 * 1024 * 1024; // 1GB
        let optimized_config = utils::optimize_gpu_config(data_size, memory_limit);

        println!("    Recommended optimizations for {} elements:", data_size);
        println!("      Batch size: {}", optimized_config.batch_size);
        println!(
            "      Half precision: {}",
            optimized_config.use_half_precision
        );
        println!("      Memory pool: {:?}", optimized_config.memory_pool_size);
    }
}

#[allow(dead_code)]
fn advanced_ml_demo() {
    println!("  Advanced ML integration with neural networks...");

    // Generate training data with complex patterns
    let mut training_data = Vec::new();

    for i in 0..500 {
        let t = i as f64;
        let trend = 0.02 * t;
        let seasonal1 = 15.0 * (2.0 * std::f64::consts::PI * t / 50.0).sin();
        let seasonal2 = 8.0 * (2.0 * std::f64::consts::PI * t / 17.0).sin();
        let noise = 2.0 * (t * 0.01).sin() * (t * 0.03).cos();

        let value = 100.0 + trend + seasonal1 + seasonal2 + noise;
        training_data.push(value);
    }

    let data_array = Array1::from_vec(training_data);
    println!(
        "  Generated {} training samples with complex patterns",
        data_array.len()
    );

    // Test different neural network configurations
    let configs = vec![
        (
            "Small LSTM",
            LSTMConfig {
                base: scirs2_series::forecasting::neural::NeuralConfig {
                    lookback_window: 12,
                    forecast_horizon: 5,
                    epochs: 20,
                    learning_rate: 0.01,
                    batch_size: 16,
                    validation_split: 0.2,
                    early_stopping_patience: Some(5),
                    random_seed: Some(42),
                },
                num_layers: 1,
                hidden_size: 16,
                dropout: 0.1,
                bidirectional: false,
            },
        ),
        (
            "Medium LSTM",
            LSTMConfig {
                base: scirs2_series::forecasting::neural::NeuralConfig {
                    lookback_window: 24,
                    forecast_horizon: 10,
                    epochs: 30,
                    learning_rate: 0.005,
                    batch_size: 32,
                    validation_split: 0.2,
                    early_stopping_patience: Some(8),
                    random_seed: Some(42),
                },
                num_layers: 2,
                hidden_size: 32,
                dropout: 0.2,
                bidirectional: false,
            },
        ),
    ];

    for (name, config) in configs {
        println!("  Training {} model...", name);

        let mut lstm_forecaster = LSTMForecaster::new(config);
        let start_time = Instant::now();

        match lstm_forecaster.fit(&data_array) {
            Ok(_) => {
                let training_duration = start_time.elapsed();
                println!(
                    "    Training completed in {:.2}s",
                    training_duration.as_secs_f64()
                );

                // Test forecasting
                let forecast_start = Instant::now();
                match lstm_forecaster.predict(10) {
                    Ok(forecast_result) => {
                        let forecast_duration = forecast_start.elapsed();
                        println!(
                            "    Forecasting completed in {:.2}ms",
                            forecast_duration.as_millis()
                        );
                        println!("    Forecast length: {}", forecast_result.forecast.len());

                        // Show sample forecast
                        let sample_forecast: Vec<String> = forecast_result
                            .forecast
                            .iter()
                            .take(5)
                            .map(|x| format!("{:.1}", x))
                            .collect();
                        println!("    Sample forecast: {:?}", sample_forecast);

                        // Test uncertainty quantification
                        match lstm_forecaster.predict_with_uncertainty(5, 0.95) {
                            Ok(uncertainty_result) => {
                                println!("    Uncertainty quantification successful");
                                let (lower, upper) =
                                    (&uncertainty_result.lower_ci, &uncertainty_result.upper_ci);
                                {
                                    println!(
                                        "    Confidence intervals available: {} points",
                                        lower.len()
                                    );
                                }
                            }
                            Err(e) => {
                                println!("    Uncertainty quantification failed: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("    Forecasting failed: {}", e);
                    }
                }

                // Show training loss history
                if let Some(loss_history) = lstm_forecaster.get_loss_history() {
                    if !loss_history.is_empty() {
                        println!(
                            "    Training loss: initial={:.4}, final={:.4}",
                            loss_history.first().unwrap_or(&0.0),
                            loss_history.last().unwrap_or(&0.0)
                        );
                    }
                }
            }
            Err(e) => {
                println!("    Training failed: {}", e);
            }
        }
    }
}

#[allow(dead_code)]
fn performance_comparison() {
    println!("  Comparing CPU vs GPU performance...");

    let series_counts = vec![10, 50, 100];
    let series_length = 200;

    for &count in &series_counts {
        println!(
            "  Testing with {} series of length {}:",
            count, series_length
        );

        // Generate test data
        let mut series_batch = Vec::new();
        for i in 0..count {
            let mut series = Array1::zeros(series_length);
            for j in 0..series_length {
                let t = j as f64;
                series[j] = 50.0 + 0.1 * i as f64 * t + 5.0 * (t * 0.1).sin();
            }
            series_batch.push(series);
        }

        // CPU processing (simulation)
        let cpu_start = Instant::now();
        let mut cpu_results = Vec::new();
        for series in &series_batch {
            // Simulate CPU forecasting
            let forecast_steps = 5;
            let mut forecast = Array1::zeros(forecast_steps);
            let last_value = series[series.len() - 1];
            forecast.fill(last_value); // Simple constant forecast
            cpu_results.push(forecast);
        }
        let cpu_duration = cpu_start.elapsed();

        // GPU processing
        let gpu_config = GpuConfig {
            batch_size: count.max(32),
            use_half_precision: false,
            enable_memory_optimization: true,
            ..Default::default()
        };

        let gpu_start = Instant::now();
        match GpuTimeSeriesProcessor::<f64>::new(gpu_config) {
            Ok(processor) => {
                match processor.batch_forecast(&series_batch, 5, ForecastMethod::LinearTrend) {
                    Ok(_gpu_results) => {
                        let gpu_duration = gpu_start.elapsed();

                        println!("    CPU time: {:.2}ms", cpu_duration.as_millis());
                        println!("    GPU time: {:.2}ms", gpu_duration.as_millis());

                        let speedup =
                            cpu_duration.as_millis() as f64 / gpu_duration.as_millis() as f64;
                        if speedup > 1.0 {
                            println!("    GPU speedup: {:.1}x", speedup);
                        } else {
                            println!("    CPU faster by: {:.1}x", 1.0 / speedup);
                        }
                    }
                    Err(e) => {
                        println!("    GPU processing failed: {}", e);
                        println!("    CPU time: {:.2}ms", cpu_duration.as_millis());
                    }
                }
            }
            Err(e) => {
                println!("    GPU initialization failed: {}", e);
            }
        }
    }
}
