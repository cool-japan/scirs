//! Integration tests for scirs2-series
//!
//! This module contains integration tests that verify cross-module functionality
//! and end-to-end workflows in the time series analysis library.

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_series::{
    // TODO: Fix these imports when modules are implemented
    // anomaly::AnomalyDetector,
    arima_models::ArimaModel,
    // biomedical::ECGAnalysis,
    // causality::GrangerCausalityTest,
    // change_point::PELTDetector,
    clustering::TimeSeriesClusterer,
    // correlation::CrossCorrelation,
    // decomposition::stl::STLDecomposer,
    // detection::pattern::PatternDetector,
    // dimensionality_reduction::FunctionalPCA,
    distributed::{ClusterConfig, DistributedProcessor, DistributedTask, TaskPriority, TaskType},
    // environmental::EnvironmentalSensorAnalysis,
    // feature_selection::filter::FilterSelector,
    // features::statistical::StatisticalFeatures,
    // financial::{bollinger_bands, garch_model, BollingerBandsConfig, MovingAverageType},
    // forecasting::neural::NeuralForecaster,
    // gpu_acceleration::GpuTimeSeriesProcessor,
    // iot_sensors::EnvironmentalSensorAnalysis as IoTEnvironmental,
    // neural_forecasting::LSTMForecaster,
    // optimization::OptimizationConfig,
    // out_of_core::{ChunkedProcessor, ProcessingConfig},
    // regression::TimeSeriesRegression,
    // sarima_models::SARIMAModel,
    // state_space::KalmanFilter,
    streaming::StreamingAnalyzer,
    // transformations::BoxCoxTransform,
    // trends::robust::RobustTrendFilter,
    utils::*,
    // validation::CrossValidator,
    // var_models::VectorAutoregression,
    // visualization::TimeSeriesPlot,
};
use statrs::statistics::Statistics;

/// Generate synthetic time series with known properties for testing
#[allow(dead_code)]
fn generate_test_series(
    length: usize,
    trend: f64,
    seasonal_period: usize,
    noise_std: f64,
) -> Array1<f64> {
    let mut series = Array1::zeros(length);
    let mut rng_state = 42u64; // Deterministic for reproducible tests

    for i in 0..length {
        // Generate deterministic random noise
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = ((rng_state % 20000) as f64 / 20000.0 - 0.5) * noise_std;

        let trend_component = (i as f64) * trend;
        let seasonal_component = (2.0 * std::f64::consts::PI * (i % seasonal_period) as f64
            / seasonal_period as f64)
            .sin()
            * 5.0;

        series[i] = 50.0 + trend_component + seasonal_component + noise;
    }

    series
}

/// Generate time series with known change points
#[allow(dead_code)]
fn generate_change_point_series(_length: usize, changepoints: &[usize]) -> Array1<f64> {
    let mut series = Array1::zeros(_length);
    let mut current_mean = 10.0;
    let mut next_change_idx = 0;

    for i in 0.._length {
        // Check if we've hit a change point
        if next_change_idx < changepoints.len() && i == changepoints[next_change_idx] {
            current_mean += 20.0; // Significant shift
            next_change_idx += 1;
        }

        // Add some noise
        let noise = ((i % 17) as f64 - 8.0) * 0.5; // Deterministic "noise"
        series[i] = current_mean + noise;
    }

    series
}

// TODO: Re-enable when PatternDetector and STLDecomposer are implemented
/*
#[test]
#[allow(dead_code)]
fn test_end_to_end_forecasting_pipeline() {
    // Create synthetic data with trend and seasonality
    let data = generate_test_series(500, 0.01, 12, 1.0);

    // 1. Detect pattern and seasonality
    let pattern_detector = PatternDetector::new();
    let period = pattern_detector.detect_period(&data).unwrap();
    assert!(period >= 10 && period <= 15); // Should detect ~12

    // 2. Decompose the series
    let decomposer = STLDecomposer::new(period, 7, 7, 1, false).unwrap();
    let decomposition = decomposer.decompose(&data).unwrap();

    // Verify decomposition components
    assert_eq!(decomposition.trend.len(), data.len());
    assert_eq!(decomposition.seasonal.len(), data.len());
    assert_eq!(decomposition.remainder.len(), data.len());

    // 3. Fit SARIMA model to the data
    let mut sarima = SARIMAModel::new(1, 1, 1, 1, 1, 1, period).unwrap();
    sarima.fit(&data).unwrap();

    // 4. Generate forecasts
    let horizon = 24;
    let forecast = sarima.forecast(horizon).unwrap();

    assert_eq!(forecast.len(), horizon);
    // Forecasts should be reasonable (not too far from last observed values)
    let last_value = data[data.len() - 1];
    for &pred in forecast.iter() {
        assert!((pred - last_value).abs() < 50.0);
    }
}
*/

// TODO: Re-enable when AnomalyDetector and PELTDetector are implemented
/*
#[test]
#[allow(dead_code)]
fn test_anomaly_detection_and_change_point_integration() {
    // Create data with anomalies and change points
    let mut data = generate_test_series(200, 0.0, 10, 1.0);

    // Insert anomalies at known positions
    data[50] = 100.0; // Outlier
    data[150] = -50.0; // Another outlier

    // Create change point at position 100
    for i in 100..200 {
        data[i] += 15.0; // Level shift
    }

    // 1. Detect anomalies
    let anomaly_detector = AnomalyDetector::new()
        .with_method(scirs2_series::anomaly::AnomalyMethod::ZScore)
        .with_threshold(3.0);
    let anomalies = anomaly_detector.detect(&data).unwrap();

    // Should detect the outliers we inserted
    assert!(anomalies[50] > 0.5); // Anomaly at position 50
    assert!(anomalies[150] > 0.5); // Anomaly at position 150

    // 2. Detect change points
    let change_detector = PELTDetector::new(10.0);
    let change_points = change_detector.detect(&data).unwrap();

    // Should detect change point around position 100
    assert!(change_points.iter().any(|&cp| (cp as i32 - 100).abs() < 10));
}

#[test]
#[allow(dead_code)]
fn test_feature_extraction_and_classification_pipeline() {
    // Generate multiple time series with different characteristics
    let series1 = generate_test_series(300, 0.02, 12, 0.5); // High trend, low noise
    let series2 = generate_test_series(300, 0.0, 24, 2.0); // No trend, high noise
    let series3 = generate_test_series(300, -0.01, 6, 1.0); // Negative trend, different seasonality

    // Stack into matrix for clustering
    let data_matrix =
        scirs2_core::ndarray::stack(Axis(0), &[series1.view(), series2.view(), series3.view()]).unwrap();

    // 1. Extract statistical features
    let feature_extractor = StatisticalFeatures::new();
    let features1 = feature_extractor.extract(&series1).unwrap();
    let features2 = feature_extractor.extract(&series2).unwrap();
    let features3 = feature_extractor.extract(&series3).unwrap();

    // Features should be different due to different series characteristics
    assert!((features1[0] - features2[0]).abs() > 0.1); // Different means
    assert!((features1[1] - features3[1]).abs() > 0.1); // Different std devs

    // 2. Perform clustering
    let clusterer = TimeSeriesClusterer::new();
    let config = scirs2_series::clustering::KMeansConfig {
        n_clusters: 3,
        distance: scirs2_series::clustering::TimeSeriesDistance::Euclidean,
        ..Default::default()
    };
    let clustering_result = clusterer.kmeans_clustering(&data_matrix, &config).unwrap();
    let cluster_assignments = clustering_result.cluster_labels;

    // Should assign different clusters to different series
    assert_eq!(cluster_assignments.len(), 3);
    // All assignments should be different (since series are quite different)
    assert!(
        cluster_assignments[0] != cluster_assignments[1]
            || cluster_assignments[1] != cluster_assignments[2]
            || cluster_assignments[0] != cluster_assignments[2]
    );
}

#[test]
#[allow(dead_code)]
fn test_causality_and_correlation_analysis() {
    let length = 300;

    // Create two series where X causes Y
    let mut x_series = generate_test_series(length, 0.0, 20, 1.0);
    let mut y_series = Array1::zeros(length);

    // Y depends on lagged X plus noise
    for i in 2..length {
        y_series[i] = 0.7 * x_series[i - 1] + 0.3 * x_series[i - 2] + ((i % 13) as f64 - 6.0) * 0.5;
        // Deterministic noise
    }

    // 1. Test Granger causality
    let granger_test = GrangerCausalityTest::new(3);
    let causality_result = granger_test.test(&x_series, &y_series).unwrap();

    // X should Granger-cause Y
    assert!(causality_result.f_statistic > 1.0);
    assert!(causality_result.p_value < 0.1); // Significant causality

    // 2. Test cross-correlation
    let correlator = CrossCorrelation::new();
    let cross_corr = correlator
        .cross_correlate(&x_series, &y_series, 10)
        .unwrap();

    // Should show strong correlation at lag 1 and 2
    assert!(cross_corr[11].abs() > 0.3); // Lag 1 (index 10 + 1)
    assert!(cross_corr[12].abs() > 0.2); // Lag 2 (index 10 + 2)
}

#[test]
#[allow(dead_code)]
fn test_distributed_processing_workflow() {
    // Large dataset for distributed processing
    let large_data = generate_test_series(50000, 0.01, 12, 1.0);

    let config = ClusterConfig::default();
    let mut processor = DistributedProcessor::new(config);

    // 1. Distributed forecasting
    let forecast = processor
        .distributed_forecast(&large_data, 20, "linear")
        .unwrap();
    assert_eq!(forecast.len(), 20);

    // 2. Distributed feature extraction
    let features = vec!["mean".to_string(), "std".to_string(), "trend".to_string()];
    let feature_matrix = processor
        .distributed_feature_extraction(&large_data, &features)
        .unwrap();
    assert!(feature_matrix.nrows() > 0);
    assert_eq!(feature_matrix.ncols(), features.len());

    // 3. Check cluster status
    let status = processor.get_cluster_status();
    assert_eq!(status.total_nodes, 1);
    assert!(status.total_completed_tasks > 0);
}

#[test]
#[allow(dead_code)]
fn test_streaming_analysis_pipeline() {
    let data = generate_test_series(1000, 0.005, 10, 1.0);

    // Initialize streaming analyzer
    let config = scirs2_series::streaming::StreamConfig {
        window_size: 50,
        ..Default::default()
    };
    let mut streaming_analyzer = StreamingAnalyzer::new(config).unwrap();

    let mut anomaly_count = 0;
    let mut trend_changes = 0;
    let mut last_trend = 0.0;

    // Process data in streaming fashion
    for (i, &value) in data.iter().enumerate() {
        streaming_analyzer.add_observation(value).unwrap();
        let stats = streaming_analyzer.get_stats();

        if i >= 50 {
            // Wait for window to be full
            // Check for anomalies (simple threshold)
            if (value - stats.mean()).abs() > 3.0 * stats.std_dev() {
                anomaly_count += 1;
            }

            // Check for trend changes (simplified - use mean as trend proxy)
            let current_trend = stats.mean();
            if (current_trend - last_trend).abs() > 0.01 {
                trend_changes += 1;
            }
            last_trend = current_trend;
        }
    }

    // Should detect some patterns in the generated data
    assert!(anomaly_count < 50); // Not too many anomalies
    assert!(trend_changes > 0); // Some trend variation
}

#[test]
#[allow(dead_code)]
fn test_financial_analysis_integration() {
    // Generate financial return series
    let returns = generate_test_series(500, 0.0, 1, 0.02); // Daily returns with volatility
    let prices =
        Array1::from_iter((0..500).map(|i| 100.0 + returns.slice(scirs2_core::ndarray::s![..=i]).sum()));

    // 1. Bollinger Bands analysis
    let bb_config = BollingerBandsConfig {
        period: 20,
        std_dev_multiplier: 2.0,
        ma_type: MovingAverageType::Simple,
    };
    let bb_result = bollinger_bands(&prices, &bb_config).unwrap();

    assert_eq!(bb_result.upper_band.len(), prices.len() - 20 + 1);
    assert_eq!(bb_result.lower_band.len(), prices.len() - 20 + 1);

    // Upper band should be above lower band
    for i in 0..bb_result.upper_band.len() {
        assert!(bb_result.upper_band[i] > bb_result.lower_band[i]);
    }

    // 2. GARCH volatility modeling
    let garch_result = garch_model(&returns, 1, 1).unwrap();

    assert_eq!(garch_result.conditional_variance.len(), returns.len());
    // All variances should be positive
    assert!(garch_result.conditional_variance.iter().all(|&v| v > 0.0));
}

#[test]
#[allow(dead_code)]
fn test_biomedical_signal_processing() {
    // Generate ECG-like signal
    let fs = 1000.0; // 1000 Hz sampling rate
    let duration = 10.0; // 10 seconds
    let length = (fs * duration) as usize;
    let mut ecg_signal = Array1::zeros(length);

    // Simulate ECG with regular heartbeats
    let heart_rate = 72.0; // 72 BPM
    let beat_interval = fs * 60.0 / heart_rate;

    for i in 0..length {
        let t = i as f64 / fs;

        // Base cardiac rhythm
        let cardiac_component = (2.0 * std::f64::consts::PI * heart_rate * t / 60.0).sin() * 0.5;

        // Add QRS complexes (simplified)
        let beat_phase = (i as f64) % beat_interval;
        let qrs_component = if beat_phase < 50.0 {
            (beat_phase / 50.0 * std::f64::consts::PI).sin() * 2.0
        } else {
            0.0
        };

        // Small amount of noise
        let noise = ((i % 17) as f64 - 8.0) * 0.01;

        ecg_signal[i] = cardiac_component + qrs_component + noise;
    }

    // Analyze ECG
    let mut ecg_analysis = ECGAnalysis::new(ecg_signal.clone(), fs).unwrap();
    let r_peaks = ecg_analysis.detect_r_peaks().unwrap();
    let r_peaks_len = r_peaks.len(); // Store length to avoid borrow conflicts

    let hrv = ecg_analysis
        .heart_rate_variability(scirs2_series::biomedical::HRVMethod::TimeDomain)
        .unwrap();

    // Should detect reasonable number of R-peaks
    let expected_beats = (duration * heart_rate / 60.0) as usize;
    assert!(r_peaks_len >= expected_beats - 5 && r_peaks_len <= expected_beats + 5);

    // HRV analysis should return reasonable values
    assert!(hrv.get("sdnn").unwrap_or(&0.0) > &0.0);
    assert!(hrv.get("rmssd").unwrap_or(&0.0) >= &0.0);
    let pnn50 = hrv.get("pnn50").unwrap_or(&0.0);
    assert!(pnn50 >= &0.0 && pnn50 <= &100.0);
}

#[test]
#[allow(dead_code)]
fn test_iot_environmental_monitoring() {
    let length = 1000;

    // Generate environmental sensor data
    let timestamps = Array1::from_iter((0..length).map(|i| i as i64));
    let temperature = Array1::from_iter((0..length).map(|i| {
        20.0 + (i as f64 * 2.0 * std::f64::consts::PI / 100.0).sin() * 5.0 + // Daily cycle
               ((i % 7) as f64 - 3.0) * 0.5 // Noise
    }));
    let humidity = Array1::from_iter((0..length).map(|i| {
        50.0 + (i as f64 * 2.0 * std::f64::consts::PI / 150.0).cos() * 10.0 + // Different cycle
               ((i % 11) as f64 - 5.0) * 0.8 // Noise
    }));

    // Create environmental analysis
    let analysis = EnvironmentalSensorAnalysis::new(timestamps, 1.0)
        .unwrap()
        .with_temperature(temperature.clone())
        .unwrap()
        .with_humidity(humidity.clone())
        .unwrap();

    // 1. Comfort index calculation
    let comfort = analysis.comfort_index().unwrap();
    assert_eq!(comfort.len(), length);
    assert!(comfort.iter().all(|&c| c >= 0.0 && c <= 100.0));

    // 2. Sensor malfunction detection
    let malfunctions = analysis.detect_sensor_malfunctions().unwrap();
    assert!(malfunctions.contains_key("Temperature"));
    assert!(malfunctions.contains_key("Humidity"));

    // 3. Environmental stress analysis
    let stress_index = analysis.environmental_stress_index().unwrap();
    assert_eq!(stress_index.len(), length);
    assert!(stress_index.iter().all(|&s| s >= 0.0));
}

#[test]
#[allow(dead_code)]
fn test_out_of_core_processing_integration() {
    // Configure out-of-core processing
    let config = ProcessingConfig::new()
        .with_chunk_size(1000)
        .with_overlap(100)
        .with_parallel_processing(false); // Disable for test simplicity

    let _processor = ChunkedProcessor::new(config);

    // Generate large dataset (simulated)
    let large_data = generate_test_series(10000, 0.01, 50, 1.0);

    // Process in chunks
    let chunk_size = 1000;
    let overlap = 100;
    let mut chunk_results = Vec::new();

    for start in (0..large_data.len()).step_by(chunk_size - overlap) {
        let end = (start + chunk_size).min(large_data.len());
        let chunk = large_data.slice(scirs2_core::ndarray::s![start..end]);

        // Simulate chunk processing (compute mean)
        let chunk_mean = chunk.mean();
        chunk_results.push(chunk_mean);

        if end >= large_data.len() {
            break;
        }
    }

    // Verify chunk processing results
    assert!(chunk_results.len() > 5); // Should have multiple chunks

    // All chunk means should be reasonable
    let overall_mean = large_data.mean();
    for &chunk_mean in &chunk_results {
        assert!((chunk_mean - overall_mean).abs() < 10.0);
    }
}

#[test]
#[allow(dead_code)]
fn test_cross_validation_workflow() {
    let data = generate_test_series(200, 0.01, 12, 1.0);

    // Create cross-validator
    let validator = CrossValidator::new(5, 0.2); // 5-fold CV with 20% holdout
    let splits = validator.time_series_split(&data).unwrap();

    assert_eq!(splits.len(), 5);

    let mut total_mse = 0.0;
    let mut fold_count = 0;

    // Evaluate ARIMA model on each fold
    for split in splits {
        let train_data = data.slice(scirs2_core::ndarray::s![split.train_start..split.train_end]);
        let test_data = data.slice(scirs2_core::ndarray::s![split.test_start..split.test_end]);

        // Fit ARIMA model
        let mut arima = ArimaModel::new(1, 1, 1).unwrap();
        if arima.fit(&train_data.to_owned()).is_ok() {
            // Generate forecasts
            let horizon = test_data.len();
            if let Ok(forecast) = arima.forecast(horizon, &data) {
                // Calculate MSE
                let mse = test_data
                    .iter()
                    .zip(forecast.iter())
                    .map(|(&actual, &pred)| (actual - pred).powi(2))
                    .sum::<f64>()
                    / test_data.len() as f64;

                total_mse += mse;
                fold_count += 1;
            }
        }
    }

    if fold_count > 0 {
        let avg_mse = total_mse / fold_count as f64;
        assert!(avg_mse < 100.0); // MSE should be reasonable
        assert!(avg_mse > 0.0); // And positive
    }
}

#[test]
#[allow(dead_code)]
fn test_dimensionality_reduction_workflow() {
    // Generate multivariate time series
    let n_series = 10;
    let length = 500;
    let mut data = Array2::zeros((length, n_series));

    // Each series has different characteristics but some correlation
    for j in 0..n_series {
        let base_series = generate_test_series(length, 0.01, 20 + j * 5, 1.0);
        for i in 0..length {
            // Add cross-correlation between series
            let correlation_component = if j > 0 { 0.3 * data[[i, j - 1]] } else { 0.0 };
            data[[i, j]] = base_series[i] + correlation_component;
        }
    }

    // Apply functional PCA
    let fpca = FunctionalPCA::new(3); // Reduce to 3 components
    let reduced_data = fpca.fit_transform(&data).unwrap();

    assert_eq!(reduced_data.ncols(), 3);
    assert_eq!(reduced_data.nrows(), length);

    // Verify dimensionality reduction preserves some structure
    // (The first few components should capture most variance)
    let original_var = data.var_axis(Axis(0), 0.0);
    let reduced_var = reduced_data.var_axis(Axis(0), 0.0);

    // Reduced data should still have reasonable variance
    assert!(reduced_var.sum() > 0.0);
}

#[test]
#[allow(dead_code)]
fn test_trend_analysis_integration() {
    // Generate data with multiple trend regimes
    let mut data = Array1::zeros(500);

    // First regime: increasing trend
    for i in 0..200 {
        data[i] = 10.0 + (i as f64) * 0.05 + ((i % 7) as f64 - 3.0) * 0.5;
    }

    // Second regime: flat trend
    for i in 200..350 {
        data[i] = data[199] + ((i % 9) as f64 - 4.0) * 0.3;
    }

    // Third regime: decreasing trend
    for i in 350..500 {
        data[i] = data[349] - ((i - 350) as f64) * 0.03 + ((i % 11) as f64 - 5.0) * 0.4;
    }

    // Apply robust trend filtering
    let trend_filter = RobustTrendFilter::new(0.1);
    let trend = trend_filter.filter(&data).unwrap();

    assert_eq!(trend.len(), data.len());

    // Trend should be smoother than original data
    let data_diff_var = data
        .windows(2)
        .into_iter()
        .map(|w| (w[1] - w[0]).powi(2))
        .sum::<f64>()
        / (data.len() - 1) as f64;
    let trend_diff_var = trend
        .windows(2)
        .into_iter()
        .map(|w| (w[1] - w[0]).powi(2))
        .sum::<f64>()
        / (trend.len() - 1) as f64;

    assert!(trend_diff_var < data_diff_var); // Trend should be smoother
}

#[test]
#[allow(dead_code)]
fn test_comprehensive_workflow() {
    // This test combines multiple modules in a realistic analysis workflow

    // 1. Generate complex synthetic data
    let base_data = generate_test_series(1000, 0.02, 24, 2.0);

    // 2. Detect and handle outliers
    let anomaly_detector = AnomalyDetector::new()
        .with_method(scirs2_series::anomaly::AnomalyMethod::InterquartileRange);
    let anomalies = anomaly_detector.detect(&base_data).unwrap();

    // Create cleaned data by replacing outliers with interpolated values
    let mut cleaned_data = base_data.clone();
    for (i, &is_anomaly) in anomalies.iter().enumerate() {
        if is_anomaly > 0.5 && i > 0 && i < cleaned_data.len() - 1 {
            cleaned_data[i] = (cleaned_data[i - 1] + cleaned_data[i + 1]) / 2.0;
        }
    }

    // 3. Decompose the cleaned series
    let decomposer = STLDecomposer::new(24, 7, 7, 1, false).unwrap();
    let decomposition = decomposer.decompose(&cleaned_data).unwrap();

    // 4. Analyze trend component for change points
    let change_detector = PELTDetector::new(5.0);
    let change_points = change_detector.detect(&decomposition.trend).unwrap();

    // 5. Extract features from each segment between change points
    let mut segments = Vec::new();
    let mut start = 0;
    for &cp in &change_points {
        if cp > start + 50 {
            // Minimum segment length
            segments.push((start, cp));
            start = cp;
        }
    }
    if start < cleaned_data.len() - 50 {
        segments.push((start, cleaned_data.len()));
    }

    let feature_extractor = StatisticalFeatures::new();
    let mut segment_features = Vec::new();

    for &(start, end) in &segments {
        let segment = cleaned_data.slice(scirs2_core::ndarray::s![start..end]);
        let features = feature_extractor.extract(&segment.to_owned()).unwrap();
        segment_features.push(features);
    }

    // 6. Fit forecasting models and generate predictions
    let mut arima = ArimaModel::new(2, 1, 2).unwrap();
    arima.fit(&cleaned_data).unwrap();
    let arima_forecast = arima.forecast(50, &cleaned_data).unwrap();

    // 7. Validate results
    assert!(segments.len() > 0);
    assert!(segment_features.len() == segments.len());
    assert_eq!(arima_forecast.len(), 50);

    // The forecast should be reasonable given the trend
    let last_values_mean = cleaned_data.slice(scirs2_core::ndarray::s![-50..]).mean();
    let forecast_mean = arima_forecast.mean();
    assert!((forecast_mean - last_values_mean).abs() < 20.0);

    println!("Comprehensive workflow test completed successfully!");
    println!(
        "Detected {} anomalies",
        anomalies.iter().filter(|&&x| x > 0.5).count()
    );
    println!("Found {} change points", change_points.len());
    println!("Analyzed {} segments", segments.len());
    println!("Generated forecast with mean: {:.2}", forecast_mean);
}
*/

// Placeholder test to ensure the test file compiles
#[test]
fn test_basic_compilation() {
    // This test ensures the file compiles with available imports
    use scirs2_series::streaming::StreamConfig;
    let _config = StreamConfig::default();
}
