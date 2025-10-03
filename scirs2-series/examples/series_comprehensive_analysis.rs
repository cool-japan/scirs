//! Comprehensive time series analysis example
//!
//! This example demonstrates the full workflow of time series analysis using scirs2-series,
//! including data preprocessing, decomposition, forecasting, and anomaly detection.

use scirs2_core::ndarray::{array, Array1};
use scirs2_series::{
    // TODO: Fix imports when modules are implemented
    // anomaly::{AnomalyDetector, IsolationForestDetector, ZScoreDetector},
    // arima_models::{auto_arima, ArimaSelectionOptions, SelectionCriterion},
    // change_point::{CusumDetector, PeltDetector},
    // clustering::{ClusteringConfig, TimeSeriesClusterer},
    // correlation::{CrossCorrelationAnalyzer, DynamicTimeWarping},
    // decomposition::{
    //     stl::{StlDecomposer, StlOptions},
    //     Decomposer,
    // },
    // features::{FeatureConfig, FeatureExtractor},
    // forecasting::{
    //     exponential_smoothing::{ExponentialSmoothingConfig, ExponentialSmoothingForecaster},
    //     neural::{LSTMConfig, LSTMForecaster, NeuralForecaster},
    // },
    streaming::{StreamConfig, StreamingAnalyzer},
    // transformations::{BoxCoxTransformer, DifferencingTransformer, Transformer},
    // validation::{cross_validate, CrossValidationConfig, ValidationMetric},
};
use statrs::statistics::Statistics;

#[allow(dead_code)]
fn main() {
    println!("Comprehensive analysis demo - TODO: Re-enable when imports are fixed");
}

// TODO: Re-enable when imports are fixed
/*
#[allow(dead_code)]
fn main() {
    println!("=== Comprehensive Time Series Analysis Demo ===\n");

    // Generate synthetic data with trend, seasonality, and noise
    let data = generate_synthetic_data();
    println!("Generated {} data points", data.len());

    // Step 1: Data preprocessing and transformations
    println!("\n1. Data Preprocessing");
    let (transformed_data, transformer) = preprocess_data(&data);
    println!("Applied Box-Cox transformation and differencing");

    // Step 2: Time series decomposition
    println!("\n2. Time Series Decomposition");
    let decomposition_result = decompose_series(&data);
    analyze_decomposition(&decomposition_result);

    // Step 3: Feature extraction
    println!("\n3. Feature Extraction");
    extract_and_analyze_features(&data);

    // Step 4: Change point and anomaly detection
    println!("\n4. Change Point and Anomaly Detection");
    detect_changes_and_anomalies(&data);

    // Step 5: Multiple forecasting methods
    println!("\n5. Forecasting Comparison");
    compare_forecasting_methods(&data);

    // Step 6: Advanced analysis
    println!("\n6. Advanced Analysis");
    advanced_analysis(&data);

    // Step 7: Streaming analysis demonstration
    println!("\n7. Streaming Analysis");
    streaming_analysis_demo(&data);

    // Step 8: Model validation
    println!("\n8. Model Validation");
    validate_models(&data);

    println!("\n=== Analysis Complete ===");
}

#[allow(dead_code)]
fn generate_synthetic_data() -> Array1<f64> {
    let n = 200;
    let mut data = Array1::zeros(n);

    for i in 0..n {
        let t = i as f64;
        // Trend component
        let trend = 0.1 * t;
        // Seasonal component (annual cycle)
        let seasonal = 10.0 * (2.0 * std::f64::consts::PI * t / 12.0).sin();
        // Weekly cycle
        let weekly = 3.0 * (2.0 * std::f64::consts::PI * t / 7.0).sin();
        // Noise
        let noise = (t * 0.1).sin() * 2.0;

        data[i] = 100.0 + trend + seasonal + weekly + noise;

        // Add some anomalies
        if i == 50 || i == 150 {
            data[i] += 20.0;
        }
    }

    data
}

#[allow(dead_code)]
fn preprocess_data(data: &Array1<f64>) -> (Array1<f64>, BoxCoxTransformer<f64>) {
    // Apply Box-Cox transformation
    let mut transformer = BoxCoxTransformer::new();
    let transformed = transformer
        .fit_transform(_data)
        .unwrap_or_else(|_| data.clone());

    // Apply differencing if needed
    let mut diff_transformer = DifferencingTransformer::new(1);
    let _differenced = diff_transformer
        .fit_transform(&transformed)
        .unwrap_or(transformed.clone());

    (transformed, transformer)
}

#[allow(dead_code)]
fn decompose_series(data: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let options = StlOptions {
        seasonal_period: 12,
        seasonal_smoother: 7,
        trend_smoother: None,
        low_pass_smoother: None,
        seasonal_degree: 1,
        trend_degree: 1,
        low_pass_degree: 1,
        robust: true,
        seasonal_jump: 1,
        trend_jump: 1,
        low_pass_jump: 1,
    };

    let mut decomposer = StlDecomposer::new(options);
    decomposer.decompose(_data).unwrap_or_else(|_| {
        // Fallback to simple decomposition
        let n = data.len();
        let trend = Array1::linspace(_data[0], data[n - 1], n);
        let seasonal = Array1::zeros(n);
        let residual = _data - &trend;
        (trend, seasonal, residual)
    })
}

#[allow(dead_code)]
fn analyze_decomposition(result: &(Array1<f64>, Array1<f64>, Array1<f64>)) {
    let (trend, seasonal, residual) = result;

    println!("  Trend variance: {:.2}", calculate_variance(trend));
    println!("  Seasonal variance: {:.2}", calculate_variance(seasonal));
    println!("  Residual variance: {:.2}", calculate_variance(residual));

    let total_var =
        calculate_variance(trend) + calculate_variance(seasonal) + calculate_variance(residual);
    println!(
        "  Trend contribution: {:.1}%",
        100.0 * calculate_variance(trend) / total_var
    );
    println!(
        "  Seasonal contribution: {:.1}%",
        100.0 * calculate_variance(seasonal) / total_var
    );
    println!(
        "  Noise contribution: {:.1}%",
        100.0 * calculate_variance(residual) / total_var
    );
}

#[allow(dead_code)]
fn calculate_variance(data: &Array1<f64>) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let mean = data.mean().unwrap_or(0.0);
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (_data.len() - 1) as f64
}

#[allow(dead_code)]
fn extract_and_analyze_features(data: &Array1<f64>) {
    let config = FeatureConfig::comprehensive();
    let mut extractor = FeatureExtractor::new(config);

    if let Ok(features) = extractor.extract_features(_data) {
        println!("  Extracted {} features", features.len());

        // Analyze key features
        if features.len() >= 10 {
            println!("  Mean: {:.2}", features[0]);
            println!("  Standard deviation: {:.2}", features[1]);
            println!("  Skewness: {:.2}", features[2]);
            println!("  Kurtosis: {:.2}", features[3]);
            println!("  Trend strength: {:.2}", features[4]);
        }
    } else {
        println!("  Failed to extract features");
    }
}

#[allow(dead_code)]
fn detect_changes_and_anomalies(data: &Array1<f64>) {
    // Change point detection
    let mut pelt_detector = PeltDetector::new(2.0, 5);
    if let Ok(change_points) = pelt_detector.detect(_data) {
        println!(
            "  Detected {} change points: {:?}",
            change_points.len(),
            change_points
        );
    }

    // Anomaly detection with multiple methods
    let mut z_detector = ZScoreDetector::new(3.0);
    if let Ok(z_anomalies) = z_detector.detect(_data) {
        let anomaly_count = z_anomalies.iter().filter(|&&x| x).count();
        println!("  Z-score anomalies: {}", anomaly_count);
    }

    let mut isolation_detector = IsolationForestDetector::new(100, 0.1, Some(42));
    if let Ok(iso_anomalies) = isolation_detector.detect(_data) {
        let anomaly_count = iso_anomalies.iter().filter(|&&x| x).count();
        println!("  Isolation forest anomalies: {}", anomaly_count);
    }
}

#[allow(dead_code)]
fn compare_forecasting_methods(data: &Array1<f64>) {
    let forecast_horizon = 10;
    let train_size = data.len() - forecast_horizon;
    let train_data = data.slice(scirs2_core::ndarray::s![..train_size]).to_owned();
    let test_data = data.slice(scirs2_core::ndarray::s![train_size..]).to_owned();

    println!(
        "  Training on {} points, testing on {} points",
        train_size,
        test_data.len()
    );

    // Method 1: ARIMA
    let arima_options = ArimaSelectionOptions {
        max_p: 3,
        max_d: 2,
        max_q: 3,
        criterion: SelectionCriterion::AIC,
        stepwise: true,
        ..Default::default()
    };

    if let Ok((mut arima_model, params)) = auto_arima(&train_data, &arima_options) {
        if let Ok(arima_forecast) = arima_model.forecast(forecast_horizon, &train_data) {
            let arima_mse = calculate_mse(&test_data, &arima_forecast.point_forecast);
            println!(
                "    ARIMA({},{},{}) MSE: {:.2}",
                params.pdq.0, params.pdq.1, params.pdq.2, arima_mse
            );
        }
    } else {
        println!("    ARIMA: Failed to fit model");
    }

    // Method 2: Exponential Smoothing
    let es_config = ExponentialSmoothingConfig {
        alpha: Some(0.3),
        beta: Some(0.1),
        gamma: Some(0.1),
        seasonal_periods: Some(12),
        damping_parameter: None,
        use_boxcox: false,
        remove_bias: true,
        use_brute: false,
    };

    let mut es_forecaster = ExponentialSmoothingForecaster::new(es_config);
    if let Ok(_) = es_forecaster.fit(&train_data) {
        if let Ok(es_forecast) = es_forecaster.predict(forecast_horizon) {
            let es_mse = calculate_mse(&test_data, &es_forecast.point_forecast);
            println!("    Exponential Smoothing MSE: {:.2}", es_mse);
        }
    } else {
        println!("    Exponential Smoothing: Failed to fit model");
    }

    // Method 3: LSTM Neural Network
    let lstm_config = LSTMConfig {
        base: scirs2,
        series: forecasting::neural::NeuralConfig {
            lookback_window: 10,
            forecast_horizon,
            epochs: 50,
            learning_rate: 0.001,
            batch_size: 16,
            validation_split: 0.2,
            early_stopping_patience: Some(5),
            random_seed: Some(42),
        },
        num_layers: 2,
        hidden_size: 32,
        dropout: 0.2,
        bidirectional: false,
    };

    let mut lstm_forecaster = LSTMForecaster::new(lstm_config);
    if let Ok(_) = lstm_forecaster.fit(&train_data) {
        if let Ok(lstm_forecast) = lstm_forecaster.predict(forecast_horizon) {
            let lstm_mse = calculate_mse(&test_data, &lstm_forecast.point_forecast);
            println!("    LSTM Neural Network MSE: {:.2}", lstm_mse);
        }
    } else {
        println!("    LSTM: Failed to fit model");
    }
}

#[allow(dead_code)]
fn calculate_mse(actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    let min_len = actual.len().min(predicted.len());
    if min_len == 0 {
        return f64::INFINITY;
    }

    let mut sum_sq_error = 0.0;
    for i in 0..min_len {
        let error = actual[i] - predicted[i];
        sum_sq_error += error * error;
    }
    sum_sq_error / min_len as f64
}

#[allow(dead_code)]
fn advanced_analysis(data: &Array1<f64>) {
    // Correlation analysis
    let mut corr_analyzer = CrossCorrelationAnalyzer::new();
    if let Ok(autocorr) = corr_analyzer.autocorrelation(_data, 20) {
        let max_autocorr = autocorr
            .iter()
            .skip(1)
            .fold(0.0, |max, &x| max.max(x.abs()));
        println!("  Maximum autocorrelation: {:.3}", max_autocorr);
    }

    // Dynamic Time Warping self-similarity
    let dtw = DynamicTimeWarping::new();
    let half_len = data.len() / 2;
    let first_half = data.slice(scirs2_core::ndarray::s![..half_len]).to_owned();
    let second_half = data.slice(scirs2_core::ndarray::s![half_len..half_len * 2]).to_owned();

    if let Ok(dtw_distance) = dtw.distance(&first_half, &second_half) {
        println!("  DTW distance between halves: {:.2}", dtw_distance);
    }

    // Clustering analysis
    let clustering_config = ClusteringConfig {
        n_clusters: 3,
        distance_metric: scirs2,
        _series: clustering::DistanceMetric::Euclidean,
        max_iterations: 100,
        tolerance: 1e-4,
        random_seed: Some(42),
    };

    // Create multiple subsequences for clustering
    let window_size = 20;
    let step_size = 10;
    let mut subsequences = Vec::new();

    for i in (0.._data.len() - window_size).step_by(step_size) {
        let subseq = data.slice(scirs2_core::ndarray::s![i..i + window_size]).to_owned();
        subsequences.push(subseq);
    }

    if subsequences.len() >= 3 {
        let mut clusterer = TimeSeriesClusterer::new(clustering_config);
        if let Ok(clusters) = clusterer.fit_predict(&subsequences) {
            println!(
                "  Clustered {} subsequences into {} groups",
                subsequences.len(),
                clusters.iter().max().unwrap_or(&0) + 1
            );
        }
    }
}

#[allow(dead_code)]
fn streaming_analysis_demo(data: &Array1<f64>) {
    let config = StreamConfig {
        window_size: 50,
        min_observations: 10,
        update_frequency: 5,
        memory_threshold: 1000,
        adaptive_windowing: false,
        change_detection_threshold: 2.5,
    };

    let mut analyzer = StreamingAnalyzer::new(config).unwrap();
    let mut detected_changes = 0;

    for (i, &value) in data.iter().enumerate() {
        if let Ok(_) = analyzer.add_observation(value) {
            let change_points = analyzer.get_change_points();
            if change_points.len() > detected_changes {
                detected_changes = change_points.len();
                println!(
                    "  Change detected at index {}: confidence {:.2}",
                    i,
                    change_points.last().unwrap().confidence
                );
            }

            // Check for outliers
            if analyzer.is_outlier(value) {
                println!("  Outlier detected at index {}: value {:.2}", i, value);
            }
        }
    }

    let stats = analyzer.get_stats();
    println!(
        "  Final streaming stats: mean={:.2}, std={:.2}, count={}",
        stats.mean(),
        stats.std_dev(),
        stats.count()
    );

    // Generate streaming forecast
    if let Ok(forecast) = analyzer.forecast(5) {
        println!("  Streaming forecast: {:?}", forecast.to_vec());
    }
}

#[allow(dead_code)]
fn validate_models(data: &Array1<f64>) {
    let cv_config = CrossValidationConfig {
        n_folds: 5,
        test_size: 0.2,
        gap: 0,
        metric: ValidationMetric::MSE,
        shuffle: false,
    };

    // Validate ARIMA model
    let arima_options = ArimaSelectionOptions {
        max_p: 2,
        max_d: 1,
        max_q: 2,
        criterion: SelectionCriterion::AIC,
        stepwise: true,
        ..Default::default()
    };

    if let Ok(cv_results) = cross_validate(_data, &arima_options, &cv_config) {
        println!("  Cross-validation results:");
        println!("    Mean MSE: {:.4}", cv_results.mean_score);
        println!("    Std MSE: {:.4}", cv_results.std_score);
        println!(
            "    Min MSE: {:.4}",
            cv_results
                .scores
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b))
        );
        println!(
            "    Max MSE: {:.4}",
            cv_results
                .scores
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        );
    } else {
        println!("  Cross-validation failed");
    }
}
*/
