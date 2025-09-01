//! Real-time streaming time series analysis demonstration
//!
//! This example shows how to use the streaming analysis capabilities for real-time
//! time series processing, including online statistics, change detection, and forecasting.

use ndarray::Array1;
use scirs2_series::streaming::{
    advanced::{
        CircularBuffer, ModelState, StreamingAnomalyDetector, StreamingForecaster,
        StreamingPatternMatcher,
    },
    MultiSeriesAnalyzer, StreamConfig, StreamingAnalyzer,
};
use statrs::statistics::Statistics;
use std::thread;
use std::time::{Duration, Instant};

#[allow(dead_code)]
fn main() {
    println!("=== Real-time Streaming Analysis Demo ===\n");

    // Demo 1: Basic streaming analysis
    println!("1. Basic Streaming Analysis");
    basic_streaming_demo();

    // Demo 2: Multi-series streaming
    println!("\n2. Multi-series Streaming Analysis");
    multi_series_demo();

    // Demo 3: Advanced streaming forecasting
    println!("\n3. Advanced Streaming Forecasting");
    streaming_forecasting_demo();

    // Demo 4: Real-time anomaly detection
    println!("\n4. Real-time Anomaly Detection");
    anomaly_detection_demo();

    // Demo 5: Pattern matching
    println!("\n5. Real-time Pattern Matching");
    pattern_matching_demo();

    // Demo 6: Memory-efficient processing
    println!("\n6. Memory-efficient Circular Buffer");
    circular_buffer_demo();

    println!("\n=== Streaming Analysis Complete ===");
}

#[allow(dead_code)]
fn basic_streaming_demo() {
    let config = StreamConfig {
        window_size: 100,
        min_observations: 5,
        update_frequency: 10,
        memory_threshold: 1000,
        adaptive_windowing: true,
        change_detection_threshold: 2.0,
    };

    let mut analyzer =
        StreamingAnalyzer::<f64>::new(config).expect("Failed to create streaming analyzer");

    println!("  Simulating real-time data stream...");

    // Simulate real-time data with different patterns
    let mut current_time = 0.0;
    let mut pattern_phase = 0;

    for i in 0..200 {
        let value = generate_streaming_value(current_time, pattern_phase);

        // Simulate real-time processing delay
        thread::sleep(Duration::from_millis(10));

        if let Ok(_) = analyzer.add_observation(value) {
            // Check for change points
            let change_points = analyzer.get_change_points();
            if !change_points.is_empty() {
                let latest = change_points.last().unwrap();
                println!(
                    "    Change detected at step {}: type={:?}, confidence={:.2}",
                    i, latest.change_type, latest.confidence
                );
            }

            // Check for outliers
            if analyzer.is_outlier(value) {
                println!("    Outlier detected at step {}: value={:.2}", i, value);
            }

            // Periodic statistics update
            if i % 50 == 49 {
                let stats = analyzer.get_stats();
                println!(
                    "    Step {}: mean={:.2}, std={:.2}, count={}",
                    i + 1,
                    stats.mean(),
                    stats.std_dev(),
                    stats.count()
                );

                if let Some(ewma_value) = analyzer.get_ewma() {
                    println!("      EWMA value: {:.2}", ewma_value);
                }

                // Generate forecast
                if let Ok(forecast) = analyzer.forecast(5) {
                    println!(
                        "      5-step forecast: {:?}",
                        forecast
                            .iter()
                            .map(|x| format!("{:.1}", x))
                            .collect::<Vec<_>>()
                    );
                }
            }
        }

        current_time += 1.0;

        // Change pattern every 60 steps
        if i % 60 == 59 {
            pattern_phase = (pattern_phase + 1) % 3;
            println!("    Pattern change to phase {}", pattern_phase);
        }
    }

    // Final summary
    let total_observations = analyzer.observation_count();
    let buffer_size = analyzer.buffer_size();
    let time_since_update = analyzer.time_since_last_update();

    println!("  Final summary:");
    println!("    Total observations: {}", total_observations);
    println!("    Buffer size: {}", buffer_size);
    println!("    Time since last update: {:?}", time_since_update);
    println!(
        "    Total change points detected: {}",
        analyzer.get_change_points().len()
    );
}

#[allow(dead_code)]
fn generate_streaming_value(time: f64, pattern: i32) -> f64 {
    let base = 50.0;

    match pattern {
        0 => {
            // Sinusoidal pattern with trend
            base + 0.1 * time + 10.0 * (0.2 * time).sin() + 2.0 * rand_noise()
        }
        1 => {
            // Step function with noise
            let step = if (time as i32) % 40 < 20 { 10.0 } else { -10.0 };
            base + step + 5.0 * rand_noise()
        }
        2 => {
            // Random walk
            base + 0.5 * time.sin() + 3.0 * ((time * 0.1).cos() + rand_noise())
        }
        _ => base + rand_noise(),
    }
}

#[allow(dead_code)]
fn rand_noise() -> f64 {
    // Simple pseudo-random noise generator
    let time_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();

    let normalized = (time_ns % 10000) as f64 / 10000.0;
    2.0 * (normalized - 0.5) // Range [-1, 1]
}

#[allow(dead_code)]
fn multi_series_demo() {
    let config = StreamConfig {
        window_size: 50,
        min_observations: 10,
        update_frequency: 5,
        memory_threshold: 500,
        adaptive_windowing: false,
        change_detection_threshold: 2.5,
    };

    let mut multi_analyzer = MultiSeriesAnalyzer::<f64>::new(config);

    // Add multiple series
    let series_names = vec!["temperature", "humidity", "pressure"];
    for name in &series_names {
        multi_analyzer
            .add_series(name.to_string())
            .expect("Failed to add series");
    }

    println!("  Added {} series for monitoring", series_names.len());

    // Simulate correlated data streams
    for i in 0..100 {
        let base_time = i as f64;

        // Temperature (base signal)
        let temp = 20.0 + 5.0 * (base_time * 0.1).sin() + rand_noise();
        multi_analyzer
            .add_observation("temperature", temp)
            .expect("Failed to add temperature");

        // Humidity (inversely correlated with temperature)
        let humidity = 60.0 - 0.8 * (temp - 20.0) + 2.0 * rand_noise();
        multi_analyzer
            .add_observation("humidity", humidity)
            .expect("Failed to add humidity");

        // Pressure (weakly correlated)
        let pressure = 1013.0 + 0.2 * temp + 3.0 * (base_time * 0.05).cos() + rand_noise();
        multi_analyzer
            .add_observation("pressure", pressure)
            .expect("Failed to add pressure");

        thread::sleep(Duration::from_millis(5));

        // Periodic analysis
        if i % 25 == 24 {
            println!("    Step {}: Analyzing correlations", i + 1);

            // Check correlations between series
            if let Ok(temp_humidity_corr) =
                multi_analyzer.get_correlation("temperature", "humidity")
            {
                println!(
                    "      Temperature-Humidity correlation: {:.3}",
                    temp_humidity_corr
                );
            }

            if let Ok(temp_pressure_corr) =
                multi_analyzer.get_correlation("temperature", "pressure")
            {
                println!(
                    "      Temperature-Pressure correlation: {:.3}",
                    temp_pressure_corr
                );
            }

            // Get statistics for each series
            for name in &series_names {
                if let Some(analyzer) = multi_analyzer.get_analyzer(name) {
                    let stats = analyzer.get_stats();
                    println!(
                        "      {}: mean={:.1}, std={:.1}",
                        name,
                        stats.mean(),
                        stats.std_dev()
                    );
                }
            }
        }
    }
}

#[allow(dead_code)]
fn streaming_forecasting_demo() {
    println!("  Setting up streaming forecaster with Holt-Winters method...");

    let mut forecaster = StreamingForecaster::<f64>::new(
        0.3,       // alpha (level)
        Some(0.1), // beta (trend)
        Some(0.1), // gamma (seasonal)
        Some(12),  // seasonal period
        100,       // max buffer size
    )
    .expect("Failed to create streaming forecaster");

    // Generate data with trend and seasonality
    for i in 0..150 {
        let t = i as f64;
        let trend = 0.1 * t;
        let seasonal = 5.0 * (2.0 * std::f64::consts::PI * t / 12.0).sin();
        let noise = rand_noise();
        let value = 100.0 + trend + seasonal + noise;

        forecaster
            .update(value)
            .expect("Failed to update forecaster");

        // Generate forecasts periodically
        if i >= 24 && i % 20 == 19 {
            if let Ok(forecast) = forecaster.forecast(5) {
                println!(
                    "    Step {}: 5-step forecast: {:?}",
                    i + 1,
                    forecast
                        .iter()
                        .map(|x| format!("{:.1}", x))
                        .collect::<Vec<_>>()
                );

                let state = forecaster.get_state();
                if let Some(level) = state.level {
                    println!(
                        "      Model state: level={:.1}, trend={:.3}, components={}",
                        level,
                        state.trend.unwrap_or(0.0),
                        state.seasonal_components.len()
                    );
                }
            }
        }

        thread::sleep(Duration::from_millis(5));
    }
}

#[allow(dead_code)]
fn anomaly_detection_demo() {
    let mut detector = StreamingAnomalyDetector::<f64>::new(
        100, // max buffer size
        3.0, // threshold
        10,  // window size
        5,   // number of features
    );

    println!("  Streaming anomaly detection with isolation forest approach...");

    // Generate mostly normal data with some anomalies
    let mut normal_data = Vec::new();

    for i in 0..200 {
        let t = i as f64;
        let normal_value = 50.0 + 10.0 * (t * 0.1).sin() + 2.0 * rand_noise();

        // Inject anomalies
        let value = if i == 50 || i == 120 || i == 180 {
            normal_value + 30.0 // Clear anomaly
        } else if i >= 80 && i <= 90 {
            normal_value + 15.0 // Anomalous period
        } else {
            normal_value
        };

        normal_data.push(value);

        // Update detector when we have enough data
        if normal_data.len() >= 10 {
            let window = &normal_data[normal_data.len() - 10..];

            if let Ok(is_anomaly) = detector.update(window) {
                if is_anomaly {
                    println!("    ANOMALY detected at step {}: value={:.1}", i + 1, value);
                }
            }
        }

        // Adapt threshold periodically
        if i % 50 == 49 && i > 50 {
            detector.adapt_threshold(2.5);
            println!("    Adapted anomaly threshold at step {}", i + 1);
        }

        thread::sleep(Duration::from_millis(5));
    }
}

#[allow(dead_code)]
fn pattern_matching_demo() {
    let mut matcher = StreamingPatternMatcher::<f64>::new(200, 0.7);

    // Define patterns to match
    let spike_pattern = vec![1.0, 2.0, 5.0, 8.0, 5.0, 2.0, 1.0];
    let dip_pattern = vec![5.0, 4.0, 2.0, 1.0, 2.0, 4.0, 5.0];
    let oscillation = vec![0.0, 3.0, 0.0, -3.0, 0.0, 3.0, 0.0];

    matcher
        .add_pattern(spike_pattern.clone(), "spike".to_string())
        .expect("Failed to add spike pattern");
    matcher
        .add_pattern(dip_pattern.clone(), "dip".to_string())
        .expect("Failed to add dip pattern");
    matcher
        .add_pattern(oscillation.clone(), "oscillation".to_string())
        .expect("Failed to add oscillation pattern");

    println!("  Added {} patterns for matching", 3);
    println!("    Spike pattern: {:?}", spike_pattern);
    println!("    Dip pattern: {:?}", dip_pattern);
    println!("    Oscillation pattern: {:?}", oscillation);

    // Generate data stream with embedded patterns
    let data_stream = vec![2.0, 3.0, 2.5, 2.8, 3.2]; // Initial data

    // Add spike pattern at position 20
    for i in 0..50 {
        let value = if i >= 15 && i < 15 + spike_pattern.len() {
            spike_pattern[i - 15] + 2.0 // Embedded spike
        } else if i >= 30 && i < 30 + dip_pattern.len() {
            dip_pattern[i - 30] + 2.0 // Embedded dip
        } else {
            3.0 + 0.5 * (i as f64 * 0.2).sin() + 0.2 * rand_noise() // Background
        };

        let matches = matcher.update(value);

        for pattern_match in matches {
            println!(
                "    PATTERN MATCH at step {}: '{}' (correlation: {:.3})",
                i + 5,
                pattern_match.pattern_name,
                pattern_match.correlation
            );
        }

        thread::sleep(Duration::from_millis(10));
    }
}

#[allow(dead_code)]
fn circular_buffer_demo() {
    let mut buffer = CircularBuffer::<f64>::new(20);

    println!("  Demonstrating memory-efficient circular buffer...");
    println!("  Buffer capacity: 20 elements");

    // Fill buffer beyond capacity to demonstrate circular behavior
    for i in 0..35 {
        let value = i as f64;
        buffer.push(value);

        if i % 5 == 4 {
            println!(
                "    Step {}: length={}, recent 5: {:?}",
                i + 1,
                buffer.len(),
                buffer.recent(5)
            );
        }

        if i == 25 {
            println!("    Buffer full - demonstrating circular overwrite");
            println!("    All values: {:?}", buffer.to_vec());
        }
    }

    // Demonstrate window statistics
    let stats = buffer.window_stats(10);
    println!("  Final window statistics (last 10 values):");
    println!("    Mean: {:.1}", stats.mean());
    println!("    Std dev: {:.1}", stats.std_dev());
    println!("    Min: {:.1}", stats.min());
    println!("    Max: {:.1}", stats.max());
    println!("    Count: {}", stats.count());
}
