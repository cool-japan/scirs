//! Example of streaming/online evaluation metrics
//!
//! This example demonstrates how to use streaming metrics for real-time
//! evaluation and large dataset handling.

use scirs2_metrics::streaming::{
    StreamingClassificationMetrics, StreamingRegressionMetrics, WindowedClassificationMetrics,
    WindowedRegressionMetrics,
};
use std::thread;
use std::time::Duration;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Streaming Metrics Example");
    println!("========================");

    // Example 1: Real-time classification monitoring
    println!("\n1. Real-time Classification Monitoring");
    println!("--------------------------------------");

    real_time_classification_example();

    // Example 2: Large dataset processing
    println!("\n2. Large Dataset Processing (Memory Efficient)");
    println!("-----------------------------------------------");

    large_dataset_example()?;

    // Example 3: Windowed metrics for concept drift detection
    println!("\n3. Windowed Metrics for Concept Drift Detection");
    println!("-----------------------------------------------");

    concept_drift_example();

    // Example 4: Streaming regression metrics
    println!("\n4. Streaming Regression Metrics");
    println!("-------------------------------");

    streaming_regression_example();

    // Example 5: Performance comparison with batch metrics
    println!("\n5. Performance Comparison");
    println!("------------------------");

    performance_comparison_example();

    println!("\nStreaming metrics example completed successfully!");
    Ok(())
}

/// Simulates real-time classification with streaming metrics
#[allow(dead_code)]
fn real_time_classification_example() {
    let mut metrics = StreamingClassificationMetrics::new();

    println!("Simulating real-time predictions...");

    // Simulate incoming predictions over time
    for i in 0..20 {
        // Simulate some pattern: accuracy starts high then degrades
        let true_label = i % 2;
        let pred_label = if i < 10 {
            true_label
        } else {
            1 - true_label // Start making mistakes
        };

        metrics.update(true_label, pred_label);

        if i % 5 == 4 {
            println!(
                "After {} predictions: Accuracy={:.3}, Precision={:.3}, Recall={:.3}, F1={:.3}",
                metrics.sample_count(),
                metrics.accuracy(),
                metrics.precision(),
                metrics.recall(),
                metrics.f1_score()
            );

            // Simulate real-time delay
            thread::sleep(Duration::from_millis(100));
        }
    }

    let (tp, fp, tn, fn_counts) = metrics.confusion_matrix();
    println!("Final confusion matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn_counts}");
}

/// Demonstrates processing large datasets without loading everything into memory
#[allow(dead_code)]
fn large_dataset_example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut metrics = StreamingClassificationMetrics::new();

    println!("Processing large dataset in chunks...");

    // Simulate processing a large dataset in chunks
    let total_samples = 100_000;
    let chunk_size = 1_000;

    for chunk_start in (0..total_samples).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(total_samples);
        let chunk_samples = chunk_end - chunk_start;

        // Generate synthetic chunk data
        let (true_labels, pred_labels) = generate_synthetic_chunk(chunk_start, chunk_samples);

        // Process chunk with batch update
        metrics.update_batch(&true_labels, &pred_labels)?;

        if chunk_start % (chunk_size * 10) == 0 {
            println!(
                "Processed {}/{} samples. Current accuracy: {:.4}",
                chunk_end,
                total_samples,
                metrics.accuracy()
            );
        }
    }

    println!(
        "Final results for {} samples: Accuracy={:.4}, F1={:.4}",
        metrics.sample_count(),
        metrics.accuracy(),
        metrics.f1_score()
    );

    Ok(())
}

/// Generates synthetic data for a chunk
#[allow(dead_code)]
fn generate_synthetic_chunk(_startidx: usize, size: usize) -> (Vec<i32>, Vec<i32>) {
    let mut true_labels = Vec::with_capacity(size);
    let mut pred_labels = Vec::with_capacity(size);

    for i in 0..size {
        let _idx = _startidx + i;
        let true_label = (_idx % 3) as i32 % 2; // Some pattern

        // Introduce some noise in predictions
        let pred_label = if (_idx * 7) % 10 < 8 {
            true_label
        } else {
            1 - true_label
        };

        true_labels.push(true_label);
        pred_labels.push(pred_label);
    }

    (true_labels, pred_labels)
}

/// Demonstrates windowed metrics for detecting concept drift
#[allow(dead_code)]
fn concept_drift_example() {
    let window_size = 50;
    let mut metrics = WindowedClassificationMetrics::new(window_size);

    println!("Monitoring for concept drift with window size {window_size}...");

    for i in 0..200 {
        let true_label = i % 2;

        // Simulate concept drift: performance degrades after step 100
        let pred_label = if i < 100 {
            // High accuracy period
            if (i * 13) % 10 < 9 {
                true_label
            } else {
                1 - true_label
            }
        } else {
            // Low accuracy period (concept drift)
            if (i * 13) % 10 < 6 {
                true_label
            } else {
                1 - true_label
            }
        };

        metrics.update(true_label, pred_label);

        if i % 25 == 24 {
            println!(
                "Step {}: Window accuracy={:.3} (window size: {}/{})",
                i + 1,
                metrics.accuracy(),
                metrics.current_window_size(),
                metrics.max_window_size()
            );

            // Alert if performance drops significantly
            if metrics.accuracy() < 0.7 && metrics.current_window_size() == window_size {
                println!("  ⚠️  ALERT: Potential concept drift detected!");
            }
        }
    }
}

/// Demonstrates streaming regression metrics
#[allow(dead_code)]
fn streaming_regression_example() {
    let mut metrics = StreamingRegressionMetrics::<f64>::new();

    println!("Streaming regression evaluation...");

    // Simulate regression predictions
    for i in 0..100 {
        let true_value = (i as f64) * 0.5 + 10.0; // Linear relationship

        // Add some noise and bias to predictions
        let noise = ((i * 7) % 10) as f64 / 10.0 - 0.5; // -0.5 to 0.4
        let bias = if i > 50 { 2.0 } else { 0.0 }; // Introduce bias later
        let pred_value = true_value + noise + bias;

        metrics.update(true_value, pred_value);

        if i % 20 == 19 {
            println!(
                "Step {}: MSE={:.4}, MAE={:.4}, RMSE={:.4}, R²={:.4}",
                i + 1,
                metrics.mse(),
                metrics.mae(),
                metrics.rmse(),
                metrics.r2_score()
            );

            if let (Some(min_err), Some(max_err)) = (metrics.min_error(), metrics.max_error()) {
                println!("  Error range: [{min_err:.4}, {max_err:.4}]");
            }
        }
    }
}

/// Compares performance of streaming vs batch computation
#[allow(dead_code)]
fn performance_comparison_example() {
    use std::time::Instant;

    println!("Comparing streaming vs batch computation performance...");

    let n_samples = 10_000;

    // Generate test data
    let true_labels: Vec<i32> = (0..n_samples).map(|i| i % 2).collect();
    let pred_labels: Vec<i32> = (0..n_samples)
        .map(|i| if (i * 7) % 10 < 8 { i % 2 } else { 1 - i % 2 })
        .collect();

    // Streaming approach
    let start = Instant::now();
    let mut streaming_metrics = StreamingClassificationMetrics::new();

    for (&true_label, &pred_label) in true_labels.iter().zip(pred_labels.iter()) {
        streaming_metrics.update(true_label, pred_label);
    }

    let streaming_time = start.elapsed();
    let streaming_accuracy = streaming_metrics.accuracy();

    // Batch approach (simulated)
    let start = Instant::now();
    let correct_predictions = true_labels
        .iter()
        .zip(pred_labels.iter())
        .filter(|(&t, &p)| t == p)
        .count();
    let batch_accuracy = correct_predictions as f64 / n_samples as f64;
    let batch_time = start.elapsed();

    println!("Results for {n_samples} samples:");
    println!("  Streaming: {streaming_accuracy:.4} accuracy in {streaming_time:?}");
    println!("  Batch:     {batch_accuracy:.4} accuracy in {batch_time:?}");
    println!(
        "  Accuracy difference: {:.6}",
        (streaming_accuracy - batch_accuracy).abs()
    );

    // Demonstrate memory efficiency with windowed metrics
    println!("\nMemory efficiency demonstration:");
    let window_size = 1000;
    let mut windowed = WindowedClassificationMetrics::new(window_size);

    println!(
        "Processing {} samples with window size {}...",
        n_samples * 2,
        window_size
    );

    for i in 0..(n_samples * 2) {
        let true_label = i % 2;
        let pred_label = if (i * 7) % 10 < 8 {
            true_label
        } else {
            1 - true_label
        };

        windowed.update(true_label, pred_label);
    }

    println!(
        "Final windowed accuracy: {:.4} (constant memory usage regardless of total samples)",
        windowed.accuracy()
    );

    // Demonstrate windowed regression
    let mut windowed_reg = WindowedRegressionMetrics::<f64>::new(100);

    for i in 0..1000 {
        let true_val = i as f64;
        let pred_val = true_val + ((i * 3) % 10) as f64 / 10.0; // Small noise
        windowed_reg.update(true_val, pred_val);
    }

    println!(
        "Windowed regression MSE: {:.4} (last 100 samples only)",
        windowed_reg.mse()
    );
}
