//! Realistic noise models demonstration
//!
//! This example demonstrates the use of realistic noise injection utilities for creating
//! datasets with missing data, outliers, and various noise patterns that mimic real-world
//! data quality issues.

use ndarray::Array2;
use scirs2_datasets::{
    add_time_series_noise, inject_missing_data, inject_outliers, load_iris, make_corrupted_dataset,
    make_time_series, MissingPattern, OutlierType,
};

fn main() {
    println!("=== Realistic Noise Models Demonstration ===\n");

    // Demonstrate missing data patterns
    println!("=== Missing Data Patterns ========================");
    demonstrate_missing_data_patterns();

    // Demonstrate outlier injection
    println!("\n=== Outlier Injection ============================");
    demonstrate_outlier_injection();

    // Demonstrate time series noise
    println!("\n=== Time Series Noise ============================");
    demonstrate_time_series_noise();

    // Demonstrate comprehensive corruption
    println!("\n=== Comprehensive Dataset Corruption =============");
    demonstrate_comprehensive_corruption();

    // Real-world applications
    println!("\n=== Real-World Applications ======================");
    demonstrate_real_world_applications();

    println!("\n=== Noise Models Demo Complete ===================");
}

fn demonstrate_missing_data_patterns() {
    println!("Testing different missing data patterns on a sample dataset:");

    let original_data = Array2::from_shape_vec(
        (8, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0, 4.0, 8.0, 12.0, 16.0, 5.0,
            10.0, 15.0, 20.0, 6.0, 12.0, 18.0, 24.0, 7.0, 14.0, 21.0, 28.0, 8.0, 16.0, 24.0, 32.0,
        ],
    )
    .unwrap();

    let patterns = [
        (MissingPattern::MCAR, "Missing Completely at Random"),
        (MissingPattern::MAR, "Missing at Random"),
        (MissingPattern::MNAR, "Missing Not at Random"),
        (MissingPattern::Block, "Block-wise Missing"),
    ];

    for (pattern, description) in patterns {
        let mut test_data = original_data.clone();
        let missing_mask = inject_missing_data(&mut test_data, 0.3, pattern, Some(42)).unwrap();

        let missing_count = missing_mask.iter().filter(|&&x| x).count();
        let total_elements = test_data.len();
        let missing_percentage = (missing_count as f64 / total_elements as f64) * 100.0;

        println!("{}:", description);
        println!(
            "  Missing elements: {} / {} ({:.1}%)",
            missing_count, total_elements, missing_percentage
        );

        // Show pattern of missing data
        print!("  Pattern (X = missing): ");
        for i in 0..test_data.nrows() {
            for j in 0..test_data.ncols() {
                if missing_mask[[i, j]] {
                    print!("X ");
                } else {
                    print!(". ");
                }
            }
            if i < test_data.nrows() - 1 {
                print!("| ");
            }
        }
        println!();
    }
}

fn demonstrate_outlier_injection() {
    println!("Testing different outlier types on a sample dataset:");

    // Create a clean dataset with known statistics
    let mut clean_data = Array2::ones((20, 3));
    // Add some structure
    for i in 0..20 {
        for j in 0..3 {
            clean_data[[i, j]] = (i as f64 + j as f64) / 2.0;
        }
    }

    let outlier_types = [
        (OutlierType::Point, "Point Outliers"),
        (OutlierType::Contextual, "Contextual Outliers"),
        (OutlierType::Collective, "Collective Outliers"),
    ];

    for (outlier_type, description) in outlier_types {
        let mut test_data = clean_data.clone();
        let original_stats = calculate_basic_stats(&test_data);

        let outlier_mask =
            inject_outliers(&mut test_data, 0.2, outlier_type, 3.0, Some(42)).unwrap();
        let corrupted_stats = calculate_basic_stats(&test_data);

        let outlier_count = outlier_mask.iter().filter(|&&x| x).count();

        println!("{}:", description);
        println!(
            "  Outliers injected: {} / {} samples",
            outlier_count,
            test_data.nrows()
        );
        println!(
            "  Mean change: {:.3} -> {:.3} (Δ={:.3})",
            original_stats.0,
            corrupted_stats.0,
            corrupted_stats.0 - original_stats.0
        );
        println!(
            "  Std change: {:.3} -> {:.3} (Δ={:.3})",
            original_stats.1,
            corrupted_stats.1,
            corrupted_stats.1 - original_stats.1
        );

        // Show which samples are outliers
        print!("  Outlier samples: ");
        for (i, &is_outlier) in outlier_mask.iter().enumerate() {
            if is_outlier {
                print!("{} ", i);
            }
        }
        println!();
    }
}

fn demonstrate_time_series_noise() {
    println!("Testing different time series noise types:");

    // Create a simple time series
    let clean_ts = make_time_series(100, 2, true, true, 0.0, Some(42)).unwrap();

    let noise_configs = [
        vec![("gaussian", 0.2)],
        vec![("spikes", 0.1)],
        vec![("drift", 0.5)],
        vec![("seasonal", 0.3)],
        vec![("autocorrelated", 0.1)],
        vec![("heteroscedastic", 0.2)],
        vec![("gaussian", 0.1), ("spikes", 0.05), ("drift", 0.2)], // Combined noise
    ];

    let noise_names = [
        "Gaussian White Noise",
        "Impulse Spikes",
        "Linear Drift",
        "Seasonal Pattern",
        "Autocorrelated Noise",
        "Heteroscedastic Noise",
        "Combined Noise",
    ];

    for (config, name) in noise_configs.iter().zip(noise_names.iter()) {
        let mut noisy_data = clean_ts.data.clone();
        let original_stats = calculate_basic_stats(&noisy_data);

        add_time_series_noise(&mut noisy_data, config, Some(42)).unwrap();
        let noisy_stats = calculate_basic_stats(&noisy_data);

        println!("{}:", name);
        println!("  Mean: {:.3} -> {:.3}", original_stats.0, noisy_stats.0);
        println!("  Std: {:.3} -> {:.3}", original_stats.1, noisy_stats.1);
        println!(
            "  Range: [{:.3}, {:.3}] -> [{:.3}, {:.3}]",
            original_stats.2, original_stats.3, noisy_stats.2, noisy_stats.3
        );
    }
}

fn demonstrate_comprehensive_corruption() {
    println!("Testing comprehensive dataset corruption:");

    // Load a real dataset
    let iris = load_iris().unwrap();
    println!(
        "Original Iris dataset: {} samples, {} features",
        iris.n_samples(),
        iris.n_features()
    );

    let original_stats = calculate_basic_stats(&iris.data);
    println!(
        "Original stats - Mean: {:.3}, Std: {:.3}",
        original_stats.0, original_stats.1
    );

    // Create different levels of corruption
    let corruption_levels = [
        (0.05, 0.02, "Light corruption"),
        (0.1, 0.05, "Moderate corruption"),
        (0.2, 0.1, "Heavy corruption"),
        (0.3, 0.15, "Severe corruption"),
    ];

    for (missing_rate, outlier_rate, description) in corruption_levels {
        let corrupted = make_corrupted_dataset(
            &iris,
            missing_rate,
            MissingPattern::MAR, // More realistic than MCAR
            outlier_rate,
            OutlierType::Point,
            2.5,
            Some(42),
        )
        .unwrap();

        // Calculate how much data is usable
        let total_elements = corrupted.data.len();
        let missing_elements = corrupted.data.iter().filter(|&&x| x.is_nan()).count();
        let usable_percentage =
            ((total_elements - missing_elements) as f64 / total_elements as f64) * 100.0;

        println!("{}:", description);
        println!("  Missing data: {:.1}%", missing_rate * 100.0);
        println!("  Outliers: {:.1}%", outlier_rate * 100.0);
        println!("  Usable data: {:.1}%", usable_percentage);

        // Show metadata
        if let Some(missing_count) = corrupted.metadata.get("missing_count") {
            println!("  Actual missing: {} elements", missing_count);
        }
        if let Some(outlier_count) = corrupted.metadata.get("outlier_count") {
            println!("  Actual outliers: {} samples", outlier_count);
        }
    }
}

fn demonstrate_real_world_applications() {
    println!("Real-world application scenarios:");

    println!("\n1. **Medical Data Simulation**:");
    let medical_data = load_iris().unwrap(); // Stand-in for medical measurements
    let _corrupted_medical = make_corrupted_dataset(
        &medical_data,
        0.15,                 // 15% missing - common in medical data
        MissingPattern::MNAR, // High values often missing (privacy, measurement issues)
        0.05,                 // 5% outliers - measurement errors
        OutlierType::Point,
        2.0,
        Some(42),
    )
    .unwrap();

    println!("  Medical dataset simulation:");
    println!("    Missing data pattern: MNAR (high values more likely missing)");
    println!("    Outliers: Point outliers (measurement errors)");
    println!("    Use case: Testing imputation algorithms for clinical data");

    println!("\n2. **Sensor Network Simulation**:");
    let sensor_data = make_time_series(200, 4, true, true, 0.1, Some(42)).unwrap();
    let mut sensor_ts = sensor_data.data.clone();

    // Add realistic sensor noise
    add_time_series_noise(
        &mut sensor_ts,
        &[
            ("gaussian", 0.05),        // Background noise
            ("spikes", 0.02),          // Electrical interference
            ("drift", 0.1),            // Sensor calibration drift
            ("heteroscedastic", 0.03), // Temperature-dependent noise
        ],
        Some(42),
    )
    .unwrap();

    // Add missing data (sensor failures)
    inject_missing_data(&mut sensor_ts, 0.08, MissingPattern::Block, Some(42)).unwrap();

    println!("  Sensor network simulation:");
    println!("    Multiple noise types: gaussian + spikes + drift + heteroscedastic");
    println!("    Missing data: Block pattern (sensor failures)");
    println!("    Use case: Testing robust time series algorithms");

    println!("\n3. **Survey Data Simulation**:");
    let survey_data = load_iris().unwrap(); // Stand-in for survey responses
    let _corrupted_survey = make_corrupted_dataset(
        &survey_data,
        0.25,                // 25% missing - typical for surveys
        MissingPattern::MAR, // Missing depends on other responses
        0.08,                // 8% outliers - data entry errors, extreme responses
        OutlierType::Contextual,
        1.5,
        Some(42),
    )
    .unwrap();

    println!("  Survey data simulation:");
    println!("    Missing data pattern: MAR (depends on other responses)");
    println!("    Outliers: Contextual (unusual response patterns)");
    println!("    Use case: Testing survey analysis robustness");

    println!("\n4. **Financial Data Simulation**:");
    let mut financial_ts = make_time_series(500, 3, false, false, 0.02, Some(42))
        .unwrap()
        .data;

    // Add financial market-specific noise
    add_time_series_noise(
        &mut financial_ts,
        &[
            ("gaussian", 0.1),        // Market volatility
            ("spikes", 0.05),         // Market shocks
            ("autocorrelated", 0.15), // Momentum effects
            ("heteroscedastic", 0.2), // Volatility clustering
        ],
        Some(42),
    )
    .unwrap();

    println!("  Financial data simulation:");
    println!("    Noise types: volatility + shocks + momentum + clustering");
    println!("    Use case: Testing financial models under realistic conditions");
}

/// Calculate basic statistics for a 2D array
fn calculate_basic_stats(data: &Array2<f64>) -> (f64, f64, f64, f64) {
    let valid_values: Vec<f64> = data.iter().filter(|&&x| !x.is_nan()).cloned().collect();

    if valid_values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
    let variance = valid_values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / valid_values.len() as f64;
    let std = variance.sqrt();
    let min = valid_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = valid_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    (mean, std, min, max)
}
