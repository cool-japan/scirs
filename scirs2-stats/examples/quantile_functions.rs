// Example file demonstrating the use of quantile-based functions

use scirs2_core::ndarray::array;
use scirs2_stats::{
    boxplot_stats, deciles, percentile, quantile, quartiles, quintiles, winsorized_mean,
    winsorized_variance, QuantileInterpolation,
};

#[allow(dead_code)]
fn main() {
    // Create a sample dataset
    let data_normal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let data_with_outlier = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];

    // Quantile functions
    println!("==== Quantile Functions ====");
    println!("\nSample data: {:?}", data_normal);

    // Basic quantiles
    let median = quantile(&data_normal.view(), 0.5, QuantileInterpolation::Linear).unwrap();
    println!("\nMedian (0.5 quantile): {:.1}", median);

    let q1 = quantile(&data_normal.view(), 0.25, QuantileInterpolation::Linear).unwrap();
    println!("First quartile (0.25 quantile): {:.1}", q1);

    let q3 = quantile(&data_normal.view(), 0.75, QuantileInterpolation::Linear).unwrap();
    println!("Third quartile (0.75 quantile): {:.1}", q3);

    // Demonstrate different interpolation methods
    println!("\nQuantile 0.4 with different interpolation methods:");
    println!(
        "  Linear: {:.2}",
        quantile(&data_normal.view(), 0.4, QuantileInterpolation::Linear).unwrap()
    );
    println!(
        "  Lower: {:.2}",
        quantile(&data_normal.view(), 0.4, QuantileInterpolation::Lower).unwrap()
    );
    println!(
        "  Higher: {:.2}",
        quantile(&data_normal.view(), 0.4, QuantileInterpolation::Higher).unwrap()
    );
    println!(
        "  Nearest: {:.2}",
        quantile(&data_normal.view(), 0.4, QuantileInterpolation::Nearest).unwrap()
    );
    println!(
        "  Midpoint: {:.2}",
        quantile(&data_normal.view(), 0.4, QuantileInterpolation::Midpoint).unwrap()
    );
    println!(
        "  Hazen: {:.2}",
        quantile(&data_normal.view(), 0.4, QuantileInterpolation::Hazen).unwrap()
    );
    println!(
        "  Weibull: {:.2}",
        quantile(&data_normal.view(), 0.4, QuantileInterpolation::Weibull).unwrap()
    );

    // Percentiles (same as quantiles but with 0-100 scale)
    println!("\nPercentiles:");
    println!(
        "  25th percentile: {:.2}",
        percentile(&data_normal.view(), 25.0, QuantileInterpolation::Linear).unwrap()
    );
    println!(
        "  50th percentile: {:.2}",
        percentile(&data_normal.view(), 50.0, QuantileInterpolation::Linear).unwrap()
    );
    println!(
        "  75th percentile: {:.2}",
        percentile(&data_normal.view(), 75.0, QuantileInterpolation::Linear).unwrap()
    );
    println!(
        "  90th percentile: {:.2}",
        percentile(&data_normal.view(), 90.0, QuantileInterpolation::Linear).unwrap()
    );

    // Quartiles, quintiles, and deciles
    println!(
        "\nQuartiles: {:?}",
        quartiles(&data_normal.view(), QuantileInterpolation::Linear).unwrap()
    );
    println!(
        "Quintiles: {:?}",
        quintiles(&data_normal.view(), QuantileInterpolation::Linear).unwrap()
    );
    println!(
        "Deciles: {:?}",
        deciles(&data_normal.view(), QuantileInterpolation::Linear).unwrap()
    );

    // Boxplot statistics
    println!("\n==== Boxplot Statistics ====");
    println!("\nData without outliers: {:?}", data_normal);
    let (q1, median, q3, whislo, whishi, outliers) = boxplot_stats(
        &data_normal.view(),
        Some(1.5),
        QuantileInterpolation::Linear,
    )
    .unwrap();
    println!("  Q1: {:.2}", q1);
    println!("  Median: {:.2}", median);
    println!("  Q3: {:.2}", q3);
    println!("  Lower whisker: {:.2}", whislo);
    println!("  Upper whisker: {:.2}", whishi);
    println!("  Outliers: {:?}", outliers);

    println!("\nData with outlier: {:?}", data_with_outlier);
    let (q1, median, q3, whislo, whishi, outliers) = boxplot_stats(
        &data_with_outlier.view(),
        Some(1.5),
        QuantileInterpolation::Linear,
    )
    .unwrap();
    println!("  Q1: {:.2}", q1);
    println!("  Median: {:.2}", median);
    println!("  Q3: {:.2}", q3);
    println!("  Lower whisker: {:.2}", whislo);
    println!("  Upper whisker: {:.2}", whishi);
    println!("  Outliers: {:?}", outliers);

    // Winsorized mean and variance
    println!("\n==== Winsorized Statistics ====");
    println!("\nRegular vs. winsorized statistics for data with outlier:");

    // Regular mean and variance
    let mean = data_with_outlier.sum() / data_with_outlier.len() as f64;
    let var = data_with_outlier
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f64>()
        / (data_with_outlier.len() - 1) as f64;

    println!("  Regular mean: {:.2}", mean);
    println!("  Regular variance (sample): {:.2}", var);

    // Winsorized mean and variance
    let win_mean_10 = winsorized_mean(&data_with_outlier.view(), 0.1).unwrap();
    let win_var_10 = winsorized_variance(&data_with_outlier.view(), 0.1, 1).unwrap();
    println!("  10% winsorized mean: {:.2}", win_mean_10);
    println!("  10% winsorized variance: {:.2}", win_var_10);

    let win_mean_20 = winsorized_mean(&data_with_outlier.view(), 0.2).unwrap();
    let win_var_20 = winsorized_variance(&data_with_outlier.view(), 0.2, 1).unwrap();
    println!("  20% winsorized mean: {:.2}", win_mean_20);
    println!("  20% winsorized variance: {:.2}", win_var_20);
}
