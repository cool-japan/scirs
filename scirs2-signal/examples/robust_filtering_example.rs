//! Example demonstrating robust filtering techniques for outlier removal
//!
//! This example shows how to use various robust filtering methods
//! to handle outliers and non-Gaussian noise in signals.

use ndarray::Array1;
use scirs2_signal::robust::{
    alpha_trimmed_filter, hampel_filter, huber_filter, robust_filter_2d, winsorize_filter,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Robust Filtering Examples");
    println!("=========================\n");

    // Create a test signal with outliers
    let clean_signal = vec![1.0, 1.2, 1.1, 1.3, 1.25, 1.15, 1.35, 1.1, 1.28, 1.22];
    let mut noisy_signal = clean_signal.clone();

    // Add some outliers
    noisy_signal[3] = 10.0; // Large positive outlier
    noisy_signal[7] = -5.0; // Large negative outlier

    let signal = Array1::from_vec(noisy_signal.clone());

    println!("Original signal with outliers:");
    println!("{:?}", noisy_signal);
    println!();

    // Example 1: Alpha-trimmed mean filter
    println!("1. Alpha-Trimmed Mean Filter");
    let alpha_filtered = alpha_trimmed_filter(&signal, 5, 0.2)?;
    println!("   Filtered signal: {:?}", alpha_filtered.to_vec());
    println!(
        "   Outlier reduction: {:.2} -> {:.2}, {:.2} -> {:.2}",
        noisy_signal[3], alpha_filtered[3], noisy_signal[7], alpha_filtered[7]
    );
    println!();

    // Example 2: Hampel filter
    println!("2. Hampel Filter (outlier detection)");
    let (hampel_filtered, outlier_indices) = hampel_filter(&signal, 5, 3.0)?;
    println!("   Filtered signal: {:?}", hampel_filtered.to_vec());
    println!("   Detected outlier indices: {:?}", outlier_indices);
    println!();

    // Example 3: Winsorize filter
    println!("3. Winsorize Filter");
    let winsorized = winsorize_filter(&signal, 5, 10.0)?;
    println!("   Filtered signal: {:?}", winsorized.to_vec());
    println!();

    // Example 4: Huber filter
    println!("4. Huber Loss-based Filter");
    let huber_filtered = huber_filter(&signal, 5, 1.35)?;
    println!("   Filtered signal: {:?}", huber_filtered.to_vec());
    println!();

    // Example 5: Comparison of methods
    println!("5. Method Comparison at Outlier Positions");
    println!("   Position 3 (outlier value 10.0):");
    println!("      Original: {:.2}", noisy_signal[3]);
    println!("      Alpha-trimmed: {:.2}", alpha_filtered[3]);
    println!("      Hampel: {:.2}", hampel_filtered[3]);
    println!("      Winsorize: {:.2}", winsorized[3]);
    println!("      Huber: {:.2}", huber_filtered[3]);
    println!();

    println!("   Position 7 (outlier value -5.0):");
    println!("      Original: {:.2}", noisy_signal[7]);
    println!("      Alpha-trimmed: {:.2}", alpha_filtered[7]);
    println!("      Hampel: {:.2}", hampel_filtered[7]);
    println!("      Winsorize: {:.2}", winsorized[7]);
    println!("      Huber: {:.2}", huber_filtered[7]);
    println!();

    // Example 6: 2D robust filtering demonstration
    println!("6. 2D Robust Filtering (Image Processing)");
    use ndarray::Array2;

    // Create a 2D signal (small image) with outliers
    let image_data = vec![
        1.0, 1.2, 1.1, 1.3, 1.1, 1.1, 10.0, 1.2, 1.4, 1.2, // Outlier at position (1,1)
        1.3, 1.1, 1.0, 1.2, 1.3, 1.2, 1.4, 1.1, -5.0, 1.1, // Outlier at position (3,3)
        1.1, 1.3, 1.2, 1.1, 1.0,
    ];

    let image = Array2::from_shape_vec((5, 5), image_data.clone())?;

    println!("   Original 2D signal:");
    for row in image.outer_iter() {
        println!("   {:?}", row.to_vec());
    }

    let filtered_2d = robust_filter_2d(&image, alpha_trimmed_filter, 3, 0.2)?;

    println!("   After 2D alpha-trimmed filtering:");
    for row in filtered_2d.outer_iter() {
        let formatted_row: Vec<String> = row.iter().map(|x| format!("{:.2}", x)).collect();
        println!("   {:?}", formatted_row);
    }

    println!("   Outlier reduction:");
    println!(
        "      Position (1,1): {:.2} -> {:.2}",
        image[[1, 1]],
        filtered_2d[[1, 1]]
    );
    println!(
        "      Position (3,3): {:.2} -> {:.2}",
        image[[3, 3]],
        filtered_2d[[3, 3]]
    );

    println!("\nRobust filtering examples completed successfully!");
    Ok(())
}
