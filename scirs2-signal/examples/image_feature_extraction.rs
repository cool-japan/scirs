// Example demonstrating image feature extraction capabilities
// This example creates synthetic images and extracts various features

use ndarray::{Array2, Array3};
use scirs2_signal::image_features::{
    extract_color_image_features, extract_image_features, ImageFeatureOptions,
};
use std::collections::HashMap;

fn main() {
    println!("Image Feature Extraction Example");
    println!("================================\n");

    // Part 1: Grayscale image feature extraction
    println!("1. Grayscale Image Features:");
    println!("---------------------------");

    // Create a synthetic grayscale image with a pattern
    let size = 64;
    let mut grayscale_image = Array2::zeros((size, size));

    // Create a gradient pattern with a circle in the center
    let center_x = size / 2;
    let center_y = size / 2;
    let radius = size / 4;

    for i in 0..size {
        for j in 0..size {
            // Base gradient
            let gradient = (i + j) as f64 / (2 * size) as f64 * 255.0;

            // Add a circle
            let dx = i as isize - center_x as isize;
            let dy = j as isize - center_y as isize;
            let distance = ((dx * dx + dy * dy) as f64).sqrt();

            if distance < radius as f64 {
                grayscale_image[[i, j]] = 200.0; // Circle interior
            } else {
                grayscale_image[[i, j]] = gradient;
            }
        }
    }

    // Extract features with default options
    let options = ImageFeatureOptions::default();
    match extract_image_features(&grayscale_image, &options) {
        Ok(features) => {
            // Print some selected features
            print_selected_features(&features, "Grayscale Image");
        }
        Err(e) => println!("Error extracting features: {:?}", e),
    }

    // Part 2: Color image feature extraction
    println!("\n2. Color Image Features:");
    println!("----------------------");

    // Create a synthetic color image
    let mut color_image = Array3::zeros((size, size, 3));

    for i in 0..size {
        for j in 0..size {
            // Create color gradients for the RGB channels
            let r_gradient = i as f64 / size as f64 * 255.0;
            let g_gradient = j as f64 / size as f64 * 255.0;
            let b_gradient = ((i as f64 / size as f64) * (j as f64 / size as f64)) * 255.0;

            // Add a color circle in the center
            let dx = i as isize - center_x as isize;
            let dy = j as isize - center_y as isize;
            let distance = ((dx * dx + dy * dy) as f64).sqrt();

            if distance < radius as f64 {
                // Inside the circle - red dominant
                color_image[[i, j, 0]] = 200.0; // R
                color_image[[i, j, 1]] = 50.0; // G
                color_image[[i, j, 2]] = 50.0; // B
            } else {
                // Outside the circle - gradients
                color_image[[i, j, 0]] = r_gradient;
                color_image[[i, j, 1]] = g_gradient;
                color_image[[i, j, 2]] = b_gradient;
            }
        }
    }

    // Extract color image features
    match extract_color_image_features(&color_image, &options) {
        Ok(features) => {
            // Print selected color features
            print_selected_features(&features, "Color Image");
        }
        Err(e) => println!("Error extracting color features: {:?}", e),
    }

    // Part 3: Compare different textures
    println!("\n3. Texture Analysis Comparison:");
    println!("-----------------------------");

    // Create two different texture patterns
    let mut smooth_texture = Array2::zeros((size, size));
    let mut rough_texture = Array2::zeros((size, size));

    for i in 0..size {
        for j in 0..size {
            // Smooth gradient
            smooth_texture[[i, j]] = (i as f64 + j as f64) / (2.0 * size as f64) * 255.0;

            // Rough checkerboard pattern
            if (i / 4 + j / 4) % 2 == 0 {
                rough_texture[[i, j]] = 255.0;
            } else {
                rough_texture[[i, j]] = 50.0;
            }
        }
    }

    // Extract and compare texture features
    let texture_options = ImageFeatureOptions {
        histogram: false,
        edges: false,
        moments: false,
        lbp: false,
        haralick: true,
        texture: true,
        ..ImageFeatureOptions::default()
    };

    let smooth_features = extract_image_features(&smooth_texture, &texture_options).unwrap();
    let rough_features = extract_image_features(&rough_texture, &texture_options).unwrap();

    println!("Smooth vs Rough Texture Features:");
    compare_features(&smooth_features, &rough_features);
}

// Helper function to print selected features
fn print_selected_features(features: &HashMap<String, f64>, image_type: &str) {
    println!("\nSelected features for {}:", image_type);
    println!("------------------------------");

    // Define categories of features to print
    let categories = [
        (
            "Basic Statistics",
            vec![
                "intensity_mean",
                "intensity_std",
                "intensity_skewness",
                "intensity_kurtosis",
            ],
        ),
        (
            "Histogram Features",
            vec!["histogram_entropy", "histogram_uniformity"],
        ),
        (
            "Edge Features",
            vec!["edge_mean_gradient", "edge_percentage"],
        ),
        (
            "Texture Features",
            vec!["texture_contrast", "texture_energy", "texture_coarseness"],
        ),
        (
            "Haralick Features",
            vec![
                "haralick_contrast",
                "haralick_energy",
                "haralick_correlation",
                "haralick_homogeneity",
            ],
        ),
        (
            "LBP Features",
            vec!["lbp_energy", "lbp_entropy", "lbp_edges", "lbp_corners"],
        ),
    ];

    if image_type.contains("Color") {
        // For color images, print color-specific features
        println!("\nColor Features:");
        println!("--------------");
        for key in &[
            "r_intensity_mean",
            "g_intensity_mean",
            "b_intensity_mean",
            "colorfulness",
            "color_homogeneity",
            "hue_entropy",
        ] {
            if let Some(value) = features.get(*key) {
                println!("{:<25}: {:.6}", key, value);
            }
        }
    }

    // Print features by category
    for (category, keys) in categories.iter() {
        let mut found = false;
        let mut category_printed = false;

        for key in keys {
            // For color images, look for color-specific keys
            let possible_keys = if image_type.contains("Color") {
                vec![
                    format!("r_{}", key),
                    format!("g_{}", key),
                    format!("b_{}", key),
                    format!("gray_{}", key),
                    key.to_string(),
                ]
            } else {
                vec![key.to_string()]
            };

            for possible_key in possible_keys {
                if let Some(value) = features.get(&possible_key) {
                    if !category_printed {
                        println!("\n{}:", category);
                        println!("{}", "-".repeat(category.len() + 1));
                        category_printed = true;
                    }
                    println!("{:<25}: {:.6}", possible_key, value);
                    found = true;
                    break;
                }
            }
        }

        if !found && !image_type.contains("Color") {
            println!("\n{}: No features found", category);
        }
    }
}

// Helper function to compare features between two textures
fn compare_features(features1: &HashMap<String, f64>, features2: &HashMap<String, f64>) {
    // Define the features to compare
    let texture_features = [
        "texture_contrast",
        "texture_energy",
        "texture_coarseness",
        "texture_directionality",
        "haralick_contrast",
        "haralick_correlation",
        "haralick_energy",
        "haralick_homogeneity",
        "haralick_entropy",
    ];

    // Print comparison
    println!(
        "{:<25} {:<15} {:<15} {:<15}",
        "Feature", "Smooth", "Rough", "Difference"
    );
    println!("{}", "-".repeat(70));

    for feature in texture_features.iter() {
        if let (Some(val1), Some(val2)) = (features1.get(*feature), features2.get(*feature)) {
            let diff = val2 - val1;
            println!(
                "{:<25} {:<15.6} {:<15.6} {:<15.6}",
                feature, val1, val2, diff
            );
        }
    }
}
