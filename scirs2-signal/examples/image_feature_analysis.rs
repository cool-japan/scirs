// Example demonstrating practical applications of image feature extraction
// for classification, segmentation, and analysis

use ndarray::Array2;
use rand::Rng;
use scirs2_signal::image_features::{extract_image_features, ImageFeatureOptions};
use std::collections::HashMap;

fn main() {
    println!("Image Feature Analysis Example");
    println!("=============================\n");

    // Generate synthetic images for analysis
    println!("Generating synthetic image dataset...");
    let (images, labels) = generate_synthetic_dataset(100, 32);
    println!("Generated {} images of size 32x32 pixels", images.len());

    // Extract features from all images
    println!("\nExtracting features from all images...");
    let features = extract_features_from_dataset(&images, &ImageFeatureOptions::default());

    // Analyze feature importance for classification
    println!("\nAnalyzing feature importance for classification...");
    analyze_feature_importance(&features, &labels);

    // Perform image segmentation using texture features
    println!("\nPerforming texture-based segmentation...");
    texture_based_segmentation();

    // Demonstrate region-based feature extraction
    println!("\nDemonstrating region-based feature extraction...");
    region_based_analysis();
}

// Generate a synthetic dataset of images for different classes
fn generate_synthetic_dataset(count: usize, size: usize) -> (Vec<Array2<f64>>, Vec<usize>) {
    let mut images = Vec::with_capacity(count);
    let mut labels = Vec::with_capacity(count);

    for i in 0..count {
        // Generate three different classes of images
        let class = i % 3;

        let mut image = Array2::zeros((size, size));

        match class {
            0 => {
                // Class 0: Gradient texture
                for i in 0..size {
                    for j in 0..size {
                        image[[i, j]] = (i as f64 / size as f64 + j as f64 / size as f64) * 127.5;
                    }
                }

                // Add random noise
                for i in 0..size {
                    for j in 0..size {
                        image[[i, j]] += rand::random::<f64>() * 20.0 - 10.0;
                    }
                }
            }
            1 => {
                // Class 1: Checkerboard pattern
                for i in 0..size {
                    for j in 0..size {
                        if (i / 4 + j / 4) % 2 == 0 {
                            image[[i, j]] = 200.0;
                        } else {
                            image[[i, j]] = 50.0;
                        }
                    }
                }

                // Add random noise
                for i in 0..size {
                    for j in 0..size {
                        image[[i, j]] += rand::random::<f64>() * 20.0 - 10.0;
                    }
                }
            }
            2 => {
                // Class 2: Circular pattern
                let center_x = size / 2;
                let center_y = size / 2;
                let radius = size / 4;

                for i in 0..size {
                    for j in 0..size {
                        let dx = i as isize - center_x as isize;
                        let dy = j as isize - center_y as isize;
                        let distance = ((dx * dx + dy * dy) as f64).sqrt();

                        if distance < radius as f64 {
                            image[[i, j]] = 180.0;
                        } else {
                            image[[i, j]] = 80.0;
                        }
                    }
                }

                // Add random noise
                for i in 0..size {
                    for j in 0..size {
                        image[[i, j]] += rand::random::<f64>() * 20.0 - 10.0;
                    }
                }
            }
            _ => unreachable!(),
        }

        images.push(image);
        labels.push(class);
    }

    (images, labels)
}

// Extract features from a dataset of images
fn extract_features_from_dataset(
    images: &[Array2<f64>],
    options: &ImageFeatureOptions,
) -> Vec<HashMap<String, f64>> {
    let mut all_features = Vec::with_capacity(images.len());

    for (i, image) in images.iter().enumerate() {
        match extract_image_features(image, options) {
            Ok(features) => {
                all_features.push(features);

                // Print progress for every 10% of images processed
                if (i + 1) % (images.len() / 10) == 0 || i + 1 == images.len() {
                    println!(
                        "  Processed {}/{} images ({:.1}%)",
                        i + 1,
                        images.len(),
                        (i + 1) as f64 / images.len() as f64 * 100.0
                    );
                }
            }
            Err(e) => {
                println!("Error extracting features for image {}: {:?}", i, e);
                all_features.push(HashMap::new());
            }
        }
    }

    all_features
}

// Analyze which features are most important for classification
fn analyze_feature_importance(features: &[HashMap<String, f64>], labels: &[usize]) {
    if features.is_empty() || labels.is_empty() || features.len() != labels.len() {
        println!("Invalid data for feature importance analysis");
        return;
    }

    // Get list of all features
    let all_feature_keys: Vec<String> = if !features[0].is_empty() {
        features[0].keys().cloned().collect()
    } else {
        println!("No features found for analysis");
        return;
    };

    // Calculate mean and standard deviation for each feature per class
    let n_classes = *labels.iter().max().unwrap() + 1;

    // For each feature, collect stats by class
    let mut feature_stats: HashMap<String, Vec<(f64, f64)>> = HashMap::new();

    for feature_name in all_feature_keys {
        let mut class_values: Vec<Vec<f64>> = vec![Vec::new(); n_classes];

        // Collect all values per class
        for (idx, label) in labels.iter().enumerate() {
            if let Some(value) = features[idx].get(&feature_name) {
                if value.is_finite() {
                    class_values[*label].push(*value);
                }
            }
        }

        // Calculate mean and stddev for each class
        let mut stats = Vec::with_capacity(n_classes);
        for class_idx in 0..n_classes {
            let values = &class_values[class_idx];
            if values.is_empty() {
                stats.push((0.0, 0.0)); // No data
                continue;
            }

            let sum: f64 = values.iter().sum();
            let mean = sum / values.len() as f64;

            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let stddev = variance.sqrt();

            stats.push((mean, stddev));
        }

        feature_stats.insert(feature_name, stats);
    }

    // Calculate feature importance based on class separation
    // A simple heuristic: (max difference between class means) / (average of standard deviations)
    let mut importance: Vec<(String, f64)> = Vec::new();

    for (feature_name, stats) in feature_stats {
        let means: Vec<f64> = stats.iter().map(|&(mean, _)| mean).collect();
        let stddevs: Vec<f64> = stats.iter().map(|&(_, stddev)| stddev).collect();

        // Find max difference between any two class means
        let mut max_diff = 0.0;
        for i in 0..means.len() {
            for j in i + 1..means.len() {
                let diff = (means[i] - means[j]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        // Calculate average standard deviation
        let avg_stddev = if !stddevs.is_empty() {
            stddevs.iter().sum::<f64>() / stddevs.len() as f64
        } else {
            1.0 // Avoid division by zero
        };

        // Calculate importance score
        let score = if avg_stddev > 1e-10 {
            max_diff / avg_stddev
        } else {
            max_diff * 1000.0 // Handle very small stddev
        };

        importance.push((feature_name, score));
    }

    // Sort by importance score
    importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Print the top features
    println!("\nTop 10 most important features for classification:");
    println!("{:<30} {:<15}", "Feature", "Importance Score");
    println!("{}", "-".repeat(45));

    for (_i, (feature, score)) in importance.iter().take(10).enumerate() {
        println!("{:<30} {:<15.4}", feature, score);
    }
}

// Demonstrate texture-based image segmentation
fn texture_based_segmentation() {
    // Create a synthetic image with different texture regions
    let size = 64;
    let mut image = Array2::zeros((size, size));

    // Region 1 (top-left): Smooth gradient
    for i in 0..size / 2 {
        for j in 0..size / 2 {
            image[[i, j]] = (i + j) as f64 / (size as f64) * 255.0;
        }
    }

    // Region 2 (top-right): Checkerboard
    for i in 0..size / 2 {
        for j in size / 2..size {
            if (i / 4 + j / 4) % 2 == 0 {
                image[[i, j]] = 220.0;
            } else {
                image[[i, j]] = 70.0;
            }
        }
    }

    // Region 3 (bottom-left): Circular pattern
    let center_x = size / 4 * 3;
    let center_y = size / 4 * 3;
    let _radius = size / 6;

    for i in size / 2..size {
        for j in 0..size / 2 {
            let dx = i as isize - center_x as isize;
            let dy = j as isize - center_y as isize;
            let distance = ((dx * dx + dy * dy) as f64).sqrt();

            if (distance / 4.0).floor() % 2.0 == 0.0 {
                image[[i, j]] = 200.0;
            } else {
                image[[i, j]] = 50.0;
            }
        }
    }

    // Region 4 (bottom-right): Random noise
    for i in size / 2..size {
        for j in size / 2..size {
            image[[i, j]] = rand::random::<f64>() * 200.0 + 25.0;
        }
    }

    // Add some random noise to the entire image
    for i in 0..size {
        for j in 0..size {
            image[[i, j]] += rand::random::<f64>() * 15.0 - 7.5;
            // Clamp values
            image[[i, j]] = image[[i, j]].max(0.0).min(255.0);
        }
    }

    println!("Created synthetic image with 4 texture regions");

    // Perform texture-based segmentation
    println!("Extracting texture features from local patches...");

    // Extract features from local patches
    let patch_size = 8;
    let step = 4;

    let options = ImageFeatureOptions {
        histogram: false,
        edges: false,
        moments: false,
        texture: true,
        haralick: true,
        lbp: true,
        histogram_bins: 8,
        cooccurrence_distance: 1,
        fast_mode: true,
        ..ImageFeatureOptions::default()
    };

    // Create a segmentation map based on texture features
    let mut segmentation = Array2::zeros((size, size));

    // Track feature vectors for each patch center
    let mut patch_features: Vec<((usize, usize), Vec<f64>)> = Vec::new();

    // Extract features for overlapping patches
    for i in 0..size - patch_size + 1 {
        if i % step != 0 {
            continue;
        }

        for j in 0..size - patch_size + 1 {
            if j % step != 0 {
                continue;
            }

            // Extract patch
            let patch =
                Array2::from_shape_fn((patch_size, patch_size), |(pi, pj)| image[[i + pi, j + pj]]);

            // Get features for this patch
            if let Ok(features) = extract_image_features(&patch, &options) {
                // Select key texture features
                let key_features = [
                    "texture_contrast",
                    "texture_energy",
                    "texture_coarseness",
                    "haralick_contrast",
                    "haralick_energy",
                    "haralick_homogeneity",
                    "lbp_energy",
                    "lbp_entropy",
                    "lbp_edges",
                ];

                let feature_vector: Vec<f64> = key_features
                    .iter()
                    .filter_map(|k| features.get(*k).copied())
                    .collect();

                if feature_vector.len() == key_features.len() {
                    // Store center coordinates and feature vector
                    let center = (i + patch_size / 2, j + patch_size / 2);
                    patch_features.push((center, feature_vector));
                }
            }
        }
    }

    println!("Extracted features from {} patches", patch_features.len());

    // Simple k-means clustering for segmentation (k=4 for our 4 regions)
    println!("Clustering texture features with k-means (k=4)...");

    // Initialize 4 random cluster centers
    let mut cluster_centers = Vec::with_capacity(4);
    for _ in 0..4 {
        let mut rng = rand::rng();
        let random_idx = rng.random_range(0..patch_features.len());
        cluster_centers.push(patch_features[random_idx].1.clone());
    }

    // Run k-means for a few iterations
    let iterations = 5;
    for iter in 0..iterations {
        // Assign patches to nearest cluster
        let mut cluster_assignments = vec![0; patch_features.len()];

        for (i, (_, features)) in patch_features.iter().enumerate() {
            let mut min_dist = f64::MAX;
            let mut best_cluster = 0;

            for (c, center) in cluster_centers.iter().enumerate() {
                // Calculate Euclidean distance
                let dist = features
                    .iter()
                    .zip(center.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = c;
                }
            }

            cluster_assignments[i] = best_cluster;
        }

        // Update cluster centers
        let mut new_centers = vec![vec![0.0; cluster_centers[0].len()]; cluster_centers.len()];
        let mut counts = vec![0; cluster_centers.len()];

        for (i, (_, features)) in patch_features.iter().enumerate() {
            let cluster = cluster_assignments[i];
            counts[cluster] += 1;

            for (j, &val) in features.iter().enumerate() {
                new_centers[cluster][j] += val;
            }
        }

        for (c, count) in counts.iter().enumerate() {
            if *count > 0 {
                for j in 0..new_centers[c].len() {
                    new_centers[c][j] /= *count as f64;
                }
            }
        }

        // Replace empty clusters with random points
        for (c, count) in counts.iter().enumerate() {
            if *count == 0 {
                let mut rng = rand::rng();
                let random_idx = rng.random_range(0..patch_features.len());
                new_centers[c] = patch_features[random_idx].1.clone();
            }
        }

        cluster_centers = new_centers;

        println!("  K-means iteration {}/{} complete", iter + 1, iterations);
    }

    // Create final segmentation map
    println!("Creating segmentation map...");

    // Assign each patch center to its cluster
    for (_i, ((center_i, center_j), features)) in patch_features.iter().enumerate() {
        let mut min_dist = f64::MAX;
        let mut best_cluster = 0;

        for (c, center) in cluster_centers.iter().enumerate() {
            // Calculate Euclidean distance
            let dist = features
                .iter()
                .zip(center.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if dist < min_dist {
                min_dist = dist;
                best_cluster = c;
            }
        }

        // Assign cluster to segmentation map
        segmentation[[*center_i, *center_j]] = best_cluster as f64;
    }

    // Interpolate to fill in gaps (simple nearest neighbor)
    for i in 0..size {
        for j in 0..size {
            if segmentation[[i, j]] == 0.0 {
                // Find nearest patch center
                let mut min_dist = f64::MAX;
                let mut best_cluster = 0;

                for ((center_i, center_j), _, cluster) in patch_features
                    .iter()
                    .enumerate()
                    .map(|(idx, ((ci, cj), _))| ((*ci, *cj), idx))
                    .map(|((ci, cj), idx)| {
                        let _i = patch_features
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i == idx)
                            .map(|(_, ((_, _), _))| idx)
                            .next()
                            .unwrap();
                        ((ci, cj), idx, segmentation[[ci, cj]])
                    })
                {
                    let dist = ((i as isize - center_i as isize).pow(2)
                        + (j as isize - center_j as isize).pow(2))
                        as f64;

                    if dist < min_dist && cluster > 0.0 {
                        min_dist = dist;
                        best_cluster = cluster as usize;
                    }
                }

                segmentation[[i, j]] = best_cluster as f64;
            }
        }
    }

    // Print segmentation results
    println!("Segmentation complete! Results for the 4 regions:");

    // Count pixels in each cluster for each region
    let mut region_stats = vec![vec![0; 4]; 4];

    // Region 1: Top-left
    for i in 0..size / 2 {
        for j in 0..size / 2 {
            let cluster = segmentation[[i, j]] as usize;
            if cluster < 4 {
                region_stats[0][cluster] += 1;
            }
        }
    }

    // Region 2: Top-right
    for i in 0..size / 2 {
        for j in size / 2..size {
            let cluster = segmentation[[i, j]] as usize;
            if cluster < 4 {
                region_stats[1][cluster] += 1;
            }
        }
    }

    // Region 3: Bottom-left
    for i in size / 2..size {
        for j in 0..size / 2 {
            let cluster = segmentation[[i, j]] as usize;
            if cluster < 4 {
                region_stats[2][cluster] += 1;
            }
        }
    }

    // Region 4: Bottom-right
    for i in size / 2..size {
        for j in size / 2..size {
            let cluster = segmentation[[i, j]] as usize;
            if cluster < 4 {
                region_stats[3][cluster] += 1;
            }
        }
    }

    // Print statistics
    println!("\nRegion Assignment Statistics (rows=regions, columns=clusters):");
    println!(
        "{:<15} {:<10} {:<10} {:<10} {:<10}",
        "Region", "Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"
    );
    println!("{}", "-".repeat(60));

    let region_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"];
    for (i, stats) in region_stats.iter().enumerate() {
        let total = stats.iter().sum::<usize>() as f64;
        let percentages: Vec<String> = stats
            .iter()
            .map(|&count| format!("{:.1}%", count as f64 / total * 100.0))
            .collect();

        println!(
            "{:<15} {:<10} {:<10} {:<10} {:<10}",
            region_names[i], percentages[0], percentages[1], percentages[2], percentages[3]
        );
    }
}

// Demonstrate region-based feature extraction
fn region_based_analysis() {
    println!("Creating a synthetic image with different regions...");

    // Create a synthetic image with distinct regions
    let size = 64;
    let mut image = Array2::zeros((size, size));

    // Define regions
    let regions = [
        // Rectangle 1 (dark)
        (10, 10, 25, 25, 50.0),
        // Rectangle 2 (medium)
        (35, 10, 55, 30, 120.0),
        // Rectangle 3 (bright)
        (10, 35, 30, 55, 200.0),
        // Circle (gradient)
        (40, 45, 15, 0, -1.0), // Special case for circle
    ];

    // Fill the background
    for i in 0..size {
        for j in 0..size {
            image[[i, j]] = 80.0;
        }
    }

    // Draw the regions
    for &(x1, y1, x2, y2, value) in &regions {
        if value < 0.0 {
            // Circle with gradient
            let radius = x2; // Last parameter is radius for circle
            let center_x = x1;
            let center_y = y1;

            for i in 0..size {
                for j in 0..size {
                    let dx = i as isize - center_x as isize;
                    let dy = j as isize - center_y as isize;
                    let distance = ((dx * dx + dy * dy) as f64).sqrt();

                    if distance < radius as f64 {
                        // Create a radial gradient
                        let intensity = 180.0 - distance / radius as f64 * 100.0;
                        image[[i, j]] = intensity;
                    }
                }
            }
        } else {
            // Rectangle
            for i in x1..x2 {
                for j in y1..y2 {
                    if i < size && j < size {
                        image[[i, j]] = value;
                    }
                }
            }
        }
    }

    // Add some noise
    for i in 0..size {
        for j in 0..size {
            image[[i, j]] += rand::random::<f64>() * 10.0 - 5.0;
            // Clamp values
            image[[i, j]] = image[[i, j]].max(0.0).min(255.0);
        }
    }

    println!("Image created with 4 regions: 3 rectangles and 1 circle");

    // Define region masks for analysis
    let mut region_masks = Vec::new();

    // Create masks for each region
    for (idx, &(x1, y1, x2, y2, value)) in regions.iter().enumerate() {
        let mut mask = Array2::zeros((size, size));

        if value < 0.0 {
            // Circle
            let radius = x2;
            let center_x = x1;
            let center_y = y1;

            for i in 0..size {
                for j in 0..size {
                    let dx = i as isize - center_x as isize;
                    let dy = j as isize - center_y as isize;
                    let distance = ((dx * dx + dy * dy) as f64).sqrt();

                    if distance < radius as f64 {
                        mask[[i, j]] = 1.0;
                    }
                }
            }
        } else {
            // Rectangle
            for i in x1..x2 {
                for j in y1..y2 {
                    if i < size && j < size {
                        mask[[i, j]] = 1.0;
                    }
                }
            }
        }

        region_masks.push((format!("Region {}", idx + 1), mask));
    }

    // Add a background mask
    let mut bg_mask = Array2::ones((size, size));
    for (_, mask) in &region_masks {
        for i in 0..size {
            for j in 0..size {
                if mask[[i, j]] > 0.0 {
                    bg_mask[[i, j]] = 0.0;
                }
            }
        }
    }
    region_masks.push(("Background".to_string(), bg_mask));

    println!("Extracting features for each region...");

    // Extract features for each region
    let options = ImageFeatureOptions {
        histogram: true,
        texture: true,
        edges: true,
        moments: true,
        haralick: false, // Haralick features need sufficient pixels
        lbp: false,      // LBP also needs sufficient pixels
        histogram_bins: 32,
        fast_mode: true,
        ..ImageFeatureOptions::default()
    };

    // Function to extract features from masked region
    let extract_region_features = |mask: &Array2<f64>| -> HashMap<String, f64> {
        // Create masked image
        let mut masked_image = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                if mask[[i, j]] > 0.0 {
                    masked_image[[i, j]] = image[[i, j]];
                }
            }
        }

        // Extract features
        match extract_image_features(&masked_image, &options) {
            Ok(features) => features,
            Err(_) => HashMap::new(),
        }
    };

    // Extract features for each region
    let mut region_features = Vec::new();
    for (name, mask) in &region_masks {
        let features = extract_region_features(mask);
        if !features.is_empty() {
            region_features.push((name.clone(), features));
        }
    }

    // Compare key features across regions
    println!("\nRegion Comparison:");
    println!("{}", "-".repeat(80));

    // Select key features to compare
    let key_features = [
        "intensity_mean",
        "intensity_std",
        "histogram_entropy",
        "texture_contrast",
        "texture_energy",
        "edge_mean_gradient",
    ];

    // Print header
    print!("{:<15}", "Feature");
    for (name, _) in &region_features {
        print!(" {:<12}", name);
    }
    println!();
    println!("{}", "-".repeat(80));

    // Print feature values
    for feature in &key_features {
        print!("{:<15}", feature);

        for (_, features) in &region_features {
            if let Some(value) = features.get(*feature) {
                print!(" {:<12.4}", value);
            } else {
                print!(" {:<12}", "N/A");
            }
        }
        println!();
    }

    println!("\nRegion-based analysis complete!");
}
