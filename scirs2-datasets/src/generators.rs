//! Dataset generators

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use std::f64::consts::PI;

/// Generate a random classification dataset with clusters
pub fn make_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_clusters_per_class: usize,
    n_informative: usize,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }

    if n_informative == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_informative must be > 0".to_string(),
        ));
    }

    if n_features < n_informative {
        return Err(DatasetsError::InvalidFormat(format!(
            "n_features ({}) must be >= n_informative ({})",
            n_features, n_informative
        )));
    }

    if n_classes < 2 {
        return Err(DatasetsError::InvalidFormat(
            "n_classes must be >= 2".to_string(),
        ));
    }

    if n_clusters_per_class == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_clusters_per_class must be > 0".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate centroids for each class and cluster
    let n_centroids = n_classes * n_clusters_per_class;
    let mut centroids = Array2::zeros((n_centroids, n_informative));
    let scale = 2.0;

    for i in 0..n_centroids {
        for j in 0..n_informative {
            centroids[[i, j]] = scale * rng.random_range(-1.0f64..1.0f64);
        }
    }

    // Generate samples
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    // Samples per class
    let samples_per_class = n_samples / n_classes;
    let remainder = n_samples % n_classes;

    let mut sample_idx = 0;

    for class in 0..n_classes {
        let n_samples_class = if class < remainder {
            samples_per_class + 1
        } else {
            samples_per_class
        };

        // Assign clusters within this class
        let samples_per_cluster = n_samples_class / n_clusters_per_class;
        let cluster_remainder = n_samples_class % n_clusters_per_class;

        for cluster in 0..n_clusters_per_class {
            let n_samples_cluster = if cluster < cluster_remainder {
                samples_per_cluster + 1
            } else {
                samples_per_cluster
            };

            let centroid_idx = class * n_clusters_per_class + cluster;

            for _ in 0..n_samples_cluster {
                // Randomly select a point near the cluster centroid
                for j in 0..n_informative {
                    data[[sample_idx, j]] =
                        centroids[[centroid_idx, j]] + 0.3 * normal.sample(&mut rng);
                }

                // Add noise features
                for j in n_informative..n_features {
                    data[[sample_idx, j]] = normal.sample(&mut rng);
                }

                target[sample_idx] = class as f64;
                sample_idx += 1;
            }
        }
    }

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Create feature names
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();

    // Create class names
    let class_names: Vec<String> = (0..n_classes).map(|i| format!("class_{}", i)).collect();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_target_names(class_names)
        .with_description(format!(
            "Synthetic classification dataset with {} classes and {} features",
            n_classes, n_features
        ));

    Ok(dataset)
}

/// Generate a random regression dataset
pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: f64,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }

    if n_informative == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_informative must be > 0".to_string(),
        ));
    }

    if n_features < n_informative {
        return Err(DatasetsError::InvalidFormat(format!(
            "n_features ({}) must be >= n_informative ({})",
            n_features, n_informative
        )));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0.0".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate the coefficients for the informative features
    let mut coef = Array1::zeros(n_features);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    for i in 0..n_informative {
        coef[i] = 100.0 * normal.sample(&mut rng);
    }

    // Generate the features
    let mut data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = normal.sample(&mut rng);
        }
    }

    // Generate the target
    let mut target = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let mut y = 0.0;
        for j in 0..n_features {
            y += data[[i, j]] * coef[j];
        }

        // Add noise
        if noise > 0.0 {
            y += normal.sample(&mut rng) * noise;
        }

        target[i] = y;
    }

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Create feature names
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_description(format!(
            "Synthetic regression dataset with {} features ({} informative)",
            n_features, n_informative
        ))
        .with_metadata("noise", &noise.to_string())
        .with_metadata("coefficients", &format!("{:?}", coef));

    Ok(dataset)
}

/// Generate a random time series dataset
pub fn make_time_series(
    n_samples: usize,
    n_features: usize,
    trend: bool,
    seasonality: bool,
    noise: f64,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0.0".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut data = Array2::zeros((n_samples, n_features));

    for feature in 0..n_features {
        let trend_coef = if trend {
            rng.random_range(0.01f64..0.1f64)
        } else {
            0.0
        };
        let seasonality_period = rng.random_range(10..=50) as f64;
        let seasonality_amplitude = if seasonality {
            rng.random_range(1.0f64..5.0f64)
        } else {
            0.0
        };

        let base_value = rng.random_range(-10.0f64..10.0f64);

        for i in 0..n_samples {
            let t = i as f64;

            // Add base value
            let mut value = base_value;

            // Add trend
            if trend {
                value += trend_coef * t;
            }

            // Add seasonality
            if seasonality {
                value += seasonality_amplitude * (2.0 * PI * t / seasonality_period).sin();
            }

            // Add noise
            if noise > 0.0 {
                value += normal.sample(&mut rng) * noise;
            }

            data[[i, feature]] = value;
        }
    }

    // Create time index (unused for now but can be useful for plotting)
    let time_index: Vec<f64> = (0..n_samples).map(|i| i as f64).collect();
    let _time_array = Array1::from(time_index);

    // Create dataset
    let mut dataset = Dataset::new(data, None);

    // Create feature names
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_description(format!(
            "Synthetic time series dataset with {} features",
            n_features
        ))
        .with_metadata("trend", &trend.to_string())
        .with_metadata("seasonality", &seasonality.to_string())
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a random blobs dataset for clustering
pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    cluster_std: f64,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }

    if centers == 0 {
        return Err(DatasetsError::InvalidFormat(
            "centers must be > 0".to_string(),
        ));
    }

    if cluster_std <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "cluster_std must be > 0.0".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate random centers
    let mut cluster_centers = Array2::zeros((centers, n_features));
    let center_box = 10.0;

    for i in 0..centers {
        for j in 0..n_features {
            cluster_centers[[i, j]] = rng.random_range(-center_box..=center_box);
        }
    }

    // Generate samples around centers
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    let normal = rand_distr::Normal::new(0.0, cluster_std).unwrap();

    // Samples per center
    let samples_per_center = n_samples / centers;
    let remainder = n_samples % centers;

    let mut sample_idx = 0;

    for center_idx in 0..centers {
        let n_samples_center = if center_idx < remainder {
            samples_per_center + 1
        } else {
            samples_per_center
        };

        for _ in 0..n_samples_center {
            for j in 0..n_features {
                data[[sample_idx, j]] = cluster_centers[[center_idx, j]] + normal.sample(&mut rng);
            }

            target[sample_idx] = center_idx as f64;
            sample_idx += 1;
        }
    }

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Create feature names
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_description(format!(
            "Synthetic clustering dataset with {} clusters and {} features",
            centers, n_features
        ))
        .with_metadata("centers", &centers.to_string())
        .with_metadata("cluster_std", &cluster_std.to_string());

    Ok(dataset)
}

/// Generate a spiral dataset for non-linear classification
pub fn make_spirals(
    n_samples: usize,
    n_spirals: usize,
    noise: f64,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_spirals == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_spirals must be > 0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0.0".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 2));
    let mut target = Array1::zeros(n_samples);

    let normal = if noise > 0.0 {
        Some(rand_distr::Normal::new(0.0, noise).unwrap())
    } else {
        None
    };

    let samples_per_spiral = n_samples / n_spirals;
    let remainder = n_samples % n_spirals;

    let mut sample_idx = 0;

    for spiral in 0..n_spirals {
        let n_samples_spiral = if spiral < remainder {
            samples_per_spiral + 1
        } else {
            samples_per_spiral
        };

        let spiral_offset = 2.0 * PI * spiral as f64 / n_spirals as f64;

        for i in 0..n_samples_spiral {
            let t = 2.0 * PI * i as f64 / n_samples_spiral as f64;
            let radius = t / (2.0 * PI);

            let mut x = radius * (t + spiral_offset).cos();
            let mut y = radius * (t + spiral_offset).sin();

            // Add noise if specified
            if let Some(ref normal_dist) = normal {
                x += normal_dist.sample(&mut rng);
                y += normal_dist.sample(&mut rng);
            }

            data[[sample_idx, 0]] = x;
            data[[sample_idx, 1]] = y;
            target[sample_idx] = spiral as f64;
            sample_idx += 1;
        }
    }

    let mut dataset = Dataset::new(data, Some(target));
    dataset = dataset
        .with_feature_names(vec!["x".to_string(), "y".to_string()])
        .with_target_names((0..n_spirals).map(|i| format!("spiral_{}", i)).collect())
        .with_description(format!("Spiral dataset with {} spirals", n_spirals))
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a moons dataset for non-linear classification
pub fn make_moons(n_samples: usize, noise: f64, random_seed: Option<u64>) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0.0".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 2));
    let mut target = Array1::zeros(n_samples);

    let normal = if noise > 0.0 {
        Some(rand_distr::Normal::new(0.0, noise).unwrap())
    } else {
        None
    };

    let samples_per_moon = n_samples / 2;
    let remainder = n_samples % 2;

    let mut sample_idx = 0;

    // Generate first moon (upper crescent)
    for i in 0..(samples_per_moon + remainder) {
        let t = PI * i as f64 / (samples_per_moon + remainder) as f64;

        let mut x = t.cos();
        let mut y = t.sin();

        // Add noise if specified
        if let Some(ref normal_dist) = normal {
            x += normal_dist.sample(&mut rng);
            y += normal_dist.sample(&mut rng);
        }

        data[[sample_idx, 0]] = x;
        data[[sample_idx, 1]] = y;
        target[sample_idx] = 0.0;
        sample_idx += 1;
    }

    // Generate second moon (lower crescent, flipped)
    for i in 0..samples_per_moon {
        let t = PI * i as f64 / samples_per_moon as f64;

        let mut x = 1.0 - t.cos();
        let mut y = 0.5 - t.sin(); // Offset vertically and flip

        // Add noise if specified
        if let Some(ref normal_dist) = normal {
            x += normal_dist.sample(&mut rng);
            y += normal_dist.sample(&mut rng);
        }

        data[[sample_idx, 0]] = x;
        data[[sample_idx, 1]] = y;
        target[sample_idx] = 1.0;
        sample_idx += 1;
    }

    let mut dataset = Dataset::new(data, Some(target));
    dataset = dataset
        .with_feature_names(vec!["x".to_string(), "y".to_string()])
        .with_target_names(vec!["moon_0".to_string(), "moon_1".to_string()])
        .with_description("Two moons dataset for non-linear classification".to_string())
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a circles dataset for non-linear classification
pub fn make_circles(
    n_samples: usize,
    factor: f64,
    noise: f64,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if factor <= 0.0 || factor >= 1.0 {
        return Err(DatasetsError::InvalidFormat(
            "factor must be between 0.0 and 1.0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0.0".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 2));
    let mut target = Array1::zeros(n_samples);

    let normal = if noise > 0.0 {
        Some(rand_distr::Normal::new(0.0, noise).unwrap())
    } else {
        None
    };

    let samples_per_circle = n_samples / 2;
    let remainder = n_samples % 2;

    let mut sample_idx = 0;

    // Generate outer circle
    for i in 0..(samples_per_circle + remainder) {
        let angle = 2.0 * PI * i as f64 / (samples_per_circle + remainder) as f64;

        let mut x = angle.cos();
        let mut y = angle.sin();

        // Add noise if specified
        if let Some(ref normal_dist) = normal {
            x += normal_dist.sample(&mut rng);
            y += normal_dist.sample(&mut rng);
        }

        data[[sample_idx, 0]] = x;
        data[[sample_idx, 1]] = y;
        target[sample_idx] = 0.0;
        sample_idx += 1;
    }

    // Generate inner circle (scaled by factor)
    for i in 0..samples_per_circle {
        let angle = 2.0 * PI * i as f64 / samples_per_circle as f64;

        let mut x = factor * angle.cos();
        let mut y = factor * angle.sin();

        // Add noise if specified
        if let Some(ref normal_dist) = normal {
            x += normal_dist.sample(&mut rng);
            y += normal_dist.sample(&mut rng);
        }

        data[[sample_idx, 0]] = x;
        data[[sample_idx, 1]] = y;
        target[sample_idx] = 1.0;
        sample_idx += 1;
    }

    let mut dataset = Dataset::new(data, Some(target));
    dataset = dataset
        .with_feature_names(vec!["x".to_string(), "y".to_string()])
        .with_target_names(vec!["outer_circle".to_string(), "inner_circle".to_string()])
        .with_description("Concentric circles dataset for non-linear classification".to_string())
        .with_metadata("factor", &factor.to_string())
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a Swiss roll dataset for dimensionality reduction
pub fn make_swiss_roll(n_samples: usize, noise: f64, random_seed: Option<u64>) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0.0".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut data = Array2::zeros((n_samples, 3));
    let mut color = Array1::zeros(n_samples); // Color parameter for visualization

    let normal = if noise > 0.0 {
        Some(rand_distr::Normal::new(0.0, noise).unwrap())
    } else {
        None
    };

    for i in 0..n_samples {
        // Parameter along the roll
        let t = 1.5 * PI * (1.0 + 2.0 * i as f64 / n_samples as f64);

        // Height parameter
        let height = 21.0 * i as f64 / n_samples as f64;

        let mut x = t * t.cos();
        let mut y = height;
        let mut z = t * t.sin();

        // Add noise if specified
        if let Some(ref normal_dist) = normal {
            x += normal_dist.sample(&mut rng);
            y += normal_dist.sample(&mut rng);
            z += normal_dist.sample(&mut rng);
        }

        data[[i, 0]] = x;
        data[[i, 1]] = y;
        data[[i, 2]] = z;
        color[i] = t; // Color based on parameter for visualization
    }

    let mut dataset = Dataset::new(data, Some(color));
    dataset = dataset
        .with_feature_names(vec!["x".to_string(), "y".to_string(), "z".to_string()])
        .with_description("Swiss roll manifold dataset for dimensionality reduction".to_string())
        .with_metadata("noise", &noise.to_string())
        .with_metadata("dimensions", "3")
        .with_metadata("manifold_dim", "2");

    Ok(dataset)
}

/// Generate anisotropic (elongated) clusters
pub fn make_anisotropic_blobs(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    cluster_std: f64,
    anisotropy_factor: f64,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_features < 2 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be >= 2 for anisotropic clusters".to_string(),
        ));
    }

    if centers == 0 {
        return Err(DatasetsError::InvalidFormat(
            "centers must be > 0".to_string(),
        ));
    }

    if cluster_std <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "cluster_std must be > 0.0".to_string(),
        ));
    }

    if anisotropy_factor <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "anisotropy_factor must be > 0.0".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate random centers
    let mut cluster_centers = Array2::zeros((centers, n_features));
    let center_box = 10.0;

    for i in 0..centers {
        for j in 0..n_features {
            cluster_centers[[i, j]] = rng.random_range(-center_box..=center_box);
        }
    }

    // Generate samples around centers with anisotropic distribution
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    let normal = rand_distr::Normal::new(0.0, cluster_std).unwrap();

    let samples_per_center = n_samples / centers;
    let remainder = n_samples % centers;

    let mut sample_idx = 0;

    for center_idx in 0..centers {
        let n_samples_center = if center_idx < remainder {
            samples_per_center + 1
        } else {
            samples_per_center
        };

        // Generate a random rotation angle for this cluster
        let rotation_angle = rng.random_range(0.0..(2.0 * PI));

        for _ in 0..n_samples_center {
            // Generate point with anisotropic distribution (elongated along first axis)
            let mut point = vec![0.0; n_features];

            // First axis has normal std, second axis has reduced std (anisotropy)
            point[0] = normal.sample(&mut rng);
            point[1] = normal.sample(&mut rng) / anisotropy_factor;

            // Remaining axes have normal std
            for item in point.iter_mut().take(n_features).skip(2) {
                *item = normal.sample(&mut rng);
            }

            // Apply rotation for 2D case
            if n_features >= 2 {
                let cos_theta = rotation_angle.cos();
                let sin_theta = rotation_angle.sin();

                let x_rot = cos_theta * point[0] - sin_theta * point[1];
                let y_rot = sin_theta * point[0] + cos_theta * point[1];

                point[0] = x_rot;
                point[1] = y_rot;
            }

            // Translate to cluster center
            for j in 0..n_features {
                data[[sample_idx, j]] = cluster_centers[[center_idx, j]] + point[j];
            }

            target[sample_idx] = center_idx as f64;
            sample_idx += 1;
        }
    }

    let mut dataset = Dataset::new(data, Some(target));
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_description(format!(
            "Anisotropic clustering dataset with {} elongated clusters and {} features",
            centers, n_features
        ))
        .with_metadata("centers", &centers.to_string())
        .with_metadata("cluster_std", &cluster_std.to_string())
        .with_metadata("anisotropy_factor", &anisotropy_factor.to_string());

    Ok(dataset)
}

/// Generate hierarchical clusters (clusters within clusters)
pub fn make_hierarchical_clusters(
    n_samples: usize,
    n_features: usize,
    n_main_clusters: usize,
    n_sub_clusters: usize,
    main_cluster_std: f64,
    sub_cluster_std: f64,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    // Validate input parameters
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }

    if n_main_clusters == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_main_clusters must be > 0".to_string(),
        ));
    }

    if n_sub_clusters == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_sub_clusters must be > 0".to_string(),
        ));
    }

    if main_cluster_std <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "main_cluster_std must be > 0.0".to_string(),
        ));
    }

    if sub_cluster_std <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "sub_cluster_std must be > 0.0".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Generate main cluster centers
    let mut main_centers = Array2::zeros((n_main_clusters, n_features));
    let center_box = 20.0;

    for i in 0..n_main_clusters {
        for j in 0..n_features {
            main_centers[[i, j]] = rng.random_range(-center_box..=center_box);
        }
    }

    let mut data = Array2::zeros((n_samples, n_features));
    let mut main_target = Array1::zeros(n_samples);
    let mut sub_target = Array1::zeros(n_samples);

    let main_normal = rand_distr::Normal::new(0.0, main_cluster_std).unwrap();
    let sub_normal = rand_distr::Normal::new(0.0, sub_cluster_std).unwrap();

    let samples_per_main = n_samples / n_main_clusters;
    let remainder = n_samples % n_main_clusters;

    let mut sample_idx = 0;

    for main_idx in 0..n_main_clusters {
        let n_samples_main = if main_idx < remainder {
            samples_per_main + 1
        } else {
            samples_per_main
        };

        // Generate sub-cluster centers within this main cluster
        let mut sub_centers = Array2::zeros((n_sub_clusters, n_features));
        for i in 0..n_sub_clusters {
            for j in 0..n_features {
                sub_centers[[i, j]] = main_centers[[main_idx, j]] + main_normal.sample(&mut rng);
            }
        }

        let samples_per_sub = n_samples_main / n_sub_clusters;
        let sub_remainder = n_samples_main % n_sub_clusters;

        for sub_idx in 0..n_sub_clusters {
            let n_samples_sub = if sub_idx < sub_remainder {
                samples_per_sub + 1
            } else {
                samples_per_sub
            };

            for _ in 0..n_samples_sub {
                for j in 0..n_features {
                    data[[sample_idx, j]] = sub_centers[[sub_idx, j]] + sub_normal.sample(&mut rng);
                }

                main_target[sample_idx] = main_idx as f64;
                sub_target[sample_idx] = (main_idx * n_sub_clusters + sub_idx) as f64;
                sample_idx += 1;
            }
        }
    }

    let mut dataset = Dataset::new(data, Some(main_target));
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_description(format!(
            "Hierarchical clustering dataset with {} main clusters, {} sub-clusters each",
            n_main_clusters, n_sub_clusters
        ))
        .with_metadata("n_main_clusters", &n_main_clusters.to_string())
        .with_metadata("n_sub_clusters", &n_sub_clusters.to_string())
        .with_metadata("main_cluster_std", &main_cluster_std.to_string())
        .with_metadata("sub_cluster_std", &sub_cluster_std.to_string())
        .with_metadata("sub_cluster_labels", &format!("{:?}", sub_target.to_vec()));

    Ok(dataset)
}

/// Missing data patterns for noise injection
#[derive(Debug, Clone, Copy)]
pub enum MissingPattern {
    /// Missing Completely at Random - uniform probability across all features
    MCAR,
    /// Missing at Random - probability depends on observed values
    MAR,
    /// Missing Not at Random - probability depends on missing values themselves
    MNAR,
    /// Block-wise missing - entire blocks of consecutive features/samples missing
    Block,
}

/// Outlier types for injection
#[derive(Debug, Clone, Copy)]
pub enum OutlierType {
    /// Point outliers - individual data points that are anomalous
    Point,
    /// Contextual outliers - points anomalous in specific contexts
    Contextual,
    /// Collective outliers - groups of points that together form an anomaly
    Collective,
}

/// Inject missing data into a dataset with realistic patterns
pub fn inject_missing_data(
    data: &mut Array2<f64>,
    missing_rate: f64,
    pattern: MissingPattern,
    random_seed: Option<u64>,
) -> Result<Array2<bool>> {
    // Validate input parameters
    if !(0.0..=1.0).contains(&missing_rate) {
        return Err(DatasetsError::InvalidFormat(
            "missing_rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let (n_samples, n_features) = data.dim();
    let mut missing_mask = Array2::from_elem((n_samples, n_features), false);

    match pattern {
        MissingPattern::MCAR => {
            // Missing Completely at Random - uniform probability
            for i in 0..n_samples {
                for j in 0..n_features {
                    if rng.random_range(0.0f64..1.0) < missing_rate {
                        missing_mask[[i, j]] = true;
                        data[[i, j]] = f64::NAN;
                    }
                }
            }
        }
        MissingPattern::MAR => {
            // Missing at Random - probability depends on first feature
            for i in 0..n_samples {
                let first_feature_val = data[[i, 0]];
                let normalized_val = (first_feature_val + 10.0) / 20.0; // Normalize roughly to [0,1]
                let adjusted_rate = missing_rate * normalized_val.clamp(0.1, 2.0);

                for j in 1..n_features {
                    // Skip first feature
                    if rng.random_range(0.0f64..1.0) < adjusted_rate {
                        missing_mask[[i, j]] = true;
                        data[[i, j]] = f64::NAN;
                    }
                }
            }
        }
        MissingPattern::MNAR => {
            // Missing Not at Random - higher values more likely to be missing
            for i in 0..n_samples {
                for j in 0..n_features {
                    let value = data[[i, j]];
                    let normalized_val = (value + 10.0) / 20.0; // Normalize roughly to [0,1]
                    let adjusted_rate = missing_rate * normalized_val.clamp(0.1, 3.0);

                    if rng.random_range(0.0f64..1.0) < adjusted_rate {
                        missing_mask[[i, j]] = true;
                        data[[i, j]] = f64::NAN;
                    }
                }
            }
        }
        MissingPattern::Block => {
            // Block-wise missing - entire blocks are missing
            let block_size = (n_features as f64 * missing_rate).ceil() as usize;
            let n_blocks = (missing_rate * n_samples as f64).ceil() as usize;

            for _ in 0..n_blocks {
                let start_row = rng.random_range(0..n_samples);
                let start_col = rng.random_range(0..n_features.saturating_sub(block_size));

                for i in start_row..n_samples.min(start_row + block_size) {
                    for j in start_col..n_features.min(start_col + block_size) {
                        missing_mask[[i, j]] = true;
                        data[[i, j]] = f64::NAN;
                    }
                }
            }
        }
    }

    Ok(missing_mask)
}

/// Inject outliers into a dataset
pub fn inject_outliers(
    data: &mut Array2<f64>,
    outlier_rate: f64,
    outlier_type: OutlierType,
    outlier_strength: f64,
    random_seed: Option<u64>,
) -> Result<Array1<bool>> {
    // Validate input parameters
    if !(0.0..=1.0).contains(&outlier_rate) {
        return Err(DatasetsError::InvalidFormat(
            "outlier_rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    if outlier_strength <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "outlier_strength must be > 0.0".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let (n_samples, n_features) = data.dim();
    let n_outliers = (n_samples as f64 * outlier_rate).ceil() as usize;
    let mut outlier_mask = Array1::from_elem(n_samples, false);

    // Calculate data statistics for outlier generation
    let mut feature_means = vec![0.0; n_features];
    let mut feature_stds = vec![0.0; n_features];

    for j in 0..n_features {
        let column = data.column(j);
        let valid_values: Vec<f64> = column.iter().filter(|&&x| !x.is_nan()).cloned().collect();

        if !valid_values.is_empty() {
            feature_means[j] = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
            let variance = valid_values
                .iter()
                .map(|&x| (x - feature_means[j]).powi(2))
                .sum::<f64>()
                / valid_values.len() as f64;
            feature_stds[j] = variance.sqrt().max(1.0); // Use minimum std of 1.0 to ensure outliers can be created
        }
    }

    match outlier_type {
        OutlierType::Point => {
            // Point outliers - individual anomalous points
            for _ in 0..n_outliers {
                let outlier_idx = rng.random_range(0..n_samples);
                outlier_mask[outlier_idx] = true;

                // Modify each feature to be an outlier
                for j in 0..n_features {
                    let direction = if rng.random_range(0.0f64..1.0) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    };
                    data[[outlier_idx, j]] =
                        feature_means[j] + direction * outlier_strength * feature_stds[j];
                }
            }
        }
        OutlierType::Contextual => {
            // Contextual outliers - anomalous in specific feature combinations
            for _ in 0..n_outliers {
                let outlier_idx = rng.random_range(0..n_samples);
                outlier_mask[outlier_idx] = true;

                // Only modify a subset of features to create contextual anomaly
                let n_features_to_modify = rng.random_range(1..=(n_features / 2).max(1));
                let mut features_to_modify: Vec<usize> = (0..n_features).collect();
                features_to_modify.shuffle(&mut rng);
                features_to_modify.truncate(n_features_to_modify);

                for &j in &features_to_modify {
                    let direction = if rng.random_range(0.0f64..1.0) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    };
                    data[[outlier_idx, j]] =
                        feature_means[j] + direction * outlier_strength * feature_stds[j];
                }
            }
        }
        OutlierType::Collective => {
            // Collective outliers - groups of points that together form anomalies
            let outliers_per_group = (n_outliers / 3).max(2); // At least 2 per group
            let n_groups = (n_outliers / outliers_per_group).max(1);

            for _ in 0..n_groups {
                // Generate cluster center for this collective outlier
                let mut outlier_center = vec![0.0; n_features];
                for j in 0..n_features {
                    let direction = if rng.random_range(0.0f64..1.0) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    };
                    outlier_center[j] =
                        feature_means[j] + direction * outlier_strength * feature_stds[j];
                }

                // Generate points around this center
                for _ in 0..outliers_per_group {
                    let outlier_idx = rng.random_range(0..n_samples);
                    outlier_mask[outlier_idx] = true;

                    for j in 0..n_features {
                        let noise = rng.random_range(-0.5f64..0.5f64) * feature_stds[j];
                        data[[outlier_idx, j]] = outlier_center[j] + noise;
                    }
                }
            }
        }
    }

    Ok(outlier_mask)
}

/// Add realistic noise patterns to time series data
pub fn add_time_series_noise(
    data: &mut Array2<f64>,
    noise_types: &[(&str, f64)], // (noise_type, strength)
    random_seed: Option<u64>,
) -> Result<()> {
    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let (n_samples, n_features) = data.dim();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    for &(noise_type, strength) in noise_types {
        match noise_type {
            "gaussian" => {
                // Add Gaussian white noise
                for i in 0..n_samples {
                    for j in 0..n_features {
                        data[[i, j]] += strength * normal.sample(&mut rng);
                    }
                }
            }
            "spikes" => {
                // Add random spikes (impulse noise)
                let n_spikes = (n_samples as f64 * strength * 0.1).ceil() as usize;
                for _ in 0..n_spikes {
                    let spike_idx = rng.random_range(0..n_samples);
                    let feature_idx = rng.random_range(0..n_features);
                    let spike_magnitude = rng.random_range(5.0..=15.0) * strength;
                    let direction = if rng.random_range(0.0f64..1.0) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    };

                    data[[spike_idx, feature_idx]] += direction * spike_magnitude;
                }
            }
            "drift" => {
                // Add gradual drift over time
                for i in 0..n_samples {
                    let drift_amount = strength * (i as f64 / n_samples as f64);
                    for j in 0..n_features {
                        data[[i, j]] += drift_amount;
                    }
                }
            }
            "seasonal" => {
                // Add seasonal pattern noise
                let period = n_samples as f64 / 4.0; // 4 seasons
                for i in 0..n_samples {
                    let seasonal_component = strength * (2.0 * PI * i as f64 / period).sin();
                    for j in 0..n_features {
                        data[[i, j]] += seasonal_component;
                    }
                }
            }
            "autocorrelated" => {
                // Add autocorrelated noise (AR(1) process)
                let ar_coeff = 0.7; // Autocorrelation coefficient
                for j in 0..n_features {
                    let mut prev_noise = 0.0;
                    for i in 0..n_samples {
                        let new_noise = ar_coeff * prev_noise + strength * normal.sample(&mut rng);
                        data[[i, j]] += new_noise;
                        prev_noise = new_noise;
                    }
                }
            }
            "heteroscedastic" => {
                // Add heteroscedastic noise (variance changes over time)
                for i in 0..n_samples {
                    let variance_factor = 1.0 + strength * (i as f64 / n_samples as f64);
                    for j in 0..n_features {
                        data[[i, j]] += variance_factor * strength * normal.sample(&mut rng);
                    }
                }
            }
            _ => {
                return Err(DatasetsError::InvalidFormat(format!(
                    "Unknown noise type: {}. Supported types: gaussian, spikes, drift, seasonal, autocorrelated, heteroscedastic",
                    noise_type
                )));
            }
        }
    }

    Ok(())
}

/// Generate a dataset with controlled corruption patterns
pub fn make_corrupted_dataset(
    base_dataset: &Dataset,
    missing_rate: f64,
    missing_pattern: MissingPattern,
    outlier_rate: f64,
    outlier_type: OutlierType,
    outlier_strength: f64,
    random_seed: Option<u64>,
) -> Result<Dataset> {
    // Validate inputs
    if !(0.0..=1.0).contains(&missing_rate) {
        return Err(DatasetsError::InvalidFormat(
            "missing_rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&outlier_rate) {
        return Err(DatasetsError::InvalidFormat(
            "outlier_rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Clone the base dataset
    let mut corrupted_data = base_dataset.data.clone();
    let corrupted_target = base_dataset.target.clone();

    // Apply missing data
    let missing_mask = inject_missing_data(
        &mut corrupted_data,
        missing_rate,
        missing_pattern,
        random_seed,
    )?;

    // Apply outliers
    let outlier_mask = inject_outliers(
        &mut corrupted_data,
        outlier_rate,
        outlier_type,
        outlier_strength,
        random_seed,
    )?;

    // Create new dataset with corruption metadata
    let mut corrupted_dataset = Dataset::new(corrupted_data, corrupted_target);

    if let Some(feature_names) = &base_dataset.feature_names {
        corrupted_dataset = corrupted_dataset.with_feature_names(feature_names.clone());
    }

    if let Some(target_names) = &base_dataset.target_names {
        corrupted_dataset = corrupted_dataset.with_target_names(target_names.clone());
    }

    corrupted_dataset = corrupted_dataset
        .with_description(format!(
            "Corrupted version of: {}",
            base_dataset
                .description
                .as_deref()
                .unwrap_or("Unknown dataset")
        ))
        .with_metadata("missing_rate", &missing_rate.to_string())
        .with_metadata("missing_pattern", &format!("{:?}", missing_pattern))
        .with_metadata("outlier_rate", &outlier_rate.to_string())
        .with_metadata("outlier_type", &format!("{:?}", outlier_type))
        .with_metadata("outlier_strength", &outlier_strength.to_string())
        .with_metadata(
            "missing_count",
            &missing_mask.iter().filter(|&&x| x).count().to_string(),
        )
        .with_metadata(
            "outlier_count",
            &outlier_mask.iter().filter(|&&x| x).count().to_string(),
        );

    Ok(corrupted_dataset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_classification_invalid_params() {
        // Test zero n_samples
        assert!(make_classification(0, 5, 2, 1, 3, None).is_err());

        // Test zero n_features
        assert!(make_classification(10, 0, 2, 1, 3, None).is_err());

        // Test zero n_informative
        assert!(make_classification(10, 5, 2, 1, 0, None).is_err());

        // Test n_features < n_informative
        assert!(make_classification(10, 3, 2, 1, 5, None).is_err());

        // Test n_classes < 2
        assert!(make_classification(10, 5, 1, 1, 3, None).is_err());

        // Test zero n_clusters_per_class
        assert!(make_classification(10, 5, 2, 0, 3, None).is_err());
    }

    #[test]
    fn test_make_regression_invalid_params() {
        // Test zero n_samples
        assert!(make_regression(0, 5, 3, 1.0, None).is_err());

        // Test zero n_features
        assert!(make_regression(10, 0, 3, 1.0, None).is_err());

        // Test zero n_informative
        assert!(make_regression(10, 5, 0, 1.0, None).is_err());

        // Test n_features < n_informative
        assert!(make_regression(10, 3, 5, 1.0, None).is_err());

        // Test negative noise
        assert!(make_regression(10, 5, 3, -1.0, None).is_err());
    }

    #[test]
    fn test_make_time_series_invalid_params() {
        // Test zero n_samples
        assert!(make_time_series(0, 3, false, false, 1.0, None).is_err());

        // Test zero n_features
        assert!(make_time_series(10, 0, false, false, 1.0, None).is_err());

        // Test negative noise
        assert!(make_time_series(10, 3, false, false, -1.0, None).is_err());
    }

    #[test]
    fn test_make_blobs_invalid_params() {
        // Test zero n_samples
        assert!(make_blobs(0, 3, 2, 1.0, None).is_err());

        // Test zero n_features
        assert!(make_blobs(10, 0, 2, 1.0, None).is_err());

        // Test zero centers
        assert!(make_blobs(10, 3, 0, 1.0, None).is_err());

        // Test zero or negative cluster_std
        assert!(make_blobs(10, 3, 2, 0.0, None).is_err());
        assert!(make_blobs(10, 3, 2, -1.0, None).is_err());
    }

    #[test]
    fn test_make_classification_valid_params() {
        let dataset = make_classification(20, 5, 3, 2, 4, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 20);
        assert_eq!(dataset.n_features(), 5);
        assert!(dataset.target.is_some());
        assert!(dataset.feature_names.is_some());
        assert!(dataset.target_names.is_some());
    }

    #[test]
    fn test_make_regression_valid_params() {
        let dataset = make_regression(15, 4, 3, 0.5, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 15);
        assert_eq!(dataset.n_features(), 4);
        assert!(dataset.target.is_some());
        assert!(dataset.feature_names.is_some());
    }

    #[test]
    fn test_make_time_series_valid_params() {
        let dataset = make_time_series(25, 3, true, true, 0.1, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 25);
        assert_eq!(dataset.n_features(), 3);
        assert!(dataset.feature_names.is_some());
        // Time series doesn't have targets by default
        assert!(dataset.target.is_none());
    }

    #[test]
    fn test_make_blobs_valid_params() {
        let dataset = make_blobs(30, 4, 3, 1.5, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 30);
        assert_eq!(dataset.n_features(), 4);
        assert!(dataset.target.is_some());
        assert!(dataset.feature_names.is_some());
    }

    #[test]
    fn test_make_spirals_invalid_params() {
        // Test zero n_samples
        assert!(make_spirals(0, 2, 0.1, None).is_err());

        // Test zero n_spirals
        assert!(make_spirals(100, 0, 0.1, None).is_err());

        // Test negative noise
        assert!(make_spirals(100, 2, -0.1, None).is_err());
    }

    #[test]
    fn test_make_spirals_valid_params() {
        let dataset = make_spirals(100, 2, 0.1, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 2);
        assert!(dataset.target.is_some());
        assert!(dataset.feature_names.is_some());

        // Check that we have the right number of spirals
        if let Some(target) = &dataset.target {
            let unique_labels: std::collections::HashSet<_> =
                target.iter().map(|&x| x as i32).collect();
            assert_eq!(unique_labels.len(), 2);
        }
    }

    #[test]
    fn test_make_moons_invalid_params() {
        // Test zero n_samples
        assert!(make_moons(0, 0.1, None).is_err());

        // Test negative noise
        assert!(make_moons(100, -0.1, None).is_err());
    }

    #[test]
    fn test_make_moons_valid_params() {
        let dataset = make_moons(100, 0.1, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 2);
        assert!(dataset.target.is_some());
        assert!(dataset.feature_names.is_some());

        // Check that we have exactly 2 classes (2 moons)
        if let Some(target) = &dataset.target {
            let unique_labels: std::collections::HashSet<_> =
                target.iter().map(|&x| x as i32).collect();
            assert_eq!(unique_labels.len(), 2);
        }
    }

    #[test]
    fn test_make_circles_invalid_params() {
        // Test zero n_samples
        assert!(make_circles(0, 0.5, 0.1, None).is_err());

        // Test invalid factor (must be between 0 and 1)
        assert!(make_circles(100, 0.0, 0.1, None).is_err());
        assert!(make_circles(100, 1.0, 0.1, None).is_err());
        assert!(make_circles(100, 1.5, 0.1, None).is_err());

        // Test negative noise
        assert!(make_circles(100, 0.5, -0.1, None).is_err());
    }

    #[test]
    fn test_make_circles_valid_params() {
        let dataset = make_circles(100, 0.5, 0.1, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 2);
        assert!(dataset.target.is_some());
        assert!(dataset.feature_names.is_some());

        // Check that we have exactly 2 classes (inner and outer circle)
        if let Some(target) = &dataset.target {
            let unique_labels: std::collections::HashSet<_> =
                target.iter().map(|&x| x as i32).collect();
            assert_eq!(unique_labels.len(), 2);
        }
    }

    #[test]
    fn test_make_swiss_roll_invalid_params() {
        // Test zero n_samples
        assert!(make_swiss_roll(0, 0.1, None).is_err());

        // Test negative noise
        assert!(make_swiss_roll(100, -0.1, None).is_err());
    }

    #[test]
    fn test_make_swiss_roll_valid_params() {
        let dataset = make_swiss_roll(100, 0.1, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 3);
        assert!(dataset.target.is_some()); // Color parameter
        assert!(dataset.feature_names.is_some());
    }

    #[test]
    fn test_make_anisotropic_blobs_invalid_params() {
        // Test zero n_samples
        assert!(make_anisotropic_blobs(0, 3, 2, 1.0, 2.0, None).is_err());

        // Test insufficient features
        assert!(make_anisotropic_blobs(100, 1, 2, 1.0, 2.0, None).is_err());

        // Test zero centers
        assert!(make_anisotropic_blobs(100, 3, 0, 1.0, 2.0, None).is_err());

        // Test invalid std
        assert!(make_anisotropic_blobs(100, 3, 2, 0.0, 2.0, None).is_err());

        // Test invalid anisotropy factor
        assert!(make_anisotropic_blobs(100, 3, 2, 1.0, 0.0, None).is_err());
    }

    #[test]
    fn test_make_anisotropic_blobs_valid_params() {
        let dataset = make_anisotropic_blobs(100, 3, 2, 1.0, 3.0, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 3);
        assert!(dataset.target.is_some());
        assert!(dataset.feature_names.is_some());

        // Check that we have the right number of clusters
        if let Some(target) = &dataset.target {
            let unique_labels: std::collections::HashSet<_> =
                target.iter().map(|&x| x as i32).collect();
            assert_eq!(unique_labels.len(), 2);
        }
    }

    #[test]
    fn test_make_hierarchical_clusters_invalid_params() {
        // Test zero n_samples
        assert!(make_hierarchical_clusters(0, 3, 2, 3, 1.0, 0.5, None).is_err());

        // Test zero features
        assert!(make_hierarchical_clusters(100, 0, 2, 3, 1.0, 0.5, None).is_err());

        // Test zero main clusters
        assert!(make_hierarchical_clusters(100, 3, 0, 3, 1.0, 0.5, None).is_err());

        // Test zero sub clusters
        assert!(make_hierarchical_clusters(100, 3, 2, 0, 1.0, 0.5, None).is_err());

        // Test invalid main cluster std
        assert!(make_hierarchical_clusters(100, 3, 2, 3, 0.0, 0.5, None).is_err());

        // Test invalid sub cluster std
        assert!(make_hierarchical_clusters(100, 3, 2, 3, 1.0, 0.0, None).is_err());
    }

    #[test]
    fn test_make_hierarchical_clusters_valid_params() {
        let dataset = make_hierarchical_clusters(120, 3, 2, 3, 2.0, 0.5, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 120);
        assert_eq!(dataset.n_features(), 3);
        assert!(dataset.target.is_some());
        assert!(dataset.feature_names.is_some());

        // Check that we have the right number of main clusters
        if let Some(target) = &dataset.target {
            let unique_labels: std::collections::HashSet<_> =
                target.iter().map(|&x| x as i32).collect();
            assert_eq!(unique_labels.len(), 2); // 2 main clusters
        }

        // Check metadata contains sub-cluster information
        assert!(dataset.metadata.contains_key("sub_cluster_labels"));
    }

    #[test]
    fn test_inject_missing_data_invalid_params() {
        let mut data = Array2::from_shape_vec((5, 3), vec![1.0; 15]).unwrap();

        // Test invalid missing rate
        assert!(inject_missing_data(&mut data, -0.1, MissingPattern::MCAR, None).is_err());
        assert!(inject_missing_data(&mut data, 1.5, MissingPattern::MCAR, None).is_err());
    }

    #[test]
    fn test_inject_missing_data_mcar() {
        let mut data =
            Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect()).unwrap();
        let original_data = data.clone();

        let missing_mask =
            inject_missing_data(&mut data, 0.3, MissingPattern::MCAR, Some(42)).unwrap();

        // Check that some data is missing
        let missing_count = missing_mask.iter().filter(|&&x| x).count();
        assert!(missing_count > 0);

        // Check that missing values are NaN
        for ((i, j), &is_missing) in missing_mask.indexed_iter() {
            if is_missing {
                assert!(data[[i, j]].is_nan());
            } else {
                assert_eq!(data[[i, j]], original_data[[i, j]]);
            }
        }
    }

    #[test]
    fn test_inject_outliers_invalid_params() {
        let mut data = Array2::from_shape_vec((5, 3), vec![1.0; 15]).unwrap();

        // Test invalid outlier rate
        assert!(inject_outliers(&mut data, -0.1, OutlierType::Point, 2.0, None).is_err());
        assert!(inject_outliers(&mut data, 1.5, OutlierType::Point, 2.0, None).is_err());

        // Test invalid outlier strength
        assert!(inject_outliers(&mut data, 0.1, OutlierType::Point, 0.0, None).is_err());
        assert!(inject_outliers(&mut data, 0.1, OutlierType::Point, -1.0, None).is_err());
    }

    #[test]
    fn test_inject_outliers_point() {
        let mut data = Array2::from_shape_vec((20, 2), vec![1.0; 40]).unwrap();

        let outlier_mask =
            inject_outliers(&mut data, 0.2, OutlierType::Point, 3.0, Some(42)).unwrap();

        // Check that some outliers were created
        let outlier_count = outlier_mask.iter().filter(|&&x| x).count();
        assert!(outlier_count > 0);

        // Check that outliers are different from original values
        for (i, &is_outlier) in outlier_mask.iter().enumerate() {
            if is_outlier {
                // At least one feature should be different from 1.0
                let row = data.row(i);
                assert!(row.iter().any(|&x| (x - 1.0).abs() > 1.0));
            }
        }
    }

    #[test]
    fn test_add_time_series_noise() {
        let mut data = Array2::zeros((100, 2));

        let noise_types = [("gaussian", 0.1), ("spikes", 0.05), ("drift", 0.2)];

        let original_data = data.clone();
        add_time_series_noise(&mut data, &noise_types, Some(42)).unwrap();

        // Check that data has been modified
        assert!(!data
            .iter()
            .zip(original_data.iter())
            .all(|(&a, &b)| (a - b).abs() < 1e-10));

        // Test invalid noise type
        let invalid_noise = [("invalid_type", 0.1)];
        let mut test_data = Array2::zeros((10, 2));
        assert!(add_time_series_noise(&mut test_data, &invalid_noise, Some(42)).is_err());
    }

    #[test]
    fn test_make_corrupted_dataset() {
        let base_dataset = make_blobs(50, 3, 2, 1.0, Some(42)).unwrap();

        let corrupted = make_corrupted_dataset(
            &base_dataset,
            0.1, // 10% missing
            MissingPattern::MCAR,
            0.05, // 5% outliers
            OutlierType::Point,
            2.0, // outlier strength
            Some(42),
        )
        .unwrap();

        // Check basic properties
        assert_eq!(corrupted.n_samples(), base_dataset.n_samples());
        assert_eq!(corrupted.n_features(), base_dataset.n_features());

        // Check metadata
        assert!(corrupted.metadata.contains_key("missing_rate"));
        assert!(corrupted.metadata.contains_key("outlier_rate"));
        assert!(corrupted.metadata.contains_key("missing_count"));
        assert!(corrupted.metadata.contains_key("outlier_count"));

        // Check some data is corrupted
        let has_nan = corrupted.data.iter().any(|&x| x.is_nan());
        assert!(has_nan, "Dataset should have some missing values");
    }

    #[test]
    fn test_make_corrupted_dataset_invalid_params() {
        let base_dataset = make_blobs(20, 2, 2, 1.0, Some(42)).unwrap();

        // Test invalid missing rate
        assert!(make_corrupted_dataset(
            &base_dataset,
            -0.1,
            MissingPattern::MCAR,
            0.0,
            OutlierType::Point,
            1.0,
            None
        )
        .is_err());
        assert!(make_corrupted_dataset(
            &base_dataset,
            1.5,
            MissingPattern::MCAR,
            0.0,
            OutlierType::Point,
            1.0,
            None
        )
        .is_err());

        // Test invalid outlier rate
        assert!(make_corrupted_dataset(
            &base_dataset,
            0.0,
            MissingPattern::MCAR,
            -0.1,
            OutlierType::Point,
            1.0,
            None
        )
        .is_err());
        assert!(make_corrupted_dataset(
            &base_dataset,
            0.0,
            MissingPattern::MCAR,
            1.5,
            OutlierType::Point,
            1.0,
            None
        )
        .is_err());
    }

    #[test]
    fn test_missing_patterns() {
        let data = Array2::from_shape_vec((20, 4), (0..80).map(|x| x as f64).collect()).unwrap();

        // Test different missing patterns
        for pattern in [
            MissingPattern::MCAR,
            MissingPattern::MAR,
            MissingPattern::MNAR,
            MissingPattern::Block,
        ] {
            let mut test_data = data.clone();
            let missing_mask = inject_missing_data(&mut test_data, 0.2, pattern, Some(42)).unwrap();

            let missing_count = missing_mask.iter().filter(|&&x| x).count();
            assert!(
                missing_count > 0,
                "Pattern {:?} should create some missing values",
                pattern
            );
        }
    }

    #[test]
    fn test_outlier_types() {
        let data = Array2::ones((30, 3));

        // Test different outlier types
        for outlier_type in [
            OutlierType::Point,
            OutlierType::Contextual,
            OutlierType::Collective,
        ] {
            let mut test_data = data.clone();
            let outlier_mask =
                inject_outliers(&mut test_data, 0.2, outlier_type, 3.0, Some(42)).unwrap();

            let outlier_count = outlier_mask.iter().filter(|&&x| x).count();
            assert!(
                outlier_count > 0,
                "Outlier type {:?} should create some outliers",
                outlier_type
            );
        }
    }
}
