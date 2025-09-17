//! Noise injection and data corruption utilities

use super::config::{MissingPattern, OutlierType};
use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Uniform;
use scirs2_core::rng;
use std::f64::consts::PI;

/// Inject missing data into a dataset with realistic patterns
#[allow(dead_code)]
pub fn inject_missing_data(
    data: &mut Array2<f64>,
    missing_rate: f64,
    pattern: MissingPattern,
    randomseed: Option<u64>,
) -> Result<Array2<bool>> {
    // Validate input parameters
    if !(0.0..=1.0).contains(&missing_rate) {
        return Err(DatasetsError::InvalidFormat(
            "missing_rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
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
                    if rng.gen_range(0.0f64..1.0) < missing_rate {
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
                    if rng.gen_range(0.0f64..1.0) < adjusted_rate {
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

                    if rng.gen_range(0.0f64..1.0) < adjusted_rate {
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
                let start_row = rng.sample(Uniform::new(0, n_samples).unwrap());
                let start_col =
                    rng.sample(Uniform::new(0, n_features.saturating_sub(block_size)).unwrap());

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
#[allow(dead_code)]
pub fn inject_outliers(
    data: &mut Array2<f64>,
    outlier_rate: f64,
    outlier_type: OutlierType,
    outlier_strength: f64,
    randomseed: Option<u64>,
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

    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
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
                let outlier_idx = rng.sample(Uniform::new(0, n_samples).unwrap());
                outlier_mask[outlier_idx] = true;

                // Modify each feature to be an outlier
                for j in 0..n_features {
                    let direction = if rng.gen_range(0.0f64..1.0) < 0.5 {
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
                let outlier_idx = rng.sample(Uniform::new(0, n_samples).unwrap());
                outlier_mask[outlier_idx] = true;

                // Only modify a subset of features to create contextual anomaly
                let n_features_to_modify =
                    rng.sample(Uniform::new(1, (n_features / 2).max(1) + 1).unwrap());
                let mut features_to_modify: Vec<usize> = (0..n_features).collect();
                features_to_modify.shuffle(&mut rng);
                features_to_modify.truncate(n_features_to_modify);

                for &j in &features_to_modify {
                    let direction = if rng.gen_range(0.0f64..1.0) < 0.5 {
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
                    let direction = if rng.gen_range(0.0f64..1.0) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    };
                    outlier_center[j] =
                        feature_means[j] + direction * outlier_strength * feature_stds[j];
                }

                // Generate points around this center
                for _ in 0..outliers_per_group {
                    let outlier_idx = rng.sample(Uniform::new(0, n_samples).unwrap());
                    outlier_mask[outlier_idx] = true;

                    for j in 0..n_features {
                        let noise = rng.gen_range(-0.5f64..0.5f64) * feature_stds[j];
                        data[[outlier_idx, j]] = outlier_center[j] + noise;
                    }
                }
            }
        }
    }

    Ok(outlier_mask)
}

/// Add realistic noise patterns to time series data
#[allow(dead_code)]
pub fn add_time_series_noise(
    data: &mut Array2<f64>,
    noise_types: &[(&str, f64)], // (noise_type, strength)
    randomseed: Option<u64>,
) -> Result<()> {
    let mut rng = match randomseed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
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
                    let spike_idx = rng.sample(Uniform::new(0, n_samples).unwrap());
                    let feature_idx = rng.sample(Uniform::new(0, n_features).unwrap());
                    let spike_magnitude = rng.gen_range(5.0..=15.0) * strength;
                    let direction = if rng.gen_range(0.0f64..1.0) < 0.5 {
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
                    "Unknown noise type: {noise_type}. Supported , types: gaussian, spikes, drift, seasonal, autocorrelated, heteroscedastic"
                )));
            }
        }
    }

    Ok(())
}

/// Generate a dataset with controlled corruption patterns
#[allow(dead_code)]
pub fn make_corrupted_dataset(
    base_dataset: &Dataset,
    missing_rate: f64,
    missing_pattern: MissingPattern,
    outlier_rate: f64,
    outlier_type: OutlierType,
    outlier_strength: f64,
    randomseed: Option<u64>,
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

    // Clone the base _dataset
    let mut corrupted_data = base_dataset.data.clone();
    let corrupted_target = base_dataset.target.clone();

    // Apply missing data
    let missing_mask = inject_missing_data(
        &mut corrupted_data,
        missing_rate,
        missing_pattern,
        randomseed,
    )?;

    // Apply outliers
    let outlier_mask = inject_outliers(
        &mut corrupted_data,
        outlier_rate,
        outlier_type,
        outlier_strength,
        randomseed,
    )?;

    // Create new _dataset with corruption metadata
    let mut corrupted_dataset = Dataset::new(corrupted_data, corrupted_target);

    if let Some(featurenames) = &base_dataset.featurenames {
        corrupted_dataset = corrupted_dataset.with_featurenames(featurenames.clone());
    }

    if let Some(targetnames) = &base_dataset.targetnames {
        corrupted_dataset = corrupted_dataset.with_targetnames(targetnames.clone());
    }

    corrupted_dataset = corrupted_dataset
        .with_description(format!(
            "Corrupted version of: {}",
            base_dataset
                .description
                .as_deref()
                .unwrap_or("Unknown _dataset")
        ))
        .with_metadata("missing_rate", &missing_rate.to_string())
        .with_metadata("missing_pattern", &format!("{missing_pattern:?}"))
        .with_metadata("outlier_rate", &outlier_rate.to_string())
        .with_metadata("outlier_type", &format!("{outlier_type:?}"))
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
