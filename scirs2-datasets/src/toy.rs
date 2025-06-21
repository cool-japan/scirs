//! Toy datasets for testing and examples
//!
//! This module provides small, synthetic datasets that are useful for
//! testing algorithms and illustrating concepts.

use crate::error::Result;
use crate::utils::Dataset;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rng;

/// Generate the classic Iris dataset
pub fn load_iris() -> Result<Dataset> {
    // Define the data
    #[rustfmt::skip]
    let data = Array2::from_shape_vec((150, 4), vec![
        5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0, 3.6, 1.4, 0.2,
        5.4, 3.9, 1.7, 0.4, 4.6, 3.4, 1.4, 0.3, 5.0, 3.4, 1.5, 0.2, 4.4, 2.9, 1.4, 0.2, 4.9, 3.1, 1.5, 0.1,
        5.4, 3.7, 1.5, 0.2, 4.8, 3.4, 1.6, 0.2, 4.8, 3.0, 1.4, 0.1, 4.3, 3.0, 1.1, 0.1, 5.8, 4.0, 1.2, 0.2,
        5.7, 4.4, 1.5, 0.4, 5.4, 3.9, 1.3, 0.4, 5.1, 3.5, 1.4, 0.3, 5.7, 3.8, 1.7, 0.3, 5.1, 3.8, 1.5, 0.3,
        5.4, 3.4, 1.7, 0.2, 5.1, 3.7, 1.5, 0.4, 4.6, 3.6, 1.0, 0.2, 5.1, 3.3, 1.7, 0.5, 4.8, 3.4, 1.9, 0.2,
        5.0, 3.0, 1.6, 0.2, 5.0, 3.4, 1.6, 0.4, 5.2, 3.5, 1.5, 0.2, 5.2, 3.4, 1.4, 0.2, 4.7, 3.2, 1.6, 0.2,
        4.8, 3.1, 1.6, 0.2, 5.4, 3.4, 1.5, 0.4, 5.2, 4.1, 1.5, 0.1, 5.5, 4.2, 1.4, 0.2, 4.9, 3.1, 1.5, 0.1,
        5.0, 3.2, 1.2, 0.2, 5.5, 3.5, 1.3, 0.2, 4.9, 3.1, 1.5, 0.1, 4.4, 3.0, 1.3, 0.2, 5.1, 3.4, 1.5, 0.2,
        5.0, 3.5, 1.3, 0.3, 4.5, 2.3, 1.3, 0.3, 4.4, 3.2, 1.3, 0.2, 5.0, 3.5, 1.6, 0.6, 5.1, 3.8, 1.9, 0.4,
        4.8, 3.0, 1.4, 0.3, 5.1, 3.8, 1.6, 0.2, 4.6, 3.2, 1.4, 0.2, 5.3, 3.7, 1.5, 0.2, 5.0, 3.3, 1.4, 0.2,
        7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3, 6.5, 2.8, 4.6, 1.5,
        5.7, 2.8, 4.5, 1.3, 6.3, 3.3, 4.7, 1.6, 4.9, 2.4, 3.3, 1.0, 6.6, 2.9, 4.6, 1.3, 5.2, 2.7, 3.9, 1.4,
        5.0, 2.0, 3.5, 1.0, 5.9, 3.0, 4.2, 1.5, 6.0, 2.2, 4.0, 1.0, 6.1, 2.9, 4.7, 1.4, 5.6, 2.9, 3.6, 1.3,
        6.7, 3.1, 4.4, 1.4, 5.6, 3.0, 4.5, 1.5, 5.8, 2.7, 4.1, 1.0, 6.2, 2.2, 4.5, 1.5, 5.6, 2.5, 3.9, 1.1,
        5.9, 3.2, 4.8, 1.8, 6.1, 2.8, 4.0, 1.3, 6.3, 2.5, 4.9, 1.5, 6.1, 2.8, 4.7, 1.2, 6.4, 2.9, 4.3, 1.3,
        6.6, 3.0, 4.4, 1.4, 6.8, 2.8, 4.8, 1.4, 6.7, 3.0, 5.0, 1.7, 6.0, 2.9, 4.5, 1.5, 5.7, 2.6, 3.5, 1.0,
        5.5, 2.4, 3.8, 1.1, 5.5, 2.4, 3.7, 1.0, 5.8, 2.7, 3.9, 1.2, 6.0, 2.7, 5.1, 1.6, 5.4, 3.0, 4.5, 1.5,
        6.0, 3.4, 4.5, 1.6, 6.7, 3.1, 4.7, 1.5, 6.3, 2.3, 4.4, 1.3, 5.6, 3.0, 4.1, 1.3, 5.5, 2.5, 4.0, 1.3,
        5.5, 2.6, 4.4, 1.2, 6.1, 3.0, 4.6, 1.4, 5.8, 2.6, 4.0, 1.2, 5.0, 2.3, 3.3, 1.0, 5.6, 2.7, 4.2, 1.3,
        5.7, 3.0, 4.2, 1.2, 5.7, 2.9, 4.2, 1.3, 6.2, 2.9, 4.3, 1.3, 5.1, 2.5, 3.0, 1.1, 5.7, 2.8, 4.1, 1.3,
        6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8, 6.5, 3.0, 5.8, 2.2,
        7.6, 3.0, 6.6, 2.1, 4.9, 2.5, 4.5, 1.7, 7.3, 2.9, 6.3, 1.8, 6.7, 2.5, 5.8, 1.8, 7.2, 3.6, 6.1, 2.5,
        6.5, 3.2, 5.1, 2.0, 6.4, 2.7, 5.3, 1.9, 6.8, 3.0, 5.5, 2.1, 5.7, 2.5, 5.0, 2.0, 5.8, 2.8, 5.1, 2.4,
        6.4, 3.2, 5.3, 2.3, 6.5, 3.0, 5.5, 1.8, 7.7, 3.8, 6.7, 2.2, 7.7, 2.6, 6.9, 2.3, 6.0, 2.2, 5.0, 1.5,
        6.9, 3.2, 5.7, 2.3, 5.6, 2.8, 4.9, 2.0, 7.7, 2.8, 6.7, 2.0, 6.3, 2.7, 4.9, 1.8, 6.7, 3.3, 5.7, 2.1,
        7.2, 3.2, 6.0, 1.8, 6.2, 2.8, 4.8, 1.8, 6.1, 3.0, 4.9, 1.8, 6.4, 2.8, 5.6, 2.1, 7.2, 3.0, 5.8, 1.6,
        7.4, 2.8, 6.1, 1.9, 7.9, 3.8, 6.4, 2.0, 6.4, 2.8, 5.6, 2.2, 6.3, 2.8, 5.1, 1.5, 6.1, 2.6, 5.6, 1.4,
        7.7, 3.0, 6.1, 2.3, 6.3, 3.4, 5.6, 2.4, 6.4, 3.1, 5.5, 1.8, 6.0, 3.0, 4.8, 1.8, 6.9, 3.1, 5.4, 2.1,
        6.7, 3.1, 5.6, 2.4, 6.9, 3.1, 5.1, 2.3, 5.8, 2.7, 5.1, 1.9, 6.8, 3.2, 5.9, 2.3, 6.7, 3.3, 5.7, 2.5,
        6.7, 3.0, 5.2, 2.3, 6.3, 2.5, 5.0, 1.9, 6.5, 3.0, 5.2, 2.0, 6.2, 3.4, 5.4, 2.3, 5.9, 3.0, 5.1, 1.8
    ]).unwrap();

    // Define the target (class labels)
    let targets = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];
    let target = Array1::from(targets);

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Add metadata
    let feature_names = vec![
        "sepal_length".to_string(),
        "sepal_width".to_string(),
        "petal_length".to_string(),
        "petal_width".to_string(),
    ];

    let target_names = vec![
        "setosa".to_string(),
        "versicolor".to_string(),
        "virginica".to_string(),
    ];

    let description = "Iris dataset: classic dataset for classification, clustering, and machine learning
    
The dataset contains 3 classes of 50 instances each, where each class refers to a type of iris plant.
One class is linearly separable from the other two; the latter are not linearly separable from each other.

Attributes:
- sepal length in cm
- sepal width in cm
- petal length in cm
- petal width in cm

Target: 
- Iris Setosa
- Iris Versicolour
- Iris Virginica".to_string();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_target_names(target_names)
        .with_description(description);

    Ok(dataset)
}

/// Generate the breast cancer dataset
pub fn load_breast_cancer() -> Result<Dataset> {
    // This is a simplified version with only 30 samples
    // In a real implementation, include the full dataset
    #[rustfmt::skip]
    let data = Array2::from_shape_vec((30, 5), vec![
        17.99, 10.38, 122.8, 1001.0, 0.1184,
        20.57, 17.77, 132.9, 1326.0, 0.08474,
        19.69, 21.25, 130.0, 1203.0, 0.1096,
        11.42, 20.38, 77.58, 386.1, 0.1425,
        20.29, 14.34, 135.1, 1297.0, 0.1003,
        12.45, 15.7, 82.57, 477.1, 0.1278,
        18.25, 19.98, 119.6, 1040.0, 0.09463,
        13.71, 20.83, 90.2, 577.9, 0.1189,
        13.0, 21.82, 87.5, 519.8, 0.1273,
        12.46, 24.04, 83.97, 475.9, 0.1186,
        16.02, 23.24, 102.7, 797.8, 0.08206,
        15.78, 17.89, 103.6, 781.0, 0.0971,
        19.17, 24.8, 132.4, 1123.0, 0.0974,
        15.85, 23.95, 103.7, 782.7, 0.08401,
        13.73, 22.61, 93.6, 578.3, 0.1131,
        14.54, 27.54, 96.73, 658.8, 0.1139,
        14.68, 20.13, 94.74, 684.5, 0.09867,
        16.13, 20.68, 108.1, 798.8, 0.117,
        19.81, 22.15, 130.0, 1260.0, 0.09831,
        13.54, 14.36, 87.46, 566.3, 0.09779,
        13.08, 15.71, 85.63, 520.0, 0.1075,
        9.504, 12.44, 60.34, 273.9, 0.1024,
        15.34, 14.26, 102.5, 704.4, 0.1073,
        21.16, 23.04, 137.2, 1404.0, 0.09428,
        16.65, 21.38, 110.0, 904.6, 0.1121,
        17.14, 16.4, 116.0, 912.7, 0.1186,
        14.58, 21.53, 97.41, 644.8, 0.1054,
        18.61, 20.25, 122.1, 1094.0, 0.0944,
        15.3, 25.27, 102.4, 732.4, 0.1082,
        17.57, 15.05, 115.0, 955.1, 0.09847
    ]).unwrap();

    // Define the target (0 = malignant, 1 = benign)
    let targets = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let target = Array1::from(targets);

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Add metadata
    let feature_names = vec![
        "mean_radius".to_string(),
        "mean_texture".to_string(),
        "mean_perimeter".to_string(),
        "mean_area".to_string(),
        "mean_smoothness".to_string(),
    ];

    let target_names = vec!["malignant".to_string(), "benign".to_string()];

    let description = "Breast Cancer Wisconsin (Diagnostic) Database
    
Features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
They describe characteristics of the cell nuclei present in the image.

(This is a simplified version of the dataset with only 5 features and 30 samples)

Target:
- Malignant
- Benign"
        .to_string();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_target_names(target_names)
        .with_description(description);

    Ok(dataset)
}

/// Generate the digits dataset
pub fn load_digits() -> Result<Dataset> {
    // Use a simplified version with fewer samples and features
    // Each digit is represented as a 4x4 image flattened to 16 features
    let n_samples = 50; // 5 samples per digit (0-9)
    let n_features = 16; // 4x4 pixels

    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    // Sample digit patterns (4x4 pixel representations of digits 0-9)
    #[rustfmt::skip]
    let digit_patterns = [
        // Digit 0
        [0., 1., 1., 0.,
         1., 0., 0., 1.,
         1., 0., 0., 1.,
         0., 1., 1., 0.],
        // Digit 1
        [0., 1., 0., 0.,
         0., 1., 0., 0.,
         0., 1., 0., 0.,
         0., 1., 0., 0.],
        // Digit 2
        [1., 1., 1., 0.,
         0., 0., 1., 0.,
         0., 1., 0., 0.,
         1., 1., 1., 1.],
        // Digit 3
        [1., 1., 1., 0.,
         0., 0., 1., 0.,
         1., 1., 1., 0.,
         0., 0., 1., 0.],
        // Digit 4
        [1., 0., 1., 0.,
         1., 0., 1., 0.,
         1., 1., 1., 1.,
         0., 0., 1., 0.],
        // Digit 5
        [1., 1., 1., 1.,
         1., 0., 0., 0.,
         1., 1., 1., 0.,
         0., 0., 1., 1.],
        // Digit 6
        [0., 1., 1., 0.,
         1., 0., 0., 0.,
         1., 1., 1., 0.,
         0., 1., 1., 0.],
        // Digit 7
        [1., 1., 1., 1.,
         0., 0., 0., 1.,
         0., 0., 1., 0.,
         0., 1., 0., 0.],
        // Digit 8
        [0., 1., 1., 0.,
         1., 0., 0., 1.,
         0., 1., 1., 0.,
         1., 0., 0., 1.],
        // Digit 9
        [0., 1., 1., 0.,
         1., 0., 0., 1.,
         0., 1., 1., 1.,
         0., 0., 1., 0.],
    ];

    // Create 5 samples per digit with small random variations
    let mut rng = rng();
    let noise_level = 0.1;

    for (digit, &pattern) in digit_patterns.iter().enumerate() {
        for sample in 0..5 {
            let idx = digit * 5 + sample;
            target[idx] = digit as f64;

            // Copy the pattern with noise
            for (j, &pixel) in pattern.iter().enumerate() {
                let noise = if pixel > 0.5 {
                    -noise_level * rng.random::<f64>()
                } else {
                    noise_level * rng.random::<f64>()
                };

                let val = pixel + noise;
                data[[idx, j]] = val.clamp(0.0, 1.0);
            }
        }
    }

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Create feature names
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("pixel_{}", i)).collect();

    let target_names: Vec<String> = (0..10).map(|i| format!("{}", i)).collect();

    let description = "Optical recognition of handwritten digits dataset
    
A simplified version with 50 samples (5 for each digit 0-9) and 16 features (4x4 pixel images).
Each feature is the grayscale value of a pixel in the image.

Target: Digit identity (0-9)"
        .to_string();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_target_names(target_names)
        .with_description(description);

    Ok(dataset)
}

/// Generate the Boston housing dataset
pub fn load_boston() -> Result<Dataset> {
    // Simplified version with fewer samples and features
    let n_samples = 30;
    let n_features = 5;

    #[rustfmt::skip]
    let data = Array2::from_shape_vec((n_samples, n_features), vec![
        0.00632, 18.0, 2.31, 0.538, 6.575,
        0.02731, 0.0, 7.07, 0.469, 6.421,
        0.02729, 0.0, 7.07, 0.469, 7.185,
        0.03237, 0.0, 2.18, 0.458, 6.998,
        0.06905, 0.0, 2.18, 0.458, 7.147,
        0.02985, 0.0, 2.18, 0.458, 6.430,
        0.08829, 12.5, 7.87, 0.524, 6.012,
        0.14455, 12.5, 7.87, 0.524, 6.172,
        0.21124, 12.5, 7.87, 0.524, 5.631,
        0.17004, 12.5, 7.87, 0.524, 6.004,
        0.22489, 12.5, 7.87, 0.524, 6.377,
        0.11747, 12.5, 7.87, 0.524, 6.009,
        0.09378, 12.5, 7.87, 0.524, 5.889,
        0.62976, 0.0, 8.14, 0.538, 5.949,
        0.63796, 0.0, 8.14, 0.538, 6.096,
        0.62739, 0.0, 8.14, 0.538, 5.834,
        1.05393, 0.0, 8.14, 0.538, 5.935,
        0.7842, 0.0, 8.14, 0.538, 5.990,
        0.80271, 0.0, 8.14, 0.538, 5.456,
        0.7258, 0.0, 8.14, 0.538, 5.727,
        1.25179, 0.0, 8.14, 0.538, 5.570,
        0.85204, 0.0, 8.14, 0.538, 5.965,
        1.23247, 0.0, 8.14, 0.538, 6.142,
        0.98843, 0.0, 8.14, 0.538, 5.813,
        0.75026, 0.0, 8.14, 0.538, 5.924,
        0.84054, 0.0, 8.14, 0.538, 5.599,
        0.67191, 0.0, 8.14, 0.538, 5.813,
        0.95577, 0.0, 8.14, 0.538, 6.047,
        0.77299, 0.0, 8.14, 0.538, 6.495,
        1.00245, 0.0, 8.14, 0.538, 6.674
    ]).unwrap();

    let targets = vec![
        24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9, 15.0, 18.9, 21.7, 20.4, 18.2,
        19.9, 23.1, 17.5, 20.2, 18.2, 13.6, 19.6, 15.2, 14.5, 15.6, 13.9, 16.6, 14.8, 18.4, 21.0,
    ];
    let target = Array1::from(targets);

    // Create dataset
    let mut dataset = Dataset::new(data, Some(target));

    // Add metadata
    let feature_names = vec![
        "CRIM".to_string(),
        "ZN".to_string(),
        "INDUS".to_string(),
        "CHAS".to_string(),
        "NOX".to_string(),
    ];

    let feature_descriptions = vec![
        "per capita crime rate by town".to_string(),
        "proportion of residential land zoned for lots over 25,000 sq.ft.".to_string(),
        "proportion of non-retail business acres per town".to_string(),
        "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)".to_string(),
        "nitric oxides concentration (parts per 10 million)".to_string(),
    ];

    let description = "Boston Housing Dataset (Simplified)
    
A simplified version of the Boston housing dataset with 30 samples and 5 features.
The target variable is the median value of owner-occupied homes in $1000s.

This is a regression dataset."
        .to_string();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_feature_descriptions(feature_descriptions)
        .with_description(description);

    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_iris() {
        let dataset = load_iris().unwrap();

        assert_eq!(dataset.n_samples(), 150);
        assert_eq!(dataset.n_features(), 4);
        assert!(dataset.target.is_some());
        assert!(dataset.description.is_some());
        assert!(dataset.feature_names.is_some());
        assert!(dataset.target_names.is_some());

        let feature_names = dataset.feature_names.as_ref().unwrap();
        assert_eq!(feature_names.len(), 4);
        assert_eq!(feature_names[0], "sepal_length");
        assert_eq!(feature_names[3], "petal_width");

        let target_names = dataset.target_names.as_ref().unwrap();
        assert_eq!(target_names.len(), 3);
        assert!(target_names.contains(&"setosa".to_string()));
        assert!(target_names.contains(&"versicolor".to_string()));
        assert!(target_names.contains(&"virginica".to_string()));

        // Check target values are in valid range (0, 1, 2)
        let target = dataset.target.as_ref().unwrap();
        for &val in target.iter() {
            assert!(val >= 0.0 && val <= 2.0);
        }
    }

    #[test]
    fn test_load_breast_cancer() {
        let dataset = load_breast_cancer().unwrap();

        assert_eq!(dataset.n_samples(), 30);
        assert_eq!(dataset.n_features(), 5);
        assert!(dataset.target.is_some());
        assert!(dataset.description.is_some());
        assert!(dataset.feature_names.is_some());
        assert!(dataset.target_names.is_some());

        let feature_names = dataset.feature_names.as_ref().unwrap();
        assert_eq!(feature_names.len(), 5);
        assert_eq!(feature_names[0], "mean_radius");
        assert_eq!(feature_names[4], "mean_smoothness");

        let target_names = dataset.target_names.as_ref().unwrap();
        assert_eq!(target_names.len(), 2);
        assert!(target_names.contains(&"malignant".to_string()));
        assert!(target_names.contains(&"benign".to_string()));

        // Check target values are binary (0 or 1)
        let target = dataset.target.as_ref().unwrap();
        for &val in target.iter() {
            assert!(val == 0.0 || val == 1.0);
        }
    }

    #[test]
    fn test_load_digits() {
        let dataset = load_digits().unwrap();

        assert_eq!(dataset.n_samples(), 50);
        assert_eq!(dataset.n_features(), 16);
        assert!(dataset.target.is_some());
        assert!(dataset.description.is_some());
        assert!(dataset.feature_names.is_some());
        assert!(dataset.target_names.is_some());

        let feature_names = dataset.feature_names.as_ref().unwrap();
        assert_eq!(feature_names.len(), 16);
        assert_eq!(feature_names[0], "pixel_0");
        assert_eq!(feature_names[15], "pixel_15");

        let target_names = dataset.target_names.as_ref().unwrap();
        assert_eq!(target_names.len(), 10);
        for i in 0..10 {
            assert!(target_names.contains(&i.to_string()));
        }

        // Check target values are digits (0-9)
        let target = dataset.target.as_ref().unwrap();
        for &val in target.iter() {
            assert!(val >= 0.0 && val <= 9.0);
        }

        // Check pixel values are in valid range [0, 1]
        for row in dataset.data.rows() {
            for &pixel in row.iter() {
                assert!(pixel >= 0.0 && pixel <= 1.0);
            }
        }
    }

    #[test]
    fn test_load_boston() {
        let dataset = load_boston().unwrap();

        assert_eq!(dataset.n_samples(), 30);
        assert_eq!(dataset.n_features(), 5);
        assert!(dataset.target.is_some());
        assert!(dataset.description.is_some());
        assert!(dataset.feature_names.is_some());
        assert!(dataset.feature_descriptions.is_some());

        let feature_names = dataset.feature_names.as_ref().unwrap();
        assert_eq!(feature_names.len(), 5);
        assert_eq!(feature_names[0], "CRIM");
        assert_eq!(feature_names[4], "NOX");

        let feature_descriptions = dataset.feature_descriptions.as_ref().unwrap();
        assert_eq!(feature_descriptions.len(), 5);
        assert!(feature_descriptions[0].contains("crime rate"));

        // Check target values are reasonable housing prices
        let target = dataset.target.as_ref().unwrap();
        for &val in target.iter() {
            assert!(val > 0.0 && val < 100.0); // Reasonable housing prices in $1000s
        }
    }

    #[test]
    fn test_all_datasets_have_consistent_shapes() {
        let datasets = vec![
            ("iris", load_iris().unwrap()),
            ("breast_cancer", load_breast_cancer().unwrap()),
            ("digits", load_digits().unwrap()),
            ("boston", load_boston().unwrap()),
        ];

        for (name, dataset) in datasets {
            // Check that data and target have consistent sample counts
            if let Some(ref target) = dataset.target {
                assert_eq!(
                    dataset.data.nrows(),
                    target.len(),
                    "Dataset '{}' has inconsistent sample counts",
                    name
                );
            }

            // Check that feature names match feature count (if present)
            if let Some(ref feature_names) = dataset.feature_names {
                assert_eq!(
                    dataset.data.ncols(),
                    feature_names.len(),
                    "Dataset '{}' has inconsistent feature count",
                    name
                );
            }

            // Check that feature descriptions match feature count (if present)
            if let Some(ref feature_descriptions) = dataset.feature_descriptions {
                assert_eq!(
                    dataset.data.ncols(),
                    feature_descriptions.len(),
                    "Dataset '{}' has inconsistent feature description count",
                    name
                );
            }

            // Check that dataset has a description
            assert!(
                dataset.description.is_some(),
                "Dataset '{}' missing description",
                name
            );
        }
    }
}
