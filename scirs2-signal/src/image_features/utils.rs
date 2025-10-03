// Utility functions for image feature extraction

use scirs2_core::ndarray::Array2;

#[allow(unused_imports)]
/// Calculate skewness of a vector
#[allow(dead_code)]
pub fn calculate_skewness(_data: &[f64], mean: f64, stddev: f64) -> f64 {
    if std_dev <= 0.0 || data.len() < 3 {
        return 0.0;
    }

    let n = data.len() as f64;
    let sum_cubed_diff: f64 = data.iter().map(|&x| (x - mean).powi(3)).sum();

    sum_cubed_diff / ((n - 1.0) * std_dev.powi(3))
}

/// Calculate kurtosis of a vector
#[allow(dead_code)]
pub fn calculate_kurtosis(_data: &[f64], mean: f64, stddev: f64) -> f64 {
    if std_dev <= 0.0 || data.len() < 4 {
        return 0.0;
    }

    let n = data.len() as f64;
    let sum_quartic_diff: f64 = data.iter().map(|&x| (x - mean).powi(4)).sum();

    sum_quartic_diff / ((n - 1.0) * std_dev.powi(4)) - 3.0 // Excess kurtosis
}

/// Calculate raw moment of an image
#[allow(dead_code)]
pub fn calculate_raw_moment(image: &Array2<f64>, p: usize, q: usize) -> f64 {
    let shape = image.shape();
    let height = shape[0];
    let width = shape[1];

    let mut moment = 0.0;
    for i in 0..height {
        for j in 0..width {
            moment += (i as f64).powi(p as i32) * (j as f64).powi(q as i32) * image[[i, j]];
        }
    }
    moment
}

/// Compute Gray Level Co-occurrence Matrix (GLCM)
#[allow(dead_code)]
pub fn compute_glcm(_image: &Array2<f64>, distance: usize, numlevels: usize) -> Array2<f64> {
    let shape = image.shape();
    let height = shape[0];
    let width = shape[1];

    if height <= distance || width <= distance {
        return Array2::zeros((num_levels, num_levels));
    }

    // Find min and max values for scaling
    let min_val = image.iter().fold(f64::MAX, |a, &b| a.min(b));
    let max_val = image.iter().fold(f64::MIN, |a, &b| a.max(b));

    // Initialize GLCM
    let mut glcm = Array2::zeros((num_levels, num_levels));

    // Compute GLCM for horizontal direction (0 degrees)
    for i in 0..height {
        for j in 0..width - distance {
            // Scale pixel values to [0, num_levels-1]
            let row: usize;
            let col: usize;

            if (max_val - min_val).abs() < 1e-10 {
                row = 0;
                col = 0;
            } else {
                row = ((_image[[i, j]] - min_val) / (max_val - min_val) * (num_levels - 1) as f64)
                    .round() as usize;
                col = ((_image[[i, j + distance]] - min_val) / (max_val - min_val)
                    * (num_levels - 1) as f64)
                    .round() as usize;
            }

            // Bound checks (shouldn't be necessary but added for safety)
            let row_idx = row.min(num_levels - 1);
            let col_idx = col.min(num_levels - 1);

            glcm[[row_idx, col_idx]] += 1.0;
        }
    }

    // Make GLCM symmetric (add transpose)
    for i in 0..num_levels {
        for j in 0..i {
            let sum = glcm[[i, j]] + glcm[[j, i]];
            glcm[[i, j]] = sum;
            glcm[[j, i]] = sum;
        }
    }

    glcm
}
