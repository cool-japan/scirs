//! Deterministic synthetic data generators for ML testing.
//!
//! All generators produce data using purely arithmetic formulas -- no randomness
//! or external crate dependencies. This ensures that test data is bit-exact
//! reproducible across platforms and runs.

use std::f64::consts::PI;

/// Generate linearly separable 2D data.
///
/// Class 0 points lie on the left side (`x < 0`), class 1 on the right (`x > 0`).
/// Points are arranged in a deterministic grid pattern.
pub fn linear_separable_2d(n_per_class: usize) -> (Vec<[f64; 2]>, Vec<usize>) {
    let mut points = Vec::with_capacity(n_per_class * 2);
    let mut labels = Vec::with_capacity(n_per_class * 2);

    let side = ((n_per_class as f64).sqrt().ceil()) as usize;
    let step = 1.0 / (side as f64 + 1.0);

    // Class 0: points in [-2, -0.5] x [-1, 1]
    for i in 0..n_per_class {
        let row = i / side;
        let col = i % side;
        let x = -2.0 + (col as f64 + 1.0) * step * 1.5;
        let y = -1.0 + (row as f64 + 1.0) * step * 2.0;
        points.push([x, y]);
        labels.push(0);
    }

    // Class 1: points in [0.5, 2] x [-1, 1]
    for i in 0..n_per_class {
        let row = i / side;
        let col = i % side;
        let x = 0.5 + (col as f64 + 1.0) * step * 1.5;
        let y = -1.0 + (row as f64 + 1.0) * step * 2.0;
        points.push([x, y]);
        labels.push(1);
    }

    (points, labels)
}

/// Generate concentric rings of points.
///
/// Each ring has `n_per_ring` points at a deterministic angular spacing.
/// Ring radii increase linearly: ring `k` has radius `k + 1`.
pub fn concentric_rings(n_per_ring: usize, n_rings: usize) -> (Vec<[f64; 2]>, Vec<usize>) {
    let mut points = Vec::with_capacity(n_per_ring * n_rings);
    let mut labels = Vec::with_capacity(n_per_ring * n_rings);

    for ring in 0..n_rings {
        let radius = (ring + 1) as f64;
        for i in 0..n_per_ring {
            let angle = 2.0 * PI * (i as f64) / (n_per_ring as f64);
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            points.push([x, y]);
            labels.push(ring);
        }
    }

    (points, labels)
}

/// Generate a simple regression dataset: `y = slope * x + intercept + noise`.
///
/// The "noise" is a deterministic triangular wave pattern with small amplitude
/// to make the data imperfect but reproducible.
pub fn linear_regression_data(n: usize, slope: f64, intercept: f64) -> (Vec<f64>, Vec<f64>) {
    let mut x_vals = Vec::with_capacity(n);
    let mut y_vals = Vec::with_capacity(n);

    for i in 0..n {
        let x = (i as f64) / (n as f64).max(1.0) * 10.0 - 5.0;
        // Deterministic small perturbation using triangle wave
        let noise = triangle_wave(i as f64 * 0.7) * 0.1;
        let y = slope * x + intercept + noise;
        x_vals.push(x);
        y_vals.push(y);
    }

    (x_vals, y_vals)
}

/// Generate exact linear regression data with no noise.
///
/// Useful for verifying that models achieve perfect fit on noiseless data.
pub fn exact_linear_data(n: usize, slope: f64, intercept: f64) -> (Vec<f64>, Vec<f64>) {
    let mut x_vals = Vec::with_capacity(n);
    let mut y_vals = Vec::with_capacity(n);

    for i in 0..n {
        let x = (i as f64) / (n as f64).max(1.0) * 10.0 - 5.0;
        let y = slope * x + intercept;
        x_vals.push(x);
        y_vals.push(y);
    }

    (x_vals, y_vals)
}

/// Generate clusters at known centroids.
///
/// Each cluster has `n_per_cluster` points arranged in a tight deterministic
/// pattern around the centroid.
pub fn clustered_data(centroids: &[[f64; 2]], n_per_cluster: usize) -> (Vec<[f64; 2]>, Vec<usize>) {
    let mut points = Vec::with_capacity(centroids.len() * n_per_cluster);
    let mut labels = Vec::with_capacity(centroids.len() * n_per_cluster);

    let spread = 0.3; // Tight clusters

    for (ci, centroid) in centroids.iter().enumerate() {
        for i in 0..n_per_cluster {
            let angle = 2.0 * PI * (i as f64) / (n_per_cluster as f64);
            // Vary radius slightly by point index
            let r = spread * (0.5 + 0.5 * triangle_wave(i as f64 * 1.3));
            let x = centroid[0] + r * angle.cos();
            let y = centroid[1] + r * angle.sin();
            points.push([x, y]);
            labels.push(ci);
        }
    }

    (points, labels)
}

/// Generate multi-dimensional feature data with known cluster structure.
///
/// Creates `n_clusters` clusters in `n_dims` dimensions, each separated
/// by at least `separation` units along the primary axis.
pub fn high_dim_clustered_data(
    n_dims: usize,
    n_clusters: usize,
    n_per_cluster: usize,
    separation: f64,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut points = Vec::with_capacity(n_clusters * n_per_cluster);
    let mut labels = Vec::with_capacity(n_clusters * n_per_cluster);

    for ci in 0..n_clusters {
        let center_offset = ci as f64 * separation;
        for i in 0..n_per_cluster {
            let mut point = vec![0.0; n_dims];
            // Place cluster center along first dimension
            point[0] = center_offset;
            // Add deterministic spread in all dimensions
            for (d, coord) in point.iter_mut().enumerate() {
                let perturbation = triangle_wave((i as f64 + d as f64 * 0.37) * 0.9) * 0.2;
                *coord += perturbation;
            }
            points.push(point);
            labels.push(ci);
        }
    }

    (points, labels)
}

/// Generate a correlation matrix with known block-diagonal structure.
///
/// Within each block, features have correlation `intra_corr`.
/// Across blocks, correlation is zero.
pub fn structured_correlation(block_sizes: &[usize], intra_corr: f64) -> Vec<Vec<f64>> {
    let n: usize = block_sizes.iter().sum();
    let mut matrix = vec![vec![0.0; n]; n];

    // Diagonal
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    // Fill blocks
    let mut offset = 0;
    for &bs in block_sizes {
        for i in 0..bs {
            for j in 0..bs {
                if i != j {
                    matrix[offset + i][offset + j] = intra_corr;
                }
            }
        }
        offset += bs;
    }

    matrix
}

/// Generate XOR-pattern data (not linearly separable).
///
/// Four quadrants; class is determined by `sign(x) == sign(y)`.
pub fn xor_pattern(n_per_quadrant: usize) -> (Vec<[f64; 2]>, Vec<usize>) {
    let mut points = Vec::with_capacity(n_per_quadrant * 4);
    let mut labels = Vec::with_capacity(n_per_quadrant * 4);

    let offsets = [
        (1.0, 1.0, 0usize), // Q1: positive
        (-1.0, -1.0, 0),    // Q3: positive (same class as Q1)
        (-1.0, 1.0, 1),     // Q2: negative
        (1.0, -1.0, 1),     // Q4: negative
    ];

    for &(cx, cy, label) in &offsets {
        for i in 0..n_per_quadrant {
            let angle = 2.0 * PI * (i as f64) / (n_per_quadrant as f64);
            let r = 0.3 * (0.5 + 0.5 * triangle_wave(i as f64 * 1.1));
            let x = cx + r * angle.cos();
            let y = cy + r * angle.sin();
            points.push([x, y]);
            labels.push(label);
        }
    }

    (points, labels)
}

/// Generate a polynomial regression dataset: `y = sum(coeffs[i] * x^i)`.
pub fn polynomial_data(n: usize, coeffs: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut x_vals = Vec::with_capacity(n);
    let mut y_vals = Vec::with_capacity(n);

    for i in 0..n {
        let x = (i as f64) / (n as f64).max(1.0) * 4.0 - 2.0;
        let mut y = 0.0;
        let mut x_power = 1.0;
        for &c in coeffs {
            y += c * x_power;
            x_power *= x;
        }
        x_vals.push(x);
        y_vals.push(y);
    }

    (x_vals, y_vals)
}

/// Deterministic triangle wave: maps any `t` to `[-1, 1]` with period 4.
fn triangle_wave(t: f64) -> f64 {
    let t_mod = ((t % 4.0) + 4.0) % 4.0;
    if t_mod < 1.0 {
        t_mod
    } else if t_mod < 3.0 {
        2.0 - t_mod
    } else {
        t_mod - 4.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_wave_bounds() {
        for i in 0..1000 {
            let t = i as f64 * 0.01 - 5.0;
            let v = triangle_wave(t);
            assert!((-1.0..=1.0).contains(&v), "triangle_wave({t}) = {v}");
        }
    }

    #[test]
    fn test_triangle_wave_period() {
        for i in 0..100 {
            let t = i as f64 * 0.1;
            let a = triangle_wave(t);
            let b = triangle_wave(t + 4.0);
            assert!(
                (a - b).abs() < 1e-12,
                "triangle_wave({t}) = {a}, triangle_wave({}) = {b}",
                t + 4.0
            );
        }
    }
}
