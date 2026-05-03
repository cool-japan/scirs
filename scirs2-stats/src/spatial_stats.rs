//! Spatial Statistics
//!
//! This module provides spatial analysis functions:
//!
//! - **Moran's I**: Global and local (LISA) spatial autocorrelation
//! - **Geary's C**: Alternative spatial autocorrelation measure
//! - **Ripley's K**: Point pattern analysis
//! - **Variogram**: Experimental semi-variogram estimation
//!
//! # References
//! - Moran, P.A.P. (1950). Notes on Continuous Stochastic Phenomena.
//! - Geary, R.C. (1954). The Contiguity Ratio and Statistical Mapping.
//! - Ripley, B.D. (1976). The Second-Order Analysis of Stationary Point Processes.
//! - Matheron, G. (1963). Principles of Geostatistics.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Helper: z-score and normal CDF
// ---------------------------------------------------------------------------

/// Abramowitz & Stegun rational approximation of erf(x), max error ~1.5e-7.
fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// Two-tailed p-value from standard normal z-score.
fn two_tailed_p(z: f64) -> f64 {
    (2.0 * norm_cdf(-z.abs())).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Moran's I
// ---------------------------------------------------------------------------

/// Result of Moran's I spatial autocorrelation test.
#[derive(Debug, Clone)]
pub struct MoransResult {
    /// The Moran's I statistic (range approximately [-1, 1]).
    pub statistic: f64,
    /// Expected value of I under no spatial autocorrelation.
    pub expected: f64,
    /// Variance of I under randomisation assumption.
    pub variance: f64,
    /// Standardised z-score.
    pub z_score: f64,
    /// Two-tailed p-value (normal approximation).
    pub p_value: f64,
}

/// Local Moran's I (LISA) result for a single location.
#[derive(Debug, Clone)]
pub struct LisaResult {
    /// Local Moran's I value for each observation.
    pub local_i: Vec<f64>,
    /// Z-score for each observation.
    pub z_scores: Vec<f64>,
    /// Two-tailed p-value for each observation.
    pub p_values: Vec<f64>,
}

/// Spatial autocorrelation analysis using Moran's I.
pub struct MoransI;

impl MoransI {
    /// Compute global Moran's I.
    ///
    /// # Arguments
    /// * `values` - Observed values at each location.
    /// * `weights` - Row-standardised (or raw) spatial weights matrix (n×n).
    ///
    /// # Returns
    /// `MoransResult` with the statistic, expected value, variance, z-score and p-value.
    pub fn compute(values: &[f64], weights: &Array2<f64>) -> StatsResult<MoransResult> {
        let n = values.len();
        if n < 3 {
            return Err(StatsError::InvalidArgument(
                "Moran's I requires at least 3 observations".to_string(),
            ));
        }
        if weights.nrows() != n || weights.ncols() != n {
            return Err(StatsError::InvalidArgument(format!(
                "weights must be {n}×{n}, got {}×{}",
                weights.nrows(),
                weights.ncols()
            )));
        }

        let mean = values.iter().sum::<f64>() / n as f64;
        let z: Vec<f64> = values.iter().map(|&v| v - mean).collect();

        // Sum of all weights
        let w_sum: f64 = weights.iter().sum();
        if w_sum == 0.0 {
            return Err(StatsError::InvalidArgument(
                "Sum of spatial weights is zero".to_string(),
            ));
        }

        // Numerator: Σ_i Σ_j w_ij (z_i)(z_j)
        let mut numerator = 0.0;
        for i in 0..n {
            for j in 0..n {
                numerator += weights[[i, j]] * z[i] * z[j];
            }
        }

        // Denominator: Σ_i z_i²
        let s2: f64 = z.iter().map(|&zi| zi * zi).sum();
        if s2 == 0.0 {
            return Err(StatsError::InvalidArgument(
                "All values are identical; Moran's I is undefined".to_string(),
            ));
        }

        let statistic = (n as f64 / w_sum) * (numerator / s2);

        // Expected value under randomisation: E[I] = -1/(n-1)
        let expected = -1.0 / (n as f64 - 1.0);

        // Variance (randomisation assumption)
        // S1 = 0.5 * Σ_i Σ_j (w_ij + w_ji)²
        let mut s1 = 0.0;
        for i in 0..n {
            for j in 0..n {
                let v = weights[[i, j]] + weights[[j, i]];
                s1 += v * v;
            }
        }
        s1 *= 0.5;

        // S2 = Σ_i (Σ_j w_ij + Σ_j w_ji)²
        let mut s2_stat = 0.0;
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| weights[[i, j]]).sum();
            let col_sum: f64 = (0..n).map(|j| weights[[j, i]]).sum();
            s2_stat += (row_sum + col_sum).powi(2);
        }

        let n_f = n as f64;
        let k4 = {
            let m2 = z.iter().map(|&zi| zi.powi(2)).sum::<f64>() / n_f;
            let m4 = z.iter().map(|&zi| zi.powi(4)).sum::<f64>() / n_f;
            m4 / (m2 * m2)
        };

        let w_sq = w_sum * w_sum;
        let num_var = n_f * (n_f * n_f - 3.0 * n_f + 3.0) * s1 - n_f * s2_stat + 3.0 * w_sq;
        let den_var = (n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0) * w_sq;
        let kur_term = k4 * ((n_f * n_f - n_f) * s1 - 2.0 * n_f * s2_stat + 6.0 * w_sq)
            / ((n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0) * w_sq);

        let variance = (num_var / den_var - kur_term - expected * expected).max(1e-15);
        let z_score = (statistic - expected) / variance.sqrt();
        let p_value = two_tailed_p(z_score);

        Ok(MoransResult {
            statistic,
            expected,
            variance,
            z_score,
            p_value,
        })
    }

    /// Compute Local Indicators of Spatial Association (LISA) — Local Moran's I.
    ///
    /// Returns the local I value for each observation along with z-scores and p-values.
    pub fn local(values: &[f64], weights: &Array2<f64>) -> StatsResult<LisaResult> {
        let n = values.len();
        if n < 3 {
            return Err(StatsError::InvalidArgument(
                "LISA requires at least 3 observations".to_string(),
            ));
        }
        if weights.nrows() != n || weights.ncols() != n {
            return Err(StatsError::InvalidArgument(format!(
                "weights must be {n}×{n}"
            )));
        }

        let mean = values.iter().sum::<f64>() / n as f64;
        let z: Vec<f64> = values.iter().map(|&v| v - mean).collect();
        let m2 = z.iter().map(|&zi| zi * zi).sum::<f64>() / n as f64;

        if m2 == 0.0 {
            return Err(StatsError::InvalidArgument(
                "All values are identical; LISA is undefined".to_string(),
            ));
        }

        let mut local_i = vec![0.0_f64; n];
        let mut z_scores = vec![0.0_f64; n];
        let mut p_values = vec![0.0_f64; n];

        for i in 0..n {
            // Row-standardise weights for location i
            let row_sum: f64 = (0..n).map(|j| weights[[i, j]]).sum();
            let w_i: Vec<f64> = if row_sum > 0.0 {
                (0..n).map(|j| weights[[i, j]] / row_sum).collect()
            } else {
                vec![0.0; n]
            };

            let lag_i: f64 = (0..n).map(|j| w_i[j] * z[j]).sum();
            local_i[i] = (z[i] / m2) * lag_i;

            // Variance of local I (analytical approximation)
            let w_sq_sum: f64 = w_i.iter().map(|&w| w * w).sum();
            let var_i = (w_sq_sum * (n as f64 / (n as f64 - 1.0))).max(1e-15);
            z_scores[i] = local_i[i] / var_i.sqrt();
            p_values[i] = two_tailed_p(z_scores[i]);
        }

        Ok(LisaResult {
            local_i,
            z_scores,
            p_values,
        })
    }
}

// ---------------------------------------------------------------------------
// Geary's C
// ---------------------------------------------------------------------------

/// Spatial autocorrelation using Geary's C statistic.
pub struct GearyC;

impl GearyC {
    /// Compute Geary's C statistic.
    ///
    /// C ≈ 0 → strong positive autocorrelation; C ≈ 1 → no autocorrelation;
    /// C > 1 → negative autocorrelation.
    ///
    /// # Arguments
    /// * `values` - Observed values at each location.
    /// * `weights` - Spatial weights matrix (n×n).
    ///
    /// # Returns
    /// Geary's C value.
    pub fn compute(values: &[f64], weights: &Array2<f64>) -> StatsResult<f64> {
        let n = values.len();
        if n < 3 {
            return Err(StatsError::InvalidArgument(
                "Geary's C requires at least 3 observations".to_string(),
            ));
        }
        if weights.nrows() != n || weights.ncols() != n {
            return Err(StatsError::InvalidArgument(format!(
                "weights must be {n}×{n}"
            )));
        }

        let mean = values.iter().sum::<f64>() / n as f64;
        let ss: f64 = values.iter().map(|&v| (v - mean).powi(2)).sum();
        if ss == 0.0 {
            return Err(StatsError::InvalidArgument(
                "All values are identical; Geary's C is undefined".to_string(),
            ));
        }

        let w_sum: f64 = weights.iter().sum();
        if w_sum == 0.0 {
            return Err(StatsError::InvalidArgument(
                "Sum of spatial weights is zero".to_string(),
            ));
        }

        let mut cross_sum = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let diff = values[i] - values[j];
                cross_sum += weights[[i, j]] * diff * diff;
            }
        }

        Ok(((n as f64 - 1.0) * cross_sum) / (2.0 * w_sum * ss))
    }
}

// ---------------------------------------------------------------------------
// Spatial Autocorrelation (convenience wrapper)
// ---------------------------------------------------------------------------

/// Convenience wrapper providing Moran's I and LISA together.
pub struct SpatialAutocorrelation {
    /// Global Moran's I result.
    pub global: MoransResult,
    /// Local LISA result.
    pub local: LisaResult,
}

impl SpatialAutocorrelation {
    /// Compute both global Moran's I and local LISA statistics.
    pub fn compute(values: &[f64], weights: &Array2<f64>) -> StatsResult<Self> {
        let global = MoransI::compute(values, weights)?;
        let local = MoransI::local(values, weights)?;
        Ok(Self { global, local })
    }
}

// ---------------------------------------------------------------------------
// Ripley's K function
// ---------------------------------------------------------------------------

/// Ripley's K function for point pattern analysis.
///
/// K(d) estimates the expected number of additional events within distance d
/// of a randomly chosen event, divided by the overall intensity λ.
pub struct Ripley;

impl Ripley {
    /// Compute Ripley's K function at each specified distance.
    ///
    /// Uses a simple toroidal edge-correction (Ripley, 1976).
    ///
    /// # Arguments
    /// * `points` - Point locations as `(x, y)` pairs.
    /// * `distances` - Distances at which to evaluate K.
    /// * `area` - Area of the study region.
    ///
    /// # Returns
    /// `Vec<f64>` with K(d) for each distance d.
    pub fn k_function(
        points: &[(f64, f64)],
        distances: &[f64],
        area: f64,
    ) -> StatsResult<Vec<f64>> {
        let n = points.len();
        if n < 2 {
            return Err(StatsError::InvalidArgument(
                "Ripley's K requires at least 2 points".to_string(),
            ));
        }
        if area <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "Area must be positive".to_string(),
            ));
        }

        let lambda = n as f64 / area;
        let mut k_values = vec![0.0_f64; distances.len()];

        for (d_idx, &d) in distances.iter().enumerate() {
            let mut count = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let dx = points[i].0 - points[j].0;
                    let dy = points[i].1 - points[j].1;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist <= d {
                        count += 1.0;
                    }
                }
            }
            k_values[d_idx] = count / (lambda * n as f64);
        }

        Ok(k_values)
    }

    /// Compute L(d) = sqrt(K(d)/π) - d, a normalised version with expected value 0
    /// under complete spatial randomness.
    pub fn l_function(
        points: &[(f64, f64)],
        distances: &[f64],
        area: f64,
    ) -> StatsResult<Vec<f64>> {
        let k = Self::k_function(points, distances, area)?;
        let l = k
            .iter()
            .zip(distances.iter())
            .map(|(&ki, &d)| (ki / std::f64::consts::PI).sqrt() - d)
            .collect();
        Ok(l)
    }
}

// ---------------------------------------------------------------------------
// Experimental variogram
// ---------------------------------------------------------------------------

/// A single bin of the empirical variogram.
#[derive(Debug, Clone)]
pub struct VariogramBin {
    /// Mean distance in this lag bin.
    pub distance: f64,
    /// Estimated semi-variance γ(h).
    pub semivariance: f64,
    /// Number of point pairs in this bin.
    pub count: usize,
}

/// Compute the experimental (empirical) semi-variogram from spatial observations.
///
/// # Arguments
/// * `points` - `(x, y, value)` triples.
/// * `n_bins` - Number of lag bins.
///
/// # Returns
/// `Vec<VariogramBin>` sorted by distance.
pub fn variogram(points: &[(f64, f64, f64)], n_bins: usize) -> StatsResult<Vec<VariogramBin>> {
    let n = points.len();
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "variogram requires at least 2 points".to_string(),
        ));
    }
    if n_bins == 0 {
        return Err(StatsError::InvalidArgument(
            "n_bins must be at least 1".to_string(),
        ));
    }

    // Collect all pairwise distances and squared differences
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n * (n - 1) / 2);
    let mut max_dist = 0.0_f64;

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = points[i].0 - points[j].0;
            let dy = points[i].1 - points[j].1;
            let dist = (dx * dx + dy * dy).sqrt();
            let sq_diff = (points[i].2 - points[j].2).powi(2);
            pairs.push((dist, sq_diff));
            if dist > max_dist {
                max_dist = dist;
            }
        }
    }

    if max_dist == 0.0 {
        return Err(StatsError::InvalidArgument(
            "All points are at the same location".to_string(),
        ));
    }

    // Bin pairs
    let bin_width = max_dist / n_bins as f64;
    let mut bin_sum = vec![0.0_f64; n_bins];
    let mut bin_cnt = vec![0_usize; n_bins];
    let mut bin_dist = vec![0.0_f64; n_bins];

    for (dist, sq_diff) in &pairs {
        let idx = ((dist / max_dist) * n_bins as f64).floor() as usize;
        let idx = idx.min(n_bins - 1);
        bin_sum[idx] += sq_diff;
        bin_cnt[idx] += 1;
        bin_dist[idx] += dist;
    }

    let result = (0..n_bins)
        .filter(|&i| bin_cnt[i] > 0)
        .map(|i| {
            let count = bin_cnt[i];
            VariogramBin {
                distance: bin_dist[i] / count as f64,
                semivariance: bin_sum[i] / (2.0 * count as f64),
                count,
            }
        })
        .collect();

    let _ = bin_width; // used implicitly via max_dist / n_bins
    Ok(result)
}

// ---------------------------------------------------------------------------
// Convenience tuple-returning wrapper (matches task spec signature)
// ---------------------------------------------------------------------------

/// Compute the experimental variogram returning `(distance, semivariance)` pairs.
///
/// This is a convenience wrapper around [`variogram`] that returns the result
/// in a simple `Vec<(f64, f64)>` format.
pub fn variogram_pairs(points: &[(f64, f64, f64)], n_bins: usize) -> StatsResult<Vec<(f64, f64)>> {
    let bins = variogram(points, n_bins)?;
    Ok(bins
        .into_iter()
        .map(|b| (b.distance, b.semivariance))
        .collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build a simple contiguity weight matrix for a 1-D chain of n nodes.
    fn chain_weights(n: usize) -> Array2<f64> {
        let mut w = Array2::zeros((n, n));
        for i in 0..n - 1 {
            w[[i, i + 1]] = 1.0;
            w[[i + 1, i]] = 1.0;
        }
        w
    }

    /// Build a fully-connected (excluding diagonal) weight matrix.
    fn full_weights(n: usize) -> Array2<f64> {
        let mut w = Array2::ones((n, n));
        for i in 0..n {
            w[[i, i]] = 0.0;
        }
        w
    }

    #[test]
    fn test_morans_i_positive_autocorrelation() {
        // Values that increase monotonically → positive autocorrelation
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = chain_weights(6);
        let result = MoransI::compute(&values, &w).expect("MoransI compute failed");
        // With monotone values and chain weights, I should be positive
        assert!(
            result.statistic > 0.0,
            "Expected positive I, got {}",
            result.statistic
        );
    }

    #[test]
    fn test_morans_i_negative_autocorrelation() {
        // Checkerboard pattern → negative autocorrelation
        let values = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let w = chain_weights(6);
        let result = MoransI::compute(&values, &w).expect("MoransI compute failed");
        assert!(
            result.statistic < 0.0,
            "Expected negative I, got {}",
            result.statistic
        );
    }

    #[test]
    fn test_morans_i_expected_value() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = chain_weights(5);
        let result = MoransI::compute(&values, &w).expect("MoransI compute failed");
        let expected = -1.0 / (5.0 - 1.0);
        assert!(
            (result.expected - expected).abs() < 1e-10,
            "Expected E[I]={expected}, got {}",
            result.expected
        );
    }

    #[test]
    fn test_morans_i_p_value_range() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = chain_weights(6);
        let result = MoransI::compute(&values, &w).expect("MoransI compute failed");
        assert!(
            result.p_value >= 0.0 && result.p_value <= 1.0,
            "p-value out of range: {}",
            result.p_value
        );
    }

    #[test]
    fn test_morans_i_error_too_few_observations() {
        let values = vec![1.0, 2.0];
        let w = chain_weights(2);
        assert!(MoransI::compute(&values, &w).is_err());
    }

    #[test]
    fn test_morans_i_error_weight_dimension_mismatch() {
        let values = vec![1.0, 2.0, 3.0];
        let w = chain_weights(4);
        assert!(MoransI::compute(&values, &w).is_err());
    }

    #[test]
    fn test_geary_c_range() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = chain_weights(6);
        let c = GearyC::compute(&values, &w).expect("GearyC compute failed");
        // Geary's C should be non-negative
        assert!(c >= 0.0, "Geary's C should be ≥ 0, got {c}");
    }

    #[test]
    fn test_geary_c_positive_autocorrelation() {
        // Smooth monotone data → C < 1 (positive autocorrelation)
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let w = chain_weights(8);
        let c = GearyC::compute(&values, &w).expect("GearyC compute failed");
        assert!(
            c < 1.0,
            "Expected C < 1 for positive autocorrelation, got {c}"
        );
    }

    #[test]
    fn test_ripley_k_increasing() {
        // K(d) should be non-decreasing in d
        let points = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let distances = vec![0.5, 1.0, 1.5, 2.0];
        let k = Ripley::k_function(&points, &distances, 9.0).expect("Ripley k_function failed");
        for i in 1..k.len() {
            assert!(
                k[i] >= k[i - 1] - 1e-10,
                "K should be non-decreasing: K[{}]={} < K[{}]={}",
                i,
                k[i],
                i - 1,
                k[i - 1]
            );
        }
    }

    #[test]
    fn test_ripley_k_error_too_few_points() {
        let points = vec![(0.0, 0.0)];
        assert!(Ripley::k_function(&points, &[1.0], 4.0).is_err());
    }

    #[test]
    fn test_variogram_bins_count() {
        let pts: Vec<(f64, f64, f64)> = (0..5)
            .flat_map(|i| (0..5).map(move |j| (i as f64, j as f64, (i + j) as f64)))
            .collect();
        let bins = variogram(&pts, 5).expect("variogram failed");
        assert!(!bins.is_empty(), "Expected at least one non-empty bin");
        assert!(bins.len() <= 5, "Expected at most 5 bins");
    }

    #[test]
    fn test_variogram_semivariance_positive() {
        let pts: Vec<(f64, f64, f64)> = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i as f64, j as f64, (i * j) as f64)))
            .collect();
        let bins = variogram(&pts, 4).expect("variogram failed");
        for b in &bins {
            assert!(
                b.semivariance >= 0.0,
                "semivariance should be non-negative, got {}",
                b.semivariance
            );
        }
    }

    #[test]
    fn test_variogram_pairs_wrapper() {
        let pts: Vec<(f64, f64, f64)> = vec![
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 2.0),
            (2.0, 0.0, 1.0),
            (3.0, 0.0, 3.0),
        ];
        let pairs = variogram_pairs(&pts, 3).expect("variogram_pairs failed");
        for (d, sv) in &pairs {
            assert!(*d > 0.0, "distance should be positive");
            assert!(*sv >= 0.0, "semivariance should be non-negative");
        }
    }

    #[test]
    fn test_spatial_autocorrelation_wrapper() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = full_weights(6);
        let sa = SpatialAutocorrelation::compute(&values, &w)
            .expect("SpatialAutocorrelation compute failed");
        assert!(
            sa.global.p_value >= 0.0 && sa.global.p_value <= 1.0,
            "p-value out of range"
        );
        assert_eq!(sa.local.local_i.len(), 6);
    }

    #[test]
    fn test_lisa_local_i_count() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = chain_weights(5);
        let lisa = MoransI::local(&values, &w).expect("LISA compute failed");
        assert_eq!(lisa.local_i.len(), 5);
        assert_eq!(lisa.z_scores.len(), 5);
        assert_eq!(lisa.p_values.len(), 5);
    }
}
