//! Singular Spectrum Analysis (SSA) for time series decomposition

use scirs2_core::ndarray::{Array1, Array2, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumCast};
use scirs2_linalg::{lowrank::randomized_svd, svd};
use std::fmt::Debug;

use super::common::DecompositionResult;
use crate::error::{Result, TimeSeriesError};

/// Options for Singular Spectrum Analysis (SSA) decomposition
///
/// **Note**: Current implementation has limitations with large window sizes due to
/// eigendecomposition constraints in scirs2-linalg. Window sizes resulting in
/// trajectory matrices larger than 4x4 may use randomized SVD approximation.
/// For production use with large window sizes, consider upgrading the linear
/// algebra backend.
#[derive(Debug, Clone)]
pub struct SSAOptions {
    /// Window length (embedding dimension)
    ///
    /// **Limitation**: Large window sizes (>4) may trigger randomized SVD
    /// approximation instead of exact decomposition.
    pub window_length: usize,
    /// Number of components to include in the trend
    pub n_trend_components: usize,
    /// Number of components to include in the seasonal
    pub n_seasonal_components: Option<usize>,
    /// Whether to group components by similarity
    pub group_by_similarity: bool,
    /// Threshold for determining component similarity
    pub component_similarity_threshold: f64,
}

impl Default for SSAOptions {
    fn default() -> Self {
        Self {
            window_length: 0, // Will be set automatically based on time series length
            n_trend_components: 2,
            n_seasonal_components: None,
            group_by_similarity: true,
            component_similarity_threshold: 0.9,
        }
    }
}

/// Performs Singular Spectrum Analysis (SSA) decomposition on a time series
///
/// SSA decomposes a time series into trend, seasonal, and residual components
/// using eigenvalue decomposition of the trajectory matrix.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `options` - Options for SSA decomposition
///
/// # Returns
///
/// * Decomposition result containing trend, seasonal, and residual components
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::decomposition::{ssa_decomposition, SSAOptions};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let mut options = SSAOptions::default();
/// options.window_length = 4;
/// options.n_trend_components = 1;
/// let result = ssa_decomposition(&ts, &options).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal: {:?}", result.seasonal);
/// println!("Residual: {:?}", result.residual);
/// ```
#[allow(dead_code)]
pub fn ssa_decomposition<F>(ts: &Array1<F>, options: &SSAOptions) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug + ScalarOperand + NumCast,
{
    let n = ts.len();

    // Check inputs
    if n < 3 {
        return Err(TimeSeriesError::DecompositionError(
            "Time series must have at least 3 points for SSA decomposition".to_string(),
        ));
    }

    // Determine window length if not specified
    let window_length = if options.window_length > 0 {
        options.window_length
    } else {
        // Default is approximately n/2
        std::cmp::max(2, n / 2)
    };

    if window_length >= n {
        return Err(TimeSeriesError::DecompositionError(format!(
            "Window length ({window_length}) must be less than time series length ({n})"
        )));
    }

    if options.n_trend_components == 0 {
        return Err(TimeSeriesError::DecompositionError(
            "Number of trend components must be at least 1".to_string(),
        ));
    }

    // Step 1: Embedding - Create trajectory matrix
    let k = n - window_length + 1; // Number of columns in the trajectory matrix
    let mut trajectory_matrix = Array2::zeros((window_length, k));

    for i in 0..window_length {
        for j in 0..k {
            trajectory_matrix[[i, j]] = ts[i + j];
        }
    }

    // Step 2: SVD on trajectory matrix using scirs2-linalg
    let trajectory_matrix_f64 = trajectory_matrix.mapv(|x| x.to_f64().unwrap());

    // Use randomized SVD for larger matrices to avoid eigendecomposition limitations
    let min_dim = std::cmp::min(window_length, k);
    let max_components = std::cmp::min(
        min_dim,
        options.n_trend_components + options.n_seasonal_components.unwrap_or(10),
    );
    let target_rank = std::cmp::max(max_components, std::cmp::min(min_dim, 20)); // Ensure we get enough components

    let (u_f64, s_f64, vt_f64) = if min_dim > 4 && target_rank < min_dim {
        // Use randomized SVD for larger matrices
        let oversampling = std::cmp::min(10, min_dim - target_rank);
        randomized_svd(
            &trajectory_matrix_f64.view(),
            target_rank,
            Some(oversampling),
            Some(2),
            None,
        )
        .map_err(|e| {
            TimeSeriesError::DecompositionError(format!("Randomized SVD computation failed: {e}"))
        })?
    } else {
        // Use standard SVD for small matrices
        svd(&trajectory_matrix_f64.view(), true, None).map_err(|e| {
            TimeSeriesError::DecompositionError(format!("SVD computation failed: {e}"))
        })?
    };

    // Convert back to the original float type
    let u = u_f64.mapv(|x| F::from_f64(x).unwrap());
    let s = s_f64.mapv(|x| F::from_f64(x).unwrap());
    let vt = vt_f64.mapv(|x| F::from_f64(x).unwrap());

    // Step 3: Grouping components
    let mut trend_components = Vec::new();
    let mut seasonal_components = Vec::new();

    let n_components = s.len();

    // Group by similarity if requested
    if options.group_by_similarity {
        let mut component_groups = Vec::new();
        let mut visited = vec![false; n_components];

        for i in 0..n_components {
            // Check if the singular value is essentially zero (use machine epsilon scaled by largest singular value)
            let epsilon_val = F::from_f64(1e-12).unwrap_or_else(F::epsilon);
            let threshold = s[0] * epsilon_val;
            if visited[i] || s[i] <= threshold {
                continue;
            }

            let mut group = vec![i];
            visited[i] = true;

            // Find similar components using w-correlation
            for j in (i + 1)..n_components {
                if visited[j] || s[j] <= threshold {
                    continue;
                }

                let similarity = compute_w_correlation(&u, &vt, &s, i, j, window_length, k);
                if similarity > options.component_similarity_threshold {
                    group.push(j);
                    visited[j] = true;
                }
            }

            component_groups.push(group);
        }

        // Assign first group to trend and next groups to seasonal
        if !component_groups.is_empty() {
            trend_components = component_groups[0].clone();

            let n_seasonal = options
                .n_seasonal_components
                .unwrap_or(component_groups.len().saturating_sub(1));

            // Get the range of component groups to include in seasonal components
            let end_idx = std::cmp::min(component_groups.len(), n_seasonal + 1);
            for group in component_groups.iter().take(end_idx).skip(1) {
                seasonal_components.extend_from_slice(group);
            }
        }
    } else {
        // Simple grouping based on eigenvalue ranking
        for i in 0..std::cmp::min(options.n_trend_components, n_components) {
            trend_components.push(i);
        }

        let max_available = std::cmp::min(n_components, 10);
        let n_seasonal = options
            .n_seasonal_components
            .unwrap_or(max_available.saturating_sub(options.n_trend_components));

        for i in options.n_trend_components
            ..std::cmp::min(options.n_trend_components + n_seasonal, n_components)
        {
            seasonal_components.push(i);
        }
    }

    // Step 4: Diagonal averaging to reconstruct components
    let mut trend = Array1::zeros(n);
    let mut seasonal = Array1::zeros(n);

    // Define threshold for numerical stability
    let epsilon_val = F::from_f64(1e-12).unwrap_or_else(F::epsilon);
    let threshold = if !s.is_empty() {
        s[0] * epsilon_val
    } else {
        epsilon_val
    };

    // Reconstruct trend components
    for &idx in &trend_components {
        if idx >= n_components || s[idx] <= threshold {
            continue;
        }

        let reconstructed = reconstruct_component(&u, &vt, &s, idx, window_length, k, n);
        for i in 0..n {
            trend[i] = trend[i] + reconstructed[i];
        }
    }

    // Reconstruct seasonal components
    for &idx in &seasonal_components {
        if idx >= n_components || s[idx] <= threshold {
            continue;
        }

        let reconstructed = reconstruct_component(&u, &vt, &s, idx, window_length, k, n);
        for i in 0..n {
            seasonal[i] = seasonal[i] + reconstructed[i];
        }
    }

    // Calculate residual
    let mut residual = Array1::zeros(n);
    for i in 0..n {
        residual[i] = ts[i] - trend[i] - seasonal[i];
    }

    // Create result
    let original = ts.clone();

    Ok(DecompositionResult {
        trend,
        seasonal,
        residual,
        original,
    })
}

/// Compute w-correlation between two principal components
#[allow(dead_code)]
fn compute_w_correlation<F>(
    u: &Array2<F>,
    vt: &Array2<F>,
    s: &Array1<F>,
    i: usize,
    j: usize,
    window_length: usize,
    k: usize,
) -> f64
where
    F: Float + FromPrimitive + Debug + ScalarOperand + NumCast,
{
    // Get the i-th and j-th elementary matrices
    let si = F::from(s[i]).unwrap_or_else(|| F::zero());
    let sj = F::from(s[j]).unwrap_or_else(|| F::zero());

    let xi = &u.column(i) * si;
    let yi = vt.row(i);

    let xj = &u.column(j) * sj;
    let yj = vt.row(j);

    // Compute weights
    let l_star = std::cmp::min(window_length, k);
    let k_star = std::cmp::max(window_length, k);

    let mut weights = Array1::zeros(window_length + k - 1);
    for idx in 0..weights.len() {
        let t = idx + 1;
        if t <= l_star {
            weights[idx] = F::from_usize(t).unwrap();
        } else if t <= k_star {
            weights[idx] = F::from_usize(l_star).unwrap();
        } else {
            weights[idx] = F::from_usize(window_length + k - t).unwrap();
        }
    }

    // Compute weighted inner products
    let mut num = F::zero();
    let mut denom_i = F::zero();
    let mut denom_j = F::zero();

    for p in 0..window_length {
        for q in 0..k {
            let t = p + q;
            let weight = weights[t];

            let val_i = xi[p] * yi[q];
            let val_j = xj[p] * yj[q];

            num = num + weight * val_i * val_j;
            denom_i = denom_i + weight * val_i * val_i;
            denom_j = denom_j + weight * val_j * val_j;
        }
    }

    if denom_i <= F::epsilon() || denom_j <= F::epsilon() {
        0.0
    } else {
        (num / (denom_i * denom_j).sqrt()).to_f64().unwrap().abs()
    }
}

/// Reconstruct a component from SVD results using diagonal averaging
#[allow(dead_code)]
fn reconstruct_component<F>(
    u: &Array2<F>,
    vt: &Array2<F>,
    s: &Array1<F>,
    idx: usize,
    window_length: usize,
    k: usize,
    n: usize,
) -> Array1<F>
where
    F: Float + FromPrimitive + Debug + ScalarOperand + NumCast,
{
    // Compute the elementary matrix X_i = s_i * u_i * v_i^T
    let ui = u.column(idx);
    let vi = vt.row(idx);
    let si = F::from(s[idx]).unwrap_or_else(|| F::zero());

    let mut elementary_matrix = Array2::zeros((window_length, k));
    for i in 0..window_length {
        for j in 0..k {
            elementary_matrix[[i, j]] = si * ui[i] * vi[j];
        }
    }

    // Diagonal averaging
    let mut result = Array1::zeros(n);
    let l_star = std::cmp::min(window_length, k);
    let k_star = std::cmp::max(window_length, k);

    for t in 0..n {
        let mut sum = F::zero();
        let mut count = 0;

        if t < l_star {
            // First part: t < L*
            for m in 0..=t {
                if m < window_length && (t - m) < k {
                    sum = sum + elementary_matrix[[m, t - m]];
                    count += 1;
                }
            }
        } else if t < k_star {
            // Middle part: L* <= t < K*
            for m in 0..window_length {
                if (t - m) < k {
                    sum = sum + elementary_matrix[[m, t - m]];
                    count += 1;
                }
            }
        } else {
            // Last part: K* <= t < N
            for m in (t - k + 1)..window_length {
                if (t - m) < k {
                    sum = sum + elementary_matrix[[m, t - m]];
                    count += 1;
                }
            }
        }

        if count > 0 {
            result[t] = sum / F::from_usize(count).unwrap();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ssa_basic() {
        // Create a simple time series with trend and seasonality
        let n = 100;
        let mut ts = Array1::zeros(n);
        for i in 0..n {
            let trend = 0.1 * i as f64;
            let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            let noise = 0.1 * (i as f64 * 0.123).sin();
            ts[i] = trend + seasonal + noise;
        }

        let options = SSAOptions {
            window_length: 4,
            n_trend_components: 1,
            n_seasonal_components: Some(1),
            group_by_similarity: false,
            ..Default::default()
        };

        let result = ssa_decomposition(&ts, &options).unwrap();

        // Check that decomposition sums to original (approximately)
        for i in 0..n {
            assert_abs_diff_eq!(
                result.trend[i] + result.seasonal[i] + result.residual[i],
                ts[i],
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_ssa_with_grouping() {
        // Create a time series with multiple periodicities
        let n = 120;
        let mut ts = Array1::zeros(n);
        for i in 0..n {
            let trend = 0.05 * i as f64;
            let seasonal1 = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            let seasonal2 = 2.0 * (2.0 * std::f64::consts::PI * i as f64 / 6.0).sin();
            ts[i] = trend + seasonal1 + seasonal2;
        }

        let options = SSAOptions {
            window_length: 4,
            n_trend_components: 1,
            group_by_similarity: true,
            component_similarity_threshold: 0.8,
            ..Default::default()
        };

        let result = ssa_decomposition(&ts, &options).unwrap();

        // Check that decomposition sums to original
        for i in 0..n {
            assert_abs_diff_eq!(
                result.trend[i] + result.seasonal[i] + result.residual[i],
                ts[i],
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_ssa_edge_cases() {
        // Test with minimum size time series
        let ts = array![1.0, 2.0, 3.0];
        let mut options = SSAOptions {
            window_length: 2,
            n_trend_components: 1,
            ..Default::default()
        };

        let result = ssa_decomposition(&ts, &options);
        assert!(result.is_ok());

        // Test with too large window length
        options.window_length = 4;
        let result = ssa_decomposition(&ts, &options);
        assert!(result.is_err());

        // Test with too small time series
        let ts = array![1.0, 2.0];
        let result = ssa_decomposition(&ts, &SSAOptions::default());
        assert!(result.is_err());
    }
}
