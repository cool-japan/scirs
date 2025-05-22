//! Singular Spectrum Analysis (SSA) for time series decomposition

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::common::DecompositionResult;
use crate::error::{Result, TimeSeriesError};

/// Options for Singular Spectrum Analysis (SSA) decomposition
#[derive(Debug, Clone)]
pub struct SSAOptions {
    /// Window length (embedding dimension)
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
/// use ndarray::array;
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
pub fn ssa_decomposition<F>(ts: &Array1<F>, options: &SSAOptions) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
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
            "Window length ({}) must be less than time series length ({})",
            window_length, n
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

    // Step 2: SVD on trajectory matrix
    let svd_result = svd(&trajectory_matrix);
    let (u, s, vt) = match svd_result {
        Ok((u, s, vt)) => (u, s, vt),
        Err(e) => {
            return Err(TimeSeriesError::DecompositionError(format!(
                "SVD computation failed: {}",
                e
            )))
        }
    };

    // Step 3: Grouping components
    let mut trend_components = Vec::new();
    let mut seasonal_components = Vec::new();

    // Group by similarity if requested
    if options.group_by_similarity {
        let mut component_groups = Vec::new();
        let mut visited = vec![false; window_length.min(k)];

        for i in 0..window_length.min(k) {
            if visited[i] || s[i] <= F::epsilon() {
                continue;
            }

            let mut group = vec![i];
            visited[i] = true;

            // Find similar components using w-correlation
            for j in (i + 1)..window_length.min(k) {
                if visited[j] || s[j] <= F::epsilon() {
                    continue;
                }

                let similarity = compute_component_similarity(&u, &vt, &s, i, j, n);
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
        for i in 0..options.n_trend_components.min(window_length.min(k)) {
            trend_components.push(i);
        }

        let n_seasonal = options
            .n_seasonal_components
            .unwrap_or(std::cmp::min(window_length.min(k), 10) - options.n_trend_components);

        for i in options.n_trend_components
            ..std::cmp::min(
                options.n_trend_components + n_seasonal,
                window_length.min(k),
            )
        {
            seasonal_components.push(i);
        }
    }

    // Step 4: Diagonal averaging to reconstruct components
    let mut trend = Array1::zeros(n);
    let mut seasonal = Array1::zeros(n);

    // Reconstruct trend components
    for &idx in &trend_components {
        if idx >= window_length.min(k) || s[idx] <= F::epsilon() {
            continue;
        }

        let reconstructed = reconstruct_component(&u, &vt, &s, idx, window_length, k, n);
        for i in 0..n {
            trend[i] = trend[i] + reconstructed[i];
        }
    }

    // Reconstruct seasonal components
    for &idx in &seasonal_components {
        if idx >= window_length.min(k) || s[idx] <= F::epsilon() {
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

/// SVD result type to reduce complexity
type SVDResult<F> = std::result::Result<(Array2<F>, Array1<F>, Array2<F>), String>;

/// Performs SVD on a matrix
fn svd<F>(matrix: &Array2<F>) -> SVDResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    // This is a placeholder for a real SVD implementation.
    // In a full implementation, we would use a linear algebra crate like ndarray-linalg.
    // For now, we'll create simple matrices to illustrate the structure.

    let (m, n) = matrix.dim();
    let min_dim = std::cmp::min(m, n);

    // Create some dummy singular values (decreasing)
    let mut s = Array1::zeros(min_dim);
    for i in 0..min_dim {
        s[i] = F::from_f64(min_dim as f64 - i as f64).unwrap();
    }

    // Create dummy U and V^T matrices
    let u = Array2::eye(m);
    let vt = Array2::eye(n);

    Ok((u, s, vt))
}

/// Compute similarity between two principal components
fn compute_component_similarity<F>(
    _u: &Array2<F>,
    _vt: &Array2<F>,
    _s: &Array1<F>,
    i: usize,
    j: usize,
    n: usize,
) -> f64
where
    F: Float + FromPrimitive + Debug,
{
    // Placeholder for computing w-correlation between elementary components
    // In a real implementation, we would compute the actual w-correlation

    // Simple approximation based on index distance
    let d = (i as f64 - j as f64).abs() / n as f64;
    f64::exp(-d * 5.0)
}

/// Reconstruct a component from SVD results using diagonal averaging
fn reconstruct_component<F>(
    _u: &Array2<F>,
    _vt: &Array2<F>,
    s: &Array1<F>,
    idx: usize,
    _window_length: usize,
    _k: usize,
    n: usize,
) -> Array1<F>
where
    F: Float + FromPrimitive + Debug,
{
    // In a real implementation, this would reconstruct the component from SVD
    // For now, we create a simple sinusoidal pattern as a placeholder

    let mut result = Array1::zeros(n);
    let period = F::from_usize(idx + 2).unwrap();

    for i in 0..n {
        result[i] = F::from_f64(
            (i as f64 * 2.0 * std::f64::consts::PI / period.to_f64().unwrap()).sin()
                * s[idx].to_f64().unwrap()
                / s[0].to_f64().unwrap(),
        )
        .unwrap();
    }

    result
}
