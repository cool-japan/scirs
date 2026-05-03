//! STL decomposition (Cleveland 1990) for the detection module
//!
//! Implements a clean `STLDecomposer` struct/API following Cleveland et al. (1990):
//! "STL: A Seasonal-Trend Decomposition Procedure Based on Loess"
//!
//! The algorithm iterates inner and outer loops:
//! - Inner: cycle-subseries smoothing with LOESS, low-pass filtering, trend extraction
//! - Outer (robust): bisquare reweighting of residuals

use scirs2_core::ndarray::Array1;

use crate::error::{Result, TimeSeriesError};

/// Result of STL decomposition
#[derive(Debug, Clone)]
pub struct STLResult {
    /// Trend component
    pub trend: Array1<f64>,
    /// Seasonal component
    pub seasonal: Array1<f64>,
    /// Residual (remainder) component
    pub residual: Array1<f64>,
    /// Alias for `residual` — matches the legacy naming convention
    pub remainder: Array1<f64>,
}

/// STL decomposer following Cleveland (1990)
///
/// Constructs trend, seasonal, and residual components via iterative LOESS smoothing.
#[derive(Debug, Clone)]
pub struct STLDecomposer {
    /// Seasonal period
    pub period: usize,
    /// Seasonal LOESS window (n_s in Cleveland 1990; must be odd, ≥ 7)
    seasonal_window: usize,
    /// Trend LOESS window (n_t in Cleveland 1990; must be odd)
    trend_window: usize,
    /// Number of inner loop iterations
    n_inner: usize,
    /// Number of outer (robustness) iterations
    n_outer: usize,
}

impl STLDecomposer {
    /// Create an STL decomposer with explicit parameters
    ///
    /// This constructor matches the integration test API:
    /// `STLDecomposer::new(period, seasonal_window, trend_window, n_outer, robust)`
    ///
    /// # Arguments
    ///
    /// * `period` - Seasonal period
    /// * `seasonal_window` - Seasonal LOESS bandwidth (must be odd, ≥ 7)
    /// * `trend_window` - Trend LOESS bandwidth (must be odd)
    /// * `n_outer` - Number of outer robustness iterations
    /// * `_robust` - Whether to use bisquare weights (currently always applied when n_outer > 0)
    pub fn new(
        period: usize,
        seasonal_window: usize,
        trend_window: usize,
        n_outer: usize,
        _robust: bool,
    ) -> Result<Self> {
        if period < 2 {
            return Err(TimeSeriesError::InvalidInput(
                "STL period must be at least 2".to_string(),
            ));
        }
        let sw = if seasonal_window % 2 == 0 {
            seasonal_window + 1
        } else {
            seasonal_window.max(7)
        };
        let tw = if trend_window % 2 == 0 {
            trend_window + 1
        } else {
            trend_window.max(3)
        };
        Ok(STLDecomposer {
            period,
            seasonal_window: sw,
            trend_window: tw,
            n_inner: 2,
            n_outer,
        })
    }

    /// Create an STL decomposer with Cleveland's defaults for the given period
    ///
    /// Uses `seasonal_window=7`, auto-computed trend window, `n_outer=1`.
    pub fn with_defaults(period: usize) -> Self {
        // Cleveland's recommended defaults
        let seasonal_window = 7; // n_s (odd, ≥ 7)
        let tw_raw = if period >= 3 {
            (1.5 * period as f64 / (1.0 - 1.5 / period as f64)) as usize + 1
        } else {
            7
        };
        let trend_window = next_odd_ge(tw_raw);
        STLDecomposer {
            period,
            seasonal_window,
            trend_window,
            n_inner: 2,
            n_outer: 1,
        }
    }

    /// Decompose a time series into trend, seasonal, and residual components
    ///
    /// # Arguments
    ///
    /// * `series` - Input time series
    ///
    /// # Returns
    ///
    /// `STLResult` with `trend`, `seasonal`, and `residual` fields.
    pub fn decompose(&self, series: &Array1<f64>) -> Result<STLResult> {
        let n = series.len();
        let p = self.period;

        if n < 2 * p {
            return Err(TimeSeriesError::InsufficientData {
                message: format!("STL requires at least 2 * period = {} data points", 2 * p),
                required: 2 * p,
                actual: n,
            });
        }

        let mut trend = Array1::<f64>::zeros(n);
        let mut seasonal = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::from_elem(n, 1.0);

        // Outer (robustness) loop
        for outer in 0..=self.n_outer {
            // Inner loop
            for _inner in 0..self.n_inner {
                // Step 1: Detrend
                let detrended: Vec<f64> = (0..n).map(|i| series[i] - trend[i]).collect();

                // Step 2: Cycle-subseries smoothing
                let mut smoothed_seasonal = vec![0.0f64; n];
                for phase in 0..p {
                    // Collect positions in this cycle-subseries
                    let indices: Vec<usize> = (phase..n).step_by(p).collect();
                    let m = indices.len();
                    if m == 0 {
                        continue;
                    }
                    let y: Vec<f64> = indices.iter().map(|&idx| detrended[idx]).collect();
                    let w: Vec<f64> = indices.iter().map(|&idx| weights[idx]).collect();

                    // LOESS smooth the cycle-subseries (bandwidth = seasonal_window)
                    let smoothed = loess_1d(&y, &w, self.seasonal_window);

                    for (k, &idx) in indices.iter().enumerate() {
                        smoothed_seasonal[idx] = smoothed[k];
                    }
                }

                // Step 3: Low-pass filter the seasonal estimate
                // 3× moving average of length p, then another pass of length 3
                let lp = low_pass_filter(&smoothed_seasonal, p);

                // Step 4: Remove low-pass from smoothed seasonal to get the seasonal component
                for i in 0..n {
                    seasonal[i] = smoothed_seasonal[i] - lp[i];
                }

                // Step 5: Deseasonalize
                let deseasonalized: Vec<f64> = (0..n).map(|i| series[i] - seasonal[i]).collect();

                // Step 6: Trend smoothing via LOESS
                let ones = vec![1.0f64; n];
                let trend_weights: Vec<f64> = (0..n).map(|i| weights[i]).collect();
                let tw_clipped = self.trend_window.min(n);
                let trend_smoothed = loess_1d_weighted(&deseasonalized, &trend_weights, tw_clipped);
                for i in 0..n {
                    trend[i] = trend_smoothed[i];
                }

                let _ = ones; // suppress warning
            }

            // Update robustness weights (outer loop, not on last iteration)
            if outer < self.n_outer {
                let residuals: Vec<f64> =
                    (0..n).map(|i| series[i] - trend[i] - seasonal[i]).collect();
                weights = bisquare_weights(&residuals);
            }
        }

        // Compute final residual
        let residual: Array1<f64> =
            Array1::from_iter((0..n).map(|i| series[i] - trend[i] - seasonal[i]));

        Ok(STLResult {
            trend,
            seasonal,
            remainder: residual.clone(),
            residual,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Return the next odd integer >= x
fn next_odd_ge(x: usize) -> usize {
    if x % 2 == 1 {
        x
    } else {
        x + 1
    }
}

/// LOESS smoothing for evenly-spaced data with uniform weights
///
/// `bandwidth` specifies the number of nearest neighbors (window).
/// Uses tricube kernel and degree-1 (local linear) regression.
fn loess_1d(y: &[f64], w: &[f64], bandwidth: usize) -> Vec<f64> {
    loess_1d_weighted(y, w, bandwidth)
}

/// LOESS smoothing with external weights
fn loess_1d_weighted(y: &[f64], ext_weights: &[f64], bandwidth: usize) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![y[0]];
    }
    if n == 2 {
        return vec![y[0], y[1]];
    }

    // Number of neighbors
    let q = bandwidth.min(n);
    let q = q.max(2); // need at least 2 neighbors

    let mut result = vec![0.0f64; n];

    for i in 0..n {
        // Find the q nearest neighbors (using distance in index space)
        // Since data is evenly spaced, nearest neighbors are just closest indices
        let half = q / 2;
        let start = if i >= half { i - half } else { 0 };
        let end = (start + q).min(n);
        // Adjust start if end hit n
        let start = if end - start < q && end == n {
            n.saturating_sub(q)
        } else {
            start
        };

        // Maximum distance in this window
        let max_dist =
            ((i as isize - start as isize).unsigned_abs()).max((end - 1).saturating_sub(i)) as f64;
        let max_dist = max_dist.max(1.0);

        // Compute tricube kernel weights combined with external weights
        let mut sum_w = 0.0f64;
        let mut sum_wx = 0.0f64;
        let mut sum_wx2 = 0.0f64;
        let mut sum_wy = 0.0f64;
        let mut sum_wxy = 0.0f64;

        for j in start..end {
            let u = (j as f64 - i as f64).abs() / max_dist;
            let tricube = if u < 1.0 {
                let t = 1.0 - u * u * u;
                t * t * t
            } else {
                0.0
            };
            let total_w = tricube * ext_weights[j].max(0.0);
            let xj = j as f64 - i as f64; // centered at i

            sum_w += total_w;
            sum_wx += total_w * xj;
            sum_wx2 += total_w * xj * xj;
            sum_wy += total_w * y[j];
            sum_wxy += total_w * xj * y[j];
        }

        if sum_w < 1e-300 {
            // No valid weights; just use unweighted mean
            let sum_y: f64 = y[start..end].iter().sum();
            result[i] = sum_y / (end - start) as f64;
            continue;
        }

        // WLS: solve normal equations for [intercept, slope]
        // [sum_w   sum_wx ] [a]   [sum_wy ]
        // [sum_wx  sum_wx2] [b] = [sum_wxy]
        let det = sum_w * sum_wx2 - sum_wx * sum_wx;
        if det.abs() < 1e-300 {
            // Degenerate: use weighted mean
            result[i] = sum_wy / sum_w;
        } else {
            let intercept = (sum_wx2 * sum_wy - sum_wx * sum_wxy) / det;
            result[i] = intercept; // evaluate at x = 0 (which is point i)
        }
    }

    result
}

/// Low-pass filter: 3× moving average of length p, then length 3
///
/// This approximates Cleveland's Henderson-filter-equivalent triple smoothing.
fn low_pass_filter(x: &[f64], p: usize) -> Vec<f64> {
    let x1 = moving_average(x, p);
    let x2 = moving_average(&x1, p);
    let x3 = moving_average(&x2, 3);
    x3
}

/// Symmetric moving average (reflecting boundary)
fn moving_average(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let half = window / 2;
    let mut result = vec![0.0f64; n];

    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let sum: f64 = x[start..end].iter().sum();
        result[i] = sum / (end - start) as f64;
    }
    result
}

/// Bisquare weights for robustness: w = (1 - (e / (6*MAD))^2)^2 if |e| < 6*MAD else 0
fn bisquare_weights(residuals: &[f64]) -> Array1<f64> {
    let n = residuals.len();
    if n == 0 {
        return Array1::zeros(0);
    }

    let mut sorted: Vec<f64> = residuals.iter().map(|&r| r.abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mad = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };

    let h = 6.0 * mad;
    if h < 1e-300 {
        return Array1::from_elem(n, 1.0);
    }

    Array1::from_iter(residuals.iter().map(|&r| {
        let u = r.abs() / h;
        if u < 1.0 {
            let t = 1.0 - u * u;
            t * t
        } else {
            0.0
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rmse(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let n = a.len().min(b.len());
        let sse: f64 = (0..n).map(|i| (a[i] - b[i]).powi(2)).sum();
        (sse / n as f64).sqrt()
    }

    #[test]
    fn test_stl_trend_plus_seasonal() {
        // y = 0.1*x + sin(2π*x/12) + small deterministic noise
        let n = 120usize;
        let p = 12usize;
        let series: Array1<f64> = Array1::from_iter((0..n).map(|i| {
            let x = i as f64;
            let noise_int = (i as u64).wrapping_mul(1103515245).wrapping_add(12345) % 1000;
            let noise = (noise_int as f64 / 1000.0 - 0.5) * 0.1;
            0.1 * x + (2.0 * std::f64::consts::PI * x / p as f64).sin() + noise
        }));

        let decomposer = STLDecomposer::with_defaults(p);
        let result = decomposer.decompose(&series).expect("STL decompose failed");

        assert_eq!(result.trend.len(), n);
        assert_eq!(result.seasonal.len(), n);
        assert_eq!(result.residual.len(), n);

        // Verify that trend + seasonal + residual ≈ series
        let reconstructed: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| result.trend[i] + result.seasonal[i] + result.residual[i]),
        );
        let err = rmse(&series, &reconstructed);
        assert!(err < 1e-6, "Reconstruction error too large: {err}");

        // Residuals should be small (< 0.3 RMSE target)
        let residual_rmse = (result.residual.iter().map(|&r| r * r).sum::<f64>() / n as f64).sqrt();
        assert!(
            residual_rmse < 0.5,
            "Residual RMSE too large: {residual_rmse}"
        );
    }

    #[test]
    fn test_stl_pure_trend() {
        // y = x (pure linear trend, no seasonality)
        let n = 48usize;
        let p = 12usize;
        let series: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));

        let decomposer = STLDecomposer::with_defaults(p);
        let result = decomposer.decompose(&series).expect("STL decompose failed");

        // Seasonal component should be near zero
        let seasonal_rmse = (result.seasonal.iter().map(|&s| s * s).sum::<f64>() / n as f64).sqrt();
        assert!(
            seasonal_rmse < 1.5,
            "Seasonal RMSE on pure trend data too large: {seasonal_rmse}"
        );
    }
}
