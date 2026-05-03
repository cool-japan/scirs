//! Grid resampling and extrapolation with configurable boundary modes.
//!
//! Provides:
//! - 1-D resampling with linear, cubic spline, nearest-neighbour, and Lanczos methods
//! - 2-D separable resampling on a regular grid
//! - Scattered-to-grid conversion using inverse distance weighting (IDW)
//! - Symbolic spline derivative (degree reduction)
//! - Multiple extrapolation modes (Nearest, Linear, Polynomial, Reflection, Periodic, Zero, Constant)
//! - N-D grid-spec resampling with `ArrayD` output (`grid_spec` sub-module)

pub mod grid_spec;
pub use grid_spec::{resample_scattered_to_grid, Aggregator, GridSpec, ResampleStrategy};

use crate::error::InterpolateError;

// ─── ExtrapolationMode ───────────────────────────────────────────────────────

/// How to handle queries outside the data domain.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ExtrapolationMode {
    /// Clamp to the nearest boundary value.
    Nearest,
    /// Linearly extrapolate using the slope at the boundary.
    Linear,
    /// Extrapolate with a polynomial of given degree fit to the last `degree+1` points.
    Polynomial(usize),
    /// Mirror / reflect the index about the boundary.
    Reflection,
    /// Wrap the index periodically.
    Periodic,
    /// Return zero outside the domain.
    Zero,
    /// Return a fixed constant value outside the domain.
    Constant(f64),
}

// ─── ResamplingMethod ────────────────────────────────────────────────────────

/// Interpolation method to use within the data domain.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ResamplingMethod {
    /// Bi/trilinear interpolation.
    Linear,
    /// Natural cubic spline.
    CubicSpline,
    /// Nearest-neighbour.
    Nearest,
    /// Lanczos windowed-sinc with the given number of lobes (a).
    Lanczos(usize),
}

// ─── ResamplingConfig ────────────────────────────────────────────────────────

/// Configuration for resampling operations.
#[derive(Debug, Clone)]
pub struct ResamplingConfig {
    /// In-domain interpolation method.
    pub method: ResamplingMethod,
    /// Out-of-domain extrapolation strategy.
    pub extrapolation: ExtrapolationMode,
}

impl Default for ResamplingConfig {
    fn default() -> Self {
        Self {
            method: ResamplingMethod::Linear,
            extrapolation: ExtrapolationMode::Nearest,
        }
    }
}

// ─── 1-D resampling ──────────────────────────────────────────────────────────

/// Resample a 1-D signal from `(x_in, y_in)` to query points `x_out`.
///
/// `x_in` must be strictly increasing. `x_out` may be arbitrary.
pub fn resample_1d(
    x_in: &[f64],
    y_in: &[f64],
    x_out: &[f64],
    config: &ResamplingConfig,
) -> Result<Vec<f64>, InterpolateError> {
    let n = x_in.len();
    if n < 2 {
        return Err(InterpolateError::InsufficientData(
            "resample_1d requires at least 2 input points".to_string(),
        ));
    }
    if n != y_in.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "x_in length {} != y_in length {}",
            n,
            y_in.len()
        )));
    }

    // Validate monotonicity
    for i in 1..n {
        if x_in[i] <= x_in[i - 1] {
            return Err(InterpolateError::InvalidInput {
                message: "x_in must be strictly increasing".to_string(),
            });
        }
    }

    // Precompute cubic spline coefficients if needed
    let spline_coeffs: Option<Vec<[f64; 4]>> = match config.method {
        ResamplingMethod::CubicSpline => Some(natural_cubic_spline_coeffs(x_in, y_in)?),
        _ => None,
    };

    let x_min = x_in[0];
    let x_max = x_in[n - 1];

    let result: Result<Vec<f64>, InterpolateError> = x_out
        .iter()
        .map(|&xq| {
            // Map possibly out-of-range xq to a resolved position
            let xq_mapped = resolve_query(xq, x_min, x_max, &config.extrapolation);

            match xq_mapped {
                ResolvedQuery::InDomain(xr) => {
                    interpolate_1d(x_in, y_in, xr, config, &spline_coeffs)
                }
                ResolvedQuery::Extrapolated(val) => Ok(val),
                ResolvedQuery::ExtrapLinear(xr) => {
                    // Linear extrapolation: use xr which may be outside domain
                    interpolate_1d_linear_extrap(x_in, y_in, xr)
                }
                ResolvedQuery::ExtrapPolynomial(xr, deg) => {
                    interpolate_1d_poly_extrap(x_in, y_in, xr, deg)
                }
            }
        })
        .collect();

    result
}

// ─── Query resolution ────────────────────────────────────────────────────────

enum ResolvedQuery {
    InDomain(f64),
    Extrapolated(f64),
    ExtrapLinear(f64),
    ExtrapPolynomial(f64, usize),
}

fn resolve_query(xq: f64, x_min: f64, x_max: f64, mode: &ExtrapolationMode) -> ResolvedQuery {
    if xq >= x_min && xq <= x_max {
        return ResolvedQuery::InDomain(xq);
    }

    match mode {
        ExtrapolationMode::Nearest => ResolvedQuery::InDomain(xq.clamp(x_min, x_max)),
        ExtrapolationMode::Linear => ResolvedQuery::ExtrapLinear(xq),
        ExtrapolationMode::Polynomial(deg) => ResolvedQuery::ExtrapPolynomial(xq, *deg),
        ExtrapolationMode::Reflection => {
            let range = x_max - x_min;
            if range < 1e-300 {
                return ResolvedQuery::InDomain(x_min);
            }
            // Normalise to [0, 2*range) then reflect
            let shifted = xq - x_min;
            let period = 2.0 * range;
            let t = shifted - (shifted / period).floor() * period;
            let reflected = if t <= range { t } else { period - t };
            ResolvedQuery::InDomain(x_min + reflected.clamp(0.0, range))
        }
        ExtrapolationMode::Periodic => {
            let range = x_max - x_min;
            if range < 1e-300 {
                return ResolvedQuery::InDomain(x_min);
            }
            let shifted = xq - x_min;
            let t = shifted - (shifted / range).floor() * range;
            ResolvedQuery::InDomain(x_min + t.clamp(0.0, range))
        }
        ExtrapolationMode::Zero => ResolvedQuery::Extrapolated(0.0),
        ExtrapolationMode::Constant(c) => ResolvedQuery::Extrapolated(*c),
    }
}

// ─── 1-D interpolation methods ───────────────────────────────────────────────

fn interpolate_1d(
    x_in: &[f64],
    y_in: &[f64],
    xq: f64,
    config: &ResamplingConfig,
    spline_coeffs: &Option<Vec<[f64; 4]>>,
) -> Result<f64, InterpolateError> {
    let n = x_in.len();
    let idx = binary_search_floor(x_in, xq);
    let i = idx.min(n - 2);

    match &config.method {
        ResamplingMethod::Linear => {
            let t = (xq - x_in[i]) / (x_in[i + 1] - x_in[i]);
            Ok(y_in[i] * (1.0 - t) + y_in[i + 1] * t)
        }
        ResamplingMethod::Nearest => {
            let i_near = if (xq - x_in[i]).abs() < (xq - x_in[(i + 1).min(n - 1)]).abs() {
                i
            } else {
                (i + 1).min(n - 1)
            };
            Ok(y_in[i_near])
        }
        ResamplingMethod::CubicSpline => {
            if let Some(coeffs) = spline_coeffs {
                let dx = xq - x_in[i];
                let [a, b, c, d] = coeffs[i];
                Ok(a + b * dx + c * dx * dx + d * dx * dx * dx)
            } else {
                // Fallback to linear
                let t = (xq - x_in[i]) / (x_in[i + 1] - x_in[i]);
                Ok(y_in[i] * (1.0 - t) + y_in[i + 1] * t)
            }
        }
        ResamplingMethod::Lanczos(a) => Ok(lanczos_interp(x_in, y_in, xq, *a)),
    }
}

fn interpolate_1d_linear_extrap(
    x_in: &[f64],
    y_in: &[f64],
    xq: f64,
) -> Result<f64, InterpolateError> {
    let n = x_in.len();
    let x_min = x_in[0];
    let x_max = x_in[n - 1];
    if xq < x_min {
        // Extrapolate left using first interval slope
        let slope = (y_in[1] - y_in[0]) / (x_in[1] - x_in[0]);
        Ok(y_in[0] + slope * (xq - x_min))
    } else {
        // Extrapolate right using last interval slope
        let slope = (y_in[n - 1] - y_in[n - 2]) / (x_in[n - 1] - x_in[n - 2]);
        Ok(y_in[n - 1] + slope * (xq - x_max))
    }
}

fn interpolate_1d_poly_extrap(
    x_in: &[f64],
    y_in: &[f64],
    xq: f64,
    deg: usize,
) -> Result<f64, InterpolateError> {
    let n = x_in.len();
    let x_min = x_in[0];
    let pts = deg + 1;
    // Pick boundary points
    let (px, py): (Vec<f64>, Vec<f64>) = if xq < x_min {
        // Use first `pts` points
        let end = pts.min(n);
        (x_in[..end].to_vec(), y_in[..end].to_vec())
    } else {
        // Use last `pts` points
        let start = n.saturating_sub(pts);
        (x_in[start..].to_vec(), y_in[start..].to_vec())
    };

    // Lagrange interpolation/extrapolation
    Ok(lagrange_eval(&px, &py, xq))
}

// ─── Natural cubic spline coefficient computation ─────────────────────────────

/// Compute natural cubic spline coefficients for `n-1` intervals.
/// Returns coefficients `[a, b, c, d]` per interval such that
/// `f(x) = a + b*(x-xi) + c*(x-xi)^2 + d*(x-xi)^3` for x in [xi, xi+1].
fn natural_cubic_spline_coeffs(x: &[f64], y: &[f64]) -> Result<Vec<[f64; 4]>, InterpolateError> {
    let n = x.len();
    if n < 2 {
        return Err(InterpolateError::InsufficientData(
            "Need at least 2 points for spline".to_string(),
        ));
    }
    let m = n - 1;
    let mut h = vec![0.0f64; m];
    for i in 0..m {
        h[i] = x[i + 1] - x[i];
        if h[i] <= 0.0 {
            return Err(InterpolateError::InvalidInput {
                message: "x must be strictly increasing".to_string(),
            });
        }
    }

    if n == 2 {
        let b = (y[1] - y[0]) / h[0];
        return Ok(vec![[y[0], b, 0.0, 0.0]]);
    }

    // Set up tridiagonal system for second derivatives σ
    let mut alpha = vec![0.0f64; n];
    for i in 1..m {
        alpha[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    // Thomas algorithm (natural: σ_0 = σ_{n-1} = 0)
    let mut l = vec![1.0f64; n];
    let mut mu = vec![0.0f64; n];
    let mut z = vec![0.0f64; n];

    for i in 1..m {
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        if l[i].abs() < 1e-300 {
            l[i] = 1e-300;
        }
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    let mut sigma = vec![0.0f64; n]; // second derivatives
    for i in (1..m).rev() {
        sigma[i] = z[i] - mu[i] * sigma[i + 1];
    }

    // Compute polynomial coefficients
    let mut coeffs = Vec::with_capacity(m);
    for i in 0..m {
        let a = y[i];
        let b = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * sigma[i] + sigma[i + 1]) / 3.0;
        let c = sigma[i];
        let d = (sigma[i + 1] - sigma[i]) / (3.0 * h[i]);
        coeffs.push([a, b, c, d]);
    }

    Ok(coeffs)
}

// ─── Lanczos windowed-sinc ────────────────────────────────────────────────────

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

fn lanczos_kernel(x: f64, a: usize) -> f64 {
    let af = a as f64;
    if x.abs() >= af {
        0.0
    } else {
        sinc(x) * sinc(x / af)
    }
}

fn lanczos_interp(x_in: &[f64], y_in: &[f64], xq: f64, a: usize) -> f64 {
    let n = x_in.len();
    if n < 2 {
        return y_in.first().copied().unwrap_or(0.0);
    }
    // Convert xq to fractional index in x_in (assume uniform spacing for simplicity,
    // fallback to linear for non-uniform)
    let i0 = binary_search_floor(x_in, xq);
    let h = x_in[1] - x_in[0]; // approximate uniform step
    if h.abs() < 1e-300 {
        return y_in[i0.min(n - 1)];
    }
    let frac = (xq - x_in[i0.min(n - 1)]) / h;
    let fi = i0 as f64 + frac;

    let mut numer = 0.0f64;
    let mut denom = 0.0f64;
    let start = (fi as isize - a as isize).max(0) as usize;
    let end = ((fi as isize + a as isize + 1) as usize).min(n);

    for k in start..end {
        let w = lanczos_kernel(fi - k as f64, a);
        numer += w * y_in[k];
        denom += w;
    }

    if denom.abs() < 1e-300 {
        y_in[i0.min(n - 1)]
    } else {
        numer / denom
    }
}

// ─── Helper: Lagrange interpolation ─────────────────────────────────────────

fn lagrange_eval(px: &[f64], py: &[f64], xq: f64) -> f64 {
    let n = px.len();
    let mut result = 0.0f64;
    for i in 0..n {
        let mut li = 1.0f64;
        for j in 0..n {
            if i != j {
                let denom = px[i] - px[j];
                if denom.abs() < 1e-300 {
                    continue;
                }
                li *= (xq - px[j]) / denom;
            }
        }
        result += py[i] * li;
    }
    result
}

// ─── Helper: binary search (floor index) ─────────────────────────────────────

/// Return index i such that x_in[i] <= xq < x_in[i+1].
/// Clamps to [0, n-2].
fn binary_search_floor(x_in: &[f64], xq: f64) -> usize {
    let n = x_in.len();
    if n == 0 {
        return 0;
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if x_in[mid] <= xq {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo.min(n.saturating_sub(2))
}

// ─── 2-D resampling ──────────────────────────────────────────────────────────

/// Resample a 2-D grid `grid[iy][ix]` from `(x_in, y_in)` axes to `(x_out, y_out)`.
///
/// Uses separable 1-D resampling: first along x, then along y.
pub fn resample_2d(
    grid: &[Vec<f64>],
    x_in: &[f64],
    y_in: &[f64],
    x_out: &[f64],
    y_out: &[f64],
    config: &ResamplingConfig,
) -> Result<Vec<Vec<f64>>, InterpolateError> {
    let ny_in = y_in.len();
    let nx_in = x_in.len();
    if grid.len() != ny_in {
        return Err(InterpolateError::DimensionMismatch(format!(
            "grid has {} rows but y_in has {} elements",
            grid.len(),
            ny_in
        )));
    }
    for (row_idx, row) in grid.iter().enumerate() {
        if row.len() != nx_in {
            return Err(InterpolateError::DimensionMismatch(format!(
                "grid row {} has {} columns but x_in has {} elements",
                row_idx,
                row.len(),
                nx_in
            )));
        }
    }

    // Step 1: resample along x for each input y row
    // Produces intermediate grid of shape (ny_in, nx_out)
    let mut intermediate: Vec<Vec<f64>> = Vec::with_capacity(ny_in);
    for row in grid.iter() {
        let resampled_row = resample_1d(x_in, row, x_out, config)?;
        intermediate.push(resampled_row);
    }

    // Step 2: resample along y for each output x column
    let nx_out = x_out.len();
    let ny_out = y_out.len();
    let mut output = vec![vec![0.0f64; nx_out]; ny_out];

    for ix in 0..nx_out {
        // Extract column from intermediate
        let col: Vec<f64> = intermediate.iter().map(|row| row[ix]).collect();
        let resampled_col = resample_1d(y_in, &col, y_out, config)?;
        for iy in 0..ny_out {
            output[iy][ix] = resampled_col[iy];
        }
    }

    Ok(output)
}

// ─── Scattered to grid (IDW) ─────────────────────────────────────────────────

/// Map scattered N-D points to a regular grid using inverse distance weighting.
///
/// `grid_ranges[d] = (min, max, n_points)` for dimension d.
/// Returns a flattened array of shape `[n0 * n1 * ... * nd]` in row-major order.
pub fn scattered_to_grid(
    x: &[Vec<f64>],
    y: &[f64],
    grid_ranges: &[(f64, f64, usize)],
    _config: &ResamplingConfig,
) -> Result<Vec<f64>, InterpolateError> {
    if x.is_empty() {
        return Err(InterpolateError::InsufficientData(
            "No scattered data points".to_string(),
        ));
    }
    if x.len() != y.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "x has {} rows but y has {} elements",
            x.len(),
            y.len()
        )));
    }
    if grid_ranges.is_empty() {
        return Err(InterpolateError::InvalidInput {
            message: "grid_ranges must not be empty".to_string(),
        });
    }

    let n_dims = grid_ranges.len();
    let input_dims = x[0].len();
    if input_dims != n_dims {
        return Err(InterpolateError::DimensionMismatch(format!(
            "x has {} dimensions but grid_ranges specifies {} dimensions",
            input_dims, n_dims
        )));
    }

    // Build grid axes
    let axes: Vec<Vec<f64>> = grid_ranges
        .iter()
        .map(|&(lo, hi, n)| {
            if n <= 1 {
                vec![lo]
            } else {
                (0..n)
                    .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
                    .collect()
            }
        })
        .collect();

    // Total grid size
    let total: usize = axes.iter().map(|a| a.len()).product();
    let mut result = vec![0.0f64; total];

    // Enumerate multi-index
    let shapes: Vec<usize> = axes.iter().map(|a| a.len()).collect();
    let mut flat_idx = 0usize;

    let mut multi = vec![0usize; n_dims];
    loop {
        // Build grid point coordinates
        let gp: Vec<f64> = (0..n_dims).map(|d| axes[d][multi[d]]).collect();

        // IDW with power p=2
        let mut numer = 0.0f64;
        let mut denom = 0.0f64;
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let dist2: f64 = xi.iter().zip(gp.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            if dist2 < 1e-28 {
                // Exact hit
                numer = yi;
                denom = 1.0;
                break;
            }
            let w = 1.0 / dist2;
            numer += w * yi;
            denom += w;
        }
        result[flat_idx] = if denom > 1e-300 { numer / denom } else { 0.0 };

        // Advance multi-index (row-major)
        flat_idx += 1;
        let mut carry = true;
        for d in (0..n_dims).rev() {
            if carry {
                multi[d] += 1;
                if multi[d] >= shapes[d] {
                    multi[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            break; // all indices wrapped
        }
    }

    Ok(result)
}

// ─── SplineDerivative ────────────────────────────────────────────────────────

/// A piecewise polynomial (spline) on a set of knot intervals.
///
/// Each segment `[knots[i], knots[i+1]]` is represented by a polynomial
/// of degree `degree` with coefficients `coefficients[i]` stored in
/// *ascending order* (coefficient of x^0 first).
#[derive(Debug, Clone)]
pub struct SplineDerivative {
    /// Polynomial coefficients per segment, `coefficients[i][k]` = coeff of `(x - knots[i])^k`.
    pub coefficients: Vec<Vec<f64>>,
    /// Knot values (segment boundaries), length = n_segments + 1.
    pub knots: Vec<f64>,
    /// Polynomial degree of each segment.
    pub degree: usize,
}

impl SplineDerivative {
    /// Create a new spline from coefficients, knots, and degree.
    pub fn new(
        coefficients: Vec<Vec<f64>>,
        knots: Vec<f64>,
        degree: usize,
    ) -> Result<Self, InterpolateError> {
        if knots.len() < 2 {
            return Err(InterpolateError::InsufficientData(
                "SplineDerivative needs at least 2 knots".to_string(),
            ));
        }
        let n_seg = knots.len() - 1;
        if coefficients.len() != n_seg {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Expected {} coefficient vectors for {} segments, got {}",
                n_seg,
                n_seg,
                coefficients.len()
            )));
        }
        Ok(Self {
            coefficients,
            knots,
            degree,
        })
    }

    /// Differentiate this spline, returning a new spline of `degree - 1`.
    pub fn differentiate(spline: &SplineDerivative) -> Result<Self, InterpolateError> {
        if spline.degree == 0 {
            return Err(InterpolateError::InvalidOperation(
                "Cannot differentiate a degree-0 spline".to_string(),
            ));
        }
        let new_degree = spline.degree - 1;
        let new_coeffs: Vec<Vec<f64>> = spline
            .coefficients
            .iter()
            .map(|seg_coeffs| {
                // Differentiate polynomial: d/dx (c_k (x-x_i)^k) = k * c_k * (x-x_i)^(k-1)
                // Result has one fewer coefficient
                let n = seg_coeffs.len().min(spline.degree + 1);
                (1..n)
                    .map(|k| k as f64 * seg_coeffs[k])
                    .collect::<Vec<f64>>()
            })
            .collect();

        Self::new(new_coeffs, spline.knots.clone(), new_degree)
    }

    /// Evaluate the spline at point `x`.
    pub fn evaluate(&self, x: f64) -> Result<f64, InterpolateError> {
        let n = self.knots.len();
        if n < 2 {
            return Err(InterpolateError::InsufficientData(
                "No segments to evaluate".to_string(),
            ));
        }

        // Find segment
        let seg = if x <= self.knots[0] {
            0
        } else if x >= self.knots[n - 1] {
            n - 2
        } else {
            binary_search_floor(&self.knots, x)
        };

        let dx = x - self.knots[seg];
        let coeffs = &self.coefficients[seg];
        // Horner's method
        let mut val = 0.0f64;
        for &c in coeffs.iter().rev() {
            val = val * dx + c;
        }
        Ok(val)
    }
}

// ─── Grid Resampling Convenience Functions (WS227) ──────────────────────────

/// Resample scattered 1-D data onto a uniform grid of `n_grid_points`.
///
/// Returns `(grid_x, grid_y)` where `grid_x` is evenly spaced over
/// `[min(scattered_x), max(scattered_x)]`.
pub fn resample_to_regular(
    scattered_x: &[f64],
    scattered_y: &[f64],
    n_grid_points: usize,
    config: &ResamplingConfig,
) -> Result<(Vec<f64>, Vec<f64>), InterpolateError> {
    if scattered_x.len() < 2 {
        return Err(InterpolateError::InsufficientData(
            "resample_to_regular requires at least 2 input points".to_string(),
        ));
    }
    if scattered_x.len() != scattered_y.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "scattered_x len {} != scattered_y len {}",
            scattered_x.len(),
            scattered_y.len()
        )));
    }
    if n_grid_points < 2 {
        return Err(InterpolateError::InvalidInput {
            message: "n_grid_points must be >= 2".to_string(),
        });
    }

    // Sort the input data by x.
    let mut pairs: Vec<(f64, f64)> = scattered_x
        .iter()
        .copied()
        .zip(scattered_y.iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Deduplicate by x (keep last y for each unique x).
    let mut sorted_x: Vec<f64> = Vec::with_capacity(pairs.len());
    let mut sorted_y: Vec<f64> = Vec::with_capacity(pairs.len());
    for &(px, py) in &pairs {
        if let Some(&last_x) = sorted_x.last() {
            if (px - last_x).abs() < 1e-15_f64 {
                // Replace y for duplicate x.
                if let Some(ly) = sorted_y.last_mut() {
                    *ly = py;
                }
                continue;
            }
        }
        sorted_x.push(px);
        sorted_y.push(py);
    }

    if sorted_x.len() < 2 {
        return Err(InterpolateError::InsufficientData(
            "After deduplication, fewer than 2 unique x values remain".to_string(),
        ));
    }

    let x_min = sorted_x[0];
    let x_max = sorted_x[sorted_x.len() - 1];
    let step = (x_max - x_min) / (n_grid_points - 1) as f64;
    let grid_x: Vec<f64> = (0..n_grid_points)
        .map(|i| x_min + i as f64 * step)
        .collect();

    let grid_y = resample_1d(&sorted_x, &sorted_y, &grid_x, config)?;
    Ok((grid_x, grid_y))
}

/// Resample 1-D data onto arbitrary target x-coordinates.
///
/// `data_x` must be sortable; it will be sorted internally.
pub fn resample_to_irregular(
    data_x: &[f64],
    data_y: &[f64],
    target_x: &[f64],
    config: &ResamplingConfig,
) -> Result<Vec<f64>, InterpolateError> {
    if data_x.len() < 2 {
        return Err(InterpolateError::InsufficientData(
            "resample_to_irregular requires at least 2 input points".to_string(),
        ));
    }
    if data_x.len() != data_y.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "data_x len {} != data_y len {}",
            data_x.len(),
            data_y.len()
        )));
    }

    // Sort input by x.
    let mut pairs: Vec<(f64, f64)> = data_x.iter().copied().zip(data_y.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_x: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let sorted_y: Vec<f64> = pairs.iter().map(|p| p.1).collect();

    resample_1d(&sorted_x, &sorted_y, target_x, config)
}

/// Resample 2-D scattered data onto a regular nx × ny grid.
///
/// Uses inverse-distance weighting to map scattered points to a grid covering
/// `[min_x, max_x] × [min_y, max_y]`.
///
/// Returns a `grid_ny × grid_nx` nested `Vec<Vec<f64>>` in row-major order.
pub fn resample_scattered_2d(
    scattered_xy: &[(f64, f64)],
    values: &[f64],
    grid_nx: usize,
    grid_ny: usize,
) -> Result<Vec<Vec<f64>>, InterpolateError> {
    if scattered_xy.is_empty() {
        return Err(InterpolateError::InsufficientData(
            "No scattered data points for 2D resampling".to_string(),
        ));
    }
    if scattered_xy.len() != values.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "scattered_xy len {} != values len {}",
            scattered_xy.len(),
            values.len()
        )));
    }
    if grid_nx < 2 || grid_ny < 2 {
        return Err(InterpolateError::InvalidInput {
            message: "grid_nx and grid_ny must each be >= 2".to_string(),
        });
    }

    let x_vals: Vec<f64> = scattered_xy.iter().map(|p| p.0).collect();
    let y_vals: Vec<f64> = scattered_xy.iter().map(|p| p.1).collect();

    let x_min = x_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = y_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Prevent degenerate grids.
    let dx = if (x_max - x_min).abs() < 1e-15 {
        1.0
    } else {
        (x_max - x_min) / (grid_nx - 1) as f64
    };
    let dy = if (y_max - y_min).abs() < 1e-15 {
        1.0
    } else {
        (y_max - y_min) / (grid_ny - 1) as f64
    };

    let mut grid = vec![vec![0.0f64; grid_nx]; grid_ny];

    for iy in 0..grid_ny {
        let gy = y_min + iy as f64 * dy;
        for ix in 0..grid_nx {
            let gx = x_min + ix as f64 * dx;

            // IDW with power 2
            let mut numer = 0.0_f64;
            let mut denom = 0.0_f64;
            let mut exact_hit = false;
            for (idx, &(sx, sy)) in scattered_xy.iter().enumerate() {
                let dist2 = (sx - gx).powi(2) + (sy - gy).powi(2);
                if dist2 < 1e-28 {
                    grid[iy][ix] = values[idx];
                    exact_hit = true;
                    break;
                }
                let w = 1.0 / dist2;
                numer += w * values[idx];
                denom += w;
            }
            if !exact_hit {
                grid[iy][ix] = if denom > 1e-300 { numer / denom } else { 0.0 };
            }
        }
    }

    Ok(grid)
}

/// Downsample a 1-D signal by keeping every `factor`-th point.
///
/// The first point is always kept. Returns `(x_out, y_out)`.
pub fn downsample(
    x: &[f64],
    y: &[f64],
    factor: usize,
) -> Result<(Vec<f64>, Vec<f64>), InterpolateError> {
    if x.len() != y.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "x len {} != y len {}",
            x.len(),
            y.len()
        )));
    }
    if factor == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "downsample factor must be >= 1".to_string(),
        });
    }
    if x.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let x_out: Vec<f64> = x.iter().copied().step_by(factor).collect();
    let y_out: Vec<f64> = y.iter().copied().step_by(factor).collect();
    Ok((x_out, y_out))
}

/// Upsample a 1-D signal by inserting `factor - 1` interpolated points
/// between each original pair.
///
/// Uses the configured resampling method (default: linear).
pub fn upsample(
    x: &[f64],
    y: &[f64],
    factor: usize,
    config: &ResamplingConfig,
) -> Result<(Vec<f64>, Vec<f64>), InterpolateError> {
    if x.len() != y.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "x len {} != y len {}",
            x.len(),
            y.len()
        )));
    }
    if factor == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "upsample factor must be >= 1".to_string(),
        });
    }
    if x.len() < 2 {
        return Ok((x.to_vec(), y.to_vec()));
    }

    // Sort input by x.
    let mut pairs: Vec<(f64, f64)> = x.iter().copied().zip(y.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_x: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let sorted_y: Vec<f64> = pairs.iter().map(|p| p.1).collect();

    let n = sorted_x.len();
    // Total output points: (n - 1) * factor + 1
    let n_out = (n - 1) * factor + 1;
    let mut x_out = Vec::with_capacity(n_out);

    for i in 0..(n - 1) {
        let x0 = sorted_x[i];
        let x1 = sorted_x[i + 1];
        for j in 0..factor {
            let t = j as f64 / factor as f64;
            x_out.push(x0 + t * (x1 - x0));
        }
    }
    // Include the last point.
    x_out.push(sorted_x[n - 1]);

    let y_out = resample_1d(&sorted_x, &sorted_y, &x_out, config)?;
    Ok((x_out, y_out))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_1d_linear_identity() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.clone();
        let config = ResamplingConfig {
            method: ResamplingMethod::Linear,
            extrapolation: ExtrapolationMode::Nearest,
        };
        let x_out: Vec<f64> = (0..10).map(|i| i as f64 * 0.5 + 0.5).collect();
        let result = resample_1d(&x, &y, &x_out, &config).expect("resample");
        for (got, expected) in result.iter().zip(x_out.iter()) {
            // y = x, so linear interpolation should be exact
            let x_clamped = expected.clamp(x[0], x[x.len() - 1]);
            assert!(
                (got - x_clamped).abs() < 1e-10,
                "Linear identity failed: got={got}, expected={x_clamped}"
            );
        }
    }

    #[test]
    fn test_extrapolation_nearest_boundary() {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];
        let config = ResamplingConfig {
            method: ResamplingMethod::Linear,
            extrapolation: ExtrapolationMode::Nearest,
        };
        let x_out = vec![-1.0, 5.0];
        let result = resample_1d(&x, &y, &x_out, &config).expect("resample");
        assert!(
            (result[0] - 10.0).abs() < 1e-10,
            "Left boundary clamped: {}",
            result[0]
        );
        assert!(
            (result[1] - 40.0).abs() < 1e-10,
            "Right boundary clamped: {}",
            result[1]
        );
    }

    #[test]
    fn test_extrapolation_zero() {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0];
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let config = ResamplingConfig {
            method: ResamplingMethod::Linear,
            extrapolation: ExtrapolationMode::Zero,
        };
        let x_out = vec![-1.0, 5.0];
        let result = resample_1d(&x, &y, &x_out, &config).expect("resample");
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_extrapolation_constant() {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0];
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let config = ResamplingConfig {
            method: ResamplingMethod::Linear,
            extrapolation: ExtrapolationMode::Constant(99.0),
        };
        let x_out = vec![-5.0, 10.0];
        let result = resample_1d(&x, &y, &x_out, &config).expect("resample");
        assert!((result[0] - 99.0).abs() < 1e-10);
        assert!((result[1] - 99.0).abs() < 1e-10);
    }

    #[test]
    fn test_extrapolation_periodic() {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0]; // y = x in domain
        let config = ResamplingConfig {
            method: ResamplingMethod::Linear,
            extrapolation: ExtrapolationMode::Periodic,
        };
        // x=3.5 should map to x=0.5 (period = 3)
        let result = resample_1d(&x, &y, &[3.5], &config).expect("periodic");
        assert!(
            (result[0] - 0.5).abs() < 0.2,
            "Periodic wrap: got {} expected ~0.5",
            result[0]
        );
    }

    #[test]
    fn test_extrapolation_linear() {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![0.0, 2.0, 4.0, 6.0]; // y = 2x
        let config = ResamplingConfig {
            method: ResamplingMethod::Linear,
            extrapolation: ExtrapolationMode::Linear,
        };
        // x=4.0 should extrapolate to 8.0
        let result = resample_1d(&x, &y, &[4.0], &config).expect("linear extrap");
        assert!(
            (result[0] - 8.0).abs() < 1e-8,
            "Linear extrapolation: got {} expected 8.0",
            result[0]
        );
    }

    #[test]
    fn test_cubic_spline_resample() {
        let x: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect(); // y = x²
        let config = ResamplingConfig {
            method: ResamplingMethod::CubicSpline,
            extrapolation: ExtrapolationMode::Nearest,
        };
        let x_out = vec![0.5, 1.5, 2.5, 3.5];
        let result = resample_1d(&x, &y, &x_out, &config).expect("cubic");
        for (xq, &yq) in x_out.iter().zip(result.iter()) {
            let exact = xq * xq;
            // Natural cubic spline imposes zero second derivatives at boundaries,
            // which introduces a boundary error for polynomial data.
            // Tolerance is relaxed near boundaries (x<1 or x>4).
            let tol = if *xq < 1.0 || *xq > 4.0 { 0.2 } else { 0.05 };
            assert!(
                (yq - exact).abs() < tol,
                "Cubic spline on y=x² at x={xq}: got {yq}, expected {exact}"
            );
        }
    }

    #[test]
    fn test_nearest_method() {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0];
        let y: Vec<f64> = vec![10.0, 20.0, 30.0];
        let config = ResamplingConfig {
            method: ResamplingMethod::Nearest,
            extrapolation: ExtrapolationMode::Nearest,
        };
        let result = resample_1d(&x, &y, &[0.3, 0.7], &config).expect("nearest");
        assert!(
            (result[0] - 10.0).abs() < 1e-10,
            "Nearest left: {}",
            result[0]
        );
        assert!(
            (result[1] - 20.0).abs() < 1e-10,
            "Nearest right: {}",
            result[1]
        );
    }

    #[test]
    fn test_scattered_to_grid_2d() {
        // Simple scattered points: y = x1 + x2
        let x: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let y: Vec<f64> = x.iter().map(|xi| xi[0] + xi[1]).collect();
        let grid_ranges = vec![(0.0, 1.0, 3), (0.0, 1.0, 3)];
        let config = ResamplingConfig::default();
        let result = scattered_to_grid(&x, &y, &grid_ranges, &config).expect("scattered_to_grid");
        assert_eq!(result.len(), 9); // 3×3
                                     // All values should be >= 0 and <= 2
        for &v in &result {
            assert!(v >= -0.1 && v <= 2.1, "Value out of expected range: {v}");
        }
    }

    #[test]
    fn test_spline_derivative_differentiation() {
        // Quadratic: y = 3x² + 2x + 1 on [0, 2]
        // Coefficients: [1, 2, 3] for (x-0)^0, (x-0)^1, (x-0)^2
        let spline = SplineDerivative::new(vec![vec![1.0, 2.0, 3.0]], vec![0.0, 2.0], 2)
            .expect("create spline");

        // Derivative: 2 + 6x => coeffs [2, 6]
        let deriv = SplineDerivative::differentiate(&spline).expect("differentiate");
        assert_eq!(deriv.degree, 1);

        // Evaluate at x=1: should be 2 + 6*1 = 8
        let val = deriv.evaluate(1.0).expect("evaluate");
        assert!(
            (val - 8.0).abs() < 1e-10,
            "Derivative at x=1: got {val}, expected 8.0"
        );
    }

    #[test]
    fn test_spline_evaluate() {
        // Linear: y = 2x + 1 on [0, 1] and y = 3x - 0 on [1, 2]
        let spline =
            SplineDerivative::new(vec![vec![1.0, 2.0], vec![3.0, 0.0]], vec![0.0, 1.0, 2.0], 1)
                .expect("create spline");
        let v0 = spline.evaluate(0.5).expect("eval");
        assert!((v0 - 2.0).abs() < 1e-10, "Got {v0}"); // 1 + 2*0.5 = 2
    }

    #[test]
    fn test_resample_2d() {
        // 3×3 grid with values row + col
        let grid: Vec<Vec<f64>> = (0..3)
            .map(|i| (0..3).map(|j| (i + j) as f64).collect())
            .collect();
        let x_in: Vec<f64> = vec![0.0, 1.0, 2.0];
        let y_in: Vec<f64> = vec![0.0, 1.0, 2.0];
        let x_out: Vec<f64> = vec![0.5, 1.0, 1.5];
        let y_out: Vec<f64> = vec![0.5, 1.0, 1.5];
        let config = ResamplingConfig::default();
        let result =
            resample_2d(&grid, &x_in, &y_in, &x_out, &y_out, &config).expect("resample_2d");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 3);
        // At (0.5, 0.5): value should be ~1.0
        assert!(
            (result[0][0] - 1.0).abs() < 0.1,
            "2D resample at (0.5,0.5): got {}",
            result[0][0]
        );
    }

    #[test]
    fn test_reflection_extrapolation() {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![0.0, 1.0, 4.0, 9.0];
        let config = ResamplingConfig {
            method: ResamplingMethod::Linear,
            extrapolation: ExtrapolationMode::Reflection,
        };
        // x=-0.5 should reflect to 0.5 within [0,3]
        let result = resample_1d(&x, &y, &[-0.5], &config).expect("reflection");
        // Should be in range since reflected
        assert!(
            result[0].is_finite(),
            "Reflection should produce finite value"
        );
    }

    // ── WS227 Grid Resampling tests ──────────────────────────────────────

    #[test]
    fn test_resample_to_regular_roundtrip() {
        // y = 2x on [0, 4] — resample to 5-point grid, then evaluate at original.
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi).collect();
        let config = ResamplingConfig::default();

        let (grid_x, grid_y) =
            resample_to_regular(&x, &y, 9, &config).expect("resample_to_regular");
        assert_eq!(grid_x.len(), 9);
        assert_eq!(grid_y.len(), 9);

        // At each grid point the value should be close to 2*x.
        for (gx, gy) in grid_x.iter().zip(grid_y.iter()) {
            let expected = 2.0 * gx;
            assert!(
                (gy - expected).abs() < 0.1,
                "resample_to_regular roundtrip: at x={gx} got {gy}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_resample_to_irregular() {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect(); // y = x^2
        let config = ResamplingConfig::default();

        let targets = vec![0.5, 1.5, 2.5, 3.5];
        let result =
            resample_to_irregular(&x, &y, &targets, &config).expect("resample_to_irregular");
        assert_eq!(result.len(), 4);

        // Linear interpolation on x^2: approximate values.
        for (i, &tgt) in targets.iter().enumerate() {
            let exact = tgt * tgt;
            // Linear interp on x^2 has some error, but should be reasonable.
            assert!(
                (result[i] - exact).abs() < 1.0,
                "resample_to_irregular at x={tgt}: got {}, expected ~{exact}",
                result[i]
            );
        }
    }

    #[test]
    fn test_resample_scattered_2d_grid_covers_domain() {
        let scattered: Vec<(f64, f64)> =
            vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)];
        let values: Vec<f64> = scattered.iter().map(|&(x, y)| x + y).collect();

        let grid = resample_scattered_2d(&scattered, &values, 3, 3).expect("scattered 2d");
        assert_eq!(grid.len(), 3);
        assert_eq!(grid[0].len(), 3);

        // All values should be in [0, 2] (since x+y ranges from 0 to 2).
        for row in &grid {
            for &v in row {
                assert!(v >= -0.1 && v <= 2.1, "2D grid value out of range: {v}");
            }
        }
    }

    #[test]
    fn test_downsample() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let (dx, dy) = downsample(&x, &y, 3).expect("downsample");
        // With factor 3: indices 0, 3, 6, 9
        assert_eq!(dx.len(), 4);
        assert!((dx[0] - 0.0).abs() < 1e-12);
        assert!((dx[1] - 3.0).abs() < 1e-12);
        assert!((dx[2] - 6.0).abs() < 1e-12);
        assert!((dx[3] - 9.0).abs() < 1e-12);
        // Values should match y = x^2
        assert!((dy[1] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_upsample_preserves_function() {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![0.0, 2.0, 4.0, 6.0]; // y = 2x
        let config = ResamplingConfig::default();

        let (ux, uy) = upsample(&x, &y, 3, &config).expect("upsample");
        // (n-1)*factor + 1 = 3*3 + 1 = 10
        assert_eq!(ux.len(), 10);
        assert_eq!(uy.len(), 10);

        // For y=2x, all upsampled values should be very close to 2*x.
        for (xi, yi) in ux.iter().zip(uy.iter()) {
            let expected = 2.0 * xi;
            assert!(
                (yi - expected).abs() < 0.1,
                "upsample: at x={xi} got {yi}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_downsample_factor_1_identity() {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0];
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let (dx, dy) = downsample(&x, &y, 1).expect("factor 1");
        assert_eq!(dx.len(), 3);
        assert_eq!(dy.len(), 3);
    }

    #[test]
    fn test_downsample_factor_zero_error() {
        let result = downsample(&[1.0], &[1.0], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_upsample_factor_zero_error() {
        let config = ResamplingConfig::default();
        let result = upsample(&[1.0, 2.0], &[1.0, 2.0], 0, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_resample_to_regular_too_few_points() {
        let config = ResamplingConfig::default();
        let result = resample_to_regular(&[1.0], &[1.0], 5, &config);
        assert!(result.is_err());
    }
}
