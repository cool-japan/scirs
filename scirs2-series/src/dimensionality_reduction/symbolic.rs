//! Symbolic approximation methods for time series.
//!
//! This module provides symbolic approximation types, configurations, and
//! computation functions including SAX, APCA, PLA, and Persist encodings.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayStatCompat};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};
use statrs::statistics::Statistics;

/// Configuration for symbolic approximation
#[derive(Debug, Clone)]
pub struct SymbolicApproximationConfig {
    /// Approximation method
    pub method: SymbolicMethod,
    /// Number of symbols in the alphabet
    pub alphabet_size: usize,
    /// Window size for segmentation
    pub window_size: usize,
    /// Number of segments for PAA
    pub nsegments: usize,
    /// Whether to normalize data before approximation
    pub normalize_data: bool,
    /// Breakpoints for SAX (None = automatic)
    pub breakpoints: Option<Array1<f64>>,
    /// Distance metric for symbolic sequences
    pub distance_metric: SymbolicDistance,
}

impl Default for SymbolicApproximationConfig {
    fn default() -> Self {
        Self {
            method: SymbolicMethod::SAX,
            alphabet_size: 8,
            window_size: 16,
            nsegments: 10,
            normalize_data: true,
            breakpoints: None,
            distance_metric: SymbolicDistance::MINDIST,
        }
    }
}

/// Symbolic approximation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SymbolicMethod {
    /// Symbolic Aggregate approXimation (SAX)
    SAX,
    /// Adaptive Piecewise Constant Approximation (APCA)
    APCA,
    /// Piecewise Linear Approximation (PLA)
    PLA,
    /// Persist (1-dimensional representation)
    Persist,
}

/// Distance metrics for symbolic sequences
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SymbolicDistance {
    /// MINDIST lower bound
    MINDIST,
    /// Hamming distance
    Hamming,
    /// Edit distance
    Edit,
}

/// Result of symbolic approximation
#[derive(Debug, Clone)]
pub struct SymbolicApproximationResult {
    /// Symbolic representation
    pub symbolic_sequence: Vec<char>,
    /// Breakpoints used for discretization
    pub breakpoints: Array1<f64>,
    /// Piecewise Aggregate Approximation values
    pub _paavalues: Array1<f64>,
    /// Reconstruction error
    pub reconstruction_error: f64,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Distance matrix between segments (if applicable)
    pub distance_matrix: Option<Array2<f64>>,
}

/// Apply symbolic approximation to time series
///
/// # Arguments
///
/// * `_timeseries` - Input time series
/// * `config` - Symbolic approximation configuration
///
/// # Returns
///
/// Symbolic approximation result including symbolic sequence and reconstruction information
#[allow(dead_code)]
pub fn apply_symbolic_approximation(
    _timeseries: &Array1<f64>,
    config: &SymbolicApproximationConfig,
) -> Result<SymbolicApproximationResult> {
    if _timeseries.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Time _series cannot be empty".to_string(),
        ));
    }

    match config.method {
        SymbolicMethod::SAX => apply_sax(_timeseries, config),
        SymbolicMethod::APCA => apply_apca(_timeseries, config),
        SymbolicMethod::PLA => apply_pla(_timeseries, config),
        SymbolicMethod::Persist => apply_persist(_timeseries, config),
    }
}

// ---------------------------------------------------------------------------
// Symbolic approximation helper functions
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn apply_sax(
    _timeseries: &Array1<f64>,
    config: &SymbolicApproximationConfig,
) -> Result<SymbolicApproximationResult> {
    // Step 1: Normalize data if requested
    let normalized_data = if config.normalize_data {
        normalize_timeseries(_timeseries)?
    } else {
        _timeseries.clone()
    };

    // Step 2: Piecewise Aggregate Approximation (PAA)
    let _paavalues = compute_paa(&normalized_data, config.nsegments)?;

    // Step 3: Determine breakpoints
    let breakpoints = config
        .breakpoints
        .clone()
        .unwrap_or_else(|| compute_gaussian_breakpoints(config.alphabet_size));

    // Step 4: Convert PAA to symbols
    let symbolic_sequence = paa_to_symbols(&_paavalues, &breakpoints)?;

    // Step 5: Compute reconstruction error (placeholder for now)
    let reconstruction_error = 0.0; // Would compute actual reconstruction error in full implementation

    // Step 6: Compute compression ratio
    let compression_ratio = _timeseries.len() as f64 / symbolic_sequence.len() as f64;

    Ok(SymbolicApproximationResult {
        symbolic_sequence,
        breakpoints,
        _paavalues,
        reconstruction_error,
        compression_ratio,
        distance_matrix: None,
    })
}

#[allow(dead_code)]
fn apply_apca(
    timeseries: &Array1<f64>,
    config: &SymbolicApproximationConfig,
) -> Result<SymbolicApproximationResult> {
    // Adaptive Piecewise Constant Approximation (Keogh et al., 2001).
    //
    // APCA adapts segment boundaries by minimising per-segment mean squared error
    // with a fixed number of segments (`config.nsegments`).  Unlike PAA (which uses
    // equal-width segments), APCA greedily merges adjacent flat segments to minimise
    // total reconstruction error.
    //
    // Algorithm (simplified greedy):
    //  1. Start with n segments of size 1.
    //  2. Repeatedly merge the pair of adjacent segments whose union has the lowest
    //     increase in squared error until the desired number of segments is reached.
    //  3. Represent each segment by its mean.

    let n = timeseries.len();
    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must not be empty".to_string(),
        ));
    }

    let nseg = config.nsegments.min(n);

    // Normalise if requested
    let data = if config.normalize_data {
        normalize_timeseries(timeseries)?
    } else {
        timeseries.clone()
    };

    // Initial segments: each element is its own segment [start, end)
    // Represent as (start_idx, end_idx) inclusive
    let mut segments: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();

    // Merge cost: increase in SSE when merging two adjacent segments.
    // SSE of a constant approximation to data[a..=b] = sum (x - mean)^2
    //   = sum x^2 - (sum x)^2 / len
    let segment_sse = |a: usize, b: usize| -> f64 {
        let len = (b - a + 1) as f64;
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for k in a..=b {
            sum += data[k];
            sum_sq += data[k] * data[k];
        }
        sum_sq - sum * sum / len
    };

    // Merge until target segment count
    while segments.len() > nseg {
        let mut best_cost = f64::INFINITY;
        let mut best_idx = 0;

        for i in 0..segments.len() - 1 {
            let a = segments[i].0;
            let b = segments[i + 1].1;
            let cost = segment_sse(a, b);
            if cost < best_cost {
                best_cost = cost;
                best_idx = i;
            }
        }

        // Merge segments[best_idx] and segments[best_idx+1]
        let merged_end = segments[best_idx + 1].1;
        segments[best_idx].1 = merged_end;
        segments.remove(best_idx + 1);
    }

    // Build PAA values: mean of each segment
    let mut paa_values = Array1::zeros(segments.len());
    for (j, &(a, b)) in segments.iter().enumerate() {
        let len = (b - a + 1) as f64;
        let mean: f64 = data.slice(scirs2_core::ndarray::s![a..=b]).sum() / len;
        paa_values[j] = mean;
    }

    // Assign symbols using breakpoints
    let breakpoints = config
        .breakpoints
        .clone()
        .unwrap_or_else(|| compute_gaussian_breakpoints(config.alphabet_size));

    let symbolic_sequence = paa_to_symbols(&paa_values, &breakpoints)?;

    // Reconstruction error: average per-point squared error
    let mut total_sse = 0.0_f64;
    for (j, &(a, b)) in segments.iter().enumerate() {
        let mean = paa_values[j];
        for k in a..=b {
            let e = data[k] - mean;
            total_sse += e * e;
        }
    }
    let reconstruction_error = (total_sse / n as f64).sqrt();
    let compression_ratio = n as f64 / symbolic_sequence.len() as f64;

    Ok(SymbolicApproximationResult {
        symbolic_sequence,
        breakpoints,
        _paavalues: paa_values,
        reconstruction_error,
        compression_ratio,
        distance_matrix: None,
    })
}

#[allow(dead_code)]
fn apply_pla(
    timeseries: &Array1<f64>,
    config: &SymbolicApproximationConfig,
) -> Result<SymbolicApproximationResult> {
    // Piecewise Linear Approximation — sliding-window variant.
    //
    // Partition the time series into `nsegments` equal-width segments and fit
    // a least-squares linear regression line to each segment.  The representative
    // value for symbol assignment is the segment mean (midpoint value of the
    // fitted line), following the PAA-compatible encoding used by most PLA
    // implementations in symbolic sequence literature.
    //
    // Each segment covers indices [start, end) (exclusive end).  For boundary
    // segments with fewer points than the ideal width, the available points are used.

    let n = timeseries.len();
    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must not be empty".to_string(),
        ));
    }

    let nseg = config.nsegments.min(n);

    let data = if config.normalize_data {
        normalize_timeseries(timeseries)?
    } else {
        timeseries.clone()
    };

    let seg_size = n as f64 / nseg as f64;
    let mut paa_values = Array1::zeros(nseg);
    let mut total_sse = 0.0_f64;

    for j in 0..nseg {
        let start = (j as f64 * seg_size).round() as usize;
        let end = ((j + 1) as f64 * seg_size).round() as usize;
        let end = end.min(n);
        let seg_len = end - start;

        if seg_len == 0 {
            continue;
        }

        if seg_len == 1 {
            paa_values[j] = data[start];
            continue;
        }

        // Least-squares linear fit: y = a*t + b where t in [0, seg_len-1]
        let len_f = seg_len as f64;
        let t_mean = (seg_len - 1) as f64 / 2.0;
        let mut y_mean = 0.0_f64;
        let mut sxx = 0.0_f64;
        let mut sxy = 0.0_f64;

        for (k, idx) in (start..end).enumerate() {
            let t = k as f64;
            let y = data[idx];
            y_mean += y;
            sxx += (t - t_mean) * (t - t_mean);
            sxy += (t - t_mean) * y;
        }
        y_mean /= len_f;

        let slope = if sxx.abs() > 1e-12 { sxy / sxx } else { 0.0 };
        let intercept = y_mean - slope * t_mean;

        // Segment representative = mean of linear fit (which equals y_mean)
        paa_values[j] = y_mean;

        // Accumulate SSE vs linear fit
        for (k, idx) in (start..end).enumerate() {
            let fitted = slope * k as f64 + intercept;
            let e = data[idx] - fitted;
            total_sse += e * e;
        }
    }

    let breakpoints = config
        .breakpoints
        .clone()
        .unwrap_or_else(|| compute_gaussian_breakpoints(config.alphabet_size));

    let symbolic_sequence = paa_to_symbols(&paa_values, &breakpoints)?;

    let reconstruction_error = (total_sse / n as f64).sqrt();
    let compression_ratio = n as f64 / symbolic_sequence.len() as f64;

    Ok(SymbolicApproximationResult {
        symbolic_sequence,
        breakpoints,
        _paavalues: paa_values,
        reconstruction_error,
        compression_ratio,
        distance_matrix: None,
    })
}

#[allow(dead_code)]
fn apply_persist(
    timeseries: &Array1<f64>,
    config: &SymbolicApproximationConfig,
) -> Result<SymbolicApproximationResult> {
    // Persist (1-dimensional) symbolic representation.
    //
    // The Persist method (Fink & Pratt, "Indexing in time series databases using
    // the persist feature", 2003) encodes each time-point not by its absolute value
    // but by the *direction of change* from the previous point:
    //   - 'u'  (up)     if x[t] > x[t-1]
    //   - 'd'  (down)   if x[t] < x[t-1]
    //   - 's'  (steady) if x[t] == x[t-1]
    //
    // When `nsegments < n`, we first segment via PAA (equal-width means), then
    // apply directional encoding to the PAA coefficients — giving a length-nsegments
    // symbolic sequence that is compatible with the SAX infrastructure.
    //
    // For the PAA values we still use a Gaussian-breakpoint alphabet so that
    // `reconstruction_error` and `compression_ratio` are well-defined.

    let n = timeseries.len();
    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must not be empty".to_string(),
        ));
    }

    let data = if config.normalize_data {
        normalize_timeseries(timeseries)?
    } else {
        timeseries.clone()
    };

    let nseg = config.nsegments.min(n);
    let paa_values = compute_paa(&data, nseg)?;

    // Directional encoding of PAA coefficients
    // 'u', 'd', 's' → but the standard alphabet is letters; map to 'a' (down), 'b' (steady),
    // 'c' (up) to stay compatible with the SAX char alphabet ordering.
    let mut symbolic_sequence = Vec::with_capacity(nseg);
    for j in 0..nseg {
        let symbol = if j == 0 {
            // First segment: no predecessor — use 'b' (steady)
            'b'
        } else {
            match paa_values[j].partial_cmp(&paa_values[j - 1]) {
                Some(std::cmp::Ordering::Greater) => 'c', // up
                Some(std::cmp::Ordering::Less) => 'a',    // down
                _ => 'b',                                 // steady / NaN
            }
        };
        symbolic_sequence.push(symbol);
    }

    // Breakpoints (for compatibility with SymbolicApproximationResult)
    let breakpoints = config
        .breakpoints
        .clone()
        .unwrap_or_else(|| compute_gaussian_breakpoints(config.alphabet_size));

    // Reconstruction via paa_values
    let seg_size = n as f64 / nseg as f64;
    let mut total_sse = 0.0_f64;
    for j in 0..nseg {
        let start = (j as f64 * seg_size).round() as usize;
        let end = (((j + 1) as f64 * seg_size).round() as usize).min(n);
        let mean = paa_values[j];
        for idx in start..end {
            let e = data[idx] - mean;
            total_sse += e * e;
        }
    }
    let reconstruction_error = (total_sse / n as f64).sqrt();
    let compression_ratio = n as f64 / symbolic_sequence.len() as f64;

    Ok(SymbolicApproximationResult {
        symbolic_sequence,
        breakpoints,
        _paavalues: paa_values,
        reconstruction_error,
        compression_ratio,
        distance_matrix: None,
    })
}

#[allow(dead_code)]
fn normalize_timeseries(_timeseries: &Array1<f64>) -> Result<Array1<f64>> {
    let mean = _timeseries.mean_or(0.0);
    let std = _timeseries.std(0.0);

    if std == 0.0 {
        return Ok(Array1::zeros(_timeseries.len()));
    }

    let normalized = _timeseries.mapv(|x| (x - mean) / std);
    Ok(normalized)
}

#[allow(dead_code)]
fn compute_paa(_timeseries: &Array1<f64>, nsegments: usize) -> Result<Array1<f64>> {
    let n = _timeseries.len();
    let segment_size = n as f64 / nsegments as f64;

    let mut _paavalues = Array1::zeros(nsegments);

    for i in 0..nsegments {
        let start = (i as f64 * segment_size) as usize;
        let end = ((i + 1) as f64 * segment_size) as usize;
        let end = std::cmp::min(end, n);

        if start < end {
            let segment_mean = _timeseries.slice(s![start..end]).mean();
            _paavalues[i] = segment_mean;
        }
    }

    Ok(_paavalues)
}

#[allow(dead_code)]
fn compute_gaussian_breakpoints(_alphabetsize: usize) -> Array1<f64> {
    // Compute breakpoints based on Gaussian distribution
    // This is a simplified version - would use proper quantile function

    let mut breakpoints = Array1::zeros(_alphabetsize - 1);

    for i in 0.._alphabetsize - 1 {
        let quantile = (i + 1) as f64 / _alphabetsize as f64;
        // Simplified inverse normal - in practice would use proper implementation
        let breakpoint = if quantile < 0.5 {
            -(1.0 - 2.0 * quantile).sqrt()
        } else {
            (2.0 * quantile - 1.0).sqrt()
        };
        breakpoints[i] = breakpoint;
    }

    breakpoints
}

#[allow(dead_code)]
fn paa_to_symbols(_paavalues: &Array1<f64>, breakpoints: &Array1<f64>) -> Result<Vec<char>> {
    let alphabet_chars: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
    let mut symbols = Vec::new();

    for &value in _paavalues.iter() {
        let mut symbol_idx = 0;

        for &breakpoint in breakpoints.iter() {
            if value > breakpoint {
                symbol_idx += 1;
            } else {
                break;
            }
        }

        let symbol = alphabet_chars.get(symbol_idx).copied().unwrap_or('z');
        symbols.push(symbol);
    }

    Ok(symbols)
}

/// Reconstruct a time series from its SAX symbolic representation.
///
/// Maps each symbol back to the midpoint of its breakpoint interval and
/// interpolates to recover the original length.
#[allow(dead_code)]
pub fn reconstruct_from_sax(
    symbolic_sequence: &[char],
    breakpoints: &Array1<f64>,
    original_length: usize,
    nsegments: usize,
) -> Result<Array1<f64>> {
    // SAX reconstruction: map each symbol back to the mid-point of its
    // corresponding breakpoint interval, then expand PAA segments to original length.
    //
    // Given breakpoints b_0, b_1, ..., b_{k-1} (with b_0 = -∞, b_k = +∞),
    // symbol 'a' maps to interval (-∞, b_0], 'b' to (b_0, b_1], etc.
    // The representative value is the midpoint of each interval.
    // The two outermost intervals use half-width extrapolation.

    if nsegments == 0 || original_length == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "nsegments and original_length must be positive".to_string(),
        ));
    }

    let bp = breakpoints;
    let nb = bp.len(); // number of interior breakpoints

    // Build midpoints for each symbol level (0..=nb)
    // Level 0: below bp[0]  → midpoint = bp[0] - (bp[1]-bp[0])/2  (if nb>1) else bp[0]-1
    // Level i (0<i<nb): (bp[i-1]+bp[i])/2
    // Level nb: above bp[nb-1] → midpoint = bp[nb-1] + (bp[nb-1]-bp[nb-2])/2
    let mut midpoints: Vec<f64> = Vec::with_capacity(nb + 1);

    if nb == 0 {
        midpoints.push(0.0);
    } else {
        // Below lowest breakpoint
        let lower_ext = if nb > 1 {
            bp[0] - (bp[1] - bp[0]) / 2.0
        } else {
            bp[0] - 1.0
        };
        midpoints.push(lower_ext);

        // Interior intervals
        for i in 0..nb.saturating_sub(1) {
            midpoints.push((bp[i] + bp[i + 1]) / 2.0);
        }

        // Above highest breakpoint
        let upper_ext = if nb > 1 {
            bp[nb - 1] + (bp[nb - 1] - bp[nb - 2]) / 2.0
        } else {
            bp[0] + 1.0
        };
        midpoints.push(upper_ext);
    }

    // Map each symbol character to a numeric value
    let alphabet_start = 'a' as u8;
    let mut paa_values = Array1::zeros(symbolic_sequence.len());
    for (j, &sym) in symbolic_sequence.iter().enumerate() {
        let level = (sym as u8).saturating_sub(alphabet_start) as usize;
        let level = level.min(midpoints.len() - 1);
        paa_values[j] = midpoints[level];
    }

    // Expand PAA back to original length
    let nseg = nsegments.min(symbolic_sequence.len());
    let seg_size = original_length as f64 / nseg as f64;
    let mut reconstructed = Array1::zeros(original_length);

    for j in 0..nseg {
        let start = (j as f64 * seg_size).round() as usize;
        let end = (((j + 1) as f64 * seg_size).round() as usize).min(original_length);
        let val = if j < paa_values.len() {
            paa_values[j]
        } else {
            0.0
        };
        for idx in start..end {
            reconstructed[idx] = val;
        }
    }

    Ok(reconstructed)
}

/// Compute the root mean squared reconstruction error between original and
/// reconstructed time series.
#[allow(dead_code)]
pub fn compute_reconstruction_error(original: &Array1<f64>, reconstructed: &Array1<f64>) -> f64 {
    // Root mean squared error between original and reconstructed time series.
    let n = original.len().min(reconstructed.len());
    if n == 0 {
        return 0.0;
    }
    let mut sse = 0.0_f64;
    for i in 0..n {
        let e = original[i] - reconstructed[i];
        sse += e * e;
    }
    (sse / n as f64).sqrt()
}
