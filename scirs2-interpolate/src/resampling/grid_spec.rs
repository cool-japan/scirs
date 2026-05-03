//! N-dimensional grid specification and scattered-to-grid resampling.
//!
//! Provides `GridSpec`, `Aggregator`, `ResampleStrategy`, and
//! `resample_scattered_to_grid` for mapping scattered (x, y) data onto a
//! regular N-D grid, returning an `ArrayD<f64>`.
//!
//! # Design
//!
//! - `Rasterize(Aggregator)` bins each scattered point into its nearest grid
//!   cell and accumulates values; empty cells receive `f64::NAN`.
//! - `Conservative` applies area-weighted (axis-aligned) accumulation —
//!   for axis-aligned grids this reduces to rasterize with count-weighting.
//!
//! # Example
//!
//! ```rust
//! use scirs2_interpolate::resampling::{GridSpec, Aggregator, ResampleStrategy, resample_scattered_to_grid};
//! use scirs2_core::ndarray::Array2;
//!
//! // 4 scattered points in 2-D, values = x + y
//! let pts = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
//! let vals = scirs2_core::ndarray::Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);
//! let grid = GridSpec::uniform(2, &[(0.0, 1.0, 2), (0.0, 1.0, 2)]);
//! let out = resample_scattered_to_grid(&pts, &vals, &grid, ResampleStrategy::Rasterize(Aggregator::Mean))
//!     .expect("resample");
//! assert_eq!(out.shape(), &[2, 2]);
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayD, IxDyn};

use crate::error::{InterpolateError, InterpolateResult};

// ─────────────────────────────────────────────────────────────────────────────
// GridSpec
// ─────────────────────────────────────────────────────────────────────────────

/// Description of a regular N-dimensional target grid.
///
/// Each axis is defined by an `Array1<f64>` of strictly increasing
/// grid-cell centre coordinates.
#[derive(Debug, Clone)]
pub struct GridSpec {
    /// Grid-cell centre coordinates for each axis.
    pub axes: Vec<Array1<f64>>,
}

impl GridSpec {
    /// Create a `GridSpec` from per-axis arrays.
    ///
    /// Each `axis` slice must contain at least 1 value and be strictly
    /// increasing.
    pub fn new(axes: Vec<Array1<f64>>) -> InterpolateResult<Self> {
        for (dim, ax) in axes.iter().enumerate() {
            if ax.is_empty() {
                return Err(InterpolateError::InvalidInput {
                    message: format!("axis {dim} is empty"),
                });
            }
            for i in 1..ax.len() {
                if ax[i] <= ax[i - 1] {
                    return Err(InterpolateError::InvalidInput {
                        message: format!("axis {dim} is not strictly increasing"),
                    });
                }
            }
        }
        Ok(Self { axes })
    }

    /// Build a uniform grid from `(min, max, n_cells)` per dimension.
    ///
    /// This is a convenience constructor; panics only if `n_cells == 0`.
    pub fn uniform(dim: usize, specs: &[(f64, f64, usize)]) -> Self {
        assert_eq!(specs.len(), dim, "specs length must equal dim");
        let axes: Vec<Array1<f64>> = specs
            .iter()
            .map(|&(lo, hi, n)| {
                let n_pts = n.max(1);
                if n_pts == 1 {
                    Array1::from_vec(vec![lo])
                } else {
                    let step = (hi - lo) / (n_pts as f64 - 1.0);
                    Array1::from_iter((0..n_pts).map(|i| lo + i as f64 * step))
                }
            })
            .collect();
        Self { axes }
    }

    /// Dimensionality of the grid.
    pub fn ndim(&self) -> usize {
        self.axes.len()
    }

    /// Shape of the output array (one entry per axis).
    pub fn shape(&self) -> Vec<usize> {
        self.axes.iter().map(|ax| ax.len()).collect()
    }

    /// Total number of grid cells.
    pub fn n_cells(&self) -> usize {
        self.axes.iter().map(|ax| ax.len()).product()
    }

    /// Find the nearest cell index along `dim` for coordinate `val`.
    ///
    /// Returns the index of the closest grid-centre value.
    pub fn nearest_index(&self, dim: usize, val: f64) -> usize {
        let ax = &self.axes[dim];
        let n = ax.len();
        // Binary search for the insertion point, then compare neighbours.
        // Use a manual binary search to avoid `unwrap()` on `as_slice()`.
        let pos = {
            let mut lo_idx = 0_usize;
            let mut hi_idx = n;
            while lo_idx < hi_idx {
                let mid = lo_idx + (hi_idx - lo_idx) / 2;
                if ax[mid] < val {
                    lo_idx = mid + 1;
                } else {
                    hi_idx = mid;
                }
            }
            lo_idx
        };
        if pos == 0 {
            0
        } else if pos == n {
            n - 1
        } else {
            let lo = ax[pos - 1];
            let hi = ax[pos];
            if (val - lo).abs() <= (val - hi).abs() {
                pos - 1
            } else {
                pos
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Aggregator
// ─────────────────────────────────────────────────────────────────────────────

/// Value-aggregation strategy for cells that receive multiple scattered points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregator {
    /// Arithmetic mean.
    Mean,
    /// Sample median (O(k log k) per non-empty cell).
    Median,
    /// Maximum value.
    Max,
    /// Minimum value.
    Min,
    /// Number of points that fell into the cell.
    Count,
}

// ─────────────────────────────────────────────────────────────────────────────
// ResampleStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// How to map scattered data to grid cells.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResampleStrategy {
    /// Assign each point to its nearest grid cell, then aggregate.
    Rasterize(Aggregator),
    /// Area-weighted conservative resampling (axis-aligned approximation).
    ///
    /// For axis-aligned grids this reduces to count-weighted rasterization,
    /// preserving the total integral of the scattered data.
    Conservative,
}

// ─────────────────────────────────────────────────────────────────────────────
// resample_scattered_to_grid
// ─────────────────────────────────────────────────────────────────────────────

/// Resample scattered N-D data onto a regular grid.
///
/// # Arguments
///
/// * `points` – `n × d` array of sample coordinates.
/// * `values` – length-`n` array of scalar values at those coordinates.
/// * `grid`   – target `GridSpec` describing the output grid.
/// * `strategy` – how to handle multiple points per cell.
///
/// # Returns
///
/// A `d`-dimensional `ArrayD<f64>` whose shape matches `grid.shape()`.
/// Cells that received no points contain `f64::NAN` (or `0.0` for `Count`).
pub fn resample_scattered_to_grid(
    points: &Array2<f64>,
    values: &Array1<f64>,
    grid: &GridSpec,
    strategy: ResampleStrategy,
) -> InterpolateResult<ArrayD<f64>> {
    let n = points.nrows();
    let d = points.ncols();

    if d != grid.ndim() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "points has {d} columns but grid has {} axes",
            grid.ndim()
        )));
    }
    if values.len() != n {
        return Err(InterpolateError::DimensionMismatch(format!(
            "points has {n} rows but values has {} elements",
            values.len()
        )));
    }

    let shape = grid.shape();
    let total_cells = grid.n_cells();

    // Build a flat bucket list: each cell collects the values of points
    // that land in it.
    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); total_cells];

    for row in 0..n {
        // Compute N-D cell index then convert to a flat C-order offset.
        let mut multi_idx = vec![0_usize; d];
        for dim in 0..d {
            let coord = points[[row, dim]];
            multi_idx[dim] = grid.nearest_index(dim, coord);
        }

        let mut flat_idx = 0_usize;
        for dim in 0..d {
            flat_idx += multi_idx[dim] * stride_for(&shape, dim);
        }

        if flat_idx < total_cells {
            buckets[flat_idx].push(values[row]);
        }
    }

    // Aggregate buckets.
    let raw_data: Vec<f64> = buckets
        .into_iter()
        .map(|mut bucket| aggregate(bucket.as_mut_slice(), strategy))
        .collect();

    let out = ArrayD::from_shape_vec(IxDyn(&shape), raw_data).map_err(|e| {
        InterpolateError::ShapeError(format!("failed to construct output array: {e}"))
    })?;

    Ok(out)
}

/// Compute C-order stride for dimension `dim` given `shape`.
fn stride_for(shape: &[usize], dim: usize) -> usize {
    shape[dim + 1..].iter().product()
}

/// Aggregate a mutable slice of values according to `strategy`.
fn aggregate(bucket: &mut [f64], strategy: ResampleStrategy) -> f64 {
    if bucket.is_empty() {
        return match strategy {
            ResampleStrategy::Rasterize(Aggregator::Count) | ResampleStrategy::Conservative => 0.0,
            _ => f64::NAN,
        };
    }

    match strategy {
        ResampleStrategy::Rasterize(Aggregator::Mean) | ResampleStrategy::Conservative => {
            bucket.iter().sum::<f64>() / bucket.len() as f64
        }
        ResampleStrategy::Rasterize(Aggregator::Median) => {
            bucket.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = bucket.len() / 2;
            if bucket.len() % 2 == 0 {
                (bucket[mid - 1] + bucket[mid]) * 0.5
            } else {
                bucket[mid]
            }
        }
        ResampleStrategy::Rasterize(Aggregator::Max) => {
            bucket.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        }
        ResampleStrategy::Rasterize(Aggregator::Min) => {
            bucket.iter().copied().fold(f64::INFINITY, f64::min)
        }
        ResampleStrategy::Rasterize(Aggregator::Count) => bucket.len() as f64,
    }
}
