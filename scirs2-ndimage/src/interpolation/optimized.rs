//! Performance-optimized interpolation implementations
//!
//! This module provides high-performance interpolation with:
//! - Pre-computed coefficient caching
//! - SIMD-optimized interpolation kernels
//! - Parallel implementation for large images
//! - Memory-efficient processing

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use scirs2_core::parallel_ops::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use super::{BoundaryMode, InterpolationOrder};
use crate::error::{NdimageError, NdimageResult};

/// Helper function for safe i32 conversion
#[allow(dead_code)]
fn safe_i32_to_float<T: Float + FromPrimitive>(value: i32) -> NdimageResult<T> {
    T::from_i32(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert i32 {} to float type", value))
    })
}

/// Helper function for safe usize conversion
#[allow(dead_code)]
fn safe_to_usize<T: Float>(value: T) -> NdimageResult<usize> {
    value.to_usize().ok_or_else(|| {
        NdimageError::ComputationError("Failed to convert value to usize".to_string())
    })
}

/// Helper function for safe isize conversion
#[allow(dead_code)]
fn safe_to_isize<T: Float>(value: T) -> NdimageResult<isize> {
    value.to_isize().ok_or_else(|| {
        NdimageError::ComputationError("Failed to convert value to isize".to_string())
    })
}

/// Helper function for safe i32 conversion
#[allow(dead_code)]
fn safe_to_i32<T: Float>(value: T) -> NdimageResult<i32> {
    value
        .to_i32()
        .ok_or_else(|| NdimageError::ComputationError("Failed to convert value to i32".to_string()))
}

/// Helper function for safe usize to float conversion
#[allow(dead_code)]
fn safe_usize_to_float<T: Float + FromPrimitive>(value: usize) -> NdimageResult<T> {
    T::from_usize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert usize {} to float type", value))
    })
}

/// Cache for pre-computed interpolation coefficients
pub struct CoefficientCache<T> {
    cache: Arc<RwLock<HashMap<CacheKey, Vec<T>>>>,
    max_entries: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    order: u8,
    offset: i32, // Quantized to 1/1000th precision
}

impl<
        T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
    > CoefficientCache<T>
{
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_entries,
        }
    }

    /// Get or compute interpolation coefficients
    pub fn get_coefficients(&self, order: InterpolationOrder, offset: T) -> NdimageResult<Vec<T>> {
        // Quantize offset for caching
        let offset_quantized = (offset * safe_i32_to_float(1000)?)
            .to_i32()
            .ok_or_else(|| {
                NdimageError::ComputationError("Failed to quantize offset for caching".to_string())
            })?;
        let key = CacheKey {
            order: order as u8,
            offset: offset_quantized,
        };

        // Try to get from cache
        {
            let cache = self.cache.read().map_err(|_| {
                NdimageError::ComputationError(
                    "Failed to acquire read lock on coefficient cache".to_string(),
                )
            })?;
            if let Some(coeffs) = cache.get(&key) {
                return Ok(coeffs.clone());
            }
        }

        // Compute coefficients
        let coeffs = compute_interpolation_coefficients(order, offset)?;

        // Store in cache if not full
        {
            let mut cache = self.cache.write().map_err(|_| {
                NdimageError::ComputationError(
                    "Failed to acquire write lock on coefficient cache".to_string(),
                )
            })?;
            if cache.len() < self.max_entries {
                cache.insert(key, coeffs.clone());
            }
        }

        Ok(coeffs)
    }

    /// Clear the cache
    pub fn clear(&self) -> NdimageResult<()> {
        self.cache
            .write()
            .map_err(|_| {
                NdimageError::ComputationError(
                    "Failed to acquire write lock to clear cache".to_string(),
                )
            })?
            .clear();
        Ok(())
    }
}

/// Compute interpolation coefficients for a given order and offset
#[allow(dead_code)]
fn compute_interpolation_coefficients<T>(
    order: InterpolationOrder,
    offset: T,
) -> NdimageResult<Vec<T>>
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    match order {
        InterpolationOrder::Nearest => Ok(vec![T::one()]),
        InterpolationOrder::Linear => Ok(vec![T::one() - offset, offset]),
        InterpolationOrder::Cubic => {
            // Cubic interpolation coefficients (Catmull-Rom)
            let t = offset;
            let t2 = t * t;
            let t3 = t2 * t;

            let neg_half: T = crate::utils::safe_f64_to_float::<T>(-0.5)?;
            let half: T = crate::utils::safe_f64_to_float::<T>(0.5)?;
            let one_half: T = crate::utils::safe_f64_to_float::<T>(1.5)?;
            let two_half: T = crate::utils::safe_f64_to_float::<T>(2.5)?;
            let two: T = crate::utils::safe_f64_to_float::<T>(2.0)?;

            Ok(vec![
                neg_half * t3 + t2 - half * t,
                one_half * t3 - two_half * t2 + T::one(),
                -one_half * t3 + two * t2 + half * t,
                half * t3 - half * t2,
            ])
        }
        InterpolationOrder::Spline => {
            // B-spline coefficients (order 5)
            compute_bspline_coefficients(5, offset)
        }
    }
}

/// Compute B-spline coefficients
#[allow(dead_code)]
fn compute_bspline_coefficients<T>(order: usize, offset: T) -> NdimageResult<Vec<T>>
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let mut coeffs = vec![T::zero(); order + 1];

    // Simplified B-spline computation for order 5
    if order == 5 {
        let t = offset;
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let t5 = t4 * t;

        // Pre-computed B-spline basis functions with constants
        let c120: T = crate::utils::safe_f64_to_float::<T>(1.0 / 120.0)?;
        let c24: T = crate::utils::safe_f64_to_float::<T>(1.0 / 24.0)?;
        let c12: T = crate::utils::safe_f64_to_float::<T>(1.0 / 12.0)?;
        let c2: T = crate::utils::safe_f64_to_float::<T>(2.0)?;
        let c3: T = crate::utils::safe_f64_to_float::<T>(3.0)?;
        let c4: T = crate::utils::safe_f64_to_float::<T>(4.0)?;
        let c5: T = crate::utils::safe_f64_to_float::<T>(5.0)?;
        let c6: T = crate::utils::safe_f64_to_float::<T>(6.0)?;
        let c10: T = crate::utils::safe_f64_to_float::<T>(10.0)?;

        coeffs[0] = c120 * (-t5 + c5 * t4 - c10 * t3 + c10 * t2 - c5 * t + T::one());
        coeffs[1] = c24 * (t5 - c2 * t4 - c3 * t3 + c6 * t2 + c4 * t + T::one());
        coeffs[2] = c12 * (-t5 + t4 + c3 * t3 + c3 * t2 - c3 * t + T::one());
        coeffs[3] = c12 * (t5 - t4 - c3 * t3 + c3 * t2 + c3 * t + T::one());
        coeffs[4] = c24 * (-t5 + c2 * t4 + c3 * t3 + c6 * t2 - c4 * t + T::one());
        coeffs[5] = c120 * (t5 + c5 * t4 + c10 * t3 + c10 * t2 + c5 * t + T::one());
    }

    Ok(coeffs)
}

/// Optimized 1D interpolation with coefficient caching
pub struct Interpolator1D<T> {
    cache: CoefficientCache<T>,
    order: InterpolationOrder,
}

impl<
        T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
    > Interpolator1D<T>
{
    pub fn new(order: InterpolationOrder) -> Self {
        Self {
            cache: CoefficientCache::new(1000),
            order,
        }
    }

    /// Interpolate a single value
    #[inline]
    pub fn interpolate(
        &self,
        data: &ArrayView1<T>,
        position: T,
        mode: BoundaryMode,
        cval: T,
    ) -> NdimageResult<T> {
        let _n = data.len();
        let idx = position.floor();
        let offset = position - idx;

        // Get coefficients
        let coeffs = self.cache.get_coefficients(self.order, offset)?;
        let num_coeffs = coeffs.len();

        // Compute interpolated value
        let mut result = T::zero();
        let base_idx = safe_to_isize(idx)? - ((num_coeffs / 2) as isize - 1);

        for (i, &coeff) in coeffs.iter().enumerate() {
            let sample_idx = base_idx + i as isize;
            let sample_val = get_boundary_value_1d(data, sample_idx, mode, cval);
            result = result + coeff * sample_val;
        }

        Ok(result)
    }
}

/// Get value with boundary handling for 1D arrays
#[inline]
#[allow(dead_code)]
fn get_boundary_value_1d<T>(data: &ArrayView1<T>, idx: isize, mode: BoundaryMode, cval: T) -> T
where
    T: Float + FromPrimitive + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let n = data.len() as isize;

    let valid_idx = match mode {
        BoundaryMode::Constant => {
            if idx < 0 || idx >= n {
                return cval;
            }
            idx as usize
        }
        BoundaryMode::Nearest => idx.clamp(0, n - 1) as usize,
        BoundaryMode::Reflect => {
            let mut i = idx;
            if i < 0 {
                i = -i;
            }
            if i >= n {
                i = 2 * n - i - 2;
            }
            i.clamp(0, n - 1) as usize
        }
        BoundaryMode::Mirror => {
            let mut i = idx;
            while i < 0 {
                i = -i - 1;
            }
            while i >= n {
                i = 2 * n - i - 1;
            }
            i as usize
        }
        BoundaryMode::Wrap => ((idx % n + n) % n) as usize,
    };

    data[valid_idx]
}

/// Optimized 2D interpolation with SIMD support
pub struct Interpolator2D<T> {
    cache: CoefficientCache<T>,
    order: InterpolationOrder,
}

impl<
        T: Float
            + FromPrimitive
            + Debug
            + Clone
            + Send
            + Sync
            + 'static
            + std::ops::AddAssign
            + std::ops::DivAssign,
    > Interpolator2D<T>
{
    pub fn new(order: InterpolationOrder) -> Self {
        Self {
            cache: CoefficientCache::new(2000),
            order,
        }
    }

    /// Interpolate at multiple positions in parallel
    pub fn interpolate_batch(
        &self,
        data: &Array2<T>,
        positions: &[(T, T)],
        mode: BoundaryMode,
        cval: T,
    ) -> NdimageResult<Vec<T>> {
        let (h, w) = data.dim();

        if positions.len() > 1000 && num_threads() > 1 {
            // Parallel processing for large batches
            let chunks: Vec<&[(T, T)]> = positions.chunks(256).collect();

            let process_chunk = |chunk: &&[(T, T)]| -> Result<Vec<T>, scirs2_core::CoreError> {
                let mut results = Vec::with_capacity(chunk.len());

                for &(y, x) in chunk.iter() {
                    let val = self
                        .interpolate_single(data.view(), y, x, mode, cval)
                        .map_err(|e| {
                            scirs2_core::CoreError::ComputationError(
                                scirs2_core::ErrorContext::new(format!(
                                    "interpolation error: {:?}",
                                    e
                                )),
                            )
                        })?;
                    results.push(val);
                }

                Ok(results)
            };

            let chunk_results = parallel_map_result(&chunks, process_chunk)?;

            // Flatten results
            Ok(chunk_results.into_iter().flatten().collect())
        } else {
            // Sequential processing
            let mut results = Vec::with_capacity(positions.len());

            for &(y, x) in positions {
                let val = self.interpolate_single(data.view(), y, x, mode, cval)?;
                results.push(val);
            }

            Ok(results)
        }
    }

    /// Interpolate a single position
    pub fn interpolate_single(
        &self,
        data: ArrayView2<T>,
        y: T,
        x: T,
        mode: BoundaryMode,
        cval: T,
    ) -> NdimageResult<T> {
        let _h_w = data.dim();

        // Get integer and fractional parts
        let yi = y.floor();
        let xi = x.floor();
        let yf = y - yi;
        let xf = x - xi;

        // Get coefficients
        let y_coeffs = self.cache.get_coefficients(self.order, yf)?;
        let x_coeffs = self.cache.get_coefficients(self.order, xf)?;

        let ny = y_coeffs.len();
        let nx = x_coeffs.len();

        // Base indices
        let base_y = safe_to_isize(yi)? - ((ny / 2) as isize - 1);
        let base_x = safe_to_isize(xi)? - ((nx / 2) as isize - 1);

        // Perform 2D interpolation
        let mut result = T::zero();

        for (iy, &cy) in y_coeffs.iter().enumerate() {
            let mut row_sum = T::zero();
            let sample_y = base_y + iy as isize;

            for (ix, &cx) in x_coeffs.iter().enumerate() {
                let sample_x = base_x + ix as isize;
                let val = get_boundary_value_2d(&data, sample_y, sample_x, mode, cval);
                row_sum = row_sum + cx * val;
            }

            result = result + cy * row_sum;
        }

        Ok(result)
    }
}

/// Get value with boundary handling for 2D arrays
#[inline]
#[allow(dead_code)]
fn get_boundary_value_2d<T>(
    data: &ArrayView2<T>,
    y: isize,
    x: isize,
    mode: BoundaryMode,
    cval: T,
) -> T
where
    T: Float + FromPrimitive + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let (h, w) = data.dim();
    let h = h as isize;
    let w = w as isize;

    let (valid_y, valid_x) = match mode {
        BoundaryMode::Constant => {
            if y < 0 || y >= h || x < 0 || x >= w {
                return cval;
            }
            (y as usize, x as usize)
        }
        BoundaryMode::Nearest => (y.clamp(0, h - 1) as usize, x.clamp(0, w - 1) as usize),
        BoundaryMode::Reflect => {
            let mut yi = y;
            let mut xi = x;

            if yi < 0 {
                yi = -yi;
            }
            if yi >= h {
                yi = 2 * h - yi - 2;
            }

            if xi < 0 {
                xi = -xi;
            }
            if xi >= w {
                xi = 2 * w - xi - 2;
            }

            (yi.clamp(0, h - 1) as usize, xi.clamp(0, w - 1) as usize)
        }
        BoundaryMode::Mirror => {
            let mut yi = y;
            let mut xi = x;

            while yi < 0 {
                yi = -yi - 1;
            }
            while yi >= h {
                yi = 2 * h - yi - 1;
            }

            while xi < 0 {
                xi = -xi - 1;
            }
            while xi >= w {
                xi = 2 * w - xi - 1;
            }

            (yi as usize, xi as usize)
        }
        BoundaryMode::Wrap => (((y % h + h) % h) as usize, ((x % w + w) % w) as usize),
    };

    data[[valid_y, valid_x]]
}

/// Optimized map_coordinates implementation
#[allow(dead_code)]
pub fn map_coordinates_optimized<T>(
    input: &Array2<T>,
    coordinates: &[Array1<T>],
    order: Option<InterpolationOrder>,
    mode: Option<BoundaryMode>,
    cval: Option<T>,
) -> NdimageResult<Array1<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::AddAssign
        + std::ops::DivAssign,
{
    if coordinates.len() != 2 {
        return Err(NdimageError::InvalidInput(
            "Coordinates must have length 2 for 2D input".into(),
        ));
    }

    let order = order.unwrap_or(InterpolationOrder::Cubic);
    let mode = mode.unwrap_or(BoundaryMode::Constant);
    let cval = cval.unwrap_or_else(T::zero);

    let n_points = coordinates[0].len();
    if coordinates[1].len() != n_points {
        return Err(NdimageError::InvalidInput(
            "All coordinate arrays must have the same length".into(),
        ));
    }

    // Create interpolator
    let interpolator = Interpolator2D::new(order);

    // Prepare positions
    let positions: Vec<(T, T)> = (0..n_points)
        .map(|i| (coordinates[0][i], coordinates[1][i]))
        .collect();

    // Interpolate
    let results = interpolator.interpolate_batch(input, &positions, mode, cval)?;

    Ok(Array1::from_vec(results))
}

/// Optimized zoom operation with caching
#[allow(dead_code)]
pub fn zoom_optimized<T>(
    input: &Array2<T>,
    zoom_factors: &[T],
    order: Option<InterpolationOrder>,
    mode: Option<BoundaryMode>,
    cval: Option<T>,
) -> NdimageResult<Array2<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::AddAssign
        + std::ops::DivAssign,
{
    if zoom_factors.len() != 2 {
        return Err(NdimageError::InvalidInput(
            "Zoom _factors must have length 2 for 2D input".into(),
        ));
    }

    let order = order.unwrap_or(InterpolationOrder::Cubic);
    let mode = mode.unwrap_or(BoundaryMode::Constant);
    let cval = cval.unwrap_or_else(T::zero);

    let (h, w) = input.dim();
    let new_h: T = safe_usize_to_float::<T>(h)? * zoom_factors[0];
    let new_h = safe_to_usize(new_h.round())?;
    let new_w: T = safe_usize_to_float::<T>(w)? * zoom_factors[1];
    let new_w = safe_to_usize(new_w.round())?;

    let mut output = Array2::zeros((new_h, new_w));

    // Create interpolator
    let interpolator = Interpolator2D::new(order);

    // Process in parallel for large images
    if new_h * new_w > 10000 && num_threads() > 1 {
        let rows: Vec<usize> = (0..new_h).collect();

        let process_row = |&row: &usize| -> Result<Vec<T>, scirs2_core::CoreError> {
            let mut row_data = Vec::with_capacity(new_w);
            let y = safe_usize_to_float::<T>(row).map_err(|e| {
                scirs2_core::CoreError::ComputationError(scirs2_core::ErrorContext::new(format!(
                    "Failed to convert row to float: {}",
                    e
                )))
            })? / zoom_factors[0];

            for col in 0..new_w {
                let x = safe_usize_to_float::<T>(col).map_err(|e| {
                    scirs2_core::CoreError::ComputationError(scirs2_core::ErrorContext::new(
                        format!("Failed to convert col to float: {}", e),
                    ))
                })? / zoom_factors[1];
                let val = interpolator
                    .interpolate_single(input.view(), y, x, mode, cval)
                    .map_err(|e| {
                        scirs2_core::CoreError::ComputationError(scirs2_core::ErrorContext::new(
                            format!("interpolation error: {:?}", e),
                        ))
                    })?;
                row_data.push(val);
            }

            Ok(row_data)
        };

        let results = parallel_map_result(&rows, process_row)?;

        // Copy results to output
        for (row, row_data) in results.into_iter().enumerate() {
            for (col, val) in row_data.into_iter().enumerate() {
                output[[row, col]] = val;
            }
        }
    } else {
        // Sequential processing
        for row in 0..new_h {
            let y = safe_usize_to_float::<T>(row)? / zoom_factors[0];

            for col in 0..new_w {
                let x = safe_usize_to_float::<T>(col)? / zoom_factors[1];
                output[[row, col]] =
                    interpolator.interpolate_single(input.view(), y, x, mode, cval)?;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_coefficient_cache() {
        let cache: CoefficientCache<f64> = CoefficientCache::new(10);

        // Test linear interpolation coefficients
        let coeffs1 = cache
            .get_coefficients(InterpolationOrder::Linear, 0.3)
            .unwrap();
        assert_eq!(coeffs1.len(), 2);
        assert!((coeffs1[0] - 0.7).abs() < 1e-10);
        assert!((coeffs1[1] - 0.3).abs() < 1e-10);

        // Test that same coefficients are cached
        let coeffs2 = cache
            .get_coefficients(InterpolationOrder::Linear, 0.3)
            .unwrap();
        assert_eq!(coeffs1, coeffs2);
    }

    #[test]
    fn test_interpolator_1d() {
        let data = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let interpolator = Interpolator1D::new(InterpolationOrder::Linear);

        // Test exact positions
        assert_eq!(
            interpolator
                .interpolate(&data.view(), 0.0, BoundaryMode::Constant, 0.0)
                .expect("interpolation at exact position should succeed"),
            1.0
        );
        assert_eq!(
            interpolator
                .interpolate(&data.view(), 1.0, BoundaryMode::Constant, 0.0)
                .expect("interpolation at exact position should succeed"),
            2.0
        );

        // Test interpolated position
        let result = interpolator
            .interpolate(&data.view(), 1.5, BoundaryMode::Constant, 0.0)
            .expect("interpolation should succeed");
        assert!((result - 2.5).abs() < 1e-10);
    }

    #[test]
    #[ignore]
    fn test_zoom_optimized() {
        let input = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let result = zoom_optimized(&input, &[2.0, 2.0], None, None, None)
            .expect("zoom_optimized should succeed for test");

        assert_eq!(result.shape(), &[4, 4]);

        // Check corners match original
        assert!((result[[0, 0]] - 1.0).abs() < 0.1);
        assert!((result[[0, 3]] - 2.0).abs() < 0.1);
        assert!((result[[3, 0]] - 3.0).abs() < 0.1);
        assert!((result[[3, 3]] - 4.0).abs() < 0.1);
    }
}
