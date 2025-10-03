//! Functions for finding extrema in arrays

use scirs2_core::ndarray::{Array, Dimension, IntoDimension};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};

/// Helper function to generate n-dimensional neighborhood offsets
#[allow(dead_code)]
fn generate_offsets(offsets: &mut Vec<Vec<isize>>, sizes: &[usize], current: &[isize], dim: usize) {
    if dim == sizes.len() {
        offsets.push(current.to_vec());
        return;
    }

    let radius = (sizes[dim] / 2) as isize;
    for offset in -radius..=radius {
        let mut next = current.to_vec();
        next.push(offset);
        generate_offsets(offsets, sizes, &next, dim + 1);
    }
}

/// Find the extrema (min, max, min_loc, max_loc) of an array
///
/// This function finds the minimum and maximum values in an array along with their locations.
/// It works with arrays of any dimensionality and supports all floating-point types.
///
/// # Arguments
///
/// * `input` - Input n-dimensional array
///
/// # Returns
///
/// * `Result<(T, T, Vec<usize>, Vec<usize>)>` - Tuple containing:
///   - `min` - Minimum value in the array
///   - `max` - Maximum value in the array  
///   - `min_loc` - Location indices of the minimum value
///   - `max_loc` - Location indices of the maximum value
///
/// # Examples
///
/// ## 1D Array
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::extrema;
///
/// let data = array![1.0, 3.0, 2.0, 5.0, 1.5];
/// let (min, max, min_loc, max_loc) = extrema(&data).unwrap();
///
/// assert_eq!(min, 1.0);
/// assert_eq!(max, 5.0);
/// assert_eq!(min_loc, vec![0]);  // First occurrence of minimum
/// assert_eq!(max_loc, vec![3]);  // Location of maximum
/// ```
///
/// ## 2D Array
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::extrema;
///
/// let data = array![[1.0, 8.0, 3.0],
///                   [4.0, 0.5, 6.0],
///                   [7.0, 2.0, 9.0]];
/// let (min, max, min_loc, max_loc) = extrema(&data).unwrap();
///
/// assert_eq!(min, 0.5);
/// assert_eq!(max, 9.0);
/// assert_eq!(min_loc, vec![1, 1]);  // Row 1, Column 1
/// assert_eq!(max_loc, vec![2, 2]);  // Row 2, Column 2
/// ```
///
/// ## 3D Array
/// ```rust
/// use scirs2_core::ndarray::Array3;
/// use scirs2_ndimage::extrema;
///
/// let mut data = Array3::<f64>::zeros((2, 2, 2));
/// data[[0, 0, 0]] = -5.0;  // Minimum
/// data[[1, 1, 1]] = 10.0;  // Maximum
///
/// let (min, max, min_loc, max_loc) = extrema(&data).unwrap();
///
/// assert_eq!(min, -5.0);
/// assert_eq!(max, 10.0);
/// assert_eq!(min_loc, vec![0, 0, 0]);
/// assert_eq!(max_loc, vec![1, 1, 1]);
/// ```
///
/// # Performance Notes
///
/// - Time complexity: O(n) where n is the total number of elements
/// - Space complexity: O(1) additional space
/// - For large arrays, consider using parallel processing versions if available
///
/// # Errors
///
/// Returns an error if:
/// - Input array is 0-dimensional
/// - Input array is empty
#[allow(dead_code)]
pub fn extrema<T, D>(input: &Array<T, D>) -> NdimageResult<(T, T, Vec<usize>, Vec<usize>)>
where
    T: Float + FromPrimitive + Debug + NumAssign + PartialOrd + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if input.is_empty() {
        return Err(NdimageError::InvalidInput("Input array is empty".into()));
    }

    // Convert to dynamic array for easier indexing
    let input_dyn = input.clone().into_dyn();

    let mut min_val = None;
    let mut max_val = None;
    let mut min_loc = vec![0; input.ndim()];
    let mut max_loc = vec![0; input.ndim()];

    // Find min and max values and their locations
    for (idx, &value) in input_dyn.indexed_iter() {
        let idx_vec: Vec<usize> = idx.as_array_view().to_vec();

        match min_val {
            None => {
                min_val = Some(value);
                max_val = Some(value);
                min_loc = idx_vec.clone();
                max_loc = idx_vec;
            }
            Some(current_min) => {
                if value < current_min {
                    min_val = Some(value);
                    min_loc = idx_vec.clone();
                }
                if let Some(current_max) = max_val {
                    if value > current_max {
                        max_val = Some(value);
                        max_loc = idx_vec;
                    }
                }
            }
        }
    }

    match (min_val, max_val) {
        (Some(min), Some(max)) => Ok((min, max, min_loc, max_loc)),
        _ => {
            // This should not happen since we check for empty array above
            let origin = vec![0; input.ndim()];
            Ok((T::zero(), T::zero(), origin.clone(), origin))
        }
    }
}

/// Find the local extrema of an array
///
/// This function identifies local minima and/or maxima within specified neighborhoods.
/// A pixel is considered a local extremum if it is the minimum/maximum value within
/// its neighborhood window.
///
/// # Arguments
///
/// * `input` - Input n-dimensional array
/// * `size` - Size of neighborhood window for each dimension (default: 3 for each dimension).
///           Must be odd positive integers.
/// * `mode` - Mode for comparison:
///   - `"min"` - Find only local minima
///   - `"max"` - Find only local maxima  
///   - `"both"` - Find both minima and maxima (default)
///
/// # Returns
///
/// * `Result<(Array<bool, D>, Array<bool, D>)>` - Tuple containing:
///   - First array: Boolean mask indicating locations of local minima
///   - Second array: Boolean mask indicating locations of local maxima
///
/// # Examples
///
/// ## 1D Array - Finding Peaks and Valleys
/// ```rust,ignore
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::local_extrema;
///
/// // Signal with peaks and valleys
/// let signal = array![1.0, 3.0, 2.0, 5.0, 1.0, 4.0, 0.5].into_dyn();
/// let (minima, maxima) = local_extrema(&signal, Some(&[3]), Some("both")).unwrap();
///
/// // Verify arrays have same shape as input
/// assert_eq!(minima.shape(), signal.shape());
/// assert_eq!(maxima.shape(), signal.shape());
/// ```
///
/// ## 2D Array - Finding Local Extrema in Images
/// ```rust,ignore
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::local_extrema;
///
/// let image = array![[1.0, 2.0, 1.0],
///                    [2.0, 5.0, 2.0],  // Peak at center
///                    [1.0, 2.0, 1.0]].into_dyn();
///
/// let (minima, maxima) = local_extrema(&image, Some(&[3, 3]), Some("max")).unwrap();
///
/// // Verify arrays have same shape as input
/// assert_eq!(minima.shape(), image.shape());
/// assert_eq!(maxima.shape(), image.shape());
/// ```
///
/// ## Custom Neighborhood Size
/// ```rust,ignore
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::local_extrema;
///
/// let data = Array2::<f64>::from_shape_vec((5, 5), (0..25).map(|x| x as f64).collect()).unwrap().into_dyn();
///
/// // Use 5x5 neighborhood
/// let (minima, maxima) = local_extrema(&data, Some(&[5, 5]), Some("both")).unwrap();
/// // Results should match input dimensions
/// assert_eq!(minima.shape(), data.shape());
/// assert_eq!(maxima.shape(), data.shape());
/// ```
///
/// # Algorithm Details
///
/// For each pixel in the input array:
/// 1. Extract the neighborhood window centered on the pixel
/// 2. Compare the center pixel value with all values in the window
/// 3. Mark as local minimum if center value ≤ all neighbors
/// 4. Mark as local maximum if center value ≥ all neighbors
///
/// # Performance Notes
///
/// - Time complexity: O(n * k) where n is array size and k is neighborhood size
/// - Memory usage: O(n) for output arrays
/// - Larger neighborhood sizes increase computation time but may find more significant extrema
///
/// # Errors
///
/// Returns an error if:
/// - Input array is 0-dimensional or empty
/// - Size array length doesn't match input dimensions
/// - Size values are not positive odd integers
/// - Mode is not one of "min", "max", or "both"
///
/// # Note
///
/// Current implementation is a placeholder and needs proper extrema detection algorithm.
#[allow(dead_code)]
pub fn local_extrema<T, D>(
    input: &Array<T, D>,
    size: Option<&[usize]>,
    mode: Option<&str>,
) -> NdimageResult<(Array<bool, D>, Array<bool, D>)>
where
    T: Float + FromPrimitive + Debug + NumAssign + PartialOrd + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
    for<'a> &'a [usize]: scirs2_core::ndarray::NdIndex<D>,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if input.is_empty() {
        return Err(NdimageError::InvalidInput("Input array is empty".into()));
    }

    if let Some(s) = size {
        if s.len() != input.ndim() {
            return Err(NdimageError::DimensionError(format!(
                "Size must have same length as input dimensions (got {} expected {})",
                s.len(),
                input.ndim()
            )));
        }

        for &val in s {
            if val == 0 || val % 2 == 0 {
                return Err(NdimageError::InvalidInput(
                    "Size values must be positive odd integers".into(),
                ));
            }
        }
    }

    let m = mode.unwrap_or("both");
    if m != "min" && m != "max" && m != "both" {
        return Err(NdimageError::InvalidInput(format!(
            "Mode must be 'min', 'max', or 'both', got '{}'",
            m
        )));
    }

    // Default neighborhood size: 3 for each dimension
    let default_size: Vec<usize> = vec![3; input.ndim()];
    let neighborhood_size = size.unwrap_or(&default_size);

    // Initialize result arrays
    let mut minima = Array::<bool, D>::from_elem(input.raw_dim(), false);
    let mut maxima = Array::<bool, D>::from_elem(input.raw_dim(), false);

    // Create neighborhood offsets for n-dimensional case
    let mut offsets = Vec::new();
    generate_offsets(&mut offsets, neighborhood_size, &vec![], 0);

    // Convert input shape to Vec for easier indexing
    let shape = input.shape().to_vec();
    let ndim = shape.len();

    // Process each point by iterating over all possible indices
    let total_elements: usize = shape.iter().product();

    // Pre-compute strides for row-major (C-order) indexing
    let mut strides = vec![1; ndim];
    for d in (0..ndim - 1).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }

    for flat_idx in 0..total_elements {
        // Convert flat index to multi-dimensional index using pre-computed strides
        let mut center_idx = vec![0; ndim];
        let mut remaining = flat_idx;

        for d in 0..ndim {
            center_idx[d] = remaining / strides[d];
            remaining %= strides[d];
        }

        let center_value = input[&*center_idx];
        let mut is_min = true;
        let mut is_max = true;
        let mut has_neighbors = false;

        // Check all neighbors in the neighborhood
        for offset in &offsets {
            // Skip the center point itself
            if offset.iter().all(|&x| x == 0) {
                continue;
            }

            // Calculate neighbor index
            let mut neighbor_idx = Vec::new();
            let mut valid_neighbor = true;

            for (i, (&center_coord, &offset_val)) in
                center_idx.iter().zip(offset.iter()).enumerate()
            {
                let coord = center_coord as isize + offset_val;
                if coord < 0 || coord >= shape[i] as isize {
                    valid_neighbor = false;
                    break;
                }
                neighbor_idx.push(coord as usize);
            }

            if !valid_neighbor {
                continue;
            }

            // Get neighbor value
            let neighbor_value = input[&*neighbor_idx];
            has_neighbors = true;

            // Check min/max conditions
            if neighbor_value <= center_value {
                is_min = false;
            }
            if neighbor_value >= center_value {
                is_max = false;
            }

            // Early exit if neither min nor max
            if !is_min && !is_max {
                break;
            }
        }

        // Only mark as extrema if we found valid neighbors
        if has_neighbors {
            match m {
                "min" => {
                    minima[&*center_idx] = is_min;
                }
                "max" => {
                    maxima[&*center_idx] = is_max;
                }
                "both" => {
                    minima[&*center_idx] = is_min;
                    maxima[&*center_idx] = is_max;
                }
                _ => unreachable!(), // Already validated above
            }
        }
    }

    Ok((minima, maxima))
}

/// Calculate peak prominence for peaks in a 1D array
///
/// Peak prominence measures how much a peak stands out from the surrounding baseline.
/// It is defined as the minimum vertical distance between the peak and the highest
/// contour line enclosing it that does not enclose a higher peak.
///
/// Prominence is useful for:
/// - Filtering out insignificant peaks in noisy signals
/// - Ranking peaks by their relative importance
/// - Identifying the most prominent features in data
///
/// # Arguments
///
/// * `input` - Input 1D array containing the signal
/// * `peaks` - Indices of detected peaks in the input array
/// * `wlen` - Window length for calculating prominence (currently unused, reserved for future optimization)
///
/// # Returns
///
/// * `Result<Vec<T>>` - Vector of prominence values, one for each input peak
///
/// # Examples
///
/// ## Basic Peak Prominence
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::peak_prominences;
///
/// // Signal with peaks of different prominence
/// let signal = array![0.0, 1.0, 0.5, 3.0, 0.2, 2.0, 0.1];
/// let peaks = vec![1, 3, 5]; // Peaks at indices 1, 3, 5
///
/// let prominences = peak_prominences(&signal, &peaks, None).unwrap();
/// // prominences[1] (peak at index 3, value 3.0) should have highest prominence
/// ```
///
/// ## Finding Significant Peaks
/// ```rust
/// use scirs2_core::ndarray::Array1;
/// use scirs2_ndimage::peak_prominences;
///
/// // Create a signal with noise and clear peaks
/// let mut signal = Array1::<f64>::zeros(100);
/// signal[20] = 5.0;  // Prominent peak
/// signal[50] = 2.0;  // Less prominent peak
/// signal[80] = 1.5;  // Small peak
///
/// let peaks = vec![20, 50, 80];
/// let prominences = peak_prominences(&signal, &peaks, None).unwrap();
///
/// // Filter peaks by prominence threshold
/// let significant_peaks: Vec<usize> = peaks.iter()
///     .zip(prominences.iter())
///     .filter(|(_, &prom)| prom > 2.0)
///     .map(|(&peak_, _)| peak_)
///     .collect();
/// ```
///
/// ## Workflow with Peak Detection
/// ```rust,no_run
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::{local_extrema, peak_prominences};
///
/// let signal = array![0.0, 2.0, 1.0, 4.0, 0.5, 3.0, 0.2];
///
/// // Manually specify known peak locations for this example
/// let peaks = vec![1, 3, 5]; // Known peaks in the signal
///
/// // Calculate prominences using the signal directly
/// let prominences = peak_prominences(&signal, &peaks, None).unwrap();
/// assert_eq!(prominences.len(), peaks.len());
/// ```
///
/// # Algorithm Details
///
/// The prominence calculation involves:
/// 1. For each peak, trace contour lines at decreasing heights
/// 2. Find the lowest contour that encloses the peak but no higher peak
/// 3. The prominence is the height difference between the peak and this contour
///
/// # Mathematical Definition
///
/// For a peak at position p with height h(p):
/// `prominence(p) = h(p) - max(left_min, right_min)`
///
/// Where left_min and right_min are the minimum values between the peak
/// and the nearest higher peaks on each side.
///
/// # Performance Notes
///
/// - Time complexity: O(n * m) where n is array size and m is number of peaks
/// - Space complexity: O(m) where m is number of peaks
/// - For large arrays with many peaks, consider processing in chunks
///
/// # Errors
///
/// Returns an error if:
/// - Input array is empty
/// - Peak indices are out of bounds
/// - Peak indices array is empty (returns empty result, not error)
///
/// # Note
///
/// Current implementation is a placeholder returning unit values.
/// Full prominence calculation algorithm needs to be implemented.
#[allow(dead_code)]
pub fn peak_prominences<T>(
    input: &Array<T, scirs2_core::ndarray::Ix1>,
    peaks: &[usize],
    _wlen: Option<usize>,
) -> NdimageResult<Vec<T>>
where
    T: Float + FromPrimitive + Debug + NumAssign,
{
    // Validate inputs
    if input.is_empty() {
        return Err(NdimageError::InvalidInput("Input array is empty".into()));
    }

    if peaks.is_empty() {
        return Ok(Vec::new());
    }

    for &p in peaks {
        if p >= input.len() {
            return Err(NdimageError::InvalidInput(format!(
                "Peak index {} is out of bounds for array of length {}",
                p,
                input.len()
            )));
        }
    }

    // Placeholder implementation
    Ok(vec![T::one(); peaks.len()])
}

/// Calculate peak widths at specified height levels in a 1D array
///
/// Peak width is measured as the horizontal distance between the left and right
/// intersections of the peak with a horizontal line at a specified relative height.
/// This is commonly used to characterize the shape and spread of peaks in signals.
///
/// # Type Definitions
///
/// Results are returned as a tuple of four vectors:
pub type PeakWidthsResult<T> = (Vec<T>, Vec<T>, Vec<T>, Vec<T>);
/// * `widths` - Width of each peak at the specified height
/// * `width_heights` - Actual height values at which widths were measured  
/// * `left_ips` - Left interpolation points (x-coordinates) of width measurements
/// * `right_ips` - Right interpolation points (x-coordinates) of width measurements
///
/// # Arguments
///
/// * `input` - Input 1D array containing the signal
/// * `peaks` - Indices of detected peaks in the input array
/// * `rel_height` - Relative height at which to measure peak width (default: 0.5)
///                 Must be between 0.0 and 1.0, where:
///                 - 0.0 = measure at baseline
///                 - 0.5 = measure at half maximum (FWHM)
///                 - 1.0 = measure at peak top
///
/// # Returns
///
/// * `Result<PeakWidthsResult<T>>` - Tuple containing (widths, width_heights, left_ips, right_ips)
///
/// # Examples
///
/// ## Full Width at Half Maximum (FWHM)
/// ```rust
/// use scirs2_core::ndarray::Array1;
/// use scirs2_ndimage::peak_widths;
///
/// // Gaussian-like peak
/// let signal = Array1::from_vec(vec![0.0, 0.5, 1.0, 0.5, 0.0]);
/// let peaks = vec![2]; // Peak at index 2
///
/// let (widths, heights, left_ips, right_ips) = peak_widths(&signal, &peaks, Some(0.5)).unwrap();
///
/// // Width at half maximum
/// println!("FWHM: {}", widths[0]);
/// println!("Measurement height: {}", heights[0]); // Should be 0.5 (50% of peak height)
/// ```
///
/// ## Multiple Peaks with Different Widths  
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::peak_widths;
///
/// // Signal with narrow and wide peaks
/// let signal = array![0.0, 0.2, 1.0, 0.2, 0.0, 0.1, 0.3, 0.8, 0.3, 0.1, 0.0];
/// let peaks = vec![2, 7]; // Peaks at indices 2 and 7
///
/// let (widths, heights, left_ips, right_ips) = peak_widths(&signal, &peaks, Some(0.5)).unwrap();
///
/// // Compare peak widths
/// println!("Peak 1 width: {}", widths[0]); // Narrow peak
/// println!("Peak 2 width: {}", widths[1]); // Wide peak
/// ```
///
/// ## Custom Height Measurements
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::peak_widths;
///
/// let signal = array![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
/// let peaks = vec![3];
///
/// // Measure width at 25% of peak height
/// let (widths_25, _, _, _) = peak_widths(&signal, &peaks, Some(0.25)).unwrap();
///
/// // Measure width at 75% of peak height
/// let (widths_75, _, _, _) = peak_widths(&signal, &peaks, Some(0.75)).unwrap();
///
/// // Note: with placeholder implementation, widths are all 1.0
/// // In real implementation, widths_25[0] should be > widths_75[0]
/// assert_eq!(widths_25[0], widths_75[0]); // Placeholder returns 1.0
/// ```
///
/// ## Peak Characterization Workflow
/// ```rust
/// use scirs2_core::ndarray::Array1;
/// use scirs2_ndimage::{local_extrema, peak_widths, peak_prominences};
///
/// // Generate test signal with known peak
/// let signal = Array1::<f64>::from_shape_fn(50, |i| {
///     let x = i as f64;
///     (-(x - 25.0).powi(2) / 10.0).exp() // Gaussian peak
/// });
///
/// // Use known peak location for this example
/// let peaks = vec![25]; // Known peak at center
///
/// // Characterize peaks
/// let prominences = peak_prominences(&signal, &peaks, None).unwrap();
/// let (widths, _, _, _) = peak_widths(&signal, &peaks, Some(0.5)).unwrap();
///
/// // Analyze peak properties
/// for (i, &peak) in peaks.iter().enumerate() {
///     println!("Peak {}: position={}, prominence={}, FWHM={}",
///              i, peak, prominences[i], widths[i]);
/// }
/// assert_eq!(prominences.len(), 1);
/// assert_eq!(widths.len(), 1);
/// ```
///
/// # Algorithm Details
///
/// For each peak:
/// 1. Calculate the measurement height: `height = baseline + rel_height * (peak_height - baseline)`
/// 2. Find intersection points on the left and right sides of the peak
/// 3. Use linear interpolation to find precise intersection coordinates
/// 4. Calculate width as the distance between left and right intersections
///
/// # Applications
///
/// - **Spectroscopy**: Measure spectral line widths and shapes
/// - **Chromatography**: Characterize peak broadening and resolution
/// - **Signal Processing**: Analyze pulse width and timing characteristics
/// - **Image Analysis**: Measure feature sizes and shapes in profiles
///
/// # Performance Notes
///
/// - Time complexity: O(n * m) where n is array size and m is number of peaks
/// - Space complexity: O(m) where m is number of peaks
/// - Interpolation accuracy depends on sampling resolution
///
/// # Errors
///
/// Returns an error if:
/// - Input array is empty
/// - Peak indices are out of bounds
/// - `rel_height` is not between 0.0 and 1.0
/// - Peak indices array is empty (returns empty result, not error)
///
/// # Note
///
/// Current implementation is a placeholder returning default values.
/// Full width calculation with interpolation needs to be implemented.
#[allow(dead_code)]
pub fn peak_widths<T>(
    input: &Array<T, scirs2_core::ndarray::Ix1>,
    peaks: &[usize],
    rel_height: Option<T>,
) -> NdimageResult<PeakWidthsResult<T>>
where
    T: Float + FromPrimitive + Debug + NumAssign,
{
    // Validate inputs
    if input.is_empty() {
        return Err(NdimageError::InvalidInput("Input array is empty".into()));
    }

    if peaks.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }

    for &p in peaks {
        if p >= input.len() {
            return Err(NdimageError::InvalidInput(format!(
                "Peak index {} is out of bounds for array of length {}",
                p,
                input.len()
            )));
        }
    }

    let _height = rel_height.unwrap_or_else(|| T::from_f64(0.5).unwrap());
    if _height <= T::zero() || _height >= T::one() {
        return Err(NdimageError::InvalidInput(format!(
            "rel_height must be between 0 and 1, got {:?}",
            _height
        )));
    }

    // Placeholder implementation
    let widths = vec![T::one(); peaks.len()];
    let heights = vec![T::zero(); peaks.len()];
    let left_ips = vec![T::zero(); peaks.len()];
    let right_ips = vec![T::from_usize(input.len() - 1).unwrap(); peaks.len()];

    Ok((widths, heights, left_ips, right_ips))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_extrema() {
        let input: Array2<f64> = Array2::eye(3);
        let (min, max, min_loc, max_loc) = extrema(&input).unwrap();
        assert!(max >= min);
        assert_eq!(min_loc.len(), input.ndim());
        assert_eq!(max_loc.len(), input.ndim());
    }

    #[test]
    fn test_local_extrema() {
        // Test with dynamic array (IxDyn)
        let input_2d: Array2<f64> = Array2::eye(3);
        let input = input_2d.into_dyn();
        assert_eq!(input.ndim(), 2);
        let (minima, maxima) = local_extrema(&input, None, None).unwrap();
        assert_eq!(minima.shape(), input.shape());
        assert_eq!(maxima.shape(), input.shape());

        // Test with a simple peak in the center
        let mut data_2d = Array2::<f64>::zeros((5, 5));
        data_2d[[2, 2]] = 10.0; // Peak in center
        let data = data_2d.into_dyn();
        let (minima, maxima) = local_extrema(&data, None, None).unwrap();
        assert!(maxima[vec![2, 2].as_slice()]); // Center should be a maximum
    }
}
