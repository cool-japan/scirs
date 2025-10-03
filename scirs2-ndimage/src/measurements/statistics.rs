//! Statistical measurement functions for labeled arrays

use scirs2_core::ndarray::{Array, Array1, Dimension};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};

/// Helper function for safe usize conversion
#[allow(dead_code)]
fn safe_usize_to_float<T: Float + FromPrimitive>(value: usize) -> NdimageResult<T> {
    T::from_usize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert usize {} to float type", value))
    })
}

/// Sum the values of an array for each labeled region
///
/// This function computes the sum of pixel values within each labeled region,
/// which is useful for analyzing connected components or regions of interest.
/// Background pixels (label 0) are automatically excluded from calculations.
///
/// # Arguments
///
/// * `input` - Input array containing values to sum
/// * `labels` - Label array defining regions (same shape as input)
/// * `index` - Specific labels to include (if None, includes all non-zero labels)
///
/// # Returns
///
/// * `Result<Array1<T>>` - Array containing sum of values for each region, indexed by label order
///
/// # Examples
///
/// ## Basic usage with 2D arrays
/// ```
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::sum_labels;
///
/// // Create input image and corresponding labels
/// let image = array![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0]
/// ];
///
/// let labels = array![
///     [1, 1, 2],
///     [1, 2, 2],
///     [3, 3, 3]
/// ];
///
/// let sums = sum_labels(&image, &labels, None).unwrap();
/// // sums[0] = sum of region 1 = 1+2+4 = 7
/// // sums[1] = sum of region 2 = 3+5+6 = 14  
/// // sums[2] = sum of region 3 = 7+8+9 = 24
/// ```
///
/// ## Computing specific region sums
/// ```
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::sum_labels;
///
/// let values = array![
///     [10.0, 20.0, 30.0],
///     [40.0, 50.0, 60.0]
/// ];
///
/// let regions = array![
///     [1, 2, 3],
///     [1, 2, 3]
/// ];
///
/// // Only compute sums for regions 1 and 3
/// let partial_sums = sum_labels(&values, &regions, Some(&[1, 3])).unwrap();
/// // Returns sums only for the specified regions
/// assert_eq!(partial_sums.len(), 2);
/// ```
///
/// ## Processing segmented image data
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::sum_labels;
///
/// // Simulate segmented image with intensity values
/// let intensityimage = Array2::from_shape_fn((100, 100), |(i, j)| {
///     ((i + j) as f64).sin().abs() * 255.0
/// });
///
/// // Simulate segmentation labels (e.g., from watershed)
/// let segmentation = Array2::from_shape_fn((100, 100), |(i, j)| {
///     ((i / 10) * 10 + (j / 10)) + 1  // Create grid-like segments
/// });
///
/// let total_intensities = sum_labels(&intensityimage, &segmentation, None).unwrap();
/// // Each element contains total intensity for that segment
/// ```
#[allow(dead_code)]
pub fn sum_labels<T, D>(
    input: &Array<T, D>,
    labels: &Array<usize, D>,
    index: Option<&[usize]>,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + NumAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if labels.shape() != input.shape() {
        return Err(NdimageError::DimensionError(
            "Labels array must have same shape as input array".to_string(),
        ));
    }

    // Find unique labels
    let unique_labels: std::collections::HashSet<usize> = if let Some(idx) = index {
        idx.iter().cloned().collect()
    } else {
        labels.iter().cloned().collect()
    };

    let mut sorted_labels: Vec<usize> = unique_labels.into_iter().collect();
    sorted_labels.sort();

    // Remove label 0 if it exists (typically background)
    if sorted_labels.first() == Some(&0) {
        sorted_labels.remove(0);
    }

    if sorted_labels.is_empty() {
        return Ok(Array1::<T>::zeros(0));
    }

    // Create a mapping from label to index
    let label_to_idx: std::collections::HashMap<usize, usize> = sorted_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    // Initialize sums array
    let mut sums = vec![T::zero(); sorted_labels.len()];

    // Sum values for each label
    for (input_val, label_val) in input.iter().zip(labels.iter()) {
        if let Some(&idx) = label_to_idx.get(label_val) {
            sums[idx] += *input_val;
        }
    }

    Ok(Array1::from_vec(sums))
}

/// Calculate the mean (average) value for each labeled region
///
/// This function computes the arithmetic mean of pixel values within each labeled region.
/// It's fundamental for analyzing region characteristics and is commonly used in
/// image segmentation, object analysis, and feature extraction workflows.
///
/// # Arguments
///
/// * `input` - Input array containing values to average
/// * `labels` - Label array defining regions (same shape as input)
/// * `index` - Specific labels to include (if None, includes all non-zero labels)
///
/// # Returns
///
/// * `Result<Array1<T>>` - Array containing mean value for each region, ordered by label
///
/// # Examples
///
/// ## Basic mean calculation for segmented image
/// ```rust
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::mean_labels;
///
/// // Image with different intensity regions
/// let image = array![
///     [10.0, 20.0, 100.0],
///     [15.0, 25.0, 110.0],
///     [80.0, 85.0, 90.0]
/// ];
///
/// let regions = array![
///     [1, 1, 2],      // Region 1: bright spot
///     [1, 1, 2],      // Region 2: very bright area  
///     [3, 3, 3]       // Region 3: medium bright area
/// ];
///
/// let means = mean_labels(&image, &regions, None).unwrap();
/// // means[0] = mean of region 1 = (10+20+15+25)/4 = 17.5
/// // means[1] = mean of region 2 = (100+110)/2 = 105.0
/// // means[2] = mean of region 3 = (80+85+90)/3 = 85.0
/// ```
///
/// ## Region-based intensity analysis
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::mean_labels;
///
/// // Simulate medical image with different tissue types
/// let intensity_map = Array2::from_shape_fn((50, 50), |(i, j)| {
///     match (i / 10, j / 10) {
///         (0..=1, 0..=1) => 50.0,   // Tissue type A
///         (2..=3, 2..=3) => 150.0,  // Tissue type B  
///         _ => 100.0,               // Background tissue
///     }
/// });
///
/// let tissue_labels = Array2::from_shape_fn((50, 50), |(i, j)| {
///     match (i / 10, j / 10) {
///         (0..=1, 0..=1) => 1,  // Label 1: Type A
///         (2..=3, 2..=3) => 2,  // Label 2: Type B
///         _ => 3,               // Label 3: Background
///     }
/// });
///
/// let tissue_means = mean_labels(&intensity_map, &tissue_labels, None).unwrap();
/// // Analyze mean intensities of different tissue types
/// for (i, &mean_intensity) in tissue_means.iter().enumerate() {
///     println!("Tissue type {}: mean intensity = {:.1}", i+1, mean_intensity);
/// }
/// ```
///
/// ## Quality control: analyzing specific regions only
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::measurements::mean_labels;
///
/// let values = array![
///     [1.0, 2.0, 3.0, 4.0],
///     [5.0, 6.0, 7.0, 8.0],
///     [9.0, 10.0, 11.0, 12.0]
/// ];
///
/// let labels = array![
///     [1, 2, 3, 4],
///     [1, 2, 3, 4],
///     [1, 2, 3, 4]
/// ];
///
/// // Only analyze regions 2 and 4
/// let selected_means = mean_labels(&values, &labels, Some(&[2, 4])).unwrap();
/// // Returns means only for regions 2 and 4
/// assert_eq!(selected_means.len(), 2);
/// ```
///
/// ## Workflow: segmentation → feature extraction
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::mean_labels;
///
/// // Simulate workflow after watershed segmentation
/// let originalimage = Array2::<f64>::from_shape_fn((20, 20), |(i, j)| {
///     ((i as f64 - 10.0).powi(2) + (j as f64 - 10.0).powi(2)).sqrt()
/// });
///
/// let segmentation_result = Array2::from_shape_fn((20, 20), |(i, j)| {
///     if i < 10 && j < 10 { 1 }
///     else if i >= 10 && j < 10 { 2 }
///     else if i < 10 && j >= 10 { 3 }
///     else { 4 }
/// });
///
/// // Extract mean intensity for each segment
/// let segmentfeatures = mean_labels(&originalimage, &segmentation_result, None).unwrap();
///
/// // Filter segments by mean intensity threshold
/// let bright_segments: Vec<usize> = segmentfeatures.iter()
///     .enumerate()
///     .filter(|(_, &mean)| mean > 10.0)
///     .map(|(i_, _)| i_ + 1)  // Convert to 1-based labeling
///     .collect();
/// ```
///
/// # Applications
///
/// - **Medical Imaging**: Analyze tissue intensity characteristics
/// - **Materials Science**: Measure average properties in different phases
/// - **Quality Control**: Check uniformity within regions of interest
/// - **Feature Extraction**: Create descriptive statistics for machine learning
/// - **Image Segmentation**: Validate segmentation quality by intensity homogeneity
///
/// # Performance Notes
///
/// - Time complexity: O(n) where n is the total number of pixels
/// - Space complexity: O(r) where r is the number of unique regions
/// - Computation involves two passes: sum calculation then division by count
/// - For very large images, consider chunked processing
///
/// # Related Functions
///
/// - [`sum_labels`]: Get total values instead of averages
/// - [`variance_labels`]: Measure spread within regions
/// - [`count_labels`]: Get region sizes
///
/// # Errors
///
/// Returns an error if:
/// - Input array is 0-dimensional
/// - Input and labels arrays have different shapes
/// - Region has zero pixels (should not occur with valid labels)
#[allow(dead_code)]
pub fn mean_labels<T, D>(
    input: &Array<T, D>,
    labels: &Array<usize, D>,
    index: Option<&[usize]>,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + NumAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if labels.shape() != input.shape() {
        return Err(NdimageError::DimensionError(
            "Labels array must have same shape as input array".to_string(),
        ));
    }

    // Get sums and counts for each label
    let sums = sum_labels(input, labels, index)?;
    let counts = count_labels(labels, index)?;

    if sums.len() != counts.len() {
        return Err(NdimageError::InvalidInput(
            "Mismatch between sums and counts arrays".into(),
        ));
    }

    // Calculate means (sum / count for each label)
    let means: Vec<T> = sums
        .iter()
        .zip(counts.iter())
        .map(|(&sum, &count)| {
            if count > 0 {
                sum / safe_usize_to_float(count).unwrap_or(T::one())
            } else {
                T::zero()
            }
        })
        .collect();

    Ok(Array1::from_vec(means))
}

/// Calculate the variance of an array for each labeled region
///
/// Computes the sample variance for pixel values within each labeled region.
/// Variance measures the spread of values around the mean, useful for analyzing
/// region homogeneity and texture properties.
///
/// # Arguments
///
/// * `input` - Input array containing values to analyze
/// * `labels` - Label array defining regions (same shape as input)  
/// * `index` - Specific labels to include (if None, includes all non-zero labels)
///
/// # Returns
///
/// * `Result<Array1<T>>` - Array containing variance for each region
///
/// # Examples
///
/// ## Basic variance calculation
/// ```
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::variance_labels;
///
/// let values = array![
///     [1.0, 2.0, 10.0],
///     [1.5, 2.5, 10.5],
///     [5.0, 5.0, 5.0]
/// ];
///
/// let regions = array![
///     [1, 1, 2],
///     [1, 1, 2],
///     [3, 3, 3]
/// ];
///
/// let variances = variance_labels(&values, &regions, None).unwrap();
/// // Region 1: variance of [1.0, 2.0, 1.5, 2.5]
/// // Region 2: variance of [10.0, 10.5]
/// // Region 3: variance of [5.0, 5.0, 5.0] = 0.0 (no variation)
/// ```
#[allow(dead_code)]
pub fn variance_labels<T, D>(
    input: &Array<T, D>,
    labels: &Array<usize, D>,
    index: Option<&[usize]>,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + NumAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if labels.shape() != input.shape() {
        return Err(NdimageError::DimensionError(
            "Labels array must have same shape as input array".to_string(),
        ));
    }

    // Find unique labels
    let unique_labels: std::collections::HashSet<usize> = if let Some(idx) = index {
        idx.iter().cloned().collect()
    } else {
        labels.iter().cloned().collect()
    };

    let mut sorted_labels: Vec<usize> = unique_labels.into_iter().collect();
    sorted_labels.sort();

    // Remove label 0 if it exists (typically background)
    if sorted_labels.first() == Some(&0) {
        sorted_labels.remove(0);
    }

    if sorted_labels.is_empty() {
        return Ok(Array1::<T>::zeros(0));
    }

    // Create a mapping from label to index
    let label_to_idx: std::collections::HashMap<usize, usize> = sorted_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    // First pass: calculate means
    let mut sums = vec![T::zero(); sorted_labels.len()];
    let mut counts = vec![0usize; sorted_labels.len()];

    for (input_val, label_val) in input.iter().zip(labels.iter()) {
        if let Some(&idx) = label_to_idx.get(label_val) {
            sums[idx] += *input_val;
            counts[idx] += 1;
        }
    }

    // Calculate means
    let means: Vec<T> = sums
        .iter()
        .zip(&counts)
        .map(|(&sum, &count)| {
            if count > 0 {
                sum / safe_usize_to_float(count).unwrap_or(T::one())
            } else {
                T::zero()
            }
        })
        .collect();

    // Second pass: calculate variances
    let mut variance_sums = vec![T::zero(); sorted_labels.len()];

    for (input_val, label_val) in input.iter().zip(labels.iter()) {
        if let Some(&idx) = label_to_idx.get(label_val) {
            let diff = *input_val - means[idx];
            variance_sums[idx] += diff * diff;
        }
    }

    // Calculate sample variances (divide by n-1)
    let variances: Vec<T> = variance_sums
        .iter()
        .zip(&counts)
        .map(|(&var_sum, &count)| {
            if count > 1 {
                var_sum / safe_usize_to_float(count - 1).unwrap_or(T::one())
            } else {
                T::zero() // Single pixel regions have zero variance
            }
        })
        .collect();

    Ok(Array1::from_vec(variances))
}

/// Count the number of pixels/elements in each labeled region
///
/// This function computes the size (area in 2D, volume in 3D) of each labeled region
/// by counting the number of pixels/elements belonging to each label. Region size is
/// a fundamental geometric property used for filtering, analysis, and feature extraction.
///
/// # Arguments
///
/// * `labels` - Label array defining regions where each unique value represents a region
/// * `index` - Specific labels to count (if None, counts all non-zero labels)
///
/// # Returns
///
/// * `Result<Array1<usize>>` - Array containing pixel count for each region, ordered by label
///
/// # Examples
///
/// ## Basic region size measurement
/// ```rust
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::count_labels;
///
/// let segmentation = array![
///     [1, 1, 2, 2],
///     [1, 0, 2, 3],  // 0 = background (excluded)
///     [4, 4, 4, 3]
/// ];
///
/// let counts = count_labels(&segmentation, None).unwrap();
/// // counts[0] = 3 pixels in region 1
/// // counts[1] = 3 pixels in region 2  
/// // counts[2] = 2 pixels in region 3
/// // counts[3] = 3 pixels in region 4
/// // Background (label 0) is automatically excluded
/// ```
///
/// ## Object size filtering workflow
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::count_labels;
///
/// // Simulate segmented image with objects of various sizes
/// let labels = Array2::from_shape_fn((100, 100), |(i, j)| {
///     match (i / 20, j / 20) {
///         (0, 0) => 1,  // Large object
///         (1, 1) => 2,  // Medium object  
///         (2, 2) => 3,  // Small object
///         (3, 3) => 4,  // Tiny object
///         _ => 0,       // Background
///     }
/// });
///
/// let object_sizes = count_labels(&labels, None).unwrap();
///
/// // Filter objects by minimum size (remove small noise)
/// let min_size = 100;
/// let large_objects: Vec<usize> = object_sizes.iter()
///     .enumerate()
///     .filter(|(_, &size)| size >= min_size)
///     .map(|(i_, _)| i_ + 1)  // Convert to 1-based label
///     .collect();
///
/// println!("Large objects (>= {} pixels): {:?}", min_size, large_objects);
/// ```
///
/// ## Cell counting and size analysis
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::measurements::count_labels;
///
/// // Simulated cell segmentation result
/// let cell_labels = array![
///     [1, 1, 1, 0, 2, 2],
///     [1, 1, 1, 0, 2, 2],
///     [1, 1, 0, 0, 0, 0],
///     [0, 0, 0, 3, 3, 3],
///     [4, 4, 5, 3, 3, 3],
///     [4, 4, 5, 5, 5, 5]
/// ];
///
/// let cell_sizes = count_labels(&cell_labels, None).unwrap();
///
/// // Analyze cell size distribution
/// let total_cells = cell_sizes.len();
/// let total_area: usize = cell_sizes.iter().sum();
/// let average_size = total_area as f64 / total_cells as f64;
///
/// println!("Found {} cells", total_cells);
/// println!("Average cell size: {:.1} pixels", average_size);
///
/// // Identify unusually large or small cells
/// for (i, &size) in cell_sizes.iter().enumerate() {
///     let cell_id = i + 1;
///     if size < 3 {
///         println!("Cell {} is very small ({} pixels)", cell_id, size);
///     } else if size > 10 {
///         println!("Cell {} is very large ({} pixels)", cell_id, size);
///     }
/// }
/// ```
///
/// ## Quality control: count specific regions only
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::measurements::count_labels;
///
/// let labels = array![
///     [1, 2, 3, 4, 5],
///     [1, 2, 3, 4, 5],
///     [1, 2, 3, 4, 5]
/// ];
///
/// // Only count regions 2, 4, and 5
/// let selected_counts = count_labels(&labels, Some(&[2, 4, 5])).unwrap();
/// assert_eq!(selected_counts.len(), 3);
/// // Each selected region has 3 pixels
/// assert_eq!(selected_counts[0], 3); // Region 2
/// assert_eq!(selected_counts[1], 3); // Region 4  
/// assert_eq!(selected_counts[2], 3); // Region 5
/// ```
///
/// ## Segmentation validation workflow
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::count_labels;
///
/// // After watershed or other segmentation algorithm
/// let segmentation_result = Array2::from_shape_fn((50, 50), |(i, j)| {
///     // Simulate segmentation with regions of different sizes
///     if i < 25 && j < 25 { 1 }
///     else if i >= 25 && j < 25 { 2 }
///     else if i < 25 && j >= 25 { 3 }
///     else { 4 }
/// });
///
/// let region_sizes = count_labels(&segmentation_result, None).unwrap();
///
/// // Validation: check for reasonable region sizes
/// let image_area = 50 * 50;
/// let total_segmented: usize = region_sizes.iter().sum();
///
/// println!("Segmentation coverage: {:.1}%",
///          100.0 * total_segmented as f64 / image_area as f64);
///
/// // Check for undersegmentation (too few, too large regions)
/// let oversized_threshold = image_area / 10;
/// let oversized_regions = region_sizes.iter()
///     .filter(|&&size| size > oversized_threshold)
///     .count();
///
/// if oversized_regions > 0 {
///     println!("Warning: {} oversized regions detected", oversized_regions);
/// }
/// ```
///
/// # Applications
///
/// - **Object Detection**: Count and size objects in segmented images
/// - **Cell Biology**: Measure cell sizes and count cell populations
/// - **Quality Control**: Filter regions by size criteria
/// - **Materials Science**: Analyze particle size distributions
/// - **Medical Imaging**: Measure lesion or organ sizes
/// - **Segmentation Validation**: Check segmentation quality
///
/// # Performance Notes
///
/// - Time complexity: O(n) where n is the total number of pixels
/// - Space complexity: O(r) where r is the number of unique regions
/// - Very efficient single-pass algorithm
/// - For memory-critical applications, consider processing in chunks
///
/// # Related Functions
///
/// - [`sum_labels`]: Get total intensity values per region
/// - [`mean_labels`]: Get average intensity per region  
/// - [`variance_labels`]: Measure intensity variation within regions
///
/// # Errors
///
/// Returns an error if:
/// - Label array is 0-dimensional
/// - No valid labels found (returns empty array, not error)
#[allow(dead_code)]
pub fn count_labels<D>(
    labels: &Array<usize, D>,
    index: Option<&[usize]>,
) -> NdimageResult<Array1<usize>>
where
    D: Dimension,
{
    // Validate inputs
    if labels.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Labels array cannot be 0-dimensional".into(),
        ));
    }

    // Find unique labels
    let unique_labels: std::collections::HashSet<usize> = if let Some(idx) = index {
        idx.iter().cloned().collect()
    } else {
        labels.iter().cloned().collect()
    };

    let mut sorted_labels: Vec<usize> = unique_labels.into_iter().collect();
    sorted_labels.sort();

    // Remove label 0 if it exists (typically background)
    if sorted_labels.first() == Some(&0) {
        sorted_labels.remove(0);
    }

    if sorted_labels.is_empty() {
        return Ok(Array1::<usize>::zeros(0));
    }

    // Create a mapping from label to index
    let label_to_idx: std::collections::HashMap<usize, usize> = sorted_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    // Initialize counts array
    let mut counts = vec![0usize; sorted_labels.len()];

    // Count occurrences of each label
    for &label_val in labels.iter() {
        if let Some(&idx) = label_to_idx.get(&label_val) {
            counts[idx] += 1;
        }
    }

    Ok(Array1::from_vec(counts))
}

/// Calculate histogram of values within labeled regions
///
/// This function computes histograms of pixel values within specified labeled regions.
/// It's useful for analyzing intensity distributions, texture characteristics, and
/// statistical properties of different regions in segmented images.
///
/// # Arguments
///
/// * `input` - Input array containing values to histogram
/// * `min` - Minimum value of histogram range (values below this are ignored)
/// * `max` - Maximum value of histogram range (values above this are ignored)
/// * `bins` - Number of histogram bins to create
/// * `labels` - Label array defining regions (if None, uses entire input array)
/// * `index` - Specific labels to include (if None, includes all non-zero labels)
///
/// # Returns
///
/// * `Result<(Array1<usize>, Array1<T>)>` - Tuple containing:
///   - Histogram counts for each bin
///   - Bin edges (length = bins + 1)
///
/// # Examples
///
/// ## Basic histogram of entire image
/// ```rust
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::histogram;
///
/// let image = array![
///     [0.0, 0.5, 1.0],
///     [0.2, 0.8, 1.2],
///     [0.4, 0.6, 0.9]
/// ];
///
/// // Create histogram with 5 bins from 0.0 to 1.0
/// let (counts, edges) = histogram(&image, 0.0, 1.0, 5, None, None).unwrap();
///
/// // edges will be [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
/// // counts shows how many pixels fall in each bin
/// println!("Histogram counts: {:?}", counts);
/// println!("Bin edges: {:?}", edges);
/// ```
///
/// ## Histogram of specific regions only
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::measurements::histogram;
///
/// let intensities = array![
///     [10.0, 20.0, 100.0, 110.0],
///     [15.0, 25.0, 105.0, 115.0],
///     [12.0, 22.0, 102.0, 112.0]
/// ];
///
/// let regions = array![
///     [1, 1, 2, 2],
///     [1, 1, 2, 2],
///     [1, 1, 2, 2]
/// ];
///
/// // Histogram only region 1 (low intensity values)
/// let (hist_region1, edges1) = histogram(&intensities, 10.0, 30.0, 4,
///                                        Some(&regions), Some(&[1])).unwrap();
///
/// // Histogram only region 2 (high intensity values)  
/// let (hist_region2, edges2) = histogram(&intensities, 100.0, 120.0, 4,
///                                        Some(&regions), Some(&[2])).unwrap();
/// ```
///
/// ## Texture analysis workflow
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::histogram;
///
/// // Simulate textured image regions
/// let textureimage = Array2::from_shape_fn((50, 50), |(i, j)| {
///     let noise = ((i * 7 + j * 11) % 100) as f64 / 100.0;
///     match (i / 25, j / 25) {
///         (0, 0) => 0.2 + 0.1 * noise,  // Low intensity, low variation
///         (0, 1) => 0.8 + 0.1 * noise,  // High intensity, low variation
///         (1, 0) => 0.4 + 0.3 * noise,  // Medium intensity, high variation
///         _ => 0.6 + 0.2 * noise,       // Medium intensity, medium variation
///     }
/// });
///
/// let texture_labels = Array2::from_shape_fn((50, 50), |(i, j)| {
///     match (i / 25, j / 25) {
///         (0, 0) => 1,
///         (0, 1) => 2,
///         (1, 0) => 3,
///         _ => 4,
///     }
/// });
///
/// // Analyze histogram characteristics of each texture region
/// for region_id in 1..=4 {
///     let (hist_counts, _hist_edges) = histogram(&textureimage, 0.0, 1.0, 10,
///                               Some(&texture_labels), Some(&[region_id])).unwrap();
///
///     // Calculate histogram statistics
///     let total_pixels: usize = hist_counts.iter().sum();
///     let entropy = -hist_counts.iter()
///         .map(|&count| {
///             if count > 0 {
///                 let p = count as f64 / total_pixels as f64;
///                 p * p.log2()
///             } else {
///                 0.0
///             }
///         })
///         .sum::<f64>();
///     
///     println!("Region {}: entropy = {:.3}", region_id, entropy);
/// }
/// ```
///
/// ## Medical image intensity analysis
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::histogram;
///
/// // Simulate medical image with different tissue types
/// let medicalimage = Array2::from_shape_fn((100, 100), |(i, j)| {
///     match (i / 33, j / 33) {
///         (0, _) => 50.0 + (i % 10) as f64,   // Tissue A: 50-60 range
///         (1, _) => 150.0 + (j % 15) as f64,  // Tissue B: 150-165 range
///         _ => 100.0 + ((i + j) % 20) as f64, // Tissue C: 100-120 range
///     }
/// });
///
/// let tissue_segmentation = Array2::from_shape_fn((100, 100), |(i, j)| {
///     match (i / 33, j / 33) {
///         (0, _) => 1,  // Tissue A
///         (1, _) => 2,  // Tissue B
///         _ => 3,       // Tissue C
///     }
/// });
///
/// // Analyze intensity distribution for each tissue type
/// for tissue_type in 1..=3 {
///     let (tissue_hist, bin_edges) = histogram(&medicalimage, 0.0, 200.0, 20,
///                                              Some(&tissue_segmentation),
///                                              Some(&[tissue_type])).unwrap();
///     
///     // Find peak intensity for this tissue type
///     let peak_bin = tissue_hist.iter()
///         .enumerate()
///         .max_by_key(|(_, &count)| count)
///         .map(|(i_, _)| i_)
///         .unwrap_or(0);
///     
///     let peak_intensity = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0;
///     println!("Tissue {}: peak intensity = {:.1}", tissue_type, peak_intensity);
/// }
/// ```
///
/// ## Quality control: histogram-based thresholding
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::histogram;
///
/// let image = Array2::from_shape_fn((50, 50), |(i, j)| {
///     let di = i as i32 - 25;
///     let dj = j as i32 - 25;
///     if di * di + dj * dj < 400 {
///         200.0  // Bright circular object
///     } else {
///         50.0   // Dark background
///     }
/// });
///
/// // Calculate histogram to find optimal threshold
/// let (hist, edges) = histogram(&image, 0.0, 255.0, 25, None, None).unwrap();
///
/// // Find valley between peaks (simplified Otsu-like approach)
/// let peak1 = hist.iter().take(10).enumerate().max_by_key(|(_, &v)| v).unwrap().0;
/// let peak2 = hist.iter().skip(15).enumerate().max_by_key(|(_, &v)| v).unwrap().0 + 15;
///
/// // Threshold is approximately between the peaks
/// let threshold = (edges[peak1] + edges[peak2]) / 2.0;
/// println!("Suggested threshold: {:.1}", threshold);
/// ```
///
/// # Applications
///
/// - **Medical Imaging**: Analyze tissue intensity distributions
/// - **Materials Science**: Study grain intensity characteristics
/// - **Quality Control**: Detect anomalies in intensity patterns
/// - **Texture Analysis**: Characterize surface textures
/// - **Image Enhancement**: Guide contrast adjustment algorithms
/// - **Segmentation**: Determine optimal thresholds
///
/// # Performance Notes
///
/// - Time complexity: O(n) where n is the number of pixels
/// - Space complexity: O(b) where b is the number of bins
/// - For large images, consider processing regions separately
/// - Bin count affects both precision and computation speed
///
/// # Implementation Note
///
/// Current implementation is a placeholder and returns empty histogram.
/// Full histogram calculation needs to be implemented with proper binning logic.
///
/// # Errors
///
/// Returns an error if:
/// - Input array is 0-dimensional
/// - min >= max (invalid range)
/// - bins = 0 (invalid bin count)
/// - Labels array shape doesn't match input array shape
#[allow(dead_code)]
pub fn histogram<T, D>(
    input: &Array<T, D>,
    min: T,
    max: T,
    bins: usize,
    labels: Option<&Array<usize, D>>,
    _index: Option<&[usize]>,
) -> NdimageResult<(Array1<usize>, Array1<T>)>
where
    T: Float + FromPrimitive + Debug + NumAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if min >= max {
        return Err(NdimageError::InvalidInput(format!(
            "min must be less than max (got min={:?}, max={:?})",
            min, max
        )));
    }

    if bins == 0 {
        return Err(NdimageError::InvalidInput(
            "bins must be greater than 0".into(),
        ));
    }

    if let Some(lab) = labels {
        if lab.shape() != input.shape() {
            return Err(NdimageError::DimensionError(
                "Labels array must have same shape as input array".to_string(),
            ));
        }
    }

    // Create bin edges
    let bin_width = (max - min) / T::from_usize(bins).unwrap();
    let mut edges = Array1::<T>::zeros(bins + 1);
    for i in 0..=bins {
        edges[i] = min + T::from_usize(i).unwrap() * bin_width;
    }

    // Initialize histogram
    let mut hist = Array1::<usize>::zeros(bins);

    // Compute histogram
    match labels {
        None => {
            // Process all values in the array
            for &value in input.iter() {
                if value >= min && value < max {
                    let bin_idx = ((value - min) / bin_width).to_usize().unwrap_or(0);
                    let bin_idx = bin_idx.min(bins - 1); // Ensure we don't exceed bounds
                    hist[bin_idx] += 1;
                } else if value == max {
                    // Special case for values exactly at max
                    hist[bins - 1] += 1;
                }
            }
        }
        Some(label_array) => {
            // Process only values with specific labels
            for (value, &label) in input.iter().zip(label_array.iter()) {
                // Skip background (label 0) if not explicitly requested
                if label > 0 && *value >= min && *value < max {
                    let bin_idx = ((*value - min) / bin_width).to_usize().unwrap_or(0);
                    let bin_idx = bin_idx.min(bins - 1);
                    hist[bin_idx] += 1;
                } else if label > 0 && *value == max {
                    hist[bins - 1] += 1;
                }
            }
        }
    }

    Ok((hist, edges))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2, Array3};

    #[test]
    fn test_sum_labels_basic() {
        let input: Array2<f64> = Array2::eye(3);
        let labels: Array2<usize> = Array2::from_elem((3, 3), 1);
        let result =
            sum_labels(&input, &labels, None).expect("sum_labels should succeed for basic test");

        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0], 3.0, epsilon = 1e-10); // Sum of diagonal elements
    }

    #[test]
    fn test_sum_labels_multiple_regions() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let labels = array![[1, 1, 2], [1, 2, 2], [3, 3, 3]];

        let sums = sum_labels(&input, &labels, None)
            .expect("sum_labels should succeed for multiple regions test");
        assert_eq!(sums.len(), 3);
        assert_abs_diff_eq!(sums[0], 1.0 + 2.0 + 4.0, epsilon = 1e-10); // Region 1
        assert_abs_diff_eq!(sums[1], 3.0 + 5.0 + 6.0, epsilon = 1e-10); // Region 2
        assert_abs_diff_eq!(sums[2], 7.0 + 8.0 + 9.0, epsilon = 1e-10); // Region 3
    }

    #[test]
    fn test_sum_labels_with_background() {
        let input = array![[1.0, 2.0], [3.0, 4.0]];
        let labels = array![[0, 1], [1, 2]]; // Label 0 is background

        let sums = sum_labels(&input, &labels, None)
            .expect("sum_labels should succeed with background test");
        assert_eq!(sums.len(), 2); // Should exclude background (label 0)
        assert_abs_diff_eq!(sums[0], 2.0 + 3.0, epsilon = 1e-10); // Region 1
        assert_abs_diff_eq!(sums[1], 4.0, epsilon = 1e-10); // Region 2
    }

    #[test]
    fn test_sum_labels_selective_index() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let labels = array![[1, 2, 3], [1, 2, 3]];

        let sums = sum_labels(&input, &labels, Some(&[1, 3]))
            .expect("sum_labels should succeed with selective index test");
        assert_eq!(sums.len(), 2); // Only regions 1 and 3
        assert_abs_diff_eq!(sums[0], 1.0 + 4.0, epsilon = 1e-10); // Region 1
        assert_abs_diff_eq!(sums[1], 3.0 + 6.0, epsilon = 1e-10); // Region 3
    }

    #[test]
    fn test_sum_labels_edge_cases() {
        // Empty result case
        let input = array![[1.0, 2.0]];
        let labels = array![[0, 0]]; // All background
        let sums = sum_labels(&input, &labels, None)
            .expect("sum_labels should succeed for empty result test");
        assert_eq!(sums.len(), 0);

        // Single pixel regions
        let input2 = array![[1.0, 2.0, 3.0]];
        let labels2 = array![[1, 2, 3]];
        let sums2 = sum_labels(&input2, &labels2, None)
            .expect("sum_labels should succeed for single pixel test");
        assert_eq!(sums2.len(), 3);
        assert_abs_diff_eq!(sums2[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sums2[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sums2[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sum_labels_3d() {
        let input = Array3::from_shape_fn((2, 2, 2), |(i, j, k)| (i + j + k) as f64);
        let labels = Array3::from_shape_fn((2, 2, 2), |(i, j, _k)| if i == j { 1 } else { 2 });

        let sums =
            sum_labels(&input, &labels, None).expect("sum_labels should succeed for 3D test");
        assert_eq!(sums.len(), 2);
        assert!(sums[0] > 0.0);
        assert!(sums[1] > 0.0);
    }

    #[test]
    fn test_mean_labels_basic() {
        let input: Array2<f64> = Array2::eye(3);
        let labels: Array2<usize> = Array2::from_elem((3, 3), 1);
        let result =
            mean_labels(&input, &labels, None).expect("mean_labels should succeed for basic test");

        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0], 3.0 / 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_labels_multiple_regions() {
        let input = array![[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]];
        let labels = array![[1, 1, 2], [1, 2, 2]];

        let means = mean_labels(&input, &labels, None)
            .expect("mean_labels should succeed for multiple regions test");
        assert_eq!(means.len(), 2);
        assert_abs_diff_eq!(means[0], (2.0 + 4.0 + 8.0) / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(means[1], (6.0 + 10.0 + 12.0) / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_variance_labels_basic() {
        let input = array![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
        let labels = array![[1, 1, 2], [1, 2, 2]];

        let variances = variance_labels(&input, &labels, None)
            .expect("variance_labels should succeed for basic test");
        assert_eq!(variances.len(), 2);

        // Manual variance calculation for region 1: [1, 3, 2]
        // Mean = 2.0, variance = ((1-2)² + (3-2)² + (2-2)²) / (3-1) = (1+1+0)/2 = 1.0
        assert_abs_diff_eq!(variances[0], 1.0, epsilon = 1e-10);

        // Manual variance calculation for region 2: [5, 4, 6]
        // Mean = 5.0, variance = ((5-5)² + (4-5)² + (6-5)²) / (3-1) = (0+1+1)/2 = 1.0
        assert_abs_diff_eq!(variances[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_variance_labels_zero_variance() {
        let input = array![[5.0, 5.0, 3.0], [5.0, 3.0, 3.0]];
        let labels = array![[1, 1, 2], [1, 2, 2]];

        let variances = variance_labels(&input, &labels, None)
            .expect("variance_labels should succeed for zero variance test");
        assert_eq!(variances.len(), 2);
        assert_abs_diff_eq!(variances[0], 0.0, epsilon = 1e-10); // All values are 5.0
        assert_abs_diff_eq!(variances[1], 0.0, epsilon = 1e-10); // All values are 3.0
    }

    #[test]
    fn test_variance_labels_single_pixel() {
        let input = array![[1.0, 2.0, 3.0]];
        let labels = array![[1, 2, 3]]; // Each pixel is its own region

        let variances = variance_labels(&input, &labels, None)
            .expect("variance_labels should succeed for single pixel test");
        assert_eq!(variances.len(), 3);
        // Single pixel regions should have zero variance
        assert_abs_diff_eq!(variances[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(variances[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(variances[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_count_labels_basic() {
        let labels: Array2<usize> = Array2::from_elem((3, 3), 1);
        let result =
            count_labels(&labels, None).expect("count_labels should succeed for basic test");

        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 9); // 3x3 grid with all label 1
    }

    #[test]
    fn test_count_labels_multiple_regions() {
        let labels = array![[1, 1, 2, 2], [1, 3, 3, 2], [4, 4, 4, 4]];

        let counts = count_labels(&labels, None)
            .expect("count_labels should succeed for multiple regions test");
        assert_eq!(counts.len(), 4);
        assert_eq!(counts[0], 3); // Region 1: 3 pixels
        assert_eq!(counts[1], 3); // Region 2: 3 pixels
        assert_eq!(counts[2], 2); // Region 3: 2 pixels
        assert_eq!(counts[3], 4); // Region 4: 4 pixels
    }

    #[test]
    fn test_count_labels_with_background() {
        let labels = array![[0, 1, 1], [0, 2, 2], [0, 0, 3]];

        let counts =
            count_labels(&labels, None).expect("count_labels should succeed with background test");
        assert_eq!(counts.len(), 3); // Should exclude background (label 0)
        assert_eq!(counts[0], 2); // Region 1: 2 pixels
        assert_eq!(counts[1], 2); // Region 2: 2 pixels
        assert_eq!(counts[2], 1); // Region 3: 1 pixel
    }

    #[test]
    fn test_error_handling() {
        // Test dimension mismatch
        let input = array![[1.0, 2.0]];
        let labels = array![[1], [2]]; // Wrong shape

        assert!(sum_labels(&input, &labels, None).is_err());
        assert!(mean_labels(&input, &labels, None).is_err());
        assert!(variance_labels(&input, &labels, None).is_err());

        // Test 0-dimensional input
        let input_0d = scirs2_core::ndarray::arr0(1.0);
        let labels_0d = scirs2_core::ndarray::arr0(1);

        assert!(sum_labels(&input_0d, &labels_0d, None).is_err());
        assert!(mean_labels(&input_0d, &labels_0d, None).is_err());
        assert!(variance_labels(&input_0d, &labels_0d, None).is_err());
        assert!(count_labels(&labels_0d, None).is_err());
    }

    #[test]
    fn test_high_dimensional_arrays() {
        // Test 4D arrays
        let input = Array::from_shape_fn((2, 2, 2, 2), |(i, j, k, l)| (i + j + k + l) as f64);
        let labels = Array::from_shape_fn((2, 2, 2, 2), |(i, j, _k, _l)| i + j + 1);

        let sums =
            sum_labels(&input, &labels, None).expect("sum_labels should succeed for 4D test");
        let means =
            mean_labels(&input, &labels, None).expect("mean_labels should succeed for 4D test");
        let variances = variance_labels(&input, &labels, None)
            .expect("variance_labels should succeed for 4D test");
        let counts = count_labels(&labels, None).expect("count_labels should succeed for 4D test");

        assert!(!sums.is_empty());
        assert!(!means.is_empty());
        assert!(!variances.is_empty());
        assert!(!counts.is_empty());
        assert_eq!(sums.len(), means.len());
        assert_eq!(means.len(), variances.len());
        assert_eq!(variances.len(), counts.len());
    }
}
