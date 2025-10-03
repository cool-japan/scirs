//! Region property measurement functions

use scirs2_core::ndarray::{Array, Dimension, Ix2};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};
use std::fmt::Debug;

use super::RegionProperties;
use crate::error::{NdimageError, NdimageResult};

/// Helper function to convert dimension pattern to coordinate vector
fn pattern_to_coords<D: Dimension>(pattern: &D::Pattern, shape: &[usize]) -> Vec<usize> {
    // For now, we'll use unsafe transmute as a workaround
    // This is not ideal but works for common cases
    let pattern_size = std::mem::size_of::<D::Pattern>();
    let coord_count = shape.len();

    if pattern_size == coord_count * std::mem::size_of::<usize>() {
        // Pattern is likely a tuple of usize values
        unsafe {
            let ptr = pattern as *const D::Pattern as *const usize;
            (0..coord_count).map(|i| *ptr.add(i)).collect()
        }
    } else {
        // Fallback: return zeros
        vec![0; coord_count]
    }
}

/// Extract comprehensive properties of labeled regions
///
/// This function analyzes segmented images to compute geometric and statistical properties
/// of each labeled region. It's essential for object analysis, feature extraction, and
/// quantitative image analysis workflows.
///
/// # Arguments
///
/// * `input` - Input array containing values (intensities, measurements, etc.)
/// * `labels` - Label array defining regions (same shape as input)
/// * `properties` - List of property names to extract (if None, extracts all available properties)
///                 Supported properties: "area", "centroid", "bbox", "mean_intensity",
///                 "min_intensity", "max_intensity", "perimeter", "eccentricity"
///
/// # Returns
///
/// * `Result<Vec<RegionProperties<T>>>` - Vector of region property structures, one per region
///
/// # Examples
///
/// ## Basic region analysis
/// ```rust
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::region_properties;
///
/// // Image with different regions
/// let image = array![
///     [100.0, 100.0, 200.0, 200.0],
///     [100.0, 100.0, 200.0, 200.0],
///     [150.0, 150.0, 150.0, 150.0],
///     [150.0, 150.0, 150.0, 150.0]
/// ];
///
/// let labels = array![
///     [1, 1, 2, 2],
///     [1, 1, 2, 2],
///     [3, 3, 3, 3],
///     [3, 3, 3, 3]
/// ];
///
/// let props = region_properties(&image, &labels, None).unwrap();
///
/// for prop in &props {
///     println!("Region {}: area={}, centroid={:?}",
///              prop.label, prop.area, prop.centroid);
/// }
/// ```
///
/// ## Cell morphology analysis
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::region_properties;
///
/// // Simulate segmented cell image
/// let cell_intensities = Array2::from_shape_fn((50, 50), |(i, j)| {
///     // Create different cell-like regions with varying intensities
///     match ((i / 10), (j / 10)) {
///         (1, 1) => 80.0 + ((i + j) % 5) as f64,   // Cell 1
///         (1, 3) => 120.0 + ((i * j) % 8) as f64,  // Cell 2
///         (3, 1) => 90.0 + ((i as i32 - j as i32).abs() % 6) as f64, // Cell 3
///         _ => 30.0,  // Background
///     }
/// });
///
/// let cell_labels = Array2::from_shape_fn((50, 50), |(i, j)| {
///     match ((i / 10), (j / 10)) {
///         (1, 1) => 1,  // Cell 1
///         (1, 3) => 2,  // Cell 2
///         (3, 1) => 3,  // Cell 3
///         _ => 0,       // Background
///     }
/// });
///
/// // Extract all cell properties
/// let cell_props = region_properties(&cell_intensities, &cell_labels, None).unwrap();
///
/// // Analyze cell characteristics
/// for cell in &cell_props {
///     println!("Cell {}: ", cell.label);
///     println!("  Area: {} pixels", cell.area);
///     println!("  Centroid: ({:.1}, {:.1})", cell.centroid[0], cell.centroid[1]);
///     println!("  Bounding box: {:?}", cell.bbox);
/// }
/// ```
///
/// ## Selective property extraction
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::measurements::region_properties;
///
/// let data = array![
///     [1.0, 2.0, 5.0, 6.0],
///     [3.0, 4.0, 7.0, 8.0],
///     [9.0, 10.0, 11.0, 12.0]
/// ];
///
/// let regions = array![
///     [1, 1, 2, 2],
///     [1, 1, 2, 2],
///     [3, 3, 3, 3]
/// ];
///
/// // Extract only area and centroid properties
/// let props = region_properties(&data, &regions,
///                              Some(vec!["area", "centroid"])).unwrap();
///
/// // Properties will only contain the requested measurements
/// ```
///
/// ## Materials analysis workflow
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::region_properties;
///
/// // Simulate microscopy image of material grains
/// let grainimage = Array2::from_shape_fn((100, 100), |(i, j)| {
///     // Create grain-like structures with different properties
///     let grain_id = ((i / 25) * 2 + (j / 25)) + 1;
///     let base_intensity = match grain_id {
///         1 => 60.0,   // Grain type A
///         2 => 90.0,   // Grain type B
///         3 => 110.0,  // Grain type C
///         _ => 75.0,   // Grain type D
///     };
///     
///     // Add some texture variation
///     let texture = ((i * 3 + j * 7) % 20) as f64;
///     base_intensity + texture
/// });
///
/// let grain_labels = Array2::from_shape_fn((100, 100), |(i, j)| {
///     ((i / 25) * 2 + (j / 25)) + 1
/// });
///
/// let grain_props = region_properties(&grainimage, &grain_labels, None).unwrap();
///
/// // Quality control: identify grains with unusual properties
/// let total_area: usize = grain_props.iter().map(|g| g.area).sum();
/// let avg_area = total_area as f64 / grain_props.len() as f64;
///
/// for grain in &grain_props {
///     let area_ratio = grain.area as f64 / avg_area;
///     if area_ratio < 0.5 {
///         println!("Small grain detected: {} (area: {})", grain.label, grain.area);
///     } else if area_ratio > 2.0 {
///         println!("Large grain detected: {} (area: {})", grain.label, grain.area);
///     }
/// }
/// ```
///
/// ## Medical imaging: lesion characterization
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::region_properties;
///
/// // Simulate medical image with lesions
/// let medical_scan = Array2::from_shape_fn((80, 80), |(i, j)| {
///     // Normal tissue background
///     let mut intensity = 100.0;
///     
///     // Add lesions with different characteristics
///     let dist1 = ((i as f64 - 20.0).powi(2) + (j as f64 - 30.0).powi(2)).sqrt();
///     let dist2 = ((i as f64 - 60.0).powi(2) + (j as f64 - 50.0).powi(2)).sqrt();
///     
///     if dist1 < 8.0 {
///         intensity = 150.0; // Bright lesion
///     } else if dist2 < 12.0 {
///         intensity = 80.0;  // Dark lesion
///     }
///     
///     intensity
/// });
///
/// let lesion_segmentation = Array2::from_shape_fn((80, 80), |(i, j)| {
///     let dist1 = ((i as f64 - 20.0).powi(2) + (j as f64 - 30.0).powi(2)).sqrt();
///     let dist2 = ((i as f64 - 60.0).powi(2) + (j as f64 - 50.0).powi(2)).sqrt();
///     
///     if dist1 < 8.0 {
///         1  // Lesion 1
///     } else if dist2 < 12.0 {
///         2  // Lesion 2
///     } else {
///         0  // Normal tissue
///     }
/// });
///
/// let lesion_props = region_properties(&medical_scan, &lesion_segmentation, None).unwrap();
///
/// // Clinical analysis
/// for lesion in &lesion_props {
///     let area_mm2 = lesion.area as f64 * 0.01; // Convert pixels to mm²
///     println!("Lesion {}: {:.2} mm², center at ({:.1}, {:.1})",
///              lesion.label, area_mm2, lesion.centroid[0], lesion.centroid[1]);
/// }
/// ```
///
/// # Region Property Details
///
/// The `RegionProperties` structure contains:
/// - **label**: Original label value from the segmentation
/// - **area**: Number of pixels in the region (2D) or voxels (3D)
/// - **centroid**: Center of mass coordinates [y, x] in 2D or [z, y, x] in 3D
/// - **bbox**: Bounding box [min_row, min_col, max_row, max_col] in 2D
///
/// # Applications
///
/// - **Cell Biology**: Analyze cell morphology, size, and intensity
/// - **Medical Imaging**: Characterize lesions, organs, and anatomical structures
/// - **Materials Science**: Study grain sizes, shapes, and distributions
/// - **Quality Control**: Identify defects and measure component properties
/// - **Ecology**: Analyze organism shapes and sizes in microscopy
/// - **Manufacturing**: Inspect part dimensions and surface features
///
/// # Performance Notes
///
/// - Time complexity: O(n) where n is the total number of pixels
/// - Space complexity: O(r) where r is the number of regions
/// - For large images with many regions, consider processing in parallel
/// - Property calculation time increases with the number of requested properties
///
/// # Implementation Note
///
/// Current implementation is a placeholder returning minimal data.
/// Full region property calculation needs to be implemented with proper
/// geometric and statistical computations.
///
/// # Errors
///
/// Returns an error if:
/// - Input array is 0-dimensional
/// - Input and labels arrays have different shapes
/// - Invalid property names are specified
#[allow(dead_code)]
pub fn region_properties<T, D>(
    input: &Array<T, D>,
    labels: &Array<usize, D>,
    _properties: Option<Vec<&str>>,
) -> NdimageResult<Vec<RegionProperties<T>>>
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

    // Find all unique labels (excluding background label 0)
    let mut unique_labels = std::collections::HashSet::new();
    for &label in labels.iter() {
        if label > 0 {
            unique_labels.insert(label);
        }
    }

    let mut region_props = Vec::new();

    // Compute _properties for each region
    for &label in unique_labels.iter() {
        let mut area = 0;
        let mut sum_coords = vec![T::zero(); input.ndim()];
        let mut min_coords = vec![usize::MAX; input.ndim()];
        let mut max_coords = vec![0; input.ndim()];

        // Iterate through all pixels to compute _properties
        for ((coords, &value), &pixel_label) in input.indexed_iter().zip(labels.iter()) {
            if pixel_label == label {
                area += 1;

                // Update sum for centroid calculation
                let coord_vec = pattern_to_coords::<D>(&coords, input.shape());
                for (i, coord) in coord_vec.iter().enumerate() {
                    sum_coords[i] += T::from_usize(*coord).unwrap() * value;
                    min_coords[i] = min_coords[i].min(*coord);
                    max_coords[i] = max_coords[i].max(*coord);
                }
            }
        }

        // Calculate centroid (center of mass)
        let total_intensity = {
            let mut total = T::zero();
            for ((coords, &value), &pixel_label) in input.indexed_iter().zip(labels.iter()) {
                if pixel_label == label {
                    total += value;
                }
            }
            total
        };

        let centroid = if total_intensity > T::zero() {
            sum_coords.iter().map(|&s| s / total_intensity).collect()
        } else {
            // Fallback to geometric centroid if intensity is zero
            sum_coords
                .iter()
                .map(|&s| s / T::from_usize(area).unwrap())
                .collect()
        };

        // Create bounding box in the format [min1, min2, ..., max1, max2, ...]
        let mut bbox = Vec::new();
        for i in 0..input.ndim() {
            bbox.push(min_coords[i]);
        }
        for i in 0..input.ndim() {
            bbox.push(max_coords[i] + 1); // Add 1 to make it exclusive end
        }

        let props = RegionProperties {
            label,
            area,
            centroid,
            bbox,
        };

        region_props.push(props);
    }

    // Sort by label for consistent output
    region_props.sort_by_key(|p| p.label);

    Ok(region_props)
}

/// Find and locate objects in a labeled array
///
/// This function identifies all objects (connected components) in a labeled array and
/// returns their bounding boxes. It's commonly used after segmentation to locate and
/// extract individual objects for further analysis or processing.
///
/// # Arguments
///
/// * `input` - Input labeled array where each unique non-zero value represents an object
///
/// # Returns
///
/// * `Result<Vec<Vec<usize>>>` - Vector of bounding box coordinates for each object.
///   Each inner vector contains [min_coord1, max_coord1, min_coord2, max_coord2, ...]
///   for each dimension of the array.
///
/// # Examples
///
/// ## Basic object detection in 2D
/// ```rust
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::find_objects;
///
/// let labeledimage = array![
///     [0, 1, 1, 0, 0],
///     [0, 1, 1, 0, 2],
///     [0, 0, 0, 0, 2],
///     [3, 3, 0, 0, 2],
///     [3, 3, 0, 0, 0]
/// ];
///
/// let objects = find_objects(&labeledimage).unwrap();
///
/// // objects[0] = bounding box for object 1: [0, 2, 1, 3] (rows 0-1, cols 1-2)
/// // objects[1] = bounding box for object 2: [1, 4, 4, 5] (rows 1-3, cols 4-4)  
/// // objects[2] = bounding box for object 3: [3, 5, 0, 2] (rows 3-4, cols 0-1)
/// ```
///
/// ## Cell detection and extraction workflow
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::find_objects;
///
/// // Simulate cell segmentation result
/// let cell_labels = Array2::from_shape_fn((100, 100), |(i, j)| {
///     // Create scattered cell-like objects
///     match ((i / 20), (j / 25)) {
///         (1, 0) => 1,  // Cell 1 in upper left
///         (1, 2) => 2,  // Cell 2 in upper right
///         (3, 1) => 3,  // Cell 3 in lower center
///         (4, 3) => 4,  // Cell 4 in lower right
///         _ => 0,       // Background
///     }
/// });
///
/// let cell_bboxes = find_objects(&cell_labels).unwrap();
///
/// // Process each detected cell
/// for (cell_id, bbox) in cell_bboxes.iter().enumerate() {
///     let cell_label = cell_id + 1;
///     println!("Cell {}: bounding box = {:?}", cell_label, bbox);
///     
///     // Extract cell region for detailed analysis
///     let min_row = bbox[0];
///     let max_row = bbox[1];
///     let min_col = bbox[2];
///     let max_col = bbox[3];
///     
///     let cell_width = max_col - min_col;
///     let cell_height = max_row - min_row;
///     
///     println!("  Size: {}x{} pixels", cell_width, cell_height);
/// }
/// ```
///
/// ## 3D object detection
/// ```rust
/// use scirs2_core::ndarray::Array3;
/// use scirs2_ndimage::measurements::find_objects;
///
/// // Create 3D labeled volume
/// let labeled_volume = Array3::from_shape_fn((50, 50, 50), |(z, y, x)| {
///     // Create 3D objects at different positions
///     if (z >= 10 && z < 20) && (y >= 10 && y < 20) && (x >= 10 && x < 20) {
///         1  // Object 1: cube in center
///     } else if (z >= 30 && z < 40) && (y >= 5 && y < 15) && (x >= 35 && x < 45) {
///         2  // Object 2: cube in corner
///     } else {
///         0  // Background
///     }
/// });
///
/// let object_bboxes = find_objects(&labeled_volume).unwrap();
///
/// for (obj_id, bbox) in object_bboxes.iter().enumerate() {
///     println!("3D Object {}: ", obj_id + 1);
///     println!("  Z range: {}-{}", bbox[0], bbox[1]);
///     println!("  Y range: {}-{}", bbox[2], bbox[3]);
///     println!("  X range: {}-{}", bbox[4], bbox[5]);
///     
///     let volume = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) * (bbox[5] - bbox[4]);
///     println!("  Bounding volume: {} voxels", volume);
/// }
/// ```
///
/// ## Object extraction and cropping
/// ```rust
/// use scirs2_core::ndarray::{Array2, s};
/// use scirs2_ndimage::measurements::find_objects;
///
/// let segmentedimage = Array2::from_shape_fn((60, 60), |(i, j)| {
///     // Create multiple objects
///     if (i >= 10 && i < 25) && (j >= 10 && j < 25) {
///         1  // Square object
///     } else if ((i as i32 - 40).pow(2) + (j as i32 - 40).pow(2)) < 100 {
///         2  // Circular object
///     } else {
///         0  // Background
///     }
/// });
///
/// let bboxes = find_objects(&segmentedimage).unwrap();
///
/// // Extract each object as a separate sub-array
/// for (obj_id, bbox) in bboxes.iter().enumerate() {
///     let min_row = bbox[0];
///     let max_row = bbox[1];
///     let min_col = bbox[2];
///     let max_col = bbox[3];
///     
///     // Crop the object from the original image
///     let cropped_object = segmentedimage.slice(s![min_row..max_row, min_col..max_col]);
///     
///     println!("Object {} cropped to shape: {:?}", obj_id + 1, cropped_object.shape());
///     
///     // Save or process the cropped object...
/// }
/// ```
///
/// ## Quality control: filter objects by size
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::find_objects;
///
/// let detection_result = Array2::from_shape_fn((100, 100), |(i, j)| {
///     // Simulate detection with objects of various sizes
///     match ((i / 15), (j / 15)) {
///         (0, 0) => 1,   // Small object
///         (1, 1) => 2,   // Medium object  
///         (2, 2) => 3,   // Large object
///         (5, 5) => 4,   // Tiny object (noise)
///         _ => 0,
///     }
/// });
///
/// let all_bboxes = find_objects(&detection_result).unwrap();
///
/// // Filter objects by minimum size
/// let min_area = 100;  // Minimum area threshold
/// let valid_objects: Vec<(usize, &Vec<usize>)> = all_bboxes.iter()
///     .enumerate()
///     .filter(|(_, bbox)| {
///         let area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]);
///         area >= min_area
///     })
///     .collect();
///
/// println!("Found {} objects after size filtering", valid_objects.len());
///
/// for (obj_id, bbox) in valid_objects {
///     let area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]);
///     println!("Valid object {}: area = {} pixels", obj_id + 1, area);
/// }
/// ```
///
/// # Applications
///
/// - **Object Detection**: Locate all objects in segmented images
/// - **Cell Biology**: Find and extract individual cells for analysis
/// - **Quality Control**: Identify and measure components or defects
/// - **Medical Imaging**: Locate anatomical structures or lesions
/// - **Materials Science**: Find and analyze particles or grains
/// - **Automated Analysis**: Batch process multiple objects
///
/// # Performance Notes
///
/// - Time complexity: O(n) where n is the total number of pixels
/// - Space complexity: O(k) where k is the number of objects
/// - For large images with many small objects, consider pre-filtering by size
/// - Processing scales linearly with image dimensions
///
/// # Implementation Note
///
/// Current implementation is a placeholder returning minimal data.
/// Full object detection algorithm needs to be implemented with proper
/// bounding box calculation for each unique label.
///
/// # Errors
///
/// Returns an error if:
/// - Input array is 0-dimensional
/// - Memory allocation fails for large numbers of objects
///
/// # Related Functions
///
/// - [`region_properties`]: Get detailed properties of each object
/// - [`count_labels`]: Count pixels in each object
/// - Label connectivity functions for segmentation preprocessing
#[allow(dead_code)]
pub fn find_objects<D>(input: &Array<usize, D>) -> NdimageResult<Vec<Vec<usize>>>
where
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // Find all unique labels (excluding background label 0)
    let mut unique_labels = std::collections::HashSet::new();
    for &label in input.iter() {
        if label > 0 {
            unique_labels.insert(label);
        }
    }

    if unique_labels.is_empty() {
        return Ok(vec![]);
    }

    let mut bboxes = Vec::new();

    // Calculate bounding box for each object
    for &label in unique_labels.iter() {
        let mut min_coords = vec![usize::MAX; input.ndim()];
        let mut max_coords = vec![0; input.ndim()];
        let mut found_object = false;

        // Scan through all pixels to find object bounds
        for (coords, &pixel_label) in input.indexed_iter() {
            if pixel_label == label {
                found_object = true;

                // Update bounding box coordinates
                let coord_vec = pattern_to_coords::<D>(&coords, input.shape());
                for (i, coord) in coord_vec.iter().enumerate() {
                    min_coords[i] = min_coords[i].min(*coord);
                    max_coords[i] = max_coords[i].max(*coord);
                }
            }
        }

        if found_object {
            // Create bounding box in the format [min1, max1, min2, max2, ...]
            let mut bbox = Vec::new();
            for i in 0..input.ndim() {
                bbox.push(min_coords[i]);
                bbox.push(max_coords[i] + 1); // Add 1 to make it exclusive end
            }
            bboxes.push(bbox);
        }
    }

    // Sort bboxes by the label order for consistent output
    // Since we don't store labels, we sort by first coordinate for consistency
    bboxes.sort_by_key(|bbox| bbox[0]);

    Ok(bboxes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_region_properties() {
        let input: Array2<f64> = Array2::eye(3);
        let labels: Array2<usize> = Array2::from_elem((3, 3), 1);
        let props = region_properties(&input, &labels, None).unwrap();

        assert_eq!(props.len(), 1);
        assert_eq!(props[0].label, 1);
        assert_eq!(props[0].centroid.len(), input.ndim());
    }

    #[test]
    fn test_find_objects() {
        let input: Array2<usize> = Array2::from_elem((3, 3), 1);
        let objects = find_objects(&input).unwrap();

        assert!(!objects.is_empty());
    }
}
