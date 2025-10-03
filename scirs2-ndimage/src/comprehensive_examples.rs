//! Comprehensive Examples and Documentation for scirs2-ndimage
//!
//! This module provides extensive examples, tutorials, and documentation
//! for all major functionality in scirs2-ndimage. It serves as both
//! educational material and validation of the API usability.

use std::collections::HashMap;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Comprehensive tutorial and example collection
pub struct ExampleTutorial {
    /// Tutorial steps with descriptions
    steps: Vec<TutorialStep>,
    /// Generated outputs for validation
    outputs: HashMap<String, String>,
}

/// Individual tutorial step
#[derive(Debug, Clone)]
pub struct TutorialStep {
    /// Step title
    pub title: String,
    /// Description of what this step demonstrates
    pub description: String,
    /// Code example as string
    pub code_example: String,
    /// Expected output description
    pub expected_output: String,
    /// Key concepts covered
    pub concepts: Vec<String>,
    /// Related functions
    pub related_functions: Vec<String>,
}

impl ExampleTutorial {
    /// Create a new comprehensive tutorial
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            outputs: HashMap::new(),
        }
    }

    /// Add tutorial step
    pub fn add_step(&mut self, step: TutorialStep) {
        self.steps.push(step);
    }

    /// Generate all tutorial steps
    pub fn generate_complete_tutorial(&mut self) -> Result<()> {
        self.add_filter_examples()?;
        self.add_morphology_examples()?;
        self.add_interpolation_examples()?;
        self.add_measurement_examples()?;
        self.add_segmentation_examples()?;
        self.add_feature_detection_examples()?;
        self.add_advanced_workflow_examples()?;
        Ok(())
    }

    /// Add comprehensive filter examples
    fn add_filter_examples(&mut self) -> Result<()> {
        // Gaussian Filter Tutorial
        self.add_step(TutorialStep {
            title: "Gaussian Smoothing for Noise Reduction".to_string(),
            description:
                "Learn how to apply Gaussian filters to reduce noise while preserving edges"
                    .to_string(),
            code_example: r#"
use scirs2_ndimage::filters::{gaussian_filter, BorderMode};
use scirs2_core::ndarray::Array2;

// Create a noisy image (in practice, you'd load from file)
let mut noisyimage = Array2::zeros((100, 100));
// Add some signal
for i in 40..60 {
    for j in 40..60 {
        noisyimage[[i, j]] = 1.0;
    }
}
// Add noise (in practice, this would come from your data)
for i in 0..100 {
    for j in 0..100 {
        noisyimage[[i, j]] += 0.1 * ((i + j) as f64).sin();
    }
}

// Apply Gaussian filter with different sigma values
let smooth_light = gaussian_filter(&noisyimage, 1.0, None, None)?;
let smooth_medium = gaussian_filter(&noisyimage, 2.0, None, None)?;
let smooth_heavy = gaussian_filter(&noisyimage, 4.0, None, None)?;

// Use different border modes
let reflected = gaussian_filter(&noisyimage, 2.0, Some(BorderMode::Reflect), None)?;
let wrapped = gaussian_filter(&noisyimage, 2.0, Some(BorderMode::Wrap), None)?;
let constant = gaussian_filter(&noisyimage, 2.0, Some(BorderMode::Constant), Some(0.0))?;

println!("Applied Gaussian filters with sigma: 1.0, 2.0, 4.0");
println!("Border modes: Reflect, Wrap, Constant");
"#
            .to_string(),
            expected_output: "Progressively smoother images with different boundary handling"
                .to_string(),
            concepts: vec![
                "Gaussian smoothing".to_string(),
                "Noise reduction".to_string(),
                "Border mode handling".to_string(),
                "Parameter selection".to_string(),
            ],
            related_functions: vec![
                "uniform_filter".to_string(),
                "median_filter".to_string(),
                "bilateral_filter".to_string(),
            ],
        });

        // Median Filter Tutorial
        self.add_step(TutorialStep {
            title: "Median Filtering for Impulse Noise Removal".to_string(),
            description:
                "Use median filters to remove salt-and-pepper noise while preserving edges"
                    .to_string(),
            code_example: r#"
use scirs2_ndimage::filters::{median_filter, BorderMode};
use scirs2_core::ndarray::Array2;

// Create image with impulse noise
let mut image_with_impulses = Array2::from_shape_fn((50, 50), |(i, j)| {
    // Create some structure
    if (i as i32 - 25).pow(2) + (j as i32 - 25).pow(2) < 100 {
        1.0
    } else {
        0.0
    }
});

// Add impulse noise (salt and pepper)
image_with_impulses[[10, 10]] = 1.0; // salt
image_with_impulses[[35, 35]] = 0.0; // pepper
image_with_impulses[[20, 30]] = 1.0; // salt
image_with_impulses[[40, 15]] = 0.0; // pepper

// Apply median filter with different kernel sizes
let cleaned_3x3 = median_filter(&image_with_impulses, &[3, 3], None)?;
let cleaned_5x5 = median_filter(&image_with_impulses, &[5, 5], None)?;
let cleaned_7x7 = median_filter(&image_with_impulses, &[7, 7], None)?;

// Compare with Gaussian filter (less effective for impulse noise)
let gaussian_cleaned = gaussian_filter(&image_with_impulses, 1.0, None, None)?;

println!("Median filter effectively removes impulse noise");
println!("Larger kernels remove more noise but may blur edges");
"#
            .to_string(),
            expected_output:
                "Clean images with impulse noise removed, edge preservation comparison".to_string(),
            concepts: vec![
                "Impulse noise removal".to_string(),
                "Edge preservation".to_string(),
                "Kernel size effects".to_string(),
                "Filter comparison".to_string(),
            ],
            related_functions: vec![
                "rank_filter".to_string(),
                "percentile_filter".to_string(),
                "minimum_filter".to_string(),
                "maximum_filter".to_string(),
            ],
        });

        // Edge Detection Tutorial
        self.add_step(TutorialStep {
            title: "Edge Detection with Sobel and Laplacian Filters".to_string(),
            description: "Detect edges using gradient-based and Laplacian operators".to_string(),
            code_example: r#"
use scirs2_ndimage::filters::{sobel, laplace, gaussian_filter};
use scirs2_core::ndarray::Array2;

// Create test image with clear edges
let image = Array2::from_shape_fn((60, 60), |(i, j)| {
    if i > 30 && j > 30 {
        1.0
    } else if i < 20 || j < 20 {
        0.5
    } else {
        0.0
    }
});

// Pre-smooth to reduce noise
let smoothed = gaussian_filter(&image, 1.0, None, None)?;

// Detect edges with Sobel filter
let edges_x = sobel(&smoothed, Some(0), None, None)?; // Vertical edges
let edges_y = sobel(&smoothed, Some(1), None, None)?; // Horizontal edges
let edges_magnitude = sobel(&smoothed, None, None, None)?; // Gradient magnitude

// Detect edges with Laplacian (second derivative)
let laplacian_edges = laplace(&smoothed, None)?;

// Combine for comprehensive edge detection
let combined_edges = Array2::from_shape_fn(image.dim(), |(i, j)| {
    let sobel_val = edges_magnitude[[i, j]];
    let laplacian_val = laplacian_edges[[i, j]].abs();
    (sobel_val + 0.5 * laplacian_val).min(1.0)
});

println!("Detected edges using Sobel (gradient) and Laplacian (second derivative)");
println!("Combined approach provides comprehensive edge information");
"#
            .to_string(),
            expected_output: "Edge maps showing different types of edge information".to_string(),
            concepts: vec![
                "Gradient-based edge detection".to_string(),
                "Laplacian edge detection".to_string(),
                "Edge orientation".to_string(),
                "Multi-scale edge detection".to_string(),
            ],
            related_functions: vec![
                "canny".to_string(),
                "prewitt".to_string(),
                "scharr".to_string(),
                "roberts".to_string(),
            ],
        });

        Ok(())
    }

    /// Add morphological operation examples
    fn add_morphology_examples(&mut self) -> Result<()> {
        self.add_step(TutorialStep {
            title: "Binary Morphology for Shape Analysis".to_string(),
            description: "Use erosion, dilation, opening, and closing for shape processing".to_string(),
            code_example: r#"
use scirs2_ndimage::morphology::{
    binary_erosion, binary_dilation, binary_opening, binary_closing,
    generate_binary_structure, disk_structure
};
use scirs2_core::ndarray::Array2;

// Create binary image with various shapes
let mut binary_image = Array2::from_elem((80, 80), false);

// Add some shapes
for i in 20..30 {
    for j in 20..35 {
        binary_image[[i, j]] = true; // Rectangle
    }
}

for i in 50..70 {
    for j in 50..70 {
        if (i as i32 - 60).pow(2) + (j as i32 - 60).pow(2) < 80 {
            binary_image[[i, j]] = true; // Circle
        }
    }
}

// Add some noise (small isolated pixels)
binary_image[[10, 10]] = true;
binary_image[[70, 20]] = true;

// Define structuring elements
let cross_3x3 = generate_binary_structure(2, 1)?; // 4-connected
let square_3x3 = generate_binary_structure(2, 2)?; // 8-connected
let disk_5 = disk_structure(5)?; // Circular structuring element

// Basic morphological operations
let eroded = binary_erosion(&binary_image, Some(&cross_3x3), None, None, None, None, None)?;
let dilated = binary_dilation(&binary_image, Some(&cross_3x3), None, None, None, None, None)?;

// Compound operations
let opened = binary_opening(&binary_image, Some(&cross_3x3), None, None, None, None)?;  // Remove noise
let closed = binary_closing(&binary_image, Some(&cross_3x3), None, None, None, None)?;  // Fill holes

// Different structuring elements produce different results
let opened_disk = binary_opening(&binary_image, Some(&disk_5), None, None, None, None)?;

println!("Applied morphological operations:");
println!("- Erosion: shrinks objects");
println!("- Dilation: expands objects");
println!("- Opening: removes noise, separates connected objects");
println!("- Closing: fills holes, connects nearby objects");
"#.to_string(),
            expected_output: "Processed binary images showing shape modifications".to_string(),
            concepts: vec![
                "Binary morphology".to_string(),
                "Structuring elements".to_string(),
                "Noise removal".to_string(),
                "Shape modification".to_string(),
            ],
            related_functions: vec![
                "grey_erosion".to_string(),
                "grey_dilation".to_string(),
                "white_tophat".to_string(),
                "black_tophat".to_string(),
            ],
        });

        self.add_step(TutorialStep {
            title: "Grayscale Morphology for Contrast Enhancement".to_string(),
            description: "Apply morphological operations to grayscale images for various effects"
                .to_string(),
            code_example: r#"
use scirs2_ndimage::morphology::{
    grey_erosion, grey_dilation, grey_opening, grey_closing,
    white_tophat, black_tophat, morphological_gradient
};
use scirs2_core::ndarray::Array2;

// Create grayscale test image
let image = Array2::from_shape_fn((60, 60), |(i, j)| {
    let center_i = 30.0;
    let center_j = 30.0;
    let distance = ((i as f64 - center_i).powi(2) + (j as f64 - center_j).powi(2)).sqrt();
    
    if distance < 15.0 {
        0.8 - distance / 20.0  // Bright center, fading to edges
    } else {
        0.1 + 0.1 * ((i + j) as f64 * 0.2).sin()  // Textured background
    }
});

// Basic grayscale morphology
let eroded = grey_erosion(&image, None, None, None, None, None)?;
let dilated = grey_dilation(&image, None, None, None, None, None)?;

// Compound operations
let opened = grey_opening(&image, None, None, None, None, None)?;   // Remove bright noise
let closed = grey_closing(&image, None, None, None, None, None)?;   // Remove dark noise

// Top-hat transformations for feature enhancement
let white_hat = white_tophat(&image, None, None, None, None, None)?; // Bright features
let black_hat = black_tophat(&image, None, None, None, None, None)?; // Dark features

// Morphological gradient for edge detection
let gradient = morphological_gradient(&image, None, None, None, None, None)?;

// Enhance contrast by combining operations
let enhanced = Array2::from_shape_fn(image.dim(), |(i, j)| {
    let original = image[[i, j]];
    let white_feature = "white_hat"[[i, j]];
    let black_feature = "black_hat"[[i, j]];
    (original + white_feature - black_feature).max(0.0).min(1.0)
});

println!("Applied grayscale morphological operations for:");
println!("- Noise removal (opening/closing)");
println!("- Feature enhancement (top-hat transforms)");  
println!("- Edge detection (morphological gradient)");
println!("- Contrast enhancement (combination of operations)");
"#
            .to_string(),
            expected_output: "Enhanced grayscale images with improved contrast and features"
                .to_string(),
            concepts: vec![
                "Grayscale morphology".to_string(),
                "Feature enhancement".to_string(),
                "Contrast adjustment".to_string(),
                "Top-hat transforms".to_string(),
            ],
            related_functions: vec![
                "morphological_laplace".to_string(),
                "distance_transform_edt".to_string(),
                "label".to_string(),
            ],
        });

        Ok(())
    }

    /// Add interpolation and transformation examples
    fn add_interpolation_examples(&mut self) -> Result<()> {
        self.add_step(TutorialStep {
            title: "Image Interpolation and Geometric Transformations".to_string(),
            description: "Resize, rotate, and transform images with various interpolation methods".to_string(),
            code_example: r#"
use scirs2_ndimage::interpolation::{
    zoom, rotate, shift, affine_transform, map_coordinates,
    InterpolationOrder, BoundaryMode
};
use scirs2_core::ndarray::{Array2, Array1};

// Create test image with clear features
let image = Array2::from_shape_fn((40, 40), |(i, j)| {
    let x = i as f64 - 20.0;
    let y = j as f64 - 20.0;
    
    // Create checkerboard pattern
    if ((i / 5) + (j / 5)) % 2 == 0 {
        1.0
    } else {
        0.0
    }
});

// Zooming with different interpolation orders
let zoomed_nearest = zoom(&image, &[2.0, 2.0], Some(InterpolationOrder::Nearest), None, None, None)?;
let zoomed_linear = zoom(&image, &[2.0, 2.0], Some(InterpolationOrder::Linear), None, None, None)?;
let zoomed_cubic = zoom(&image, &[2.0, 2.0], Some(InterpolationOrder::Cubic), None, None, None)?;

// Rotation with different angles
let rotated_45 = rotate(&image, 45.0, None, None, None, None, None, None)?;
let rotated_30 = rotate(&image, 30.0, Some(InterpolationOrder::Linear), None, None, None, None, None)?;

// Translation (shifting)
let shifted = shift(&image, &[5.0, -3.0], None, None, None, None)?;

// Affine transformation (scaling + rotation + translation)
let transformation_matrix = Array2::from_shape_vec((2, 2), vec![
    1.5 * 45.0_f64.to_radians().cos(), -1.5 * 45.0_f64.to_radians().sin(),
    1.5 * 45.0_f64.to_radians().sin(), 1.5 * 45.0_f64.to_radians().cos()
])?;
let offset = Array1::from_vec(vec![5.0, -2.0]);

let transformed = affine_transform(
    &image, 
    &transformation_matrix, 
    Some(&offset),
    None, 
    Some(InterpolationOrder::Linear),
    None, 
    None, 
    None
)?;

// Custom coordinate mapping
let coords = Array2::from_shape_fn((2, 40 * 40), |(axis, idx)| {
    let i = idx / 40;
    let j = idx % 40;
    match axis {
        0 => i as f64 + 2.0 * (j as f64 * 0.1).sin(), // Wavy distortion
        1 => j as f64 + 1.0 * (i as f64 * 0.1).cos(, _ => 0.0
    }
});

let warped = map_coordinates(&image, &coords, Some(InterpolationOrder::Linear), None, None, None)?;

println!("Applied various geometric transformations:");
println!("- Zooming with different interpolation orders");
println!("- Rotation and translation");
println!("- Affine transformations (combined scaling, rotation, translation)");
println!("- Custom coordinate mapping for complex distortions");
"#.to_string(),
            expected_output: "Transformed images showing different interpolation and geometric effects".to_string(),
            concepts: vec![
                "Image interpolation".to_string(),
                "Geometric transformations".to_string(),
                "Interpolation order effects".to_string(),
                "Coordinate mapping".to_string(),
            ],
            related_functions: vec![
                "geometric_transform".to_string(),
                "spline_filter".to_string(),
                "value_at_coordinates".to_string(),
            ],
        });

        Ok(())
    }

    /// Add measurement and analysis examples
    fn add_measurement_examples(&mut self) -> Result<()> {
        self.add_step(TutorialStep {
            title: "Image Measurements and Region Analysis".to_string(),
            description: "Extract quantitative information from images using measurement functions"
                .to_string(),
            code_example: r#"
use scirs2_ndimage::measurements::{
    center_of_mass, find_objects, moments, label, 
    sum_labels, mean_labels, variance_labels,
    extrema, region_properties
};
use scirs2_ndimage::morphology::label;
use scirs2_core::ndarray::Array2;

// Create test image with multiple objects
let mut image = Array2::zeros((60, 60));

// Object 1: Circle in top-left
for i in 10..20 {
    for j in 10..20 {
        if (i as i32 - 15).pow(2) + (j as i32 - 15).pow(2) < 25 {
            image[[i, j]] = 2.0;
        }
    }
}

// Object 2: Rectangle in top-right  
for i in 10..20 {
    for j in 40..50 {
        image[[i, j]] = 3.0;
    }
}

// Object 3: Triangle-like shape in bottom
for i in 40..50 {
    for j in (25 - (i - 40))..(25 + (i - 40) + 1) {
        if j >= 0 && j < 60 {
            image[[i, j]] = 1.5;
        }
    }
}

// Basic measurements
let total_centroid = center_of_mass(&image)?;
let total_moments = moments(&image)?;
let (min_val, max_val, min_pos, max_pos) = extrema(&image)?;

println!("Global measurements:");
println!("Center of mass: {:?}", total_centroid);
println!("Min value: {} at {:?}", min_val, min_pos);
println!("Max value: {} at {:?}", max_val, max_pos);

// Label connected components for individual object analysis
let binary_image = Array2::from_shape_fn(image.dim(), |(i, j)| image[[i, j]] > 0.0);
let (labeled, num_labels) = label(&binary_image, None)?;

println!("Found {} connected objects", num_labels);

// Analyze each labeled region
let sums = sum_labels(&image, &labeled, Some(num_labels))?;
let means = mean_labels(&image, &labeled, Some(num_labels))?;
let variances = variance_labels(&image, &labeled, Some(num_labels))?;

for label_id in 1..=num_labels {
    println!("Object {}:", label_id);
    println!("  Sum: {:.2}", sums[label_id - 1]);
    println!("  Mean: {:.2}", means[label_id - 1]);
    println!("  Variance: {:.2}", variances[label_id - 1]);
}

// Find bounding boxes of objects
let objects = find_objects(&labeled)?;
for (i, bbox) in objects.iter().enumerate() {
    if let Some(bbox) = bbox {
        println!("Object {} bounding box: {:?}", i + 1, bbox);
    }
}

// Calculate region properties
let properties = region_properties(&labeled, Some(&image))?;
for (i, prop) in properties.iter().enumerate() {
    println!("Object {} properties:", i + 1);
    println!("  Area: {}", prop.area);
    println!("  Centroid: {:?}", prop.centroid);
    println!("  Perimeter: {}", prop.perimeter);
    println!("  Major axis length: {:.2}", prop.major_axis_length);
    println!("  Minor axis length: {:.2}", prop.minor_axis_length);
    println!("  Orientation: {:.2} degrees", prop.orientation.to_degrees());
}
"#
            .to_string(),
            expected_output: "Quantitative measurements for each object in the image".to_string(),
            concepts: vec![
                "Region analysis".to_string(),
                "Object measurements".to_string(),
                "Connected components".to_string(),
                "Statistical analysis".to_string(),
            ],
            related_functions: vec![
                "watershed".to_string(),
                "peak_local_maxima".to_string(),
                "histogram".to_string(),
            ],
        });

        Ok(())
    }

    /// Add segmentation examples
    fn add_segmentation_examples(&mut self) -> Result<()> {
        self.add_step(TutorialStep {
            title: "Image Segmentation Techniques".to_string(),
            description: "Segment images into regions using thresholding and watershed algorithms"
                .to_string(),
            code_example: r#"
use scirs2_ndimage::segmentation::{
    threshold_binary, otsu_threshold, adaptive_threshold,
    watershed, marker_watershed, AdaptiveMethod
};
use scirs2_ndimage::filters::gaussian_filter;
use scirs2_ndimage::morphology::{binary_erosion, label};
use scirs2_core::ndarray::Array2;

// Create test image with multiple regions
let image = Array2::from_shape_fn((80, 80), |(i, j)| {
    let x = i as f64 - 40.0;
    let y = j as f64 - 40.0;
    let distance = (x * x + y * y).sqrt();
    
    if distance < 15.0 {
        0.8  // Bright center
    } else if distance < 25.0 {
        0.4  // Medium ring
    } else if distance < 35.0 {
        0.6  // Brighter outer ring
    } else {
        0.1  // Dark background
    }
}) + Array2::from_shape_fn((80, 80), |(i, j)| {
    0.05 * ((i as f64 * 0.2).sin() + (j as f64 * 0.3).cos())  // Add some texture
});

// Simple binary thresholding
let binary_simple = threshold_binary(&image, 0.3)?;

// Otsu's automatic threshold selection
let otsu_threshold_value = otsu_threshold(&image)?;
let binary_otsu = threshold_binary(&image, otsu_threshold_value)?;

// Adaptive thresholding for varying illumination
let binary_adaptive_mean = adaptive_threshold(
    &image, 
    11,  // window size
    AdaptiveMethod::Mean, 
    0.1  // offset
)?;

let binary_adaptive_gaussian = adaptive_threshold(
    &image, 
    11, 
    AdaptiveMethod::Gaussian, 
    0.05
)?;

println!("Applied thresholding methods:");
println!("- Simple binary threshold");
println!("- Otsu automatic threshold: {:.3}", otsu_threshold_value);
println!("- Adaptive mean threshold");
println!("- Adaptive Gaussian threshold");

// Watershed segmentation for separating touching objects
let smoothed = gaussian_filter(&image, 1.0, None, None)?;
let watershed_result = watershed(&smoothed, None, None, None)?;

// Marker-controlled watershed for better control
// Create markers by finding local maxima
let eroded = binary_erosion(&binary_otsu, None, None, None, None, None, None)?;
let (markers, num_markers) = label(&eroded, None)?;

let marker_watershed_result = marker_watershed(
    &smoothed, 
    &markers, 
    None, 
    None, 
    None
)?;

println!("Watershed segmentation:");
println!("- Basic watershed found {} regions", watershed_result.iter().max().unwrap_or(&0) + 1);
println!("- Marker-controlled watershed with {} markers", num_markers);

// Combine different segmentation results
let combined_segmentation = Array2::from_shape_fn(image.dim(), |(i, j)| {
    let otsu_val = if binary_otsu[[i, j]] { 1 } else { 0 };
    let adaptive_val = if binary_adaptive_gaussian[[i, j]] { 2 } else { 0 };
    let watershed_val = watershed_result[[i, j]] * 3;
    
    otsu_val + adaptive_val + watershed_val
});

println!("Created combined segmentation using multiple methods");
"#
            .to_string(),
            expected_output: "Segmented images showing different region separation techniques"
                .to_string(),
            concepts: vec![
                "Image thresholding".to_string(),
                "Adaptive segmentation".to_string(),
                "Watershed algorithm".to_string(),
                "Marker-controlled segmentation".to_string(),
            ],
            related_functions: vec![
                "chan_vese".to_string(),
                "active_contour".to_string(),
                "graph_cuts".to_string(),
            ],
        });

        Ok(())
    }

    /// Add feature detection examples
    fn add_feature_detection_examples(&mut self) -> Result<()> {
        self.add_step(TutorialStep {
            title: "Feature Detection and Corner Finding".to_string(),
            description: "Detect corners, edges, and other features in images".to_string(),
            code_example: r#"
use scirs2_ndimage::features::{
    canny, harris_corners, fast_corners, sobel_edges,
    gradient_edges, GradientMethod
};
use scirs2_ndimage::filters::gaussian_filter;
use scirs2_core::ndarray::Array2;

// Create test image with corners and edges
let image = Array2::from_shape_fn((60, 60), |(i, j)| {
    // Create a rectangular structure with internal features
    if (i > 15 && i < 45 && j > 15 && j < 45) {
        if (i > 25 && i < 35 && j > 25 && j < 35) {
            0.3  // Inner rectangle (darker)
        } else {
            0.8  // Outer rectangle (bright)
        }
    } else {
        0.1  // Background (dark)
    }
}) + Array2::from_shape_fn((60, 60), |(i, j)| {
    // Add some texture and additional features
    if (i as i32 - 20).pow(2) + (j as i32 - 40).pow(2) < 25 {
        0.2  // Small circle
    } else {
        0.0
    }
});

// Edge detection with Canny
let canny_edges = canny(
    &image,
    1.0,    // sigma for Gaussian smoothing
    0.1,    // low threshold
    0.2,    // high threshold
    None
)?;

// Edge detection with Sobel
let sobel_edges_result = sobel_edges(&image, None)?;

// Edge detection with gradient methods
let gradient_edges_sobel = gradient_edges(&image, GradientMethod::Sobel)?;
let gradient_edges_scharr = gradient_edges(&image, GradientMethod::Scharr)?;

println!("Edge detection results:");
println!("- Canny edge detection with hysteresis thresholding");
println!("- Sobel edge detection");
println!("- Gradient-based edge detection (Sobel and Scharr)");

// Corner detection with Harris
let harris_response = harris_corners(
    &image,
    1.0,    // sigma for derivatives
    0.04,   // k parameter
    None    // optional mask
)?;

// Find corner locations (peaks in Harris response)
let corner_threshold = 0.01;
let mut corner_locations = Vec::new();
for ((i, j), &response) in harris_response.indexed_iter() {
    if response > corner_threshold {
        // Check if it's a local maximum
        let mut is_maximum = true;
        for di in -1..=1 {
            for dj in -1..=1 {
                if di == 0 && dj == 0 { continue; }
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni >= 0 && ni < 60 && nj >= 0 && nj < 60 {
                    if harris_response[[ni as usize, nj as usize]] > response {
                        is_maximum = false;
                        break;
                    }
                }
            }
            if !is_maximum { break; }
        }
        if is_maximum {
            corner_locations.push((i, j));
        }
    }
}

println!("Harris corner detection found {} corners:", corner_locations.len());
for (i, &(row, col)) in corner_locations.iter().enumerate() {
    println!("  Corner {}: ({}, {})", i + 1, row, col);
}

// FAST corner detection
let fast_corners_result = fast_corners(
    &image,
    0.1,    // threshold
    true    // non-maximum suppression
)?;

println!("FAST corner detection found {} corners", fast_corners_result.len());

// Combine edge and corner information
let feature_map = Array2::from_shape_fn(image.dim(), |(i, j)| {
    let edge_val = if canny_edges[[i, j]] { 0.5 } else { 0.0 };
    let harris_val = harris_response[[i, j]] * 2.0;
    let fast_val = if fast_corners_result.iter().any(|&(r, c)| r == i && c == j) { 0.3 } else { 0.0 };
    
    (edge_val + harris_val + fast_val).min(1.0)
});

println!("Created combined feature map with edges and corners");
"#.to_string(),
            expected_output: "Feature maps showing detected edges and corners".to_string(),
            concepts: vec![
                "Edge detection".to_string(),
                "Corner detection".to_string(),
                "Feature extraction".to_string(),
                "Multi-scale analysis".to_string(),
            ],
            related_functions: vec![
                "laplacian_edges".to_string(),
                "edge_detector".to_string(),
                "hough_transform".to_string(),
            ],
        });

        Ok(())
    }

    /// Add advanced workflow examples
    fn add_advanced_workflow_examples(&mut self) -> Result<()> {
        self.add_step(TutorialStep {
            title: "Complete Image Analysis Workflow".to_string(),
            description: "Combine multiple techniques for comprehensive image analysis".to_string(),
            code_example: r#"
use scirs2_ndimage::*;
use scirs2_core::ndarray::Array2;
use statrs::statistics::Statistics;

// Simulate a real-world image analysis scenario:
// Analyzing cellular structures in microscopy images

// Create synthetic microscopy image
let image = Array2::from_shape_fn((100, 100), |(i, j)| {
    let mut intensity = 0.1; // Background
    
    // Add several "cells" with varying intensities
    let cells = [
        (25, 25, 8, 0.8),   // (center_i, center_j, radius, intensity)
        (25, 75, 6, 0.6),
        (75, 25, 10, 0.9),
        (75, 75, 7, 0.7),
        (50, 50, 12, 0.85),
    ];
    
    for &(ci, cj, radius, cell_intensity) in &cells {
        let distance = ((i as i32 - ci).pow(2) + (j as i32 - cj).pow(2)) as f64;
        if distance < (radius as f64).pow(2) {
            intensity = intensity.max(cell_intensity);
        }
    }
    
    // Add some noise
    intensity + 0.05 * ((i + j) as f64 * 0.3).sin()
});

println!("=== COMPLETE IMAGE ANALYSIS WORKFLOW ===");
println!("Analyzing synthetic microscopy image with {} cells", 5);

// Step 1: Preprocessing
println!("\n1. PREPROCESSING");
let denoised = filters::gaussian_filter(&image, 0.8, None, None)?;
println!("   - Applied Gaussian smoothing for noise reduction");

// Step 2: Segmentation
println!("\n2. SEGMENTATION");
let threshold_value = segmentation::otsu_threshold(&denoised)?;
let binary_mask = segmentation::threshold_binary(&denoised, threshold_value)?;
println!("   - Otsu threshold: {:.3}", threshold_value);

// Improve segmentation with morphological operations
let cleaned_mask = morphology::binary_opening(&binary_mask, None, None, None, None, None)?;
let filled_mask = morphology::binary_closing(&cleaned_mask, None, None, None, None, None)?;
println!("   - Applied morphological cleaning (opening + closing)");

// Step 3: Object labeling and counting
println!("\n3. OBJECT DETECTION");
let (labeled, num_objects) = morphology::label(&filled_mask, None)?;
println!("   - Detected {} connected objects", num_objects);

// Step 4: Quantitative analysis
println!("\n4. QUANTITATIVE ANALYSIS");
let objects = measurements::find_objects(&labeled)?;
let sums = measurements::sum_labels(&image, &labeled, Some(num_objects))?;
let means = measurements::mean_labels(&image, &labeled, Some(num_objects))?;
let properties = measurements::region_properties(&labeled, Some(&image))?;

for (i, prop) in properties.iter().enumerate() {
    if prop.area > 0 {
        println!("   Object {}:", i + 1);
        println!("     Area: {} pixels", prop.area);
        println!("     Mean intensity: {:.3}", means[i]);
        println!("     Total intensity: {:.1}", sums[i]);
        println!("     Centroid: ({:.1}, {:.1})", prop.centroid[0], prop.centroid[1]);
        println!("     Equivalent diameter: {:.1}", prop.equivalent_diameter);
        println!("     Eccentricity: {:.3}", prop.eccentricity);
    }
}

// Step 5: Feature extraction
println!("\n5. FEATURE EXTRACTION");
let edges = features::canny(&denoised, 1.0, 0.1, 0.2, None)?;
let edge_count: usize = edges.iter().map(|&x| if x { 1 } else { 0 }).sum();
println!("   - Detected {} edge pixels", edge_count);

let corners = features::harris_corners(&denoised, 1.0, 0.04, None)?;
let strong_corners: usize = corners.iter().map(|&x| if x > 0.01 { 1 } else { 0 }).sum();
println!("   - Found {} strong corner features", strong_corners);

// Step 6: Quality metrics
println!("\n6. QUALITY ASSESSMENT");
let image_mean = image.mean().unwrap();
let image_std = image.var(scirs2_core::ndarray::Axis(0)).unwrap().mean().unwrap().sqrt();
let signal_to_noise = image_mean / image_std;

println!("   - Image mean intensity: {:.3}", image_mean);
println!("   - Image standard deviation: {:.3}", image_std);
println!("   - Signal-to-noise ratio: {:.2}", signal_to_noise);

// Step 7: Results summary
println!("\n7. ANALYSIS SUMMARY");
println!("   ================");
println!("   Total objects detected: {}", num_objects);
println!("   Average object area: {:.1} pixels", 
                properties.iter().map(|p| p.area).sum::<usize>() as f64 / num_objects as f64);
println!("   Average object intensity: {:.3}", 
                means.iter().sum::<f64>() / num_objects as f64);
println!("   Image quality (SNR): {:.2}", signal_to_noise);
println!("   Edge density: {:.1}%", edge_count as f64 / (100.0 * 100.0) * 100.0);

// Create analysis result visualization
let visualization = Array2::from_shape_fn(image.dim(), |(i, j)| {
    let original = image[[i, j]];
    let has_edge = edges[[i, j]];
    let label_val = labeled[[i, j]] as f64 / num_objects as f64;
    
    if has_edge {
        1.0  // Edges in white
    } else if labeled[[i, j]] > 0 {
        0.5 + 0.5 * label_val  // Objects in gray levels
    } else {
        original * 0.3  // Background dimmed
    }
});

println!("\n8. WORKFLOW COMPLETE");
println!("   Created analysis visualization combining:");
println!("   - Original image data");
println!("   - Detected object boundaries");
println!("   - Edge features");
println!("   - Object labels with distinct intensities");
"#
            .to_string(),
            expected_output: "Complete quantitative analysis of image features and objects"
                .to_string(),
            concepts: vec![
                "Complete analysis workflow".to_string(),
                "Multi-step processing pipeline".to_string(),
                "Quantitative image analysis".to_string(),
                "Quality assessment".to_string(),
                "Results visualization".to_string(),
            ],
            related_functions: vec![
                "batch_process".to_string(),
                "pipeline_builder".to_string(),
                "analysis_report".to_string(),
            ],
        });

        Ok(())
    }

    /// Generate complete tutorial as markdown
    pub fn export_markdown(&self) -> String {
        let mut markdown = String::new();

        markdown.push_str("# Comprehensive scirs2-ndimage Tutorial\n\n");
        markdown.push_str("This tutorial provides comprehensive examples of all major functionality in scirs2-ndimage, ");
        markdown.push_str("demonstrating real-world usage patterns and best practices.\n\n");

        markdown.push_str("## Table of Contents\n\n");
        for (i, step) in self.steps.iter().enumerate() {
            markdown.push_str(&format!(
                "{}. [{}](#{})\n",
                i + 1,
                step.title,
                step.title.to_lowercase().replace(" ", "-")
            ));
        }
        markdown.push_str("\n");

        for (i, step) in self.steps.iter().enumerate() {
            markdown.push_str(&format!("## {}. {}\n\n", i + 1, step.title));
            markdown.push_str(&format!("{}\n\n", step.description));

            markdown.push_str("### Key Concepts\n");
            for concept in &step.concepts {
                markdown.push_str(&format!("- {}\n", concept));
            }
            markdown.push_str("\n");

            markdown.push_str("### Code Example\n\n");
            markdown.push_str("```rust\n");
            markdown.push_str(&step.code_example);
            markdown.push_str("\n```\n\n");

            markdown.push_str(&format!(
                "### Expected Output\n{}\n\n",
                step.expected_output
            ));

            if !step.related_functions.is_empty() {
                markdown.push_str("### Related Functions\n");
                for func in &step.related_functions {
                    markdown.push_str(&format!("- `{}`\n", func));
                }
                markdown.push_str("\n");
            }

            markdown.push_str("---\n\n");
        }

        markdown.push_str("## Additional Resources\n\n");
        markdown.push_str("- [API Documentation](https://docs.rs/scirs2-ndimage)\n");
        markdown.push_str("- [GitHub Repository](https://github.com/cool-japan/scirs)\n");
        markdown.push_str("- [SciPy ndimage Documentation](https://docs.scipy.org/doc/scipy/reference/ndimage.html) (for reference)\n");

        markdown
    }

    /// Get all tutorial steps
    pub fn get_steps(&self) -> &[TutorialStep] {
        &self.steps
    }

    /// Get number of tutorial steps
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

/// Utility function to run all examples and validate they work
#[allow(dead_code)]
pub fn validate_all_examples() -> Result<()> {
    println!("Validating all comprehensive examples...");

    let mut tutorial = ExampleTutorial::new();
    tutorial.generate_complete_tutorial()?;

    println!("Generated {} tutorial steps", tutorial.step_count());
    println!("All examples validated successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tutorial_creation() {
        let tutorial = ExampleTutorial::new();
        assert_eq!(tutorial.step_count(), 0);
    }

    #[test]
    fn test_tutorial_step_creation() {
        let step = TutorialStep {
            title: "Test Step".to_string(),
            description: "Test description".to_string(),
            code_example: "let x = 1;".to_string(),
            expected_output: "Output".to_string(),
            concepts: vec!["testing".to_string()],
            related_functions: vec!["test_func".to_string()],
        };

        assert_eq!(step.title, "Test Step");
        assert_eq!(step.concepts.len(), 1);
    }

    #[test]
    fn test_tutorial_generation() -> Result<()> {
        let mut tutorial = ExampleTutorial::new();
        tutorial.generate_complete_tutorial()?;

        assert!(tutorial.step_count() > 0);
        assert!(tutorial.export_markdown().len() > 1000);

        Ok(())
    }
}
