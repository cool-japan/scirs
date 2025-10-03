//! Tutorial Generation System
//!
//! This module contains functions to build comprehensive tutorials and examples
//! for the SciRS2 NDImage library documentation.

use crate::documentation::types::{DocumentationSite, Example, Result, Tutorial};

impl DocumentationSite {
    /// Build comprehensive tutorials for the documentation
    pub fn build_tutorials(&mut self) -> Result<()> {
        self.tutorials = vec![
            build_getting_started_tutorial(),
            build_advanced_filtering_tutorial(),
            build_morphological_operations_tutorial(),
            build_performance_optimization_tutorial(),
        ];
        Ok(())
    }

    /// Build comprehensive examples for the documentation
    pub fn build_examples(&mut self) -> Result<()> {
        self.examples = vec![
            build_medical_imaging_example(),
            build_satellite_analysis_example(),
            build_realtime_video_example(),
            build_scientific_analysis_example(),
        ];
        Ok(())
    }
}

/// Build the "Getting Started" tutorial
pub fn build_getting_started_tutorial() -> Tutorial {
    let content = r#"
# Getting Started with SciRS2 NDImage

## Introduction

SciRS2 NDImage is a comprehensive n-dimensional image processing library for Rust that provides
high-performance implementations of common image processing operations. This tutorial will guide
you through the basics of using the library.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-ndimage = "0.1.0-beta.4"
ndarray = "0.16"
```

## Basic Usage

### Creating Arrays

```rust
use scirs2_core::ndarray::{Array2, Array3};

// Create a 2D array (image)
let image = Array2::from_elem((100, 100), 0.5f64);

// Create a 3D array (volume)
let volume = Array3::from_elem((50, 50, 50), 1.0f64);
```

### Applying Filters

```rust
use scirs2_ndimage::filters::gaussian_filter;

let filtered = gaussian_filter(&image, 2.0);
```

## Next Steps

- Learn about morphological operations
- Explore geometric transformations
- Try advanced filtering techniques
"#;

    let mut tutorial = Tutorial::beginner(
        "Getting Started with SciRS2 NDImage",
        "Introduction to n-dimensional image processing in Rust",
        content,
    );

    tutorial.add_code_example("Basic array creation and manipulation");
    tutorial.add_code_example("Simple filtering operations");

    tutorial
}

/// Build the "Advanced Filtering Techniques" tutorial
pub fn build_advanced_filtering_tutorial() -> Tutorial {
    let content = r#"
# Advanced Filtering Techniques

## Edge Detection

Edge detection is crucial for feature extraction and object recognition.

### Sobel Filter

```rust
use scirs2_ndimage::filters::sobel_filter;

let edges = sobel_filter(&image);
```

### Canny Edge Detection

```rust
use scirs2_ndimage::filters::canny_edge_detector;

let edges = canny_edge_detector(&image, 0.1, 0.2);
```

## Noise Reduction

### Bilateral Filter

Preserves edges while reducing noise:

```rust
use scirs2_ndimage::filters::bilateral_filter;

let denoised = bilateral_filter(&noisyimage, 5.0, 10.0);
```

### Non-local Means

Advanced denoising technique:

```rust
use scirs2_ndimage::filters::non_local_means;

let denoised = non_local_means(&noisyimage, 0.1, 7, 21);
```
"#;

    let mut tutorial = Tutorial::intermediate(
        "Advanced Filtering Techniques",
        "Master advanced filtering operations for noise reduction and feature enhancement",
        content,
    );

    tutorial.add_code_example("Edge detection pipeline");
    tutorial.add_code_example("Noise reduction comparison");

    tutorial
}

/// Build the "Morphological Operations" tutorial
pub fn build_morphological_operations_tutorial() -> Tutorial {
    let content = r#"
# Morphological Operations

## Understanding Mathematical Morphology

Mathematical morphology is a theory and technique for analyzing shapes and structures
in images. It's particularly useful for binary images but can be extended to grayscale.

## Basic Operations

### Erosion and Dilation

```rust
use scirs2_ndimage::morphology::{binary_erosion, binary_dilation};
use scirs2_core::ndarray::Array2;

let structure = Array2::from_elem((3, 3), true);
let eroded = binary_erosion(&binary_image, &structure);
let dilated = binary_dilation(&binary_image, &structure);
```

### Opening and Closing

```rust
use scirs2_ndimage::morphology::{binary_opening, binary_closing};

let opened = binary_opening(&binary_image, &structure);
let closed = binary_closing(&binary_image, &structure);
```

## Advanced Applications

### Skeletonization

```rust
use scirs2_ndimage::morphology::skeletonize;

let skeleton = skeletonize(&binary_image);
```

### Distance Transform

```rust
use scirs2_ndimage::morphology::distance_transform_edt;

let distances = distance_transform_edt(&binary_image);
```
"#;

    let mut tutorial = Tutorial::intermediate(
        "Morphological Operations",
        "Shape analysis and morphological transformations",
        content,
    );

    tutorial.add_code_example("Shape analysis pipeline");
    tutorial.add_code_example("Binary image processing");

    tutorial
}

/// Build the "Performance Optimization" tutorial
pub fn build_performance_optimization_tutorial() -> Tutorial {
    let content = r#"
# Performance Optimization

## SIMD Acceleration

SciRS2 NDImage automatically uses SIMD instructions when available:

```rust
// Enable SIMD features
use scirs2_ndimage::filters::gaussian_filter_simd;

let filtered = gaussian_filter_simd(&largeimage, 2.0);
```

## Parallel Processing

Large arrays are automatically processed in parallel:

```rust
use scirs2_ndimage::parallel::ParallelConfig;

// Configure parallel processing
ParallelConfig::set_num_threads(8);

// Operations automatically use parallel processing for large arrays
let result = expensive_operation(&hugeimage);
```

## GPU Acceleration

For supported operations, GPU acceleration provides significant speedup:

```rust
use scirs2_ndimage::gpu::{GpuContext, gpu_gaussian_filter};

let gpu_ctx = GpuContext::new()?;
let gpu_result = gpu_gaussian_filter(&gpu_ctx, &image, 2.0)?;
```

## Memory Optimization

### Streaming Processing

For very large datasets that don't fit in memory:

```rust
use scirs2_ndimage::streaming::StreamProcessor;

let processor = StreamProcessor::new("largeimage.tiff")?;
let result = processor.apply_filter(gaussian_filter, 2.0)?;
```

### In-place Operations

Reduce memory usage with in-place operations:

```rust
use scirs2_ndimage::filters::gaussian_filter_inplace;

gaussian_filter_inplace(&mut image, 2.0);
```
"#;

    let mut tutorial = Tutorial::advanced(
        "Performance Optimization",
        "Optimize performance with SIMD, parallel processing, and GPU acceleration",
        content,
    );

    tutorial.add_code_example("Performance benchmarking");
    tutorial.add_code_example("Memory-efficient processing");

    tutorial
}

/// Build medical imaging example
pub fn build_medical_imaging_example() -> Example {
    let code = r#"
use scirs2_ndimage::domain_specific::medical::*;
use scirs2_core::ndarray::Array3;

// Load medical volume (e.g., CT scan)
let ct_volume = Array3::from_elem((256, 256, 100), 1000.0f64);

// Apply Frangi vesselness filter for blood vessel detection
let vessels = frangi_vesselness_filter(&ct_volume, &FrangiParams::default());

// Segment bones using threshold and morphology
let bones = bone_enhancement_filter(&ct_volume, 400.0);

// Detect lung nodules
let nodules = lung_nodule_detector(&ct_volume, &NoduleDetectionParams::default());

println!("Detected {} potential nodules", nodules.len());
"#;

    Example::with_output(
        "Medical Image Processing",
        "Process medical images with specialized filters and analysis",
        code,
        "Medical",
        "Medical image processing completed successfully",
    )
}

/// Build satellite analysis example
pub fn build_satellite_analysis_example() -> Example {
    let code = r#"
use scirs2_ndimage::domain_specific::satellite::*;
use scirs2_core::ndarray::Array3;

// Multi-spectral satellite image (bands: R, G, B, NIR)
let satelliteimage = Array3::from_elem((1000, 1000, 4), 0.5f64);

// Calculate vegetation indices
let ndvi = compute_ndvi(&satelliteimage);
let ndwi = compute_ndwi(&satelliteimage);

// Water body detection
let water_mask = detect_water_bodies(&satelliteimage, 0.3);

// Cloud detection and removal
let cloud_mask = detect_clouds(&satelliteimage, &CloudDetectionParams::default());
let cloud_free = remove_clouds(&satelliteimage, &cloud_mask);

// Pan-sharpening for higher resolution
let panchromatic = Array3::from_elem((4000, 4000, 1), 0.7f64);
let sharpened = pan_sharpen(&satelliteimage, &panchromatic);

println!("Processed satellite image with {} water pixels", water_mask.iter().filter(|&&x| x).count());
"#;

    Example::with_output(
        "Satellite Image Analysis",
        "Analyze satellite imagery for environmental monitoring",
        code,
        "Remote Sensing",
        "Satellite image analysis completed",
    )
}

/// Build real-time video processing example
pub fn build_realtime_video_example() -> Example {
    let code = r#"
use scirs2_ndimage::streaming::*;
use scirs2_ndimage::features::*;
use scirs2_core::ndarray::Array3;

// Setup streaming video processor
let mut video_processor = StreamProcessor::new_video("input.mp4")?;

// Configure real-time processing pipeline
let pipeline = ProcessingPipeline::new()
    .add_filter(gaussian_filter, 1.0)
    .add_detector(corner_detector, &CornerParams::default())
    .add_tracker(object_tracker, &TrackerParams::default());

// Process frames in real-time
while let Some(frame) = video_processor.next_frame()? {
    let processed = pipeline.process(&frame)?;

    // Extract features
    let corners = detect_corners(&processed, &CornerParams::default());
    let edges = detect_edges(&processed, &EdgeParams::default());

    // Track objects across frames
    let tracked_objects = update_tracking(&corners, &edges);

    // Output processed frame
    video_processor.write_frame(&processed)?;

    println!("Frame processed: {} corners, {} edges", corners.len(), edges.len());
}
"#;

    Example::with_output(
        "Real-time Video Processing",
        "Process video frames in real-time with optimized algorithms",
        code,
        "Computer Vision",
        "Real-time video processing pipeline completed",
    )
}

/// Build scientific analysis example
pub fn build_scientific_analysis_example() -> Example {
    let code = r#"
use scirs2_ndimage::measurements::*;
use scirs2_ndimage::segmentation::*;
use scirs2_core::ndarray::Array2;

// Scientific image (e.g., microscopy, astronomy)
let scientificimage = Array2::from_elem((2048, 2048), 0.0f64);

// Advanced segmentation using watershed
let markers = find_local_maxima(&scientificimage, 10);
let segmented = watershed_segmentation(&scientificimage, &markers);

// Measure region properties
let regions = analyze_regions(&segmented);
for region in &regions {
    println!("Region {}: area={}, centroid={:?}, eccentricity={:.3}",
             region.label, region.area, region.centroid, region.eccentricity);
}

// Statistical analysis
let moments = compute_moments(&scientificimage);
let hu_moments = compute_hu_moments(&moments);

// Feature extraction for classification
let texturefeatures = extracttexturefeatures(&scientificimage);
let shapefeatures = extractshapefeatures(&segmented);

println!("Analysis complete: {} regions found", regions.len());
"#;

    Example::with_output(
        "Scientific Image Analysis",
        "Advanced analysis techniques for scientific imaging",
        code,
        "Scientific",
        "Scientific image analysis completed with region measurements",
    )
}

/// Helper function to create a comprehensive tutorial with common sections
pub fn create_comprehensive_tutorial(
    title: &str,
    description: &str,
    difficulty: &str,
) -> Tutorial {
    let mut tutorial = Tutorial::new(title, description, "", difficulty);

    // Add common sections that most tutorials would have
    let common_content = format!(
        r#"
# {}

## Overview

{}

## Prerequisites

- Basic knowledge of Rust programming
- Familiarity with the ndarray crate
- Understanding of image processing concepts

## Learning Objectives

By the end of this tutorial, you will:
- Understand the core concepts
- Be able to apply the techniques in practice
- Know how to optimize performance
- Understand common pitfalls and how to avoid them

"#,
        title, description
    );

    tutorial.content = common_content;
    tutorial
}

/// Helper function to add common examples to tutorials
pub fn add_common_tutorial_examples(tutorial: &mut Tutorial) {
    tutorial.add_code_example("Step-by-step implementation");
    tutorial.add_code_example("Performance considerations");
    tutorial.add_code_example("Common use cases");
    tutorial.add_code_example("Error handling patterns");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_getting_started_tutorial() {
        let tutorial = build_getting_started_tutorial();
        assert_eq!(tutorial.title, "Getting Started with SciRS2 NDImage");
        assert_eq!(tutorial.difficulty, "Beginner");
        assert!(!tutorial.content.is_empty());
        assert!(!tutorial.code_examples.is_empty());
    }

    #[test]
    fn test_advanced_filtering_tutorial() {
        let tutorial = build_advanced_filtering_tutorial();
        assert_eq!(tutorial.title, "Advanced Filtering Techniques");
        assert_eq!(tutorial.difficulty, "Intermediate");
        assert!(tutorial.content.contains("Edge Detection"));
        assert!(tutorial.content.contains("Noise Reduction"));
    }

    #[test]
    fn test_morphological_operations_tutorial() {
        let tutorial = build_morphological_operations_tutorial();
        assert_eq!(tutorial.title, "Morphological Operations");
        assert_eq!(tutorial.difficulty, "Intermediate");
        assert!(tutorial.content.contains("Mathematical Morphology"));
    }

    #[test]
    fn test_performance_optimization_tutorial() {
        let tutorial = build_performance_optimization_tutorial();
        assert_eq!(tutorial.title, "Performance Optimization");
        assert_eq!(tutorial.difficulty, "Advanced");
        assert!(tutorial.content.contains("SIMD"));
        assert!(tutorial.content.contains("GPU"));
    }

    #[test]
    fn test_medical_imaging_example() {
        let example = build_medical_imaging_example();
        assert_eq!(example.title, "Medical Image Processing");
        assert_eq!(example.category, "Medical");
        assert!(example.expected_output.is_some());
        assert!(example.code.contains("ct_volume"));
    }

    #[test]
    fn test_satellite_analysis_example() {
        let example = build_satellite_analysis_example();
        assert_eq!(example.title, "Satellite Image Analysis");
        assert_eq!(example.category, "Remote Sensing");
        assert!(example.code.contains("ndvi"));
        assert!(example.code.contains("water_mask"));
    }

    #[test]
    fn test_tutorial_builder() {
        let mut site = DocumentationSite::new();
        let result = site.build_tutorials();

        assert!(result.is_ok());
        assert_eq!(site.tutorials.len(), 4);

        // Check tutorial difficulties
        let difficulties: Vec<_> = site.tutorials.iter().map(|t| &t.difficulty).collect();
        assert!(difficulties.contains(&&"Beginner".to_string()));
        assert!(difficulties.contains(&&"Intermediate".to_string()));
        assert!(difficulties.contains(&&"Advanced".to_string()));
    }

    #[test]
    fn test_examples_builder() {
        let mut site = DocumentationSite::new();
        let result = site.build_examples();

        assert!(result.is_ok());
        assert_eq!(site.examples.len(), 4);

        // Check example categories
        let categories: Vec<_> = site.examples.iter().map(|e| &e.category).collect();
        assert!(categories.contains(&&"Medical".to_string()));
        assert!(categories.contains(&&"Remote Sensing".to_string()));
        assert!(categories.contains(&&"Computer Vision".to_string()));
        assert!(categories.contains(&&"Scientific".to_string()));
    }

    #[test]
    fn test_comprehensive_tutorial_creation() {
        let tutorial = create_comprehensive_tutorial(
            "Test Tutorial",
            "A test tutorial for validation",
            "Beginner",
        );

        assert_eq!(tutorial.title, "Test Tutorial");
        assert_eq!(tutorial.difficulty, "Beginner");
        assert!(tutorial.content.contains("Prerequisites"));
        assert!(tutorial.content.contains("Learning Objectives"));
    }
}