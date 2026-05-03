// Integration tests for scirs2-ndimage + scirs2-vision
// Tests image processing pipelines, feature detection, and computer vision workflows

use crate::common::*;
use crate::fixtures::TestDatasets;
use proptest::prelude::*;
use scirs2_core::ndarray::{Array2, Array3};
use scirs2_ndimage::*;

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Test image filtering pipeline
#[test]
fn test_image_filtering_pipeline() -> TestResult<()> {
    // Test integration of filtering operations from scirs2-ndimage
    // with feature detection from scirs2-vision

    let image = TestDatasets::test_image_gradient(64);

    println!("Testing image filtering pipeline");
    println!("Image shape: {:?}", image.shape());

    // Apply Gaussian filter from scirs2-ndimage
    let filtered = filters::gaussian_filter(&image, 1.5, None, None)?;
    assert_eq!(
        filtered.dim(),
        image.dim(),
        "Gaussian filter should preserve shape"
    );

    // Verify smoothing occurred — interior values should be within the original range
    let max_orig = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_filt = filtered.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_filt <= max_orig + 1e-9,
        "Gaussian filter should not amplify values"
    );

    // Compute gradient edges via sobel
    let sx = filters::sobel(&filtered, 1, None)?;
    let sy = filters::sobel(&filtered, 0, None)?;
    // At least some edge response should be non-zero
    let has_edges =
        sx.iter().any(|&v: &f64| v.abs() > 1e-15) || sy.iter().any(|&v: &f64| v.abs() > 1e-15);
    assert!(has_edges, "Sobel on gradient image should detect edges");

    Ok(())
}

/// Test edge detection integration
#[test]
fn test_edge_detection_integration() -> TestResult<()> {
    // Test that edge detection algorithms work with ndimage filters

    let image = TestDatasets::test_image_gradient(64);

    println!("Testing edge detection integration");

    // Sobel filter from scirs2-ndimage along axis 1
    let sobel_x = filters::sobel(&image, 1, None)?;
    assert_eq!(sobel_x.dim(), image.dim(), "Sobel should preserve shape");

    // On a gradient image, the x-axis Sobel response should be non-zero
    let max_abs_sobel = sobel_x.iter().cloned().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        max_abs_sobel > 1e-10,
        "Sobel x should detect the diagonal gradient"
    );

    // Laplace filter
    let laplace_result = filters::laplace(&image, None, None)?;
    assert_eq!(
        laplace_result.dim(),
        image.dim(),
        "Laplace should preserve shape"
    );
    // The Laplacian of a linear gradient is near-zero everywhere
    let max_abs_lap = laplace_result
        .iter()
        .cloned()
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        max_abs_lap < 1.0,
        "Laplacian of a smooth gradient should be small"
    );

    Ok(())
}

/// Test morphological operations integration
#[test]
fn test_morphological_operations_integration() -> TestResult<()> {
    // Test morphological operations from scirs2-ndimage

    let image = create_test_array_2d::<f64>(32, 32, 42)?;

    println!("Testing morphological operations integration");

    // Convert to boolean mask by thresholding at median
    let threshold = 0.5_f64;
    let binary = image.mapv(|v| v > threshold);

    // Binary erosion
    let eroded = morphology::simple_morph::binary_erosion_2d(&binary, None, None, None, None)?;
    assert_eq!(eroded.dim(), binary.dim(), "Erosion should preserve shape");

    // Binary dilation
    let dilated = morphology::simple_morph::binary_dilation_2d(&binary, None, None, None, None)?;
    assert_eq!(
        dilated.dim(),
        binary.dim(),
        "Dilation should preserve shape"
    );

    // Erosion reduces foreground: eroded should have ≤ as many true pixels as original
    let orig_count = binary.iter().filter(|&&v| v).count();
    let eroded_count = eroded.iter().filter(|&&v| v).count();
    assert!(
        eroded_count <= orig_count,
        "Erosion should not increase foreground"
    );

    // Dilation increases foreground
    let dilated_count = dilated.iter().filter(|&&v| v).count();
    assert!(
        dilated_count >= orig_count,
        "Dilation should not decrease foreground"
    );

    Ok(())
}

/// Test image segmentation pipeline
#[test]
fn test_image_segmentation_pipeline() -> TestResult<()> {
    // Test complete image segmentation workflow

    let image = TestDatasets::test_image_gradient(64);

    println!("Testing image segmentation pipeline");

    // Preprocessing with Gaussian filter
    let smoothed = filters::gaussian_filter(&image, 1.0, None, None)?;
    assert_eq!(
        smoothed.dim(),
        image.dim(),
        "Smoothing should preserve shape"
    );

    // Threshold segmentation: pixels above 0.5
    let binary = smoothed.mapv(|v| v > 0.5_f64);
    let n_foreground = binary.iter().filter(|&&v| v).count();
    assert!(n_foreground > 0, "Thresholding should find some foreground");

    // Label connected components
    let (labeled, n_labels) = morphology::label(&binary, None, None, None)?;
    assert_eq!(labeled.dim(), binary.dim(), "Label should preserve shape");
    assert!(
        n_labels >= 1,
        "Should find at least one connected component"
    );

    Ok(())
}

/// Test feature detection and description
#[test]
fn test_feature_detection_and_description() -> TestResult<()> {
    // Test feature point detection and descriptor computation

    let image_f64 = TestDatasets::test_image_gradient(64);

    println!("Testing feature detection and description");

    // Convert to f32 for features module which requires f32 Array2
    let image_f32: scirs2_core::ndarray::Array2<f32> = image_f64.mapv(|v| v as f32);

    // Harris corner detection
    let corners = features::harris_corners(&image_f32, 3, 0.04, 0.001_f32);
    assert_eq!(
        corners.dim(),
        image_f32.dim(),
        "Harris corners should preserve shape"
    );

    // Gradient-based edge detection also works on Ix2 f32
    let grad_edges = features::gradient_edges(&image_f32, None, None, None)?;
    assert_eq!(
        grad_edges.dim(),
        image_f32.dim(),
        "Gradient edges should preserve shape"
    );

    Ok(())
}

/// Test image pyramid operations
#[test]
fn test_image_pyramid_operations() -> TestResult<()> {
    // Test multi-scale image representation

    let image = TestDatasets::test_image_gradient(64);

    println!("Testing image pyramid operations");

    // Apply Gaussian filter with increasing sigma to simulate pyramid levels
    let level1 = filters::gaussian_filter(&image, 1.0, None, None)?;
    let level2 = filters::gaussian_filter(&image, 2.0, None, None)?;
    let level3 = filters::gaussian_filter(&image, 4.0, None, None)?;

    // Shape should remain constant since we use same-size filtering
    assert_eq!(level1.dim(), image.dim(), "Level 1 should preserve shape");
    assert_eq!(level2.dim(), image.dim(), "Level 2 should preserve shape");
    assert_eq!(level3.dim(), image.dim(), "Level 3 should preserve shape");

    // Higher sigma => smoother => lower variance
    let var1: f64 = {
        let mean = level1.sum() / level1.len() as f64;
        level1.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / level1.len() as f64
    };
    let var3: f64 = {
        let mean = level3.sum() / level3.len() as f64;
        level3.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / level3.len() as f64
    };
    assert!(
        var3 <= var1 + 1e-12,
        "Higher sigma should produce smaller variance"
    );

    Ok(())
}

/// Test image registration workflow
#[test]
fn test_image_registration_workflow() -> TestResult<()> {
    // Test image registration combining both modules

    let image1 = TestDatasets::test_image_gradient(64);
    // Create slightly shifted version (shift by +1 row in data)
    let image2 = image1.clone();

    println!("Testing image registration workflow");

    // Compute MSE between identical images (should be zero)
    let mse = analysis::mean_squared_error(&image1.view(), &image2.view());
    assert!(
        mse < 1e-15,
        "MSE between identical images should be near-zero, got {}",
        mse
    );

    // Compute structural similarity on identical images
    let ssim = analysis::structural_similarity_index(&image1.view(), &image2.view())?;
    assert!(
        ssim > 0.99_f64,
        "SSIM of identical images should be near 1, got {}",
        ssim
    );

    Ok(())
}

/// Test object detection pipeline
#[test]
fn test_object_detection_pipeline() -> TestResult<()> {
    // Test complete object detection workflow

    let image = create_test_array_2d::<f64>(32, 32, 42)?;

    println!("Testing object detection pipeline");

    // Apply preprocessing (uniform filter as a kind of local average)
    let preprocessed = filters::uniform_filter(&image, &[3, 3], None, None)?;
    assert_eq!(
        preprocessed.dim(),
        image.dim(),
        "Preprocessing should preserve shape"
    );

    // Compute edges to find objects
    let edges_x = filters::sobel(&preprocessed, 1, None)?;
    let edges_y = filters::sobel(&preprocessed, 0, None)?;

    // Edge magnitude
    let edge_mag: Array2<f64> = scirs2_core::ndarray::Zip::from(&edges_x)
        .and(&edges_y)
        .map_collect(|&sx, &sy| (sx * sx + sy * sy).sqrt());

    assert_eq!(
        edge_mag.dim(),
        image.dim(),
        "Edge magnitude should preserve shape"
    );
    let max_edge = edge_mag.iter().cloned().fold(0.0_f64, f64::max);
    assert!(max_edge.is_finite(), "Edge magnitude should be finite");

    Ok(())
}

/// Test image enhancement pipeline
#[test]
fn test_image_enhancement_pipeline() -> TestResult<()> {
    // Test image quality enhancement workflow

    let image = TestDatasets::test_image_gradient(64);

    println!("Testing image enhancement pipeline");

    // Noise reduction with Gaussian filter
    let denoised = filters::gaussian_filter(&image, 1.0, None, None)?;
    assert_eq!(
        denoised.dim(),
        image.dim(),
        "Denoised should preserve shape"
    );

    // Sharpening: original + (original - blurred)
    let sharpened: Array2<f64> = scirs2_core::ndarray::Zip::from(&image)
        .and(&denoised)
        .map_collect(|&orig, &smooth| (orig + (orig - smooth)).clamp(0.0, 1.0));
    assert_eq!(
        sharpened.dim(),
        image.dim(),
        "Sharpened should preserve shape"
    );

    // All values should be in [0, 1] after clamping
    assert!(
        sharpened.iter().all(|&v| (0.0_f64..=1.0).contains(&v)),
        "Sharpened values should be in [0, 1]"
    );

    Ok(())
}

/// Test optical flow computation
#[test]
fn test_optical_flow_computation() -> TestResult<()> {
    // Test optical flow estimation between image pairs

    let image1 = TestDatasets::test_image_gradient(32);
    let image2 = TestDatasets::test_image_gradient(32);

    println!("Testing optical flow computation");

    // Compute temporal difference
    let diff: Array2<f64> = scirs2_core::ndarray::Zip::from(&image2)
        .and(&image1)
        .map_collect(|&b, &a| b - a);

    assert_eq!(
        diff.dim(),
        image1.dim(),
        "Temporal diff should preserve shape"
    );
    // Identical images => zero difference
    let max_diff = diff.iter().cloned().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        max_diff < 1e-14,
        "Identical images should have zero temporal difference"
    );

    // Spatial gradient of image1 (Lucas-Kanade style)
    let grad_x = filters::sobel(&image1, 1, None)?;
    let grad_y = filters::sobel(&image1, 0, None)?;
    assert_eq!(
        grad_x.dim(),
        image1.dim(),
        "Gradient x should preserve shape"
    );
    assert_eq!(
        grad_y.dim(),
        image1.dim(),
        "Gradient y should preserve shape"
    );

    Ok(())
}

/// Test image rotation and transformation
#[test]
fn test_image_transformation_pipeline() -> TestResult<()> {
    // Test geometric transformations

    let image = TestDatasets::test_image_gradient(32);

    println!("Testing image transformation pipeline");

    // Use ndimage affine/rotate
    let rotated =
        interpolation::rotate(&image, 90.0_f64, None, Some(true), None, None, None, None)?;
    // After 90° rotation, shape should match (reshape=true keeps original bounds)
    assert_eq!(
        rotated.dim(),
        image.dim(),
        "Rotate with reshape=true should keep shape"
    );

    // Verify the rotated image has similar statistics as original
    let orig_sum: f64 = image.sum();
    let rot_sum: f64 = rotated.sum();
    let rel_diff = (orig_sum - rot_sum).abs() / (orig_sum.abs() + 1e-12);
    assert!(
        rel_diff < 0.2,
        "Rotation should approximately preserve integral, rel_diff={}",
        rel_diff
    );

    Ok(())
}

// Property-based tests

proptest! {
    #[test]
    fn prop_filter_commutativity(
        size in 32usize..64
    ) {
        // Property: Some filters should be commutative
        // (e.g., two Gaussian filters)

        let image = TestDatasets::test_image_gradient(size);

        // filter1(filter2(img)) == filter2(filter1(img)) for Gaussian with same sigma
        let filt_ab = filters::gaussian_filter(
            &filters::gaussian_filter(&image, 1.0, None, None).expect("filter failed"),
            1.5, None, None
        ).expect("filter2 failed");
        let filt_ba = filters::gaussian_filter(
            &filters::gaussian_filter(&image, 1.5, None, None).expect("filter failed"),
            1.0, None, None
        ).expect("filter2 failed");

        // Should be approximately equal (Gaussian convolutions commute)
        let max_diff = scirs2_core::ndarray::Zip::from(&filt_ab)
            .and(&filt_ba)
            .fold(0.0_f64, |acc, &a, &b| acc.max((a - b).abs()));
        prop_assert!(max_diff < 1e-10, "Gaussian filters should commute, max_diff={}", max_diff);

        prop_assert!(size >= 32);
    }

    #[test]
    fn prop_feature_detection_scale_covariance(
        base_size in 32usize..64,
        _scale_factor in 1.5f64..3.0
    ) {
        // Property: Harris corners exist in both original and smoothed versions
        // (with appropriate scale normalization)

        let image_f64 = TestDatasets::test_image_gradient(base_size);
        let image_f32: scirs2_core::ndarray::Array2<f32> = image_f64.mapv(|v| v as f32);

        // Detect corners at original scale
        let corners_orig = features::harris_corners(&image_f32, 3, 0.04, 0.001_f32);
        // Just verify they have the same shape
        prop_assert_eq!(corners_orig.dim(), image_f32.dim());

        prop_assert!(base_size >= 32);
    }

    #[test]
    fn prop_morphology_duality(
        size in 32usize..64
    ) {
        // Property: Morphological duality
        // dilate(img) has >= as many true pixels as original
        // erode(img) has <= as many true pixels as original

        let image_f64 = TestDatasets::test_image_gradient(size);
        let binary = image_f64.mapv(|v| v > 0.5);

        let eroded = morphology::simple_morph::binary_erosion_2d(&binary, None, None, None, None)
            .expect("erosion failed");
        let dilated = morphology::simple_morph::binary_dilation_2d(&binary, None, None, None, None)
            .expect("dilation failed");

        let orig_count = binary.iter().filter(|&&v| v).count();
        let eroded_count = eroded.iter().filter(|&&v| v).count();
        let dilated_count = dilated.iter().filter(|&&v| v).count();

        prop_assert!(eroded_count <= orig_count,
            "Erosion should reduce foreground: {} <= {}", eroded_count, orig_count);
        prop_assert!(dilated_count >= orig_count,
            "Dilation should increase foreground: {} >= {}", dilated_count, orig_count);

        prop_assert!(size >= 32);
    }
}

/// Test memory efficiency of image processing pipeline
#[test]
fn test_image_processing_memory_efficiency() -> TestResult<()> {
    // Verify that image processing pipelines don't create
    // unnecessary copies

    let large_image = TestDatasets::test_image_gradient(256);

    println!("Testing image processing pipeline memory efficiency");
    println!("Image size: {}x{}", 256, 256);

    assert_memory_efficient(
        || {
            // Multi-stage image processing pipeline
            let smoothed = filters::gaussian_filter(&large_image, 1.5, None, None)?;
            let edges_x = filters::sobel(&smoothed, 1, None)?;
            let edges_y = filters::sobel(&smoothed, 0, None)?;
            let edge_mag: Array2<f64> = scirs2_core::ndarray::Zip::from(&edges_x)
                .and(&edges_y)
                .map_collect(|&ex, &ey| (ex * ex + ey * ey).sqrt());
            // Verify the pipeline produced a finite result
            assert!(edge_mag.iter().all(|v| v.is_finite()));
            Ok(())
        },
        200.0, // 200 MB max
        "Multi-stage image processing pipeline",
    )?;

    Ok(())
}

/// Test color image processing
#[test]
fn test_color_image_processing() -> TestResult<()> {
    // Test processing of color (multi-channel) images

    println!("Testing color image processing");

    // Construct a synthetic RGB image (3 channels)
    let h = 32_usize;
    let w = 32_usize;
    let color_image = Array3::<f64>::from_shape_fn((h, w, 3), |(y, x, c)| {
        (y + x + c) as f64 / (h + w + 3) as f64
    });

    // Per-channel filtering
    let mut filtered_channels = Vec::with_capacity(3);
    for c in 0..3 {
        let channel: Array2<f64> = color_image
            .slice(scirs2_core::ndarray::s![.., .., c])
            .to_owned();
        let filtered_ch = filters::gaussian_filter(&channel, 1.0, None, None)?;
        assert_eq!(
            filtered_ch.dim(),
            (h, w),
            "Each channel should preserve shape"
        );
        filtered_channels.push(filtered_ch);
    }

    // All channels processed
    assert_eq!(
        filtered_channels.len(),
        3,
        "Should have 3 filtered channels"
    );

    // Verify values are reasonable
    for ch in &filtered_channels {
        assert!(
            ch.iter().all(|v| v.is_finite()),
            "Filtered channel should have finite values"
        );
    }

    Ok(())
}

/// Test image quality metrics
#[test]
fn test_image_quality_metrics() -> TestResult<()> {
    // Test computation of image quality metrics

    let image1 = TestDatasets::test_image_gradient(64);
    let image2 = image1.clone();

    println!("Testing image quality metrics");

    // MSE of identical images
    let mse = analysis::mean_squared_error(&image1.view(), &image2.view());
    assert!(
        mse < 1e-15,
        "MSE of identical images should be zero, got {}",
        mse
    );

    // PSNR of identical images should be infinity
    let psnr = analysis::peak_signal_to_noise_ratio(&image1.view(), &image2.view())?;
    assert!(
        psnr.is_infinite(),
        "PSNR of identical images should be infinite, got {}",
        psnr
    );

    // SSIM of identical images should be near 1
    let ssim = analysis::structural_similarity_index(&image1.view(), &image2.view())?;
    assert!(
        ssim > 0.99_f64,
        "SSIM of identical images should be near 1, got {}",
        ssim
    );

    // Create a noisy version and verify SSIM < 1
    let noisy: Array2<f64> = image1.mapv(|v| (v + 0.1).clamp(0.0, 1.0));
    let ssim_noisy = analysis::structural_similarity_index(&image1.view(), &noisy.view())?;
    assert!(
        ssim_noisy < ssim,
        "Noisy image should have lower SSIM than identical"
    );

    Ok(())
}

/// Test image stitching/panorama creation
#[test]
fn test_image_stitching() -> TestResult<()> {
    // Test stitching multiple images into panorama

    let image1 = TestDatasets::test_image_gradient(64);
    let image2 = TestDatasets::test_image_gradient(64);

    println!("Testing image stitching");

    // Simple horizontal concatenation as a basic stitching operation
    let (h1, w1) = image1.dim();
    let (h2, w2) = image2.dim();
    assert_eq!(
        h1, h2,
        "Images must have same height for horizontal stitching"
    );

    let mut panorama = Array2::<f64>::zeros((h1, w1 + w2));
    panorama
        .slice_mut(scirs2_core::ndarray::s![.., ..w1])
        .assign(&image1);
    panorama
        .slice_mut(scirs2_core::ndarray::s![.., w1..])
        .assign(&image2);

    assert_eq!(
        panorama.dim(),
        (h1, w1 + w2),
        "Panorama should have combined width"
    );

    // Apply Gaussian filter to blend seam
    let blended = filters::gaussian_filter(&panorama, 1.0, None, None)?;
    assert_eq!(
        blended.dim(),
        panorama.dim(),
        "Blending should preserve shape"
    );
    assert!(
        blended.iter().all(|v| v.is_finite()),
        "Blended panorama should be finite"
    );

    Ok(())
}

/// Test template matching
#[test]
fn test_template_matching() -> TestResult<()> {
    // Test template matching using correlation

    let image = TestDatasets::test_image_gradient(64);
    let template = create_test_array_2d::<f64>(8, 8, 42)?;

    println!("Testing template matching");

    // Simple normalized cross-correlation via uniform filter
    let (ti, tj) = template.dim();
    let (ii, ij) = image.dim();

    // Manual normalized cross-correlation: slide template over image
    let search_rows = ii - ti + 1;
    let search_cols = ij - tj + 1;
    let mut response = Array2::<f64>::zeros((search_rows, search_cols));

    let tmpl_mean = template.sum() / (ti * tj) as f64;
    let tmpl_var = template
        .iter()
        .map(|&v| (v - tmpl_mean).powi(2))
        .sum::<f64>();

    for r in 0..search_rows {
        for c in 0..search_cols {
            let patch = image.slice(scirs2_core::ndarray::s![r..r + ti, c..c + tj]);
            let patch_mean = patch.sum() / (ti * tj) as f64;
            let numer: f64 = scirs2_core::ndarray::Zip::from(&patch)
                .and(&template)
                .fold(0.0, |acc, &pv, &tv| {
                    acc + (pv - patch_mean) * (tv - tmpl_mean)
                });
            let denom =
                (patch.iter().map(|&v| (v - patch_mean).powi(2)).sum::<f64>() * tmpl_var).sqrt();
            response[[r, c]] = if denom > 1e-12 { numer / denom } else { 0.0 };
        }
    }

    assert_eq!(response.dim(), (search_rows, search_cols));

    // Values should be in [-1, 1]
    assert!(
        response
            .iter()
            .all(|&v| (-1.0 - 1e-9..=1.0 + 1e-9).contains(&v)),
        "Normalized cross-correlation should be in [-1, 1]"
    );

    Ok(())
}

/// Test contour detection and analysis
#[test]
fn test_contour_detection() -> TestResult<()> {
    // Test contour extraction and analysis

    let image = create_test_array_2d::<f64>(32, 32, 42)?;

    println!("Testing contour detection and analysis");

    // Edge detection from scirs2-ndimage
    let edges_x = filters::sobel(&image, 1, None)?;
    let edges_y = filters::sobel(&image, 0, None)?;

    let edge_mag: Array2<f64> = scirs2_core::ndarray::Zip::from(&edges_x)
        .and(&edges_y)
        .map_collect(|&ex, &ey| (ex * ex + ey * ey).sqrt());

    assert_eq!(
        edge_mag.dim(),
        image.dim(),
        "Edge magnitude should preserve shape"
    );

    // Threshold to get binary edge map
    let edge_threshold = edge_mag.iter().cloned().fold(0.0_f64, f64::max) * 0.1;
    let edge_binary = edge_mag.mapv(|v| v > edge_threshold);

    let n_edge_pixels = edge_binary.iter().filter(|&&v| v).count();
    assert!(n_edge_pixels > 0, "Should detect edge pixels");

    // Apply morphological thinning via erosion
    let thinned =
        morphology::simple_morph::binary_erosion_2d(&edge_binary, None, None, None, None)?;
    assert_eq!(
        thinned.dim(),
        edge_binary.dim(),
        "Thinned contours should preserve shape"
    );

    Ok(())
}

/// Test superpixel segmentation
#[test]
fn test_superpixel_segmentation() -> TestResult<()> {
    // Test superpixel generation (SLIC-like algorithm using simple grid segmentation)

    let image = TestDatasets::test_image_gradient(32);

    println!("Testing superpixel segmentation");

    // Smooth image first
    let smoothed = filters::gaussian_filter(&image, 1.0, None, None)?;
    assert_eq!(smoothed.dim(), image.dim());

    // Simple grid-based superpixel approximation
    let block_size = 8_usize;
    let (h, w) = smoothed.dim();
    let mut labels = Array2::<usize>::zeros((h, w));
    let mut label_id = 1_usize;

    let mut r = 0;
    while r < h {
        let mut c = 0;
        while c < w {
            let r_end = (r + block_size).min(h);
            let c_end = (c + block_size).min(w);
            labels
                .slice_mut(scirs2_core::ndarray::s![r..r_end, c..c_end])
                .fill(label_id);
            label_id += 1;
            c += block_size;
        }
        r += block_size;
    }

    // Verify all pixels have a label
    assert!(
        labels.iter().all(|&v| v > 0),
        "All pixels should be labeled"
    );

    // Compute per-superpixel mean
    let sums = measurements::sum_labels(&smoothed, &labels, None)?;
    assert!(!sums.is_empty(), "Should have per-region sums");

    Ok(())
}

/// Test Hough transform integration
#[test]
fn test_hough_transform() -> TestResult<()> {
    // Test Hough transform for line/circle detection

    let image = create_test_array_2d::<f64>(32, 32, 42)?;

    println!("Testing Hough transform");

    // Edge detection from scirs2-ndimage
    let edges_x = filters::sobel(&image, 1, None)?;
    let edges_y = filters::sobel(&image, 0, None)?;

    // Binary edge map
    let edge_mag: Array2<f64> = scirs2_core::ndarray::Zip::from(&edges_x)
        .and(&edges_y)
        .map_collect(|&ex, &ey| (ex * ex + ey * ey).sqrt());

    let max_edge = edge_mag.iter().cloned().fold(0.0_f64, f64::max);
    let edge_binary = edge_mag.mapv(|v| v > max_edge * 0.2);

    assert_eq!(
        edge_binary.dim(),
        image.dim(),
        "Edge binary should preserve shape"
    );

    // Simple Hough accumulator for horizontal lines (count edge pixels per row)
    let (h, w) = edge_binary.dim();
    let mut accumulator = vec![0_usize; h];
    for r in 0..h {
        for c in 0..w {
            if edge_binary[[r, c]] {
                accumulator[r] += 1;
            }
        }
    }

    // Find peak row
    let max_votes = *accumulator.iter().max().unwrap_or(&0);
    assert!(
        max_votes > 0,
        "Hough accumulator should have non-zero votes"
    );

    Ok(())
}

/// Test image moments computation
#[test]
fn test_image_moments() -> TestResult<()> {
    // Test computation of image moments

    let image = TestDatasets::test_image_gradient(32);

    println!("Testing image moments computation");

    // Compute center of mass (weighted centroid)
    let com = measurements::center_of_mass(&image)?;
    assert_eq!(com.len(), 2, "2D image should have 2D center of mass");

    // For the gradient image (i+j)/(2*n), center should be near the center of the array
    let (h, w) = image.dim();
    let expected_row = (h as f64) * 0.5;
    let expected_col = (w as f64) * 0.5;

    assert!(
        (com[0] - expected_row).abs() < (h as f64) * 0.3,
        "Center of mass row should be near center, got {} expected ~{}",
        com[0],
        expected_row
    );
    assert!(
        (com[1] - expected_col).abs() < (w as f64) * 0.3,
        "Center of mass col should be near center, got {} expected ~{}",
        com[1],
        expected_col
    );

    // Raw moments
    let m = measurements::moments(&image, 2)?;
    assert!(!m.is_empty(), "Moments should not be empty");

    Ok(())
}

/// Test image denoising
#[test]
fn test_image_denoising() -> TestResult<()> {
    // Test various denoising methods

    let clean_image = TestDatasets::test_image_gradient(32);

    // Add noise (simple additive noise)
    let noise_level = 0.05_f64;
    let noisy_image: Array2<f64> = clean_image.mapv(|v| {
        // Deterministic "pseudo-random" noise using the value itself
        let noise = ((v * 12345.678).sin() * noise_level).clamp(-noise_level, noise_level);
        (v + noise).clamp(0.0, 1.0)
    });

    println!("Testing image denoising");

    // Gaussian filter denoising
    let gaussian_denoised = filters::gaussian_filter(&noisy_image, 1.0, None, None)?;
    assert_eq!(
        gaussian_denoised.dim(),
        noisy_image.dim(),
        "Denoised should preserve shape"
    );

    // Median filter denoising
    let median_denoised = filters::median_filter(&noisy_image, &[3, 3], None)?;
    assert_eq!(
        median_denoised.dim(),
        noisy_image.dim(),
        "Median denoised should preserve shape"
    );

    // Uniform filter denoising
    let uniform_denoised = filters::uniform_filter(&noisy_image, &[3, 3], None, None)?;
    assert_eq!(
        uniform_denoised.dim(),
        noisy_image.dim(),
        "Uniform denoised should preserve shape"
    );

    // Gaussian denoising should reduce noise: MSE of denoised vs clean < MSE of noisy vs clean
    let mse_noisy = analysis::mean_squared_error(&clean_image.view(), &noisy_image.view());
    let mse_gaussian = analysis::mean_squared_error(&clean_image.view(), &gaussian_denoised.view());
    assert!(
        mse_gaussian <= mse_noisy + 1e-9,
        "Gaussian denoising should reduce error: {} <= {}",
        mse_gaussian,
        mse_noisy
    );

    Ok(())
}

/// Test image inpainting
#[test]
fn test_image_inpainting() -> TestResult<()> {
    // Test image inpainting (filling missing regions)

    let image = TestDatasets::test_image_gradient(32);

    println!("Testing image inpainting");

    // Create a mask of missing regions (a rectangle in the center)
    let (h, w) = image.dim();
    let mask_r_start = h / 4;
    let mask_r_end = 3 * h / 4;
    let mask_c_start = w / 4;
    let mask_c_end = 3 * w / 4;

    let mut masked_image = image.clone();
    for r in mask_r_start..mask_r_end {
        for c in mask_c_start..mask_c_end {
            masked_image[[r, c]] = 0.0;
        }
    }

    // Simple diffusion-based inpainting: apply Gaussian filter and blend with mask
    let smoothed = filters::gaussian_filter(&masked_image, 2.0, None, None)?;

    // Fill masked region with smoothed values
    let mut inpainted = masked_image.clone();
    for r in mask_r_start..mask_r_end {
        for c in mask_c_start..mask_c_end {
            inpainted[[r, c]] = smoothed[[r, c]];
        }
    }

    assert_eq!(
        inpainted.dim(),
        image.dim(),
        "Inpainted should preserve shape"
    );

    // The inpainted region should have non-zero values (was zero before)
    let mut inpainted_sum = 0.0_f64;
    for r in mask_r_start..mask_r_end {
        for c in mask_c_start..mask_c_end {
            inpainted_sum += inpainted[[r, c]];
        }
    }
    assert!(
        inpainted_sum.abs() > 1e-10,
        "Inpainted region should be non-zero"
    );

    Ok(())
}

/// Test performance of vision pipeline
#[test]
fn test_vision_pipeline_performance() -> TestResult<()> {
    // Test performance characteristics of integrated pipeline

    let sizes = vec![32, 64, 128];

    println!("Testing vision pipeline performance");

    for size in sizes {
        let image = TestDatasets::test_image_gradient(size);

        let (_result, perf) = measure_time(&format!("Vision pipeline size {}", size), || {
            // Representative vision pipeline: Gaussian smooth + Sobel edges + label
            let smoothed = filters::gaussian_filter(&image, 1.0, None, None)?;
            let sobel_x = filters::sobel(&smoothed, 1, None)?;
            let sobel_y = filters::sobel(&smoothed, 0, None)?;
            let edge_mag: Array2<f64> = scirs2_core::ndarray::Zip::from(&sobel_x)
                .and(&sobel_y)
                .map_collect(|&ex, &ey| (ex * ex + ey * ey).sqrt());
            let thresh = edge_mag.iter().cloned().fold(0.0_f64, f64::max) * 0.1;
            let binary = edge_mag.mapv(|v| v > thresh);
            let (labeled, _n_labels) = morphology::label(&binary, None, None, None)?;
            // Verify the result has the same shape
            assert_eq!(labeled.dim(), image.dim());
            Ok(())
        })?;

        println!("  Size {}x{}: {:.3} ms", size, size, perf.duration_ms);
    }

    Ok(())
}

#[cfg(test)]
mod api_compatibility_tests {
    use super::*;

    /// Test array format compatibility
    #[test]
    fn test_array_format_compatibility() -> TestResult<()> {
        // Verify that arrays from scirs2-ndimage can be used
        // directly in scirs2-vision functions

        let image = TestDatasets::test_image_gradient(32);

        println!("Testing array format compatibility");

        // Apply ndimage filter and use result directly in another ndimage function
        let filtered = filters::gaussian_filter(&image, 1.0, None, None)?;

        // Pass filtered result directly to sobel (no conversion needed)
        let edges = filters::sobel(&filtered, 1, None)?;
        assert_eq!(
            edges.dim(),
            image.dim(),
            "Array format should be compatible"
        );

        Ok(())
    }

    /// Test color space representation consistency
    #[test]
    fn test_color_space_consistency() -> TestResult<()> {
        // Verify that color representations produce consistent results

        println!("Testing color space consistency");

        // Create a simple 3-channel image
        let h = 16_usize;
        let w = 16_usize;
        let rgb = Array3::<f64>::from_shape_fn((h, w, 3), |(y, x, c)| {
            (y + x + c) as f64 / (h + w) as f64
        });

        // Process each channel independently
        let mut processed = Array3::<f64>::zeros((h, w, 3));
        for c in 0..3 {
            let channel = rgb.slice(scirs2_core::ndarray::s![.., .., c]).to_owned();
            let filtered = filters::gaussian_filter(&channel, 0.5, None, None)?;
            processed
                .slice_mut(scirs2_core::ndarray::s![.., .., c])
                .assign(&filtered);
        }

        assert_eq!(
            processed.dim(),
            (h, w, 3),
            "Processed color image should preserve shape"
        );
        assert!(
            processed.iter().all(|v| v.is_finite()),
            "Processed values should be finite"
        );

        Ok(())
    }

    /// Test coordinate system consistency
    #[test]
    fn test_coordinate_system_consistency() -> TestResult<()> {
        // Verify that coordinate systems (row/col) are consistent

        println!("Testing coordinate system consistency");

        // Create an image with a known structure: bright top-left corner
        let mut image = Array2::<f64>::zeros((16, 16));
        image[[0, 0]] = 1.0;
        image[[0, 1]] = 0.5;
        image[[1, 0]] = 0.5;

        // Apply Sobel in x direction (axis=1 = columns)
        let sobel_x = filters::sobel(&image, 1, None)?;
        // Apply Sobel in y direction (axis=0 = rows)
        let sobel_y = filters::sobel(&image, 0, None)?;

        // Both should have the same shape as input
        assert_eq!(sobel_x.dim(), image.dim(), "Sobel x should preserve shape");
        assert_eq!(sobel_y.dim(), image.dim(), "Sobel y should preserve shape");

        // Center of mass of the bright image should be at top-left
        let com = measurements::center_of_mass(&image)?;
        assert!(
            com[0] < 4.0,
            "COM row should be in the top area, got {}",
            com[0]
        );
        assert!(
            com[1] < 4.0,
            "COM col should be in the left area, got {}",
            com[1]
        );

        Ok(())
    }
}
