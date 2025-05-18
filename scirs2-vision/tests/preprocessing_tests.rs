use image::{DynamicImage, GrayImage, RgbImage};
use scirs2_vision::preprocessing::{
    bilateral_filter, clahe, gaussian_blur, median_filter, normalize_brightness, unsharp_mask,
};

#[test]
fn test_gaussian_blur() {
    // Create a simple test image
    let mut img = RgbImage::new(5, 5);

    // Set image to all white (255) except for the center pixel which is black (0)
    for y in 0..5 {
        for x in 0..5 {
            let color = if x == 2 && y == 2 {
                [0, 0, 0]
            } else {
                [255, 255, 255]
            };
            img.put_pixel(x, y, image::Rgb(color));
        }
    }

    let dynamic_img = DynamicImage::ImageRgb8(img);

    // Apply Gaussian blur
    let blurred = gaussian_blur(&dynamic_img, 1.0).unwrap();

    // Center pixel should be lightened by blur
    let center_value = blurred.to_luma8().get_pixel(2, 2)[0];
    assert!(
        center_value > 0,
        "Gaussian blur should have lightened the center pixel"
    );

    // Surrounding pixels should be darkened by blur
    let surround_value = blurred.to_luma8().get_pixel(1, 1)[0];
    assert!(
        surround_value < 255,
        "Gaussian blur should have darkened surrounding pixels"
    );
}

#[test]
fn test_bilateral_filter_grayscale() {
    // Create a test image with a sharp edge
    let mut img = GrayImage::new(10, 10);

    // Left half is black, right half is white
    for y in 0..10 {
        for x in 0..10 {
            let value = if x < 5 { 0 } else { 255 };
            img.put_pixel(x, y, image::Luma([value]));
        }
    }

    let dynamic_img = DynamicImage::ImageLuma8(img);

    // Apply bilateral filter
    let filtered = bilateral_filter(&dynamic_img, 3, 10.0, 100.0).unwrap();

    // Edge should be preserved - check pixels just on either side of edge
    let left_edge = filtered.to_luma8().get_pixel(4, 5)[0];
    let right_edge = filtered.to_luma8().get_pixel(5, 5)[0];

    // There should still be a significant difference at the edge
    assert!(
        right_edge - left_edge > 200,
        "Edge should be preserved by bilateral filter"
    );

    // Test invalid parameters
    assert!(bilateral_filter(&dynamic_img, 0, 10.0, 100.0).is_err());
    assert!(bilateral_filter(&dynamic_img, 2, 10.0, 100.0).is_err()); // Even numbers invalid
    assert!(bilateral_filter(&dynamic_img, 3, -1.0, 100.0).is_err());
    assert!(bilateral_filter(&dynamic_img, 3, 10.0, -1.0).is_err());
}

#[test]
fn test_bilateral_filter_color() {
    // Create a test color image with a sharp edge
    let mut img = RgbImage::new(10, 10);

    // Left half is red, right half is blue
    for y in 0..10 {
        for x in 0..10 {
            let color = if x < 5 {
                [255, 0, 0] // Red
            } else {
                [0, 0, 255] // Blue
            };
            img.put_pixel(x, y, image::Rgb(color));
        }
    }

    let dynamic_img = DynamicImage::ImageRgb8(img);

    // Apply bilateral filter
    let filtered = bilateral_filter(&dynamic_img, 3, 10.0, 100.0).unwrap();

    // Convert to RGB for checking
    let filtered_rgb = filtered.to_rgb8();

    // Edge should be preserved - check pixels just on either side of edge
    let left_edge = filtered_rgb.get_pixel(4, 5);
    let right_edge = filtered_rgb.get_pixel(5, 5);

    // Check red channel - should have significant drop at the edge
    assert!(
        left_edge[0] > 200,
        "Left edge should remain predominantly red"
    );
    assert!(right_edge[0] < 50, "Right edge should have little red");

    // Check blue channel - should have significant increase at the edge
    assert!(left_edge[2] < 50, "Left edge should have little blue");
    assert!(
        right_edge[2] > 200,
        "Right edge should remain predominantly blue"
    );
}

#[test]
fn test_median_filter() {
    // Create a test image with salt and pepper noise
    let mut img = GrayImage::new(10, 10);

    // Fill with middle gray
    for y in 0..10 {
        for x in 0..10 {
            img.put_pixel(x, y, image::Luma([128]));
        }
    }

    // Add salt and pepper noise
    let noise_coords = [(2, 2), (5, 5), (8, 8), (3, 7)];
    for (x, y) in noise_coords.iter() {
        // Add salt (white pixel)
        img.put_pixel(*x, *y, image::Luma([255]));
    }

    let dynamic_img = DynamicImage::ImageLuma8(img);

    // Apply median filter
    let filtered = median_filter(&dynamic_img, 3).unwrap();

    // Noise should be removed, check the noise coordinates
    for (x, y) in noise_coords.iter() {
        let value = filtered.to_luma8().get_pixel(*x, *y)[0];
        assert_eq!(value, 128, "Median filter should remove salt noise");
    }

    // Test invalid parameters
    assert!(median_filter(&dynamic_img, 0).is_err());
    assert!(median_filter(&dynamic_img, 2).is_err()); // Even numbers invalid
}

#[test]
fn test_unsharp_mask() {
    // Create a simple test image with a blurred edge
    let mut img = GrayImage::new(10, 10);

    // Create a gradient in the middle
    for y in 0..10 {
        for x in 0..10 {
            let value = if x < 3 {
                0
            } else if x > 6 {
                255
            } else {
                ((x - 3) as f32 * 255.0 / 3.0) as u8
            };
            img.put_pixel(x, y, image::Luma([value]));
        }
    }

    let dynamic_img = DynamicImage::ImageLuma8(img);

    // Apply unsharp mask
    let sharpened = unsharp_mask(&dynamic_img, 1.0, 1.0).unwrap();

    // Edge should be enhanced
    // The gradient in the middle should have more contrast now
    // Check pixels within the gradient area where we can see the enhancement
    let original_diff =
        dynamic_img.to_luma8().get_pixel(5, 5)[0] - dynamic_img.to_luma8().get_pixel(4, 5)[0];
    let sharpened_diff =
        sharpened.to_luma8().get_pixel(5, 5)[0] - sharpened.to_luma8().get_pixel(4, 5)[0];

    assert!(
        sharpened_diff > original_diff,
        "Unsharp mask should enhance edges"
    );

    // Test invalid parameter
    assert!(unsharp_mask(&dynamic_img, 1.0, -1.0).is_err());
}

#[test]
fn test_normalize_brightness() {
    // Create a simple test image with varying brightness
    let mut img = GrayImage::new(10, 10);

    // Fill with varying grays (50-200)
    for y in 0..10 {
        for x in 0..10 {
            let value = 50 + ((x + y) as f32 * 150.0 / 18.0) as u8; // 50-200 range
            img.put_pixel(x, y, image::Luma([value]));
        }
    }

    let dynamic_img = DynamicImage::ImageLuma8(img);

    // Apply brightness normalization to 0.2-0.8 range
    let normalized = normalize_brightness(&dynamic_img, 0.2, 0.8).unwrap();

    // Check darkest and brightest pixels
    let darkest_original = dynamic_img.to_luma8().get_pixel(0, 0)[0];
    let brightest_original = dynamic_img.to_luma8().get_pixel(9, 9)[0];

    let darkest_normalized = normalized.to_luma8().get_pixel(0, 0)[0];
    let brightest_normalized = normalized.to_luma8().get_pixel(9, 9)[0];

    // Darkest should be around 0.2*255 = ~51, brightest around 0.8*255 = ~204
    assert!(
        darkest_normalized >= 50 && darkest_normalized <= 52,
        "Normalized darkest value should be around 51, got {}",
        darkest_normalized
    );
    assert!(
        brightest_normalized >= 203 && brightest_normalized <= 205,
        "Normalized brightest value should be around 204, got {}",
        brightest_normalized
    );

    // Original dark and bright values for comparison
    assert_eq!(darkest_original, 50);
    assert_eq!(brightest_original, 200);

    // Test invalid parameters
    assert!(normalize_brightness(&dynamic_img, -0.1, 0.5).is_err());
    assert!(normalize_brightness(&dynamic_img, 0.1, 1.1).is_err());
    assert!(normalize_brightness(&dynamic_img, 0.5, 0.3).is_err());
}

#[test]
fn test_clahe() {
    // Create a test image with varying contrast
    let mut img = GrayImage::new(64, 64);

    // Create a gradient image with a low-contrast region
    for y in 0..64 {
        for x in 0..64 {
            // Left side: low contrast (values between 100-120)
            // Right side: high contrast (values between 50-200)
            let value = if x < 32 {
                100 + (x as f32 / 32.0 * 20.0) as u8
            } else {
                50 + ((x - 32) as f32 / 32.0 * 150.0) as u8
            };
            img.put_pixel(x, y, image::Luma([value]));
        }
    }

    let dynamic_img = DynamicImage::ImageLuma8(img);

    // Apply CLAHE with different parameters
    let enhanced = clahe(&dynamic_img, 8, 2.0).unwrap();

    // Test that low contrast region now has more contrast
    let left_original_min = dynamic_img.to_luma8().get_pixel(0, 0)[0];
    let left_original_max = dynamic_img.to_luma8().get_pixel(31, 0)[0];
    let left_original_diff = left_original_max - left_original_min;

    let left_enhanced_min = enhanced.to_luma8().get_pixel(0, 0)[0];
    let left_enhanced_max = enhanced.to_luma8().get_pixel(31, 0)[0];
    let left_enhanced_diff = left_enhanced_max - left_enhanced_min;

    // Enhanced image should have more contrast (greater difference between min/max)
    assert!(
        left_enhanced_diff > left_original_diff,
        "CLAHE should increase contrast in low-contrast regions"
    );

    // Test invalid parameters
    assert!(clahe(&dynamic_img, 0, 2.0).is_err());
    assert!(clahe(&dynamic_img, 8, 0.5).is_err());
}
