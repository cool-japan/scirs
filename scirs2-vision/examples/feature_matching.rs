//! Feature Matching Example
//!
//! This example demonstrates advanced feature matching algorithms including:
//! - Brute force matching with various distance metrics
//! - FLANN-based approximate nearest neighbor matching
//! - Lowe's ratio test for robust matching
//! - Cross-check validation
//! - RANSAC-based outlier rejection

use image::{DynamicImage, ImageBuffer, Luma, Rgb, RgbImage};
use scirs2_vision::error::Result;
use scirs2_vision::feature::{
    detect_and_compute, detect_and_compute_orb, match_descriptors, match_orb_descriptors,
    matching::*, OrbConfig,
};

fn main() -> Result<()> {
    println!("Feature Matching Example");
    println!("========================");

    // Create two synthetic images with similar features but different perspectives
    let (img1, img2) = create_test_images()?;

    // Save the test images for visualization
    std::fs::create_dir_all("output").ok();
    img1.save("output/test_image1.png").ok();
    img2.save("output/test_image2.png").ok();
    println!("Created test images: output/test_image1.png and output/test_image2.png");

    // Test different matching approaches
    test_descriptor_matching(&img1, &img2)?;
    test_orb_matching(&img1, &img2)?;
    test_various_matchers(&img1, &img2)?;
    test_ransac_filtering(&img1, &img2)?;

    println!("\nExample completed successfully!");
    println!("Check the 'output' directory for visualization images.");

    Ok(())
}

/// Test SIFT-like descriptor matching
fn test_descriptor_matching(img1: &DynamicImage, img2: &DynamicImage) -> Result<()> {
    println!("\n1. SIFT-like Descriptor Matching");
    println!("=================================");

    // Detect and compute descriptors
    let descriptors1 = detect_and_compute(img1, 50, 0.01)?;
    let descriptors2 = detect_and_compute(img2, 50, 0.01)?;

    println!("Image 1: {} features", descriptors1.len());
    println!("Image 2: {} features", descriptors2.len());

    if descriptors1.is_empty() || descriptors2.is_empty() {
        println!("Warning: Not enough features detected for matching");
        return Ok(());
    }

    // Basic matching
    let basic_matches = match_descriptors(&descriptors1, &descriptors2, 0.7);
    println!("Basic matches: {}", basic_matches.len());

    // Advanced brute force matching
    let bf_matcher = BruteForceMatcher::new(BruteForceConfig {
        distance_metric: DistanceMetric::Euclidean,
        max_distance: 0.8,
        cross_check: true,
        use_ratio_test: true,
        ratio_threshold: 0.75,
    });

    let bf_matches = bf_matcher.match_descriptors(&descriptors1, &descriptors2)?;
    println!(
        "Brute force matches (with cross-check): {}",
        bf_matches.len()
    );

    // Calculate match statistics
    let stats = utils::calculate_match_statistics(&bf_matches);
    println!("Match statistics:");
    println!("  Mean distance: {:.4}", stats.mean_distance);
    println!("  Mean confidence: {:.4}", stats.mean_confidence);
    println!(
        "  Min/Max distance: {:.4}/{:.4}",
        stats.min_distance, stats.max_distance
    );

    // Filter high-confidence matches
    let high_conf_matches = utils::filter_by_confidence(&bf_matches, 0.7);
    println!(
        "High confidence matches (>0.7): {}",
        high_conf_matches.len()
    );

    // Visualize matches
    if !bf_matches.is_empty() {
        let visualization =
            visualize_matches(img1, img2, &descriptors1, &descriptors2, &bf_matches)?;
        visualization.save("output/descriptor_matches.png").ok();
        println!("Saved match visualization: output/descriptor_matches.png");
    }

    Ok(())
}

/// Test ORB descriptor matching
fn test_orb_matching(img1: &DynamicImage, img2: &DynamicImage) -> Result<()> {
    println!("\n2. ORB Descriptor Matching");
    println!("===========================");

    // Detect and compute ORB descriptors
    let orb_config = OrbConfig {
        num_features: 100,
        scale_factor: 1.2,
        num_levels: 8,
        fast_threshold: 20,
        use_harris_detector: true,
        patch_size: 31,
    };

    let orb_desc1 = detect_and_compute_orb(img1, &orb_config)?;
    let orb_desc2 = detect_and_compute_orb(img2, &orb_config)?;

    println!("ORB Image 1: {} features", orb_desc1.len());
    println!("ORB Image 2: {} features", orb_desc2.len());

    if orb_desc1.is_empty() || orb_desc2.is_empty() {
        println!("Warning: Not enough ORB features detected for matching");
        return Ok(());
    }

    // Basic ORB matching
    let basic_orb_matches = match_orb_descriptors(&orb_desc1, &orb_desc2, 80);
    println!("Basic ORB matches: {}", basic_orb_matches.len());

    // Advanced binary matching
    let orb_descriptors1: Vec<Vec<u32>> = orb_desc1.iter().map(|d| d.descriptor.clone()).collect();
    let orb_descriptors2: Vec<Vec<u32>> = orb_desc2.iter().map(|d| d.descriptor.clone()).collect();

    let bf_config = BruteForceConfig {
        distance_metric: DistanceMetric::Hamming,
        max_distance: 80.0,
        cross_check: true,
        use_ratio_test: true,
        ratio_threshold: 0.8,
    };

    let bf_matcher = BruteForceMatcher::new(bf_config);
    let binary_matches =
        bf_matcher.match_binary_descriptors(&orb_descriptors1, &orb_descriptors2)?;
    println!(
        "Binary matches (with cross-check): {}",
        binary_matches.len()
    );

    // Convert ORB matches for visualization
    if !binary_matches.is_empty() {
        let orb_keypoints1: Vec<_> = orb_desc1.iter().map(|d| d.keypoint.clone()).collect();
        let orb_keypoints2: Vec<_> = orb_desc2.iter().map(|d| d.keypoint.clone()).collect();

        let visualization = visualize_binary_matches(
            img1,
            img2,
            &orb_keypoints1,
            &orb_keypoints2,
            &binary_matches,
        )?;
        visualization.save("output/orb_matches.png").ok();
        println!("Saved ORB match visualization: output/orb_matches.png");
    }

    Ok(())
}

/// Test various matching algorithms
fn test_various_matchers(img1: &DynamicImage, img2: &DynamicImage) -> Result<()> {
    println!("\n3. Various Matching Algorithms");
    println!("===============================");

    // Get descriptors
    let descriptors1 = detect_and_compute(img1, 30, 0.01)?;
    let descriptors2 = detect_and_compute(img2, 30, 0.01)?;

    if descriptors1.is_empty() || descriptors2.is_empty() {
        println!("Warning: Not enough features for advanced matching tests");
        return Ok(());
    }

    // Test FLANN matcher
    println!("\nFLANN Matcher:");
    let flann_matcher = FlannMatcher::new(FlannConfig {
        trees: 4,
        checks: 50,
        use_lsh: false,
        hash_tables: 12,
        key_size: 20,
    });

    let flann_matches = flann_matcher.match_descriptors(&descriptors1, &descriptors2)?;
    println!("  FLANN matches: {}", flann_matches.len());

    // Test ratio test matcher
    println!("\nRatio Test Matcher:");
    let ratio_matcher = RatioTestMatcher::new(0.75, DistanceMetric::Euclidean);
    let ratio_matches = ratio_matcher.match_descriptors(&descriptors1, &descriptors2)?;
    println!("  Ratio test matches: {}", ratio_matches.len());

    // Test cross-check matcher
    println!("\nCross-Check Matcher:");
    let cross_matcher = CrossCheckMatcher::new(DistanceMetric::Euclidean);
    let cross_matches = cross_matcher.match_descriptors(&descriptors1, &descriptors2)?;
    println!("  Cross-check matches: {}", cross_matches.len());

    // Test different distance metrics
    println!("\nDistance Metrics Comparison:");
    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::Cosine,
    ] {
        let matcher = BruteForceMatcher::new(BruteForceConfig {
            distance_metric: metric,
            max_distance: 1.0,
            cross_check: false,
            use_ratio_test: false,
            ratio_threshold: 1.0,
        });

        let matches = matcher.match_descriptors(&descriptors1, &descriptors2)?;
        println!("  {:?}: {} matches", metric, matches.len());
    }

    Ok(())
}

/// Test RANSAC filtering
fn test_ransac_filtering(img1: &DynamicImage, img2: &DynamicImage) -> Result<()> {
    println!("\n4. RANSAC Outlier Rejection");
    println!("============================");

    // Get descriptors and matches
    let descriptors1 = detect_and_compute(img1, 50, 0.01)?;
    let descriptors2 = detect_and_compute(img2, 50, 0.01)?;

    if descriptors1.is_empty() || descriptors2.is_empty() {
        println!("Warning: Not enough features for RANSAC testing");
        return Ok(());
    }

    let bf_matcher = BruteForceMatcher::new_default();
    let initial_matches = bf_matcher.match_descriptors(&descriptors1, &descriptors2)?;

    if initial_matches.len() < 8 {
        println!("Warning: Not enough matches for RANSAC (need at least 8)");
        return Ok(());
    }

    println!("Initial matches: {}", initial_matches.len());

    // Extract keypoints
    let keypoints1: Vec<_> = descriptors1.iter().map(|d| d.keypoint.clone()).collect();
    let keypoints2: Vec<_> = descriptors2.iter().map(|d| d.keypoint.clone()).collect();

    // Test different RANSAC models
    for model in [GeometricModel::Homography, GeometricModel::Affine] {
        let ransac_config = RansacMatcherConfig {
            max_iterations: 1000,
            threshold: 3.0,
            min_inliers: 4,
            confidence: 0.99,
            model_type: model,
        };

        let ransac_matcher = RansacMatcher::new(ransac_config);
        let filtered_matches =
            ransac_matcher.filter_matches(&initial_matches, &keypoints1, &keypoints2)?;

        println!(
            "RANSAC {:?}: {} inliers from {} matches",
            model,
            filtered_matches.len(),
            initial_matches.len()
        );

        if !filtered_matches.is_empty() {
            let inlier_ratio = filtered_matches.len() as f32 / initial_matches.len() as f32;
            println!("  Inlier ratio: {:.2}%", inlier_ratio * 100.0);

            // Visualize RANSAC results
            let filename = format!("output/ransac_{:?}_matches.png", model).to_lowercase();
            let visualization =
                visualize_matches(img1, img2, &descriptors1, &descriptors2, &filtered_matches)?;
            visualization.save(&filename).ok();
            println!("  Saved visualization: {}", filename);
        }
    }

    Ok(())
}

/// Create two test images with known correspondences
fn create_test_images() -> Result<(DynamicImage, DynamicImage)> {
    let width = 400;
    let height = 300;

    // Create first image
    let mut img1 = ImageBuffer::new(width, height);

    // Fill with gradient background
    for y in 0..height {
        for x in 0..width {
            let intensity = ((x + y) % 256) as u8 / 4;
            img1.put_pixel(x, y, Luma([intensity]));
        }
    }

    // Add distinctive features to first image
    let features = vec![
        (50, 50, 20),   // Top-left circle
        (350, 50, 15),  // Top-right circle
        (50, 250, 18),  // Bottom-left circle
        (350, 250, 12), // Bottom-right circle
        (200, 150, 25), // Center circle
        (150, 100, 10), // Additional features
        (250, 200, 14),
        (100, 180, 16),
        (300, 120, 11),
    ];

    for (cx, cy, radius) in &features {
        draw_circle(&mut img1, *cx, *cy, *radius, 255);
    }

    // Add some corner patterns
    let corners = vec![
        (80, 80),
        (320, 80),
        (80, 220),
        (320, 220),
        (200, 100),
        (150, 200),
        (250, 120),
    ];

    for (cx, cy) in corners {
        draw_corner(&mut img1, cx, cy, 8, 200);
    }

    // Create second image with transformation
    let mut img2 = ImageBuffer::new(width, height);

    // Fill with slightly different gradient
    for y in 0..height {
        for x in 0..width {
            let intensity = ((x + y + 30) % 256) as u8 / 4;
            img2.put_pixel(x, y, Luma([intensity]));
        }
    }

    // Add transformed features (with slight translation and rotation simulation)
    let transformed_features = vec![
        (60, 60, 20), // Translated features
        (360, 60, 15),
        (60, 260, 18),
        (360, 260, 12),
        (210, 160, 25),
        (160, 110, 10),
        (260, 210, 14),
        (110, 190, 16),
        (310, 130, 11),
    ];

    for (cx, cy, radius) in &transformed_features {
        if *cx < width && *cy < height {
            draw_circle(&mut img2, *cx, *cy, *radius, 255);
        }
    }

    // Add transformed corners
    let transformed_corners = vec![
        (90, 90),
        (330, 90),
        (90, 230),
        (330, 230),
        (210, 110),
        (160, 210),
        (260, 130),
    ];

    for (cx, cy) in transformed_corners {
        if cx < width && cy < height {
            draw_corner(&mut img2, cx, cy, 8, 200);
        }
    }

    Ok((
        DynamicImage::ImageLuma8(img1),
        DynamicImage::ImageLuma8(img2),
    ))
}

/// Draw a filled circle
fn draw_circle(
    img: &mut ImageBuffer<Luma<u8>, Vec<u8>>,
    cx: u32,
    cy: u32,
    radius: u32,
    intensity: u8,
) {
    let (width, height) = img.dimensions();
    let r_sq = (radius * radius) as i32;

    for y in cy.saturating_sub(radius)..=cy.saturating_add(radius).min(height - 1) {
        for x in cx.saturating_sub(radius)..=cx.saturating_add(radius).min(width - 1) {
            let dx = x as i32 - cx as i32;
            let dy = y as i32 - cy as i32;

            if dx * dx + dy * dy <= r_sq {
                img.put_pixel(x, y, Luma([intensity]));
            }
        }
    }
}

/// Draw a corner pattern
fn draw_corner(
    img: &mut ImageBuffer<Luma<u8>, Vec<u8>>,
    cx: u32,
    cy: u32,
    size: u32,
    intensity: u8,
) {
    let (width, height) = img.dimensions();

    // Draw horizontal line
    for x in cx.saturating_sub(size)..=cx.saturating_add(size).min(width - 1) {
        if cy < height {
            img.put_pixel(x, cy, Luma([intensity]));
        }
    }

    // Draw vertical line
    for y in cy.saturating_sub(size)..=cy.saturating_add(size).min(height - 1) {
        if cx < width {
            img.put_pixel(cx, y, Luma([intensity]));
        }
    }
}

/// Visualize descriptor matches
fn visualize_matches(
    img1: &DynamicImage,
    img2: &DynamicImage,
    descriptors1: &[scirs2_vision::feature::Descriptor],
    descriptors2: &[scirs2_vision::feature::Descriptor],
    matches: &[DescriptorMatch],
) -> Result<RgbImage> {
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();
    let (w1, h1) = gray1.dimensions();
    let (w2, h2) = gray2.dimensions();

    // Create side-by-side image
    let total_width = w1 + w2;
    let total_height = h1.max(h2);
    let mut result = RgbImage::new(total_width, total_height);

    // Copy first image
    for y in 0..h1 {
        for x in 0..w1 {
            let gray_val = gray1.get_pixel(x, y)[0];
            result.put_pixel(x, y, Rgb([gray_val, gray_val, gray_val]));
        }
    }

    // Copy second image
    for y in 0..h2 {
        for x in 0..w2 {
            let gray_val = gray2.get_pixel(x, y)[0];
            result.put_pixel(x + w1, y, Rgb([gray_val, gray_val, gray_val]));
        }
    }

    // Draw matches
    for (i, m) in matches.iter().enumerate() {
        let kp1 = &descriptors1[m.query_idx].keypoint;
        let kp2 = &descriptors2[m.train_idx].keypoint;

        let x1 = kp1.x as u32;
        let y1 = kp1.y as u32;
        let x2 = (kp2.x as u32) + w1;
        let y2 = kp2.y as u32;

        // Color based on confidence
        let confidence = m.confidence.clamp(0.0, 1.0);
        let red = (255.0 * (1.0 - confidence)) as u8;
        let green = (255.0 * confidence) as u8;
        let blue = 128;

        // Draw keypoints
        if x1 < w1 && y1 < h1 {
            draw_keypoint(&mut result, x1, y1, Rgb([red, green, blue]));
        }
        if x2 < total_width && y2 < total_height {
            draw_keypoint(&mut result, x2, y2, Rgb([red, green, blue]));
        }

        // Draw connection line (only for first 20 matches to avoid clutter)
        if i < 20 && x1 < w1 && y1 < h1 && x2 < total_width && y2 < total_height {
            draw_line(&mut result, x1, y1, x2, y2, Rgb([0, 255, 255]));
        }
    }

    Ok(result)
}

/// Visualize binary descriptor matches
fn visualize_binary_matches(
    img1: &DynamicImage,
    img2: &DynamicImage,
    keypoints1: &[scirs2_vision::feature::KeyPoint],
    keypoints2: &[scirs2_vision::feature::KeyPoint],
    matches: &[DescriptorMatch],
) -> Result<RgbImage> {
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();
    let (w1, h1) = gray1.dimensions();
    let (w2, h2) = gray2.dimensions();

    // Create side-by-side image
    let total_width = w1 + w2;
    let total_height = h1.max(h2);
    let mut result = RgbImage::new(total_width, total_height);

    // Copy first image
    for y in 0..h1 {
        for x in 0..w1 {
            let gray_val = gray1.get_pixel(x, y)[0];
            result.put_pixel(x, y, Rgb([gray_val, gray_val, gray_val]));
        }
    }

    // Copy second image
    for y in 0..h2 {
        for x in 0..w2 {
            let gray_val = gray2.get_pixel(x, y)[0];
            result.put_pixel(x + w1, y, Rgb([gray_val, gray_val, gray_val]));
        }
    }

    // Draw matches
    for (i, m) in matches.iter().enumerate() {
        let kp1 = &keypoints1[m.query_idx];
        let kp2 = &keypoints2[m.train_idx];

        let x1 = kp1.x as u32;
        let y1 = kp1.y as u32;
        let x2 = (kp2.x as u32) + w1;
        let y2 = kp2.y as u32;

        // Color based on distance (lower is better for Hamming)
        let normalized_dist = (m.distance / 256.0).clamp(0.0, 1.0);
        let red = (255.0 * normalized_dist) as u8;
        let green = (255.0 * (1.0 - normalized_dist)) as u8;
        let blue = 128;

        // Draw keypoints
        if x1 < w1 && y1 < h1 {
            draw_keypoint(&mut result, x1, y1, Rgb([red, green, blue]));
        }
        if x2 < total_width && y2 < total_height {
            draw_keypoint(&mut result, x2, y2, Rgb([red, green, blue]));
        }

        // Draw connection line (only for first 15 matches)
        if i < 15 && x1 < w1 && y1 < h1 && x2 < total_width && y2 < total_height {
            draw_line(&mut result, x1, y1, x2, y2, Rgb([255, 255, 0]));
        }
    }

    Ok(result)
}

/// Draw a keypoint marker
fn draw_keypoint(img: &mut RgbImage, x: u32, y: u32, color: Rgb<u8>) {
    let (width, height) = img.dimensions();

    // Draw a small cross
    for dx in -3..=3 {
        for dy in -3..=3 {
            let px = (x as i32 + dx) as u32;
            let py = (y as i32 + dy) as u32;

            if px < width && py < height && (dx == 0 || dy == 0) {
                img.put_pixel(px, py, color);
            }
        }
    }
}

/// Draw a line between two points
fn draw_line(img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>) {
    let (width, height) = img.dimensions();

    // Simple line drawing using linear interpolation
    let dx = x2 as i32 - x1 as i32;
    let dy = y2 as i32 - y1 as i32;
    let steps = dx.abs().max(dy.abs()) as u32;

    if steps == 0 {
        return;
    }

    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = (x1 as f32 + t * dx as f32) as u32;
        let y = (y1 as f32 + t * dy as f32) as u32;

        if x < width && y < height {
            img.put_pixel(x, y, color);
        }
    }
}
