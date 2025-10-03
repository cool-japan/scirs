//! Comprehensive demonstration of vision module features

use image::GenericImageView;
use scirs2_vision::feature::{
    canny, gabor_filter, hough_lines, lbp, template_match, GaborParams, HoughParams, LBPType,
    MatchMethod, PreprocessMode,
};
use scirs2_vision::quality::{psnr, ssim, SSIMParams};
use scirs2_vision::segmentation::slic;
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("SciRS2 Vision Module - Comprehensive Demo\n");

    // Load test image
    let img = image::open("examples/input/input.jpg")?;
    println!("Loaded image: {:?}", img.dimensions());

    // 1. Edge Detection and Line Detection
    println!("\n1. Edge Detection and Line Detection");
    let edges = canny(
        &img,
        1.0,
        Some(0.05),
        Some(0.1),
        None,
        false,
        PreprocessMode::Reflect,
    )?;
    edges.save("examples/output/demo_edges.jpg")?;

    let lines = hough_lines(&edges, &HoughParams::default())?;
    println!("   Detected {} lines", lines.len());

    // 2. Texture Analysis with LBP
    println!("\n2. Texture Analysis");
    let lbp_img = lbp(
        &img,
        LBPType::Uniform {
            radius: 1.0,
            points: 8,
        },
    )?;
    lbp_img.save("examples/output/demo_lbp.jpg")?;

    // 3. Gabor Filters
    println!("\n3. Gabor Filter Bank");
    let gabor_params = GaborParams {
        wavelength: 10.0,
        orientation: std::f32::consts::PI / 4.0,
        phase: 0.0,
        aspect_ratio: 0.5,
        sigma: 5.0,
    };
    let gabor_response = gabor_filter(&img, &gabor_params)?;
    gabor_response.save("examples/output/demo_gabor.jpg")?;

    // 4. Superpixel Segmentation
    println!("\n4. Superpixel Segmentation");
    let superpixels = slic(&img, 100, 10.0, 10, 1.0)?;
    let unique_labels: std::collections::HashSet<_> = superpixels.iter().cloned().collect();
    println!("   Created {} superpixels", unique_labels.len());

    // 5. Template Matching
    println!("\n5. Template Matching");
    let template = img.crop_imm(100, 100, 50, 50);
    let match_scores = template_match(&img, &template, MatchMethod::NormalizedCrossCorrelation)?;
    let (height, width) = match_scores.dim();

    // Find best match
    let mut best_score = 0.0;
    let mut best_pos = (0, 0);
    for y in 0..height {
        for x in 0..width {
            if match_scores[[y, x]] > best_score {
                best_score = match_scores[[y, x]];
                best_pos = (x, y);
            }
        }
    }
    println!(
        "   Best match at ({}, {}) with score {:.3}",
        best_pos.0, best_pos.1, best_score
    );

    // 6. Image Quality Assessment
    println!("\n6. Image Quality Assessment");

    // Add some noise to create a distorted version
    let mut noisy_img = img.to_rgba8();
    for pixel in noisy_img.pixels_mut() {
        let noise = (scirs2_core::random::random::<f32>() - 0.5) * 20.0;
        pixel[0] = pixel[0].saturating_add(noise as u8);
        pixel[1] = pixel[1].saturating_add(noise as u8);
        pixel[2] = pixel[2].saturating_add(noise as u8);
    }
    let noisy_img = image::DynamicImage::ImageRgba8(noisy_img);

    let psnr_value = psnr(&img, &noisy_img, 255.0)?;
    let ssim_value = ssim(&img, &noisy_img, &SSIMParams::default())?;

    println!("   PSNR: {psnr_value:.2} dB");
    println!("   SSIM: {ssim_value:.4}");

    println!("\nDemo complete! Check examples/output/ for generated images.");

    Ok(())
}
