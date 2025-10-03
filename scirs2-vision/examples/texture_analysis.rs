//! Example demonstrating texture analysis using Local Binary Patterns (LBP)

use image::{DynamicImage, GenericImageView};
use scirs2_vision::feature::{lbp, lbp_histogram, multi_scale_lbp, LBPType};
use statrs::statistics::Statistics;
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    // Load input image
    let img = image::open("examples/input/input.jpg")?;

    println!("Running texture analysis with Local Binary Patterns...\n");

    // Test different LBP variants
    println!("1. Original LBP (3x3 neighborhood)");
    let lbp_original = lbp(&img, LBPType::Original)?;
    lbp_original.save("examples/output/lbp_original.jpg")?;
    let hist = lbp_histogram(&lbp_original, 256, true)?;
    println!("   Histogram entropy: {:.4}", compute_entropy(&hist));

    println!("\n2. Extended LBP (radius=1.5, 8 points)");
    let lbp_extended = lbp(
        &img,
        LBPType::Extended {
            radius: 1.5,
            points: 8,
        },
    )?;
    lbp_extended.save("examples/output/lbp_extended.jpg")?;

    println!("\n3. Uniform LBP (radius=2.0, 16 points)");
    let lbp_uniform = lbp(
        &img,
        LBPType::Uniform {
            radius: 2.0,
            points: 16,
        },
    )?;
    lbp_uniform.save("examples/output/lbp_uniform.jpg")?;

    println!("\n4. Rotation Invariant LBP (radius=1.0, 8 points)");
    let lbp_rotation = lbp(
        &img,
        LBPType::RotationInvariant {
            radius: 1.0,
            points: 8,
        },
    )?;
    lbp_rotation.save("examples/output/lbp_rotation_invariant.jpg")?;

    // Multi-scale LBP for robust texture features
    println!("\n5. Multi-scale LBP features");
    let scales = vec![(1.0, 8), (2.0, 16), (3.0, 24)];
    let features = multi_scale_lbp(&img, &scales)?;
    println!("   Feature vector length: {}", features.len());
    println!("   Feature mean: {:.4}", features.mean().unwrap_or(0.0));
    println!(
        "   Feature std: {:.4}",
        features.std_axis(scirs2_core::ndarray::Axis(0), 0.0)
    );

    // Compare textures in different regions
    println!("\n6. Texture comparison between image regions");
    compare_regions(&img)?;

    println!("\nTexture analysis complete!");
    Ok(())
}

/// Compute entropy of a histogram
#[allow(dead_code)]
fn compute_entropy(hist: &scirs2_core::ndarray::Array1<f32>) -> f32 {
    let mut entropy = 0.0;
    for &p in hist.iter() {
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Compare textures in different regions of the image
#[allow(dead_code)]
fn compare_regions(img: &DynamicImage) -> Result<(), Box<dyn Error>> {
    let (width, height) = img.dimensions();

    // Define regions (top-left, top-right, bottom-left, bottom-right)
    let regions = vec![
        ("Top-left", 0, 0, width / 2, height / 2),
        ("Top-right", width / 2, 0, width / 2, height / 2),
        ("Bottom-left", 0, height / 2, width / 2, height / 2),
        ("Bottom-right", width / 2, height / 2, width / 2, height / 2),
    ];

    let mut histograms = Vec::new();

    for (name, x, y, w, h) in &regions {
        // Extract region
        let region = img.crop_imm(*x, *y, *w, *h);

        // Compute LBP
        let lbp_img = lbp(
            &region,
            LBPType::Uniform {
                radius: 1.0,
                points: 8,
            },
        )?;

        // Compute histogram
        let hist = lbp_histogram(&lbp_img, 59, true)?; // 58 uniform patterns + 1
        let entropy = compute_entropy(&hist);

        println!("   {name} - Entropy: {entropy:.4}");
        histograms.push(hist);
    }

    // Compute similarity between regions using chi-square distance
    println!("\n   Texture similarity (chi-square distance):");
    for i in 0..regions.len() {
        for j in i + 1..regions.len() {
            let distance = chi_square_distance(&histograms[i], &histograms[j]);
            println!(
                "     {} vs {} = {:.4}",
                regions[i].0, regions[j].0, distance
            );
        }
    }

    Ok(())
}

/// Compute chi-square distance between histograms
#[allow(dead_code)]
fn chi_square_distance(
    hist1: &scirs2_core::ndarray::Array1<f32>,
    hist2: &scirs2_core::ndarray::Array1<f32>,
) -> f32 {
    let mut distance = 0.0;
    for i in 0..hist1.len() {
        let sum = hist1[i] + hist2[i];
        if sum > 0.0 {
            distance += (hist1[i] - hist2[i]).powi(2) / sum;
        }
    }
    distance / 2.0
}
