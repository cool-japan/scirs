//! Advanced filtering examples demonstrating guided filter and oriented gradients

use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use scirs2_core::ndarray::{Array2, Array3};
use scirs2_vision::error::Result;
use scirs2_vision::feature::{
    compute_gradients, sobel_edges_oriented, visualize_gradient_orientation,
};
use scirs2_vision::preprocessing::{fast_guided_filter, guided_filter, guided_filter_color};

#[allow(dead_code)]
fn main() -> Result<()> {
    // Load input image
    let img_path = "examples/input/input.jpg";
    let img = image::open(img_path).expect("Failed to load image");

    println!("Demonstrating advanced filtering techniques...");

    // 1. Guided Filter for edge-preserving smoothing
    demonstrate_guided_filter(&img)?;

    // 2. Sobel edge detection with gradient orientation
    demonstrate_oriented_gradients(&img)?;

    println!("\nAll examples completed successfully!");
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_guided_filter(img: &DynamicImage) -> Result<()> {
    println!("\n1. Guided Filter Examples:");

    // Convert to grayscale for processing
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Convert to ndarray
    let mut array = Array2::zeros((height as usize, width as usize));
    for y in 0..height {
        for x in 0..width {
            array[[y as usize, x as usize]] = gray.get_pixel(x, y)[0] as f32 / 255.0;
        }
    }

    // Apply guided filter with different parameters
    let radii = [2, 5, 10];
    let epsilons = [0.01, 0.1, 0.5];

    for (&radius, &epsilon) in radii.iter().zip(epsilons.iter()) {
        println!("  - Applying guided filter with radius={radius}, epsilon={epsilon}");

        let filtered = guided_filter(&array, &array, radius, epsilon)?;

        // Convert back to image
        let mut output_img = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val = (filtered[[y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
                output_img.put_pixel(x, y, Luma([val]));
            }
        }

        let output_path = format!("examples/output/guided_filter_r{radius}_e{epsilon}.png");
        output_img.save(&output_path).expect("Failed to save image");
        println!("    Saved: {output_path}");
    }

    // Demonstrate fast guided filter
    println!("  - Testing fast guided filter (with subsampling)");
    let fast_filtered = fast_guided_filter(&array, &array, 10, 0.1, 4)?;

    let mut fast_output = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let val = (fast_filtered[[y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
            fast_output.put_pixel(x, y, Luma([val]));
        }
    }
    fast_output
        .save("examples/output/fast_guided_filter.png")
        .expect("Failed to save fast guided filter output");

    // Demonstrate color guided filter
    if img.color() != image::ColorType::L8 {
        println!("  - Applying guided filter to color image");

        let rgb = img.to_rgb8();
        let mut color_array = Array3::zeros((height as usize, width as usize, 3));

        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    color_array[[y as usize, x as usize, c]] = pixel[c] as f32 / 255.0;
                }
            }
        }

        let color_filtered = guided_filter_color(&color_array, &color_array, 5, 0.1)?;

        let mut color_output = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r =
                    (color_filtered[[y as usize, x as usize, 0]] * 255.0).clamp(0.0, 255.0) as u8;
                let g =
                    (color_filtered[[y as usize, x as usize, 1]] * 255.0).clamp(0.0, 255.0) as u8;
                let b =
                    (color_filtered[[y as usize, x as usize, 2]] * 255.0).clamp(0.0, 255.0) as u8;
                color_output.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        color_output
            .save("examples/output/guided_filter_color.png")
            .expect("Failed to save color guided filter output");
    }

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_oriented_gradients(img: &DynamicImage) -> Result<()> {
    println!("\n2. Oriented Gradient Examples:");

    // Basic Sobel edge detection
    println!("  - Computing Sobel edges");
    let (edges, _orientations) = sobel_edges_oriented(img, 0.1, true)?;
    edges
        .save("examples/output/sobel_edges.png")
        .expect("Failed to save Sobel edges");

    // Compute raw gradients
    println!("  - Computing gradient magnitude and orientation");
    let (magnitude, orientation) = compute_gradients(img)?;

    // Save gradient magnitude as image
    let (height, width) = magnitude.dim();
    let max_mag = magnitude.iter().fold(0.0f32, |a, &b| a.max(b));

    let mut mag_img = ImageBuffer::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let val = ((magnitude[[y, x]] / max_mag) * 255.0).clamp(0.0, 255.0) as u8;
            mag_img.put_pixel(x as u32, y as u32, Luma([val]));
        }
    }
    mag_img
        .save("examples/output/gradient_magnitude.png")
        .expect("Failed to save gradient magnitude");

    // Visualize gradient orientation with color coding
    println!("  - Creating color-coded orientation visualization");
    let orientation_vis = visualize_gradient_orientation(&magnitude, &orientation, 0.05)?;
    orientation_vis
        .save("examples/output/gradient_orientation.png")
        .expect("Failed to save gradient orientation");

    // Create orientation visualization with different thresholds
    let thresholds = vec![0.02, 0.1, 0.2];
    for threshold in thresholds {
        let vis = visualize_gradient_orientation(&magnitude, &orientation, threshold)?;
        let path = format!("examples/output/gradient_orientation_t{threshold}.png");
        vis.save(&path)
            .expect("Failed to save orientation visualization");
        println!("    Saved orientation visualization with threshold={threshold}: {path}");
    }

    Ok(())
}
