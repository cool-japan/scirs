//! Create a test image for examples to use
//!
//! This utility generates a simple test image with various features
//! that are suitable for edge detection, corner detection, and other
//! computer vision algorithms.

use image::{Rgb, RgbImage};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating test image for scirs2-vision examples...");

    // Create a 400x400 RGB image
    let mut img = RgbImage::new(400, 400);

    // Background gradient
    for y in 0..400 {
        for x in 0..400 {
            let r = (x as f32 / 400.0 * 128.0) as u8 + 64;
            let g = (y as f32 / 400.0 * 128.0) as u8 + 64;
            let b = 128;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    // Draw a white rectangle
    for y in 50..150 {
        for x in 50..150 {
            img.put_pixel(x, y, Rgb([255, 255, 255]));
        }
    }

    // Draw a black square inside the white rectangle
    for y in 75..125 {
        for x in 75..125 {
            img.put_pixel(x, y, Rgb([0, 0, 0]));
        }
    }

    // Draw a red circle
    let center_x = 300.0;
    let center_y = 100.0;
    let radius = 50.0;

    for y in 0..400 {
        for x in 0..400 {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist <= radius {
                img.put_pixel(x, y, Rgb([255, 0, 0]));
            }
        }
    }

    // Draw a green triangle
    let triangle_points = [(150.0, 300.0), (250.0, 300.0), (200.0, 200.0)];

    // Simple filled triangle using barycentric coordinates
    for y in 200..=300 {
        for x in 150..=250 {
            let p = (x as f32, y as f32);
            if point_in_triangle(p, &triangle_points) {
                img.put_pixel(x, y, Rgb([0, 255, 0]));
            }
        }
    }

    // Draw blue diagonal stripes
    for y in 250..350 {
        for x in 250..350 {
            if (x + y) % 20 < 10 {
                img.put_pixel(x, y, Rgb([0, 0, 255]));
            }
        }
    }

    // Add some noise
    use scirs2_core::random::Rng;
    let mut rng = scirs2_core::random::rng();

    for _ in 0..1000 {
        let x = rng.random_range(0..400);
        let y = rng.random_range(0..400);
        let val = rng.random_range(0..256) as u8;
        img.put_pixel(x, y, Rgb([val, val, val]));
    }

    // Save the image
    img.save("examples/input/input.jpg")?;
    println!("Test image saved to examples/input/input.jpg");

    // Also create a grayscale version
    let gray_img = image::DynamicImage::ImageRgb8(img).to_luma8();
    gray_img.save("examples/input/input_gray.jpg")?;
    println!("Grayscale test image saved to examples/input/input_gray.jpg");

    Ok(())
}

#[allow(dead_code)]
fn sign(p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) -> f32 {
    (p1.0 - p3.0) * (p2.1 - p3.1) - (p2.0 - p3.0) * (p1.1 - p3.1)
}

#[allow(dead_code)]
fn point_in_triangle(pt: (f32, f32), triangle: &[(f32, f32); 3]) -> bool {
    let d1 = sign(pt, triangle[0], triangle[1]);
    let d2 = sign(pt, triangle[1], triangle[2]);
    let d3 = sign(pt, triangle[2], triangle[0]);

    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

    !(has_neg && has_pos)
}
