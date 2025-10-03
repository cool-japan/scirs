//! Image sequence and animation example
//!
//! This example demonstrates how to work with animated images and sequences

use scirs2_core::ndarray::Array3;
use scirs2_io::image::{
    save_image, AnimationData, ColorMode, ImageData, ImageFormat, ImageMetadata,
};
use std::error::Error;
use std::fs;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Image Sequence Example ===\n");

    // Create output directory
    fs::create_dir_all("animation_frames")?;

    println!("1. Creating a simple animation sequence...");

    // Create 10 frames of a moving ball
    let frames = 10;
    let size = 100;
    let mut frame_data = Vec::new();

    for frame in 0..frames {
        let mut image_array = Array3::zeros((size, size, 3));

        // Create a moving colored ball
        let ball_x = (frame * 10) % (size - 20);
        let ball_y = size / 2 - 10;
        let ball_radius = 10;

        for y in 0..size {
            for x in 0..size {
                let dx = x as i32 - (ball_x + ball_radius) as i32;
                let dy = y as i32 - (ball_y + ball_radius) as i32;
                let distance = ((dx * dx + dy * dy) as f32).sqrt();

                if distance <= ball_radius as f32 {
                    // Inside the ball - make it colorful
                    image_array[[y, x, 0]] = 255; // Red
                    image_array[[y, x, 1]] = ((frame as f32 / frames as f32) * 255.0) as u8; // Green varies
                    image_array[[y, x, 2]] = 100; // Blue
                } else {
                    // Background - dark blue gradient
                    image_array[[y, x, 0]] = 0;
                    image_array[[y, x, 1]] = 0;
                    image_array[[y, x, 2]] = ((y as f32 / size as f32) * 100.0) as u8;
                }
            }
        }

        let metadata = ImageMetadata {
            width: size as u32,
            height: size as u32,
            color_mode: ColorMode::RGB,
            format: ImageFormat::PNG,
            file_size: 0,
            exif: None,
        };

        let image_data = ImageData {
            data: image_array,
            metadata,
        };

        frame_data.push(image_data);
    }

    println!("✓ Created {} frames", frames);

    // Save individual frames
    println!("\n2. Saving individual frames...");
    for (i, frame) in frame_data.iter().enumerate() {
        let filename = format!("animation_frames/frame_{:03}.png", i);
        save_image(frame, &filename, Some(ImageFormat::PNG))?;
    }
    println!("✓ Saved {} individual frame files", frames);

    // Create an animation structure (for demonstration)
    println!("\n3. Creating animation data structure...");
    let animation = AnimationData {
        frames: frame_data,
        delays: vec![100; frames], // 100ms per frame
        loop_count: 0,             // Infinite loop
    };
    println!("✓ Animation structure created:");
    println!("  - Frames: {}", animation.frames.len());
    println!("  - Frame delay: {}ms", animation.delays[0]);
    println!("  - Loop count: {} (0 = infinite)", animation.loop_count);

    // Demonstrate loading a GIF (if available)
    println!("\n4. Example of loading animated GIF...");

    // First create a simple test GIF by saving our frames
    // Note: This is a demonstration - actual GIF writing would require the frames
    // to be saved as a proper animated GIF file

    println!("Note: To load actual animated GIFs, use:");
    println!("  let animation = load_animation(\"animated.gif\")?;");
    println!("  for (i, frame) in animation.frames.iter().enumerate() {{");
    println!(
        "      println!(\"Frame {{}}: {{}}x{{}}\", i, frame.metadata.width, frame.metadata.height);"
    );
    println!("  }}");

    // Create a tiled image from frames for visualization
    println!("\n5. Creating a tiled visualization of all frames...");
    let tile_cols = 5;
    let tile_rows = frames.div_ceil(tile_cols);
    let tiled_width = size * tile_cols;
    let tiled_height = size * tile_rows;
    let mut tiled_image = Array3::zeros((tiled_height, tiled_width, 3));

    for (i, frame) in animation.frames.iter().enumerate() {
        let row = i / tile_cols;
        let col = i % tile_cols;
        let start_y = row * size;
        let start_x = col * size;

        for y in 0..size {
            for x in 0..size {
                for c in 0..3 {
                    tiled_image[[start_y + y, start_x + x, c]] = frame.data[[y, x, c]];
                }
            }
        }
    }

    let tiled_metadata = ImageMetadata {
        width: tiled_width as u32,
        height: tiled_height as u32,
        color_mode: ColorMode::RGB,
        format: ImageFormat::PNG,
        file_size: 0,
        exif: None,
    };

    let tiled_data = ImageData {
        data: tiled_image,
        metadata: tiled_metadata,
    };

    save_image(&tiled_data, "animation_tiled.png", Some(ImageFormat::PNG))?;
    println!("✓ Created tiled visualization: animation_tiled.png");

    // Clean up
    println!("\n6. Cleaning up...");
    fs::remove_dir_all("animation_frames")?;
    fs::remove_file("animation_tiled.png")?;
    println!("✓ Cleaned up temporary files");

    println!("\n✓ All examples completed successfully!");

    Ok(())
}
