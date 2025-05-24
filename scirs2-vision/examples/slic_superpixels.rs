//! Example demonstrating SLIC superpixel segmentation

use scirs2_vision::segmentation::{draw_superpixel_boundaries, slic};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Load input image
    let img = image::open("examples/input/input.jpg")?;

    println!("Running SLIC superpixel segmentation...");

    // Run SLIC with different parameters
    let test_cases = vec![
        (50, 10.0, "50 superpixels, compactness=10"),
        (100, 10.0, "100 superpixels, compactness=10"),
        (200, 10.0, "200 superpixels, compactness=10"),
        (100, 5.0, "100 superpixels, compactness=5 (more irregular)"),
        (100, 20.0, "100 superpixels, compactness=20 (more compact)"),
    ];

    for (n_segments, compactness, description) in test_cases {
        println!("\n{}", description);

        // Run SLIC
        let labels = slic(&img, n_segments, compactness, 10, 1.0)?;

        // Get unique label count
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        println!("  Created {} superpixels", unique_labels.len());

        // Draw boundaries
        let result = draw_superpixel_boundaries(&img, &labels, [255, 0, 0]);

        // Save result
        let filename = format!(
            "examples/output/slic_n{}_c{}.jpg",
            n_segments, compactness as i32
        );
        result.save(&filename)?;
        println!("  Saved to: {}", filename);
    }

    println!("\nSLIC superpixel segmentation complete!");

    Ok(())
}
