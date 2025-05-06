use ndarray::Array2;
use scirs2_ndimage::morphology::{
    binary_closing, binary_dilation, binary_erosion, binary_opening, generate_binary_structure,
    Connectivity,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Binary Morphology Examples\n");

    // Create a simple 5x5 test pattern with an object and a hole
    let mut image = Array2::from_elem((5, 5), false);

    // Create a square pattern with a hole in the center
    image[[1, 1]] = true;
    image[[1, 2]] = true;
    image[[1, 3]] = true;
    image[[2, 1]] = true;
    image[[2, 3]] = true;
    image[[3, 1]] = true;
    image[[3, 2]] = true;
    image[[3, 3]] = true;

    println!("Original image:");
    print_binary_image(&image);

    // Create a structuring element (cross shape)
    let structure = generate_binary_structure(2, Connectivity::Face)?;

    println!("\nStructuring element (face connectivity):");
    print_binary_image(&structure.into_dimensionality::<ndarray::Ix2>()?);

    // Perform erosion
    let eroded = binary_erosion(&image, None, None, None, None, None, None)?;
    println!("\nEroded image:");
    print_binary_image(&eroded);

    // Perform dilation
    let dilated = binary_dilation(&image, None, None, None, None, None, None)?;
    println!("\nDilated image:");
    print_binary_image(&dilated);

    // Perform opening (erosion followed by dilation)
    let opened = binary_opening(&image, None, None, None, None, None, None)?;
    println!("\nOpened image:");
    print_binary_image(&opened);

    // Perform closing (dilation followed by erosion)
    let closed = binary_closing(&image, None, None, None, None, None, None)?;
    println!("\nClosed image (should fill the hole):");
    print_binary_image(&closed);

    // Create another example with small isolated objects
    let mut sparse_image = Array2::from_elem((7, 7), false);

    // Place a few small objects and one larger object
    sparse_image[[1, 1]] = true; // Single pixel (should be removed by opening)

    // Small 2x2 object (may be removed by opening depending on structure)
    sparse_image[[3, 1]] = true;
    sparse_image[[3, 2]] = true;
    sparse_image[[4, 1]] = true;
    sparse_image[[4, 2]] = true;

    // Larger object
    for i in 2..6 {
        for j in 4..7 {
            sparse_image[[i, j]] = true;
        }
    }

    println!("\n\nSecond example with isolated objects:");
    print_binary_image(&sparse_image);

    // Use opening to remove small objects
    let opened_sparse = binary_opening(&sparse_image, None, Some(1), None, None, None, None)?;
    println!("\nAfter opening (small objects removed):");
    print_binary_image(&opened_sparse);

    // Create a third example with gaps in objects
    let mut gapped_image = Array2::from_elem((7, 7), false);

    // Create a line with a gap
    for i in 1..6 {
        if i != 3 {
            // Gap at position 3
            gapped_image[[i, 3]] = true;
        }
    }

    println!("\n\nThird example with a gap in a line:");
    print_binary_image(&gapped_image);

    // Use closing to bridge the gap
    let closed_gapped = binary_closing(&gapped_image, None, Some(1), None, None, None, None)?;
    println!("\nAfter closing (gap should be bridged):");
    print_binary_image(&closed_gapped);

    Ok(())
}

// Helper function to print a binary image in a readable format
fn print_binary_image(image: &Array2<bool>) {
    for i in 0..image.shape()[0] {
        for j in 0..image.shape()[1] {
            print!("{}", if image[[i, j]] { "■ " } else { "□ " });
        }
        println!();
    }
}
