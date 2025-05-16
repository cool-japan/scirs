//! Example demonstrating distance transform functions
//!
//! This example shows how to use the distance transform functions provided by the
//! morphology module. Distance transforms are useful for measuring the distance from
//! each foreground pixel to the nearest background pixel in a binary image.

use ndarray::{Array2, IxDyn};
use scirs2_ndimage::morphology::{
    distance_transform_bf, distance_transform_cdt, distance_transform_edt,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating distance transform functions\n");

    // Create a test binary image
    let mut input = Array2::from_elem((10, 10), false);

    // Create a pattern with a clear boundary
    for i in 3..7 {
        for j in 3..7 {
            input[[i, j]] = true;
        }
    }

    println!("Input binary image:");
    print_binary_2d(&input);

    // Calculate the Euclidean distance transform
    let input_dyn = input.clone().into_dimensionality::<IxDyn>().unwrap();
    let (edt, _) = distance_transform_edt(&input_dyn, None, true, false);
    let edt = edt.unwrap().into_dimensionality::<ndarray::Ix2>().unwrap();

    println!(
        "\nEuclidean Distance Transform (distance from each true pixel to nearest false pixel):"
    );
    print_distance_2d(&edt);

    // Calculate the city block distance transform
    let (cdt, _) = distance_transform_cdt(&input_dyn, "cityblock", true, false);
    let cdt = cdt.unwrap().into_dimensionality::<ndarray::Ix2>().unwrap();

    println!("\nCity Block (Manhattan) Distance Transform:");
    print_integer_distance_2d(&cdt);

    // Calculate the chessboard distance transform
    let (chess, _) = distance_transform_cdt(&input_dyn, "chessboard", true, false);
    let chess = chess
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    println!("\nChessboard Distance Transform:");
    print_integer_distance_2d(&chess);

    // Calculate the distance transform with brute force
    let (bf_edt, _) = distance_transform_bf(&input_dyn, "euclidean", None, true, false);
    let bf_edt = bf_edt
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    println!("\nBrute Force Euclidean Distance Transform:");
    print_distance_2d(&bf_edt);

    // Demonstrate distance transform with indices
    let (_, indices) = distance_transform_edt(&input_dyn, None, false, true);
    let indices = indices.unwrap();

    println!("\nDistance Transform with Indices (showing a slice of the indices):");
    println!("For each true pixel, these indices show the coordinates of the nearest false pixel");
    println!("Dimension 0 (row) indices:");
    print_indices_2d(&indices, 0);

    println!("\nDimension 1 (column) indices:");
    print_indices_2d(&indices, 1);

    // Demonstrate custom sampling
    let sampling = [1.0, 0.5]; // Non-uniform pixel spacing
    let (custom_edt, _) = distance_transform_edt(&input_dyn, Some(&sampling), true, false);
    let custom_edt = custom_edt
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    println!("\nEuclidean Distance Transform with Custom Sampling [1.0, 0.5]:");
    println!("(Distances in column direction are halved due to sampling)");
    print_distance_2d(&custom_edt);

    Ok(())
}

// Helper function to print a 2D binary array
fn print_binary_2d(arr: &Array2<bool>) {
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            if arr[[i, j]] {
                print!("█ ");
            } else {
                print!("· ");
            }
        }
        println!();
    }
}

// Helper function to print a 2D floating-point distance array
fn print_distance_2d(arr: &Array2<f64>) {
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            let val = arr[[i, j]];
            if val < 0.01 {
                print!("0.0 ");
            } else {
                print!("{:.1} ", val);
            }
        }
        println!();
    }
}

// Helper function to print a 2D integer distance array
fn print_integer_distance_2d(arr: &Array2<i32>) {
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            let val = arr[[i, j]];
            print!("{:2} ", val);
        }
        println!();
    }
}

// Helper function to print indices from a distance transform
fn print_indices_2d(indices: &ndarray::ArrayBase<ndarray::OwnedRepr<i32>, IxDyn>, dim: usize) {
    // The indices array has an extra dimension: [dimension, rows, columns]
    for i in 0..indices.shape()[1] {
        for j in 0..indices.shape()[2] {
            print!("{:2} ", indices[[dim, i, j]]);
        }
        println!();
    }
}
