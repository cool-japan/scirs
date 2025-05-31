use ndarray::array;
use scirs2_spatial::transform::Rotation;
use std::f64::consts::PI;

fn main() {
    println!("=== Test Specific ZYX Case ===\n");

    // Use a simpler test case
    println!("Test 1: Single axis rotations");

    // Z rotation only
    let angles_z = array![PI / 4.0, 0.0, 0.0]; // 45° around Z
    let rot_z = Rotation::from_euler(&angles_z.view(), "zyx").unwrap();
    let recovered_z = rot_z.as_euler("zyx").unwrap();
    println!(
        "Z only: input=[{:.3}, {:.3}, {:.3}], recovered=[{:.3}, {:.3}, {:.3}]",
        angles_z[0], angles_z[1], angles_z[2], recovered_z[0], recovered_z[1], recovered_z[2]
    );

    // Y rotation only
    let angles_y = array![0.0, PI / 4.0, 0.0]; // 45° around Y
    let rot_y = Rotation::from_euler(&angles_y.view(), "zyx").unwrap();
    let recovered_y = rot_y.as_euler("zyx").unwrap();
    println!(
        "Y only: input=[{:.3}, {:.3}, {:.3}], recovered=[{:.3}, {:.3}, {:.3}]",
        angles_y[0], angles_y[1], angles_y[2], recovered_y[0], recovered_y[1], recovered_y[2]
    );

    // X rotation only
    let angles_x = array![0.0, 0.0, PI / 4.0]; // 45° around X
    let rot_x = Rotation::from_euler(&angles_x.view(), "zyx").unwrap();
    let recovered_x = rot_x.as_euler("zyx").unwrap();
    println!(
        "X only: input=[{:.3}, {:.3}, {:.3}], recovered=[{:.3}, {:.3}, {:.3}]",
        angles_x[0], angles_x[1], angles_x[2], recovered_x[0], recovered_x[1], recovered_x[2]
    );

    println!("\nTest 2: Combined small angles");
    let small_angles = array![0.1, 0.2, 0.3];
    let rot_small = Rotation::from_euler(&small_angles.view(), "zyx").unwrap();
    let recovered_small = rot_small.as_euler("zyx").unwrap();
    println!(
        "Small: input=[{:.3}, {:.3}, {:.3}], recovered=[{:.3}, {:.3}, {:.3}]",
        small_angles[0],
        small_angles[1],
        small_angles[2],
        recovered_small[0],
        recovered_small[1],
        recovered_small[2]
    );

    // Check the matrix for small angles
    let matrix = rot_small.as_matrix();
    println!("\nMatrix for small angles:");
    for i in 0..3 {
        println!(
            "  [{:.6}, {:.6}, {:.6}]",
            matrix[[i, 0]],
            matrix[[i, 1]],
            matrix[[i, 2]]
        );
    }

    // Let's trace through the extraction
    println!("\nExtraction trace:");
    let beta = (-matrix[[0, 2]]).asin();
    println!(
        "beta = asin(-R[0,2]) = asin(-{:.6}) = {:.6}",
        matrix[[0, 2]],
        beta
    );

    let alpha = matrix[[0, 1]].atan2(matrix[[0, 0]]);
    println!(
        "alpha = atan2(R[0,1], R[0,0]) = atan2({:.6}, {:.6}) = {:.6}",
        matrix[[0, 1]],
        matrix[[0, 0]],
        alpha
    );

    let gamma = matrix[[1, 2]].atan2(matrix[[2, 2]]);
    println!(
        "gamma = atan2(R[1,2], R[2,2]) = atan2({:.6}, {:.6}) = {:.6}",
        matrix[[1, 2]],
        matrix[[2, 2]],
        gamma
    );
}
