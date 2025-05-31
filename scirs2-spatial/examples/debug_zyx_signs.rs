use ndarray::array;
use scirs2_spatial::transform::Rotation;
use std::f64::consts::PI;

fn main() {
    println!("=== Debug ZYX Sign Issue ===\n");

    let angles = array![PI / 6.0, PI / 4.0, PI / 3.0]; // 30°, 45°, 60°
    println!(
        "Input angles: [{:.3}, {:.3}, {:.3}] rad = [30°, 45°, 60°]",
        angles[0], angles[1], angles[2]
    );

    // Create rotation
    let rot = Rotation::from_euler(&angles.view(), "zyx").unwrap();
    let quat = rot.as_quat();
    println!(
        "Quaternion: [{:.4}, {:.4}, {:.4}, {:.4}]",
        quat[0], quat[1], quat[2], quat[3]
    );

    // Get rotation matrix
    let matrix = rot.as_matrix();
    println!("\nRotation matrix:");
    for i in 0..3 {
        println!(
            "  [{:.4}, {:.4}, {:.4}]",
            matrix[[i, 0]],
            matrix[[i, 1]],
            matrix[[i, 2]]
        );
    }

    // Extract angles back
    let angles_back = rot.as_euler("zyx").unwrap();
    println!(
        "\nRecovered angles: [{:.3}, {:.3}, {:.3}] rad",
        angles_back[0], angles_back[1], angles_back[2]
    );

    // Let's manually extract to debug
    println!("\nManual extraction:");
    println!("R[0,2] = {:.4}", matrix[[0, 2]]);
    println!(
        "Y angle = asin(-R[0,2]) = asin({:.4}) = {:.3} rad",
        -matrix[[0, 2]],
        (-matrix[[0, 2]]).asin()
    );

    println!(
        "\nR[0,1] = {:.4}, R[0,0] = {:.4}",
        matrix[[0, 1]],
        matrix[[0, 0]]
    );
    println!(
        "Z angle = atan2(R[0,1], R[0,0]) = atan2({:.4}, {:.4}) = {:.3} rad",
        matrix[[0, 1]],
        matrix[[0, 0]],
        matrix[[0, 1]].atan2(matrix[[0, 0]])
    );

    println!(
        "\nR[1,2] = {:.4}, R[2,2] = {:.4}",
        matrix[[1, 2]],
        matrix[[2, 2]]
    );
    println!(
        "X angle = atan2(R[1,2], R[2,2]) = atan2({:.4}, {:.4}) = {:.3} rad",
        matrix[[1, 2]],
        matrix[[2, 2]],
        matrix[[1, 2]].atan2(matrix[[2, 2]])
    );

    // Test alternate angle representation
    println!("\nAlternate representation test:");
    let alt_angles = array![-angles[0], -angles[1], -angles[2]];
    let rot_alt = Rotation::from_euler(&alt_angles.view(), "zyx").unwrap();
    let quat_alt = rot_alt.as_quat();
    println!(
        "Alt angles: [{:.3}, {:.3}, {:.3}]",
        alt_angles[0], alt_angles[1], alt_angles[2]
    );
    println!(
        "Alt quaternion: [{:.4}, {:.4}, {:.4}, {:.4}]",
        quat_alt[0], quat_alt[1], quat_alt[2], quat_alt[3]
    );

    // Compare rotations
    let test_point = array![1.0, 0.0, 0.0];
    let rotated1 = rot.apply(&test_point.view());
    let rotated2 = rot_alt.apply(&test_point.view());
    println!("\nRotating [1,0,0]:");
    println!("Original: {:?}", rotated1);
    println!("Alternate: {:?}", rotated2);
}
