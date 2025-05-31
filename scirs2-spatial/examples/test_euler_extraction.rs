use ndarray::array;
use scirs2_spatial::transform::Rotation;
use std::f64::consts::PI;

fn main() {
    println!("=== Testing Euler Angle Extraction ===\n");

    // Test 1: Simple rotations
    println!("Test 1: Simple single-axis rotations");

    // X-axis rotation
    let angles_x = array![PI / 4.0, 0.0, 0.0];
    test_euler_roundtrip(&angles_x, "X(45°)");

    // Y-axis rotation
    let angles_y = array![0.0, PI / 4.0, 0.0];
    test_euler_roundtrip(&angles_y, "Y(45°)");

    // Z-axis rotation
    let angles_z = array![0.0, 0.0, PI / 4.0];
    test_euler_roundtrip(&angles_z, "Z(45°)");

    // Test 2: Combined rotation
    println!("\nTest 2: Combined rotation");
    let angles_combined = array![PI / 6.0, PI / 4.0, PI / 3.0];
    test_euler_roundtrip(&angles_combined, "XYZ(30°,45°,60°)");

    // Test 3: Check rotation matrix elements
    println!("\nTest 3: Rotation matrix analysis");
    let rot = Rotation::from_euler(&angles_combined.view(), "xyz").unwrap();
    let matrix = rot.as_matrix();

    println!("Rotation matrix:");
    for i in 0..3 {
        println!(
            "  [{:.4}, {:.4}, {:.4}]",
            matrix[[i, 0]],
            matrix[[i, 1]],
            matrix[[i, 2]]
        );
    }

    // For XYZ convention, the matrix elements relate to angles as:
    // R = Rz(c) * Ry(b) * Rx(a)
    let a = angles_combined[0];
    let b = angles_combined[1];
    let c = angles_combined[2];

    let ca = a.cos();
    let sa = a.sin();
    let cb = b.cos();
    let sb = b.sin();
    let cc = c.cos();
    let sc = c.sin();

    // Expected matrix elements for XYZ intrinsic (= ZYX extrinsic)
    let r00 = cb * cc;
    let r01 = -cb * sc;
    let r02 = sb;
    let r10 = sa * sb * cc + ca * sc;
    let r11 = -sa * sb * sc + ca * cc;
    let r12 = -sa * cb;
    let r20 = -ca * sb * cc + sa * sc;
    let r21 = ca * sb * sc + sa * cc;
    let r22 = ca * cb;

    println!("\nExpected matrix elements:");
    println!("  [{:.4}, {:.4}, {:.4}]", r00, r01, r02);
    println!("  [{:.4}, {:.4}, {:.4}]", r10, r11, r12);
    println!("  [{:.4}, {:.4}, {:.4}]", r20, r21, r22);

    // Extracting angles from matrix (for XYZ)
    println!("\nExtracting angles from matrix:");
    println!("  sin(b) = R[0,2] = {:.4}", matrix[[0, 2]]);
    println!("  Expected: sin({:.4}) = {:.4}", b, sb);

    let extracted_b = matrix[[0, 2]].asin();
    println!(
        "  Extracted b = {:.4} rad = {:.1}°",
        extracted_b,
        extracted_b * 180.0 / PI
    );

    if extracted_b.abs() < PI / 2.0 - 1e-6 {
        // Not in gimbal lock
        let extracted_a = (-matrix[[1, 2]]).atan2(matrix[[2, 2]]);
        let extracted_c = (-matrix[[0, 1]]).atan2(matrix[[0, 0]]);

        println!(
            "  Extracted a = {:.4} rad = {:.1}°",
            extracted_a,
            extracted_a * 180.0 / PI
        );
        println!(
            "  Extracted c = {:.4} rad = {:.1}°",
            extracted_c,
            extracted_c * 180.0 / PI
        );
    }
}

fn test_euler_roundtrip(angles: &ndarray::Array1<f64>, label: &str) {
    let rot = Rotation::from_euler(&angles.view(), "xyz").unwrap();
    let angles_back = rot.as_euler("xyz").unwrap();

    println!(
        "{}: input=[{:.4}, {:.4}, {:.4}], recovered=[{:.4}, {:.4}, {:.4}]",
        label, angles[0], angles[1], angles[2], angles_back[0], angles_back[1], angles_back[2]
    );

    // Test that the rotation is preserved
    let test_point = array![1.0, 0.0, 0.0];
    let rotated1 = rot.apply(&test_point.view());

    let rot2 = Rotation::from_euler(&angles_back.view(), "xyz").unwrap();
    let rotated2 = rot2.apply(&test_point.view());

    let diff = ((rotated1[0] - rotated2[0]).powi(2)
        + (rotated1[1] - rotated2[1]).powi(2)
        + (rotated1[2] - rotated2[2]).powi(2))
    .sqrt();

    println!("  Rotation difference: {:.2e}", diff);
}
