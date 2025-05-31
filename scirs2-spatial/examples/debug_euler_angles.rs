use ndarray::array;
use scirs2_spatial::transform::Rotation;
use std::f64::consts::PI;

fn main() {
    println!("=== Debugging Euler Angle Conversions ===\n");

    // Test 1: Simple 90-degree rotation around X-axis
    println!("Test 1: 90° rotation around X-axis");
    let angles_x = array![PI / 2.0, 0.0, 0.0];
    let rot_x = Rotation::from_euler(&angles_x.view(), "xyz").unwrap();
    println!("Input angles (rad): {:?}", angles_x);
    println!("Quaternion: {:?}", rot_x.as_quat());

    // Expected: Should rotate Y to Z, Z to -Y
    let test_y = array![0.0, 1.0, 0.0];
    let test_z = array![0.0, 0.0, 1.0];
    println!(
        "Rotate Y axis: {:?} -> {:?}",
        test_y,
        rot_x.apply(&test_y.view())
    );
    println!(
        "Rotate Z axis: {:?} -> {:?}",
        test_z,
        rot_x.apply(&test_z.view())
    );
    println!();

    // Test 2: Simple 90-degree rotation around Y-axis
    println!("Test 2: 90° rotation around Y-axis");
    let angles_y = array![0.0, PI / 2.0, 0.0];
    let rot_y = Rotation::from_euler(&angles_y.view(), "xyz").unwrap();
    println!("Input angles (rad): {:?}", angles_y);
    println!("Quaternion: {:?}", rot_y.as_quat());

    // Expected: Should rotate Z to X, X to -Z
    let test_x = array![1.0, 0.0, 0.0];
    println!(
        "Rotate X axis: {:?} -> {:?}",
        test_x,
        rot_y.apply(&test_x.view())
    );
    println!(
        "Rotate Z axis: {:?} -> {:?}",
        test_z,
        rot_y.apply(&test_z.view())
    );
    println!();

    // Test 3: Simple 90-degree rotation around Z-axis
    println!("Test 3: 90° rotation around Z-axis");
    let angles_z = array![0.0, 0.0, PI / 2.0];
    let rot_z = Rotation::from_euler(&angles_z.view(), "xyz").unwrap();
    println!("Input angles (rad): {:?}", angles_z);
    println!("Quaternion: {:?}", rot_z.as_quat());

    // Expected: Should rotate X to Y, Y to -X
    println!(
        "Rotate X axis: {:?} -> {:?}",
        test_x,
        rot_z.apply(&test_x.view())
    );
    println!(
        "Rotate Y axis: {:?} -> {:?}",
        test_y,
        rot_z.apply(&test_y.view())
    );
    println!();

    // Test 4: Combined rotation - manual calculation
    println!("Test 4: Manual quaternion calculation for XYZ intrinsic");
    let angles = array![PI / 4.0, PI / 3.0, PI / 6.0]; // 45°, 60°, 30°

    // Half angles
    let a = angles[0] / 2.0; // X rotation / 2
    let b = angles[1] / 2.0; // Y rotation / 2
    let c = angles[2] / 2.0; // Z rotation / 2

    let ca = a.cos();
    let sa = a.sin();
    let cb = b.cos();
    let sb = b.sin();
    let cc = c.cos();
    let sc = c.sin();

    // Current implementation in rotation.rs (lines 303-309)
    let q_current = [
        ca * cb * cc - sa * cb * sc, // w
        sa * cb * cc + ca * cb * sc, // x
        ca * sb * cc - sa * sb * sc, // y
        ca * cb * sc + sa * sb * cc, // z
    ];

    // Correct formula for intrinsic XYZ (R = Rx * Ry * Rz)
    // This is actually the formula for ZYX! The current implementation has XYZ and ZYX swapped.
    let q_correct = [
        ca * cb * cc + sa * sb * sc, // w
        sa * cb * cc - ca * sb * sc, // x
        ca * sb * cc + sa * cb * sc, // y
        ca * cb * sc - sa * sb * cc, // z
    ];

    println!(
        "Current implementation: w={:.4}, x={:.4}, y={:.4}, z={:.4}",
        q_current[0], q_current[1], q_current[2], q_current[3]
    );
    println!(
        "Correct formula:        w={:.4}, x={:.4}, y={:.4}, z={:.4}",
        q_correct[0], q_correct[1], q_correct[2], q_correct[3]
    );
    println!();

    // Test 5: Verify with known rotation
    println!("Test 5: Verify quaternion multiplication manually");

    // Individual quaternions
    let qx = [a.cos(), a.sin(), 0.0, 0.0]; // Rotation around X
    let qy = [b.cos(), 0.0, b.sin(), 0.0]; // Rotation around Y
    let qz = [c.cos(), 0.0, 0.0, c.sin()]; // Rotation around Z

    // qx * qy
    let qxy = quaternion_multiply(&qx, &qy);
    // (qx * qy) * qz
    let qxyz = quaternion_multiply(&qxy, &qz);

    println!("Step-by-step quaternion multiplication:");
    println!(
        "qx = [{:.4}, {:.4}, {:.4}, {:.4}]",
        qx[0], qx[1], qx[2], qx[3]
    );
    println!(
        "qy = [{:.4}, {:.4}, {:.4}, {:.4}]",
        qy[0], qy[1], qy[2], qy[3]
    );
    println!(
        "qz = [{:.4}, {:.4}, {:.4}, {:.4}]",
        qz[0], qz[1], qz[2], qz[3]
    );
    println!(
        "qx * qy = [{:.4}, {:.4}, {:.4}, {:.4}]",
        qxy[0], qxy[1], qxy[2], qxy[3]
    );
    println!(
        "(qx * qy) * qz = [{:.4}, {:.4}, {:.4}, {:.4}]",
        qxyz[0], qxyz[1], qxyz[2], qxyz[3]
    );

    // Compare with what's actually computed
    let rot_xyz = Rotation::from_euler(&angles.view(), "xyz").unwrap();
    let actual_quat = rot_xyz.as_quat();
    println!(
        "\nActual from Rotation::from_euler: [{:.4}, {:.4}, {:.4}, {:.4}]",
        actual_quat[0], actual_quat[1], actual_quat[2], actual_quat[3]
    );

    // Compare with ZYX convention
    println!("\nTest 6: Compare with ZYX convention");
    let rot_zyx = Rotation::from_euler(&angles.view(), "zyx").unwrap();
    let zyx_quat = rot_zyx.as_quat();

    println!(
        "XYZ quaternion: [{:.4}, {:.4}, {:.4}, {:.4}]",
        actual_quat[0], actual_quat[1], actual_quat[2], actual_quat[3]
    );
    println!(
        "ZYX quaternion: [{:.4}, {:.4}, {:.4}, {:.4}]",
        zyx_quat[0], zyx_quat[1], zyx_quat[2], zyx_quat[3]
    );

    // Apply to test vector
    let test_vec = array![1.0, 0.0, 0.0];
    println!("\nApplying to X-axis vector [1, 0, 0]:");
    println!("XYZ result: {:?}", rot_xyz.apply(&test_vec.view()));
    println!("ZYX result: {:?}", rot_zyx.apply(&test_vec.view()));

    // Test 7: Check with SciPy's known results
    println!("\nTest 7: Checking a specific case");
    let test_angles = array![PI / 4.0, 0.0, 0.0]; // 45° around X only
    let rot_test = Rotation::from_euler(&test_angles.view(), "xyz").unwrap();
    let q_test = rot_test.as_quat();
    println!("45° X rotation: angles={:?}", test_angles);
    println!(
        "Quaternion: [{:.4}, {:.4}, {:.4}, {:.4}]",
        q_test[0], q_test[1], q_test[2], q_test[3]
    );

    // Expected quaternion for 45° X rotation: [cos(22.5°), sin(22.5°), 0, 0]
    let expected_w = (PI / 8.0).cos();
    let expected_x = (PI / 8.0).sin();
    println!(
        "Expected: [{:.4}, {:.4}, 0.0000, 0.0000]",
        expected_w, expected_x
    );
}

fn quaternion_multiply(q1: &[f64; 4], q2: &[f64; 4]) -> [f64; 4] {
    // q1 = [w1, x1, y1, z1], q2 = [w2, x2, y2, z2]
    // q1 * q2 = [w1*w2 - x1*x2 - y1*y2 - z1*z2,
    //            w1*x2 + x1*w2 + y1*z2 - z1*y2,
    //            w1*y2 - x1*z2 + y1*w2 + z1*x2,
    //            w1*z2 + x1*y2 - y1*x2 + z1*w2]
    [
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
    ]
}
