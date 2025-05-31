use ndarray::array;
use scirs2_spatial::transform::Rotation;
use std::f64::consts::PI;

fn main() {
    println!("=== Verifying Different Euler Conventions ===\n");

    let angles = array![PI / 6.0, PI / 4.0, PI / 3.0]; // 30°, 45°, 60°

    println!(
        "Input angles: [{:.1}°, {:.1}°, {:.1}°]",
        angles[0] * 180.0 / PI,
        angles[1] * 180.0 / PI,
        angles[2] * 180.0 / PI
    );

    // Test XYZ convention
    let rot_xyz = Rotation::from_euler(&angles.view(), "xyz").unwrap();
    let q_xyz = rot_xyz.as_quat();
    println!(
        "\nXYZ quaternion: [{:.4}, {:.4}, {:.4}, {:.4}]",
        q_xyz[0], q_xyz[1], q_xyz[2], q_xyz[3]
    );

    // Test ZYX convention
    let rot_zyx = Rotation::from_euler(&angles.view(), "zyx").unwrap();
    let q_zyx = rot_zyx.as_quat();
    println!(
        "ZYX quaternion: [{:.4}, {:.4}, {:.4}, {:.4}]",
        q_zyx[0], q_zyx[1], q_zyx[2], q_zyx[3]
    );

    // These should be different!
    println!(
        "\nQuaternions equal? {}",
        (q_xyz[0] - q_zyx[0]).abs() < 1e-10
            && (q_xyz[1] - q_zyx[1]).abs() < 1e-10
            && (q_xyz[2] - q_zyx[2]).abs() < 1e-10
            && (q_xyz[3] - q_zyx[3]).abs() < 1e-10
    );

    // Apply to test vectors
    let test_x = array![1.0, 0.0, 0.0];
    let test_y = array![0.0, 1.0, 0.0];
    let test_z = array![0.0, 0.0, 1.0];

    println!("\nApplying XYZ rotation:");
    println!("X-axis -> {:?}", rot_xyz.apply(&test_x.view()));
    println!("Y-axis -> {:?}", rot_xyz.apply(&test_y.view()));
    println!("Z-axis -> {:?}", rot_xyz.apply(&test_z.view()));

    println!("\nApplying ZYX rotation:");
    println!("X-axis -> {:?}", rot_zyx.apply(&test_x.view()));
    println!("Y-axis -> {:?}", rot_zyx.apply(&test_y.view()));
    println!("Z-axis -> {:?}", rot_zyx.apply(&test_z.view()));

    // Check against SciPy
    println!("\nExpected from SciPy:");
    println!("XYZ: q=[0.8224, 0.2006, 0.5320, 0.0223] (w,x,y,z order)");
    println!("ZYX: q=[0.7233, 0.4397, 0.3604, 0.3919] (w,x,y,z order)");
}
