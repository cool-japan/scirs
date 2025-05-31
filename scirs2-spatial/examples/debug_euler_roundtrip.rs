use ndarray::array;
use scirs2_spatial::transform::Rotation;
use std::f64::consts::PI;

fn main() {
    println!("=== Debug Euler Angle Roundtrip ===\n");

    // Test each convention like the test does
    let conventions = ["xyz", "zyx", "xyx", "xzx", "yxy", "yzy", "zxz", "zyz"];

    for convention in &conventions {
        println!("Testing convention: {}", convention);

        let angles = match *convention {
            "xyz" | "zyx" => array![PI / 6.0, PI / 4.0, PI / 3.0], // 30, 45, 60 degrees
            "xyx" | "xzx" => array![0.0, PI / 4.0, 0.0], // 45 degrees around Y (middle axis)
            "yxy" | "yzy" => array![PI / 4.0, 0.0, 0.0], // 45 degrees around Y (first axis)
            "zxz" | "zyz" => array![0.0, PI / 4.0, 0.0], // 45 degrees around Z (middle axis)
            _ => unreachable!(),
        };

        println!(
            "  Input angles: [{:.3}, {:.3}, {:.3}] rad",
            angles[0], angles[1], angles[2]
        );

        // Create rotation from angles
        let rotation = Rotation::from_euler(&angles.view(), convention).unwrap();
        let quat = rotation.as_quat();
        println!(
            "  Quaternion: [{:.4}, {:.4}, {:.4}, {:.4}]",
            quat[0], quat[1], quat[2], quat[3]
        );

        // Convert back to angles
        let angles_back = rotation.as_euler(convention).unwrap();
        println!(
            "  Recovered angles: [{:.3}, {:.3}, {:.3}] rad",
            angles_back[0], angles_back[1], angles_back[2]
        );

        // Check if angles are close
        let angle_diff = (angles[0] - angles_back[0]).abs()
            + (angles[1] - angles_back[1]).abs()
            + (angles[2] - angles_back[2]).abs();
        println!("  Total angle difference: {:.2e}", angle_diff);

        // Test rotation preservation
        let point = array![1.0, 1.0, 1.0];
        let rotated1 = rotation.apply(&point.view());

        let rotation2 = Rotation::from_euler(&angles_back.view(), convention).unwrap();
        let rotated2 = rotation2.apply(&point.view());

        let rotation_diff = ((rotated1[0] - rotated2[0]).powi(2)
            + (rotated1[1] - rotated2[1]).powi(2)
            + (rotated1[2] - rotated2[2]).powi(2))
        .sqrt();
        println!("  Rotation difference: {:.2e}", rotation_diff);

        if rotation_diff > 1e-10 {
            println!("  ERROR: Rotation not preserved!");
            println!(
                "    rotated1: [{:.6}, {:.6}, {:.6}]",
                rotated1[0], rotated1[1], rotated1[2]
            );
            println!(
                "    rotated2: [{:.6}, {:.6}, {:.6}]",
                rotated2[0], rotated2[1], rotated2[2]
            );
        }

        println!();
    }
}
