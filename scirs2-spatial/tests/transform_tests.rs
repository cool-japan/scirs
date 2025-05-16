use approx::assert_relative_eq;
use ndarray::{array, Array1};
use std::f64::consts::PI;

use scirs2_spatial::transform::{Rotation, RotationSpline, Slerp};

#[test]
fn test_rotation_basic() {
    // Create a rotation from various representations
    let rot_identity = Rotation::identity();

    // Apply to a test point
    let point = array![1.0, 2.0, 3.0];
    let rotated = rot_identity.apply(&point.view());

    // Identity rotation should return the same point
    assert_relative_eq!(rotated[0], point[0], epsilon = 1e-10);
    assert_relative_eq!(rotated[1], point[1], epsilon = 1e-10);
    assert_relative_eq!(rotated[2], point[2], epsilon = 1e-10);

    // Create a 90-degree rotation around Z
    let rot_z = Rotation::from_euler(&array![0.0, 0.0, PI / 2.0], "xyz").unwrap();
    let rotated_z = rot_z.apply(&array![1.0, 0.0, 0.0].view());

    // Should map [1, 0, 0] to approximately [0, 1, 0]
    assert_relative_eq!(rotated_z[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_z[1], 1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_z[2], 0.0, epsilon = 1e-10);

    // Test inverse rotation
    let rot_z_inv = rot_z.inv();
    let point_back = rot_z_inv.apply(&rotated_z.view());

    // Should get back original point [1, 0, 0]
    assert_relative_eq!(point_back[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(point_back[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(point_back[2], 0.0, epsilon = 1e-10);
}

#[test]
fn test_rotation_euler_conventions() {
    let test_conventions = ["xyz", "zyx", "xyx", "xzx", "yxy", "yzy", "zxz", "zyz"];

    // Test each Euler angle convention
    for &convention in &test_conventions {
        // Create a simple rotation using this convention
        let angles = match convention {
            "xyz" | "zyx" => array![PI / 4.0, 0.0, 0.0], // 45 degrees around X
            "xyx" | "xzx" => array![0.0, PI / 4.0, 0.0], // 45 degrees around Y (middle axis)
            "yxy" | "yzy" => array![PI / 4.0, 0.0, 0.0], // 45 degrees around Y (first axis)
            "zxz" | "zyz" => array![0.0, PI / 4.0, 0.0], // 45 degrees around Z (middle axis)
            _ => unreachable!(),
        };

        let rotation = Rotation::from_euler(&angles.view(), convention).unwrap();

        // Get back the Euler angles
        let angles_back = rotation.as_euler(convention).unwrap();

        // For simple rotations, the angles should be recoverable
        // (allowing for different but equivalent representations)
        let point = array![1.0, 1.0, 1.0];
        let rotated1 = rotation.apply(&point.view());

        let rotation2 = Rotation::from_euler(&angles_back.view(), convention).unwrap();
        let rotated2 = rotation2.apply(&point.view());

        // Both rotations should produce the same result
        assert_relative_eq!(rotated1[0], rotated2[0], epsilon = 1e-10);
        assert_relative_eq!(rotated1[1], rotated2[1], epsilon = 1e-10);
        assert_relative_eq!(rotated1[2], rotated2[2], epsilon = 1e-10);
    }
}

#[test]
fn test_slerp_basic() {
    // Create two rotations
    let rot1 = Rotation::identity();
    let rot2 = Rotation::from_euler(&array![0.0, 0.0, PI], "xyz").unwrap(); // 180 degrees around Z

    // Create a slerp interpolator
    let slerp = Slerp::new(rot1, rot2).unwrap();

    // Test interpolation at various parameters
    let test_point = array![1.0, 0.0, 0.0];

    // Test t=0 and t=1 (endpoints)
    let rot_0 = slerp.interpolate(0.0);
    let rot_1 = slerp.interpolate(1.0);

    let rotated_0 = rot_0.apply(&test_point.view());
    let rotated_1 = rot_1.apply(&test_point.view());

    // Should match the original rotations
    assert_relative_eq!(rotated_0[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_0[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_0[2], 0.0, epsilon = 1e-10);

    assert_relative_eq!(rotated_1[0], -1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_1[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_1[2], 0.0, epsilon = 1e-10);

    // Test midpoint (should be 90 degrees around Z)
    let rot_half = slerp.interpolate(0.5);
    let rotated_half = rot_half.apply(&test_point.view());

    assert_relative_eq!(rotated_half[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_half[1], 1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_half[2], 0.0, epsilon = 1e-10);
}

#[test]
fn test_rotation_spline_slerp() {
    // Create a rotation spline
    let rotations = vec![
        Rotation::identity(),
        Rotation::from_euler(&array![0.0, 0.0, PI / 2.0], "xyz").unwrap(),
        Rotation::from_euler(&array![0.0, 0.0, PI], "xyz").unwrap(),
    ];
    let times = vec![0.0, 1.0, 2.0];

    let spline = RotationSpline::new(&rotations, &times).unwrap();

    // Test that the default interpolation type is "slerp"
    assert_eq!(spline.interpolation_type(), "slerp");

    // Test interpolation at key times
    let test_point = array![1.0, 0.0, 0.0];

    let rot_0 = spline.interpolate(0.0);
    let rot_1 = spline.interpolate(1.0);
    let rot_2 = spline.interpolate(2.0);

    let rotated_0 = rot_0.apply(&test_point.view());
    let rotated_1 = rot_1.apply(&test_point.view());
    let rotated_2 = rot_2.apply(&test_point.view());

    // Check that they match the original rotations
    assert_relative_eq!(rotated_0[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_0[1], 0.0, epsilon = 1e-10);

    assert_relative_eq!(rotated_1[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_1[1], 1.0, epsilon = 1e-10);

    assert_relative_eq!(rotated_2[0], -1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_2[1], 0.0, epsilon = 1e-10);

    // Test interpolation at midpoints
    let rot_05 = spline.interpolate(0.5);
    let rot_15 = spline.interpolate(1.5);

    let rotated_05 = rot_05.apply(&test_point.view());
    let rotated_15 = rot_15.apply(&test_point.view());

    // Check that they produce expected intermediate rotations
    // t=0.5 should be 45 degrees around Z
    assert_relative_eq!(rotated_05[0], 0.7071, epsilon = 0.001);
    assert_relative_eq!(rotated_05[1], 0.7071, epsilon = 0.001);

    // t=1.5 should be 135 degrees around Z
    assert_relative_eq!(rotated_15[0], -0.7071, epsilon = 0.001);
    assert_relative_eq!(rotated_15[1], 0.7071, epsilon = 0.001);
}

#[test]
fn test_rotation_spline_cubic() {
    // Create a rotation spline
    let rotations = vec![
        Rotation::identity(),
        Rotation::from_euler(&array![0.0, 0.0, PI / 2.0], "xyz").unwrap(),
        Rotation::from_euler(&array![0.0, 0.0, PI], "xyz").unwrap(),
    ];
    let times = vec![0.0, 1.0, 2.0];

    let mut spline = RotationSpline::new(&rotations, &times).unwrap();

    // Switch to cubic interpolation
    spline.set_interpolation_type("cubic").unwrap();
    assert_eq!(spline.interpolation_type(), "cubic");

    // Test that the velocities have been computed
    // (This is an implementation detail, so we don't directly test the velocities)

    // Test interpolation at key times
    let test_point = array![1.0, 0.0, 0.0];

    let rot_0 = spline.interpolate(0.0);
    let rot_1 = spline.interpolate(1.0);
    let rot_2 = spline.interpolate(2.0);

    let rotated_0 = rot_0.apply(&test_point.view());
    let rotated_1 = rot_1.apply(&test_point.view());
    let rotated_2 = rot_2.apply(&test_point.view());

    // Check that they match the original rotations
    assert_relative_eq!(rotated_0[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_0[1], 0.0, epsilon = 1e-10);

    assert_relative_eq!(rotated_1[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_1[1], 1.0, epsilon = 1e-10);

    assert_relative_eq!(rotated_2[0], -1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_2[1], 0.0, epsilon = 1e-10);

    // Test interpolation at midpoints
    // The results will be different from SLERP due to the cubic interpolation
    // But we can't directly test the values without knowing the implementation details
    // Instead, we make sure that it produces valid rotations
    let rot_05 = spline.interpolate(0.5);
    let rot_15 = spline.interpolate(1.5);

    let rotated_05 = rot_05.apply(&test_point.view());
    let rotated_15 = rot_15.apply(&test_point.view());

    // Check that the results are unit vectors (valid rotations)
    let norm_05 = (rotated_05[0] * rotated_05[0]
        + rotated_05[1] * rotated_05[1]
        + rotated_05[2] * rotated_05[2])
        .sqrt();
    let norm_15 = (rotated_15[0] * rotated_15[0]
        + rotated_15[1] * rotated_15[1]
        + rotated_15[2] * rotated_15[2])
        .sqrt();

    assert_relative_eq!(norm_05, 1.0, epsilon = 1e-10);
    assert_relative_eq!(norm_15, 1.0, epsilon = 1e-10);
}

#[test]
fn test_rotation_spline_angular_velocity() {
    // Create a simple rotation spline (rotation around Z-axis)
    let rotations = vec![
        Rotation::identity(),
        Rotation::from_euler(&array![0.0, 0.0, PI], "xyz").unwrap(), // 180 degrees around Z in 1 second
    ];
    let times = vec![0.0, 1.0];

    let spline = RotationSpline::new(&rotations, &times).unwrap();

    // Test angular velocity at midpoint
    let velocity = spline.angular_velocity(0.5);

    // Should be approximately [0, 0, PI] rad/s
    assert_relative_eq!(velocity[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(velocity[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(velocity[2], PI, epsilon = 1e-10);

    // Test at the endpoints (should be zero or close to velocity at nearby points)
    let velocity_start = spline.angular_velocity(0.0);
    let velocity_end = spline.angular_velocity(1.0);

    // For SLERP, velocity is constant throughout
    assert_relative_eq!(velocity_start[2], 0.0, epsilon = 1e-10);
    assert_relative_eq!(velocity_end[2], 0.0, epsilon = 1e-10);

    // Test with cubic interpolation
    let mut cubic_spline = RotationSpline::new(&rotations, &times).unwrap();
    cubic_spline.set_interpolation_type("cubic").unwrap();

    // Velocity at midpoint should still be around [0, 0, PI]
    let cubic_velocity = cubic_spline.angular_velocity(0.5);

    // Magnitude should be close to PI
    let magnitude = (cubic_velocity[0] * cubic_velocity[0]
        + cubic_velocity[1] * cubic_velocity[1]
        + cubic_velocity[2] * cubic_velocity[2])
        .sqrt();

    assert_relative_eq!(magnitude, PI, epsilon = 0.1);
}

#[test]
fn test_rotation_spline_angular_acceleration() {
    // Create a rotation spline
    let rotations = vec![
        Rotation::identity(),
        Rotation::from_euler(&array![0.0, 0.0, PI / 2.0], "xyz").unwrap(),
        Rotation::from_euler(&array![0.0, 0.0, PI], "xyz").unwrap(),
    ];
    let times = vec![0.0, 1.0, 2.0];

    // With SLERP, acceleration should be zero
    let spline = RotationSpline::new(&rotations, &times).unwrap();
    let accel = spline.angular_acceleration(0.5);

    assert_relative_eq!(accel[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(accel[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(accel[2], 0.0, epsilon = 1e-10);

    // With cubic interpolation, acceleration should be non-zero
    let mut cubic_spline = RotationSpline::new(&rotations, &times).unwrap();
    cubic_spline.set_interpolation_type("cubic").unwrap();

    // Create a more complex rotation sequence to better test acceleration
    let complex_rotations = vec![
        Rotation::identity(),
        Rotation::from_euler(&array![0.0, PI / 4.0, 0.0], "xyz").unwrap(),
        Rotation::from_euler(&array![PI / 4.0, PI / 4.0, 0.0], "xyz").unwrap(),
        Rotation::from_euler(&array![0.0, 0.0, 0.0], "xyz").unwrap(),
    ];
    let complex_times = vec![0.0, 1.0, 2.0, 3.0];

    let mut complex_spline = RotationSpline::new(&complex_rotations, &complex_times).unwrap();
    complex_spline.set_interpolation_type("cubic").unwrap();

    // Test at a few points
    let accel_1 = complex_spline.angular_acceleration(0.5);
    let accel_2 = complex_spline.angular_acceleration(1.5);
    let accel_3 = complex_spline.angular_acceleration(2.5);

    // Accelerations should be non-zero for a complex spline
    // We don't know the exact values, so just check that at least one component is non-zero
    let has_accel_1 = accel_1[0].abs() > 1e-6 || accel_1[1].abs() > 1e-6 || accel_1[2].abs() > 1e-6;
    let has_accel_2 = accel_2[0].abs() > 1e-6 || accel_2[1].abs() > 1e-6 || accel_2[2].abs() > 1e-6;
    let has_accel_3 = accel_3[0].abs() > 1e-6 || accel_3[1].abs() > 1e-6 || accel_3[2].abs() > 1e-6;

    // At least one of the tested points should have non-zero acceleration
    assert!(has_accel_1 || has_accel_2 || has_accel_3);
}
