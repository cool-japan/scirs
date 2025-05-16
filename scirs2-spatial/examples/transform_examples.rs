use ndarray::array;
use scirs2_spatial::transform::{RigidTransform, Rotation, RotationSpline, Slerp};
use scirs2_spatial::SpatialResult;
use std::f64::consts::PI;

fn main() -> SpatialResult<()> {
    println!("Spatial Transformations Examples");
    println!("==============================\n");

    // Demonstrate Rotation
    rotation_examples()?;

    // Demonstrate RigidTransform
    rigid_transform_examples()?;

    // Demonstrate Slerp
    slerp_examples()?;

    // Demonstrate RotationSpline
    rotation_spline_examples()?;

    Ok(())
}

fn rotation_examples() -> SpatialResult<()> {
    println!("Rotation Examples");
    println!("-----------------");

    // Create rotations using different representations
    println!("Creating rotations using different representations:");

    // 1. From quaternion
    let quat = array![0.7071, 0.7071, 0.0, 0.0]; // 90-degree rotation around X
    let rot_quat = Rotation::from_quat(&quat.view())?;
    println!("  1. From quaternion [0.7071, 0.7071, 0.0, 0.0] (90° around X)");

    // 2. From Euler angles
    let euler = array![0.0, 0.0, PI / 2.0]; // 90-degree rotation around Z
    let rot_euler = Rotation::from_euler(&euler.view(), "xyz")?;
    println!("  2. From Euler angles [0, 0, π/2] (90° around Z)");

    // 3. From rotation matrix
    let matrix = array![[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]; // 90-degree rotation around Z
    let _rot_matrix = Rotation::from_matrix(&matrix.view())?;
    println!("  3. From rotation matrix (90° around Z)");

    // 4. From axis-angle
    let rotvec = array![0.0, 0.0, PI / 2.0]; // 90-degree rotation around Z
    let _rot_rotvec = Rotation::from_rotvec(&rotvec.view())?;
    println!("  4. From rotation vector [0, 0, π/2] (90° around Z)");

    // Apply rotations to a test point
    println!("\nApplying rotations to point [1, 0, 0]:");
    let point = array![1.0, 0.0, 0.0];

    let rotated_quat = rot_quat.apply(&point.view());
    println!(
        "  X-rotation: [{:.4}, {:.4}, {:.4}]",
        rotated_quat[0], rotated_quat[1], rotated_quat[2]
    );

    let rotated_euler = rot_euler.apply(&point.view());
    println!(
        "  Z-rotation: [{:.4}, {:.4}, {:.4}]",
        rotated_euler[0], rotated_euler[1], rotated_euler[2]
    );

    // Convert between representations
    println!("\nConverting between representations:");

    // Quaternion of the Z-rotation
    let quat_z = rot_euler.as_quat();
    println!(
        "  Quaternion of Z-rotation: [{:.4}, {:.4}, {:.4}, {:.4}]",
        quat_z[0], quat_z[1], quat_z[2], quat_z[3]
    );

    // Euler angles of the X-rotation (xyz convention)
    let euler_x = rot_quat.as_euler("xyz")?;
    println!(
        "  Euler angles of X-rotation: [{:.4}, {:.4}, {:.4}]",
        euler_x[0], euler_x[1], euler_x[2]
    );

    // Rotation vector of the Z-rotation
    let rotvec_z = rot_euler.as_rotvec();
    println!(
        "  Rotation vector of Z-rotation: [{:.4}, {:.4}, {:.4}]",
        rotvec_z[0], rotvec_z[1], rotvec_z[2]
    );

    // Compose rotations
    println!("\nComposing rotations:");
    let composed = rot_quat.compose(&rot_euler);
    let composed_point = composed.apply(&point.view());
    println!("  X-rotation followed by Z-rotation applied to [1, 0, 0]:");
    println!(
        "  Result: [{:.4}, {:.4}, {:.4}]",
        composed_point[0], composed_point[1], composed_point[2]
    );

    // Inverse rotation
    println!("\nInverse rotation:");
    let rot_inv = rot_euler.inv();
    let point_rotated = rot_euler.apply(&point.view());
    let point_back = rot_inv.apply(&point_rotated.view());
    println!(
        "  Original point: [{:.4}, {:.4}, {:.4}]",
        point[0], point[1], point[2]
    );
    println!(
        "  Rotated by Z-rotation: [{:.4}, {:.4}, {:.4}]",
        point_rotated[0], point_rotated[1], point_rotated[2]
    );
    println!(
        "  Then by inverse Z-rotation: [{:.4}, {:.4}, {:.4}]",
        point_back[0], point_back[1], point_back[2]
    );

    println!();
    Ok(())
}

fn rigid_transform_examples() -> SpatialResult<()> {
    println!("RigidTransform Examples");
    println!("----------------------");

    // Create a rigid transform from rotation and translation
    // Create an array first, then pass its view
    let euler_angles = array![0.0, 0.0, PI / 2.0];
    let rotation = Rotation::from_euler(&euler_angles.view(), "xyz")?;
    let translation = array![1.0, 2.0, 3.0];

    let transform =
        RigidTransform::from_rotation_and_translation(rotation.clone(), &translation.view())?;
    println!("Created a rigid transform with:");
    println!("  - 90° rotation around Z");
    println!("  - Translation [1, 2, 3]");

    // Create a rigid transform from a 4x4 matrix
    let matrix = array![
        [0.0, -1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ];
    let _transform_from_matrix = RigidTransform::from_matrix(&matrix.view())?;
    println!("\nCreated an equivalent rigid transform from 4x4 matrix");

    // Apply the transform to points
    let point1 = array![1.0, 0.0, 0.0];
    let point2 = array![0.0, 0.0, 0.0];

    let transformed1 = transform.apply(&point1.view());
    let transformed2 = transform.apply(&point2.view());

    println!("\nApplying the transform to points:");
    println!(
        "  [1, 0, 0] -> [{:.4}, {:.4}, {:.4}]",
        transformed1[0], transformed1[1], transformed1[2]
    );
    println!(
        "  [0, 0, 0] -> [{:.4}, {:.4}, {:.4}]",
        transformed2[0], transformed2[1], transformed2[2]
    );

    // Inverse transform
    let inverse = transform.inv();
    println!("\nApplying the inverse transform:");

    let back1 = inverse.apply(&transformed1.view());
    println!(
        "  [{:.4}, {:.4}, {:.4}] -> [{:.4}, {:.4}, {:.4}]",
        transformed1[0], transformed1[1], transformed1[2], back1[0], back1[1], back1[2]
    );

    // Composition of transforms
    let transform2 = RigidTransform::from_rotation_and_translation(
        {
            let angles = array![PI / 2.0, 0.0, 0.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
        &array![0.0, 0.0, 1.0].view(),
    )?;

    let composed = transform.compose(&transform2);

    println!("\nComposing transforms:");
    println!("  Transform 1: 90° rotation around Z + translation [1, 2, 3]");
    println!("  Transform 2: 90° rotation around X + translation [0, 0, 1]");

    let composed_applied = composed.apply(&point1.view());
    println!(
        "  [1, 0, 0] -> [{:.4}, {:.4}, {:.4}]",
        composed_applied[0], composed_applied[1], composed_applied[2]
    );

    println!();
    Ok(())
}

fn slerp_examples() -> SpatialResult<()> {
    println!("Slerp (Spherical Linear Interpolation) Examples");
    println!("---------------------------------------------");

    // Create two rotations to interpolate between
    let rot1 = Rotation::identity(); // No rotation
    let euler_angles2 = array![0.0, 0.0, PI];
    let rot2 = Rotation::from_euler(&euler_angles2.view(), "xyz")?; // 180° around Z

    // Create a Slerp interpolator
    let slerp = Slerp::new(rot1, rot2)?;
    println!("Created a Slerp interpolator between:");
    println!("  - Identity rotation");
    println!("  - 180° rotation around Z");

    // Sample at different interpolation parameters
    println!("\nSampling at different interpolation parameters:");
    let test_point = array![1.0, 0.0, 0.0];

    let t_values = [0.0, 0.25, 0.5, 0.75, 1.0];
    for &t in &t_values {
        let rot_t = slerp.interpolate(t);
        let rotated = rot_t.apply(&test_point.view());

        println!(
            "  t = {:.2}: [{:.4}, {:.4}, {:.4}]",
            t, rotated[0], rotated[1], rotated[2]
        );
    }

    // Times for constant angular velocity
    println!("\nGenerating times for constant angular velocity:");
    let times = slerp.times(5);
    println!(
        "  Times: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
        times[0], times[1], times[2], times[3], times[4]
    );

    println!();
    Ok(())
}

fn rotation_spline_examples() -> SpatialResult<()> {
    println!("RotationSpline Examples");
    println!("---------------------");

    // Create a sequence of rotations
    let rotations = vec![
        Rotation::identity(),
        {
            let angles = array![0.0, 0.0, PI / 2.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
        {
            let angles = array![PI / 2.0, 0.0, PI / 2.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
        {
            let angles = array![PI / 2.0, PI / 2.0, PI / 2.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
    ];

    // Define times at which these rotations occur
    let times = vec![0.0, 1.0, 2.0, 3.0];

    // Create a rotation spline using default Slerp interpolation
    let mut spline = RotationSpline::new(&rotations, &times)?;
    println!("Created a rotation spline with 4 key rotations at times [0, 1, 2, 3]");
    println!(
        "Default interpolation type: {}",
        spline.interpolation_type()
    );

    // Sample the spline at various times using Slerp interpolation
    println!("\nSampling the spline at various times (Slerp interpolation):");
    let test_point = array![1.0, 0.0, 0.0];

    let sample_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    for &t in &sample_times {
        let rot_t = spline.interpolate(t);
        let rotated = rot_t.apply(&test_point.view());

        println!(
            "  t = {:.1}: [{:.4}, {:.4}, {:.4}]",
            t, rotated[0], rotated[1], rotated[2]
        );
    }

    // Generate evenly spaced samples
    println!("\nGenerating 7 evenly spaced samples (Slerp):");
    let (sample_times, sample_rotations) = spline.sample(7);

    for i in 0..7 {
        let rotated = sample_rotations[i].apply(&test_point.view());
        println!(
            "  t = {:.2}: [{:.4}, {:.4}, {:.4}]",
            sample_times[i], rotated[0], rotated[1], rotated[2]
        );
    }

    // Switch to cubic interpolation
    println!("\nSwitching to cubic interpolation");
    spline.set_interpolation_type("cubic")?;
    println!("New interpolation type: {}", spline.interpolation_type());

    // Sample the spline with cubic interpolation
    println!("\nSampling the spline with cubic interpolation:");
    for &t in &sample_times {
        let rot_t = spline.interpolate(t);
        let rotated = rot_t.apply(&test_point.view());

        println!(
            "  t = {:.1}: [{:.4}, {:.4}, {:.4}]",
            t, rotated[0], rotated[1], rotated[2]
        );
    }

    // Calculate angular velocities
    println!("\nCalculating angular velocities at sample times:");
    for &t in &sample_times {
        let velocity = spline.angular_velocity(t);

        println!(
            "  t = {:.1}: [{:.4}, {:.4}, {:.4}] rad/s",
            t, velocity[0], velocity[1], velocity[2]
        );
    }

    // Calculate angular accelerations
    println!("\nCalculating angular accelerations at sample times:");
    for &t in &sample_times {
        let acceleration = spline.angular_acceleration(t);

        println!(
            "  t = {:.1}: [{:.4}, {:.4}, {:.4}] rad/s²",
            t, acceleration[0], acceleration[1], acceleration[2]
        );
    }

    // Create a more complex spline for animation
    println!("\nCreating a complex spline for animation:");
    let animation_rotations = vec![
        Rotation::identity(),
        {
            let angles = array![0.0, PI / 4.0, 0.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
        {
            let angles = array![PI / 4.0, PI / 4.0, 0.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
        {
            let angles = array![PI / 4.0, 0.0, PI / 4.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
        {
            let angles = array![0.0, 0.0, 0.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
    ];

    let animation_times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let mut animation_spline = RotationSpline::new(&animation_rotations, &animation_times)?;

    // Use cubic interpolation for smooth animation
    animation_spline.set_interpolation_type("cubic")?;

    println!("Creating a smooth animation with 9 frames:");
    let (anim_times, anim_rotations) = animation_spline.sample(9);

    for i in 0..9 {
        let rotated = anim_rotations[i].apply(&test_point.view());
        let velocity = animation_spline.angular_velocity(anim_times[i]);

        println!(
            "  Frame {}: position [{:.4}, {:.4}, {:.4}], velocity magnitude: {:.4} rad/s",
            i,
            rotated[0],
            rotated[1],
            rotated[2],
            (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2])
                .sqrt()
        );
    }

    println!();
    Ok(())
}
