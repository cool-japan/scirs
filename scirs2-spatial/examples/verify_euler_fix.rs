use std::f64::consts::PI;

fn main() {
    println!("=== Verifying Euler Angle Formulas ===\n");

    // Test individual axis rotations
    println!("Individual axis rotations (90 degrees each):");

    // X-axis rotation
    let qx_90 = euler_to_quat_xyz(PI / 2.0, 0.0, 0.0);
    println!(
        "X(90°): [{:.4}, {:.4}, {:.4}, {:.4}]",
        qx_90[0], qx_90[1], qx_90[2], qx_90[3]
    );
    println!("  Expected: [std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0000, 0.0000]");

    // Y-axis rotation
    let qy_90 = euler_to_quat_xyz(0.0, PI / 2.0, 0.0);
    println!(
        "Y(90°): [{:.4}, {:.4}, {:.4}, {:.4}]",
        qy_90[0], qy_90[1], qy_90[2], qy_90[3]
    );
    println!("  Expected: [std::f64::consts::FRAC_1_SQRT_2, 0.0000, std::f64::consts::FRAC_1_SQRT_2, 0.0000]");

    // Z-axis rotation
    let qz_90 = euler_to_quat_xyz(0.0, 0.0, PI / 2.0);
    println!(
        "Z(90°): [{:.4}, {:.4}, {:.4}, {:.4}]",
        qz_90[0], qz_90[1], qz_90[2], qz_90[3]
    );
    println!("  Expected: [std::f64::consts::FRAC_1_SQRT_2, 0.0000, 0.0000, std::f64::consts::FRAC_1_SQRT_2]");

    println!("\nCombined rotation test:");
    let q_combined = euler_to_quat_xyz(PI / 4.0, PI / 3.0, PI / 6.0);
    println!(
        "XYZ(45°,60°,30°): [{:.4}, {:.4}, {:.4}, {:.4}]",
        q_combined[0], q_combined[1], q_combined[2], q_combined[3]
    );

    // Verify by manual multiplication
    let qx = quat_from_axis_angle([1.0, 0.0, 0.0], PI / 4.0);
    let qy = quat_from_axis_angle([0.0, 1.0, 0.0], PI / 3.0);
    let qz = quat_from_axis_angle([0.0, 0.0, 1.0], PI / 6.0);

    let qxy = quat_multiply(&qx, &qy);
    let qxyz = quat_multiply(&qxy, &qz);

    println!(
        "Manual multiplication: [{:.4}, {:.4}, {:.4}, {:.4}]",
        qxyz[0], qxyz[1], qxyz[2], qxyz[3]
    );

    println!("\nComparing current (wrong) vs correct formulas:");
    let (wrong, correct) = compare_formulas(PI / 4.0, PI / 3.0, PI / 6.0);
    println!(
        "Current (wrong): [{:.4}, {:.4}, {:.4}, {:.4}]",
        wrong[0], wrong[1], wrong[2], wrong[3]
    );
    println!(
        "Correct:         [{:.4}, {:.4}, {:.4}, {:.4}]",
        correct[0], correct[1], correct[2], correct[3]
    );
}

fn euler_to_quat_xyz(x: f64, y: f64, z: f64) -> [f64; 4] {
    // Correct formula for intrinsic XYZ rotation
    let a = x / 2.0;
    let b = y / 2.0;
    let c = z / 2.0;

    let ca = a.cos();
    let sa = a.sin();
    let cb = b.cos();
    let sb = b.sin();
    let cc = c.cos();
    let sc = c.sin();

    // This is the correct formula for XYZ
    [
        ca * cb * cc + sa * sb * sc, // w
        sa * cb * cc - ca * sb * sc, // x
        ca * sb * cc + sa * cb * sc, // y
        ca * cb * sc - sa * sb * cc, // z
    ]
}

fn compare_formulas(x: f64, y: f64, z: f64) -> ([f64; 4], [f64; 4]) {
    let a = x / 2.0;
    let b = y / 2.0;
    let c = z / 2.0;

    let ca = a.cos();
    let sa = a.sin();
    let cb = b.cos();
    let sb = b.sin();
    let cc = c.cos();
    let sc = c.sin();

    // Current wrong formula (actually for some other convention)
    let wrong = [
        ca * cb * cc - sa * cb * sc,
        sa * cb * cc + ca * cb * sc,
        ca * sb * cc - sa * sb * sc,
        ca * cb * sc + sa * sb * cc,
    ];

    // Correct formula for XYZ
    let correct = [
        ca * cb * cc + sa * sb * sc,
        sa * cb * cc - ca * sb * sc,
        ca * sb * cc + sa * cb * sc,
        ca * cb * sc - sa * sb * cc,
    ];

    (wrong, correct)
}

fn quat_from_axis_angle(axis: [f64; 3], angle: f64) -> [f64; 4] {
    let half_angle = angle / 2.0;
    let s = half_angle.sin();
    [half_angle.cos(), axis[0] * s, axis[1] * s, axis[2] * s]
}

fn quat_multiply(q1: &[f64; 4], q2: &[f64; 4]) -> [f64; 4] {
    [
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
    ]
}
