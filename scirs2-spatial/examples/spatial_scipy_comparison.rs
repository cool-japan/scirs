use std::f64::consts::PI;

fn main() {
    println!("=== SciPy vs Our Implementation Comparison ===\n");

    // SciPy results for individual rotations (converted from [x,y,z,w] to [w,x,y,z])
    println!("Individual rotations (90°):");
    println!("X: SciPy=[std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0000, 0.0000]");
    println!("Y: SciPy=[std::f64::consts::FRAC_1_SQRT_2, 0.0000, std::f64::consts::FRAC_1_SQRT_2, 0.0000]");
    println!("Z: SciPy=[std::f64::consts::FRAC_1_SQRT_2, 0.0000, 0.0000, std::f64::consts::FRAC_1_SQRT_2]");

    println!("\nCombined XYZ(45°,60°,30°):");
    println!("SciPy result: [w=0.8224, x=0.2006, y=0.5320, z=0.0223]");
    println!("Manual mult:  [w=0.7233, x=0.4397, y=0.3604, z=0.3919]");

    // The key insight: SciPy's XYZ means apply X first, then Y, then Z
    // But their manual multiplication shows something different!

    // Let's trace through what's happening
    println!("\nDebugging the quaternion multiplication order:");

    // For intrinsic XYZ: First rotate around X, then around the new Y, then around the new Z
    // The quaternion multiplication order for intrinsic rotations is: Qz * Qy * Qx
    // (rightmost rotation is applied first)

    let qx = quat_from_axis_angle([1.0, 0.0, 0.0], PI / 4.0);
    let qy = quat_from_axis_angle([0.0, 1.0, 0.0], PI / 3.0);
    let qz = quat_from_axis_angle([0.0, 0.0, 1.0], PI / 6.0);

    println!("\nIndividual quaternions:");
    println!(
        "Qx(45°): [{:.4}, {:.4}, {:.4}, {:.4}]",
        qx[0], qx[1], qx[2], qx[3]
    );
    println!(
        "Qy(60°): [{:.4}, {:.4}, {:.4}, {:.4}]",
        qy[0], qy[1], qy[2], qy[3]
    );
    println!(
        "Qz(30°): [{:.4}, {:.4}, {:.4}, {:.4}]",
        qz[0], qz[1], qz[2], qz[3]
    );

    // For intrinsic XYZ, we need: Qz * Qy * Qx
    let qyx = quat_multiply(&qy, &qx);
    let qzyx = quat_multiply(&qz, &qyx);

    println!("\nIntrinsic XYZ (Qz * Qy * Qx):");
    println!(
        "Qy * Qx = [{:.4}, {:.4}, {:.4}, {:.4}]",
        qyx[0], qyx[1], qyx[2], qyx[3]
    );
    println!(
        "Qz * (Qy * Qx) = [{:.4}, {:.4}, {:.4}, {:.4}]",
        qzyx[0], qzyx[1], qzyx[2], qzyx[3]
    );

    // Compare with the direct formula
    let q_formula = euler_to_quat_xyz_intrinsic(PI / 4.0, PI / 3.0, PI / 6.0);
    println!(
        "\nDirect formula result: [{:.4}, {:.4}, {:.4}, {:.4}]",
        q_formula[0], q_formula[1], q_formula[2], q_formula[3]
    );
}

fn euler_to_quat_xyz_intrinsic(x: f64, y: f64, z: f64) -> [f64; 4] {
    // For intrinsic XYZ, the combined rotation is Qz * Qy * Qx
    // Using the quaternion multiplication formula
    let a = x / 2.0;
    let b = y / 2.0;
    let c = z / 2.0;

    let ca = a.cos();
    let sa = a.sin();
    let cb = b.cos();
    let sb = b.sin();
    let cc = c.cos();
    let sc = c.sin();

    // Result of Qz * Qy * Qx multiplication
    [
        cc * cb * ca + sc * sb * sa, // w
        cc * cb * sa - sc * sb * ca, // x
        cc * sb * ca + sc * cb * sa, // y
        sc * cb * ca - cc * sb * sa, // z
    ]
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
