use std::f64::consts::PI;

fn main() {
    println!("=== Debug ZYX Step by Step ===\n");

    // For ZYX with angles [30°, 45°, 60°]:
    // angles[0] = 30° is Z rotation
    // angles[1] = 45° is Y rotation
    // angles[2] = 60° is X rotation

    let z_angle = 30.0 * PI / 180.0;
    let y_angle = 45.0 * PI / 180.0;
    let x_angle = 60.0 * PI / 180.0;

    println!("ZYX intrinsic rotations:");
    println!("1. First rotate 30° around Z");
    println!("2. Then rotate 45° around new Y");
    println!("3. Then rotate 60° around new X");
    println!();

    // Individual quaternions
    let qz = quat_from_axis_angle([0.0, 0.0, 1.0], z_angle);
    let qy = quat_from_axis_angle([0.0, 1.0, 0.0], y_angle);
    let qx = quat_from_axis_angle([1.0, 0.0, 0.0], x_angle);

    println!("Individual quaternions:");
    println!(
        "Qz(30°): [{:.4}, {:.4}, {:.4}, {:.4}]",
        qz[0], qz[1], qz[2], qz[3]
    );
    println!(
        "Qy(45°): [{:.4}, {:.4}, {:.4}, {:.4}]",
        qy[0], qy[1], qy[2], qy[3]
    );
    println!(
        "Qx(60°): [{:.4}, {:.4}, {:.4}, {:.4}]",
        qx[0], qx[1], qx[2], qx[3]
    );
    println!();

    // For intrinsic ZYX, the multiplication order is Qx * Qy * Qz
    let qy_qz = quat_multiply(&qy, &qz);
    let qx_qy_qz = quat_multiply(&qx, &qy_qz);

    println!("Step-by-step multiplication (Qx * Qy * Qz):");
    println!(
        "Qy * Qz = [{:.4}, {:.4}, {:.4}, {:.4}]",
        qy_qz[0], qy_qz[1], qy_qz[2], qy_qz[3]
    );
    println!(
        "Qx * (Qy * Qz) = [{:.4}, {:.4}, {:.4}, {:.4}]",
        qx_qy_qz[0], qx_qy_qz[1], qx_qy_qz[2], qx_qy_qz[3]
    );
    println!();

    // Now using the direct formula
    let z_half = z_angle / 2.0;
    let y_half = y_angle / 2.0;
    let x_half = x_angle / 2.0;

    let cx = x_half.cos();
    let sx = x_half.sin();
    let cy = y_half.cos();
    let sy = y_half.sin();
    let cz = z_half.cos();
    let sz = z_half.sin();

    println!("Half angles:");
    println!(
        "z/2 = {:.4} rad, y/2 = {:.4} rad, x/2 = {:.4} rad",
        z_half, y_half, x_half
    );
    println!("cos(z/2) = {:.4}, sin(z/2) = {:.4}", cz, sz);
    println!("cos(y/2) = {:.4}, sin(y/2) = {:.4}", cy, sy);
    println!("cos(x/2) = {:.4}, sin(x/2) = {:.4}", cx, sx);
    println!();

    // Direct formula for Qx * Qy * Qz
    let q_direct = [
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ];

    println!("Direct formula result:");
    println!(
        "[{:.4}, {:.4}, {:.4}, {:.4}]",
        q_direct[0], q_direct[1], q_direct[2], q_direct[3]
    );

    println!("\nExpected from SciPy: [0.7233, 0.5320, 0.2006, 0.3919]");
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
