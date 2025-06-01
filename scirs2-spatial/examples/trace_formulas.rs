use std::f64::consts::PI;

fn main() {
    println!("=== Tracing Quaternion Formulas ===\n");

    let angles = [30.0 * PI / 180.0, 45.0 * PI / 180.0, 60.0 * PI / 180.0];

    // For XYZ: angles[0] = X rotation, angles[1] = Y rotation, angles[2] = Z rotation
    let a = angles[0] / 2.0; // X half-angle
    let b = angles[1] / 2.0; // Y half-angle
    let c = angles[2] / 2.0; // Z half-angle

    let ca = a.cos();
    let sa = a.sin();
    let cb = b.cos();
    let sb = b.sin();
    let cc = c.cos();
    let sc = c.sin();

    println!("Half angles:");
    println!("a (X/2) = {:.4} rad", a);
    println!("b (Y/2) = {:.4} rad", b);
    println!("c (Z/2) = {:.4} rad", c);
    println!();

    // Current wrong formula (what's in the code now for XYZ)
    let q_wrong = [
        cc * cb * ca + sc * sb * sa,
        cc * cb * sa - sc * sb * ca,
        cc * sb * ca + sc * cb * sa,
        sc * cb * ca - cc * sb * sa,
    ];

    println!("Current XYZ formula result:");
    println!(
        "[{:.4}, {:.4}, {:.4}, {:.4}]",
        q_wrong[0], q_wrong[1], q_wrong[2], q_wrong[3]
    );

    // Manual quaternion multiplication for XYZ
    // Qx = [cos(a), sin(a), 0, 0]
    // Qy = [cos(b), 0, sin(b), 0]
    // Qz = [cos(c), 0, 0, sin(c)]

    let qx = [ca, sa, 0.0, 0.0];
    let qy = [cb, 0.0, sb, 0.0];
    let qz = [cc, 0.0, 0.0, sc];

    println!("\nIndividual quaternions:");
    println!(
        "Qx: [{:.4}, {:.4}, {:.4}, {:.4}]",
        qx[0], qx[1], qx[2], qx[3]
    );
    println!(
        "Qy: [{:.4}, {:.4}, {:.4}, {:.4}]",
        qy[0], qy[1], qy[2], qy[3]
    );
    println!(
        "Qz: [{:.4}, {:.4}, {:.4}, {:.4}]",
        qz[0], qz[1], qz[2], qz[3]
    );

    // For intrinsic XYZ: Qz * Qy * Qx
    let qyx = quat_multiply(&qy, &qx);
    let qzyx = quat_multiply(&qz, &qyx);

    println!("\nIntrinsic XYZ (Qz * Qy * Qx):");
    println!(
        "Result: [{:.4}, {:.4}, {:.4}, {:.4}]",
        qzyx[0], qzyx[1], qzyx[2], qzyx[3]
    );

    // For ZYX convention with same angles
    // For intrinsic ZYX: Qx * Qy * Qz
    let qyz = quat_multiply(&qy, &qz);
    let qxyz = quat_multiply(&qx, &qyz);

    println!("\nIntrinsic ZYX (Qx * Qy * Qz):");
    println!(
        "Result: [{:.4}, {:.4}, {:.4}, {:.4}]",
        qxyz[0], qxyz[1], qxyz[2], qxyz[3]
    );

    // Wait, I think I'm confusing myself. Let me be very clear about the conventions:
    // For XYZ intrinsic: First rotate around body X, then around new body Y, then around new body Z
    // For ZYX intrinsic: First rotate around body Z, then around new body Y, then around new body X

    // But the angles array is always in the order of the convention name!
    // So for XYZ: angles[0] is X rotation, angles[1] is Y rotation, angles[2] is Z rotation
    // For ZYX: angles[0] is Z rotation, angles[1] is Y rotation, angles[2] is X rotation

    println!("\n=== Correct interpretation ===");
    println!("\nFor XYZ convention with angles [30°, 45°, 60°]:");
    println!("- First rotate 30° around X");
    println!("- Then rotate 45° around new Y");
    println!("- Then rotate 60° around new Z");

    println!("\nFor ZYX convention with angles [30°, 45°, 60°]:");
    println!("- First rotate 30° around Z");
    println!("- Then rotate 45° around new Y");
    println!("- Then rotate 60° around new X");

    // So for ZYX, we need different angle assignments!
    let z_angle = angles[0] / 2.0; // First angle is Z for ZYX
    let y_angle = angles[1] / 2.0; // Second angle is Y
    let x_angle = angles[2] / 2.0; // Third angle is X

    let cx = x_angle.cos();
    let sx = x_angle.sin();
    let cy = y_angle.cos();
    let sy = y_angle.sin();
    let cz = z_angle.cos();
    let sz = z_angle.sin();

    let qx_zyx = [cx, sx, 0.0, 0.0];
    let qy_zyx = [cy, 0.0, sy, 0.0];
    let qz_zyx = [cz, 0.0, 0.0, sz];

    // For intrinsic ZYX: Qx * Qy * Qz
    let qy_qz = quat_multiply(&qy_zyx, &qz_zyx);
    let qx_qy_qz = quat_multiply(&qx_zyx, &qy_qz);

    println!("\nCorrect ZYX result (Qx * Qy * Qz with proper angle mapping):");
    println!(
        "Result: [{:.4}, {:.4}, {:.4}, {:.4}]",
        qx_qy_qz[0], qx_qy_qz[1], qx_qy_qz[2], qx_qy_qz[3]
    );
}

fn quat_multiply(q1: &[f64; 4], q2: &[f64; 4]) -> [f64; 4] {
    [
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
    ]
}
