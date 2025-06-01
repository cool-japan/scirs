use std::f64::consts::PI;

fn main() {
    println!("=== Debug Euler Conventions ===\n");

    // For debugging, let's use the exact formulas
    let angles = [30.0 * PI / 180.0, 45.0 * PI / 180.0, 60.0 * PI / 180.0];

    println!("Input angles: [30°, 45°, 60°]\n");

    // XYZ convention
    println!("XYZ Convention:");
    println!("- angles[0] = 30° is rotation around X");
    println!("- angles[1] = 45° is rotation around Y");
    println!("- angles[2] = 60° is rotation around Z");

    let a = angles[0] / 2.0; // X angle / 2
    let b = angles[1] / 2.0; // Y angle / 2
    let c = angles[2] / 2.0; // Z angle / 2

    let q_xyz = compute_xyz_quaternion(a, b, c);
    println!(
        "XYZ quaternion: [{:.4}, {:.4}, {:.4}, {:.4}]",
        q_xyz[0], q_xyz[1], q_xyz[2], q_xyz[3]
    );

    // ZYX convention
    println!("\nZYX Convention:");
    println!("- angles[0] = 30° is rotation around Z");
    println!("- angles[1] = 45° is rotation around Y");
    println!("- angles[2] = 60° is rotation around X");

    // For ZYX, the angle mapping is different!
    let z_half = angles[0] / 2.0; // Z angle / 2
    let y_half = angles[1] / 2.0; // Y angle / 2
    let x_half = angles[2] / 2.0; // X angle / 2

    let q_zyx = compute_zyx_quaternion(z_half, y_half, x_half);
    println!(
        "ZYX quaternion: [{:.4}, {:.4}, {:.4}, {:.4}]",
        q_zyx[0], q_zyx[1], q_zyx[2], q_zyx[3]
    );

    println!("\nExpected from SciPy:");
    println!("XYZ: [0.8224, 0.0223, 0.4397, 0.3604]");
    println!("ZYX: [0.7233, 0.5320, 0.2006, 0.3919]");
}

fn compute_xyz_quaternion(x_half: f64, y_half: f64, z_half: f64) -> [f64; 4] {
    // For intrinsic XYZ: Qz * Qy * Qx
    let cx = x_half.cos();
    let sx = x_half.sin();
    let cy = y_half.cos();
    let sy = y_half.sin();
    let cz = z_half.cos();
    let sz = z_half.sin();

    // Direct formula for Qz * Qy * Qx
    [
        cz * cy * cx + sz * sy * sx,
        cz * cy * sx - sz * sy * cx,
        cz * sy * cx + sz * cy * sx,
        sz * cy * cx - cz * sy * sx,
    ]
}

fn compute_zyx_quaternion(z_half: f64, y_half: f64, x_half: f64) -> [f64; 4] {
    // For intrinsic ZYX: Qx * Qy * Qz
    let cx = x_half.cos();
    let sx = x_half.sin();
    let cy = y_half.cos();
    let sy = y_half.sin();
    let cz = z_half.cos();
    let sz = z_half.sin();

    // Direct formula for Qx * Qy * Qz
    [
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ]
}
