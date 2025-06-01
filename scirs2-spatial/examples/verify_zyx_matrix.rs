use std::f64::consts::PI;

fn main() {
    println!("=== Verify ZYX Rotation Matrix ===\n");

    // For ZYX with angles [Z, Y, X]
    let z_angle = 30.0 * PI / 180.0;
    let y_angle = 45.0 * PI / 180.0;
    let x_angle = 60.0 * PI / 180.0;

    let cx = x_angle.cos();
    let sx = x_angle.sin();
    let cy = y_angle.cos();
    let sy = y_angle.sin();
    let cz = z_angle.cos();
    let sz = z_angle.sin();

    println!("Angles: Z=30°, Y=45°, X=60°");

    // Individual rotation matrices
    println!("\nRz(30°) =");
    println!("  [{:.4}, {:.4},  0.0000]", cz, -sz);
    println!("  [{:.4},  {:.4},  0.0000]", sz, cz);
    println!("  [ 0.0000,  0.0000,  1.0000]");

    println!("\nRy(45°) =");
    println!("  [ {:.4},  0.0000, {:.4}]", cy, sy);
    println!("  [ 0.0000,  1.0000,  0.0000]");
    println!("  [{:.4},  0.0000,  {:.4}]", -sy, cy);

    println!("\nRx(60°) =");
    println!("  [ 1.0000,  0.0000,  0.0000]");
    println!("  [ 0.0000,  {:.4}, {:.4}]", cx, -sx);
    println!("  [ 0.0000,  {:.4},  {:.4}]", sx, cx);

    // For intrinsic ZYX: first Z, then Y around new axis, then X around new axis
    // This corresponds to extrinsic multiplication: Rx * Ry * Rz
    println!("\nFor intrinsic ZYX, the combined matrix is Rx * Ry * Rz:");

    // Compute Ry * Rz first
    let r00_yz = cy * cz;
    let r01_yz = cy * (-sz);
    let r02_yz = sy;
    let r10_yz = sz;
    let r11_yz = cz;
    let r12_yz = 0.0;
    let r20_yz = -sy * cz;
    let r21_yz = -sy * (-sz);
    let r22_yz = cy;

    println!("\nRy * Rz =");
    println!("  [{:.4}, {:.4}, {:.4}]", r00_yz, r01_yz, r02_yz);
    println!("  [{:.4}, {:.4}, {:.4}]", r10_yz, r11_yz, r12_yz);
    println!("  [{:.4}, {:.4}, {:.4}]", r20_yz, r21_yz, r22_yz);

    // Now compute Rx * (Ry * Rz)
    let r00 = r00_yz;
    let r01 = r01_yz;
    let r02 = r02_yz;
    let r10 = cx * r10_yz + sx * r20_yz;
    let r11 = cx * r11_yz + sx * r21_yz;
    let r12 = cx * r12_yz + sx * r22_yz;
    let r20 = -sx * r10_yz + cx * r20_yz;
    let r21 = -sx * r11_yz + cx * r21_yz;
    let r22 = -sx * r12_yz + cx * r22_yz;

    println!("\nRx * Ry * Rz =");
    println!("  [{:.4}, {:.4}, {:.4}]", r00, r01, r02);
    println!("  [{:.4}, {:.4}, {:.4}]", r10, r11, r12);
    println!("  [{:.4}, {:.4}, {:.4}]", r20, r21, r22);

    println!("\nExpected from our quaternion multiplication:");
    println!("  [0.6124, -0.3536, std::f64::consts::FRAC_1_SQRT_2]");
    println!("  [0.7803, 0.1268, -0.6124]");
    println!("  [0.1268, 0.9268, 0.3536]");

    // The issue might be in the extraction
    println!("\nExtracting angles from this matrix:");
    println!("sin(Y) = -R[0,2] = -{:.4} = {:.4}", r02, -r02);
    println!(
        "Y = asin({:.4}) = {:.3} rad = {:.1}°",
        -r02,
        (-r02).asin(),
        (-r02).asin() * 180.0 / PI
    );

    // But wait, our input Y was positive 45°, and we're getting -45°
    // This suggests we might need a different extraction formula
}
