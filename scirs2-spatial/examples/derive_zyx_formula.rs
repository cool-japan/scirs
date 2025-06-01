use std::f64::consts::PI;

fn main() {
    println!("=== Deriving ZYX Formula ===\n");

    // For ZYX intrinsic: Qx * Qy * Qz
    // Qz = [cz, 0, 0, sz]
    // Qy = [cy, 0, sy, 0]
    // Qx = [cx, sx, 0, 0]

    println!("Step 1: Qy * Qz");
    println!("Qy = [cy, 0, sy, 0]");
    println!("Qz = [cz, 0, 0, sz]");
    println!();
    println!("Using quaternion multiplication formula:");
    println!("(w1, x1, y1, z1) * (w2, x2, y2, z2) = ");
    println!("  w: w1*w2 - x1*x2 - y1*y2 - z1*z2");
    println!("  x: w1*x2 + x1*w2 + y1*z2 - z1*y2");
    println!("  y: w1*y2 - x1*z2 + y1*w2 + z1*x2");
    println!("  z: w1*z2 + x1*y2 - y1*x2 + z1*w2");
    println!();

    println!("Qy * Qz = ");
    println!("  w: cy*cz - 0*0 - sy*0 - 0*sz = cy*cz");
    println!("  x: cy*0 + 0*cz + sy*sz - 0*0 = sy*sz");
    println!("  y: cy*0 - 0*sz + sy*cz + 0*0 = sy*cz");
    println!("  z: cy*sz + 0*0 - sy*0 + 0*cz = cy*sz");
    println!();
    println!("Qy * Qz = [cy*cz, sy*sz, sy*cz, cy*sz]");
    println!();

    println!("Step 2: Qx * (Qy * Qz)");
    println!("Qx = [cx, sx, 0, 0]");
    println!("Qy*Qz = [cy*cz, sy*sz, sy*cz, cy*sz]");
    println!();

    println!("Qx * (Qy * Qz) = ");
    println!("  w: cx*(cy*cz) - sx*(sy*sz) - 0*(sy*cz) - 0*(cy*sz)");
    println!("    = cx*cy*cz - sx*sy*sz");
    println!();
    println!("  x: cx*(sy*sz) + sx*(cy*cz) + 0*(cy*sz) - 0*(sy*cz)");
    println!("    = cx*sy*sz + sx*cy*cz");
    println!();
    println!("  y: cx*(sy*cz) - sx*(cy*sz) + 0*(cy*cz) + 0*(sy*sz)");
    println!("    = cx*sy*cz - sx*cy*sz");
    println!();
    println!("  z: cx*(cy*sz) + sx*(sy*cz) - 0*(sy*sz) + 0*(cy*cz)");
    println!("    = cx*cy*sz + sx*sy*cz");
    println!();

    println!("Final formula for ZYX:");
    println!("w = cx*cy*cz - sx*sy*sz");
    println!("x = cx*sy*sz + sx*cy*cz");
    println!("y = cx*sy*cz - sx*cy*sz");
    println!("z = cx*cy*sz + sx*sy*cz");
    println!();

    // Test with actual values
    let z_angle = 30.0 * PI / 180.0;
    let y_angle = 45.0 * PI / 180.0;
    let x_angle = 60.0 * PI / 180.0;

    let cx = (x_angle / 2.0).cos();
    let sx = (x_angle / 2.0).sin();
    let cy = (y_angle / 2.0).cos();
    let sy = (y_angle / 2.0).sin();
    let cz = (z_angle / 2.0).cos();
    let sz = (z_angle / 2.0).sin();

    let w = cx * cy * cz - sx * sy * sz;
    let x = cx * sy * sz + sx * cy * cz;
    let y = cx * sy * cz - sx * cy * sz;
    let z = cx * cy * sz + sx * sy * cz;

    println!("Test with Z=30°, Y=45°, X=60°:");
    println!("Result: [{:.4}, {:.4}, {:.4}, {:.4}]", w, x, y, z);
    println!("Expected: [0.7233, 0.5320, 0.2006, 0.3919]");
}
