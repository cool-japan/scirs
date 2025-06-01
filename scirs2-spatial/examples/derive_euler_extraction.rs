use std::f64::consts::PI;

fn main() {
    println!("=== Deriving Euler Angle Extraction Formulas ===\n");

    // For XYZ intrinsic rotations (equivalent to ZYX extrinsic)
    // The combined rotation matrix is: R = Rz(c) * Ry(b) * Rx(a)

    println!("For intrinsic XYZ rotations:");
    println!("R = Rz(γ) * Ry(β) * Rx(α)\n");

    // Let's work out the matrix multiplication symbolically
    println!("Individual rotation matrices:");
    println!("Rx(α) = | 1     0      0   |");
    println!("        | 0   cos(α) -sin(α)|");
    println!("        | 0   sin(α)  cos(α)|");
    println!();
    println!("Ry(β) = | cos(β)  0  sin(β)|");
    println!("        |   0     1    0   |");
    println!("        |-sin(β)  0  cos(β)|");
    println!();
    println!("Rz(γ) = | cos(γ) -sin(γ) 0 |");
    println!("        | sin(γ)  cos(γ) 0 |");
    println!("        |   0       0    1 |");
    println!();

    // The combined matrix R = Rz * Ry * Rx gives:
    println!("Combined matrix R = Rz(γ) * Ry(β) * Rx(α):");
    println!();
    println!("R[0,0] = cos(β)*cos(γ)");
    println!("R[0,1] = sin(α)*sin(β)*cos(γ) - cos(α)*sin(γ)");
    println!("R[0,2] = cos(α)*sin(β)*cos(γ) + sin(α)*sin(γ)");
    println!();
    println!("R[1,0] = cos(β)*sin(γ)");
    println!("R[1,1] = sin(α)*sin(β)*sin(γ) + cos(α)*cos(γ)");
    println!("R[1,2] = cos(α)*sin(β)*sin(γ) - sin(α)*cos(γ)");
    println!();
    println!("R[2,0] = -sin(β)");
    println!("R[2,1] = sin(α)*cos(β)");
    println!("R[2,2] = cos(α)*cos(β)");
    println!();

    println!("Extraction formulas:");
    println!("1. β = asin(-R[2,0])");
    println!("2. If cos(β) ≠ 0:");
    println!("   α = atan2(R[2,1]/cos(β), R[2,2]/cos(β)) = atan2(R[2,1], R[2,2])");
    println!("   γ = atan2(R[1,0]/cos(β), R[0,0]/cos(β)) = atan2(R[1,0], R[0,0])");
    println!();

    // Test with known values
    let alpha = 30.0 * PI / 180.0;
    let beta = 45.0 * PI / 180.0;
    let gamma = 60.0 * PI / 180.0;

    let ca = alpha.cos();
    let sa = alpha.sin();
    let cb = beta.cos();
    let sb = beta.sin();
    let cg = gamma.cos();
    let sg = gamma.sin();

    println!("Test with α=30°, β=45°, γ=60°:");

    let r00 = cb * cg;
    let _r01 = sa * sb * cg - ca * sg;
    let _r02 = ca * sb * cg + sa * sg;
    let r10 = cb * sg;
    let _r11 = sa * sb * sg + ca * cg;
    let _r12 = ca * sb * sg - sa * cg;
    let r20 = -sb;
    let r21 = sa * cb;
    let r22 = ca * cb;

    println!("Matrix elements:");
    println!(
        "R[2,0] = {:.4} (should be -sin(45°) = -std::f64::consts::FRAC_1_SQRT_2)",
        r20
    );
    println!("R[2,1] = {:.4}", r21);
    println!("R[2,2] = {:.4}", r22);
    println!();

    let extracted_beta = (-r20).asin();
    let extracted_alpha = r21.atan2(r22);
    let extracted_gamma = r10.atan2(r00);

    println!("Extracted angles:");
    println!("α = {:.1}° (expected 30°)", extracted_alpha * 180.0 / PI);
    println!("β = {:.1}° (expected 45°)", extracted_beta * 180.0 / PI);
    println!("γ = {:.1}° (expected 60°)", extracted_gamma * 180.0 / PI);
}
