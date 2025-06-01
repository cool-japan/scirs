use std::f64::consts::PI;

fn main() {
    println!("=== Deriving ZYX Euler Angle Extraction ===\n");

    // For ZYX intrinsic rotations with angles[0]=Z, angles[1]=Y, angles[2]=X
    // The combined rotation matrix is: R = Rx * Ry * Rz

    println!("For intrinsic ZYX rotations:");
    println!("R = Rx(angles[2]) * Ry(angles[1]) * Rz(angles[0])\n");

    // Using standard notation: angles[0]=α (Z), angles[1]=β (Y), angles[2]=γ (X)
    println!("Let α = Z angle, β = Y angle, γ = X angle");
    println!("R = Rx(γ) * Ry(β) * Rz(α)\n");

    println!("The combined matrix elements are:");
    println!("R[0,0] = cos(β)*cos(α)");
    println!("R[0,1] = cos(β)*sin(α)");
    println!("R[0,2] = -sin(β)");
    println!();
    println!("R[1,0] = sin(γ)*sin(β)*cos(α) - cos(γ)*sin(α)");
    println!("R[1,1] = sin(γ)*sin(β)*sin(α) + cos(γ)*cos(α)");
    println!("R[1,2] = sin(γ)*cos(β)");
    println!();
    println!("R[2,0] = cos(γ)*sin(β)*cos(α) + sin(γ)*sin(α)");
    println!("R[2,1] = cos(γ)*sin(β)*sin(α) - sin(γ)*cos(α)");
    println!("R[2,2] = cos(γ)*cos(β)");
    println!();

    println!("Extraction formulas:");
    println!("1. β = asin(-R[0,2])");
    println!("2. If cos(β) ≠ 0:");
    println!("   α = atan2(R[0,1], R[0,0])");
    println!("   γ = atan2(R[1,2], R[2,2])");
    println!();
    println!("Remember: angles[0] = α (Z), angles[1] = β (Y), angles[2] = γ (X)");
    println!();

    // Test with known values
    let z_angle = 30.0 * PI / 180.0; // α
    let y_angle = 45.0 * PI / 180.0; // β
    let x_angle = 60.0 * PI / 180.0; // γ

    let ca = z_angle.cos();
    let sa = z_angle.sin();
    let cb = y_angle.cos();
    let sb = y_angle.sin();
    let cg = x_angle.cos();
    let sg = x_angle.sin();

    println!("Test with Z=30°, Y=45°, X=60°:");

    let r02 = -sb;
    let r01 = cb * sa;
    let r00 = cb * ca;
    let r12 = sg * cb;
    let r22 = cg * cb;

    println!("Matrix elements:");
    println!(
        "R[0,2] = {:.4} (should be -sin(45°) = -std::f64::consts::FRAC_1_SQRT_2)",
        r02
    );
    println!("R[0,1] = {:.4}", r01);
    println!("R[0,0] = {:.4}", r00);
    println!("R[1,2] = {:.4}", r12);
    println!("R[2,2] = {:.4}", r22);
    println!();

    let extracted_beta = (-r02).asin();
    let extracted_alpha = r01.atan2(r00);
    let extracted_gamma = r12.atan2(r22);

    println!("Extracted angles:");
    println!(
        "α (Z) = {:.1}° (expected 30°)",
        extracted_alpha * 180.0 / PI
    );
    println!("β (Y) = {:.1}° (expected 45°)", extracted_beta * 180.0 / PI);
    println!(
        "γ (X) = {:.1}° (expected 60°)",
        extracted_gamma * 180.0 / PI
    );
}
