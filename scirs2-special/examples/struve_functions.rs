use scirs2_special::{it2_struve0, it_mod_struve0, it_struve0, mod_struve, struve};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Struve Functions Example");
    println!("=======================\n");

    // Evaluate Struve functions at several points
    let test_points = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];

    println!("Struve Function Values H_v(x):\n");
    println!("{:^10} | {:^12} | {:^12}", "x", "H_0(x)", "H_1(x)");
    println!("{:-^10} | {:-^12} | {:-^12}", "", "", "");

    for &x in &test_points {
        let h0_x = struve(0.0, x)?;
        let h1_x = struve(1.0, x)?;

        println!("{:^10.2} | {:^12.6} | {:^12.6}", x, h0_x, h1_x);
    }

    println!("\nModified Struve Function Values L_v(x):\n");
    println!("{:^10} | {:^12} | {:^12}", "x", "L_0(x)", "L_1(x)");
    println!("{:-^10} | {:-^12} | {:-^12}", "", "", "");

    for &x in &test_points {
        let l0_x = mod_struve(0.0, x)?;
        let l1_x = mod_struve(1.0, x)?;

        println!("{:^10.2} | {:^12.6} | {:^12.6}", x, l0_x, l1_x);
    }

    println!("\nIntegrated Struve Functions:\n");
    println!(
        "{:^10} | {:^16} | {:^16} | {:^16}",
        "x", "∫H_0(t)dt", "∫∫H_0(t)dt", "∫L_0(t)dt"
    );
    println!("{:-^10} | {:-^16} | {:-^16} | {:-^16}", "", "", "", "");

    for &x in &test_points {
        let ith0_x = it_struve0(x)?;
        let it2h0_x = it2_struve0(x)?;
        let itl0_x = it_mod_struve0(x)?;

        println!(
            "{:^10.2} | {:^16.6} | {:^16.6} | {:^16.6}",
            x, ith0_x, it2h0_x, itl0_x
        );
    }

    // Demonstrate the relationship between Struve and Bessel functions for large arguments
    println!("\nRelationship with Bessel Functions for Large Arguments:");
    println!("For large x, H_v(x) ≈ Y_v(x) + correction term");

    let large_x = 50.0;
    let h0_large = struve(0.0, large_x)?;
    let h1_large = struve(1.0, large_x)?;

    println!("H_0({}) = {}", large_x, h0_large);
    println!("H_1({}) = {}", large_x, h1_large);

    // Show special cases and boundary conditions
    println!("\nSpecial Cases:");
    println!("H_0(0) = {}", struve(0.0, 0.0)?);
    println!("L_0(0) = {}", mod_struve(0.0, 0.0)?);
    println!("∫H_0(t)dt from 0 to 0 = {}", it_struve0(0.0)?);

    Ok(())
}
