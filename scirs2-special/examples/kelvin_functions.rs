use scirs2_special::{bei, beip, ber, berp, kei, keip, kelvin, ker, kerp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Kelvin Functions Example");
    println!("=======================\n");

    // Evaluate Kelvin functions at several points
    let test_points = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];

    println!("Individual Kelvin Function Values:\n");
    println!(
        "{:^10} | {:^12} | {:^12} | {:^12} | {:^12}",
        "x", "ber(x)", "bei(x)", "ker(x)", "kei(x)"
    );
    println!(
        "{:-^10} | {:-^12} | {:-^12} | {:-^12} | {:-^12}",
        "", "", "", "", ""
    );

    for &x in &test_points {
        let ber_x = ber(x)?;
        let bei_x = bei(x)?;
        let ker_x = ker(x)?;
        let kei_x = kei(x)?;

        println!(
            "{:^10.2} | {:^12.6} | {:^12.6} | {:^12.6} | {:^12.6}",
            x, ber_x, bei_x, ker_x, kei_x
        );
    }

    println!("\nDerivatives of Kelvin Functions:\n");
    println!(
        "{:^10} | {:^12} | {:^12} | {:^12} | {:^12}",
        "x", "berp(x)", "beip(x)", "kerp(x)", "keip(x)"
    );
    println!(
        "{:-^10} | {:-^12} | {:-^12} | {:-^12} | {:-^12}",
        "", "", "", "", ""
    );

    for &x in &test_points {
        let berp_x = berp(x)?;
        let beip_x = beip(x)?;
        let kerp_x = kerp(x)?;
        let keip_x = keip(x)?;

        println!(
            "{:^10.2} | {:^12.6} | {:^12.6} | {:^12.6} | {:^12.6}",
            x, berp_x, beip_x, kerp_x, keip_x
        );
    }

    // Using the combined function
    println!("\nKelvin Function as Complex Values:\n");
    println!(
        "{:^10} | {:^25} | {:^25}",
        "x", "ber(x) + i·bei(x)", "ker(x) + i·kei(x)"
    );
    println!("{:-^10} | {:-^25} | {:-^25}", "", "", "");

    for &x in &test_points {
        let (be, ke, _, _) = kelvin(x)?;

        println!(
            "{:^10.2} | {:^12.6} + {:^10.6}i | {:^12.6} + {:^10.6}i",
            x, be.re, be.im, ke.re, ke.im
        );
    }

    // Verify the relationship with Bessel functions
    println!("\nVerifying Relationship with Bessel Functions:\n");
    println!("For x = 1.0:");

    let x = 1.0;
    let (be, ke, bep, kep) = kelvin(x)?;

    // The relationship: ber(x) + i·bei(x) ≈ J₀(x·e^(3πi/4))
    println!("ber(x) + i·bei(x) = {} + {}i", be.re, be.im);

    // The relationship: ker(x) + i·kei(x) ≈ K₀(x·e^(π/4))
    println!("ker(x) + i·kei(x) = {} + {}i", ke.re, ke.im);

    // Derivatives
    println!("\nDerivatives at x = 1.0:");
    println!("berp(x) + i·beip(x) = {} + {}i", bep.re, bep.im);
    println!("kerp(x) + i·keip(x) = {} + {}i", kep.re, kep.im);

    // Asymptotic behavior
    println!("\nAsymptotic Behavior as x approaches zero:");
    let small_x = 1e-5;
    let (be_small, ke_small, _, _) = kelvin(small_x)?;

    println!(
        "ber({}) + i·bei({}) ≈ {} + {}i",
        small_x, small_x, be_small.re, be_small.im
    );
    println!("Expected: 1 + 0i");

    println!(
        "ker({}) + i·kei({}) ≈ {} + {}i",
        small_x, small_x, ke_small.re, ke_small.im
    );
    println!("Expected: -ln(x/2) - γ + 0i (where γ is Euler's constant)");

    Ok(())
}
