use scirs2_special::{fresnel, fresnelc, fresnels, mod_fresnel_minus, mod_fresnel_plus};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fresnel Integrals Example");
    println!("========================\n");

    // Evaluate Fresnel integrals at several points
    let test_points = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];

    println!("Fresnel Sine and Cosine Integrals:\n");
    println!("{:^10} | {:^12} | {:^12}", "x", "S(x)", "C(x)");
    println!("{:-^10} | {:-^12} | {:-^12}", "", "", "");

    for &x in &test_points {
        let (s, c) = fresnel(x)?;

        println!("{:^10.2} | {:^12.6} | {:^12.6}", x, s, c);
    }

    // Demonstrate individual functions
    println!("\nIndividual Fresnel Functions:\n");
    println!(
        "{:^10} | {:^12} | {:^12}",
        "x", "fresnels(x)", "fresnelc(x)"
    );
    println!("{:-^10} | {:-^12} | {:-^12}", "", "", "");

    for &x in &test_points {
        let s = fresnels(x)?;
        let c = fresnelc(x)?;

        println!("{:^10.2} | {:^12.6} | {:^12.6}", x, s, c);
    }

    // Modified Fresnel Plus Integrals
    println!("\nModified Fresnel Plus Integrals:\n");
    println!("{:^10} | {:^25} | {:^25}", "x", "F₊(x)", "K₊(x)");
    println!("{:-^10} | {:-^25} | {:-^25}", "", "", "");

    for &x in &test_points {
        let (f_plus, k_plus) = mod_fresnel_plus(x)?;

        println!(
            "{:^10.2} | {:^12.6} + {:^10.6}i | {:^12.6} + {:^10.6}i",
            x, f_plus.re, f_plus.im, k_plus.re, k_plus.im
        );
    }

    // Modified Fresnel Minus Integrals
    println!("\nModified Fresnel Minus Integrals:\n");
    println!("{:^10} | {:^25} | {:^25}", "x", "F₋(x)", "K₋(x)");
    println!("{:-^10} | {:-^25} | {:-^25}", "", "", "");

    for &x in &test_points {
        let (f_minus, k_minus) = mod_fresnel_minus(x)?;

        println!(
            "{:^10.2} | {:^12.6} + {:^10.6}i | {:^12.6} + {:^10.6}i",
            x, f_minus.re, f_minus.im, k_minus.re, k_minus.im
        );
    }

    // Relationship with error function
    println!("\nRelationship with the Error Function:");
    println!("For z = x(1-i)/√2:");
    println!("C(x) + iS(x) = (1+i)/2 * erf(z)\n");

    for &x in &test_points {
        // Compute the actual Fresnel integrals
        let (s, c) = fresnel(x)?;

        // Compute the value using the relationship with erf
        // In a real implementation, we would use erf for complex arguments
        // Here we just show the relationship conceptually
        println!("At x = {}, C(x) + iS(x) = {} + {}i", x, c, s);
    }

    // Asymptotic behavior
    println!("\nAsymptotic Behavior:");
    println!("As x approaches infinity, S(x) and C(x) approach 0.5");

    let large_x = 100.0;
    let (s_large, c_large) = fresnel(large_x)?;

    println!("S({}) = {}", large_x, s_large);
    println!("C({}) = {}", large_x, c_large);

    Ok(())
}
