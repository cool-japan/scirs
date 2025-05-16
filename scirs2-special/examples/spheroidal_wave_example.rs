use scirs2_special::{
    obl_ang1, obl_cv, obl_rad1, obl_rad2, pro_ang1, pro_cv, pro_cv_seq, pro_rad1, pro_rad2,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Spheroidal Wave Functions Example");
    println!("=================================\n");

    // Prolate spheroidal characteristic values
    println!("Prolate Spheroidal Characteristic Values");
    println!("---------------------------------------");

    // For c=0, the characteristic values are n(n+1)
    let c = 0.0;
    println!("For c = {}", c);
    for n in 0..5 {
        let cv = pro_cv(0, n, c)?;
        println!("λ_{{0,{}}}({}) = {:.6}", n, c, cv);
    }
    println!();

    // For c=1.0, use the perturbation approximation
    let c = 1.0;
    println!("For c = {}", c);
    for n in 0..5 {
        match pro_cv(0, n, c) {
            Ok(cv) => println!("λ_{{0,{}}}({}) = {:.6}", n, c, cv),
            Err(_) => println!("λ_{{0,{}}}({}) = [Not implemented for this value]", n, c),
        }
    }
    println!();

    // Sequence of prolate characteristic values
    println!("Sequence of Prolate Spheroidal Characteristic Values");
    println!("--------------------------------------------------");
    let m = 1; // Order
    let n_max = 5; // Maximum degree
    let c = 0.0;

    match pro_cv_seq(m, n_max, c) {
        Ok(values) => {
            println!("For m = {}, c = {}, and n = {}..{}", m, c, m, n_max);
            for (i, val) in values.iter().enumerate() {
                println!("λ_{{{},{}}}({}) = {:.6}", m, m + i as i32, c, val);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // Oblate spheroidal characteristic values
    println!("Oblate Spheroidal Characteristic Values");
    println!("--------------------------------------");

    // For c=0, the characteristic values are n(n+1)
    let c = 0.0;
    println!("For c = {}", c);
    for n in 0..5 {
        let cv = obl_cv(0, n, c)?;
        println!("λ_{{0,{}}}({}) = {:.6}", n, c, cv);
    }
    println!();

    // For c=1.0, use the perturbation approximation
    let c = 1.0;
    println!("For c = {}", c);
    for n in 0..5 {
        match obl_cv(0, n, c) {
            Ok(cv) => println!("λ_{{0,{}}}({}) = {:.6}", n, c, cv),
            Err(_) => println!("λ_{{0,{}}}({}) = [Not implemented for this value]", n, c),
        }
    }
    println!();

    // Angular and Radial Functions
    println!("Spheroidal Angular and Radial Functions");
    println!("--------------------------------------");

    // Try to compute some angular function values
    let m = 0;
    let n = 0;
    let c = 0.5;
    let x = 0.5; // Angular coordinate in [-1, 1]

    println!("Prolate Angular Function:");
    match pro_ang1(m, n, c, x) {
        Ok((val, deriv)) => {
            println!("S_{{{},{}}}^{{(1)}}({}, {}) = {:.6}", m, n, c, x, val);
            println!("S_{{{},{}}}^{{(1)'}}({}, {}) = {:.6}", m, n, c, x, deriv);
        }
        Err(e) => println!("Not fully implemented yet: {}", e),
    }
    println!();

    // Try to compute some radial function values
    let x_rad = 1.5; // Radial coordinate x ≥ 1

    println!("Prolate Radial Functions:");
    match pro_rad1(m, n, c, x_rad) {
        Ok((val, deriv)) => {
            println!("R_{{{},{}}}^{{(1)}}({}, {}) = {:.6}", m, n, c, x_rad, val);
            println!(
                "R_{{{},{}}}^{{(1)'}}({}, {}) = {:.6}",
                m, n, c, x_rad, deriv
            );
        }
        Err(e) => println!("Not fully implemented yet: {}", e),
    }

    match pro_rad2(m, n, c, x_rad) {
        Ok((val, deriv)) => {
            println!("R_{{{},{}}}^{{(2)}}({}, {}) = {:.6}", m, n, c, x_rad, val);
            println!(
                "R_{{{},{}}}^{{(2)'}}({}, {}) = {:.6}",
                m, n, c, x_rad, deriv
            );
        }
        Err(e) => println!("Not fully implemented yet: {}", e),
    }
    println!();

    // Oblate angular and radial functions
    println!("Oblate Angular Function:");
    match obl_ang1(m, n, c, x) {
        Ok((val, deriv)) => {
            println!("S_{{{},{}}}^{{(1)}}({}, {}) = {:.6}", m, n, c, x, val);
            println!("S_{{{},{}}}^{{(1)'}}({}, {}) = {:.6}", m, n, c, x, deriv);
        }
        Err(e) => println!("Not fully implemented yet: {}", e),
    }
    println!();

    // For oblate radial, x ≥ 0
    let x_obl = 0.5;

    println!("Oblate Radial Functions:");
    match obl_rad1(m, n, c, x_obl) {
        Ok((val, deriv)) => {
            println!("R_{{{},{}}}^{{(1)}}({}, {}) = {:.6}", m, n, c, x_obl, val);
            println!(
                "R_{{{},{}}}^{{(1)'}}({}, {}) = {:.6}",
                m, n, c, x_obl, deriv
            );
        }
        Err(e) => println!("Not fully implemented yet: {}", e),
    }

    match obl_rad2(m, n, c, x_obl) {
        Ok((val, deriv)) => {
            println!("R_{{{},{}}}^{{(2)}}({}, {}) = {:.6}", m, n, c, x_obl, val);
            println!(
                "R_{{{},{}}}^{{(2)'}}({}, {}) = {:.6}",
                m, n, c, x_obl, deriv
            );
        }
        Err(e) => println!("Not fully implemented yet: {}", e),
    }

    Ok(())
}
