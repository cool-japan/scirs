use scirs2_special::{wright_bessel, wright_bessel_zeros};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Wright Bessel Functions Example");
    println!("==============================\n");

    // Compute Wright Bessel function for different parameters
    println!("Wright Bessel Function Values:");
    println!("----------------------------");

    // Standard case - rho=1, beta=1
    let rho = 1.0;
    let beta = 1.0;

    println!("J_{{1,1}}(z) for various z:");
    for z in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0] {
        match wright_bessel(rho, beta, z) {
            Ok(value) => println!("J_{{1,1}}({}) = {:.6}", z, value),
            Err(e) => println!("J_{{1,1}}({}) = Error: {}", z, e),
        }
    }
    println!();

    // Different rho values
    let beta = 1.0;
    let z = 1.0;

    println!("J_{{rho,1}}(1.0) for various rho:");
    for rho in [0.5, 1.0, 1.5, 2.0, 3.0] {
        match wright_bessel(rho, beta, z) {
            Ok(value) => println!("J_{{{},1}}(1.0) = {:.6}", rho, value),
            Err(e) => println!("J_{{{},1}}(1.0) = Error: {}", rho, e),
        }
    }
    println!();

    // Different beta values
    let rho = 1.0;
    let z = 1.0;

    println!("J_{{1,beta}}(1.0) for various beta:");
    for beta in [0.5, 1.0, 1.5, 2.0, 3.0] {
        match wright_bessel(rho, beta, z) {
            Ok(value) => println!("J_{{1,{}}}(1.0) = {:.6}", beta, value),
            Err(e) => println!("J_{{1,{}}}(1.0) = Error: {}", beta, e),
        }
    }
    println!();

    // Try to compute zeros
    println!("Wright Bessel Function Zeros:");
    println!("---------------------------");

    let rho = 1.0;
    let beta = 1.0;
    let n_zeros = 5;

    match wright_bessel_zeros(rho, beta, n_zeros) {
        Ok(zeros) => {
            println!("First {} zeros of J_{{1,1}}(z):", n_zeros);
            for (i, zero) in zeros.iter().enumerate() {
                println!("Zero #{}: {:.8}", i + 1, zero);
            }
        }
        Err(e) => println!("Could not compute zeros: {}", e),
    }

    Ok(())
}
