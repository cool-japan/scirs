use scirs2_special::*;

fn main() {
    // Test j0 values
    println!("J0 function values:");
    println!("j0(0.5) = {}", j0(0.5));
    println!("j0(1.0) = {}", j0(1.0));
    println!("j0(5.0) = {}", j0(5.0));
    println!("j0(10.0) = {}", j0(10.0));
    println!();

    // Test j1 values
    println!("J1 function values:");
    println!("j1(0.5) = {}", j1(0.5));
    println!("j1(1.0) = {}", j1(1.0));
    println!("j1(5.0) = {}", j1(5.0));
    println!("j1(10.0) = {}", j1(10.0));
    println!();

    // Test i0 values
    println!("I0 function values:");
    println!("i0(0.5) = {}", i0(0.5));
    println!("i0(1.0) = {}", i0(1.0));
    println!("i0(5.0) = {}", i0(5.0));
    println!();

    // Test i1 values
    println!("I1 function values:");
    println!("i1(0.5) = {}", i1(0.5));
    println!("i1(1.0) = {}", i1(1.0));
    println!("i1(5.0) = {}", i1(5.0));
    println!();

    // Test y0 values
    println!("Y0 function values:");
    println!("y0(0.5) = {}", y0(0.5));
    println!("y0(1.0) = {}", y0(1.0));
    println!("y0(5.0) = {}", y0(5.0));
    println!("y0(10.0) = {}", y0(10.0));
    println!();

    // Test jn values for n=2
    println!("Jn function values for n=2:");
    println!("jn(2, 0.5) = {}", jn(2, 0.5));
    println!("jn(2, 1.0) = {}", jn(2, 1.0));
    println!("jn(2, 5.0) = {}", jn(2, 5.0));
    println!();

    // Test iv values for v=2
    println!("Iv function values for v=2:");
    println!("iv(2.0, 0.5) = {}", iv(2.0, 0.5));
    println!("iv(2.0, 1.0) = {}", iv(2.0, 1.0));
    println!("iv(2.0, 5.0) = {}", iv(2.0, 5.0));
    println!();

    // Test gamma function values
    println!("Gamma function values:");
    println!("gamma(0.5) = {}", gamma(0.5));
    println!("gamma(1.0) = {}", gamma(1.0));
    println!("gamma(5.0) = {}", gamma(5.0));
    println!();

    // Test gammaln function values
    println!("Gammaln function values:");
    println!("gammaln(0.5) = {}", gammaln(0.5));
    println!("gammaln(1.0) = {}", gammaln(1.0));
    println!("gammaln(5.0) = {}", gammaln(5.0));
    println!();

    // Test beta function values
    println!("Beta function values:");
    println!("beta(0.5, 0.5) = {}", beta(0.5, 0.5));
    println!("beta(1.0, 1.0) = {}", beta(1.0, 1.0));
    println!("beta(5.0, 2.0) = {}", beta(5.0, 2.0));
    println!();

    // Test betaln function values
    println!("Betaln function values:");
    println!("betaln(0.5, 0.5) = {}", betaln(0.5, 0.5));
    println!("betaln(1.0, 1.0) = {}", betaln(1.0, 1.0));
    println!("betaln(5.0, 2.0) = {}", betaln(5.0, 2.0));
    println!();

    // Test Wright omega values
    println!("Wright omega values:");
    println!(
        "wright_omega_real(0.0, 1e-10) = {}",
        wright_omega_real(0.0, 1e-10).unwrap()
    );
    println!(
        "wright_omega_real(1.0, 1e-10) = {}",
        wright_omega_real(1.0, 1e-10).unwrap()
    );
}
