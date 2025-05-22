// Example program to test spherical Bessel functions
// This demonstrates the use of both regular and scaled spherical Bessel functions
// with various argument ranges

use scirs2_special::bessel::spherical::{spherical_jn, spherical_jn_scaled, spherical_yn};

fn main() {
    // Test small arguments
    println!("\nSmall Arguments (x = 0.1):");
    let x_small: f64 = 0.1;
    for n in 0..=5 {
        let jn: f64 = spherical_jn(n, x_small);
        println!("j_{}({}) = {:.10e}", n, x_small, jn);
    }

    // Test medium arguments
    println!("\nMedium Arguments (x = 5.0):");
    let x_medium: f64 = 5.0;
    for n in 0..=5 {
        let jn: f64 = spherical_jn(n, x_medium);
        let yn: f64 = spherical_yn(n, x_medium);
        println!(
            "j_{}({}) = {:.10e}, y_{}({}) = {:.10e}",
            n, x_medium, jn, n, x_medium, yn
        );
    }

    // Large arguments - demonstrate scaled functions
    println!("\nLarge Arguments (x = 100.0):");
    let x_large: f64 = 100.0;

    // Compare regular and scaled versions
    println!("Comparison of regular and scaled spherical Bessel functions:");
    for n in 0..=3 {
        let jn: f64 = spherical_jn(n, x_large);
        let jn_scaled: f64 = spherical_jn_scaled(n, x_large);
        let jn_reconstructed: f64 = jn_scaled * x_large.sin() / x_large;

        println!("j_{}({}) = {:.10e}", n, x_large, jn);
        println!("j_{}_scaled({}) = {:.10e}", n, x_large, jn_scaled);
        println!(
            "j_{}_reconstructed({}) = {:.10e}",
            n, x_large, jn_reconstructed
        );
        println!(
            "Relative error: {:.10e}\n",
            ((jn - jn_reconstructed) / jn).abs()
        );
    }

    // Demonstrate recurrence relation stability
    println!("\nRecurrence Relation Test (x = 10.0):");
    let x_rec: f64 = 10.0;
    let j0: f64 = spherical_jn(0, x_rec);
    let j1: f64 = spherical_jn(1, x_rec);
    println!("j_0({}) = {:.10e}", x_rec, j0);
    println!("j_1({}) = {:.10e}", x_rec, j1);

    let j2_recurrence: f64 = 3.0 / x_rec * j1 - j0;
    let j2_direct: f64 = spherical_jn(2, x_rec);
    println!("j_2({}) via recurrence = {:.10e}", x_rec, j2_recurrence);
    println!("j_2({}) direct = {:.10e}", x_rec, j2_direct);
    println!("Difference: {:.10e}", (j2_recurrence - j2_direct).abs());
}
