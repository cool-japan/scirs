//! Simple test of Butterworth filter design

use scirs2_signal::filter::butter;

fn main() {
    println!("Testing Butterworth Filter Design");

    // Test the specific cases that are failing
    println!("\nTest 1: 1st order lowpass at 0.5");
    match butter(1, 0.5, "lowpass") {
        Ok((b, a)) => {
            println!("  B coeffs: {:?}", b);
            println!("  A coeffs: {:?}", a);
            println!("  B length: {}, A length: {}", b.len(), a.len());
        }
        Err(e) => println!("  Failed: {:?}", e),
    }

    println!("\nTest 2: 1st order highpass at 0.5");
    match butter(1, 0.5, "highpass") {
        Ok((b, a)) => {
            println!("  B coeffs: {:?}", b);
            println!("  A coeffs: {:?}", a);
            println!("  B length: {}, A length: {}", b.len(), a.len());
        }
        Err(e) => println!("  Failed: {:?}", e),
    }

    println!("\nTest 3: 2nd order lowpass at 0.3");
    match butter(2, 0.3, "lowpass") {
        Ok((b, a)) => {
            println!("  B coeffs: {:?}", b);
            println!("  A coeffs: {:?}", a);
            println!("  B length: {}, A length: {}", b.len(), a.len());
        }
        Err(e) => println!("  Failed: {:?}", e),
    }

    println!("\nButterworth filter design test completed!");
}
