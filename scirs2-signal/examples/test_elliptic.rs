//! Simple test of elliptic filter design

use scirs2_signal::filter::ellip;

fn main() {
    println!("Testing Elliptic Filter Design");

    // Test different elliptic filter configurations
    let test_cases = vec![
        (3, 0.5, 30.0, 0.3, "lowpass"),
        (4, 0.5, 40.0, 0.3, "lowpass"),
        (5, 1.0, 35.0, 0.25, "lowpass"),
        (3, 0.5, 30.0, 0.4, "highpass"),
        (4, 1.0, 40.0, 0.35, "highpass"),
    ];

    for (i, (order, ripple, atten, cutoff, ftype)) in test_cases.iter().enumerate() {
        match ellip(*order, *ripple, *atten, *cutoff, *ftype) {
            Ok((b, a)) => {
                println!(
                    "\nTest {}: Order={}, Ripple={:.1}dB, Atten={:.0}dB, Type={}",
                    i + 1,
                    order,
                    ripple,
                    atten,
                    ftype
                );
                println!("  B coeffs: {:?}", format_coeffs(&b));
                println!("  A coeffs: {:?}", format_coeffs(&a));
                println!("  B length: {}, A length: {}", b.len(), a.len());
            }
            Err(e) => {
                println!("Test {}: Failed - {:?}", i + 1, e);
            }
        }
    }

    println!("\nElliptic filter design test completed!");
}

/// Format coefficients for display
fn format_coeffs(coeffs: &[f64]) -> String {
    coeffs
        .iter()
        .map(|&c| format!("{:.4}", c))
        .collect::<Vec<_>>()
        .join(", ")
}
