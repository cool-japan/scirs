//! Example of Parks-McClellan optimal FIR filter design using the Remez exchange algorithm

use scirs2_fft::fft;
use scirs2_signal::filter::remez;
use std::f64::consts::PI;

fn main() {
    println!("Parks-McClellan Optimal FIR Filter Design Example");
    println!("=================================================\n");

    // Example 1: Lowpass filter
    println!("1. Lowpass Filter Design");
    println!("------------------------");

    let numtaps = 65;
    let bands = vec![0.0, 0.3, 0.35, 1.0]; // Passband: 0-0.3, Stopband: 0.35-1.0
    let desired = vec![1.0, 1.0, 0.0, 0.0];
    let weights = vec![1.0, 1.0, 10.0, 10.0]; // Emphasize stopband attenuation

    let h_lp = remez(numtaps, &bands, &desired, Some(&weights), None, None).unwrap();

    println!("Lowpass filter coefficients (first 10):");
    for (i, &coeff) in h_lp.iter().take(10).enumerate() {
        println!("  h[{}] = {:.6}", i, coeff);
    }

    // Compute frequency response
    let nfft = 512;
    let mut h_padded = vec![0.0; nfft];
    h_padded[..numtaps].copy_from_slice(&h_lp);

    let h_fft = fft(&h_padded, None).unwrap();
    let _freqs: Vec<f64> = (0..nfft / 2).map(|i| i as f64 / nfft as f64).collect();

    println!("\nFrequency response at key points:");
    println!(
        "  f=0.0:   |H(f)| = {:.4} dB",
        20.0 * h_fft[0].norm().log10()
    );
    println!(
        "  f=0.15:  |H(f)| = {:.4} dB",
        20.0 * h_fft[(0.15 * nfft as f64) as usize].norm().log10()
    );
    println!(
        "  f=0.3:   |H(f)| = {:.4} dB",
        20.0 * h_fft[(0.3 * nfft as f64) as usize].norm().log10()
    );
    println!(
        "  f=0.35:  |H(f)| = {:.4} dB",
        20.0 * h_fft[(0.35 * nfft as f64) as usize].norm().log10()
    );
    println!(
        "  f=0.5:   |H(f)| = {:.4} dB",
        20.0 * h_fft[(0.5 * nfft as f64) as usize].norm().log10()
    );

    // Example 2: Bandpass filter
    println!("\n\n2. Bandpass Filter Design");
    println!("-------------------------");

    let numtaps = 101;
    let bands = vec![0.0, 0.2, 0.25, 0.45, 0.5, 1.0];
    let desired = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    let weights = vec![10.0, 10.0, 1.0, 1.0, 10.0, 10.0]; // Emphasize rejection bands

    let h_bp = remez(numtaps, &bands, &desired, Some(&weights), None, None).unwrap();

    println!("Bandpass filter coefficients (center 11):");
    let center = numtaps / 2;
    #[allow(clippy::needless_range_loop)]
    for i in (center - 5)..=(center + 5) {
        println!("  h[{}] = {:.6}", i, h_bp[i]);
    }

    // Example 3: Multiband filter
    println!("\n\n3. Multiband Filter Design");
    println!("--------------------------");

    let numtaps = 121;
    let bands = vec![0.0, 0.1, 0.15, 0.25, 0.3, 0.4, 0.45, 0.55, 0.6, 1.0];
    let desired = vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5];

    let h_mb = remez(numtaps, &bands, &desired, None, None, None).unwrap();

    println!("Multiband filter has {} taps", h_mb.len());

    // Verify symmetry
    let mut max_asymmetry = 0.0f64;
    for i in 0..(numtaps / 2) {
        let diff = (h_mb[i] - h_mb[numtaps - 1 - i]).abs();
        max_asymmetry = max_asymmetry.max(diff);
    }
    println!("Maximum asymmetry: {:.2e}", max_asymmetry);

    // Example 4: Differentiator
    println!("\n\n4. Differentiator Design");
    println!("------------------------");

    let numtaps = 81;
    let bands = vec![0.05, 0.45]; // Avoid DC and Nyquist
    let desired = vec![0.05 * PI, 0.45 * PI]; // Linear slope

    let h_diff = remez(numtaps, &bands, &desired, None, None, None).unwrap();

    println!("Differentiator coefficients (first 10):");
    for (i, &coeff) in h_diff.iter().take(10).enumerate() {
        println!("  h[{}] = {:.6}", i, coeff);
    }

    // Example 5: Hilbert transformer
    println!("\n\n5. Hilbert Transformer Design");
    println!("-----------------------------");

    let _numtaps = 81; // Must be odd for Type III filter
    let _bands = [0.05, 0.95]; // Avoid DC and Nyquist
    let _desired = [1.0, 1.0]; // Constant magnitude

    // For Hilbert transformer, we need to design an antisymmetric filter
    // This is a simplified version - a full implementation would need Type III/IV support
    println!("Hilbert transformer would require Type III filter support");
    println!("(antisymmetric impulse response with odd length)");

    // Performance characteristics
    println!("\n\nPerformance Characteristics");
    println!("---------------------------");
    println!("The Parks-McClellan algorithm:");
    println!("- Produces optimal filters in the minimax sense");
    println!("- Minimizes the maximum error between desired and actual response");
    println!("- Achieves equiripple behavior in passbands and stopbands");
    println!("- Provides the narrowest transition band for given specifications");
    println!("- Computational complexity: O(N² log N) where N is the filter order");
}
