//! Advanced Filter Design Example
//!
//! This example demonstrates the advanced filter design capabilities including:
//! - Improved Butterworth filters with proper analog prototypes
//! - Chebyshev Type I and Type II filters
//! - Bessel filters with maximally flat group delay
//! - Comb filters for periodic signal processing
//! - Notch and peak filters for frequency-specific processing
//! - Allpass filters for phase equalization

use scirs2_signal::filter::{
    allpass_filter, bessel, butter, cheby1, cheby2, comb_filter, ellip, group_delay, lfilter,
    notch_filter, peak_filter,
};
use std::f64::consts::PI;

fn main() {
    println!("Advanced Filter Design Example");
    println!("============================\n");

    // Example 1: Improved Butterworth Filter Design
    println!("1. Improved Butterworth Filter Design");
    println!("------------------------------------");

    let (b_butter, a_butter) = butter(4, 0.3, "lowpass").unwrap();
    println!("4th-order Butterworth lowpass filter:");
    println!("  Numerator:   {:?}", format_coeffs(&b_butter));
    println!("  Denominator: {:?}", format_coeffs(&a_butter));

    let (b_butter_hp, a_butter_hp) = butter(3, 0.4, "highpass").unwrap();
    println!("3rd-order Butterworth highpass filter:");
    println!("  Numerator:   {:?}", format_coeffs(&b_butter_hp));
    println!("  Denominator: {:?}", format_coeffs(&a_butter_hp));

    // Example 2: Chebyshev Type I Filter
    println!("\n\n2. Chebyshev Type I Filter Design");
    println!("--------------------------------");

    let (b_cheb1, a_cheb1) = cheby1(4, 0.5, 0.3, "lowpass").unwrap(); // 0.5 dB ripple
    println!("4th-order Chebyshev I filter (0.5 dB ripple):");
    println!("  Numerator:   {:?}", format_coeffs(&b_cheb1));
    println!("  Denominator: {:?}", format_coeffs(&a_cheb1));

    // Example 3: Chebyshev Type II Filter
    println!("\n\n3. Chebyshev Type II Filter Design");
    println!("---------------------------------");

    let (b_cheb2, a_cheb2) = cheby2(4, 40.0, 0.3, "lowpass").unwrap(); // 40 dB stopband attenuation
    println!("4th-order Chebyshev II filter (40 dB stopband attenuation):");
    println!("  Numerator:   {:?}", format_coeffs(&b_cheb2));
    println!("  Denominator: {:?}", format_coeffs(&a_cheb2));

    // Example 4: Elliptic (Cauer) Filter Design
    println!("\\n\\n4. Elliptic (Cauer) Filter Design");
    println!("--------------------------------");

    let (b_ellip, a_ellip) = ellip(4, 0.5, 40.0, 0.3, "lowpass").unwrap(); // 0.5 dB ripple, 40 dB attenuation
    println!("4th-order Elliptic filter (0.5 dB ripple, 40 dB stopband attenuation):");
    println!("  Numerator:   {:?}", format_coeffs(&b_ellip));
    println!("  Denominator: {:?}", format_coeffs(&a_ellip));

    // Example 5: Bessel Filter with Flat Group Delay
    println!("\n\n5. Bessel Filter Design");
    println!("----------------------");

    let (b_bessel, a_bessel) = bessel(4, 0.3, "lowpass").unwrap();
    println!("4th-order Bessel filter (maximally flat group delay):");
    println!("  Numerator:   {:?}", format_coeffs(&b_bessel));
    println!("  Denominator: {:?}", format_coeffs(&a_bessel));

    // Compare group delays
    let frequencies = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
    let gd_butter = group_delay(&b_butter, &a_butter, &frequencies).unwrap();
    let gd_bessel = group_delay(&b_bessel, &a_bessel, &frequencies).unwrap();

    println!("\nGroup delay comparison (samples):");
    println!("Freq (norm) | Butterworth | Bessel");
    println!("------------|-------------|--------");
    for (i, &freq) in frequencies.iter().enumerate() {
        println!(
            "     {:.1}    |    {:.2}    | {:.2}",
            freq, gd_butter[i], gd_bessel[i]
        );
    }

    // Example 6: Comb Filters
    println!("\n\n6. Comb Filter Design");
    println!("--------------------");

    // FIR comb filter for echo enhancement
    let (b_comb_fir, _a_comb_fir) = comb_filter(20, 0.0, 0.5).unwrap();
    println!("FIR comb filter (20-sample delay, 0.5 gain):");
    println!("  Numerator length: {}", b_comb_fir.len());
    println!(
        "  Non-zero coefficients: b[0]={:.1}, b[20]={:.1}",
        b_comb_fir[0], b_comb_fir[20]
    );

    // IIR comb filter for echo removal
    let (_b_comb_iir, a_comb_iir) = comb_filter(15, -0.7, 1.0).unwrap();
    println!("IIR comb filter (15-sample delay, -0.7 gain):");
    println!("  Denominator length: {}", a_comb_iir.len());
    println!(
        "  Non-zero coefficients: a[0]={:.1}, a[15]={:.1}",
        a_comb_iir[0], a_comb_iir[15]
    );

    // Example 7: Notch and Peak Filters
    println!("\n\n7. Notch and Peak Filters");
    println!("------------------------");

    // Notch filter to remove 60 Hz interference (assuming 1000 Hz sample rate)
    let (b_notch, a_notch) = notch_filter(0.12, 30.0).unwrap(); // 60/(1000/2) = 0.12
    println!("Notch filter (60 Hz removal, Q=30):");
    println!("  Numerator:   {:?}", format_coeffs(&b_notch));
    println!("  Denominator: {:?}", format_coeffs(&a_notch));

    // Peak filter to enhance a specific frequency
    let (b_peak, a_peak) = peak_filter(0.2, 5.0, 6.0).unwrap(); // 6 dB boost at 0.2 normalized freq
    println!("Peak filter (0.2 normalized freq, Q=5, +6dB gain):");
    println!("  Numerator:   {:?}", format_coeffs(&b_peak));
    println!("  Denominator: {:?}", format_coeffs(&a_peak));

    // Example 8: Allpass Filters
    println!("\n\n8. Allpass Filters for Phase Equalization");
    println!("----------------------------------------");

    // First-order allpass
    let (b_ap1, a_ap1) = allpass_filter(0.25, 0.9).unwrap();
    println!("1st-order allpass filter (0.25 normalized freq):");
    println!("  Numerator:   {:?}", format_coeffs(&b_ap1));
    println!("  Denominator: {:?}", format_coeffs(&a_ap1));

    // Second-order allpass
    let (b_ap2, a_ap2) = allpass_filter(0.3, 0.8).unwrap();
    println!("2nd-order allpass filter (0.3 normalized freq, Q=2):");
    println!("  Numerator:   {:?}", format_coeffs(&b_ap2));
    println!("  Denominator: {:?}", format_coeffs(&a_ap2));

    // Example 9: Filter Performance Comparison
    println!("\n\n9. Filter Performance Comparison");
    println!("-------------------------------");

    // Create test signal: sine wave + noise + interference
    let fs = 1000.0;
    let t: Vec<f64> = (0..1000).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .enumerate()
        .map(|(i, &time)| {
            0.8 * (2.0 * PI * 50.0 * time).sin()  // 50 Hz signal
                + 0.3 * (2.0 * PI * 60.0 * time).sin()  // 60 Hz interference
                + 0.1 * (2.0 * PI * 200.0 * time).sin() // High frequency component
                + 0.05 * ((i as f64 * 0.1).sin()) // Low frequency noise
        })
        .collect();

    // Apply different filters
    let filtered_butter = lfilter(&b_butter, &a_butter, &signal).unwrap();
    let filtered_cheb1 = lfilter(&b_cheb1, &a_cheb1, &signal).unwrap();
    let filtered_ellip = lfilter(&b_ellip, &a_ellip, &signal).unwrap();
    let filtered_bessel = lfilter(&b_bessel, &a_bessel, &signal).unwrap();
    let filtered_notch = lfilter(&b_notch, &a_notch, &signal).unwrap();

    // Calculate RMS values for comparison
    let rms_original = calculate_rms(&signal);
    let rms_butter = calculate_rms(&filtered_butter);
    let rms_cheb1 = calculate_rms(&filtered_cheb1);
    let rms_ellip = calculate_rms(&filtered_ellip);
    let rms_bessel = calculate_rms(&filtered_bessel);
    let rms_notch = calculate_rms(&filtered_notch);

    println!("Signal RMS comparison:");
    println!("  Original:     {:.4}", rms_original);
    println!(
        "  Butterworth:  {:.4} ({:.1}% of original)",
        rms_butter,
        100.0 * rms_butter / rms_original
    );
    println!(
        "  Chebyshev I:  {:.4} ({:.1}% of original)",
        rms_cheb1,
        100.0 * rms_cheb1 / rms_original
    );
    println!(
        "  Elliptic:     {:.4} ({:.1}% of original)",
        rms_ellip,
        100.0 * rms_ellip / rms_original
    );
    println!(
        "  Bessel:       {:.4} ({:.1}% of original)",
        rms_bessel,
        100.0 * rms_bessel / rms_original
    );
    println!(
        "  Notch (60Hz): {:.4} ({:.1}% of original)",
        rms_notch,
        100.0 * rms_notch / rms_original
    );

    // Example 10: Filter Design Guidelines
    println!("\n\n10. Filter Design Guidelines");
    println!("--------------------------");

    println!("Filter Selection Guide:");
    println!("┌─────────────┬─────────────────┬─────────────────┬─────────────────┐");
    println!("│Filter Type  │ Passband        │ Stopband        │ Group Delay     │");
    println!("├─────────────┼─────────────────┼─────────────────┼─────────────────┤");
    println!("│Butterworth  │ Maximally flat  │ Monotonic       │ Moderate        │");
    println!("│Chebyshev I  │ Equiripple      │ Monotonic       │ Nonlinear       │");
    println!("│Chebyshev II │ Monotonic       │ Equiripple      │ Nonlinear       │");
    println!("│Elliptic     │ Equiripple      │ Equiripple      │ Highly nonlinear│");
    println!("│Bessel       │ Gradual rolloff │ Poor rejection  │ Maximally flat  │");
    println!("└─────────────┴─────────────────┴─────────────────┴─────────────────┘");

    println!("\nApplication Guidelines:");
    println!("- Butterworth: General purpose, good compromise");
    println!("- Chebyshev I: Sharp cutoff, can tolerate passband ripple");
    println!("- Chebyshev II: Linear phase important, can tolerate gradual rolloff");
    println!("- Elliptic: Sharpest possible transition, minimal filter order");
    println!("- Bessel: Waveform preservation critical (pulse shaping, audio)");
    println!("- Comb: Periodic signal enhancement/suppression");
    println!("- Notch: Narrow-band interference removal");
    println!("- Allpass: Phase/group delay equalization");

    println!("\nDigital Filter Design Complete!");
    println!("Rust implementations provide excellent performance and numerical stability.");
}

/// Calculate RMS (Root Mean Square) of a signal
fn calculate_rms(signal: &[f64]) -> f64 {
    let sum_squares: f64 = signal.iter().map(|&x| x * x).sum();
    (sum_squares / signal.len() as f64).sqrt()
}

/// Format coefficients for display
fn format_coeffs(coeffs: &[f64]) -> String {
    coeffs
        .iter()
        .map(|&c| format!("{:.4}", c))
        .collect::<Vec<_>>()
        .join(", ")
}
