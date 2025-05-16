use scirs2_signal::window::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Create windows of length 64
    let length = 64;

    // Generate different window types
    let hamming_win = hamming(length, true)?;
    let hann_win = hann(length, true)?;
    let blackman_win = blackman(length, true)?;
    let flattop_win = flattop(length, true)?;
    let kaiser_win = kaiser(length, 8.6, true)?;
    let boxcar_win = boxcar(length, true)?;
    let tukey_win = tukey(length, 0.5, true)?;

    // Print statistics about each window
    println!("Window comparison (length = {}):\n", length);

    println!("Hamming window:");
    print_stats(&hamming_win);

    println!("\nHann window:");
    print_stats(&hann_win);

    println!("\nBlackman window:");
    print_stats(&blackman_win);

    println!("\nFlat top window:");
    print_stats(&flattop_win);

    println!("\nKaiser window (beta=8.6):");
    print_stats(&kaiser_win);

    println!("\nBoxcar window:");
    print_stats(&boxcar_win);

    println!("\nTukey window (alpha=0.5):");
    print_stats(&tukey_win);

    Ok(())
}

fn print_stats(window: &[f64]) {
    // Calculate some basic statistics
    let min = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f64 = window.iter().sum();
    let mean = sum / window.len() as f64;

    // Calculate energy (sum of squares)
    let energy: f64 = window.iter().map(|&x| x * x).sum();

    println!("  Min value: {:.6}", min);
    println!("  Max value: {:.6}", max);
    println!("  Mean value: {:.6}", mean);
    println!("  Sum: {:.6}", sum);
    println!("  Energy: {:.6}", energy);

    // Print first and last few values
    println!(
        "  First 5 values: {:.6}, {:.6}, {:.6}, {:.6}, {:.6}",
        window[0], window[1], window[2], window[3], window[4]
    );
    let len = window.len();
    println!(
        "  Last 5 values: {:.6}, {:.6}, {:.6}, {:.6}, {:.6}",
        window[len - 5],
        window[len - 4],
        window[len - 3],
        window[len - 2],
        window[len - 1]
    );
}
