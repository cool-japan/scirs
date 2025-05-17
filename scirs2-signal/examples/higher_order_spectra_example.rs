use ndarray::{Array1, Array2};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

use scirs2_signal::higher_order::{
    biamplitude, bicoherence, bispectrum, cumulative_bispectrum, detect_phase_coupling,
    skewness_spectrum, trispectrum,
};

fn main() {
    println!("Higher-Order Spectral Analysis Examples");
    println!("---------------------------------------");

    // Create test signals
    println!("\nGenerating test signals...");
    let signal_phase_coupled = generate_phase_coupled_signal();
    let signal_no_coupling = generate_uncoupled_signal();

    // Parameters for spectral analysis
    let fs = 1000.0; // Sampling frequency in Hz
    let nfft = 256; // FFT size

    // Demonstrate bispectrum on coupled signal
    println!("\n1. Computing bispectrum of signal with phase coupling...");
    let (bis_coupled, f1_axis, f2_axis) =
        bispectrum(&signal_phase_coupled, nfft, Some("hann"), None, fs).unwrap();

    // Save bispectrum to CSV for visualization
    save_matrix_to_csv("bispectrum_coupled.csv", &bis_coupled, &f1_axis, &f2_axis);
    println!("   Saved bispectrum to bispectrum_coupled.csv");
    println!("   Look for peaks at (50 Hz, 120 Hz) and related frequencies");

    // Demonstrate bispectrum on uncoupled signal
    println!("\n2. Computing bispectrum of signal without phase coupling...");
    let (bis_uncoupled, _, _) =
        bispectrum(&signal_no_coupling, nfft, Some("hann"), None, fs).unwrap();

    save_matrix_to_csv(
        "bispectrum_uncoupled.csv",
        &bis_uncoupled,
        &f1_axis,
        &f2_axis,
    );
    println!("   Saved bispectrum to bispectrum_uncoupled.csv");
    println!("   Note the absence of strong coupling peaks");

    // Demonstrate bicoherence
    println!("\n3. Computing bicoherence (normalized bispectrum)...");
    let (bicoh_coupled, (_, _)) =
        bicoherence(&signal_phase_coupled, nfft, Some("hann"), None, fs).unwrap();

    let (bicoh_uncoupled, (_, _)) =
        bicoherence(&signal_no_coupling, nfft, Some("hann"), None, fs).unwrap();

    save_matrix_to_csv(
        "bicoherence_coupled.csv",
        &bicoh_coupled,
        &f1_axis,
        &f2_axis,
    );
    save_matrix_to_csv(
        "bicoherence_uncoupled.csv",
        &bicoh_uncoupled,
        &f1_axis,
        &f2_axis,
    );
    println!("   Saved bicoherence to bicoherence_coupled.csv and bicoherence_uncoupled.csv");
    println!("   Bicoherence values are between 0 and 1, with 1 indicating perfect phase coupling");

    // Automatic detection of phase coupling
    println!("\n4. Automatically detecting phase coupling...");
    let coupling_peaks =
        detect_phase_coupling(&signal_phase_coupled, nfft, Some("hann"), fs, Some(0.6)).unwrap();

    println!("   Detected phase coupling at frequencies:");
    for (i, (f1, f2, bic)) in coupling_peaks.iter().enumerate().take(5) {
        println!(
            "     {}: ({:.1} Hz, {:.1} Hz) with bicoherence {:.3}",
            i + 1,
            f1,
            f2,
            bic
        );
    }

    // Demonstrate trispectrum (partial implementation, 2D slice)
    println!("\n5. Computing trispectrum (2D slice T(f1, f2, f1, f2))...");
    let tri_coupled = trispectrum(&signal_phase_coupled, nfft, Some("hann"), fs).unwrap();

    save_matrix_to_csv("trispectrum_slice.csv", &tri_coupled, &f1_axis, &f2_axis);
    println!("   Saved trispectrum slice to trispectrum_slice.csv");

    // Demonstrate biamplitude
    println!("\n6. Computing biamplitude (detects amplitude coupling)...");
    let (biamp, (_, _)) = biamplitude(&signal_phase_coupled, nfft, Some("hann"), fs).unwrap();

    save_matrix_to_csv("biamplitude.csv", &biamp, &f1_axis, &f2_axis);
    println!("   Saved biamplitude to biamplitude.csv");

    // Demonstrate cumulative bispectrum
    println!("\n7. Computing cumulative bispectrum...");
    let (cum_bis, bandwidth) =
        cumulative_bispectrum(&signal_phase_coupled, nfft, Some("hann"), fs).unwrap();

    save_array_to_csv("cumulative_bispectrum.csv", &cum_bis, &bandwidth);
    println!("   Saved cumulative bispectrum to cumulative_bispectrum.csv");

    // Demonstrate skewness spectrum
    println!("\n8. Computing skewness spectrum...");
    let (skewness, freq) =
        skewness_spectrum(&signal_phase_coupled, nfft, Some("hann"), fs).unwrap();

    save_array_to_csv("skewness_spectrum.csv", &skewness, &freq);
    println!("   Saved skewness spectrum to skewness_spectrum.csv");

    // Generate signals with different coupling angles
    println!("\n9. Analyzing signals with different phase coupling angles...");
    let angles = [0.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0];

    for (i, &angle) in angles.iter().enumerate() {
        let signal = generate_phase_coupled_signal_with_angle(angle);

        let (bicoh, (_, _)) = bicoherence(&signal, nfft, Some("hann"), None, fs).unwrap();

        save_matrix_to_csv(
            &format!("bicoherence_angle_{}.csv", i),
            &bicoh,
            &f1_axis,
            &f2_axis,
        );
    }

    println!("   Saved bicoherence with different coupling angles");
    println!("   Note how the phase coupling strength varies with the coupling angle");

    println!("\nDone! All results saved to CSV files for plotting.");
}

/// Generates a signal with quadratic phase coupling
fn generate_phase_coupled_signal() -> Array1<f64> {
    // Signal parameters
    let n_samples = 2048;
    let fs = 1000.0;
    let t = Array1::linspace(0.0, (n_samples as f64 - 1.0) / fs, n_samples);

    // Frequency components
    let f1 = 50.0;
    let f2 = 120.0;
    let f3 = f1 + f2; // Sum frequency will have phase coupling

    // Generate signal with phase coupling
    // x(t) = sin(2πf₁t) + sin(2πf₂t) + sin(2π(f₁+f₂)t + φ)
    // When φ = 0, we have perfect phase coupling
    let phase_coupling = 0.0;

    let signal = t.mapv(|ti| {
        (2.0 * PI * f1 * ti).sin()
            + (2.0 * PI * f2 * ti).sin()
            + 0.5 * (2.0 * PI * f3 * ti + phase_coupling).sin()
    });

    // Add some noise
    let noise_level = 0.1;
    let mut rng = rand::rng();
    let noise = Array1::from_iter(
        (0..n_samples)
            .map(|_| noise_level * (2.0 * rand::Rng::random_range(&mut rng, 0.0..1.0) - 1.0)),
    );

    signal + noise
}

/// Generates a signal with the same frequency components but without phase coupling
fn generate_uncoupled_signal() -> Array1<f64> {
    // Signal parameters
    let n_samples = 2048;
    let fs = 1000.0;
    let t = Array1::linspace(0.0, (n_samples as f64 - 1.0) / fs, n_samples);

    // Frequency components
    let f1 = 50.0;
    let f2 = 120.0;
    let f3 = f1 + f2; // Same sum frequency but with random phase

    // Generate signal without phase coupling by adding random phase to f3
    let mut rng = rand::rng();
    let random_phase = rand::Rng::random_range(&mut rng, 0.0..2.0 * PI);

    let signal = t.mapv(|ti| {
        (2.0 * PI * f1 * ti).sin()
            + (2.0 * PI * f2 * ti).sin()
            + 0.5 * (2.0 * PI * f3 * ti + random_phase).sin()
    });

    // Add some noise
    let noise_level = 0.1;
    let noise = Array1::from_iter(
        (0..n_samples)
            .map(|_| noise_level * (2.0 * rand::Rng::random_range(&mut rng, 0.0..1.0) - 1.0)),
    );

    signal + noise
}

/// Generates a signal with phase coupling and a specific coupling angle
fn generate_phase_coupled_signal_with_angle(angle: f64) -> Array1<f64> {
    // Signal parameters
    let n_samples = 2048;
    let fs = 1000.0;
    let t = Array1::linspace(0.0, (n_samples as f64 - 1.0) / fs, n_samples);

    // Frequency components
    let f1 = 50.0;
    let f2 = 120.0;
    let f3 = f1 + f2; // Sum frequency

    // Generate signal with specified phase coupling angle
    let signal = t.mapv(|ti| {
        (2.0 * PI * f1 * ti).sin()
            + (2.0 * PI * f2 * ti).sin()
            + 0.5 * (2.0 * PI * f3 * ti + angle).sin()
    });

    // Add some noise
    let noise_level = 0.1;
    let mut rng = rand::rng();
    let noise = Array1::from_iter(
        (0..n_samples)
            .map(|_| noise_level * (2.0 * rand::Rng::random_range(&mut rng, 0.0..1.0) - 1.0)),
    );

    signal + noise
}

/// Saves a 2D matrix to CSV with row and column headers
fn save_matrix_to_csv(
    filename: &str,
    matrix: &Array2<f64>,
    row_labels: &Array1<f64>,
    col_labels: &Array1<f64>,
) {
    let mut file = File::create(filename).expect(&format!("Failed to create {}", filename));

    // Write header with column labels
    write!(file, "f1/f2").expect("Failed to write header");
    for &col in col_labels.iter() {
        write!(file, ",{:.2}", col).expect("Failed to write header");
    }
    writeln!(file).expect("Failed to write header");

    // Write data with row labels
    for (i, &row_label) in row_labels.iter().enumerate() {
        write!(file, "{:.2}", row_label).expect("Failed to write data");

        for j in 0..col_labels.len() {
            write!(file, ",{:.6e}", matrix[(i, j)]).expect("Failed to write data");
        }
        writeln!(file).expect("Failed to write data");
    }
}

/// Saves two 1D arrays to CSV as columns
fn save_array_to_csv(filename: &str, array1: &Array1<f64>, array2: &Array1<f64>) {
    let mut file = File::create(filename).expect(&format!("Failed to create {}", filename));

    // Write header
    writeln!(file, "x,y").expect("Failed to write header");

    // Write data
    for i in 0..array1.len().min(array2.len()) {
        writeln!(file, "{:.6e},{:.6e}", array2[i], array1[i]).expect("Failed to write data");
    }
}
