//! Example demonstrating window analysis and comparison

// use scirs2_core::ndarray::Array1;  // Unused import
use scirs2_fft::window::{get_window, Window};
use scirs2_fft::window_extended::{
    analyze_window, compare_windows, get_extended_window, ExtendedWindow,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Window Analysis Example ===");
    println!();

    // Example 1: Analyze common windows
    println!("1. Analysis of Common Windows:");
    let n = 256;
    let sample_rate = 44100.0;

    let windows = vec![
        ("Rectangular", get_window(Window::Rectangular, n, true)?),
        ("Hann", get_window(Window::Hann, n, true)?),
        ("Hamming", get_window(Window::Hamming, n, true)?),
        ("Blackman", get_window(Window::Blackman, n, true)?),
        (
            "BlackmanHarris",
            get_window(Window::BlackmanHarris, n, true)?,
        ),
        ("FlatTop", get_window(Window::FlatTop, n, true)?),
    ];

    for (name, window) in &windows {
        let props = analyze_window(window, Some(sample_rate))?;
        println!("   {name}:");
        println!("     Main lobe width: {:.2} Hz", props.main_lobe_width);
        println!("     Sidelobe level: {:.1} dB", props.sidelobe_level_db);
        println!("     Coherent gain: {:.4}", props.coherent_gain);
        println!("     Processing gain: {:.4}", props.processing_gain);
        println!("     ENBW: {:.4} bins", props.enbw);
        println!("     Scalloping loss: {:.2} dB", props.scalloping_loss_db);
        println!();
    }

    // Example 2: Extended windows
    println!("2. Extended Window Types:");

    let extended_windows = vec![
        (
            "Chebyshev (60dB)",
            get_extended_window(
                ExtendedWindow::Chebyshev {
                    attenuation_db: 60.0,
                },
                n,
            )?,
        ),
        (
            "Slepian (0.3)",
            get_extended_window(ExtendedWindow::Slepian { width: 0.3 }, n)?,
        ),
        ("Lanczos", get_extended_window(ExtendedWindow::Lanczos, n)?),
        (
            "Poisson (2.0)",
            get_extended_window(ExtendedWindow::Poisson { alpha: 2.0 }, n)?,
        ),
        (
            "HannPoisson (2.0)",
            get_extended_window(ExtendedWindow::HannPoisson { alpha: 2.0 }, n)?,
        ),
        (
            "Taylor (4, -30dB)",
            get_extended_window(
                ExtendedWindow::Taylor {
                    n_sidelobes: 4,
                    sidelobe_level_db: -30.0,
                },
                n,
            )?,
        ),
    ];

    for (name, window) in &extended_windows {
        let props = analyze_window(window, Some(sample_rate))?;
        println!("   {name}:");
        println!("     Main lobe width: {:.2} Hz", props.main_lobe_width);
        println!("     Sidelobe level: {:.1} dB", props.sidelobe_level_db);
        println!("     Processing gain: {:.4}", props.processing_gain);
        println!();
    }

    // Example 3: Window comparison for specific applications
    println!("3. Application-Specific Window Selection:");

    // For spectral analysis (low sidelobes)
    println!("   Best for spectral analysis (lowest sidelobes):");
    let spectral_windows = vec![
        ("Blackman", get_window(Window::Blackman, n, true)?),
        (
            "BlackmanHarris",
            get_window(Window::BlackmanHarris, n, true)?,
        ),
        (
            "Chebyshev (80dB)",
            get_extended_window(
                ExtendedWindow::Chebyshev {
                    attenuation_db: 80.0,
                },
                n,
            )?,
        ),
        (
            "Taylor (6, -40dB)",
            get_extended_window(
                ExtendedWindow::Taylor {
                    n_sidelobes: 6,
                    sidelobe_level_db: -40.0,
                },
                n,
            )?,
        ),
    ];

    let mut best_sidelobe = ("", 0.0);
    for (name, window) in &spectral_windows {
        let props = analyze_window(window, Some(sample_rate))?;
        if props.sidelobe_level_db < best_sidelobe.1 {
            best_sidelobe = (name, props.sidelobe_level_db);
        }
        println!("     {}: {:.1} dB sidelobes", name, props.sidelobe_level_db);
    }
    println!(
        "     Best: {} with {:.1} dB",
        best_sidelobe.0, best_sidelobe.1
    );
    println!();

    // For amplitude accuracy (low scalloping loss)
    println!("   Best for amplitude accuracy (lowest scalloping loss):");
    let amplitude_windows = vec![
        ("FlatTop", get_window(Window::FlatTop, n, true)?),
        ("Hann", get_window(Window::Hann, n, true)?),
        ("Hamming", get_window(Window::Hamming, n, true)?),
    ];

    let mut best_scalloping = ("", f64::INFINITY);
    for (name, window) in &amplitude_windows {
        let props = analyze_window(window, Some(sample_rate))?;
        if props.scalloping_loss_db < best_scalloping.1 {
            best_scalloping = (name, props.scalloping_loss_db);
        }
        println!(
            "     {}: {:.3} dB scalloping loss",
            name, props.scalloping_loss_db
        );
    }
    println!(
        "     Best: {} with {:.3} dB",
        best_scalloping.0, best_scalloping.1
    );
    println!();

    // Example 4: Parameterized windows
    println!("4. Parameterized Window Comparison:");

    // Kaiser window with different beta values
    println!("   Kaiser window (varying beta):");
    let kaiser_betas = [2.0, 5.0, 8.0, 12.0];
    let mut kaiser_windows = Vec::new();

    for &beta in &kaiser_betas {
        let window = get_window(Window::Kaiser(beta), n, true)?;
        let props = analyze_window(&window, Some(sample_rate))?;
        kaiser_windows.push((format!("Kaiser(β={beta})"), window));
        println!(
            "     β={}: main lobe {:.2} Hz, sidelobes {:.1} dB",
            beta, props.main_lobe_width, props.sidelobe_level_db
        );
    }
    println!();

    // Tukey window with different alpha values
    println!("   Tukey window (varying alpha):");
    let tukey_alphas = [0.0, 0.25, 0.5, 0.75, 1.0];

    for &alpha in &tukey_alphas {
        let window = get_window(Window::Tukey(alpha), n, true)?;
        let props = analyze_window(&window, Some(sample_rate))?;
        println!(
            "     α={}: main lobe {:.2} Hz, sidelobes {:.1} dB",
            alpha, props.main_lobe_width, props.sidelobe_level_db
        );
    }
    println!();

    // Example 5: Window comparison summary
    println!("5. Window Comparison Summary:");

    let all_windows = vec![
        (
            "Rectangular".to_string(),
            get_window(Window::Rectangular, n, true)?,
        ),
        ("Hann".to_string(), get_window(Window::Hann, n, true)?),
        ("Hamming".to_string(), get_window(Window::Hamming, n, true)?),
        (
            "Blackman".to_string(),
            get_window(Window::Blackman, n, true)?,
        ),
        (
            "Kaiser(8)".to_string(),
            get_window(Window::Kaiser(8.0), n, true)?,
        ),
        (
            "Chebyshev(60dB)".to_string(),
            get_extended_window(
                ExtendedWindow::Chebyshev {
                    attenuation_db: 60.0,
                },
                n,
            )?,
        ),
    ];

    let comparison = compare_windows(&all_windows)?;

    // Create comparison table
    println!("   Window          | Main Lobe | Sidelobes | ENBW  | Scalloping");
    println!("   ----------------|-----------|-----------|-------|----------");
    for (name, props) in comparison {
        println!(
            "   {:<15} | {:>8.2} Hz | {:>8.1} dB | {:>5.2} | {:>7.2} dB",
            name,
            props.main_lobe_width,
            props.sidelobe_level_db,
            props.enbw,
            props.scalloping_loss_db
        );
    }
    println!();

    // Example 6: Custom window design recommendations
    println!("6. Window Selection Guide:");
    println!("   - For general purpose: Hann or Hamming");
    println!("   - For high dynamic range: Blackman-Harris or Chebyshev");
    println!("   - For amplitude accuracy: FlatTop");
    println!("   - For transient analysis: Rectangular or low-alpha Tukey");
    println!("   - For narrow-band signals: High-beta Kaiser or Taylor");
    println!("   - For minimizing leakage: Slepian (DPSS) windows");

    Ok(())
}
