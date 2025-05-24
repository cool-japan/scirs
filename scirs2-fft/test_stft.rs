use scirs2_fft::spectrogram::stft;
use scirs2_fft::window::Window;
use std::f64::consts::PI;

fn main() {
    // Generate a chirp signal
    let fs = 1000.0; // 1 kHz sampling rate
    let t = (0..1000).map(|i| i as f64 / fs).collect::<Vec<_>>();
    let chirp = t.iter().map(|&ti| (2.0 * PI * (10.0 + 10.0 * ti) * ti).sin()).collect::<Vec<_>>();

    // Compute STFT
    let result = stft(
        &chirp,
        Window::Hann,
        256,
        Some(128),
        None,
        Some(fs),
        Some(true),
        Some(true),
        None,
    );
    
    match result {
        Ok((f, t, zxx)) => {
            println!("Success! f.len()={}, t.len()={}, zxx.shape()={:?}", f.len(), t.len(), zxx.shape());
        }
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
}