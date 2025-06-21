//! Example demonstrating the use of the phase vocoder for time stretching and pitch shifting
//!
//! This example generates a chirp signal and applies different time stretching
//! and pitch shifting operations using the phase vocoder.

use scirs2_signal::phase_vocoder::{phase_vocoder, PhaseVocoderConfig};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Phase Vocoder Example for Time Stretching and Pitch Shifting");
    println!("=========================================================");

    // Parameters for the test signal
    let sample_rate = 44100; // 44.1 kHz
    let duration = 1.0; // 1 second
    let samples = (sample_rate as f64 * duration) as usize;

    println!("Generating a chirp signal (440 Hz to 880 Hz)");
    println!(
        "Sample rate: {} Hz, Duration: {} seconds",
        sample_rate, duration
    );

    // Generate a chirp signal that goes from 440 Hz to 880 Hz
    let signal = generate_chirp(samples, 440.0, 880.0, sample_rate as f64);

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output").unwrap_or(());

    // Time stretching examples
    println!("\nApplying time stretching with different factors:");

    let stretch_factors = [0.5, 1.5, 2.0];

    for &factor in &stretch_factors {
        println!("  - Time stretch factor: {:.1}", factor);

        // Create a phase vocoder configuration
        let config = PhaseVocoderConfig {
            time_stretch: factor,
            window_size: 2048,
            hop_size: 512,
            ..Default::default()
        };

        // Apply phase vocoder
        let result = phase_vocoder(&signal, &config)?;

        let new_duration = duration * factor;
        println!("    Output duration: {:.2} seconds", new_duration);
        println!("    Output samples: {}", result.len());

        // Calculate signal statistics
        let rms = calculate_rms(&result);
        println!("    Output RMS: {:.4}", rms);
    }

    // Pitch shifting examples
    println!("\nApplying pitch shifting with different semitones:");

    let pitch_shifts = [-12.0, 7.0, 12.0]; // -1 octave, perfect fifth, +1 octave

    for &semitones in &pitch_shifts {
        let note_name = semitone_to_note_name(semitones);
        println!(
            "  - Pitch shift: {:.1} semitones ({})",
            semitones, note_name
        );

        // Create a phase vocoder configuration
        let config = PhaseVocoderConfig {
            time_stretch: 1.0, // No time stretching
            pitch_shift: Some(semitones),
            window_size: 2048,
            hop_size: 512,
            ..Default::default()
        };

        // Apply phase vocoder
        let result = phase_vocoder(&signal, &config)?;

        println!("    Output samples: {}", result.len());

        // Calculate signal statistics
        let rms = calculate_rms(&result);
        println!("    Output RMS: {:.4}", rms);
    }

    // Combined time stretching and pitch shifting example
    println!("\nCombining time stretching and pitch shifting:");

    let config = PhaseVocoderConfig {
        time_stretch: 1.5,      // 50% longer
        pitch_shift: Some(7.0), // Perfect fifth up
        window_size: 2048,
        hop_size: 512,
        ..Default::default()
    };

    // Apply phase vocoder
    let result = phase_vocoder(&signal, &config)?;

    println!(
        "  - Time stretch: {:.1}x, Pitch shift: +7 semitones (perfect fifth)",
        config.time_stretch
    );
    println!(
        "    Output duration: {:.2} seconds",
        duration * config.time_stretch
    );
    println!("    Output samples: {}", result.len());

    // Calculate signal statistics
    let rms = calculate_rms(&result);
    println!("    Output RMS: {:.4}", rms);

    // Formant preservation example
    println!("\nPitch shifting with formant preservation:");

    let config = PhaseVocoderConfig {
        time_stretch: 1.0,
        pitch_shift: Some(12.0), // One octave up
        preserve_formants: true,
        window_size: 2048,
        hop_size: 512,
        ..Default::default()
    };

    // Apply phase vocoder
    let result = phase_vocoder(&signal, &config)?;

    println!("  - Pitch shift: +12 semitones (octave) with formant preservation");
    println!("    Output samples: {}", result.len());

    // Calculate signal statistics
    let rms = calculate_rms(&result);
    println!("    Output RMS: {:.4}", rms);

    println!("\nNote: In a real application, you would save the processed audio to a file");
    println!("      or play it through an audio output device.");

    Ok(())
}

/// Generate a chirp signal (frequency sweep)
fn generate_chirp(samples: usize, start_freq: f64, end_freq: f64, sample_rate: f64) -> Vec<f64> {
    let mut signal = Vec::with_capacity(samples);

    for i in 0..samples {
        let t = i as f64 / sample_rate;
        let duration = samples as f64 / sample_rate;

        // Linear frequency sweep
        let _freq = start_freq + (end_freq - start_freq) * t / duration;

        // Calculate phase (integral of frequency)
        let phase =
            2.0 * PI * (start_freq * t + (end_freq - start_freq) * t * t / (2.0 * duration));

        signal.push(phase.sin());
    }

    signal
}

/// Calculate the RMS (Root Mean Square) value of a signal
fn calculate_rms(signal: &[f64]) -> f64 {
    let sum_squared: f64 = signal.iter().map(|&x| x * x).sum();
    (sum_squared / signal.len() as f64).sqrt()
}

/// Convert a semitone shift to a musical note name
fn semitone_to_note_name(semitones: f64) -> String {
    if semitones == 0.0 {
        return "unison".to_string();
    }

    let semitones_int = semitones.round() as i32;

    match semitones_int.abs() % 12 {
        0 => format!("{} octave", if semitones_int > 0 { "+" } else { "-" }),
        1 => format!(
            "minor 2nd {}",
            if semitones_int > 0 { "up" } else { "down" }
        ),
        2 => format!(
            "major 2nd {}",
            if semitones_int > 0 { "up" } else { "down" }
        ),
        3 => format!(
            "minor 3rd {}",
            if semitones_int > 0 { "up" } else { "down" }
        ),
        4 => format!(
            "major 3rd {}",
            if semitones_int > 0 { "up" } else { "down" }
        ),
        5 => format!(
            "perfect 4th {}",
            if semitones_int > 0 { "up" } else { "down" }
        ),
        6 => format!("tritone {}", if semitones_int > 0 { "up" } else { "down" }),
        7 => format!(
            "perfect 5th {}",
            if semitones_int > 0 { "up" } else { "down" }
        ),
        8 => format!(
            "minor 6th {}",
            if semitones_int > 0 { "up" } else { "down" }
        ),
        9 => format!(
            "major 6th {}",
            if semitones_int > 0 { "up" } else { "down" }
        ),
        10 => format!(
            "minor 7th {}",
            if semitones_int > 0 { "up" } else { "down" }
        ),
        11 => format!(
            "major 7th {}",
            if semitones_int > 0 { "up" } else { "down" }
        ),
        _ => unreachable!(),
    }
}
