use ndarray::{Array1, Array2, Axis};
use plotters::prelude::*;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use scirs2_signal::{kalman, SignalResult};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

fn main() -> SignalResult<()> {
    println!("Kalman Filter Examples");

    // Example 1: Basic Kalman filtering for a noisy sine wave
    basic_kalman_filtering()?;

    // Example 2: Extended Kalman filter for nonlinear system
    extended_kalman_filtering()?;

    // Example 3: Unscented Kalman filter for nonlinear system
    unscented_kalman_filtering()?;

    // Example 4: Kalman filtering for signal denoising
    kalman_denoising()?;

    // Example 5: Adaptive Kalman filtering for time-varying noise
    adaptive_kalman_filtering()?;

    // Example 6: Robust Kalman filtering for signals with outliers
    robust_kalman_filtering()?;

    Ok(())
}

/// Generate a noisy sine wave with optional outliers
fn generate_noisy_sine(
    n_samples: usize,
    amplitude: f64,
    frequency: f64,
    noise_level: f64,
    outlier_prob: Option<f64>,
    outlier_scale: Option<f64>,
) -> Array1<f64> {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, noise_level).unwrap();
    let t = Array1::linspace(0.0, 10.0, n_samples);

    let mut signal = Array1::zeros(n_samples);
    for i in 0..n_samples {
        signal[i] = amplitude * (2.0 * PI * frequency * t[i]).sin();

        // Add Gaussian noise
        signal[i] += normal.sample(&mut rng);

        // Add outliers with specified probability
        if let (Some(prob), Some(scale)) = (outlier_prob, outlier_scale) {
            if rand::random::<f64>() < prob {
                signal[i] += normal.sample(&mut rng) * scale;
            }
        }
    }

    signal
}

/// Export signal data to CSV for external plotting
fn export_to_csv(file_name: &str, signals: &[(&str, &Array1<f64>)]) -> SignalResult<()> {
    let mut file = File::create(file_name)?;

    // Write header
    let header = signals
        .iter()
        .map(|(name, _)| name.to_string())
        .collect::<Vec<String>>()
        .join(",");
    writeln!(file, "{}", header)?;

    // Find common signal length
    let min_len = signals.iter().map(|(_, data)| data.len()).min().unwrap();

    // Write data
    for i in 0..min_len {
        let line = signals
            .iter()
            .map(|(_, data)| data[i].to_string())
            .collect::<Vec<String>>()
            .join(",");
        writeln!(file, "{}", line)?;
    }

    println!("Data exported to {}", file_name);
    Ok(())
}

/// Basic Kalman filtering example
fn basic_kalman_filtering() -> SignalResult<()> {
    println!("Basic Kalman Filtering Example");

    // Generate test signal: sine wave with noise
    let n_samples = 200;
    let amplitude = 1.0;
    let frequency = 0.5;
    let noise_level = 0.3;

    let true_signal = Array1::linspace(0.0, 10.0, n_samples)
        .map(|t| amplitude * (2.0 * PI * frequency * t).sin());

    let noisy_signal =
        generate_noisy_sine(n_samples, amplitude, frequency, noise_level, None, None);

    // Define a simple constant-velocity model
    // State is [position, velocity]
    // x(k+1) = F*x(k) + w(k)
    // z(k) = H*x(k) + v(k)
    let f = Array2::from_shape_vec((2, 2), vec![1.0, 0.05, 0.0, 1.0])?;
    let h = Array2::from_shape_vec((1, 2), vec![1.0, 0.0])?;

    // Initial state
    let initial_x = Array1::from_vec(vec![noisy_signal[0], 0.0]);

    // Configure Kalman filter
    let mut config = kalman::KalmanConfig::default();
    config.process_noise_scale = 1e-3;
    config.measurement_noise_scale = noise_level.powi(2);

    // Apply Kalman filter
    let filtered_states = kalman::kalman_filter(&noisy_signal, &f, &h, Some(initial_x), &config)?;

    // Extract position component
    let filtered_signal = filtered_states.slice(s![.., 0]).to_owned();

    // Apply optimal smoother
    let smoothed_signal = kalman::kalman_smooth(&noisy_signal, &f, &h, &config)?;

    // Calculate error metrics
    let filtered_mse = filtered_signal
        .iter()
        .zip(true_signal.iter())
        .map(|(&f, &t)| (f - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    let smoothed_mse = smoothed_signal
        .iter()
        .zip(true_signal.iter())
        .map(|(&s, &t)| (s - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    let noisy_mse = noisy_signal
        .iter()
        .zip(true_signal.iter())
        .map(|(&n, &t)| (n - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    println!("Noisy signal MSE: {:.6}", noisy_mse);
    println!("Filtered signal MSE: {:.6}", filtered_mse);
    println!("Smoothed signal MSE: {:.6}", smoothed_mse);

    // Export data for plotting
    export_to_csv(
        "kalman_basic.csv",
        &[
            ("True", &true_signal),
            ("Noisy", &noisy_signal),
            ("Filtered", &filtered_signal),
            ("Smoothed", &smoothed_signal),
        ],
    )?;

    Ok(())
}

/// Extended Kalman filter for nonlinear systems
fn extended_kalman_filtering() -> SignalResult<()> {
    println!("Extended Kalman Filtering Example");

    // Generate a nonlinear system with sinusoidal dynamics
    let n_samples = 200;
    let dt = 0.1;

    // State transition function:
    // x1(k+1) = x1(k) + dt * x2(k)
    // x2(k+1) = x2(k) + dt * (-0.1 * x2(k) - sin(x1(k)))
    let f_func = |x: &Array1<f64>| -> Array1<f64> {
        let mut x_new = Array1::zeros(2);
        x_new[0] = x[0] + dt * x[1];
        x_new[1] = x[1] + dt * (-0.1 * x[1] - (x[0]).sin());
        x_new
    };

    // Measurement function (nonlinear): z = x1^2
    let h_func = |x: &Array1<f64>| -> Array1<f64> { Array1::from_vec(vec![x[0].powi(2)]) };

    // Jacobian of state transition function
    let f_jacobian = |x: &Array1<f64>| -> Array2<f64> {
        Array2::from_shape_vec((2, 2), vec![1.0, dt, -dt * (x[0]).cos(), 1.0 - 0.1 * dt]).unwrap()
    };

    // Jacobian of measurement function
    let h_jacobian = |x: &Array1<f64>| -> Array2<f64> {
        Array2::from_shape_vec((1, 2), vec![2.0 * x[0], 0.0]).unwrap()
    };

    // Generate true state trajectory
    let mut x_true = Array1::from_vec(vec![0.0, 1.0]);
    let mut true_states = Vec::with_capacity(n_samples);
    let mut true_measurements = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        true_states.push(x_true.clone());
        true_measurements.push(h_func(&x_true)[0]);
        x_true = f_func(&x_true);
    }

    // Convert to Array1
    let true_states_x = Array1::from_vec(true_states.iter().map(|x| x[0]).collect());
    let true_states_v = Array1::from_vec(true_states.iter().map(|x| x[1]).collect());
    let true_measurements = Array1::from_vec(true_measurements);

    // Add noise to measurements
    let noise_level = 0.2;
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, noise_level).unwrap();

    let noisy_measurements = true_measurements.mapv(|x| x + normal.sample(&mut rng));

    // Create measurement array for EKF
    let mut z = Array2::zeros((n_samples, 1));
    for i in 0..n_samples {
        z[[i, 0]] = noisy_measurements[i];
    }

    // Configure EKF
    let mut config = kalman::KalmanConfig::default();
    config.process_noise_scale = 1e-2;
    config.measurement_noise_scale = noise_level.powi(2);

    // Initial state estimate
    let initial_x = Array1::from_vec(vec![0.1, 0.9]);

    // Apply Extended Kalman Filter
    let ekf_states = kalman::extended_kalman_filter(
        &z, f_func, h_func, f_jacobian, h_jacobian, initial_x, &config,
    )?;

    // Extract state components
    let ekf_states_x = ekf_states.slice(s![.., 0]).to_owned();
    let ekf_states_v = ekf_states.slice(s![.., 1]).to_owned();

    // Calculate error metrics
    let position_mse = ekf_states_x
        .iter()
        .zip(true_states_x.iter())
        .map(|(&e, &t)| (e - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    let velocity_mse = ekf_states_v
        .iter()
        .zip(true_states_v.iter())
        .map(|(&e, &t)| (e - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    println!("EKF position MSE: {:.6}", position_mse);
    println!("EKF velocity MSE: {:.6}", velocity_mse);

    // Export data for plotting
    export_to_csv(
        "kalman_ekf_position.csv",
        &[("True", &true_states_x), ("EKF", &ekf_states_x)],
    )?;

    export_to_csv(
        "kalman_ekf_velocity.csv",
        &[("True", &true_states_v), ("EKF", &ekf_states_v)],
    )?;

    export_to_csv(
        "kalman_ekf_measurement.csv",
        &[("True", &true_measurements), ("Noisy", &noisy_measurements)],
    )?;

    Ok(())
}

/// Unscented Kalman filter for nonlinear systems
fn unscented_kalman_filtering() -> SignalResult<()> {
    println!("Unscented Kalman Filtering Example");

    // Generate a nonlinear system with sinusoidal dynamics (same as EKF example)
    let n_samples = 200;
    let dt = 0.1;

    // State transition function
    let f_func = |x: &Array1<f64>| -> Array1<f64> {
        let mut x_new = Array1::zeros(2);
        x_new[0] = x[0] + dt * x[1];
        x_new[1] = x[1] + dt * (-0.1 * x[1] - (x[0]).sin());
        x_new
    };

    // Measurement function (nonlinear): z = x1^2
    let h_func = |x: &Array1<f64>| -> Array1<f64> { Array1::from_vec(vec![x[0].powi(2)]) };

    // Generate true state trajectory
    let mut x_true = Array1::from_vec(vec![0.0, 1.0]);
    let mut true_states = Vec::with_capacity(n_samples);
    let mut true_measurements = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        true_states.push(x_true.clone());
        true_measurements.push(h_func(&x_true)[0]);
        x_true = f_func(&x_true);
    }

    // Convert to Array1
    let true_states_x = Array1::from_vec(true_states.iter().map(|x| x[0]).collect());
    let true_states_v = Array1::from_vec(true_states.iter().map(|x| x[1]).collect());
    let true_measurements = Array1::from_vec(true_measurements);

    // Add noise to measurements
    let noise_level = 0.2;
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, noise_level).unwrap();

    let noisy_measurements = true_measurements.mapv(|x| x + normal.sample(&mut rng));

    // Create measurement array for UKF
    let mut z = Array2::zeros((n_samples, 1));
    for i in 0..n_samples {
        z[[i, 0]] = noisy_measurements[i];
    }

    // Configure UKF
    let mut config = kalman::KalmanConfig::default();
    config.process_noise_scale = 1e-2;
    config.measurement_noise_scale = noise_level.powi(2);

    // Initial state estimate
    let initial_x = Array1::from_vec(vec![0.1, 0.9]);

    // UKF parameters
    let alpha = 0.1; // Small alpha for high concentration around mean
    let beta = 2.0; // Optimal for Gaussian distributions
    let kappa = 0.0; // Secondary scaling parameter

    // Apply Unscented Kalman Filter
    let ukf_states = kalman::unscented_kalman_filter(
        &z, f_func, h_func, initial_x, &config, alpha, beta, kappa,
    )?;

    // Extract state components
    let ukf_states_x = ukf_states.slice(s![.., 0]).to_owned();
    let ukf_states_v = ukf_states.slice(s![.., 1]).to_owned();

    // Calculate error metrics
    let position_mse = ukf_states_x
        .iter()
        .zip(true_states_x.iter())
        .map(|(&u, &t)| (u - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    let velocity_mse = ukf_states_v
        .iter()
        .zip(true_states_v.iter())
        .map(|(&u, &t)| (u - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    println!("UKF position MSE: {:.6}", position_mse);
    println!("UKF velocity MSE: {:.6}", velocity_mse);

    // Export data for plotting
    export_to_csv(
        "kalman_ukf_position.csv",
        &[("True", &true_states_x), ("UKF", &ukf_states_x)],
    )?;

    export_to_csv(
        "kalman_ukf_velocity.csv",
        &[("True", &true_states_v), ("UKF", &ukf_states_v)],
    )?;

    Ok(())
}

/// Kalman filter for signal denoising
fn kalman_denoising() -> SignalResult<()> {
    println!("Kalman Denoising Example");

    // Generate a test signal with different noise characteristics in different regions
    let n_samples = 300;

    // Create piecewise signal with jumps and smooth regions
    let mut true_signal = Array1::zeros(n_samples);

    for i in 0..n_samples {
        if i < 100 {
            // Sine wave
            true_signal[i] = (i as f64 * 0.1).sin();
        } else if i < 200 {
            // Step function
            true_signal[i] = 1.0;
        } else {
            // Quadratic
            let x = (i - 200) as f64 / 100.0;
            true_signal[i] = x * x - 0.5;
        }
    }

    // Add noise
    let noise_level = 0.2;
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, noise_level).unwrap();

    let noisy_signal = true_signal.mapv(|x| x + normal.sample(&mut rng));

    // Apply different denoising methods

    // 1. Basic Kalman filter for denoising
    let denoised_basic =
        kalman::kalman_denoise_1d(&noisy_signal, Some(1e-4), Some(noise_level.powi(2)))?;

    // 2. Estimate noise parameters from the signal
    let (process_var, measurement_var) = kalman::estimate_noise_parameters(&noisy_signal, 10)?;
    println!("Estimated process variance: {:.6}", process_var);
    println!("Estimated measurement variance: {:.6}", measurement_var);

    let denoised_adaptive =
        kalman::kalman_denoise_1d(&noisy_signal, Some(process_var), Some(measurement_var))?;

    // 3. Robust Kalman filtering
    let denoised_robust = kalman::robust_kalman_filter(
        &noisy_signal,
        3.0, // outlier threshold
    )?;

    // Calculate error metrics
    let noisy_mse = noisy_signal
        .iter()
        .zip(true_signal.iter())
        .map(|(&n, &t)| (n - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    let basic_mse = denoised_basic
        .iter()
        .zip(true_signal.iter())
        .map(|(&d, &t)| (d - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    let adaptive_mse = denoised_adaptive
        .iter()
        .zip(true_signal.iter())
        .map(|(&d, &t)| (d - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    let robust_mse = denoised_robust
        .iter()
        .zip(true_signal.iter())
        .map(|(&d, &t)| (d - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    println!("Noisy signal MSE: {:.6}", noisy_mse);
    println!("Basic Kalman denoising MSE: {:.6}", basic_mse);
    println!("Adaptive Kalman denoising MSE: {:.6}", adaptive_mse);
    println!("Robust Kalman denoising MSE: {:.6}", robust_mse);

    // Export data for plotting
    export_to_csv(
        "kalman_denoising.csv",
        &[
            ("True", &true_signal),
            ("Noisy", &noisy_signal),
            ("Basic", &denoised_basic),
            ("Adaptive", &denoised_adaptive),
            ("Robust", &denoised_robust),
        ],
    )?;

    Ok(())
}

/// Example of adaptive Kalman filtering for time-varying noise
fn adaptive_kalman_filtering() -> SignalResult<()> {
    println!("Adaptive Kalman Filtering Example");

    // Generate a signal with time-varying noise
    let n_samples = 300;

    // Create sine wave signal
    let true_signal = Array1::linspace(0.0, 15.0, n_samples).mapv(|t| (t * 0.5).sin());

    // Add time-varying noise
    let mut noisy_signal = true_signal.clone();
    let mut rng = thread_rng();

    for i in 0..n_samples {
        // Noise variance increases with time
        let noise_level = 0.1 + (i as f64 / n_samples as f64) * 0.5;
        let normal = Normal::new(0.0, noise_level).unwrap();
        noisy_signal[i] += normal.sample(&mut rng);
    }

    // Apply standard Kalman filter
    let standard_filtered = kalman::kalman_denoise_1d(
        &noisy_signal,
        Some(1e-4),
        Some(0.3.powi(2)), // Fixed noise variance
    )?;

    // Apply adaptive Kalman filter
    let adaptive_filtered = kalman::adaptive_kalman_filter(
        &noisy_signal,
        15,   // Adaptive window size
        0.98, // Forgetting factor
    )?;

    // Calculate error metrics
    let noisy_mse = noisy_signal
        .iter()
        .zip(true_signal.iter())
        .map(|(&n, &t)| (n - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    let standard_mse = standard_filtered
        .iter()
        .zip(true_signal.iter())
        .map(|(&s, &t)| (s - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    let adaptive_mse = adaptive_filtered
        .iter()
        .zip(true_signal.iter())
        .map(|(&a, &t)| (a - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    println!("Noisy signal MSE: {:.6}", noisy_mse);
    println!("Standard Kalman MSE: {:.6}", standard_mse);
    println!("Adaptive Kalman MSE: {:.6}", adaptive_mse);

    // Export data for plotting
    export_to_csv(
        "kalman_adaptive.csv",
        &[
            ("True", &true_signal),
            ("Noisy", &noisy_signal),
            ("Standard", &standard_filtered),
            ("Adaptive", &adaptive_filtered),
        ],
    )?;

    Ok(())
}

/// Example of robust Kalman filtering for signals with outliers
fn robust_kalman_filtering() -> SignalResult<()> {
    println!("Robust Kalman Filtering Example");

    // Generate a signal with outliers
    let n_samples = 300;
    let amplitude = 1.0;
    let frequency = 0.5;
    let noise_level = 0.2;
    let outlier_prob = 0.05; // 5% chance of outlier
    let outlier_scale = 5.0; // 5x regular noise

    let true_signal = Array1::linspace(0.0, 10.0, n_samples)
        .mapv(|t| amplitude * (2.0 * PI * frequency * t).sin());

    let noisy_signal = generate_noisy_sine(
        n_samples,
        amplitude,
        frequency,
        noise_level,
        Some(outlier_prob),
        Some(outlier_scale),
    );

    // Apply standard Kalman filter
    let standard_filtered =
        kalman::kalman_denoise_1d(&noisy_signal, Some(1e-4), Some(noise_level.powi(2)))?;

    // Apply robust Kalman filter
    let robust_filtered = kalman::robust_kalman_filter(
        &noisy_signal,
        3.0, // outlier threshold (3-sigma)
    )?;

    // Calculate error metrics
    let noisy_mse = noisy_signal
        .iter()
        .zip(true_signal.iter())
        .map(|(&n, &t)| (n - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    let standard_mse = standard_filtered
        .iter()
        .zip(true_signal.iter())
        .map(|(&s, &t)| (s - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    let robust_mse = robust_filtered
        .iter()
        .zip(true_signal.iter())
        .map(|(&r, &t)| (r - t).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    println!("Noisy signal MSE: {:.6}", noisy_mse);
    println!("Standard Kalman MSE: {:.6}", standard_mse);
    println!("Robust Kalman MSE: {:.6}", robust_mse);

    // Export data for plotting
    export_to_csv(
        "kalman_robust.csv",
        &[
            ("True", &true_signal),
            ("Noisy", &noisy_signal),
            ("Standard", &standard_filtered),
            ("Robust", &robust_filtered),
        ],
    )?;

    Ok(())
}
