//! System Identification Example
//!
//! This example demonstrates various system identification techniques
//! available in the scirs2-signal crate for estimating mathematical
//! models of dynamic systems from input-output data.

use ndarray::Array1;
use scirs2_signal::parametric::{ARMethod, OrderSelection};
use scirs2_signal::sysid::{
    estimate_frequency_response, estimate_transfer_function, identify_ar_model,
    identify_arma_model, validate_model, FreqResponseMethod, RecursiveLeastSquares, SysIdConfig,
    TfEstimationMethod,
};
use scirs2_signal::waveforms::chirp;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== System Identification Example ===\n");

    // Generate test system and signals
    let n = 500;
    let fs = 100.0;
    let t = Array1::linspace(0.0, (n - 1) as f64 / fs, n);

    // Example 1: Transfer Function Estimation
    println!("1. Transfer Function Estimation");
    println!("--------------------------------");

    // Create chirp input signal for good frequency coverage
    let t_slice = t.as_slice().unwrap();
    let input_vec = chirp(t_slice, 1.0, t[t.len() - 1], 20.0, "linear", 0.0)?;
    let input = Array1::from(input_vec);

    // Simulate a second-order system: H(s) = 1 / (s^2 + 0.8*s + 1)
    let mut output = Array1::zeros(n);
    let mut y_prev1 = 0.0;
    let mut y_prev2 = 0.0;
    let _dt = 1.0 / fs;

    for i in 0..n {
        let u = input[i];

        // Discrete-time approximation of the continuous system
        let y = 0.98 * y_prev1 - 0.96 * y_prev2 + 0.02 * u;
        output[i] = y;

        y_prev2 = y_prev1;
        y_prev1 = y;
    }

    // Add some noise
    for i in 0..n {
        output[i] += 0.01 * (i as f64 * 0.1).sin();
    }

    // Estimate transfer function using least squares
    let tf_result =
        estimate_transfer_function(&input, &output, fs, 2, 2, TfEstimationMethod::LeastSquares)?;

    println!(
        "Estimated numerator coefficients: {:?}",
        tf_result.numerator
    );
    println!(
        "Estimated denominator coefficients: {:?}",
        tf_result.denominator
    );
    println!("Model fit: {:.2}%", tf_result.fit_percentage);
    println!("Error variance: {:.6}", tf_result.error_variance);

    // Example 2: Frequency Response Estimation
    println!("\n2. Frequency Response Estimation");
    println!("----------------------------------");

    let config = SysIdConfig {
        fs,
        window: "hann".to_string(),
        overlap: 0.5,
        nfft: Some(256),
        ..Default::default()
    };

    let freq_result =
        estimate_frequency_response(&input, &output, fs, FreqResponseMethod::Welch, &config)?;

    println!(
        "Frequency response estimated at {} frequencies",
        freq_result.frequencies.len()
    );
    println!(
        "Frequency range: {:.2} to {:.2} Hz",
        freq_result.frequencies[0],
        freq_result.frequencies[freq_result.frequencies.len() - 1]
    );

    // Show coherence at a few points
    let mid_idx = freq_result.coherence.len() / 2;
    println!(
        "Coherence at {:.1} Hz: {:.3}",
        freq_result.frequencies[mid_idx], freq_result.coherence[mid_idx]
    );

    // Example 3: AR Model Identification
    println!("\n3. Autoregressive (AR) Model Identification");
    println!("--------------------------------------------");

    // Generate AR process for testing
    let mut ar_signal = Array1::zeros(200);
    for i in 2..200 {
        ar_signal[i] =
            0.7 * ar_signal[i - 1] + 0.2 * ar_signal[i - 2] + 0.1 * (i as f64 * 0.05).sin();
    }

    let ar_result = identify_ar_model(&ar_signal, 10, ARMethod::Burg, OrderSelection::AIC)?;

    println!("Optimal AR order: {}", ar_result.model_order.0);
    println!(
        "AR coefficients: {:?}",
        ar_result.ar_coefficients.slice(ndarray::s![0..5])
    );
    println!("Noise variance: {:.6}", ar_result.noise_variance);
    println!("AIC value: {:.3}", ar_result.information_criterion);

    // Example 4: ARMA Model Identification
    println!("\n4. ARMA Model Identification");
    println!("------------------------------");

    let arma_result = identify_arma_model(&ar_signal, 3, 2, OrderSelection::BIC)?;

    println!(
        "ARMA model order: AR({}) MA({})",
        arma_result.model_order.0, arma_result.model_order.1
    );
    println!("AR coefficients: {:?}", arma_result.ar_coefficients);
    if let Some(ref ma_coeffs) = arma_result.ma_coefficients {
        println!("MA coefficients: {:?}", ma_coeffs);
    }
    println!("BIC value: {:.3}", arma_result.information_criterion);

    // Example 5: Model Validation
    println!("\n5. Model Validation");
    println!("--------------------");

    // Generate predictions using the estimated transfer function
    let mut predicted = Array1::zeros(n);
    let a = &tf_result.denominator;
    let b = &tf_result.numerator;

    for i in 2..n {
        let mut pred = 0.0;

        // AR part (denominator)
        for j in 1..a.len().min(i + 1) {
            if j <= i {
                pred += a[j] * predicted[i - j];
            }
        }

        // MA part (numerator)
        for j in 0..b.len().min(i + 1) {
            if j <= i {
                pred += b[j] * input[i - j];
            }
        }

        predicted[i] = pred;
    }

    let validation = validate_model(&predicted, &output, a.len() + b.len(), true)?;

    println!("Model validation results:");
    println!("  Fit percentage: {:.2}%", validation.fit_percentage);
    println!("  R-squared: {:.4}", validation.r_squared);
    println!("  MSE: {:.6}", validation.mse);
    println!("  AIC: {:.3}", validation.aic);
    println!("  BIC: {:.3}", validation.bic);
    println!("  Whiteness test p-value: {:.3}", validation.whiteness_test);

    // Example 6: Recursive Least Squares
    println!("\n6. Recursive Least Squares (Online Estimation)");
    println!("------------------------------------------------");

    let mut rls = RecursiveLeastSquares::new(3, 0.98, 100.0);

    // Simulate online estimation with streaming data
    let _true_params = [1.5, -0.8, 0.3];
    let mut estimation_errors = Vec::new();

    for i in 10..100 {
        // Create regression vector from past outputs and inputs
        let regression = Array1::from_vec(vec![output[i - 1], output[i - 2], input[i - 1]]);

        // Current output
        let current_output = output[i];

        // Update RLS estimator
        let pred_error = rls.update(&regression, current_output)?;
        estimation_errors.push(pred_error.abs());

        if i % 20 == 0 {
            let params = rls.get_parameters();
            println!(
                "  Step {}: Parameters = [{:.3}, {:.3}, {:.3}], Error = {:.4}",
                i,
                params[0],
                params[1],
                params[2],
                pred_error.abs()
            );
        }
    }

    let final_params = rls.get_parameters();
    println!(
        "Final RLS parameters: [{:.3}, {:.3}, {:.3}]",
        final_params[0], final_params[1], final_params[2]
    );

    let avg_error = estimation_errors.iter().sum::<f64>() / estimation_errors.len() as f64;
    println!("Average prediction error: {:.4}", avg_error);

    // Example 7: Compare Multiple Methods
    println!("\n7. Method Comparison");
    println!("---------------------");

    let methods = vec![
        ("Least Squares", TfEstimationMethod::LeastSquares),
        ("Frequency Domain", TfEstimationMethod::FrequencyDomain),
        (
            "Instrumental Variable",
            TfEstimationMethod::InstrumentalVariable,
        ),
    ];

    for (name, method) in methods {
        match estimate_transfer_function(&input, &output, fs, 2, 2, method) {
            Ok(result) => {
                println!(
                    "  {}: Fit = {:.1}%, Error Var = {:.6}",
                    name, result.fit_percentage, result.error_variance
                );
            }
            Err(e) => {
                println!("  {}: Failed - {}", name, e);
            }
        }
    }

    println!("\n=== System Identification Example Complete ===");

    Ok(())
}
