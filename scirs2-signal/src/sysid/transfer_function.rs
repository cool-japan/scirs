// Transfer function estimation methods for system identification

use super::frequency_response::{
    estimate_frequency_response, fit_parametric_to_frequency_response,
};
use super::n4sid_identification;
use super::types::{FreqResponseMethod, SysIdConfig, TfEstimationMethod, TfEstimationResult};
use super::utils::{calculate_fit_percentage, solve_linear_system};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Estimate transfer function from input-output data
///
/// # Arguments
/// * `input` - Input signal
/// * `output` - Output signal
/// * `fs` - Sampling frequency
/// * `num_order` - Numerator order
/// * `den_order` - Denominator order
/// * `method` - Estimation method
///
/// # Returns
/// * Transfer function estimation result
///
/// # Example
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_signal::sysid::{estimate_transfer_function, TfEstimationMethod};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
///
/// // Create longer test signals to avoid singular matrix
/// let input = Array1::from_vec(vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0]);
/// let output = Array1::from_vec(vec![0.0, 0.5, 0.65, 0.725, 0.7625, 0.68125, 0.540625, 0.2703125, 0.13515625, 0.067578125]);
/// let fs = 1.0;
///
/// let result = estimate_transfer_function(
///     &input, &output, fs, 1, 1, TfEstimationMethod::LeastSquares
/// )?;
///
/// // Should estimate something like H(z) = 0.5 / (z - 0.5)
/// println!("Estimated transfer function with {} numerator and {} denominator coefficients",
///          result.numerator.len(), result.denominator.len());
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn estimate_transfer_function(
    input: &Array1<f64>,
    output: &Array1<f64>,
    fs: f64,
    num_order: usize,
    den_order: usize,
    method: TfEstimationMethod,
) -> SignalResult<TfEstimationResult> {
    if input.len() != output.len() {
        return Err(SignalError::ValueError(
            "Input and output signals must have the same length".to_string(),
        ));
    }

    if input.len() < num_order + den_order + 1 {
        return Err(SignalError::ValueError(
            "Signal length insufficient for specified model orders".to_string(),
        ));
    }

    match method {
        TfEstimationMethod::LeastSquares => {
            estimate_tf_least_squares(input, output, fs, num_order, den_order)
        }
        TfEstimationMethod::FrequencyDomain => {
            estimate_tf_frequency_domain(input, output, fs, num_order, den_order)
        }
        TfEstimationMethod::InstrumentalVariable => {
            estimate_tf_instrumental_variable(input, output, fs, num_order, den_order)
        }
        TfEstimationMethod::Subspace => {
            estimate_tf_subspace(input, output, fs, num_order, den_order)
        }
    }
}

/// Least squares transfer function estimation
#[allow(dead_code)]
fn estimate_tf_least_squares(
    input: &Array1<f64>,
    output: &Array1<f64>,
    _fs: f64,
    num_order: usize,
    den_order: usize,
) -> SignalResult<TfEstimationResult> {
    let n = input.len();
    let total_order = num_order + den_order;

    if n <= total_order {
        return Err(SignalError::ValueError(
            "Insufficient data for specified model orders".to_string(),
        ));
    }

    // Build the regression matrix for ARX model: A(z)y(k) = B(z)u(k) + e(k)
    let data_length = n - total_order;
    let param_count = num_order + den_order + 1;

    let mut phi = Array2::<f64>::zeros((data_length, param_count));
    let mut y_vec = Array1::<f64>::zeros(data_length);

    for i in 0..data_length {
        let t = i + total_order;

        // Output regression vector (negative AR terms)
        for j in 1..=den_order {
            phi[[i, j - 1]] = -output[t - j];
        }

        // Input regression vector
        for j in 0..=num_order {
            if t >= j {
                phi[[i, den_order + j]] = input[t - j];
            }
        }

        y_vec[i] = output[t];
    }

    // Solve least squares problem: phi * theta = y
    let phi_t = phi.t();
    let phi_t_phi = phi_t.dot(&phi);
    let phi_t_y = phi_t.dot(&y_vec);

    // Add regularization if needed
    let ata = phi_t_phi;
    if let Some(reg) = None::<f64> {
        let mut ata_mut = ata.clone();
        for i in 0..ata_mut.nrows() {
            ata_mut[[i, i]] += reg;
        }
        let theta = solve_linear_system(&ata_mut, &phi_t_y)?;
        return build_tf_result(num_order, den_order, &theta, &phi, &y_vec);
    }

    let theta = solve_linear_system(&ata, &phi_t_y)?;
    build_tf_result(num_order, den_order, &theta, &phi, &y_vec)
}

/// Extract TfEstimationResult from solved parameter vector
fn build_tf_result(
    num_order: usize,
    den_order: usize,
    theta: &Array1<f64>,
    phi: &Array2<f64>,
    y_vec: &Array1<f64>,
) -> SignalResult<TfEstimationResult> {
    // Extract denominator and numerator coefficients
    let mut denominator = Array1::<f64>::zeros(den_order + 1);
    denominator[0] = 1.0;
    for i in 1..=den_order {
        denominator[i] = theta[i - 1];
    }

    let mut numerator = Array1::<f64>::zeros(num_order + 1);
    for i in 0..=num_order {
        numerator[i] = theta[den_order + i];
    }

    // Calculate model fit
    let y_pred = phi.dot(theta);
    let fit_percentage = calculate_fit_percentage(y_vec, &y_pred);

    // Calculate error variance
    let residuals = y_vec - &y_pred;
    let sq = residuals.mapv(|x| x * x);
    let error_variance = if !sq.is_empty() {
        sq.sum() / sq.len() as f64
    } else {
        0.0
    };

    Ok(TfEstimationResult {
        numerator,
        denominator,
        fit_percentage,
        error_variance,
        frequency_response: None,
        frequencies: None,
    })
}

/// Frequency domain transfer function estimation using spectral methods
#[allow(dead_code)]
fn estimate_tf_frequency_domain(
    input: &Array1<f64>,
    output: &Array1<f64>,
    fs: f64,
    num_order: usize,
    den_order: usize,
) -> SignalResult<TfEstimationResult> {
    // Estimate frequency response first
    let freq_result = estimate_frequency_response(
        input,
        output,
        fs,
        FreqResponseMethod::Welch,
        &SysIdConfig::default(),
    )?;

    // Fit parametric model to frequency response
    fit_parametric_to_frequency_response(
        &freq_result.frequency_response,
        &freq_result.frequencies,
        num_order,
        den_order,
    )
}

/// Instrumental variable method for transfer function estimation
#[allow(dead_code)]
fn estimate_tf_instrumental_variable(
    input: &Array1<f64>,
    output: &Array1<f64>,
    _fs: f64,
    num_order: usize,
    den_order: usize,
) -> SignalResult<TfEstimationResult> {
    // For now, use a simplified IV approach where instruments are delayed inputs
    let n = input.len();
    let total_order = num_order + den_order;
    let delay = 1; // Instrument delay

    if n <= total_order + delay {
        return Err(SignalError::ValueError(
            "Insufficient data for IV estimation".to_string(),
        ));
    }

    let data_length = n - total_order - delay;
    let param_count = num_order + den_order + 1;

    let mut phi = Array2::<f64>::zeros((data_length, param_count));
    let mut z = Array2::<f64>::zeros((data_length, param_count)); // Instruments
    let mut y_vec = Array1::<f64>::zeros(data_length);

    for i in 0..data_length {
        let t = i + total_order + delay;

        // Regression vector
        for j in 1..=den_order {
            phi[[i, j - 1]] = -output[t - j];
        }
        for j in 0..=num_order {
            if t >= j {
                phi[[i, den_order + j]] = input[t - j];
            }
        }

        // Instruments (delayed inputs and past outputs)
        for j in 1..=den_order {
            z[[i, j - 1]] = -output[t - j - delay];
        }
        for j in 0..=num_order {
            if t >= j + delay {
                z[[i, den_order + j]] = input[t - j - delay];
            }
        }

        y_vec[i] = output[t];
    }

    // IV estimation: theta = (Z'Phi)^(-1) Z'y
    let z_t = z.t();
    let z_t_phi = z_t.dot(&phi);
    let z_t_y = z_t.dot(&y_vec);

    let theta = solve_linear_system(&z_t_phi, &z_t_y)?;

    // Extract coefficients
    let mut denominator = Array1::<f64>::zeros(den_order + 1);
    denominator[0] = 1.0;
    for i in 1..=den_order {
        denominator[i] = theta[i - 1];
    }

    let mut numerator = Array1::<f64>::zeros(num_order + 1);
    for i in 0..=num_order {
        numerator[i] = theta[den_order + i];
    }

    // Calculate fit
    let y_pred = phi.dot(&theta);
    let fit_percentage = calculate_fit_percentage(&y_vec, &y_pred);
    let residuals = &y_vec - &y_pred;
    let sq = residuals.mapv(|x| x * x);
    let error_variance = if !sq.is_empty() {
        sq.sum() / sq.len() as f64
    } else {
        0.0
    };

    Ok(TfEstimationResult {
        numerator,
        denominator,
        fit_percentage,
        error_variance,
        frequency_response: None,
        frequencies: None,
    })
}

/// Subspace-based transfer function estimation via the N4SID algorithm.
///
/// Identifies a SISO state-space model `(A, B, C, D)` of order `den_order`
/// from the input/output data using [`n4sid_identification`] (a genuine
/// Hankel/SVD-based subspace method), then converts it to an equivalent
/// transfer function via the Faddeev-LeVerrier algorithm, which computes
/// the characteristic polynomial `det(zI-A)` and the adjugate `adj(zI-A)`
/// simultaneously -- the standard, numerically stable state-space-to-
/// transfer-function conversion. This replaces a previous stand-in that
/// silently fell back to ordinary least squares regardless of the
/// requested method.
///
/// The resulting numerator/denominator use the same `z^-1` (delay
/// operator) convention as [`estimate_tf_least_squares`]: `denominator =
/// [1, a_1, ..., a_den_order]` and `numerator = [b_0, b_1, ..., b_num_order]`
/// such that `(1 + a_1 z^-1 + ...) Y(z) = (b_0 + b_1 z^-1 + ...) U(z)`. Since
/// subspace identification naturally produces a numerator of the same
/// order as the identified state dimension (`den_order`), a requested
/// `num_order` smaller than `den_order` truncates the higher-delay
/// coefficients, and a larger one zero-pads.
#[allow(dead_code)]
fn estimate_tf_subspace(
    input: &Array1<f64>,
    output: &Array1<f64>,
    _fs: f64,
    num_order: usize,
    den_order: usize,
) -> SignalResult<TfEstimationResult> {
    let state_order = den_order.max(1);
    let n_samples = input.len();

    // Choose the largest past/future Hankel horizon (>= state_order + 1,
    // so the block Hankel matrices have enough rows to identify a model of
    // this order) that still leaves enough data columns.
    let mut horizon = state_order + 1;
    while horizon > state_order && n_samples < 2 * horizon + state_order + 2 {
        horizon -= 1;
    }
    if n_samples < 2 * horizon + state_order + 2 {
        return Err(SignalError::ValueError(format!(
            "Insufficient data for subspace (N4SID) estimation: need at least {} samples for state order {}, got {}",
            2 * (state_order + 1) + state_order + 2,
            state_order,
            n_samples
        )));
    }

    let (a_mat, b_mat, c_mat, d_mat) =
        n4sid_identification(input, output, state_order, horizon, horizon)?;

    let n = state_order;
    let d_scalar = d_mat[[0, 0]];

    // Faddeev-LeVerrier algorithm: computes det(zI-A) = z^n + a_1 z^{n-1}
    // + ... + a_n (coefficients `char_coeffs[1..=n]`) together with the
    // matrix-polynomial coefficients of adj(zI-A) = M_0 z^{n-1} + M_1
    // z^{n-2} + ... + M_{n-1} z^0, via the recursion M_0 = I,
    // a_k = -trace(A*M_{k-1})/k, M_k = A*M_{k-1} + a_k*I.
    let mut char_coeffs = vec![0.0; n + 1];
    char_coeffs[0] = 1.0;
    let mut m_prev = Array2::<f64>::eye(n);
    let mut adjugate_terms: Vec<Array2<f64>> = Vec::with_capacity(n);
    adjugate_terms.push(m_prev.clone());

    for k in 1..=n {
        let a_m = a_mat.dot(&m_prev);
        let trace: f64 = (0..n).map(|i| a_m[[i, i]]).sum();
        let coeff = -trace / k as f64;
        char_coeffs[k] = coeff;

        let mut m_k = a_m;
        for i in 0..n {
            m_k[[i, i]] += coeff;
        }
        if k < n {
            adjugate_terms.push(m_k.clone());
        }
        m_prev = m_k;
    }

    // Numerator (z^-1 convention): b_0 = D; b_{k+1} = C*M_k*B + D*a_{k+1}
    // for k = 0..n-1 (from H(z) = [D*det(zI-A) + C*adj(zI-A)*B] /
    // det(zI-A), rewritten in powers of z^-1 by dividing through by z^n).
    let mut numerator_full = vec![0.0; n + 1];
    numerator_full[0] = d_scalar;
    for k in 0..n {
        let cmb = c_mat.dot(&adjugate_terms[k]).dot(&b_mat);
        numerator_full[k + 1] = cmb[[0, 0]] + d_scalar * char_coeffs[k + 1];
    }
    let denominator_full = char_coeffs;

    // Match the caller's requested orders by truncating or zero-padding.
    let mut numerator = Array1::<f64>::zeros(num_order + 1);
    for (i, coeff) in numerator.iter_mut().enumerate() {
        *coeff = numerator_full.get(i).copied().unwrap_or(0.0);
    }
    let mut denominator = Array1::<f64>::zeros(den_order + 1);
    denominator[0] = 1.0;
    for i in 1..=den_order {
        denominator[i] = denominator_full.get(i).copied().unwrap_or(0.0);
    }

    // Evaluate one-step-ahead ARX-style prediction fit quality, exactly as
    // the other estimation methods report it, for a fair comparison.
    let total_order = num_order + den_order;
    let (fit_percentage, error_variance) = if n_samples > total_order {
        let data_length = n_samples - total_order;
        let mut y_pred = Array1::<f64>::zeros(data_length);
        let mut y_actual = Array1::<f64>::zeros(data_length);
        for i in 0..data_length {
            let t = i + total_order;
            let mut pred = 0.0;
            for j in 1..=den_order {
                pred -= denominator[j] * output[t - j];
            }
            for j in 0..=num_order {
                if t >= j {
                    pred += numerator[j] * input[t - j];
                }
            }
            y_pred[i] = pred;
            y_actual[i] = output[t];
        }
        let fit = calculate_fit_percentage(&y_actual, &y_pred);
        let residuals = &y_actual - &y_pred;
        let sq = residuals.mapv(|x| x * x);
        let variance = if !sq.is_empty() {
            sq.sum() / sq.len() as f64
        } else {
            0.0
        };
        (fit, variance)
    } else {
        (0.0, 0.0)
    };

    Ok(TfEstimationResult {
        numerator,
        denominator,
        fit_percentage,
        error_variance,
        frequency_response: None,
        frequencies: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate a discrete-time SISO state-space system
    /// `x[k+1] = A x[k] + B u[k]`, `y[k] = C x[k] + D u[k]` given a driving
    /// input sequence, returning the resulting output sequence.
    fn simulate_second_order_system(
        a: [[f64; 2]; 2],
        b: [f64; 2],
        c: [f64; 2],
        d: f64,
        input: &[f64],
    ) -> Vec<f64> {
        let mut x = [0.0_f64, 0.0_f64];
        let mut output = Vec::with_capacity(input.len());
        for &u in input {
            let y = c[0] * x[0] + c[1] * x[1] + d * u;
            output.push(y);
            let x_next = [
                a[0][0] * x[0] + a[0][1] * x[1] + b[0] * u,
                a[1][0] * x[0] + a[1][1] * x[1] + b[1] * u,
            ];
            x = x_next;
        }
        output
    }

    /// Deterministic pseudo-random sequence in `[-0.5, 0.5]` (xorshift64),
    /// used as a persistently-exciting input for identification.
    fn pseudo_random_sequence(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f64 / u64::MAX as f64) - 0.5
            })
            .collect()
    }

    #[test]
    fn test_estimate_tf_subspace_recovers_known_system() {
        // A known, stable 2nd-order system with characteristic polynomial
        // det(zI-A) = z^2 - 0.8z + 0.16, i.e. denominator [1, -0.8, 0.16]
        // in this module's z^-1 convention.
        let a = [[0.0, 1.0], [-0.16, 0.8]];
        let b = [0.0, 1.0];
        let c = [1.0, 0.0];
        let d = 0.0;

        let n = 400;
        let input = pseudo_random_sequence(n, 12345);
        let output = simulate_second_order_system(a, b, c, d, &input);

        let input = Array1::from_vec(input);
        let output = Array1::from_vec(output);

        let result =
            estimate_transfer_function(&input, &output, 1.0, 2, 2, TfEstimationMethod::Subspace)
                .expect("subspace estimation should succeed");

        // The fabricated implementation always silently aliased to
        // ordinary least squares; a genuine subspace identification should
        // recover the true denominator (up to numerical precision) from
        // this noiseless simulation.
        assert!(
            result.fit_percentage > 95.0,
            "fit_percentage={}",
            result.fit_percentage
        );
        assert!(
            (result.denominator[1] - (-0.8)).abs() < 0.05,
            "denominator={:?}",
            result.denominator
        );
        assert!(
            (result.denominator[2] - 0.16).abs() < 0.05,
            "denominator={:?}",
            result.denominator
        );
    }

    #[test]
    fn test_estimate_tf_subspace_differs_from_least_squares() {
        // On noiseless data both a genuine subspace estimator and ARX
        // least squares are consistent and converge to (numerically) the
        // same truth, so this adds measurement noise on the output (which
        // biases ordinary ARX least squares differently than the subspace
        // approach) to verify the two methods are not simply computing the
        // same thing end-to-end -- the old stub always returned the plain
        // least-squares estimate verbatim for the Subspace method.
        let a = [[0.0, 1.0], [-0.25, 0.6]];
        let b = [0.0, 1.0];
        let c = [1.0, 0.3];
        let d = 0.0;

        let n = 200;
        let input = pseudo_random_sequence(n, 999);
        let clean_output = simulate_second_order_system(a, b, c, d, &input);
        let measurement_noise = pseudo_random_sequence(n, 42);
        let output: Vec<f64> = clean_output
            .iter()
            .zip(measurement_noise.iter())
            .map(|(&y, &noise)| y + 0.3 * noise)
            .collect();

        let input = Array1::from_vec(input);
        let output = Array1::from_vec(output);

        let subspace =
            estimate_transfer_function(&input, &output, 1.0, 2, 2, TfEstimationMethod::Subspace)
                .expect("subspace estimation should succeed");
        let least_squares = estimate_transfer_function(
            &input,
            &output,
            1.0,
            2,
            2,
            TfEstimationMethod::LeastSquares,
        )
        .expect("least squares estimation should succeed");

        let denom_diff: f64 = subspace
            .denominator
            .iter()
            .zip(least_squares.denominator.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(
            denom_diff > 1e-6,
            "subspace and least-squares denominators are identical: {:?} vs {:?}",
            subspace.denominator,
            least_squares.denominator
        );
    }
}
