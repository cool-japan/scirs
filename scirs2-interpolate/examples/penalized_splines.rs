use ndarray::{array, Array1, Array2, ArrayView1};
use scirs2_interpolate::bspline::ExtrapolateMode;
use scirs2_interpolate::penalized::{
    cross_validate_lambda, pspline_with_custom_penalty, PSpline, PenaltyType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Penalized Splines (P-splines) Examples");
    println!("======================================\n");

    // Example 1: Basic P-spline with synthetic data
    println!("Example 1: Basic P-spline with different penalties");

    // Create synthetic data: y = sin(2πx) + noise
    let x = Array1::linspace(0.0, 1.0, 20);
    let mut y = Array1::zeros(x.len());

    for (i, &xi) in x.iter().enumerate() {
        let true_value = (2.0 * std::f64::consts::PI * xi).sin();
        // Add noise
        y[i] = true_value + 0.1 * (i as f64 * 7.9).sin();
    }

    // Create the true curve for comparison
    let x_fine = Array1::linspace(0.0, 1.0, 100);
    let mut y_true = Array1::zeros(x_fine.len());
    for (i, &xi) in x_fine.iter().enumerate() {
        y_true[i] = (2.0 * std::f64::consts::PI * xi).sin();
    }

    println!("  Data points: {}", x.len());
    println!("  True function: y = sin(2πx)");

    // Fit P-splines with different penalty types
    println!("\n  Fitting with different penalties (λ = 0.1):");

    for penalty_type in &[
        PenaltyType::Ridge,
        PenaltyType::FirstDerivative,
        PenaltyType::SecondDerivative,
        PenaltyType::ThirdDerivative,
    ] {
        let pspline = PSpline::new(
            &x.view(),
            &y.view(),
            15,  // Number of knots
            3,   // Cubic spline
            0.1, // Lambda
            *penalty_type,
            ExtrapolateMode::Extrapolate,
        )?;

        // Evaluate the fitted spline
        let y_fit = pspline.evaluate_array(&x_fine.view())?;

        // Calculate mean squared error
        let mut mse = 0.0;
        for i in 0..y_true.len() {
            mse += (y_fit[i] - y_true[i]).powi(2);
        }
        mse /= y_true.len() as f64;

        match penalty_type {
            PenaltyType::Ridge => println!("  Ridge penalty MSE: {:.6}", mse),
            PenaltyType::FirstDerivative => println!("  First derivative penalty MSE: {:.6}", mse),
            PenaltyType::SecondDerivative => {
                println!("  Second derivative penalty MSE: {:.6}", mse)
            }
            PenaltyType::ThirdDerivative => println!("  Third derivative penalty MSE: {:.6}", mse),
        }
    }

    // Example 2: Effect of lambda on smoothing
    println!("\nExample 2: Effect of lambda on smoothing");

    // Use the same data as in Example 1
    // Fit P-splines with different lambda values
    let lambda_values = [0.0001, 0.01, 0.1, 1.0, 10.0];

    println!("\n  Fitting with different lambda values (second derivative penalty):");

    for &lambda in &lambda_values {
        let pspline = PSpline::new(
            &x.view(),
            &y.view(),
            15, // Number of knots
            3,  // Cubic spline
            lambda,
            PenaltyType::SecondDerivative,
            ExtrapolateMode::Extrapolate,
        )?;

        // Evaluate the fitted spline
        let y_fit = pspline.evaluate_array(&x_fine.view())?;

        // Calculate mean squared error
        let mut mse = 0.0;
        for i in 0..y_true.len() {
            mse += (y_fit[i] - y_true[i]).powi(2);
        }
        mse /= y_true.len() as f64;

        // Calculate roughness penalty (integral of squared second derivative)
        let mut roughness = 0.0;
        for i in 0..x_fine.len() - 1 {
            let x_mid = (x_fine[i] + x_fine[i + 1]) / 2.0;
            let d2 = pspline.derivative(x_mid, 2)?;
            roughness += d2 * d2 * (x_fine[i + 1] - x_fine[i]);
        }

        println!(
            "  Lambda = {:.4}: MSE = {:.6}, Roughness = {:.6}",
            lambda, mse, roughness
        );
    }

    // Example 3: Cross-validation to find optimal lambda
    println!("\nExample 3: Cross-validation to find optimal lambda");

    // Create array of lambda values to test
    let cv_lambda_values = array![0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0];

    let (best_lambda, cv_error) = cross_validate_lambda(
        &x.view(),
        &y.view(),
        15,
        3,
        &cv_lambda_values.view(),
        PenaltyType::SecondDerivative,
        ExtrapolateMode::Extrapolate,
    )?;

    println!("  Lambda values tested: {:?}", cv_lambda_values);
    println!("  Best lambda from cross-validation: {:.4}", best_lambda);
    println!("  Cross-validation error: {:.6}", cv_error);

    // Fit with the optimal lambda
    let best_pspline = PSpline::new(
        &x.view(),
        &y.view(),
        15, // Number of knots
        3,  // Cubic spline
        best_lambda,
        PenaltyType::SecondDerivative,
        ExtrapolateMode::Extrapolate,
    )?;

    // Evaluate the fitted spline
    let y_best = best_pspline.evaluate_array(&x_fine.view())?;

    // Calculate mean squared error
    let mut best_mse = 0.0;
    for i in 0..y_true.len() {
        best_mse += (y_best[i] - y_true[i]).powi(2);
    }
    best_mse /= y_true.len() as f64;

    println!("  MSE with best lambda: {:.6}", best_mse);

    // Example 4: Custom penalty matrix
    println!("\nExample 4: Custom penalty matrix");

    // Generate knots
    let knots = match generate_uniform_knots(&x.view(), 15, 3) {
        Ok(k) => k,
        Err(_) => {
            // Fallback if our helper function fails
            Array1::linspace(0.0, 1.0, 15 + 4)
        }
    };

    // Create a custom penalty matrix that penalizes large coefficients at the boundaries
    let n_basis = knots.len() - 3 - 1;
    let mut penalty = Array2::zeros((n_basis, n_basis));

    // Add stronger penalties at the boundaries (first and last 3 coefficients)
    for i in 0..n_basis {
        if i < 3 || i >= n_basis - 3 {
            penalty[[i, i]] = 10.0; // Higher penalty for boundaries
        } else {
            penalty[[i, i]] = 1.0; // Lower penalty for interior
        }
    }

    // Also add second derivative penalty
    for i in 0..n_basis - 2 {
        // Diagonal elements
        penalty[[i, i]] += 1.0;
        penalty[[i + 1, i + 1]] += 4.0;
        penalty[[i + 2, i + 2]] += 1.0;

        // Off-diagonal elements
        penalty[[i, i + 1]] += -2.0;
        penalty[[i + 1, i]] += -2.0;

        penalty[[i, i + 2]] += 1.0;
        penalty[[i + 2, i]] += 1.0;

        penalty[[i + 1, i + 2]] += -2.0;
        penalty[[i + 2, i + 1]] += -2.0;
    }

    // Fit with the custom penalty
    let custom_pspline = pspline_with_custom_penalty(
        &x.view(),
        &y.view(),
        &knots.view(),
        3,
        0.1,
        &penalty.view(),
        ExtrapolateMode::Extrapolate,
    )?;

    // Evaluate the fitted spline
    let y_custom = custom_pspline.evaluate_array(&x_fine.view())?;

    // Calculate mean squared error
    let mut custom_mse = 0.0;
    for i in 0..y_true.len() {
        custom_mse += (y_custom[i] - y_true[i]).powi(2);
    }
    custom_mse /= y_true.len() as f64;

    println!("  Custom penalty MSE: {:.6}", custom_mse);

    // Example 5: Working with derivatives
    println!("\nExample 5: Working with derivatives");

    // Create data that follows a simple parabola: y = x^2
    let x_quad = Array1::linspace(0.0, 1.0, 11);
    let mut y_quad = Array1::zeros(x_quad.len());

    for (i, &xi) in x_quad.iter().enumerate() {
        y_quad[i] = xi * xi;
    }

    // Fit a P-spline
    let pspline_quad = PSpline::new(
        &x_quad.view(),
        &y_quad.view(),
        10,
        3,
        0.001, // Small lambda to fit the data closely
        PenaltyType::SecondDerivative,
        ExtrapolateMode::Extrapolate,
    )?;

    // Evaluate the spline and its derivatives at a few points
    let eval_points = array![0.25, 0.5, 0.75];

    println!("  Derivatives of fitted P-spline for y = x^2:");
    println!("  x | y | dy/dx | d²y/dx² | d³y/dx³");
    println!("  --------------------------------");

    for &x_val in eval_points.iter() {
        let y_val = pspline_quad.evaluate(x_val)?;
        let d1 = pspline_quad.derivative(x_val, 1)?;
        let d2 = pspline_quad.derivative(x_val, 2)?;
        let d3 = pspline_quad.derivative(x_val, 3)?;

        println!(
            "  {:.2} | {:.4} | {:.4} | {:.4} | {:.4}",
            x_val, y_val, d1, d2, d3
        );

        // Expected values for y = x²:
        // y = x²
        // dy/dx = 2x
        // d²y/dx² = 2
        // d³y/dx³ = 0
        println!(
            "  Exact: {:.4} | {:.4} | {:.4} | {:.4}",
            x_val * x_val,
            2.0 * x_val,
            2.0,
            0.0
        );
        println!("  --------------------------------");
    }

    Ok(())
}

/// Helper function to generate uniformly spaced knots
fn generate_uniform_knots(
    x: &ArrayView1<f64>,
    n_knots: usize,
    degree: usize,
) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    if x.is_empty() {
        return Err("Empty x array".into());
    }

    // Calculate min and max x values
    let x_min = *x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let x_max = *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    // Create knot vector
    let mut knots = Array1::zeros(n_knots + degree + 1);

    // First degree+1 knots are at x_min
    for i in 0..=degree {
        knots[i] = x_min;
    }

    // Middle knots are uniformly spaced
    for i in 1..n_knots - degree {
        let t = i as f64 / (n_knots - degree) as f64;
        knots[i + degree] = x_min + t * (x_max - x_min);
    }

    // Last degree+1 knots are at x_max
    for i in 0..=degree {
        knots[n_knots + i] = x_max;
    }

    Ok(knots)
}
