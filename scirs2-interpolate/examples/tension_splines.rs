use ndarray::Array1;
use scirs2_interpolate::{cubic_interpolate, make_tension_spline, ExtrapolateMode};

fn main() {
    println!("Tension Splines Example");
    println!("======================\n");

    // Create some sample data
    let x = Array1::linspace(0.0, 10.0, 11);

    // Create oscillating data to demonstrate tension effects
    let y = x.mapv(|v| f64::sin(v) + 0.1 * v);

    println!("Example 1: Comparing Different Tension Values");
    println!("-------------------------------------------");

    // Create splines with different tension parameters
    let spline_low =
        make_tension_spline(&x.view(), &y.view(), 0.01, ExtrapolateMode::Error).unwrap();
    let spline_med =
        make_tension_spline(&x.view(), &y.view(), 1.0, ExtrapolateMode::Error).unwrap();
    let spline_high =
        make_tension_spline(&x.view(), &y.view(), 10.0, ExtrapolateMode::Error).unwrap();

    // Create a fine grid for evaluation
    let x_fine = Array1::linspace(0.0, 10.0, 101);

    // Evaluate all splines on the fine grid
    let y_low = spline_low.evaluate(&x_fine.view()).unwrap();
    let y_med = spline_med.evaluate(&x_fine.view()).unwrap();
    let y_high = spline_high.evaluate(&x_fine.view()).unwrap();

    // Print values at some sample points to observe the effect of tension
    println!("Comparing interpolated values at some sample points:");
    println!("   x   |  Original  |  Tension=0.01  |  Tension=1.0  |  Tension=10.0");
    println!("---------------------------------------------------------------");

    for i in [10, 25, 40, 60, 75, 90] {
        println!(
            " {:.2} |    {:.4}    |     {:.4}     |    {:.4}    |     {:.4}",
            x_fine[i],
            f64::sin(x_fine[i]) + 0.1 * x_fine[i],
            y_low[i],
            y_med[i],
            y_high[i]
        );
    }

    // Calculate total curvature as a measure of smoothness
    let calc_curvature = |y: &Array1<f64>| -> f64 {
        let mut curvature = 0.0;
        for i in 1..y.len() - 1 {
            let dy = y[i + 1] - 2.0 * y[i] + y[i - 1];
            curvature += dy.powi(2);
        }
        curvature
    };

    let curv_low = calc_curvature(&y_low);
    let curv_med = calc_curvature(&y_med);
    let curv_high = calc_curvature(&y_high);

    println!("\nTotal curvature (lower means smoother):");
    println!("  Tension=0.01: {:.6}", curv_low);
    println!("  Tension=1.0:  {:.6}", curv_med);
    println!("  Tension=10.0: {:.6}", curv_high);

    println!("\nExample 2: Working with Derivatives");
    println!("---------------------------------");

    // Create a quadratic function
    let x_quad = Array1::linspace(0.0, 5.0, 6);
    let y_quad = x_quad.mapv(|x| f64::powi(x, 2));

    // Create a tension spline for this quadratic data
    let quad_spline =
        make_tension_spline(&x_quad.view(), &y_quad.view(), 0.5, ExtrapolateMode::Error).unwrap();

    // Sample points for derivative evaluation
    let x_sample = Array1::linspace(0.5, 4.5, 9);

    // Evaluate first derivative (should be approximately 2*x)
    let deriv1 = quad_spline.derivative(1, &x_sample.view()).unwrap();

    // Evaluate second derivative (should be approximately 2)
    let deriv2 = quad_spline.derivative(2, &x_sample.view()).unwrap();

    println!("For y = x²:");
    println!("   x   | First Derivative | Second Derivative | Exact First | Exact Second");
    println!("------------------------------------------------------------------------");

    for i in 0..x_sample.len() {
        let exact_first = 2.0 * x_sample[i];
        let exact_second = 2.0;

        println!(
            " {:.1} |      {:.4}      |       {:.4}      |    {:.4}    |     {:.4}",
            x_sample[i], deriv1[i], deriv2[i], exact_first, exact_second
        );
    }

    println!("\nExample 3: Handling Discontinuities");
    println!("----------------------------------");

    // Create data with a sharp transition
    let x_sharp = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let y_sharp = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);

    // Create splines with different tension values
    let sharp_low = make_tension_spline(
        &x_sharp.view(),
        &y_sharp.view(),
        0.1,
        ExtrapolateMode::Error,
    )
    .unwrap();
    let sharp_med = make_tension_spline(
        &x_sharp.view(),
        &y_sharp.view(),
        2.0,
        ExtrapolateMode::Error,
    )
    .unwrap();
    let sharp_high = make_tension_spline(
        &x_sharp.view(),
        &y_sharp.view(),
        20.0,
        ExtrapolateMode::Error,
    )
    .unwrap();

    // Evaluate at points around the transition
    let x_trans = Array1::linspace(4.0, 6.0, 21);
    let y_trans_low = sharp_low.evaluate(&x_trans.view()).unwrap();
    let y_trans_med = sharp_med.evaluate(&x_trans.view()).unwrap();
    let y_trans_high = sharp_high.evaluate(&x_trans.view()).unwrap();

    println!("Values around the transition point (x=5.0):");
    println!("   x   | Tension=0.1 | Tension=2.0 | Tension=20.0");
    println!("-------------------------------------------------");

    for i in [0, 5, 10, 15, 20] {
        println!(
            " {:.2} |    {:.4}   |    {:.4}   |     {:.4}",
            x_trans[i], y_trans_low[i], y_trans_med[i], y_trans_high[i]
        );
    }

    println!("\nObservation: Higher tension values produce sharper transitions");

    println!("\nExample 4: Extrapolation Behavior");
    println!("-------------------------------");

    // Create simple linear data
    let x_simple = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_simple = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    // Create splines with different extrapolation modes
    let extrap_error = make_tension_spline(
        &x_simple.view(),
        &y_simple.view(),
        1.0,
        ExtrapolateMode::Error,
    );
    let extrap_allow = make_tension_spline(
        &x_simple.view(),
        &y_simple.view(),
        1.0,
        ExtrapolateMode::Extrapolate,
    )
    .unwrap();
    let extrap_nearest = make_tension_spline(
        &x_simple.view(),
        &y_simple.view(),
        1.0,
        ExtrapolateMode::Constant,
    )
    .unwrap();

    // Test points outside the domain
    let x_outside = Array1::from_vec(vec![-2.0, -1.0, 6.0, 7.0]);

    println!("Extrapolation results with different modes:");
    println!("   x   | Extrapolate | Constant | Error mode");
    println!("---------------------------------------------------");

    for &x_val in x_outside.iter() {
        let result_allow = extrap_allow
            .evaluate(&Array1::from_vec(vec![x_val]).view())
            .unwrap()[0];
        let result_nearest = extrap_nearest
            .evaluate(&Array1::from_vec(vec![x_val]).view())
            .unwrap()[0];
        let result_error = match extrap_error
            .as_ref()
            .unwrap()
            .evaluate(&Array1::from_vec(vec![x_val]).view())
        {
            Ok(v) => format!("{:.4}", v[0]),
            Err(_) => "Error".to_string(),
        };

        println!(
            " {:.1} |    {:.4}   |      {:.4}     | {}",
            x_val, result_allow, result_nearest, result_error
        );
    }

    println!("\nExample 5: Comparison with Cubic Splines");
    println!("---------------------------------------");

    // Create data with oscillation to show the difference
    let x_osc = Array1::linspace(0.0, 4.0 * std::f64::consts::PI, 9);
    let y_osc = x_osc.mapv(f64::sin);

    // Create tension splines with different tension values
    let tens_0 =
        make_tension_spline(&x_osc.view(), &y_osc.view(), 0.0, ExtrapolateMode::Error).unwrap();
    let tens_1 =
        make_tension_spline(&x_osc.view(), &y_osc.view(), 1.0, ExtrapolateMode::Error).unwrap();
    let tens_5 =
        make_tension_spline(&x_osc.view(), &y_osc.view(), 5.0, ExtrapolateMode::Error).unwrap();

    // Evaluate at fine points
    let x_fine_osc = Array1::linspace(0.0, 4.0 * std::f64::consts::PI, 101);
    let y_exact = x_fine_osc.mapv(f64::sin);

    // Also create a standard cubic spline for comparison - handle potential errors
    let y_cubic = match cubic_interpolate(&x_osc.view(), &y_osc.view(), &x_fine_osc.view()) {
        Ok(result) => result,
        Err(_) => {
            // If interpolation fails, create a dummy array matching the size of y_exact
            println!("Note: Cubic interpolation failed, using zeros for comparison");
            Array1::zeros(y_exact.len())
        }
    };

    // Handle potential errors with evaluate
    let y_tens_0 = match tens_0.evaluate(&x_fine_osc.view()) {
        Ok(result) => result,
        Err(_) => {
            println!("Note: Tension=0.0 evaluation failed, using zeros for comparison");
            Array1::zeros(y_exact.len())
        }
    };

    let y_tens_1 = match tens_1.evaluate(&x_fine_osc.view()) {
        Ok(result) => result,
        Err(_) => {
            println!("Note: Tension=1.0 evaluation failed, using zeros for comparison");
            Array1::zeros(y_exact.len())
        }
    };

    let y_tens_5 = match tens_5.evaluate(&x_fine_osc.view()) {
        Ok(result) => result,
        Err(_) => {
            println!("Note: Tension=5.0 evaluation failed, using zeros for comparison");
            Array1::zeros(y_exact.len())
        }
    };

    // Calculate root mean square error for each spline
    let calc_rmse = |y_pred: &Array1<f64>, y_true: &Array1<f64>| -> f64 {
        let squared_errors = y_pred
            .iter()
            .zip(y_true.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>();
        (squared_errors / y_pred.len() as f64).sqrt()
    };

    let rmse_tens_0 = calc_rmse(&y_tens_0, &y_exact);
    let rmse_tens_1 = calc_rmse(&y_tens_1, &y_exact);
    let rmse_tens_5 = calc_rmse(&y_tens_5, &y_exact);
    let rmse_cubic = calc_rmse(&y_cubic, &y_exact);

    println!("Root Mean Square Error compared to true sine function:");
    println!(
        "  Tension=0.0: {:.6} (should be similar to cubic)",
        rmse_tens_0
    );
    println!("  Tension=1.0: {:.6}", rmse_tens_1);
    println!("  Tension=5.0: {:.6}", rmse_tens_5);
    println!("  Cubic spline: {:.6}", rmse_cubic);

    println!("\nNote that tension=0 should be equivalent to a cubic spline.");
    println!("The small differences are due to implementation details.");

    println!("\nAll examples completed successfully!");
}
