use ndarray::Array1;
use scirs2_interpolate::{
    make_hermite_spline, make_hermite_spline_with_derivatives, make_natural_hermite_spline,
    make_periodic_hermite_spline, make_quintic_hermite_spline, DerivativeSpec, ExtrapolateMode,
    HermiteSpline,
};

fn main() {
    println!("Hermite Splines Example");
    println!("======================\n");

    // Example 1: Basic Hermite spline with automatic derivatives
    println!("Example 1: Basic Hermite Spline");
    println!("------------------------------");

    // Create some sample data - a parabola
    let x = Array1::linspace(0.0, 10.0, 6);
    let y = x.mapv(|v| f64::powi(v, 2));

    // Create a Hermite spline with automatic derivative estimation
    let spline = make_hermite_spline(&x.view(), &y.view(), ExtrapolateMode::Error).unwrap();

    // Evaluate at more points to see the interpolation
    let x_fine = Array1::linspace(0.0, 10.0, 21);
    let y_interp = spline.evaluate(&x_fine.view()).unwrap();
    let y_exact = x_fine.mapv(|v| f64::powi(v, 2));

    println!("Interpolation of y = x² with 6 data points:");
    println!("  x   |  Estimated  |  Exact  |  Error");
    println!("----------------------------------------");

    for i in 0..x_fine.len() {
        if i % 4 == 0 {
            // Show just a subset of points
            let error = f64::abs(y_interp[i] - y_exact[i]);
            println!(
                "{:5.2} | {:11.6} | {:7.6} | {:8.6}",
                x_fine[i], y_interp[i], y_exact[i], error
            );
        }
    }

    // Calculate RMSE
    let mse = y_interp
        .iter()
        .zip(y_exact.iter())
        .map(|(a, b)| f64::powi(a - b, 2))
        .sum::<f64>()
        / y_interp.len() as f64;
    let rmse = mse.sqrt();

    println!("\nRoot Mean Square Error: {:.6}", rmse);

    // Show the estimated derivatives
    let derivatives = spline.get_derivatives();
    println!("\nEstimated derivatives at data points:");
    println!("  x   | Estimated | Exact (2x)");
    println!("-----------------------------");

    for i in 0..x.len() {
        let exact_deriv = 2.0 * x[i];
        println!(
            "{:5.2} | {:9.4} | {:9.4}",
            x[i], derivatives[i], exact_deriv
        );
    }

    // Example 2: Hermite spline with exact derivatives
    println!("\nExample 2: Hermite Spline with Exact Derivatives");
    println!("----------------------------------------------");

    // For y = x², the derivative is 2*x
    let deriv_exact = x.mapv(|v| 2.0 * v);

    // Create a Hermite spline with exact derivatives
    let spline_exact = make_hermite_spline_with_derivatives(
        &x.view(),
        &y.view(),
        &deriv_exact.view(),
        ExtrapolateMode::Error,
    )
    .unwrap();

    // Evaluate at more points
    let y_interp_exact = spline_exact.evaluate(&x_fine.view()).unwrap();

    // Calculate RMSE with exact derivatives
    let mse_exact = y_interp_exact
        .iter()
        .zip(y_exact.iter())
        .map(|(a, b)| f64::powi(a - b, 2))
        .sum::<f64>()
        / y_interp_exact.len() as f64;
    let rmse_exact = mse_exact.sqrt();

    println!("Comparing interpolation accuracy:");
    println!("  With estimated derivatives: RMSE = {:.6}", rmse);
    println!("  With exact derivatives:     RMSE = {:.6}", rmse_exact);
    println!("\nWith exact derivatives, the RMSE should be very small or zero,");
    println!("since cubic Hermite can exactly represent quadratic functions.");

    // Example 3: Derivatives and continuity
    println!("\nExample 3: Derivative Continuity");
    println!("------------------------------");

    // Create sample data with a sharp feature
    let x_sharp = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let mut y_sharp = Array1::zeros(x_sharp.len());

    // Create a function with a sharp peak at x=5
    for i in 0..y_sharp.len() {
        let x = x_sharp[i];
        if x < 5.0 {
            y_sharp[i] = x / 5.0;
        } else {
            y_sharp[i] = (10.0 - x) / 5.0;
        }
    }

    // Create Hermite splines with different derivative specifications
    let spline_auto =
        make_hermite_spline(&x_sharp.view(), &y_sharp.view(), ExtrapolateMode::Error).unwrap();

    // Create a spline with a fixed derivative at the peak point (index 5, x=5.0)
    // Setting it to 0 to create a smoother peak
    let spline_fixed = HermiteSpline::new(
        &x_sharp.view(),
        &y_sharp.view(),
        None,
        DerivativeSpec::Fixed(5, 0.0),
        ExtrapolateMode::Error,
    )
    .unwrap();

    // Compare the derivatives at the peak
    let derivatives_auto = spline_auto.get_derivatives();
    let derivatives_fixed = spline_fixed.get_derivatives();

    println!("Derivatives at the sharp peak (x=5.0):");
    println!("  With automatic estimation: {:.6}", derivatives_auto[5]);
    println!("  With fixed value (0.0):    {:.6}", derivatives_fixed[5]);

    // Evaluate at fine points around the peak
    let x_peak = Array1::linspace(4.0, 6.0, 21);
    let y_auto = spline_auto.evaluate(&x_peak.view()).unwrap();
    let y_fixed = spline_fixed.evaluate(&x_peak.view()).unwrap();

    println!("\nValues around the peak:");
    println!("   x   | Auto-deriv | Fixed-deriv");
    println!("---------------------------------");

    for i in 0..x_peak.len() {
        if i % 4 == 0 {
            println!(
                " {:5.2} | {:9.6} | {:9.6}",
                x_peak[i], y_auto[i], y_fixed[i]
            );
        }
    }

    println!("\nNote how the fixed derivative creates a smoother, more rounded peak");
    println!("compared to the automatic derivative estimation.");

    // Example 4: Natural and Periodic Boundary Conditions
    println!("\nExample 4: Natural and Periodic Boundary Conditions");
    println!("-----------------------------------------------");

    // Create a sine wave from 0 to 2π
    let x_sine = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, 7);
    let y_sine = x_sine.mapv(f64::sin);

    // Create splines with different boundary conditions
    let spline_auto_sine =
        make_hermite_spline(&x_sine.view(), &y_sine.view(), ExtrapolateMode::Extrapolate).unwrap();

    let spline_natural =
        make_natural_hermite_spline(&x_sine.view(), &y_sine.view(), ExtrapolateMode::Extrapolate)
            .unwrap();

    let spline_periodic =
        make_periodic_hermite_spline(&x_sine.view(), &y_sine.view(), ExtrapolateMode::Extrapolate)
            .unwrap();

    // Get the derivatives at the endpoints
    let derivs_auto = spline_auto_sine.get_derivatives();
    let derivs_natural = spline_natural.get_derivatives();
    let derivs_periodic = spline_periodic.get_derivatives();

    println!("Endpoint derivatives for sine wave from 0 to 2π:");
    println!("  Boundary Condition |   Start   |    End");
    println!("-------------------------------------------");
    println!(
        "  Automatic          | {:9.6} | {:9.6}",
        derivs_auto[0],
        derivs_auto[derivs_auto.len() - 1]
    );
    println!(
        "  Natural (zero)     | {:9.6} | {:9.6}",
        derivs_natural[0],
        derivs_natural[derivs_natural.len() - 1]
    );
    println!(
        "  Periodic           | {:9.6} | {:9.6}",
        derivs_periodic[0],
        derivs_periodic[derivs_periodic.len() - 1]
    );

    println!("\nExact values for sine function:");
    println!("  Start: cos(0) = 1.0");
    println!("  End:   cos(2π) = 1.0");

    println!("\nNote that:");
    println!("  - Natural splines force zero derivatives at endpoints");
    println!("  - Periodic splines make the end derivative match the start");
    println!("  - Auto-estimation recovers approximate correct derivatives");

    // Test extrapolation behavior
    let x_extend = Array1::linspace(-std::f64::consts::PI, 3.0 * std::f64::consts::PI, 5);
    let y_auto = spline_auto_sine.evaluate(&x_extend.view()).unwrap();
    let y_natural = spline_natural.evaluate(&x_extend.view()).unwrap();
    let y_periodic = spline_periodic.evaluate(&x_extend.view()).unwrap();
    let y_exact = x_extend.mapv(f64::sin);

    println!("\nExtrapolation behavior:");
    println!("     x     |   Auto   |  Natural |  Periodic |   Exact");
    println!("----------------------------------------------------------");

    for i in 0..x_extend.len() {
        println!(
            " {:9.6} | {:8.4} | {:8.4} | {:8.4} | {:8.4}",
            x_extend[i], y_auto[i], y_natural[i], y_periodic[i], y_exact[i]
        );
    }

    println!("\nPeriodic splines should continue the pattern for better extrapolation");

    // Example 5: Quintic Hermite Splines
    println!("\nExample 5: Quintic Hermite Splines");
    println!("-------------------------------");

    // Create data for a cubic function y = x³
    let x_cubic = Array1::linspace(0.0, 5.0, 6);
    let y_cubic = x_cubic.mapv(|v| f64::powi(v, 3));

    // For y = x³, first derivative is 3x², second derivative is 6x
    let first_derivs = x_cubic.mapv(|v| 3.0 * f64::powi(v, 2));
    let second_derivs = x_cubic.mapv(|v| 6.0 * v);

    // Create cubic and quintic Hermite splines
    let cubic_spline = make_hermite_spline_with_derivatives(
        &x_cubic.view(),
        &y_cubic.view(),
        &first_derivs.view(),
        ExtrapolateMode::Error,
    )
    .unwrap();

    let quintic_spline = make_quintic_hermite_spline(
        &x_cubic.view(),
        &y_cubic.view(),
        &first_derivs.view(),
        &second_derivs.view(),
        ExtrapolateMode::Error,
    )
    .unwrap();

    // Evaluate at fine points
    let x_fine_cubic = Array1::linspace(0.0, 5.0, 51);
    let y_cubic_exact = x_fine_cubic.mapv(|v| f64::powi(v, 3));

    let y_cubic_hermite = cubic_spline.evaluate(&x_fine_cubic.view()).unwrap();
    let y_quintic_hermite = quintic_spline.evaluate(&x_fine_cubic.view()).unwrap();

    // Calculate errors
    let mse_cubic = y_cubic_hermite
        .iter()
        .zip(y_cubic_exact.iter())
        .map(|(a, b)| f64::powi(a - b, 2))
        .sum::<f64>()
        / y_cubic_hermite.len() as f64;

    let mse_quintic = y_quintic_hermite
        .iter()
        .zip(y_cubic_exact.iter())
        .map(|(a, b)| f64::powi(a - b, 2))
        .sum::<f64>()
        / y_quintic_hermite.len() as f64;

    println!("Interpolating y = x³ with different Hermite splines:");
    println!("  Cubic Hermite RMSE:  {:.8}", mse_cubic.sqrt());
    println!("  Quintic Hermite RMSE: {:.8}", mse_quintic.sqrt());

    // Find maximum errors
    let max_err_cubic = y_cubic_hermite
        .iter()
        .zip(y_cubic_exact.iter())
        .map(|(a, b)| f64::abs(a - b))
        .fold(0.0, |max, err| if err > max { err } else { max });

    let max_err_quintic = y_quintic_hermite
        .iter()
        .zip(y_cubic_exact.iter())
        .map(|(a, b)| f64::abs(a - b))
        .fold(0.0, |max, err| if err > max { err } else { max });

    println!("\nMaximum absolute errors:");
    println!("  Cubic Hermite:  {:.8}", max_err_cubic);
    println!("  Quintic Hermite: {:.8}", max_err_quintic);

    println!("\nNote: Quintic Hermite splines can exactly represent cubic functions");
    println!("when provided with exact first and second derivatives!");

    println!("\nAll examples completed successfully!");
}
