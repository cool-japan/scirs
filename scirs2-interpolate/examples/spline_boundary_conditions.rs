use ndarray::{array, Array1, ArrayView1};
use scirs2_interpolate::spline::make_interp_spline;

fn main() {
    println!("Cubic Spline Boundary Conditions Example");
    println!("=======================================\n");

    // Create sample data
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = array![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2

    // Compare different boundary conditions
    compare_boundary_conditions(&x.view(), &y.view());

    // Show derivatives and integration
    demonstrate_derivatives_and_integration(&x.view(), &y.view());

    // Demonstrate periodic boundary conditions
    demonstrate_periodic_splines();
}

fn compare_boundary_conditions(x: &ArrayView1<f64>, y: &ArrayView1<f64>) {
    println!("1. Comparing Different Boundary Conditions");
    println!("----------------------------------------");

    // Create cubic splines with different boundary conditions
    let natural_spline = make_interp_spline(x, y, "natural", None).unwrap();
    let not_a_knot_spline = make_interp_spline(x, y, "not-a-knot", None).unwrap();

    // For clamped boundary conditions, we need to provide first derivatives at endpoints
    // For a parabola y = x^2, the first derivative is y' = 2x
    let first_deriv_start = 0.0; // y'(0) = 0
    let first_deriv_end = 8.0; // y'(4) = 8
    let bc_params = array![first_deriv_start, first_deriv_end];
    let clamped_spline = make_interp_spline(x, y, "clamped", Some(&bc_params.view())).unwrap();

    println!("\nSample data (y = x²):");
    for i in 0..x.len() {
        println!("  x = {}, y = {}", x[i], y[i]);
    }

    // Generate test points
    let test_points = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

    println!("\nComparison of interpolation results:");
    println!("   x   | Natural | Not-a-knot | Clamped |   x²   ");
    println!("-------|---------|------------|---------|--------");

    for &test_point in test_points.iter() {
        let natural_value = natural_spline.evaluate(test_point).unwrap();
        let not_a_knot_value = not_a_knot_spline.evaluate(test_point).unwrap();
        let clamped_value = clamped_spline.evaluate(test_point).unwrap();
        let true_value = test_point * test_point;

        println!(
            " {:.1} |  {:.4} |   {:.4}   |  {:.4} | {:.4}",
            test_point, natural_value, not_a_knot_value, clamped_value, true_value
        );
    }

    // Check which method best approximates the true function
    let test_points = array![0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75];

    let natural_values = natural_spline.evaluate_array(&test_points.view()).unwrap();
    let not_a_knot_values = not_a_knot_spline
        .evaluate_array(&test_points.view())
        .unwrap();
    let clamped_values = clamped_spline.evaluate_array(&test_points.view()).unwrap();

    let mut natural_mse = 0.0;
    let mut not_a_knot_mse = 0.0;
    let mut clamped_mse = 0.0;

    for i in 0..test_points.len() {
        let true_value = test_points[i] * test_points[i];
        natural_mse += f64::powi(natural_values[i] - true_value, 2);
        not_a_knot_mse += f64::powi(not_a_knot_values[i] - true_value, 2);
        clamped_mse += f64::powi(clamped_values[i] - true_value, 2);
    }

    natural_mse /= test_points.len() as f64;
    not_a_knot_mse /= test_points.len() as f64;
    clamped_mse /= test_points.len() as f64;

    println!("\nMean Squared Error comparison:");
    println!("  Natural spline:     {:.8}", natural_mse);
    println!("  Not-a-knot spline:  {:.8}", not_a_knot_mse);
    println!("  Clamped spline:     {:.8}", clamped_mse);

    // Find the method with the lowest MSE
    let min_mse = natural_mse.min(not_a_knot_mse.min(clamped_mse));
    if min_mse == natural_mse {
        println!("\nNatural spline has the lowest error for this dataset");
    } else if min_mse == not_a_knot_mse {
        println!("\nNot-a-knot spline has the lowest error for this dataset");
    } else {
        println!("\nClamped spline has the lowest error for this dataset");
    }

    println!();
}

fn demonstrate_derivatives_and_integration(x: &ArrayView1<f64>, y: &ArrayView1<f64>) {
    println!("2. Derivatives and Integration");
    println!("-----------------------------");

    // Create a cubic spline with clamped boundary conditions
    // For a parabola y = x^2, the first derivative is y' = 2x
    let first_deriv_start = 0.0; // y'(0) = 0
    let first_deriv_end = 8.0; // y'(4) = 8
    let bc_params = array![first_deriv_start, first_deriv_end];
    let spline = make_interp_spline(x, y, "clamped", Some(&bc_params.view())).unwrap();

    // Test derivatives
    let test_points = [0.0, 1.0, 2.0, 3.0, 4.0];

    println!("\nDerivatives comparison:");
    println!("   x   | First Derivative | Second Derivative | True f'(x) = 2x");
    println!("-------|------------------|-------------------|----------------");

    for &test_point in test_points.iter() {
        let first_deriv = spline.derivative(test_point).unwrap();
        let second_deriv = spline.second_derivative(test_point).unwrap();
        let true_first_deriv = 2.0 * test_point; // For y = x^2, y' = 2x

        println!(
            " {:.1} |      {:.4}      |       {:.4}      |      {:.4}",
            test_point, first_deriv, second_deriv, true_first_deriv
        );
    }

    // Test integration
    let integration_intervals = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (0.0, 4.0)];

    println!("\nDefinite integrals comparison:");
    println!("Interval | Spline Integral | True Integral (∫x² dx = x³/3)");
    println!("---------|----------------|------------------------------");

    for &(a, b) in integration_intervals.iter() {
        let spline_integral = spline.integrate(a, b).unwrap();

        // For y = x^2, the indefinite integral is x^3/3
        let true_integral = (f64::powi(b, 3) - f64::powi(a, 3)) / 3.0;

        println!(
            "[{:.1}, {:.1}] |     {:.4}      |          {:.4}",
            a, b, spline_integral, true_integral
        );
    }

    println!();
}

fn demonstrate_periodic_splines() {
    println!("3. Periodic Splines");
    println!("------------------");

    // Create sample data for a periodic function
    // Using sin(x) over [0, 2π] as an example
    let _n_points = 9;
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, std::f64::consts::TAU];
    let mut y = Array1::zeros(x.len());

    // Fill y with sin(x) values
    for i in 0..x.len() {
        y[i] = f64::sin(x[i]);
    }

    println!("\nSample data (y = sin(x) over [0, 2π]):");
    for i in 0..x.len() {
        println!("  x = {:.4}, y = {:.4}", x[i], y[i]);
    }

    // Ensure the endpoints have the same y value (required for periodic splines)
    // For sin(x), sin(0) = sin(2π) = 0, so we're good

    // Create a periodic spline
    let periodic_spline = make_interp_spline(&x.view(), &y.view(), "periodic", None).unwrap();

    // Create a natural spline for comparison
    let natural_spline = make_interp_spline(&x.view(), &y.view(), "natural", None).unwrap();

    // Generate test points
    let test_points = array![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.0];

    println!("\nComparison of interpolation results:");
    println!("   x   | Periodic Spline | Natural Spline |   sin(x)   ");
    println!("-------|-----------------|----------------|------------");

    for i in 0..test_points.len() {
        let test_point = test_points[i];
        let periodic_value = periodic_spline.evaluate(test_point).unwrap();
        let natural_value = natural_spline.evaluate(test_point).unwrap();
        let true_value = f64::sin(test_point);

        println!(
            " {:.1} |     {:.6}     |    {:.6}    |  {:.6}",
            test_point, periodic_value, natural_value, true_value
        );
    }

    // Demonstrate the periodicity by evaluating outside the original domain
    println!("\nDemonstrating periodicity by evaluating outside the original domain:");
    println!("Points outside [0, 2π] will produce errors with natural spline but work with periodic spline");
    println!();

    let outside_points = array![-1.0, 7.0, 8.0];

    println!("   x   | Periodic Spline |   sin(x)   |  Natural Spline  ");
    println!("-------|-----------------|------------|------------------");

    for i in 0..outside_points.len() {
        let test_point = outside_points[i];
        let periodic_value = periodic_spline.evaluate(test_point);
        let true_value = f64::sin(test_point);
        let natural_value = natural_spline.evaluate(test_point);

        let periodic_str = match periodic_value {
            Ok(val) => format!("{:.6}", val),
            Err(_) => "Error".to_string(),
        };

        let natural_str = match natural_value {
            Ok(val) => format!("{:.6}", val),
            Err(_) => "Error (out of range)".to_string(),
        };

        println!(
            " {:.1} |     {:<10}    |  {:.6}  |  {:<16}",
            test_point, periodic_str, true_value, natural_str
        );
    }

    println!();
}
