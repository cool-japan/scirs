use plotters::prelude::*;
use scirs2_core::ndarray::{array, Array2};
use scirs2_stats::huber_regression;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Huber Robust Regression Example");
    println!("==============================\n");

    // Create data with outliers
    let x_values = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

    // Add outliers
    y[2] = 20.0; // Strong positive outlier
    y[7] = 0.0; // Strong negative outlier

    // Create design matrix (add as column to x)
    let mut x = Array2::zeros((x_values.len(), 1));
    for i in 0..x_values.len() {
        x[[i, 0]] = x_values[i];
    }

    // Perform Huber regression with different epsilon values
    let huber_result_default = huber_regression(
        &x.view(),
        &y.view(),
        None,       // epsilon (default: 1.345)
        Some(true), // fit_intercept
        None,       // scale
        None,       // max_iter
        None,       // tol
        None,       // conf_level
    )
    .unwrap();

    // More robust (lower epsilon)
    let huber_result_robust = huber_regression(
        &x.view(),
        &y.view(),
        Some(0.8),  // epsilon (more robust to outliers)
        Some(true), // fit_intercept
        None,       // scale
        None,       // max_iter
        None,       // tol
        None,       // conf_level
    )
    .unwrap();

    // Less robust (higher epsilon, more like OLS)
    let huber_result_less_robust = huber_regression(
        &x.view(),
        &y.view(),
        Some(2.5),  // epsilon (less robust, more like OLS)
        Some(true), // fit_intercept
        None,       // scale
        None,       // max_iter
        None,       // tol
        None,       // conf_level
    )
    .unwrap();

    // Compute ordinary least squares regression for comparison
    let ols_result = scirs2_stats::linregress(&x_values.view(), &y.view()).unwrap();
    let (ols_slope, ols_intercept___, _, _, _) = ols_result;

    // Print results
    println!("Huber Regression Results (default epsilon=1.345):");
    println!("Intercept: {:.4}", huber_result_default.coefficients[0]);
    println!("Slope: {:.4}", huber_result_default.coefficients[1]);
    println!("R²: {:.4}", huber_result_default.r_squared);
    println!();

    println!("Huber Regression Results (robust with epsilon=0.8):");
    println!("Intercept: {:.4}", huber_result_robust.coefficients[0]);
    println!("Slope: {:.4}", huber_result_robust.coefficients[1]);
    println!("R²: {:.4}", huber_result_robust.r_squared);
    println!();

    println!("Huber Regression Results (less robust with epsilon=2.5):");
    println!("Intercept: {:.4}", huber_result_less_robust.coefficients[0]);
    println!("Slope: {:.4}", huber_result_less_robust.coefficients[1]);
    println!("R²: {:.4}", huber_result_less_robust.r_squared);
    println!();

    println!("Ordinary Least Squares Results:");
    println!("Intercept: {:.4}", ols_intercept___);
    println!("Slope: {:.4}", ols_slope);
    println!();

    // Create a plot to visualize the results
    let root = BitMapBackend::new("huber_regression.png", (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;

    let min_x = *x_values
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let max_x = *x_values
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let min_y = *y
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let max_y = *y
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    // Add some padding to the plot ranges
    let x_range = min_x - 0.5..max_x + 0.5;
    let y_range = min_y - 1.0..max_y + 1.0;

    let mut chart = ChartBuilder::on(&root)
        .caption("Huber vs OLS Regression", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().x_desc("X").y_desc("Y").draw()?;

    // Plot the data points
    chart.draw_series(
        x_values
            .iter()
            .zip(y.iter())
            .map(|(x, y)| Circle::new((*x, *y), 5, BLUE.filled())),
    )?;

    // Plot Huber regression lines
    let huber_default_intercept = huber_result_default.coefficients[0];
    let huber_default_slope = huber_result_default.coefficients[1];

    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = huber_default_intercept + huber_default_slope * x;
                (x, y)
            }),
            &RED,
        ))?
        .label("Huber (ε=1.345)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    let huber_robust_intercept = huber_result_robust.coefficients[0];
    let huber_robust_slope = huber_result_robust.coefficients[1];

    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = huber_robust_intercept + huber_robust_slope * x;
                (x, y)
            }),
            &MAGENTA,
        ))?
        .label("Huber (ε=0.8)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], MAGENTA));

    let huber_less_robust_intercept = huber_result_less_robust.coefficients[0];
    let huber_less_robust_slope = huber_result_less_robust.coefficients[1];

    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = huber_less_robust_intercept + huber_less_robust_slope * x;
                (x, y)
            }),
            &CYAN,
        ))?
        .label("Huber (ε=2.5)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], CYAN));

    // Plot OLS regression line
    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = ols_intercept___ + ols_slope * x;
                (x, y)
            }),
            &GREEN,
        ))?
        .label("OLS")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    println!("Plot saved as 'huber_regression.png'");

    Ok(())
}
