use plotters::prelude::*;
use scirs2_core::ndarray::array;
use scirs2_stats::ransac;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RANSAC Robust Regression Example");
    println!("===============================\n");

    // Create data with outliers
    let x_values = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

    // Convert to 2D array for RANSAC
    let mut x = scirs2_core::ndarray::Array2::zeros((x_values.len(), 1));
    for i in 0..x_values.len() {
        x[[i, 0]] = x_values[i];
    }

    // Add outliers
    y[2] = 20.0; // Strong positive outlier
    y[7] = 0.0; // Strong negative outlier

    // Perform RANSAC regression
    let ransac_result = ransac(
        &x.view(),
        &y.view(),
        None,       // min_samples (default: 2)
        Some(3.0),  // residual_threshold
        Some(100),  // max_trials
        Some(0.99), // stop_probability
        Some(42),   // random_seed
    )
    .unwrap();

    // Compute ordinary least squares regression for comparison
    let ols_result = scirs2_stats::linregress(&x_values.view(), &y.view()).unwrap();
    let (ols_slope, ols_intercept, r, p_value, std_err) = ols_result;

    // Print results
    println!("RANSAC Regression Results:");
    println!("Intercept: {:.4}", ransac_result.coefficients[0]);
    println!("Slope: {:.4}", ransac_result.coefficients[1]);
    println!("R²: {:.4}", ransac_result.r_squared);
    println!();

    println!("Ordinary Least Squares Results:");
    println!("Intercept: {:.4}", ols_intercept);
    println!("Slope: {:.4}", ols_slope);
    println!();

    // Create a plot to visualize the results
    let root = BitMapBackend::new("ransac_regression.png", (800, 600)).into_drawing_area();

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
        .caption("RANSAC vs OLS Regression", ("sans-serif", 30))
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

    // Plot RANSAC regression line
    let ransac_intercept = ransac_result.coefficients[0];
    let ransac_slope = ransac_result.coefficients[1];

    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = ransac_intercept + ransac_slope * x;
                (x, y)
            }),
            &RED,
        ))?
        .label("RANSAC")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    // Plot OLS regression line
    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = ols_intercept + ols_slope * x;
                (x, y)
            }),
            &GREEN,
        ))?
        .label("OLS")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    // Plot inlier/outlier boundaries for RANSAC
    let threshold = 3.0; // Same as the residual_threshold parameter
    chart.draw_series(LineSeries::new(
        (0..100).map(|i| {
            let x = min_x + (max_x - min_x) * i as f64 / 99.0;
            let y = ransac_intercept + ransac_slope * x + threshold;
            (x, y)
        }),
        RED.mix(0.5),
    ))?;

    chart.draw_series(LineSeries::new(
        (0..100).map(|i| {
            let x = min_x + (max_x - min_x) * i as f64 / 99.0;
            let y = ransac_intercept + ransac_slope * x - threshold;
            (x, y)
        }),
        RED.mix(0.5),
    ))?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    println!("Plot saved as 'ransac_regression.png'");

    Ok(())
}
