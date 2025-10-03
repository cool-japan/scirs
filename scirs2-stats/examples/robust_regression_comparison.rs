use plotters::prelude::*;
use scirs2_core::ndarray::{array, Array2};
use scirs2_stats::{huber_regression, ransac, theilslopes};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Robust Regression Methods Comparison");
    println!("===================================\n");

    // Create data with outliers
    let x_values = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

    // Add outliers
    y[2] = 20.0; // Strong positive outlier
    y[7] = 0.0; // Strong negative outlier

    // Create design matrix for Huber regression
    let mut x = Array2::zeros((x_values.len(), 1));
    for i in 0..x_values.len() {
        x[[i, 0]] = x_values[i];
    }

    // Perform Theil-Sen regression
    let theilsen_result = theilslopes(&x_values.view(), &y.view(), None, None).unwrap();
    let theilsen_intercept = theilsen_result.intercept;
    let theilsen_slope = theilsen_result.slope;

    // Perform RANSAC regression
    let ransac_result = ransac(
        &x.view(), // Use the 2D array we created for Huber
        &y.view(),
        None,      // min_samples
        Some(3.0), // residual_threshold
        None,      // max_trials
        None,      // stop_probability
        Some(42),  // random_seed
    )
    .unwrap();
    let ransac_intercept = ransac_result.coefficients[0];
    let ransac_slope = ransac_result.coefficients[1];

    // Perform Huber regression
    let huber_result = huber_regression(
        &x.view(),
        &y.view(),
        None,       // epsilon
        Some(true), // fit_intercept
        None,       // scale
        None,       // max_iter
        None,       // tol
        None,       // conf_level
    )
    .unwrap();
    let huber_intercept = huber_result.coefficients[0];
    let huber_slope = huber_result.coefficients[1];

    // Compute ordinary least squares regression for comparison
    let ols_result = scirs2_stats::linregress(&x_values.view(), &y.view()).unwrap();
    let (ols_slope, ols_intercept___, _, _, _) = ols_result;

    // Print results
    println!("Theil-Sen Regression Results:");
    println!("Intercept: {:.4}", theilsen_intercept);
    println!("Slope: {:.4}", theilsen_slope);
    println!(
        "Slope confidence interval: [{:.4}, {:.4}]",
        theilsen_result.slope_low, theilsen_result.slope_high
    );
    println!();

    println!("RANSAC Regression Results:");
    println!("Intercept: {:.4}", ransac_intercept);
    println!("Slope: {:.4}", ransac_slope);
    println!("R²: {:.4}", ransac_result.r_squared);
    println!();

    println!("Huber Regression Results:");
    println!("Intercept: {:.4}", huber_intercept);
    println!("Slope: {:.4}", huber_slope);
    println!("R²: {:.4}", huber_result.r_squared);
    println!();

    println!("Ordinary Least Squares Results:");
    println!("Intercept: {:.4}", ols_intercept___);
    println!("Slope: {:.4}", ols_slope);
    println!();

    // Create a plot to visualize the results
    let root =
        BitMapBackend::new("robust_regression_comparison.png", (800, 600)).into_drawing_area();

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
    let x_range_original = min_x - 0.5..max_x + 0.5;
    let y_range = min_y - 1.0..max_y + 1.0;

    let mut chart = ChartBuilder::on(&root)
        .caption("Robust Regression Methods Comparison", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range_original.clone(), y_range)?;

    chart.configure_mesh().x_desc("X").y_desc("Y").draw()?;

    // Plot the data points
    chart.draw_series(
        x_values
            .iter()
            .zip(y.iter())
            .map(|(x, y)| Circle::new((*x, *y), 5, BLUE.filled())),
    )?;

    // Plot Theil-Sen regression line
    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = theilsen_intercept + theilsen_slope * x;
                (x, y)
            }),
            &RED,
        ))?
        .label("Theil-Sen")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    // Plot RANSAC regression line
    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = ransac_intercept + ransac_slope * x;
                (x, y)
            }),
            &GREEN,
        ))?
        .label("RANSAC")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    // Plot Huber regression line
    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = huber_intercept + huber_slope * x;
                (x, y)
            }),
            &MAGENTA,
        ))?
        .label("Huber")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], MAGENTA));

    // Plot OLS regression line
    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = ols_intercept___ + ols_slope * x;
                (x, y)
            }),
            &CYAN,
        ))?
        .label("OLS")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], CYAN));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    println!("Plot saved as 'robust_regression_comparison.png'");

    // Create a second plot focusing on true line and predictions (excluding outliers)
    let root =
        BitMapBackend::new("robust_regression_comparison_zoom.png", (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;

    // Use tighter y-range excluding outliers (focus on the trend around y = 2x)
    let y_range_tight = 0.0..25.0;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Robust Regression Methods Comparison (Zoomed)",
            ("sans-serif", 30),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range_original.clone(), y_range_tight)?;

    chart.configure_mesh().x_desc("X").y_desc("Y").draw()?;

    // Plot the data points
    chart.draw_series(
        x_values
            .iter()
            .zip(y.iter())
            .map(|(x, y)| Circle::new((*x, *y), 5, BLUE.filled())),
    )?;

    // Plot the true line (y = 2x)
    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = 0.0 + 2.0 * x; // True line is y = 2x
                (x, y)
            }),
            &BLACK,
        ))?
        .label("True Line (y = 2x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

    // Plot all regression lines
    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = theilsen_intercept + theilsen_slope * x;
                (x, y)
            }),
            &RED,
        ))?
        .label("Theil-Sen")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = ransac_intercept + ransac_slope * x;
                (x, y)
            }),
            &GREEN,
        ))?
        .label("RANSAC")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = huber_intercept + huber_slope * x;
                (x, y)
            }),
            &MAGENTA,
        ))?
        .label("Huber")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], MAGENTA));

    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = ols_intercept___ + ols_slope * x;
                (x, y)
            }),
            &CYAN,
        ))?
        .label("OLS")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], CYAN));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    println!("Zoomed plot saved as 'robust_regression_comparison_zoom.png'");

    // Print a summary of each method's performance
    println!("\nSummary of Performance:");
    println!("Method      | Intercept | Slope   | Deviation from True Model (y = 2x)");
    println!("------------------------------------------------------------------------");
    println!("True Model  |  0.0000   | 2.0000  |  -");
    println!(
        "OLS         | {:8.4}  | {:7.4} | {:7.4}%",
        ols_intercept___,
        ols_slope,
        (((ols_slope - 2.0) / 2.0) * 100.0).abs()
    );
    println!(
        "Theil-Sen   | {:8.4}  | {:7.4} | {:7.4}%",
        theilsen_intercept,
        theilsen_slope,
        (((theilsen_slope - 2.0) / 2.0) * 100.0).abs()
    );
    println!(
        "RANSAC      | {:8.4}  | {:7.4} | {:7.4}%",
        ransac_intercept,
        ransac_slope,
        (((ransac_slope - 2.0) / 2.0) * 100.0).abs()
    );
    println!(
        "Huber       | {:8.4}  | {:7.4} | {:7.4}%",
        huber_intercept,
        huber_slope,
        (((huber_slope - 2.0) / 2.0) * 100.0).abs()
    );

    Ok(())
}
