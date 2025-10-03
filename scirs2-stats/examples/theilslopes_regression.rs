use plotters::prelude::*;
use scirs2_core::ndarray::array;
use scirs2_stats::theilslopes;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Theil-Sen Robust Regression Example");
    println!("===================================\n");

    // Create some data with outliers
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut y = array![1.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 11.0, 13.0, 14.0];

    // Add outliers
    y[3] = 20.0; // Strong positive outlier
    y[7] = 0.0; // Strong negative outlier

    // Compute Theil-Sen regression
    let result = theilslopes(&x.view(), &y.view(), None, None).unwrap();

    // Compute ordinary least squares regression for comparison
    let ols_result = scirs2_stats::linregress(&x.view(), &y.view()).unwrap();
    let (ols_slope, ols_intercept___, _, _, _) = ols_result;

    // Print results
    println!("Theil-Sen Regression Results:");
    println!("Slope: {:.4}", result.slope);
    println!("Intercept: {:.4}", result.intercept);
    println!(
        "Slope confidence interval: [{:.4}, {:.4}]",
        result.slope_low, result.slope_high
    );
    println!();

    println!("Ordinary Least Squares Results:");
    println!("Slope: {:.4}", ols_slope);
    println!("Intercept: {:.4}", ols_intercept___);
    println!();

    // Create a plot to visualize the results
    let root = BitMapBackend::new("theilslopes_regression.png", (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;

    let min_x = *x
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let max_x = *x
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
        .caption("Theil-Sen vs OLS Regression", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().x_desc("X").y_desc("Y").draw()?;

    // Plot the data points
    chart.draw_series(
        x.iter()
            .zip(y.iter())
            .map(|(x, y)| Circle::new((*x, *y), 5, BLUE.filled())),
    )?;

    // Plot Theil-Sen regression line
    chart
        .draw_series(LineSeries::new(
            (0..100).map(|i| {
                let x = min_x + (max_x - min_x) * i as f64 / 99.0;
                let y = result.intercept + result.slope * x;
                (x, y)
            }),
            &RED,
        ))?
        .label("Theil-Sen")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

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

    // Plot confidence interval lines for Theil-Sen
    chart.draw_series(LineSeries::new(
        (0..100).map(|i| {
            let x = min_x + (max_x - min_x) * i as f64 / 99.0;
            let y = result.intercept + result.slope_low * x;
            (x, y)
        }),
        RED.mix(0.5),
    ))?;

    chart.draw_series(LineSeries::new(
        (0..100).map(|i| {
            let x = min_x + (max_x - min_x) * i as f64 / 99.0;
            let y = result.intercept + result.slope_high * x;
            (x, y)
        }),
        RED.mix(0.5),
    ))?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    println!("Plot saved as 'theilslopes_regression.png'");

    Ok(())
}
