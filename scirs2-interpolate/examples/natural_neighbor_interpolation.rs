//! Example demonstrating Natural Neighbor interpolation
//!
//! This example shows how to use the Voronoi-based Natural Neighbor
//! interpolation methods for scattered 2D data.

use plotters::prelude::*;
use plotters::style::colors::{BLACK, BLUE, RED};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Rng;
use scirs2_interpolate::voronoi::{make_laplace_interpolator, make_sibson_interpolator};
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    // Generate scattered data points
    let n_points = 30;
    let mut rng = scirs2_core::random::rng();

    // Create points in a 2D domain
    let mut points_vec = Vec::with_capacity(n_points * 2);
    for _ in 0..n_points {
        let x = rng.random_range(0.0..=10.0);
        let y = rng.random_range(0.0..=10.0);
        points_vec.push(x);
        points_vec.push(y);
    }

    let points = Array2::from_shape_vec((n_points, 2), points_vec)?;

    // Create values with a test function
    let mut values_vec = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let x = points[[i, 0]];
        let y = points[[i, 1]];

        // Test function: peaks with exponential decay
        let value = 2.0 * f64::exp(-(f64::powi(x - 5.0, 2) + f64::powi(y - 5.0, 2) / 8.0))
            + f64::exp(-(f64::powi(x - 2.0, 2) + f64::powi(y - 7.0, 2) / 2.0))
            + 0.5 * f64::exp(-(f64::powi(x - 8.0, 2) + f64::powi(y - 3.0, 2) / 4.0));

        values_vec.push(value);
    }

    let values = Array1::from_vec(values_vec);

    // Create interpolators with different methods
    let sibson_interpolator = make_sibson_interpolator(points.clone(), values.clone())?;
    let laplace_interpolator = make_laplace_interpolator(points.clone(), values.clone())?;

    // Create the plot
    let root =
        BitMapBackend::new("natural_neighbor_comparison.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let _plot_title = "Natural Neighbor Interpolation Comparison";

    // Split the drawing area into two parts
    let areas = root.split_evenly((2, 1));

    // Define plot area
    let mut cc = ChartBuilder::on(&areas[0])
        .caption("Sibson Method", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..10.0, 0.0..10.0)?;

    cc.configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_desc("X")
        .y_desc("Y")
        .draw()?;

    // Create a grid for interpolation visualization
    let grid_size = 100;
    let x_vals = Array1::linspace(0.0, 10.0, grid_size);
    let y_vals = Array1::linspace(0.0, 10.0, grid_size);

    // Create the Sibson interpolation grid
    let mut sibson_grid = Array2::zeros((grid_size, grid_size));

    for (i, &x) in x_vals.iter().enumerate() {
        for (j, &y) in y_vals.iter().enumerate() {
            let query = Array1::from_vec(vec![x, y]);
            match sibson_interpolator.interpolate(&query.view()) {
                Ok(value) => sibson_grid[[i, j]] = value,
                Err(_) => sibson_grid[[i, j]] = 0.0, // Handle error case
            }
        }
    }

    // Find min and max values for color mapping
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;

    for &v in values.iter() {
        min_value = min_value.min(v);
        max_value = max_value.max(v);
    }

    // Normalize to 0-1 range for color mapping
    let normalize = |v: f64| -> f64 {
        if max_value == min_value {
            0.5
        } else {
            (v - min_value) / (max_value - min_value)
        }
    };

    // Plot the scattered points with their interpolated values
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = x_vals[i];
            let y = y_vals[j];

            let value = sibson_grid[[i, j]];
            let norm_value = normalize(value);

            // Create a color gradient from blue to red
            let r = (norm_value * 255.0) as u8;
            let g = 0;
            let b = ((1.0 - norm_value) * 255.0) as u8;

            let point_color = RGBColor(r, g, b);

            let point_size = 3;
            cc.draw_series(PointSeries::of_element(
                vec![(x, y)],
                point_size,
                point_color,
                &|c, s, st| Circle::new(c, s, st),
            ))?;
        }
    }

    // Plot the original data points as black circles
    for i in 0..n_points {
        let x = points[[i, 0]];
        let y = points[[i, 1]];

        cc.draw_series(PointSeries::of_element(
            vec![(x, y)],
            5,
            BLACK.filled(),
            &|c, s, st| Circle::new(c, s, st.stroke_width(1)),
        ))?;
    }

    // Now create a similar plot for Laplace interpolation
    let mut cc = ChartBuilder::on(&areas[1])
        .caption("Laplace Method", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..10.0, 0.0..10.0)?;

    cc.configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_desc("X")
        .y_desc("Y")
        .draw()?;

    // Create the Laplace interpolation grid
    let mut laplace_grid = Array2::zeros((grid_size, grid_size));

    for (i, &x) in x_vals.iter().enumerate() {
        for (j, &y) in y_vals.iter().enumerate() {
            let query = Array1::from_vec(vec![x, y]);
            match laplace_interpolator.interpolate(&query.view()) {
                Ok(value) => laplace_grid[[i, j]] = value,
                Err(_) => laplace_grid[[i, j]] = 0.0, // Handle error case
            }
        }
    }

    // Plot the scattered points with their interpolated values
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = x_vals[i];
            let y = y_vals[j];

            let value = laplace_grid[[i, j]];
            let norm_value = normalize(value);

            // Create a color gradient from blue to red
            let r = (norm_value * 255.0) as u8;
            let g = 0;
            let b = ((1.0 - norm_value) * 255.0) as u8;

            let point_color = RGBColor(r, g, b);

            let point_size = 3;
            cc.draw_series(PointSeries::of_element(
                vec![(x, y)],
                point_size,
                point_color,
                &|c, s, st| Circle::new(c, s, st),
            ))?;
        }
    }

    // Plot the original data points as black circles
    for i in 0..n_points {
        let x = points[[i, 0]];
        let y = points[[i, 1]];

        cc.draw_series(PointSeries::of_element(
            vec![(x, y)],
            5,
            BLACK.filled(),
            &|c, s, st| Circle::new(c, s, st.stroke_width(1)),
        ))?;
    }

    // Add a title to the plot
    root.present().expect("Failed to present plot");
    println!("Plot has been saved as 'natural_neighbor_comparison.png'");

    // Now create a 1D slice comparison to better visualize differences
    // between Sibson and Laplace methods
    let slice_root =
        BitMapBackend::new("natural_neighbor_slice.png", (800, 400)).into_drawing_area();
    slice_root.fill(&WHITE)?;

    let mut cc = ChartBuilder::on(&slice_root)
        .caption(
            "Natural Neighbor Methods Comparison - Slice at y=5.0",
            ("sans-serif", 20),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0f64..10.0f64, 0.0f64..2.0f64)?;

    cc.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .x_desc("X")
        .y_desc("Value")
        .draw()?;

    // Get slices across y=5.0
    let slice_y_idx = grid_size / 2; // Middle of the grid

    // Extract slices
    let sibson_slice = sibson_grid.column(slice_y_idx).to_owned();
    let laplace_slice = laplace_grid.column(slice_y_idx).to_owned();

    // Plot the slices
    cc.draw_series(LineSeries::new(
        x_vals
            .iter()
            .zip(sibson_slice.iter())
            .map(|(&x, &y)| (x, y)),
        RED.mix(0.8).stroke_width(3),
    ))?
    .label("Sibson")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.8).stroke_width(3)));

    cc.draw_series(LineSeries::new(
        x_vals
            .iter()
            .zip(laplace_slice.iter())
            .map(|(&x, &y)| (x, y)),
        BLUE.mix(0.8).stroke_width(3),
    ))?
    .label("Laplace")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.mix(0.8).stroke_width(3)));

    // Plot the original data points that are close to y=5.0
    let threshold = 0.5; // Points within this distance of y=5.0 will be shown
    let close_points: Vec<_> = (0..n_points)
        .filter(|&i| (points[[i, 1]] - 5.0).abs() < threshold)
        .map(|i| (points[[i, 0]], values[i]))
        .collect();

    cc.draw_series(
        close_points
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 5, BLACK.filled())),
    )?
    .label("Data points")
    .legend(|(x, y)| Circle::new((x + 10, y), 5, BLACK.filled()));

    cc.configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    slice_root.present()?;
    println!("Slice comparison plot has been saved as 'natural_neighbor_slice.png'");

    Ok(())
}
