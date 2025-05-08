// Example comparing different optimizers on a simple quadratic loss function

use ndarray::Array1;
use plotters::prelude::*;
use scirs2_optim::optimizers::{Adam, AdamW, Optimizer, RAdam, RMSprop, SGD};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define a simple 2D quadratic function: f(x, y) = x^2 + 2y^2
    // Minimum at (0, 0)
    let params = Array1::from_vec(vec![5.0, 5.0]);

    // Configure optimizers
    let learning_rate = 0.1;
    let num_iterations = 100;

    let mut optimizers: Vec<(String, Box<dyn Optimizer<f64, ndarray::Ix1>>)> = vec![
        ("SGD".to_string(), Box::new(SGD::new(learning_rate))),
        ("Adam".to_string(), Box::new(Adam::new(learning_rate))),
        ("AdamW".to_string(), Box::new(AdamW::new(learning_rate))),
        ("RAdam".to_string(), Box::new(RAdam::new(learning_rate))),
        ("RMSprop".to_string(), Box::new(RMSprop::new(learning_rate))),
    ];

    // Track function values for each optimizer
    let mut paths: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut loss_histories: Vec<Vec<f64>> = Vec::new();

    // Run optimization for each optimizer
    for (name, optimizer) in optimizers.iter_mut() {
        println!("Running {} optimizer...", name);

        let mut current_params = params.clone();
        let mut path = Vec::new();
        let mut losses = Vec::new();

        path.push((current_params[0], current_params[1]));
        let current_loss = function_value(&current_params);
        losses.push(current_loss);

        for i in 0..num_iterations {
            // Compute gradients for the current parameters
            let gradients = compute_gradients(&current_params);

            // Apply optimizer step
            current_params = optimizer.step(&current_params, &gradients)?;

            // Record path and loss
            path.push((current_params[0], current_params[1]));
            let current_loss = function_value(&current_params);
            losses.push(current_loss);

            if i % 10 == 0 {
                println!(
                    "  Iteration {}: params = {:?}, loss = {}",
                    i, current_params, current_loss
                );
            }
        }

        println!("  Final loss: {}", function_value(&current_params));
        println!("  Final parameters: {:?}", current_params);

        paths.push(path);
        loss_histories.push(losses);
    }

    // Plot optimization paths
    plot_paths("optimizer_paths.png", &paths, &optimizers)?;

    // Plot loss history
    plot_loss_history("optimizer_loss_history.png", &loss_histories, &optimizers)?;

    Ok(())
}

// Define the function: f(x, y) = x^2 + 2y^2
fn function_value(params: &Array1<f64>) -> f64 {
    params[0].powi(2) + 2.0 * params[1].powi(2)
}

// Define the gradients: ∇f = [2x, 4y]
fn compute_gradients(params: &Array1<f64>) -> Array1<f64> {
    Array1::from_vec(vec![2.0 * params[0], 4.0 * params[1]])
}

// Plot optimization paths
fn plot_paths(
    filename: &str,
    paths: &[Vec<(f64, f64)>],
    optimizers: &[(String, Box<dyn Optimizer<f64, ndarray::Ix1>>)],
) -> Result<(), Box<dyn std::error::Error>> {
    // Determine plot boundaries
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for path in paths {
        for &(x, y) in path {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
    }

    // Add some padding
    let x_padding = (x_max - x_min) * 0.1;
    let y_padding = (y_max - y_min) * 0.1;

    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Optimizer Comparison", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            (x_min - x_padding)..(x_max + x_padding),
            (y_min - y_padding)..(y_max + y_padding),
        )?;

    chart.configure_mesh().draw()?;

    // Plot contour lines
    let contour_density = 40;
    let contour_values = [0.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0];

    for &value in &contour_values {
        let points: Vec<(f64, f64)> = (0..contour_density)
            .flat_map(|i| {
                let theta = 2.0 * std::f64::consts::PI * (i as f64) / (contour_density as f64);
                let r = (value / (f64::cos(theta).powi(2) + 2.0 * f64::sin(theta).powi(2))).sqrt();
                let x = r * f64::cos(theta);
                let y = r * f64::sin(theta);
                Some((x, y))
            })
            .collect();

        chart.draw_series(LineSeries::new(points, &BLACK.mix(0.3)))?;
    }

    // Plot optimizer paths
    let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &YELLOW, &BLACK];

    for (i, (path, (name, _))) in paths.iter().zip(optimizers).enumerate() {
        let color = colors[i % colors.len()];

        // Draw path
        chart
            .draw_series(LineSeries::new(path.clone(), color))?
            .label(name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));

        // Draw markers at start and end
        chart.draw_series(PointSeries::of_element(
            vec![path.first().unwrap().clone()],
            5,
            color,
            &|c, s, st| Circle::new(c, s, st.filled()),
        ))?;

        chart.draw_series(PointSeries::of_element(
            vec![path.last().unwrap().clone()],
            5,
            color,
            &|c, s, st| {
                EmptyElement::at(c)
                    + Circle::new((0, 0), s, st.filled())
                    + Text::new(format!("{}", name), (10, 0), ("sans-serif", 10).into_font())
            },
        ))?;
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    println!("Path plot saved to {}", filename);

    Ok(())
}

// Plot loss history
fn plot_loss_history(
    filename: &str,
    loss_histories: &[Vec<f64>],
    optimizers: &[(String, Box<dyn Optimizer<f64, ndarray::Ix1>>)],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_loss = loss_histories
        .iter()
        .flat_map(|losses| losses.iter().cloned())
        .fold(0.0, f64::max);

    let max_iter = loss_histories
        .iter()
        .map(|losses| losses.len())
        .max()
        .unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss History Comparison", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..(max_iter as f64), 0.0..max_loss)?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc("Loss")
        .draw()?;

    let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &YELLOW, &BLACK];

    for (i, (losses, (name, _))) in loss_histories.iter().zip(optimizers).enumerate() {
        let color = colors[i % colors.len()];

        let losses_with_idx: Vec<(f64, f64)> = losses
            .iter()
            .enumerate()
            .map(|(idx, &loss)| (idx as f64, loss))
            .collect();

        chart
            .draw_series(LineSeries::new(losses_with_idx, color))?
            .label(name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    println!("Loss history plot saved to {}", filename);

    Ok(())
}
