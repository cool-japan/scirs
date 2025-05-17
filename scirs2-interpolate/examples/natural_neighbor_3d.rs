//! Example demonstrating 3D Natural Neighbor interpolation
//!
//! This example shows how to use the Voronoi-based Natural Neighbor
//! interpolation methods for scattered 3D data.

use ndarray::{Array1, Array2};
use rand::Rng;
use scirs2_interpolate::voronoi::{make_laplace_interpolator, make_sibson_interpolator};
use std::error::Error;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn Error>> {
    // Generate scattered data points in 3D
    let n_points = 100;
    let mut rng = rand::rng();

    // Create points in a 3D domain
    let mut points_vec = Vec::with_capacity(n_points * 3);
    for _ in 0..n_points {
        let x = rng.random_range(0.0..=10.0);
        let y = rng.random_range(0.0..=10.0);
        let z = rng.random_range(0.0..=10.0);
        points_vec.push(x);
        points_vec.push(y);
        points_vec.push(z);
    }

    let points = Array2::from_shape_vec((n_points, 3), points_vec)?;

    // Create values with a test function
    let mut values_vec = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let x = points[[i, 0]];
        let y = points[[i, 1]];
        let z = points[[i, 2]];

        // Test function: 3D peaks with exponential decay
        let value =
            2.0 * f64::exp(
                -(f64::powi(x - 5.0, 2) + f64::powi(y - 5.0, 2) + f64::powi(z - 5.0, 2) / 8.0),
            ) + f64::exp(
                -(f64::powi(x - 2.0, 2) + f64::powi(y - 7.0, 2) + f64::powi(z - 3.0, 2) / 2.0),
            ) + 0.5
                * f64::exp(
                    -(f64::powi(x - 8.0, 2) + f64::powi(y - 3.0, 2) + f64::powi(z - 7.0, 2) / 4.0),
                );

        values_vec.push(value);
    }

    let values = Array1::from_vec(values_vec);

    // Create interpolators with different methods
    println!("Creating Sibson interpolator...");
    let sibson_interpolator = make_sibson_interpolator(points.clone(), values.clone())?;

    println!("Creating Laplace interpolator...");
    let laplace_interpolator = make_laplace_interpolator(points.clone(), values.clone())?;

    // Generate a grid of query points for visualization
    // Since we can't easily visualize 3D data in a console, we'll:
    // 1. Create a regular grid in 3D
    // 2. Interpolate at each grid point
    // 3. Save the results to a file for external visualization

    let grid_size = 20; // Smaller for demonstration
    let x_vals = Array1::linspace(0.0, 10.0, grid_size);
    let y_vals = Array1::linspace(0.0, 10.0, grid_size);
    let z_vals = Array1::linspace(0.0, 10.0, grid_size);

    // Create output file for Sibson interpolation results
    let mut sibson_file = File::create("sibson_3d_interpolation.csv")?;
    writeln!(sibson_file, "x,y,z,value")?;

    // Create output file for Laplace interpolation results
    let mut laplace_file = File::create("laplace_3d_interpolation.csv")?;
    writeln!(laplace_file, "x,y,z,value")?;

    // Interpolate on a grid and save results
    println!("Interpolating on a 3D grid (this may take a while)...");
    let total_points = grid_size.pow(3);
    let mut processed = 0;

    for &x in x_vals.iter() {
        for &y in y_vals.iter() {
            for &z in z_vals.iter() {
                let query = Array1::from_vec(vec![x, y, z]);

                // Interpolate using Sibson method
                match sibson_interpolator.interpolate(&query.view()) {
                    Ok(value) => {
                        writeln!(sibson_file, "{},{},{},{}", x, y, z, value)?;
                    }
                    Err(_) => {
                        writeln!(sibson_file, "{},{},{},NaN", x, y, z)?;
                    }
                }

                // Interpolate using Laplace method
                match laplace_interpolator.interpolate(&query.view()) {
                    Ok(value) => {
                        writeln!(laplace_file, "{},{},{},{}", x, y, z, value)?;
                    }
                    Err(_) => {
                        writeln!(laplace_file, "{},{},{},NaN", x, y, z)?;
                    }
                }

                processed += 1;
                if processed % (total_points / 10) == 0 {
                    println!(
                        "Progress: {:.1}%",
                        100.0 * processed as f64 / total_points as f64
                    );
                }
            }
        }
    }

    // Also save the original scattered data points
    let mut data_file = File::create("original_3d_data.csv")?;
    writeln!(data_file, "x,y,z,value")?;

    for i in 0..n_points {
        let x = points[[i, 0]];
        let y = points[[i, 1]];
        let z = points[[i, 2]];
        let value = values[i];

        writeln!(data_file, "{},{},{},{}", x, y, z, value)?;
    }

    println!("Interpolation complete!");
    println!("Results saved to:");
    println!("  - sibson_3d_interpolation.csv");
    println!("  - laplace_3d_interpolation.csv");
    println!("  - original_3d_data.csv");
    println!("\nYou can visualize these files using a 3D plotting tool such as:");
    println!("  - Python with matplotlib, plotly, or mayavi");
    println!("  - ParaView");
    println!("  - VisIt");

    // Example Python code for visualization
    println!("\nExample Python code for visualization:");
    println!("```python");
    println!("import pandas as pd");
    println!("import plotly.graph_objects as go");
    println!();
    println!("# Load the data");
    println!("sibson_data = pd.read_csv('sibson_3d_interpolation.csv')");
    println!("original_data = pd.read_csv('original_3d_data.csv')");
    println!();
    println!("# Create a 3D scatter plot for the interpolated values");
    println!("fig = go.Figure(data=[");
    println!("    go.Scatter3d(");
    println!("        x=sibson_data['x'],");
    println!("        y=sibson_data['y'],");
    println!("        z=sibson_data['z'],");
    println!("        mode='markers',");
    println!("        marker=dict(");
    println!("            size=5,");
    println!("            color=sibson_data['value'],");
    println!("            colorscale='Viridis',");
    println!("            opacity=0.8,");
    println!("            showscale=True");
    println!("        ),");
    println!("        name='Interpolated Values'");
    println!("    ),");
    println!("    go.Scatter3d(");
    println!("        x=original_data['x'],");
    println!("        y=original_data['y'],");
    println!("        z=original_data['z'],");
    println!("        mode='markers',");
    println!("        marker=dict(");
    println!("            size=10,");
    println!("            color=original_data['value'],");
    println!("            colorscale='Viridis',");
    println!("            symbol='circle',");
    println!("            opacity=1.0");
    println!("        ),");
    println!("        name='Original Data Points'");
    println!("    )");
    println!("]);");
    println!();
    println!("fig.update_layout(");
    println!("    title='3D Natural Neighbor Interpolation',");
    println!("    scene=dict(");
    println!("        xaxis_title='X',");
    println!("        yaxis_title='Y',");
    println!("        zaxis_title='Z',");
    println!("    )");
    println!(")");
    println!();
    println!("fig.show()");
    println!("```");

    Ok(())
}
