use ndarray::{array, Array1};
use plotters::prelude::*;
use scirs2_spatial::transform::{Rotation, RotationSpline};
use std::error::Error;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn Error>> {
    println!("RotationSpline Example");
    println!("=====================\n");

    // Create a sequence of rotations
    let rotations = vec![
        Rotation::identity(),
        {
            let angles = array![0.0, 0.0, PI / 2.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
        {
            let angles = array![0.0, 0.0, PI];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
        {
            let angles = array![0.0, 0.0, 3.0 * PI / 2.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
        {
            let angles = array![0.0, 0.0, 2.0 * PI];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
    ];

    // Define times
    let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];

    // Create spline
    let mut spline = RotationSpline::new(&rotations, &times)?;
    println!(
        "Created spline with default interpolation: {}",
        spline.interpolation_type()
    );

    // Test point
    let point = array![1.0, 0.0, 0.0];

    // Interpolate at various times
    let test_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

    println!("\nSampling at various times with SLERP:");
    for &t in &test_times {
        let rot = spline.interpolate(t);
        let rotated = rot.apply(&point.view());
        println!(
            "t = {:.1}: [{:.4}, {:.4}, {:.4}]",
            t, rotated[0], rotated[1], rotated[2]
        );
    }

    // Change interpolation type
    spline.set_interpolation_type("cubic")?;
    println!(
        "\nChanged interpolation to: {}",
        spline.interpolation_type()
    );

    println!("\nSampling at various times with cubic spline:");
    for &t in &test_times {
        let rot = spline.interpolate(t);
        let rotated = rot.apply(&point.view());
        println!(
            "t = {:.1}: [{:.4}, {:.4}, {:.4}]",
            t, rotated[0], rotated[1], rotated[2]
        );
    }

    // Calculate angular velocities
    println!("\nCalculating angular velocities at sample times:");
    for &t in &test_times {
        let velocity = spline.angular_velocity(t);

        println!(
            "  t = {:.1}: [{:.4}, {:.4}, {:.4}] rad/s",
            t, velocity[0], velocity[1], velocity[2]
        );
    }

    // Sample
    println!("\nGenerating uniform samples with cubic spline:");
    let (sample_times, sample_rotations) = spline.sample(9);

    for i in 0..9 {
        let rotated = sample_rotations[i].apply(&point.view());
        println!(
            "t = {:.2}: [{:.4}, {:.4}, {:.4}]",
            sample_times[i], rotated[0], rotated[1], rotated[2]
        );
    }

    // Demonstrate a 3D rotation path
    println!("\nDemonstrating 3D rotation path:");

    // Create more complex rotations in 3D
    let rotations_3d = vec![
        {
            let angles = array![0.0, 0.0, 0.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
        {
            let angles = array![PI / 4.0, 0.0, 0.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
        {
            let angles = array![PI / 4.0, PI / 4.0, 0.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
        {
            let angles = array![PI / 4.0, PI / 4.0, PI / 4.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
        {
            let angles = array![0.0, 0.0, 0.0];
            Rotation::from_euler(&angles.view(), "xyz")?
        },
    ];

    let mut spline_3d = RotationSpline::new(&rotations_3d, &times)?;
    spline_3d.set_interpolation_type("cubic")?;

    // Test point for 3D
    let point_3d = array![1.0, 0.0, 0.0];

    // Sample 3D path
    let (sample_times_3d, sample_rotations_3d) = spline_3d.sample(9);

    for i in 0..9 {
        let rotated = sample_rotations_3d[i].apply(&point_3d.view());
        println!(
            "t = {:.2}: [{:.4}, {:.4}, {:.4}]",
            sample_times_3d[i], rotated[0], rotated[1], rotated[2]
        );
    }

    // Generate 2D visualization
    visualize_spline_rotations(&spline, &point)?;

    // Generate 3D visualization
    visualize_spline_rotations_3d(&spline_3d, &point_3d)?;

    println!("\nExample completed successfully!");

    Ok(())
}

/// Visualize the effect of rotation spline on a point in 2D
fn visualize_spline_rotations(
    spline: &RotationSpline,
    point: &Array1<f64>,
) -> Result<(), Box<dyn Error>> {
    // Create a drawing area
    let root =
        BitMapBackend::new("rotation_spline_visualization_2d.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Define chart area
    let mut chart = ChartBuilder::on(&root)
        .caption("Rotation Spline Path (2D)", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.5f64..1.5f64, -1.5f64..1.5f64)?;

    // Draw mesh and axes
    chart
        .configure_mesh()
        .x_desc("X coordinate")
        .y_desc("Y coordinate")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    // Sample the spline with high resolution for visualization
    const NUM_SAMPLES: usize = 100;
    let (_times, rotations) = spline.sample(NUM_SAMPLES);

    // Extract rotation path
    let mut path = Vec::with_capacity(NUM_SAMPLES);
    for rotation in rotations.iter().take(NUM_SAMPLES) {
        let rotated = rotation.apply(&point.view());
        path.push((rotated[0], rotated[1]));
    }

    // Draw the path
    chart.draw_series(LineSeries::new(path, &RED.mix(0.8)).point_size(3))?;

    // Create more visible dots for the control points
    let mut control_points = Vec::with_capacity(spline.rotations().len());
    for rot in spline.rotations() {
        let rotated = rot.apply(&point.view());
        control_points.push((rotated[0], rotated[1]));
    }

    chart.draw_series(PointSeries::of_element(
        control_points,
        5,
        &BLUE.mix(0.8),
        &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
    ))?;

    // Draw interpolation type and info
    root.draw_text(
        &format!("Interpolation: {}", spline.interpolation_type()),
        &TextStyle::from(("sans-serif", 20)).color(&BLACK),
        (50, 20),
    )?;

    root.present()?;
    println!("\nVisualization saved to rotation_spline_visualization_2d.png");

    Ok(())
}

/// Visualize the effect of rotation spline on a point in 3D
fn visualize_spline_rotations_3d(
    spline: &RotationSpline,
    point: &Array1<f64>,
) -> Result<(), Box<dyn Error>> {
    // Create a drawing area
    let root =
        BitMapBackend::new("rotation_spline_visualization_3d.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create 3D projections as multiple 2D views
    let plotting_area = root.titled("Rotation Spline Path (3D Projections)", ("sans-serif", 30))?;

    let (top, bottom) = plotting_area.split_vertically(300);
    let (left, right) = bottom.split_horizontally(400);

    // Create three chart areas for different projections (XY, XZ, YZ)
    let mut xy_chart = ChartBuilder::on(&top)
        .margin(10)
        .caption("XY Projection", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.5f64..1.5f64, -1.5f64..1.5f64)?;

    let mut xz_chart = ChartBuilder::on(&left)
        .margin(10)
        .caption("XZ Projection", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.5f64..1.5f64, -1.5f64..1.5f64)?;

    let mut yz_chart = ChartBuilder::on(&right)
        .margin(10)
        .caption("YZ Projection", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.5f64..1.5f64, -1.5f64..1.5f64)?;

    // Draw mesh and axes for each projection
    xy_chart
        .configure_mesh()
        .x_desc("X coordinate")
        .y_desc("Y coordinate")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    xz_chart
        .configure_mesh()
        .x_desc("X coordinate")
        .y_desc("Z coordinate")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    yz_chart
        .configure_mesh()
        .x_desc("Y coordinate")
        .y_desc("Z coordinate")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    // Sample the spline with high resolution for visualization
    const NUM_SAMPLES: usize = 100;
    let (_times, rotations) = spline.sample(NUM_SAMPLES);

    // Extract rotation paths for each projection
    let mut xy_path = Vec::with_capacity(NUM_SAMPLES);
    let mut xz_path = Vec::with_capacity(NUM_SAMPLES);
    let mut yz_path = Vec::with_capacity(NUM_SAMPLES);

    for rotation in rotations.iter().take(NUM_SAMPLES) {
        let rotated = rotation.apply(&point.view());
        xy_path.push((rotated[0], rotated[1]));
        xz_path.push((rotated[0], rotated[2]));
        yz_path.push((rotated[1], rotated[2]));
    }

    // Draw the paths
    xy_chart.draw_series(LineSeries::new(xy_path, &RED.mix(0.8)).point_size(3))?;
    xz_chart.draw_series(LineSeries::new(xz_path, &GREEN.mix(0.8)).point_size(3))?;
    yz_chart.draw_series(LineSeries::new(yz_path, &BLUE.mix(0.8)).point_size(3))?;

    // Create more visible dots for the control points in each projection
    let mut xy_control_points = Vec::with_capacity(spline.rotations().len());
    let mut xz_control_points = Vec::with_capacity(spline.rotations().len());
    let mut yz_control_points = Vec::with_capacity(spline.rotations().len());

    for rot in spline.rotations() {
        let rotated = rot.apply(&point.view());
        xy_control_points.push((rotated[0], rotated[1]));
        xz_control_points.push((rotated[0], rotated[2]));
        yz_control_points.push((rotated[1], rotated[2]));
    }

    // Draw control points for each projection
    xy_chart.draw_series(PointSeries::of_element(
        xy_control_points,
        5,
        &RED.mix(0.8),
        &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
    ))?;

    xz_chart.draw_series(PointSeries::of_element(
        xz_control_points,
        5,
        &GREEN.mix(0.8),
        &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
    ))?;

    yz_chart.draw_series(PointSeries::of_element(
        yz_control_points,
        5,
        &BLUE.mix(0.8),
        &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
    ))?;

    // Draw interpolation type and info
    root.draw_text(
        &format!("Interpolation: {}", spline.interpolation_type()),
        &TextStyle::from(("sans-serif", 20)).color(&BLACK),
        (50, 20),
    )?;

    root.present()?;
    println!("\nVisualization saved to rotation_spline_visualization_3d.png");

    Ok(())
}
