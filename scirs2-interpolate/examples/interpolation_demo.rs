use ndarray::{array, Array1, Array2}; // ArrayView1は使用していない
use scirs2_interpolate::{
    cubic_interpolate,

    // N-dimensional interpolation
    interpnd::{
        make_interp_nd,
        make_interp_scattered,
        ExtrapolateMode, // map_coordinatesは使用していない
        InterpolationMethod as InterpNDMethod,
        ScatteredInterpolationMethod,
        ScatteredInterpolatorParams,
    },
    linear_interpolate,
    make_interp_spline,

    nearest_interpolate,
    // Spline interpolation
    // CubicSpline - 現在未使用
    // 1D interpolation
    Interp1d,
    InterpolationMethod as Interp1DMethod,
};

fn main() {
    println!("SciRS2 Interpolation Methods Demonstration");
    println!("========================================");

    // 1D Interpolation Demo
    demo_1d_interpolation();

    // Spline Interpolation Demo
    demo_spline_interpolation();

    // N-dimensional Interpolation Demo
    demo_nd_interpolation();
}

fn demo_1d_interpolation() {
    println!("\n1. One-dimensional Interpolation");
    println!("-------------------------------");

    // Create sample data
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = array![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x²

    // Import ExtrapolateMode for Interp1d
    use scirs2_interpolate::interp1d::ExtrapolateMode;

    // Create interpolator objects
    let interp_nearest = Interp1d::new(
        &x.view(),
        &y.view(),
        Interp1DMethod::Nearest,
        ExtrapolateMode::Error,
    )
    .unwrap();

    let interp_linear = Interp1d::new(
        &x.view(),
        &y.view(),
        Interp1DMethod::Linear,
        ExtrapolateMode::Error,
    )
    .unwrap();

    let interp_cubic = Interp1d::new(
        &x.view(),
        &y.view(),
        Interp1DMethod::Cubic,
        ExtrapolateMode::Error,
    )
    .unwrap();

    // Points to evaluate
    let x_new = array![0.5, 1.5, 2.2, 3.7];
    println!("Sample points: [0, 1, 2, 3, 4] → [0, 1, 4, 9, 16]  (y = x²)");
    println!("\nEvaluation points:");

    // For the 1D interpolation, we need to evaluate point by point
    let mut y_nearest = Vec::with_capacity(x_new.len());
    let mut y_linear = Vec::with_capacity(x_new.len());
    let mut y_cubic = Vec::with_capacity(x_new.len());

    for &x in x_new.iter() {
        y_nearest.push(interp_nearest.evaluate(x).unwrap());
        y_linear.push(interp_linear.evaluate(x).unwrap());
        y_cubic.push(interp_cubic.evaluate(x).unwrap());
    }

    println!("  x   | Nearest |  Linear  |   Cubic  | Actual (x²)");
    println!("------|---------|----------|----------|------------");
    for i in 0..x_new.len() {
        println!(
            " {:.1} |   {:.1}   |   {:.2}   |   {:.2}   |    {:.2}",
            x_new[i],
            y_nearest[i],
            y_linear[i],
            y_cubic[i],
            x_new[i] * x_new[i]
        );
    }

    // Direct function demonstration
    println!("\nDirect function calls:");
    let x_direct = array![2.5];
    let y_direct_nearest = nearest_interpolate(&x.view(), &y.view(), &x_direct.view()).unwrap();
    let y_direct_linear = linear_interpolate(&x.view(), &y.view(), &x_direct.view()).unwrap();
    let y_direct_cubic = cubic_interpolate(&x.view(), &y.view(), &x_direct.view()).unwrap();

    println!(
        "  x = 2.5  | Nearest: {:.2} | Linear: {:.2} | Cubic: {:.2} | Actual: {:.2}",
        y_direct_nearest[0],
        y_direct_linear[0],
        y_direct_cubic[0],
        2.5 * 2.5
    );
}

fn demo_spline_interpolation() {
    println!("\n2. Spline Interpolation");
    println!("----------------------");

    // Create sample data (non-uniform spacing)
    let x = array![0.0, 0.5, 2.0, 3.5, 4.0, 6.0];
    let y = array![0.0, 0.25, 0.5, 0.1, -0.5, -1.0];

    println!("Sample points:");
    for i in 0..x.len() {
        print!("({:.1}, {:.2}) ", x[i], y[i]);
    }
    println!();

    // Create a cubic spline (with natural boundary conditions)
    let cs = make_interp_spline(&x.view(), &y.view(), "natural", None).unwrap();

    // Evaluate the spline at new points
    let x_new = array![1.0, 2.5, 4.5, 5.0];

    // For CubicSpline, evaluate each point individually
    let mut y_spline = Vec::with_capacity(x_new.len());
    for &x in x_new.iter() {
        y_spline.push(cs.evaluate(x).unwrap());
    }

    // For demonstration purposes, we'll calculate derivatives at each point
    let mut y_derivatives = Vec::with_capacity(x_new.len());
    let mut y_second_derivatives = Vec::with_capacity(x_new.len());

    for &x in x_new.iter() {
        y_derivatives.push(cs.derivative(x).unwrap());

        // For second derivative, we'd need another method or to calculate it numerically
        // This is just a placeholder
        y_second_derivatives.push(0.0);
    }

    println!("\nSpline evaluation:");
    println!("  x   |  Value  | 1st Deriv | 2nd Deriv");
    println!("------|---------|-----------|----------");
    for i in 0..x_new.len() {
        println!(
            " {:.1} |  {:.3}  |   {:.3}   |   {:.3}",
            x_new[i], y_spline[i], y_derivatives[i], y_second_derivatives[i]
        );
    }
}

fn demo_nd_interpolation() {
    println!("\n3. N-dimensional Interpolation");
    println!("----------------------------");

    // 3.1 Regular grid interpolation
    println!("\n3.1 Regular grid interpolation");

    // Create a 2D grid
    let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    let points = vec![x, y];

    // Create values on the grid (z = x^2 + y^2)
    let mut values = ndarray::Array::zeros(ndarray::IxDyn(&[3, 4]));
    for i in 0..3 {
        for j in 0..4 {
            let idx = [i, j];
            values[idx.as_slice()] = (i * i + j * j) as f64;
        }
    }

    println!("Grid values (z = x² + y²):");
    for i in 0..3 {
        for j in 0..4 {
            let idx = [i, j];
            print!("{:.1} ", values[idx.as_slice()]);
        }
        println!();
    }

    // Create the interpolator
    let interp = make_interp_nd(
        points,
        values,
        InterpNDMethod::Linear,
        ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // Test points for interpolation
    let test_points = Array2::from_shape_vec(
        (3, 2),
        vec![
            0.5, 1.5, // Between grid points
            1.5, 2.5, // Between grid points
            1.0, 1.0, // On grid point
        ],
    )
    .unwrap();

    // Perform interpolation
    let results = interp.__call__(&test_points.view()).unwrap();

    println!("\nInterpolation results:");
    for i in 0..test_points.shape()[0] {
        println!(
            "f({:.1}, {:.1}) = {:.2}",
            test_points[[i, 0]],
            test_points[[i, 1]],
            results[i]
        );
    }

    // 3.2 Scattered data interpolation
    println!("\n3.2 Scattered data interpolation");

    // Create scattered points in 2D
    let points = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    .unwrap();

    // Create values at those points (z = x^2 + y^2)
    let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 0.5]);

    println!("Scattered data points (z = x² + y²):");
    for i in 0..points.shape()[0] {
        println!(
            "f({:.1}, {:.1}) = {:.1}",
            points[[i, 0]],
            points[[i, 1]],
            values[i]
        );
    }

    // Create IDW interpolator
    let interp_idw = make_interp_scattered(
        points.clone(),
        values.clone(),
        ScatteredInterpolationMethod::IDW,
        ExtrapolateMode::Extrapolate,
        Some(ScatteredInterpolatorParams::IDW { power: 2.0 }),
    )
    .unwrap();

    // Test points
    let test_points = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.75, 0.75]).unwrap();

    // Perform interpolation
    let results = interp_idw.__call__(&test_points.view()).unwrap();

    println!("\nIDW Interpolation results:");
    for i in 0..test_points.shape()[0] {
        println!(
            "f({:.2}, {:.2}) = {:.3}",
            test_points[[i, 0]],
            test_points[[i, 1]],
            results[i]
        );
    }
}
