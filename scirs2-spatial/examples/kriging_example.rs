//! Kriging Interpolation Example
//!
//! This example demonstrates Kriging (Gaussian process regression) interpolation,
//! which is widely used in geostatistics for spatial prediction. Kriging provides
//! the Best Linear Unbiased Estimator (BLUE) and includes uncertainty quantification.

use ndarray::{array, Array1, Array2};
use scirs2_spatial::kriging::{OrdinaryKriging, SimpleKriging, VariogramModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Kriging Interpolation Example ===\n");

    // Example 1: Basic Ordinary Kriging
    println!("1. Basic Ordinary Kriging");
    basic_ordinary_kriging_example()?;
    println!();

    // Example 2: Variogram model comparison
    println!("2. Variogram Model Comparison");
    variogram_comparison_example()?;
    println!();

    // Example 3: Simple Kriging
    println!("3. Simple Kriging with Known Mean");
    simple_kriging_example()?;
    println!();

    // Example 4: Batch prediction
    println!("4. Batch Prediction");
    batch_prediction_example()?;
    println!();

    // Example 5: Cross-validation
    println!("5. Cross-validation Assessment");
    cross_validation_example()?;
    println!();

    // Example 6: 1D Kriging
    println!("6. 1D Kriging Example");
    kriging_1d_example()?;
    println!();

    // Example 7: Uncertainty quantification
    println!("7. Uncertainty Quantification");
    uncertainty_example()?;

    Ok(())
}

fn basic_ordinary_kriging_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 2D spatial dataset
    let points = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0]
    ];

    // Simulated temperature data with spatial correlation
    let values = array![20.0, 22.5, 25.0, 18.5, 21.0, 23.5, 17.0, 19.5, 22.0];

    println!("Data points (x, y, temperature):");
    for i in 0..points.nrows() {
        println!(
            "  ({:.1}, {:.1}) = {:.1}°C",
            points[[i, 0]],
            points[[i, 1]],
            values[i]
        );
    }

    // Create spherical variogram model
    let variogram = VariogramModel::spherical(1.5, 4.0, 0.5);
    println!("Using spherical variogram: range=1.5, sill=4.0, nugget=0.5");

    // Create and fit Kriging model
    let mut kriging = OrdinaryKriging::new(&points.view(), &values.view(), variogram)?;
    kriging.fit()?;

    // Predict at several locations
    let test_locations = vec![
        [0.5, 0.5],
        [1.5, 1.5],
        [0.25, 1.75],
        [2.5, 1.0], // Outside convex hull
    ];

    println!("Predictions:");
    for location in test_locations {
        let prediction = kriging.predict(&location)?;
        println!(
            "  At ({:.2}, {:.2}): {:.2}°C ± {:.2}°C (std dev)",
            location[0],
            location[1],
            prediction.value,
            prediction.variance.sqrt()
        );
    }

    Ok(())
}

fn variogram_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    // Same dataset as before
    let points = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [2.0, 1.0]
    ];
    let values = array![10.0, 12.0, 11.0, 13.0, 14.0, 15.0];

    let test_point = [0.5, 0.5];

    // Different variogram models
    let variograms = vec![
        ("Spherical", VariogramModel::spherical(1.5, 2.0, 0.3)),
        ("Exponential", VariogramModel::exponential(1.0, 2.0, 0.3)),
        ("Gaussian", VariogramModel::gaussian(1.0, 2.0, 0.3)),
        ("Linear", VariogramModel::linear(1.0, 0.3)),
    ];

    println!(
        "Comparing variogram models at point ({:.1}, {:.1}):",
        test_point[0], test_point[1]
    );

    for (name, variogram) in variograms {
        let kriging = OrdinaryKriging::new(&points.view(), &values.view(), variogram)?;
        let prediction = kriging.predict(&test_point)?;

        println!(
            "  {}: {:.3} ± {:.3}",
            name,
            prediction.value,
            prediction.variance.sqrt()
        );
    }

    Ok(())
}

fn simple_kriging_example() -> Result<(), Box<dyn std::error::Error>> {
    // Dataset with known global mean
    let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];

    // Data with anomalies around known mean of 100
    let values = array![98.5, 101.2, 99.8, 102.1, 100.3];
    let known_mean = 100.0;

    println!("Simple Kriging with known mean = {:.1}", known_mean);
    println!("Data values: {:?}", values);

    let variogram = VariogramModel::exponential(0.8, 1.5, 0.2);
    let kriging = SimpleKriging::new(&points.view(), &values.view(), known_mean, variogram)?;

    let test_locations = vec![[0.25, 0.25], [0.75, 0.75], [1.5, 0.5]];

    println!("Simple Kriging predictions:");
    for location in test_locations {
        let prediction = kriging.predict(&location)?;
        println!(
            "  At ({:.2}, {:.2}): {:.3} ± {:.3}",
            location[0],
            location[1],
            prediction.value,
            prediction.variance.sqrt()
        );
    }

    Ok(())
}

fn batch_prediction_example() -> Result<(), Box<dyn std::error::Error>> {
    // Regular grid of data points
    let points = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0]
    ];

    // Elevation data
    let values = array![100.0, 110.0, 125.0, 105.0, 120.0, 135.0, 115.0, 130.0, 150.0];

    let variogram = VariogramModel::spherical(2.0, 400.0, 25.0);
    let mut kriging = OrdinaryKriging::new(&points.view(), &values.view(), variogram)?;
    kriging.fit()?;

    // Create grid of prediction locations
    let mut prediction_points = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            let x = i as f64 * 0.5;
            let y = j as f64 * 0.5;
            prediction_points.push([x, y]);
        }
    }

    let prediction_array = Array2::from_shape_fn((25, 2), |(i, j)| {
        let row_idx = i / 5;
        let col_idx = i % 5;
        if j == 0 {
            row_idx as f64 * 0.5
        } else {
            col_idx as f64 * 0.5
        }
    });

    println!("Batch prediction on 5x5 grid (25 points):");
    let predictions = kriging.predict_batch(&prediction_array.view())?;

    println!("Grid of predicted elevations (rows: y=0 to y=2, cols: x=0 to x=2):");
    for i in 0..5 {
        print!("  ");
        for j in 0..5 {
            let idx = i * 5 + j;
            print!("{:6.1} ", predictions[idx].value);
        }
        println!();
    }

    // Calculate prediction statistics
    let values: Vec<f64> = predictions.iter().map(|p| p.value).collect();
    let variances: Vec<f64> = predictions.iter().map(|p| p.variance).collect();

    let mean_prediction = values.iter().sum::<f64>() / values.len() as f64;
    let mean_variance = variances.iter().sum::<f64>() / variances.len() as f64;

    println!("Statistics:");
    println!("  Mean predicted value: {:.2}", mean_prediction);
    println!("  Mean prediction variance: {:.2}", mean_variance);
    println!("  Mean prediction std dev: {:.2}", mean_variance.sqrt());

    Ok(())
}

fn cross_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create more substantial dataset for cross-validation
    let points = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [3.0, 2.0],
        [0.5, 0.5],
        [1.5, 1.5],
        [2.5, 0.5]
    ];

    // Synthetic data with spatial trend
    let values =
        array![5.0, 7.0, 9.0, 11.0, 6.0, 8.0, 10.0, 12.0, 7.0, 9.0, 11.0, 13.0, 6.5, 9.5, 10.5];

    println!("Cross-validation with {} data points", points.nrows());

    // Test different variogram models
    let variograms = vec![
        (
            "Spherical (range=2.0)",
            VariogramModel::spherical(2.0, 4.0, 0.5),
        ),
        (
            "Spherical (range=3.0)",
            VariogramModel::spherical(3.0, 4.0, 0.5),
        ),
        ("Exponential", VariogramModel::exponential(2.0, 4.0, 0.5)),
        ("Gaussian", VariogramModel::gaussian(1.5, 4.0, 0.5)),
    ];

    for (name, variogram) in variograms {
        let kriging = OrdinaryKriging::new(&points.view(), &values.view(), variogram)?;
        let errors = kriging.cross_validate()?;

        let mean_error = errors.sum() / errors.len() as f64;
        let mse = errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64;
        let rmse = mse.sqrt();

        println!("  {}:", name);
        println!("    Mean error: {:.4}", mean_error);
        println!("    RMSE: {:.4}", rmse);
        println!(
            "    Max absolute error: {:.4}",
            errors.iter().map(|e| e.abs()).fold(0.0f64, f64::max)
        );
    }

    Ok(())
}

fn kriging_1d_example() -> Result<(), Box<dyn std::error::Error>> {
    // 1D example: measurements along a transect
    let points = array![[0.0], [1.0], [2.0], [4.0], [6.0], [8.0], [10.0]];

    // Measured values along the transect
    let values = array![2.1, 3.8, 4.2, 5.9, 4.8, 3.1, 1.7];

    println!("1D Kriging along a transect:");
    println!("Data points (distance, value):");
    for i in 0..points.nrows() {
        println!("  {:.1} km: {:.1}", points[[i, 0]], values[i]);
    }

    let variogram = VariogramModel::spherical(3.5, 2.0, 0.3);
    let kriging = OrdinaryKriging::new(&points.view(), &values.view(), variogram)?;

    // Predict at intermediate locations
    let test_points = vec![0.5, 1.5, 3.0, 5.0, 7.0, 9.0];

    println!("Predictions at intermediate locations:");
    for &x in &test_points {
        let prediction = kriging.predict(&[x])?;
        println!(
            "  {:.1} km: {:.2} ± {:.2}",
            x,
            prediction.value,
            prediction.variance.sqrt()
        );
    }

    Ok(())
}

fn uncertainty_example() -> Result<(), Box<dyn std::error::Error>> {
    // Sparse data to highlight uncertainty
    let points = array![[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0], [2.5, 2.5]];

    let values = array![10.0, 15.0, 12.0, 18.0, 14.0];

    let variogram = VariogramModel::spherical(3.0, 8.0, 1.0);
    let kriging = OrdinaryKriging::new(&points.view(), &values.view(), variogram)?;

    println!("Uncertainty quantification with sparse data:");
    println!("Data locations show low uncertainty, interpolated areas show higher uncertainty");

    // Test different locations - some near data, some far
    let test_cases = vec![
        ([0.1, 0.1], "Very close to data point"),
        ([2.5, 2.6], "Very close to central data point"),
        ([1.0, 1.0], "Moderate distance from data"),
        ([3.5, 1.0], "Between data points"),
        ([6.0, 6.0], "Far from all data points"),
        ([2.5, 5.5], "At edge of data region"),
    ];

    for (location, description) in test_cases {
        let prediction = kriging.predict(&location)?;
        let std_dev = prediction.variance.sqrt();
        let confidence_95 = 1.96 * std_dev; // Approximate 95% confidence interval

        println!(
            "  {} at ({:.1}, {:.1}):",
            description, location[0], location[1]
        );
        println!("    Prediction: {:.2} ± {:.2}", prediction.value, std_dev);
        println!(
            "    95% CI: [{:.2}, {:.2}]",
            prediction.value - confidence_95,
            prediction.value + confidence_95
        );

        // Classify uncertainty level
        let uncertainty_level = if std_dev < 1.0 {
            "Low"
        } else if std_dev < 2.0 {
            "Medium"
        } else {
            "High"
        };
        println!("    Uncertainty level: {}", uncertainty_level);
        println!();
    }

    Ok(())
}

/// Helper function to create synthetic spatial data
#[allow(dead_code)]
fn create_synthetic_data(n_points: usize, noise_level: f64) -> (Array2<f64>, Array1<f64>) {
    use rand::Rng;
    let mut rng = rand::rng();

    let mut points = Array2::zeros((n_points, 2));
    let mut values = Array1::zeros(n_points);

    for i in 0..n_points {
        let x = rng.random_range(0.0..10.0);
        let y = rng.random_range(0.0..10.0);

        points[[i, 0]] = x;
        points[[i, 1]] = y;

        // Synthetic function with spatial correlation
        let true_value =
            10.0 + 2.0 * x + 0.5 * y + 3.0 * (0.5 * x as f64).sin() * (0.3 * y as f64).cos();
        let noise = rng.random_range(-noise_level..noise_level);
        values[i] = true_value + noise;
    }

    (points, values)
}

/// Display variogram characteristics
#[allow(dead_code)]
fn display_variogram_info(variogram: &VariogramModel) {
    println!("Variogram characteristics:");
    println!("  Type: {:?}", variogram);
    println!("  Effective range: {:.2}", variogram.effective_range());
    println!("  Sill: {:.2}", variogram.sill());
    println!("  Nugget: {:.2}", variogram.nugget());

    // Sample variogram values at different distances
    println!("  Values at distances:");
    let distances = [0.0, 0.5, 1.0, 2.0, 5.0];
    for &dist in &distances {
        println!(
            "    h = {:.1}: γ(h) = {:.3}",
            dist,
            variogram.evaluate(dist)
        );
    }
}
