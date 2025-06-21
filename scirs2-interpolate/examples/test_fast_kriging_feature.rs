//! Test Fast Kriging functionality
//!
//! This example demonstrates the Fast Kriging implementation for large-scale datasets.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "linalg")]
    {
        use ndarray::{Array1, Array2};
        use scirs2_interpolate::advanced::fast_kriging::{
            make_fixed_rank_kriging, make_local_kriging, make_tapered_kriging, FastKrigingBuilder,
            FastKrigingMethod,
        };
        use scirs2_interpolate::advanced::kriging::CovarianceFunction;
        println!("Testing Fast Kriging with linalg feature...");

        // Create sample data
        let n_points = 100;
        let n_dims = 2;
        let mut points = Array2::zeros((n_points, n_dims));
        let mut values = Array1::zeros(n_points);

        // Generate a simple test dataset
        for i in 0..n_points {
            let x = (i as f64) / (n_points as f64);
            let y = (i as f64 * 2.0) / (n_points as f64);
            points[[i, 0]] = x;
            points[[i, 1]] = y;
            values[i] = x * x + y * y; // Simple quadratic function
        }

        println!(
            "Created dataset with {} points in {} dimensions",
            n_points, n_dims
        );

        // Test 1: Local Kriging
        println!("\n1. Testing Local Kriging...");
        let local_kriging = make_local_kriging(
            &points.view(),
            &values.view(),
            CovarianceFunction::Matern52,
            1.0, // length_scale
            20,  // max_neighbors
        )?;

        // Test prediction
        let query_points = Array2::from_shape_vec(
            (3, 2),
            vec![
                0.25, 0.25, // Point 1
                0.5, 0.5, // Point 2
                0.75, 0.75, // Point 3
            ],
        )
        .unwrap();

        let local_result = local_kriging.predict(&query_points.view())?;
        println!("Local Kriging predictions: {:?}", local_result.value);
        println!("Local Kriging variances: {:?}", local_result.variance);

        // Test 2: Fixed Rank Kriging
        println!("\n2. Testing Fixed Rank Kriging...");
        let fixed_rank_kriging = make_fixed_rank_kriging(
            &points.view(),
            &values.view(),
            CovarianceFunction::Exponential,
            1.0, // length_scale
            10,  // rank
        )?;

        let fixed_rank_result = fixed_rank_kriging.predict(&query_points.view())?;
        println!("Fixed Rank predictions: {:?}", fixed_rank_result.value);

        // Test 3: Tapered Kriging
        println!("\n3. Testing Tapered Kriging...");
        let tapered_kriging = make_tapered_kriging(
            &points.view(),
            &values.view(),
            CovarianceFunction::SquaredExponential,
            1.0, // length_scale
            2.0, // taper_range
        )?;

        let tapered_result = tapered_kriging.predict(&query_points.view())?;
        println!("Tapered predictions: {:?}", tapered_result.value);

        // Test 4: Builder Pattern
        println!("\n4. Testing Builder Pattern...");
        let builder_kriging = FastKrigingBuilder::<f64>::new()
            .points(points.clone())
            .values(values.clone())
            .covariance_function(CovarianceFunction::Matern32)
            .approximation_method(FastKrigingMethod::Local)
            .max_neighbors(30)
            .build()?;

        let builder_result = builder_kriging.predict(&query_points.view())?;
        println!("Builder pattern predictions: {:?}", builder_result.value);

        // Test 5: Different approximation methods comparison
        println!("\n5. Comparing approximation methods...");
        let methods = vec![
            ("Local", FastKrigingMethod::Local),
            ("FixedRank(15)", FastKrigingMethod::FixedRank(15)),
            ("Tapering(1.5)", FastKrigingMethod::Tapering(1.5)),
            ("HODLR(32)", FastKrigingMethod::HODLR(32)),
        ];

        for (name, method) in methods {
            let kriging = FastKrigingBuilder::<f64>::new()
                .points(points.clone())
                .values(values.clone())
                .covariance_function(CovarianceFunction::Matern52)
                .approximation_method(method)
                .build()?;

            let result = kriging.predict(&query_points.view())?;
            println!(
                "{}: [{:.4}, {:.4}, {:.4}]",
                name, result.value[0], result.value[1], result.value[2]
            );
        }

        // Test 6: Simple confidence intervals (manual calculation)
        println!("\n6. Simple confidence intervals (95%):");
        let z_score = 1.96; // 95% confidence
        for i in 0..local_result.value.len() {
            let std_dev = local_result.variance[i].sqrt();
            let lower = local_result.value[i] - std_dev * z_score;
            let upper = local_result.value[i] + std_dev * z_score;
            println!("  Point {}: [{:.4}, {:.4}]", i + 1, lower, upper);
        }

        println!("\n✅ Fast Kriging tests completed successfully!");
    }

    #[cfg(not(feature = "linalg"))]
    {
        println!("Fast Kriging requires linalg feature. Skipping tests.");
        println!("Run with: cargo run --example test_fast_kriging_feature --features linalg");
    }

    Ok(())
}
