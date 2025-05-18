use ndarray::ArrayView1;
use scirs2_optimize::global::{multi_start, MultiStartOptions, StartingPointStrategy};
use scirs2_optimize::unconstrained::{minimize, Bounds, Method, Options};

fn main() {
    // Simple sphere function
    let sphere = |x: &ArrayView1<f64>| x.iter().map(|&xi| xi.powi(2)).sum::<f64>();

    // First test direct optimization
    println!("Testing direct optimization:");
    let x0 = vec![1.0, 1.0];
    let result_direct = minimize(sphere, &x0, Method::LBFGS, None).unwrap();
    println!(
        "Direct LBFGS result: x = {:?}, f(x) = {}",
        result_direct.x, result_direct.fun
    );

    // Try with BFGS
    let result_bfgs = minimize(sphere, &x0, Method::BFGS, None).unwrap();
    println!(
        "Direct BFGS result: x = {:?}, f(x) = {}",
        result_bfgs.x, result_bfgs.fun
    );

    // Try with NelderMead
    let result_nm = minimize(sphere, &x0, Method::NelderMead, None).unwrap();
    println!(
        "Direct NelderMead result: x = {:?}, f(x) = {}",
        result_nm.x, result_nm.fun
    );

    // Test with bounds
    println!("\nTesting optimization with bounds:");
    let mut options = Options::default();
    let bounds =
        Bounds::from_vecs(vec![Some(-5.0), Some(-5.0)], vec![Some(5.0), Some(5.0)]).unwrap();
    options.bounds = Some(bounds);
    let result_bounded = minimize(sphere, &x0, Method::LBFGS, Some(options)).unwrap();
    println!(
        "Bounded result: x = {:?}, f(x) = {}",
        result_bounded.x, result_bounded.fun
    );

    // Now test multi-start
    println!("\nTesting multi-start:");
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

    let options = MultiStartOptions {
        n_starts: 5,
        parallel: false,
        seed: Some(42),
        strategy: StartingPointStrategy::Random,
        ..Default::default()
    };

    let result = multi_start(sphere, bounds.clone(), Some(options)).unwrap();

    println!("Success: {}", result.success);
    println!("Result x: {:?}", result.x);
    println!("Result fun: {}", result.fun);
    println!("Iterations: {}", result.iterations);
    println!("Message: {}", result.message);

    // Test more starts
    println!("\nTesting with more starts:");
    let options2 = MultiStartOptions {
        n_starts: 20,
        parallel: false,
        seed: Some(42),
        strategy: StartingPointStrategy::Random,
        ..Default::default()
    };

    let result2 = multi_start(sphere, bounds.clone(), Some(options2)).unwrap();
    println!("Result x: {:?}", result2.x);
    println!("Result fun: {}", result2.fun);
}
