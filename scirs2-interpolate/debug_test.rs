use scirs2_core::ndarray::array;
use scirs2_interpolate::{
    bspline::{BSpline, ExtrapolateMode},
    simd_bspline::SimdBSplineEvaluator,
};

fn main() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let coefficients = array![1.0, 2.0, 3.0, 4.0];

    let spline = BSpline::new(
        &knots.view(),
        &coefficients.view(),
        3,
        ExtrapolateMode::Extrapolate,
    ).unwrap();

    // Test regular evaluation
    println!("Regular spline evaluation:");
    for &x in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let result = spline.eval(x).unwrap();
        println!("  spline.eval({}) = {}", x, result);
    }

    // Test SIMD evaluation
    let mut evaluator = SimdBSplineEvaluator::new(spline);
    let points = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let results = evaluator.eval_batch(&points).unwrap();
    
    println!("\nSIMD batch evaluation:");
    for (i, (&x, &result)) in points.iter().zip(results.iter()).enumerate() {
        println!("  results[{}] = eval({}) = {}", i, x, result);
    }
}
