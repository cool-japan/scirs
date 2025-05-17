use scirs2_optimize::{minimize, Method, Options, Bounds};
use ndarray::{Array1, ArrayView1};

fn main() {
    // Test function: quadratic with minimum at (2, 3)
    let quadratic = |x: &ArrayView1<f64>| -> f64 { 
        (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) 
    };

    let x0 = Array1::from_vec(vec![0.0, 0.0]);
    
    // Test without bounds
    let options_no_bounds = Options::default();
    let result_no_bounds = minimize(quadratic, x0.clone(), Method::ConjugateGradient, options_no_bounds).unwrap();
    println!("Without bounds: x = {:?}, success = {}", result_no_bounds.x, result_no_bounds.success);
    
    // Test with bounds [0, 1] x [0, 1]  
    let mut options_with_bounds = Options::default();
    let bounds = Bounds::new(&[(Some(0.0), Some(1.0)), (Some(0.0), Some(1.0))]);
    options_with_bounds.bounds = Some(bounds);
    
    let result_with_bounds = minimize(quadratic, x0.clone(), Method::ConjugateGradient, options_with_bounds).unwrap();
    println!("With bounds [0,1]x[0,1]: x = {:?}, success = {}", result_with_bounds.x, result_with_bounds.success);
    
    // Expected: with bounds should be [1.0, 1.0] since that's closest to (2,3) within bounds
}