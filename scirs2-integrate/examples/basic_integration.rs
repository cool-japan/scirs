use scirs2_integrate::quad;

fn main() {
    println!("Basic numerical integration examples");

    // Example 1: Integrate x^2 from 0 to 1
    // Exact result: 1/3
    let f1 = |x: f64| x * x;
    let result1 = quad(f1, 0.0, 1.0, None).unwrap();
    println!("∫x² dx from 0 to 1:");
    println!("  Calculated: {}", result1.value);
    println!("  Exact:      {}", 1.0 / 3.0);
    println!("  Error:      {}", result1.abs_error);
    println!("  Evaluations: {}", result1.n_evals);
    println!("  Converged:  {}", result1.converged);
    println!();

    // Example 2: Integrate sin(x) from 0 to π
    // Exact result: 2
    let f2 = |x: f64| x.sin();
    let pi = std::f64::consts::PI;
    let result2 = quad(f2, 0.0, pi, None).unwrap();
    println!("∫sin(x) dx from 0 to π:");
    println!("  Calculated: {}", result2.value);
    println!("  Exact:      {}", 2.0);
    println!("  Error:      {}", result2.abs_error);
    println!("  Evaluations: {}", result2.n_evals);
    println!("  Converged:  {}", result2.converged);
}
