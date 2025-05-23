use scirs2_integrate::tanhsinh::{tanhsinh, TanhSinhOptions};

fn main() {
    // Test 1: Simple polynomial x^2 from 0 to 1
    println!("Test 1: Integrate x^2 from 0 to 1");
    let result = tanhsinh(|x| x * x, 0.0, 1.0, None).unwrap();
    println!("Result: {}", result.integral);
    println!("Expected: {}", 1.0/3.0);
    println!("Error: {}", result.error);
    println!("Actual error: {}", (result.integral - 1.0/3.0).abs());
    println!("Success: {}", result.success);
    println!("Max level: {}", result.max_level);
    println!("Function evaluations: {}", result.nfev);
    println!();

    // Test 2: With custom options
    println!("Test 2: With custom options");
    let options = TanhSinhOptions {
        atol: 1e-10,
        rtol: 1e-10,
        max_level: 10,
        min_level: 3,
        ..Default::default()
    };
    let result = tanhsinh(|x| x * x, 0.0, 1.0, Some(options)).unwrap();
    println!("Result: {}", result.integral);
    println!("Expected: {}", 1.0/3.0);
    println!("Error: {}", result.error);
    println!("Actual error: {}", (result.integral - 1.0/3.0).abs());
    println!("Success: {}", result.success);
    println!("Max level: {}", result.max_level);
    println!("Function evaluations: {}", result.nfev);
    println!();

    // Test 3: Harder function - exp(x)
    println!("Test 3: Integrate exp(x) from 0 to 1");
    let result = tanhsinh(|x| x.exp(), 0.0, 1.0, None).unwrap();
    let expected = 1_f64.exp() - 1.0;
    println!("Result: {}", result.integral);
    println!("Expected: {}", expected);
    println!("Error: {}", result.error);
    println!("Actual error: {}", (result.integral - expected).abs());
    println!("Success: {}", result.success);
    println!();

    // Test 4: Sine function
    println!("Test 4: Integrate sin(x) from 0 to π");
    let result = tanhsinh(|x| x.sin(), 0.0, std::f64::consts::PI, None).unwrap();
    let expected = 2.0;
    println!("Result: {}", result.integral);
    println!("Expected: {}", expected);
    println!("Error: {}", result.error);
    println!("Actual error: {}", (result.integral - expected).abs());
    println!("Success: {}", result.success);
}