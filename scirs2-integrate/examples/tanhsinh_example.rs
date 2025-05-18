use scirs2_integrate::tanhsinh::{nsum, tanhsinh, TanhSinhOptions};
use std::f64::consts::PI;

fn main() {
    println!("Tanh-Sinh Quadrature Examples");
    println!("=============================\n");

    // Example 1: Basic integral
    println!("Example 1: Integrate x^2 from 0 to 1");
    println!("Exact result: 1/3 = 0.3333...");

    let result = tanhsinh(|x| x * x, 0.0, 1.0, None).unwrap();

    println!("Tanh-sinh result: {:.10}", result.integral);
    println!("Error estimate: {:.10e}", result.error);
    println!("Actual error: {:.10e}", (result.integral - 1.0 / 3.0).abs());
    println!("Function evaluations: {}", result.nfev);
    println!("Maximum level: {}", result.max_level);
    println!("Success: {}\n", result.success);

    // Example 2: Trigonometric function
    println!("Example 2: Integrate sin(x) from 0 to pi");
    println!("Exact result: 2.0");

    let result = tanhsinh(|x| x.sin(), 0.0, PI, None).unwrap();

    println!("Tanh-sinh result: {:.10}", result.integral);
    println!("Error estimate: {:.10e}", result.error);
    println!("Actual error: {:.10e}", (result.integral - 2.0).abs());
    println!("Function evaluations: {}", result.nfev);
    println!("Maximum level: {}", result.max_level);
    println!("Success: {}\n", result.success);

    // Example 3: Integral with endpoint singularity
    println!("Example 3: Integrate 1/sqrt(x) from 0 to 1");
    println!("Exact result: 2.0");

    let result = tanhsinh(|x| 1.0 / x.sqrt(), 0.0, 1.0, None).unwrap();

    println!("Tanh-sinh result: {:.10}", result.integral);
    println!("Error estimate: {:.10e}", result.error);
    println!("Actual error: {:.10e}", (result.integral - 2.0).abs());
    println!("Function evaluations: {}", result.nfev);
    println!("Maximum level: {}", result.max_level);
    println!("Success: {}\n", result.success);

    // Example 4: Semi-infinite range
    println!("Example 4: Integrate e^(-x) from 0 to infinity");
    println!("Exact result: 1.0");

    let result = tanhsinh(|x| (-x).exp(), 0.0, f64::INFINITY, None).unwrap();

    println!("Tanh-sinh result: {:.10}", result.integral);
    println!("Error estimate: {:.10e}", result.error);
    println!("Actual error: {:.10e}", (result.integral - 1.0).abs());
    println!("Function evaluations: {}", result.nfev);
    println!("Maximum level: {}", result.max_level);
    println!("Success: {}\n", result.success);

    // Example 5: Double-infinite range
    println!("Example 5: Integrate e^(-x^2) from -infinity to infinity");
    println!("Exact result: sqrt(pi) = {:.10}", PI.sqrt());

    let result = tanhsinh(|x| (-x * x).exp(), f64::NEG_INFINITY, f64::INFINITY, None).unwrap();

    println!("Tanh-sinh result: {:.10}", result.integral);
    println!("Error estimate: {:.10e}", result.error);
    println!("Actual error: {:.10e}", (result.integral - PI.sqrt()).abs());
    println!("Function evaluations: {}", result.nfev);
    println!("Maximum level: {}", result.max_level);
    println!("Success: {}\n", result.success);

    // Example 6: High-precision requirements
    println!("Example 6: Higher precision for sin(x) from 0 to pi");
    println!("Exact result: 2.0");

    let options = TanhSinhOptions {
        atol: 0.0,
        rtol: 1e-14,
        max_level: 12,
        ..Default::default()
    };

    let result = tanhsinh(|x| x.sin(), 0.0, PI, Some(options)).unwrap();

    println!("Tanh-sinh result: {:.16}", result.integral);
    println!("Error estimate: {:.10e}", result.error);
    println!("Actual error: {:.16e}", (result.integral - 2.0).abs());
    println!("Function evaluations: {}", result.nfev);
    println!("Maximum level: {}", result.max_level);
    println!("Success: {}\n", result.success);

    // Example 7: Log-space integration for numerical stability
    println!("Example 7: Log-space integration");
    println!("Integrate e^(-1000*x^2) from -1 to 1");
    println!("Exact result: sqrt(pi/1000) = {:.10}", (PI / 1000.0).sqrt());

    let options = TanhSinhOptions {
        log: true,
        ..Default::default()
    };

    let result = tanhsinh(|x| -1000.0 * x * x, -1.0, 1.0, Some(options)).unwrap();

    println!("Tanh-sinh result (log): {:.10}", result.integral);
    println!("Tanh-sinh result (exp): {:.10}", result.integral.exp());
    println!("Error estimate (log): {:.10e}", result.error);
    println!(
        "Actual error: {:.10e}",
        (result.integral.exp() - (PI / 1000.0).sqrt()).abs()
    );
    println!("Function evaluations: {}", result.nfev);
    println!("Maximum level: {}", result.max_level);
    println!("Success: {}\n", result.success);

    // Example 8: Finite sum using nsum
    println!("Example 8: Sum of first 100 integers");
    println!("Exact result: 5050");

    let result = nsum(|n| n, 1.0, 100.0, 1.0, None, None).unwrap();

    println!("nsum result: {:.10}", result.integral);
    println!("Error estimate: {:.10e}", result.error);
    println!("Actual error: {:.10e}", (result.integral - 5050.0).abs());
    println!("Function evaluations: {}", result.nfev);
    println!("Success: {}\n", result.success);

    // Example 9: Infinite sum using nsum (Riemann zeta function at s=2)
    println!("Example 9: Sum of 1/n^2 from n=1 to infinity");
    println!("Exact result: pi^2/6 = {:.10}", PI * PI / 6.0);

    let result = nsum(|n| 1.0 / (n * n), 1.0, f64::INFINITY, 1.0, None, None).unwrap();

    println!("nsum result: {:.10}", result.integral);
    println!("Error estimate: {:.10e}", result.error);
    println!(
        "Actual error: {:.10e}",
        (result.integral - PI * PI / 6.0).abs()
    );
    println!("Function evaluations: {}", result.nfev);
    println!("Maximum level: {}", result.max_level);
    println!("Success: {}\n", result.success);

    // Example 10: Alternating sum using nsum
    println!("Example 10: Sum of (-1)^(n+1)/n from n=1 to infinity (log(2))");
    println!("Exact result: log(2) = {:.10}", 2.0_f64.ln());

    let result = nsum(
        |n| (-1.0_f64).powf(n + 1.0) / n,
        1.0,
        f64::INFINITY,
        1.0,
        None,
        None,
    )
    .unwrap();

    println!("nsum result: {:.10}", result.integral);
    println!("Error estimate: {:.10e}", result.error);
    println!(
        "Actual error: {:.10e}",
        (result.integral - 2.0_f64.ln()).abs()
    );
    println!("Function evaluations: {}", result.nfev);
    println!("Maximum level: {}", result.max_level);
    println!("Success: {}", result.success);
}
