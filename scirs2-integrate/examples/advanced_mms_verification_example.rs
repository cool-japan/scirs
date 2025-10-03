//! Advanced Method of Manufactured Solutions (MMS) verification example
//!
//! This example demonstrates the enhanced MMS toolkit with new solution types,
//! 3D support, and automated verification workflows.

use scirs2_core::ndarray::Array1;
use scirs2_integrate::{
    ode::{solve_ivp, ODEMethod, ODEOptions},
    verification::{
        combined_solution, exponential_solution, polynomial_solution, trigonometric_solution_2d,
        trigonometric_solution_3d, ConvergenceAnalysis, MMSODEProblem, MMSPDEProblem,
        VerificationTestCase, VerificationWorkflow,
    },
};

use scirs2_core::ndarray::ArrayView1;
use scirs2_integrate::IntegrateError;
use std::f64::consts::PI;

type SolverFunction = Box<dyn Fn(&[f64]) -> Result<f64, IntegrateError>>;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Method of Manufactured Solutions Verification ===\n");

    // Example 1: Enhanced ODE verification with exponential solutions
    exponential_ode_verification()?;

    // Example 2: Combined solution verification
    combined_solution_verification()?;

    // Example 3: 3D PDE verification
    three_d_pde_verification()?;

    // Example 4: Helmholtz equation verification
    helmholtz_verification()?;

    // Example 5: Automated verification workflow
    automated_workflow_example()?;

    Ok(())
}

#[allow(dead_code)]
fn exponential_ode_verification() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Exponential ODE Verification");
    println!("{}", "=".repeat(50));

    // Create manufactured problem with exponential exact solution: y(t) = 2*exp(-3t)
    let exact_solution = exponential_solution(2.0, -3.0);
    let problem = MMSODEProblem::new(exact_solution, [0.0, 1.0]);

    println!("Exact solution: y(t) = 2*exp(-3t)");
    println!("Manufactured ODE: y'(t) = -6*exp(-3t) (derivative of exact solution)");

    // Test with different step sizes for implicit method
    let step_sizes = vec![0.1, 0.05, 0.025, 0.0125];
    let mut errors = Vec::new();

    println!(
        "\nStep Size   Final Error   Expected: y(1) = 2*exp(-3) ≈ {:.6}",
        2.0 * (-3.0_f64).exp()
    );
    println!("{}", "─".repeat(50));

    for &h in &step_sizes {
        // Solve the manufactured ODE with implicit method
        let manufactured_rhs = |t: f64, _y: scirs2_core::ndarray::ArrayView1<f64>| {
            Array1::from_vec(vec![problem.source_term(t)])
        };

        let options = ODEOptions {
            method: ODEMethod::Radau, // Good for stiff exponential decay
            rtol: 1e-12,
            atol: 1e-12,
            max_step: Some(h),
            ..Default::default()
        };

        let result = solve_ivp(
            manufactured_rhs,
            problem.time_span(),
            Array1::from_vec(vec![problem.initial_condition()]),
            Some(options),
        )?;

        let numerical_final = result.y.last().unwrap()[0];
        let exact_final = problem.exact_at(1.0);
        let error = (numerical_final - exact_final).abs();

        println!("{h:8.4}   {error:11.2e}   Numerical: {numerical_final:.6}");
        errors.push(error);
    }

    // Analyze convergence
    if let Ok(analysis) = ConvergenceAnalysis::compute_order(step_sizes, errors) {
        println!("\n📊 Convergence Analysis:");
        println!("Estimated order of accuracy: {:.2}", analysis.order);

        if analysis.verify_order(5.0, 1.0) {
            println!("✅ High-order accuracy confirmed (Radau method)");
        } else {
            println!(
                "⚠️  Order: {:.2} (Radau typically achieves 5th order)",
                analysis.order
            );
        }
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn combined_solution_verification() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 Combined Solution Verification");
    println!("{}", "=".repeat(50));

    // Create combined solution: polynomial + exponential
    let poly = polynomial_solution(vec![1.0, 0.5, 0.1]); // 1 + 0.5t + 0.1t²
    let exp = exponential_solution(0.5, -1.0); // 0.5*exp(-t)

    let combined = combined_solution(1)
        .with_polynomial(poly)
        .with_exponential(exp);

    let problem = MMSODEProblem::new(combined, [0.0, 2.0]);

    println!("Exact solution: y(t) = 1 + 0.5t + 0.1t² + 0.5*exp(-t)");
    println!("This combines polynomial growth with exponential decay");

    // Test accuracy at several points
    let test_points = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    println!("\nTime    Exact Value   Derivative");
    println!("{}", "─".repeat(35));

    for &t in &test_points {
        let exact_val = problem.exact_at(t);
        let derivative = problem.source_term(t);
        println!("{t:4.1}    {exact_val:10.6}   {derivative:10.6}");
    }

    println!("\n✅ Combined solutions enable verification of methods on diverse function types");
    println!();
    Ok(())
}

#[allow(dead_code)]
fn three_d_pde_verification() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 3D PDE Verification");
    println!("{}", "=".repeat(50));

    // Create 3D Poisson problem: -∇²u = f
    // Exact solution: u(x,y,z) = sin(πx) * cos(πy) * sin(πz)
    let exact_solution = trigonometric_solution_3d(PI, PI, PI);
    let problem = MMSPDEProblem::new_poisson_3d(exact_solution, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);

    println!("Exact solution: u(x,y,z) = sin(πx) * cos(πy) * sin(πz)");
    println!("3D Poisson equation: -∇²u = 3π²sin(πx)cos(πy)sin(πz)");

    // Test different grid sizes
    let grid_sizes = vec![0.2, 0.1, 0.05];

    println!("\nGrid Size   Source Term at (0.5,0,0.5)");
    println!("{}", "─".repeat(35));

    for &h in &grid_sizes {
        let coords = [0.5, 0.0, 0.5];
        let source = problem.source_term(&coords);
        let exact_val = problem.exact_at_3d(coords[0], coords[1], coords[2]);

        println!("{h:8.2}   {source:15.6}");

        // Verify source term: should be 3π²u
        let expected_source = 3.0 * PI * PI * exact_val;
        let relative_error = (source - expected_source).abs() / expected_source.abs();
        assert!(relative_error < 1e-10, "Source term verification failed");
    }

    println!("\n📊 3D verification allows testing finite element, finite difference,");
    println!("    and spectral methods on realistic 3D domains");
    println!("✅ Source term computation verified for 3D Poisson equation");
    println!();
    Ok(())
}

#[allow(dead_code)]
fn helmholtz_verification() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Helmholtz Equation Verification");
    println!("{}", "=".repeat(50));

    // Create 2D Helmholtz problem: ∇²u + k²u = f
    let exact_solution = trigonometric_solution_2d(2.0 * PI, PI);
    let k = 3.0; // Wave number
    let problem = MMSPDEProblem::new_helmholtz_2d(exact_solution, [0.0, 1.0], [0.0, 1.0], k);

    println!("Exact solution: u(x,y) = sin(2πx) * cos(πy)");
    println!("Helmholtz equation: ∇²u + {k}²u = f");
    println!("Applications: acoustics, electromagnetics, quantum mechanics");

    // Calculate theoretical source term
    // For u = sin(2πx)cos(πy):
    // ∇²u = -(2π)²sin(2πx)cos(πy) - π²sin(2πx)cos(πy) = -(4π² + π²)u = -5π²u
    // So f = ∇²u + k²u = -5π²u + k²u = (k² - 5π²)u

    let test_points = vec![
        [0.25, 0.0], // cos(πy) = 1
        [0.5, 0.0],  // sin(2πx) = 0
        [0.25, 0.5], // cos(πy) = 0
    ];

    println!("\nPoint (x,y)    Exact u    Source f    Theoretical f");
    println!("{}", "─".repeat(50));

    for coords in &test_points {
        let u_exact = problem.exact_at(coords[0], coords[1]);
        let f_computed = problem.source_term(coords);
        let f_theoretical = (k * k - 5.0 * PI * PI) * u_exact;

        println!(
            "({:4.2},{:4.1})   {:8.4}   {:9.4}   {:11.4}",
            coords[0], coords[1], u_exact, f_computed, f_theoretical
        );

        // Verify computation
        if u_exact.abs() > 1e-10 {
            let relative_error = (f_computed - f_theoretical).abs() / f_theoretical.abs();
            assert!(
                relative_error < 1e-10,
                "Helmholtz source term verification failed"
            );
        }
    }

    println!("\n✅ Helmholtz equation verification successful");
    println!("   This enables testing wave propagation and scattering solvers");
    println!();
    Ok(())
}

#[allow(dead_code)]
fn automated_workflow_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Automated Verification Workflow");
    println!("{}", "=".repeat(50));

    let mut workflow = VerificationWorkflow::new();

    // Add test cases for different orders
    let test_cases = vec![
        VerificationTestCase {
            name: "First-order Forward Euler".to_string(),
            expected_order: 1.0,
            order_tolerance: 0.2,
            grid_sizes: vec![0.1, 0.05, 0.025, 0.0125],
            expected_errors: None,
        },
        VerificationTestCase {
            name: "Second-order Trapezoid Rule".to_string(),
            expected_order: 2.0,
            order_tolerance: 0.2,
            grid_sizes: vec![0.1, 0.05, 0.025, 0.0125],
            expected_errors: None,
        },
        VerificationTestCase {
            name: "Fourth-order Runge-Kutta".to_string(),
            expected_order: 4.0,
            order_tolerance: 0.5,
            grid_sizes: vec![0.1, 0.05, 0.025, 0.0125],
            expected_errors: None,
        },
    ];

    for test_case in test_cases {
        workflow.add_test_case(test_case);
    }

    // Mock solvers with different orders of accuracy
    let solvers: Vec<(&str, SolverFunction)> = vec![
        (
            "First-order",
            Box::new(|h: &[f64]| -> Result<f64, IntegrateError> { Ok(0.1 * h[0]) }),
        ), // O(h)
        (
            "Second-order",
            Box::new(|h: &[f64]| -> Result<f64, IntegrateError> { Ok(0.1 * h[0] * h[0]) }),
        ), // O(h²)
        (
            "Fourth-order",
            Box::new(|h: &[f64]| -> Result<f64, IntegrateError> { Ok(0.1 * h[0].powi(4)) }),
        ), // O(h⁴)
    ];

    println!("Running automated verification workflow...\n");

    for (i, (solver_name, solver)) in solvers.iter().enumerate() {
        println!("Testing {solver_name}");

        // Run verification for this solver on the corresponding test case
        let single_case_workflow = VerificationWorkflow {
            test_cases: vec![workflow.test_cases[i].clone()],
        };

        let results = single_case_workflow.run_verification(solver);

        for result in &results {
            println!("  Test: {}", result.test_name);
            println!(
                "  Status: {}",
                if result.passed {
                    "✅ PASSED"
                } else {
                    "❌ FAILED"
                }
            );
            if let Some(order) = result.computed_order {
                println!("  Computed order: {order:.2}");
            }
            if let Some(ref error) = result.error_message {
                println!("  Error: {error}");
            }
            println!();
        }
    }

    println!("📊 Automated workflows enable:");
    println!("   • Continuous integration testing");
    println!("   • Regression detection");
    println!("   • Method comparison");
    println!("   • Performance benchmarking");
    println!();

    Ok(())
}
