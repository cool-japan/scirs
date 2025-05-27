use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use std::time::Instant;

fn main() {
    println!("Enhanced LSODA Solver Example");
    println!("----------------------------");
    println!("This example demonstrates the enhanced LSODA implementation");
    println!("which features improved stiffness detection and method switching.");
    println!();

    // First, test standard vs enhanced LSODA on a simple test problem
    println!("Simple Problem: Exponential Decay");
    println!("y' = -y, y(0) = 1, exact solution: y(t) = exp(-t)");

    // Exponential decay function
    let decay = |_t: f64, y: ArrayView1<f64>| array![-y[0]];

    // Solve with standard LSODA
    let standard_start = Instant::now();
    let standard_result = solve_ivp(
        decay,
        [0.0, 10.0],
        array![1.0],
        Some(ODEOptions {
            method: ODEMethod::LSODA,
            rtol: 1e-6,
            atol: 1e-8,
            ..Default::default()
        }),
    )
    .unwrap();
    let standard_duration = standard_start.elapsed();

    // Solve with enhanced LSODA
    let enhanced_start = Instant::now();
    let enhanced_result = solve_ivp(
        decay,
        [0.0, 10.0],
        array![1.0],
        Some(ODEOptions {
            method: ODEMethod::EnhancedLSODA,
            rtol: 1e-6,
            atol: 1e-8,
            ..Default::default()
        }),
    )
    .unwrap();
    let enhanced_duration = enhanced_start.elapsed();

    // Print comparison for simple case
    println!("\nComparison for simple problem:");
    println!("--------------------------------------------------");
    println!("                           Standard      Enhanced");
    println!("--------------------------------------------------");
    println!(
        "Time (ms):                 {:.2}          {:.2}",
        standard_duration.as_secs_f64() * 1000.0,
        enhanced_duration.as_secs_f64() * 1000.0
    );
    println!(
        "Function evaluations:      {}           {}",
        standard_result.n_eval, enhanced_result.n_eval
    );
    println!(
        "Steps taken:               {}           {}",
        standard_result.n_steps, enhanced_result.n_steps
    );
    println!(
        "Final error:               {:.2e}       {:.2e}",
        (standard_result.y.last().unwrap()[0] - (-10.0f64).exp()).abs(),
        (enhanced_result.y.last().unwrap()[0] - (-10.0f64).exp()).abs()
    );
    println!("--------------------------------------------------");

    // Now test with a stiff problem
    println!("\nStiff Problem: Van der Pol Oscillator with μ = 1000");
    println!("y'' - 1000(1-y²)y' + y = 0");
    println!("(extremely stiff problem requiring method switching)");

    // Van der Pol oscillator with large mu (very stiff)
    let mu = 1000.0;
    let van_der_pol =
        move |_t: f64, y: ArrayView1<f64>| array![y[1], mu * (1.0 - y[0].powi(2)) * y[1] - y[0]];

    // Solve with standard LSODA
    let vdp_standard_start = Instant::now();
    let vdp_standard_result = solve_ivp(
        van_der_pol,
        [0.0, 3000.0],
        array![2.0, 0.0],
        Some(ODEOptions {
            method: ODEMethod::LSODA,
            rtol: 1e-6,
            atol: 1e-8,
            max_steps: 10000,
            ..Default::default()
        }),
    );
    let vdp_standard_duration = vdp_standard_start.elapsed();

    // Solve with enhanced LSODA
    let vdp_enhanced_start = Instant::now();
    let vdp_enhanced_result = solve_ivp(
        van_der_pol,
        [0.0, 3000.0],
        array![2.0, 0.0],
        Some(ODEOptions {
            method: ODEMethod::EnhancedLSODA,
            rtol: 1e-6,
            atol: 1e-8,
            max_steps: 10000,
            ..Default::default()
        }),
    );
    let vdp_enhanced_duration = vdp_enhanced_start.elapsed();

    // Print comparison for stiff case
    println!("\nComparison for stiff problem:");
    println!("--------------------------------------------------");
    println!("                           Standard      Enhanced");
    println!("--------------------------------------------------");

    // Handle potential failures for either solver
    match (&vdp_standard_result, &vdp_enhanced_result) {
        (Ok(std_res), Ok(enh_res)) => {
            println!(
                "Time (ms):                 {:.2}          {:.2}",
                vdp_standard_duration.as_secs_f64() * 1000.0,
                vdp_enhanced_duration.as_secs_f64() * 1000.0
            );
            println!(
                "Function evaluations:      {}           {}",
                std_res.n_eval, enh_res.n_eval
            );
            println!(
                "Steps taken:               {}           {}",
                std_res.n_steps, enh_res.n_steps
            );
            println!(
                "Jacobian evaluations:      {}           {}",
                std_res.n_jac, enh_res.n_jac
            );
            println!(
                "LU decompositions:         {}           {}",
                std_res.n_lu, enh_res.n_lu
            );

            // Print final state
            println!("\nFinal state (y0, y1):");
            println!(
                "Standard: [{:.6}, {:.6}]",
                std_res.y.last().unwrap()[0],
                std_res.y.last().unwrap()[1]
            );
            println!(
                "Enhanced: [{:.6}, {:.6}]",
                enh_res.y.last().unwrap()[0],
                enh_res.y.last().unwrap()[1]
            );

            // Print method switching information from enhanced LSODA
            if let Some(msg) = &enh_res.message {
                println!("\nEnhanced LSODA Method Switching Information:");
                println!("{}", msg);
            }
        }
        (Ok(_), Err(e)) => {
            println!("Standard: Completed successfully");
            println!("Enhanced: Failed with error: {}", e);
        }
        (Err(e), Ok(_)) => {
            println!("Standard: Failed with error: {}", e);
            println!("Enhanced: Completed successfully");
        }
        (Err(e1), Err(e2)) => {
            println!("Both solvers failed:");
            println!("Standard error: {}", e1);
            println!("Enhanced error: {}", e2);
        }
    }
    println!("--------------------------------------------------");

    // Final summary and recommendations
    println!("\nEnhanced LSODA Features:");
    println!("1. Better stiffness detection using multiple indicators");
    println!("2. Improved method switching logic to reduce unnecessary switches");
    println!("3. More robust Jacobian approximation and reuse strategy");
    println!("4. Enhanced error estimation for better step size control");
    println!("5. Diagnostic information about method switching decisions");

    println!("\nRecommendations:");
    println!("- For simple problems, standard LSODA may be faster");
    println!("- For stiff problems, enhanced LSODA often requires fewer function evaluations");
    println!("- When diagnostic information is needed, enhanced LSODA provides more details");
    println!("- Both methods support the same interface and can be used interchangeably");
}
