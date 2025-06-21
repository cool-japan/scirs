//! Robin and mixed boundary conditions example
//!
//! This example demonstrates how to solve boundary value problems with
//! Robin boundary conditions (a*u + b*u' = c) and other advanced boundary
//! condition types.

use ndarray::{Array1, ArrayView1};
use scirs2_integrate::{
    bvp::BVPOptions,
    bvp_extended::{
        solve_bvp_extended, BoundaryConditionType, ExtendedBoundaryConditions, RobinBC,
    },
};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Robin and Mixed Boundary Conditions Examples ===\n");

    // Example 1: Heat conduction with convective boundary conditions
    heat_conduction_example()?;

    // Example 2: Beam deflection with mixed boundary conditions
    beam_deflection_example()?;

    // Example 3: Robin boundary conditions for reaction-diffusion
    reaction_diffusion_example()?;

    // Example 4: Periodic boundary conditions
    periodic_example()?;

    Ok(())
}

fn heat_conduction_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Heat Conduction with Convective Boundary Conditions");
    println!("{}", "=".repeat(60));

    // Solve: u'' = -q (constant heat generation)
    // with convective boundary conditions:
    // Left: -k*u'(0) = h*(u(0) - T_env)  =>  u'(0) + (h/k)*u(0) = (h/k)*T_env
    // Right: -k*u'(L) = h*(u(L) - T_env) => -u'(L) + (h/k)*u(L) = (h/k)*T_env

    let q = 1000.0; // Heat generation rate [W/m³]
    let k = 50.0; // Thermal conductivity [W/m·K]
    let h = 25.0; // Heat transfer coefficient [W/m²·K]
    let t_env = 20.0; // Environment temperature [°C]
    let length = 1.0; // Rod length [m]

    println!("Problem: Heat conduction in 1D rod");
    println!("Equation: u'' = -q/k = -{:.0}", q / k);
    println!("Left BC: u'(0) + (h/k)*u(0) = (h/k)*T_env");
    println!("Right BC: -u'(L) + (h/k)*u(L) = (h/k)*T_env");
    println!(
        "Parameters: q={} W/m³, k={} W/m·K, h={} W/m²·K, T_env={}°C",
        q, k, h, t_env
    );

    // Define the ODE: u'' = -q/k, so if y = [u, u'], then y' = [u', -q/k]
    let ode_fn = |_x: f64, y: ArrayView1<f64>| Array1::from_vec(vec![y[1], -q / k]);

    // Set up Robin boundary conditions
    let h_over_k = h / k;
    let boundary_conditions = ExtendedBoundaryConditions::robin(
        vec![(h_over_k, 1.0, h_over_k * t_env)], // Left: (h/k)*u + 1*u' = (h/k)*T_env
        vec![(h_over_k, -1.0, h_over_k * t_env)], // Right: (h/k)*u - 1*u' = (h/k)*T_env
    );

    let options = BVPOptions {
        max_iter: 100,
        tol: 1e-8,
        n_nodes: 20,
        fixed_mesh: true,
        ..Default::default()
    };

    let result = solve_bvp_extended(
        ode_fn,
        [0.0, length],
        boundary_conditions,
        21,
        Some(options),
    )?;

    if result.success {
        println!("\n✅ Solution converged in {} iterations", result.n_iter);
        println!("Final residual: {:.2e}", result.residual_norm);

        // Analytical solution for validation
        // For constant q, the solution is u(x) = A + B*x - (q/(2k))*x²
        // With convective BCs, this becomes more complex, but we can check some properties

        println!("\nTemperature profile:");
        println!("x [m]     T [°C]     u' [°C/m]");
        println!("{}", "─".repeat(35));

        let n_print = 11;
        for i in 0..n_print {
            let idx = i * result.x.len() / (n_print - 1);
            let idx = idx.min(result.x.len() - 1);
            let x = result.x[idx];
            let u = result.y[idx][0];
            let u_prime = result.y[idx][1];

            println!("{:5.2}     {:6.2}     {:8.2}", x, u, u_prime);
        }

        // Check heat flux balance at boundaries
        let q_left = -k * result.y[0][1]; // -k * u'(0)
        let q_right = -k * result.y.last().unwrap()[1]; // -k * u'(L)
        let q_conv_left = h * (result.y[0][0] - t_env);
        let q_conv_right = h * (result.y.last().unwrap()[0] - t_env);

        println!("\nHeat flux balance check:");
        println!(
            "Left:  Conduction = {:.2} W/m², Convection = {:.2} W/m²",
            q_left, q_conv_left
        );
        println!(
            "Right: Conduction = {:.2} W/m², Convection = {:.2} W/m²",
            q_right, q_conv_right
        );
    } else {
        println!("❌ Solution failed to converge");
        if let Some(msg) = result.message {
            println!("Error: {}", msg);
        }
    }

    println!();
    Ok(())
}

fn beam_deflection_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️  Beam Deflection with Mixed Boundary Conditions");
    println!("{}", "=".repeat(60));

    // Solve beam deflection: EI * u'''' = q(x)
    // Convert to first-order system: y = [u, u', u'', u''']
    // y' = [u', u'', u''', q(x)/(EI)]
    //
    // Mixed boundary conditions:
    // Left (clamped): u(0) = 0, u'(0) = 0
    // Right (simply supported): u(L) = 0, u''(L) = 0 (zero moment)

    let ei = 1000.0; // Flexural rigidity [N·m²]
    let q0 = 100.0; // Distributed load [N/m]
    let length = 2.0; // Beam length [m]

    println!("Problem: Beam deflection under uniform load");
    println!("Equation: EI*u'''' = q₀ = {}", q0);
    println!("Left BC (clamped): u(0) = 0, u'(0) = 0");
    println!("Right BC (simply supported): u(L) = 0, u''(L) = 0");
    println!(
        "Parameters: EI = {} N·m², q₀ = {} N/m, L = {} m",
        ei, q0, length
    );

    // Define the ODE system
    let ode_fn = |_x: f64, y: ArrayView1<f64>| {
        Array1::from_vec(vec![
            y[1],    // u' = u'
            y[2],    // u'' = u''
            y[3],    // u''' = u'''
            q0 / ei, // u'''' = q₀/(EI)
        ])
    };

    // Set up mixed boundary conditions
    let left_bcs = vec![
        BoundaryConditionType::Dirichlet { value: 0.0 }, // u(0) = 0
        BoundaryConditionType::Dirichlet { value: 0.0 }, // u'(0) = 0
    ];

    let right_bcs = vec![
        BoundaryConditionType::Dirichlet { value: 0.0 }, // u(L) = 0
        BoundaryConditionType::Dirichlet { value: 0.0 }, // u''(L) = 0
    ];

    let boundary_conditions = ExtendedBoundaryConditions::mixed(left_bcs, right_bcs);

    let options = BVPOptions {
        max_iter: 100,
        tol: 1e-10,
        n_nodes: 30,
        fixed_mesh: true,
        ..Default::default()
    };

    let result = solve_bvp_extended(
        ode_fn,
        [0.0, length],
        boundary_conditions,
        31,
        Some(options),
    )?;

    if result.success {
        println!("\n✅ Solution converged in {} iterations", result.n_iter);
        println!("Final residual: {:.2e}", result.residual_norm);

        // Analytical solution for uniform load with clamped-simply supported beam
        // is complex, but we can check basic properties

        println!("\nDeflection profile:");
        println!("x [m]     u [m]        u' [rad]     M [N·m]");
        println!("{}", "─".repeat(45));

        let n_print = 11;
        for i in 0..n_print {
            let idx = i * result.x.len() / (n_print - 1);
            let idx = idx.min(result.x.len() - 1);
            let x = result.x[idx];
            let u = result.y[idx][0];
            let u_prime = result.y[idx][1];
            let u_double_prime = result.y[idx][2];
            let moment = -ei * u_double_prime; // Bending moment M = -EI*u''

            println!(
                "{:5.2}     {:8.5}     {:8.5}     {:7.1}",
                x, u, u_prime, moment
            );
        }

        // Find maximum deflection
        let max_deflection = result
            .y
            .iter()
            .map(|y| y[0].abs())
            .fold(0.0f64, |a, b| a.max(b));

        println!("\nMaximum deflection: {:.5} m", max_deflection);

        // Check boundary conditions
        let u0 = result.y[0][0];
        let u_prime_0 = result.y[0][1];
        let u_l = result.y.last().unwrap()[0];
        let u_double_prime_l = result.y.last().unwrap()[2];

        println!("Boundary condition verification:");
        println!("u(0) = {:.2e} (should be 0)", u0);
        println!("u'(0) = {:.2e} (should be 0)", u_prime_0);
        println!("u(L) = {:.2e} (should be 0)", u_l);
        println!("u''(L) = {:.2e} (should be 0)", u_double_prime_l);
    } else {
        println!("❌ Solution failed to converge");
        if let Some(msg) = result.message {
            println!("Error: {}", msg);
        }
    }

    println!();
    Ok(())
}

fn reaction_diffusion_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Reaction-Diffusion with Robin Boundary Conditions");
    println!("{}", "=".repeat(60));

    // Solve: D*u'' - k*u + S = 0 (steady-state reaction-diffusion)
    // Robin BC: a*u + b*u' = c at both ends

    let d = 0.1; // Diffusion coefficient
    let k = 1.0; // Reaction rate
    let s = 5.0; // Source term

    println!("Problem: Steady-state reaction-diffusion");
    println!("Equation: {}*u'' - {}*u + {} = 0", d, k, s);
    println!("Robin BC: 2*u + u' = 10 at x=0");
    println!("Robin BC: u - 0.5*u' = 3 at x=1");

    // Define the ODE: if y = [u, u'], then y' = [u', (k*u - S)/D]
    let ode_fn = |_x: f64, y: ArrayView1<f64>| Array1::from_vec(vec![y[1], (k * y[0] - s) / d]);

    // Set up Robin boundary conditions using the builder
    let left_robin = RobinBC::new(2.0, 1.0, 10.0); // 2*u + u' = 10
    let right_robin = RobinBC::new(1.0, -0.5, 3.0); // u - 0.5*u' = 3

    let boundary_conditions = ExtendedBoundaryConditions::robin(
        vec![(left_robin.a, left_robin.b, left_robin.c)],
        vec![(right_robin.a, right_robin.b, right_robin.c)],
    );

    let options = BVPOptions {
        max_iter: 50,
        tol: 1e-8,
        fixed_mesh: true,
        ..Default::default()
    };

    let result = solve_bvp_extended(ode_fn, [0.0, 1.0], boundary_conditions, 21, Some(options))?;

    if result.success {
        println!("\n✅ Solution converged in {} iterations", result.n_iter);
        println!("Final residual: {:.2e}", result.residual_norm);

        println!("\nConcentration profile:");
        println!("x        u        u'       Reaction");
        println!("{}", "─".repeat(35));

        for (i, &x) in result.x.iter().enumerate() {
            let u = result.y[i][0];
            let u_prime = result.y[i][1];
            let reaction_rate = k * u;

            if i % 4 == 0 || i == result.x.len() - 1 {
                println!(
                    "{:5.2}    {:6.3}    {:6.3}    {:6.3}",
                    x, u, u_prime, reaction_rate
                );
            }
        }

        // Verify boundary conditions
        let u0 = result.y[0][0];
        let u_prime_0 = result.y[0][1];
        let bc_left = 2.0 * u0 + u_prime_0;

        let u1 = result.y.last().unwrap()[0];
        let u_prime_1 = result.y.last().unwrap()[1];
        let bc_right = u1 - 0.5 * u_prime_1;

        println!("\nBoundary condition verification:");
        println!("Left BC: 2*u(0) + u'(0) = {:.3} (should be 10.0)", bc_left);
        println!(
            "Right BC: u(1) - 0.5*u'(1) = {:.3} (should be 3.0)",
            bc_right
        );
    } else {
        println!("❌ Solution failed to converge");
        if let Some(msg) = result.message {
            println!("Error: {}", msg);
        }
    }

    println!();
    Ok(())
}

fn periodic_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Periodic Boundary Conditions");
    println!("{}", "=".repeat(60));

    // Solve: u'' + u = sin(2πx) with periodic BCs: u(0) = u(1), u'(0) = u'(1)

    println!("Problem: u'' + u = sin(2πx)");
    println!("Periodic BC: u(0) = u(1), u'(0) = u'(1)");

    // Define the ODE: if y = [u, u'], then y' = [u', -u + sin(2πx)]
    let ode_fn =
        |x: f64, y: ArrayView1<f64>| Array1::from_vec(vec![y[1], -y[0] + (2.0 * PI * x).sin()]);

    // Set up periodic boundary conditions
    let boundary_conditions = ExtendedBoundaryConditions::periodic(2);

    let options = BVPOptions {
        max_iter: 100,
        tol: 1e-8,
        fixed_mesh: true,
        ..Default::default()
    };

    let result = solve_bvp_extended(ode_fn, [0.0, 1.0], boundary_conditions, 21, Some(options))?;

    if result.success {
        println!("\n✅ Solution converged in {} iterations", result.n_iter);
        println!("Final residual: {:.2e}", result.residual_norm);

        println!("\nPeriodic solution profile:");
        println!("x        u        u'");
        println!("{}", "─".repeat(25));

        for (i, &x) in result.x.iter().enumerate() {
            let u = result.y[i][0];
            let u_prime = result.y[i][1];

            if i % 4 == 0 || i == result.x.len() - 1 {
                println!("{:5.2}    {:6.3}    {:6.3}", x, u, u_prime);
            }
        }

        // Verify periodic boundary conditions
        let u0 = result.y[0][0];
        let u_prime_0 = result.y[0][1];
        let u1 = result.y.last().unwrap()[0];
        let u_prime_1 = result.y.last().unwrap()[1];

        println!("\nPeriodic boundary condition verification:");
        println!(
            "u(0) = {:.6}, u(1) = {:.6}, diff = {:.2e}",
            u0,
            u1,
            (u0 - u1).abs()
        );
        println!(
            "u'(0) = {:.6}, u'(1) = {:.6}, diff = {:.2e}",
            u_prime_0,
            u_prime_1,
            (u_prime_0 - u_prime_1).abs()
        );
    } else {
        println!("❌ Solution failed to converge");
        if let Some(msg) = result.message {
            println!("Error: {}", msg);
        }
    }

    println!();
    Ok(())
}
