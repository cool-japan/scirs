//! Chemical Kinetics Integration Demo
//!
//! This example demonstrates the specialized numerical integration methods for
//! chemical kinetics and reaction networks, including stiff reaction systems,
//! enzyme kinetics, and various reaction mechanisms.

use scirs2_integrate::ode::chemical::{ChemicalIntegrator, StiffIntegrationMethod};
use scirs2_integrate::ode::chemical_systems;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Chemical Kinetics Integration Demo\n");

    // Example 1: Simple first-order reaction A -> B
    println!("1. First-Order Reaction: A -> B");
    demonstrate_first_order_reaction()?;
    println!();

    // Example 2: Reversible reaction A <-> B
    println!("2. Reversible Reaction: A <-> B");
    demonstrate_reversible_reaction()?;
    println!();

    // Example 3: Enzyme kinetics (Michaelis-Menten)
    println!("3. Enzyme Kinetics: E + S <-> ES -> E + P");
    demonstrate_enzyme_kinetics()?;
    println!();

    // Example 4: Competitive reactions
    println!("4. Competitive Reactions: A + B -> C, A + D -> E");
    demonstrate_competitive_reactions()?;
    println!();

    // Example 5: Stiff reaction system
    println!("5. Stiff Reaction System (Multiple Time Scales)");
    demonstrate_stiff_reactions()?;
    println!();

    // Example 6: Integration method comparison
    println!("6. Integration Method Comparison for Stiff Systems");
    demonstrate_method_comparison()?;
    println!();

    println!("All chemical kinetics demonstrations completed successfully!");

    Ok(())
}

/// Demonstrate simple first-order reaction kinetics
fn demonstrate_first_order_reaction() -> Result<(), Box<dyn std::error::Error>> {
    let rate_constant = 0.5; // s^-1
    let initial_a = 1.0; // Initial concentration of A
    let initial_b = 0.0; // Initial concentration of B

    let (config, properties, initial_state) =
        chemical_systems::first_order_reaction(rate_constant, initial_a, initial_b);

    let mut integrator = ChemicalIntegrator::new(config, properties);
    let mut state = initial_state;
    let dt = 0.1;
    let n_steps = 50;

    println!("   Rate constant k = {} s^-1", rate_constant);
    println!("   Analytical solution: [A](t) = [A]₀ * exp(-kt)");
    println!(
        "   Initial: [A] = {:.3} M, [B] = {:.3} M",
        initial_a, initial_b
    );

    let mut concentrations_a = Vec::new();
    let mut concentrations_b = Vec::new();
    let mut times = Vec::new();

    for i in 0..n_steps {
        let t = i as f64 * dt;
        let result = integrator.step(t, &state)?;
        state = result.state;

        concentrations_a.push(state.concentrations[0]);
        concentrations_b.push(state.concentrations[1]);
        times.push(t);

        // Print status every 10 steps
        if i % 10 == 0 {
            let analytical_a = initial_a * (-rate_constant * t).exp();
            let analytical_b = initial_a - analytical_a;
            let error_a = (state.concentrations[0] - analytical_a).abs();
            let error_b = (state.concentrations[1] - analytical_b).abs();

            println!(
                "   t = {:.1}s: [A] = {:.4} M (analytical: {:.4}, error: {:.2e})",
                t, state.concentrations[0], analytical_a, error_a
            );
            println!(
                "             [B] = {:.4} M (analytical: {:.4}, error: {:.2e})",
                state.concentrations[1], analytical_b, error_b
            );
        }
    }

    // Calculate final conversion
    let conversion = (initial_a - state.concentrations[0]) / initial_a * 100.0;
    println!("   Final conversion: {:.1}%", conversion);
    println!(
        "   Mass balance: {:.6} M (should be {:.6} M)",
        state.concentrations.sum(),
        initial_a + initial_b
    );

    Ok(())
}

/// Demonstrate reversible reaction kinetics
fn demonstrate_reversible_reaction() -> Result<(), Box<dyn std::error::Error>> {
    let k_forward = 0.3; // Forward rate constant
    let k_reverse = 0.1; // Reverse rate constant
    let initial_a = 1.0;
    let initial_b = 0.0;

    let (config, properties, initial_state) =
        chemical_systems::reversible_reaction(k_forward, k_reverse, initial_a, initial_b);

    let mut integrator = ChemicalIntegrator::new(config, properties);
    let mut state = initial_state;
    let dt = 0.05;
    let n_steps = 200;

    println!(
        "   Forward rate k₁ = {} s^-1, Reverse rate k₋₁ = {} s^-1",
        k_forward, k_reverse
    );
    println!(
        "   Equilibrium constant K = k₁/k₋₁ = {:.2}",
        k_forward / k_reverse
    );

    // Calculate analytical equilibrium concentrations
    let total_conc = initial_a + initial_b;
    let keq = k_forward / k_reverse;
    let a_eq = total_conc / (1.0 + keq);
    let b_eq = total_conc * keq / (1.0 + keq);

    println!(
        "   Equilibrium: [A]_eq = {:.4} M, [B]_eq = {:.4} M",
        a_eq, b_eq
    );

    let mut reached_equilibrium = false;

    for i in 0..n_steps {
        let t = i as f64 * dt;
        let result = integrator.step(t, &state)?;
        state = result.state;

        // Check if we're close to equilibrium
        let approach_a = (state.concentrations[0] - a_eq).abs() / a_eq;
        let approach_b = (state.concentrations[1] - b_eq).abs() / b_eq;

        if !reached_equilibrium && approach_a < 0.01 && approach_b < 0.01 {
            println!("   Approached equilibrium at t = {:.2}s", t);
            println!(
                "   [A] = {:.4} M (equilibrium: {:.4}), [B] = {:.4} M (equilibrium: {:.4})",
                state.concentrations[0], a_eq, state.concentrations[1], b_eq
            );
            reached_equilibrium = true;
        }

        // Print status every 40 steps
        if i % 40 == 0 {
            println!(
                "   t = {:.1}s: [A] = {:.4} M, [B] = {:.4} M, conservation error = {:.2e}",
                t, state.concentrations[0], state.concentrations[1], result.conservation_error
            );
        }
    }

    let stats = integrator.get_system_stats();
    println!(
        "   Final stiffness estimate: {:.2}",
        stats.stiffness_estimate
    );

    Ok(())
}

/// Demonstrate enzyme kinetics (Michaelis-Menten mechanism)
fn demonstrate_enzyme_kinetics() -> Result<(), Box<dyn std::error::Error>> {
    let k1 = 1.0; // Forward binding rate (M^-1 s^-1)
    let k_minus_1 = 0.5; // Reverse binding rate (s^-1)
    let k2 = 0.3; // Catalytic rate (s^-1)
    let initial_enzyme = 0.01; // 10 μM enzyme
    let initial_substrate = 0.1; // 100 μM substrate
    let initial_product = 0.0;

    let (mut config, properties, initial_state) = chemical_systems::enzyme_kinetics(
        k1,
        k_minus_1,
        k2,
        initial_enzyme,
        initial_substrate,
        initial_product,
    );

    // Use smaller time step for enzyme kinetics
    config.dt = 0.01;
    let dt = config.dt;

    let mut integrator = ChemicalIntegrator::new(config, properties);
    let mut state = initial_state;
    let n_steps = 1000;

    println!("   Enzyme kinetics parameters:");
    println!(
        "   k₁ = {} M⁻¹s⁻¹, k₋₁ = {} s⁻¹, k₂ = {} s⁻¹",
        k1, k_minus_1, k2
    );

    // Calculate Michaelis constant
    let km = (k_minus_1 + k2) / k1;
    let vmax = k2 * initial_enzyme;
    println!("   Kₘ = {:.4} M, Vₘₐₓ = {:.4} M/s", km, vmax);

    println!(
        "   Initial: [E] = {:.3} M, [S] = {:.3} M, [P] = {:.3} M",
        initial_enzyme, initial_substrate, initial_product
    );

    let mut max_es_concentration = 0.0_f64;
    let mut _time_to_steady_state = 0.0;
    let mut steady_state_reached = false;

    for i in 0..n_steps {
        let t = i as f64 * dt;
        let result = integrator.step(t, &state)?;
        state = result.state;

        let es_conc = state.concentrations[3]; // ES complex concentration
        max_es_concentration = max_es_concentration.max(es_conc);

        // Check for steady state in ES (rapid equilibrium approximation)
        if !steady_state_reached && i > 50 {
            let es_prev = if i >= 10 {
                state.concentrations[3]
            } else {
                0.0
            };
            let es_change_rate = if i >= 10 {
                (es_conc - es_prev).abs() / (10.0 * dt)
            } else {
                1.0
            };

            if es_change_rate < 1e-6 {
                _time_to_steady_state = t;
                steady_state_reached = true;
                println!(
                    "   ES steady state reached at t = {:.3}s, [ES] = {:.6} M",
                    t, es_conc
                );
            }
        }

        // Print status every 200 steps
        if i % 200 == 0 {
            println!(
                "   t = {:.2}s: [E] = {:.5} M, [S] = {:.5} M, [P] = {:.5} M, [ES] = {:.6} M",
                t,
                state.concentrations[0],
                state.concentrations[1],
                state.concentrations[2],
                state.concentrations[3]
            );

            if result.stats.stiffness_ratio > 1.0 {
                println!("     Stiffness ratio: {:.1}", result.stats.stiffness_ratio);
            }
        }
    }

    // Calculate final product formation rate
    let final_rate = vmax * state.concentrations[1] / (km + state.concentrations[1]);
    println!(
        "   Final substrate concentration: [S] = {:.5} M",
        state.concentrations[1]
    );
    println!(
        "   Final product concentration: [P] = {:.5} M",
        state.concentrations[2]
    );
    println!("   Predicted final rate: v = {:.6} M/s", final_rate);
    println!("   Maximum ES concentration: {:.6} M", max_es_concentration);

    // Check enzyme conservation
    let total_enzyme = state.concentrations[0] + state.concentrations[3]; // E + ES
    let enzyme_conservation_error = (total_enzyme - initial_enzyme).abs() / initial_enzyme;
    println!(
        "   Enzyme conservation error: {:.2e}",
        enzyme_conservation_error
    );

    Ok(())
}

/// Demonstrate competitive reactions
fn demonstrate_competitive_reactions() -> Result<(), Box<dyn std::error::Error>> {
    let k1 = 0.2; // A + B -> C
    let k2 = 0.3; // A + D -> E
    let initial_a = 1.0;
    let initial_b = 0.5;
    let initial_d = 0.3;

    let (config, properties, initial_state) =
        chemical_systems::competitive_reactions(k1, k2, initial_a, initial_b, initial_d);

    let mut integrator = ChemicalIntegrator::new(config, properties);
    let mut state = initial_state;
    let dt = 0.1;
    let n_steps = 100;

    println!("   Reaction 1: A + B -> C (k₁ = {} M⁻¹s⁻¹)", k1);
    println!("   Reaction 2: A + D -> E (k₂ = {} M⁻¹s⁻¹)", k2);
    println!(
        "   Initial: [A] = {} M, [B] = {} M, [D] = {} M",
        initial_a, initial_b, initial_d
    );

    // Calculate selectivity
    let selectivity_factor = k2 * initial_d / (k1 * initial_b);
    println!(
        "   Selectivity factor (k₂[D]₀)/(k₁[B]₀) = {:.2}",
        selectivity_factor
    );

    let mut product_c_max = 0.0_f64;
    let mut product_e_max = 0.0_f64;

    for i in 0..n_steps {
        let t = i as f64 * dt;
        let result = integrator.step(t, &state)?;
        state = result.state;

        product_c_max = product_c_max.max(state.concentrations[2]);
        product_e_max = product_e_max.max(state.concentrations[4]);

        // Print status every 20 steps
        if i % 20 == 0 {
            let conversion_a = (initial_a - state.concentrations[0]) / initial_a * 100.0;
            println!(
                "   t = {:.1}s: [A] = {:.4} M ({:.1}% consumed)",
                t, state.concentrations[0], conversion_a
            );
            println!(
                "           [C] = {:.4} M, [E] = {:.4} M (C:E ratio = {:.2})",
                state.concentrations[2],
                state.concentrations[4],
                if state.concentrations[4] > 1e-10 {
                    state.concentrations[2] / state.concentrations[4]
                } else {
                    f64::INFINITY
                }
            );
        }
    }

    println!(
        "   Final selectivity C:E = {:.2}",
        if product_e_max > 1e-10 {
            product_c_max / product_e_max
        } else {
            f64::INFINITY
        }
    );

    Ok(())
}

/// Demonstrate stiff reaction system
fn demonstrate_stiff_reactions() -> Result<(), Box<dyn std::error::Error>> {
    let fast_rate = 1000.0; // Very fast reaction
    let slow_rate = 0.001; // Very slow reaction
    let initial_concentrations = vec![1.0, 0.0, 0.0]; // A, B, C

    let (mut config, properties, initial_state) = chemical_systems::stiff_reaction_system(
        fast_rate,
        slow_rate,
        initial_concentrations.clone(),
    );

    // Use small time step appropriate for stiff system
    config.dt = 0.0001;
    let dt = config.dt;
    let stiff_method = config.stiff_method;

    println!("   Fast reaction: A -> B (k₁ = {} s⁻¹)", fast_rate);
    println!("   Slow reaction: B -> C (k₂ = {} s⁻¹)", slow_rate);
    println!("   Stiffness ratio: {:.0}", fast_rate / slow_rate);

    let mut integrator = ChemicalIntegrator::new(config, properties);
    let mut state = initial_state;
    let n_steps = 50000; // Integrate for 5 seconds

    println!(
        "   Using {} integration method with dt = {} s",
        match stiff_method {
            StiffIntegrationMethod::BDF2 => "BDF2",
            StiffIntegrationMethod::ImplicitEuler => "Implicit Euler",
            _ => "Other",
        },
        dt
    );

    let mut fast_equilibrium_time = None;
    let mut slow_timescale_started = false;

    for i in 0..n_steps {
        let t = i as f64 * dt;
        let result = integrator.step(t, &state)?;
        state = result.state;

        // Check when fast reaction reaches quasi-equilibrium
        if fast_equilibrium_time.is_none() && state.concentrations[0] < 0.01 {
            fast_equilibrium_time = Some(t);
            println!(
                "   Fast reaction quasi-equilibrium reached at t = {:.4}s",
                t
            );
            println!(
                "   [A] = {:.6} M, [B] = {:.6} M, [C] = {:.6} M",
                state.concentrations[0], state.concentrations[1], state.concentrations[2]
            );
        }

        // Check when slow timescale becomes dominant
        if !slow_timescale_started && state.concentrations[2] > 0.1 {
            slow_timescale_started = true;
            println!("   Slow timescale regime started at t = {:.2}s", t);
        }

        // Print status every 10000 steps (every 1 second)
        if i % 10000 == 0 {
            println!(
                "   t = {:.1}s: [A] = {:.6} M, [B] = {:.6} M, [C] = {:.6} M",
                t, state.concentrations[0], state.concentrations[1], state.concentrations[2]
            );
            println!(
                "           Stiffness ratio: {:.1}",
                result.stats.stiffness_ratio
            );
        }
    }

    // Verify mass conservation
    let total_mass = state.concentrations.sum();
    let initial_mass = initial_concentrations.iter().sum::<f64>();
    println!(
        "   Mass conservation: {:.8} M (initial: {:.8} M, error: {:.2e})",
        total_mass,
        initial_mass,
        (total_mass - initial_mass).abs()
    );

    Ok(())
}

/// Compare different integration methods for stiff systems
fn demonstrate_method_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let fast_rate = 500.0;
    let slow_rate = 0.002;
    let initial_concentrations = vec![1.0, 0.0, 0.0];

    let methods = vec![
        ("BDF2", StiffIntegrationMethod::BDF2),
        ("Implicit Euler", StiffIntegrationMethod::ImplicitEuler),
        ("Rosenbrock", StiffIntegrationMethod::Rosenbrock),
        ("Adaptive", StiffIntegrationMethod::Adaptive),
    ];

    println!("   Comparing integration methods for stiff system:");
    println!(
        "   Fast rate: {} s⁻¹, Slow rate: {} s⁻¹",
        fast_rate, slow_rate
    );

    for (method_name, method) in methods {
        let (mut config, properties, initial_state) = chemical_systems::stiff_reaction_system(
            fast_rate,
            slow_rate,
            initial_concentrations.clone(),
        );

        config.stiff_method = method;
        config.dt = 0.001; // Larger time step to test stability
        let dt = config.dt;

        let mut integrator = ChemicalIntegrator::new(config, properties);
        let mut state = initial_state;
        let n_steps = 5000; // Integrate for 5 seconds

        let start_time = std::time::Instant::now();
        let mut total_function_evals = 0;
        let mut total_newton_iters = 0;
        let mut max_stiffness = 0.0_f64;

        for i in 0..n_steps {
            let t = i as f64 * dt;
            match integrator.step(t, &state) {
                Ok(result) => {
                    state = result.state;
                    total_function_evals += result.stats.function_evaluations;
                    total_newton_iters += result.stats.newton_iterations;
                    max_stiffness = max_stiffness.max(result.stats.stiffness_ratio);
                }
                Err(_) => {
                    println!("     {}: Integration failed at t = {:.3}s", method_name, t);
                    break;
                }
            }
        }

        let elapsed_time = start_time.elapsed().as_secs_f64();
        let final_conversion = (1.0 - state.concentrations[0]) * 100.0;

        println!("     {}:", method_name);
        println!(
            "       Final concentrations: [A] = {:.6}, [B] = {:.6}, [C] = {:.6}",
            state.concentrations[0], state.concentrations[1], state.concentrations[2]
        );
        println!("       Conversion: {:.2}%", final_conversion);
        println!("       Computation time: {:.4}s", elapsed_time);
        println!("       Function evaluations: {}", total_function_evals);
        println!("       Newton iterations: {}", total_newton_iters);
        println!("       Max stiffness ratio: {:.1}", max_stiffness);
        println!();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_first_order_demo() {
        assert!(demonstrate_first_order_reaction().is_ok());
    }

    #[test]
    fn test_reversible_demo() {
        assert!(demonstrate_reversible_reaction().is_ok());
    }

    #[test]
    fn test_enzyme_kinetics_demo() {
        assert!(demonstrate_enzyme_kinetics().is_ok());
    }

    #[test]
    fn test_competitive_reactions_demo() {
        assert!(demonstrate_competitive_reactions().is_ok());
    }

    #[test]
    fn test_stiff_reactions_demo() {
        assert!(demonstrate_stiff_reactions().is_ok());
    }

    #[test]
    fn test_method_comparison_demo() {
        assert!(demonstrate_method_comparison().is_ok());
    }

    #[test]
    fn test_analytical_vs_numerical() {
        // Test first-order reaction against analytical solution
        let rate_constant = 0.1;
        let initial_a = 1.0;
        let initial_b = 0.0;

        let (config, properties, initial_state) =
            chemical_systems::first_order_reaction(rate_constant, initial_a, initial_b);

        let mut integrator = ChemicalIntegrator::new(config, properties);
        let mut state = initial_state;

        // Integrate for several time steps
        for i in 0..10 {
            let t = i as f64 * 0.1;
            let result = integrator.step(t, &state).unwrap();
            state = result.state;
        }

        // Compare with analytical solution at t = 1.0
        let t_final = 1.0;
        let analytical_a = initial_a * (-rate_constant * t_final).exp();
        let analytical_b = initial_a - analytical_a;

        assert_abs_diff_eq!(state.concentrations[0], analytical_a, epsilon = 0.1);
        assert_abs_diff_eq!(state.concentrations[1], analytical_b, epsilon = 0.1);
    }

    #[test]
    fn test_mass_conservation() {
        let (config, properties, initial_state) =
            chemical_systems::reversible_reaction(0.1, 0.05, 1.0, 0.0);

        let mut integrator = ChemicalIntegrator::new(config, properties);
        let mut state = initial_state.clone();
        let initial_total = initial_state.concentrations.sum();

        // Integrate for many steps
        for i in 0..100 {
            let t = i as f64 * 0.05;
            let result = integrator.step(t, &state).unwrap();
            state = result.state;
        }

        let final_total = state.concentrations.sum();
        assert_abs_diff_eq!(initial_total, final_total, epsilon = 1e-8);
    }
}
