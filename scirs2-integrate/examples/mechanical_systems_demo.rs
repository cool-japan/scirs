//! Mechanical Systems Integration Demo
//!
//! This example demonstrates the specialized numerical integration methods for
//! mechanical systems including rigid body dynamics, constrained multibody systems,
//! and various integration schemes optimized for mechanical systems.

use ndarray::{Array1, Array2};
use scirs2_integrate::ode::mechanical::{systems, MechanicalIntegrator, PositionIntegrationMethod};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mechanical Systems Integration Demo\n");

    // Example 1: Simple rigid body dynamics
    println!("1. Rigid Body Dynamics");
    demonstrate_rigid_body()?;
    println!();

    // Example 2: Damped harmonic oscillator
    println!("2. Damped Harmonic Oscillator");
    demonstrate_damped_oscillator()?;
    println!();

    // Example 3: Double pendulum (multibody system)
    println!("3. Double Pendulum (Multibody System)");
    demonstrate_double_pendulum()?;
    println!();

    // Example 4: Energy conservation comparison
    println!("4. Energy Conservation Comparison");
    demonstrate_energy_conservation()?;
    println!();

    // Example 5: Integration method comparison
    println!("5. Integration Method Comparison");
    demonstrate_integration_methods()?;
    println!();

    println!("All mechanical systems demonstrations completed successfully!");

    Ok(())
}

/// Demonstrate basic rigid body dynamics
fn demonstrate_rigid_body() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple rigid body
    let mass = 2.0;
    let inertia = Array2::eye(3) * 0.1; // Spherical body
    let initial_position = Array1::from_vec(vec![0.0, 0.0, 0.0]);
    let initial_velocity = Array1::from_vec(vec![1.0, 0.5, 0.0]);
    let initial_orientation = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]); // Identity quaternion
    let initial_angular_velocity = Array1::from_vec(vec![0.0, 0.0, 1.0]); // Rotating about z-axis

    let (config, properties, initial_state) = systems::rigid_body(
        mass,
        inertia,
        initial_position,
        initial_velocity,
        initial_orientation,
        initial_angular_velocity,
    );

    let mut integrator = MechanicalIntegrator::new(config, properties);
    let mut state = initial_state;
    let dt = 0.01;
    let n_steps = 100;

    println!(
        "   Integrating rigid body motion for {} steps (dt = {})",
        n_steps, dt
    );
    println!(
        "   Initial position: [{:.3}, {:.3}, {:.3}]",
        state.position[0], state.position[1], state.position[2]
    );
    println!(
        "   Initial velocity: [{:.3}, {:.3}, {:.3}]",
        state.velocity[0], state.velocity[1], state.velocity[2]
    );

    // Integrate for several steps
    for i in 0..n_steps {
        let t = i as f64 * dt;
        let result = integrator.step(t, &state)?;
        state = result.state;

        // Print status every 20 steps
        if i % 20 == 0 {
            println!(
                "   Step {}: position = [{:.3}, {:.3}, {:.3}], energy drift = {:.2e}",
                i, state.position[0], state.position[1], state.position[2], result.energy_drift
            );
        }
    }

    let (relative_drift, max_drift, current_energy) = integrator.energy_statistics();
    println!(
        "   Final position: [{:.3}, {:.3}, {:.3}]",
        state.position[0], state.position[1], state.position[2]
    );
    println!(
        "   Energy statistics: current = {:.6}, relative drift = {:.2e}, max drift = {:.2e}",
        current_energy, relative_drift, max_drift
    );

    Ok(())
}

/// Demonstrate damped harmonic oscillator
fn demonstrate_damped_oscillator() -> Result<(), Box<dyn std::error::Error>> {
    let mass = 1.0;
    let stiffness = 10.0; // k = 10 N/m
    let damping = 0.2; // Light damping
    let initial_position = 1.0; // Displaced 1 meter
    let initial_velocity = 0.0;

    let (mut config, properties, initial_state) =
        systems::damped_oscillator(mass, stiffness, damping, initial_position, initial_velocity);

    // Use smaller time step for better accuracy
    config.dt = 0.001;
    let dt = config.dt;

    let mut integrator = MechanicalIntegrator::new(config, properties);
    let mut state = initial_state;
    let n_steps = 1000;

    println!(
        "   Damped oscillator: m = {}, k = {}, c = {}",
        mass, stiffness, damping
    );
    println!(
        "   Natural frequency: {:.3} rad/s",
        (stiffness / mass).sqrt()
    );
    println!(
        "   Damping ratio: {:.3}",
        damping / (2.0 * (stiffness * mass).sqrt())
    );

    let mut max_position = initial_position;
    let mut positions = Vec::new();
    let mut times = Vec::new();

    for i in 0..n_steps {
        let t = i as f64 * dt;
        let result = integrator.step(t, &state)?;
        state = result.state;

        positions.push(state.position[0]);
        times.push(t);
        max_position = max_position.max(state.position[0].abs());

        // Print oscillation peaks
        if i > 0 && positions.len() > 2 {
            let prev_pos = positions[positions.len() - 2];
            let curr_pos = positions[positions.len() - 1];
            if positions.len() > 3 {
                let prev_prev_pos = positions[positions.len() - 3];
                if prev_pos > prev_prev_pos && prev_pos > curr_pos && prev_pos.abs() > 0.01 {
                    println!(
                        "   Peak at t = {:.3}s: position = {:.4} m",
                        t - dt,
                        prev_pos
                    );
                }
            }
        }
    }

    println!("   Maximum displacement: {:.4} m", max_position);
    println!("   Final position: {:.4} m", state.position[0]);
    println!("   Final velocity: {:.4} m/s", state.velocity[0]);

    Ok(())
}

/// Demonstrate double pendulum system
fn demonstrate_double_pendulum() -> Result<(), Box<dyn std::error::Error>> {
    let m1 = 1.0; // Mass of first pendulum
    let m2 = 0.5; // Mass of second pendulum
    let l1 = 1.0; // Length of first pendulum
    let l2 = 0.8; // Length of second pendulum
    let initial_angles = [PI / 6.0, PI / 4.0]; // Initial angles (30° and 45°)
    let initial_velocities = [0.0, 0.0]; // Start from rest

    let (mut config, properties, initial_state) =
        systems::double_pendulum(m1, m2, l1, l2, initial_angles, initial_velocities);

    config.dt = 0.001; // Small time step for accuracy
    config.constraint_tolerance = 1e-10;
    let dt = config.dt;

    let mut integrator = MechanicalIntegrator::new(config, properties);
    let mut state = initial_state;
    let n_steps = 5000;

    println!(
        "   Double pendulum: m1 = {}, m2 = {}, l1 = {}, l2 = {}",
        m1, m2, l1, l2
    );
    println!(
        "   Initial angles: {:.1}° and {:.1}°",
        initial_angles[0] * 180.0 / PI,
        initial_angles[1] * 180.0 / PI
    );

    let mut energy_history = Vec::new();
    let mut constraint_violations = Vec::new();

    for i in 0..n_steps {
        let t = i as f64 * dt;
        let result = integrator.step(t, &state)?;
        state = result.state;

        energy_history.push(result.energy_drift);
        constraint_violations.push(result.constraint_violation);

        // Print status every 1000 steps
        if i % 1000 == 0 {
            let x1 = state.position[0];
            let y1 = state.position[1];
            let x2 = state.position[3];
            let y2 = state.position[4];

            let angle1 = if x1.abs() < 1e-10 && y1.abs() < 1e-10 {
                0.0 // Default angle when position is at origin
            } else {
                y1.atan2(x1) + PI / 2.0
            };
            let angle2 = if (x2 - x1).abs() < 1e-10 && (y2 - y1).abs() < 1e-10 {
                0.0 // Default angle when positions are coincident
            } else {
                (y2 - y1).atan2(x2 - x1) + PI / 2.0
            };

            println!(
                "   Step {}: angles = [{:.1}°, {:.1}°], constraint violation = {:.2e}",
                i,
                angle1 * 180.0 / PI,
                angle2 * 180.0 / PI,
                result.constraint_violation
            );
        }
    }

    // Calculate statistics
    let avg_energy_drift = energy_history.iter().sum::<f64>() / energy_history.len() as f64;
    let max_constraint_violation = constraint_violations.iter().fold(0.0f64, |a, &b| a.max(b));

    println!("   Average energy drift: {:.2e}", avg_energy_drift);
    println!(
        "   Maximum constraint violation: {:.2e}",
        max_constraint_violation
    );
    println!(
        "   Final constraint violation: {:.2e}",
        constraint_violations.last().unwrap()
    );

    Ok(())
}

/// Demonstrate energy conservation with different integrators
fn demonstrate_energy_conservation() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Comparing energy conservation for different integration methods");

    // Test system: undamped harmonic oscillator
    let mass = 1.0;
    let stiffness = 25.0; // Higher frequency for more challenging test
    let damping = 0.0; // No damping for energy conservation
    let initial_position = 1.0;
    let initial_velocity = 0.0;

    let integration_methods = vec![
        ("Verlet", PositionIntegrationMethod::Verlet),
        ("Velocity Verlet", PositionIntegrationMethod::VelocityVerlet),
        (
            "Newmark-β",
            PositionIntegrationMethod::NewmarkBeta {
                beta: 0.25,
                gamma: 0.5,
            },
        ),
        (
            "Central Difference",
            PositionIntegrationMethod::CentralDifference,
        ),
    ];

    for (method_name, method) in integration_methods {
        let (mut config, properties, initial_state) = systems::damped_oscillator(
            mass,
            stiffness,
            damping,
            initial_position,
            initial_velocity,
        );

        config.dt = 0.0005; // Even smaller time step for better energy conservation
        config.position_method = method;
        let dt = config.dt;

        let mut integrator = MechanicalIntegrator::new(config, properties);
        let mut state = initial_state;
        let n_steps = 20000; // Integrate for 10 seconds with even smaller dt

        // Integrate
        for i in 0..n_steps {
            let t = i as f64 * dt;
            let result = integrator.step(t, &state)?;
            state = result.state;
        }

        let (relative_drift, max_drift, _) = integrator.energy_statistics();
        println!(
            "   {}: relative drift = {:.2e}, max drift = {:.2e}",
            method_name, relative_drift, max_drift
        );
    }

    Ok(())
}

/// Demonstrate comparison of different integration methods
fn demonstrate_integration_methods() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Comparing accuracy and stability of integration methods");

    // Test system: lightly damped oscillator
    let mass = 1.0;
    let stiffness = 100.0; // High frequency system
    let damping = 0.1; // Light damping
    let initial_position = 1.0;
    let initial_velocity = 0.0;

    let time_steps = vec![0.01, 0.005, 0.001];

    for &dt in &time_steps {
        println!("   Time step dt = {}", dt);

        let (mut config, properties, initial_state) = systems::damped_oscillator(
            mass,
            stiffness,
            damping,
            initial_position,
            initial_velocity,
        );

        config.dt = dt;
        config.position_method = PositionIntegrationMethod::VelocityVerlet;

        let mut integrator = MechanicalIntegrator::new(config, properties);
        let mut state = initial_state;
        let n_steps = (1.0 / dt) as usize; // Integrate for 1 second

        let mut max_constraint_iterations = 0;
        let mut total_force_time = 0.0;
        let mut total_constraint_time = 0.0;

        for i in 0..n_steps {
            let t = i as f64 * dt;
            let result = integrator.step(t, &state)?;
            state = result.state;

            max_constraint_iterations =
                max_constraint_iterations.max(result.stats.constraint_iterations);
            total_force_time += result.stats.force_computation_time;
            total_constraint_time += result.stats.constraint_time;
        }

        let (relative_drift, _max_drift, final_energy) = integrator.energy_statistics();

        println!("     Final position: {:.6} m", state.position[0]);
        println!("     Final energy: {:.6} J", final_energy);
        println!("     Energy drift: {:.2e}", relative_drift);
        println!(
            "     Max constraint iterations: {}",
            max_constraint_iterations
        );
        println!(
            "     Avg force computation time: {:.2e} s",
            total_force_time / n_steps as f64
        );
        println!(
            "     Avg constraint time: {:.2e} s",
            total_constraint_time / n_steps as f64
        );
        println!();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_rigid_body_integration() {
        let mass = 1.0;
        let inertia = Array2::eye(3);
        let initial_position = Array1::zeros(3);
        let initial_velocity = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let initial_orientation = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let initial_angular_velocity = Array1::zeros(3);

        let (config, properties, initial_state) = systems::rigid_body(
            mass,
            inertia,
            initial_position,
            initial_velocity,
            initial_orientation,
            initial_angular_velocity,
        );

        let mut integrator = MechanicalIntegrator::new(config, properties);
        let result = integrator.step(0.0, &initial_state).unwrap();

        // After one time step, position should have changed
        assert!(result.state.position[0] > 0.0);
        assert_eq!(result.stats.converged, true);
    }

    #[test]
    fn test_oscillator_period() {
        let mass = 1.0;
        let stiffness = 4.0 * PI * PI; // Natural frequency = 2π rad/s, period = 1s
        let damping = 0.0;
        let initial_position = 1.0;
        let initial_velocity = 0.0;

        let (mut config, properties, initial_state) = systems::damped_oscillator(
            mass,
            stiffness,
            damping,
            initial_position,
            initial_velocity,
        );

        config.dt = 0.001;
        let dt = config.dt; // Save dt before moving config
        let mut integrator = MechanicalIntegrator::new(config, properties);
        let mut state = initial_state;

        // Integrate for one period (1000 steps)
        for i in 0..1000 {
            let t = i as f64 * dt;
            let result = integrator.step(t, &state).unwrap();
            state = result.state;
        }

        // Should return approximately to initial position
        assert_abs_diff_eq!(state.position[0], initial_position, epsilon = 0.1);
        assert_abs_diff_eq!(state.velocity[0], initial_velocity, epsilon = 0.5);
    }

    #[test]
    fn test_energy_conservation() {
        let mass = 1.0;
        let stiffness = 10.0;
        let damping = 0.0; // No damping
        let initial_position = 1.0;
        let initial_velocity = 0.0;

        let (mut config, properties, initial_state) = systems::damped_oscillator(
            mass,
            stiffness,
            damping,
            initial_position,
            initial_velocity,
        );

        config.dt = 0.001;
        let dt = config.dt; // Save dt before moving config
        let mut integrator = MechanicalIntegrator::new(config, properties);
        let mut state = initial_state;

        // Integrate for many steps
        for i in 0..500 {
            let t = i as f64 * dt;
            let result = integrator.step(t, &state).unwrap();
            state = result.state;
        }

        let (relative_drift, _max_drift, _current_energy) = integrator.energy_statistics();

        // Energy should be well conserved for undamped system
        assert!(
            relative_drift < 0.01,
            "Energy drift too large: {}",
            relative_drift
        );
    }
}
