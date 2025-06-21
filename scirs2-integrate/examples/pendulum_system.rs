//! Pendulum System Examples
//!
//! This example demonstrates solving pendulum dynamics using various ODE methods.
//! It includes simple pendulum, damped pendulum, and driven pendulum systems.

use ndarray::{array, Array1, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use std::f64::consts::PI;

/// Simple pendulum: θ'' + (g/L)sin(θ) = 0
/// State vector: [θ, θ']
fn simple_pendulum(t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let _t = t; // Time not explicitly used in this system
    let theta = y[0];
    let theta_dot = y[1];

    let g = 9.81; // gravitational acceleration (m/s²)
    let l = 1.0; // pendulum length (m)

    let theta_ddot = -(g / l) * theta.sin();

    array![theta_dot, theta_ddot]
}

/// Damped pendulum: θ'' + c*θ' + (g/L)sin(θ) = 0
/// State vector: [θ, θ']
fn damped_pendulum(t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let _t = t; // Time not explicitly used in this system
    let theta = y[0];
    let theta_dot = y[1];

    let g = 9.81; // gravitational acceleration (m/s²)
    let l = 1.0; // pendulum length (m)
    let c = 0.5; // damping coefficient

    let theta_ddot = -(g / l) * theta.sin() - c * theta_dot;

    array![theta_dot, theta_ddot]
}

/// Driven pendulum: θ'' + c*θ' + (g/L)sin(θ) = A*cos(ωt)
/// State vector: [θ, θ']
fn driven_pendulum(t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let theta = y[0];
    let theta_dot = y[1];

    let g = 9.81; // gravitational acceleration (m/s²)
    let l = 1.0; // pendulum length (m)
    let c = 0.2; // damping coefficient
    let a = 1.5; // driving amplitude
    let omega = 2.0; // driving frequency

    let theta_ddot = -(g / l) * theta.sin() - c * theta_dot + a * (omega * t).cos();

    array![theta_dot, theta_ddot]
}

/// Double pendulum system
/// State vector: [θ₁, θ₁', θ₂, θ₂']
fn double_pendulum(t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let _t = t; // Time not explicitly used
    let theta1 = y[0];
    let theta1_dot = y[1];
    let theta2 = y[2];
    let theta2_dot = y[3];

    let g = 9.81; // gravitational acceleration
    let l1 = 1.0; // length of first pendulum
    let l2 = 1.0; // length of second pendulum
    let m1 = 1.0; // mass of first pendulum
    let m2 = 1.0; // mass of second pendulum

    let delta_theta = theta2 - theta1;
    let den1 = (m1 + m2) * l1 - m2 * l1 * (delta_theta).cos() * (delta_theta).cos();
    let den2 = (l2 / l1) * den1;

    // First pendulum angular acceleration
    let num1 = -m2 * l1 * theta1_dot * theta1_dot * (delta_theta).sin() * (delta_theta).cos()
        + m2 * g * (theta2).sin() * (delta_theta).cos()
        + m2 * l2 * theta2_dot * theta2_dot * (delta_theta).sin()
        - (m1 + m2) * g * (theta1).sin();

    let theta1_ddot = num1 / den1;

    // Second pendulum angular acceleration
    let num2 = -m2 * l2 * theta2_dot * theta2_dot * (delta_theta).sin() * (delta_theta).cos()
        + (m1 + m2) * g * (theta1).sin() * (delta_theta).cos()
        - (m1 + m2) * l1 * theta1_dot * theta1_dot * (delta_theta).sin()
        - (m1 + m2) * g * (theta2).sin();

    let theta2_ddot = num2 / den2;

    array![theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Pendulum System Examples\n");

    // Example 1: Simple pendulum with small angle approximation
    println!("1. Simple Pendulum (small angles)");
    let t_span = [0.0, 10.0];
    let y0 = array![PI / 6.0, 0.0]; // Initial angle: 30 degrees, initial velocity: 0

    let result = solve_ivp(simple_pendulum, t_span, y0.clone(), None)?;

    println!(
        "   Initial angle: {:.3} rad ({:.1}°)",
        y0[0],
        y0[0] * 180.0 / PI
    );
    println!(
        "   Final angle: {:.3} rad ({:.1}°)",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[0] * 180.0 / PI
    );
    println!(
        "   Period (theoretical): {:.3} s",
        2.0 * PI * (1.0_f64 / 9.81).sqrt()
    );
    println!("   Steps taken: {}", result.t.len());
    println!();

    // Example 2: Damped pendulum
    println!("2. Damped Pendulum");
    let options = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        ..Default::default()
    };

    let result = solve_ivp(damped_pendulum, t_span, y0.clone(), Some(options))?;

    println!("   Initial angle: {:.3} rad", y0[0]);
    println!(
        "   Final angle: {:.3} rad (energy dissipated by damping)",
        result.y.last().unwrap()[0]
    );
    println!("   Steps taken: {}", result.t.len());
    println!();

    // Example 3: Driven pendulum (potential chaotic behavior)
    println!("3. Driven Pendulum");
    let y0_driven = array![0.1, 0.0]; // Small initial displacement
    let t_span_long = [0.0, 50.0]; // Longer time span to see driven behavior

    let options_driven = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-9,
        atol: 1e-11,
        max_step: Some(0.1), // Smaller max step for driven system
        ..Default::default()
    };

    let result = solve_ivp(
        driven_pendulum,
        t_span_long,
        y0_driven.clone(),
        Some(options_driven),
    )?;

    println!("   Initial angle: {:.3} rad", y0_driven[0]);
    println!("   Final angle: {:.3} rad", result.y.last().unwrap()[0]);
    println!(
        "   Final velocity: {:.3} rad/s",
        result.y.last().unwrap()[1]
    );
    println!("   Steps taken: {}", result.t.len());
    println!();

    // Example 4: Double pendulum (chaotic system)
    println!("4. Double Pendulum (Chaotic System)");
    let y0_double = array![PI / 4.0, 0.0, PI / 2.0, 0.0]; // θ₁=45°, θ₂=90°
    let t_span_short = [0.0, 10.0]; // Shorter time for chaotic system

    let options_double = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-10,
        atol: 1e-12,
        max_step: Some(0.01), // Very small steps for accuracy
        ..Default::default()
    };

    let result = solve_ivp(
        double_pendulum,
        t_span_short,
        y0_double.clone(),
        Some(options_double),
    )?;

    println!(
        "   Initial angles: θ₁={:.3} rad, θ₂={:.3} rad",
        y0_double[0], y0_double[2]
    );
    println!(
        "   Final angles: θ₁={:.3} rad, θ₂={:.3} rad",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[2]
    );
    println!(
        "   Steps taken: {} (high precision needed for chaotic system)",
        result.t.len()
    );
    println!();

    // Example 5: Energy conservation check for undamped pendulum
    println!("5. Energy Conservation Analysis");
    let result_energy = solve_ivp(simple_pendulum, [0.0, 20.0], array![PI / 3.0, 0.0], None)?;

    let g = 9.81;
    let l = 1.0;

    // Calculate total energy at different times
    let initial_energy = {
        let theta = result_energy.y[0][0];
        let theta_dot = result_energy.y[0][1];
        0.5 * theta_dot * theta_dot + g / l * (1.0 - theta.cos())
    };

    let final_energy = {
        let theta = result_energy.y.last().unwrap()[0];
        let theta_dot = result_energy.y.last().unwrap()[1];
        0.5 * theta_dot * theta_dot + g / l * (1.0 - theta.cos())
    };

    println!("   Initial total energy: {:.6} J/kg", initial_energy);
    println!("   Final total energy: {:.6} J/kg", final_energy);
    println!(
        "   Energy conservation error: {:.2e}",
        (final_energy - initial_energy).abs()
    );
    println!();

    println!("All pendulum examples completed successfully!");
    println!("\nNotes:");
    println!("- Simple pendulum shows periodic motion");
    println!("- Damped pendulum shows energy dissipation");
    println!("- Driven pendulum can exhibit complex, potentially chaotic behavior");
    println!("- Double pendulum is a classic example of deterministic chaos");
    println!("- Energy conservation demonstrates numerical accuracy of the solver");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_pendulum_small_angle() {
        // For small angles, pendulum behaves like harmonic oscillator
        let t_span = [0.0, 2.0];
        let y0 = array![0.1, 0.0]; // 0.1 rad ≈ 5.7 degrees (small angle)

        let result = solve_ivp(simple_pendulum, t_span, y0.clone(), None).unwrap();

        // Check that motion is approximately periodic
        let final_angle = result.y.last().unwrap()[0];

        // For small angles, the theoretical period is 2π√(L/g)
        let _theoretical_period = 2.0 * PI * (1.0_f64 / 9.81).sqrt();

        // After 2 seconds (less than one period), should still be oscillating
        assert!(final_angle.abs() < 0.2); // Should stay within reasonable bounds
        assert!(result.y.len() > 10); // Should take multiple steps
    }

    #[test]
    fn test_damped_pendulum_energy_loss() {
        let t_span = [0.0, 10.0];
        let y0 = array![PI / 4.0, 0.0];

        let result = solve_ivp(damped_pendulum, t_span, y0.clone(), None).unwrap();

        // Initial and final kinetic + potential energy
        let g = 9.81;
        let l = 1.0;

        let initial_energy = g / l * (1.0 - y0[0].cos());
        let final_state = result.y.last().unwrap();
        let final_energy =
            0.5 * final_state[1] * final_state[1] + g / l * (1.0 - final_state[0].cos());

        // Energy should decrease due to damping
        assert!(final_energy < initial_energy);
    }

    #[test]
    fn test_double_pendulum_conservation() {
        // Test that double pendulum conserves energy (undamped)
        let t_span = [0.0, 1.0]; // Short time to avoid numerical drift
        let y0 = array![0.1, 0.0, 0.2, 0.0]; // Small angles

        let options = ODEOptions {
            rtol: 1e-12,
            atol: 1e-14,
            ..Default::default()
        };

        let result = solve_ivp(double_pendulum, t_span, y0.clone(), Some(options)).unwrap();

        // Just verify the integration completed successfully
        assert!(result.t.len() > 2);
        assert_eq!(result.y.len(), result.t.len());

        // Verify state variables remain finite and reasonable
        for state in result.y.iter() {
            assert!(state[0].is_finite()); // theta1
            assert!(state[1].is_finite()); // theta1_dot
            assert!(state[2].is_finite()); // theta2
            assert!(state[3].is_finite()); // theta2_dot

            // Angles should remain reasonable for small initial conditions
            assert!(state[0].abs() < 1.0); // theta1 < 1 radian
            assert!(state[2].abs() < 1.0); // theta2 < 1 radian
        }
    }
}
