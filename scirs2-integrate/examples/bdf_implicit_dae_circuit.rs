//! BDF-specific DAE solver for an RLC circuit
//!
//! This example demonstrates the usage of specialized BDF methods for
//! solving an RLC circuit represented as a fully implicit DAE.
//!
//! The implicit DAE form is:
//! F(t, y, y') = 0
//!
//! For an RLC circuit with a voltage source, the equations are:
//! C*v' + i + v/R = 0
//! L*i' - v + v_source(t) = 0

use plotters::prelude::*;
use scirs2_core::ndarray::{array, ArrayView1};
use scirs2_core::numeric::Float;
use scirs2_integrate::{bdf_implicit_dae, DAEIndex, DAEOptions, DAEType};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Circuit parameters
    let r = 1.0; // Resistance (Ohms)
    let l = 1.0; // Inductance (Henries)
    let c = 0.1; // Capacitance (Farads)

    // Resonant frequency
    let omega_res = 1.0 / (l * c).sqrt();
    let damping = r / (2.0 * (l / c).sqrt());

    println!("RLC Circuit parameters:");
    println!("  Resonant frequency: {omega_res:.6} rad/s");
    println!("  Damping coefficient: {damping:.6}");

    // Initial conditions
    let v0 = 0.0; // Initial voltage across capacitor
    let i0 = 0.0; // Initial current through inductor

    // Initial state and derivative
    let y0 = array![v0, i0];

    // Initial derivatives (v', i')
    // For voltage source Vs(t) = sin(t), at t=0, Vs(0) = 0
    // From the circuit equations:
    // C*v' + i + v/R = 0  => v' = -(i + v/R)/C
    // L*i' - v + Vs(t) = 0  => i' = (v - Vs(t))/L
    let v_prime0 = -(i0 + v0 / r) / c;
    let i_prime0 = (v0 - 0.0) / l; // Vs(0) = 0
    let y_prime0 = array![v_prime0, i_prime0];

    // Define the time span for the integration
    let t_span = [0.0, 20.0];

    // Define the implicit DAE function F(t, y, y')
    let f = |t: f64, y: ArrayView1<f64>, y_prime: ArrayView1<f64>| {
        let v = y[0];
        let i = y[1];
        let v_prime = y_prime[0];
        let i_prime = y_prime[1];

        // External voltage source: sine wave
        let v_source = t.sin();

        // Residual equations:
        // F1 = C*v' + i + v/R
        // F2 = L*i' - v + v_source
        array![c * v_prime + i + v / r, l * i_prime - v + v_source]
    };

    // Set solver options
    let options = DAEOptions {
        dae_type: DAEType::FullyImplicit,
        index: DAEIndex::Index1,
        rtol: 1e-6,
        atol: 1e-8,
        max_steps: 10_000,
        max_newton_iterations: 10,
        newton_tol: 1e-8,
        h0: Some(0.01),
        min_step: Some(1e-10),
        max_step: Some(0.1),
        max_order: Some(5),
        ..Default::default()
    };

    // Solve the DAE system using specialized BDF method
    let result = bdf_implicit_dae(f, t_span, y0, y_prime0, options)?;

    // Print some performance statistics
    println!("BDF Implicit DAE solver performance metrics:");
    println!("  Steps taken: {}", result.n_steps);
    println!("  Steps accepted: {}", result.n_accepted);
    println!("  Steps rejected: {}", result.n_rejected);
    println!("  Function evaluations: {}", result.n_eval);
    println!("  Jacobian evaluations: {}", result.n_jac);
    println!("  LU decompositions: {}", result.n_lu);
    println!("  Success: {}", result.success);

    if let Some(msg) = &result.message {
        println!("  Message: {msg}");
    }

    // Calculate the input voltage and total energy at each time point
    let mut v_source = Vec::new();
    let mut energy = Vec::new();

    for i in 0..result.t.len() {
        let t = result.t[i];
        let v = result.x[i][0];
        let i_current = result.x[i][1];

        // External voltage source: sine wave
        v_source.push(t.sin());

        // Energy stored in capacitor (0.5 * C * v²)
        let capacitor_energy = 0.5 * c * v * v;

        // Energy stored in inductor (0.5 * L * i²)
        let inductor_energy = 0.5 * l * i_current * i_current;

        // Total energy
        energy.push(capacitor_energy + inductor_energy);
    }

    // Create a plot of the circuit variables
    let root = BitMapBackend::new("rlc_circuit_bdf.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_v = result
        .x
        .iter()
        .map(|y| y[0].abs())
        .fold(0.0, |a, b| a.max(b));
    let max_i = result
        .x
        .iter()
        .map(|y| y[1].abs())
        .fold(0.0, |a, b| a.max(b));
    let max_val = max_v.max(max_i) * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("RLC Circuit Response", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..t_span[1], -max_val..max_val)?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_desc("Time (s)")
        .y_desc("Voltage (V) / Current (A)")
        .draw()?;

    // Draw the voltage
    chart
        .draw_series(LineSeries::new(
            result
                .t
                .iter()
                .zip(result.x.iter())
                .map(|(&t, y)| (t, y[0])),
            &RED,
        ))?
        .label("Capacitor Voltage")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    // Draw the current
    chart
        .draw_series(LineSeries::new(
            result
                .t
                .iter()
                .zip(result.x.iter())
                .map(|(&t, y)| (t, y[1])),
            &GREEN,
        ))?
        .label("Inductor Current")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    // Draw the input voltage
    chart
        .draw_series(LineSeries::new(
            result.t.iter().zip(v_source.iter()).map(|(&t, &v)| (t, v)),
            &BLUE,
        ))?
        .label("Input Voltage")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    // Plot the energy over time
    let root = BitMapBackend::new("rlc_energy_bdf.png", (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_energy = energy.iter().cloned().fold(0.0, |a, b| a.max(b)) * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Circuit Energy", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..t_span[1], 0.0..max_energy)?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_desc("Time (s)")
        .y_desc("Energy (J)")
        .draw()?;

    // Draw the energy plot
    chart.draw_series(LineSeries::new(
        result.t.iter().zip(energy.iter()).map(|(&t, &e)| (t, e)),
        &BLUE,
    ))?;

    println!("Plots generated: rlc_circuit_bdf.png and rlc_energy_bdf.png");

    Ok(())
}
