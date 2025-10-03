use scirs2_core::ndarray::{array, Array1, ArrayView1};
use scirs2_integrate::dae::{solve_semi_explicit_dae, DAEOptions};
use scirs2_integrate::ode::ODEMethod;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RLC Circuit as a DAE System Example");
    println!("==================================\n");

    // Circuit parameters
    let r = 10.0; // Resistance (ohms)
    let l = 0.1; // Inductance (henries)
    let c = 1e-3; // Capacitance (farads)
    let v_in = 10.0; // Input voltage (volts)

    // Time span: 0 to 2 seconds
    let t_span = [0.0, 2.0];

    // Initial conditions
    // - i_l: Current through the inductor (amperes)
    // - v_c: Voltage across the capacitor (volts)
    let x0 = array![0.0, 0.0]; // [i_l, v_c] - start with zero current and voltage

    // Algebraic variables
    // - i_r: Current through the resistor (amperes)
    // - i_c: Current through the capacitor (amperes)
    let y0 = array![0.0, 0.0]; // [i_r, i_c] - consistent with initial x values

    // Function that provides the input voltage profile
    let input_voltage = |t: f64| -> f64 {
        if t < 0.1 {
            // Linear ramp-up from 0 to v_in during first 0.1 seconds
            v_in * t / 0.1
        } else if t < 1.0 {
            // Constant voltage v_in from 0.1 to 1.0 seconds
            v_in
        } else {
            // Exponential decay after 1.0 seconds
            v_in * (-5.0 * (t - 1.0)).exp()
        }
    };

    // Differential equations for the circuit
    // di_l/dt = (v_in - v_c - r*i_r) / l
    // dv_c/dt = i_c / c
    let f = move |t: f64, x: ArrayView1<f64>, y: ArrayView1<f64>| -> Array1<f64> {
        let _i_l = x[0]; // Inductor current
        let v_c = x[1]; // Capacitor voltage
        let i_r = y[0]; // Resistor current
        let i_c = y[1]; // Capacitor current

        let v_source = input_voltage(t);

        array![
            (v_source - v_c - r * i_r) / l, // di_l/dt
            i_c / c                         // dv_c/dt
        ]
    };

    // Constraint equations (Kirchhoff's current law)
    // i_l - i_r - i_c = 0 (current at node)
    // i_r - v_c/r = 0 (Ohm's law for resistor)
    let g_constraint = |_t: f64, x: ArrayView1<f64>, y: ArrayView1<f64>| -> Array1<f64> {
        let i_l = x[0];
        let v_c = x[1];
        let i_r = y[0];
        let i_c = y[1];

        array![
            i_l - i_r - i_c, // KCL at node
            i_r - v_c / r    // Ohm's law
        ]
    };

    // DAE options
    let options = DAEOptions {
        method: ODEMethod::Radau, // Use Radau for better stability with DAEs
        rtol: 1e-6,
        atol: 1e-8,
        max_steps: 1000,
        ..Default::default()
    };

    // Solve the DAE system
    println!("Solving RLC circuit DAE system...");
    let result = solve_semi_explicit_dae(f, g_constraint, t_span, x0, y0, Some(options))?;

    println!("Solution completed with {} steps.\n", result.n_steps);

    // Calculate the theoretical natural frequency and damping ratio
    let omega_n = (1.0 / (l * c)).sqrt();
    let zeta = r / 2.0 * (c / l).sqrt();
    let damped_freq = omega_n * (1.0 - zeta * zeta).sqrt();

    println!("Circuit characteristics:");
    println!("Natural frequency: {omega_n:.3} rad/s");
    println!("Damping ratio: {zeta:.3}");
    if zeta < 1.0 {
        println!("Damped frequency: {damped_freq:.3} rad/s");
        println!(
            "Damped period: {:.3} s",
            2.0 * std::f64::consts::PI / damped_freq
        );
    }
    println!();

    // Print headers
    println!(
        "{:<10} {:<12} {:<12} {:<12} {:<12} {:<12}",
        "Time", "Input V", "Cap V", "Ind I", "Res I", "Cap I"
    );
    println!("{:-<70}", "");

    // Number of points to print
    let num_print = 10.min(result.t.len());

    // Print first few points
    for i in 0..num_print {
        let t = result.t[i];
        let v_source = input_voltage(t);
        let i_l = result.x[i][0];
        let v_c = result.x[i][1];
        let i_r = result.y[i][0];
        let i_c = result.y[i][1];

        println!("{t:<10.3} {v_source:<12.6} {v_c:<12.6} {i_l:<12.6} {i_r:<12.6} {i_c:<12.6}");
    }

    if result.t.len() > 2 * num_print {
        println!("{:^70}", "...");
    }

    // Print last few points
    if result.t.len() > num_print {
        for i in (result.t.len() - num_print)..result.t.len() {
            let t = result.t[i];
            let v_source = input_voltage(t);
            let i_l = result.x[i][0];
            let v_c = result.x[i][1];
            let i_r = result.y[i][0];
            let i_c = result.y[i][1];

            println!("{t:<10.3} {v_source:<12.6} {v_c:<12.6} {i_l:<12.6} {i_r:<12.6} {i_c:<12.6}");
        }
    }

    println!("\nConstraint Satisfaction Check (should be close to zero):");
    for i in [0, result.t.len() / 2, result.t.len() - 1] {
        let t = result.t[i];
        let constraints = g_constraint(t, result.x[i].view(), result.y[i].view());
        println!(
            "t = {:<8.3}: KCL error = {:.3e}, Ohm's law error = {:.3e}",
            t,
            constraints[0].abs(),
            constraints[1].abs()
        );
    }

    // Ensure conservation laws are satisfied
    println!("\nEnergy Analysis:");
    analyze_energy(&result.t, &result.x, &result.y, r, l, c, &input_voltage);

    Ok(())
}

/// Analyze the energy in different components of the circuit
#[allow(dead_code)]
fn analyze_energy(
    t: &[f64],
    x: &[Array1<f64>],
    y: &[Array1<f64>],
    r: f64,
    l: f64,
    c: f64,
    input_voltage: &dyn Fn(f64) -> f64,
) {
    // Calculate energy at select points
    println!(
        "{:<10} {:<12} {:<12} {:<12} {:<12}",
        "Time", "Inductor E", "Capacitor E", "Power In", "Power Diss"
    );
    println!("{:-<60}", "");

    let num_print = 10.min(t.len());
    let mut total_energy_in = 0.0;
    let mut total_energy_dissipated = 0.0;

    // Calculate energy at each time point
    for i in 0..num_print {
        let time = t[i];
        let i_l = x[i][0]; // Inductor current
        let v_c = x[i][1]; // Capacitor _voltage
        let i_r = y[i][0]; // Resistor current

        let v_source = input_voltage(time);

        // Inductor energy: E_l = 0.5 * L * i_l^2
        let inductor_energy = 0.5 * l * i_l * i_l;

        // Capacitor energy: E_c = 0.5 * C * v_c^2
        let capacitor_energy = 0.5 * c * v_c * v_c;

        // Instantaneous power from the source: P_in = v_source * i_l
        let power_in = v_source * i_l;

        // Instantaneous power dissipated in the resistor: P_r = i_r^2 * R
        let power_dissipated = i_r * i_r * r;

        println!(
            "{time:<10.3} {inductor_energy:<12.6e} {capacitor_energy:<12.6e} {power_in:<12.6e} {power_dissipated:<12.6e}"
        );

        // For energy calculations, we'd need to integrate over time
        if i > 0 {
            let dt = t[i] - t[i - 1];
            // Trapezoidal rule for power integration
            let avg_power_in = (power_in + v_source * x[i - 1][0]) / 2.0;
            let avg_power_diss = (power_dissipated + y[i - 1][0] * y[i - 1][0] * r) / 2.0;

            total_energy_in += avg_power_in * dt;
            total_energy_dissipated += avg_power_diss * dt;
        }
    }

    // Final energy in reactive components
    let final_i_l = x[t.len() - 1][0];
    let final_v_c = x[t.len() - 1][1];
    let final_inductor_energy = 0.5 * l * final_i_l * final_i_l;
    let final_capacitor_energy = 0.5 * c * final_v_c * final_v_c;
    let final_stored_energy = final_inductor_energy + final_capacitor_energy;

    println!("\nEnergy balance at final time t = {:.3}:", t[t.len() - 1]);
    println!("Total energy input from source: {total_energy_in:.6e}");
    println!("Total energy dissipated in resistor: {total_energy_dissipated:.6e}");
    println!("Final energy stored in L and C: {final_stored_energy:.6e}");
    println!(
        "Energy balance (input - dissipated - stored): {:.6e}",
        total_energy_in - total_energy_dissipated - final_stored_energy
    );
}
