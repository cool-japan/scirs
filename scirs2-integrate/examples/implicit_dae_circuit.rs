use scirs2_core::ndarray::{array, Array1, ArrayView1};
use scirs2_integrate::dae::{solve_implicit_dae, DAEOptions, DAEType};
use scirs2_integrate::ode::ODEMethod;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Implicit DAE Solver Example - RLC Circuit");
    println!("==========================================\n");

    // Circuit parameters
    let r = 10.0; // Resistance (ohms)
    let l = 0.1; // Inductance (henries)
    let c = 1e-3; // Capacitance (farads)
    let v_in = 10.0; // Input voltage (volts)

    // Time span: 0 to 2 seconds
    let t_span = [0.0, 2.0];

    // Initial conditions
    // y = [i_l, v_c, i_r, i_c]
    // i_l: Current through the inductor (amperes)
    // v_c: Voltage across the capacitor (volts)
    // i_r: Current through the resistor (amperes)
    // i_c: Current through the capacitor (amperes)
    let y0 = array![0.0, 0.0, 0.0, 0.0];

    // Initial derivatives
    // y' = [i_l', v_c', i_r', i_c']
    // For the initial condition, we need consistent derivatives
    // Based on the circuit equations and initial values:
    let yprime0 = array![v_in / l, 0.0, 0.0, 0.0];

    // Define the input voltage function
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

    // Define the DAE residual function
    // F(t, y, y') = 0
    // For the RLC circuit, we have:
    // F1: l * i_l' - v_in + v_c + r * i_r = 0 (Kirchhoff's voltage law)
    // F2: v_c' - i_c / c = 0 (Capacitor relation)
    // F3: i_l - i_r - i_c = 0 (Kirchhoff's current law)
    // F4: i_r - v_c / r = 0 (Ohm's law)
    let residual_fn = move |t: f64, y: ArrayView1<f64>, yprime: ArrayView1<f64>| -> Array1<f64> {
        let i_l = y[0]; // Inductor current
        let v_c = y[1]; // Capacitor voltage
        let i_r = y[2]; // Resistor current
        let i_c = y[3]; // Capacitor current

        let i_l_prime = yprime[0]; // Derivative of inductor current
        let v_c_prime = yprime[1]; // Derivative of capacitor voltage

        let v_source = input_voltage(t);

        array![
            l * i_l_prime - v_source + v_c + r * i_r, // Kirchhoff's voltage law
            v_c_prime - i_c / c,                      // Capacitor relation
            i_l - i_r - i_c,                          // Kirchhoff's current law
            i_r - v_c / r                             // Ohm's law
        ]
    };

    // DAE options
    let options = DAEOptions {
        dae_type: DAEType::FullyImplicit, // Explicit indication of DAE type
        method: ODEMethod::Radau,         // Radau method is effective for DAEs
        rtol: 1e-6,
        atol: 1e-8,
        max_steps: 1000,
        max_newton_iterations: 10,
        newton_tol: 1e-8,
        ..Default::default()
    };

    // Solve the DAE system
    println!("Solving implicit DAE system...");
    let result = solve_implicit_dae(residual_fn, t_span, y0, yprime0, Some(options))?;

    println!(
        "Solution completed with {} steps ({} accepted, {} rejected).\n",
        result.n_steps, result.n_accepted, result.n_rejected
    );

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
        let i_r = result.x[i][2];
        let i_c = result.x[i][3];

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
            let i_r = result.x[i][2];
            let i_c = result.x[i][3];

            println!("{t:<10.3} {v_source:<12.6} {v_c:<12.6} {i_l:<12.6} {i_r:<12.6} {i_c:<12.6}");
        }
    }

    println!("\nResidual Check (should be close to zero):");
    for i in [0, result.t.len() / 2, result.t.len() - 1] {
        let t = result.t[i];
        let y = &result.x[i];

        // At these points, we don't have the derivatives stored,
        // so we estimate them using finite difference
        let yprime = if i > 0 && i < result.t.len() - 1 {
            // Central difference for interior points
            let dt_prev = result.t[i] - result.t[i - 1];
            let dt_next = result.t[i + 1] - result.t[i];
            let _weight_prev = dt_next / (dt_prev + dt_next);
            let _weight_next = dt_prev / (dt_prev + dt_next);

            (&result.x[i + 1] - &result.x[i - 1]) / (dt_prev + dt_next)
        } else if i > 0 {
            // Backward difference for the last point
            let dt = result.t[i] - result.t[i - 1];
            (&result.x[i] - &result.x[i - 1]) / dt
        } else {
            // Forward difference for the first point
            let dt = result.t[i + 1] - result.t[i];
            (&result.x[i + 1] - &result.x[i]) / dt
        };

        let residual = residual_fn(t, y.view(), yprime.view());
        let residual_norm = residual.iter().fold(0.0, |acc, &x| acc + x * x).sqrt();

        println!("t = {t:<8.3}: Residual norm = {residual_norm:.3e}");
    }

    // Verify that the algebraic constraints are satisfied
    println!("\nAlgebraic Constraint Check:");
    for i in [0, result.t.len() / 2, result.t.len() - 1] {
        let t = result.t[i];
        let y = &result.x[i];

        // Check Kirchhoff's current law: i_l - i_r - i_c = 0
        let kcl_error = y[0] - y[2] - y[3];

        // Check Ohm's law: i_r - v_c / r = 0
        let ohm_error = y[2] - y[1] / r;

        println!(
            "t = {:<8.3}: KCL error = {:.3e}, Ohm's law error = {:.3e}",
            t,
            kcl_error.abs(),
            ohm_error.abs()
        );
    }

    // Ensure energy is conserved
    println!("\nEnergy Analysis:");
    println!(
        "{:<10} {:<12} {:<12} {:<12} {:<12}",
        "Time", "Inductor E", "Capacitor E", "Power In", "Power Diss"
    );
    println!("{:-<60}", "");

    let mut total_energy_in = 0.0;
    let mut total_energy_dissipated = 0.0;

    // Calculate energy at select points
    for i in 0..num_print {
        let time = result.t[i];
        let i_l = result.x[i][0]; // Inductor current
        let v_c = result.x[i][1]; // Capacitor voltage
        let i_r = result.x[i][2]; // Resistor current

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
            let dt = result.t[i] - result.t[i - 1];
            // Trapezoidal rule for power integration
            let avg_power_in =
                (power_in + input_voltage(result.t[i - 1]) * result.x[i - 1][0]) / 2.0;
            let avg_power_diss =
                (power_dissipated + result.x[i - 1][2] * result.x[i - 1][2] * r) / 2.0;

            total_energy_in += avg_power_in * dt;
            total_energy_dissipated += avg_power_diss * dt;
        }
    }

    // Final energy in reactive components
    let final_i_l = result.x[result.t.len() - 1][0];
    let final_v_c = result.x[result.t.len() - 1][1];
    let final_inductor_energy = 0.5 * l * final_i_l * final_i_l;
    let final_capacitor_energy = 0.5 * c * final_v_c * final_v_c;
    let final_stored_energy = final_inductor_energy + final_capacitor_energy;

    println!(
        "\nEnergy balance at final time t = {:.3}:",
        result.t[result.t.len() - 1]
    );
    println!("Total energy input from source: {total_energy_in:.6e}");
    println!("Total energy dissipated in resistor: {total_energy_dissipated:.6e}");
    println!("Final energy stored in L and C: {final_stored_energy:.6e}");
    println!(
        "Energy balance (input - dissipated - stored): {:.6e}",
        total_energy_in - total_energy_dissipated - final_stored_energy
    );

    // The energy balance should be close to zero if energy is conserved
    let energy_balance = (total_energy_in - total_energy_dissipated - final_stored_energy).abs();
    let energy_input = total_energy_in.abs();
    let relative_error = if energy_input > 1e-10 {
        energy_balance / energy_input
    } else {
        energy_balance
    };

    println!("Relative energy error: {relative_error:.6e}");

    Ok(())
}
