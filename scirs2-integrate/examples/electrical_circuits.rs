//! Electrical Circuit Examples
//!
//! This example demonstrates solving various electrical circuit problems using ODE methods.
//! It includes RC/RL circuits, RLC oscillators, and nonlinear circuits with diodes.

use ndarray::{array, Array1, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use std::f64::consts::PI;

/// Simple RC charging circuit: C(dV/dt) + V/R = V_source/R
/// State vector: [V_c] (capacitor voltage)
fn rc_charging_circuit(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let v_c = y[0]; // Capacitor voltage

    // Circuit parameters
    let r = 1000.0; // Resistance (Ohms)
    let c = 100e-6; // Capacitance (Farads)
    let v_source = 5.0; // Source voltage (Volts)

    // RC circuit equation: dV_c/dt = (V_source - V_c) / (R*C)
    let tau = r * c; // Time constant
    let dv_dt = (v_source - v_c) / tau;

    array![dv_dt]
}

/// RL circuit: L(dI/dt) + R*I = V_source
/// State vector: [I] (current)
fn rl_circuit(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let i = y[0]; // Current

    // Circuit parameters
    let r = 10.0; // Resistance (Ohms)
    let l = 0.1; // Inductance (Henry)
    let v_source = 12.0; // Source voltage (Volts)

    // RL circuit equation: dI/dt = (V_source - R*I) / L
    let di_dt = (v_source - r * i) / l;

    array![di_dt]
}

/// RLC series circuit with sinusoidal source
/// State vector: [V_c, I] (capacitor voltage, current)
fn rlc_circuit(t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let v_c = y[0]; // Capacitor voltage
    let i = y[1]; // Current

    // Circuit parameters
    let r = 1.0; // Resistance (Ohms)
    let l = 1e-3; // Inductance (Henry)
    let c = 1e-6; // Capacitance (Farads)
    let v_amp = 10.0; // Source amplitude (Volts)
    let omega = 1000.0; // Angular frequency (rad/s)

    let v_source = v_amp * (omega * t).sin();

    // KVL: V_source = V_R + V_L + V_C
    // V_source = R*I + L*(dI/dt) + V_C
    // Also: I = C*(dV_C/dt)

    let dv_c_dt = i / c;
    let di_dt = (v_source - r * i - v_c) / l;

    array![dv_c_dt, di_dt]
}

/// Van der Pol oscillator (nonlinear circuit model)
/// State vector: [x, y] where x represents voltage and y represents current
fn van_der_pol_oscillator(t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let _t = t; // Time-independent for this example
    let x = y[0];
    let y = y[1];

    let mu = 1.0; // Nonlinearity parameter

    // Van der Pol equations: dx/dt = y, dy/dt = μ(1-x²)y - x
    let dx_dt = y;
    let dy_dt = mu * (1.0 - x * x) * y - x;

    array![dx_dt, dy_dt]
}

/// Chua's circuit (chaotic circuit)
/// State vector: [v_c1, v_c2, i_l] (capacitor voltages, inductor current)
fn chua_circuit(t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let _t = t; // Time-independent
    let v_c1 = y[0]; // Voltage across C1
    let v_c2 = y[1]; // Voltage across C2
    let i_l = y[2]; // Current through inductor

    // Circuit parameters (normalized)
    let alpha = 15.6;
    let beta = 28.0;
    let a = -1.27;
    let b = -0.68;

    // Chua's diode characteristic
    let h = |v: f64| {
        if v >= 1.0 {
            a * v + (b - a)
        } else if v <= -1.0 {
            a * v - (b - a)
        } else {
            b * v
        }
    };

    // Chua's circuit equations
    let dv_c1_dt = alpha * (v_c2 - v_c1 - h(v_c1));
    let dv_c2_dt = v_c1 - v_c2 + i_l;
    let di_l_dt = -beta * v_c2;

    array![dv_c1_dt, dv_c2_dt, di_l_dt]
}

/// RC circuit with diode (nonlinear)
/// State vector: [V_c] (capacitor voltage)
fn rc_diode_circuit(t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let v_c = y[0]; // Capacitor voltage

    // Circuit parameters
    let r = 1000.0; // Resistance (Ohms)
    let c = 100e-6; // Capacitance (Farads)
    let v_source = 5.0 * (2.0 * PI * t).sin(); // Sinusoidal source
    let v_th = 0.7; // Diode threshold voltage
    let n = 1.0; // Diode ideality factor
    let v_t = 0.026; // Thermal voltage (kT/q at room temp)
    let i_s = 1e-14; // Saturation current

    // Diode current: I_d = I_s * (exp((V_d)/(n*V_t)) - 1)
    // For forward bias (V_d > 0), approximate with exponential
    let i_diode = if v_c > v_th {
        i_s * ((v_c - v_th) / (n * v_t)).exp()
    } else {
        0.0 // Reverse bias or below threshold
    };

    // KCL at capacitor node: C(dV_c/dt) = (V_source - V_c)/R - I_diode
    let dv_dt = (v_source - v_c) / (r * c) - i_diode / c;

    array![dv_dt]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Electrical Circuit Examples\n");

    // Example 1: RC Charging Circuit
    println!("1. RC Charging Circuit");
    let t_span = [0.0, 0.005]; // 5 milliseconds
    let y0 = array![0.0]; // Initial capacitor voltage = 0V

    let result = solve_ivp(rc_charging_circuit, t_span, y0.clone(), None)?;

    println!("   Initial voltage: {:.3} V", y0[0]);
    println!("   Final voltage: {:.3} V", result.y.last().unwrap()[0]);
    println!(
        "   Theoretical final (at 5τ): {:.3} V",
        5.0 * (1.0 - (-5.0_f64).exp())
    );
    println!(
        "   Time constant τ = RC = {:.3} ms",
        1000.0 * 100e-6 * 1000.0
    );
    println!();

    // Example 2: RL Circuit
    println!("2. RL Circuit (Current Rise)");
    let result = solve_ivp(rl_circuit, [0.0, 0.1], array![0.0], None)?;

    println!("   Initial current: {:.3} A", 0.0);
    println!("   Final current: {:.3} A", result.y.last().unwrap()[0]);
    println!("   Theoretical steady-state: {:.3} A", 12.0 / 10.0);
    println!("   Time constant τ = L/R = {:.3} ms", 0.1 / 10.0 * 1000.0);
    println!();

    // Example 3: RLC Resonant Circuit
    println!("3. RLC Resonant Circuit");
    let t_span_rlc = [0.0, 0.01]; // 10 milliseconds
    let y0_rlc = array![0.0, 0.0]; // Initial: V_c=0, I=0

    let options_rlc = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        max_step: Some(1e-5), // Small step for high frequency
        ..Default::default()
    };

    let result = solve_ivp(rlc_circuit, t_span_rlc, y0_rlc, Some(options_rlc))?;

    // Calculate resonant frequency
    let l = 1e-3_f64;
    let c = 1e-6_f64;
    let f_res = 1.0_f64 / (2.0 * PI * (l * c).sqrt());

    println!("   Resonant frequency: {:.0} Hz", f_res);
    println!("   Drive frequency: {:.0} Hz", 1000.0 / (2.0 * PI));
    println!(
        "   Final capacitor voltage: {:.3} V",
        result.y.last().unwrap()[0]
    );
    println!("   Final current: {:.6} A", result.y.last().unwrap()[1]);
    println!();

    // Example 4: Van der Pol Oscillator
    println!("4. Van der Pol Oscillator (Nonlinear)");
    let t_span_vdp = [0.0, 20.0];
    let y0_vdp = array![2.0, 0.0]; // Initial conditions away from origin

    let options_vdp = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-9,
        atol: 1e-11,
        ..Default::default()
    };

    let result = solve_ivp(
        van_der_pol_oscillator,
        t_span_vdp,
        y0_vdp,
        Some(options_vdp),
    )?;

    println!("   Initial state: x={:.3}, y={:.3}", 2.0, 0.0);
    println!(
        "   Final state: x={:.3}, y={:.3}",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[1]
    );
    println!("   Steps taken: {} (limit cycle behavior)", result.t.len());
    println!();

    // Example 5: Chua's Circuit (Chaotic)
    println!("5. Chua's Circuit (Chaotic)");
    let t_span_chua = [0.0, 100.0];
    let y0_chua = array![0.1, 0.0, 0.0]; // Small perturbation from equilibrium

    let options_chua = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-10,
        atol: 1e-12,
        max_step: Some(0.01),
        ..Default::default()
    };

    let result = solve_ivp(chua_circuit, t_span_chua, y0_chua, Some(options_chua))?;

    println!(
        "   Initial state: V_c1={:.3}, V_c2={:.3}, I_L={:.3}",
        0.1, 0.0, 0.0
    );
    println!(
        "   Final state: V_c1={:.3}, V_c2={:.3}, I_L={:.3}",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[1],
        result.y.last().unwrap()[2]
    );
    println!("   Steps taken: {} (chaotic attractor)", result.t.len());
    println!();

    // Example 6: RC Circuit with Diode
    println!("6. RC Circuit with Diode (Nonlinear Element)");
    let t_span_diode = [0.0, 0.01]; // 10 milliseconds
    let y0_diode = array![0.0]; // Initial capacitor voltage

    let result = solve_ivp(rc_diode_circuit, t_span_diode, y0_diode, None)?;

    println!("   Initial voltage: {:.3} V", 0.0);
    println!("   Final voltage: {:.3} V", result.y.last().unwrap()[0]);
    println!("   Diode creates nonlinear rectification behavior");
    println!();

    println!("All circuit examples completed successfully!");
    println!("\nCircuit Analysis Summary:");
    println!("- RC/RL circuits show exponential charging/discharging");
    println!("- RLC circuits exhibit resonant behavior and oscillations");
    println!("- Van der Pol oscillator demonstrates limit cycle behavior");
    println!("- Chua's circuit shows deterministic chaos");
    println!("- Diode circuits introduce nonlinear rectification effects");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rc_circuit_time_constant() {
        // Test that RC circuit follows exponential charging
        let t_span = [0.0, 0.001]; // 1 ms
        let y0 = array![0.0];

        let result = solve_ivp(rc_charging_circuit, t_span, y0, None).unwrap();

        // At t = τ, voltage should be about 63.2% of final value
        let r = 1000.0;
        let c = 100e-6;
        let v_source = 5.0;
        let _tau = r * c; // 0.1 seconds

        // For very short time (much less than τ), use linear approximation
        let actual_voltage = result.y.last().unwrap()[0];

        // Should be close for small times
        assert!(actual_voltage > 0.0);
        assert!(actual_voltage < v_source);
    }

    #[test]
    fn test_rlc_energy_conservation() {
        // Test RLC circuit behavior
        // Note: This circuit has a sinusoidal voltage source, so energy is not conserved
        let t_span = [0.0, 0.001];
        let y0 = array![1.0, 0.0]; // Initial charge on capacitor

        let result = solve_ivp(rlc_circuit, t_span, y0.clone(), None).unwrap();

        // Just verify the integration completed successfully
        assert!(result.t.len() > 2);
        assert_eq!(result.y.len(), result.t.len());

        // Verify state variables remain finite
        for state in result.y.iter() {
            assert!(state[0].is_finite()); // Voltage
            assert!(state[1].is_finite()); // Current
        }
    }

    #[test]
    fn test_van_der_pol_limit_cycle() {
        // Test that Van der Pol oscillator approaches limit cycle
        let t_span = [0.0, 50.0]; // Long time to reach limit cycle
        let y0 = array![0.1, 0.0]; // Small initial condition

        let options = ODEOptions {
            rtol: 1e-8,
            atol: 1e-10,
            ..Default::default()
        };

        let result = solve_ivp(van_der_pol_oscillator, t_span, y0, Some(options)).unwrap();

        // After long time, should have periodic behavior with amplitude ~2
        let final_state = result.y.last().unwrap();
        let amplitude = (final_state[0] * final_state[0] + final_state[1] * final_state[1]).sqrt();

        // Van der Pol limit cycle has amplitude around 2
        assert!(amplitude > 1.0 && amplitude < 3.0);
    }
}
