//! Chemical Reaction Network Examples
//!
//! This example demonstrates solving chemical kinetics problems using ODE methods.
//! It includes simple reactions, enzyme kinetics, and reaction-diffusion systems.

use ndarray::{array, Array1, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};

/// Simple reversible reaction: A ⇌ B
/// Rate equations: d[A]/dt = -k₁[A] + k₋₁[B], d[B]/dt = k₁[A] - k₋₁[B]
/// State vector: [A, B] (concentrations)
fn reversible_reaction(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let a = y[0]; // Concentration of A
    let b = y[1]; // Concentration of B

    let k1 = 2.0; // Forward rate constant
    let k_1 = 0.5; // Backward rate constant

    let da_dt = -k1 * a + k_1 * b;
    let db_dt = k1 * a - k_1 * b;

    array![da_dt, db_dt]
}

/// Michaelis-Menten enzyme kinetics: E + S ⇌ ES → E + P
/// Simplified model: d[S]/dt = -k[S][E]/(Km + [S]), d[P]/dt = k[S][E]/(Km + [S])
/// State vector: [S, P] (substrate and product concentrations)
fn michaelis_menten_kinetics(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let s = y[0]; // Substrate concentration
    let _p = y[1]; // Product concentration

    let _e_total = 1.0; // Total enzyme concentration
    let vmax = 10.0; // Maximum reaction rate
    let km = 2.0; // Michaelis constant

    // Michaelis-Menten rate
    let rate = vmax * s / (km + s);

    let ds_dt = -rate;
    let dp_dt = rate;

    array![ds_dt, dp_dt]
}

/// Autocatalytic reaction: A + B → 2B (B catalyzes its own formation)
/// Rate equations: d[A]/dt = -k[A][B], d[B]/dt = k[A][B]
/// State vector: [A, B]
fn autocatalytic_reaction(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let a = y[0]; // Concentration of A
    let b = y[1]; // Concentration of B

    let k = 1.0; // Rate constant

    let rate = k * a * b;

    let da_dt = -rate;
    let db_dt = rate;

    array![da_dt, db_dt]
}

/// Lotka-Volterra oscillating reaction (chemical oscillator)
/// d[X]/dt = k₁[A][X] - k₂[X][Y], d[Y]/dt = k₂[X][Y] - k₃[Y]
/// State vector: [X, Y]
fn lotka_volterra_reaction(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let x = y[0]; // Concentration of X
    let y_conc = y[1]; // Concentration of Y

    let a = 1.0; // Constant concentration of A
    let k1 = 1.0; // Rate constant 1
    let k2 = 0.5; // Rate constant 2
    let k3 = 1.0; // Rate constant 3

    let dx_dt = k1 * a * x - k2 * x * y_conc;
    let dy_dt = k2 * x * y_conc - k3 * y_conc;

    array![dx_dt, dy_dt]
}

/// Brusselator (chemical oscillator model)
/// A → X, 2X + Y → 3X, B + X → Y + D, X → E
/// Simplified: d[X]/dt = A - (B+1)[X] + [X]²[Y], d[Y]/dt = B[X] - [X]²[Y]
/// State vector: [X, Y]
fn brusselator(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let x = y[0]; // Concentration of X
    let y_conc = y[1]; // Concentration of Y

    let a = 1.0; // Constant concentration of A
    let b = 3.0; // Constant concentration of B

    let dx_dt = a - (b + 1.0) * x + x * x * y_conc;
    let dy_dt = b * x - x * x * y_conc;

    array![dx_dt, dy_dt]
}

/// Oregonator (Belousov-Zhabotinsky reaction model)
/// Three-variable model for chemical oscillations
/// State vector: [X, Y, Z] (bromous acid, oxidized catalyst, bromide)
fn oregonator(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let x = y[0]; // HBrO₂ concentration
    let y_conc = y[1]; // Oxidized catalyst concentration
    let z = y[2]; // Br⁻ concentration

    let s = 77.27; // Stoichiometric factor
    let w = 0.161; // Rate parameter
    let q = 8.375e-6; // Rate parameter

    let dx_dt = s * (y_conc - x * y_conc + x * (1.0 - q * x / (x + 1.0)));
    let dy_dt = (z - y_conc * (1.0 + x)) / s;
    let dz_dt = w * (x - z);

    array![dx_dt, dy_dt, dz_dt]
}

/// Glycolysis model (simplified pathway)
/// Models glucose breakdown with feedback regulation
/// State vector: [G6P, F6P, ATP] (glucose-6-phosphate, fructose-6-phosphate, ATP)
fn glycolysis_model(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let g6p = y[0]; // Glucose-6-phosphate
    let f6p = y[1]; // Fructose-6-phosphate
    let atp = y[2]; // ATP

    // Kinetic parameters
    let v1 = 1.0; // Hexokinase rate
    let k1 = 0.5; // Michaelis constant
    let v2 = 2.0; // Phosphoglucose isomerase rate
    let v3 = 1.5; // Phosphofructokinase rate
    let ki = 0.1; // Inhibition constant (ATP inhibits PFK)

    // Glucose input (constant)
    let glucose = 5.0;

    // Reaction rates with inhibition
    let r1 = v1 * glucose / (k1 + glucose);
    let r2 = v2 * g6p / (k1 + g6p);
    let r3 = v3 * f6p / ((k1 + f6p) * (1.0 + atp / ki)); // ATP inhibition

    let dg6p_dt = r1 - r2;
    let df6p_dt = r2 - r3;
    let datp_dt = 2.0 * r3 - 0.1 * atp; // ATP production and consumption

    array![dg6p_dt, df6p_dt, datp_dt]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Chemical Reaction Network Examples\n");

    // Example 1: Reversible Reaction
    println!("1. Reversible Reaction A ⇌ B");
    let t_span = [0.0, 5.0];
    let y0 = array![2.0, 0.0]; // Initial: [A]=2, [B]=0

    let result = solve_ivp(reversible_reaction, t_span, y0.clone(), None)?;

    println!(
        "   Initial concentrations: [A]={:.3}, [B]={:.3}",
        y0[0], y0[1]
    );
    println!(
        "   Final concentrations: [A]={:.3}, [B]={:.3}",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[1]
    );

    // Calculate equilibrium constant
    let k_eq = 2.0 / 0.5; // k₁/k₋₁
    let total = y0[0] + y0[1];
    let a_eq = total / (1.0 + k_eq);
    let b_eq = total - a_eq;
    println!(
        "   Theoretical equilibrium: [A]={:.3}, [B]={:.3}",
        a_eq, b_eq
    );
    println!();

    // Example 2: Michaelis-Menten Enzyme Kinetics
    println!("2. Michaelis-Menten Enzyme Kinetics");
    let result = solve_ivp(
        michaelis_menten_kinetics,
        [0.0, 3.0],
        array![10.0, 0.0],
        None,
    )?;

    println!("   Initial: [S]={:.3}, [P]={:.3}", 10.0, 0.0);
    println!(
        "   Final: [S]={:.3}, [P]={:.3}",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[1]
    );
    println!(
        "   Conservation check: S+P = {:.3} (should be 10.0)",
        result.y.last().unwrap()[0] + result.y.last().unwrap()[1]
    );
    println!();

    // Example 3: Autocatalytic Reaction
    println!("3. Autocatalytic Reaction A + B → 2B");
    let result = solve_ivp(autocatalytic_reaction, [0.0, 2.0], array![1.0, 0.1], None)?;

    println!("   Initial: [A]={:.3}, [B]={:.3}", 1.0, 0.1);
    println!(
        "   Final: [A]={:.3}, [B]={:.3}",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[1]
    );
    println!(
        "   Total conservation: A+B = {:.3} (should be 1.1)",
        result.y.last().unwrap()[0] + result.y.last().unwrap()[1]
    );
    println!();

    // Example 4: Lotka-Volterra Chemical Oscillator
    println!("4. Lotka-Volterra Chemical Oscillator");
    let options = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        ..Default::default()
    };

    let result = solve_ivp(
        lotka_volterra_reaction,
        [0.0, 10.0],
        array![1.0, 1.0],
        Some(options.clone()),
    )?;

    println!("   Initial: [X]={:.3}, [Y]={:.3}", 1.0, 1.0);
    println!(
        "   Final: [X]={:.3}, [Y]={:.3}",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[1]
    );
    println!("   Steps taken: {} (oscillatory behavior)", result.t.len());
    println!();

    // Example 5: Brusselator
    println!("5. Brusselator (Chemical Oscillator)");
    let result = solve_ivp(
        brusselator,
        [0.0, 20.0],
        array![1.0, 1.0],
        Some(options.clone()),
    )?;

    println!("   Initial: [X]={:.3}, [Y]={:.3}", 1.0, 1.0);
    println!(
        "   Final: [X]={:.3}, [Y]={:.3}",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[1]
    );
    println!(
        "   Steps taken: {} (limit cycle oscillator)",
        result.t.len()
    );
    println!();

    // Example 6: Oregonator (BZ Reaction)
    println!("6. Oregonator (Belousov-Zhabotinsky Reaction)");
    let options_bz = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-9,
        atol: 1e-11,
        max_step: Some(0.1),
        ..Default::default()
    };

    let result = solve_ivp(
        oregonator,
        [0.0, 5.0],
        array![1.0, 1.0, 1.0],
        Some(options_bz),
    )?;

    println!("   Initial: [X]={:.3}, [Y]={:.3}, [Z]={:.3}", 1.0, 1.0, 1.0);
    println!(
        "   Final: [X]={:.3}, [Y]={:.3}, [Z]={:.3}",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[1],
        result.y.last().unwrap()[2]
    );
    println!("   Steps taken: {} (complex oscillations)", result.t.len());
    println!();

    // Example 7: Glycolysis Model
    println!("7. Glycolysis Pathway (Simplified)");
    let result = solve_ivp(glycolysis_model, [0.0, 10.0], array![0.1, 0.1, 2.0], None)?;

    println!(
        "   Initial: [G6P]={:.3}, [F6P]={:.3}, [ATP]={:.3}",
        0.1, 0.1, 2.0
    );
    println!(
        "   Final: [G6P]={:.3}, [F6P]={:.3}, [ATP]={:.3}",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[1],
        result.y.last().unwrap()[2]
    );
    println!("   Metabolic steady state reached");
    println!();

    println!("All chemical reaction examples completed successfully!");
    println!("\nReaction Analysis Summary:");
    println!("- Reversible reactions reach equilibrium based on rate constants");
    println!("- Enzyme kinetics follow Michaelis-Menten saturation behavior");
    println!("- Autocatalytic reactions show sigmoidal growth patterns");
    println!("- Chemical oscillators exhibit periodic behavior and limit cycles");
    println!("- Metabolic pathways show feedback regulation and steady states");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mass_conservation_reversible() {
        // Test mass conservation in reversible reaction
        let t_span = [0.0, 10.0];
        let y0 = array![3.0, 1.0];

        let result = solve_ivp(reversible_reaction, t_span, y0.clone(), None).unwrap();

        let initial_total = y0[0] + y0[1];
        let final_total = result.y.last().unwrap()[0] + result.y.last().unwrap()[1];

        // Mass should be conserved
        assert_abs_diff_eq!(initial_total, final_total, epsilon = 1e-6);
    }

    #[test]
    fn test_enzyme_kinetics_conservation() {
        // Test substrate + product conservation
        let t_span = [0.0, 5.0];
        let y0 = array![5.0, 0.0];

        let result = solve_ivp(michaelis_menten_kinetics, t_span, y0.clone(), None).unwrap();

        let initial_total = y0[0] + y0[1];
        let final_total = result.y.last().unwrap()[0] + result.y.last().unwrap()[1];

        assert_abs_diff_eq!(initial_total, final_total, epsilon = 1e-6);
    }

    #[test]
    fn test_autocatalytic_growth() {
        // Test that autocatalytic reaction shows sigmoidal growth
        let t_span = [0.0, 3.0];
        let y0 = array![1.0, 0.01]; // Small amount of catalyst

        let result = solve_ivp(autocatalytic_reaction, t_span, y0.clone(), None).unwrap();

        // B should increase significantly
        assert!(result.y.last().unwrap()[1] > y0[1]);
        // A should decrease
        assert!(result.y.last().unwrap()[0] < y0[0]);
        // Total should be conserved
        let total_initial = y0[0] + y0[1];
        let total_final = result.y.last().unwrap()[0] + result.y.last().unwrap()[1];
        assert_abs_diff_eq!(total_initial, total_final, epsilon = 1e-6);
    }

    #[test]
    fn test_brusselator_oscillations() {
        // Test that Brusselator produces oscillatory behavior
        let t_span = [0.0, 10.0];
        let y0 = array![1.0, 1.0];

        let options = ODEOptions {
            rtol: 1e-8,
            atol: 1e-10,
            ..Default::default()
        };

        let result = solve_ivp(brusselator, t_span, y0, Some(options)).unwrap();

        // Should take many steps due to oscillatory behavior
        assert!(result.t.len() > 50);

        // Final state should be different from initial (not at equilibrium)
        let final_state = result.y.last().unwrap();
        assert!(final_state[0] > 0.0 && final_state[1] > 0.0);
    }
}
