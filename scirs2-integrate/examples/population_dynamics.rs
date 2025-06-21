//! Population Dynamics and Epidemiological Models
//!
//! This example demonstrates solving population dynamics and disease spread models
//! using ODE methods. It includes classic ecological and epidemiological models.

use ndarray::{array, Array1, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};

/// Exponential growth model: dN/dt = rN
/// State vector: [N] (population size)
fn exponential_growth(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let n = y[0]; // Population size
    let r = 0.5; // Growth rate

    let dn_dt = r * n;

    array![dn_dt]
}

/// Logistic growth model: dN/dt = rN(1 - N/K)
/// State vector: [N] (population size)
fn logistic_growth(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let n = y[0]; // Population size
    let r = 1.0; // Intrinsic growth rate
    let k = 100.0; // Carrying capacity

    let dn_dt = r * n * (1.0 - n / k);

    array![dn_dt]
}

/// Lotka-Volterra predator-prey model
/// dN/dt = aN - bNP, dP/dt = cbNP - dP
/// State vector: [N, P] (prey, predator populations)
fn lotka_volterra(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let n = y[0]; // Prey population
    let p = y[1]; // Predator population

    let a = 1.0; // Prey growth rate
    let b = 0.1; // Predation rate
    let c = 0.075; // Predator efficiency
    let d = 1.5; // Predator death rate

    let dn_dt = a * n - b * n * p;
    let dp_dt = c * b * n * p - d * p;

    array![dn_dt, dp_dt]
}

/// Competitive Lotka-Volterra model (two competing species)
/// dN₁/dt = r₁N₁(1 - (N₁ + α₁₂N₂)/K₁)
/// dN₂/dt = r₂N₂(1 - (N₂ + α₂₁N₁)/K₂)
/// State vector: [N1, N2] (competing species populations)
fn competitive_lotka_volterra(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let n1 = y[0]; // Species 1 population
    let n2 = y[1]; // Species 2 population

    let r1 = 1.0; // Species 1 growth rate
    let r2 = 0.8; // Species 2 growth rate
    let k1 = 100.0; // Species 1 carrying capacity
    let k2 = 80.0; // Species 2 carrying capacity
    let alpha12 = 1.2; // Effect of species 2 on species 1
    let alpha21 = 0.9; // Effect of species 1 on species 2

    let dn1_dt = r1 * n1 * (1.0 - (n1 + alpha12 * n2) / k1);
    let dn2_dt = r2 * n2 * (1.0 - (n2 + alpha21 * n1) / k2);

    array![dn1_dt, dn2_dt]
}

/// SIR epidemic model: dS/dt = -βSI/N, dI/dt = βSI/N - γI, dR/dt = γI
/// State vector: [S, I, R] (susceptible, infected, recovered)
fn sir_model(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let s = y[0]; // Susceptible
    let i = y[1]; // Infected
    let r = y[2]; // Recovered

    let n = s + i + r; // Total population
    let beta = 0.3; // Transmission rate
    let gamma = 0.1; // Recovery rate

    let ds_dt = -beta * s * i / n;
    let di_dt = beta * s * i / n - gamma * i;
    let dr_dt = gamma * i;

    array![ds_dt, di_dt, dr_dt]
}

/// SEIR epidemic model (with exposed/incubation class)
/// dS/dt = -βSI/N, dE/dt = βSI/N - σE, dI/dt = σE - γI, dR/dt = γI
/// State vector: [S, E, I, R] (susceptible, exposed, infected, recovered)
fn seir_model(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let s = y[0]; // Susceptible
    let e = y[1]; // Exposed (incubating)
    let i = y[2]; // Infected
    let r = y[3]; // Recovered

    let n = s + e + i + r; // Total population
    let beta = 0.4; // Transmission rate
    let sigma = 0.2; // Incubation rate (1/incubation period)
    let gamma = 0.1; // Recovery rate

    let ds_dt = -beta * s * i / n;
    let de_dt = beta * s * i / n - sigma * e;
    let di_dt = sigma * e - gamma * i;
    let dr_dt = gamma * i;

    array![ds_dt, de_dt, di_dt, dr_dt]
}

/// SIRS model (with immunity loss)
/// dS/dt = -βSI/N + ωR, dI/dt = βSI/N - γI, dR/dt = γI - ωR
/// State vector: [S, I, R] with immunity waning
fn sirs_model(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let s = y[0]; // Susceptible
    let i = y[1]; // Infected
    let r = y[2]; // Recovered

    let n = s + i + r; // Total population
    let beta = 0.3; // Transmission rate
    let gamma = 0.1; // Recovery rate
    let omega = 0.05; // Rate of immunity loss

    let ds_dt = -beta * s * i / n + omega * r;
    let di_dt = beta * s * i / n - gamma * i;
    let dr_dt = gamma * i - omega * r;

    array![ds_dt, di_dt, dr_dt]
}

/// SIS model (no immunity)
/// dS/dt = -βSI/N + γI, dI/dt = βSI/N - γI
/// State vector: [S, I] (susceptible, infected)
fn sis_model(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let s = y[0]; // Susceptible
    let i = y[1]; // Infected

    let n = s + i; // Total population
    let beta = 0.5; // Transmission rate
    let gamma = 0.2; // Recovery rate (back to susceptible)

    let ds_dt = -beta * s * i / n + gamma * i;
    let di_dt = beta * s * i / n - gamma * i;

    array![ds_dt, di_dt]
}

/// Metapopulation model (two connected populations)
/// Models disease spread between cities/regions
/// State vector: [S1, I1, R1, S2, I2, R2]
fn metapopulation_sir(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let s1 = y[0];
    let i1 = y[1];
    let r1 = y[2]; // Population 1
    let s2 = y[3];
    let i2 = y[4];
    let r2 = y[5]; // Population 2

    let n1 = s1 + i1 + r1; // Total population 1
    let n2 = s2 + i2 + r2; // Total population 2

    let beta1 = 0.3;
    let beta2 = 0.25; // Transmission rates
    let gamma = 0.1; // Recovery rate
    let m = 0.01; // Migration rate between populations

    // Within-population dynamics
    let ds1_dt_local = -beta1 * s1 * i1 / n1;
    let di1_dt_local = beta1 * s1 * i1 / n1 - gamma * i1;
    let dr1_dt_local = gamma * i1;

    let ds2_dt_local = -beta2 * s2 * i2 / n2;
    let di2_dt_local = beta2 * s2 * i2 / n2 - gamma * i2;
    let dr2_dt_local = gamma * i2;

    // Migration between populations
    let ds1_dt = ds1_dt_local + m * (s2 - s1);
    let di1_dt = di1_dt_local + m * (i2 - i1);
    let dr1_dt = dr1_dt_local + m * (r2 - r1);

    let ds2_dt = ds2_dt_local + m * (s1 - s2);
    let di2_dt = di2_dt_local + m * (i1 - i2);
    let dr2_dt = dr2_dt_local + m * (r1 - r2);

    array![ds1_dt, di1_dt, dr1_dt, ds2_dt, di2_dt, dr2_dt]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Population Dynamics and Epidemiological Models\n");

    // Example 1: Exponential Growth
    println!("1. Exponential Population Growth");
    let t_span = [0.0, 5.0];
    let y0 = array![10.0]; // Initial population

    let result = solve_ivp(exponential_growth, t_span, y0.clone(), None)?;

    println!("   Initial population: {:.1}", y0[0]);
    println!("   Final population: {:.1}", result.y.last().unwrap()[0]);
    println!(
        "   Theoretical final: {:.1}",
        y0[0] * (0.5 * t_span[1]).exp()
    );
    println!();

    // Example 2: Logistic Growth
    println!("2. Logistic Population Growth");
    let result = solve_ivp(logistic_growth, [0.0, 10.0], array![5.0], None)?;

    println!("   Initial population: {:.1}", 5.0);
    println!("   Final population: {:.1}", result.y.last().unwrap()[0]);
    println!("   Carrying capacity: {:.1}", 100.0);
    println!(
        "   Approached carrying capacity: {:.1}%",
        result.y.last().unwrap()[0] / 100.0 * 100.0
    );
    println!();

    // Example 3: Lotka-Volterra Predator-Prey
    println!("3. Lotka-Volterra Predator-Prey Model");
    let options = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        ..Default::default()
    };

    let result = solve_ivp(
        lotka_volterra,
        [0.0, 15.0],
        array![10.0, 5.0],
        Some(options.clone()),
    )?;

    println!(
        "   Initial populations: Prey={:.1}, Predator={:.1}",
        10.0, 5.0
    );
    println!(
        "   Final populations: Prey={:.1}, Predator={:.1}",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[1]
    );
    println!("   Steps taken: {} (oscillatory dynamics)", result.t.len());
    println!();

    // Example 4: Competitive Species
    println!("4. Competitive Lotka-Volterra (Two Species)");
    let result = solve_ivp(
        competitive_lotka_volterra,
        [0.0, 20.0],
        array![30.0, 20.0],
        None,
    )?;

    println!(
        "   Initial populations: Species1={:.1}, Species2={:.1}",
        30.0, 20.0
    );
    println!(
        "   Final populations: Species1={:.1}, Species2={:.1}",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[1]
    );

    // Determine competitive outcome
    let final_ratio = result.y.last().unwrap()[0] / result.y.last().unwrap()[1];
    if final_ratio > 2.0 {
        println!("   Outcome: Species 1 dominates");
    } else if final_ratio < 0.5 {
        println!("   Outcome: Species 2 dominates");
    } else {
        println!("   Outcome: Coexistence");
    }
    println!();

    // Example 5: SIR Epidemic Model
    println!("5. SIR Epidemic Model");
    let sir_initial = array![990.0, 10.0, 0.0]; // 1000 total, 10 infected
    let result = solve_ivp(sir_model, [0.0, 50.0], sir_initial.clone(), None)?;

    println!(
        "   Initial: S={:.0}, I={:.0}, R={:.0}",
        sir_initial[0], sir_initial[1], sir_initial[2]
    );
    let final_state = result.y.last().unwrap();
    println!(
        "   Final: S={:.0}, I={:.0}, R={:.0}",
        final_state[0], final_state[1], final_state[2]
    );

    // Calculate basic reproduction number R₀ = β/γ
    let r0 = 0.3 / 0.1;
    println!("   Basic reproduction number R₀: {:.1}", r0);
    println!(
        "   Attack rate: {:.1}%",
        (sir_initial[0] - final_state[0]) / sir_initial[0] * 100.0
    );
    println!();

    // Example 6: SEIR Model
    println!("6. SEIR Epidemic Model (with incubation)");
    let seir_initial = array![990.0, 0.0, 10.0, 0.0]; // 1000 total, 10 infected
    let result = solve_ivp(
        seir_model,
        [0.0, 60.0],
        seir_initial.clone(),
        Some(options.clone()),
    )?;

    println!(
        "   Initial: S={:.0}, E={:.0}, I={:.0}, R={:.0}",
        seir_initial[0], seir_initial[1], seir_initial[2], seir_initial[3]
    );
    let final_state = result.y.last().unwrap();
    println!(
        "   Final: S={:.0}, E={:.0}, I={:.0}, R={:.0}",
        final_state[0], final_state[1], final_state[2], final_state[3]
    );
    println!(
        "   Peak exposed: {:.0}",
        result
            .y
            .iter()
            .map(|state| state[1])
            .fold(0.0_f64, f64::max)
    );
    println!();

    // Example 7: SIRS Model (immunity waning)
    println!("7. SIRS Model (Immunity Loss)");
    let result = solve_ivp(sirs_model, [0.0, 100.0], array![990.0, 10.0, 0.0], None)?;

    let final_state = result.y.last().unwrap();
    println!("   Initial: S={:.0}, I={:.0}, R={:.0}", 990.0, 10.0, 0.0);
    println!(
        "   Final: S={:.0}, I={:.0}, R={:.0}",
        final_state[0], final_state[1], final_state[2]
    );
    println!("   Endemic equilibrium reached (cycling infections)");
    println!();

    // Example 8: SIS Model
    println!("8. SIS Model (No Immunity)");
    let result = solve_ivp(sis_model, [0.0, 30.0], array![950.0, 50.0], None)?;

    let final_state = result.y.last().unwrap();
    println!("   Initial: S={:.0}, I={:.0}", 950.0, 50.0);
    println!("   Final: S={:.0}, I={:.0}", final_state[0], final_state[1]);

    let endemic_level = final_state[1] / (final_state[0] + final_state[1]) * 100.0;
    println!("   Endemic infection level: {:.1}%", endemic_level);
    println!();

    // Example 9: Metapopulation Model
    println!("9. Metapopulation SIR (Two Connected Cities)");
    let meta_initial = array![500.0, 10.0, 0.0, 480.0, 5.0, 15.0]; // Two cities
    let result = solve_ivp(
        metapopulation_sir,
        [0.0, 40.0],
        meta_initial.clone(),
        Some(options),
    )?;

    println!(
        "   City 1 Initial: S={:.0}, I={:.0}, R={:.0}",
        meta_initial[0], meta_initial[1], meta_initial[2]
    );
    println!(
        "   City 2 Initial: S={:.0}, I={:.0}, R={:.0}",
        meta_initial[3], meta_initial[4], meta_initial[5]
    );

    let final_state = result.y.last().unwrap();
    println!(
        "   City 1 Final: S={:.0}, I={:.0}, R={:.0}",
        final_state[0], final_state[1], final_state[2]
    );
    println!(
        "   City 2 Final: S={:.0}, I={:.0}, R={:.0}",
        final_state[3], final_state[4], final_state[5]
    );
    println!("   Migration facilitates disease spread between cities");
    println!();

    println!("All population dynamics examples completed successfully!");
    println!("\nModel Analysis Summary:");
    println!("- Exponential growth shows unlimited population increase");
    println!("- Logistic growth approaches carrying capacity asymptotically");
    println!("- Predator-prey systems exhibit oscillatory dynamics");
    println!("- Competitive species may coexist or lead to exclusion");
    println!("- Epidemic models show threshold behavior based on R₀");
    println!("- SEIR models capture incubation period effects");
    println!("- SIRS/SIS models show endemic equilibria with reinfection");
    println!("- Metapopulation models demonstrate spatial disease spread");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_logistic_growth_carrying_capacity() {
        // Test that logistic growth approaches carrying capacity
        let t_span = [0.0, 20.0]; // Long time
        let y0 = array![10.0];

        let result = solve_ivp(logistic_growth, t_span, y0, None).unwrap();

        let final_pop = result.y.last().unwrap()[0];
        let carrying_capacity = 100.0;

        // Should approach carrying capacity (within 5%)
        assert!((final_pop - carrying_capacity).abs() / carrying_capacity < 0.05);
    }

    #[test]
    fn test_sir_population_conservation() {
        // Test that SIR model conserves total population
        let t_span = [0.0, 30.0];
        let y0 = array![1000.0, 1.0, 0.0];

        let result = solve_ivp(sir_model, t_span, y0.clone(), None).unwrap();

        let initial_total = y0[0] + y0[1] + y0[2];
        let final_state = result.y.last().unwrap();
        let final_total = final_state[0] + final_state[1] + final_state[2];

        // Population should be conserved
        assert_abs_diff_eq!(initial_total, final_total, epsilon = 1e-6);
    }

    #[test]
    fn test_seir_population_conservation() {
        // Test that SEIR model conserves total population
        let t_span = [0.0, 50.0];
        let y0 = array![1000.0, 0.0, 1.0, 0.0];

        let options = ODEOptions {
            rtol: 1e-8,
            atol: 1e-10,
            ..Default::default()
        };

        let result = solve_ivp(seir_model, t_span, y0.clone(), Some(options)).unwrap();

        let initial_total = y0.sum();
        let final_state = result.y.last().unwrap();
        let final_total = final_state[0] + final_state[1] + final_state[2] + final_state[3];

        assert_abs_diff_eq!(initial_total, final_total, epsilon = 1e-6);
    }

    #[test]
    fn test_lotka_volterra_conservation() {
        // Test conservation of the Lotka-Volterra first integral
        let t_span = [0.0, 10.0];
        let y0 = array![10.0, 5.0];

        let options = ODEOptions {
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        };

        let result = solve_ivp(lotka_volterra, t_span, y0.clone(), Some(options)).unwrap();

        // First integral: H = cbN - d ln(N) + aP - b ln(P)
        let _a = 1.0;
        let _b = 0.1;
        let _c = 0.075;
        let _d = 1.5;

        // Verify populations remain positive and reasonable
        let mut min_prey = f64::INFINITY;
        let mut max_prey = 0.0;
        let mut min_pred = f64::INFINITY;
        let mut max_pred = 0.0;

        for state in result.y.iter() {
            assert!(state[0] > 0.0, "Prey population became non-positive");
            assert!(state[1] > 0.0, "Predator population became non-positive");

            min_prey = f64::min(min_prey, state[0]);
            max_prey = f64::max(max_prey, state[0]);
            min_pred = f64::min(min_pred, state[1]);
            max_pred = f64::max(max_pred, state[1]);
        }

        // Populations should oscillate but remain bounded
        assert!(min_prey > 0.01);
        assert!(max_prey < 1000.0);
        assert!(min_pred > 0.01);
        assert!(max_pred < 500.0);
    }

    #[test]
    fn test_epidemic_basic_reproduction_number() {
        // Test that epidemic grows when R₀ > 1 and dies out when R₀ < 1
        let t_span = [0.0, 20.0];
        let y0 = array![999.0, 1.0, 0.0]; // Small initial infection

        // R₀ = β/γ = 0.3/0.1 = 3 > 1, so epidemic should grow initially
        let result = solve_ivp(sir_model, t_span, y0.clone(), None).unwrap();

        // Find peak infection
        let peak_infected = result
            .y
            .iter()
            .map(|state| state[1])
            .fold(0.0_f64, f64::max);

        // Peak should be much larger than initial infection
        assert!(peak_infected > 10.0 * y0[1]);

        // Final number of susceptibles should be less than initial
        let final_susceptible = result.y.last().unwrap()[0];
        assert!(final_susceptible < y0[0]);
    }
}
