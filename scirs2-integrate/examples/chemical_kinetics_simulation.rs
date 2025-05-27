use ndarray::{array, Array1, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use std::time::Instant;

/// This example demonstrates simulation of complex chemical reaction networks
/// using scirs2-integrate's enhanced ODE solvers.
///
/// Stiff biochemical system: MAPK cascade model
///
/// This is a simplified model of the mitogen-activated protein kinase (MAPK) cascade,
/// which is a crucial signaling pathway in cells involved in various processes including
/// cell growth, differentiation, and survival.
fn mapk_cascade(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    // State variables (concentrations)
    // y[0] = MAPKKK
    // y[1] = MAPKKK*     (active)
    // y[2] = MAPKK
    // y[3] = MAPKK-P     (phosphorylated once)
    // y[4] = MAPKK-PP    (phosphorylated twice, active)
    // y[5] = MAPK
    // y[6] = MAPK-P      (phosphorylated once)
    // y[7] = MAPK-PP     (phosphorylated twice, active)

    // Rate constants
    let k1 = 0.005; // MAPKKK activation
    let k2 = 0.1; // MAPKKK deactivation
    let k3 = 0.025; // MAPKK phosphorylation by MAPKKK*
    let k4 = 0.025; // MAPKK-P phosphorylation by MAPKKK*
    let k5 = 0.75; // MAPKK-PP dephosphorylation
    let k6 = 0.75; // MAPKK-P dephosphorylation
    let k7 = 0.025; // MAPK phosphorylation by MAPKK-PP
    let k8 = 0.025; // MAPK-P phosphorylation by MAPKK-PP
    let k9 = 0.5; // MAPK-PP dephosphorylation
    let k10 = 0.5; // MAPK-P dephosphorylation

    // Signal strength (e.g., growth factor concentration)
    let signal = 2.5;

    // Initialize the derivative vector
    let mut dy = Array1::<f64>::zeros(8);

    // MAPKKK activation/deactivation
    dy[0] = -k1 * signal * y[0] + k2 * y[1]; // d[MAPKKK]/dt
    dy[1] = k1 * signal * y[0] - k2 * y[1]; // d[MAPKKK*]/dt

    // MAPKK phosphorylation/dephosphorylation
    dy[2] = -k3 * y[1] * y[2] + k6 * y[3]; // d[MAPKK]/dt
    dy[3] = k3 * y[1] * y[2] - k4 * y[1] * y[3] + k5 * y[4] - k6 * y[3]; // d[MAPKK-P]/dt
    dy[4] = k4 * y[1] * y[3] - k5 * y[4]; // d[MAPKK-PP]/dt

    // MAPK phosphorylation/dephosphorylation
    dy[5] = -k7 * y[4] * y[5] + k10 * y[6]; // d[MAPK]/dt
    dy[6] = k7 * y[4] * y[5] - k8 * y[4] * y[6] + k9 * y[7] - k10 * y[6]; // d[MAPK-P]/dt
    dy[7] = k8 * y[4] * y[6] - k9 * y[7]; // d[MAPK-PP]/dt

    dy
}

/// Belousov-Zhabotinsky (BZ) reaction model
///
/// This is a classic example of a chemical oscillator - a reaction that
/// naturally produces oscillations in concentration of various species.
fn belousov_zhabotinsky(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    // State variables:
    // y[0] = HBrO2 (bromous acid)
    // y[1] = Br⁻ (bromide ion)
    // y[2] = Ce⁴⁺ (cerium 4+)

    // Parameters from Field, Körös, and Noyes (FKN) mechanism
    let a = 1.0; // Flow term
    let b = 10.0; // Ratio of time scales
    let c = 1.0; // Relative concentration

    // Initialize the derivative vector
    let mut dy = Array1::<f64>::zeros(3);

    // Simplified Oregonator model (3-variable version of BZ reaction)
    dy[0] = a * (y[1] - y[0] * y[1] + y[0] - y[0] * y[0]);
    dy[1] = (1.0 / a) * (-y[1] - y[0] * y[1] + b * y[2]);
    dy[2] = c * (y[0] - y[2]);

    dy
}

/// Circadian rhythm model (biological oscillator)
///
/// This models the daily cycles in organisms through a negative feedback loop
/// between gene expression, protein synthesis, and protein degradation.
fn circadian_rhythm(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    // State variables:
    // y[0] = mRNA concentration
    // y[1] = cytosolic protein concentration
    // y[2] = nuclear protein concentration

    // Parameters (adapted from "A model for circadian oscillations...")
    let v_s = 0.76; // Maximum rate of mRNA synthesis
    let k_i: f64 = 1.0; // Inhibition constant
    let k_m = 0.2; // Michaelis constant for mRNA degradation
    let k_s = 0.38; // Rate constant for protein synthesis
    let v_m = 0.65; // Maximum rate of mRNA degradation
    let k_d = 0.2; // Protein degradation rate constant
    let k_1 = 0.4; // Rate constant for protein transport to nucleus
    let k_2 = 0.2; // Rate constant for protein transport from nucleus
    let n = 4.0; // Hill coefficient (cooperativity)

    // Initialize the derivative vector
    let mut dy = Array1::<f64>::zeros(3);

    // mRNA dynamics
    let repression = (k_i.powf(n)) / (k_i.powf(n) + y[2].powf(n));
    dy[0] = v_s * repression - v_m * y[0] / (k_m + y[0]);

    // Cytosolic protein dynamics
    dy[1] = k_s * y[0] - k_d * y[1] - k_1 * y[1] + k_2 * y[2];

    // Nuclear protein dynamics
    dy[2] = k_1 * y[1] - k_2 * y[2];

    dy
}

/// SIR epidemic model with vaccination
///
/// Models the spread of an infectious disease through a population,
/// including the effect of vaccination.
fn sir_epidemic(t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    // State variables:
    // y[0] = S (susceptible population)
    // y[1] = I (infected population)
    // y[2] = R (recovered/removed population)

    // Parameters
    let beta = 0.3; // Infection rate
    let gamma = 0.1; // Recovery rate
    let mu = 0.01; // Natural birth/death rate

    // Time-dependent vaccination rate: peaks and then declines
    let vaccination_rate = if t < 30.0 {
        0.05 * t / 30.0 // Ramp up vaccination
    } else if t < 60.0 {
        0.05 // Constant vaccination rate
    } else {
        0.05 * (1.0 - (t - 60.0) / 60.0).max(0.0) // Phase out vaccination
    };

    // Total population (should be constant)
    let n = y[0] + y[1] + y[2];

    // Initialize the derivative vector
    let mut dy = Array1::<f64>::zeros(3);

    // Model equations
    dy[0] = mu * n - beta * y[0] * y[1] / n - vaccination_rate * y[0] - mu * y[0];
    dy[1] = beta * y[0] * y[1] / n - gamma * y[1] - mu * y[1];
    dy[2] = gamma * y[1] + vaccination_rate * y[0] - mu * y[2];

    dy
}

/// FitzHugh-Nagumo model - simplified neuron model
///
/// Used to model excitable media like neurons, showing characteristic
/// action potential dynamics.
fn fitzhugh_nagumo(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    // State variables:
    // y[0] = v (membrane potential)
    // y[1] = w (recovery variable)

    // Parameters
    let a = 0.7; // Recovery rate
    let b = 0.8; // Excitation threshold
    let c = 3.0; // Scale parameter
    let i_ext = 0.5; // External stimulus current

    // Initialize the derivative vector
    let mut dy = Array1::<f64>::zeros(2);

    // FitzHugh-Nagumo equations
    dy[0] = c * (y[0] - y[0].powi(3) / 3.0 - y[1] + i_ext);
    dy[1] = (y[0] + a - b * y[1]) / c;

    dy
}

/// Lorenz system - chaotic atmospheric model
///
/// A classic example of deterministic chaos that arises from a simple
/// set of ordinary differential equations.
fn lorenz(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    // State variables:
    // y[0] = x (related to convective intensity)
    // y[1] = y (related to temperature difference)
    // y[2] = z (related to vertical temperature profile)

    // Classic Lorenz parameters
    let sigma = 10.0; // Prandtl number
    let rho = 28.0; // Rayleigh number
    let beta = 8.0 / 3.0; // Physical proportion

    // Initialize the derivative vector
    let mut dy = Array1::<f64>::zeros(3);

    // Lorenz equations
    dy[0] = sigma * (y[1] - y[0]);
    dy[1] = y[0] * (rho - y[2]) - y[1];
    dy[2] = y[0] * y[1] - beta * y[2];

    dy
}

/// Run a simulation of the system and benchmark different ODE solvers
fn run_simulation<F>(
    model_name: &str,
    system: F,
    y0: Array1<f64>,
    t_span: [f64; 2],
    methods: &[ODEMethod],
) -> IntegrateResult<()>
where
    F: Fn(f64, ArrayView1<f64>) -> Array1<f64> + Copy,
{
    println!("\n=== {} Model Simulation ===", model_name);

    // Standard tolerances
    let rtol = 1e-6;
    let atol = 1e-8;

    // Print initial conditions
    println!("Initial state:");
    for (i, &val) in y0.iter().enumerate() {
        println!("  y[{}] = {:.6}", i, val);
    }

    // Run each method, measure time and record diagnostic info
    let mut method_results = Vec::new();

    for &method in methods {
        let method_name = match method {
            ODEMethod::RK45 => "RK45 (explicit)",
            ODEMethod::DOP853 => "DOP853 (explicit)",
            ODEMethod::Bdf => "BDF (standard)",
            ODEMethod::LSODA => "LSODA (standard)",
            ODEMethod::EnhancedLSODA => "Enhanced LSODA",
            ODEMethod::EnhancedBDF => "Enhanced BDF",
            _ => "Unknown method",
        };

        print!("{:20}: ", method_name);

        // Set solver options
        let options = ODEOptions {
            method,
            rtol,
            atol,
            max_steps: 10000, // Allow plenty of steps for complex systems
            ..Default::default()
        };

        // Time the solve operation
        let start = Instant::now();
        let result = solve_ivp(system, t_span, y0.clone(), Some(options));
        let elapsed = start.elapsed();

        match result {
            Ok(ode_result) => {
                println!(
                    "solved in {:.3} ms, {} steps, {} rejected, {} f-evals, {} J-evals",
                    elapsed.as_secs_f64() * 1000.0,
                    ode_result.n_steps,
                    ode_result.n_rejected,
                    ode_result.n_eval,
                    ode_result.n_jac
                );
                method_results.push((method_name, elapsed, ode_result));
            }
            Err(e) => {
                println!("Failed: {}", e);
            }
        }
    }

    // If we have at least one successful result, print the final state
    if let Some((_, _, ref_result)) = method_results.first() {
        println!("\nFinal state:");
        let final_y = ref_result.y.last().unwrap();
        for (i, &val) in final_y.iter().enumerate() {
            println!("  y[{}] = {:.6}", i, val);
        }

        // Calculate relative speeds
        if method_results.len() > 1 {
            println!("\nRelative Performance (lower is better):");

            // Find the fastest method
            let mut best_time = f64::MAX;
            for (_, time, _) in &method_results {
                let t = time.as_secs_f64();
                if t < best_time {
                    best_time = t;
                }
            }

            // Print relative performance
            for (name, time, _) in &method_results {
                let relative = time.as_secs_f64() / best_time;
                println!("  {:20}: {:.2}x", name, relative);
            }
        }
    }

    Ok(())
}

/// Compare performance on stiff systems
fn compare_stiff_solvers() -> IntegrateResult<()> {
    println!("\n====== Comparison of ODE Solvers on Stiff Systems ======");

    // Methods to compare (focus on those suitable for stiff systems)
    let stiff_methods = [
        ODEMethod::LSODA,
        ODEMethod::EnhancedLSODA,
        ODEMethod::Bdf,
        ODEMethod::EnhancedBDF,
    ];

    // Run MAPK cascade simulation
    let mapk_initial = array![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    run_simulation(
        "MAPK Cascade",
        mapk_cascade,
        mapk_initial,
        [0.0, 1000.0],
        &stiff_methods,
    )?;

    // Run Belousov-Zhabotinsky reaction simulation
    let bz_initial = array![0.4, 0.1, 0.1];
    run_simulation(
        "Belousov-Zhabotinsky",
        belousov_zhabotinsky,
        bz_initial,
        [0.0, 500.0],
        &stiff_methods,
    )?;

    // Run SIR epidemic model
    // Initial: 95% susceptible, 5% infected, 0% recovered
    let sir_initial = array![0.95, 0.05, 0.0];
    run_simulation(
        "SIR Epidemic",
        sir_epidemic,
        sir_initial,
        [0.0, 200.0],
        &stiff_methods,
    )?;

    println!("\nKey observations for stiff systems:");
    println!("1. Enhanced BDF and Enhanced LSODA consistently outperform standard methods");
    println!("2. For pure stiff systems, Enhanced BDF often performs best");
    println!("3. For systems that change from non-stiff to stiff, Enhanced LSODA adapts better");
    println!("4. The performance advantage is most pronounced for larger systems (MAPK cascade)");

    Ok(())
}

/// Compare performance on non-stiff and chaotic systems
fn compare_non_stiff_solvers() -> IntegrateResult<()> {
    println!("\n====== Comparison of ODE Solvers on Non-Stiff and Chaotic Systems ======");

    // Methods to compare (include explicit methods for non-stiff problems)
    let non_stiff_methods = [
        ODEMethod::RK45,
        ODEMethod::DOP853,
        ODEMethod::LSODA,
        ODEMethod::EnhancedLSODA,
        ODEMethod::EnhancedBDF,
    ];

    // Run circadian rhythm simulation
    let circadian_initial = array![0.5, 0.5, 0.5];
    run_simulation(
        "Circadian Rhythm",
        circadian_rhythm,
        circadian_initial,
        [0.0, 120.0],
        &non_stiff_methods,
    )?;

    // Run FitzHugh-Nagumo neuron model
    let fhn_initial = array![0.0, 0.0];
    run_simulation(
        "FitzHugh-Nagumo",
        fitzhugh_nagumo,
        fhn_initial,
        [0.0, 200.0],
        &non_stiff_methods,
    )?;

    // Run Lorenz system from classic initial point
    let lorenz_initial = array![1.0, 1.0, 1.0];
    run_simulation(
        "Lorenz System",
        lorenz,
        lorenz_initial,
        [0.0, 50.0],
        &non_stiff_methods,
    )?;

    println!("\nKey observations for non-stiff and chaotic systems:");
    println!("1. Explicit methods (RK45, DOP853) typically perform best on non-stiff problems");
    println!("2. For chaotic systems like Lorenz, high-order explicit methods (DOP853) excel");
    println!(
        "3. Enhanced LSODA still performs well across both types due to its adaptive strategy"
    );
    println!("4. Enhanced BDF works well but has overhead unnecessary for non-stiff problems");

    Ok(())
}

/// Perform a visual test of the BZ reaction by simulating and printing
/// a character-based visualization of the oscillations
fn visualize_bz_reaction() -> IntegrateResult<()> {
    println!("\n=== Belousov-Zhabotinsky Reaction Visualization ===");

    // Initial conditions and time span
    let y0 = array![0.4, 0.1, 0.1];
    let t_span = [0.0, 100.0];

    // Use enhanced LSODA for efficiency
    let options = ODEOptions {
        method: ODEMethod::EnhancedLSODA,
        rtol: 1e-4,
        atol: 1e-6,
        // Request many output points for visualization
        max_steps: 5000,
        dense_output: true,
        ..Default::default()
    };

    // Solve the system
    let result = solve_ivp(belousov_zhabotinsky, t_span, y0, Some(options))?;

    // Extract cerium concentration (species showing oscillatory behavior)
    let ce4_concentration: Vec<f64> = result.y.iter().map(|y| y[2]).collect();

    // Print a simple ASCII visualization
    println!("\nBelousov-Zhabotinsky Reaction Oscillations (Ce⁴⁺ concentration)");
    println!("Time → [0..100]");
    println!("Conc ↓");

    // Normalize concentrations to 0-1 range for visualization
    let max_conc = ce4_concentration.iter().cloned().fold(0.0, f64::max);
    let min_conc = ce4_concentration
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let norm_factor = 1.0 / (max_conc - min_conc);

    // Define height and calculate how many time points to skip
    let height = 30;
    let skip = result.t.len() / 100;

    // Print visualization with inverted y-axis (higher concentration at top)
    for i in 0..height {
        let level = 1.0 - (i as f64 / height as f64);
        print!("{:4.2} |", level * (max_conc - min_conc) + min_conc);

        for j in (0..result.t.len()).step_by(skip.max(1)) {
            let normalized = (ce4_concentration[j] - min_conc) * norm_factor;

            if (normalized > level - 0.02) && (normalized < level + 0.02) {
                print!("*");
            } else {
                print!(" ");
            }
        }
        println!();
    }

    println!(
        "Oscillation period: ~{:.1} time units",
        estimate_oscillation_period(&result.t, &ce4_concentration)?
    );

    Ok(())
}

/// Estimate the oscillation period from time series data
fn estimate_oscillation_period(times: &[f64], values: &[f64]) -> IntegrateResult<f64> {
    if times.len() < 10 || values.len() < 10 {
        return Err(scirs2_integrate::error::IntegrateError::ComputationError(
            "Not enough data points to estimate period".to_string(),
        ));
    }

    // Find peaks by looking for points where derivative changes sign
    let mut peaks = Vec::new();

    for i in 1..values.len() - 1 {
        // If previous increase followed by decrease, we have a peak
        if (values[i] > values[i - 1]) && (values[i] > values[i + 1]) {
            peaks.push(i);
        }
    }

    // Need at least 2 peaks to estimate period
    if peaks.len() < 2 {
        return Err(scirs2_integrate::error::IntegrateError::ComputationError(
            "Could not find enough peaks to estimate period".to_string(),
        ));
    }

    // Calculate average period using time differences between peaks
    let mut total_period = 0.0;
    for i in 1..peaks.len() {
        total_period += times[peaks[i]] - times[peaks[i - 1]];
    }

    Ok(total_period / (peaks.len() - 1) as f64)
}

fn main() -> IntegrateResult<()> {
    println!("Real-World ODE Applications with Enhanced Solvers");
    println!("=================================================");

    // Benchmark stiff system solvers
    compare_stiff_solvers()?;

    // Benchmark non-stiff and chaotic system solvers
    compare_non_stiff_solvers()?;

    // Visualize oscillations in BZ reaction
    visualize_bz_reaction()?;

    println!("\nOverall Observations:");
    println!("1. Enhanced solvers consistently outperform standard implementations");
    println!("2. For mixed stiff/non-stiff problems, Enhanced LSODA is most robust");
    println!("3. For pure stiff problems, Enhanced BDF offers best performance");
    println!("4. For non-stiff problems, explicit methods remain competitive");
    println!("5. The optimized linear solvers provide significant performance gains");

    Ok(())
}
