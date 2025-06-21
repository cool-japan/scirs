//! Advanced Enzyme Kinetics and Metabolic Pathway Demo
//!
//! This example demonstrates the advanced enzyme kinetics models and
//! metabolic pathway simulation capabilities, including multi-substrate
//! mechanisms, allosteric regulation, and pathway analysis.

use ndarray::{Array1, ArrayView1};
use scirs2_integrate::ode::{
    enzyme_kinetics::{pathways, EnzymeMechanism, EnzymeParameters, RegulationType},
    solve_ivp, ODEMethod, ODEOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Advanced Enzyme Kinetics and Metabolic Pathway Demo\n");

    // Example 1: Multi-substrate enzyme mechanisms
    println!("1. Multi-Substrate Enzyme Mechanisms");
    demonstrate_multisubstrate_mechanisms()?;
    println!();

    // Example 2: Allosteric enzyme regulation
    println!("2. Allosteric Enzyme Regulation");
    demonstrate_allosteric_regulation()?;
    println!();

    // Example 3: Simple metabolic pathway
    println!("3. Simple Glycolysis Pathway Simulation");
    demonstrate_glycolysis_pathway()?;
    println!();

    // Example 4: TCA cycle dynamics
    println!("4. TCA Cycle Dynamics");
    demonstrate_tca_cycle()?;
    println!();

    // Example 5: Metabolic control analysis
    println!("5. Metabolic Control Analysis");
    demonstrate_control_analysis()?;
    println!();

    // Example 6: Temperature and pH effects
    println!("6. Temperature and pH Effects on Enzyme Activity");
    demonstrate_environmental_effects()?;
    println!();

    // Example 7: Pathway regulation and feedback
    println!("7. Pathway Regulation and Feedback Inhibition");
    demonstrate_pathway_regulation()?;
    println!();

    println!("All enzyme kinetics and metabolic pathway demonstrations completed successfully!");

    Ok(())
}

/// Demonstrate different multi-substrate enzyme mechanisms
fn demonstrate_multisubstrate_mechanisms() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Comparing different bi-substrate enzyme mechanisms");

    // Test substrates A and B concentrations
    let substrate_a_range = vec![0.1, 0.5, 1.0, 2.0, 5.0];
    let substrate_b = 1.0; // Fixed concentration of substrate B

    // Ordered sequential mechanism
    let ordered_params = EnzymeParameters {
        mechanism: EnzymeMechanism::OrderedSequential {
            ka: 0.1,
            kb: 0.2,
            kp: 0.05,
            kq: 0.03,
            kcat: 100.0,
        },
        temperature: 310.15,
        ph: 7.4,
        ionic_strength: 0.15,
        temperature_params: None,
        ph_params: None,
    };

    // Random sequential mechanism
    let random_params = EnzymeParameters {
        mechanism: EnzymeMechanism::RandomSequential {
            ka: 0.1,
            kb: 0.2,
            kp: 0.05,
            kq: 0.03,
            kcat: 100.0,
            alpha: 0.5,
        },
        temperature: 310.15,
        ph: 7.4,
        ionic_strength: 0.15,
        temperature_params: None,
        ph_params: None,
    };

    // Ping-pong mechanism
    let pingpong_params = EnzymeParameters {
        mechanism: EnzymeMechanism::PingPong {
            ka: 0.1,
            kb: 0.2,
            kp: 0.05,
            kq: 0.03,
            kcat1: 80.0,
            kcat2: 120.0,
        },
        temperature: 310.15,
        ph: 7.4,
        ionic_strength: 0.15,
        temperature_params: None,
        ph_params: None,
    };

    println!("   [A] (mM)  | Ordered Seq. | Random Seq. | Ping-Pong |");
    println!("   ----------|--------------|-------------|-----------|");

    for &substrate_a in &substrate_a_range {
        let concentrations = vec![substrate_a, substrate_b, 0.0, 0.0]; // A, B, P, Q

        let rate_ordered = ordered_params.calculate_rate(&concentrations);
        let rate_random = random_params.calculate_rate(&concentrations);
        let rate_pingpong = pingpong_params.calculate_rate(&concentrations);

        println!(
            "   {:8.1}  | {:10.2}   | {:9.2}   | {:7.2}   |",
            substrate_a, rate_ordered, rate_random, rate_pingpong
        );
    }

    println!("   Note: All rates in μM/s units");
    Ok(())
}

/// Demonstrate allosteric enzyme regulation
fn demonstrate_allosteric_regulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Allosteric enzyme with activator and inhibitor");

    let allosteric_params = EnzymeParameters::allosteric(
        1.0,   // Km = 1.0 mM
        200.0, // Vmax = 200 μM/s
        0.5,   // Ka_act = 0.5 mM (activator)
        2.0,   // Ka_inh = 2.0 mM (inhibitor)
        2.0,   // n_act = 2.0 (positive cooperativity)
        3.0,   // n_inh = 3.0 (strong inhibition)
    );

    let substrate_conc = 1.0; // Fixed at Km value
    let effector_range = vec![0.0, 0.1, 0.5, 1.0, 2.0, 5.0];

    println!(
        "   Substrate concentration: {:.1} mM (= Km)",
        substrate_conc
    );
    println!();
    println!("   Effector | With Activator | With Inhibitor | Both        |");
    println!("   (mM)     | Only          | Only          | A=0.5, I=var |");
    println!("   ---------|---------------|---------------|-------------|");

    for &effector_conc in &effector_range {
        // With activator only
        let rate_activator =
            allosteric_params.calculate_rate(&[substrate_conc, effector_conc, 0.0]);

        // With inhibitor only
        let rate_inhibitor =
            allosteric_params.calculate_rate(&[substrate_conc, 0.0, effector_conc]);

        // With both (fixed activator, variable inhibitor)
        let rate_both = allosteric_params.calculate_rate(&[substrate_conc, 0.5, effector_conc]);

        println!(
            "   {:6.1}   | {:11.1}     | {:11.1}     | {:9.1}   |",
            effector_conc, rate_activator, rate_inhibitor, rate_both
        );
    }

    // Demonstrate cooperativity
    println!();
    println!("   Cooperativity demonstration (Hill plot data):");
    println!("   Substrate (mM) | Rate (μM/s) | log[S]  | log(v/(Vmax-v)) |");
    println!("   ---------------|-------------|---------|-----------------|");

    let vmax = 200.0;
    for i in 0..10 {
        let s = 0.1 * 2.0_f64.powf(i as f64); // Exponential substrate range
        let rate = allosteric_params.calculate_rate(&[s, 0.0, 0.0]);
        let log_s = s.log10();
        let log_ratio = if rate < vmax * 0.99 {
            (rate / (vmax - rate)).log10()
        } else {
            f64::NAN
        };

        if log_ratio.is_finite() {
            println!(
                "   {:12.3}   | {:9.1}   | {:5.2}   | {:13.2}   |",
                s, rate, log_s, log_ratio
            );
        }
    }

    Ok(())
}

/// Demonstrate simple glycolysis pathway simulation
fn demonstrate_glycolysis_pathway() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Simulating simplified glycolysis pathway dynamics");

    let pathway = pathways::simple_glycolysis();

    println!("   Pathway: {}", pathway.name);
    println!("   Enzymes: {}", pathway.enzymes.len());
    println!("   Metabolites: {}", pathway.metabolites.len());

    // Initial concentrations (mM)
    let initial_concentrations = Array1::from_vec(vec![
        5.0,  // Glucose (external, maintained)
        0.1,  // G6P
        0.1,  // F6P
        0.05, // FBP
        0.05, // PEP
        0.1,  // Pyruvate (external, maintained)
    ]);

    println!();
    println!("   Initial concentrations:");
    for (i, (name, &conc)) in pathway
        .metabolites
        .iter()
        .zip(initial_concentrations.iter())
        .enumerate()
    {
        println!(
            "   {}: {:.3} mM{}",
            name,
            conc,
            if pathway.external_metabolites.contains_key(&i) {
                " (external)"
            } else {
                ""
            }
        );
    }

    // Set up ODE system for pathway simulation
    let pathway_clone = pathway.clone();
    let ode_fn = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
        pathway_clone.calculate_derivatives(&y.to_owned())
    };

    // Solve the ODE system
    let options = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-6,
        atol: 1e-9,
        max_step: Some(0.1),
        ..Default::default()
    };

    let t_span = [0.0, 10.0]; // Simulate for 10 seconds
    let result = solve_ivp(
        ode_fn,
        t_span,
        initial_concentrations.clone(),
        Some(options),
    )?;

    // Display results at key time points
    let time_points = vec![0.0, 1.0, 2.0, 5.0, 10.0];
    println!();
    println!("   Time course results:");
    println!("   Time | G6P   | F6P   | FBP   | PEP   | Flux  |");
    println!("   (s)  | (mM)  | (mM)  | (mM)  | (mM)  |(μM/s) |");
    println!("   -----|-------|-------|-------|-------|-------|");

    for &t in &time_points {
        // Find closest time point in solution
        let idx = result
            .t
            .iter()
            .position(|&x| x >= t)
            .unwrap_or(result.t.len() - 1);

        let y = &result.y[idx];
        let rates = pathway.calculate_reaction_rates(y);
        let total_flux = rates.sum() * 1000.0; // Convert to μM/s

        println!(
            "   {:3.0}  | {:5.3} | {:5.3} | {:5.3} | {:5.3} | {:5.1} |",
            t, y[1], y[2], y[3], y[4], total_flux
        );
    }

    Ok(())
}

/// Demonstrate TCA cycle dynamics
fn demonstrate_tca_cycle() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Simulating TCA cycle dynamics");

    let pathway = pathways::tca_cycle();

    // Initial concentrations for TCA cycle metabolites (mM)
    let initial_concentrations = Array1::from_vec(vec![
        0.5,  // Acetyl-CoA
        0.3,  // Citrate
        0.2,  // Isocitrate
        0.1,  // α-Ketoglutarate
        0.05, // Succinyl-CoA
        0.1,  // Succinate
        0.08, // Fumarate
        0.15, // Malate
    ]);

    println!("   Metabolites in TCA cycle:");
    for (name, &conc) in pathway
        .metabolites
        .iter()
        .zip(initial_concentrations.iter())
    {
        println!("   {}: {:.3} mM", name, conc);
    }

    // Calculate initial fluxes
    let initial_rates = pathway.calculate_reaction_rates(&initial_concentrations);
    println!();
    println!("   Initial enzyme fluxes:");
    for (enzyme, &rate) in pathway.enzymes.iter().zip(initial_rates.iter()) {
        println!("   {}: {:.2} μM/s", enzyme.name, rate * 1000.0);
    }

    // Set up ODE system
    let pathway_clone = pathway.clone();
    let ode_fn = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
        pathway_clone.calculate_derivatives(&y.to_owned())
    };

    // Solve for short-term dynamics
    let options = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-6,
        atol: 1e-9,
        max_step: Some(0.01),
        ..Default::default()
    };

    let t_span = [0.0, 1.0]; // Simulate for 1 second
    let result = solve_ivp(
        ode_fn,
        t_span,
        initial_concentrations.clone(),
        Some(options),
    )?;

    // Check for steady-state approach
    let final_y = &result.y[result.y.len() - 1];
    let final_rates = pathway.calculate_reaction_rates(final_y);

    println!();
    println!("   After 1 second:");
    println!("   Metabolite changes:");
    for (i, name) in pathway.metabolites.iter().enumerate() {
        let change = ((final_y[i] - initial_concentrations[i]) / initial_concentrations[i]) * 100.0;
        println!("   {}: {:+.1}% change", name, change);
    }

    println!();
    println!("   Final enzyme fluxes:");
    for (enzyme, &rate) in pathway.enzymes.iter().zip(final_rates.iter()) {
        println!("   {}: {:.2} μM/s", enzyme.name, rate * 1000.0);
    }

    Ok(())
}

/// Demonstrate metabolic control analysis
fn demonstrate_control_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Performing metabolic control analysis on glycolysis");

    let pathway = pathways::simple_glycolysis();

    // Steady-state concentrations (estimated)
    let steady_state = Array1::from_vec(vec![
        5.0, // Glucose (external)
        1.0, // G6P
        0.5, // F6P
        0.3, // FBP
        0.2, // PEP
        0.1, // Pyruvate (external)
    ]);

    println!("   Assumed steady-state concentrations:");
    for (name, &conc) in pathway.metabolites.iter().zip(steady_state.iter()) {
        println!("   {}: {:.3} mM", name, conc);
    }

    // Perform control analysis
    let analysis = pathway.control_analysis(&steady_state);

    println!();
    println!("   Flux Control Coefficients:");
    for (i, enzyme) in pathway.enzymes.iter().enumerate() {
        let fcc = analysis.flux_control_coefficients[i];
        println!("   {}: {:.3}", enzyme.name, fcc);
    }

    println!();
    println!(
        "   Sum of FCCs: {:.3} (should be 1.0)",
        analysis.flux_control_coefficients.sum()
    );

    println!();
    println!("   Elasticity Coefficients (enzyme response to metabolite changes):");
    println!("   Enzyme \\ Metabolite | G6P   | F6P   | FBP   | PEP   |");
    println!("   -------------------|-------|-------|-------|-------|");

    for (i, enzyme) in pathway.enzymes.iter().enumerate() {
        print!("   {:17} |", enzyme.name);
        for j in 1..5 {
            // Skip external metabolites (0 and 5)
            let elasticity = analysis.elasticity_coefficients[(i, j)];
            print!(" {:5.2} |", elasticity);
        }
        println!();
    }

    println!();
    println!("   Steady-state fluxes:");
    for (i, enzyme) in pathway.enzymes.iter().enumerate() {
        let flux = analysis.steady_state_fluxes[i] * 1000.0; // Convert to μM/s
        println!("   {}: {:.1} μM/s", enzyme.name, flux);
    }

    Ok(())
}

/// Demonstrate temperature and pH effects on enzyme activity
fn demonstrate_environmental_effects() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Effects of temperature and pH on enzyme activity");

    let mut base_params = EnzymeParameters::michaelis_menten(1.0, 100.0);
    let substrate_conc = 1.0; // At Km

    // Temperature effects
    println!("   Temperature effects (pH = 7.4):");
    println!("   Temp (°C) | Rate (μM/s) | Relative Activity |");
    println!("   ----------|-------------|-------------------|");

    let temperatures = vec![15.0, 25.0, 37.0, 45.0, 55.0, 65.0];
    let base_rate = {
        base_params.temperature = 310.15; // 37°C
        base_params.calculate_rate(&[substrate_conc])
    };

    for &temp_c in &temperatures {
        base_params.temperature = temp_c + 273.15; // Convert to Kelvin
        let rate = base_params.calculate_rate(&[substrate_conc]);
        let relative = rate / base_rate;

        println!(
            "   {:7.0}   | {:9.1}   | {:15.2}   |",
            temp_c, rate, relative
        );
    }

    // pH effects
    println!();
    println!("   pH effects (Temperature = 37°C):");
    println!("   pH    | Rate (μM/s) | Relative Activity |");
    println!("   ------|-------------|-------------------|");

    base_params.temperature = 310.15; // Reset to 37°C
    let ph_values = vec![5.0, 6.0, 7.0, 7.4, 8.0, 9.0, 10.0];

    for &ph in &ph_values {
        base_params.ph = ph;
        let rate = base_params.calculate_rate(&[substrate_conc]);
        let relative = rate / base_rate;

        println!("   {:4.1}  | {:9.1}   | {:15.2}   |", ph, rate, relative);
    }

    // Combined effects
    println!();
    println!("   Optimal conditions:");
    println!("   Finding temperature and pH that maximize activity...");

    let mut max_rate = 0.0;
    let mut optimal_temp = 0.0;
    let mut optimal_ph = 0.0;

    for temp_c in (20..60).step_by(5) {
        for ph_x10 in (60..85).step_by(2) {
            let temp = temp_c as f64;
            let ph = ph_x10 as f64 / 10.0;

            base_params.temperature = temp + 273.15;
            base_params.ph = ph;
            let rate = base_params.calculate_rate(&[substrate_conc]);

            if rate > max_rate {
                max_rate = rate;
                optimal_temp = temp;
                optimal_ph = ph;
            }
        }
    }

    println!("   Optimal temperature: {:.0}°C", optimal_temp);
    println!("   Optimal pH: {:.1}", optimal_ph);
    println!("   Maximum rate: {:.1} μM/s", max_rate);
    println!(
        "   Improvement over standard conditions: {:.1}×",
        max_rate / base_rate
    );

    Ok(())
}

/// Demonstrate pathway regulation and feedback inhibition
fn demonstrate_pathway_regulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating pathway regulation and feedback inhibition");

    // Create a simple linear pathway with feedback
    let pathway = pathways::purine_biosynthesis();

    println!("   Pathway: {}", pathway.name);
    println!("   Enzymes: {}", pathway.enzymes.len());

    // Initial concentrations
    let mut concentrations = Array1::from_vec(vec![
        1.0, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    ]);

    println!();
    println!("   Initial state (no product feedback):");
    let initial_rates = pathway.calculate_reaction_rates(&concentrations);
    for (i, enzyme) in pathway.enzymes.iter().enumerate() {
        println!("   {}: {:.2} μM/s", enzyme.name, initial_rates[i] * 1000.0);
    }

    // Simulate with increasing end product (IMP) concentration
    println!();
    println!("   Effect of IMP feedback inhibition:");
    println!("   IMP (mM) | First Enzyme | Last Enzyme | Pathway Flux |");
    println!("   ---------|--------------|-------------|--------------|");

    let imp_concentrations = vec![0.0, 0.1, 0.5, 1.0, 2.0, 5.0];

    for &imp_conc in &imp_concentrations {
        concentrations[9] = imp_conc; // Set IMP concentration
        let rates = pathway.calculate_reaction_rates(&concentrations);
        let first_enzyme_rate = rates[0] * 1000.0;
        let last_enzyme_rate = rates[8] * 1000.0;
        let pathway_flux = rates.iter().take(9).sum::<f64>() * 1000.0 / 9.0; // Average flux

        println!(
            "   {:6.1}   | {:10.2}   | {:9.2}   | {:10.2}   |",
            imp_conc, first_enzyme_rate, last_enzyme_rate, pathway_flux
        );
    }

    // Demonstrate multiple regulation types
    println!();
    println!("   Comparing different regulation types:");

    let _base_rate = 50.0; // Base enzyme rate
    let effector_conc: f64 = 1.0;
    let ki: f64 = 0.5; // Inhibition constant

    let regulation_types = vec![
        ("Competitive", RegulationType::CompetitiveInhibition),
        ("Non-competitive", RegulationType::NonCompetitiveInhibition),
        ("Allosteric", RegulationType::AllostericInhibition),
        ("Feedback", RegulationType::FeedbackInhibition),
    ];

    println!("   Regulation Type    | Inhibition Factor | Remaining Activity |");
    println!("   -------------------|-------------------|--------------------|");

    for (name, reg_type) in regulation_types {
        let inhibition_factor = match reg_type {
            RegulationType::CompetitiveInhibition => 1.0 / (1.0 + effector_conc / ki),
            RegulationType::NonCompetitiveInhibition => 1.0 / (1.0 + effector_conc / ki),
            RegulationType::AllostericInhibition => {
                1.0 / (1.0 + (effector_conc / ki).powf(2.0))
            }
            RegulationType::FeedbackInhibition => {
                1.0 / (1.0 + (effector_conc / ki).powf(4.0))
            }
            _ => 1.0,
        };

        let remaining_activity = inhibition_factor * 100.0;

        println!(
            "   {:17} | {:15.3}   | {:16.1}%    |",
            name, inhibition_factor, remaining_activity
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_enzyme_kinetics_demo() {
        // Test that all demo functions run without errors
        assert!(demonstrate_multisubstrate_mechanisms().is_ok());
        assert!(demonstrate_allosteric_regulation().is_ok());
        assert!(demonstrate_environmental_effects().is_ok());
        assert!(demonstrate_pathway_regulation().is_ok());
    }

    #[test]
    fn test_glycolysis_pathway_integration() {
        // Test the glycolysis pathway simulation
        let result = demonstrate_glycolysis_pathway();
        assert!(result.is_ok());
    }

    #[test]
    fn test_control_analysis() {
        // Test metabolic control analysis
        let result = demonstrate_control_analysis();
        assert!(result.is_ok());
    }

    #[test]
    fn test_tca_cycle_simulation() {
        // Test TCA cycle dynamics
        let result = demonstrate_tca_cycle();
        assert!(result.is_ok());
    }

    #[test]
    fn test_pathway_calculations() {
        let pathway = pathways::simple_glycolysis();
        let concentrations = Array1::from_vec(vec![5.0, 1.0, 0.5, 0.3, 0.2, 0.1]);

        let rates = pathway.calculate_reaction_rates(&concentrations);

        // All rates should be non-negative
        for &rate in rates.iter() {
            assert!(rate >= 0.0);
        }

        let derivatives = pathway.calculate_derivatives(&concentrations);

        // External metabolites should have zero derivatives
        assert_abs_diff_eq!(derivatives[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(derivatives[5], 0.0, epsilon = 1e-10);
    }
}
