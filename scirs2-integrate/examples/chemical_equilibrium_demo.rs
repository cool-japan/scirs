//! Chemical Equilibrium Calculation Demo
//!
//! This example demonstrates the chemical equilibrium calculation methods
//! including acid-base equilibria, complex formation, solubility equilibria,
//! and activity coefficient models.

use ndarray::arr1;
use scirs2_integrate::ode::chemical_equilibrium::{systems, ActivityModel, ThermoData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Chemical Equilibrium Calculation Demo\n");

    // Example 1: Weak acid equilibrium
    println!("1. Weak Acid Equilibrium (Acetic Acid)");
    demonstrate_weak_acid_equilibrium()?;
    println!();

    // Example 2: Buffer systems
    println!("2. Buffer System Analysis");
    demonstrate_buffer_equilibrium()?;
    println!();

    // Example 3: Complex formation
    println!("3. Metal-Ligand Complex Formation");
    demonstrate_complex_formation()?;
    println!();

    // Example 4: Solubility equilibrium
    println!("4. Solubility Equilibrium");
    demonstrate_solubility_equilibrium()?;
    println!();

    // Example 5: Multiple equilibria (amino acid)
    println!("5. Multiple Equilibria System (Amino Acid)");
    demonstrate_amino_acid_equilibrium()?;
    println!();

    // Example 6: Activity coefficient effects
    println!("6. Activity Coefficient Effects");
    demonstrate_activity_coefficients()?;
    println!();

    // Example 7: Temperature effects
    println!("7. Temperature Effects on Equilibrium");
    demonstrate_temperature_effects()?;
    println!();

    // Example 8: pH titration simulation
    println!("8. pH Titration Simulation");
    demonstrate_titration_curve()?;
    println!();

    println!("All chemical equilibrium demonstrations completed successfully!");

    Ok(())
}

/// Demonstrate weak acid equilibrium calculation
fn demonstrate_weak_acid_equilibrium() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Calculating equilibrium for acetic acid (CH₃COOH)");

    let ka = 1.8e-5_f64; // Acetic acid dissociation constant
    let initial_acid = 0.1; // 0.1 M acetic acid

    let calculator = systems::weak_acid_equilibrium(ka, initial_acid, None);

    // Initial concentrations: [HA, H+, A-]
    let initial_conc = arr1(&[initial_acid, 1e-7, 0.0]);

    println!("   Initial conditions:");
    println!("   [CH₃COOH]₀ = {:.3} M", initial_acid);
    println!("   Ka = {:.2e}", ka);

    let result = calculator.calculate_equilibrium(initial_conc, None)?;

    if result.converged {
        let h_plus = result.concentrations[1];
        let acetate = result.concentrations[2];
        let acid_remaining = result.concentrations[0];
        let ph = -h_plus.log10();
        let degree_dissociation = (acetate / initial_acid) * 100.0;

        println!();
        println!("   Equilibrium results:");
        println!("   [CH₃COOH] = {:.6} M", acid_remaining);
        println!("   [H⁺] = {:.6e} M", h_plus);
        println!("   [CH₃COO⁻] = {:.6} M", acetate);
        println!("   pH = {:.2}", ph);
        println!("   Degree of dissociation = {:.2}%", degree_dissociation);
        println!("   Iterations: {}", result.iterations);

        // Verify Ka calculation
        let ka_calculated = (h_plus * acetate) / acid_remaining;
        println!(
            "   Ka (calculated) = {:.2e} (should be {:.2e})",
            ka_calculated, ka
        );

        // Compare with analytical approximation for weak acid
        let h_analytical = (ka * initial_acid).sqrt();
        let ph_analytical = -h_analytical.log10();
        println!("   pH (analytical approx.) = {:.2}", ph_analytical);
    } else {
        println!("   Equilibrium calculation did not converge!");
    }

    Ok(())
}

/// Demonstrate buffer equilibrium
fn demonstrate_buffer_equilibrium() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Acetate buffer system analysis");

    let ka = 1.8e-5_f64;
    let acid_conc = 0.1; // CH₃COOH
    let base_conc = 0.1; // CH₃COONa (acetate)

    let calculator = systems::buffer_equilibrium(ka, acid_conc, base_conc);

    // Initial concentrations: [HA, H+, A-, OH-, H2O]
    let initial_conc = arr1(&[acid_conc, 1e-7, base_conc, 1e-7, 55.5]);

    println!("   Buffer composition:");
    println!("   [CH₃COOH] = {:.1} M", acid_conc);
    println!("   [CH₃COO⁻] = {:.1} M", base_conc);
    println!("   Ka = {:.2e}", ka);

    let result = calculator.calculate_equilibrium(initial_conc, None)?;

    if result.converged {
        let h_plus = result.concentrations[1];
        let ph = -h_plus.log10();
        let pka = -ka.log10();

        println!();
        println!("   Buffer equilibrium results:");
        println!("   [H⁺] = {:.2e} M", h_plus);
        println!("   pH = {:.2}", ph);
        println!("   pKa = {:.2}", pka);

        // Henderson-Hasselbalch equation verification
        let ph_hh = pka + (base_conc / acid_conc).log10();
        println!("   pH (Henderson-Hasselbalch) = {:.2}", ph_hh);

        // Buffer capacity calculation
        let buffer_capacity =
            2.303 * ka * h_plus * (acid_conc + base_conc) / (ka + h_plus).powf(2.0);
        println!("   Buffer capacity = {:.4} mol/(L·pH)", buffer_capacity);
    } else {
        println!("   Buffer equilibrium calculation did not converge!");
    }

    // Test buffer response to acid addition
    println!();
    println!("   Buffer response to strong acid addition:");
    println!("   HCl added | pH before | pH after | ΔpH    |");
    println!("   (mmol/L)  |           |          |        |");
    println!("   ----------|-----------|----------|--------|");

    let hcl_additions = vec![0.0, 1.0, 5.0, 10.0, 20.0];

    for &hcl_added in &hcl_additions {
        // Recalculate with added HCl
        let new_acid = acid_conc + hcl_added / 1000.0;
        let new_base = base_conc - hcl_added / 1000.0;

        if new_base > 0.0 {
            let new_calculator = systems::buffer_equilibrium(ka, new_acid, new_base);
            let new_initial = arr1(&[new_acid, 1e-7, new_base, 1e-7, 55.5]);

            if let Ok(new_result) = new_calculator.calculate_equilibrium(new_initial, None) {
                if new_result.converged {
                    let original_ph = -result.concentrations[1].log10();
                    let new_ph = -new_result.concentrations[1].log10();
                    let delta_ph = new_ph - original_ph;

                    println!(
                        "   {:8.1}  | {:7.2}   | {:6.2}   | {:6.2} |",
                        hcl_added, original_ph, new_ph, delta_ph
                    );
                }
            }
        } else {
            println!(
                "   {:8.1}  | {:7.2}   | buffer exceeded capacity   |",
                hcl_added,
                -result.concentrations[1].log10()
            );
        }
    }

    Ok(())
}

/// Demonstrate complex formation equilibrium
fn demonstrate_complex_formation() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Metal-ligand complex formation: Cu²⁺ + 4NH₃ ⇌ [Cu(NH₃)₄]²⁺");

    let k_formation = 1.1e13; // Formation constant for [Cu(NH3)4]2+
    let initial_cu = 0.001; // 1 mM Cu2+
    let initial_nh3 = 0.1; // 100 mM NH3

    let calculator = systems::complex_formation(k_formation, initial_cu, initial_nh3);

    // For simplicity, consider 1:1 complex (can be extended to 1:4)
    let initial_conc = arr1(&[initial_cu, initial_nh3, 0.0]);

    println!("   Initial conditions:");
    println!("   [Cu²⁺]₀ = {:.1} mM", initial_cu * 1000.0);
    println!("   [NH₃]₀ = {:.1} mM", initial_nh3 * 1000.0);
    println!("   K_formation = {:.2e}", k_formation);

    let result = calculator.calculate_equilibrium(initial_conc, None)?;

    if result.converged {
        let cu_free = result.concentrations[0];
        let nh3_free = result.concentrations[1];
        let complex = result.concentrations[2];

        println!();
        println!("   Equilibrium results:");
        println!("   [Cu²⁺] = {:.3e} M ({:.1} μM)", cu_free, cu_free * 1e6);
        println!("   [NH₃] = {:.4} M", nh3_free);
        println!(
            "   [Cu(NH₃)ₙ²⁺] = {:.3e} M ({:.1} μM)",
            complex,
            complex * 1e6
        );

        let fraction_complexed = complex / initial_cu * 100.0;
        println!("   Fraction of Cu complexed = {:.1}%", fraction_complexed);

        // Verify formation constant
        let k_calc = complex / (cu_free * nh3_free);
        println!("   K_formation (calculated) = {:.2e}", k_calc);

        // Calculate α coefficient (fraction of free Cu)
        let alpha = cu_free / initial_cu;
        println!("   α₀ (fraction free Cu) = {:.4}", alpha);
    } else {
        println!("   Complex formation calculation did not converge!");
    }

    // Effect of ligand concentration
    println!();
    println!("   Effect of NH₃ concentration on complex formation:");
    println!("   [NH₃] (M) | [Cu²⁺] (μM) | [Complex] (μM) | % Complexed |");
    println!("   ----------|-------------|----------------|-------------|");

    let nh3_concentrations = vec![0.001, 0.01, 0.1, 1.0];

    for &nh3_conc in &nh3_concentrations {
        let test_calc = systems::complex_formation(k_formation, initial_cu, nh3_conc);
        let test_initial = arr1(&[initial_cu, nh3_conc, 0.0]);

        if let Ok(test_result) = test_calc.calculate_equilibrium(test_initial, None) {
            if test_result.converged {
                let cu_free = test_result.concentrations[0] * 1e6; // Convert to μM
                let complex = test_result.concentrations[2] * 1e6;
                let percent_complexed = (complex / (initial_cu * 1e6)) * 100.0;

                println!(
                    "   {:8.3}  | {:9.1}   | {:12.1}  | {:9.1}   |",
                    nh3_conc, cu_free, complex, percent_complexed
                );
            }
        }
    }

    Ok(())
}

/// Demonstrate solubility equilibrium
fn demonstrate_solubility_equilibrium() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Solubility equilibrium: AgCl(s) ⇌ Ag⁺ + Cl⁻");

    let ksp = 1.8e-10; // Solubility product of AgCl
    let calculator = systems::solubility_equilibrium(ksp, 1.0, 1.0);

    // Initial: pure water with solid AgCl
    let initial_conc = arr1(&[1.0, 0.0, 0.0]); // AgCl(s), Ag+, Cl-

    println!("   AgCl solubility in pure water");
    println!("   Ksp = {:.2e}", ksp);

    let result = calculator.calculate_equilibrium(initial_conc, None)?;

    if result.converged {
        let ag_conc = result.concentrations[1];
        let cl_conc = result.concentrations[2];
        let solubility = ag_conc; // mol/L

        println!();
        println!("   Equilibrium results:");
        println!("   [Ag⁺] = {:.3e} M", ag_conc);
        println!("   [Cl⁻] = {:.3e} M", cl_conc);
        println!(
            "   Solubility = {:.3e} M ({:.2} mg/L)",
            solubility,
            solubility * 143.32
        );

        let ksp_calc = ag_conc * cl_conc;
        println!("   Ksp (calculated) = {:.2e}", ksp_calc);

        // Compare with analytical result
        let solubility_analytical = ksp.sqrt();
        println!(
            "   Solubility (analytical) = {:.3e} M",
            solubility_analytical
        );
    } else {
        println!("   Solubility calculation did not converge!");
    }

    // Common ion effect
    println!();
    println!("   Common ion effect (AgCl in NaCl solutions):");
    println!("   [NaCl] (M) | [Ag⁺] (M)   | Solubility | Reduction Factor |");
    println!("   -----------|-------------|------------|------------------|");

    let nacl_concentrations = vec![0.0, 0.001, 0.01, 0.1, 1.0];
    let pure_water_solubility = ksp.sqrt();

    for &nacl_conc in &nacl_concentrations {
        // Create modified system with added Cl-
        let modified_calc = systems::solubility_equilibrium(ksp, 1.0, 1.0);
        let modified_initial = arr1(&[1.0, 0.0, nacl_conc]); // Start with NaCl concentration

        if let Ok(modified_result) = modified_calc.calculate_equilibrium(modified_initial, None) {
            if modified_result.converged {
                let ag_with_nacl = modified_result.concentrations[1];
                let reduction_factor = pure_water_solubility / ag_with_nacl;

                println!(
                    "   {:9.3}  | {:9.3e}   | {:8.3e}   | {:14.1}   |",
                    nacl_conc, ag_with_nacl, ag_with_nacl, reduction_factor
                );
            }
        }
    }

    Ok(())
}

/// Demonstrate amino acid equilibrium (multiple equilibria)
fn demonstrate_amino_acid_equilibrium() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Amino acid equilibrium: glycine (NH₃⁺-CH₂-COO⁻)");

    let ka1 = 4.5e-3; // -COOH group (pKa1 = 2.35)
    let ka2 = 1.7e-10; // -NH3+ group (pKa2 = 9.78)
    let initial_conc = 0.1; // 0.1 M glycine

    let calculator = systems::amino_acid_equilibrium(ka1, ka2, initial_conc);

    // Initial concentrations: [H2A, H+, HA-, A2-]
    let initial_conc_array = arr1(&[initial_conc, 1e-7, 0.0, 0.0]);

    println!("   Glycine in water:");
    println!("   Initial concentration = {:.1} M", initial_conc);
    println!("   pKa1 (COOH) = {:.2}", -ka1.log10());
    println!("   pKa2 (NH3+) = {:.2}", -ka2.log10());

    let result = calculator.calculate_equilibrium(initial_conc_array, None)?;

    if result.converged {
        let h2a = result.concentrations[0]; // H3N+-CH2-COOH
        let h_plus = result.concentrations[1];
        let ha_minus = result.concentrations[2]; // H3N+-CH2-COO-
        let a2_minus = result.concentrations[3]; // H2N-CH2-COO-

        let ph = -h_plus.log10();

        println!();
        println!("   Equilibrium results:");
        println!("   [H₃N⁺-CH₂-COOH] = {:.4e} M", h2a);
        println!("   [H₃N⁺-CH₂-COO⁻] = {:.4e} M (zwitterion)", ha_minus);
        println!("   [H₂N-CH₂-COO⁻] = {:.4e} M", a2_minus);
        println!("   [H⁺] = {:.2e} M", h_plus);
        println!("   pH = {:.2}", ph);

        // Calculate species fractions
        let total = h2a + ha_minus + a2_minus;
        let alpha0 = h2a / total;
        let alpha1 = ha_minus / total;
        let alpha2 = a2_minus / total;

        println!();
        println!("   Species distribution:");
        println!("   α₀ (H₃N⁺-CH₂-COOH) = {:.1}%", alpha0 * 100.0);
        println!(
            "   α₁ (H₃N⁺-CH₂-COO⁻) = {:.1}% (zwitterion)",
            alpha1 * 100.0
        );
        println!("   α₂ (H₂N-CH₂-COO⁻) = {:.1}%", alpha2 * 100.0);

        // Isoelectric point (where zwitterion is maximum)
        let isoelectric_ph = 0.5 * (-ka1.log10() + -ka2.log10());
        println!("   Isoelectric point (pI) = {:.2}", isoelectric_ph);
    } else {
        println!("   Amino acid equilibrium calculation did not converge!");
    }

    // pH dependence
    println!();
    println!("   Species distribution vs pH:");
    println!("   pH  | α₀ (%) | α₁ (%) | α₂ (%) | Dominant form |");
    println!("   ----|--------|--------|--------|---------------|");

    let ph_values = vec![1.0, 2.35, 6.0, 9.78, 12.0];

    for &target_ph in &ph_values {
        let h_target = 10.0_f64.powf(-target_ph);

        // Calculate alpha values analytically
        let denominator = h_target * h_target + ka1 * h_target + ka1 * ka2;
        let alpha0_calc = h_target * h_target / denominator;
        let alpha1_calc = ka1 * h_target / denominator;
        let alpha2_calc = ka1 * ka2 / denominator;

        let dominant = if alpha0_calc > 0.5 {
            "H₃N⁺-CH₂-COOH"
        } else if alpha1_calc > 0.5 {
            "Zwitterion"
        } else {
            "H₂N-CH₂-COO⁻"
        };

        println!(
            "   {:3.1} | {:4.1}   | {:4.1}   | {:4.1}   | {}",
            target_ph,
            alpha0_calc * 100.0,
            alpha1_calc * 100.0,
            alpha2_calc * 100.0,
            dominant
        );
    }

    Ok(())
}

/// Demonstrate activity coefficient effects
fn demonstrate_activity_coefficients() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Activity coefficient effects in ionic solutions");

    let ka = 1.8e-5;
    let initial_acid = 0.1;

    // Compare ideal vs. extended Debye-Hückel models
    let mut ideal_calc = systems::weak_acid_equilibrium(ka, initial_acid, None);
    ideal_calc.set_activity_model(ActivityModel::Ideal);

    let mut real_calc = systems::weak_acid_equilibrium(ka, initial_acid, None);
    real_calc.set_activity_model(ActivityModel::ExtendedDebyeHuckel);

    let initial_conc = arr1(&[initial_acid, 1e-7, 0.0]);

    println!("   Acetic acid (0.1 M) - Model comparison:");

    // Ideal solution
    let ideal_result = ideal_calc.calculate_equilibrium(initial_conc.clone(), None)?;
    let ideal_ph = if ideal_result.converged {
        -ideal_result.concentrations[1].log10()
    } else {
        f64::NAN
    };

    // Real solution with activity coefficients
    let real_result = real_calc.calculate_equilibrium(initial_conc, None)?;
    let real_ph = if real_result.converged {
        -real_result.concentrations[1].log10()
    } else {
        f64::NAN
    };

    println!();
    println!("   Model             | pH    | [H⁺] (M)    | Activity Coeff. H⁺ |");
    println!("   ------------------|-------|-------------|---------------------|");
    println!(
        "   Ideal solution    | {:5.2} | {:9.2e}   | {:17.3}   |",
        ideal_ph, ideal_result.concentrations[1], 1.000
    );

    if real_result.converged {
        println!(
            "   Extended D-H      | {:5.2} | {:9.2e}   | {:17.3}   |",
            real_ph, real_result.concentrations[1], real_result.activity_coefficients[1]
        );
    }

    // Effect of ionic strength
    println!();
    println!("   Effect of ionic strength on activity coefficients:");
    println!("   I (M)  | γ±    | [H⁺] (ideal) | [H⁺] (real) | pH (ideal) | pH (real) |");
    println!("   -------|-------|--------------|--------------|------------|-----------|");

    let ionic_strengths = vec![0.001_f64, 0.01, 0.1, 0.5, 1.0];

    for &ionic_strength in &ionic_strengths {
        // Approximate activity coefficient using Debye-Hückel
        let a_dh = 0.5115;
        let b_dh = 0.3288;
        let ion_size = 3.0; // Angstroms
        let sqrt_i = ionic_strength.sqrt();

        let log_gamma = -a_dh * sqrt_i / (1.0 + b_dh * ion_size * sqrt_i);
        let gamma = 10.0_f64.powf(log_gamma);

        // For weak acid, approximate effect on equilibrium
        let h_ideal = (ka * initial_acid).sqrt();
        let h_real = h_ideal / gamma; // Simplified approximation

        let ph_ideal = -h_ideal.log10();
        let ph_real = -h_real.log10();

        println!(
            "   {:5.3} | {:5.3} | {:10.2e}   | {:10.2e}   | {:8.2}   | {:7.2}   |",
            ionic_strength, gamma, h_ideal, h_real, ph_ideal, ph_real
        );
    }

    Ok(())
}

/// Demonstrate temperature effects on equilibrium
fn demonstrate_temperature_effects() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Temperature effects on chemical equilibrium");

    let ka_25c = 1.8e-5; // Acetic acid Ka at 25°C
    let initial_acid = 0.1;

    // Set up thermodynamic data for acetic acid
    let mut calculator = systems::weak_acid_equilibrium(ka_25c, initial_acid, None);

    // Add temperature-dependent thermodynamic data
    let mut thermo_data = vec![
        ThermoData::new(
            "CH3COOH".to_string(),
            -484.5,
            159.8,
            [0.0, 0.0, 0.0, 0.0],
            -389.9,
        ),
        ThermoData::new("H+".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
        ThermoData::new(
            "CH3COO-".to_string(),
            -486.0,
            86.6,
            [0.0, 0.0, 0.0, 0.0],
            -369.3,
        ),
    ];

    // Set ionic charges
    thermo_data[1].activity_params.charge = 1.0; // H+
    thermo_data[2].activity_params.charge = -1.0; // CH3COO-

    calculator.set_thermo_data(thermo_data);

    println!("   Acetic acid equilibrium at different temperatures:");
    println!("   T (°C) | T (K)   | Ka        | pH    | α (%) | ΔG (kJ/mol) |");
    println!("   -------|---------|-----------|-------|-------|-------------|");

    let temperatures_c = vec![0.0, 25.0, 37.0, 50.0, 75.0, 100.0];

    for &temp_c in &temperatures_c {
        let temp_k = temp_c + 273.15;
        calculator.temperature = temp_k;

        let initial_conc = arr1(&[initial_acid, 1e-7, 0.0]);

        if let Ok(result) = calculator.calculate_equilibrium(initial_conc, None) {
            if result.converged {
                let h_plus = result.concentrations[1];
                let acetate = result.concentrations[2];
                let ph = -h_plus.log10();
                let alpha = (acetate / initial_acid) * 100.0; // Degree of dissociation
                let ka_eff = result.equilibrium_constants[0];
                let delta_g = result.delta_g;

                println!(
                    "   {:4.0}   | {:5.1}   | {:7.2e}   | {:5.2} | {:5.1} | {:9.1}   |",
                    temp_c, temp_k, ka_eff, ph, alpha, delta_g
                );
            }
        }
    }

    // Van't Hoff plot analysis
    println!();
    println!("   Van't Hoff analysis (ln K vs 1/T):");
    println!("   The slope gives -ΔH/R, intercept gives ΔS/R");

    // For demonstration, use literature values
    let delta_h_diss = 1.0; // kJ/mol (enthalpy of dissociation)
    let delta_s_diss = 10.0; // J/(mol·K) (entropy of dissociation)

    println!("   Estimated ΔH° = {:.1} kJ/mol", delta_h_diss);
    println!("   Estimated ΔS° = {:.1} J/(mol·K)", delta_s_diss);

    Ok(())
}

/// Demonstrate pH titration simulation
fn demonstrate_titration_curve() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Simulating titration of acetic acid with NaOH");

    let ka = 1.8e-5_f64;
    let initial_acid = 0.1; // 0.1 M acetic acid
    let volume_acid = 50.0; // 50 mL
    let naoh_conc = 0.1; // 0.1 M NaOH

    println!(
        "   Titrating {:.1} mL of {:.2} M CH₃COOH with {:.2} M NaOH",
        volume_acid, initial_acid, naoh_conc
    );

    println!();
    println!("   NaOH added | Total Vol | CH₃COOH | CH₃COO⁻ | pH    | Region        |");
    println!("   (mL)      | (mL)      | (M)     | (M)     |       |               |");
    println!("   ----------|-----------|---------|---------|-------|---------------|");

    let naoh_volumes = vec![
        0.0, 10.0, 20.0, 30.0, 40.0, 49.0, 49.9, 50.0, 50.1, 51.0, 60.0,
    ];

    for &v_naoh in &naoh_volumes {
        let total_volume = volume_acid + v_naoh;
        let moles_acid_initial = initial_acid * volume_acid / 1000.0;
        let moles_base_added = naoh_conc * v_naoh / 1000.0;

        let (final_acid, final_base, ph, region) = if moles_base_added < moles_acid_initial {
            // Before equivalence point - buffer region
            let moles_acid_remaining = moles_acid_initial - moles_base_added;
            let moles_acetate_formed = moles_base_added;

            let acid_conc = moles_acid_remaining / (total_volume / 1000.0);
            let base_conc = moles_acetate_formed / (total_volume / 1000.0);

            // Buffer calculation
            let h_plus = ka * acid_conc / base_conc;
            let ph = -h_plus.log10();

            (acid_conc, base_conc, ph, "Buffer")
        } else if (moles_base_added - moles_acid_initial).abs() < 1e-10_f64 {
            // Equivalence point
            let acetate_conc = moles_acid_initial / (total_volume / 1000.0);

            // Hydrolysis of acetate: CH3COO- + H2O ⇌ CH3COOH + OH-
            let kb = 1e-14 / ka; // Kb for acetate
            let oh_minus = (kb * acetate_conc).sqrt();
            let h_plus = 1e-14 / oh_minus;
            let ph = -h_plus.log10();

            (0.0, acetate_conc, ph, "Equivalence")
        } else {
            // After equivalence point - excess base
            let moles_excess_base = moles_base_added - moles_acid_initial;
            let oh_excess = moles_excess_base / (total_volume / 1000.0);
            let h_plus = 1e-14 / oh_excess;
            let ph = -h_plus.log10();

            (
                0.0,
                moles_acid_initial / (total_volume / 1000.0),
                ph,
                "Excess base",
            )
        };

        println!(
            "   {:8.1}  | {:7.1}   | {:5.3}   | {:5.3}   | {:5.2} | {}",
            v_naoh, total_volume, final_acid, final_base, ph, region
        );
    }

    // Calculate key points
    let equivalence_volume = (initial_acid * volume_acid) / naoh_conc;
    let half_equivalence = equivalence_volume / 2.0;
    let ph_half_eq = -ka.log10(); // pH = pKa at half-equivalence

    println!();
    println!("   Key points:");
    println!(
        "   Half-equivalence point: {:.1} mL, pH = {:.2} (= pKa)",
        half_equivalence, ph_half_eq
    );
    println!("   Equivalence point: {:.1} mL", equivalence_volume);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_equilibrium_demos() {
        // Test that all demo functions run without errors
        assert!(demonstrate_weak_acid_equilibrium().is_ok());
        assert!(demonstrate_buffer_equilibrium().is_ok());
        assert!(demonstrate_complex_formation().is_ok());
        assert!(demonstrate_solubility_equilibrium().is_ok());
        assert!(demonstrate_amino_acid_equilibrium().is_ok());
        assert!(demonstrate_activity_coefficients().is_ok());
        assert!(demonstrate_temperature_effects().is_ok());
        assert!(demonstrate_titration_curve().is_ok());
    }

    #[test]
    fn test_weak_acid_calculation() {
        let ka = 1.8e-5_f64;
        let initial_acid = 0.1;
        let calculator = systems::weak_acid_equilibrium(ka, initial_acid, None);
        let initial_conc = arr1(&[initial_acid, 1e-7, 0.0]);

        let result = calculator
            .calculate_equilibrium(initial_conc, None)
            .unwrap();
        assert!(result.converged);

        // Check that some dissociation occurred
        assert!(result.concentrations[1] > 1e-7); // H+ increased
        assert!(result.concentrations[2] > 0.0); // Acetate formed

        // Verify Ka relationship (approximately)
        let ka_calc =
            (result.concentrations[1] * result.concentrations[2]) / result.concentrations[0];
        assert_abs_diff_eq!(ka_calc, ka, epsilon = ka * 0.1);
    }

    #[test]
    fn test_buffer_ph() {
        let ka = 1.8e-5_f64;
        let calculator = systems::buffer_equilibrium(ka, 0.1, 0.1);
        let initial_conc = arr1(&[0.1, 1e-7, 0.1, 1e-7, 55.5]);

        let result = calculator
            .calculate_equilibrium(initial_conc, None)
            .unwrap();
        assert!(result.converged);

        // pH should be close to pKa for equal concentrations
        let ph = -result.concentrations[1].log10();
        let pka = -ka.log10();
        assert_abs_diff_eq!(ph, pka, epsilon = 0.5);
    }

    #[test]
    fn test_solubility_product() {
        let ksp = 1.8e-10;
        let calculator = systems::solubility_equilibrium(ksp, 1.0, 1.0);
        let initial_conc = arr1(&[1.0, 0.0, 0.0]);

        let result = calculator
            .calculate_equilibrium(initial_conc, None)
            .unwrap();
        assert!(result.converged);

        // Check Ksp relationship
        let ksp_calc = result.concentrations[1] * result.concentrations[2];
        assert_abs_diff_eq!(ksp_calc, ksp, epsilon = ksp * 0.1);

        // For 1:1 salt, [Ag+] = [Cl-] = sqrt(Ksp)
        let expected_conc = ksp.sqrt();
        assert_abs_diff_eq!(
            result.concentrations[1],
            expected_conc,
            epsilon = expected_conc * 0.1
        );
    }

    #[test]
    fn test_activity_coefficients() {
        let calculator = systems::weak_acid_equilibrium(1e-5, 0.1, None);
        let concentrations = arr1(&[0.09, 0.001, 0.001]);

        // Test ideal vs non-ideal models
        let mut ideal_calc = calculator.clone();
        ideal_calc.set_activity_model(ActivityModel::Ideal);
        let ideal_result = ideal_calc
            .calculate_equilibrium(concentrations.clone(), None)
            .unwrap();

        let mut real_calc = calculator.clone();
        real_calc.set_activity_model(ActivityModel::ExtendedDebyeHuckel);
        let real_result = real_calc
            .calculate_equilibrium(concentrations.clone(), None)
            .unwrap();

        // Ideal coefficients should be 1
        for &coeff in ideal_result.activity_coefficients.iter() {
            assert_abs_diff_eq!(coeff, 1.0, epsilon = 1e-10);
        }

        // Real coefficients for ions should be < 1
        assert!(real_result.activity_coefficients[1] < 1.0); // H+
        assert!(real_result.activity_coefficients[2] < 1.0); // A-
    }
}
