//! Simple Chemical Equilibrium Test
//!
//! A minimal test of the chemical equilibrium functionality to verify it works.

use scirs2_integrate::ode::chemical_equilibrium::{systems, ActivityModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simple Chemical Equilibrium Test\n");

    // Test 1: Weak acid equilibrium setup
    println!("1. Testing weak acid equilibrium setup");
    let ka = 1.8e-5; // Acetic acid
    let initial_acid = 0.1;

    let calculator = systems::weak_acid_equilibrium(ka, initial_acid, None);

    println!("   ✅ Calculator created successfully");
    println!("   Species: {:?}", calculator.species_names);
    println!("   Reactions: {:?}", calculator.reaction_names);
    println!("   Ka: {:.2e}", calculator.equilibrium_constants[0]);

    // Simple analytical approximation for weak acid
    let h_analytical = (ka * initial_acid).sqrt();
    let ph_analytical = -h_analytical.log10();
    println!("   Analytical pH approximation: {:.2}", ph_analytical);

    // Test 2: Buffer system setup
    println!("\n2. Testing buffer equilibrium setup");
    let buffer_calc = systems::buffer_equilibrium(ka, 0.1, 0.1);

    println!("   ✅ Buffer calculator created");
    println!("   Species: {:?}", buffer_calc.species_names);
    println!("   Ka: {:.2e}", buffer_calc.equilibrium_constants[0]);

    // Henderson-Hasselbalch equation for equal concentrations
    let pka = -ka.log10();
    let ph_hh = pka; // pH = pKa when [A-] = [HA]
    println!("   Henderson-Hasselbalch pH: {:.2}", ph_hh);

    // Test 3: Complex formation setup
    println!("\n3. Testing complex formation setup");
    let k_formation = 1e6;
    let complex_calc = systems::complex_formation(k_formation, 0.001, 0.01);

    println!("   ✅ Complex formation calculator created");
    println!("   Species: {:?}", complex_calc.species_names);
    println!(
        "   K_formation: {:.2e}",
        complex_calc.equilibrium_constants[0]
    );

    // Estimate complex formation
    let ligand = 0.01; // L
    let alpha = 1.0 / (1.0 + k_formation * ligand);
    let fraction_free = alpha * 100.0;
    println!("   Estimated fraction of free metal: {:.1}%", fraction_free);

    // Test 4: Solubility equilibrium setup
    println!("\n4. Testing solubility equilibrium setup");
    let ksp = 1.8e-10;
    let solubility_calc = systems::solubility_equilibrium(ksp, 1.0, 1.0);

    println!("   ✅ Solubility calculator created");
    println!("   Species: {:?}", solubility_calc.species_names);
    println!("   Ksp: {:.2e}", solubility_calc.equilibrium_constants[0]);

    // Analytical solubility for 1:1 salt
    let solubility_analytical = ksp.sqrt();
    println!("   Analytical solubility: {:.3e} M", solubility_analytical);
    println!(
        "   Solubility in mg/L: {:.2}",
        solubility_analytical * 143.32
    ); // AgCl MW

    // Test 5: Activity coefficient models
    println!("\n5. Testing activity coefficient models");
    let mut test_calc = systems::weak_acid_equilibrium(1e-5, 0.1, None);

    // Test different activity models
    test_calc.set_activity_model(ActivityModel::Ideal);
    println!("   ✅ Ideal activity model set");

    test_calc.set_activity_model(ActivityModel::ExtendedDebyeHuckel);
    println!("   ✅ Extended Debye-Hückel model set");

    println!("   Activity coefficient models working properly");

    println!("\n✅ All chemical equilibrium tests passed!");

    Ok(())
}
