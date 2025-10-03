//! Simple Enzyme Kinetics Test
//!
//! A minimal test of the enzyme kinetics functionality to verify it works.

use scirs2_core::ndarray::Array1;
use scirs2_integrate::ode::enzyme_kinetics::{pathways, EnzymeParameters};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simple Enzyme Kinetics Test\n");

    // Test 1: Michaelis-Menten kinetics
    println!("1. Testing Michaelis-Menten kinetics");
    let mut params = EnzymeParameters::michaelis_menten(1.0, 100.0);
    params.temperature = 298.15; // Set to reference temperature to avoid correction
    let substrate_conc = 1.0; // At Km
    let rate = params.calculate_rate(&[substrate_conc]);
    println!("   Rate at Km = {rate:.1} μM/s (should be ~50)");
    assert!((rate - 50.0).abs() < 1.0);

    // Test 2: Hill kinetics
    println!("\n2. Testing Hill kinetics");
    let mut hill_params = EnzymeParameters::hill(1.0, 100.0, 2.0);
    hill_params.temperature = 298.15; // Set to reference temperature to avoid correction
    let rate_hill = hill_params.calculate_rate(&[substrate_conc]);
    println!("   Hill rate at Kd = {rate_hill:.1} μM/s (should be ~50)");
    assert!((rate_hill - 50.0).abs() < 1.0);

    // Test 3: Simple glycolysis pathway
    println!("\n3. Testing simple glycolysis pathway");
    let pathway = pathways::simple_glycolysis();
    println!("   Pathway: {}", pathway.name);
    println!("   Enzymes: {}", pathway.enzymes.len());
    println!("   Metabolites: {}", pathway.metabolites.len());

    // Test rate calculation
    let concentrations = Array1::from_vec(vec![5.0, 1.0, 0.5, 0.3, 0.2, 0.1]);
    let rates = pathway.calculate_reaction_rates(&concentrations);
    println!("   Reaction rates calculated: {} reactions", rates.len());

    for (i, &rate) in rates.iter().enumerate() {
        println!("   Reaction {}: {:.3e} μM/s", i + 1, rate * 1000.0);
        assert!(rate >= 0.0); // All rates should be non-negative
    }

    // Test 4: TCA cycle pathway
    println!("\n4. Testing TCA cycle pathway");
    let tca_pathway = pathways::tca_cycle();
    println!("   TCA Pathway: {}", tca_pathway.name);
    println!("   Enzymes: {}", tca_pathway.enzymes.len());

    let tca_concentrations = Array1::from_vec(vec![1.0; 8]);
    let tca_rates = tca_pathway.calculate_reaction_rates(&tca_concentrations);
    println!("   TCA rates calculated: {} reactions", tca_rates.len());

    for (i, &rate) in tca_rates.iter().enumerate() {
        println!("   TCA reaction {}: {:.3e} μM/s", i + 1, rate * 1000.0);
        assert!(rate >= 0.0);
    }

    // Test 5: Temperature effects
    println!("\n5. Testing temperature effects");
    let mut temp_params = EnzymeParameters::michaelis_menten(1.0, 100.0);

    temp_params.temperature = 298.15; // 25°C
    let rate_25c = temp_params.calculate_rate(&[1.0]);

    temp_params.temperature = 310.15; // 37°C
    let rate_37c = temp_params.calculate_rate(&[1.0]);

    println!("   Rate at 25°C: {rate_25c:.1} μM/s");
    println!("   Rate at 37°C: {rate_37c:.1} μM/s");
    println!("   Temperature factor: {:.2}", rate_37c / rate_25c);

    assert!(rate_37c > rate_25c); // Rate should increase with temperature

    println!("\n✅ All enzyme kinetics tests passed!");

    Ok(())
}
