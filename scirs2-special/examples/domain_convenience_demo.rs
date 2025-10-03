//! Demonstration of domain-specific convenience functions
//!
//! This example shows various convenience functions for different
//! scientific and engineering domains.

use scirs2_core::ndarray::Array1;
use scirs2_special::convenience::*;

#[allow(dead_code)]
fn main() {
    println!("=== Domain-Specific Convenience Functions Demo ===\n");

    // Physics examples
    println!("1. Physics Functions");
    println!("--------------------");

    // Particle in a box
    let x = Array1::linspace(0.0, 1.0, 100);
    let psi = physics::particle_in_box_wavefunction(2, &x.view(), true).unwrap();
    println!(
        "Particle in box (n=2): max ψ = {:.3}",
        psi.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    );

    // Hydrogen radial wavefunction
    let r = Array1::linspace(0.1, 20.0, 50);
    let psi_r = physics::hydrogen_radial_wavefunction(2, 1, &r.view(), 1.0).unwrap();
    println!(
        "Hydrogen 2p radial: max = {:.3e}",
        psi_r
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    );

    // Fermi-Dirac distribution
    let energy = Array1::linspace(-2.0, 2.0, 50);
    let fermi = physics::fermi_dirac_distribution(&energy.view(), 0.0, 300.0).unwrap();
    println!("Fermi-Dirac at EF: f(0) = {:.3}", fermi[25]);

    // Engineering examples
    println!("\n2. Engineering Functions");
    println!("------------------------");

    // Q-function for communications
    let snr = Array1::from(vec![0.0, 3.0, 6.0, 9.0]);
    let q_vals = engineering::q_function(&snr.view());
    println!("Q-function values: {:?}", q_vals.to_vec());

    // Antenna array factor
    let theta = Array1::linspace(0.0, std::f64::consts::PI, 180);
    let af = engineering::antenna_array_factor(&theta.view(), 4, 0.5, 0.0);
    println!(
        "Array factor max: {:.3}",
        af.iter()
            .map(|c| c.norm())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    );

    // Data Science examples
    println!("\n3. Data Science Functions");
    println!("-------------------------");

    // Gini coefficient
    let income = Array1::from(vec![10.0, 20.0, 30.0, 40.0, 100.0]);
    let gini = data_science::gini_coefficient(&income.view()).unwrap();
    println!("Gini coefficient: {gini:.3}");

    // Shannon entropy
    let probs = Array1::from(vec![0.25, 0.25, 0.25, 0.25]);
    let entropy = data_science::shannon_entropy(&probs.view(), 2.0).unwrap();
    println!("Shannon entropy (uniform): {entropy:.3} bits");

    // KL divergence
    let p = Array1::from(vec![0.4, 0.3, 0.2, 0.1]);
    let q = Array1::from(vec![0.25, 0.25, 0.25, 0.25]);
    let kl = data_science::kl_divergence(&p.view(), &q.view()).unwrap();
    println!("KL divergence: {kl:.3}");

    // Signal Processing examples
    println!("\n4. Signal Processing Functions");
    println!("------------------------------");

    // Gaussian window
    let gauss_win = signal_processing::gaussian_window(64, 0.4);
    println!("Gaussian window sum: {:.3}", gauss_win.sum());

    // Kaiser window
    let kaiser_win = signal_processing::kaiser_window(64, 8.6);
    println!("Kaiser window sum: {:.3}", kaiser_win.sum());

    // Finance examples
    println!("\n5. Finance Functions");
    println!("--------------------");

    // Black-Scholes option pricing
    let call_price = finance::black_scholes_call(100.0, 95.0, 0.05, 0.2, 0.25).unwrap();
    println!("Black-Scholes call price: ${call_price:.2}");

    // Implied volatility
    let impl_vol = finance::implied_volatility(10.0, 100.0, 95.0, 0.05, 0.25, 100).unwrap();
    println!("Implied volatility: {:.1}%", impl_vol * 100.0);

    // Bioinformatics examples
    println!("\n6. Bioinformatics Functions");
    println!("---------------------------");

    // Jukes-Cantor distance
    let jc_dist = bioinformatics::jukes_cantor_distance(0.2).unwrap();
    println!("Jukes-Cantor distance (p=0.2): {jc_dist:.3}");

    // Michaelis-Menten kinetics
    let substrate = Array1::linspace(0.0, 10.0, 50);
    let velocity = bioinformatics::michaelis_menten(&substrate.view(), 100.0, 2.0).unwrap();
    println!(
        "Michaelis-Menten Vmax/2 at: S = {:.1}",
        substrate[velocity.iter().position(|&v| v >= 50.0).unwrap()]
    );

    // Logistic growth
    let time = Array1::linspace(0.0, 10.0, 50);
    let population = bioinformatics::logistic_growth(&time.view(), 1000.0, 0.5, 10.0).unwrap();
    println!("Logistic growth at t=5: {:.0}", population[25]);

    // Geophysics examples
    println!("\n7. Geophysics Functions");
    println!("-----------------------");

    // Acoustic impedance
    let z = geophysics::acoustic_impedance(3000.0, 2500.0).unwrap();
    println!("Acoustic impedance: {z:.0} kg/(m²·s)");

    // Richter magnitude
    let magnitude = geophysics::richter_magnitude(1000.0, 100.0).unwrap();
    println!("Richter magnitude: {magnitude:.1}");

    // Barometric pressure
    let altitudes = Array1::from(vec![0.0, 1000.0, 5000.0, 10000.0]);
    let pressures = geophysics::barometric_pressure(&altitudes.view(), 101325.0).unwrap();
    println!("Pressure at 5km: {:.0} Pa", pressures[2]);

    // Chemistry examples
    println!("\n8. Chemistry Functions");
    println!("----------------------");

    // Arrhenius rate
    let temps = Array1::from(vec![300.0, 400.0, 500.0]);
    let rates = chemistry::arrhenius_rate(&temps.view(), 50000.0, 1e10).unwrap();
    println!(
        "Arrhenius rates: {:?}",
        rates.mapv(|r| format!("{r:.2e}")).to_vec()
    );

    // Debye-Hückel activity
    let activity = chemistry::debye_huckel_activity(0.01, 2.0, 298.15).unwrap();
    println!("Activity coefficient: {activity:.3}");

    // van der Waals pressure
    let pressure = chemistry::van_der_waals_pressure(0.03, 300.0, 1.355, 0.0322).unwrap();
    println!("van der Waals pressure: {pressure:.0} Pa");

    // Astronomy examples
    println!("\n9. Astronomy Functions");
    println!("----------------------");

    // Stellar luminosity
    let luminosity = astronomy::stellar_luminosity(2.0, 10000.0).unwrap();
    println!("Stellar luminosity (R=2R☉, T=10000K): {luminosity:.1} L☉");

    // Saha equation
    let ionization_fraction = astronomy::saha_equation(5000.0, 1e20, 13.6).unwrap();
    println!("Saha ionization fraction: {ionization_fraction:.2e}");

    // Jeans mass
    let jeans_m = astronomy::jeans_mass(10.0, 1e-19, 2.0).unwrap();
    println!("Jeans mass: {jeans_m:.2e} M☉");

    // Redshift
    let z = astronomy::velocity_to_redshift(30000000.0, true).unwrap();
    println!("Redshift at 0.1c: z = {z:.3}");

    println!("\n=== Demo Complete ===");
}
