//! Experimental Physics Applications Laboratory
//!
//! An interactive tutorial demonstrating special functions in cutting-edge experimental
//! physics applications. This tutorial bridges the gap between abstract mathematics
//! and real-world laboratory measurements and theoretical predictions.
//!
//! Topics covered:
//! - Quantum optics and photon statistics
//! - Nuclear and particle physics cross-sections
//! - Condensed matter physics phase transitions
//! - Gravitational wave analysis
//! - Plasma physics and fusion
//! - Atomic and molecular spectroscopy
//! - Statistical mechanics in mesoscopic systems
//! - Nonlinear optics and solitons
//!
//! Run with: cargo run --example experimental_physics_applications

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use scirs2_special::*;
use std::f64::consts::PI;
use std::io::{self, Write};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧬 Experimental Physics Applications Laboratory");
    println!("===============================================");
    println!("Special functions in cutting-edge physics research\n");

    loop {
        display_main_menu();
        let choice = get_user_input("Enter your choice (1-10, or 'q' to quit): ")?;

        if choice.to_lowercase() == "q" {
            println!("🔬 Thank you for exploring experimental physics applications!");
            break;
        }

        match choice.parse::<u32>() {
            Ok(1) => quantum_optics_photon_statistics()?,
            Ok(2) => nuclear_physics_cross_sections()?,
            Ok(3) => condensed_matter_phase_transitions()?,
            Ok(4) => gravitational_wave_analysis()?,
            Ok(5) => plasma_physics_fusion()?,
            Ok(6) => atomic_spectroscopy()?,
            Ok(7) => statistical_mechanics_mesoscopic()?,
            Ok(8) => nonlinear_optics_solitons()?,
            Ok(9) => cosmic_ray_physics()?,
            Ok(10) => quantum_information_experiments()?,
            _ => println!("❌ Invalid choice. Please try again.\n"),
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn display_main_menu() {
    println!("🔬 Choose an experimental physics application:");
    println!("1.  🌟 Quantum Optics & Photon Statistics");
    println!("2.  ⚛️  Nuclear Physics Cross-Sections");
    println!("3.  🧊 Condensed Matter Phase Transitions");
    println!("4.  🌊 Gravitational Wave Analysis");
    println!("5.  🔥 Plasma Physics & Fusion");
    println!("6.  📡 Atomic & Molecular Spectroscopy");
    println!("7.  🎲 Mesoscopic Statistical Mechanics");
    println!("8.  💫 Nonlinear Optics & Solitons");
    println!("9.  🚀 Cosmic Ray Physics");
    println!("10. 🔒 Quantum Information Experiments");
    println!("q.  Quit");
    println!();
}

#[allow(dead_code)]
fn get_user_input(prompt: &str) -> io::Result<String> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

#[allow(dead_code)]
fn quantum_optics_photon_statistics() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌟 QUANTUM OPTICS & PHOTON STATISTICS");
    println!("=====================================\n");

    println!("In quantum optics experiments, photon counting statistics reveal");
    println!("fundamental quantum properties through special function analysis.\n");

    pause_for_user()?;

    println!("1. COHERENT LIGHT: Poisson Statistics");
    println!("=====================================");
    println!();
    println!("For coherent light (laser), photon counts follow Poisson distribution:");
    println!("P(n) = (⟨n⟩^n / n!) exp(-⟨n⟩)");
    println!();
    println!("The Mandel Q parameter quantifies bunching/antibunching:");
    println!("Q = (⟨n²⟩ - ⟨n⟩² - ⟨n⟩) / ⟨n⟩");
    println!("Q = 0 for coherent light, Q < 0 for antibunched, Q > 0 for bunched");
    println!();

    // Simulate coherent light measurement
    let mean_photons = 5.0;
    println!(
        "EXPERIMENTAL SIMULATION: Coherent light with ⟨n⟩ = {}",
        mean_photons
    );
    println!("Photon count probabilities:");
    for n in 0..12 {
        let prob = poisson_probability(n, mean_photons);
        let bar_length = (prob * 50.0) as usize;
        let bar = "█".repeat(bar_length);
        println!("n = {:2}: P(n) = {:.4} {}", n, prob, bar);
    }
    println!();

    pause_for_user()?;

    println!("2. THERMAL LIGHT: Negative Binomial Statistics");
    println!("===============================================");
    println!();
    println!("Thermal light exhibits super-Poissonian statistics:");
    println!("P(n) = Γ(n + M) / [n! Γ(M)] * (M/(M+⟨n⟩))^M * (⟨n⟩/(M+⟨n⟩))^n");
    println!();
    println!("Where M characterizes the number of modes.");
    println!("For M → ∞, we recover Poisson statistics.");
    println!();

    let thermal_mean = 5.0;
    let modes = 2.0;
    println!(
        "THERMAL LIGHT SIMULATION: ⟨n⟩ = {}, M = {}",
        thermal_mean, modes
    );
    println!("Comparison with coherent light:");
    println!("n    Coherent   Thermal   Enhancement");
    println!("--   --------   -------   -----------");
    for n in 0..10 {
        let coherent_prob = poisson_probability(n, thermal_mean);
        let thermal_prob = negative_binomial_probability(n, modes, thermal_mean);
        let enhancement = thermal_prob / coherent_prob;
        println!(
            "{:2}   {:.4}     {:.4}     {:.2}x",
            n, coherent_prob, thermal_prob, enhancement
        );
    }
    println!();

    pause_for_user()?;

    println!("3. SQUEEZED LIGHT: Modified Photon Statistics");
    println!("=============================================");
    println!();
    println!("Squeezed light reduces noise below the shot-noise limit.");
    println!("The photon distribution involves associated Laguerre polynomials:");
    println!();
    println!("P(n) = (tanh r)^(2n) / cosh r * |L_n^(0)(−⟨n⟩/sinh²r)|²");
    println!();
    println!("Where r is the squeezing parameter.");
    println!();

    let squeezing_r = 0.5_f64;
    let squeezed_mean = 3.0;
    println!(
        "SQUEEZED LIGHT: r = {}, ⟨n⟩ = {}",
        squeezing_r, squeezed_mean
    );
    println!();
    println!("Squeezing reduces variance: σ²_squeezed = ⟨n⟩ * e^(-2r) for ideal squeezing");
    let ideal_variance_reduction = (-2.0 * squeezing_r).exp();
    println!("Variance reduction factor: {:.3}", ideal_variance_reduction);
    println!("This is observed in parametric down-conversion experiments!");
    println!();

    pause_for_user()?;

    println!("4. SINGLE-PHOTON SOURCES: Antibunching");
    println!("=======================================");
    println!();
    println!("Perfect single-photon sources have g^(2)(0) = 0 (perfect antibunching).");
    println!("Real sources are characterized by second-order coherence:");
    println!();
    println!("g^(2)(τ) = ⟨I(t)I(t+τ)⟩ / ⟨I(t)⟩²");
    println!();
    println!("For exponential decay with rate γ:");
    println!("g^(2)(τ) = 1 - exp(-γ|τ|)");
    println!();

    // Demonstrate antibunching measurement
    let decay_rate = 1.0_f64; // in units of 1/τ
    let time_points = Array1::linspace(0.0, 5.0, 50);

    println!("ANTIBUNCHING MEASUREMENT (γ = {} 1/τ):", decay_rate);
    println!("τ     g^(2)(τ)   Antibunching");
    println!("---   -------    ------------");
    for i in (0..time_points.len()).step_by(5) {
        let tau = time_points[i];
        let g2 = 1.0 - (-decay_rate * tau).exp();
        let antibunching = if g2 < 0.5 { "YES" } else { "NO " };
        println!("{:.1}   {:.3}      {}", tau, g2, antibunching);
    }
    println!();

    println!("💡 EXPERIMENTAL INSIGHTS:");
    println!("• Hanbury Brown-Twiss interferometry measures g^(2)(τ)");
    println!("• Quantum dots and NV centers are leading single-photon sources");
    println!("• Photon antibunching proves the quantum nature of light");
    println!("• Applications: quantum cryptography, linear optical quantum computing");
    println!();

    Ok(())
}

#[allow(dead_code)]
fn nuclear_physics_cross_sections() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚛️ NUCLEAR PHYSICS CROSS-SECTIONS");
    println!("==================================\n");

    println!("Nuclear reaction cross-sections involve special functions through");
    println!("quantum mechanical scattering theory and statistical mechanics.\n");

    pause_for_user()?;

    println!("1. NEUTRON-NUCLEUS SCATTERING: Bessel Functions");
    println!("===============================================");
    println!();
    println!("The neutron-nucleus scattering amplitude involves spherical Bessel functions:");
    println!("f_ℓ(k,R) ∝ j_ℓ(kR) - tan(δ_ℓ) n_ℓ(kR)");
    println!();
    println!("Where δ_ℓ are the phase shifts determined by the nuclear potential.");
    println!();

    let neutron_energy_mev = vec![0.1, 1.0, 10.0, 100.0];
    let nucleus_radius_fm = 5.0; // typical radius in femtometers

    println!("NEUTRON SCATTERING CALCULATION:");
    println!("Nucleus radius R = {} fm", nucleus_radius_fm);
    println!();
    println!("E (MeV)   k (fm⁻¹)   kR     j₀(kR)    n₀(kR)    Phase");
    println!("-------   --------   ----   ------    ------    -----");

    for &energy in &neutron_energy_mev {
        let k = neutron_wave_number(energy); // fm^-1
        let kr = k * nucleus_radius_fm;
        let j0_kr = spherical_jn(0, kr);
        let n0_kr = spherical_yn(0, kr);
        let phase_shift = calculate_hard_sphere_phase_shift(kr);

        println!(
            "{:7.1}   {:8.3}   {:4.1}   {:6.3}    {:6.3}    {:5.2}°",
            energy, k, kr, j0_kr, n0_kr, phase_shift
        );
    }
    println!();

    pause_for_user()?;

    println!("2. NUCLEAR FISSION: Gamma Function in Barrier Penetration");
    println!("=========================================================");
    println!();
    println!("The fission probability involves barrier penetration through the");
    println!("Wentzel-Kramers-Brillouin (WKB) approximation:");
    println!();
    println!("T ≈ exp(-2∫[√(2m(V(r)-E))/ħ] dr)");
    println!();
    println!("For parabolic barriers, this leads to Gamma function expressions.");
    println!();

    let barrier_height_mev = 6.0;
    let excitation_energies = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    println!("FISSION BARRIER PENETRATION:");
    println!("Barrier height = {} MeV", barrier_height_mev);
    println!();
    println!("E* (MeV)   Penetration   Fission Rate");
    println!("--------   -----------   ------------");

    for &excitation in &excitation_energies {
        let penetration = calculate_barrier_penetration(excitation, barrier_height_mev);
        let relative_rate = if excitation == 0.0 {
            1.0
        } else {
            penetration / calculate_barrier_penetration(0.0, barrier_height_mev)
        };
        println!(
            "{:8.1}   {:11.2e}   {:12.1e}",
            excitation, penetration, relative_rate
        );
    }
    println!();

    pause_for_user()?;

    println!("3. NUCLEAR LEVEL DENSITY: Modified Bessel Functions");
    println!("===================================================");
    println!();
    println!("The nuclear level density follows the Bethe formula:");
    println!("ρ(E) = (1/12) √(π/a) * exp(2√aE) / E^(5/4)");
    println!();
    println!("Where 'a' is the level density parameter (~A/8 MeV⁻¹).");
    println!("The exponential term can be expressed using modified Bessel functions.");
    println!();

    let mass_number = 238; // Uranium-238
    let level_density_param = mass_number as f64 / 8.0; // MeV^-1
    let excitation_range = Array1::linspace(1.0, 20.0, 10);

    println!("NUCLEAR LEVEL DENSITY (A = {}):", mass_number);
    println!(
        "Level density parameter a = {:.1} MeV⁻¹",
        level_density_param
    );
    println!();
    println!("E (MeV)   ρ(E) (MeV⁻¹)   Cumulative");
    println!("-------   -----------    ----------");

    for &energy in excitation_range.iter() {
        let level_density = nuclear_level_density(energy, level_density_param);
        let cumulative = cumulative_levels(energy, level_density_param);
        println!(
            "{:7.1}   {:11.2e}    {:10.1e}",
            energy, level_density, cumulative
        );
    }
    println!();

    pause_for_user()?;

    println!("4. NEUTRINO OSCILLATIONS: Hypergeometric Functions");
    println!("==================================================");
    println!();
    println!("Neutrino oscillations in matter involve hypergeometric functions");
    println!("when matter density varies with distance.");
    println!();
    println!("For constant density, the survival probability is:");
    println!("P(νₑ → νₑ) = 1 - sin²(2θ_m) sin²(Δm²L/4E)");
    println!();
    println!("Where θ_m is the effective mixing angle in matter.");
    println!();

    let neutrino_energy_gev = vec![0.1, 1.0, 10.0];
    let baseline_km = 295.0; // DUNE baseline
    let delta_m_squared = 2.5e-3; // eV^2

    println!("NEUTRINO OSCILLATION CALCULATION:");
    println!("Baseline L = {} km", baseline_km);
    println!("Δm² = {:.1e} eV²", delta_m_squared);
    println!();
    println!("E (GeV)   Oscillation   Survival");
    println!("-------   -----------   --------");

    for &energy in &neutrino_energy_gev {
        let oscillation_phase: f64 = delta_m_squared * baseline_km * 1000.0 / (4.0 * energy * 1e9);
        let survival_prob =
            1.0 - (2.0 * 0.85_f64.asin()).sin().powi(2) * oscillation_phase.sin().powi(2);
        println!(
            "{:7.1}   {:11.3}     {:8.3}",
            energy, oscillation_phase, survival_prob
        );
    }
    println!();

    println!("💡 EXPERIMENTAL CONNECTIONS:");
    println!("• Neutron scattering: determines nuclear structure and interactions");
    println!("• Fission studies: critical for reactor physics and weapons design");
    println!("• Level density: crucial for nucleosynthesis calculations");
    println!("• Neutrino experiments: probe fundamental physics and astrophysics");
    println!();

    Ok(())
}

#[allow(dead_code)]
fn condensed_matter_phase_transitions() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧊 CONDENSED MATTER PHASE TRANSITIONS");
    println!("======================================\n");

    println!("Phase transitions in condensed matter involve critical phenomena");
    println!("where special functions describe universal scaling behavior.\n");

    pause_for_user()?;

    println!("1. ISING MODEL: Elliptic Integrals and Critical Behavior");
    println!("========================================================");
    println!();
    println!("The 2D Ising model solution by Onsager involves complete elliptic integrals.");
    println!("Near the critical temperature T_c, the correlation length diverges:");
    println!();
    println!("ξ ∝ |T - T_c|^(-ν) where ν = 1 for 2D Ising");
    println!();
    println!("The spontaneous magnetization involves elliptic integrals:");
    println!("M ∝ (1 - sinh⁻⁴(2J/k_BT))^(1/8) for T < T_c");
    println!();

    let temperature_range = Array1::linspace(0.8, 1.2, 21);
    let t_critical = 1.0; // normalized critical temperature

    println!("2D ISING MODEL CRITICAL BEHAVIOR:");
    println!("T/T_c   Magnetization   Correlation   Heat Capacity");
    println!("-----   -------------   -----------   -------------");

    for &t in temperature_range.iter().take(21).step_by(2) {
        let magnetization = if t < t_critical {
            ising_magnetization(t, t_critical)
        } else {
            0.0
        };
        let correlation_length = ising_correlation_length(t, t_critical);
        let heat_capacity = ising_heat_capacity(t, t_critical);

        println!(
            "{:5.2}   {:13.3}   {:11.1}   {:13.1}",
            t, magnetization, correlation_length, heat_capacity
        );
    }
    println!();

    pause_for_user()?;

    println!("2. SUPERCONDUCTING TRANSITION: BCS Theory");
    println!("=========================================");
    println!();
    println!("The BCS theory gap equation involves elliptic integrals:");
    println!("1 = (V/2) ∫₀^ωD dε / √(ε² + Δ²) tanh(√(ε² + Δ²)/2k_BT)");
    println!();
    println!("At T = 0: Δ₀ = 2ωD exp(-1/N(0)V)");
    println!("Where N(0) is the density of states at the Fermi level.");
    println!();

    let temperature_sc = Array1::linspace(0.0, 1.2, 13);
    let t_c_sc = 1.0; // normalized T_c

    println!("BCS SUPERCONDUCTOR:");
    println!("T/T_c   Gap Δ/Δ₀   Heat Capacity   Critical Field");
    println!("-----   --------   -------------   --------------");

    for &t in temperature_sc.iter() {
        let gap_ratio = if t < t_c_sc {
            bcs_gap_ratio(t, t_c_sc)
        } else {
            0.0
        };
        let heat_capacity_sc = bcs_heat_capacity(t, t_c_sc);
        let critical_field = bcs_critical_field(t, t_c_sc);

        println!(
            "{:5.2}   {:8.3}   {:13.3}   {:14.3}",
            t, gap_ratio, heat_capacity_sc, critical_field
        );
    }
    println!();

    pause_for_user()?;

    println!("3. QUANTUM PHASE TRANSITIONS: Scaling Functions");
    println!("===============================================");
    println!();
    println!("Quantum phase transitions at T = 0 involve quantum fluctuations.");
    println!("The scaling hypothesis gives universal functions near criticality:");
    println!();
    println!("χ(g,T) = t^(-γ) F_χ(T/t^(zν))");
    println!();
    println!("Where g is the tuning parameter, t = |g - g_c|/g_c, and");
    println!("z, ν, γ are critical exponents.");
    println!();

    let tuning_parameter = Array1::linspace(-0.2, 0.2, 21);
    let _quantum_critical_point = 0.0;

    println!("QUANTUM CRITICAL SCALING:");
    println!("(g-g_c)/g_c   Susceptibility   Correlation   Scaling Function");
    println!("-----------   -------------   -----------   ----------------");

    for &g in tuning_parameter.iter().take(21).step_by(2) {
        let reduced_g = g;
        let susceptibility = quantum_susceptibility(reduced_g);
        let correlation_qcp = quantum_correlation_length(reduced_g);
        let scaling_func = quantum_scaling_function(reduced_g);

        println!(
            "{:11.3}   {:13.2}   {:11.2}   {:16.3}",
            reduced_g, susceptibility, correlation_qcp, scaling_func
        );
    }
    println!();

    pause_for_user()?;

    println!("4. TOPOLOGICAL PHASE TRANSITIONS: Berry Phases");
    println!("==============================================");
    println!();
    println!("Topological phase transitions involve changes in topological invariants.");
    println!("The Berry phase for adiabatic evolution involves gamma functions:");
    println!();
    println!("γ = i ∮ ⟨ψ(R)|∇_R|ψ(R)⟩ · dR");
    println!();
    println!("For spin-1/2 in rotating magnetic field:");
    println!("γ = π(1 - cos θ) = 2π(1 + n̂·ẑ)/2");
    println!();

    let berry_phase_angles = Array1::linspace(0.0, PI, 11);

    println!("BERRY PHASE CALCULATION:");
    println!("θ (degrees)   Berry Phase   Topological Number");
    println!("-----------   -----------   ------------------");

    for &theta in berry_phase_angles.iter() {
        let berry_phase = PI * (1.0 - theta.cos());
        let topological_number = berry_phase / (2.0 * PI);

        println!(
            "{:11.1}   {:11.3}   {:18.3}",
            theta * 180.0 / PI,
            berry_phase,
            topological_number
        );
    }
    println!();

    println!("💡 EXPERIMENTAL REALIZATIONS:");
    println!("• Ising model: quantum gas microscopes with advancedcold atoms");
    println!("• BCS superconductors: cuprates, iron-based, organic superconductors");
    println!("• Quantum criticality: heavy fermion compounds, quantum magnets");
    println!("• Topological phases: quantum Hall systems, topological insulators");
    println!();

    Ok(())
}

#[allow(dead_code)]
fn gravitational_wave_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌊 GRAVITATIONAL WAVE ANALYSIS");
    println!("===============================\n");

    println!("Gravitational wave detection involves sophisticated signal processing");
    println!("where special functions appear in waveform modeling and data analysis.\n");

    pause_for_user()?;

    println!("1. INSPIRAL WAVEFORMS: Hypergeometric Functions");
    println!("===============================================");
    println!();
    println!("The gravitational wave frequency evolution during inspiral:");
    println!("f(t) = f₀ (1 - t/τ)^(-3/8)");
    println!();
    println!("Where τ = 5c⁵/(256πG)^(1/3) M^(-5/3) f₀^(-8/3) is the coalescence time.");
    println!("The phase evolution involves hypergeometric functions.");
    println!();

    let initial_frequency = 35.0; // Hz
    let total_mass = 65.0; // solar masses
    let time_to_merger = chirp_time(total_mass, initial_frequency);

    println!("BINARY INSPIRAL ANALYSIS:");
    println!("Initial frequency f₀ = {} Hz", initial_frequency);
    println!("Total mass M = {} M☉", total_mass);
    println!("Time to merger τ = {:.1} s", time_to_merger);
    println!();

    println!("Time (s)   Frequency (Hz)   Strain h₊");
    println!("--------   --------------   ----------");

    let time_points = Array1::linspace(0.0, time_to_merger * 0.95, 10);
    for &t in time_points.iter() {
        let frequency = inspiral_frequency(t, initial_frequency, time_to_merger);
        let strain = inspiral_strain(t, total_mass, time_to_merger);
        println!("{:8.2}   {:14.1}   {:10.2e}", t, frequency, strain);
    }
    println!();

    pause_for_user()?;

    println!("2. MATCHED FILTERING: Wiener Filtering Theory");
    println!("=============================================");
    println!();
    println!("Optimal detection uses matched filtering with the Wiener filter:");
    println!("Q = 4 Re ∫₀^∞ h̃*(f) s̃(f) / S_n(f) df");
    println!();
    println!("Where h̃(f) is the template, s̃(f) is the data, and S_n(f) is the noise PSD.");
    println!("The false alarm rate involves the gamma function through χ² statistics.");
    println!();

    let snr_threshold = 8.0;
    let false_alarm_rate = 1e-6; // per second

    println!("DETECTION STATISTICS:");
    println!("SNR threshold = {}", snr_threshold);
    println!("Target FAR = {:.0e} Hz⁻¹", false_alarm_rate);
    println!();

    let snr_values = vec![5.0, 8.0, 12.0, 20.0, 35.0];
    println!("SNR    Detection Prob   FAR (Hz⁻¹)   Significance");
    println!("-----  --------------   ----------   ------------");

    for &snr in &snr_values {
        let detection_prob = detection_probability(snr, snr_threshold);
        let far = false_alarm_rate_from_snr(snr);
        let significance = gaussian_significance(snr);
        println!(
            "{:5.1}  {:14.3}   {:10.2e}   {:12.1}σ",
            snr, detection_prob, far, significance
        );
    }
    println!();

    pause_for_user()?;

    println!("3. PARAMETER ESTIMATION: Fisher Information Matrix");
    println!("==================================================");
    println!();
    println!("Parameter uncertainties are given by the Cramér-Rao bound:");
    println!("Δθᵢ ≥ √(Γ⁻¹)ᵢᵢ");
    println!();
    println!("Where Γᵢⱼ = (∂h/∂θᵢ | ∂h/∂θⱼ) is the Fisher information matrix.");
    println!("The inner product involves integrals over the detector bandwidth.");
    println!();

    let gw_events = vec![
        ("GW150914", 36.0, 29.0, 410.0),
        ("GW151226", 14.0, 8.0, 440.0),
        ("GW170817", 1.17, 1.60, 40.0),
    ];

    println!("PARAMETER ESTIMATION FOR LIGO EVENTS:");
    println!("Event      m₁ (M☉)   m₂ (M☉)   Distance (Mpc)   Δm₁/m₁   ΔD/D");
    println!("--------   -------   -------   --------------   -------   ----");

    for &(event, m1, m2, distance) in &gw_events {
        let delta_m1_rel = parameter_uncertainty_mass(m1, m2, distance);
        let delta_d_rel = parameter_uncertainty_distance(m1, m2, distance);
        println!(
            "{:8}   {:7.1}   {:7.1}   {:14.0}     {:7.3}   {:4.2}",
            event, m1, m2, distance, delta_m1_rel, delta_d_rel
        );
    }
    println!();

    pause_for_user()?;

    println!("4. BAYESIAN INFERENCE: Model Selection");
    println!("======================================");
    println!();
    println!("Model comparison uses Bayesian evidence ratios:");
    println!("B₁₂ = Z₁/Z₂ = ∫ L(d|θ,M₁) π(θ|M₁) dθ / ∫ L(d|θ,M₂) π(θ|M₂) dθ");
    println!();
    println!("Evidence integrals often involve gamma functions through");
    println!("marginalization over parameter posteriors.");
    println!();

    let models = vec![
        ("Point particle", 1.0),
        ("Tidal deformation", 0.1),
        ("Eccentric orbit", 0.01),
        ("Precessing spins", 10.0),
    ];

    println!("MODEL COMPARISON (Bayes factors relative to point particle):");
    println!("Model               log B    Odds Ratio   Interpretation");
    println!("------------------  -------  -----------  --------------");

    for &(model, bayes_factor) in &models {
        let bayes_factor: f64 = bayes_factor;
        let log_b = bayes_factor.ln();
        let interpretation = if log_b > 5.0 {
            "Strong evidence"
        } else if log_b > 2.5 {
            "Moderate evidence"
        } else if log_b > 1.0 {
            "Weak evidence"
        } else if log_b > -1.0 {
            "Inconclusive"
        } else {
            "Disfavored"
        };
        println!(
            "{:18}  {:7.2}  {:11.1}  {}",
            model, log_b, bayes_factor, interpretation
        );
    }
    println!();

    println!("💡 LIGO/VIRGO DISCOVERIES:");
    println!("• Direct confirmation of Einstein's general relativity");
    println!("• Binary black hole mergers: most are ~30 M☉, unexpectedly heavy");
    println!("• Neutron star merger GW170817: r-process nucleosynthesis site");
    println!("• Tests of fundamental physics: graviton mass limits, Lorentz violation");
    println!();

    Ok(())
}

#[allow(dead_code)]
fn plasma_physics_fusion() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔥 PLASMA PHYSICS & FUSION");
    println!("==========================\n");

    println!("Plasma physics involves special functions in kinetic theory,");
    println!("magnetohydrodynamics, and fusion reaction calculations.\n");

    pause_for_user()?;

    println!("1. MAXWELL-BOLTZMANN DISTRIBUTION: Gamma Functions");
    println!("==================================================");
    println!();
    println!("The velocity distribution in thermal equilibrium:");
    println!("f(v) = 4π (m/2πkT)^(3/2) v² exp(-mv²/2kT)");
    println!();
    println!("Average values involve gamma function integrals:");
    println!("⟨v⟩ = √(8kT/πm), ⟨v²⟩ = 3kT/m, ⟨v³⟩ = 8√(2kT³/πm³)");
    println!();

    let temperatures_kev = vec![1.0, 10.0, 50.0, 100.0];
    let particle_mass = 3.34e-27; // deuteron mass in kg

    println!("THERMAL PLASMA PROPERTIES (Deuterons):");
    println!("T (keV)   ⟨v⟩ (km/s)   v_th (km/s)   Most Probable v (km/s)");
    println!("-------   ----------   -----------   ----------------------");

    for &temp_kev in &temperatures_kev {
        let temp_k = temp_kev * 1.16e7; // convert keV to K
        let avg_velocity = average_thermal_velocity(temp_k, particle_mass);
        let thermal_velocity = thermal_velocity(temp_k, particle_mass);
        let most_probable_v = most_probable_velocity(temp_k, particle_mass);

        println!(
            "{:7.0}   {:10.0}   {:11.0}   {:22.0}",
            temp_kev,
            avg_velocity / 1000.0,
            thermal_velocity / 1000.0,
            most_probable_v / 1000.0
        );
    }
    println!();

    pause_for_user()?;

    println!("2. FUSION REACTION RATES: Gamow Peak");
    println!("====================================");
    println!();
    println!("The fusion cross-section involves the Gamow factor:");
    println!("σ(E) = S(E)/E × exp(-2πη) where η = Z₁Z₂e²/ħv");
    println!();
    println!("Convolving with Maxwell-Boltzmann gives the Gamow peak:");
    println!("⟨σv⟩ ∝ ∫₀^∞ σ(E) E exp(-E/kT) dE");
    println!();

    let dt_reactions = vec![
        ("D-T", 1.0, 3.0, 1074.0),  // peak cross-section at ~100 keV
        ("D-D", 1.0, 2.0, 96.0),    // peak cross-section at ~1 MeV
        ("D-³He", 1.0, 2.0, 678.0), // peak cross-section at ~200 keV
    ];

    println!("FUSION REACTION RATES:");
    println!("Reaction   T (keV)   ⟨σv⟩ (cm³/s)   Gamow Peak (keV)   Power (MW/m³)");
    println!("--------   -------   ------------   ----------------   -------------");

    for &(reaction, z1, z2, q_value) in &dt_reactions {
        for &temp in &[10.0, 50.0, 100.0] {
            let reaction_rate = fusion_reaction_rate(temp, z1, z2);
            let gamow_peak = gamow_peak_energy(temp, z1, z2);
            let power_density = fusion_power_density(temp, reaction_rate, q_value);

            println!(
                "{:8}   {:7.0}   {:12.2e}   {:16.1}     {:13.2}",
                reaction, temp, reaction_rate, gamow_peak, power_density
            );
        }
    }
    println!();

    pause_for_user()?;

    println!("3. PLASMA OSCILLATIONS: Plasma Dispersion Function");
    println!("==================================================");
    println!();
    println!("The plasma dispersion function Z(ζ) appears in kinetic theory:");
    println!("Z(ζ) = (1/√π) ∫₋∞^∞ exp(-t²)/(t-ζ) dt");
    println!();
    println!("This is related to the Faddeeva function and error functions.");
    println!("Landau damping rates involve Im[Z(ζ)].");
    println!();

    let wave_phase_velocities = Array1::linspace(0.5, 3.0, 11);

    println!("LANDAU DAMPING CALCULATION:");
    println!("v_φ/v_th   Re[Z(ζ)]   Im[Z(ζ)]   Damping Rate");
    println!("---------   --------   --------   ------------");

    for &vph_vth in wave_phase_velocities.iter() {
        let zeta = Complex64::new(vph_vth, 0.0);
        let z_val = plasma_dispersion_function(zeta);
        let damping_rate = -zeta.im * z_val.im;

        println!(
            "{:9.1}   {:8.3}   {:8.3}   {:12.4}",
            vph_vth, z_val.re, z_val.im, damping_rate
        );
    }
    println!();

    pause_for_user()?;

    println!("4. MAGNETIC CONFINEMENT: Adiabatic Invariants");
    println!("=============================================");
    println!();
    println!("Charged particle motion in tokamaks involves three adiabatic invariants:");
    println!("μ = mv_⊥²/2B (magnetic moment)");
    println!("J = ∮ mv_∥ dl (longitudinal invariant)");
    println!("Φ = ∮ ψ dφ (flux invariant)");
    println!();
    println!("Orbit calculations involve elliptic integrals and special functions.");
    println!();

    let magnetic_field_t = vec![1.0, 2.0, 5.0, 10.0];
    let particle_energy_kev = 100.0;

    println!("PARTICLE CONFINEMENT (100 keV deuterons):");
    println!("B (T)   r_L (cm)   f_c (MHz)   Banana Width (cm)");
    println!("-----   --------   ---------   -----------------");

    for &b_field in &magnetic_field_t {
        let larmor_radius = larmor_radius_cm(particle_energy_kev, b_field);
        let cyclotron_freq = cyclotron_frequency_mhz(b_field);
        let banana_width = banana_orbit_width(particle_energy_kev, b_field);

        println!(
            "{:5.0}   {:8.2}   {:9.1}   {:17.1}",
            b_field, larmor_radius, cyclotron_freq, banana_width
        );
    }
    println!();

    println!("💡 FUSION ENERGY PROSPECTS:");
    println!("• ITER: 500 MW fusion power, Q = 10, first plasma ~2035");
    println!("• Stellarators: Alternative to tokamaks, inherently steady-state");
    println!("• Inertial confinement: NIF achieved ignition in December 2022");
    println!("• Private ventures: Commonwealth Fusion, TAE, Helion targeting 2030s");
    println!();

    Ok(())
}

#[allow(dead_code)]
fn atomic_spectroscopy() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📡 ATOMIC & MOLECULAR SPECTROSCOPY");
    println!("===================================\n");

    println!("Spectroscopy reveals atomic structure through special functions");
    println!("in quantum mechanics, selection rules, and line shape analysis.\n");

    pause_for_user()?;

    println!("1. HYDROGEN SPECTRUM: Rydberg Formula & Quantum Defects");
    println!("=======================================================");
    println!();
    println!("The Rydberg formula for hydrogen-like atoms:");
    println!("1/λ = R_∞ Z²eff (1/n₁² - 1/n₂²)");
    println!();
    println!("For alkali atoms, quantum defects δ_ℓ modify the formula:");
    println!("E_n,ℓ = -Ry/(n - δ_ℓ)²");
    println!();

    let hydrogen_transitions = vec![
        ("Lyman α", 1, 2),
        ("Lyman β", 1, 3),
        ("Balmer α", 2, 3),
        ("Balmer β", 2, 4),
        ("Paschen α", 3, 4),
    ];

    println!("HYDROGEN SPECTRUM:");
    println!("Transition   n₁ → n₂   Wavelength (nm)   Energy (eV)   Series");
    println!("-----------  -------   ---------------   -----------   ------");

    for &(name, n1, n2) in &hydrogen_transitions {
        let wavelength_nm = hydrogen_wavelength(n1, n2);
        let energy_ev = 1240.0 / wavelength_nm; // hc/λ in eV
        let series = if n1 == 1 {
            "Lyman"
        } else if n1 == 2 {
            "Balmer"
        } else {
            "Paschen"
        };

        println!(
            "{:11}  {:7}   {:15.1}   {:11.3}   {}",
            name,
            format!("{} → {}", n1, n2),
            wavelength_nm,
            energy_ev,
            series
        );
    }
    println!();

    pause_for_user()?;

    println!("2. FINE STRUCTURE: Dirac Equation & j-j Coupling");
    println!("=================================================");
    println!();
    println!("Fine structure from spin-orbit coupling:");
    println!("ΔE_fs = α² Ry Z⁴/n³ [1/j+1/2 - 3/4n] for j = ℓ ± 1/2");
    println!();
    println!("The fine structure constant α ≈ 1/137 determines the splitting magnitude.");
    println!();

    let sodium_d_lines = vec![
        ("D₁", 3, 0, 0.5, 1.5), // 3s₁/₂ → 3p₁/₂
        ("D₂", 3, 0, 0.5, 0.5), // 3s₁/₂ → 3p₃/₂
    ];

    println!("SODIUM D-LINE FINE STRUCTURE:");
    println!("Line   Transition     Wavelength (nm)   Splitting (meV)");
    println!("----   ----------     ---------------   ---------------");

    for &(line, n, l, j_lower, j_upper) in &sodium_d_lines {
        let wavelength = sodium_d_line_wavelength(line);
        let splitting = fine_structure_splitting(n, l);

        println!(
            "{:4}   {}s₁/₂ → {}p₃/₂   {:15.1}   {:15.1}",
            line, n, n, wavelength, splitting
        );
    }
    println!();

    pause_for_user()?;

    println!("3. HYPERFINE STRUCTURE: Nuclear Magnetic Moments");
    println!("================================================");
    println!();
    println!("Hyperfine splitting from nuclear magnetic moment interaction:");
    println!("ΔE_hfs = (2π/3) α² Ry (m_e/m_p) Z³ μ_N g_I |ψ(0)|²");
    println!();
    println!("For hydrogen 21-cm line: ΔE = 5.9 μeV, λ = 21.1 cm");
    println!();

    let hyperfine_transitions = vec![
        ("¹H", 1, 0.5, 1420.4, 5.9),    // 21-cm line
        ("²H", 1, 1.0, 327.4, 1.4),     // deuterium
        ("³He⁺", 1, 0.5, 8665.6, 36.0), // helium-3 ion
    ];

    println!("HYPERFINE TRANSITIONS:");
    println!("Isotope   n   I   Frequency (MHz)   Energy (μeV)   λ (cm)");
    println!("-------   -   -   ---------------   ------------   ------");

    for &(isotope, n, nuclear_spin, freq_mhz, energy_uev) in &hyperfine_transitions {
        let wavelength_cm = 29979.2458 / freq_mhz; // c/f in cm

        println!(
            "{:7}   {}   {}   {:15.1}   {:12.1}   {:6.1}",
            isotope, n, nuclear_spin, freq_mhz, energy_uev, wavelength_cm
        );
    }
    println!();

    pause_for_user()?;

    println!("4. LINE BROADENING: Voigt Profiles");
    println!("==================================");
    println!();
    println!("Spectral line shapes involve convolution of Gaussian and Lorentzian:");
    println!("Voigt profile = Gaussian ⊗ Lorentzian");
    println!();
    println!("V(ν) = ∫₋∞^∞ G(ν') L(ν-ν') dν' = Re[w(z)]");
    println!("where w(z) is the Faddeeva function.");
    println!();

    let line_broadening_mechanisms = vec![
        ("Doppler", 300.0_f64, 0.1_f64, 10.0_f64), // thermal motion at 300K
        ("Natural", 0.0_f64, 1.0_f64, 16.0_f64),   // spontaneous emission
        ("Pressure", 1000.0_f64, 5.0_f64, 3.2_f64), // collisional at 1 atm
        ("Stark", 0.0_f64, 10.0_f64, 1.6_f64),     // electric field
    ];

    println!("LINE BROADENING ANALYSIS (λ = 589 nm):");
    println!("Mechanism   T (K)   FWHM (pm)   Lifetime (ns)   Dominant Range");
    println!("---------   -----   ---------   -------------   --------------");

    for &(mechanism, temp, fwhm_pm, lifetime_ns) in &line_broadening_mechanisms {
        let dominant_regime = match mechanism {
            "Doppler" => "Low pressure",
            "Natural" => "All conditions",
            "Pressure" => "High pressure",
            "Stark" => "Strong E-field",
            _ => "Special conditions",
        };

        println!(
            "{:9}   {:5.0}   {:9.1}   {:13.1}   {}",
            mechanism, temp, fwhm_pm, lifetime_ns, dominant_regime
        );
    }
    println!();

    pause_for_user()?;

    println!("5. MOLECULAR SPECTROSCOPY: Rotational-Vibrational Structure");
    println!("===========================================================");
    println!();
    println!("Diatomic molecule energy levels:");
    println!("E(v,J) = ωₑ(v + 1/2) + BₑJ(J+1) - αₑ(v + 1/2)J(J+1)");
    println!();
    println!("Selection rules: Δv = ±1, ΔJ = ±1 for fundamental transitions");
    println!();

    let co_molecule_constants = (2170.2, 1.931, 0.0175); // ωₑ, Bₑ, αₑ in cm⁻¹

    println!("CO MOLECULE ROVIBRATIONAL SPECTRUM:");
    println!("v   J   Energy (cm⁻¹)   P-branch (cm⁻¹)   R-branch (cm⁻¹)");
    println!("-   -   -------------   ----------------   ----------------");

    for v in 0..3 {
        for j in 0..6 {
            let energy = vibrational_rotational_energy(v, j, co_molecule_constants);
            let p_branch = if j > 0 {
                Some(co_fundamental_p_branch(j))
            } else {
                None
            };
            let r_branch = co_fundamental_r_branch(j);

            match p_branch {
                Some(p) => println!(
                    "{}   {}   {:13.1}   {:16.1}   {:16.1}",
                    v, j, energy, p, r_branch
                ),
                None => println!(
                    "{}   {}   {:13.1}   {:16}   {:16.1}",
                    v, j, energy, "--", r_branch
                ),
            }
        }
    }
    println!();

    println!("💡 SPECTROSCOPIC APPLICATIONS:");
    println!("• Atomic clocks: Cs 133 hyperfine transition defines the second");
    println!("• Laser cooling: Doppler and sub-Doppler cooling mechanisms");
    println!("• Astrophysics: Element abundances, stellar velocities, exoplanet detection");
    println!("• Quantum metrology: Frequency standards, fundamental constant measurements");
    println!();

    Ok(())
}

#[allow(dead_code)]
fn statistical_mechanics_mesoscopic() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎲 MESOSCOPIC STATISTICAL MECHANICS");
    println!("====================================\n");

    println!("Mesoscopic systems bridge quantum and classical regimes,");
    println!("involving special functions in fluctuation phenomena.\n");

    pause_for_user()?;

    println!("1. QUANTUM DOTS: Single-Electron Statistics");
    println!("===========================================");
    println!();
    println!("Coulomb blockade in quantum dots leads to discrete charging:");
    println!("E(N) = N²e²/2C + N(μ - e²/2C)");
    println!();
    println!("The addition energy involves special functions through");
    println!("electrostatic interactions and exchange effects.");
    println!();

    let quantum_dot_params = vec![
        ("Small", 1e-18, 50), // Capacitance (F), Level spacing (μeV)
        ("Medium", 1e-17, 10),
        ("Large", 1e-16, 2),
    ];

    println!("QUANTUM DOT CHARGING ENERGY:");
    println!("Size     C (F)     Δ (μeV)   E_c (meV)   Nmax(300K)");
    println!("------   -------   -------   ---------   -----------");

    for &(size, capacitance, level_spacing) in &quantum_dot_params {
        let charging_energy = charging_energy_mev(capacitance);
        let max_electrons = max_electrons_thermal(charging_energy, 300.0);

        println!(
            "{:6}   {:7.0e}   {:7}   {:9.2}   {:11.0}",
            size, capacitance, level_spacing, charging_energy, max_electrons
        );
    }
    println!();

    pause_for_user()?;

    println!("2. RANDOM MATRIX THEORY: Wigner-Dyson Statistics");
    println!("================================================");
    println!();
    println!("Level spacing distribution in chaotic quantum systems:");
    println!("P(s) = (π/2) s exp(-πs²/4)  (Gaussian Orthogonal Ensemble)");
    println!("P(s) = (32/π²) s² exp(-4s²/π)  (Gaussian Unitary Ensemble)");
    println!();

    let spacing_values = Array1::linspace(0.0, 3.0, 16);

    println!("LEVEL SPACING STATISTICS:");
    println!("s     P_GOE(s)   P_GUE(s)   P_Poisson(s)");
    println!("---   --------   --------   ------------");

    for &s in spacing_values.iter().take(16).step_by(2) {
        let p_goe = wigner_surmise_goe(s);
        let p_gue = wigner_surmise_gue(s);
        let p_poisson = (-s).exp(); // Poisson statistics

        println!(
            "{:.1}   {:8.3}   {:8.3}   {:12.3}",
            s, p_goe, p_gue, p_poisson
        );
    }
    println!();

    pause_for_user()?;

    println!("3. SHOT NOISE: Full Counting Statistics");
    println!("=======================================");
    println!();
    println!("Current fluctuations in mesoscopic conductors involve");
    println!("cumulant generating functions and special functions:");
    println!();
    println!("⟨⟨I^n⟩⟩ = ∂ⁿ ln⟨e^λI⟩/∂λⁿ|_{{λ=0}}");
    println!();
    println!("For tunnel junctions: ⟨⟨I²⟩⟩ = eI (shot noise)");
    println!();

    let voltage_bias = Array1::linspace(0.0, 5.0, 11);
    let tunnel_resistance = 1e6; // ohms

    println!("SHOT NOISE IN TUNNEL JUNCTION:");
    println!("V (mV)   I (nA)   ⟨I²⟩ (pA²/Hz)   Fano Factor");
    println!("------   ------   -------------   -----------");

    for &voltage in voltage_bias.iter() {
        let current = voltage / tunnel_resistance * 1e9; // nA
        let shot_noise = shot_noise_tunnel_junction(current * 1e-9); // A²/Hz
        let fano_factor = if current > 0.0 {
            shot_noise / (2.0 * 1.6e-19 * current * 1e-9)
        } else {
            0.0
        };

        println!(
            "{:6.1}   {:6.1}   {:13.1}   {:11.3}",
            voltage * 1000.0,
            current,
            shot_noise * 1e24,
            fano_factor
        );
    }
    println!();

    pause_for_user()?;

    println!("4. BROWNIAN MOTION: Fluctuation-Dissipation Theorem");
    println!("====================================================");
    println!();
    println!("Einstein relation connects diffusion and mobility:");
    println!("D = μkT where μ = 1/(6πηr) for spherical particles");
    println!();
    println!("The power spectral density involves Lorentzian profiles:");
    println!("S(ω) = 4kTγ/(γ² + ω²) where γ = 6πηr/m");
    println!();

    let particlesizes = vec![1e-9, 10e-9, 100e-9, 1e-6]; // meters
    let temperature = 300.0; // K
    let viscosity = 1e-3; // Pa·s (water)

    println!("BROWNIAN MOTION PARAMETERS (T = 300 K, water):");
    println!("r (nm)   D (m²/s)   τ_c (ns)   ⟨x²⟩ (nm²/s)");
    println!("------   --------   --------   -------------");

    for &radius in &particlesizes {
        let diffusion_coeff = diffusion_coefficient(radius, temperature, viscosity);
        let correlation_time = momentum_correlation_time(radius, viscosity);
        let mean_square_displacement = 2.0 * diffusion_coeff * 1e18; // nm²/s

        println!(
            "{:6.0}   {:8.2e}   {:8.1}   {:13.1}",
            radius * 1e9,
            diffusion_coeff,
            correlation_time * 1e9,
            mean_square_displacement
        );
    }
    println!();

    pause_for_user()?;

    println!("5. QUANTUM PHASE TRANSITIONS: Finite-Size Scaling");
    println!("=================================================");
    println!();
    println!("Near quantum critical points, finite-size scaling gives:");
    println!("χ(L,t) = L^(γ/ν) f_χ(tL^(1/ν))");
    println!();
    println!("Where L is system size, t = (g-g_c)/g_c, and f_χ is a universal function.");
    println!();

    let systemsizes = vec![4, 8, 16, 32, 64];
    let critical_exponents = (1.0, 0.75, 2.0); // ν, γ, β

    println!("FINITE-SIZE SCALING ANALYSIS:");
    println!("L    Correlation ξ/L   Susceptibility χL^(-γ/ν)   Order Parameter");
    println!("--   --------------   ----------------------   ----------------");

    for &size in &systemsizes {
        let correlation_ratio = finitesize_correlation(size as f64, critical_exponents.0);
        let susceptibility_scaled = finitesize_susceptibility(size as f64, critical_exponents);
        let order_parameter = finitesize_order_parameter(size as f64, critical_exponents.2);

        println!(
            "{:2}   {:14.3}   {:22.3}   {:16.3}",
            size, correlation_ratio, susceptibility_scaled, order_parameter
        );
    }
    println!();

    println!("💡 MESOSCOPIC PHENOMENA:");
    println!("• Quantum dots: artificial atoms for quantum computation");
    println!("• Random matrix theory: universal correlations in complex systems");
    println!("• Shot noise: fundamental limit in electronic measurements");
    println!("• Brownian motion: foundation of stochastic thermodynamics");
    println!("• Finite-size scaling: bridge between microscopic and macroscopic physics");
    println!();

    Ok(())
}

#[allow(dead_code)]
fn nonlinear_optics_solitons() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💫 NONLINEAR OPTICS & SOLITONS");
    println!("===============================\n");

    println!("Nonlinear optics involves special functions in wave propagation,");
    println!("soliton solutions, and nonlinear Schrödinger equations.\n");

    pause_for_user()?;

    println!("1. OPTICAL SOLITONS: Nonlinear Schrödinger Equation");
    println!("===================================================");
    println!();
    println!("The nonlinear Schrödinger equation:");
    println!("i∂u/∂z + (1/2)∂²u/∂t² + γ|u|²u = 0");
    println!();
    println!("Fundamental soliton solution:");
    println!("u(z,t) = η sech(ηt) exp(iη²z/2)");
    println!();

    let soliton_parameters = vec![
        ("Low power", 1.0, 1.0), // η, peak power
        ("Medium power", 2.0, 4.0),
        ("High power", 3.0, 9.0),
    ];

    println!("OPTICAL SOLITON PROPERTIES:");
    println!("Regime        η    P₀    FWHM (ps)   Period (km)");
    println!("-----------   --   ----   ---------   -----------");

    for &(regime, eta, peak_power) in &soliton_parameters {
        let fwhm_ps = 2.0 * (2.0_f64.ln()).sqrt() / eta; // pulse width
        let period_km = 2.0 * PI / (eta * eta); // soliton period

        println!(
            "{:11}   {:2.0}   {:4.0}   {:9.1}   {:11.1}",
            regime, eta, peak_power, fwhm_ps, period_km
        );
    }
    println!();

    pause_for_user()?;

    println!("2. ELLIPTIC FUNCTION SOLUTIONS: Jacobi cn and dn");
    println!("===============================================");
    println!();
    println!("Periodic solutions involve Jacobi elliptic functions:");
    println!("u(z,t) = A cn(Bt + φ, m) exp(iCz)");
    println!();
    println!("Where cn is the Jacobi cosine amplitude function and m is the modulus.");
    println!("These interpolate between sinusoidal (m → 0) and soliton (m → 1) limits.");
    println!();

    let modulus_values = vec![0.0, 0.25, 0.5, 0.75, 0.9, 0.99];

    println!("ELLIPTIC FUNCTION SOLUTIONS:");
    println!("Modulus m   cn(u,m) at u=0   Period K(m)   Soliton Limit");
    println!("---------   --------------   -----------   -------------");

    for &m in &modulus_values {
        let cn_zero = jacobi_cn(0.0, m);
        let period = complete_elliptic_k(m);
        let limit_type = if m < 0.1 {
            "Sinusoidal"
        } else if m > 0.9 {
            "Solitonic"
        } else {
            "Intermediate"
        };

        println!(
            "{:9.2}   {:14.3}   {:11.3}   {}",
            m, cn_zero, period, limit_type
        );
    }
    println!();

    pause_for_user()?;

    println!("3. SUPERCONTINUUM GENERATION: Modulation Instability");
    println!("====================================================");
    println!();
    println!("Modulation instability leads to spectral broadening:");
    println!("Gain: g(Ω) = 2γP₀√(1 - (Ω/Ω_c)²) for |Ω| < Ω_c");
    println!();
    println!("Where Ω_c = 2√(γP₀/|β₂|) is the cutoff frequency.");
    println!();

    let input_powers = vec![0.1, 0.5, 1.0, 2.0, 5.0]; // W
    let gamma: f64 = 0.001; // W^-1 m^-1
    let beta2: f64 = -20e-27; // s^2/m

    println!("MODULATION INSTABILITY:");
    println!("P₀ (W)   Ω_c (THz)   Max Gain (m⁻¹)   Bandwidth (nm)");
    println!("------   ----------   ---------------   --------------");

    for &power in &input_powers {
        let cutoff_freq = 2.0 * (gamma * power / beta2.abs()).sqrt() / (2.0 * PI); // THz
        let max_gain = gamma * power;
        let bandwidth_nm = cutoff_freq * 1550.0 * 1550.0 / 299.8; // approximate

        println!(
            "{:6.1}   {:10.1}   {:15.3}   {:14.1}",
            power,
            cutoff_freq / 1e12,
            max_gain,
            bandwidth_nm
        );
    }
    println!();

    pause_for_user()?;

    println!("4. ROGUE WAVES: Peregrine Solitons");
    println!("==================================");
    println!();
    println!("The Peregrine soliton describes rational rogue wave solutions:");
    println!("u(z,t) = [1 - 4(1+2iz)/(1+4z²+4t²)] exp(iz)");
    println!();
    println!("These are localized in both space and time, unlike periodic cnoidal waves.");
    println!();

    let time_points = Array1::linspace(-3.0, 3.0, 13);
    let propagation_distance = 0.0; // z = 0

    println!("PEREGRINE SOLITON PROFILE (z = 0):");
    println!("t     |u(0,t)|²   Phase (rad)   Amplification");
    println!("---   ---------   -----------   -------------");

    for &t in time_points.iter() {
        let amplitude_squared = peregrine_amplitude_squared(propagation_distance, t);
        let phase = peregrine_phase(propagation_distance, t);
        let amplification = amplitude_squared; // relative to background

        println!(
            "{:3.0}   {:9.2}   {:11.2}   {:13.2}",
            t, amplitude_squared, phase, amplification
        );
    }
    println!();

    pause_for_user()?;

    println!("5. OPTICAL FREQUENCY COMBS: Kerr Microresonators");
    println!("===============================================");
    println!();
    println!("Kerr comb formation involves Mathieu functions and modulation instability.");
    println!("The driven-damped nonlinear Schrödinger equation:");
    println!("∂A/∂t = -iδA - A/2τ - iγ|A|²A + √(P/τ)");
    println!();

    let detuning_values = vec![-2.0, -1.0, 0.0, 1.0, 2.0]; // normalized detuning
    let pump_power = 1.0; // normalized

    println!("KERR COMB FORMATION:");
    println!("Detuning   Threshold   Comb Lines   Coherence");
    println!("--------   ---------   ----------   ---------");

    for &detuning in &detuning_values {
        let threshold = kerr_comb_threshold(detuning);
        let num_lines = estimate_comb_lines(detuning, pump_power);
        let coherence = if detuning < 0.0 { "High" } else { "Low" };

        println!(
            "{:8.1}   {:9.2}   {:10.0}   {}",
            detuning, threshold, num_lines, coherence
        );
    }
    println!();

    println!("💡 APPLICATIONS & DISCOVERIES:");
    println!("• Optical solitons: optimized telecommunications, pulse compression");
    println!("• Supercontinuum: spectroscopy, optical coherence tomography");
    println!("• Rogue waves: understanding extreme events in optics and oceanography");
    println!("• Frequency combs: precision metrology, Nobel Prize 2005 (Hänsch, Hall)");
    println!("• Kerr combs: chip-scale frequency combs for portable atomic clocks");
    println!();

    Ok(())
}

#[allow(dead_code)]
fn cosmic_ray_physics() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 COSMIC RAY PHYSICS");
    println!("======================\n");

    println!("Cosmic ray physics involves special functions in energy spectra,");
    println!("atmospheric interactions, and detection statistics.\n");

    pause_for_user()?;

    println!("1. COSMIC RAY ENERGY SPECTRUM: Power Laws");
    println!("=========================================");
    println!();
    println!("The cosmic ray spectrum follows broken power laws:");
    println!("dN/dE ∝ E^(-γ) where γ changes at characteristic energies");
    println!();
    println!("• Below knee (~3×10¹⁵ eV): γ ≈ 2.7");
    println!("• Above knee: γ ≈ 3.0");
    println!("• Above ankle (~3×10¹⁸ eV): γ ≈ 2.6");
    println!("• GZK cutoff: ~6×10¹⁹ eV");
    println!();

    let energy_ranges = vec![
        ("Low energy", 1e12, 1e15, 2.7),
        ("Knee region", 1e15, 1e17, 3.0),
        ("Ankle region", 1e17, 1e19, 2.6),
        ("Advanced-high", 1e19, 1e21, 4.0),
    ];

    println!("COSMIC RAY FLUX:");
    println!("Region        Emin (eV)   Emax (eV)   Index γ   Flux at 10¹⁵ eV");
    println!("------------  -----------  -----------  -------   ----------------");

    for &(region, emin, emax, gamma) in &energy_ranges {
        let flux_reference = cosmic_ray_flux(1e15, gamma);

        println!(
            "{:12}  {:11.0e}  {:11.0e}  {:7.1}   {:16.2e}",
            region, emin, emax, gamma, flux_reference
        );
    }
    println!();

    pause_for_user()?;

    println!("2. ATMOSPHERIC SHOWERS: Heitler Model");
    println!("=====================================");
    println!();
    println!("Electromagnetic cascades follow the Heitler model:");
    println!("N(t) = 2^t for t < tmax");
    println!("N(t) = 2^tmax exp(-(t-tmax)/λ) for t > tmax");
    println!();
    println!("Where t = depth/X₀ and tmax ≈ ln(E₀/E_c)/ln(2)");
    println!();

    let primary_energies = vec![1e12, 1e14, 1e16, 1e18]; // eV
    let critical_energy: f64 = 81e6; // eV (for air)
    let radiation_length: f64 = 37.15; // g/cm² for air

    println!("ELECTROMAGNETIC SHOWER DEVELOPMENT:");
    println!("E₀ (eV)     tmax   Nmax       Xmax (g/cm²)   Depth (km)");
    println!("---------   -----   ---------   -------------   ----------");

    for &energy in &primary_energies {
        let tmax = (energy / critical_energy).ln() / 2.0_f64.ln();
        let nmax = 2.0_f64.powf(tmax);
        let xmax = tmax * radiation_length;
        let depth_km = xmax / 1030.0 * 10.0; // approximate conversion to km

        println!(
            "{:9.0e}   {:5.1}   {:9.1e}   {:13.1}   {:10.2}",
            energy, tmax, nmax, xmax, depth_km
        );
    }
    println!();

    pause_for_user()?;

    println!("3. MUON PRODUCTION: Pion Decay Chain");
    println!("====================================");
    println!();
    println!("Hadronic showers produce muons through pion decay:");
    println!("π⁺ → μ⁺ + ν_μ (τ = 26 ns)");
    println!("μ⁺ → e⁺ + ν_e + ν̄_μ (τ = 2.2 μs)");
    println!();
    println!("Muon flux at sea level depends on zenith angle θ:");
    println!("dN/dE ∝ E^(-2.7) sec(θ) for low energies");
    println!();

    let zenith_angles = vec![0.0, 30.0, 45.0, 60.0, 75.0]; // degrees
    let muon_energy = 1e9; // eV

    println!("MUON FLUX vs ZENITH ANGLE:");
    println!("θ (°)   sec(θ)   Relative Flux   Path Length (km)");
    println!("-----   ------   -------------   ----------------");

    for &angle in &zenith_angles {
        let theta_rad = angle * PI / 180.0;
        let sec_theta = 1.0 / theta_rad.cos();
        let relative_flux = sec_theta;
        let path_length = 15.0 * sec_theta; // approximate atmosphere thickness

        println!(
            "{:5.0}   {:6.2}   {:13.2}   {:16.1}",
            angle, sec_theta, relative_flux, path_length
        );
    }
    println!();

    pause_for_user()?;

    println!("4. DETECTOR STATISTICS: Poisson Fluctuations");
    println!("============================================");
    println!();
    println!("Cosmic ray detection involves Poisson statistics:");
    println!("P(n;λ) = λⁿe^(-λ)/n! where λ = rate × time");
    println!();
    println!("For rare events, waiting time distribution is exponential:");
    println!("P(t) = λe^(-λt)");
    println!();

    let detector_rates = vec![
        ("Muon counter", 1.0, 3600.0),         // 1 Hz, 1 hour
        ("Neutron monitor", 100.0, 60.0),      // 100 Hz, 1 minute
        ("Air shower", 0.01, 86400.0),         // 0.01 Hz, 1 day
        ("Advanced-high E", 1e-6, 31536000.0), // 1 per year
    ];

    println!("DETECTION STATISTICS:");
    println!("Detector        Rate (Hz)   Time (s)   ⟨N⟩   P(0)     P(≥1)");
    println!("-------------   ---------   --------   ----   ------   ------");

    for &(detector, rate, time_period) in &detector_rates {
        let expected_counts: f64 = rate * time_period;
        let prob_zero = (-expected_counts).exp();
        let prob_one_or_more = 1.0 - prob_zero;

        println!(
            "{:13}   {:9.2e}   {:8.0e}   {:4.1}   {:6.3}   {:6.3}",
            detector, rate, time_period, expected_counts, prob_zero, prob_one_or_more
        );
    }
    println!();

    pause_for_user()?;

    println!("5. GZK CUTOFF: Photodisintegration Process");
    println!("==========================================");
    println!();
    println!("Advanced-high energy protons interact with CMB photons:");
    println!("p + γ_CMB → Δ⁺ → p + π⁰ or n + π⁺");
    println!();
    println!("The threshold energy is E_th ≈ 6×10¹⁹ eV for head-on collision.");
    println!("Energy loss length: λ(E) ≈ 50 Mpc for E > E_th");
    println!();

    let distances_mpc = vec![10.0, 50.0, 100.0, 200.0, 500.0];
    let gzk_energy = 6e19; // eV

    println!("GZK ENERGY LOSS:");
    println!("Distance (Mpc)   Survival Prob   Final Energy (eV)   Attenuation");
    println!("--------------   -------------   -----------------   -----------");

    for &distance in &distances_mpc {
        let survival_prob = (-distance / 50.0_f64).exp(); // λ ≈ 50 Mpc
        let final_energy = gzk_energy * survival_prob;
        let attenuation_db = -10.0 * (survival_prob).log10();

        println!(
            "{:14.0}   {:13.3}   {:17.1e}   {:11.1} dB",
            distance, survival_prob, final_energy, attenuation_db
        );
    }
    println!();

    println!("💡 COSMIC RAY DISCOVERIES:");
    println!("• Victor Hess (1912): Discovery of cosmic radiation, Nobel Prize 1936");
    println!("• Positron discovery (1932): First antimatter particle by Carl Anderson");
    println!("• Muon discovery (1936): \"Who ordered that?\" - I.I. Rabi");
    println!("• Pierre Auger Observatory: Advanced-high energy cosmic ray studies");
    println!("• IceCube: Neutrino astronomy with cubic kilometer detector");
    println!();

    Ok(())
}

#[allow(dead_code)]
fn quantum_information_experiments() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔒 QUANTUM INFORMATION EXPERIMENTS");
    println!("===================================\n");

    println!("Quantum information science involves special functions in");
    println!("entanglement measures, quantum error correction, and fidelity metrics.\n");

    pause_for_user()?;

    println!("1. ENTANGLEMENT MEASURES: Von Neumann Entropy");
    println!("=============================================");
    println!();
    println!("For bipartite quantum states, entanglement entropy:");
    println!("S(ρ_A) = -Tr(ρ_A ln ρ_A) where ρ_A = Tr_B(|ψ⟩⟨ψ|)");
    println!();
    println!("For Bell states and Werner states, this involves logarithmic functions.");
    println!();

    let werner_state_params = vec![0.0, 0.25, 0.5, 0.75, 1.0]; // mixing parameter p

    println!("WERNER STATE ENTANGLEMENT:");
    println!("p     S(ρ_A) (bits)   Concurrence   PPT Criterion");
    println!("----  -------------   -----------   -------------");

    for &p in &werner_state_params {
        let entropy = werner_state_entropy(p);
        let concurrence = werner_state_concurrence(p);
        let ppt_separable = if p < 0.5 { "Separable" } else { "Entangled" };

        println!(
            "{:.2}  {:13.3}   {:11.3}   {}",
            p, entropy, concurrence, ppt_separable
        );
    }
    println!();

    pause_for_user()?;

    println!("2. QUANTUM FIDELITY: Uhlmann's Theorem");
    println!("======================================");
    println!();
    println!("Fidelity between quantum states ρ and σ:");
    println!("F(ρ,σ) = Tr(√(√ρ σ √ρ))");
    println!();
    println!("For pure states: F(|ψ⟩,|φ⟩) = |⟨ψ|φ⟩|²");
    println!("Bures distance: d_B(ρ,σ) = √(2(1-F(ρ,σ)))");
    println!();

    let overlap_angles = Array1::linspace(0.0, PI / 2.0, 11);

    println!("QUANTUM STATE FIDELITY:");
    println!("θ (°)   |⟨ψ|φ⟩|²   Fidelity   Bures Distance   Classical Analog");
    println!("-----   --------   --------   --------------   ----------------");

    for &theta in overlap_angles.iter() {
        let overlap_squared = theta.cos().powi(2);
        let fidelity = overlap_squared;
        let bures_distance = (2.0 * (1.0 - fidelity)).sqrt();
        let classical = if theta < PI / 4.0 {
            "Distinguishable"
        } else {
            "Overlapping"
        };

        println!(
            "{:5.0}   {:8.3}   {:8.3}   {:14.3}   {}",
            theta * 180.0 / PI,
            overlap_squared,
            fidelity,
            bures_distance,
            classical
        );
    }
    println!();

    pause_for_user()?;

    println!("3. QUANTUM ERROR CORRECTION: Threshold Theorem");
    println!("==============================================");
    println!();
    println!("Error threshold for quantum computation depends on error rates:");
    println!("p_th ≈ 10⁻⁴ for surface codes (most promising)");
    println!("p_th ≈ 10⁻⁶ for concatenated codes");
    println!();
    println!("Logical error rate: p_L ≈ (p/p_th)^((d+1)/2) for distance d codes");
    println!();

    let physical_error_rates = vec![1e-2, 1e-3, 1e-4, 1e-5, 1e-6];
    let threshold: f64 = 1e-4;
    let code_distances = vec![3, 5, 7, 9];

    println!("QUANTUM ERROR CORRECTION:");
    println!("p_phys    d=3      d=5      d=7      d=9      Status");
    println!("------   -------  -------  -------  -------  ----------");

    for &p_phys in &physical_error_rates {
        print!("{:6.0e}", p_phys);
        for &d in &code_distances {
            let logical_error = if p_phys < threshold {
                (p_phys / threshold).powf((d + 1) as f64 / 2.0)
            } else {
                1.0 // Above threshold
            };
            print!("   {:7.0e}", logical_error);
        }
        let status = if p_phys < threshold {
            "Below threshold"
        } else {
            "Above threshold"
        };
        println!("  {}", status);
    }
    println!();

    pause_for_user()?;

    println!("4. QUANTUM TELEPORTATION: Fidelity Analysis");
    println!("===========================================");
    println!();
    println!("Teleportation fidelity with noisy entanglement:");
    println!("F = (2F_ent + 1)/3 where F_ent is entanglement fidelity");
    println!();
    println!("For Werner states: F_ent = (1+2p)/3 where p is mixture parameter");
    println!();

    let entanglement_fidelities = Array1::linspace(0.5, 1.0, 11);

    println!("QUANTUM TELEPORTATION FIDELITY:");
    println!("F_ent   F_teleportation   Classical Limit   Advantage");
    println!("-----   ---------------   ---------------   ---------");

    for &f_ent in entanglement_fidelities.iter() {
        let f_teleportation = (2.0 * f_ent + 1.0) / 3.0;
        let classical_limit = 2.0 / 3.0; // Random guessing
        let advantage = f_teleportation > classical_limit;

        println!(
            "{:.2}   {:15.3}   {:15.3}   {}",
            f_ent,
            f_teleportation,
            classical_limit,
            if advantage { "Quantum" } else { "Classical" }
        );
    }
    println!();

    pause_for_user()?;

    println!("5. QUANTUM ALGORITHMS: Computational Complexity");
    println!("===============================================");
    println!();
    println!("Quantum speedup for various problems:");
    println!("• Factoring (Shor): exponential speedup");
    println!("• Search (Grover): quadratic speedup");
    println!("• Simulation: exponential for many-body quantum systems");
    println!();

    let problemsizes = vec![100, 1000, 10000, 100000];

    println!("QUANTUM vs CLASSICAL COMPLEXITY:");
    println!("Problem Size   Classical (factoring)   Quantum (Shor)   Speedup");
    println!("------------   --------------------   ---------------   -------");

    for &n in &problemsizes {
        let classical_time = shor_classical_complexity(n as f64);
        let quantum_time = shor_quantum_complexity(n as f64);
        let speedup = classical_time / quantum_time;

        println!(
            "{:12}   {:20.1e}   {:15.1e}   {:7.0e}",
            n, classical_time, quantum_time, speedup
        );
    }
    println!();

    println!("GROVER SEARCH ALGORITHM:");
    println!("Database Size   Classical   Quantum (Grover)   Speedup");
    println!("-------------   ---------   ----------------   -------");

    for &n in &problemsizes {
        let classical_search = n as f64 / 2.0; // average case
        let quantum_search = (n as f64).sqrt() * PI / 4.0;
        let grover_speedup = classical_search / quantum_search;

        println!(
            "{:13}   {:9.1e}   {:16.1e}   {:7.1}",
            n, classical_search, quantum_search, grover_speedup
        );
    }
    println!();

    println!("💡 QUANTUM INFORMATION MILESTONES:");
    println!("• 1982: Feynman proposes quantum computers");
    println!("• 1994: Shor's factoring algorithm threatens RSA cryptography");
    println!("• 1997: First quantum teleportation experiments");
    println!("• 2019: Google claims quantum supremacy with 53-qubit Sycamore");
    println!("• 2021: IBM unveils 127-qubit Eagle processor");
    println!("• Current: Race to fault-tolerant quantum computing");
    println!();

    Ok(())
}

// Helper functions for pause and calculations
#[allow(dead_code)]
fn pause_for_user() -> Result<(), Box<dyn std::error::Error>> {
    print!("Press Enter to continue...");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(())
}

// Physics calculation helper functions
#[allow(dead_code)]
fn poisson_probability(n: usize, mean: f64) -> f64 {
    let n_factorial = (1..=n).map(|i| i as f64).product::<f64>();
    mean.powi(n as i32) * (-mean).exp() / n_factorial
}

#[allow(dead_code)]
fn negative_binomial_probability(n: usize, modes: f64, mean: f64) -> f64 {
    let p = modes / (modes + mean);
    let gamma_ratio = gamma(n as f64 + modes) / (gamma(n as f64 + 1.0) * gamma(modes));
    gamma_ratio * p.powf(modes) * (1.0 - p).powi(n as i32)
}

#[allow(dead_code)]
fn neutron_wave_number(energy_mev: f64) -> f64 {
    // k = √(2mE)/ħ in fm^-1
    let mass_mev = 939.565; // neutron mass in MeV
    let hbar_c = 197.327; // ħc in MeV·fm
    (2.0 * mass_mev * energy_mev).sqrt() / hbar_c
}

#[allow(dead_code)]
fn calculate_hard_sphere_phase_shift(kr: f64) -> f64 {
    // Simple hard sphere approximation
    -kr * 180.0 / PI
}

#[allow(dead_code)]
fn calculate_barrier_penetration(excitation: f64, barrier_height: f64) -> f64 {
    if excitation >= barrier_height {
        1.0
    } else {
        let argument: f64 = -2.0 * (barrier_height - excitation).sqrt();
        argument.exp()
    }
}

#[allow(dead_code)]
fn nuclear_level_density(energy: f64, a: f64) -> f64 {
    let sqrt_term = (PI / a).sqrt();
    let exp_term = (2.0 * (a * energy).sqrt()).exp();
    sqrt_term * exp_term / (12.0 * energy.powf(1.25))
}

#[allow(dead_code)]
fn cumulative_levels(energy: f64, a: f64) -> f64 {
    // Approximate integral of level density
    nuclear_level_density(energy, a) * energy / 2.0
}

#[allow(dead_code)]
fn ising_magnetization(t: f64, tc: f64) -> f64 {
    if t >= tc {
        0.0
    } else {
        (1.0 - t / tc).powf(0.125)
    }
}

#[allow(dead_code)]
fn ising_correlation_length(t: f64, tc: f64) -> f64 {
    (t - tc).abs().powf(-1.0)
}

#[allow(dead_code)]
fn ising_heat_capacity(t: f64, tc: f64) -> f64 {
    if (t - tc).abs() < 1e-6 {
        1000.0 // Divergent at critical point
    } else {
        (t - tc).abs().powf(-0.1)
    }
}

#[allow(dead_code)]
fn bcs_gap_ratio(t: f64, tc: f64) -> f64 {
    if t >= tc {
        0.0
    } else {
        (1.0 - t / tc).sqrt()
    }
}

#[allow(dead_code)]
fn bcs_heat_capacity(t: f64, tc: f64) -> f64 {
    if t >= tc {
        t / tc // Normal state
    } else {
        let gap_ratio = bcs_gap_ratio(t, tc);
        gap_ratio * gap_ratio * (-1.76 * tc / t).exp()
    }
}

#[allow(dead_code)]
fn bcs_critical_field(t: f64, tc: f64) -> f64 {
    if t >= tc {
        0.0
    } else {
        (1.0 - (t / tc).powi(2)).sqrt()
    }
}

#[allow(dead_code)]
fn quantum_susceptibility(g: f64) -> f64 {
    g.abs().powf(-0.75) // Critical exponent γ/ν
}

#[allow(dead_code)]
fn quantum_correlation_length(g: f64) -> f64 {
    g.abs().powf(-1.0) // Critical exponent ν
}

#[allow(dead_code)]
fn quantum_scaling_function(g: f64) -> f64 {
    (-g.abs()).exp() // Simplified scaling function
}

#[allow(dead_code)]
fn chirp_time(_mass_solar: f64, f0hz: f64) -> f64 {
    // Simplified formula for binary inspiral time
    let total_mass_kg = _mass_solar * 1.989e30;
    let g: f64 = 6.674e-11;
    let c: f64 = 2.998e8;

    5.0 * c.powi(5) / (256.0 * PI) * (total_mass_kg * g / c.powi(3)).powf(-5.0 / 3.0)
        / f0hz.powf(8.0 / 3.0)
}

#[allow(dead_code)]
fn inspiral_frequency(t: f64, f0: f64, tau: f64) -> f64 {
    f0 * (1.0 - t / tau).powf(-3.0 / 8.0)
}

#[allow(dead_code)]
fn inspiral_strain(t: f64, mass: f64, tau: f64) -> f64 {
    // Simplified strain calculation
    let distance = 410e6 * 3.086e16; // 410 Mpc in meters
    1e-21 * (mass / 30.0) * (100e6 * 3.086e16 / distance) * (1.0 - t / tau).powf(-1.0 / 4.0)
}

#[allow(dead_code)]
fn detection_probability(snr: f64, threshold: f64) -> f64 {
    if snr >= threshold {
        0.999
    } else {
        (snr / threshold).powi(2)
    }
}

#[allow(dead_code)]
fn false_alarm_rate_from_snr(snr: f64) -> f64 {
    (-snr.powi(2) / 2.0).exp() / 1e6
}

#[allow(dead_code)]
fn gaussian_significance(snr: f64) -> f64 {
    snr / (2.0_f64).sqrt()
}

#[allow(dead_code)]
fn parameter_uncertainty_mass(m1: f64, m2: f64, distance: f64) -> f64 {
    // Simplified Fisher matrix estimate
    0.1 / (distance / 100.0).sqrt()
}

#[allow(dead_code)]
fn parameter_uncertainty_distance(_m1: f64, m2: f64, distance: f64) -> f64 {
    // Simplified distance uncertainty
    0.5 * (distance / 100.0).sqrt()
}

#[allow(dead_code)]
fn average_thermal_velocity(_temp_k: f64, masskg: f64) -> f64 {
    let k_b = 1.381e-23;
    (8.0 * k_b * _temp_k / (PI * masskg)).sqrt()
}

#[allow(dead_code)]
fn thermal_velocity(_temp_k: f64, masskg: f64) -> f64 {
    let k_b = 1.381e-23;
    (k_b * _temp_k / masskg).sqrt()
}

#[allow(dead_code)]
fn most_probable_velocity(_temp_k: f64, masskg: f64) -> f64 {
    let k_b = 1.381e-23;
    (2.0 * k_b * _temp_k / masskg).sqrt()
}

#[allow(dead_code)]
fn fusion_reaction_rate(_tempkev: f64, z1: f64, z2: f64) -> f64 {
    // Simplified Gamow peak calculation
    let gamow_energy = gamow_peak_energy(_tempkev, z1, z2);
    1e-16 * (gamow_energy / _tempkev).exp() // cm³/s
}

#[allow(dead_code)]
fn gamow_peak_energy(_tempkev: f64, z1: f64, z2: f64) -> f64 {
    // Energy at Gamow peak
    1.22 * (z1 * z2).powf(2.0 / 3.0) * _tempkev.powf(1.0 / 3.0)
}

#[allow(dead_code)]
fn fusion_power_density(_temp_kev: f64, rate: f64, qvalue: f64) -> f64 {
    // Power density in MW/m³
    let density = 1e20; // particles/m³
    rate * 1e-6 * density * density * qvalue * 1.602e-13 * 1e-6
}

#[allow(dead_code)]
fn plasma_dispersion_function(zeta: Complex64) -> Complex64 {
    // Simplified plasma dispersion function
    // This would need proper implementation of Faddeeva function
    Complex64::new(1.0 / zeta.re, -PI.sqrt() * (-zeta.re.powi(2)).exp())
}

#[allow(dead_code)]
fn larmor_radius_cm(_energy_kev: f64, b_fieldt: f64) -> f64 {
    let mass_kg = 3.34e-27; // deuteron
    let charge = 1.602e-19;
    let velocity = (2.0 * _energy_kev * 1000.0 * 1.602e-19 / mass_kg).sqrt();
    mass_kg * velocity / (charge * b_fieldt) * 100.0 // convert to cm
}

#[allow(dead_code)]
fn cyclotron_frequency_mhz(_b_fieldt: f64) -> f64 {
    let charge = 1.602e-19;
    let mass_kg = 3.34e-27;
    charge * _b_fieldt / (2.0 * PI * mass_kg) / 1e6
}

#[allow(dead_code)]
fn banana_orbit_width(_energy_kev: f64, b_fieldt: f64) -> f64 {
    // Simplified banana orbit width
    larmor_radius_cm(_energy_kev, b_fieldt) * 2.0
}

#[allow(dead_code)]
fn hydrogen_wavelength(n1: i32, n2: i32) -> f64 {
    let rydberg = 1.097e7; // m^-1
    let wavelength_m = 1.0 / (rydberg * (1.0 / (n1 * n1) as f64 - 1.0 / (n2 * n2) as f64));
    wavelength_m * 1e9 // convert to nm
}

#[allow(dead_code)]
fn sodium_d_line_wavelength(line: &str) -> f64 {
    match line {
        "D₁" => 589.6,
        "D₂" => 589.3,
        _ => 589.0, // Default sodium D-line wavelength
    }
}

#[allow(dead_code)]
fn fine_structure_splitting(n: i32, l: i32) -> f64 {
    // Simplified fine structure splitting in meV
    let alpha = 1.0 / 137.0;
    let rydberg_ev = 13.6;
    alpha * alpha * rydberg_ev * 1000.0 / (n as f64).powi(3)
}

#[allow(dead_code)]
fn vibrational_rotational_energy(v: i32, j: i32, constants: (f64, f64, f64)) -> f64 {
    let (omega_e, b_e, alpha_e) = constants;
    omega_e * (v as f64 + 0.5) + b_e * j as f64 * (j + 1) as f64
        - alpha_e * (v as f64 + 0.5) * j as f64 * (j + 1) as f64
}

#[allow(dead_code)]
fn co_fundamental_p_branch(j: i32) -> f64 {
    2170.2 - 2.0 * 1.931 * j as f64
}

#[allow(dead_code)]
fn co_fundamental_r_branch(j: i32) -> f64 {
    2170.2 + 2.0 * 1.931 * (j + 1) as f64
}

#[allow(dead_code)]
fn charging_energy_mev(capacitance: f64) -> f64 {
    let e = 1.602e-19;
    e * e / (2.0 * capacitance) / 1.602e-16 // convert to meV
}

#[allow(dead_code)]
fn max_electrons_thermal(_charging_energy_mev: f64, tempk: f64) -> f64 {
    let k_b_mev = 8.617e-5 * 1000.0; // k_B in meV/K
    _charging_energy_mev / (k_b_mev * tempk)
}

#[allow(dead_code)]
fn wigner_surmise_goe(s: f64) -> f64 {
    (PI / 2.0) * s * (-PI * s * s / 4.0).exp()
}

#[allow(dead_code)]
fn wigner_surmise_gue(s: f64) -> f64 {
    (32.0 / (PI * PI)) * s * s * (-4.0 * s * s / PI).exp()
}

#[allow(dead_code)]
fn shot_noise_tunnel_junction(current: f64) -> f64 {
    2.0 * 1.602e-19 * current // 2eI in A²/Hz
}

#[allow(dead_code)]
fn diffusion_coefficient(radius: f64, temp: f64, viscosity: f64) -> f64 {
    let k_b = 1.381e-23;
    k_b * temp / (6.0 * PI * viscosity * radius)
}

#[allow(dead_code)]
fn momentum_correlation_time(radius: f64, viscosity: f64) -> f64 {
    let mass = 4.0 / 3.0 * PI * radius.powi(3) * 1000.0; // assume water density
    mass / (6.0 * PI * viscosity * radius)
}

#[allow(dead_code)]
fn finitesize_correlation(size: f64, nu: f64) -> f64 {
    1.0 / size.powf(1.0 / nu)
}

#[allow(dead_code)]
fn finitesize_susceptibility(size: f64, exponents: (f64, f64, f64)) -> f64 {
    let (nu, gamma, beta) = exponents;
    size.powf(-gamma / nu)
}

#[allow(dead_code)]
fn finitesize_order_parameter(size: f64, beta: f64) -> f64 {
    size.powf(-beta)
}

#[allow(dead_code)]
fn jacobi_cn(u: f64, m: f64) -> f64 {
    // Simplified Jacobi cn function - would need proper implementation
    if m < 0.1 {
        u.cos()
    } else if m > 0.9 {
        1.0 / u.cosh()
    } else {
        (1.0 - m * u.sin().powi(2)).sqrt()
    }
}

#[allow(dead_code)]
fn complete_elliptic_k(m: f64) -> f64 {
    // Simplified complete elliptic integral K(m)
    if m < 0.1 {
        PI / 2.0
    } else {
        PI / 2.0 * (1.0 + m / 4.0)
    }
}

#[allow(dead_code)]
fn peregrine_amplitude_squared(z: f64, t: f64) -> f64 {
    let denominator = 1.0 + 4.0 * z * z + 4.0 * t * t;
    let numerator = 4.0 * (1.0 + 2.0 * z);
    (1.0 - numerator / denominator).powi(2)
}

#[allow(dead_code)]
fn peregrine_phase(z: f64, t: f64) -> f64 {
    z + 2.0 * (2.0 * z / (1.0 + 4.0 * z * z + 4.0 * t * t)).atan()
}

#[allow(dead_code)]
fn kerr_comb_threshold(detuning: f64) -> f64 {
    // Simplified threshold calculation
    1.0 + detuning.abs()
}

#[allow(dead_code)]
fn estimate_comb_lines(detuning: f64, power: f64) -> f64 {
    if detuning < 0.0 && power > kerr_comb_threshold(detuning) {
        10.0 * (-detuning).sqrt()
    } else {
        1.0
    }
}

#[allow(dead_code)]
fn cosmic_ray_flux(energy: f64, gamma: f64) -> f64 {
    // Flux in particles/(m²·s·sr·eV)
    1e4 * energy.powf(-gamma)
}

#[allow(dead_code)]
fn werner_state_entropy(p: f64) -> f64 {
    if p == 0.0 || p == 1.0 {
        0.0
    } else {
        let lambda1 = (1.0 + 3.0 * p) / 4.0;
        let lambda2 = (1.0 - p) / 4.0;
        -lambda1 * lambda1.log2() - 3.0 * lambda2 * lambda2.log2()
    }
}

#[allow(dead_code)]
fn werner_state_concurrence(p: f64) -> f64 {
    if p > 1.0 / 3.0 {
        3.0 * p - 1.0
    } else {
        0.0
    }
}

#[allow(dead_code)]
fn shor_classical_complexity(n: f64) -> f64 {
    // Classical factoring complexity ~ exp(n^(1/3))
    (n.powf(1.0 / 3.0)).exp()
}

#[allow(dead_code)]
fn shor_quantum_complexity(n: f64) -> f64 {
    // Quantum factoring complexity ~ n³
    n.powi(3)
}
