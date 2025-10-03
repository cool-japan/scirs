//! Educational Applications of Special Functions
//!
//! This example demonstrates practical applications of special functions
//! in various scientific and engineering domains.
//!
//! Run with: cargo run --example educational_applications

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use scirs2_special::*;
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎓 Educational Applications of Special Functions");
    println!("===============================================\n");

    // Physics Applications
    physics_applications()?;

    // Engineering Applications
    engineering_applications()?;

    // Statistics and Probability
    statistics_applications()?;

    // Signal Processing
    signal_processing_applications()?;

    // Computer Graphics
    computer_graphics_applications()?;

    // Financial Mathematics
    financial_applications()?;

    println!("🎉 All applications demonstrated successfully!");
    Ok(())
}

#[allow(dead_code)]
fn physics_applications() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚛️ PHYSICS APPLICATIONS");
    println!("=======================\n");

    // Quantum Mechanics: Hydrogen Atom
    println!("🔬 Quantum Mechanics: Hydrogen Atom");
    println!("The radial part of hydrogen wavefunctions involves Laguerre polynomials:");
    println!("R_nl(r) ∝ (2r/na₀)^l e^(-r/na₀) L^(2l+1)_(n-l-1)(2r/na₀)");
    println!();

    let a0 = 1.0; // Bohr radius (atomic units)
    let _r_values = Array1::linspace(0.1, 20.0, 100);

    // Calculate radial wavefunctions for different quantum numbers
    println!("Hydrogen atom radial probability densities:");
    println!("n  l  |ψ(r=a₀)|²   |ψ(r=2a₀)|²   |ψ(r=5a₀)|²");
    println!("-------------------------------------------");

    for n in 1..=3 {
        for l in 0..n {
            let prob_a0 = hydrogen_radial_probability(n, l, a0)?;
            let prob_2a0 = hydrogen_radial_probability(n, l, 2.0 * a0)?;
            let prob_5a0 = hydrogen_radial_probability(n, l, 5.0 * a0)?;

            println!(
                "{}  {}  {:10.6}   {:10.6}   {:10.6}",
                n, l, prob_a0, prob_2a0, prob_5a0
            );
        }
    }

    // Electromagnetic Theory: Multipole Expansion
    println!("\n🌐 Electromagnetic Theory: Multipole Expansion");
    println!("Electric multipole moments involve spherical harmonics:");
    println!("φ(r,θ,φ) = Σ_l Σ_m A_lm r^l Y_l^m(θ,φ)");
    println!();

    // Calculate electric field from a quadrupole
    let theta_vals = vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI];
    println!("Quadrupole field strength vs angle (θ):");
    for &theta in &theta_vals {
        let field_strength = quadrupole_field_strength(theta, 0.0, 1.0, 2.0)?;
        println!(
            "θ = {:5.2} rad ({:3.0}°): E = {:.6}",
            theta,
            theta * 180.0 / PI,
            field_strength
        );
    }

    // Statistical Mechanics: Maxwell-Boltzmann Distribution
    println!("\n🌡️ Statistical Mechanics: Maxwell-Boltzmann Distribution");
    println!("The speed distribution involves the gamma function:");
    println!("f(v) = 4π(m/2πkT)^(3/2) v² e^(-mv²/2kT)");
    println!();

    let temperature = 300.0; // Kelvin
    let mass = 28.0; // Approximate mass of N₂ molecule (atomic mass units)

    println!("Maxwell-Boltzmann distribution for N₂ at {}K:", temperature);
    println!("Speed (m/s)  Probability Density");
    println!("--------------------------------");

    for v in [100.0, 200.0, 400.0, 600.0, 800.0, 1000.0] {
        let prob_density = maxwell_boltzmann_speed(v, mass, temperature);
        println!("{:8.0}     {:12.6e}", v, prob_density);
    }

    // Calculate most probable speed
    let k_b = 1.380649e-23; // Boltzmann constant
    let m_kg = mass * 1.66054e-27; // Convert to kg
    let most_probable_speed = (2.0 * k_b * temperature / m_kg).sqrt();
    println!("Most probable speed: {:.1} m/s", most_probable_speed);

    // Optics: Fresnel Diffraction
    println!("\n🔍 Optics: Fresnel Diffraction");
    println!("Fresnel integrals describe diffraction at a straight edge:");
    println!("C(t) = ∫₀ᵗ cos(πs²/2) ds,  S(t) = ∫₀ᵗ sin(πs²/2) ds");
    println!();

    println!("Fresnel diffraction parameters:");
    println!("t      C(t)      S(t)      Intensity");
    println!("------------------------------------");

    for i in 0..=10 {
        let t = i as f64 * 0.5;
        let (c_val, s_val) = fresnel(t)?;
        let intensity = (0.5_f64 + c_val).powi(2) + (0.5_f64 + s_val).powi(2);
        println!("{:4.1}  {:8.5}  {:8.5}  {:8.5}", t, c_val, s_val, intensity);
    }

    // Heat Conduction: Error Functions
    println!("\n🔥 Heat Conduction");
    println!("Temperature distribution in a semi-infinite rod:");
    println!("T(x,t) = T₀ erfc(x / 2√(αt))");
    println!();

    let t0 = 100.0; // Initial temperature
    let alpha: f64 = 1e-6; // Thermal diffusivity (m²/s)
    let times = vec![60.0, 300.0, 900.0, 3600.0]; // seconds

    println!("Temperature distribution:");
    println!("Distance(cm)  t=1min   t=5min   t=15min  t=1hour");
    println!("------------------------------------------------");

    for x_cm in [0.0, 1.0, 2.0, 5.0, 10.0] {
        let x = x_cm / 100.0; // Convert to meters
        print!("{:8.1}    ", x_cm);

        for &t in &times {
            let temp = t0 * erfc(x / (2.0_f64 * (alpha * t).sqrt()));
            print!("{:7.1}  ", temp);
        }
        println!();
    }

    println!("\n✅ Physics applications completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn engineering_applications() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 ENGINEERING APPLICATIONS");
    println!("===========================\n");

    // Vibrations and Acoustics: Bessel Functions
    println!("🎵 Vibrations: Circular Membrane");
    println!("Natural frequencies of a circular membrane involve Bessel function zeros:");
    println!("ω_mn = (α_mn c) / R, where α_mn are zeros of J_m(x)");
    println!();

    let radius = 0.1; // meters
    let wave_speed = 343.0; // m/s (sound speed in air)

    println!("Natural frequencies of circular membrane (R={}m):", radius);
    println!("Mode (m,n)  Frequency (Hz)  Wavelength (m)");
    println!("------------------------------------------");

    // Calculate first few zeros of J0 and J1
    for n in 1..=3 {
        let zero: f64 = j0_zeros(n)?;
        let frequency = zero * wave_speed / (2.0 * PI * radius);
        let wavelength = wave_speed / frequency;
        println!("(0,{})      {:8.1}        {:8.4}", n, frequency, wavelength);
    }

    for n in 1..=3 {
        let zero: f64 = j1_zeros(n)?;
        let frequency = zero * wave_speed / (2.0 * PI * radius);
        let wavelength = wave_speed / frequency;
        println!("(1,{})      {:8.1}        {:8.4}", n, frequency, wavelength);
    }

    // Control Systems: Step Response
    println!("\n🎛️ Control Systems: Step Response");
    println!("Second-order system step response involves exponentials and trigonometry:");
    println!("c(t) = 1 - e^(-ζωₙt)[cos(ωdt) + (ζ/√(1-ζ²))sin(ωdt)]");
    println!();

    let wn = 10.0; // Natural frequency (rad/s)
    let damping_ratios = vec![0.1, 0.3, 0.7, 1.0, 1.5];

    println!("Step response characteristics:");
    println!("ζ     Overshoot  Settling Time  Peak Time");
    println!("------------------------------------------");

    for &zeta in &damping_ratios {
        let (overshoot, settling_time, peak_time) = step_response_characteristics(wn, zeta);
        println!(
            "{:.1}   {:7.1}%    {:8.3}s      {:6.3}s",
            zeta,
            overshoot * 100.0,
            settling_time,
            peak_time
        );
    }

    // Signal Processing: Window Functions
    println!("\n📡 Signal Processing: Window Functions");
    println!("Window functions for spectral analysis often involve special functions.");
    println!();

    let n_points = 64;
    let _n = Array1::linspace(0.0, (n_points - 1) as f64, n_points);

    // Calculate different window functions
    println!("Window function comparison (N={}):", n_points);
    println!("n    Hann      Hamming   Blackman   Kaiser(β=5)");
    println!("----------------------------------------------");

    for i in [0, 8, 16, 24, 31, 32, 40, 48, 56, 63] {
        let hann = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n_points - 1) as f64).cos());
        let hamming = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n_points - 1) as f64).cos();
        let blackman = 0.42 - 0.5 * (2.0 * PI * i as f64 / (n_points - 1) as f64).cos()
            + 0.08 * (4.0 * PI * i as f64 / (n_points - 1) as f64).cos();

        // Kaiser window using modified Bessel function
        let beta = 5.0;
        let alpha = (n_points - 1) as f64 / 2.0;
        let arg = beta * (1.0 - ((i as f64 - alpha) / alpha).powi(2)).sqrt();
        let kaiser = i0(arg) / i0(beta);

        println!(
            "{:2}   {:7.4}   {:7.4}   {:7.4}    {:7.4}",
            i, hann, hamming, blackman, kaiser
        );
    }

    // Antenna Theory: Radiation Patterns
    println!("\n📡 Antenna Theory: Array Radiation Pattern");
    println!("Linear antenna arrays have radiation patterns involving sinc functions:");
    println!("F(θ) = sinc(Nπd sin(θ)/λ) / sinc(πd sin(θ)/λ)");
    println!();

    let wavelength = 1.0; // Normalized wavelength
    let element_spacing = 0.5; // λ/2 spacing
    let num_elements = 8;

    println!("Radiation pattern for {}-element array:", num_elements);
    println!("Angle(°)  Pattern(dB)  Normalized");
    println!("--------------------------------");

    for angle_deg in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0] {
        let angle_rad = angle_deg * PI / 180.0;
        let pattern = array_radiation_pattern(num_elements, element_spacing, wavelength, angle_rad);
        let pattern_db = 20.0 * pattern.log10();
        println!(
            "{:6.0}    {:8.2}     {:7.4}",
            angle_deg, pattern_db, pattern
        );
    }

    // Structural Engineering: Beam Deflection
    println!("\n🏗️ Structural Engineering: Beam with Distributed Load");
    println!("Beam deflection under distributed loads can involve special functions.");
    println!();

    let beam_length = 10.0; // meters
    let elastic_modulus = 200e9; // Pa (steel)
    let moment_inertia = 1e-4; // m⁴
    let distributed_load = 10000.0; // N/m

    println!("Simply supported beam deflection:");
    println!("Position(m)  Deflection(mm)  Slope(mrad)");
    println!("---------------------------------------");

    for x in [0.0, 1.0, 2.5, 5.0, 7.5, 9.0, 10.0] {
        let (deflection, slope) = beam_deflection_distributed(
            x,
            beam_length,
            distributed_load,
            elastic_modulus,
            moment_inertia,
        );
        println!(
            "{:8.1}    {:10.2}      {:8.2}",
            x,
            deflection * 1000.0,
            slope * 1000.0
        );
    }

    // Fluid Mechanics: Boundary Layer
    println!("\n🌊 Fluid Mechanics: Boundary Layer Profile");
    println!("Blasius boundary layer solution involves error functions.");
    println!();

    let free_stream_velocity = 10.0; // m/s
    let kinematic_viscosity = 1.5e-5; // m²/s (air)
    let distance = 1.0; // m from leading edge

    println!("Boundary layer velocity profile at x={}m:", distance);
    println!("y/δ     u/U∞    Dimensionless y");
    println!("-----------------------------");

    let reynolds_x: f64 = free_stream_velocity * distance / kinematic_viscosity;
    let delta = 5.0 * distance / reynolds_x.sqrt(); // Boundary layer thickness

    for y_over_delta in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2] {
        let y = y_over_delta * delta;
        let eta = y * (free_stream_velocity / (kinematic_viscosity * distance)).sqrt() / 2.0;
        let u_ratio = blasius_velocity_profile(eta);
        println!("{:5.1}   {:6.3}   {:10.2}", y_over_delta, u_ratio, eta);
    }

    println!("\n✅ Engineering applications completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn statistics_applications() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 STATISTICS AND PROBABILITY");
    println!("=============================\n");

    // Central Limit Theorem Demonstration
    println!("📈 Central Limit Theorem");
    println!("Sum of random variables approaches normal distribution.");
    println!("The error function appears in the cumulative distribution.");
    println!();

    // Calculate probabilities using error function
    let z_scores = vec![-2.0, -1.0, 0.0, 1.0, 1.96, 2.0, 2.58];
    println!("Standard normal probabilities:");
    println!("z-score  P(Z ≤ z)  P(|Z| ≤ z)  Percentile");
    println!("------------------------------------------");

    for &z in &z_scores {
        let prob_less = 0.5 * (1.0 + erf(z / 2.0_f64.sqrt()));
        let prob_abs = erf(z.abs() / 2.0_f64.sqrt());
        let percentile = prob_less * 100.0;
        println!(
            "{:7.2}  {:8.5}   {:8.5}    {:7.2}%",
            z, prob_less, prob_abs, percentile
        );
    }

    // Chi-Square Distribution
    println!("\n🔲 Chi-Square Distribution");
    println!("Chi-square distribution involves incomplete gamma functions:");
    println!("F(x;k) = γ(k/2, x/2) / Γ(k/2)");
    println!();

    let degrees_freedom = vec![1, 2, 3, 5, 10];
    let x_values = vec![1.0, 2.0, 5.0, 10.0, 15.0];

    println!("Chi-square cumulative probabilities:");
    print!("x\\df  ");
    for &df in &degrees_freedom {
        print!("  df={}  ", df);
    }
    println!();

    for &x in &x_values {
        print!("{:4.1}  ", x);
        for &df in &degrees_freedom {
            let prob = chi_square_cdf(x, df as f64)?;
            print!("{:6.3}  ", prob);
        }
        println!();
    }

    // Student's t-Distribution
    println!("\n📚 Student's t-Distribution");
    println!("Critical values for hypothesis testing:");
    println!();

    let confidence_levels = vec![0.90, 0.95, 0.99];
    let dfs = vec![1, 2, 5, 10, 20, 30];

    println!("t-distribution critical values:");
    println!("df    90% CI    95% CI    99% CI");
    println!("-------------------------------");

    for &df in &dfs {
        print!("{:2}  ", df);
        for &conf_level in &confidence_levels {
            let alpha = 1.0 - conf_level;
            let t_critical = t_inverse_cdf(1.0 - alpha / 2.0, df as f64)?;
            print!("{:8.3}  ", t_critical);
        }
        println!();
    }

    // Beta Distribution in Bayesian Analysis
    println!("\n🎯 Bayesian Analysis: Beta Distribution");
    println!("Beta distribution is the conjugate prior for binomial likelihood.");
    println!("Beta(x; α, β) = x^(α-1) (1-x)^(β-1) / B(α,β)");
    println!();

    // Prior and posterior analysis
    let prior_alpha = 2.0;
    let prior_beta = 5.0;
    let successes = 7;
    let trials = 10;

    let posterior_alpha = prior_alpha + successes as f64;
    let posterior_beta = prior_beta + (trials - successes) as f64;

    println!("Bayesian update example:");
    println!("Prior: Beta({}, {})", prior_alpha, prior_beta);
    println!("Data: {} successes in {} trials", successes, trials);
    println!("Posterior: Beta({}, {})", posterior_alpha, posterior_beta);
    println!();

    let x_vals = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    println!("x     Prior    Posterior  Evidence Ratio");
    println!("----------------------------------------");

    for &x in &x_vals {
        let prior_density = beta_density(x, prior_alpha, prior_beta)?;
        let posterior_density = beta_density(x, posterior_alpha, posterior_beta)?;
        let evidence_ratio = posterior_density / prior_density;
        println!(
            "{:.1}   {:7.4}    {:7.4}     {:8.2}",
            x, prior_density, posterior_density, evidence_ratio
        );
    }

    // Reliability Engineering: Weibull Distribution
    println!("\n⚡ Reliability Engineering: Weibull Distribution");
    println!("Weibull distribution models failure times and involves gamma functions.");
    println!();

    let shape_parameters = vec![0.5, 1.0, 1.5, 2.0, 3.0];
    let scale = 1000.0; // hours
    let time = 500.0; // hours

    println!("Weibull reliability at t={}h (λ={}):", time, scale);
    println!("Shape(k)  Reliability  Hazard Rate");
    println!("----------------------------------");

    for &k in &shape_parameters {
        let reliability = weibull_reliability(time, k, scale);
        let hazard_rate = weibull_hazard_rate(time, k, scale);
        println!("{:6.1}    {:8.5}     {:8.5}", k, reliability, hazard_rate);
    }

    // Quality Control: Control Charts
    println!("\n📋 Quality Control: Control Chart Limits");
    println!("Control limits based on normal distribution properties.");
    println!();

    let samplesizes = vec![1, 4, 9, 16, 25];
    let sigma_levels = [2.0, 3.0];

    println!("Control chart factors:");
    println!("n     2σ limits   3σ limits   LCL      UCL");
    println!("------------------------------------------");

    for &n in &samplesizes {
        if let Some(&sigma) = sigma_levels.first() {
            let factor = sigma / (n as f64).sqrt();
            let lcl = -factor;
            let ucl = factor;
            println!(
                "{:2}    {:8.4}    {:8.4}   {:6.3}   {:6.3}",
                n, factor, factor, lcl, ucl
            );
        }
    }

    println!("\n✅ Statistics applications completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn signal_processing_applications() -> Result<(), Box<dyn std::error::Error>> {
    println!("📡 SIGNAL PROCESSING APPLICATIONS");
    println!("=================================\n");

    // Digital Filter Design
    println!("🎛️ Digital Filter Design: Bessel Filters");
    println!("Bessel filters have maximally flat group delay.");
    println!();

    let frequencies = Array1::logspace(10.0, -1.0, 1.0, 50); // 0.1 to 1.0 Hz
    let cutoff_freq = 1.0; // Normalized cutoff frequency
    let filter_order = 4;

    println!("Bessel filter response (order {}):", filter_order);
    println!("Freq(Hz)  Magnitude(dB)  Phase(deg)  Group Delay");
    println!("-----------------------------------------------");

    for i in [5, 10, 15, 20, 25, 30, 35, 40, 45] {
        let freq = frequencies[i];
        let (magnitude_db, phase_deg, group_delay) =
            bessel_filter_response(freq, cutoff_freq, filter_order);
        println!(
            "{:7.3}    {:9.2}     {:8.1}     {:8.2}",
            freq, magnitude_db, phase_deg, group_delay
        );
    }

    // Communication Systems: Error Probability
    println!("\n📶 Communication Systems: Bit Error Rate");
    println!("Error probability in digital communication involves complementary error function.");
    println!();

    let snr_db_values = vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];

    println!("Bit error rates for different modulation schemes:");
    println!("SNR(dB)   BPSK       QPSK       16-QAM     64-QAM");
    println!("---------------------------------------------------");

    for &snr_db in &snr_db_values {
        let snr = 10.0_f64.powf(snr_db / 10.0);

        let ber_bpsk = 0.5 * erfc((snr).sqrt());
        let ber_qpsk = 0.5 * erfc((snr).sqrt()); // Same as BPSK for Gray coding
        let ber_16qam = 0.375 * erfc((0.4 * snr).sqrt());
        let ber_64qam = 0.4375 * erfc((snr / 21.0).sqrt());

        println!(
            "{:6.1}  {:9.2e}  {:9.2e}  {:9.2e}  {:9.2e}",
            snr_db, ber_bpsk, ber_qpsk, ber_16qam, ber_64qam
        );
    }

    // Radar Systems: Detection Probability
    println!("\n🛰️ Radar Systems: Detection Performance");
    println!("Radar detection involves modified Bessel functions in Rician distributions.");
    println!();

    let snr_values = vec![0.0, 5.0, 10.0, 15.0, 20.0];
    let false_alarm_rates = vec![1e-6, 1e-8, 1e-10];

    println!("Detection probability vs SNR:");
    println!("SNR(dB)    Pfa=1e-6   Pfa=1e-8   Pfa=1e-10");
    println!("------------------------------------------");

    for &snr_db in &snr_values {
        let snr_linear = 10.0_f64.powf(snr_db / 10.0);
        print!("{:6.1}    ", snr_db);

        for &pfa in &false_alarm_rates {
            let pd = radar_detection_probability(snr_linear, pfa)?;
            print!("{:8.4}   ", pd);
        }
        println!();
    }

    // Audio Processing: Psychoacoustic Masking
    println!("\n🎵 Audio Processing: Psychoacoustic Masking");
    println!("Auditory masking curves approximate Bessel-like functions.");
    println!();

    let masker_freq = 1000.0; // Hz
    let masker_level = 60.0; // dB SPL

    println!("Masking threshold around {}Hz masker:", masker_freq);
    println!("Freq(Hz)  Mask Threshold(dB)  Spread Function");
    println!("----------------------------------------------");

    let test_frequencies = vec![250.0, 500.0, 800.0, 1000.0, 1200.0, 1600.0, 2000.0, 4000.0];
    for &freq in &test_frequencies {
        let (threshold, spread) = psychoacoustic_masking(freq, masker_freq, masker_level);
        println!(
            "{:7.0}      {:11.1}        {:10.4}",
            freq, threshold, spread
        );
    }

    // Image Processing: Gaussian Blur
    println!("\n🖼️ Image Processing: Gaussian Kernels");
    println!("Gaussian blur kernels are based on the error function.");
    println!();

    let sigma_values = vec![0.5, 1.0, 2.0];
    println!("Gaussian kernel weights (normalized):");

    for &sigma in &sigma_values {
        println!("\nσ = {}:", sigma);
        print!("x: ");
        for x in -3..=3 {
            print!("{:6}", x);
        }
        println!();
        print!("w: ");

        let mut weights = Vec::new();
        for x in -3..=3 {
            let weight = gaussian_kernel_weight(x as f64, sigma);
            weights.push(weight);
            print!("{:6.3}", weight);
        }
        println!();

        let sum: f64 = weights.iter().sum();
        println!("Sum: {:.6} (should be close to 1.0)", sum);
    }

    // Time-Frequency Analysis: Wavelets
    println!("\n〰️ Time-Frequency Analysis: Morlet Wavelet");
    println!("Morlet wavelets involve Gaussian functions and complex exponentials.");
    println!();

    let center_freq = 1.0; // Hz
    let time_points = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

    println!("Morlet wavelet at different time points:");
    println!("Time(s)   Real Part   Imag Part   Magnitude");
    println!("-------------------------------------------");

    for &t in &time_points {
        let morlet = morlet_wavelet(t, center_freq);
        println!(
            "{:6.1}    {:8.4}    {:8.4}     {:8.4}",
            t,
            morlet.re,
            morlet.im,
            morlet.norm()
        );
    }

    println!("\n✅ Signal processing applications completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn computer_graphics_applications() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 COMPUTER GRAPHICS APPLICATIONS");
    println!("=================================\n");

    // Spherical Harmonics Lighting
    println!("💡 Spherical Harmonics Lighting");
    println!("Environmental lighting using spherical harmonics basis functions.");
    println!();

    let light_directions = [
        (0.0, 0.0),            // Overhead
        (PI / 2.0, 0.0),       // Horizon, front
        (PI / 2.0, PI / 2.0),  // Horizon, side
        (PI / 2.0, PI),        // Horizon, back
        (3.0 * PI / 4.0, 0.0), // Low angle
    ];

    println!("Spherical harmonics coefficients for lighting:");
    println!("Direction        Y₀⁰      Y₁⁻¹     Y₁⁰      Y₁¹      Y₂⁰");
    println!("---------------------------------------------------------------");

    for (i, &(theta, phi)) in light_directions.iter().enumerate() {
        let y00 = sph_harm(0, 0, theta, phi)?;
        let y1m1 = sph_harm(1, -1, theta, phi)?;
        let y10 = sph_harm(1, 0, theta, phi)?;
        let y11 = sph_harm(1, 1, theta, phi)?;
        let y20 = sph_harm(2, 0, theta, phi)?;

        let direction_name = match i {
            0 => "Overhead     ",
            1 => "Horizon Front",
            2 => "Horizon Side ",
            3 => "Horizon Back ",
            4 => "Low Angle    ",
            _ => "Unknown      ",
        };

        println!(
            "{}  {:7.4}  {:7.4}  {:7.4}  {:7.4}  {:7.4}",
            direction_name, y00, y1m1, y10, y11, y20
        );
    }

    // Noise Functions: Perlin Noise
    println!("\n🌊 Procedural Generation: Noise Functions");
    println!("Smooth noise functions use interpolation and basis functions.");
    println!();

    println!("Gradient noise evaluation:");
    println!("x      y      Noise Value  Turbulence");
    println!("------------------------------------");

    for i in 0..8 {
        let x = i as f64 * 0.5;
        let y = i as f64 * 0.3;
        let noise_val = gradient_noise_2d(x, y);
        let turbulence = fractal_noise_2d(x, y, 4);
        println!(
            "{:4.1}   {:4.1}   {:9.4}    {:9.4}",
            x, y, noise_val, turbulence
        );
    }

    // Bezier Curves: Bernstein Polynomials
    println!("\n🎯 Bezier Curves: Bernstein Basis");
    println!("Bezier curves use Bernstein polynomials which involve binomial coefficients.");
    println!();

    // Cubic Bezier curve
    let control_points = vec![
        (0.0, 0.0), // P0
        (1.0, 2.0), // P1
        (3.0, 2.0), // P2
        (4.0, 0.0), // P3
    ];

    println!("Cubic Bezier curve evaluation:");
    println!("t     x(t)    y(t)    Curvature");
    println!("------------------------------");

    for i in 0..=10 {
        let t = i as f64 / 10.0;
        let (x, y, curvature) = evaluate_cubic_bezier(&control_points, t);
        println!("{:.1}   {:6.3}  {:6.3}   {:8.4}", t, x, y, curvature);
    }

    // Surface Modeling: B-Splines
    println!("\n🏔️ Surface Modeling: B-Spline Basis Functions");
    println!("B-splines provide smooth basis functions for surface modeling.");
    println!();

    let degree = 3;
    let knot_vector = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];

    println!("B-spline basis functions (degree {}):", degree);
    println!("u     N₀(u)   N₁(u)   N₂(u)   N₃(u)   N₄(u)   N₅(u)");
    println!("---------------------------------------------------");

    for i in 0..=6 {
        let u = i as f64 * 0.5;
        print!("{:.1}   ", u);

        for j in 0..6 {
            let basis_val = b_spline_basis(j, degree, u, &knot_vector);
            print!("{:6.3}  ", basis_val);
        }
        println!();
    }

    // Ray Tracing: Spherical Coordinates
    println!("\n☀️ Ray Tracing: Spherical Sampling");
    println!("Monte Carlo ray tracing uses spherical coordinate sampling.");
    println!();

    println!("Hemisphere sampling for global illumination:");
    println!("Sample#  θ(rad)  φ(rad)   x       y       z      PDF");
    println!("----------------------------------------------------");

    for i in 0..8 {
        let u1 = (i as f64 + 0.5) / 8.0;
        let u2 = (i as f64 * 0.7 + 0.3) % 1.0;

        let (theta, phi, pdf) = cosine_weighted_hemisphere_sample(u1, u2);
        let x = theta.sin() * phi.cos();
        let y = theta.sin() * phi.sin();
        let z = theta.cos();

        println!(
            "{:7}  {:6.3}  {:6.3}  {:6.3}  {:6.3}  {:6.3}  {:6.3}",
            i, theta, phi, x, y, z, pdf
        );
    }

    // Animation: Easing Functions
    println!("\n🎬 Animation: Easing Functions");
    println!("Smooth animation transitions using mathematical curves.");
    println!();

    println!("Easing function comparison:");
    println!("t     Linear   Ease-In  Ease-Out  Smooth   Bounce");
    println!("------------------------------------------------");

    for i in 0..=10 {
        let t = i as f64 / 10.0;

        let linear = t;
        let ease_in = t * t;
        let ease_out = 1.0 - (1.0 - t) * (1.0 - t);
        let smooth = 3.0 * t * t - 2.0 * t * t * t;
        let bounce = bounce_easing(t);

        println!(
            "{:.1}   {:6.3}  {:6.3}   {:6.3}   {:6.3}  {:6.3}",
            t, linear, ease_in, ease_out, smooth, bounce
        );
    }

    // Mesh Processing: Gaussian Curvature
    println!("\n🕸️ Mesh Processing: Surface Curvature");
    println!("Surface analysis using differential geometry and special functions.");
    println!();

    // Simulate a sphere (analytical curvature = 1/r²)
    let sphere_radius = 2.0;
    let analytical_curvature = 1.0 / (sphere_radius * sphere_radius);

    println!("Sphere surface analysis (radius = {}):", sphere_radius);
    println!("Point      Gaussian K   Mean H      Principal κ₁  κ₂");
    println!("------------------------------------------------");

    let test_points = [
        (0.0, 0.0, sphere_radius),
        (
            sphere_radius / 2.0_f64.sqrt(),
            sphere_radius / 2.0_f64.sqrt(),
            0.0,
        ),
        (sphere_radius, 0.0, 0.0),
    ];

    for (i, &_x_y_z) in test_points.iter().enumerate() {
        let gaussian_curvature = analytical_curvature;
        let mean_curvature = 1.0 / sphere_radius;
        let k1 = 1.0 / sphere_radius;
        let k2 = 1.0 / sphere_radius;

        println!(
            "Point {}    {:8.4}   {:8.4}   {:8.4}   {:8.4}",
            i, gaussian_curvature, mean_curvature, k1, k2
        );
    }

    println!("\n✅ Computer graphics applications completed!\n");
    Ok(())
}

#[allow(dead_code)]
fn financial_applications() -> Result<(), Box<dyn std::error::Error>> {
    println!("💰 FINANCIAL MATHEMATICS");
    println!("========================\n");

    // Black-Scholes Option Pricing
    println!("📈 Black-Scholes Option Pricing");
    println!("Option values involve cumulative normal distribution (error function).");
    println!();

    let spot_price = 100.0;
    let strike_prices = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let risk_free_rate = 0.05;
    let volatility = 0.2;
    let time_to_expiry = 0.25; // 3 months

    println!(
        "European option prices (S={}, r={:.1}%, σ={:.1}%, T={:.2}y):",
        spot_price,
        risk_free_rate * 100.0,
        volatility * 100.0,
        time_to_expiry
    );
    println!("Strike   Call Price   Put Price   Call Delta   Put Delta");
    println!("------------------------------------------------------");

    for &strike in &strike_prices {
        let (call_price, put_price, call_delta, put_delta) = black_scholes_option(
            spot_price,
            strike,
            risk_free_rate,
            volatility,
            time_to_expiry,
        )?;

        println!(
            "{:6.0}    {:8.2}     {:8.2}     {:8.4}     {:8.4}",
            strike, call_price, put_price, call_delta, put_delta
        );
    }

    // Value at Risk (VaR) Calculation
    println!("\n📊 Value at Risk (VaR) Calculation");
    println!("VaR calculations use inverse normal distribution.");
    println!();

    let portfolio_value = 1_000_000.0;
    let confidence_levels = vec![0.90, 0.95, 0.99];
    let volatilities = vec![0.10, 0.15, 0.20, 0.25];
    let time_horizon = 1.0; // days

    println!("Daily VaR for ${:.0}k portfolio:", portfolio_value / 1000.0);
    println!("Volatility   90% VaR    95% VaR    99% VaR");
    println!("-----------------------------------------");

    for &vol in &volatilities {
        print!("{:8.1}%  ", vol * 100.0);

        for &conf in &confidence_levels {
            let z_score = ndtri(conf)?; // Inverse normal CDF
            let var = portfolio_value * vol * (time_horizon / 365.0_f64).sqrt() * z_score;
            print!("{:8.0}   ", var);
        }
        println!();
    }

    // Credit Risk: Default Probability
    println!("\n💳 Credit Risk: Default Probability Models");
    println!("Merton model uses normal distribution for default probability.");
    println!();

    let asset_values = vec![80.0, 90.0, 100.0, 110.0, 120.0];
    let debt_value = 100.0;
    let asset_volatility = 0.3;
    let time_horizon = 1.0; // years

    println!(
        "Merton model default probabilities (Debt = {}):",
        debt_value
    );
    println!("Asset Value   Default Prob   Distance to Default");
    println!("---------------------------------------------");

    for &asset_val in &asset_values {
        let (default_prob, distance_to_default) = merton_default_probability(
            asset_val,
            debt_value,
            risk_free_rate,
            asset_volatility,
            time_horizon,
        )?;

        println!(
            "{:10.0}      {:8.4}        {:11.2}",
            asset_val, default_prob, distance_to_default
        );
    }

    // Interest Rate Models: Vasicek Model
    println!("\n🏦 Interest Rate Models: Vasicek Model");
    println!("Bond pricing in Vasicek model involves exponential integrals.");
    println!();

    let current_rate = 0.03;
    let mean_reversion_speed = 0.1;
    let long_term_mean = 0.04;
    let rate_volatility = 0.01;
    let maturities = vec![0.25, 0.5, 1.0, 2.0, 5.0, 10.0];

    println!("Vasicek bond prices and yields:");
    println!("Maturity   Bond Price   Yield(%)   Duration");
    println!("------------------------------------------");

    for &maturity in &maturities {
        let (bond_price, yield_rate, duration) = vasicek_bond_pricing(
            current_rate,
            mean_reversion_speed,
            long_term_mean,
            rate_volatility,
            maturity,
        );

        println!(
            "{:7.2}     {:8.4}    {:6.2}    {:7.2}",
            maturity,
            bond_price,
            yield_rate * 100.0,
            duration
        );
    }

    // Portfolio Optimization: Mean-Variance
    println!("\n📊 Portfolio Optimization: Efficient Frontier");
    println!("Mean-variance optimization involves quadratic forms and matrix operations.");
    println!();

    let expected_returns = vec![0.08, 0.12, 0.15]; // Asset expected returns
    let volatilities = vec![0.15, 0.20, 0.25]; // Asset volatilities
    let correlations = vec![
        vec![1.0, 0.3, 0.1],
        vec![0.3, 1.0, 0.2],
        vec![0.1, 0.2, 1.0],
    ];

    println!("Efficient frontier points:");
    println!("Target Return   Portfolio Risk   Sharpe Ratio");
    println!("--------------------------------------------");

    for i in 0..8 {
        let target_return = 0.06 + i as f64 * 0.02;
        let (portfolio_risk, sharpe_ratio) = efficient_frontier_point(
            target_return,
            &expected_returns,
            &volatilities,
            &correlations,
            risk_free_rate,
        );

        println!(
            "{:11.1}%      {:10.1}%      {:10.2}",
            target_return * 100.0,
            portfolio_risk * 100.0,
            sharpe_ratio
        );
    }

    // Monte Carlo Simulation: Asian Options
    println!("\n🎲 Monte Carlo Simulation: Asian Option Pricing");
    println!("Path-dependent options require numerical simulation.");
    println!();

    let num_simulations = 10000;
    let num_time_steps = 252; // Daily steps for 1 year
    let asian_strikes = vec![95.0, 100.0, 105.0];

    println!(
        "Asian option prices (arithmetic average, {} simulations):",
        num_simulations
    );
    println!("Strike   Call Price   Put Price   Std Error");
    println!("------------------------------------------");

    for &strike in &asian_strikes {
        let (call_price, put_price, std_error) = asian_option_monte_carlo(
            spot_price,
            strike,
            risk_free_rate,
            volatility,
            time_to_expiry,
            num_time_steps,
            num_simulations,
        );

        println!(
            "{:6.0}    {:8.2}     {:8.2}     {:7.4}",
            strike, call_price, put_price, std_error
        );
    }

    println!("\n✅ Financial mathematics applications completed!\n");
    Ok(())
}

// Helper functions for the applications
#[allow(dead_code)]
fn hydrogen_radial_probability(
    n: usize,
    l: usize,
    r: f64,
) -> Result<f64, Box<dyn std::error::Error>> {
    // Simplified hydrogen radial wavefunction probability density
    // This is a basic approximation for demonstration
    let alpha = 2.0 / (n as f64);
    let rho = alpha * r;

    if l == 0 {
        // s orbital approximation
        let laguerre_val = laguerre(n - 1, rho);
        let radial = rho.powf(l as f64) * (-rho / 2.0).exp() * laguerre_val;
        Ok(radial * radial * r * r)
    } else {
        // Simplified for other orbitals
        let radial = rho.powf(l as f64) * (-rho / 2.0).exp();
        Ok(radial * radial * r * r)
    }
}

#[allow(dead_code)]
fn quadrupole_field_strength(
    theta: f64,
    phi: f64,
    moment: f64,
    distance: f64,
) -> Result<f64, Box<dyn std::error::Error>> {
    // Electric field from quadrupole moment
    let y_20 = sph_harm(2, 0, theta, phi)?;
    Ok(moment * y_20 / (distance.powi(4)))
}

#[allow(dead_code)]
fn maxwell_boltzmann_speed(v: f64, mass: f64, temperature: f64) -> f64 {
    // Maxwell-Boltzmann speed distribution
    let k_b = 1.380649e-23; // Boltzmann constant
    let m = mass * 1.66054e-27; // Convert atomic mass units to kg

    let coeff = 4.0 * PI * (m / (2.0 * PI * k_b * temperature)).powf(1.5);
    coeff * v * v * (-m * v * v / (2.0 * k_b * temperature)).exp()
}

#[allow(dead_code)]
fn step_response_characteristics(wn: f64, zeta: f64) -> (f64, f64, f64) {
    // Second-order system step response characteristics
    if zeta < 1.0 {
        let wd = wn * (1.0 - zeta * zeta).sqrt();
        let overshoot = (-PI * zeta / (1.0 - zeta * zeta).sqrt()).exp();
        let settling_time = 4.0 / (zeta * wn); // 2% settling time
        let peak_time = PI / wd;
        (overshoot, settling_time, peak_time)
    } else {
        (0.0, 4.0 / (zeta * wn), 0.0) // No overshoot for overdamped
    }
}

#[allow(dead_code)]
fn array_radiation_pattern(n: usize, d: f64, lambda: f64, theta: f64) -> f64 {
    // Linear antenna array radiation pattern
    let beta = 2.0 * PI / lambda;
    let psi = beta * d * theta.sin();

    if psi.abs() < 1e-10 {
        1.0
    } else {
        let numerator = (n as f64 * psi / 2.0).sin();
        let denominator = n as f64 * (psi / 2.0).sin();
        (numerator / denominator).abs()
    }
}

#[allow(dead_code)]
fn beam_deflection_distributed(x: f64, l: f64, w: f64, e: f64, i: f64) -> (f64, f64) {
    // Simply supported beam with uniform distributed load
    let deflection = w * x / (24.0 * e * i) * (l.powi(3) - 2.0 * l * x * x + x.powi(3));
    let slope = w / (24.0 * e * i) * (l.powi(3) - 6.0 * l * x * x + 4.0 * x.powi(3));
    (deflection, slope)
}

#[allow(dead_code)]
fn blasius_velocity_profile(eta: f64) -> f64 {
    // Blasius boundary layer velocity profile (approximation)
    if eta >= 5.0 {
        1.0
    } else {
        // Approximate solution using polynomial
        let f_eta = eta - eta.powi(3) / 6.0 + eta.powi(5) / 120.0;
        f_eta.tanh()
    }
}

#[allow(dead_code)]
fn chi_square_cdf(x: f64, df: f64) -> Result<f64, Box<dyn std::error::Error>> {
    // Chi-square cumulative distribution function
    if x <= 0.0 {
        Ok(0.0)
    } else {
        Ok(gammainc(df / 2.0, x / 2.0)? / gamma(df / 2.0))
    }
}

#[allow(dead_code)]
fn t_inverse_cdf(p: f64, df: f64) -> Result<f64, Box<dyn std::error::Error>> {
    // Student's t-distribution inverse CDF (approximation)
    // This is a simplified version; real implementation would be more complex
    let z = ndtri(p)?; // Normal inverse
    if df > 30.0 {
        Ok(z) // Approximates normal for large df
    } else {
        // Cornish-Fisher expansion approximation
        let c1 = z.powi(3) + z;
        let correction = c1 / (4.0 * df);
        Ok(z + correction)
    }
}

#[allow(dead_code)]
fn beta_density(x: f64, alpha: f64, betaparam: f64) -> Result<f64, Box<dyn std::error::Error>> {
    // Beta distribution probability density function
    if x <= 0.0 || x >= 1.0 {
        Ok(0.0)
    } else {
        let beta_function = beta(alpha, betaparam);
        Ok(x.powf(alpha - 1.0) * (1.0 - x).powf(betaparam - 1.0) / beta_function)
    }
}

#[allow(dead_code)]
fn weibull_reliability(t: f64, k: f64, lambda: f64) -> f64 {
    // Weibull reliability function
    (-(t / lambda).powf(k)).exp()
}

#[allow(dead_code)]
fn weibull_hazard_rate(t: f64, k: f64, lambda: f64) -> f64 {
    // Weibull hazard rate function
    (k / lambda) * (t / lambda).powf(k - 1.0)
}

#[allow(dead_code)]
fn bessel_filter_response(freq: f64, fc: f64, order: usize) -> (f64, f64, f64) {
    // Bessel filter frequency response (simplified)
    let _s = Complex64::new(0.0, 2.0 * PI * freq / fc);

    // Bessel polynomial evaluation would go here
    // This is a simplified approximation
    let magnitude: f64 = 1.0 / (1.0 + (freq / fc).powi(2 * order as i32)).sqrt();
    let magnitude_db = 20.0 * magnitude.log10();
    let phase_deg = -(order as f64) * (freq / fc).atan() * 180.0 / PI;
    let group_delay = (order as f64) / (2.0 * PI * fc);

    (magnitude_db, phase_deg, group_delay)
}

#[allow(dead_code)]
fn radar_detection_probability(snr: f64, pfa: f64) -> Result<f64, Box<dyn std::error::Error>> {
    // Radar detection probability for Swerling I target
    let threshold = ndtri(1.0 - pfa)?; // Detection threshold
    let pd: f64 = 0.5 * (1.0 + erf((snr - threshold) / (2.0_f64.sqrt())));
    Ok(pd.clamp(0.0, 1.0))
}

#[allow(dead_code)]
fn psychoacoustic_masking(_freq: f64, masker_freq: f64, maskerlevel: f64) -> (f64, f64) {
    // Simplified psychoacoustic masking model
    let freq_ratio = _freq / masker_freq;
    let bark_diff = 13.0 * freq_ratio.ln() + 3.5 * (freq_ratio.ln()).atan();

    let spread = (-2.5 * bark_diff.abs()).exp();
    let threshold = maskerlevel + 10.0 * spread.log10() - 15.0;

    (threshold.max(0.0), spread)
}

#[allow(dead_code)]
fn gaussian_kernel_weight(x: f64, sigma: f64) -> f64 {
    // Gaussian kernel weight
    let coefficient = 1.0 / (sigma * (2.0 * PI).sqrt());
    coefficient * (-x * x / (2.0 * sigma * sigma)).exp()
}

#[allow(dead_code)]
fn morlet_wavelet(t: f64, fc: f64) -> Complex64 {
    // Morlet wavelet
    let sigma = 1.0;
    let gaussian = (-t * t / (2.0 * sigma * sigma)).exp();
    let complex_exp = Complex64::new(0.0, 2.0 * PI * fc * t).exp();
    gaussian * complex_exp
}

#[allow(dead_code)]
fn gradient_noise_2d(x: f64, y: f64) -> f64 {
    // Simplified gradient noise (Perlin-like)
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - x.floor();
    let yf = y - y.floor();

    // Simple hash function for gradients
    let hash = |x: i32, y: i32| -> f64 {
        let h = (x.wrapping_mul(374761393) + y.wrapping_mul(668265263)) as u32;
        (h as f64 / u32::MAX as f64) * 2.0 - 1.0
    };

    let g00 = hash(xi, yi);
    let g10 = hash(xi + 1, yi);
    let g01 = hash(xi, yi + 1);
    let g11 = hash(xi + 1, yi + 1);

    // Bilinear interpolation with smoothstep
    let u = xf * xf * (3.0 - 2.0 * xf);
    let v = yf * yf * (3.0 - 2.0 * yf);

    let n0 = g00 * (1.0 - u) + g10 * u;
    let n1 = g01 * (1.0 - u) + g11 * u;
    n0 * (1.0 - v) + n1 * v
}

#[allow(dead_code)]
fn fractal_noise_2d(x: f64, y: f64, octaves: usize) -> f64 {
    // Fractal noise using multiple octaves
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;

    for _ in 0..octaves {
        value += gradient_noise_2d(x * frequency, y * frequency) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    value
}

#[allow(dead_code)]
fn evaluate_cubic_bezier(_controlpoints: &[(f64, f64)], t: f64) -> (f64, f64, f64) {
    // Evaluate cubic Bezier curve and curvature
    let p0 = _controlpoints[0];
    let p1 = _controlpoints[1];
    let p2 = _controlpoints[2];
    let p3 = _controlpoints[3];

    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;

    // Position
    let x = mt3 * p0.0 + 3.0 * mt2 * t * p1.0 + 3.0 * mt * t2 * p2.0 + t3 * p3.0;
    let y = mt3 * p0.1 + 3.0 * mt2 * t * p1.1 + 3.0 * mt * t2 * p2.1 + t3 * p3.1;

    // First derivative
    let dx = 3.0 * mt2 * (p1.0 - p0.0) + 6.0 * mt * t * (p2.0 - p1.0) + 3.0 * t2 * (p3.0 - p2.0);
    let dy = 3.0 * mt2 * (p1.1 - p0.1) + 6.0 * mt * t * (p2.1 - p1.1) + 3.0 * t2 * (p3.1 - p2.1);

    // Second derivative
    let ddx = 6.0 * mt * (p2.0 - 2.0 * p1.0 + p0.0) + 6.0 * t * (p3.0 - 2.0 * p2.0 + p1.0);
    let ddy = 6.0 * mt * (p2.1 - 2.0 * p1.1 + p0.1) + 6.0 * t * (p3.1 - 2.0 * p2.1 + p1.1);

    // Curvature
    let curvature = (dx * ddy - dy * ddx) / (dx * dx + dy * dy).powf(1.5);

    (x, y, curvature)
}

#[allow(dead_code)]
fn b_spline_basis(i: usize, p: usize, u: f64, knots: &[f64]) -> f64 {
    // B-spline basis function (Cox-de Boor recursion)
    if p == 0 {
        if i < knots.len() - 1 && u >= knots[i] && u < knots[i + 1] {
            1.0
        } else {
            0.0
        }
    } else {
        let mut left = 0.0;
        let mut right = 0.0;

        if i + p < knots.len() && (knots[i + p] - knots[i]).abs() > 1e-10 {
            left = (u - knots[i]) / (knots[i + p] - knots[i]) * b_spline_basis(i, p - 1, u, knots);
        }

        if i + 1 < knots.len()
            && i + p + 1 < knots.len()
            && (knots[i + p + 1] - knots[i + 1]).abs() > 1e-10
        {
            right = (knots[i + p + 1] - u) / (knots[i + p + 1] - knots[i + 1])
                * b_spline_basis(i + 1, p - 1, u, knots);
        }

        left + right
    }
}

#[allow(dead_code)]
fn cosine_weighted_hemisphere_sample(u1: f64, u2: f64) -> (f64, f64, f64) {
    // Cosine-weighted hemisphere sampling for ray tracing
    let theta = (u1).sqrt().acos();
    let phi = 2.0 * PI * u2;
    let pdf = theta.cos() / PI;
    (theta, phi, pdf)
}

#[allow(dead_code)]
fn bounce_easing(t: f64) -> f64 {
    // Bounce easing function
    if t < 1.0 / 2.75 {
        7.5625 * t * t
    } else if t < 2.0 / 2.75 {
        let t2 = t - 1.5 / 2.75;
        7.5625 * t2 * t2 + 0.75
    } else if t < 2.5 / 2.75 {
        let t2 = t - 2.25 / 2.75;
        7.5625 * t2 * t2 + 0.9375
    } else {
        let t2 = t - 2.625 / 2.75;
        7.5625 * t2 * t2 + 0.984375
    }
}

#[allow(dead_code)]
fn black_scholes_option(
    s: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
) -> Result<(f64, f64, f64, f64), Box<dyn std::error::Error>> {
    // Black-Scholes option pricing
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    let nd1 = 0.5 * (1.0 + erf(d1 / 2.0_f64.sqrt()));
    let nd2 = 0.5 * (1.0 + erf(d2 / 2.0_f64.sqrt()));

    let nmd1 = 1.0 - nd1;
    let nmd2 = 1.0 - nd2;

    let call_price = s * nd1 - k * (-r * t).exp() * nd2;
    let put_price = k * (-r * t).exp() * nmd2 - s * nmd1;

    let call_delta = nd1;
    let put_delta = nd1 - 1.0;

    Ok((call_price, put_price, call_delta, put_delta))
}

#[allow(dead_code)]
fn merton_default_probability(
    v: f64,
    d: f64,
    r: f64,
    sigma: f64,
    t: f64,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    // Merton model default probability
    let distance_to_default = ((v / d).ln() + (r - 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let default_prob = 0.5 * (1.0 - erf(distance_to_default / 2.0_f64.sqrt()));
    Ok((default_prob, distance_to_default))
}

#[allow(dead_code)]
fn vasicek_bond_pricing(r: f64, a: f64, b: f64, sigma: f64, t: f64) -> (f64, f64, f64) {
    // Vasicek model bond pricing
    let bt = (1.0 - (-a * t).exp()) / a;
    let at =
        (bt - t) * (a * a * b - 0.5 * sigma * sigma) / (a * a) - 0.25 * sigma * sigma * bt * bt / a;

    let bond_price = (at - r * bt).exp();
    let yield_rate = -at / t + r * bt / t;
    let duration = bt;

    (bond_price, yield_rate, duration)
}

#[allow(dead_code)]
fn efficient_frontier_point(
    target_return: f64,
    returns: &[f64],
    vols: &[f64],
    corr: &[Vec<f64>],
    rf: f64,
) -> (f64, f64) {
    // Simplified efficient frontier calculation
    // In practice, this would use quadratic programming
    let n = returns.len();
    let mut min_risk = f64::INFINITY;

    // Grid search over possible weights (simplified)
    for i in 0..=100 {
        for j in 0..=(100 - i) {
            let w1 = i as f64 / 100.0;
            let w2 = j as f64 / 100.0;
            let w3 = 1.0 - w1 - w2;

            if w3 >= 0.0 {
                let weights = [w1, w2, w3];

                // Calculate portfolio _return
                let port_return: f64 = weights.iter().zip(returns.iter()).map(|(w, r)| w * r).sum();

                if (port_return - target_return).abs() < 0.001 {
                    // Calculate portfolio risk
                    let mut risk_squared = 0.0;
                    for i in 0..n {
                        for j in 0..n {
                            risk_squared +=
                                weights[i] * weights[j] * vols[i] * vols[j] * corr[i][j];
                        }
                    }
                    let risk = risk_squared.sqrt();
                    min_risk = min_risk.min(risk);
                }
            }
        }
    }

    let sharpe_ratio = (target_return - rf) / min_risk;
    (min_risk, sharpe_ratio)
}

#[allow(dead_code)]
fn asian_option_monte_carlo(
    s: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    steps: usize,
    simulations: usize,
) -> (f64, f64, f64) {
    // Monte Carlo pricing for Asian options
    let dt = t / steps as f64;
    let mut call_payoffs = Vec::new();
    let mut put_payoffs = Vec::new();

    for _ in 0..simulations {
        let mut price_sum = 0.0;
        let mut price = s;

        for _ in 0..steps {
            // Simple random walk (should use proper random number generation)
            let z = 0.0; // Would be random normal in real implementation
            price *= (r * dt + sigma * dt.sqrt() * z).exp();
            price_sum += price;
        }

        let average_price = price_sum / steps as f64;
        call_payoffs.push((average_price - k).max(0.0) * (-r * t).exp());
        put_payoffs.push((k - average_price).max(0.0) * (-r * t).exp());
    }

    let call_price: f64 = call_payoffs.iter().sum::<f64>() / simulations as f64;
    let put_price: f64 = put_payoffs.iter().sum::<f64>() / simulations as f64;

    // Standard error calculation
    let call_variance: f64 = call_payoffs
        .iter()
        .map(|x| (x - call_price).powi(2))
        .sum::<f64>()
        / simulations as f64;
    let std_error = (call_variance / simulations as f64).sqrt();

    (call_price, put_price, std_error)
}
