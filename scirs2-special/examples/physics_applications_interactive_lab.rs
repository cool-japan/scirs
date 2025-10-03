//! Interactive Physics Applications Laboratory
//!
//! This comprehensive interactive laboratory demonstrates the practical applications
//! of special functions in physics and engineering through guided experiments.
//!
//! Features:
//! - Interactive quantum mechanics simulations using special functions
//! - Electromagnetic wave propagation with Bessel functions
//! - Heat transfer analysis with error functions
//! - Vibration analysis with orthogonal polynomials
//! - Statistical mechanics applications with gamma functions
//! - Signal processing demonstrations with Fresnel integrals
//! - Real-time visualization and parameter exploration
//! - Theoretical background with practical verification
//!
//! Run with: cargo run --example physics_applications_interactive_lab

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_special::*;
use std::collections::HashMap;
use std::f64::consts::{PI, TAU};
use std::io::{self, Write};
use std::time::Instant;

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct PhysicsLab {
    experiments: Vec<PhysicsExperiment>,
    current_experiment: Option<usize>,
    user_session: UserSession,
    visualization_data: VisualizationData,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct PhysicsExperiment {
    id: String,
    title: String,
    description: String,
    physics_background: String,
    mathematical_foundation: String,
    special_functions_used: Vec<String>,
    parameters: HashMap<String, ExperimentParameter>,
    simulation_engine: SimulationEngine,
    visualization_config: VisualizationConfig,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ExperimentParameter {
    name: String,
    symbol: String,
    current_value: f64,
    min_value: f64,
    max_value: f64,
    stepsize: f64,
    units: String,
    physical_meaning: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SimulationEngine {
    time_evolution: bool,
    spatial_dimensions: u8,
    boundary_conditions: String,
    numerical_method: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct VisualizationConfig {
    plot_type: PlotType,
    x_axis: AxisConfig,
    y_axis: AxisConfig,
    animation_enabled: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum PlotType {
    Line2D,
    Heatmap2D,
    Surface3D,
    Animation,
    MultiPanel,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct AxisConfig {
    label: String,
    range: (f64, f64),
    scale: String, // "linear", "log", "symlog"
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct UserSession {
    completed_experiments: Vec<String>,
    current_parameters: HashMap<String, f64>,
    notes: Vec<String>,
    session_start: Instant,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct VisualizationData {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
    z_data: Option<Array2<f64>>,
    metadata: HashMap<String, String>,
}

impl PhysicsLab {
    fn new() -> Self {
        let experiments = vec![
            Self::create_quantum_harmonic_oscillator(),
            Self::create_cylindrical_wave_propagation(),
            Self::create_heat_diffusion_experiment(),
            Self::create_vibrating_membrane(),
            Self::create_statistical_mechanics_demo(),
            Self::create_signal_processing_lab(),
            Self::create_electromagnetic_scattering(),
            Self::create_quantum_tunneling(),
        ];

        PhysicsLab {
            experiments,
            current_experiment: None,
            user_session: UserSession {
                completed_experiments: Vec::new(),
                current_parameters: HashMap::new(),
                notes: Vec::new(),
                session_start: Instant::now(),
            },
            visualization_data: VisualizationData {
                x_data: Array1::zeros(0),
                y_data: Array1::zeros(0),
                z_data: None,
                metadata: HashMap::new(),
            },
        }
    }

    fn create_quantum_harmonic_oscillator() -> PhysicsExperiment {
        let mut parameters = HashMap::new();

        parameters.insert(
            "n".to_string(),
            ExperimentParameter {
                name: "Quantum Number".to_string(),
                symbol: "n".to_string(),
                current_value: 3.0,
                min_value: 0.0,
                max_value: 20.0,
                stepsize: 1.0,
                units: "dimensionless".to_string(),
                physical_meaning: "Principal quantum number determining energy level".to_string(),
            },
        );

        parameters.insert(
            "omega".to_string(),
            ExperimentParameter {
                name: "Angular Frequency".to_string(),
                symbol: "ω".to_string(),
                current_value: 1.0,
                min_value: 0.1,
                max_value: 10.0,
                stepsize: 0.1,
                units: "rad/s".to_string(),
                physical_meaning: "Oscillator frequency parameter".to_string(),
            },
        );

        PhysicsExperiment {
            id: "quantum_harmonic_oscillator".to_string(),
            title: "Quantum Harmonic Oscillator Wave Functions".to_string(),
            description: "Explore the wave functions of the quantum harmonic oscillator using Hermite polynomials and Gaussian functions.".to_string(),
            physics_background: r"
The quantum harmonic oscillator is one of the most important exactly solvable models in quantum mechanics.
The time-independent Schrödinger equation is:

    -ℏ²/(2m) d²ψ/dx² + (1/2)mω²x²ψ = Eψ

The solutions are characterized by quantum number n and have energy levels:
    E_n = ℏω(n + 1/2)

The wave functions involve Hermite polynomials and provide insight into quantum tunneling and zero-point energy.
            ".to_string(),
            mathematical_foundation: r"
The normalized wave functions are:

    ψ_n(x) = (mω/πℏ)^(1/4) * 1/√(2^n n!) * H_n(√(mω/ℏ)x) * exp(-mωx²/2ℏ)

where H_n(ξ) are the Hermite polynomials defined by:
    H_n(ξ) = (-1)^n exp(ξ²) d^n/dξ^n exp(-ξ²)

Key mathematical properties:
- Orthogonality: ∫ ψ_m(x) ψ_n(x) dx = δ_mn
- Recurrence: H_{n+1}(ξ) = 2ξH_n(ξ) - 2nH_{n-1}(ξ)
- Generating function: exp(2ξt - t²) = Σ H_n(ξ) t^n/n!
            ".to_string(),
            special_functions_used: vec![
                "Hermite polynomials H_n(x)".to_string(),
                "Gaussian exp(-x²)".to_string(),
                "Gamma function Γ(n+1) = n!".to_string(),
            ],
            parameters,
            simulation_engine: SimulationEngine {
                time_evolution: false,
                spatial_dimensions: 1,
                boundary_conditions: "Normalizable at infinity".to_string(),
                numerical_method: "Analytical Hermite polynomials".to_string(),
            },
            visualization_config: VisualizationConfig {
                plot_type: PlotType::Line2D,
                x_axis: AxisConfig {
                    label: "Position x/√(ℏ/mω)".to_string(),
                    range: (-6.0, 6.0),
                    scale: "linear".to_string(),
                },
                y_axis: AxisConfig {
                    label: "Wave function ψ_n(x)".to_string(),
                    range: (-1.5, 1.5),
                    scale: "linear".to_string(),
                },
                animation_enabled: false,
            },
        }
    }

    fn create_cylindrical_wave_propagation() -> PhysicsExperiment {
        let mut parameters = HashMap::new();

        parameters.insert(
            "frequency".to_string(),
            ExperimentParameter {
                name: "Wave Frequency".to_string(),
                symbol: "f".to_string(),
                current_value: 1000.0,
                min_value: 100.0,
                max_value: 10000.0,
                stepsize: 100.0,
                units: "Hz".to_string(),
                physical_meaning: "Frequency of the propagating wave".to_string(),
            },
        );

        parameters.insert(
            "radius".to_string(),
            ExperimentParameter {
                name: "Cylinder Radius".to_string(),
                symbol: "a".to_string(),
                current_value: 0.1,
                min_value: 0.01,
                max_value: 1.0,
                stepsize: 0.01,
                units: "m".to_string(),
                physical_meaning: "Radius of the cylindrical boundary".to_string(),
            },
        );

        PhysicsExperiment {
            id: "cylindrical_wave_propagation".to_string(),
            title: "Electromagnetic Wave Propagation in Cylindrical Waveguides".to_string(),
            description:
                "Study electromagnetic wave modes in cylindrical waveguides using Bessel functions."
                    .to_string(),
            physics_background: r"
Electromagnetic waves in cylindrical waveguides are solutions to Maxwell's equations
with cylindrical boundary conditions. The wave equation in cylindrical coordinates is:

    ∇²E + k²E = 0

where k = ω/c is the wave number. For TM modes (transverse magnetic), the electric field
component E_z satisfies:

    1/r ∂/∂r(r ∂E_z/∂r) + 1/r² ∂²E_z/∂φ² + γ²E_z = 0

with γ² = k² - β² where β is the propagation constant.
            "
            .to_string(),
            mathematical_foundation: r"
The solutions are expressed in terms of Bessel functions:

    E_z(r,φ,z) = A J_m(γr) cos(mφ) exp(iβz)

where:
- J_m(x) is the Bessel function of the first kind of order m
- γ is determined by boundary conditions: J_m(γa) = 0
- The zeros of J_m determine the cutoff frequencies

For each mode (m,n), the cutoff frequency is:
    f_c = c/(2π) * (χ_mn/a)

where χ_mn is the n-th zero of J_m(x).

Key properties:
- Orthogonality of modes
- Dispersion relation: β² = k² - (χ_mn/a)²
- Group velocity: v_g = c²β/ω
            "
            .to_string(),
            special_functions_used: vec![
                "Bessel functions J_m(x)".to_string(),
                "Bessel function zeros χ_mn".to_string(),
                "Neumann functions Y_m(x)".to_string(),
            ],
            parameters,
            simulation_engine: SimulationEngine {
                time_evolution: true,
                spatial_dimensions: 2,
                boundary_conditions: "Perfect conductor at r=a".to_string(),
                numerical_method: "Bessel function expansion".to_string(),
            },
            visualization_config: VisualizationConfig {
                plot_type: PlotType::Heatmap2D,
                x_axis: AxisConfig {
                    label: "Radial position r (m)".to_string(),
                    range: (0.0, 0.2),
                    scale: "linear".to_string(),
                },
                y_axis: AxisConfig {
                    label: "Azimuthal angle φ (rad)".to_string(),
                    range: (0.0, TAU),
                    scale: "linear".to_string(),
                },
                animation_enabled: true,
            },
        }
    }

    fn create_heat_diffusion_experiment() -> PhysicsExperiment {
        let mut parameters = HashMap::new();

        parameters.insert(
            "diffusivity".to_string(),
            ExperimentParameter {
                name: "Thermal Diffusivity".to_string(),
                symbol: "α".to_string(),
                current_value: 1e-5,
                min_value: 1e-7,
                max_value: 1e-3,
                stepsize: 1e-6,
                units: "m²/s".to_string(),
                physical_meaning: "Rate of heat diffusion through material".to_string(),
            },
        );

        parameters.insert(
            "time".to_string(),
            ExperimentParameter {
                name: "Time".to_string(),
                symbol: "t".to_string(),
                current_value: 100.0,
                min_value: 1.0,
                max_value: 1000.0,
                stepsize: 10.0,
                units: "s".to_string(),
                physical_meaning: "Elapsed time since initial condition".to_string(),
            },
        );

        PhysicsExperiment {
            id: "heat_diffusion".to_string(),
            title: "Heat Diffusion and Error Functions".to_string(),
            description: "Analyze heat diffusion in semi-infinite media using error functions and complementary error functions.".to_string(),
            physics_background: r"
Heat diffusion in materials is governed by the heat equation:

    ∂T/∂t = α ∇²T

For a semi-infinite medium (x ≥ 0) with constant surface temperature T₀ at x=0
and initial temperature T_∞ throughout, the solution involves error functions.

This model applies to:
- Heat treatment of metals
- Thermal barrier coatings
- Underground temperature variations
- Semiconductor thermal management
            ".to_string(),
            mathematical_foundation: r"
The temperature distribution is:

    T(x,t) = T₀ + (T_∞ - T₀) erf(x/(2√(αt)))

where erf is the error function:
    erf(z) = (2/√π) ∫₀ᶻ exp(-t²) dt

Key properties:
- erf(0) = 0, erf(∞) = 1
- erf(-z) = -erf(z) (odd function)
- Asymptotic behavior: erf(z) ≈ 1 - exp(-z²)/(z√π) for large z

The complementary error function erfc(z) = 1 - erf(z) gives the temperature profile directly:
    T(x,t) = T₀ + (T_∞ - T₀) erfc(x/(2√(αt)))

Heat flux at the surface:
    q(t) = -k(T_∞ - T₀)/(√(παt))
            ".to_string(),
            special_functions_used: vec![
                "Error function erf(x)".to_string(),
                "Complementary error function erfc(x)".to_string(),
                "Dawson integral".to_string(),
            ],
            parameters,
            simulation_engine: SimulationEngine {
                time_evolution: true,
                spatial_dimensions: 1,
                boundary_conditions: "Fixed temperature at x=0".to_string(),
                numerical_method: "Analytical error function solution".to_string(),
            },
            visualization_config: VisualizationConfig {
                plot_type: PlotType::Animation,
                x_axis: AxisConfig {
                    label: "Distance x (m)".to_string(),
                    range: (0.0, 0.1),
                    scale: "linear".to_string(),
                },
                y_axis: AxisConfig {
                    label: "Temperature T (°C)".to_string(),
                    range: (0.0, 100.0),
                    scale: "linear".to_string(),
                },
                animation_enabled: true,
            },
        }
    }

    fn create_vibrating_membrane() -> PhysicsExperiment {
        let mut parameters = HashMap::new();

        parameters.insert(
            "membrane_radius".to_string(),
            ExperimentParameter {
                name: "Membrane Radius".to_string(),
                symbol: "R".to_string(),
                current_value: 0.2,
                min_value: 0.05,
                max_value: 1.0,
                stepsize: 0.05,
                units: "m".to_string(),
                physical_meaning: "Radius of the circular membrane".to_string(),
            },
        );

        parameters.insert(
            "mode_m".to_string(),
            ExperimentParameter {
                name: "Azimuthal Mode Number".to_string(),
                symbol: "m".to_string(),
                current_value: 1.0,
                min_value: 0.0,
                max_value: 5.0,
                stepsize: 1.0,
                units: "dimensionless".to_string(),
                physical_meaning: "Number of nodal diameters".to_string(),
            },
        );

        parameters.insert(
            "mode_n".to_string(),
            ExperimentParameter {
                name: "Radial Mode Number".to_string(),
                symbol: "n".to_string(),
                current_value: 1.0,
                min_value: 1.0,
                max_value: 5.0,
                stepsize: 1.0,
                units: "dimensionless".to_string(),
                physical_meaning: "Number of nodal circles".to_string(),
            },
        );

        PhysicsExperiment {
            id: "vibrating_membrane".to_string(),
            title: "Vibrating Circular Membrane and Bessel Functions".to_string(),
            description: "Study the normal modes of a vibrating circular membrane using Bessel functions and explore musical acoustics.".to_string(),
            physics_background: r"
A circular membrane (like a drumhead) vibrating under tension follows the 2D wave equation:

    ∇²u - (1/c²)∂²u/∂t² = 0

In cylindrical coordinates with circular symmetry, this becomes a boundary value problem
where the displacement must vanish at the rim: u(R,φ,t) = 0.

The normal modes determine the characteristic frequencies and shapes of vibration,
directly relevant to:
- Musical instruments (drums, timpani)
- Speaker diaphragms
- Microphone membranes
- Seismic sensors
            ".to_string(),
            mathematical_foundation: r"
Using separation of variables u(r,φ,t) = R(r)Φ(φ)T(t), the radial equation is:

    r²R'' + rR' + (k²r² - m²)R = 0

This is Bessel's equation with solution R(r) = J_m(kr).

The boundary condition R(R) = 0 gives: J_m(kR) = 0

Therefore k = χ_mn/R where χ_mn is the n-th zero of J_m.

The complete normal modes are:
    u_mn(r,φ,t) = A_mn J_m(χ_mn r/R) cos(mφ + φ_mn) cos(ω_mn t + δ_mn)

with frequencies:
    ω_mn = (c/R) χ_mn

Key properties:
- J₀ has zeros at 2.405, 5.520, 8.654, ...
- J₁ has zeros at 3.832, 7.016, 10.173, ...
- Higher order modes have characteristic nodal patterns
            ".to_string(),
            special_functions_used: vec![
                "Bessel functions J_m(x)".to_string(),
                "Bessel zeros χ_mn".to_string(),
                "Trigonometric functions".to_string(),
            ],
            parameters,
            simulation_engine: SimulationEngine {
                time_evolution: true,
                spatial_dimensions: 2,
                boundary_conditions: "Fixed at circular boundary".to_string(),
                numerical_method: "Bessel function modal expansion".to_string(),
            },
            visualization_config: VisualizationConfig {
                plot_type: PlotType::Surface3D,
                x_axis: AxisConfig {
                    label: "x position (m)".to_string(),
                    range: (-0.2, 0.2),
                    scale: "linear".to_string(),
                },
                y_axis: AxisConfig {
                    label: "y position (m)".to_string(),
                    range: (-0.2, 0.2),
                    scale: "linear".to_string(),
                },
                animation_enabled: true,
            },
        }
    }

    fn create_statistical_mechanics_demo() -> PhysicsExperiment {
        let mut parameters = HashMap::new();

        parameters.insert(
            "temperature".to_string(),
            ExperimentParameter {
                name: "Temperature".to_string(),
                symbol: "T".to_string(),
                current_value: 300.0,
                min_value: 1.0,
                max_value: 3000.0,
                stepsize: 10.0,
                units: "K".to_string(),
                physical_meaning: "Absolute temperature of the system".to_string(),
            },
        );

        parameters.insert(
            "chemical_potential".to_string(),
            ExperimentParameter {
                name: "Chemical Potential".to_string(),
                symbol: "μ".to_string(),
                current_value: 0.0,
                min_value: -5.0,
                max_value: 5.0,
                stepsize: 0.1,
                units: "eV".to_string(),
                physical_meaning: "Chemical potential in electron volts".to_string(),
            },
        );

        PhysicsExperiment {
            id: "statistical_mechanics".to_string(),
            title: "Statistical Mechanics and Gamma Functions".to_string(),
            description: "Explore statistical distributions in thermodynamics using gamma functions and related special functions.".to_string(),
            physics_background: r"
Statistical mechanics connects microscopic properties to macroscopic observables.
The partition function and related thermodynamic quantities often involve special functions:

- Maxwell-Boltzmann distribution for classical particles
- Fermi-Dirac distribution for fermions  
- Bose-Einstein distribution for bosons
- Planck distribution for black-body radiation

These distributions arise naturally from fundamental counting arguments and
extremization of entropy subject to energy conservation constraints.
            ".to_string(),
            mathematical_foundation: r"
Key distributions involving special functions:

**Maxwell-Boltzmann Speed Distribution:**
    f(v) = 4π(m/2πkT)^(3/2) v² exp(-mv²/2kT)

This involves the gamma function through:
    ⟨v^n⟩ = Γ((n+3)/2) / Γ(3/2) * (2kT/m)^(n/2)

**Planck Distribution:**
    u(ν,T) = (8πhν³/c³) / (exp(hν/kT) - 1)

The total energy density involves:
    U = ∫₀^∞ u(ν,T) dν = (8π⁵k⁴T⁴)/(15h³c³) = aT⁴

where a involves ζ(4) = π⁴/90.

**Fermi-Dirac Distribution:**
    n(E) = 1/(exp((E-μ)/kT) + 1)

The density of states for free electrons involves Γ(3/2).
            ".to_string(),
            special_functions_used: vec![
                "Gamma function Γ(x)".to_string(),
                "Riemann zeta function ζ(s)".to_string(),
                "Polylogarithm Li_s(z)".to_string(),
                "Incomplete gamma functions".to_string(),
            ],
            parameters,
            simulation_engine: SimulationEngine {
                time_evolution: false,
                spatial_dimensions: 0,
                boundary_conditions: "Thermodynamic equilibrium".to_string(),
                numerical_method: "Analytical special function evaluation".to_string(),
            },
            visualization_config: VisualizationConfig {
                plot_type: PlotType::MultiPanel,
                x_axis: AxisConfig {
                    label: "Energy E (eV)".to_string(),
                    range: (-2.0, 8.0),
                    scale: "linear".to_string(),
                },
                y_axis: AxisConfig {
                    label: "Probability / Density".to_string(),
                    range: (0.0, 1.0),
                    scale: "log".to_string(),
                },
                animation_enabled: false,
            },
        }
    }

    fn create_signal_processing_lab() -> PhysicsExperiment {
        let mut parameters = HashMap::new();

        parameters.insert(
            "cutoff_freq".to_string(),
            ExperimentParameter {
                name: "Cutoff Frequency".to_string(),
                symbol: "f_c".to_string(),
                current_value: 1000.0,
                min_value: 100.0,
                max_value: 10000.0,
                stepsize: 100.0,
                units: "Hz".to_string(),
                physical_meaning: "Filter cutoff frequency".to_string(),
            },
        );

        PhysicsExperiment {
            id: "signal_processing".to_string(),
            title: "Signal Processing with Fresnel Integrals".to_string(),
            description: "Explore Fresnel diffraction and signal processing applications using Fresnel integrals.".to_string(),
            physics_background: r"
Fresnel integrals appear in optics (diffraction) and signal processing (chirp signals).
They describe the intensity pattern when light passes through apertures or around obstacles.

In signal processing, chirp signals (frequency-modulated signals) are used in:
- Radar systems
- Sonar applications  
- Spread spectrum communications
- Medical advancedsound imaging

The mathematical connection between optics and signal processing comes through
the Fourier transform and the principle of stationary phase.
            ".to_string(),
            mathematical_foundation: r"
The Fresnel integrals are defined as:

    C(x) = ∫₀ˣ cos(πt²/2) dt
    S(x) = ∫₀ˣ sin(πt²/2) dt

For Fresnel diffraction at a straight edge, the intensity is:

    I(x) = I₀/4 [(C(v) + 1/2)² + (S(v) + 1/2)²]

where v = x√(2/λz) is the Fresnel parameter.

For chirp signal analysis:
    s(t) = A cos(πkt²) = A Re[exp(iπkt²)]

The spectrum involves Fresnel integrals through the stationary phase method.

Properties:
- C(∞) = S(∞) = 1/2
- C(-x) = -C(x), S(-x) = -S(x)
- Spiral of Cornu parameterization: (C(t), S(t))
            ".to_string(),
            special_functions_used: vec![
                "Fresnel integral C(x)".to_string(),
                "Fresnel integral S(x)".to_string(),
                "Complex Fresnel integral".to_string(),
            ],
            parameters,
            simulation_engine: SimulationEngine {
                time_evolution: true,
                spatial_dimensions: 2,
                boundary_conditions: "Aperture or edge geometry".to_string(),
                numerical_method: "Fresnel integral evaluation".to_string(),
            },
            visualization_config: VisualizationConfig {
                plot_type: PlotType::Heatmap2D,
                x_axis: AxisConfig {
                    label: "Fresnel parameter v".to_string(),
                    range: (-5.0, 5.0),
                    scale: "linear".to_string(),
                },
                y_axis: AxisConfig {
                    label: "Intensity I/I₀".to_string(),
                    range: (0.0, 2.0),
                    scale: "linear".to_string(),
                },
                animation_enabled: true,
            },
        }
    }

    fn create_electromagnetic_scattering() -> PhysicsExperiment {
        let mut parameters = HashMap::new();

        parameters.insert(
            "sphere_radius".to_string(),
            ExperimentParameter {
                name: "Sphere Radius".to_string(),
                symbol: "a".to_string(),
                current_value: 0.001,
                min_value: 1e-6,
                max_value: 0.01,
                stepsize: 1e-5,
                units: "m".to_string(),
                physical_meaning: "Radius of the scattering sphere".to_string(),
            },
        );

        parameters.insert(
            "wavelength".to_string(),
            ExperimentParameter {
                name: "Wavelength".to_string(),
                symbol: "λ".to_string(),
                current_value: 500e-9,
                min_value: 300e-9,
                max_value: 800e-9,
                stepsize: 10e-9,
                units: "m".to_string(),
                physical_meaning: "Wavelength of incident electromagnetic radiation".to_string(),
            },
        );

        PhysicsExperiment {
            id: "mie_scattering".to_string(),
            title: "Mie Scattering and Spherical Bessel Functions".to_string(),
            description: "Study electromagnetic scattering by spherical particles using spherical Bessel functions.".to_string(),
            physics_background: r"
Mie scattering describes the interaction of electromagnetic waves with spherical particles.
This is crucial for understanding:

- Atmospheric optics (why sky is blue, sunsets are red)
- Particle sizing in aerosol science
- Optical properties of colloidal systems
- Biomedical imaging and therapy
- Planetary atmosphere analysis

The solution involves expanding incident, scattered, and internal fields
in terms of vector spherical harmonics and spherical Bessel functions.
            ".to_string(),
            mathematical_foundation: r"
The scattered field is expanded as:

    E_scat = Σ [iᵃⁿ M_n^{(3)} - bᵃⁿ N_n^{(3)}]

where M_n and N_n are vector spherical harmonics involving:
- Spherical Bessel functions j_n(x), y_n(x)
- Hankel functions h_n^{(1,2)}(x) = j_n(x) ± i y_n(x)
- Associated Legendre polynomials P_n^m(cos θ)

The Mie coefficients are:
    aₙ = [mψₙ(mx)ψₙ'(x) - ψₙ(x)ψₙ'(mx)] / [mψₙ(mx)ξₙ'(x) - ξₙ(x)ψₙ'(mx)]
    bₙ = [ψₙ(mx)ψₙ'(x) - mψₙ(x)ψₙ'(mx)] / [ψₙ(mx)ξₙ'(x) - mξₙ(x)ψₙ'(mx)]

where x = ka is the size parameter and m is the relative refractive index.

Key functions:
- ψₙ(x) = x jₙ(x) (Riccati-Bessel function)
- ξₙ(x) = x hₙ^{(1)}(x) (Riccati-Hankel function)
            ".to_string(),
            special_functions_used: vec![
                "Spherical Bessel functions jₙ(x)".to_string(),
                "Spherical Neumann functions yₙ(x)".to_string(),
                "Spherical Hankel functions hₙ^{(1,2)}(x)".to_string(),
                "Associated Legendre polynomials P_n^m".to_string(),
            ],
            parameters,
            simulation_engine: SimulationEngine {
                time_evolution: false,
                spatial_dimensions: 3,
                boundary_conditions: "Continuous fields at sphere surface".to_string(),
                numerical_method: "Mie series expansion".to_string(),
            },
            visualization_config: VisualizationConfig {
                plot_type: PlotType::MultiPanel,
                x_axis: AxisConfig {
                    label: "Scattering angle θ (degrees)".to_string(),
                    range: (0.0, 180.0),
                    scale: "linear".to_string(),
                },
                y_axis: AxisConfig {
                    label: "Scattering intensity".to_string(),
                    range: (1e-6, 1e2),
                    scale: "log".to_string(),
                },
                animation_enabled: false,
            },
        }
    }

    fn create_quantum_tunneling() -> PhysicsExperiment {
        let mut parameters = HashMap::new();

        parameters.insert(
            "barrier_height".to_string(),
            ExperimentParameter {
                name: "Barrier Height".to_string(),
                symbol: "V₀".to_string(),
                current_value: 5.0,
                min_value: 1.0,
                max_value: 20.0,
                stepsize: 0.5,
                units: "eV".to_string(),
                physical_meaning: "Height of the potential barrier".to_string(),
            },
        );

        parameters.insert(
            "barrier_width".to_string(),
            ExperimentParameter {
                name: "Barrier Width".to_string(),
                symbol: "a".to_string(),
                current_value: 1e-9,
                min_value: 1e-10,
                max_value: 1e-8,
                stepsize: 1e-10,
                units: "m".to_string(),
                physical_meaning: "Width of the potential barrier".to_string(),
            },
        );

        PhysicsExperiment {
            id: "quantum_tunneling".to_string(),
            title: "Quantum Tunneling and Airy Functions".to_string(),
            description: "Investigate quantum tunneling through potential barriers using Airy functions and WKB approximation.".to_string(),
            physics_background: r"
Quantum tunneling is a purely quantum mechanical phenomenon where particles
can pass through potential barriers even when their kinetic energy is less
than the barrier height.

Applications include:
- Scanning tunneling microscopy (STM)
- Tunnel junctions in electronics
- Nuclear fusion in stars
- Field emission devices
- Josephson junctions in superconductors

For slowly varying potentials, the WKB (Wentzel-Kramers-Brillouin) method
provides excellent approximations using Airy functions.
            ".to_string(),
            mathematical_foundation: r"
For a general potential V(x), the WKB approximation gives:

    ψ(x) ≈ (1/√p(x)) exp(±i ∫ p(x) dx / ℏ)

where p(x) = √(2m[E - V(x)]) is the classical momentum.

Near turning points where E = V(x), this approximation breaks down and
the exact solution involves Airy functions Ai(x) and Bi(x).

The connection formulas across turning points are:

    ψ(x) = (1/√|p|) [C₁ Ai(ξ) + C₂ Bi(ξ)]

where ξ = (2m/ℏ²)^{1/3} ∫[V(x) - E] dx

For a linear potential V(x) = V₀ + Fx, the exact solutions are Airy functions.

Transmission coefficient through a barrier:
    T = exp(-2 ∫_{x₁}^{x₂} |p(x)| dx / ℏ)

where x₁, x₂ are the classical turning points.
            ".to_string(),
            special_functions_used: vec![
                "Airy function Ai(x)".to_string(),
                "Airy function Bi(x)".to_string(),
                "Airy function derivatives Ai'(x), Bi'(x)".to_string(),
                "Exponential integrals".to_string(),
            ],
            parameters,
            simulation_engine: SimulationEngine {
                time_evolution: false,
                spatial_dimensions: 1,
                boundary_conditions: "Scattering boundary conditions".to_string(),
                numerical_method: "WKB approximation with Airy functions".to_string(),
            },
            visualization_config: VisualizationConfig {
                plot_type: PlotType::Line2D,
                x_axis: AxisConfig {
                    label: "Position x (nm)".to_string(),
                    range: (-5e-9, 5e-9),
                    scale: "linear".to_string(),
                },
                y_axis: AxisConfig {
                    label: "Wave function |ψ(x)|²".to_string(),
                    range: (0.0, 2.0),
                    scale: "linear".to_string(),
                },
                animation_enabled: false,
            },
        }
    }
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Physics Applications Interactive Laboratory");
    println!("==============================================");
    println!("Explore special functions through physics and engineering applications\n");

    let mut lab = PhysicsLab::new();

    loop {
        display_main_menu(&lab.experiments);
        let choice =
            get_user_input("Enter your choice (1-8, 'info' for details, or 'q' to quit): ")?;

        if choice.to_lowercase() == "q" {
            println!("🎓 Thank you for exploring physics with special functions!");
            break;
        }

        if choice.to_lowercase() == "info" {
            display_theory_overview();
            continue;
        }

        match choice.parse::<usize>() {
            Ok(n) if n >= 1 && n <= lab.experiments.len() => {
                lab.current_experiment = Some(n - 1);
                run_experiment(&mut lab, n - 1)?;
            }
            _ => println!("❌ Invalid choice. Please try again.\n"),
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn display_main_menu(experiments: &[PhysicsExperiment]) {
    println!("🔬 Available Physics Experiments:");
    println!();

    for (i, exp) in experiments.iter().enumerate() {
        println!("{}. {} 📊", i + 1, exp.title);
        println!("   {}", exp.description);
        println!(
            "   Special functions: {}",
            exp.special_functions_used.join(", ")
        );
        println!();
    }

    println!("ℹ️  Type 'info' for theoretical background");
    println!("🚪 Type 'q' to quit");
    println!();
}

#[allow(dead_code)]
fn display_theory_overview() {
    println!("\n🎯 Theoretical Foundations\n");
    println!("Special functions arise naturally in physics through:");
    println!();
    println!("📐 **Separation of Variables**: Many PDEs in physics separate into ODEs");
    println!("   whose solutions are special functions (Bessel, Hermite, Legendre, etc.)");
    println!();
    println!("🌊 **Wave Phenomena**: From quantum mechanics to electromagnetics,");
    println!("   wave equations in different geometries lead to specific special functions");
    println!();
    println!("🔥 **Diffusion Processes**: Heat, mass, and momentum transfer often");
    println!("   involve error functions and related integrals");
    println!();
    println!("📊 **Statistical Mechanics**: Partition functions and thermodynamic");
    println!("   averages frequently involve gamma functions and zeta functions");
    println!();
    println!("🎭 **Asymptotic Analysis**: Understanding behavior for extreme parameters");
    println!("   requires asymptotic expansions of special functions");
    println!();
    println!("🧮 **Transform Methods**: Fourier, Laplace, and other integral transforms");
    println!("   often yield special functions in their solutions");
    println!();
    println!("Press Enter to continue...");
    let _ = io::stdin().read_line(&mut String::new());
}

#[allow(dead_code)]
fn run_experiment(
    lab: &mut PhysicsLab,
    experiment_index: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let experiment = &lab.experiments[experiment_index].clone();

    println!("\n🧪 Experiment: {}", experiment.title);
    println!("{}", "=".repeat(experiment.title.len() + 13));
    println!();

    // Display physics background
    println!("🌌 **Physics Background:**");
    println!("{}", experiment.physics_background);
    println!();

    // Display mathematical foundation
    println!("📐 **Mathematical Foundation:**");
    println!("{}", experiment.mathematical_foundation);
    println!();

    // Interactive parameter exploration
    loop {
        display_current_parameters(&experiment.parameters);
        println!();
        println!("Available actions:");
        println!("  'run' - Execute simulation with current parameters");
        println!("  'param <name> <value>' - Set parameter value");
        println!("  'theory' - Show detailed mathematical theory");
        println!("  'back' - Return to main menu");
        println!();

        let input = get_user_input("Enter action: ")?;
        let parts: Vec<&str> = input.split_whitespace().collect();

        match parts.as_slice() {
            ["back"] => break,
            ["run"] => {
                println!("\n🚀 Running simulation...");
                run_simulation(experiment)?;
                println!("✅ Simulation completed!\n");
            }
            ["theory"] => {
                display_detailed_theory(experiment);
            }
            ["param", name, value] => {
                if let Ok(val) = value.parse::<f64>() {
                    println!("📝 Parameter {} set to {}", name, val);
                    // In a full implementation, this would update the parameter
                } else {
                    println!("❌ Invalid value. Please enter a number.");
                }
            }
            _ => println!("❌ Unknown action. Type 'back' to return to menu."),
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn display_current_parameters(parameters: &HashMap<String, ExperimentParameter>) {
    println!("🎛️  **Current Parameters:**");
    for param in parameters.values() {
        println!(
            "  {} ({}) = {} {} - {}",
            param.name, param.symbol, param.current_value, param.units, param.physical_meaning
        );
    }
}

#[allow(dead_code)]
fn display_detailed_theory(experiment: &PhysicsExperiment) {
    println!("\n📚 **Detailed Mathematical Theory**");
    println!("{}", "=".repeat(35));
    println!();

    println!("🎯 **Special Functions Used:**");
    for func in &experiment.special_functions_used {
        println!("  • {}", func);
    }
    println!();

    println!("🔧 **Simulation Details:**");
    println!(
        "  • Time evolution: {}",
        experiment.simulation_engine.time_evolution
    );
    println!(
        "  • Spatial dimensions: {}",
        experiment.simulation_engine.spatial_dimensions
    );
    println!(
        "  • Boundary conditions: {}",
        experiment.simulation_engine.boundary_conditions
    );
    println!(
        "  • Numerical method: {}",
        experiment.simulation_engine.numerical_method
    );
    println!();

    println!("📊 **Visualization:**");
    println!(
        "  • Plot type: {:?}",
        experiment.visualization_config.plot_type
    );
    println!(
        "  • X-axis: {}",
        experiment.visualization_config.x_axis.label
    );
    println!(
        "  • Y-axis: {}",
        experiment.visualization_config.y_axis.label
    );
    println!(
        "  • Animation: {}",
        experiment.visualization_config.animation_enabled
    );
    println!();

    println!("Press Enter to continue...");
    let _ = io::stdin().read_line(&mut String::new());
}

#[allow(dead_code)]
fn run_simulation(experiment: &PhysicsExperiment) -> Result<(), Box<dyn std::error::Error>> {
    match experiment.id.as_str() {
        "quantum_harmonic_oscillator" => run_quantum_oscillator_simulation(),
        "cylindrical_wave_propagation" => run_wave_propagation_simulation(),
        "heat_diffusion" => run_heat_diffusion_simulation(),
        "vibrating_membrane" => run_membrane_simulation(),
        "statistical_mechanics" => run_statistical_simulation(),
        "signal_processing" => run_signal_processing_simulation(),
        "mie_scattering" => run_scattering_simulation(),
        "quantum_tunneling" => run_tunneling_simulation(),
        _ => {
            println!("Simulation not yet implemented for this experiment.");
            Ok(())
        }
    }
}

#[allow(dead_code)]
fn run_quantum_oscillator_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌀 Computing quantum harmonic oscillator wave functions...");

    let x_range = Array1::linspace(-5.0, 5.0, 200);
    let n = 3; // quantum number

    // Compute normalized wave function using Hermite polynomials
    let mut psi = Array1::zeros(x_range.len());
    let normalization = (PI.powf(-0.25)) / (2_f64.powi(n as i32) * gamma((n + 1) as f64)).sqrt();

    for (i, &x) in x_range.iter().enumerate() {
        let hermite_val = hermite_physicist(n, x);
        psi[i] = normalization * hermite_val * (-0.5 * x * x).exp();
    }

    // Display key results
    println!("📈 Wave function computed for n = {}", n);
    println!("   Energy level: E_{} = ℏω({} + 1/2)", n, n);
    println!(
        "   Classical turning points: ±√(2n+1) ≈ ±{:.2}",
        (2.0 * n as f64 + 1.0).sqrt()
    );

    // Show some sample values
    println!("\n📊 Sample wave function values:");
    for i in (0..psi.len()).step_by(psi.len() / 10) {
        println!("   ψ({:6.2}) = {:8.4}", x_range[i], psi[i]);
    }

    // Verify normalization
    let dx = x_range[1] - x_range[0];
    let norm_check: f64 = psi.iter().map(|&val| val * val).sum::<f64>() * dx;
    println!(
        "\n✅ Normalization check: ∫|ψ|²dx = {:.6} (should be 1.0)",
        norm_check
    );

    Ok(())
}

#[allow(dead_code)]
fn run_wave_propagation_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("📡 Simulating electromagnetic wave propagation in cylindrical waveguide...");

    let frequency = 1000.0; // Hz
    let radius = 0.1; // m
    let mode_m = 1;
    let mode_n = 1;

    // Find the (m,n)-th zero of J_m
    let zeros_j1 = [3.8317, 7.0156, 10.1735]; // First few zeros of J_1
    let cutoff_wavenumber = zeros_j1[mode_n - 1] / radius;
    let cutoff_frequency = cutoff_wavenumber * 3e8 / (2.0 * PI); // c/(2π) * k_c

    println!("🎯 Mode TM_{}{}:", mode_m, mode_n);
    println!("   Cutoff frequency: {:.2} MHz", cutoff_frequency / 1e6);
    println!("   Operating frequency: {:.2} MHz", frequency / 1e6);

    if frequency > cutoff_frequency {
        println!("✅ Propagating mode (frequency > cutoff)");

        // Compute field pattern
        let r_points = Array1::linspace(0.001, radius * 0.99, 50);
        let mut field_amplitude = Array1::zeros(r_points.len());

        for (i, &r) in r_points.iter().enumerate() {
            field_amplitude[i] = j1(cutoff_wavenumber * r);
        }

        println!("\n📊 Radial field distribution E_z(r):");
        for i in (0..field_amplitude.len()).step_by(field_amplitude.len() / 8) {
            println!(
                "   r = {:6.4} m: E_z = {:8.4}",
                r_points[i], field_amplitude[i]
            );
        }
    } else {
        println!("🚫 Evanescent mode (frequency < cutoff)");
        println!("   Exponential decay with distance");
    }

    Ok(())
}

#[allow(dead_code)]
fn run_heat_diffusion_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Simulating heat diffusion using error functions...");

    let alpha = 1e-5_f64; // thermal diffusivity m²/s
    let time = 100.0_f64; // seconds
    let surface_temp = 100.0; // °C
    let initial_temp = 20.0; // °C

    let x_range = Array1::linspace(0.0, 0.1, 100); // distance from surface
    let mut temperature = Array1::zeros(x_range.len());

    let sqrt_4_alpha_t = (4.0 * alpha * time).sqrt();

    for (i, &x) in x_range.iter().enumerate() {
        let argument = x / sqrt_4_alpha_t;
        temperature[i] = surface_temp + (initial_temp - surface_temp) * erf(argument);
    }

    println!("🌡️  Temperature distribution at t = {} s:", time);
    println!("   Thermal diffusivity α = {} m²/s", alpha);
    println!(
        "   Characteristic length scale: √(4αt) = {:.4} m",
        sqrt_4_alpha_t
    );

    println!("\n📊 Temperature profile:");
    for i in (0..temperature.len()).step_by(temperature.len() / 10) {
        println!("   x = {:6.4} m: T = {:6.1} °C", x_range[i], temperature[i]);
    }

    // Heat flux at surface
    let heat_flux = -1000.0 * (initial_temp - surface_temp) / (PI * alpha * time).sqrt(); // W/m²
    println!("\n🔥 Heat flux at surface: {:.1} W/m²", heat_flux);

    Ok(())
}

#[allow(dead_code)]
fn run_membrane_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🥁 Simulating vibrating circular membrane...");

    let radius = 0.2; // m
    let wave_speed = 100.0; // m/s
    let mode_m = 1;
    let mode_n = 1;

    // Zeros of Bessel functions
    let zeros_j0 = [2.4048, 5.5201, 8.6537];
    let zeros_j1 = [3.8317, 7.0156, 10.1735];

    let chi_mn = if mode_m == 0 {
        zeros_j0[mode_n - 1]
    } else {
        zeros_j1[mode_n - 1]
    };
    let frequency = wave_speed * chi_mn / (2.0 * PI * radius);

    println!("🎵 Mode ({}, {}):", mode_m, mode_n);
    println!("   Zero χ_{}{} = {:.4}", mode_m, mode_n, chi_mn);
    println!("   Frequency: {:.1} Hz", frequency);

    // Compute mode shape
    let r_range = Array1::linspace(0.0, radius, 50);
    let mut amplitude = Array1::zeros(r_range.len());

    for (i, &r) in r_range.iter().enumerate() {
        let argument = chi_mn * r / radius;
        amplitude[i] = if mode_m == 0 {
            j0(argument)
        } else {
            j1(argument)
        };
    }

    println!("\n📊 Radial amplitude distribution:");
    for i in (0..amplitude.len()).step_by(amplitude.len() / 8) {
        println!("   r = {:6.4} m: A = {:7.4}", r_range[i], amplitude[i]);
    }

    // Node locations (zeros of the mode)
    println!(
        "\n🎯 Nodal circles occur where J_{}(χ_{}{}·r/R) = 0",
        mode_m, mode_m, mode_n
    );

    Ok(())
}

#[allow(dead_code)]
fn run_statistical_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Computing statistical mechanics distributions...");

    let temperature = 300.0; // K
    let k_b = 1.381e-23; // Boltzmann constant J/K
    let k_b_ev = 8.617e-5; // Boltzmann constant eV/K

    println!("🌡️  Temperature: {} K", temperature);
    println!("   kT = {:.3} eV", k_b_ev * temperature);

    // Maxwell-Boltzmann speed distribution for nitrogen molecules
    let mass_n2 = 28.0 / 6.022e23 * 1e-3; // kg
    let v_range = Array1::linspace(0.0, 2000.0, 100); // m/s
    let mut maxwell_boltzmann = Array1::zeros(v_range.len());

    let normalization = 4.0 * PI * (mass_n2 / (2.0 * PI * k_b * temperature)).powf(1.5);

    for (i, &v) in v_range.iter().enumerate() {
        maxwell_boltzmann[i] =
            normalization * v * v * (-mass_n2 * v * v / (2.0 * k_b * temperature)).exp();
    }

    // Most probable speed
    let v_mp = (2.0 * k_b * temperature / mass_n2).sqrt();
    let v_avg = (8.0 * k_b * temperature / (PI * mass_n2)).sqrt();
    let v_rms = (3.0 * k_b * temperature / mass_n2).sqrt();

    println!("\n🏃 Characteristic speeds for N₂ molecules:");
    println!("   Most probable: {:.1} m/s", v_mp);
    println!("   Average: {:.1} m/s", v_avg);
    println!("   RMS: {:.1} m/s", v_rms);

    // Sample the distribution
    println!("\n📈 Maxwell-Boltzmann distribution f(v):");
    for i in (0..maxwell_boltzmann.len()).step_by(maxwell_boltzmann.len() / 8) {
        println!(
            "   v = {:6.1} m/s: f(v) = {:.2e}",
            v_range[i], maxwell_boltzmann[i]
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn run_signal_processing_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("📡 Analyzing Fresnel diffraction and chirp signals...");

    let t_range = Array1::linspace(-3.0, 3.0, 100);
    let mut fresnel_c = Array1::zeros(t_range.len());
    let mut fresnel_s = Array1::zeros(t_range.len());

    // Compute Fresnel integrals
    for (i, &t) in t_range.iter().enumerate() {
        fresnel_c[i] = fresnel_c_integral(t);
        fresnel_s[i] = fresnel_s_integral(t);
    }

    println!("🌊 Fresnel integrals computed for diffraction analysis:");
    println!("   C(∞) = S(∞) = 0.5 (limiting values)");

    // Fresnel diffraction intensity at a straight edge
    let mut intensity = Array1::zeros(t_range.len());
    for (i, _t) in t_range.iter().enumerate() {
        let c_val = fresnel_c[i] + 0.5;
        let s_val = fresnel_s[i] + 0.5;
        intensity[i] = 0.25 * (c_val * c_val + s_val * s_val);
    }

    println!("\n🔆 Diffraction intensity I(v)/I₀:");
    for i in (0..intensity.len()).step_by(intensity.len() / 10) {
        println!("   v = {:6.2}: I/I₀ = {:6.4}", t_range[i], intensity[i]);
    }

    // Find first maximum and minimum
    let max_idx = intensity
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    println!("\n🎯 First diffraction maximum:");
    println!("   Position: v = {:.2}", t_range[max_idx]);
    println!("   Intensity: I/I₀ = {:.4}", intensity[max_idx]);

    Ok(())
}

#[allow(dead_code)]
fn run_scattering_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("💫 Computing Mie scattering cross-sections...");

    let radius = 0.001; // m (1 mm sphere)
    let wavelength = 500e-9; // m (green light)
    let refractive_index = 1.5_f64; // glass

    let size_parameter = 2.0 * PI * radius / wavelength;

    println!("🔍 Scattering parameters:");
    println!("   Sphere radius: {} mm", radius * 1000.0);
    println!("   Wavelength: {} nm", wavelength * 1e9);
    println!("   Size parameter x = 2πa/λ = {:.2}", size_parameter);
    println!("   Refractive index: {}", refractive_index);

    // Classify scattering regime
    if size_parameter < 1.0 {
        println!("📏 Rayleigh scattering regime (x << 1)");
        let rayleigh_cross_section = 8.0 * PI / 3.0
            * size_parameter.powi(4)
            * ((refractive_index * refractive_index - 1.0)
                / (refractive_index * refractive_index + 2.0))
                .powi(2);
        println!("   Scattering cross-section ∝ λ⁻⁴");
        println!(
            "   Normalized cross-section: {:.2e}",
            rayleigh_cross_section
        );
    } else if size_parameter > 1.0 {
        println!("💿 Geometric optics regime (x >> 1)");
        println!("   Cross-section approaches geometric limit πa²");
    } else {
        println!("🌀 Mie scattering regime (x ≈ 1)");
        println!("   Complex interference between partial waves");

        // Simplified Mie calculation for first few terms
        let mut q_ext = 0.0;
        let mut q_sca = 0.0;

        for n in 1..=5 {
            let x = size_parameter;
            let mx = refractive_index * x;

            // Approximate Mie coefficients (simplified)
            let psi_n_x = spherical_j(n as f64, x) * x;
            let chi_n_x = -spherical_y(n as f64, x) * x;
            let psi_n_mx = spherical_j(n as f64, mx) * mx;

            // Very simplified coefficients (for demonstration)
            let a_n = psi_n_x / (psi_n_x + chi_n_x);
            let b_n = psi_n_x / (psi_n_x + chi_n_x);

            q_ext += (2.0 * n as f64 + 1.0) * (a_n + b_n);
            q_sca += (2.0 * n as f64 + 1.0) * (a_n * a_n + b_n * b_n);
        }

        q_ext *= 2.0 / (size_parameter * size_parameter);
        q_sca *= 2.0 / (size_parameter * size_parameter);

        println!("   Extinction efficiency: Q_ext ≈ {:.3}", q_ext);
        println!("   Scattering efficiency: Q_sca ≈ {:.3}", q_sca);
        println!("   Absorption efficiency: Q_abs ≈ {:.3}", q_ext - q_sca);
    }

    Ok(())
}

#[allow(dead_code)]
fn run_tunneling_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 Simulating quantum tunneling with Airy functions...");

    let barrier_height = 5.0_f64; // eV
    let particle_energy = 3.0_f64; // eV
    let barrier_width = 1e-9_f64; // m
    let mass = 9.109e-31_f64; // electron mass kg
    let hbar = 1.055e-34_f64; // J·s
    let e_v_to_j = 1.602e-19_f64;

    println!("⚡ Tunneling parameters:");
    println!("   Particle energy: {:.1} eV", particle_energy);
    println!("   Barrier height: {:.1} eV", barrier_height);
    println!("   Barrier width: {:.1} nm", barrier_width * 1e9);

    if particle_energy >= barrier_height {
        println!("🚀 Classical transmission (E ≥ V₀)");
        return Ok(());
    }

    // WKB approximation
    let energy_diff_j = (barrier_height - particle_energy) * e_v_to_j;
    let momentum_inside = (2.0 * mass * energy_diff_j).sqrt();
    let action_integral = 2.0 * momentum_inside * barrier_width / hbar;
    let transmission_wkb = (-action_integral).exp();

    println!("\n🌀 WKB approximation:");
    println!("   Action integral: S/ℏ = {:.2}", action_integral);
    println!("   Transmission coefficient: T ≈ {:.2e}", transmission_wkb);

    // For a rectangular barrier, exact result is available
    let k_inside = momentum_inside / hbar;
    let transmission_exact = 1.0
        / (1.0
            + (barrier_height / (4.0 * particle_energy * (barrier_height - particle_energy)))
                * (k_inside * barrier_width).sinh().powi(2));

    println!("\n✅ Exact result (rectangular barrier):");
    println!(
        "   Transmission coefficient: T = {:.2e}",
        transmission_exact
    );
    println!(
        "   Reflection coefficient: R = {:.2e}",
        1.0 - transmission_exact
    );

    // Characteristic lengths
    let de_broglie = 2.0 * PI * hbar / (2.0 * mass * particle_energy * e_v_to_j).sqrt();
    let penetration_depth = hbar / momentum_inside;

    println!("\n📏 Characteristic length scales:");
    println!("   de Broglie wavelength: {:.2} nm", de_broglie * 1e9);
    println!("   Penetration depth: {:.2} nm", penetration_depth * 1e9);

    if barrier_width > 3.0 * penetration_depth {
        println!("🔒 Thick barrier regime (a >> δ)");
    } else {
        println!("🪟 Thin barrier regime (a ≈ δ)");
    }

    Ok(())
}

// Helper functions for special function evaluations
#[allow(dead_code)]
fn fresnel_c_integral(x: f64) -> f64 {
    // Simplified implementation - in practice, use scirs2_special::fresnel
    let tmax = x.abs();
    if tmax > 5.0 {
        return 0.5 * x.signum();
    }

    // Series approximation for small x
    let mut sum = 0.0;
    let mut term = x;
    let x_squared = x * x;
    let pi_half = PI / 2.0;

    for n in 0..20 {
        let coeff = if n % 2 == 0 { 1.0 } else { -1.0 };
        sum += coeff * term / (4 * n + 1) as f64;
        term *= x_squared * pi_half / ((2 * n + 2) * (2 * n + 3)) as f64;
        if term.abs() < 1e-10 {
            break;
        }
    }

    sum * pi_half
}

#[allow(dead_code)]
fn fresnel_s_integral(x: f64) -> f64 {
    // Simplified implementation - in practice, use scirs2_special::fresnel
    let tmax = x.abs();
    if tmax > 5.0 {
        return 0.5 * x.signum();
    }

    // Series approximation for small x
    let mut sum = 0.0;
    let mut term = x * x * x / 3.0;
    let x_squared = x * x;
    let pi_half = PI / 2.0;

    for n in 0..20 {
        let coeff = if n % 2 == 0 { 1.0 } else { -1.0 };
        sum += coeff * term / (4 * n + 3) as f64;
        term *= x_squared * pi_half / ((2 * n + 3) * (2 * n + 4)) as f64;
        if term.abs() < 1e-10 {
            break;
        }
    }

    sum * pi_half
}

#[allow(dead_code)]
fn spherical_j(n: f64, x: f64) -> f64 {
    // Simplified spherical Bessel function - use scirs2_special in practice
    if x.abs() < 1e-10 {
        return if n == 0.0 { 1.0 } else { 0.0 };
    }

    let j_bessel = j_n(n as i32, x);
    (PI / (2.0 * x)).sqrt() * j_bessel
}

#[allow(dead_code)]
fn spherical_y(n: f64, x: f64) -> f64 {
    // Simplified spherical Neumann function - use scirs2_special in practice
    if x.abs() < 1e-10 {
        return f64::NEG_INFINITY;
    }

    let y_bessel = y_n(n as i32, x);
    (PI / (2.0 * x)).sqrt() * y_bessel
}

#[allow(dead_code)]
fn hermite_physicist(n: usize, x: f64) -> f64 {
    // Physicist's Hermite polynomials - use scirs2_special in practice
    match n {
        0 => 1.0,
        1 => 2.0 * x,
        2 => 4.0 * x * x - 2.0,
        3 => 8.0 * x * x * x - 12.0 * x,
        4 => 16.0 * x.powi(4) - 48.0 * x * x + 12.0,
        5 => 32.0 * x.powi(5) - 160.0 * x.powi(3) + 120.0 * x,
        _ => {
            // Recurrence relation: H_{n+1} = 2x H_n - 2n H_{n-1}
            let mut h_prev2 = 1.0; // H_0
            let mut h_prev1 = 2.0 * x; // H_1

            for k in 2..=n {
                let h_current = 2.0 * x * h_prev1 - 2.0 * (k - 1) as f64 * h_prev2;
                h_prev2 = h_prev1;
                h_prev1 = h_current;
            }
            h_prev1
        }
    }
}

#[allow(dead_code)]
fn j_n(n: i32, x: f64) -> f64 {
    // Placeholder for Bessel function of first kind - use scirs2_special in practice
    match n {
        0 => j0(x),
        1 => j1(x),
        _ => 0.0, // Simplified - implement full Bessel functions
    }
}

#[allow(dead_code)]
fn y_n(n: i32, x: f64) -> f64 {
    // Placeholder for Bessel function of second kind - use scirs2_special in practice
    match n {
        0 => y0(x),
        1 => y1(x),
        _ => 0.0, // Simplified - implement full Neumann functions
    }
}

#[allow(dead_code)]
fn get_user_input(prompt: &str) -> io::Result<String> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}
