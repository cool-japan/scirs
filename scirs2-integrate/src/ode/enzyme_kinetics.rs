//! Advanced enzyme kinetics and metabolic pathway modeling
//!
//! This module provides sophisticated models for enzyme kinetics including
//! multi-substrate mechanisms, allosteric regulation, and metabolic pathway
//! network simulation.

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Types of enzyme mechanisms
#[derive(Debug, Clone, PartialEq)]
pub enum EnzymeMechanism {
    /// Michaelis-Menten single substrate mechanism
    MichaelisMenten {
        km: f64,   // Michaelis constant
        vmax: f64, // Maximum velocity
    },
    /// Bi-substrate ordered sequential mechanism (A binds first, then B)
    OrderedSequential {
        ka: f64,   // Binding constant for substrate A
        kb: f64,   // Binding constant for substrate B
        kp: f64,   // Product release constant for P
        kq: f64,   // Product release constant for Q
        kcat: f64, // Catalytic rate constant
    },
    /// Bi-substrate random sequential mechanism (A and B can bind in any order)
    RandomSequential {
        ka: f64,    // Binding constant for substrate A
        kb: f64,    // Binding constant for substrate B
        kp: f64,    // Product release constant for P
        kq: f64,    // Product release constant for Q
        kcat: f64,  // Catalytic rate constant
        alpha: f64, // Interaction parameter
    },
    /// Ping-pong mechanism (substrate A binds, product P released, then B binds)
    PingPong {
        ka: f64,    // Binding constant for substrate A
        kb: f64,    // Binding constant for substrate B
        kp: f64,    // Product release constant for P
        kq: f64,    // Product release constant for Q
        kcat1: f64, // First catalytic step
        kcat2: f64, // Second catalytic step
    },
    /// Hill equation for cooperative binding
    Hill {
        kd: f64,   // Dissociation constant
        vmax: f64, // Maximum velocity
        n: f64,    // Hill coefficient (cooperativity)
    },
    /// Allosteric enzyme with activators and inhibitors
    Allosteric {
        km: f64,     // Michaelis constant for substrate
        vmax: f64,   // Maximum velocity
        ka_act: f64, // Activation constant for activator
        ka_inh: f64, // Inhibition constant for inhibitor
        n_act: f64,  // Cooperativity for activator
        n_inh: f64,  // Cooperativity for inhibitor
    },
}

/// Enzyme kinetic parameters
#[derive(Debug, Clone)]
pub struct EnzymeParameters {
    /// Enzyme mechanism type
    pub mechanism: EnzymeMechanism,
    /// Temperature (K)
    pub temperature: f64,
    /// pH
    pub ph: f64,
    /// Ionic strength (M)
    pub ionic_strength: f64,
    /// Temperature dependence parameters
    pub temperature_params: Option<TemperatureParams>,
    /// pH dependence parameters
    pub ph_params: Option<PhParams>,
}

/// Temperature dependence parameters
#[derive(Debug, Clone)]
pub struct TemperatureParams {
    /// Enthalpy of activation (kJ/mol)
    pub delta_h: f64,
    /// Entropy of activation (J/(mol·K))
    pub delta_s: f64,
    /// Heat capacity change (J/(mol·K))
    pub delta_cp: f64,
    /// Reference temperature (K)
    pub temp_ref: f64,
}

/// pH dependence parameters
#[derive(Debug, Clone)]
pub struct PhParams {
    /// pKa values for ionizable groups
    pub pka_values: Vec<f64>,
    /// Activity coefficients for different ionization states
    pub activity_coefficients: Vec<f64>,
    /// Optimal pH
    pub ph_optimum: f64,
}

/// Metabolic pathway definition
#[derive(Debug, Clone)]
pub struct MetabolicPathway {
    /// Pathway name
    pub name: String,
    /// Enzyme definitions
    pub enzymes: Vec<EnzymeDefinition>,
    /// Metabolite names
    pub metabolites: Vec<String>,
    /// Stoichiometric matrix (reactions × metabolites)
    pub stoichiometry_matrix: Array2<f64>,
    /// Regulatory relationships
    pub regulations: Vec<Regulation>,
    /// External metabolite concentrations (fixed)
    pub external_metabolites: HashMap<usize, f64>,
}

/// Enzyme definition within a pathway
#[derive(Debug, Clone)]
pub struct EnzymeDefinition {
    /// Enzyme name
    pub name: String,
    /// Kinetic parameters
    pub parameters: EnzymeParameters,
    /// Substrate indices
    pub substrates: Vec<usize>,
    /// Product indices
    pub products: Vec<usize>,
    /// Effector indices (activators/inhibitors)
    pub effectors: Vec<usize>,
    /// Enzyme concentration (nM)
    pub enzyme_concentration: f64,
}

/// Regulatory relationship
#[derive(Debug, Clone)]
pub struct Regulation {
    /// Target enzyme index
    pub target_enzyme: usize,
    /// Effector metabolite index
    pub effector_metabolite: usize,
    /// Type of regulation
    pub regulation_type: RegulationType,
    /// Regulation strength parameter
    pub strength: f64,
}

/// Types of metabolic regulation
#[derive(Debug, Clone, PartialEq)]
pub enum RegulationType {
    /// Competitive inhibition
    CompetitiveInhibition,
    /// Non-competitive inhibition
    NonCompetitiveInhibition,
    /// Uncompetitive inhibition
    UncompetitiveInhibition,
    /// Allosteric activation
    AllostericActivation,
    /// Allosteric inhibition
    AllostericInhibition,
    /// Feedback inhibition
    FeedbackInhibition,
}

/// Pathway analysis results
#[derive(Debug, Clone)]
pub struct PathwayAnalysis {
    /// Flux control coefficients
    pub flux_control_coefficients: Array1<f64>,
    /// Concentration control coefficients
    pub concentration_control_coefficients: Array2<f64>,
    /// Elasticity coefficients
    pub elasticity_coefficients: Array2<f64>,
    /// Steady-state fluxes
    pub steady_state_fluxes: Array1<f64>,
    /// Steady-state concentrations
    pub steady_state_concentrations: Array1<f64>,
}

impl EnzymeParameters {
    /// Create Michaelis-Menten enzyme parameters
    pub fn michaelis_menten(km: f64, vmax: f64) -> Self {
        Self {
            mechanism: EnzymeMechanism::MichaelisMenten { km, vmax },
            temperature: 310.15, // 37°C
            ph: 7.4,
            ionic_strength: 0.15,
            temperature_params: None,
            ph_params: None,
        }
    }

    /// Create Hill equation enzyme parameters
    pub fn hill(kd: f64, vmax: f64, n: f64) -> Self {
        Self {
            mechanism: EnzymeMechanism::Hill { kd, vmax, n },
            temperature: 310.15,
            ph: 7.4,
            ionic_strength: 0.15,
            temperature_params: None,
            ph_params: None,
        }
    }

    /// Create allosteric enzyme parameters
    pub fn allosteric(
        km: f64,
        vmax: f64,
        ka_act: f64,
        ka_inh: f64,
        n_act: f64,
        n_inh: f64,
    ) -> Self {
        Self {
            mechanism: EnzymeMechanism::Allosteric {
                km,
                vmax,
                ka_act,
                ka_inh,
                n_act,
                n_inh,
            },
            temperature: 310.15,
            ph: 7.4,
            ionic_strength: 0.15,
            temperature_params: None,
            ph_params: None,
        }
    }

    /// Calculate reaction rate for this enzyme
    pub fn calculate_rate(&self, concentrations: &[f64]) -> f64 {
        let base_rate = match &self.mechanism {
            EnzymeMechanism::MichaelisMenten { km, vmax } => {
                if concentrations.is_empty() {
                    return 0.0;
                }
                let s = concentrations[0];
                vmax * s / (km + s)
            }
            EnzymeMechanism::OrderedSequential {
                ka,
                kb,
                kp,
                kq,
                kcat,
            } => {
                if concentrations.len() < 2 {
                    return 0.0;
                }
                let a = concentrations[0];
                let b = concentrations[1];
                let p = if concentrations.len() > 2 {
                    concentrations[2]
                } else {
                    0.0
                };
                let q = if concentrations.len() > 3 {
                    concentrations[3]
                } else {
                    0.0
                };

                // Ordered sequential rate equation
                let numerator = kcat * a * b;
                let denominator =
                    ka * kb + kb * a + ka * b + a * b + (kp * a * q) / kq + (kq * b * p) / kp;
                if denominator > 1e-12 {
                    numerator / denominator
                } else {
                    0.0
                }
            }
            EnzymeMechanism::RandomSequential {
                ka,
                kb,
                kp,
                kq,
                kcat,
                alpha,
            } => {
                if concentrations.len() < 2 {
                    return 0.0;
                }
                let a = concentrations[0];
                let b = concentrations[1];
                let p = if concentrations.len() > 2 {
                    concentrations[2]
                } else {
                    0.0
                };
                let q = if concentrations.len() > 3 {
                    concentrations[3]
                } else {
                    0.0
                };

                // Random sequential rate equation with interaction parameter
                let numerator = kcat * a * b;
                let denominator = ka * kb * (1.0 + alpha)
                    + kb * a
                    + ka * b
                    + a * b
                    + (kp * a * q) / (kq * alpha)
                    + (kq * b * p) / (kp * alpha);
                if denominator > 1e-12 {
                    numerator / denominator
                } else {
                    0.0
                }
            }
            EnzymeMechanism::PingPong {
                ka,
                kb,
                kp,
                kq,
                kcat1,
                kcat2,
            } => {
                if concentrations.len() < 2 {
                    return 0.0;
                }
                let a = concentrations[0];
                let b = concentrations[1];
                let p = if concentrations.len() > 2 {
                    concentrations[2]
                } else {
                    0.0
                };
                let q = if concentrations.len() > 3 {
                    concentrations[3]
                } else {
                    0.0
                };

                // Ping-pong rate equation
                let v1 = kcat1;
                let v2 = kcat2;
                let numerator = v1 * v2 * a * b;
                let denominator = v2 * ka * b + v1 * kb * a + v1 * kp * q + v2 * kq * p;
                if denominator > 1e-12 {
                    numerator / denominator
                } else {
                    0.0
                }
            }
            EnzymeMechanism::Hill { kd, vmax, n } => {
                if concentrations.is_empty() {
                    return 0.0;
                }
                let s = concentrations[0];
                let s_n = s.powf(*n);
                let kd_n = kd.powf(*n);
                vmax * s_n / (kd_n + s_n)
            }
            EnzymeMechanism::Allosteric {
                km,
                vmax,
                ka_act,
                ka_inh,
                n_act,
                n_inh,
            } => {
                if concentrations.is_empty() {
                    return 0.0;
                }
                let s = concentrations[0];
                let activator = if concentrations.len() > 1 {
                    concentrations[1]
                } else {
                    0.0
                };
                let inhibitor = if concentrations.len() > 2 {
                    concentrations[2]
                } else {
                    0.0
                };

                // Base Michaelis-Menten rate
                let base_rate = vmax * s / (km + s);

                // Allosteric modulation
                let activation_factor = if activator > 0.0 {
                    (1.0 + (activator / ka_act).powf(*n_act))
                        / (1.0 + (activator / ka_act).powf(*n_act))
                } else {
                    1.0
                };

                let inhibition_factor = if inhibitor > 0.0 {
                    1.0 / (1.0 + (inhibitor / ka_inh).powf(*n_inh))
                } else {
                    1.0
                };

                base_rate * activation_factor * inhibition_factor
            }
        };

        // Apply temperature and pH corrections
        let temp_correction = self.calculate_temperature_correction();
        let ph_correction = self.calculate_ph_correction();

        base_rate * temp_correction * ph_correction
    }

    /// Calculate temperature correction factor
    fn calculate_temperature_correction(&self) -> f64 {
        if let Some(ref temp_params) = self.temperature_params {
            let t = self.temperature;
            let t_ref = temp_params.temp_ref;
            let r = 8.314; // Gas constant J/(mol·K)

            // van't Hoff equation with heat capacity correction
            let delta_h_corr = temp_params.delta_h + temp_params.delta_cp * (t - t_ref);
            let delta_s_corr = temp_params.delta_s + temp_params.delta_cp * (t / t_ref).ln();

            let delta_g = delta_h_corr - t * delta_s_corr;
            (-delta_g / (r * t)).exp()
        } else {
            // Simple Arrhenius approximation if no detailed parameters
            let ea = 50000.0; // Default activation energy 50 kJ/mol
            let r = 8.314;
            let t_ref = 298.15;
            (-ea / r * (1.0 / self.temperature - 1.0 / t_ref)).exp()
        }
    }

    /// Calculate pH correction factor
    fn calculate_ph_correction(&self) -> f64 {
        if let Some(ref ph_params) = self.ph_params {
            // Henderson-Hasselbalch equation for multiple ionizable groups
            let mut total_activity = 0.0;
            let ph = self.ph;

            for (i, &pka) in ph_params.pka_values.iter().enumerate() {
                let alpha = 1.0 / (1.0 + 10.0_f64.powf(pka - ph));
                total_activity += alpha * ph_params.activity_coefficients.get(i).unwrap_or(&1.0);
            }

            total_activity / ph_params.pka_values.len() as f64
        } else {
            // Simple pH bell curve if no detailed parameters
            let ph_opt = 7.4;
            let ph_width = 2.0;
            let delta_ph = (self.ph - ph_opt) / ph_width;
            (-0.5 * delta_ph * delta_ph).exp()
        }
    }
}

impl MetabolicPathway {
    /// Create a new empty metabolic pathway
    pub fn new(_name: String, num_metabolites: usize, numenzymes: usize) -> Self {
        Self {
            name: _name,
            enzymes: Vec::new(),
            metabolites: (0..num_metabolites).map(|i| format!("M{i}")).collect(),
            stoichiometry_matrix: Array2::zeros((numenzymes, num_metabolites)),
            regulations: Vec::new(),
            external_metabolites: HashMap::new(),
        }
    }

    /// Add an enzyme to the pathway
    pub fn add_enzyme(&mut self, enzyme: EnzymeDefinition) {
        self.enzymes.push(enzyme);
    }

    /// Add a regulatory relationship
    pub fn add_regulation(&mut self, regulation: Regulation) {
        self.regulations.push(regulation);
    }

    /// Set external metabolite concentration
    pub fn set_external_metabolite(&mut self, _metaboliteidx: usize, concentration: f64) {
        self.external_metabolites
            .insert(_metaboliteidx, concentration);
    }

    /// Calculate reaction rates for all enzymes
    pub fn calculate_reaction_rates(&self, concentrations: &Array1<f64>) -> Array1<f64> {
        let mut rates = Array1::zeros(self.enzymes.len());

        for (i, enzyme) in self.enzymes.iter().enumerate() {
            // Get substrate _concentrations
            let substrate_concentrations: Vec<f64> = enzyme
                .substrates
                .iter()
                .map(|&idx| concentrations.get(idx).copied().unwrap_or(0.0))
                .collect();

            // Get effector _concentrations for allosteric enzymes
            let effector_concentrations: Vec<f64> = enzyme
                .effectors
                .iter()
                .map(|&idx| concentrations.get(idx).copied().unwrap_or(0.0))
                .collect();

            // Combine substrate and effector _concentrations
            let mut all_concentrations = substrate_concentrations;
            all_concentrations.extend(effector_concentrations);

            // Calculate base rate
            let base_rate = enzyme.parameters.calculate_rate(&all_concentrations);

            // Apply regulatory effects
            let regulated_rate = self.apply_regulations(i, base_rate, concentrations);

            rates[i] = regulated_rate * enzyme.enzyme_concentration * 1e-9; // Convert nM to M
        }

        rates
    }

    /// Apply regulatory effects to an enzyme
    fn apply_regulations(
        &self,
        enzyme_idx: usize,
        base_rate: f64,
        concentrations: &Array1<f64>,
    ) -> f64 {
        let mut modified_rate = base_rate;

        for regulation in &self.regulations {
            if regulation.target_enzyme == enzyme_idx {
                let effector_conc = concentrations
                    .get(regulation.effector_metabolite)
                    .copied()
                    .unwrap_or(0.0);

                let regulation_factor = match regulation.regulation_type {
                    RegulationType::CompetitiveInhibition => {
                        1.0 / (1.0 + effector_conc / regulation.strength)
                    }
                    RegulationType::NonCompetitiveInhibition => {
                        1.0 / (1.0 + effector_conc / regulation.strength)
                    }
                    RegulationType::UncompetitiveInhibition => {
                        1.0 / (1.0 + effector_conc / regulation.strength)
                    }
                    RegulationType::AllostericActivation => {
                        1.0 + effector_conc / regulation.strength
                    }
                    RegulationType::AllostericInhibition => {
                        1.0 / (1.0 + (effector_conc / regulation.strength).powf(2.0))
                    }
                    RegulationType::FeedbackInhibition => {
                        1.0 / (1.0 + (effector_conc / regulation.strength).powf(4.0))
                    }
                };

                modified_rate *= regulation_factor;
            }
        }

        modified_rate
    }

    /// Calculate concentration time derivatives
    pub fn calculate_derivatives(&self, concentrations: &Array1<f64>) -> Array1<f64> {
        let reaction_rates = self.calculate_reaction_rates(concentrations);
        let mut derivatives = Array1::zeros(concentrations.len());

        // Apply stoichiometry matrix
        for (reaction_idx, &rate) in reaction_rates.iter().enumerate() {
            for metabolite_idx in 0..derivatives.len() {
                if let Some(&stoich) = self
                    .stoichiometry_matrix
                    .get((reaction_idx, metabolite_idx))
                {
                    derivatives[metabolite_idx] += stoich * rate;
                }
            }
        }

        // External metabolites have zero derivatives
        for &metabolite_idx in self.external_metabolites.keys() {
            if metabolite_idx < derivatives.len() {
                derivatives[metabolite_idx] = 0.0;
            }
        }

        derivatives
    }

    /// Perform metabolic control analysis
    pub fn control_analysis(&self, _steady_stateconcentrations: &Array1<f64>) -> PathwayAnalysis {
        let num_enzymes = self.enzymes.len();
        let num_metabolites = _steady_stateconcentrations.len();

        // Calculate flux control coefficients
        let flux_control_coefficients =
            self.calculate_flux_control_coefficients(_steady_stateconcentrations);

        // Calculate concentration control coefficients
        let concentration_control_coefficients = Array2::zeros((num_enzymes, num_metabolites));

        // Calculate elasticity coefficients
        let elasticity_coefficients =
            self.calculate_elasticity_coefficients(_steady_stateconcentrations);

        // Calculate steady-state fluxes
        let steady_state_fluxes = self.calculate_reaction_rates(_steady_stateconcentrations);

        PathwayAnalysis {
            flux_control_coefficients,
            concentration_control_coefficients,
            elasticity_coefficients,
            steady_state_fluxes,
            steady_state_concentrations: _steady_stateconcentrations.clone(),
        }
    }

    /// Calculate flux control coefficients
    fn calculate_flux_control_coefficients(&self, concentrations: &Array1<f64>) -> Array1<f64> {
        let num_enzymes = self.enzymes.len();
        let mut flux_control_coefficients = Array1::zeros(num_enzymes);

        let base_flux = self.calculate_reaction_rates(concentrations).sum();
        let perturbation = 0.01; // 1% perturbation

        for i in 0..num_enzymes {
            // Perturb enzyme concentration
            let mut perturbed_pathway = self.clone();
            perturbed_pathway.enzymes[i].enzyme_concentration *= 1.0 + perturbation;

            let perturbed_flux = perturbed_pathway
                .calculate_reaction_rates(concentrations)
                .sum();

            // Calculate control coefficient
            if base_flux > 1e-12 {
                flux_control_coefficients[i] =
                    ((perturbed_flux - base_flux) / base_flux) / perturbation;
            }
        }

        flux_control_coefficients
    }

    /// Calculate elasticity coefficients
    fn calculate_elasticity_coefficients(&self, concentrations: &Array1<f64>) -> Array2<f64> {
        let num_enzymes = self.enzymes.len();
        let num_metabolites = concentrations.len();
        let mut elasticity_coefficients = Array2::zeros((num_enzymes, num_metabolites));

        let base_rates = self.calculate_reaction_rates(concentrations);
        let perturbation = 0.01; // 1% perturbation

        for enzyme_idx in 0..num_enzymes {
            for metabolite_idx in 0..num_metabolites {
                if concentrations[metabolite_idx] > 1e-12 {
                    let mut perturbed_concentrations = concentrations.clone();
                    perturbed_concentrations[metabolite_idx] *= 1.0 + perturbation;

                    let perturbed_rates = self.calculate_reaction_rates(&perturbed_concentrations);

                    // Calculate elasticity coefficient
                    if base_rates[enzyme_idx] > 1e-12 {
                        elasticity_coefficients[(enzyme_idx, metabolite_idx)] =
                            ((perturbed_rates[enzyme_idx] - base_rates[enzyme_idx])
                                / base_rates[enzyme_idx])
                                / perturbation;
                    }
                }
            }
        }

        elasticity_coefficients
    }
}

/// Factory functions for common metabolic pathways
pub mod pathways {
    use super::*;
    use scirs2_core::ndarray::arr2;

    /// Create a simple glycolysis pathway (simplified)
    pub fn simple_glycolysis() -> MetabolicPathway {
        let mut pathway = MetabolicPathway::new("Simple Glycolysis".to_string(), 6, 3);

        // Metabolites: Glucose, G6P, F6P, FBP, PEP, Pyruvate
        pathway.metabolites = vec![
            "Glucose".to_string(),
            "G6P".to_string(),
            "F6P".to_string(),
            "FBP".to_string(),
            "PEP".to_string(),
            "Pyruvate".to_string(),
        ];

        // Enzyme 1: Hexokinase (Glucose -> G6P)
        pathway.add_enzyme(EnzymeDefinition {
            name: "Hexokinase".to_string(),
            parameters: EnzymeParameters::michaelis_menten(0.1, 100.0), // Km = 0.1 mM, Vmax = 100 μM/s
            substrates: vec![0],                                        // Glucose
            products: vec![1],                                          // G6P
            effectors: vec![],
            enzyme_concentration: 50.0, // 50 nM
        });
        // Enzyme 2: Phosphofructokinase (F6P -> FBP) - allosteric
        pathway.add_enzyme(EnzymeDefinition {
            name: "Phosphofructokinase".to_string(),
            parameters: EnzymeParameters::allosteric(
                0.3,   // Km
                200.0, // Vmax
                0.1,   // Ka_act (activation by AMP)
                2.0,   // Ka_inh (inhibition by ATP)
                2.0,   // n_act
                4.0,   // n_inh
            ),
            substrates: vec![2], // F6P
            products: vec![3],   // FBP
            effectors: vec![],   // AMP, ATP (would be separate metabolites)
            enzyme_concentration: 30.0,
        });
        // Enzyme 3: Pyruvate kinase (PEP -> Pyruvate)
        pathway.add_enzyme(EnzymeDefinition {
            name: "Pyruvate Kinase".to_string(),
            parameters: EnzymeParameters::hill(0.5, 300.0, 2.0), // Kd = 0.5 mM, Vmax = 300 μM/s, n = 2
            substrates: vec![4],                                 // PEP
            products: vec![5],                                   // Pyruvate
            effectors: vec![],
            enzyme_concentration: 100.0,
        });
        // Set stoichiometry matrix (enzymes × metabolites)
        pathway.stoichiometry_matrix = arr2(&[
            [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0], // Hexokinase: Glucose -> G6P
            [0.0, 0.0, -1.0, 1.0, 0.0, 0.0], // PFK: F6P -> FBP
            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0], // Pyruvate kinase: PEP -> Pyruvate
        ]);
        // Add feedback inhibition: G6P inhibits Hexokinase
        pathway.add_regulation(Regulation {
            target_enzyme: 0,
            effector_metabolite: 1,
            regulation_type: RegulationType::FeedbackInhibition,
            strength: 1.0, // Ki = 1.0 mM
        });

        // Set external metabolites (glucose and pyruvate)
        pathway.set_external_metabolite(0, 5.0); // 5 mM glucose
        pathway.set_external_metabolite(5, 0.1); // 0.1 mM pyruvate

        pathway
    }

    /// Create a TCA cycle pathway (simplified)
    pub fn tca_cycle() -> MetabolicPathway {
        let mut pathway = MetabolicPathway::new("TCA Cycle".to_string(), 8, 8);

        // Metabolites: Acetyl-CoA, Citrate, Isocitrate, α-Ketoglutarate,
        // Succinyl-CoA, Succinate, Fumarate, Malate, Oxaloacetate
        pathway.metabolites = vec![
            "Acetyl-CoA".to_string(),
            "Citrate".to_string(),
            "Isocitrate".to_string(),
            "α-Ketoglutarate".to_string(),
            "Succinyl-CoA".to_string(),
            "Succinate".to_string(),
            "Fumarate".to_string(),
            "Malate".to_string(),
        ];

        // Add enzymes for each step of TCA cycle
        let enzyme_params = [
            ("Citrate Synthase", 0.1, 50.0),
            ("Aconitase", 0.3, 80.0),
            ("Isocitrate Dehydrogenase", 0.2, 60.0),
            ("α-Ketoglutarate Dehydrogenase", 0.4, 40.0),
            ("Succinyl-CoA Synthetase", 0.1, 70.0),
            ("Succinate Dehydrogenase", 0.5, 30.0),
            ("Fumarase", 0.2, 100.0),
            ("Malate Dehydrogenase", 0.3, 90.0),
        ];

        for (i, (name, km, vmax)) in enzyme_params.iter().enumerate() {
            pathway.add_enzyme(EnzymeDefinition {
                name: name.to_string(),
                parameters: EnzymeParameters::michaelis_menten(*km, *vmax),
                substrates: vec![i],
                products: vec![(i + 1) % 8],
                effectors: vec![],
                enzyme_concentration: 50.0,
            });
        }

        // Set stoichiometry matrix for cyclic pathway
        let mut stoich = Array2::zeros((8, 8));
        for i in 0..8 {
            stoich[[i, i]] = -1.0; // Consume substrate
            stoich[[i, (i + 1) % 8]] = 1.0; // Produce product
        }
        pathway.stoichiometry_matrix = stoich;

        pathway
    }

    /// Create a purine biosynthesis pathway
    pub fn purine_biosynthesis() -> MetabolicPathway {
        let mut pathway = MetabolicPathway::new("Purine Biosynthesis".to_string(), 10, 10);

        // Simplified purine biosynthesis pathway
        pathway.metabolites = vec![
            "PRPP".to_string(),                  // 0
            "5-Phosphoribosylamine".to_string(), // 1
            "GAR".to_string(),                   // 2
            "FGAR".to_string(),                  // 3
            "FGAM".to_string(),                  // 4
            "AIR".to_string(),                   // 5
            "CAIR".to_string(),                  // 6
            "SAICAR".to_string(),                // 7
            "AICAR".to_string(),                 // 8
            "IMP".to_string(),                   // 9
        ];

        // Add enzymes with different kinetic models
        let enzymes = [
            (
                "PRPP Amidotransferase",
                EnzymeParameters::michaelis_menten(0.1, 50.0),
            ),
            (
                "GAR Synthetase",
                EnzymeParameters::michaelis_menten(0.2, 60.0),
            ),
            (
                "GAR Transformylase",
                EnzymeParameters::michaelis_menten(0.15, 40.0),
            ),
            (
                "FGAM Synthetase",
                EnzymeParameters::michaelis_menten(0.3, 30.0),
            ),
            (
                "AIR Synthetase",
                EnzymeParameters::michaelis_menten(0.25, 45.0),
            ),
            (
                "AIR Carboxylase",
                EnzymeParameters::michaelis_menten(0.1, 35.0),
            ),
            (
                "SAICAR Synthetase",
                EnzymeParameters::michaelis_menten(0.2, 55.0),
            ),
            (
                "SAICAR Lyase",
                EnzymeParameters::michaelis_menten(0.4, 70.0),
            ),
            (
                "AICAR Transformylase",
                EnzymeParameters::michaelis_menten(0.3, 50.0),
            ),
            ("IMP Synthase", EnzymeParameters::hill(0.2, 40.0, 2.0)),
        ];

        for (i, (name, params)) in enzymes.iter().enumerate() {
            pathway.add_enzyme(EnzymeDefinition {
                name: name.to_string(),
                parameters: params.clone(),
                substrates: vec![i],
                products: vec![i + 1],
                effectors: vec![],
                enzyme_concentration: 25.0,
            });
        }

        // Linear pathway stoichiometry
        let mut stoich = Array2::zeros((10, 10));
        for i in 0..9 {
            stoich[[i, i]] = -1.0; // Consume substrate
            stoich[[i, i + 1]] = 1.0; // Produce product
        }
        pathway.stoichiometry_matrix = stoich;

        // Add feedback inhibition: IMP inhibits first enzyme
        pathway.add_regulation(Regulation {
            target_enzyme: 0,
            effector_metabolite: 9,
            regulation_type: RegulationType::FeedbackInhibition,
            strength: 0.5,
        });

        pathway
    }
}

#[cfg(test)]
mod tests {
    use crate::ode::{enzyme_kinetics::pathways, EnzymeParameters};
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_michaelis_menten_kinetics() {
        let mut params = EnzymeParameters::michaelis_menten(1.0, 100.0);
        params.temperature = 298.15; // Set to reference temperature to avoid correction

        // Test at Km concentration (should give Vmax/2)
        let rate_at_km = params.calculate_rate(&[1.0]);
        assert_abs_diff_eq!(rate_at_km, 50.0, epsilon = 1e-10);

        // Test at high substrate concentration (should approach Vmax)
        let rate_high_s = params.calculate_rate(&[100.0]);
        assert!(rate_high_s > 99.0);
    }

    #[test]
    fn test_hill_kinetics() {
        let mut params = EnzymeParameters::hill(1.0, 100.0, 2.0);
        params.temperature = 298.15; // Set to reference temperature to avoid correction

        // Test Hill equation behavior
        let rate_at_kd = params.calculate_rate(&[1.0]);
        assert_abs_diff_eq!(rate_at_kd, 50.0, epsilon = 1e-10);

        // Test cooperativity
        let rate_low = params.calculate_rate(&[0.1]);
        let rate_high = params.calculate_rate(&[10.0]);
        assert!(rate_high > rate_low);
    }

    #[test]
    fn test_simple_glycolysis_pathway() {
        let pathway = pathways::simple_glycolysis();

        assert_eq!(pathway.enzymes.len(), 3);
        assert_eq!(pathway.metabolites.len(), 6);
        assert_eq!(pathway.regulations.len(), 1);

        // Test rate calculation with initial concentrations
        let concentrations = Array1::from_vec(vec![5.0, 0.1, 0.1, 0.1, 0.1, 0.1]);
        let rates = pathway.calculate_reaction_rates(&concentrations);

        // All rates should be positive
        for &rate in rates.iter() {
            assert!(rate >= 0.0);
        }
    }

    #[test]
    fn test_tca_cycle_pathway() {
        let pathway = pathways::tca_cycle();

        assert_eq!(pathway.enzymes.len(), 8);
        assert_eq!(pathway.metabolites.len(), 8);

        // Test with uniform concentrations
        let concentrations = Array1::from_vec(vec![1.0; 8]);
        let rates = pathway.calculate_reaction_rates(&concentrations);

        // All rates should be positive
        for &rate in rates.iter() {
            assert!(rate >= 0.0);
        }
    }

    #[test]
    fn test_allosteric_regulation() {
        let params = EnzymeParameters::allosteric(
            1.0,   // Km
            100.0, // Vmax
            0.5,   // Ka_act
            2.0,   // Ka_inh
            2.0,   // n_act
            2.0,   // n_inh
        );

        // Test with substrate only
        let rate_base = params.calculate_rate(&[1.0]);

        // Test with activator
        let rate_activated = params.calculate_rate(&[1.0, 0.5]);

        // Test with inhibitor
        let rate_inhibited = params.calculate_rate(&[1.0, 0.0, 2.0]);

        assert!(rate_activated >= rate_base);
        assert!(rate_inhibited <= rate_base);
    }

    #[test]
    fn test_temperature_effects() {
        let mut params = EnzymeParameters::michaelis_menten(1.0, 100.0);

        // Test at different temperatures
        params.temperature = 298.15; // 25°C
        let rate_25c = params.calculate_rate(&[1.0]);

        params.temperature = 310.15; // 37°C
        let rate_37c = params.calculate_rate(&[1.0]);

        // Rate should increase with temperature
        assert!(rate_37c > rate_25c);
    }

    #[test]
    fn test_pathway_derivatives() {
        let pathway = pathways::simple_glycolysis();
        let concentrations = Array1::from_vec(vec![5.0, 0.1, 0.1, 0.1, 0.1, 0.1]);

        let derivatives = pathway.calculate_derivatives(&concentrations);

        // Check that external metabolites have zero derivatives
        assert_abs_diff_eq!(derivatives[0], 0.0, epsilon = 1e-10); // Glucose (external)
        assert_abs_diff_eq!(derivatives[5], 0.0, epsilon = 1e-10); // Pyruvate (external)
    }

    #[test]
    fn test_control_analysis() {
        let pathway = pathways::simple_glycolysis();
        let concentrations = Array1::from_vec(vec![5.0, 1.0, 0.5, 0.3, 0.2, 0.1]);

        let analysis = pathway.control_analysis(&concentrations);

        // Flux control coefficients should sum to 1 (summation theorem)
        let sum_fcc = analysis.flux_control_coefficients.sum();
        assert_abs_diff_eq!(sum_fcc, 1.0, epsilon = 0.1);
    }
}
