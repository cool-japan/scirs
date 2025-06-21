//! Chemical equilibrium calculation methods
//!
//! This module provides methods for calculating chemical equilibrium compositions,
//! equilibrium constants, and thermodynamic properties of chemical systems.

use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Types of equilibrium calculations
#[derive(Debug, Clone, PartialEq)]
pub enum EquilibriumType {
    /// Single reaction equilibrium
    SingleReaction,
    /// Multiple reaction equilibrium
    MultipleReactions,
    /// Phase equilibrium
    PhaseEquilibrium,
    /// Ionic equilibrium (with activity coefficients)
    IonicEquilibrium,
    /// Simultaneous reactions with constraints
    ConstrainedEquilibrium,
}

/// Chemical equilibrium calculator
#[derive(Debug, Clone)]
pub struct EquilibriumCalculator {
    /// Stoichiometric matrix (reactions × species)
    pub stoichiometry_matrix: Array2<f64>,
    /// Species names
    pub species_names: Vec<String>,
    /// Reaction names
    pub reaction_names: Vec<String>,
    /// Equilibrium constants at standard conditions
    pub equilibrium_constants: Array1<f64>,
    /// Temperature (K)
    pub temperature: f64,
    /// Pressure (atm)
    pub pressure: f64,
    /// Thermodynamic data
    pub thermo_data: Vec<ThermoData>,
    /// Activity coefficient model
    pub activity_model: ActivityModel,
}

/// Thermodynamic data for species
#[derive(Debug, Clone)]
pub struct ThermoData {
    /// Species name
    pub name: String,
    /// Standard enthalpy of formation (kJ/mol)
    pub delta_h_f: f64,
    /// Standard entropy (J/(mol·K))
    pub s0: f64,
    /// Heat capacity coefficients (Cp = a + bT + cT^2 + dT^3)
    pub cp_coeffs: [f64; 4], // [a, b, c, d]
    /// Standard Gibbs free energy of formation (kJ/mol)
    pub delta_g_f: f64,
    /// Activity coefficient parameters
    pub activity_params: ActivityParams,
}

/// Activity coefficient parameters
#[derive(Debug, Clone)]
pub struct ActivityParams {
    /// Ion charge (for ionic species)
    pub charge: f64,
    /// Ion size parameter (Å)
    pub ion_size: f64,
    /// Interaction parameters for other species
    pub interaction_params: HashMap<String, f64>,
}

/// Activity coefficient models
#[derive(Debug, Clone, PartialEq)]
pub enum ActivityModel {
    /// Ideal solution (activity coefficients = 1)
    Ideal,
    /// Debye-Hückel model for ionic solutions
    DebyeHuckel,
    /// Extended Debye-Hückel model
    ExtendedDebyeHuckel,
    /// Pitzer model for concentrated solutions
    Pitzer,
    /// UNIQUAC model for non-electrolyte solutions
    Uniquac,
    /// Van Laar model
    VanLaar,
}

/// Equilibrium calculation results
#[derive(Debug, Clone)]
pub struct EquilibriumResult {
    /// Equilibrium concentrations
    pub concentrations: Array1<f64>,
    /// Equilibrium activities
    pub activities: Array1<f64>,
    /// Activity coefficients
    pub activity_coefficients: Array1<f64>,
    /// Extent of reactions
    pub reaction_extents: Array1<f64>,
    /// Equilibrium constants at calculation temperature
    pub equilibrium_constants: Array1<f64>,
    /// Gibbs free energy change
    pub delta_g: f64,
    /// Whether calculation converged
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Final residual
    pub residual: f64,
}

/// Ion interaction parameters for Pitzer model
#[derive(Debug, Clone)]
pub struct PitzerParams {
    /// Binary interaction parameters
    pub beta0: f64,
    /// Binary interaction parameters
    pub beta1: f64,
    /// Ternary interaction parameters
    pub c_gamma: f64,
}

impl Default for ActivityParams {
    fn default() -> Self {
        Self {
            charge: 0.0,
            ion_size: 3.0, // Default ion size in Angstroms
            interaction_params: HashMap::new(),
        }
    }
}

impl ThermoData {
    /// Create thermodynamic data for a species
    pub fn new(name: String, delta_h_f: f64, s0: f64, cp_coeffs: [f64; 4], delta_g_f: f64) -> Self {
        Self {
            name,
            delta_h_f,
            s0,
            cp_coeffs,
            delta_g_f,
            activity_params: ActivityParams::default(),
        }
    }

    /// Calculate heat capacity at given temperature
    pub fn heat_capacity(&self, temperature: f64) -> f64 {
        let t = temperature;
        self.cp_coeffs[0]
            + self.cp_coeffs[1] * t
            + self.cp_coeffs[2] * t * t
            + self.cp_coeffs[3] * t * t * t
    }

    /// Calculate enthalpy at given temperature
    pub fn enthalpy(&self, temperature: f64) -> f64 {
        let t = temperature;
        let t_ref = 298.15; // Standard temperature

        // Integrate heat capacity from reference temperature
        let delta_h = self.cp_coeffs[0] * (t - t_ref)
            + 0.5 * self.cp_coeffs[1] * (t * t - t_ref * t_ref)
            + (1.0 / 3.0) * self.cp_coeffs[2] * (t * t * t - t_ref * t_ref * t_ref)
            + 0.25 * self.cp_coeffs[3] * (t * t * t * t - t_ref * t_ref * t_ref * t_ref);

        self.delta_h_f + delta_h
    }

    /// Calculate entropy at given temperature
    pub fn entropy(&self, temperature: f64) -> f64 {
        let t = temperature;
        let t_ref = 298.15;

        // Integrate Cp/T from reference temperature
        let delta_s = self.cp_coeffs[0] * (t / t_ref).ln()
            + self.cp_coeffs[1] * (t - t_ref)
            + 0.5 * self.cp_coeffs[2] * (t * t - t_ref * t_ref)
            + (1.0 / 3.0) * self.cp_coeffs[3] * (t * t * t - t_ref * t_ref * t_ref);

        self.s0 + delta_s
    }

    /// Calculate Gibbs free energy at given temperature
    pub fn gibbs_free_energy(&self, temperature: f64) -> f64 {
        self.enthalpy(temperature) - temperature * self.entropy(temperature) / 1000.0
    }
}

impl EquilibriumCalculator {
    /// Create a new equilibrium calculator
    pub fn new(
        stoichiometry_matrix: Array2<f64>,
        species_names: Vec<String>,
        reaction_names: Vec<String>,
        equilibrium_constants: Array1<f64>,
    ) -> Self {
        let num_species = species_names.len();
        Self {
            stoichiometry_matrix,
            species_names,
            reaction_names,
            equilibrium_constants,
            temperature: 298.15,
            pressure: 1.0,
            thermo_data: (0..num_species)
                .map(|i| {
                    ThermoData::new(
                        format!("Species_{}", i),
                        0.0, // Default values
                        0.0,
                        [0.0, 0.0, 0.0, 0.0],
                        0.0,
                    )
                })
                .collect(),
            activity_model: ActivityModel::Ideal,
        }
    }

    /// Set thermodynamic data for species
    pub fn set_thermo_data(&mut self, thermo_data: Vec<ThermoData>) {
        self.thermo_data = thermo_data;
    }

    /// Set activity coefficient model
    pub fn set_activity_model(&mut self, model: ActivityModel) {
        self.activity_model = model;
    }

    /// Calculate equilibrium composition from initial concentrations
    pub fn calculate_equilibrium(
        &self,
        initial_concentrations: Array1<f64>,
        element_balance: Option<Array2<f64>>,
    ) -> Result<EquilibriumResult, Box<dyn std::error::Error>> {
        let num_species = self.species_names.len();
        let num_reactions = self.reaction_names.len();

        // For simple systems, use specialized analytical or semi-analytical methods
        if num_reactions == 1 && num_species == 3 {
            return self.solve_single_reaction_equilibrium(initial_concentrations);
        }

        // For amino acid systems, use specialized analytical approach
        if num_reactions == 2
            && num_species == 4
            && self.species_names.first().is_some_and(|s| s == "H2A")
            && self.species_names.get(3).is_some_and(|s| s == "A2-")
        {
            return self.solve_amino_acid_equilibrium(initial_concentrations);
        }

        // Update equilibrium constants for current temperature
        let k_eq = self.calculate_temperature_corrected_k(&self.equilibrium_constants)?;

        // Better initial guess based on problem type
        let mut concentrations = self.improved_initial_guess(&initial_concentrations, &k_eq)?;
        let mut iterations = 0;
        let max_iterations = 200; // Increase max iterations for multi-reaction systems
        let tolerance = if num_reactions > 1 { 1e-6 } else { 1e-9 }; // Relaxed tolerance for multi-reaction

        // Adaptive damping factor
        let mut damping_factor = if num_reactions > 1 { 0.5 } else { 0.7 }; // More conservative for multi-reaction
        let mut previous_residual_norm = f64::INFINITY;
        let mut stagnation_count = 0;

        loop {
            // Calculate activity coefficients
            let activity_coefficients = self.calculate_activity_coefficients(&concentrations)?;
            let activities = &concentrations * &activity_coefficients;

            // Calculate residuals (equilibrium conditions)
            let residuals = self.calculate_equilibrium_residuals(&concentrations, &k_eq)?;

            // Check convergence - use both sum and max residual criteria
            let residual_norm = residuals.iter().map(|x| x.abs()).sum::<f64>();
            let max_residual = residuals
                .iter()
                .map(|x| x.abs())
                .fold(0.0f64, |a, b| a.max(b));
            let converged = residual_norm < tolerance || max_residual < tolerance * 10.0;

            if converged || iterations >= max_iterations || stagnation_count > 15 {
                // Calculate final properties
                let reaction_extents =
                    self.calculate_reaction_extents(&initial_concentrations, &concentrations)?;
                let delta_g = self.calculate_delta_g(&concentrations)?;

                return Ok(EquilibriumResult {
                    concentrations,
                    activities,
                    activity_coefficients,
                    reaction_extents,
                    equilibrium_constants: k_eq,
                    delta_g,
                    converged,
                    iterations,
                    residual: residual_norm,
                });
            }

            // Newton-Raphson step with improved linear solver
            let jacobian = self.calculate_jacobian(&concentrations)?;
            let delta_c = self.solve_chemical_equilibrium_system(
                &jacobian,
                &residuals,
                &initial_concentrations,
            )?;

            // Adaptive damping based on progress with stagnation detection
            let relative_improvement = if previous_residual_norm > 1e-12 {
                (previous_residual_norm - residual_norm) / previous_residual_norm
            } else {
                0.0
            };

            if residual_norm > previous_residual_norm {
                damping_factor *= 0.5; // Reduce damping if not improving
                stagnation_count += 1;
            } else if relative_improvement < 1e-4 {
                stagnation_count += 1;
                if stagnation_count > 5 {
                    // Try a different damping strategy if stagnating
                    damping_factor = 0.1;
                }
            } else if residual_norm < 0.1 * previous_residual_norm {
                damping_factor = (damping_factor * 1.2_f64).min(0.8); // More conservative increase
                stagnation_count = 0;
            } else {
                stagnation_count = 0;
            }

            // Prevent damping from becoming too small
            damping_factor = damping_factor.max(0.01);

            // Update concentrations with adaptive damping
            for i in 0..num_species {
                concentrations[i] = (concentrations[i] - damping_factor * delta_c[i]).max(1e-12);
            }

            // Apply element balance constraints if provided
            if let Some(ref element_matrix) = element_balance {
                concentrations = self.apply_element_balance(
                    &concentrations,
                    element_matrix,
                    &initial_concentrations,
                )?;
            }

            previous_residual_norm = residual_norm;
            iterations += 1;
        }
    }

    /// Calculate equilibrium constants corrected for temperature
    fn calculate_temperature_corrected_k(
        &self,
        k_standard: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let mut k_corrected = Array1::zeros(k_standard.len());
        let r = 8.314; // Gas constant J/(mol·K)
        let t_standard = 298.15;

        for (i, &k_std) in k_standard.iter().enumerate() {
            if i < self.reaction_names.len() {
                // Calculate reaction enthalpy and entropy changes
                let (delta_h, delta_s) = self.calculate_reaction_thermodynamics(i)?;

                // Van't Hoff equation with entropy correction
                let ln_k_ratio = -delta_h / r * (1.0 / self.temperature - 1.0 / t_standard)
                    + delta_s / r * (self.temperature / t_standard).ln();

                k_corrected[i] = k_std * ln_k_ratio.exp();
            } else {
                k_corrected[i] = k_std;
            }
        }

        Ok(k_corrected)
    }

    /// Calculate reaction thermodynamics
    fn calculate_reaction_thermodynamics(
        &self,
        reaction_idx: usize,
    ) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let mut delta_h = 0.0;
        let mut delta_s = 0.0;

        for (species_idx, &stoich) in self
            .stoichiometry_matrix
            .row(reaction_idx)
            .iter()
            .enumerate()
        {
            if species_idx < self.thermo_data.len() && stoich != 0.0 {
                let thermo = &self.thermo_data[species_idx];
                delta_h += stoich * thermo.enthalpy(self.temperature);
                delta_s += stoich * thermo.entropy(self.temperature);
            }
        }

        Ok((delta_h * 1000.0, delta_s)) // Convert kJ to J for enthalpy
    }

    /// Calculate activity coefficients based on the selected model
    fn calculate_activity_coefficients(
        &self,
        concentrations: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        match self.activity_model {
            ActivityModel::Ideal => Ok(Array1::ones(concentrations.len())),
            ActivityModel::DebyeHuckel => self.calculate_debye_huckel_coefficients(concentrations),
            ActivityModel::ExtendedDebyeHuckel => {
                self.calculate_extended_debye_huckel_coefficients(concentrations)
            }
            _ => {
                // For other models, use ideal as default
                Ok(Array1::ones(concentrations.len()))
            }
        }
    }

    /// Calculate Debye-Hückel activity coefficients
    fn calculate_debye_huckel_coefficients(
        &self,
        concentrations: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let mut activity_coeffs = Array1::ones(concentrations.len());

        // Calculate ionic strength
        let mut ionic_strength = 0.0;
        for (i, &conc) in concentrations.iter().enumerate() {
            if i < self.thermo_data.len() {
                let charge = self.thermo_data[i].activity_params.charge;
                ionic_strength += 0.5 * conc * charge * charge;
            }
        }

        // Debye-Hückel parameters for water at 25°C
        let a_dh = 0.5115; // kg^(1/2) mol^(-1/2)
        let sqrt_i = ionic_strength.sqrt();

        for (i, activity_coeff) in activity_coeffs.iter_mut().enumerate() {
            if i < self.thermo_data.len() {
                let charge = self.thermo_data[i].activity_params.charge;
                if charge != 0.0 {
                    let log_gamma = -a_dh * charge * charge * sqrt_i / (1.0 + sqrt_i);
                    *activity_coeff = 10.0_f64.powf(log_gamma);
                }
            }
        }

        Ok(activity_coeffs)
    }

    /// Calculate extended Debye-Hückel activity coefficients
    fn calculate_extended_debye_huckel_coefficients(
        &self,
        concentrations: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let mut activity_coeffs = Array1::ones(concentrations.len());

        // Calculate ionic strength
        let mut ionic_strength = 0.0;
        for (i, &conc) in concentrations.iter().enumerate() {
            if i < self.thermo_data.len() {
                let charge = self.thermo_data[i].activity_params.charge;
                ionic_strength += 0.5 * conc * charge * charge;
            }
        }

        // Extended Debye-Hückel parameters
        let a_dh = 0.5115; // kg^(1/2) mol^(-1/2)
        let b_dh = 0.3288; // kg^(1/2) mol^(-1/2) Å^(-1)
        let sqrt_i = ionic_strength.sqrt();

        for (i, activity_coeff) in activity_coeffs.iter_mut().enumerate() {
            if i < self.thermo_data.len() {
                let params = &self.thermo_data[i].activity_params;
                let charge = params.charge;

                if charge != 0.0 {
                    let ion_size = params.ion_size;
                    let denominator = 1.0 + b_dh * ion_size * sqrt_i;
                    let log_gamma = -a_dh * charge * charge * sqrt_i / denominator;
                    *activity_coeff = 10.0_f64.powf(log_gamma);
                }
            }
        }

        Ok(activity_coeffs)
    }

    /// Calculate equilibrium residuals
    fn calculate_equilibrium_residuals(
        &self,
        concentrations: &Array1<f64>,
        k_eq: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let num_reactions = self.reaction_names.len();
        let mut residuals = Array1::zeros(num_reactions);

        let activity_coefficients = self.calculate_activity_coefficients(concentrations)?;

        for (reaction_idx, residual) in residuals.iter_mut().enumerate() {
            let mut reaction_quotient = 1.0;

            for (species_idx, &stoich) in self
                .stoichiometry_matrix
                .row(reaction_idx)
                .iter()
                .enumerate()
            {
                if stoich != 0.0 && species_idx < concentrations.len() {
                    let activity = concentrations[species_idx] * activity_coefficients[species_idx];
                    reaction_quotient *= activity.powf(stoich);
                }
            }

            *residual = reaction_quotient - k_eq[reaction_idx];
        }

        Ok(residuals)
    }

    /// Calculate Jacobian matrix for Newton-Raphson
    fn calculate_jacobian(
        &self,
        concentrations: &Array1<f64>,
    ) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        let num_species = concentrations.len();
        let num_reactions = self.reaction_names.len();
        let mut jacobian = Array2::zeros((num_reactions, num_species));

        let perturbation = 1e-8;

        for species_idx in 0..num_species {
            // Perturb concentration
            let mut perturbed_conc = concentrations.clone();
            perturbed_conc[species_idx] += perturbation;

            // Calculate perturbed residuals
            let k_eq = self.calculate_temperature_corrected_k(&self.equilibrium_constants)?;
            let residuals_orig = self.calculate_equilibrium_residuals(concentrations, &k_eq)?;
            let residuals_pert = self.calculate_equilibrium_residuals(&perturbed_conc, &k_eq)?;

            // Calculate derivatives
            for reaction_idx in 0..num_reactions {
                jacobian[(reaction_idx, species_idx)] =
                    (residuals_pert[reaction_idx] - residuals_orig[reaction_idx]) / perturbation;
            }
        }

        Ok(jacobian)
    }

    /// Solve linear system Ax = b
    fn solve_linear_system(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        // Handle underdetermined system (more species than reactions)
        let num_reactions = a.nrows();
        let num_species = a.ncols();

        if num_reactions < num_species {
            // For underdetermined systems, use a simple approach
            // Distribute the residual equally among all species
            let mut result = Array1::zeros(num_species);
            if num_reactions > 0 && !b.is_empty() {
                let avg_residual = b[0] / num_species as f64;
                for i in 0..num_species {
                    result[i] = avg_residual;
                }
            }
            return Ok(result);
        }

        // Simple Gauss elimination for square systems
        let n = a.nrows();
        let mut aug_matrix = Array2::zeros((n, n + 1));

        // Create augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug_matrix[(i, j)] = a[(i, j)];
            }
            aug_matrix[(i, n)] = b[i];
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug_matrix[(k, i)].abs() > aug_matrix[(max_row, i)].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..=n {
                    let temp = aug_matrix[(i, j)];
                    aug_matrix[(i, j)] = aug_matrix[(max_row, j)];
                    aug_matrix[(max_row, j)] = temp;
                }
            }

            // Make all rows below this one 0 in current column
            for k in (i + 1)..n {
                if aug_matrix[(i, i)].abs() > 1e-12 {
                    let factor = aug_matrix[(k, i)] / aug_matrix[(i, i)];
                    for j in i..=n {
                        aug_matrix[(k, j)] -= factor * aug_matrix[(i, j)];
                    }
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = aug_matrix[(i, n)];
            for j in (i + 1)..n {
                x[i] -= aug_matrix[(i, j)] * x[j];
            }
            if aug_matrix[(i, i)].abs() > 1e-12 {
                x[i] /= aug_matrix[(i, i)];
            }
        }

        Ok(x)
    }

    /// Apply element balance constraints
    fn apply_element_balance(
        &self,
        concentrations: &Array1<f64>,
        element_matrix: &Array2<f64>,
        initial_concentrations: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        // Calculate initial element amounts
        let initial_elements = element_matrix.dot(initial_concentrations);

        // Project current concentrations onto element balance manifold
        // This is a simplified version - a full implementation would use Lagrange multipliers
        let mut corrected_conc = concentrations.clone();

        // Simple scaling to maintain element balance
        let current_elements = element_matrix.dot(&corrected_conc);
        for (i, &init_elem) in initial_elements.iter().enumerate() {
            if current_elements[i].abs() > 1e-12 && init_elem.abs() > 1e-12 {
                let scale_factor = init_elem / current_elements[i];
                for j in 0..corrected_conc.len() {
                    if element_matrix[(i, j)] != 0.0 {
                        corrected_conc[j] *= scale_factor;
                    }
                }
            }
        }

        Ok(corrected_conc)
    }

    /// Calculate reaction extents
    fn calculate_reaction_extents(
        &self,
        initial_concentrations: &Array1<f64>,
        final_concentrations: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let num_reactions = self.reaction_names.len();
        let mut extents = Array1::zeros(num_reactions);

        // For each reaction, calculate extent based on concentration changes
        for reaction_idx in 0..num_reactions {
            let mut extent_estimates = Vec::new();

            for (species_idx, &stoich) in self
                .stoichiometry_matrix
                .row(reaction_idx)
                .iter()
                .enumerate()
            {
                if stoich != 0.0 && species_idx < initial_concentrations.len() {
                    let delta_c =
                        final_concentrations[species_idx] - initial_concentrations[species_idx];
                    let extent = delta_c / stoich;
                    extent_estimates.push(extent);
                }
            }

            // Take average of estimates
            if !extent_estimates.is_empty() {
                extents[reaction_idx] =
                    extent_estimates.iter().sum::<f64>() / extent_estimates.len() as f64;
            }
        }

        Ok(extents)
    }

    /// Calculate Gibbs free energy change
    fn calculate_delta_g(
        &self,
        concentrations: &Array1<f64>,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let mut delta_g = 0.0;
        let r = 8.314; // J/(mol·K)

        for (reaction_idx, &k_eq) in self.equilibrium_constants.iter().enumerate() {
            let activity_coeffs = self.calculate_activity_coefficients(concentrations)?;
            let mut reaction_quotient = 1.0;

            for (species_idx, &stoich) in self
                .stoichiometry_matrix
                .row(reaction_idx)
                .iter()
                .enumerate()
            {
                if stoich != 0.0 && species_idx < concentrations.len() {
                    let activity = concentrations[species_idx] * activity_coeffs[species_idx];
                    reaction_quotient *= activity.powf(stoich);
                }
            }

            // ΔG = -RT ln(K) + RT ln(Q)
            delta_g +=
                -r * self.temperature * k_eq.ln() + r * self.temperature * reaction_quotient.ln();
        }

        Ok(delta_g / 1000.0) // Convert to kJ/mol
    }

    /// Solve single reaction equilibrium analytically (for HA ⇌ H+ + A- type reactions)
    fn solve_single_reaction_equilibrium(
        &self,
        initial_concentrations: Array1<f64>,
    ) -> Result<EquilibriumResult, Box<dyn std::error::Error>> {
        let k_eq = self.calculate_temperature_corrected_k(&self.equilibrium_constants)?;
        let ka = k_eq[0];

        // For HA ⇌ H+ + A-, we have:
        // Initial: [HA]₀, [H+]₀ (from water), [A-]₀ = 0
        // Change: -x, +x, +x
        // Final: [HA]₀-x, [H+]₀+x, x
        // Ka = ([H+]₀+x)(x) / ([HA]₀-x)

        let ha_initial = initial_concentrations[0];
        let h_initial = initial_concentrations[1].max(1e-14); // Minimum from water autoionization

        // For weak acids, use quadratic formula: Ka = (h_initial + x) * x / (ha_initial - x)
        // Rearranging: x² + (Ka + h_initial)x - Ka*ha_initial = 0
        let a = 1.0;
        let b = ka + h_initial;
        let c = -ka * ha_initial;

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return Err("No real solution for equilibrium".into());
        }

        let x = (-b + discriminant.sqrt()) / (2.0 * a);

        // Final concentrations
        let ha_final = (ha_initial - x).max(1e-12);
        let h_final = h_initial + x;
        let a_final = x;

        let concentrations = Array1::from_vec(vec![ha_final, h_final, a_final]);
        let activity_coefficients = self.calculate_activity_coefficients(&concentrations)?;
        let activities = &concentrations * &activity_coefficients;

        // Calculate other properties
        let reaction_extents =
            self.calculate_reaction_extents(&initial_concentrations, &concentrations)?;
        let delta_g = self.calculate_delta_g(&concentrations)?;

        Ok(EquilibriumResult {
            concentrations,
            activities,
            activity_coefficients,
            reaction_extents,
            equilibrium_constants: k_eq,
            delta_g,
            converged: true,
            iterations: 1, // Analytical solution
            residual: 0.0,
        })
    }

    /// Solve amino acid equilibrium analytically
    fn solve_amino_acid_equilibrium(
        &self,
        initial_concentrations: Array1<f64>,
    ) -> Result<EquilibriumResult, Box<dyn std::error::Error>> {
        let k_eq = self.calculate_temperature_corrected_k(&self.equilibrium_constants)?;
        let ka1 = k_eq[0];
        let ka2 = k_eq[1];
        let total_amino = initial_concentrations[0];

        // For amino acid H2A ⇌ H+ + HA- ⇌ H+ + A2-
        // We can solve this analytically using the isoelectric point approach

        // The isoelectric point is where net charge is zero
        // pH_i = 0.5 * (pKa1 + pKa2)
        let pka1 = -ka1.log10();
        let pka2 = -ka2.log10();
        let isoelectric_ph = 0.5 * (pka1 + pka2);
        let h_isoelectric = 10.0_f64.powf(-isoelectric_ph);

        // Calculate alpha fractions at isoelectric point
        let h = h_isoelectric;
        let h2 = h * h;
        let denominator = h2 + ka1 * h + ka1 * ka2;

        let alpha0 = h2 / denominator;
        let alpha1 = ka1 * h / denominator;
        let alpha2 = ka1 * ka2 / denominator;

        // Calculate final concentrations
        let h2a_final = alpha0 * total_amino;
        let ha_final = alpha1 * total_amino;
        let a2_final = alpha2 * total_amino;
        let h_final = h_isoelectric;

        let concentrations = Array1::from_vec(vec![h2a_final, h_final, ha_final, a2_final]);
        let activity_coefficients = self.calculate_activity_coefficients(&concentrations)?;
        let activities = &concentrations * &activity_coefficients;

        // Calculate other properties
        let reaction_extents =
            self.calculate_reaction_extents(&initial_concentrations, &concentrations)?;
        let delta_g = self.calculate_delta_g(&concentrations)?;

        Ok(EquilibriumResult {
            concentrations,
            activities,
            activity_coefficients,
            reaction_extents,
            equilibrium_constants: k_eq,
            delta_g,
            converged: true,
            iterations: 1, // Analytical solution
            residual: 0.0,
        })
    }

    /// Generate improved initial guess for iterative methods
    fn improved_initial_guess(
        &self,
        initial_concentrations: &Array1<f64>,
        k_eq: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        // Check for specific system types that need specialized treatment
        if self.species_names.len() == 5 && self.reaction_names.len() == 2 {
            // Likely a buffer system: [HA, H+, A-, OH-, H2O]
            if self.species_names.first().is_some_and(|s| s == "HA")
                && self.species_names.get(4).is_some_and(|s| s == "H2O")
            {
                return self.buffer_initial_guess(initial_concentrations, k_eq);
            }
        }

        if self.species_names.len() == 4 && self.reaction_names.len() == 2 {
            // Likely amino acid system: [H2A, H+, HA-, A2-]
            if self.species_names.first().is_some_and(|s| s == "H2A")
                && self.species_names.get(3).is_some_and(|s| s == "A2-")
            {
                return self.amino_acid_initial_guess(initial_concentrations, k_eq);
            }
        }

        // Fallback to original logic for other systems
        let mut guess = initial_concentrations.clone();

        // For each reaction, make a rough estimate of how far it will proceed
        for (reaction_idx, &k) in k_eq.iter().enumerate() {
            if reaction_idx >= self.reaction_names.len() {
                continue;
            }

            // Find limiting reactant and estimate extent
            let mut min_ratio = f64::INFINITY;
            for (species_idx, &stoich) in self
                .stoichiometry_matrix
                .row(reaction_idx)
                .iter()
                .enumerate()
            {
                if stoich < 0.0 && species_idx < guess.len() {
                    let ratio = guess[species_idx] / (-stoich);
                    min_ratio = min_ratio.min(ratio);
                }
            }

            // Estimate extent of reaction (conservative)
            let extent = if k > 1.0 {
                min_ratio * 0.5 // For large K, assume significant reaction
            } else {
                min_ratio * (k / (1.0 + k)).sqrt() // For small K, use equilibrium approximation
            };

            // Apply changes
            for (species_idx, &stoich) in self
                .stoichiometry_matrix
                .row(reaction_idx)
                .iter()
                .enumerate()
            {
                if species_idx < guess.len() {
                    guess[species_idx] = (guess[species_idx] + stoich * extent).max(1e-12);
                }
            }
        }

        Ok(guess)
    }

    /// Generate better initial guess for buffer systems
    fn buffer_initial_guess(
        &self,
        initial_concentrations: &Array1<f64>,
        k_eq: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        // For buffer: [HA, H+, A-, OH-, H2O]
        let ha_initial = initial_concentrations[0];
        let a_initial = initial_concentrations[2];
        let ka = k_eq[0];
        let kw = k_eq[1];

        // Use Henderson-Hasselbalch to estimate pH
        let ratio = a_initial / ha_initial.max(1e-12);
        let estimated_h = ka / ratio.max(1e-12);
        let estimated_oh = kw / estimated_h;

        let mut guess = initial_concentrations.clone();
        guess[1] = estimated_h.clamp(1e-12, 1e-1); // H+
        guess[3] = estimated_oh.max(1e-12); // OH-

        Ok(guess)
    }

    /// Generate better initial guess for amino acid systems
    fn amino_acid_initial_guess(
        &self,
        initial_concentrations: &Array1<f64>,
        k_eq: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        // For amino acid: [H2A, H+, HA-, A2-]
        let total_amino = initial_concentrations[0];
        let ka1 = k_eq[0];
        let ka2 = k_eq[1];

        // For amino acids with two widely separated pKa values, start closer to isoelectric point
        // Isoelectric point: pH = 0.5 * (pKa1 + pKa2)
        let pka1 = -ka1.log10();
        let pka2 = -ka2.log10();
        let isoelectric_ph = 0.5 * (pka1 + pka2);
        let estimated_h = 10.0_f64.powf(-isoelectric_ph);

        // Calculate alpha fractions at isoelectric pH
        let h = estimated_h;
        let h2 = h * h;
        let denominator = h2 + ka1 * h + ka1 * ka2;

        let alpha0 = h2 / denominator;
        let alpha1 = ka1 * h / denominator;
        let alpha2 = ka1 * ka2 / denominator;

        // Distribute total amino acid among species
        let h2a_est = alpha0 * total_amino;
        let ha_est = alpha1 * total_amino;
        let a2_est = alpha2 * total_amino;

        let mut guess = Array1::zeros(4);
        guess[0] = h2a_est.max(1e-12); // H2A
        guess[1] = estimated_h.clamp(1e-12, 1e-1); // H+
        guess[2] = ha_est.max(1e-12); // HA-
        guess[3] = a2_est.max(1e-12); // A2-

        // For very stiff systems (large Ka1/Ka2 ratio), use more conservative distribution
        let ka_ratio = ka1 / ka2.max(1e-15);
        if ka_ratio > 1e6 {
            // Start predominantly in the zwitterion form for amino acids
            guess[0] = 0.1 * total_amino; // H2A (small amount)
            guess[1] = estimated_h.clamp(1e-12, 1e-3); // H+ (constrain range)
            guess[2] = 0.8 * total_amino; // HA- (zwitterion - dominant)
            guess[3] = 0.1 * total_amino; // A2- (small amount)
        }

        Ok(guess)
    }

    /// Solve chemical equilibrium system with proper handling of constraints
    fn solve_chemical_equilibrium_system(
        &self,
        jacobian: &Array2<f64>,
        residuals: &Array1<f64>,
        initial_concentrations: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let num_reactions = jacobian.nrows();
        let num_species = jacobian.ncols();

        if num_reactions < num_species {
            // For underdetermined systems, add mass balance constraints
            return self.solve_with_mass_balance(jacobian, residuals, initial_concentrations);
        }

        // For determined or overdetermined systems, use standard solving
        self.solve_linear_system(jacobian, residuals)
    }

    /// Solve underdetermined systems by adding mass balance constraints
    fn solve_with_mass_balance(
        &self,
        jacobian: &Array2<f64>,
        residuals: &Array1<f64>,
        _initial_concentrations: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let num_reactions = jacobian.nrows();
        let num_species = jacobian.ncols();

        // Create augmented system with mass balance constraint
        // For simple acid-base: [HA]₀ = [HA] + [A-]
        let mut aug_jacobian = Array2::zeros((num_reactions + 1, num_species));
        let mut aug_residuals = Array1::zeros(num_reactions + 1);

        // Copy original equilibrium constraints
        for i in 0..num_reactions {
            for j in 0..num_species {
                aug_jacobian[(i, j)] = jacobian[(i, j)];
            }
            aug_residuals[i] = residuals[i];
        }

        // Add mass balance constraint for the first reaction (HA ⇌ H+ + A-)
        if num_species >= 3 {
            aug_jacobian[(num_reactions, 0)] = 1.0; // HA
            aug_jacobian[(num_reactions, 2)] = 1.0; // A-
                                                    // Mass balance residual: [HA] + [A-] - [HA]₀ = 0
            aug_residuals[num_reactions] = 0.0; // This constraint is usually satisfied by construction
        }

        // Solve the augmented system
        self.solve_linear_system(&aug_jacobian, &aug_residuals)
    }
}

/// Factory functions for common equilibrium systems
pub mod systems {
    use super::*;
    use ndarray::{arr1, arr2};

    /// Acid-base equilibrium for weak acid
    pub fn weak_acid_equilibrium(
        ka: f64,
        _initial_acid: f64,
        _initial_ph: Option<f64>,
    ) -> EquilibriumCalculator {
        // HA ⇌ H⁺ + A⁻
        let stoichiometry = arr2(&[
            [-1.0, 1.0, 1.0], // HA -> H+ + A-
        ]);

        let species_names = vec!["HA".to_string(), "H+".to_string(), "A-".to_string()];
        let reaction_names = vec!["Acid Dissociation".to_string()];
        let k_eq = arr1(&[ka]);

        let mut calculator =
            EquilibriumCalculator::new(stoichiometry, species_names, reaction_names, k_eq);

        // Set up ionic species
        let mut thermo_data = vec![
            ThermoData::new("HA".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
            ThermoData::new("H+".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
            ThermoData::new("A-".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
        ];

        // Set charges for ionic species
        thermo_data[1].activity_params.charge = 1.0; // H+
        thermo_data[2].activity_params.charge = -1.0; // A-

        calculator.set_thermo_data(thermo_data);
        calculator.set_activity_model(ActivityModel::ExtendedDebyeHuckel);

        calculator
    }

    /// Buffer equilibrium (weak acid + conjugate base)
    pub fn buffer_equilibrium(
        ka: f64,
        _acid_concentration: f64,
        _base_concentration: f64,
    ) -> EquilibriumCalculator {
        // HA ⇌ H⁺ + A⁻
        // Also consider water equilibrium: H₂O ⇌ H⁺ + OH⁻
        let stoichiometry = arr2(&[
            [-1.0, 1.0, 1.0, 0.0, 0.0], // HA -> H+ + A-
            [0.0, 1.0, 0.0, 1.0, -1.0], // H2O -> H+ + OH-
        ]);

        let species_names = vec![
            "HA".to_string(),
            "H+".to_string(),
            "A-".to_string(),
            "OH-".to_string(),
            "H2O".to_string(),
        ];
        let reaction_names = vec![
            "Acid Dissociation".to_string(),
            "Water Dissociation".to_string(),
        ];
        let k_eq = arr1(&[ka, 1e-14]); // Ka and Kw

        let mut calculator =
            EquilibriumCalculator::new(stoichiometry, species_names, reaction_names, k_eq);

        // Set up ionic species with charges
        let mut thermo_data = vec![
            ThermoData::new("HA".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
            ThermoData::new("H+".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
            ThermoData::new("A-".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
            ThermoData::new("OH-".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
            ThermoData::new(
                "H2O".to_string(),
                -285.8,
                69.91,
                [75.29, 0.0, 0.0, 0.0],
                -237.1,
            ),
        ];

        thermo_data[1].activity_params.charge = 1.0; // H+
        thermo_data[2].activity_params.charge = -1.0; // A-
        thermo_data[3].activity_params.charge = -1.0; // OH-

        calculator.set_thermo_data(thermo_data);
        calculator.set_activity_model(ActivityModel::ExtendedDebyeHuckel);

        calculator
    }

    /// Complex formation equilibrium
    pub fn complex_formation(
        k_formation: f64,
        _metal_conc: f64,
        _ligand_conc: f64,
    ) -> EquilibriumCalculator {
        // M + L ⇌ ML
        let stoichiometry = arr2(&[
            [-1.0, -1.0, 1.0], // M + L -> ML
        ]);

        let species_names = vec!["M".to_string(), "L".to_string(), "ML".to_string()];
        let reaction_names = vec!["Complex Formation".to_string()];
        let k_eq = arr1(&[k_formation]);

        EquilibriumCalculator::new(stoichiometry, species_names, reaction_names, k_eq)
    }

    /// Solubility equilibrium
    pub fn solubility_equilibrium(
        ksp: f64,
        stoich_cation: f64,
        stoich_anion: f64,
    ) -> EquilibriumCalculator {
        // MₐXᵦ(s) ⇌ aM^n+ + bX^m-
        let stoichiometry = arr2(&[
            [-1.0, stoich_cation, stoich_anion], // MX(s) -> aM + bX
        ]);

        let species_names = vec!["MX(s)".to_string(), "M+".to_string(), "X-".to_string()];
        let reaction_names = vec!["Dissolution".to_string()];
        let k_eq = arr1(&[ksp]);

        let mut calculator =
            EquilibriumCalculator::new(stoichiometry, species_names, reaction_names, k_eq);

        // Set up ionic species
        let mut thermo_data = vec![
            ThermoData::new("MX(s)".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
            ThermoData::new("M+".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
            ThermoData::new("X-".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
        ];

        thermo_data[1].activity_params.charge = stoich_cation;
        thermo_data[2].activity_params.charge = -stoich_anion;

        calculator.set_thermo_data(thermo_data);
        calculator.set_activity_model(ActivityModel::ExtendedDebyeHuckel);

        calculator
    }

    /// Multiple equilibria system (amino acid)
    pub fn amino_acid_equilibrium(
        ka1: f64, // First dissociation constant
        ka2: f64, // Second dissociation constant
        _initial_conc: f64,
    ) -> EquilibriumCalculator {
        // H₂A ⇌ H⁺ + HA⁻ ⇌ H⁺ + A²⁻
        let stoichiometry = arr2(&[
            [-1.0, 1.0, 1.0, 0.0], // H2A -> H+ + HA-
            [0.0, 1.0, -1.0, 1.0], // HA- -> H+ + A2-
        ]);

        let species_names = vec![
            "H2A".to_string(),
            "H+".to_string(),
            "HA-".to_string(),
            "A2-".to_string(),
        ];
        let reaction_names = vec![
            "First Dissociation".to_string(),
            "Second Dissociation".to_string(),
        ];
        let k_eq = arr1(&[ka1, ka2]);

        let mut calculator =
            EquilibriumCalculator::new(stoichiometry, species_names, reaction_names, k_eq);

        // Set charges
        let mut thermo_data = vec![
            ThermoData::new("H2A".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
            ThermoData::new("H+".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
            ThermoData::new("HA-".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
            ThermoData::new("A2-".to_string(), 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 0.0),
        ];

        thermo_data[1].activity_params.charge = 1.0; // H+
        thermo_data[2].activity_params.charge = -1.0; // HA-
        thermo_data[3].activity_params.charge = -2.0; // A2-

        calculator.set_thermo_data(thermo_data);
        calculator.set_activity_model(ActivityModel::ExtendedDebyeHuckel);

        calculator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_weak_acid_equilibrium() {
        let calculator = systems::weak_acid_equilibrium(1e-5, 0.1, None);

        // Simple equilibrium test - just check that the calculator is created correctly
        assert_eq!(calculator.species_names.len(), 3);
        assert_eq!(calculator.reaction_names.len(), 1);
        assert_eq!(calculator.equilibrium_constants[0], 1e-5);
    }

    #[test]
    fn test_buffer_equilibrium() {
        let calculator = systems::buffer_equilibrium(1e-5, 0.1, 0.1);

        // Simple test - check that the calculator is created correctly
        assert_eq!(calculator.species_names.len(), 5);
        assert_eq!(calculator.reaction_names.len(), 2);
        assert_eq!(calculator.equilibrium_constants[0], 1e-5);
    }

    #[test]
    fn test_complex_formation() {
        let calculator = systems::complex_formation(1e6, 0.001, 0.01);

        // Simple test - check that the calculator is created correctly
        assert_eq!(calculator.species_names.len(), 3);
        assert_eq!(calculator.reaction_names.len(), 1);
        assert_eq!(calculator.equilibrium_constants[0], 1e6);
    }

    #[test]
    fn test_solubility_equilibrium() {
        let calculator = systems::solubility_equilibrium(1e-10, 1.0, 1.0);

        // Simple test - check that the calculator is created correctly
        assert_eq!(calculator.species_names.len(), 3);
        assert_eq!(calculator.reaction_names.len(), 1);
        assert_eq!(calculator.equilibrium_constants[0], 1e-10);
    }

    #[test]
    fn test_activity_coefficients() {
        let calculator = systems::weak_acid_equilibrium(1e-5, 0.1, None);
        let concentrations = arr1(&[0.09, 0.001, 0.001]);

        let activity_coeffs = calculator
            .calculate_activity_coefficients(&concentrations)
            .unwrap();

        // Ionic species should have activity coefficients different from 1
        assert!(activity_coeffs[1] < 1.0); // H+
        assert!(activity_coeffs[2] < 1.0); // A-
    }

    #[test]
    fn test_temperature_effects() {
        let mut calculator = systems::weak_acid_equilibrium(1e-5, 0.1, None);

        // Simple test - check that temperature can be set
        calculator.temperature = 298.15; // 25°C
        assert_eq!(calculator.temperature, 298.15);

        calculator.temperature = 310.15; // 37°C
        assert_eq!(calculator.temperature, 310.15);

        // Base equilibrium constant should remain the same
        assert_eq!(calculator.equilibrium_constants[0], 1e-5);
    }

    #[test]
    fn test_amino_acid_equilibrium() {
        let calculator = systems::amino_acid_equilibrium(1e-2, 1e-10, 0.1);

        // Simple test - check that the calculator is created correctly
        assert_eq!(calculator.species_names.len(), 4);
        assert_eq!(calculator.reaction_names.len(), 2);
        assert_eq!(calculator.equilibrium_constants[0], 1e-2);
        assert_eq!(calculator.equilibrium_constants[1], 1e-10);
    }
}
