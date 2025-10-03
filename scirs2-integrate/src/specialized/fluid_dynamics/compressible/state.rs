//! Compressible flow state representation and manipulation
//!
//! This module provides the core data structures for representing compressible fluid states
//! and methods for state manipulation and conversion between primitive and conservative variables.

use scirs2_core::ndarray::Array3;

/// Compressible fluid state representation
///
/// This structure holds all the field variables for a compressible fluid flow simulation,
/// including both conservative variables (density, momentum, energy) and derived quantities
/// (pressure, temperature, Mach number).
#[derive(Debug, Clone)]
pub struct CompressibleState {
    /// Density field (ρ)
    pub density: Array3<f64>,
    /// Momentum fields [ρu, ρv, ρw] - conservative momentum components
    pub momentum: Vec<Array3<f64>>,
    /// Total energy density field (E = ρe + ½ρ(u² + v² + w²))
    pub energy: Array3<f64>,
    /// Pressure field (derived quantity)
    pub pressure: Array3<f64>,
    /// Temperature field (derived quantity)
    pub temperature: Array3<f64>,
    /// Mach number field (derived quantity)
    pub mach: Array3<f64>,
}

impl CompressibleState {
    /// Create a new compressible state with given dimensions
    ///
    /// # Arguments
    /// * `nx` - Grid points in x-direction
    /// * `ny` - Grid points in y-direction
    /// * `nz` - Grid points in z-direction
    ///
    /// # Returns
    /// A new `CompressibleState` with initialized fields
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let density = Array3::ones((nx, ny, nz));
        let momentum = vec![
            Array3::zeros((nx, ny, nz)), // ρu
            Array3::zeros((nx, ny, nz)), // ρv
            Array3::zeros((nx, ny, nz)), // ρw
        ];
        let energy = Array3::from_elem((nx, ny, nz), 2.5); // Initial energy
        let pressure = Array3::ones((nx, ny, nz));
        let temperature = Array3::from_elem((nx, ny, nz), 300.0);
        let mach = Array3::zeros((nx, ny, nz));

        CompressibleState {
            density,
            momentum,
            energy,
            pressure,
            temperature,
            mach,
        }
    }

    /// Get the grid dimensions
    ///
    /// # Returns
    /// A tuple `(nx, ny, nz)` representing the grid dimensions
    pub fn dimensions(&self) -> (usize, usize, usize) {
        let (nx, ny, nz) = self.density.dim();
        (nx, ny, nz)
    }

    /// Extract velocity components from momentum and density
    ///
    /// # Returns
    /// A vector of velocity components [u, v, w]
    pub fn velocity_components(&self) -> Vec<Array3<f64>> {
        self.momentum
            .iter()
            .map(|mom| mom / &self.density)
            .collect()
    }

    /// Calculate kinetic energy density (½ρ(u² + v² + w²))
    ///
    /// # Returns
    /// An `Array3<f64>` containing the kinetic energy density at each grid point
    pub fn kinetic_energy_density(&self) -> Array3<f64> {
        let velocities = self.velocity_components();
        let u = &velocities[0];
        let v = &velocities[1];
        let w = &velocities[2];

        0.5 * &self.density * (u * u + v * v + w * w)
    }

    /// Calculate internal energy density (E - kinetic energy)
    ///
    /// # Returns
    /// An `Array3<f64>` containing the internal energy density at each grid point
    pub fn internal_energy_density(&self) -> Array3<f64> {
        &self.energy - &self.kinetic_energy_density()
    }

    /// Calculate sound speed field using ideal gas law
    ///
    /// # Arguments
    /// * `gamma` - Specific heat ratio (typically 1.4 for air)
    ///
    /// # Returns
    /// An `Array3<f64>` containing the sound speed at each grid point
    pub fn sound_speed(&self, gamma: f64) -> Array3<f64> {
        // c = √(γp/ρ)
        (gamma * &self.pressure / &self.density).mapv(|x| x.sqrt())
    }

    /// Update Mach number field based on current velocity and pressure
    ///
    /// # Arguments
    /// * `gamma` - Specific heat ratio
    pub fn update_mach_number(&mut self, gamma: f64) {
        let velocities = self.velocity_components();
        let u = &velocities[0];
        let v = &velocities[1];
        let w = &velocities[2];

        let velocity_magnitude = (u * u + v * v + w * w).mapv(|x| x.sqrt());
        let sound_speed = self.sound_speed(gamma);

        self.mach = velocity_magnitude / sound_speed;
    }

    /// Check if the state is physically valid
    ///
    /// # Returns
    /// `true` if density, pressure, and temperature are all positive everywhere
    pub fn is_physically_valid(&self) -> bool {
        // Check density is positive
        if self.density.iter().any(|&rho| rho <= 0.0) {
            return false;
        }

        // Check pressure is positive
        if self.pressure.iter().any(|&p| p <= 0.0) {
            return false;
        }

        // Check temperature is positive
        if self.temperature.iter().any(|&t| t <= 0.0) {
            return false;
        }

        // Check for NaN or infinite values
        if self.density.iter().any(|&x| !x.is_finite())
            || self.pressure.iter().any(|&x| !x.is_finite())
            || self.temperature.iter().any(|&x| !x.is_finite())
            || self.energy.iter().any(|&x| !x.is_finite())
        {
            return false;
        }

        for momentum_component in &self.momentum {
            if momentum_component.iter().any(|&x| !x.is_finite()) {
                return false;
            }
        }

        true
    }

    /// Set uniform initial conditions
    ///
    /// # Arguments
    /// * `rho0` - Initial density
    /// * `u0` - Initial x-velocity
    /// * `v0` - Initial y-velocity
    /// * `w0` - Initial z-velocity
    /// * `p0` - Initial pressure
    /// * `t0` - Initial temperature
    pub fn set_uniform_conditions(
        &mut self,
        rho0: f64,
        u0: f64,
        v0: f64,
        w0: f64,
        p0: f64,
        t0: f64,
    ) {
        self.density.fill(rho0);
        self.momentum[0].fill(rho0 * u0);
        self.momentum[1].fill(rho0 * v0);
        self.momentum[2].fill(rho0 * w0);
        self.pressure.fill(p0);
        self.temperature.fill(t0);

        // Calculate energy from pressure and kinetic energy
        let kinetic = 0.5 * rho0 * (u0 * u0 + v0 * v0 + w0 * w0);
        // For ideal gas: internal energy = p/(γ-1)
        let gamma = 1.4; // Default value for air
        let internal = p0 / (gamma - 1.0);
        self.energy.fill(internal + kinetic);

        // Calculate Mach number
        let c = (gamma * p0 / rho0).sqrt();
        let vel_mag = (u0 * u0 + v0 * v0 + w0 * w0).sqrt();
        self.mach.fill(vel_mag / c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = CompressibleState::new(10, 10, 10);
        assert_eq!(state.dimensions(), (10, 10, 10));
        assert_eq!(state.momentum.len(), 3);

        // Check initial conditions
        assert!(state.density.iter().all(|&x| x == 1.0));
        assert!(state.momentum[0].iter().all(|&x| x == 0.0));
        assert!(state.pressure.iter().all(|&x| x == 1.0));
        assert!(state.temperature.iter().all(|&x| x == 300.0));
    }

    #[test]
    fn test_velocity_components() {
        let mut state = CompressibleState::new(2, 2, 2);
        state.set_uniform_conditions(1.0, 10.0, 5.0, 2.0, 1.0, 300.0);

        let velocities = state.velocity_components();
        assert!(velocities[0].iter().all(|&x| (x - 10.0).abs() < 1e-10));
        assert!(velocities[1].iter().all(|&x| (x - 5.0).abs() < 1e-10));
        assert!(velocities[2].iter().all(|&x| (x - 2.0).abs() < 1e-10));
    }

    #[test]
    fn test_physical_validity() {
        let mut state = CompressibleState::new(5, 5, 5);
        assert!(state.is_physically_valid());

        // Make state invalid with negative density
        state.density[[0, 0, 0]] = -1.0;
        assert!(!state.is_physically_valid());
    }

    #[test]
    fn test_sound_speed() {
        let mut state = CompressibleState::new(2, 2, 2);
        state.set_uniform_conditions(1.0, 0.0, 0.0, 0.0, 1.0, 300.0);

        let gamma = 1.4;
        let c = state.sound_speed(gamma);
        let expected_c = (gamma * 1.0 / 1.0).sqrt();

        assert!(c.iter().all(|&x| (x - expected_c).abs() < 1e-10));
    }

    #[test]
    fn test_mach_number_update() {
        let mut state = CompressibleState::new(2, 2, 2);
        state.set_uniform_conditions(1.0, 10.0, 0.0, 0.0, 1.0, 300.0);

        let gamma = 1.4;
        state.update_mach_number(gamma);

        let expected_c = (gamma * 1.0 / 1.0).sqrt();
        let expected_mach = 10.0 / expected_c;

        assert!(state
            .mach
            .iter()
            .all(|&x| (x - expected_mach).abs() < 1e-10));
    }
}
