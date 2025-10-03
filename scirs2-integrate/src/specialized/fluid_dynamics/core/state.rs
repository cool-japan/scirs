//! Fluid state representations for 2D and 3D computational fluid dynamics.

use scirs2_core::ndarray::{Array2, Array3};

/// Fluid state representation for 2D fluid dynamics
#[derive(Debug, Clone)]
pub struct FluidState {
    /// Velocity field (u, v) for 2D
    pub velocity: Vec<Array2<f64>>,
    /// Pressure field
    pub pressure: Array2<f64>,
    /// Temperature field (optional)
    pub temperature: Option<Array2<f64>>,
    /// Time
    pub time: f64,
    /// Grid spacing
    pub dx: f64,
    pub dy: f64,
}

/// 3D fluid state representation for 3D fluid dynamics
#[derive(Debug, Clone)]
pub struct FluidState3D {
    /// Velocity field (u, v, w)
    pub velocity: Vec<Array3<f64>>,
    /// Pressure field
    pub pressure: Array3<f64>,
    /// Temperature field (optional)
    pub temperature: Option<Array3<f64>>,
    /// Time
    pub time: f64,
    /// Grid spacing
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl FluidState {
    /// Create a new 2D fluid state with the given dimensions and grid spacing
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        Self {
            velocity: vec![
                Array2::zeros((nx, ny)), // u-component
                Array2::zeros((nx, ny)), // v-component
            ],
            pressure: Array2::zeros((nx, ny)),
            temperature: None,
            time: 0.0,
            dx,
            dy,
        }
    }

    /// Create a new 2D fluid state with temperature field
    pub fn new_with_temperature(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        Self {
            velocity: vec![
                Array2::zeros((nx, ny)), // u-component
                Array2::zeros((nx, ny)), // v-component
            ],
            pressure: Array2::zeros((nx, ny)),
            temperature: Some(Array2::zeros((nx, ny))),
            time: 0.0,
            dx,
            dy,
        }
    }

    /// Get the grid dimensions (nx, ny)
    pub fn dimensions(&self) -> (usize, usize) {
        let shape = self.pressure.raw_dim();
        (shape[0], shape[1])
    }

    /// Check if the state has temperature field
    pub fn has_temperature(&self) -> bool {
        self.temperature.is_some()
    }
}

impl FluidState3D {
    /// Create a new 3D fluid state with the given dimensions and grid spacing
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        Self {
            velocity: vec![
                Array3::zeros((nx, ny, nz)), // u-component
                Array3::zeros((nx, ny, nz)), // v-component
                Array3::zeros((nx, ny, nz)), // w-component
            ],
            pressure: Array3::zeros((nx, ny, nz)),
            temperature: None,
            time: 0.0,
            dx,
            dy,
            dz,
        }
    }

    /// Create a new 3D fluid state with temperature field
    pub fn new_with_temperature(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> Self {
        Self {
            velocity: vec![
                Array3::zeros((nx, ny, nz)), // u-component
                Array3::zeros((nx, ny, nz)), // v-component
                Array3::zeros((nx, ny, nz)), // w-component
            ],
            pressure: Array3::zeros((nx, ny, nz)),
            temperature: Some(Array3::zeros((nx, ny, nz))),
            time: 0.0,
            dx,
            dy,
            dz,
        }
    }

    /// Get the grid dimensions (nx, ny, nz)
    pub fn dimensions(&self) -> (usize, usize, usize) {
        let shape = self.pressure.raw_dim();
        (shape[0], shape[1], shape[2])
    }

    /// Check if the state has temperature field
    pub fn has_temperature(&self) -> bool {
        self.temperature.is_some()
    }
}
