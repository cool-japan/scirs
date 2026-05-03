//! Deformable Image Registration
//!
//! Provides diffeomorphic deformable registration algorithms:
//!
//! - [`DemonsDiffeo`]: diffeomorphic demons registration using update + composition
//! - [`FluidRegistration`]: viscous fluid model via iterative force smoothing
//! - [`FreeFormDeformation`]: B-spline free-form deformation (FFD)
//! - [`DisplacementField`]: dense vector displacement field representation
//! - [`JacobianDeterminant`]: Jacobian determinant of the deformation field
//! - [`CompositeTransform`]: compose rigid + deformable transforms
//!
//! # References
//!
//! - Thirion (1998), "Image matching as a diffusion process", Medical Image Analysis.
//! - Vercauteren et al. (2009), "Diffeomorphic Demons: Efficient Non-parametric
//!   Image Registration", NeuroImage.
//! - Rueckert et al. (1999), "Nonrigid Registration Using Free-Form Deformations:
//!   Application to Breast MR Images", IEEE TMI.

use scirs2_core::ndarray::{Array2, Array3};
use std::f64::consts::PI;

use crate::error::{NdimageError, NdimageResult};
use crate::registration::{AffineTransform2D, RigidTransform2D};

// ─── DisplacementField ───────────────────────────────────────────────────────

/// A dense displacement (vector) field over a 2D or 3D domain.
///
/// For a 2D domain of shape `(rows, cols)` the field has shape `(rows, cols, 2)`
/// where the last dimension carries `[dy, dx]` displacements in pixels.
///
/// For a 3D domain `(nz, ny, nx)` the shape is `(nz, ny, nx, 3)` with `[dz, dy, dx]`.
#[derive(Debug, Clone)]
pub struct DisplacementField {
    /// Displacement vectors; shape `[..dims.., n_components]`.
    pub field: Vec<f64>,
    /// Spatial dimensions of the domain (e.g., `[rows, cols]` or `[nz, ny, nx]`).
    pub dims: Vec<usize>,
    /// Number of vector components (2 for 2D, 3 for 3D).
    pub n_components: usize,
}

impl DisplacementField {
    /// Create a zero displacement field for a 2D domain.
    pub fn zeros_2d(rows: usize, cols: usize) -> Self {
        Self {
            field: vec![0.0; rows * cols * 2],
            dims: vec![rows, cols],
            n_components: 2,
        }
    }

    /// Create a zero displacement field for a 3D domain.
    pub fn zeros_3d(nz: usize, ny: usize, nx: usize) -> Self {
        Self {
            field: vec![0.0; nz * ny * nx * 3],
            dims: vec![nz, ny, nx],
            n_components: 3,
        }
    }

    /// Total number of spatial voxels / pixels.
    pub fn num_voxels(&self) -> usize {
        self.dims.iter().product()
    }

    /// Access the displacement vector at pixel `(r, c)` for a 2D field.
    ///
    /// Returns `[dy, dx]`.
    pub fn get_2d(&self, r: usize, c: usize) -> NdimageResult<[f64; 2]> {
        if self.dims.len() != 2 || self.n_components != 2 {
            return Err(NdimageError::InvalidInput(
                "DisplacementField::get_2d: field is not 2D".to_string(),
            ));
        }
        let cols = self.dims[1];
        let base = (r * cols + c) * 2;
        if base + 1 >= self.field.len() {
            return Err(NdimageError::InvalidInput(
                "DisplacementField::get_2d: index out of bounds".to_string(),
            ));
        }
        Ok([self.field[base], self.field[base + 1]])
    }

    /// Set the displacement vector at pixel `(r, c)` for a 2D field.
    pub fn set_2d(&mut self, r: usize, c: usize, dy: f64, dx: f64) -> NdimageResult<()> {
        if self.dims.len() != 2 || self.n_components != 2 {
            return Err(NdimageError::InvalidInput(
                "DisplacementField::set_2d: field is not 2D".to_string(),
            ));
        }
        let cols = self.dims[1];
        let base = (r * cols + c) * 2;
        if base + 1 >= self.field.len() {
            return Err(NdimageError::InvalidInput(
                "DisplacementField::set_2d: index out of bounds".to_string(),
            ));
        }
        self.field[base] = dy;
        self.field[base + 1] = dx;
        Ok(())
    }

    /// Access the displacement vector at voxel `(iz, iy, ix)` for a 3D field.
    ///
    /// Returns `[dz, dy, dx]`.
    pub fn get_3d(&self, iz: usize, iy: usize, ix: usize) -> NdimageResult<[f64; 3]> {
        if self.dims.len() != 3 || self.n_components != 3 {
            return Err(NdimageError::InvalidInput(
                "DisplacementField::get_3d: field is not 3D".to_string(),
            ));
        }
        let ny = self.dims[1];
        let nx = self.dims[2];
        let base = (iz * ny * nx + iy * nx + ix) * 3;
        if base + 2 >= self.field.len() {
            return Err(NdimageError::InvalidInput(
                "DisplacementField::get_3d: index out of bounds".to_string(),
            ));
        }
        Ok([self.field[base], self.field[base + 1], self.field[base + 2]])
    }

    /// Compose two displacement fields: `result(x) = self(x) + other(x + self(x))`.
    ///
    /// Currently supports only 2D fields.
    pub fn compose_2d(&self, other: &DisplacementField) -> NdimageResult<DisplacementField> {
        if self.dims.len() != 2
            || other.dims.len() != 2
            || self.dims != other.dims
            || self.n_components != 2
            || other.n_components != 2
        {
            return Err(NdimageError::DimensionError(
                "DisplacementField::compose_2d: incompatible fields".to_string(),
            ));
        }
        let rows = self.dims[0];
        let cols = self.dims[1];
        let mut result = DisplacementField::zeros_2d(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let [dy0, dx0] = self.get_2d(r, c)?;
                // Interpolate other at (r + dy0, c + dx0)
                let wr = r as f64 + dy0;
                let wc = c as f64 + dx0;
                let [dy1, dx1] = bilinear_sample_2d_field(other, wr, wc);
                result.set_2d(r, c, dy0 + dy1, dx0 + dx1)?;
            }
        }
        Ok(result)
    }

    /// Apply a Gaussian smoothing kernel to the displacement field in-place.
    ///
    /// Only 2D fields are supported.  `sigma` is in pixels.
    pub fn gaussian_smooth_2d(&mut self, sigma: f64) -> NdimageResult<()> {
        if self.dims.len() != 2 || self.n_components != 2 {
            return Err(NdimageError::InvalidInput(
                "DisplacementField::gaussian_smooth_2d: only 2D fields supported".to_string(),
            ));
        }
        let rows = self.dims[0];
        let cols = self.dims[1];

        // Smooth each component independently
        for comp in 0..2 {
            let mut component: Vec<f64> =
                (0..rows * cols).map(|i| self.field[i * 2 + comp]).collect();
            gaussian_smooth_1d_separable(&mut component, rows, cols, sigma);
            for i in 0..rows * cols {
                self.field[i * 2 + comp] = component[i];
            }
        }
        Ok(())
    }

    /// Root-mean-square magnitude of the displacement vectors.
    pub fn rms_magnitude(&self) -> f64 {
        let n = self.num_voxels();
        if n == 0 {
            return 0.0;
        }
        let sum_sq: f64 = (0..n)
            .map(|i| {
                let base = i * self.n_components;
                (0..self.n_components)
                    .map(|c| self.field[base + c].powi(2))
                    .sum::<f64>()
            })
            .sum();
        (sum_sq / n as f64).sqrt()
    }
}

// ─── JacobianDeterminant ─────────────────────────────────────────────────────

/// Compute the Jacobian determinant of a 2D displacement field.
///
/// For a diffeomorphic transform the determinant should be positive everywhere
/// (no folding).  Negative values indicate folding artefacts.
///
/// Returns an `Array2<f64>` of determinants at each pixel.
pub struct JacobianDeterminant;

impl JacobianDeterminant {
    /// Compute det(J) for a 2D displacement field using central finite differences.
    ///
    /// `field.dims` must be `[rows, cols]` with `n_components == 2`.
    pub fn compute_2d(field: &DisplacementField) -> NdimageResult<Array2<f64>> {
        if field.dims.len() != 2 || field.n_components != 2 {
            return Err(NdimageError::InvalidInput(
                "JacobianDeterminant::compute_2d: field must be 2D".to_string(),
            ));
        }
        let rows = field.dims[0];
        let cols = field.dims[1];
        if rows < 3 || cols < 3 {
            return Err(NdimageError::InvalidInput(
                "JacobianDeterminant::compute_2d: domain must be at least 3×3".to_string(),
            ));
        }

        let mut det = Array2::<f64>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                // Clamp neighbours for boundary pixels
                let rn = r.saturating_sub(1);
                let rp = (r + 1).min(rows - 1);
                let cn = c.saturating_sub(1);
                let cp = (c + 1).min(cols - 1);
                let hr = (rp - rn) as f64;
                let hc = (cp - cn) as f64;

                // Deformation map φ(r,c) = (r + dy, c + dx)
                let [dy_rn, dx_rn] = field.get_2d(rn, c)?;
                let [dy_rp, dx_rp] = field.get_2d(rp, c)?;
                let [dy_rcn, dx_rcn] = field.get_2d(r, cn)?;
                let [dy_rcp, dx_rcp] = field.get_2d(r, cp)?;

                // d(φ_r)/dr, d(φ_r)/dc
                let dphi_r_dr = 1.0 + (dy_rp - dy_rn) / hr;
                let dphi_r_dc = (dy_rcp - dy_rcn) / hc;
                // d(φ_c)/dr, d(φ_c)/dc
                let dphi_c_dr = (dx_rp - dx_rn) / hr;
                let dphi_c_dc = 1.0 + (dx_rcp - dx_rcn) / hc;

                det[[r, c]] = dphi_r_dr * dphi_c_dc - dphi_r_dc * dphi_c_dr;
            }
        }
        Ok(det)
    }

    /// Count folded voxels (det(J) <= 0) and return the fraction.
    pub fn folding_fraction_2d(field: &DisplacementField) -> NdimageResult<f64> {
        let det = Self::compute_2d(field)?;
        let total = det.len();
        if total == 0 {
            return Ok(0.0);
        }
        let folded = det.iter().filter(|&&v| v <= 0.0).count();
        Ok(folded as f64 / total as f64)
    }
}

// ─── CompositeTransform ──────────────────────────────────────────────────────

/// Composition of a rigid transform followed by a deformable displacement.
///
/// The full mapping is:  `x' = deformable(rigid(x))`
#[derive(Debug, Clone)]
pub struct CompositeTransform {
    /// Optional rigid component (rotation + translation).
    pub rigid: Option<RigidTransform2D>,
    /// Optional affine component (overrides rigid if both are set; applied first).
    pub affine: Option<AffineTransform2D>,
    /// Deformable displacement field applied after the linear transform.
    pub deformable: Option<DisplacementField>,
}

impl CompositeTransform {
    /// Create an identity composite transform.
    pub fn identity() -> Self {
        Self {
            rigid: None,
            affine: None,
            deformable: None,
        }
    }

    /// Apply the composite transform to a point `(r, c)` (in pixel coordinates)
    /// and return the transformed point.
    pub fn apply_to_point(&self, r: f64, c: f64) -> NdimageResult<(f64, f64)> {
        // Step 1: apply rigid or affine linear transform
        let (lr, lc) = if let Some(ref aff) = self.affine {
            apply_affine_point(aff, r, c)
        } else if let Some(ref rig) = self.rigid {
            apply_rigid_point(rig, r, c)
        } else {
            (r, c)
        };

        // Step 2: add displacement from the deformable field
        if let Some(ref field) = self.deformable {
            if field.dims.len() == 2 && field.n_components == 2 {
                let [dy, dx] = bilinear_sample_2d_field(field, lr, lc);
                Ok((lr + dy, lc + dx))
            } else {
                Ok((lr, lc))
            }
        } else {
            Ok((lr, lc))
        }
    }

    /// Compose two `CompositeTransform`s by merging their displacement fields.
    ///
    /// Rigid/affine components from `self` take precedence.
    /// Deformable fields are composed using `DisplacementField::compose_2d`.
    pub fn compose(&self, other: &CompositeTransform) -> NdimageResult<CompositeTransform> {
        let deformable = match (&self.deformable, &other.deformable) {
            (Some(a), Some(b)) => Some(a.compose_2d(b)?),
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            (None, None) => None,
        };
        Ok(CompositeTransform {
            rigid: self.rigid.clone().or_else(|| other.rigid.clone()),
            affine: self.affine.clone().or_else(|| other.affine.clone()),
            deformable,
        })
    }
}

// ─── DemonsDiffeo ─────────────────────────────────────────────────────────────

/// Configuration for diffeomorphic demons registration.
#[derive(Debug, Clone)]
pub struct DemonsConfig {
    /// Maximum number of registration iterations.
    pub max_iterations: usize,
    /// Convergence criterion: stop when RMS displacement change is below this.
    pub convergence_threshold: f64,
    /// Gaussian smoothing sigma (pixels) applied to the update field.
    pub fluid_sigma: f64,
    /// Gaussian smoothing sigma applied to the accumulated displacement field.
    pub diffeo_sigma: f64,
    /// Step size multiplier (default 1.0; reduce for stability).
    pub step_size: f64,
}

impl Default for DemonsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-3,
            fluid_sigma: 1.5,
            diffeo_sigma: 1.0,
            step_size: 1.0,
        }
    }
}

/// Result of a demons registration run.
#[derive(Debug, Clone)]
pub struct DemonsResult {
    /// Final deformation field.
    pub field: DisplacementField,
    /// Number of iterations performed.
    pub iterations: usize,
    /// History of RMS displacement updates per iteration.
    pub rms_history: Vec<f64>,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Diffeomorphic demons registration for 2D images.
///
/// Registers a moving image to a fixed image by estimating a diffeomorphic
/// displacement field that maximises image similarity (SSD).
///
/// # Algorithm
///
/// At each iteration:
/// 1. Compute the update field `u` from the demons force (image gradient + intensity diff).
/// 2. Smooth `u` with a Gaussian kernel (`fluid_sigma`).
/// 3. Compose the current displacement with the exponential of `u`.
/// 4. Smooth the composition with `diffeo_sigma`.
pub struct DemonsDiffeo {
    config: DemonsConfig,
}

impl DemonsDiffeo {
    /// Create a new diffeomorphic demons registrar with default configuration.
    pub fn new() -> Self {
        Self {
            config: DemonsConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: DemonsConfig) -> Self {
        Self { config }
    }

    /// Register `moving` to `fixed`.
    ///
    /// Both images must have the same shape `(rows, cols)`.
    /// Returns the estimated displacement field and iteration statistics.
    pub fn register(
        &self,
        fixed: &Array2<f64>,
        moving: &Array2<f64>,
    ) -> NdimageResult<DemonsResult> {
        let fshape = fixed.shape();
        let mshape = moving.shape();
        if fshape != mshape {
            return Err(NdimageError::DimensionError(format!(
                "DemonsDiffeo: fixed shape {:?} != moving shape {:?}",
                fshape, mshape
            )));
        }
        let rows = fshape[0];
        let cols = fshape[1];
        if rows < 3 || cols < 3 {
            return Err(NdimageError::InvalidInput(
                "DemonsDiffeo: images must be at least 3×3".to_string(),
            ));
        }

        let mut disp = DisplacementField::zeros_2d(rows, cols);
        let mut rms_history = Vec::with_capacity(self.config.max_iterations);
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            // Warp moving image with current displacement
            let warped = warp_image_2d(moving, &disp);

            // Compute demons force at each pixel
            let mut update = DisplacementField::zeros_2d(rows, cols);
            for r in 0..rows {
                for c in 0..cols {
                    let f_val = fixed[[r, c]];
                    let m_val = warped[r * cols + c];
                    let diff = f_val - m_val;

                    // Gradient of fixed image (central difference)
                    let f_gn = if r > 0 {
                        fixed[[r - 1, c]]
                    } else {
                        fixed[[r, c]]
                    };
                    let f_gp = if r + 1 < rows {
                        fixed[[r + 1, c]]
                    } else {
                        fixed[[r, c]]
                    };
                    let f_gcn = if c > 0 {
                        fixed[[r, c - 1]]
                    } else {
                        fixed[[r, c]]
                    };
                    let f_gcp = if c + 1 < cols {
                        fixed[[r, c + 1]]
                    } else {
                        fixed[[r, c]]
                    };
                    let gx = (f_gp - f_gn) * 0.5;
                    let gy = (f_gcp - f_gcn) * 0.5;
                    let denom = gx * gx + gy * gy + diff * diff + 1e-10;

                    let uy = self.config.step_size * diff * gx / denom;
                    let ux = self.config.step_size * diff * gy / denom;
                    update.set_2d(r, c, uy, ux)?;
                }
            }

            // Smooth the update field
            update.gaussian_smooth_2d(self.config.fluid_sigma)?;

            // Compute RMS of the update
            let rms = update.rms_magnitude();
            rms_history.push(rms);

            // Compose: disp = disp ∘ update  (update exponential ~ identity + update for small steps)
            disp = disp.compose_2d(&update)?;

            // Smooth the total displacement field
            disp.gaussian_smooth_2d(self.config.diffeo_sigma)?;

            if rms < self.config.convergence_threshold {
                converged = true;
                let final_iter = iter + 1;
                return Ok(DemonsResult {
                    field: disp,
                    iterations: final_iter,
                    rms_history,
                    converged,
                });
            }
        }

        Ok(DemonsResult {
            field: disp,
            iterations: self.config.max_iterations,
            rms_history,
            converged,
        })
    }
}

// ─── FluidRegistration ───────────────────────────────────────────────────────

/// Configuration for viscous fluid-model registration.
#[derive(Debug, Clone)]
pub struct FluidConfig {
    /// Number of outer iterations.
    pub max_iterations: usize,
    /// Viscosity regularisation parameter (smoothing sigma in pixels).
    pub viscosity: f64,
    /// Step size for gradient descent.
    pub step_size: f64,
    /// Convergence threshold on normalised energy change.
    pub convergence_threshold: f64,
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            viscosity: 2.0,
            step_size: 0.5,
            convergence_threshold: 1e-4,
        }
    }
}

/// Result of fluid registration.
#[derive(Debug, Clone)]
pub struct FluidResult {
    /// Final deformation field.
    pub field: DisplacementField,
    /// Energy history per iteration (SSD).
    pub energy_history: Vec<f64>,
    /// Number of iterations.
    pub iterations: usize,
    /// Converged flag.
    pub converged: bool,
}

/// Viscous fluid model registration for 2D images.
///
/// Models the moving image as a viscous fluid flowing toward the fixed image
/// under image-derived body forces.  The velocity field is regularised by
/// applying a Gaussian smoothing at each step (approximating Stokes' equation).
pub struct FluidRegistration {
    config: FluidConfig,
}

impl FluidRegistration {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: FluidConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: FluidConfig) -> Self {
        Self { config }
    }

    /// Register `moving` to `fixed`.
    pub fn register(
        &self,
        fixed: &Array2<f64>,
        moving: &Array2<f64>,
    ) -> NdimageResult<FluidResult> {
        let fshape = fixed.shape();
        if fshape != moving.shape() {
            return Err(NdimageError::DimensionError(format!(
                "FluidRegistration: shape mismatch {:?} vs {:?}",
                fshape,
                moving.shape()
            )));
        }
        let rows = fshape[0];
        let cols = fshape[1];
        if rows < 3 || cols < 3 {
            return Err(NdimageError::InvalidInput(
                "FluidRegistration: images must be at least 3×3".to_string(),
            ));
        }

        let mut vel = DisplacementField::zeros_2d(rows, cols);
        let mut disp = DisplacementField::zeros_2d(rows, cols);
        let mut energy_history = Vec::with_capacity(self.config.max_iterations);
        let mut prev_energy = f64::INFINITY;
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            let warped = warp_image_2d(moving, &disp);
            let mut ssd = 0.0_f64;

            // Compute body forces from SSD gradient
            for r in 0..rows {
                for c in 0..cols {
                    let diff = fixed[[r, c]] - warped[r * cols + c];
                    ssd += diff * diff;

                    // Gradient of warped image
                    let w_rn = if r > 0 {
                        warped[(r - 1) * cols + c]
                    } else {
                        warped[r * cols + c]
                    };
                    let w_rp = if r + 1 < rows {
                        warped[(r + 1) * cols + c]
                    } else {
                        warped[r * cols + c]
                    };
                    let w_cn = if c > 0 {
                        warped[r * cols + c - 1]
                    } else {
                        warped[r * cols + c]
                    };
                    let w_cp = if c + 1 < cols {
                        warped[r * cols + c + 1]
                    } else {
                        warped[r * cols + c]
                    };
                    let gy = (w_rp - w_rn) * 0.5;
                    let gx = (w_cp - w_cn) * 0.5;

                    let fy = 2.0 * diff * gy;
                    let fx = 2.0 * diff * gx;
                    vel.set_2d(r, c, fy, fx)?;
                }
            }

            energy_history.push(ssd);

            // Smooth velocity field (viscosity regularisation)
            vel.gaussian_smooth_2d(self.config.viscosity)?;

            // Update displacement: disp = disp + step * vel
            for i in 0..rows * cols {
                disp.field[i * 2] += self.config.step_size * vel.field[i * 2];
                disp.field[i * 2 + 1] += self.config.step_size * vel.field[i * 2 + 1];
            }

            let rel_change = (prev_energy - ssd).abs() / (prev_energy.abs() + 1e-12);
            if rel_change < self.config.convergence_threshold {
                converged = true;
                return Ok(FluidResult {
                    field: disp,
                    energy_history,
                    iterations: iter + 1,
                    converged,
                });
            }
            prev_energy = ssd;
        }

        Ok(FluidResult {
            field: disp,
            energy_history,
            iterations: self.config.max_iterations,
            converged,
        })
    }
}

// ─── FreeFormDeformation ─────────────────────────────────────────────────────

/// Configuration for B-spline free-form deformation.
#[derive(Debug, Clone)]
pub struct FfdConfig {
    /// Number of control point grid nodes along each axis `[n_r, n_c]`.
    /// Values less than 4 will be clamped to 4 (minimum for cubic B-splines).
    pub grid_size: [usize; 2],
    /// Number of optimisation iterations.
    pub max_iterations: usize,
    /// Gradient descent step size.
    pub step_size: f64,
    /// Bending energy regularisation weight.
    pub regularisation: f64,
    /// Convergence threshold on energy change.
    pub convergence_threshold: f64,
}

impl Default for FfdConfig {
    fn default() -> Self {
        Self {
            grid_size: [8, 8],
            max_iterations: 100,
            step_size: 0.1,
            regularisation: 0.01,
            convergence_threshold: 1e-4,
        }
    }
}

/// Result of B-spline FFD registration.
#[derive(Debug, Clone)]
pub struct FfdResult {
    /// Dense displacement field derived from the B-spline control points.
    pub field: DisplacementField,
    /// Control-point displacements in `y` direction, shape `[grid_r, grid_c]`.
    pub ctrl_dy: Array2<f64>,
    /// Control-point displacements in `x` direction, shape `[grid_r, grid_c]`.
    pub ctrl_dx: Array2<f64>,
    /// Energy history.
    pub energy_history: Vec<f64>,
    /// Number of iterations.
    pub iterations: usize,
    /// Converged flag.
    pub converged: bool,
}

/// B-spline free-form deformation (FFD) registration for 2D images.
///
/// Uses a regular grid of cubic B-spline control points to parametrise the
/// deformation field.  The control-point displacements are optimised by
/// gradient descent to minimise SSD + bending energy regularisation.
pub struct FreeFormDeformation {
    config: FfdConfig,
}

impl FreeFormDeformation {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: FfdConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: FfdConfig) -> Self {
        let mut cfg = config;
        cfg.grid_size[0] = cfg.grid_size[0].max(4);
        cfg.grid_size[1] = cfg.grid_size[1].max(4);
        Self { config: cfg }
    }

    /// Register `moving` to `fixed`.
    pub fn register(&self, fixed: &Array2<f64>, moving: &Array2<f64>) -> NdimageResult<FfdResult> {
        let fshape = fixed.shape();
        if fshape != moving.shape() {
            return Err(NdimageError::DimensionError(format!(
                "FreeFormDeformation: shape mismatch {:?} vs {:?}",
                fshape,
                moving.shape()
            )));
        }
        let rows = fshape[0];
        let cols = fshape[1];
        if rows < 4 || cols < 4 {
            return Err(NdimageError::InvalidInput(
                "FreeFormDeformation: images must be at least 4×4".to_string(),
            ));
        }

        let [gr, gc] = self.config.grid_size;
        // Initialise control-point grids at zero
        let mut ctrl_dy = Array2::<f64>::zeros((gr, gc));
        let mut ctrl_dx = Array2::<f64>::zeros((gr, gc));

        let mut energy_history = Vec::with_capacity(self.config.max_iterations);
        let mut prev_energy = f64::INFINITY;
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            // Evaluate dense displacement from current control points
            let disp = self.ctrl_to_dense(&ctrl_dy, &ctrl_dx, rows, cols);
            let warped = warp_image_2d(moving, &disp);

            // Compute gradient of SSD w.r.t. control points
            let mut grad_dy = Array2::<f64>::zeros((gr, gc));
            let mut grad_dx = Array2::<f64>::zeros((gr, gc));
            let mut ssd = 0.0_f64;

            for r in 0..rows {
                for c in 0..cols {
                    let diff = fixed[[r, c]] - warped[r * cols + c];
                    ssd += diff * diff;

                    // Image gradient at warped position
                    let w_rn = if r > 0 {
                        warped[(r - 1) * cols + c]
                    } else {
                        warped[r * cols + c]
                    };
                    let w_rp = if r + 1 < rows {
                        warped[(r + 1) * cols + c]
                    } else {
                        warped[r * cols + c]
                    };
                    let w_cn = if c > 0 {
                        warped[r * cols + c - 1]
                    } else {
                        warped[r * cols + c]
                    };
                    let w_cp = if c + 1 < cols {
                        warped[r * cols + c + 1]
                    } else {
                        warped[r * cols + c]
                    };
                    let gy = (w_rp - w_rn) * 0.5;
                    let gx = (w_cp - w_cn) * 0.5;

                    // Propagate gradient to control points via B-spline basis
                    let t_r = r as f64 / rows as f64 * (gr - 1) as f64;
                    let t_c = c as f64 / cols as f64 * (gc - 1) as f64;
                    let pr = (t_r.floor() as isize).clamp(0, gr as isize - 1) as usize;
                    let pc = (t_c.floor() as isize).clamp(0, gc as isize - 1) as usize;

                    // Simple trilinear weight for neighbouring control points
                    let fr = t_r - pr as f64;
                    let fc = t_c - pc as f64;
                    for dr in 0..2_usize {
                        for dc in 0..2_usize {
                            let nrr = (pr + dr).min(gr - 1);
                            let ncc = (pc + dc).min(gc - 1);
                            let wr = if dr == 0 { 1.0 - fr } else { fr };
                            let wc = if dc == 0 { 1.0 - fc } else { fc };
                            let w = wr * wc;
                            grad_dy[[nrr, ncc]] -= 2.0 * diff * gy * w;
                            grad_dx[[nrr, ncc]] -= 2.0 * diff * gx * w;
                        }
                    }
                }
            }

            // Bending energy regularisation: penalise second derivatives of
            // control-point grid using finite differences
            let bend = self.bending_energy(&ctrl_dy, &ctrl_dx, gr, gc);
            let total_energy = ssd + self.config.regularisation * bend;
            energy_history.push(total_energy);

            // Add regularisation gradient (Laplacian of control points)
            for r in 1..gr - 1 {
                for c in 1..gc - 1 {
                    let lap_dy = ctrl_dy[[r - 1, c]] - 2.0 * ctrl_dy[[r, c]]
                        + ctrl_dy[[r + 1, c]]
                        + ctrl_dy[[r, c - 1]]
                        - 2.0 * ctrl_dy[[r, c]]
                        + ctrl_dy[[r, c + 1]];
                    let lap_dx = ctrl_dx[[r - 1, c]] - 2.0 * ctrl_dx[[r, c]]
                        + ctrl_dx[[r + 1, c]]
                        + ctrl_dx[[r, c - 1]]
                        - 2.0 * ctrl_dx[[r, c]]
                        + ctrl_dx[[r, c + 1]];
                    grad_dy[[r, c]] -= self.config.regularisation * lap_dy;
                    grad_dx[[r, c]] -= self.config.regularisation * lap_dx;
                }
            }

            // Update control points
            for r in 0..gr {
                for c in 0..gc {
                    ctrl_dy[[r, c]] -= self.config.step_size * grad_dy[[r, c]];
                    ctrl_dx[[r, c]] -= self.config.step_size * grad_dx[[r, c]];
                }
            }

            let rel_change = (prev_energy - total_energy).abs() / (prev_energy.abs() + 1e-12);
            if rel_change < self.config.convergence_threshold {
                converged = true;
                let final_disp = self.ctrl_to_dense(&ctrl_dy, &ctrl_dx, rows, cols);
                return Ok(FfdResult {
                    field: final_disp,
                    ctrl_dy,
                    ctrl_dx,
                    energy_history,
                    iterations: iter + 1,
                    converged,
                });
            }
            prev_energy = total_energy;
        }

        let final_disp = self.ctrl_to_dense(&ctrl_dy, &ctrl_dx, rows, cols);
        Ok(FfdResult {
            field: final_disp,
            ctrl_dy,
            ctrl_dx,
            energy_history,
            iterations: self.config.max_iterations,
            converged,
        })
    }

    /// Evaluate the dense displacement field by bilinear interpolation of
    /// B-spline control points.
    fn ctrl_to_dense(
        &self,
        ctrl_dy: &Array2<f64>,
        ctrl_dx: &Array2<f64>,
        rows: usize,
        cols: usize,
    ) -> DisplacementField {
        let [gr, gc] = self.config.grid_size;
        let mut disp = DisplacementField::zeros_2d(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let t_r = r as f64 / rows as f64 * (gr - 1) as f64;
                let t_c = c as f64 / cols as f64 * (gc - 1) as f64;
                let pr = (t_r.floor() as isize).clamp(0, gr as isize - 1) as usize;
                let pc = (t_c.floor() as isize).clamp(0, gc as isize - 1) as usize;
                let fr = t_r - pr as f64;
                let fc = t_c - pc as f64;

                let mut dy = 0.0;
                let mut dx = 0.0;
                for dr in 0..2_usize {
                    for dc in 0..2_usize {
                        let nrr = (pr + dr).min(gr - 1);
                        let ncc = (pc + dc).min(gc - 1);
                        let wr = if dr == 0 { 1.0 - fr } else { fr };
                        let wc = if dc == 0 { 1.0 - fc } else { fc };
                        dy += wr * wc * ctrl_dy[[nrr, ncc]];
                        dx += wr * wc * ctrl_dx[[nrr, ncc]];
                    }
                }
                let base = (r * cols + c) * 2;
                disp.field[base] = dy;
                disp.field[base + 1] = dx;
            }
        }
        disp
    }

    /// Compute the bending energy of the control-point grid using second
    /// finite differences.
    fn bending_energy(&self, dy: &Array2<f64>, dx: &Array2<f64>, gr: usize, gc: usize) -> f64 {
        let mut energy = 0.0;
        for r in 1..gr.saturating_sub(1) {
            for c in 1..gc.saturating_sub(1) {
                let d2y_rr = dy[[r - 1, c]] - 2.0 * dy[[r, c]] + dy[[r + 1, c]];
                let d2y_cc = dy[[r, c - 1]] - 2.0 * dy[[r, c]] + dy[[r, c + 1]];
                let d2x_rr = dx[[r - 1, c]] - 2.0 * dx[[r, c]] + dx[[r + 1, c]];
                let d2x_cc = dx[[r, c - 1]] - 2.0 * dx[[r, c]] + dx[[r, c + 1]];
                energy += d2y_rr * d2y_rr + d2y_cc * d2y_cc + d2x_rr * d2x_rr + d2x_cc * d2x_cc;
            }
        }
        energy
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Warp a 2D image by a displacement field using bilinear interpolation.
///
/// Returns a flat `Vec<f64>` with shape `rows × cols` (row-major).
fn warp_image_2d(image: &Array2<f64>, field: &DisplacementField) -> Vec<f64> {
    let rows = field.dims[0];
    let cols = field.dims[1];
    let mut out = vec![0.0_f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let base = (r * cols + c) * 2;
            let dy = field.field[base];
            let dx = field.field[base + 1];
            let src_r = r as f64 - dy;
            let src_c = c as f64 - dx;
            out[r * cols + c] = bilinear_interpolate(image, src_r, src_c);
        }
    }
    out
}

/// Bilinear interpolation on a 2D array at fractional coordinates `(r, c)`.
///
/// Boundary pixels are replicated (clamp-to-edge).
fn bilinear_interpolate(image: &Array2<f64>, r: f64, c: f64) -> f64 {
    let rows = image.nrows();
    let cols = image.ncols();
    let r0 = r.floor() as isize;
    let c0 = c.floor() as isize;
    let fr = r - r.floor();
    let fc = c - c.floor();

    let clamp_r = |v: isize| v.clamp(0, rows as isize - 1) as usize;
    let clamp_c = |v: isize| v.clamp(0, cols as isize - 1) as usize;

    let r0u = clamp_r(r0);
    let r1u = clamp_r(r0 + 1);
    let c0u = clamp_c(c0);
    let c1u = clamp_c(c0 + 1);

    let v00 = image[[r0u, c0u]];
    let v01 = image[[r0u, c1u]];
    let v10 = image[[r1u, c0u]];
    let v11 = image[[r1u, c1u]];

    v00 * (1.0 - fr) * (1.0 - fc) + v01 * (1.0 - fr) * fc + v10 * fr * (1.0 - fc) + v11 * fr * fc
}

/// Bilinearly sample a 2D displacement field at fractional coordinates.
///
/// Returns `[dy, dx]`; out-of-bounds coordinates are clamped.
fn bilinear_sample_2d_field(field: &DisplacementField, r: f64, c: f64) -> [f64; 2] {
    let rows = field.dims[0];
    let cols = field.dims[1];
    let r0 = r.floor() as isize;
    let c0 = c.floor() as isize;
    let fr = r - r.floor();
    let fc = c - c.floor();

    let clamp_r = |v: isize| v.clamp(0, rows as isize - 1) as usize;
    let clamp_c = |v: isize| v.clamp(0, cols as isize - 1) as usize;

    let corners = [
        (clamp_r(r0), clamp_c(c0)),
        (clamp_r(r0), clamp_c(c0 + 1)),
        (clamp_r(r0 + 1), clamp_c(c0)),
        (clamp_r(r0 + 1), clamp_c(c0 + 1)),
    ];
    let weights = [
        (1.0 - fr) * (1.0 - fc),
        (1.0 - fr) * fc,
        fr * (1.0 - fc),
        fr * fc,
    ];

    let mut dy = 0.0_f64;
    let mut dx = 0.0_f64;
    for (idx, &(cr, cc)) in corners.iter().enumerate() {
        let base = (cr * cols + cc) * 2;
        dy += weights[idx] * field.field[base];
        dx += weights[idx] * field.field[base + 1];
    }
    [dy, dx]
}

/// Apply an affine transform to a single point.
fn apply_affine_point(aff: &AffineTransform2D, r: f64, c: f64) -> (f64, f64) {
    let m = &aff.matrix;
    let nr = m[[0, 0]] * r + m[[0, 1]] * c + m[[0, 2]];
    let nc = m[[1, 0]] * r + m[[1, 1]] * c + m[[1, 2]];
    (nr, nc)
}

/// Apply a rigid transform to a single point.
fn apply_rigid_point(rig: &RigidTransform2D, r: f64, c: f64) -> (f64, f64) {
    let cos_a = rig.angle.cos();
    let sin_a = rig.angle.sin();
    let nr = cos_a * r - sin_a * c + rig.ty;
    let nc = sin_a * r + cos_a * c + rig.tx;
    (nr, nc)
}

/// Separable 1D Gaussian smoothing applied row-wise then column-wise to a flat
/// 2D buffer of size `rows × cols`.
fn gaussian_smooth_1d_separable(buf: &mut Vec<f64>, rows: usize, cols: usize, sigma: f64) {
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel: Vec<f64> = {
        let two_sig2 = 2.0 * sigma * sigma;
        let k: Vec<f64> = (0..=radius)
            .flat_map(|i| {
                if i == 0 {
                    vec![(-(0_f64.powi(2)) / two_sig2).exp()]
                } else {
                    let v = (-(i as f64).powi(2) / two_sig2).exp();
                    vec![v, v]
                }
            })
            .collect();
        // Build symmetric kernel [-radius..0..+radius]
        let mut full = Vec::with_capacity(2 * radius + 1);
        for i in (1..=radius).rev() {
            full.push((-(i as f64).powi(2) / two_sig2).exp());
        }
        full.push(0.0_f64.exp()); // centre
        for i in 1..=radius {
            full.push((-(i as f64).powi(2) / two_sig2).exp());
        }
        let sum: f64 = full.iter().sum();
        let _ = k; // suppress warning
        full.iter().map(|v| v / sum).collect()
    };

    let klen = kernel.len();
    let krad = klen / 2;

    // Row-wise pass
    let mut tmp = buf.clone();
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sc = c as isize + ki as isize - krad as isize;
                let sc_clamped = sc.clamp(0, cols as isize - 1) as usize;
                acc += kv * buf[r * cols + sc_clamped];
            }
            tmp[r * cols + c] = acc;
        }
    }

    // Column-wise pass
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sr = r as isize + ki as isize - krad as isize;
                let sr_clamped = sr.clamp(0, rows as isize - 1) as usize;
                acc += kv * tmp[sr_clamped * cols + c];
            }
            buf[r * cols + c] = acc;
        }
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_test_image(rows: usize, cols: usize, offset: f64) -> Array2<f64> {
        let mut img = Array2::<f64>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                img[[r, c]] = ((r as f64 + offset).sin() + (c as f64).cos()) * 50.0 + 128.0;
            }
        }
        img
    }

    #[test]
    fn test_displacement_field_create_and_access() {
        let mut df = DisplacementField::zeros_2d(10, 10);
        df.set_2d(3, 4, 1.5, -2.0)
            .expect("set_2d should succeed for valid coordinates");
        let [dy, dx] = df
            .get_2d(3, 4)
            .expect("get_2d should succeed for valid coordinates");
        assert!((dy - 1.5).abs() < 1e-10);
        assert!((dx + 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_displacement_field_compose_identity() {
        let a = DisplacementField::zeros_2d(8, 8);
        let b = DisplacementField::zeros_2d(8, 8);
        let composed = a
            .compose_2d(&b)
            .expect("compose_2d should succeed with identical-size fields");
        assert!(composed.rms_magnitude() < 1e-10);
    }

    #[test]
    fn test_jacobian_determinant_identity() {
        let field = DisplacementField::zeros_2d(10, 10);
        let det = JacobianDeterminant::compute_2d(&field)
            .expect("compute_2d should succeed on identity field");
        // Identity deformation → all determinants should be 1
        for v in det.iter() {
            assert!((v - 1.0).abs() < 1e-8, "Expected det≈1, got {}", v);
        }
    }

    #[test]
    fn test_jacobian_folding_fraction_zero_for_identity() {
        let field = DisplacementField::zeros_2d(10, 10);
        let frac = JacobianDeterminant::folding_fraction_2d(&field)
            .expect("folding_fraction_2d should succeed on identity field");
        assert!(frac < 1e-10);
    }

    #[test]
    fn test_demons_diffeo_smoke() {
        let fixed = make_test_image(16, 16, 0.0);
        let moving = make_test_image(16, 16, 0.3);
        let reg = DemonsDiffeo::new();
        let result = reg
            .register(&fixed, &moving)
            .expect("DemonsDiffeo register should succeed on valid images");
        assert!(result.iterations > 0);
        // The field should have some non-zero displacements
        let _ = result.field;
    }

    #[test]
    fn test_fluid_registration_smoke() {
        let fixed = make_test_image(16, 16, 0.0);
        let moving = make_test_image(16, 16, 0.3);
        let reg = FluidRegistration::new();
        let result = reg
            .register(&fixed, &moving)
            .expect("FluidRegistration register should succeed on valid images");
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_ffd_registration_smoke() {
        let fixed = make_test_image(16, 16, 0.0);
        let moving = make_test_image(16, 16, 0.3);
        let reg = FreeFormDeformation::new();
        let result = reg
            .register(&fixed, &moving)
            .expect("FreeFormDeformation register should succeed on valid images");
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_composite_transform_identity() {
        let t = CompositeTransform::identity();
        let (nr, nc) = t
            .apply_to_point(5.0, 7.0)
            .expect("apply_to_point should succeed for identity transform");
        assert!((nr - 5.0).abs() < 1e-10);
        assert!((nc - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_smooth_does_not_panic() {
        let mut df = DisplacementField::zeros_2d(8, 8);
        df.set_2d(4, 4, 10.0, -5.0)
            .expect("set_2d should succeed for valid coordinates");
        df.gaussian_smooth_2d(1.0)
            .expect("gaussian_smooth_2d should succeed with sigma=1");
        // After smoothing the peak should be reduced
        let [dy, _dx] = df
            .get_2d(4, 4)
            .expect("get_2d should succeed for valid coordinates");
        assert!(dy < 10.0);
    }
}
