//! Spectral Methods for solving PDEs
//!
//! This module provides implementations of spectral methods for solving PDEs,
//! which approximate solutions using global basis functions like Fourier series,
//! Chebyshev polynomials, or Legendre polynomials.
//!
//! Key features:
//! - Fourier spectral methods for periodic problems
//! - Chebyshev spectral methods for non-periodic problems
//! - Legendre spectral methods for non-periodic problems
//! - Spectral element methods for complex geometries
//! - Fast transforms using FFT
//! - High accuracy for smooth solutions

use ndarray::{Array1, Array2, ArrayView1};
use std::f64::consts::PI;
use std::time::Instant;

// FFT functions would be implemented here or imported from another crate
// For now, let's create stubs for these functions
// use crate::fft::{fft, ifft, irfft, rfft};
use crate::gaussian::GaussLegendreQuadrature;
use crate::pde::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, PDEError, PDEResult,
    PDESolution, PDESolverInfo,
};

/// Type of spectral basis functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectralBasis {
    /// Fourier basis (exp(ikx)) for periodic problems
    Fourier,

    /// Chebyshev polynomials (T_n(x)) for non-periodic problems
    Chebyshev,

    /// Legendre polynomials (P_n(x)) for non-periodic problems
    Legendre,
}

/// Options for spectral solvers
#[derive(Debug, Clone)]
pub struct SpectralOptions {
    /// Type of spectral basis to use
    pub basis: SpectralBasis,

    /// Number of modes/coefficients to use
    pub num_modes: usize,

    /// Maximum iterations for iterative solvers
    pub max_iterations: usize,

    /// Tolerance for convergence
    pub tolerance: f64,

    /// Whether to save convergence history
    pub save_convergence_history: bool,

    /// Whether to use real FFT (more efficient for real-valued functions)
    pub use_real_transform: bool,

    /// Whether to use dealiasing (e.g., 2/3 rule for Fourier)
    pub use_dealiasing: bool,

    /// Print detailed progress information
    pub verbose: bool,
}

impl Default for SpectralOptions {
    fn default() -> Self {
        SpectralOptions {
            basis: SpectralBasis::Fourier,
            num_modes: 64,
            max_iterations: 1000,
            tolerance: 1e-10,
            save_convergence_history: false,
            use_real_transform: true,
            use_dealiasing: true,
            verbose: false,
        }
    }
}

/// Result of spectral method solutions
#[derive(Debug, Clone)]
pub struct SpectralResult {
    /// Solution values on the grid
    pub u: Array1<f64>,

    /// Spectral coefficients
    pub coefficients: Array1<f64>,

    /// Grid points
    pub grid: Array1<f64>,

    /// Residual norm
    pub residual_norm: f64,

    /// Number of iterations performed
    pub num_iterations: usize,

    /// Computation time
    pub computation_time: f64,

    /// Convergence history
    pub convergence_history: Option<Vec<f64>>,
}

/// Create a Chebyshev differentiation matrix of size n×n
///
/// This matrix D satisfies Du = u' where u is a vector of function values
/// at Chebyshev points x_j = cos(jπ/n), j=0...n
pub fn chebyshev_diff_matrix(n: usize) -> Array2<f64> {
    let mut d = Array2::zeros((n, n));

    // Compute Chebyshev points
    let mut x = Array1::zeros(n);
    for j in 0..n {
        x[j] = (j as f64 * PI / (n - 1) as f64).cos();
    }

    // Compute the matrix entries
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let c_i = if i == 0 || i == n - 1 { 2.0 } else { 1.0 };
                let c_j = if j == 0 || j == n - 1 { 2.0 } else { 1.0 };

                d[[i, j]] = c_i / c_j * (-1.0_f64).powf((i + j) as f64) / (x[i] - x[j]);
            }
        }
    }

    // Compute diagonal entries by enforcing row sums to be zero
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            if i != j {
                row_sum += d[[i, j]];
            }
        }
        d[[i, i]] = -row_sum;
    }

    d
}

/// Create a second-derivative Chebyshev differentiation matrix
pub fn chebyshev_diff2_matrix(n: usize) -> Array2<f64> {
    let d1 = chebyshev_diff_matrix(n);
    d1.dot(&d1) // D^2 = D * D
}

/// Generate Chebyshev grid points x_j = cos(jπ/n), j=0...n-1
pub fn chebyshev_points(n: usize) -> Array1<f64> {
    let mut x = Array1::zeros(n);
    for j in 0..n {
        x[j] = (j as f64 * PI / (n - 1) as f64).cos();
    }
    x
}

/// Transform from physical space to Chebyshev coefficient space
pub fn chebyshev_transform(u: &ArrayView1<f64>) -> Array1<f64> {
    let n = u.len();
    let mut coeffs = Array1::zeros(n);

    // Simple direct transform
    for k in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            let _x_j = (j as f64 * PI / (n - 1) as f64).cos();
            sum += u[j] * (k as f64 * j as f64 * PI / (n - 1) as f64).cos();
        }

        let norm = if k == 0 || k == n - 1 {
            n - 1
        } else {
            2 * (n - 1)
        };
        coeffs[k] = 2.0 * sum / norm as f64;
    }

    coeffs
}

/// Transform from Chebyshev coefficient space to physical space
pub fn chebyshev_inverse_transform(coeffs: &ArrayView1<f64>) -> Array1<f64> {
    let n = coeffs.len();
    let mut u = Array1::zeros(n);

    // Generate Chebyshev points
    let _x = chebyshev_points(n);

    // Evaluate the Chebyshev series at each point
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..n {
            sum += coeffs[k] * (k as f64 * j as f64 * PI / (n - 1) as f64).cos();
        }
        u[j] = sum;
    }

    u
}

/// Generate Legendre-Gauss-Lobatto quadrature points and weights
///
/// These are the zeros of (1-x²)P'ₙ₋₁(x) in [-1, 1] where Pₙ is the
/// Legendre polynomial of degree n. The points include the endpoints ±1.
pub fn legendre_points(n: usize) -> (Array1<f64>, Array1<f64>) {
    if n <= 1 {
        return (Array1::zeros(1), Array1::ones(1));
    }

    // For small n, we can use precomputed Gauss-Legendre quadrature
    // and add the endpoints
    if n <= 12 {
        // Get the inner Gauss-Legendre points (n-2 points)
        let quadrature = GaussLegendreQuadrature::<f64>::new(n - 2).unwrap();

        // Create arrays for Gauss-Lobatto points (including endpoints)
        let mut points = Array1::zeros(n);
        let mut weights = Array1::zeros(n);

        // Add endpoints
        points[0] = -1.0;
        points[n - 1] = 1.0;

        // Add interior points (reversed because GaussLegendreQuadrature gives ascending points)
        for i in 0..n - 2 {
            points[i + 1] = quadrature.nodes[n - 3 - i]; // Reverse order
        }

        // Compute weights: the formula for Legendre-Gauss-Lobatto weights is
        // wⱼ = 2/[n(n-1)[Pₙ₋₁(xⱼ)]²] for all j
        let factor = 2.0 / (n as f64 * (n - 1) as f64);

        // Endpoints have the same weight
        weights[0] = factor;
        weights[n - 1] = factor;

        // Interior weights
        for i in 1..n - 1 {
            let x = points[i];
            let p = legendre_polynomial(n - 1, x);
            weights[i] = factor / (p * p);
        }

        return (points, weights);
    }

    // For larger n, we need to compute the points by finding the zeros of the derivative
    // of the Legendre polynomial, and then compute the weights separately

    // Initial guess: Chebyshev points (good approximation for Legendre)
    let mut points = Array1::zeros(n);
    for i in 0..n {
        points[i] = -(i as f64 * PI / (n - 1) as f64).cos();
    }

    // Refine using Newton-Raphson iteration
    for i in 1..n - 1 {
        // Skip endpoints that are always ±1
        let mut x = points[i];
        let mut delta;

        // Newton-Raphson to find zeros of (1-x²)P'ₙ₋₁(x)
        // We compute P'ₙ₋₁(x) and P''ₙ₋₁(x) using recurrence relations
        for _ in 0..10 {
            // Usually converges in a few iterations
            let _p = legendre_polynomial(n - 1, x);
            let dp = legendre_polynomial_derivative(n - 1, x);
            // d/dx[(1-x²)P'ₙ₋₁(x)] = -2x·P'ₙ₋₁(x) + (1-x²)P''ₙ₋₁(x)
            // We approximate P''ₙ₋₁(x) using a central difference on P'ₙ₋₁
            let h = 1e-6;
            let dp_plus = legendre_polynomial_derivative(n - 1, x + h);
            let dp_minus = legendre_polynomial_derivative(n - 1, x - h);
            let ddp = (dp_plus - dp_minus) / (2.0 * h);

            // Function value: (1-x²)P'ₙ₋₁(x)
            let f = (1.0 - x * x) * dp;
            // Derivative: -2x·P'ₙ₋₁(x) + (1-x²)P''ₙ₋₁(x)
            let df = -2.0 * x * dp + (1.0 - x * x) * ddp;

            // Newton update
            delta = f / df;
            x -= delta;

            if delta.abs() < 1e-12 {
                break;
            }
        }

        points[i] = x;
    }

    // Compute weights using the formula wⱼ = 2/[n(n-1)[Pₙ₋₁(xⱼ)]²]
    let mut weights = Array1::zeros(n);
    let factor = 2.0 / (n as f64 * (n - 1) as f64);

    for i in 0..n {
        let x = points[i];
        let p = legendre_polynomial(n - 1, x);
        weights[i] = factor / (p * p);
    }

    (points, weights)
}

/// Evaluate the Legendre polynomial of degree n at point x
fn legendre_polynomial(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }

    // Use recurrence relation for Legendre polynomials:
    // (n+1)Pₙ₊₁(x) = (2n+1)xPₙ(x) - nPₙ₋₁(x)
    let mut p_prev = 1.0; // P₀(x)
    let mut p = x; // P₁(x)

    for k in 2..=n {
        let k_f64 = k as f64;
        let p_next = ((2.0 * k_f64 - 1.0) * x * p - (k_f64 - 1.0) * p_prev) / k_f64;
        p_prev = p;
        p = p_next;
    }

    p
}

/// Evaluate the derivative of the Legendre polynomial of degree n at point x
fn legendre_polynomial_derivative(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return 1.0;
    }

    // Use relation: (1-x²)P'ₙ(x) = n(Pₙ₋₁(x) - xPₙ(x))
    // => P'ₙ(x) = n(Pₙ₋₁(x) - xPₙ(x))/(1-x²)

    let pn = legendre_polynomial(n, x);
    let pn_minus_1 = legendre_polynomial(n - 1, x);

    // Handle numerical issues at endpoints where the denominator is zero
    if (1.0 - x * x).abs() < 1e-10 {
        if x > 0.0 {
            // At x = 1, use P'ₙ(1) = n(n+1)/2
            return n as f64 * (n + 1) as f64 / 2.0;
        } else {
            // At x = -1, use P'ₙ(-1) = (-1)^(n+1) * n(n+1)/2
            return if n % 2 == 0 { -1.0 } else { 1.0 } * n as f64 * (n + 1) as f64 / 2.0;
        }
    }

    n as f64 * (pn_minus_1 - x * pn) / (1.0 - x * x)
}

/// Create a Legendre differentiation matrix of size n×n
///
/// This matrix D satisfies Du = u' where u is a vector of function values
/// at Legendre-Gauss-Lobatto points
pub fn legendre_diff_matrix(n: usize) -> Array2<f64> {
    let mut d = Array2::zeros((n, n));

    // Compute Legendre-Gauss-Lobatto points and weights
    let (x, _) = legendre_points(n);

    // Compute the differentiation matrix entries
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let p_i = legendre_polynomial(n - 1, x[i]);
                let p_j = legendre_polynomial(n - 1, x[j]);

                d[[i, j]] = p_i / (p_j * (x[i] - x[j]));

                if j == 0 || j == n - 1 {
                    d[[i, j]] *= 2.0;
                }
            }
        }
    }

    // Compute diagonal entries by enforcing row sum = 0
    for i in 0..n {
        d[[i, i]] = 0.0;
        for j in 0..n {
            if i != j {
                d[[i, i]] -= d[[i, j]];
            }
        }
    }

    d
}

/// Create a second-derivative Legendre differentiation matrix
pub fn legendre_diff2_matrix(n: usize) -> Array2<f64> {
    let d1 = legendre_diff_matrix(n);
    d1.dot(&d1) // D^2 = D * D
}

/// Transform from physical space to Legendre coefficient space
pub fn legendre_transform(u: &ArrayView1<f64>) -> Array1<f64> {
    let n = u.len();
    let mut coeffs = Array1::zeros(n);

    // Get Legendre-Gauss-Lobatto points and weights
    let (x, w) = legendre_points(n);

    // Compute coefficients using quadrature
    for k in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            let p_k = legendre_polynomial(k, x[j]);
            sum += u[j] * p_k * w[j];
        }

        // Normalization: (k+1/2) for the orthogonality relation of Legendre polynomials
        let norm = k as f64 + 0.5;
        coeffs[k] = sum * norm;
    }

    coeffs
}

/// Transform from Legendre coefficient space to physical space
pub fn legendre_inverse_transform(
    coeffs: &ArrayView1<f64>,
    x_points: Option<&ArrayView1<f64>>,
) -> Array1<f64> {
    let n = coeffs.len();
    let mut u = Array1::zeros(n);

    // Use provided points or generate Legendre-Gauss-Lobatto points
    let x = if let Some(points) = x_points {
        points.to_owned()
    } else {
        legendre_points(n).0
    };

    // Evaluate the Legendre series at each point
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..n {
            // Normalization factor
            let norm = 1.0 / (k as f64 + 0.5);
            sum += coeffs[k] * norm * legendre_polynomial(k, x[j]);
        }
        u[j] = sum;
    }

    u
}

/// Fourier spectral solver for 1D periodic PDEs
pub struct FourierSpectralSolver1D {
    /// Domain for the problem
    domain: Domain,

    /// Source term function f(x)
    source_term: Box<dyn Fn(f64) -> f64 + Send + Sync>,

    /// Boundary conditions (must be periodic for Fourier methods)
    #[allow(dead_code)]
    boundary_conditions: Vec<BoundaryCondition<f64>>,

    /// Solver options
    options: SpectralOptions,
}

impl FourierSpectralSolver1D {
    /// Create a new Fourier spectral solver for 1D periodic PDEs
    pub fn new(
        domain: Domain,
        source_term: impl Fn(f64) -> f64 + Send + Sync + 'static,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<SpectralOptions>,
    ) -> PDEResult<Self> {
        // Validate domain
        if domain.dimensions() != 1 {
            return Err(PDEError::DomainError(
                "Domain must be 1-dimensional for 1D Fourier spectral solver".to_string(),
            ));
        }

        // Check that the domain is suitable for Fourier methods (periodic)
        let mut has_periodic = false;
        for bc in &boundary_conditions {
            if bc.bc_type == BoundaryConditionType::Periodic {
                has_periodic = true;
                break;
            }
        }

        if !has_periodic {
            return Err(PDEError::BoundaryConditions(
                "Fourier spectral methods require periodic boundary conditions".to_string(),
            ));
        }

        let mut options = options.unwrap_or_default();
        options.basis = SpectralBasis::Fourier; // Ensure Fourier basis

        Ok(FourierSpectralSolver1D {
            domain,
            source_term: Box::new(source_term),
            boundary_conditions,
            options,
        })
    }

    /// Solve a 1D periodic PDE using Fourier spectral method
    ///
    /// Solves the equation: d²u/dx² = f(x) with periodic boundary conditions
    pub fn solve(&self) -> PDEResult<SpectralResult> {
        let start_time = Instant::now();

        // Extract domain information
        let range = &self.domain.ranges[0];
        let length = range.end - range.start;
        let n = self.options.num_modes;

        // Create uniform grid
        let mut grid = Array1::zeros(n);
        for i in 0..n {
            grid[i] = range.start + i as f64 * length / n as f64;
        }

        // Compute source term on the grid
        let mut f_values = Array1::zeros(n);
        for (i, &x) in grid.iter().enumerate() {
            f_values[i] = (self.source_term)(x);
        }

        // Transform source term to frequency domain
        let f_hat = if self.options.use_real_transform {
            rfft(&f_values.to_owned())
        } else {
            fft(&f_values.to_owned())
        };

        // Set up wavenumbers (accounting for rfft vs fft)
        let n_freq = if self.options.use_real_transform {
            n / 2 + 1
        } else {
            n
        };
        let mut k = Array1::zeros(n_freq);

        for i in 0..n_freq {
            if i <= n / 2 {
                k[i] = 2.0 * PI * i as f64 / length;
            } else {
                k[i] = -2.0 * PI * (n - i) as f64 / length;
            }
        }

        // Solve in frequency domain: -k²û = f̂ => û = -f̂/k²
        let mut u_hat = Array1::from_elem(f_hat.len(), num_complex::Complex::new(0.0, 0.0));

        for i in 0..n_freq {
            if i == 0 {
                // k=0 mode corresponds to the constant/mean of the solution
                // For Poisson's equation, this is determined by the source term's mean
                // Typically, we set it to zero (solution is determined up to a constant)
                u_hat[i] = num_complex::Complex::new(0.0, 0.0);
            } else {
                // For all other modes, solve using the inverse Laplacian
                u_hat[i] = -f_hat[i] / (k[i] * k[i]);
            }
        }

        // Apply dealiasing if requested (2/3 rule)
        if self.options.use_dealiasing {
            let cutoff = 2 * n / 3;
            for i in cutoff..n_freq {
                u_hat[i] = num_complex::Complex::new(0.0, 0.0);
            }
        }

        // Transform back to physical space
        let u = if self.options.use_real_transform {
            irfft(&u_hat)
        } else {
            ifft(&u_hat).mapv(|c| c.re)
        };

        // Compute residual norm
        let mut residual = Array1::zeros(n);

        // Second derivative using spectral accuracy
        let u_xx = if self.options.use_real_transform {
            // Compute second derivative in frequency domain
            let mut u_xx_hat = Array1::zeros(u_hat.len());
            for i in 0..n_freq {
                u_xx_hat[i] = -k[i] * k[i] * u_hat[i];
            }
            irfft(&u_xx_hat)
        } else {
            // Compute second derivative in frequency domain
            let mut u_xx_hat = Array1::zeros(u_hat.len());
            for i in 0..n_freq {
                u_xx_hat[i] = -k[i] * k[i] * u_hat[i];
            }
            ifft(&u_xx_hat).mapv(|c| c.re)
        };

        // Residual: d²u/dx² - f(x)
        for i in 0..n {
            residual[i] = u_xx[i] - f_values[i];
        }

        // Compute L2 norm of residual
        let residual_norm = (residual.mapv(|r| r * r).sum() / n as f64).sqrt();

        let computation_time = start_time.elapsed().as_secs_f64();

        Ok(SpectralResult {
            u,
            coefficients: u_hat.mapv(|c| c.re),
            grid,
            residual_norm,
            num_iterations: 1, // Direct method, one iteration
            computation_time,
            convergence_history: None,
        })
    }
}

/// Chebyshev spectral solver for 1D non-periodic PDEs
pub struct ChebyshevSpectralSolver1D {
    /// Domain for the problem
    domain: Domain,

    /// Source term function f(x)
    source_term: Box<dyn Fn(f64) -> f64 + Send + Sync>,

    /// Boundary conditions (Dirichlet, Neumann, or Robin)
    boundary_conditions: Vec<BoundaryCondition<f64>>,

    /// Solver options
    options: SpectralOptions,
}

impl ChebyshevSpectralSolver1D {
    /// Create a new Chebyshev spectral solver for 1D non-periodic PDEs
    pub fn new(
        domain: Domain,
        source_term: impl Fn(f64) -> f64 + Send + Sync + 'static,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<SpectralOptions>,
    ) -> PDEResult<Self> {
        // Validate domain
        if domain.dimensions() != 1 {
            return Err(PDEError::DomainError(
                "Domain must be 1-dimensional for 1D Chebyshev spectral solver".to_string(),
            ));
        }

        // Validate boundary conditions
        if boundary_conditions.len() != 2 {
            return Err(PDEError::BoundaryConditions(
                "1D Chebyshev spectral solver requires exactly 2 boundary conditions".to_string(),
            ));
        }

        let mut has_lower = false;
        let mut has_upper = false;

        for bc in &boundary_conditions {
            match bc.location {
                BoundaryLocation::Lower => has_lower = true,
                BoundaryLocation::Upper => has_upper = true,
            }

            // Periodic boundary conditions are not suitable for Chebyshev methods
            if bc.bc_type == BoundaryConditionType::Periodic {
                return Err(PDEError::BoundaryConditions(
                    "Chebyshev spectral methods are not suitable for periodic boundary conditions"
                        .to_string(),
                ));
            }
        }

        if !has_lower || !has_upper {
            return Err(PDEError::BoundaryConditions(
                "Chebyshev spectral solver requires boundary conditions at both domain endpoints"
                    .to_string(),
            ));
        }

        let mut options = options.unwrap_or_default();
        options.basis = SpectralBasis::Chebyshev; // Ensure Chebyshev basis

        Ok(ChebyshevSpectralSolver1D {
            domain,
            source_term: Box::new(source_term),
            boundary_conditions,
            options,
        })
    }

    /// Solve a 1D non-periodic PDE using Chebyshev spectral method
    ///
    /// Solves the equation: d²u/dx² = f(x) with Dirichlet, Neumann, or Robin boundary conditions
    pub fn solve(&self) -> PDEResult<SpectralResult> {
        let start_time = Instant::now();

        // Extract domain information
        let range = &self.domain.ranges[0];
        let a = range.start;
        let b = range.end;

        // Number of Chebyshev points
        let n = self.options.num_modes;

        // Generate Chebyshev grid in [-1, 1]
        let mut cheb_grid = Array1::zeros(n);
        for j in 0..n {
            cheb_grid[j] = (j as f64 * PI / (n - 1) as f64).cos();
        }

        // Map Chebyshev grid to the domain [a, b]
        let mut grid = Array1::zeros(n);
        for j in 0..n {
            grid[j] = a + (b - a) * (cheb_grid[j] + 1.0) / 2.0;
        }

        // Compute source term on the grid
        let mut f_values = Array1::zeros(n);
        for j in 0..n {
            f_values[j] = (self.source_term)(grid[j]);
        }

        // Create second-derivative Chebyshev differentiation matrix
        let mut d2 = chebyshev_diff2_matrix(n);

        // Scale the differentiation matrix to account for the domain mapping
        let scale = 4.0 / ((b - a) * (b - a));
        d2.mapv_inplace(|x| x * scale);

        // Set up the linear system Au = f
        let mut a_matrix = d2;
        let mut rhs = f_values;

        // Apply boundary conditions
        for bc in &self.boundary_conditions {
            match bc.location {
                BoundaryLocation::Lower => {
                    // Lower boundary (x = a, or Chebyshev point = 1)
                    let j = 0; // First grid point (Chebyshev coordinate 1)

                    match bc.bc_type {
                        BoundaryConditionType::Dirichlet => {
                            // u(a) = bc.value
                            for k in 0..n {
                                a_matrix[[j, k]] = 0.0;
                            }
                            a_matrix[[j, j]] = 1.0;
                            rhs[j] = bc.value;
                        }
                        BoundaryConditionType::Neumann => {
                            // du/dx(a) = bc.value
                            // Use first-derivative Chebyshev differentiation matrix
                            let d1 = chebyshev_diff_matrix(n);

                            // Scale the derivative matrix for domain mapping
                            let deriv_scale = 2.0 / (b - a);

                            for k in 0..n {
                                a_matrix[[j, k]] = d1[[j, k]] * deriv_scale;
                            }
                            rhs[j] = bc.value;
                        }
                        BoundaryConditionType::Robin => {
                            // a*u(a) + b*du/dx(a) = c
                            if let Some([a_coef, b_coef, c_coef]) = bc.coefficients {
                                // Use first-derivative Chebyshev differentiation matrix for the derivative term
                                let d1 = chebyshev_diff_matrix(n);

                                // Scale the derivative matrix for domain mapping
                                let deriv_scale = 2.0 / (b - a);

                                for k in 0..n {
                                    a_matrix[[j, k]] = a_coef + b_coef * d1[[j, k]] * deriv_scale;
                                }
                                rhs[j] = c_coef;
                            }
                        }
                        _ => {
                            return Err(PDEError::BoundaryConditions(
                                "Unsupported boundary condition type for Chebyshev spectral method"
                                    .to_string(),
                            ))
                        }
                    }
                }
                BoundaryLocation::Upper => {
                    // Upper boundary (x = b, or Chebyshev point = -1)
                    let j = n - 1; // Last grid point (Chebyshev coordinate -1)

                    match bc.bc_type {
                        BoundaryConditionType::Dirichlet => {
                            // u(b) = bc.value
                            for k in 0..n {
                                a_matrix[[j, k]] = 0.0;
                            }
                            a_matrix[[j, j]] = 1.0;
                            rhs[j] = bc.value;
                        }
                        BoundaryConditionType::Neumann => {
                            // du/dx(b) = bc.value
                            // Use first-derivative Chebyshev differentiation matrix
                            let d1 = chebyshev_diff_matrix(n);

                            // Scale the derivative matrix for domain mapping
                            let deriv_scale = 2.0 / (b - a);

                            for k in 0..n {
                                a_matrix[[j, k]] = d1[[j, k]] * deriv_scale;
                            }
                            rhs[j] = bc.value;
                        }
                        BoundaryConditionType::Robin => {
                            // a*u(b) + b*du/dx(b) = c
                            if let Some([a_coef, b_coef, c_coef]) = bc.coefficients {
                                // Use first-derivative Chebyshev differentiation matrix for the derivative term
                                let d1 = chebyshev_diff_matrix(n);

                                // Scale the derivative matrix for domain mapping
                                let deriv_scale = 2.0 / (b - a);

                                for k in 0..n {
                                    a_matrix[[j, k]] = a_coef + b_coef * d1[[j, k]] * deriv_scale;
                                }
                                rhs[j] = c_coef;
                            }
                        }
                        _ => {
                            return Err(PDEError::BoundaryConditions(
                                "Unsupported boundary condition type for Chebyshev spectral method"
                                    .to_string(),
                            ))
                        }
                    }
                }
            }
        }

        // Solve the linear system
        let u = self.solve_linear_system(&a_matrix, &rhs.view())?;

        // Compute Chebyshev coefficients (optional)
        let coefficients = chebyshev_transform(&u.view());

        // Compute residual
        let mut residual = Array1::zeros(n);
        let a_times_u = a_matrix.dot(&u);

        for i in 0..n {
            residual[i] = a_times_u[i] - rhs[i];
        }

        // Exclude boundary points from residual calculation
        let mut residual_norm = 0.0;
        for i in 1..n - 1 {
            residual_norm += residual[i] * residual[i];
        }
        residual_norm = (residual_norm / (n - 2) as f64).sqrt();

        let computation_time = start_time.elapsed().as_secs_f64();

        Ok(SpectralResult {
            u,
            coefficients,
            grid,
            residual_norm,
            num_iterations: 1, // Direct method, one iteration
            computation_time,
            convergence_history: None,
        })
    }

    /// Solve the linear system Ax = b
    fn solve_linear_system(&self, a: &Array2<f64>, b: &ArrayView1<f64>) -> PDEResult<Array1<f64>> {
        let n = b.len();

        // Simple Gaussian elimination with partial pivoting
        // For a real implementation, use a specialized linear algebra library

        // Create copies of A and b
        let mut a_copy = a.clone();
        let mut b_copy = b.to_owned();

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_val = a_copy[[i, i]].abs();
            let mut max_row = i;

            for k in i + 1..n {
                if a_copy[[k, i]].abs() > max_val {
                    max_val = a_copy[[k, i]].abs();
                    max_row = k;
                }
            }

            // Check if matrix is singular
            if max_val < 1e-10 {
                return Err(PDEError::Other(
                    "Matrix is singular or nearly singular".to_string(),
                ));
            }

            // Swap rows if necessary
            if max_row != i {
                for j in i..n {
                    let temp = a_copy[[i, j]];
                    a_copy[[i, j]] = a_copy[[max_row, j]];
                    a_copy[[max_row, j]] = temp;
                }

                let temp = b_copy[i];
                b_copy[i] = b_copy[max_row];
                b_copy[max_row] = temp;
            }

            // Eliminate below
            for k in i + 1..n {
                let factor = a_copy[[k, i]] / a_copy[[i, i]];

                for j in i..n {
                    a_copy[[k, j]] -= factor * a_copy[[i, j]];
                }

                b_copy[k] -= factor * b_copy[i];
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in i + 1..n {
                sum += a_copy[[i, j]] * x[j];
            }

            x[i] = (b_copy[i] - sum) / a_copy[[i, i]];
        }

        Ok(x)
    }
}

/// Convert SpectralResult to PDESolution
/// Legendre spectral solver for 1D non-periodic PDEs
pub struct LegendreSpectralSolver1D {
    /// Domain for the problem
    domain: Domain,

    /// Source term function f(x)
    source_term: Box<dyn Fn(f64) -> f64 + Send + Sync>,

    /// Boundary conditions (Dirichlet, Neumann, or Robin)
    boundary_conditions: Vec<BoundaryCondition<f64>>,

    /// Solver options
    options: SpectralOptions,
}

impl LegendreSpectralSolver1D {
    /// Create a new Legendre spectral solver for 1D non-periodic PDEs
    pub fn new(
        domain: Domain,
        source_term: impl Fn(f64) -> f64 + Send + Sync + 'static,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<SpectralOptions>,
    ) -> PDEResult<Self> {
        // Validate domain
        if domain.dimensions() != 1 {
            return Err(PDEError::DomainError(
                "Domain must be 1-dimensional for 1D Legendre spectral solver".to_string(),
            ));
        }

        // Validate boundary conditions
        if boundary_conditions.len() != 2 {
            return Err(PDEError::BoundaryConditions(
                "1D Legendre spectral solver requires exactly 2 boundary conditions".to_string(),
            ));
        }

        let mut has_lower = false;
        let mut has_upper = false;

        for bc in &boundary_conditions {
            match bc.location {
                BoundaryLocation::Lower => has_lower = true,
                BoundaryLocation::Upper => has_upper = true,
            }

            // Periodic boundary conditions are not suitable for Legendre methods
            if bc.bc_type == BoundaryConditionType::Periodic {
                return Err(PDEError::BoundaryConditions(
                    "Legendre spectral methods are not suitable for periodic boundary conditions"
                        .to_string(),
                ));
            }
        }

        if !has_lower || !has_upper {
            return Err(PDEError::BoundaryConditions(
                "Legendre spectral solver requires boundary conditions at both domain endpoints"
                    .to_string(),
            ));
        }

        let mut options = options.unwrap_or_default();
        options.basis = SpectralBasis::Legendre; // Ensure Legendre basis

        Ok(LegendreSpectralSolver1D {
            domain,
            source_term: Box::new(source_term),
            boundary_conditions,
            options,
        })
    }

    /// Solve a 1D non-periodic PDE using Legendre spectral method
    ///
    /// Solves the equation: d²u/dx² = f(x) with Dirichlet, Neumann, or Robin boundary conditions
    pub fn solve(&self) -> PDEResult<SpectralResult> {
        let start_time = Instant::now();

        // Extract domain information
        let range = &self.domain.ranges[0];
        let a = range.start;
        let b = range.end;

        // Number of Legendre points
        let n = self.options.num_modes;

        // Generate Legendre-Gauss-Lobatto grid in [-1, 1] and weights
        let (lgb_grid, _weights) = legendre_points(n);

        // Map Legendre grid to the domain [a, b]
        let mut grid = Array1::zeros(n);
        for j in 0..n {
            grid[j] = a + (b - a) * (lgb_grid[j] + 1.0) / 2.0;
        }

        // Compute source term on the grid
        let mut f_values = Array1::zeros(n);
        for j in 0..n {
            f_values[j] = (self.source_term)(grid[j]);
        }

        // Create second-derivative Legendre differentiation matrix
        let mut d2 = legendre_diff2_matrix(n);

        // Scale the differentiation matrix to account for the domain mapping
        let scale = 4.0 / ((b - a) * (b - a));
        d2.mapv_inplace(|x| x * scale);

        // Set up the linear system Au = f
        let mut a_matrix = d2;
        let mut rhs = f_values;

        // Apply boundary conditions
        for bc in &self.boundary_conditions {
            match bc.location {
                BoundaryLocation::Lower => {
                    // Lower boundary (x = a, first grid point)
                    let j = 0;

                    match bc.bc_type {
                        BoundaryConditionType::Dirichlet => {
                            // u(a) = bc.value
                            for k in 0..n {
                                a_matrix[[j, k]] = 0.0;
                            }
                            a_matrix[[j, j]] = 1.0;
                            rhs[j] = bc.value;
                        }
                        BoundaryConditionType::Neumann => {
                            // du/dx(a) = bc.value
                            // Use first-derivative Legendre differentiation matrix
                            let d1 = legendre_diff_matrix(n);

                            // Scale the derivative matrix for domain mapping
                            let deriv_scale = 2.0 / (b - a);

                            for k in 0..n {
                                a_matrix[[j, k]] = d1[[j, k]] * deriv_scale;
                            }
                            rhs[j] = bc.value;
                        }
                        BoundaryConditionType::Robin => {
                            // a*u(a) + b*du/dx(a) = c
                            if let Some([a_coef, b_coef, c_coef]) = bc.coefficients {
                                // Use first-derivative Legendre differentiation matrix for the derivative term
                                let d1 = legendre_diff_matrix(n);

                                // Scale the derivative matrix for domain mapping
                                let deriv_scale = 2.0 / (b - a);

                                for k in 0..n {
                                    a_matrix[[j, k]] = a_coef + b_coef * d1[[j, k]] * deriv_scale;
                                }
                                rhs[j] = c_coef;
                            }
                        }
                        _ => {
                            return Err(PDEError::BoundaryConditions(
                                "Unsupported boundary condition type for Legendre spectral method"
                                    .to_string(),
                            ))
                        }
                    }
                }
                BoundaryLocation::Upper => {
                    // Upper boundary (x = b, last grid point)
                    let j = n - 1;

                    match bc.bc_type {
                        BoundaryConditionType::Dirichlet => {
                            // u(b) = bc.value
                            for k in 0..n {
                                a_matrix[[j, k]] = 0.0;
                            }
                            a_matrix[[j, j]] = 1.0;
                            rhs[j] = bc.value;
                        }
                        BoundaryConditionType::Neumann => {
                            // du/dx(b) = bc.value
                            // Use first-derivative Legendre differentiation matrix
                            let d1 = legendre_diff_matrix(n);

                            // Scale the derivative matrix for domain mapping
                            let deriv_scale = 2.0 / (b - a);

                            for k in 0..n {
                                a_matrix[[j, k]] = d1[[j, k]] * deriv_scale;
                            }
                            rhs[j] = bc.value;
                        }
                        BoundaryConditionType::Robin => {
                            // a*u(b) + b*du/dx(b) = c
                            if let Some([a_coef, b_coef, c_coef]) = bc.coefficients {
                                // Use first-derivative Legendre differentiation matrix for the derivative term
                                let d1 = legendre_diff_matrix(n);

                                // Scale the derivative matrix for domain mapping
                                let deriv_scale = 2.0 / (b - a);

                                for k in 0..n {
                                    a_matrix[[j, k]] = a_coef + b_coef * d1[[j, k]] * deriv_scale;
                                }
                                rhs[j] = c_coef;
                            }
                        }
                        _ => {
                            return Err(PDEError::BoundaryConditions(
                                "Unsupported boundary condition type for Legendre spectral method"
                                    .to_string(),
                            ))
                        }
                    }
                }
            }
        }

        // Solve the linear system
        let u = self.solve_linear_system(&a_matrix, &rhs.view())?;

        // Compute Legendre coefficients
        let coefficients = legendre_transform(&u.view());

        // Compute residual
        let mut residual = Array1::zeros(n);
        let a_times_u = a_matrix.dot(&u);

        for i in 0..n {
            residual[i] = a_times_u[i] - rhs[i];
        }

        // Exclude boundary points from residual calculation
        let mut residual_norm = 0.0;
        for i in 1..n - 1 {
            residual_norm += residual[i] * residual[i];
        }
        residual_norm = (residual_norm / (n - 2) as f64).sqrt();

        let computation_time = start_time.elapsed().as_secs_f64();

        Ok(SpectralResult {
            u,
            coefficients,
            grid,
            residual_norm,
            num_iterations: 1, // Direct method, one iteration
            computation_time,
            convergence_history: None,
        })
    }

    /// Solve the linear system Ax = b
    fn solve_linear_system(&self, a: &Array2<f64>, b: &ArrayView1<f64>) -> PDEResult<Array1<f64>> {
        let n = b.len();

        // Simple Gaussian elimination with partial pivoting
        // For a real implementation, use a specialized linear algebra library

        // Create copies of A and b
        let mut a_copy = a.clone();
        let mut b_copy = b.to_owned();

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_val = a_copy[[i, i]].abs();
            let mut max_row = i;

            for k in i + 1..n {
                if a_copy[[k, i]].abs() > max_val {
                    max_val = a_copy[[k, i]].abs();
                    max_row = k;
                }
            }

            // Check if matrix is singular
            if max_val < 1e-10 {
                return Err(PDEError::Other(
                    "Matrix is singular or nearly singular".to_string(),
                ));
            }

            // Swap rows if necessary
            if max_row != i {
                for j in i..n {
                    let temp = a_copy[[i, j]];
                    a_copy[[i, j]] = a_copy[[max_row, j]];
                    a_copy[[max_row, j]] = temp;
                }

                let temp = b_copy[i];
                b_copy[i] = b_copy[max_row];
                b_copy[max_row] = temp;
            }

            // Eliminate below
            for k in i + 1..n {
                let factor = a_copy[[k, i]] / a_copy[[i, i]];

                for j in i..n {
                    a_copy[[k, j]] -= factor * a_copy[[i, j]];
                }

                b_copy[k] -= factor * b_copy[i];
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in i + 1..n {
                sum += a_copy[[i, j]] * x[j];
            }

            x[i] = (b_copy[i] - sum) / a_copy[[i, i]];
        }

        Ok(x)
    }
}

impl From<SpectralResult> for PDESolution<f64> {
    fn from(result: SpectralResult) -> Self {
        let grids = vec![result.grid.clone()];

        // Create solution values as a 2D array with one column
        let mut values = Vec::new();
        // Clone the result.u to avoid the move issue
        let u_clone = result.u.clone();
        let u_len = u_clone.len();
        let u_reshaped = u_clone.into_shape_with_order((u_len, 1)).unwrap();
        values.push(u_reshaped);

        // Create solver info
        let info = PDESolverInfo {
            num_iterations: result.num_iterations,
            computation_time: result.computation_time,
            residual_norm: Some(result.residual_norm),
            convergence_history: result.convergence_history,
            method: "Spectral Method".to_string(),
        };

        PDESolution {
            grids,
            values,
            error_estimate: None,
            info,
        }
    }
}

// Stub FFT implementations
// These are temporary placeholders for the missing FFT functions
// In a real implementation, these would use a proper FFT library

/// Perform a Fast Fourier Transform (FFT) on a real-valued array
///
/// # Arguments
/// * `x` - The input array
///
/// # Returns
/// * A complex-valued array containing the FFT result
fn fft(x: &Array1<f64>) -> Array1<num_complex::Complex<f64>> {
    // This is a stub implementation
    // In a real implementation, this would use a proper FFT algorithm
    let n = x.len();
    let mut result = Array1::zeros(n);

    for k in 0..n {
        let mut sum = num_complex::Complex::new(0.0, 0.0);
        for j in 0..n {
            let angle = -2.0 * PI * (j as f64) * (k as f64) / (n as f64);
            let factor = num_complex::Complex::new(angle.cos(), angle.sin());
            sum += factor * x[j];
        }
        result[k] = sum;
    }

    result
}

/// Perform an Inverse Fast Fourier Transform (IFFT) on a complex-valued array
///
/// # Arguments
/// * `x` - The input array
///
/// # Returns
/// * A complex-valued array containing the IFFT result
fn ifft(x: &Array1<num_complex::Complex<f64>>) -> Array1<num_complex::Complex<f64>> {
    // This is a stub implementation
    // In a real implementation, this would use a proper IFFT algorithm
    let n = x.len();
    let mut result = Array1::zeros(n);

    for k in 0..n {
        let mut sum = num_complex::Complex::new(0.0, 0.0);
        for j in 0..n {
            let angle = 2.0 * PI * (j as f64) * (k as f64) / (n as f64);
            let factor = num_complex::Complex::new(angle.cos(), angle.sin());
            sum += factor * x[j];
        }
        result[k] = sum / (n as f64);
    }

    result
}

/// Perform a Real Fast Fourier Transform (RFFT) on a real-valued array
///
/// # Arguments
/// * `x` - The input array
///
/// # Returns
/// * A complex-valued array containing the RFFT result
fn rfft(x: &Array1<f64>) -> Array1<num_complex::Complex<f64>> {
    // For simplicity, use the regular FFT
    fft(x)
}

/// Perform an Inverse Real Fast Fourier Transform (IRFFT) on a complex-valued array
///
/// # Arguments
/// * `x` - The input array
///
/// # Returns
/// * A real-valued array containing the IRFFT result
fn irfft(x: &Array1<num_complex::Complex<f64>>) -> Array1<f64> {
    // This is a stub implementation
    // In a real implementation, this would use a proper IRFFT algorithm
    let complex_result = ifft(x);

    // Extract real parts
    let n = complex_result.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        result[i] = complex_result[i].re;
    }

    result
}

// Import spectral element module
pub mod spectral_element;
pub use spectral_element::{
    QuadElement, SpectralElementMesh2D, SpectralElementOptions, SpectralElementPoisson2D,
    SpectralElementResult,
};
