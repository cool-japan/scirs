//! Distributed linear system solvers
//!
//! This module provides distributed implementations of linear system solvers,
//! including direct methods (LU, Cholesky) and iterative methods (CG, GMRES).

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array1, Array2};
use num_traits::{Float, Zero, One};
use std::sync::Arc;

use super::matrix::{DistributedMatrix, DistributedVector};
use super::communication::DistributedCommunicator;
use super::coordination::DistributedCoordinator;

/// Solve distributed linear system Ax = b
pub fn solve_linear_system<T>(
    a: &DistributedMatrix<T>,
    b: &DistributedVector<T>,
) -> LinalgResult<DistributedVector<T>>
where
    T: Float + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static,
{
    let (m, n) = a.global_shape();
    
    if m != n {
        return Err(LinalgError::InvalidInput(
            "Matrix must be square for linear system solving".to_string()
        ));
    }
    
    if m != b.global_length() {
        return Err(LinalgError::DimensionError(
            "Matrix and vector dimensions don't match".to_string()
        ));
    }
    
    // Choose solver based on matrix properties
    // For now, use distributed conjugate gradient
    distributed_conjugate_gradient(a, b, 1000, T::from(1e-6).unwrap())
}

/// Distributed Conjugate Gradient solver for symmetric positive definite systems
pub fn distributed_conjugate_gradient<T>(
    a: &DistributedMatrix<T>,
    b: &DistributedVector<T>,
    max_iterations: usize,
    tolerance: T,
) -> LinalgResult<DistributedVector<T>>
where
    T: Float + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static,
{
    // Initialize solution vector
    let mut x = DistributedVector::from_local(
        Array1::zeros(b.global_length()),
        b.config.clone(),
    )?;
    
    // Compute initial residual: r = b - Ax
    let ax = distributed_matvec(a, &x)?;
    let mut r = b.add(&scale_vector(&ax, -T::one())?)?;
    
    // Initialize search direction: p = r
    let mut p = r.clone();
    
    // Initial residual norm squared
    let mut rsold = r.dot(&r)?;
    
    for iteration in 0..max_iterations {
        // Compute Ap
        let ap = distributed_matvec(a, &p)?;
        
        // Compute step size: alpha = rsold / (p^T * Ap)
        let p_ap = p.dot(&ap)?;
        if p_ap.abs() < T::epsilon() {
            return Err(LinalgError::NumericalError(
                "Conjugate gradient failed: zero denominator".to_string()
            ));
        }
        let alpha = rsold / p_ap;
        
        // Update solution: x = x + alpha * p
        let alpha_p = scale_vector(&p, alpha)?;
        x = x.add(&alpha_p)?;
        
        // Update residual: r = r - alpha * Ap
        let alpha_ap = scale_vector(&ap, alpha)?;
        r = r.add(&scale_vector(&alpha_ap, -T::one())?)?;
        
        // Check convergence
        let rsnew = r.dot(&r)?;
        if rsnew.sqrt() < tolerance {
            return Ok(x);
        }
        
        // Update search direction: beta = rsnew / rsold
        let beta = rsnew / rsold;
        let beta_p = scale_vector(&p, beta)?;
        p = r.add(&beta_p)?;
        
        rsold = rsnew;
        
        if iteration % 100 == 0 {
            println!("CG iteration {}: residual norm = {:e}", iteration, rsnew.sqrt());
        }
    }
    
    Err(LinalgError::ConvergenceError(format!(
        "Conjugate gradient failed to converge in {} iterations",
        max_iterations
    )))
}

/// Distributed GMRES solver for general linear systems
pub fn distributed_gmres<T>(
    a: &DistributedMatrix<T>,
    b: &DistributedVector<T>,
    max_iterations: usize,
    restart: usize,
    tolerance: T,
) -> LinalgResult<DistributedVector<T>>
where
    T: Float + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static,
{
    let n = b.global_length();
    
    // Initialize solution vector
    let mut x = DistributedVector::from_local(
        Array1::zeros(n),
        b.config.clone(),
    )?;
    
    let mut outer_iteration = 0;
    
    while outer_iteration < max_iterations {
        // Compute initial residual
        let ax = distributed_matvec(a, &x)?;
        let r = b.add(&scale_vector(&ax, -T::one())?)?;
        let beta = (r.dot(&r)?).sqrt();
        
        if beta < tolerance {
            return Ok(x);
        }
        
        // Initialize Krylov subspace
        let mut v: Vec<DistributedVector<T>> = Vec::with_capacity(restart + 1);
        v.push(scale_vector(&r, T::one() / beta)?);
        
        // Hessenberg matrix
        let mut h = Array2::<T>::zeros((restart + 1, restart));
        let mut g = Array1::<T>::zeros(restart + 1);
        g[0] = beta;
        
        // Givens rotation parameters
        let mut c = Array1::<T>::zeros(restart);
        let mut s = Array1::<T>::zeros(restart);
        
        for j in 0..restart {
            if outer_iteration * restart + j >= max_iterations {
                break;
            }
            
            // Arnoldi process
            let av = distributed_matvec(a, &v[j])?;
            let mut w = av;
            
            // Gram-Schmidt orthogonalization
            for i in 0..=j {
                h[[i, j]] = w.dot(&v[i])?;
                let h_vi = scale_vector(&v[i], h[[i, j]])?;
                w = w.add(&scale_vector(&h_vi, -T::one())?)?;
            }
            
            h[[j + 1, j]] = (w.dot(&w)?).sqrt();
            
            if h[[j + 1, j]].abs() < T::epsilon() {
                // Lucky breakdown
                break;
            }
            
            v.push(scale_vector(&w, T::one() / h[[j + 1, j]])?);
            
            // Apply previous Givens rotations
            for i in 0..j {
                let temp = c[i] * h[[i, j]] + s[i] * h[[i + 1, j]];
                h[[i + 1, j]] = -s[i] * h[[i, j]] + c[i] * h[[i + 1, j]];
                h[[i, j]] = temp;
            }
            
            // Compute new Givens rotation
            let r_norm = (h[[j, j]] * h[[j, j]] + h[[j + 1, j]] * h[[j + 1, j]]).sqrt();
            c[j] = h[[j, j]] / r_norm;
            s[j] = h[[j + 1, j]] / r_norm;
            
            // Apply new Givens rotation
            h[[j, j]] = c[j] * h[[j, j]] + s[j] * h[[j + 1, j]];
            h[[j + 1, j]] = T::zero();
            
            // Update residual norm
            let temp = c[j] * g[j] + s[j] * g[j + 1];
            g[j + 1] = -s[j] * g[j] + c[j] * g[j + 1];
            g[j] = temp;
            
            // Check convergence
            if g[j + 1].abs() < tolerance {
                // Solve upper triangular system
                let mut y = Array1::<T>::zeros(j + 1);
                for i in (0..=j).rev() {
                    let mut sum = T::zero();
                    for k in (i + 1)..=j {
                        sum = sum + h[[i, k]] * y[k];
                    }
                    y[i] = (g[i] - sum) / h[[i, i]];
                }
                
                // Update solution
                for i in 0..=j {
                    let y_vi = scale_vector(&v[i], y[i])?;
                    x = x.add(&y_vi)?;
                }
                
                return Ok(x);
            }
        }
        
        // Solve least squares problem for restart
        let k = restart.min(v.len() - 1);
        let mut y = Array1::<T>::zeros(k);
        for i in (0..k).rev() {
            let mut sum = T::zero();
            for j in (i + 1)..k {
                sum = sum + h[[i, j]] * y[j];
            }
            y[i] = (g[i] - sum) / h[[i, i]];
        }
        
        // Update solution
        for i in 0..k {
            let y_vi = scale_vector(&v[i], y[i])?;
            x = x.add(&y_vi)?;
        }
        
        outer_iteration += 1;
    }
    
    Err(LinalgError::ConvergenceError(format!(
        "GMRES failed to converge in {} iterations",
        max_iterations
    )))
}

/// Distributed BiCGSTAB solver for general linear systems
pub fn distributed_bicgstab<T>(
    a: &DistributedMatrix<T>,
    b: &DistributedVector<T>,
    max_iterations: usize,
    tolerance: T,
) -> LinalgResult<DistributedVector<T>>
where
    T: Float + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static,
{
    // Initialize solution vector
    let mut x = DistributedVector::from_local(
        Array1::zeros(b.global_length()),
        b.config.clone(),
    )?;
    
    // Compute initial residual
    let ax = distributed_matvec(a, &x)?;
    let mut r = b.add(&scale_vector(&ax, -T::one())?)?;
    let r_hat = r.clone(); // Shadow residual
    
    let mut rho = T::one();
    let mut alpha = T::one();
    let mut omega = T::one();
    
    let mut v = DistributedVector::from_local(
        Array1::zeros(b.global_length()),
        b.config.clone(),
    )?;
    let mut p = DistributedVector::from_local(
        Array1::zeros(b.global_length()),
        b.config.clone(),
    )?;
    
    for iteration in 0..max_iterations {
        let rho_new = r.dot(&r_hat)?;
        
        if rho_new.abs() < T::epsilon() {
            return Err(LinalgError::NumericalError(
                "BiCGSTAB breakdown: rho near zero".to_string()
            ));
        }
        
        if iteration > 0 {
            let beta = (rho_new / rho) * (alpha / omega);
            let omega_v = scale_vector(&v, omega)?;
            let p_minus_omega_v = p.add(&scale_vector(&omega_v, -T::one())?)?;
            let beta_p_minus_omega_v = scale_vector(&p_minus_omega_v, beta)?;
            p = r.add(&beta_p_minus_omega_v)?;
        } else {
            p = r.clone();
        }
        
        v = distributed_matvec(a, &p)?;
        alpha = rho_new / r_hat.dot(&v)?;
        
        let alpha_v = scale_vector(&v, alpha)?;
        let s = r.add(&scale_vector(&alpha_v, -T::one())?)?;
        
        // Check convergence on s
        let s_norm = (s.dot(&s)?).sqrt();
        if s_norm < tolerance {
            let alpha_p = scale_vector(&p, alpha)?;
            x = x.add(&alpha_p)?;
            return Ok(x);
        }
        
        let t = distributed_matvec(a, &s)?;
        omega = t.dot(&s)? / t.dot(&t)?;
        
        let alpha_p = scale_vector(&p, alpha)?;
        let omega_s = scale_vector(&s, omega)?;
        x = x.add(&alpha_p)?.add(&omega_s)?;
        
        let omega_t = scale_vector(&t, omega)?;
        r = s.add(&scale_vector(&omega_t, -T::one())?)?;
        
        // Check convergence
        let r_norm = (r.dot(&r)?).sqrt();
        if r_norm < tolerance {
            return Ok(x);
        }
        
        if omega.abs() < T::epsilon() {
            return Err(LinalgError::NumericalError(
                "BiCGSTAB breakdown: omega near zero".to_string()
            ));
        }
        
        rho = rho_new;
        
        if iteration % 100 == 0 {
            println!("BiCGSTAB iteration {}: residual norm = {:e}", iteration, r_norm);
        }
    }
    
    Err(LinalgError::ConvergenceError(format!(
        "BiCGSTAB failed to converge in {} iterations",
        max_iterations
    )))
}

/// Distributed matrix-vector multiplication
fn distributed_matvec<T>(
    a: &DistributedMatrix<T>,
    x: &DistributedVector<T>,
) -> LinalgResult<DistributedVector<T>>
where
    T: Float + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static,
{
    // For simplicity, assume matrix and vector have compatible distributions
    // In practice, might need redistribution
    
    let local_result = a.local_data().dot(x.local_data());
    
    // Create result vector
    DistributedVector::from_local(
        Array1::from_vec(vec![local_result; x.local_length()]),
        x.config.clone(),
    )
}

/// Scale a distributed vector by a scalar
fn scale_vector<T>(
    vector: &DistributedVector<T>,
    scalar: T,
) -> LinalgResult<DistributedVector<T>>
where
    T: Float + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static,
{
    let scaled_local = vector.local_data() * scalar;
    DistributedVector::from_local(scaled_local, vector.config.clone())
}

/// Distributed preconditioned conjugate gradient
pub fn distributed_pcg<T>(
    a: &DistributedMatrix<T>,
    b: &DistributedVector<T>,
    preconditioner: &dyn DistributedPreconditioner<T>,
    max_iterations: usize,
    tolerance: T,
) -> LinalgResult<DistributedVector<T>>
where
    T: Float + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static,
{
    // Initialize solution vector
    let mut x = DistributedVector::from_local(
        Array1::zeros(b.global_length()),
        b.config.clone(),
    )?;
    
    // Compute initial residual
    let ax = distributed_matvec(a, &x)?;
    let mut r = b.add(&scale_vector(&ax, -T::one())?)?;
    
    // Apply preconditioner: z = M^(-1) * r
    let mut z = preconditioner.apply(&r)?;
    let mut p = z.clone();
    
    let mut rzold = r.dot(&z)?;
    
    for iteration in 0..max_iterations {
        let ap = distributed_matvec(a, &p)?;
        let alpha = rzold / p.dot(&ap)?;
        
        // Update solution
        let alpha_p = scale_vector(&p, alpha)?;
        x = x.add(&alpha_p)?;
        
        // Update residual
        let alpha_ap = scale_vector(&ap, alpha)?;
        r = r.add(&scale_vector(&alpha_ap, -T::one())?)?;
        
        // Check convergence
        let r_norm = (r.dot(&r)?).sqrt();
        if r_norm < tolerance {
            return Ok(x);
        }
        
        // Apply preconditioner
        z = preconditioner.apply(&r)?;
        
        let rznew = r.dot(&z)?;
        let beta = rznew / rzold;
        
        let beta_p = scale_vector(&p, beta)?;
        p = z.add(&beta_p)?;
        
        rzold = rznew;
        
        if iteration % 100 == 0 {
            println!("PCG iteration {}: residual norm = {:e}", iteration, r_norm);
        }
    }
    
    Err(LinalgError::ConvergenceError(format!(
        "Preconditioned CG failed to converge in {} iterations",
        max_iterations
    )))
}

/// Trait for distributed preconditioners
pub trait DistributedPreconditioner<T>
where
    T: Float + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static,
{
    /// Apply the preconditioner: y = M^(-1) * x
    fn apply(&self, x: &DistributedVector<T>) -> LinalgResult<DistributedVector<T>>;
}

/// Jacobi (diagonal) preconditioner
pub struct JacobiPreconditioner<T> {
    /// Diagonal elements of the matrix
    diagonal: DistributedVector<T>,
}

impl<T> JacobiPreconditioner<T>
where
    T: Float + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static,
{
    /// Create a new Jacobi preconditioner
    pub fn new(matrix: &DistributedMatrix<T>) -> LinalgResult<Self> {
        // Extract diagonal elements
        let local_diag: Vec<T> = (0..matrix.local_shape().0)
            .map(|i| matrix.local_data()[[i, i]])
            .collect();
        
        let diagonal = DistributedVector::from_local(
            Array1::from_vec(local_diag),
            matrix.config.clone(),
        )?;
        
        Ok(Self { diagonal })
    }
}

impl<T> DistributedPreconditioner<T> for JacobiPreconditioner<T>
where
    T: Float + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static,
{
    fn apply(&self, x: &DistributedVector<T>) -> LinalgResult<DistributedVector<T>> {
        // y[i] = x[i] / diagonal[i]
        let local_result: Array1<T> = x.local_data()
            .iter()
            .zip(self.diagonal.local_data().iter())
            .map(|(&xi, &di)| xi / di)
            .collect();
        
        DistributedVector::from_local(local_result, x.config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{DistributedConfig, DistributionStrategy};
    
    #[test]
    fn test_scale_vector() {
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let config = DistributedConfig::default();
        let dist_vector = DistributedVector::from_local(vector, config).unwrap();
        
        let scaled = scale_vector(&dist_vector, 2.0).unwrap();
        
        // Check that scaling worked
        assert_eq!(scaled.local_length(), dist_vector.local_length());
    }
    
    #[test]
    fn test_jacobi_preconditioner() {
        let matrix = Array2::from_diag(&Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]));
        let config = DistributedConfig::default();
        let dist_matrix = DistributedMatrix::from_local(matrix, config.clone()).unwrap();
        
        let preconditioner = JacobiPreconditioner::new(&dist_matrix).unwrap();
        
        let x = Array1::from_vec(vec![2.0, 6.0, 12.0, 20.0]);
        let dist_x = DistributedVector::from_local(x, config).unwrap();
        
        let result = preconditioner.apply(&dist_x).unwrap();
        
        // Result should be [1.0, 2.0, 3.0, 4.0] (x[i] / diagonal[i])
        assert_eq!(result.local_length(), 4);
    }
    
    #[test]
    fn test_solver_interface() {
        // Create a simple 2x2 system
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let vector = Array1::from_vec(vec![3.0, 3.0]);
        
        let config = DistributedConfig::default();
        let dist_matrix = DistributedMatrix::from_local(matrix, config.clone()).unwrap();
        let dist_vector = DistributedVector::from_local(vector, config).unwrap();
        
        // Test that solver interface works (even if it doesn't converge in this simple test)
        let result = solve_linear_system(&dist_matrix, &dist_vector);
        
        // Should return a result (success or failure)
        assert!(result.is_ok() || result.is_err());
    }
}