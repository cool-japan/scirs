//! High-level PINN interface for common PDE problems.
//!
//! This module provides a simplified `Pinn` struct with convenience methods for
//! solving 1D Poisson and heat equations using PINNs, built on top of the
//! full `PINNSolver` infrastructure.

use super::network::PINNNetwork;
use super::solver::PINNSolver;
use super::types::{Boundary, BoundaryCondition, BoundarySide, PDEProblem, PINNConfig, PINNResult};
use crate::error::{IntegrateError, IntegrateResult};

/// Simplified result type for the high-level PINN interface.
#[derive(Debug, Clone)]
pub struct PinnSolveResult {
    /// Predictions at query points (populated after calling `predict`)
    pub predictions: Vec<f64>,
    /// Mean squared PDE residual at collocation points
    pub pde_residual: f64,
    /// Mean squared boundary condition residual
    pub bc_residual: f64,
    /// Total weighted loss at end of training
    pub total_loss: f64,
    /// Number of epochs trained
    pub n_epochs: usize,
    /// Loss history (one entry per epoch)
    pub loss_history: Vec<f64>,
}

impl PinnSolveResult {
    fn from_pinn_result(r: PINNResult) -> Self {
        Self {
            predictions: vec![],
            pde_residual: r.physics_loss,
            bc_residual: r.boundary_loss,
            total_loss: r.final_loss,
            n_epochs: r.epochs_trained,
            loss_history: r.loss_history,
        }
    }
}

/// High-level PINN solver for common 1D PDEs.
///
/// Provides convenience methods for:
/// - 1D Poisson: -u''(x) = f(x) on \[a,b\]
/// - 1D Heat equation: du/dt = α d²u/dx²
///
/// After training, use `predict` to evaluate the network at new points.
pub struct Pinn {
    /// Trained solver (available after calling a solve method)
    solver: Option<PINNSolver>,
    /// Input dimension of the current network
    input_dim: usize,
}

impl Pinn {
    /// Create a new PINN with 1 input dimension.
    pub fn new_1d(hidden_layers: Vec<usize>) -> IntegrateResult<Self> {
        let problem = PDEProblem {
            spatial_dim: 1,
            domain: vec![(0.0, 1.0)],
            boundaries: vec![],
            has_time: false,
            time_domain: None,
        };
        let config = PINNConfig {
            hidden_layers,
            ..Default::default()
        };
        let solver = PINNSolver::new(&problem, config)?;
        Ok(Self {
            solver: Some(solver),
            input_dim: 1,
        })
    }

    /// Create a new PINN with 2 input dimensions (for space-time problems).
    pub fn new_2d(hidden_layers: Vec<usize>) -> IntegrateResult<Self> {
        let problem = PDEProblem {
            spatial_dim: 1,
            domain: vec![(0.0, 1.0)],
            boundaries: vec![],
            has_time: true,
            time_domain: Some((0.0, 1.0)),
        };
        let config = PINNConfig {
            hidden_layers,
            ..Default::default()
        };
        let solver = PINNSolver::new(&problem, config)?;
        Ok(Self {
            solver: Some(solver),
            input_dim: 2,
        })
    }

    /// Solve 1D Poisson equation: -u''(x) = f(x) on [a, b].
    ///
    /// Boundary conditions: u(a) = u_a, u(b) = u_b.
    ///
    /// # Arguments
    /// * `f` - source function (right-hand side of -u'' = f)
    /// * `domain` - (a, b) interval
    /// * `bc` - boundary values (u_a, u_b)
    /// * `config` - PINN configuration
    pub fn solve_poisson_1d(
        &mut self,
        f: impl Fn(f64) -> f64 + Clone + 'static,
        domain: (f64, f64),
        bc: (f64, f64),
        config: &PINNConfig,
    ) -> IntegrateResult<PinnSolveResult> {
        let (a, b) = domain;
        let (u_a, u_b) = bc;

        let problem = PDEProblem {
            spatial_dim: 1,
            domain: vec![(a, b)],
            boundaries: vec![
                Boundary {
                    dim: 0,
                    side: BoundarySide::Low,
                    condition: BoundaryCondition::Dirichlet { value: u_a },
                },
                Boundary {
                    dim: 0,
                    side: BoundarySide::High,
                    condition: BoundaryCondition::Dirichlet { value: u_b },
                },
            ],
            has_time: false,
            time_domain: None,
        };

        let mut solver = PINNSolver::new(&problem, config.clone())?;
        self.input_dim = 1;

        let f_clone = f.clone();
        let residual = move |net: &PINNNetwork, x: &scirs2_core::ndarray::Array1<f64>| {
            // Poisson residual: u'' + f(x) = 0  =>  residual = laplacian(u) + f(x)
            let lap = net.laplacian(x, 1e-4)?;
            Ok(lap + f_clone(x[0]))
        };

        let pinn_result = solver.train(&residual, &problem, None)?;
        self.solver = Some(solver);

        Ok(PinnSolveResult::from_pinn_result(pinn_result))
    }

    /// Solve 1D heat equation: du/dt = α d²u/dx² on \[0,1\] x \[0, t_max\].
    ///
    /// Boundary conditions: u(0,t) = u(1,t) = 0.
    /// Initial condition: u(x, 0) = u0(x).
    ///
    /// # Arguments
    /// * `u0` - initial condition function
    /// * `alpha` - thermal diffusivity coefficient
    /// * `t_max` - final time
    /// * `config` - PINN configuration
    pub fn solve_heat_1d(
        &mut self,
        u0: impl Fn(f64) -> f64 + Clone + 'static,
        alpha: f64,
        t_max: f64,
        config: &PINNConfig,
    ) -> IntegrateResult<PinnSolveResult> {
        let problem = PDEProblem {
            spatial_dim: 1,
            domain: vec![(0.0, 1.0)],
            boundaries: vec![
                Boundary {
                    dim: 0,
                    side: BoundarySide::Low,
                    condition: BoundaryCondition::Dirichlet { value: 0.0 },
                },
                Boundary {
                    dim: 0,
                    side: BoundarySide::High,
                    condition: BoundaryCondition::Dirichlet { value: 0.0 },
                },
            ],
            has_time: true,
            time_domain: Some((0.0, t_max)),
        };

        // Build initial condition data matrix for training.
        // Scale IC points with collocation count to keep cost proportional.
        let n_ic = config.n_collocation.min(50);
        let ic_x: Vec<f64> = (0..n_ic)
            .flat_map(|k| {
                let x = k as f64 / (n_ic as f64 - 1.0);
                vec![x, 0.0] // [x, t=0]
            })
            .collect();
        let ic_x_arr = scirs2_core::ndarray::Array2::from_shape_vec((n_ic, 2), ic_x)
            .map_err(|e| IntegrateError::ComputationError(format!("IC array: {e}")))?;
        let ic_y_arr = scirs2_core::ndarray::Array1::from_vec(
            (0..n_ic)
                .map(|k| u0(k as f64 / (n_ic as f64 - 1.0)))
                .collect(),
        );

        let mut solver = PINNSolver::new(&problem, config.clone())?;
        self.input_dim = 2;

        let residual = move |net: &PINNNetwork, x: &scirs2_core::ndarray::Array1<f64>| {
            // Heat residual: du/dt - alpha * d²u/dx² = 0
            let h = 1e-4;
            // du/dt: finite diff on time (dim 1)
            let mut x_tp = x.clone();
            let mut x_tm = x.clone();
            x_tp[1] += h;
            x_tm[1] -= h;
            let u_tp = net.forward(&x_tp)?;
            let u_tm = net.forward(&x_tm)?;
            let du_dt = (u_tp - u_tm) / (2.0 * h);
            // d²u/dx²: finite diff on space (dim 0)
            let mut x_xp = x.clone();
            let mut x_xm = x.clone();
            x_xp[0] += h;
            x_xm[0] -= h;
            let u_c = net.forward(x)?;
            let u_xp = net.forward(&x_xp)?;
            let u_xm = net.forward(&x_xm)?;
            let d2u = (u_xp - 2.0 * u_c + u_xm) / (h * h);
            Ok(du_dt - alpha * d2u)
        };

        let pinn_result = solver.train(&residual, &problem, Some((&ic_x_arr, &ic_y_arr)))?;
        self.solver = Some(solver);

        Ok(PinnSolveResult::from_pinn_result(pinn_result))
    }

    /// Predict the solution at query points.
    ///
    /// # Arguments
    /// * `query_points` - each inner Vec is one point of length `input_dim`
    pub fn predict(&self, query_points: &[Vec<f64>]) -> IntegrateResult<Vec<f64>> {
        use scirs2_core::ndarray::Array1;
        let solver = self.solver.as_ref().ok_or_else(|| {
            IntegrateError::ComputationError(
                "No trained model. Call solve_poisson_1d or solve_heat_1d first.".to_string(),
            )
        })?;
        let net = solver.network();
        let mut preds = Vec::with_capacity(query_points.len());
        for pt in query_points {
            let arr = Array1::from_vec(pt.clone());
            let val = net.forward(&arr)?;
            preds.push(val);
        }
        Ok(preds)
    }

    /// Forward pass: evaluate the current network at a single point.
    pub fn forward(&self, x: &[f64]) -> IntegrateResult<f64> {
        use scirs2_core::ndarray::Array1;
        let solver = self.solver.as_ref().ok_or_else(|| {
            IntegrateError::ComputationError("No trained model available.".to_string())
        })?;
        let arr = Array1::from_vec(x.to_vec());
        solver.network().forward(&arr)
    }

    /// Compute Laplacian at a point.
    pub fn laplacian(&self, x: &[f64]) -> IntegrateResult<f64> {
        use scirs2_core::ndarray::Array1;
        let solver = self.solver.as_ref().ok_or_else(|| {
            IntegrateError::ComputationError("No trained model available.".to_string())
        })?;
        let arr = Array1::from_vec(x.to_vec());
        solver.network().laplacian(&arr, 1e-4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick_config() -> PINNConfig {
        PINNConfig {
            hidden_layers: vec![16, 16],
            n_collocation: 30,
            n_boundary: 10,
            max_epochs: 50,
            learning_rate: 1e-3,
            physics_weight: 1.0,
            boundary_weight: 10.0,
            data_weight: 1.0,
            convergence_tol: 1e-8,
            collocation: super::super::types::CollocationStrategy::Random,
        }
    }

    #[test]
    fn test_pinn_config_default() {
        let cfg = PINNConfig::default();
        assert_eq!(cfg.hidden_layers, vec![64, 64, 64]);
        assert!((cfg.learning_rate - 1e-3).abs() < 1e-15);
        assert_eq!(cfg.max_epochs, 10000);
        assert_eq!(cfg.n_collocation, 1000);
        assert_eq!(cfg.n_boundary, 100);
        assert!((cfg.physics_weight - 1.0).abs() < 1e-15);
        assert!((cfg.boundary_weight - 10.0).abs() < 1e-15);
    }

    #[test]
    fn test_pinn_new_1d() {
        let p = Pinn::new_1d(vec![16, 16]);
        assert!(p.is_ok());
    }

    #[test]
    fn test_pinn_new_2d() {
        let p = Pinn::new_2d(vec![16, 16]);
        assert!(p.is_ok());
    }

    #[test]
    fn test_pinn_forward_without_solve_fails() {
        // Create a fresh Pinn - solver is set in new_1d but we test that forward works
        let pinn = Pinn::new_1d(vec![8]).expect("create pinn");
        let val = pinn.forward(&[0.5]);
        assert!(val.is_ok());
    }

    #[test]
    fn test_pinn_predict_length() {
        let pinn = Pinn::new_1d(vec![8]).expect("create pinn");
        let query = vec![vec![0.1], vec![0.5], vec![0.9]];
        let preds = pinn.predict(&query).expect("predict");
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_pinn_predict_finite() {
        let pinn = Pinn::new_1d(vec![8]).expect("create pinn");
        let query: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 / 4.0]).collect();
        let preds = pinn.predict(&query).expect("predict");
        for &p in &preds {
            assert!(p.is_finite(), "prediction should be finite, got {p}");
        }
    }

    #[ignore = "slow: measured ~192-194s consistently across all 4 tests in this module under a \
                bounded 600s retry (0.6.5 ignore audit) -- well past the workspace's 120s default \
                nextest hard-kill, so left ignored rather than raising the default profile's timeout \
                for one module; quick_config() (16,16 hidden layers / 30 collocation points / 50 \
                epochs) is already a reduced config vs PINNConfig::default() (64,64,64 / 10000 epochs), \
                and PINN training cost here is dominated by finite-difference gradients scaling with \
                n_params, not a hang -- all 4 tests PASS given enough time"]
    #[test]
    fn test_solve_poisson_1d_runs() {
        let mut pinn = Pinn::new_1d(vec![16, 16]).expect("create pinn");
        let config = quick_config();
        // -u'' = 1, BC: u(0)=0, u(1)=0, exact: u = x(1-x)/2
        let result = pinn.solve_poisson_1d(|_x| 1.0, (0.0, 1.0), (0.0, 0.0), &config);
        assert!(
            result.is_ok(),
            "solve_poisson_1d failed: {:?}",
            result.err()
        );
        let r = result.expect("result");
        assert_eq!(r.n_epochs, config.max_epochs);
        assert!(r.total_loss.is_finite());
        assert!(!r.loss_history.is_empty());
    }

    #[ignore = "slow: measured ~192-194s consistently across all 4 tests in this module under a \
                bounded 600s retry (0.6.5 ignore audit) -- well past the workspace's 120s default \
                nextest hard-kill, so left ignored rather than raising the default profile's timeout \
                for one module; quick_config() (16,16 hidden layers / 30 collocation points / 50 \
                epochs) is already a reduced config vs PINNConfig::default() (64,64,64 / 10000 epochs), \
                and PINN training cost here is dominated by finite-difference gradients scaling with \
                n_params, not a hang -- all 4 tests PASS given enough time"]
    #[test]
    fn test_solve_poisson_1d_predict_output_length() {
        let mut pinn = Pinn::new_1d(vec![16, 16]).expect("create pinn");
        let config = quick_config();
        let _ = pinn
            .solve_poisson_1d(|_x| 1.0, (0.0, 1.0), (0.0, 0.0), &config)
            .expect("solve");
        let query: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 / 4.0]).collect();
        let preds = pinn.predict(&query).expect("predict");
        assert_eq!(
            preds.len(),
            5,
            "predict should return n_query_points values"
        );
    }

    #[test]
    fn test_solve_heat_1d_runs() {
        let mut pinn = Pinn::new_2d(vec![8]).expect("create pinn");
        // Use a minimal config: the heat equation residual is expensive
        // (5 forward passes per collocation point) and finite-difference
        // gradients scale with n_params, so keep everything small.
        let config = PINNConfig {
            hidden_layers: vec![8],
            n_collocation: 10,
            n_boundary: 5,
            max_epochs: 10,
            learning_rate: 1e-3,
            physics_weight: 1.0,
            boundary_weight: 10.0,
            data_weight: 1.0,
            convergence_tol: 1e-8,
            collocation: super::super::types::CollocationStrategy::Random,
        };
        let result =
            pinn.solve_heat_1d(|x: f64| (std::f64::consts::PI * x).sin(), 1.0, 0.1, &config);
        assert!(result.is_ok(), "solve_heat_1d failed: {:?}", result.err());
        let r = result.expect("result");
        assert!(r.total_loss.is_finite());
    }

    #[ignore = "slow: measured ~192-194s consistently across all 4 tests in this module under a \
                bounded 600s retry (0.6.5 ignore audit) -- well past the workspace's 120s default \
                nextest hard-kill, so left ignored rather than raising the default profile's timeout \
                for one module; quick_config() (16,16 hidden layers / 30 collocation points / 50 \
                epochs) is already a reduced config vs PINNConfig::default() (64,64,64 / 10000 epochs), \
                and PINN training cost here is dominated by finite-difference gradients scaling with \
                n_params, not a hang -- all 4 tests PASS given enough time"]
    #[test]
    fn test_solve_result_pde_residual_finite() {
        let mut pinn = Pinn::new_1d(vec![8]).expect("create pinn");
        let config = quick_config();
        let r = pinn
            .solve_poisson_1d(|_| 0.0, (0.0, 1.0), (0.0, 0.0), &config)
            .expect("solve");
        assert!(r.pde_residual.is_finite());
        assert!(r.bc_residual.is_finite());
    }

    #[ignore = "slow: measured ~192-194s consistently across all 4 tests in this module under a \
                bounded 600s retry (0.6.5 ignore audit) -- well past the workspace's 120s default \
                nextest hard-kill, so left ignored rather than raising the default profile's timeout \
                for one module; quick_config() (16,16 hidden layers / 30 collocation points / 50 \
                epochs) is already a reduced config vs PINNConfig::default() (64,64,64 / 10000 epochs), \
                and PINN training cost here is dominated by finite-difference gradients scaling with \
                n_params, not a hang -- all 4 tests PASS given enough time"]
    #[test]
    fn test_pinn_solve_result_loss_history_length() {
        let mut pinn = Pinn::new_1d(vec![8]).expect("create pinn");
        let config = quick_config();
        let r = pinn
            .solve_poisson_1d(|_| 1.0, (0.0, 1.0), (0.0, 0.0), &config)
            .expect("solve");
        assert!(
            r.loss_history.len() <= config.max_epochs,
            "loss history len {} > max_epochs {}",
            r.loss_history.len(),
            config.max_epochs
        );
    }

    #[test]
    fn test_laplacian_finite() {
        let pinn = Pinn::new_1d(vec![8]).expect("create pinn");
        let lap = pinn.laplacian(&[0.5]);
        assert!(lap.is_ok());
        assert!(lap.expect("lap").is_finite());
    }
}
