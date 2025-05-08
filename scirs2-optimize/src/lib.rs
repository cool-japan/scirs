//! Optimization module for SciRS
//!
//! This module provides implementations of various optimization algorithms,
//! modeled after SciPy's `optimize` module.
//!

#![allow(clippy::field_reassign_with_default)]
//! ## Submodules
//!
//! * `unconstrained`: Unconstrained optimization algorithms
//! * `constrained`: Constrained optimization algorithms
//! * `least_squares`: Least squares minimization
//! * `roots`: Root finding algorithms
//!
//! ## Optimization Methods
//!
//! The following optimization methods are currently implemented:
//!
//! ### Unconstrained:
//! - **Nelder-Mead**: A derivative-free method using simplex-based approach
//! - **Powell**: Derivative-free method using conjugate directions
//! - **BFGS**: Quasi-Newton method with BFGS update
//! - **CG**: Nonlinear conjugate gradient method
//!
//! ### Constrained:
//! - **SLSQP**: Sequential Least SQuares Programming
//! - **TrustConstr**: Trust-region constrained optimizer
//!
//! ## Bounds Support
//!
//! The `unconstrained` module now supports bounds constraints for variables.
//! You can specify lower and upper bounds for each variable, and the optimizer
//! will ensure that all iterates remain within these bounds.
//!
//! The following methods support bounds constraints:
//! - Powell
//! - Nelder-Mead
//! - BFGS
//! - CG (Conjugate Gradient)
//!
//! ## Examples
//!
//! ### Basic Optimization
//!
//! ```
//! // Example of minimizing a function using BFGS
//! use ndarray::array;
//! use scirs2_optimize::unconstrained::{minimize, Method};
//!
//! fn rosenbrock(x: &[f64]) -> f64 {
//!     let a = 1.0;
//!     let b = 100.0;
//!     let x0 = x[0];
//!     let x1 = x[1];
//!     (a - x0).powi(2) + b * (x1 - x0.powi(2)).powi(2)
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let initial_guess = array![0.0, 0.0];
//! let result = minimize(rosenbrock, &initial_guess, Method::BFGS, None)?;
//!
//! println!("Solution: {:?}", result.x);
//! println!("Function value at solution: {}", result.fun);
//! println!("Number of iterations: {}", result.nit);
//! println!("Success: {}", result.success);
//! # Ok(())
//! # }
//! ```
//!
//! ### Optimization with Bounds
//!
//! ```
//! // Example of minimizing a function with bounds constraints
//! use ndarray::array;
//! use scirs2_optimize::{Bounds, unconstrained::{minimize, Method, Options}};
//!
//! // A function with minimum at (-1, -1)
//! fn func(x: &[f64]) -> f64 {
//!     (x[0] + 1.0).powi(2) + (x[1] + 1.0).powi(2)
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create bounds: x >= 0, y >= 0
//! // This will constrain the optimization to the positive quadrant
//! let bounds = Bounds::new(&[(Some(0.0), None), (Some(0.0), None)]);
//!
//! let initial_guess = array![0.5, 0.5];
//! let mut options = Options::default();
//! options.bounds = Some(bounds);
//!
//! // Use Powell's method which supports bounds
//! let result = minimize(func, &initial_guess, Method::Powell, Some(options))?;
//!
//! // The constrained minimum should be at [0, 0] with value 2.0
//! println!("Solution: {:?}", result.x);
//! println!("Function value at solution: {}", result.fun);
//! # Ok(())
//! # }
//! ```
//!
//! ### Bounds Creation Options
//!
//! ```
//! use scirs2_optimize::Bounds;
//!
//! // Create bounds from pairs
//! // Format: [(min_x1, max_x1), (min_x2, max_x2), ...] where None = unbounded
//! let bounds1 = Bounds::new(&[
//!     (Some(0.0), Some(1.0)),  // 0 <= x[0] <= 1
//!     (Some(-1.0), None),      // x[1] >= -1, no upper bound
//!     (None, Some(10.0)),      // x[2] <= 10, no lower bound
//!     (None, None)             // x[3] is completely unbounded
//! ]);
//!
//! // Alternative: create from separate lower and upper bound vectors
//! let lb = vec![Some(0.0), Some(-1.0), None, None];
//! let ub = vec![Some(1.0), None, Some(10.0), None];
//! let bounds2 = Bounds::from_vecs(lb, ub).unwrap();
//! ```

extern crate openblas_src;

// Export error types
pub mod error;
pub use error::{OptimizeError, OptimizeResult};

// Module structure
pub mod constrained;
pub mod least_squares;
pub mod roots;
pub mod roots_anderson;
pub mod roots_krylov;
pub mod unconstrained;

// Common optimization result structure
pub mod result;
pub use result::OptimizeResults;

// Convenience re-exports for common functions
pub use constrained::minimize_constrained;
pub use least_squares::least_squares;
pub use roots::root;
pub use unconstrained::{minimize, Bounds};

// Prelude module for convenient imports
pub mod prelude {
    pub use crate::constrained::{minimize_constrained, Method as ConstrainedMethod};
    pub use crate::error::{OptimizeError, OptimizeResult};
    pub use crate::least_squares::{least_squares, Method as LeastSquaresMethod};
    pub use crate::result::OptimizeResults;
    pub use crate::roots::{root, Method as RootMethod};
    pub use crate::unconstrained::{minimize, Bounds, Method as UnconstrainedMethod, Options};
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
