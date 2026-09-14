//! Multivariate distribution functions
//!
//! This module provides functionality for multivariate statistical distributions.

pub mod dirichlet;
pub mod inverse_wishart;
pub mod multinomial;
pub mod multivariate_lognormal;
pub mod normal;
pub mod student_t;
pub mod von_mises_fisher;
pub mod wishart;

pub use dirichlet::Dirichlet;
pub use inverse_wishart::InverseWishart;
pub use multinomial::Multinomial;
pub use multivariate_lognormal::MultivariateLognormal;
pub use normal::MultivariateNormal;
pub use student_t::MultivariateT;
pub use von_mises_fisher::VonMisesFisher;
pub use wishart::Wishart;

// Re-export convenience functions
pub use dirichlet::dirichlet;
pub use inverse_wishart::inverse_wishart;
pub use multinomial::multinomial;
pub use multivariate_lognormal::multivariate_lognormal;
pub use normal::multivariate_normal;
pub use student_t::multivariate_t;
pub use wishart::wishart;
