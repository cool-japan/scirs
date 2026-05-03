//! Longitudinal and Panel Data Analysis
//!
//! This module provides estimators and tests for panel data (also known as
//! longitudinal or cross-sectional time-series data).
//!
//! # Sub-modules
//!
//! | Module | Contents |
//! |--------|---------|
//! | [`fixed_effects`] | Within estimator, two-way FE, first-difference estimator |
//! | [`random_effects`] | GLS random effects, Hausman test, LMM, REML |
//! | [`dynamic`] | Arellano-Bond GMM, Blundell-Bond system GMM, serial correlation tests |
//! | [`count_models`] | Poisson FE, Negative Binomial FE, Zero-Inflated models |
//!
//! # Quick Examples
//!
//! ## Fixed Effects
//! ```rust,no_run
//! use scirs2_stats::panel::fixed_effects::FixedEffectsModel;
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! // 3 entities × 4 periods
//! let x = Array2::<f64>::ones((12, 2));
//! let y = Array1::<f64>::ones(12);
//! let entity: Vec<usize> = (0..3).flat_map(|e| std::iter::repeat(e).take(4)).collect();
//! let time:   Vec<usize> = (0..4).cycle().take(12).collect();
//!
//! let result = FixedEffectsModel::fit(&x.view(), &y.view(), &entity, &time, false)
//!     .expect("fit failed");
//! println!("β: {:?}", result.coefficients);
//! println!("R² within: {}", result.r2_within);
//! ```
//!
//! ## Random Effects + Hausman Test
//! ```rust,no_run
//! use scirs2_stats::panel::{
//!     fixed_effects::FixedEffectsModel,
//!     random_effects::{RandomEffectsModel, HausmanTest},
//! };
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! let x      = Array2::<f64>::ones((20, 1));
//! let y      = Array1::<f64>::ones(20);
//! let entity: Vec<usize> = (0..4).flat_map(|e| std::iter::repeat(e).take(5)).collect();
//! let time:   Vec<usize> = (0..5).cycle().take(20).collect();
//!
//! let fe = FixedEffectsModel::fit(&x.view(), &y.view(), &entity, &time, false).unwrap();
//! let re = RandomEffectsModel::fit(&x.view(), &y.view(), &entity, &time).unwrap();
//! let h  = HausmanTest::test(&fe, &re).unwrap();
//! println!("Hausman H={:.4}, p={:.4}", h.h_stat, h.p_value);
//! ```
//!
//! ## Dynamic Panel (Arellano-Bond)
//! ```rust,no_run
//! use scirs2_stats::panel::dynamic::ArellanoBlond;
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! let y      = Array1::<f64>::ones(30);
//! let x      = Array2::<f64>::ones((30, 1));
//! let entity: Vec<usize> = (0..5).flat_map(|e| std::iter::repeat(e).take(6)).collect();
//! let time:   Vec<usize> = (0..6).cycle().take(30).collect();
//!
//! let result = ArellanoBlond::fit(&y.view(), &x.view(), &entity, &time, 1)
//!     .expect("AB-GMM failed");
//! println!("β̂={:?}", result.coefficients);
//! println!("Sargan J={:.4} p={:.4}", result.sargan.j_stat, result.sargan.p_value);
//! ```
//!
//! ## Count Panel (Poisson FE)
//! ```rust,no_run
//! use scirs2_stats::panel::count_models::PoissonFE;
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! let x = Array2::<f64>::ones((20, 1));
//! let y = Array1::<f64>::from(vec![0.0, 1.0, 2.0, 3.0].repeat(5));
//! let entity: Vec<usize> = (0..4).flat_map(|e| std::iter::repeat(e).take(5)).collect();
//!
//! let result = PoissonFE::fit(&x.view(), &y.view(), &entity, 100, 1e-8)
//!     .expect("Poisson FE failed");
//! println!("IRR: {:?}", result.irr);
//! ```

pub mod count_models;
pub mod dynamic;
pub mod fixed_effects;
pub mod random_effects;

// ── Re-exports for convenience ──────────────────────────────────────────────

pub use count_models::{
    CountDistribution, CountPanelResult, NegBinomFE, PoissonFE, ZeroInflated, ZeroInflatedResult,
};
pub use dynamic::{
    ARTestResult, ArellanoBlond, BlundellBond, DynamicPanelResult, SarganTest, SarganTestResult,
};
pub use fixed_effects::{
    FEResult, FirstDiffEstimator, FixedEffectsModel, TwoWayFE, WithinTransform,
};
pub use random_effects::{
    HausmanTest, HausmanTestResult, LinearMixedModel, LmmConfig, LmmResult, REResult,
    RandomEffectsModel, REML,
};
