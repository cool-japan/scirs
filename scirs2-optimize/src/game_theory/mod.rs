//! Game Theory and Nash Equilibrium Solvers
//!
//! This module provides comprehensive game-theoretic algorithms including:
//! - Normal-form game analysis and Nash equilibrium computation
//! - Zero-sum game solvers (minimax, linear programming, fictitious play)
//! - Cooperative game theory (Shapley value, core, nucleolus)
//!
//! # Examples
//!
//! ## Finding Nash Equilibria in a Prisoner's Dilemma
//!
//! ```rust
//! use scirs2_optimize::game_theory::normal_form::{NormalFormGame, find_pure_nash_equilibria};
//! use scirs2_core::ndarray::array;
//!
//! // Prisoner's Dilemma: Cooperate (0) or Defect (1)
//! let payoff_1 = array![[-1.0, -3.0], [0.0, -2.0]];
//! let payoff_2 = array![[-1.0, 0.0], [-3.0, -2.0]];
//! let game = NormalFormGame::new(payoff_1, payoff_2).expect("valid input");
//! let pure_nash = find_pure_nash_equilibria(&game);
//! // (Defect, Defect) is the unique Nash equilibrium
//! assert_eq!(pure_nash.len(), 1);
//! ```
//!
//! ## Solving a Zero-Sum Game
//!
//! ```rust
//! use scirs2_optimize::game_theory::zero_sum::minimax_solve;
//! use scirs2_core::ndarray::array;
//!
//! let payoff = array![[2.0, -1.0], [-1.0, 2.0]];
//! let result = minimax_solve(payoff.view()).expect("valid input");
//! println!("Game value: {:.4}", result.game_value);
//! ```
//!
//! ## Computing Shapley Values
//!
//! ```rust
//! use scirs2_optimize::game_theory::cooperative::{CooperativeGame, shapley_value};
//!
//! let mut game = CooperativeGame::new(3);
//! game.set_value(&[0], 0.0);
//! game.set_value(&[1], 0.0);
//! game.set_value(&[2], 0.0);
//! game.set_value(&[0, 1], 0.6);
//! game.set_value(&[0, 2], 0.6);
//! game.set_value(&[1, 2], 0.6);
//! game.set_value(&[0, 1, 2], 1.0);
//! let shapley = shapley_value(&game);
//! ```

pub mod cooperative;
pub mod normal_form;
pub mod zero_sum;

pub use cooperative::*;
pub use normal_form::*;
pub use zero_sum::*;
