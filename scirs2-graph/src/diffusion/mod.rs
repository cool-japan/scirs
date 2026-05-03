//! Information diffusion models and influence maximization algorithms
//!
//! This module provides:
//!
//! - **Diffusion Models**: Independent Cascade (IC), Linear Threshold (LT),
//!   SIR (Susceptible-Infected-Recovered), and SIS epidemic models.
//! - **Influence Maximization**: Greedy Monte-Carlo (Kempe 2003), CELF, CELF++,
//!   and fast heuristics (high-degree, PageRank).
//! - **Reverse Influence Sampling**: RIS sets, RIS-based estimators, the IMM
//!   algorithm (Tang et al. 2014/2015), and the Sandwich approximation.
//!
//! # References
//!
//! - Kempe, D., Kleinberg, J., & Tardos, É. (2003). Maximizing the Spread of
//!   Influence through a Social Network. *KDD 2003*.
//! - Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J.,
//!   & Glance, N. (2007). Cost-effective Outbreak Detection in Networks. *KDD
//!   2007*. (CELF)
//! - Goyal, A., Lu, W., & Lakshmanan, L. V. S. (2011). CELF++. *WWW 2011*.
//! - Tang, Y., Xiao, X., & Shi, Y. (2014). Influence Maximization: Near-Optimal
//!   Time Complexity Meets Practical Efficiency. *SIGMOD 2014*. (IMM)
//! - Borg, I., & Groenen, P. (2005). Sandwich approximation for submodular
//!   maximization.

pub mod influence_max;
pub mod models;
pub mod ris;

pub use influence_max::{
    celf_influence_max, celf_plus_plus, degree_heuristic, greedy_influence_max, pagerank_heuristic,
    InfluenceMaxConfig, InfluenceMaxResult,
};
pub use models::{
    expected_spread, simulate_ic, simulate_lt, simulate_sir, simulate_sis, IndependentCascade,
    LinearThreshold, SIRModel, SISModel, SimulationResult, SirState,
};
pub use ris::{
    generate_rr_sets, imm_algorithm, ris_estimate, sandwich_approximation, ImmConfig, ImmResult,
    RISConfig, RRSet,
};
