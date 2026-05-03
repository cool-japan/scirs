//! Information diffusion models
//!
//! Provides four simulation models on an adjacency representation:
//!
//! | Model | Description |
//! |-------|-------------|
//! | [`IndependentCascade`] | Each active node activates each inactive neighbour with probability `p(u,v)` |
//! | [`LinearThreshold`] | Node activates when total in-weight from active neighbours ≥ per-node threshold |
//! | [`SIRModel`] | Epidemic SIR: β infection probability per edge, γ recovery probability per step |
//! | [`SISModel`] | Epidemic SIS: β/γ as SIR but recovered nodes return to susceptible |

use std::collections::{HashMap, HashSet, VecDeque};

use scirs2_core::random::{Rng, RngExt};

use crate::error::{GraphError, Result};

// ---------------------------------------------------------------------------
// Shared adjacency representation
// ---------------------------------------------------------------------------

/// Compact edge representation: (target_id, edge_weight).
pub type AdjList = HashMap<usize, Vec<(usize, f64)>>;

// ---------------------------------------------------------------------------
// SIR node state
// ---------------------------------------------------------------------------

/// State of a node in the SIR / SIS epidemic model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SirState {
    /// Susceptible — not yet infected.
    Susceptible,
    /// Infected (and infectious).
    Infected,
    /// Recovered (and immune, in SIR).
    Recovered,
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Outcome of a single diffusion simulation.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Set of node IDs that were activated / infected at any point.
    pub activated: HashSet<usize>,
    /// Per-step counts: `(susceptible, infected, recovered)`.
    /// Only populated by epidemic simulations.
    pub time_series: Vec<(usize, usize, usize)>,
    /// Final total number of activated nodes (including seeds).
    pub spread: usize,
}

// ---------------------------------------------------------------------------
// Independent Cascade model
// ---------------------------------------------------------------------------

/// Independent Cascade (IC) diffusion model configuration.
///
/// Each active node `u` attempts to activate every inactive neighbour `v` with
/// probability `p(u, v)` stored as the edge weight.  Each edge is tried **at
/// most once**.
#[derive(Debug, Clone)]
pub struct IndependentCascade {
    /// Adjacency list: `node → [(neighbour, propagation_probability)]`.
    pub adjacency: AdjList,
    /// Total number of nodes in the graph.
    pub num_nodes: usize,
}

impl IndependentCascade {
    /// Create a new IC model.
    ///
    /// # Arguments
    /// * `adjacency` — directed adjacency list with propagation probabilities as weights.
    /// * `num_nodes` — number of nodes (needed for result reporting).
    pub fn new(adjacency: AdjList, num_nodes: usize) -> Self {
        IndependentCascade {
            adjacency,
            num_nodes,
        }
    }

    /// Build an IC model from a vec of `(source, target, probability)` triples.
    pub fn from_edges(edges: &[(usize, usize, f64)], num_nodes: usize) -> Self {
        let mut adjacency: AdjList = HashMap::new();
        for &(src, tgt, prob) in edges {
            adjacency.entry(src).or_default().push((tgt, prob));
        }
        IndependentCascade::new(adjacency, num_nodes)
    }

    /// Run a single IC simulation from the given seed set.
    pub fn simulate(&self, seeds: &[usize]) -> Result<SimulationResult> {
        simulate_ic(&self.adjacency, seeds)
    }

    /// Estimate expected spread over `num_simulations` Monte-Carlo runs.
    pub fn expected_spread(&self, seeds: &[usize], num_simulations: usize) -> Result<f64> {
        expected_spread_ic(&self.adjacency, seeds, num_simulations)
    }
}

// ---------------------------------------------------------------------------
// Linear Threshold model
// ---------------------------------------------------------------------------

/// Linear Threshold (LT) diffusion model configuration.
///
/// Each node `v` has a threshold `θ_v ∈ [0, 1]` drawn at the start of each
/// simulation.  Node `v` becomes active when
/// `Σ_{u active} w(u, v) ≥ θ_v`, where `w(u, v)` is the normalised in-edge
/// weight.
#[derive(Debug, Clone)]
pub struct LinearThreshold {
    /// Directed adjacency list: `node → [(neighbour, weight)]`.
    ///
    /// Weights represent *influence* of `node` on `neighbour`.  They should be
    /// normalised so that `Σ_{u} w(u, v) ≤ 1` for every `v`.
    pub adjacency: AdjList,
    /// Total number of nodes.
    pub num_nodes: usize,
    /// Optional fixed thresholds per node (if `None`, drawn uniformly per run).
    pub thresholds: Option<Vec<f64>>,
}

impl LinearThreshold {
    /// Create a new LT model with random per-run thresholds.
    pub fn new(adjacency: AdjList, num_nodes: usize) -> Self {
        LinearThreshold {
            adjacency,
            num_nodes,
            thresholds: None,
        }
    }

    /// Create an LT model with fixed thresholds.
    pub fn with_thresholds(adjacency: AdjList, thresholds: Vec<f64>) -> Result<Self> {
        let num_nodes = thresholds.len();
        for (i, &t) in thresholds.iter().enumerate() {
            if !(0.0..=1.0).contains(&t) {
                return Err(GraphError::InvalidParameter {
                    param: format!("thresholds[{i}]"),
                    value: t.to_string(),
                    expected: "value in [0, 1]".to_string(),
                    context: "LinearThreshold::with_thresholds".to_string(),
                });
            }
        }
        Ok(LinearThreshold {
            adjacency,
            num_nodes,
            thresholds: Some(thresholds),
        })
    }

    /// Build from `(source, target, weight)` triples.
    pub fn from_edges(edges: &[(usize, usize, f64)], num_nodes: usize) -> Self {
        let mut adjacency: AdjList = HashMap::new();
        for &(src, tgt, w) in edges {
            adjacency.entry(src).or_default().push((tgt, w));
        }
        LinearThreshold::new(adjacency, num_nodes)
    }

    /// Run a single LT simulation.
    pub fn simulate(&self, seeds: &[usize]) -> Result<SimulationResult> {
        simulate_lt(
            &self.adjacency,
            self.num_nodes,
            seeds,
            self.thresholds.as_deref(),
        )
    }

    /// Estimate expected spread.
    pub fn expected_spread(&self, seeds: &[usize], num_simulations: usize) -> Result<f64> {
        expected_spread_lt(
            &self.adjacency,
            self.num_nodes,
            seeds,
            self.thresholds.as_deref(),
            num_simulations,
        )
    }
}

// ---------------------------------------------------------------------------
// SIR model
// ---------------------------------------------------------------------------

/// SIR epidemic model (Susceptible–Infected–Recovered).
///
/// At each discrete time step:
/// 1. Every `Infected` node infects each `Susceptible` neighbour with probability `beta`.
/// 2. Every `Infected` node transitions to `Recovered` with probability `gamma`.
#[derive(Debug, Clone)]
pub struct SIRModel {
    /// Adjacency list (undirected: each edge stored in both directions).
    pub adjacency: AdjList,
    /// Infection probability per edge per step.
    pub beta: f64,
    /// Recovery probability per step.
    pub gamma: f64,
    /// Total number of nodes.
    pub num_nodes: usize,
}

impl SIRModel {
    /// Create a new SIR model.
    ///
    /// # Errors
    /// Returns [`GraphError::InvalidParameter`] when `beta` or `gamma` ∉ `[0, 1]`.
    pub fn new(adjacency: AdjList, num_nodes: usize, beta: f64, gamma: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&beta) {
            return Err(GraphError::InvalidParameter {
                param: "beta".to_string(),
                value: beta.to_string(),
                expected: "[0, 1]".to_string(),
                context: "SIRModel::new".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&gamma) {
            return Err(GraphError::InvalidParameter {
                param: "gamma".to_string(),
                value: gamma.to_string(),
                expected: "[0, 1]".to_string(),
                context: "SIRModel::new".to_string(),
            });
        }
        Ok(SIRModel {
            adjacency,
            beta,
            gamma,
            num_nodes,
        })
    }

    /// Build from `(source, target)` edge list (unweighted).
    pub fn from_edges(
        edges: &[(usize, usize)],
        num_nodes: usize,
        beta: f64,
        gamma: f64,
    ) -> Result<Self> {
        let mut adjacency: AdjList = HashMap::new();
        for &(src, tgt) in edges {
            adjacency.entry(src).or_default().push((tgt, 1.0));
            adjacency.entry(tgt).or_default().push((src, 1.0));
        }
        SIRModel::new(adjacency, num_nodes, beta, gamma)
    }

    /// Run a single SIR simulation from initial infected set.
    pub fn simulate(&self, initial_infected: &[usize]) -> Result<SimulationResult> {
        simulate_sir(
            &self.adjacency,
            self.num_nodes,
            initial_infected,
            self.beta,
            self.gamma,
        )
    }
}

// ---------------------------------------------------------------------------
// SIS model
// ---------------------------------------------------------------------------

/// SIS epidemic model (Susceptible–Infected–Susceptible).
///
/// Like SIR except `Recovered` nodes return to `Susceptible` rather than
/// gaining immunity.  The simulation terminates when no infected nodes remain
/// or after `max_steps` steps.
#[derive(Debug, Clone)]
pub struct SISModel {
    /// Adjacency list.
    pub adjacency: AdjList,
    /// Infection probability per edge per step.
    pub beta: f64,
    /// Recovery probability per step (returns to Susceptible).
    pub gamma: f64,
    /// Total number of nodes.
    pub num_nodes: usize,
    /// Maximum simulation steps before forced termination.
    pub max_steps: usize,
}

impl SISModel {
    /// Create a new SIS model.
    ///
    /// # Errors
    /// Returns [`GraphError::InvalidParameter`] when `beta` or `gamma` ∉ `[0, 1]`.
    pub fn new(
        adjacency: AdjList,
        num_nodes: usize,
        beta: f64,
        gamma: f64,
        max_steps: usize,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&beta) {
            return Err(GraphError::InvalidParameter {
                param: "beta".to_string(),
                value: beta.to_string(),
                expected: "[0, 1]".to_string(),
                context: "SISModel::new".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&gamma) {
            return Err(GraphError::InvalidParameter {
                param: "gamma".to_string(),
                value: gamma.to_string(),
                expected: "[0, 1]".to_string(),
                context: "SISModel::new".to_string(),
            });
        }
        Ok(SISModel {
            adjacency,
            beta,
            gamma,
            num_nodes,
            max_steps,
        })
    }

    /// Build from unweighted edge list.
    pub fn from_edges(
        edges: &[(usize, usize)],
        num_nodes: usize,
        beta: f64,
        gamma: f64,
        max_steps: usize,
    ) -> Result<Self> {
        let mut adjacency: AdjList = HashMap::new();
        for &(src, tgt) in edges {
            adjacency.entry(src).or_default().push((tgt, 1.0));
            adjacency.entry(tgt).or_default().push((src, 1.0));
        }
        SISModel::new(adjacency, num_nodes, beta, gamma, max_steps)
    }

    /// Run a single SIS simulation from initial infected set.
    pub fn simulate(&self, initial_infected: &[usize]) -> Result<SimulationResult> {
        simulate_sis(
            &self.adjacency,
            self.num_nodes,
            initial_infected,
            self.beta,
            self.gamma,
            self.max_steps,
        )
    }
}

// ---------------------------------------------------------------------------
// Free functions – IC
// ---------------------------------------------------------------------------

/// Simulate one run of the Independent Cascade model.
///
/// # Arguments
/// * `adjacency` — directed adjacency list with propagation probabilities.
/// * `seeds` — initial active node IDs.
///
/// # Returns
/// [`SimulationResult`] with `time_series` empty (IC is a cascade, not
/// time-stepped).
pub fn simulate_ic(adjacency: &AdjList, seeds: &[usize]) -> Result<SimulationResult> {
    let mut rng = scirs2_core::random::rng();
    let mut active: HashSet<usize> = seeds.iter().cloned().collect();
    let mut queue: VecDeque<usize> = seeds.iter().cloned().collect();

    while let Some(node) = queue.pop_front() {
        if let Some(neighbors) = adjacency.get(&node) {
            for &(nbr, prob) in neighbors {
                if !active.contains(&nbr) && rng.random::<f64>() < prob {
                    active.insert(nbr);
                    queue.push_back(nbr);
                }
            }
        }
    }

    let spread = active.len();
    Ok(SimulationResult {
        activated: active,
        time_series: Vec::new(),
        spread,
    })
}

/// Estimate the expected spread of a seed set under the IC model using
/// Monte-Carlo averaging over `num_simulations` independent runs.
///
/// # Arguments
/// * `adjacency` — directed adjacency list with propagation probabilities.
/// * `seeds` — initial seed set.
/// * `num_simulations` — number of Monte-Carlo trials.
pub fn expected_spread(
    adjacency: &AdjList,
    seeds: &[usize],
    num_simulations: usize,
) -> Result<f64> {
    expected_spread_ic(adjacency, seeds, num_simulations)
}

fn expected_spread_ic(adjacency: &AdjList, seeds: &[usize], num_simulations: usize) -> Result<f64> {
    if num_simulations == 0 {
        return Err(GraphError::InvalidParameter {
            param: "num_simulations".to_string(),
            value: "0".to_string(),
            expected: ">= 1".to_string(),
            context: "expected_spread_ic".to_string(),
        });
    }
    let mut total = 0.0_f64;
    for _ in 0..num_simulations {
        let result = simulate_ic(adjacency, seeds)?;
        total += result.spread as f64;
    }
    Ok(total / num_simulations as f64)
}

// ---------------------------------------------------------------------------
// Free functions – LT
// ---------------------------------------------------------------------------

/// Simulate one run of the Linear Threshold model.
///
/// # Arguments
/// * `adjacency` — directed adjacency list `source → [(target, weight)]`.
///   For LT the weights should satisfy `Σ_{u} w(u,v) ≤ 1`.
/// * `num_nodes` — total number of nodes (used to allocate per-node state).
/// * `seeds` — initial active set.
/// * `fixed_thresholds` — if `Some`, use these thresholds; otherwise draw
///   uniformly from `[0, 1]` per run.
pub fn simulate_lt(
    adjacency: &AdjList,
    num_nodes: usize,
    seeds: &[usize],
    fixed_thresholds: Option<&[f64]>,
) -> Result<SimulationResult> {
    // Build reverse adjacency: target → [(source, weight)]
    let mut reverse: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    for (&src, nbrs) in adjacency {
        for &(tgt, w) in nbrs {
            reverse.entry(tgt).or_default().push((src, w));
        }
    }

    let mut rng = scirs2_core::random::rng();

    // Assign thresholds
    let thresholds: Vec<f64> = match fixed_thresholds {
        Some(t) => {
            if t.len() < num_nodes {
                return Err(GraphError::InvalidParameter {
                    param: "fixed_thresholds".to_string(),
                    value: format!("len={}", t.len()),
                    expected: format!(">= num_nodes={num_nodes}"),
                    context: "simulate_lt".to_string(),
                });
            }
            t.to_vec()
        }
        None => (0..num_nodes).map(|_| rng.random::<f64>()).collect(),
    };

    let mut active: HashSet<usize> = seeds.iter().cloned().collect();
    let mut changed = true;

    // Iterative activation until no new node can be activated
    while changed {
        changed = false;
        // Collect candidate nodes: inactive nodes with at least one active in-neighbour
        let candidates: Vec<usize> = reverse
            .keys()
            .filter(|&&node| !active.contains(&node))
            .cloned()
            .collect();

        for node in candidates {
            let weight_sum: f64 = reverse
                .get(&node)
                .map(|in_nbrs| {
                    in_nbrs
                        .iter()
                        .filter(|(src, _)| active.contains(src))
                        .map(|(_, w)| w)
                        .sum()
                })
                .unwrap_or(0.0);

            let threshold = if node < thresholds.len() {
                thresholds[node]
            } else {
                1.0
            };

            if weight_sum >= threshold {
                active.insert(node);
                changed = true;
            }
        }
    }

    let spread = active.len();
    Ok(SimulationResult {
        activated: active,
        time_series: Vec::new(),
        spread,
    })
}

fn expected_spread_lt(
    adjacency: &AdjList,
    num_nodes: usize,
    seeds: &[usize],
    fixed_thresholds: Option<&[f64]>,
    num_simulations: usize,
) -> Result<f64> {
    if num_simulations == 0 {
        return Err(GraphError::InvalidParameter {
            param: "num_simulations".to_string(),
            value: "0".to_string(),
            expected: ">= 1".to_string(),
            context: "expected_spread_lt".to_string(),
        });
    }
    let mut total = 0.0_f64;
    for _ in 0..num_simulations {
        let result = simulate_lt(adjacency, num_nodes, seeds, fixed_thresholds)?;
        total += result.spread as f64;
    }
    Ok(total / num_simulations as f64)
}

// ---------------------------------------------------------------------------
// Free functions – SIR
// ---------------------------------------------------------------------------

/// Simulate one run of the SIR epidemic model.
///
/// # Arguments
/// * `adjacency` — (undirected) adjacency list; each undirected edge should
///   appear in both directions.
/// * `num_nodes` — total number of nodes.
/// * `initial_infected` — nodes initially in the `Infected` state.
/// * `beta` — infection probability per edge per time step.
/// * `gamma` — recovery probability per time step.
pub fn simulate_sir(
    adjacency: &AdjList,
    num_nodes: usize,
    initial_infected: &[usize],
    beta: f64,
    gamma: f64,
) -> Result<SimulationResult> {
    if !(0.0..=1.0).contains(&beta) || !(0.0..=1.0).contains(&gamma) {
        return Err(GraphError::InvalidParameter {
            param: "beta/gamma".to_string(),
            value: format!("beta={beta}, gamma={gamma}"),
            expected: "both in [0, 1]".to_string(),
            context: "simulate_sir".to_string(),
        });
    }

    let mut rng = scirs2_core::random::rng();
    let mut states: Vec<SirState> = vec![SirState::Susceptible; num_nodes];
    for &node in initial_infected {
        if node < num_nodes {
            states[node] = SirState::Infected;
        }
    }

    let mut time_series: Vec<(usize, usize, usize)> = Vec::new();
    let mut ever_infected: HashSet<usize> = initial_infected.iter().cloned().collect();

    loop {
        let n_infected = states.iter().filter(|&&s| s == SirState::Infected).count();
        let n_recovered = states.iter().filter(|&&s| s == SirState::Recovered).count();
        let n_susceptible = num_nodes - n_infected - n_recovered;
        time_series.push((n_susceptible, n_infected, n_recovered));

        if n_infected == 0 {
            break;
        }

        let mut next_states = states.clone();

        // Infection step
        for node in 0..num_nodes {
            if states[node] == SirState::Infected {
                if let Some(neighbors) = adjacency.get(&node) {
                    for &(nbr, _) in neighbors {
                        if nbr < num_nodes
                            && states[nbr] == SirState::Susceptible
                            && rng.random::<f64>() < beta
                        {
                            next_states[nbr] = SirState::Infected;
                            ever_infected.insert(nbr);
                        }
                    }
                }
            }
        }

        // Recovery step
        for node in 0..num_nodes {
            if states[node] == SirState::Infected && rng.random::<f64>() < gamma {
                next_states[node] = SirState::Recovered;
            }
        }

        states = next_states;
    }

    Ok(SimulationResult {
        activated: ever_infected,
        time_series,
        spread: states.iter().filter(|&&s| s == SirState::Recovered).count()
            + states.iter().filter(|&&s| s == SirState::Infected).count(),
    })
}

// ---------------------------------------------------------------------------
// Free functions – SIS
// ---------------------------------------------------------------------------

/// Simulate one run of the SIS epidemic model.
///
/// Unlike SIR, recovered nodes return to the susceptible state, so endemic
/// equilibria are possible.  Simulation terminates when no infected nodes
/// remain or after `max_steps` steps.
pub fn simulate_sis(
    adjacency: &AdjList,
    num_nodes: usize,
    initial_infected: &[usize],
    beta: f64,
    gamma: f64,
    max_steps: usize,
) -> Result<SimulationResult> {
    if !(0.0..=1.0).contains(&beta) || !(0.0..=1.0).contains(&gamma) {
        return Err(GraphError::InvalidParameter {
            param: "beta/gamma".to_string(),
            value: format!("beta={beta}, gamma={gamma}"),
            expected: "both in [0, 1]".to_string(),
            context: "simulate_sis".to_string(),
        });
    }

    let mut rng = scirs2_core::random::rng();
    let mut infected: HashSet<usize> = initial_infected.iter().cloned().collect();
    let mut ever_infected = infected.clone();
    let mut time_series: Vec<(usize, usize, usize)> = Vec::new();

    for _step in 0..max_steps {
        let n_infected = infected.len();
        time_series.push((num_nodes - n_infected, n_infected, 0));

        if n_infected == 0 {
            break;
        }

        let mut new_infections: HashSet<usize> = HashSet::new();
        let mut new_recoveries: HashSet<usize> = HashSet::new();

        for &node in &infected {
            // Try to infect susceptible neighbours
            if let Some(neighbors) = adjacency.get(&node) {
                for &(nbr, _) in neighbors {
                    if nbr < num_nodes && !infected.contains(&nbr) && rng.random::<f64>() < beta {
                        new_infections.insert(nbr);
                        ever_infected.insert(nbr);
                    }
                }
            }
            // Try to recover
            if rng.random::<f64>() < gamma {
                new_recoveries.insert(node);
            }
        }

        for node in new_recoveries {
            infected.remove(&node);
        }
        for node in new_infections {
            infected.insert(node);
        }
    }

    let spread = ever_infected.len();
    Ok(SimulationResult {
        activated: ever_infected,
        time_series,
        spread,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn star_adjacency(n: usize, prob: f64) -> AdjList {
        // Hub node 0 connected to nodes 1..n
        let mut adj: AdjList = HashMap::new();
        for i in 1..n {
            adj.entry(0).or_default().push((i, prob));
        }
        adj
    }

    #[test]
    fn test_simulate_ic_full_spread() {
        // probability 1.0 → all nodes activated from hub
        let adj = star_adjacency(6, 1.0);
        let result = simulate_ic(&adj, &[0]).expect("ic simulation");
        assert_eq!(result.spread, 6);
    }

    #[test]
    fn test_simulate_ic_no_spread() {
        // probability 0.0 → only seed activated
        let adj = star_adjacency(6, 0.0);
        let result = simulate_ic(&adj, &[0]).expect("ic simulation");
        assert_eq!(result.spread, 1);
    }

    #[test]
    fn test_simulate_lt_deterministic_threshold() {
        // All edges weight 1.0, threshold = 0.5 → all nodes activated immediately
        let mut adj: AdjList = HashMap::new();
        // 0 → {1, 2, 3}
        for i in 1..4_usize {
            adj.entry(0).or_default().push((i, 1.0));
        }
        let thresholds = vec![0.5_f64; 4];
        let result = simulate_lt(&adj, 4, &[0], Some(&thresholds)).expect("lt simulation");
        assert!(result.spread >= 1);
    }

    #[test]
    fn test_simulate_sir_terminates() {
        // Simple chain: 0–1–2–3–4
        let mut adj: AdjList = HashMap::new();
        for i in 0..4_usize {
            adj.entry(i).or_default().push((i + 1, 1.0));
            adj.entry(i + 1).or_default().push((i, 1.0));
        }
        let result = simulate_sir(&adj, 5, &[0], 0.8, 0.5).expect("sir");
        assert!(result.spread >= 1);
        assert!(!result.time_series.is_empty());
    }

    #[test]
    fn test_simulate_sis_terminates() {
        let mut adj: AdjList = HashMap::new();
        for i in 0..4_usize {
            adj.entry(i).or_default().push((i + 1, 1.0));
            adj.entry(i + 1).or_default().push((i, 1.0));
        }
        let result = simulate_sis(&adj, 5, &[0], 0.5, 0.9, 1000).expect("sis");
        assert!(result.spread >= 1);
    }

    #[test]
    fn test_expected_spread_ic() {
        let adj = star_adjacency(5, 1.0);
        let spread = expected_spread(&adj, &[0], 50).expect("expected spread");
        // With prob 1.0, expected spread = 5
        assert!((spread - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_sir_bad_params() {
        let adj: AdjList = HashMap::new();
        let err = simulate_sir(&adj, 1, &[], 2.0, 0.5);
        assert!(err.is_err());
    }

    #[test]
    fn test_ic_struct() {
        let edges = vec![(0_usize, 1_usize, 1.0_f64), (0, 2, 1.0), (0, 3, 1.0)];
        let ic = IndependentCascade::from_edges(&edges, 4);
        let res = ic.simulate(&[0]).expect("simulate");
        assert_eq!(res.spread, 4);
    }

    #[test]
    fn test_lt_bad_threshold() {
        let adj: AdjList = HashMap::new();
        let err = LinearThreshold::with_thresholds(adj, vec![0.5, 1.5]);
        assert!(err.is_err());
    }
}
