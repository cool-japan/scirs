//! Search configuration for symbolic regression.

/// Available building blocks for symbolic regression search.
///
/// Each variant enables a family of operators in the candidate grammar.
/// Defaults via [`SrConfig::default`] enable arithmetic, power, trig, and
/// exp/log; hyperbolic and sqrt/abs are off by default to keep the
/// initial search frontier small.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuildingBlock {
    /// `+`, `-`, `*`, `/`
    Arithmetic,
    /// `pow`
    Pow,
    /// `sin`, `cos`, `tan`
    Trig,
    /// `exp`, `ln`
    ExpLog,
    /// `sqrt`, `abs`
    SqrtAbs,
    /// `sinh`, `cosh`, `tanh`
    Hyperbolic,
}

/// Configuration for [`fn@crate::regression::discover`].
///
/// Builder methods return `Self` so configurations can be composed with
/// fluent syntax: `SrConfig::default().with_max_iter(50).with_top_n(3)`.
#[derive(Clone, Debug)]
pub struct SrConfig {
    /// Maximum tree depth (limits formula complexity).
    pub max_depth: usize,
    /// Maximum number of nodes in the formula.
    pub max_nodes: usize,
    /// Beam width — number of best candidates kept per generation.
    pub beam_width: usize,
    /// Number of search iterations.
    pub max_iter: usize,
    /// Building blocks to use in the search.
    pub building_blocks: Vec<BuildingBlock>,
    /// Number of top formulas to return.
    pub top_n: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Tolerance: stop early if best MSE drops below this.
    pub tolerance: f64,
    /// Penalty per node (parsimony pressure).
    pub complexity_penalty: f64,
}

impl Default for SrConfig {
    fn default() -> Self {
        Self {
            max_depth: 6,
            max_nodes: 20,
            beam_width: 32,
            max_iter: 100,
            building_blocks: vec![
                BuildingBlock::Arithmetic,
                BuildingBlock::Pow,
                BuildingBlock::Trig,
                BuildingBlock::ExpLog,
            ],
            top_n: 5,
            seed: 42,
            tolerance: 1e-9,
            complexity_penalty: 0.001,
        }
    }
}

impl SrConfig {
    /// Builder method: set `max_depth`.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    /// Builder method: set `max_nodes`.
    pub fn with_max_nodes(mut self, n: usize) -> Self {
        self.max_nodes = n;
        self
    }

    /// Builder method: set `beam_width`.
    pub fn with_beam_width(mut self, b: usize) -> Self {
        self.beam_width = b;
        self
    }

    /// Builder method: set `max_iter`.
    pub fn with_max_iter(mut self, i: usize) -> Self {
        self.max_iter = i;
        self
    }

    /// Builder method: set `top_n`.
    pub fn with_top_n(mut self, n: usize) -> Self {
        self.top_n = n;
        self
    }

    /// Builder method: set the random seed.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Builder method: set the early-stopping tolerance on MSE.
    pub fn with_tolerance(mut self, t: f64) -> Self {
        self.tolerance = t;
        self
    }

    /// Builder method: set the per-node parsimony penalty.
    pub fn with_complexity_penalty(mut self, p: f64) -> Self {
        self.complexity_penalty = p;
        self
    }

    /// Builder method: set the building blocks (replaces default set).
    pub fn with_building_blocks(mut self, blocks: Vec<BuildingBlock>) -> Self {
        self.building_blocks = blocks;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_arithmetic_and_trig() {
        let c = SrConfig::default();
        assert!(c.building_blocks.contains(&BuildingBlock::Arithmetic));
        assert!(c.building_blocks.contains(&BuildingBlock::Trig));
    }

    #[test]
    fn builder_chains() {
        let c = SrConfig::default()
            .with_max_iter(7)
            .with_top_n(2)
            .with_seed(99);
        assert_eq!(c.max_iter, 7);
        assert_eq!(c.top_n, 2);
        assert_eq!(c.seed, 99);
    }
}
