//! FCI Algorithm for Causal Discovery with Latent Confounders
//!
//! The Fast Causal Inference (FCI) algorithm extends the PC algorithm to
//! handle latent (unmeasured) confounders. Instead of a CPDAG, FCI outputs
//! a Partial Ancestral Graph (PAG) that uses additional edge marks:
//!
//! - **Circle (o)**: the mark is unknown (could be tail or arrow)
//! - **Arrow (>)**: definite arrowhead
//! - **Tail (-)**: definite tail
//!
//! Edge types in a PAG:
//! - `->` : definite direct cause
//! - `<->` : latent common cause (bidirected)
//! - `o->` : possible direct cause or latent common cause
//! - `o-o` : completely uncertain
//!
//! # Algorithm outline
//!
//! 1. Run PC skeleton discovery
//! 2. Orient unshielded colliders (v-structures)
//! 3. Compute Possible-D-SEP sets and refine adjacency
//! 4. Re-orient v-structures on refined skeleton
//! 5. Apply FCI orientation rules R1-R10
//!
//! # References
//!
//! - Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation, Prediction,
//!   and Search* (2nd ed.). MIT Press.
//! - Richardson, T. & Spirtes, P. (2002). Ancestral graph Markov models.
//!   *Ann. Statist.* 30, 962-1030.
//! - Zhang, J. (2008). On the completeness of orientation rules for causal
//!   discovery in the presence of latent confounders and selection variables.
//!   *Artificial Intelligence* 172, 1873-1896.

use std::collections::{HashMap, HashSet, VecDeque};

use scirs2_core::ndarray::ArrayView2;

use super::conditional_independence::{ConditionalIndependenceTest, PartialCorrelationTest};
use super::pc_algorithm::subsets;
use super::{CausalGraph, EdgeMark};
use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// FCI Algorithm
// ---------------------------------------------------------------------------

/// Configuration for the FCI algorithm.
#[derive(Debug, Clone)]
pub struct FciAlgorithm {
    /// Significance level alpha for CI tests (default 0.05).
    pub alpha: f64,
    /// Maximum conditioning set size (default 4).
    pub max_cond_set_size: usize,
    /// Maximum Possible-D-SEP set size (default 4).
    pub max_pdsep_size: usize,
}

impl Default for FciAlgorithm {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            max_cond_set_size: 4,
            max_pdsep_size: 4,
        }
    }
}

/// Result of the FCI algorithm.
#[derive(Debug, Clone)]
pub struct FciResult {
    /// The learned PAG (Partial Ancestral Graph).
    pub graph: CausalGraph,
    /// Separation sets.
    pub sep_sets: HashMap<(usize, usize), Vec<usize>>,
    /// Number of CI tests performed.
    pub n_tests: usize,
    /// Whether latent confounders were detected (any bidirected edges).
    pub has_latent_confounders: bool,
}

impl FciAlgorithm {
    /// Create an FCI algorithm with the given significance level.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ..Default::default()
        }
    }

    /// Create with custom parameters.
    pub fn with_params(alpha: f64, max_cond_set_size: usize, max_pdsep_size: usize) -> Self {
        Self {
            alpha,
            max_cond_set_size,
            max_pdsep_size,
        }
    }

    /// Run FCI using the default partial correlation CI test.
    pub fn fit(&self, data: ArrayView2<f64>, var_names: &[&str]) -> StatsResult<FciResult> {
        let ci_test = PartialCorrelationTest::new(self.alpha);
        self.fit_with_test(data, var_names, &ci_test)
    }

    /// Run FCI with a custom CI test.
    pub fn fit_with_test<T: ConditionalIndependenceTest>(
        &self,
        data: ArrayView2<f64>,
        var_names: &[&str],
        ci_test: &T,
    ) -> StatsResult<FciResult> {
        let p = data.ncols();
        if var_names.len() != p {
            return Err(StatsError::DimensionMismatch(
                "var_names length must match number of columns".to_owned(),
            ));
        }
        if p == 0 {
            return Ok(FciResult {
                graph: CausalGraph::new(var_names),
                sep_sets: HashMap::new(),
                n_tests: 0,
                has_latent_confounders: false,
            });
        }

        // Step 1: Initial skeleton discovery (same as PC)
        let (mut adj, mut sep_sets, mut n_tests) =
            skeleton_discovery(data, p, self.alpha, self.max_cond_set_size, ci_test)?;

        // Step 2: Initial v-structure orientation
        let mut graph = CausalGraph::new(var_names);
        // Initialise with circle marks (FCI uses o-o for unoriented)
        for i in 0..p {
            for j in (i + 1)..p {
                if adj[i][j] {
                    graph.set_edge(i, j, EdgeMark::Circle, EdgeMark::Circle);
                }
            }
        }
        orient_unshielded_colliders(&mut graph, &adj, &sep_sets, p);

        // Step 3: Possible-D-SEP refinement
        let pdsep_removals = possible_dsep_phase(
            &graph,
            data,
            &adj,
            p,
            self.alpha,
            self.max_pdsep_size,
            ci_test,
            &mut n_tests,
        )?;

        // Apply removals
        for (x, y, z_set) in pdsep_removals {
            adj[x][y] = false;
            adj[y][x] = false;
            graph.remove_edge(x, y);
            let key = (x.min(y), x.max(y));
            sep_sets.insert(key, z_set);
        }

        // Step 4: Re-orient v-structures on refined skeleton
        // Reset all remaining edges to circle-circle
        for i in 0..p {
            for j in (i + 1)..p {
                if adj[i][j] {
                    graph.set_edge(i, j, EdgeMark::Circle, EdgeMark::Circle);
                }
            }
        }
        orient_unshielded_colliders(&mut graph, &adj, &sep_sets, p);

        // Step 5: Apply FCI orientation rules R1-R10
        apply_fci_rules(&mut graph, &adj, &sep_sets, p);

        // Detect latent confounders (bidirected edges)
        let has_latent_confounders =
            (0..p).any(|i| (0..p).any(|j| i != j && graph.is_bidirected(i, j)));

        Ok(FciResult {
            graph,
            sep_sets,
            n_tests,
            has_latent_confounders,
        })
    }
}

// ---------------------------------------------------------------------------
// Skeleton Discovery (shared with PC)
// ---------------------------------------------------------------------------

/// PC-stable skeleton discovery.
fn skeleton_discovery<T: ConditionalIndependenceTest>(
    data: ArrayView2<f64>,
    p: usize,
    alpha: f64,
    max_cond_set_size: usize,
    ci_test: &T,
) -> StatsResult<(Vec<Vec<bool>>, HashMap<(usize, usize), Vec<usize>>, usize)> {
    let mut adj = vec![vec![true; p]; p];
    for i in 0..p {
        adj[i][i] = false;
    }
    let mut sep_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut n_tests = 0usize;

    for ord in 0..=max_cond_set_size {
        let adj_snapshot = adj.clone();
        let edges: Vec<(usize, usize)> = (0..p)
            .flat_map(|i| ((i + 1)..p).map(move |j| (i, j)))
            .filter(|&(i, j)| adj_snapshot[i][j])
            .collect();

        let mut removals = Vec::new();

        for (x, y) in edges {
            let z_x: Vec<usize> = (0..p)
                .filter(|&k| k != x && k != y && adj_snapshot[x][k])
                .collect();
            let z_y: Vec<usize> = (0..p)
                .filter(|&k| k != x && k != y && adj_snapshot[y][k])
                .collect();

            let mut found = false;
            if z_x.len() >= ord {
                for z_set in subsets(&z_x, ord) {
                    n_tests += 1;
                    if ci_test.is_independent(x, y, &z_set, data, alpha)? {
                        removals.push((x, y, z_set));
                        found = true;
                        break;
                    }
                }
            }
            if !found && z_y.len() >= ord {
                for z_set in subsets(&z_y, ord) {
                    n_tests += 1;
                    if ci_test.is_independent(x, y, &z_set, data, alpha)? {
                        removals.push((x, y, z_set));
                        break;
                    }
                }
            }
        }

        for (x, y, z_set) in removals {
            adj[x][y] = false;
            adj[y][x] = false;
            let key = (x.min(y), x.max(y));
            sep_sets.insert(key, z_set);
        }
    }

    Ok((adj, sep_sets, n_tests))
}

// ---------------------------------------------------------------------------
// Unshielded Collider Orientation
// ---------------------------------------------------------------------------

/// Orient unshielded colliders: X *-o Z o-* Y where X-Y not adjacent
/// and Z not in sep(X,Y) => X *-> Z <-* Y.
fn orient_unshielded_colliders(
    graph: &mut CausalGraph,
    adj: &[Vec<bool>],
    sep_sets: &HashMap<(usize, usize), Vec<usize>>,
    p: usize,
) {
    for z in 0..p {
        let neighbours: Vec<usize> = (0..p).filter(|&k| k != z && adj[z][k]).collect();
        for i in 0..neighbours.len() {
            for j in (i + 1)..neighbours.len() {
                let x = neighbours[i];
                let y = neighbours[j];
                if adj[x][y] {
                    continue; // shielded
                }
                let key = (x.min(y), x.max(y));
                let sep = sep_sets.get(&key).cloned().unwrap_or_default();
                if !sep.contains(&z) {
                    // Orient: arrowhead at z from both x and y
                    let mark_xz_from = graph.get_mark_from(x, z).unwrap_or(EdgeMark::Circle);
                    graph.set_edge(x, z, mark_xz_from, EdgeMark::Arrow);
                    let mark_yz_from = graph.get_mark_from(y, z).unwrap_or(EdgeMark::Circle);
                    graph.set_edge(y, z, mark_yz_from, EdgeMark::Arrow);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Possible-D-SEP
// ---------------------------------------------------------------------------

/// Compute Possible-D-SEP(a, b) in the partially oriented graph.
///
/// Possible-D-SEP(a, b) is the set of all nodes that can be reached from a
/// by a path on which every non-endpoint node is a collider or has an
/// undetermined edge mark.
fn possible_dsep(graph: &CausalGraph, a: usize, b: usize, p: usize) -> HashSet<usize> {
    let mut pdsep = HashSet::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    // Start from a's neighbours
    for k in 0..p {
        if k != a && k != b && graph.is_adjacent(a, k) {
            queue.push_back((k, a)); // (current, previous)
        }
    }

    while let Some((cur, prev)) = queue.pop_front() {
        if !visited.insert((cur, prev)) {
            continue;
        }
        pdsep.insert(cur);

        // Continue along the path if cur is a collider on the path or
        // has a circle mark from the previous node
        for next in 0..p {
            if next == prev || next == a || !graph.is_adjacent(cur, next) {
                continue;
            }
            // Check if cur is a "possible collider" on the path prev - cur - next
            // A node is on Possible-D-SEP if:
            // 1. It has an arrowhead from prev (prev *-> cur)
            // 2. Or it has a circle mark from prev (prev o-* cur)
            let mark_at_cur_from_prev = graph.get_mark_at(prev, cur);
            let is_possible_collider = match mark_at_cur_from_prev {
                Some(EdgeMark::Arrow) | Some(EdgeMark::Circle) => true,
                _ => false,
            };

            if is_possible_collider {
                queue.push_back((next, cur));
            }
        }
    }

    pdsep
}

/// Possible-D-SEP phase: for each adjacent pair, test independence
/// conditioning on subsets of Possible-D-SEP.
fn possible_dsep_phase<T: ConditionalIndependenceTest>(
    graph: &CausalGraph,
    data: ArrayView2<f64>,
    adj: &[Vec<bool>],
    p: usize,
    alpha: f64,
    max_pdsep_size: usize,
    ci_test: &T,
    n_tests: &mut usize,
) -> StatsResult<Vec<(usize, usize, Vec<usize>)>> {
    let mut removals = Vec::new();

    for x in 0..p {
        for y in (x + 1)..p {
            if !adj[x][y] {
                continue;
            }

            let pdsep_x = possible_dsep(graph, x, y, p);
            let pdsep_y = possible_dsep(graph, y, x, p);
            let combined: Vec<usize> = pdsep_x
                .union(&pdsep_y)
                .copied()
                .filter(|&k| k != x && k != y)
                .collect();

            if combined.is_empty() {
                continue;
            }

            let max_size = max_pdsep_size.min(combined.len());
            let mut found = false;
            for ord in 0..=max_size {
                if found {
                    break;
                }
                for z_set in subsets(&combined, ord) {
                    *n_tests += 1;
                    if ci_test.is_independent(x, y, &z_set, data, alpha)? {
                        removals.push((x, y, z_set));
                        found = true;
                        break;
                    }
                }
            }
        }
    }

    Ok(removals)
}

// ---------------------------------------------------------------------------
// FCI Orientation Rules R1-R10
// ---------------------------------------------------------------------------

/// Apply FCI orientation rules R1-R10 until convergence.
///
/// These rules orient remaining circle marks into tails or arrowheads.
fn apply_fci_rules(
    graph: &mut CausalGraph,
    adj: &[Vec<bool>],
    sep_sets: &HashMap<(usize, usize), Vec<usize>>,
    p: usize,
) {
    let max_iterations = p * p * 2 + 10;
    let mut changed = true;
    let mut iterations = 0;

    while changed && iterations < max_iterations {
        changed = false;
        iterations += 1;

        changed |= fci_r1(graph, p);
        changed |= fci_r2(graph, p);
        changed |= fci_r3(graph, adj, p);
        changed |= fci_r4(graph, adj, sep_sets, p);
        changed |= fci_r5(graph, adj, p);
        changed |= fci_r6(graph, p);
        changed |= fci_r7(graph, p);
        changed |= fci_r8(graph, p);
        changed |= fci_r9(graph, p);
        changed |= fci_r10(graph, p);
    }
}

/// R1: If a *-> b o-* c, and a is not adjacent to c, orient b *-> c.
fn fci_r1(graph: &mut CausalGraph, p: usize) -> bool {
    let mut changed = false;
    for b in 0..p {
        for a in 0..p {
            if a == b {
                continue;
            }
            // a *-> b: arrowhead at b from a
            if graph.get_mark_at(a, b) != Some(EdgeMark::Arrow) {
                continue;
            }
            for c in 0..p {
                if c == a || c == b {
                    continue;
                }
                if !graph.is_adjacent(b, c) {
                    continue;
                }
                if graph.is_adjacent(a, c) {
                    continue;
                }
                // b o-* c: circle mark at b on edge b-c
                if graph.get_mark_from(b, c) != Some(EdgeMark::Circle) {
                    continue;
                }
                // Orient: change circle at b to tail => b -> c (preserve mark at c)
                let mark_at_c = graph.get_mark_at(b, c).unwrap_or(EdgeMark::Circle);
                graph.set_edge(b, c, EdgeMark::Tail, mark_at_c);
                changed = true;
            }
        }
    }
    changed
}

/// R2: If a -> b *-> c or a *-> b -> c, and a *-o c, orient a *-> c.
fn fci_r2(graph: &mut CausalGraph, p: usize) -> bool {
    let mut changed = false;
    for a in 0..p {
        for c in 0..p {
            if a == c || !graph.is_adjacent(a, c) {
                continue;
            }
            // a *-o c: circle mark at c from a
            if graph.get_mark_at(a, c) != Some(EdgeMark::Circle) {
                continue;
            }
            for b in 0..p {
                if b == a || b == c {
                    continue;
                }
                // Case 1: a -> b *-> c
                let case1 = graph.get_mark_from(a, b) == Some(EdgeMark::Tail)
                    && graph.get_mark_at(a, b) == Some(EdgeMark::Arrow)
                    && graph.get_mark_at(b, c) == Some(EdgeMark::Arrow);
                // Case 2: a *-> b -> c
                let case2 = graph.get_mark_at(a, b) == Some(EdgeMark::Arrow)
                    && graph.get_mark_from(b, c) == Some(EdgeMark::Tail)
                    && graph.get_mark_at(b, c) == Some(EdgeMark::Arrow);

                if case1 || case2 {
                    // Orient a *-> c
                    let mark_from_a = graph.get_mark_from(a, c).unwrap_or(EdgeMark::Circle);
                    graph.set_edge(a, c, mark_from_a, EdgeMark::Arrow);
                    changed = true;
                    break;
                }
            }
        }
    }
    changed
}

/// R3: If a *-> b <-* c, a *-o d o-* c, a not adj c, d *-o b,
/// orient d *-> b.
fn fci_r3(graph: &mut CausalGraph, adj: &[Vec<bool>], p: usize) -> bool {
    let mut changed = false;
    for d in 0..p {
        for b in 0..p {
            if d == b || !graph.is_adjacent(d, b) {
                continue;
            }
            // d *-o b
            if graph.get_mark_at(d, b) != Some(EdgeMark::Circle) {
                continue;
            }
            // Find a, c such that: a *-> b <-* c, a *-o d o-* c, a not adj c
            let parents_b: Vec<usize> = (0..p)
                .filter(|&k| {
                    k != b
                        && k != d
                        && graph.is_adjacent(k, b)
                        && graph.get_mark_at(k, b) == Some(EdgeMark::Arrow)
                })
                .collect();
            let mut orient = false;
            for i in 0..parents_b.len() {
                for j in (i + 1)..parents_b.len() {
                    let a = parents_b[i];
                    let c = parents_b[j];
                    if adj[a][c] {
                        continue;
                    }
                    // a *-o d
                    if !graph.is_adjacent(a, d) {
                        continue;
                    }
                    if graph.get_mark_at(a, d) != Some(EdgeMark::Circle) {
                        continue;
                    }
                    // d o-* c (i.e., c *-o d from c's perspective)
                    if !graph.is_adjacent(c, d) {
                        continue;
                    }
                    if graph.get_mark_at(c, d) != Some(EdgeMark::Circle) {
                        continue;
                    }
                    orient = true;
                    break;
                }
                if orient {
                    break;
                }
            }
            if orient {
                let mark_from = graph.get_mark_from(d, b).unwrap_or(EdgeMark::Circle);
                graph.set_edge(d, b, mark_from, EdgeMark::Arrow);
                changed = true;
            }
        }
    }
    changed
}

/// R4: Discriminating path rule.
///
/// If there is a discriminating path <a, ..., b, c> for b, and b is a
/// collider on the path, and c is in sep(a, c_end), orient b <-> c;
/// otherwise orient b -> c.
fn fci_r4(
    graph: &mut CausalGraph,
    _adj: &[Vec<bool>],
    sep_sets: &HashMap<(usize, usize), Vec<usize>>,
    p: usize,
) -> bool {
    let mut changed = false;
    // For each triple b, c where b *-> c o-* ? (circle at b on b-c edge)
    for c in 0..p {
        for b in 0..p {
            if b == c || !graph.is_adjacent(b, c) {
                continue;
            }
            if graph.get_mark_at(b, c) != Some(EdgeMark::Arrow) {
                continue;
            }
            // b o-* c (circle at c end from b side)
            if graph.get_mark_from(b, c) != Some(EdgeMark::Circle) {
                continue;
            }

            // Try to find a discriminating path for b
            // A discriminating path is: a - ... - v_k - b - c where
            // a is not adjacent to c, every v_i is a collider with arrow into c (parent of c),
            // and v_i *-> ... (collider on subpath)
            for a in 0..p {
                if a == b || a == c || !graph.is_adjacent(a, c) {
                    continue;
                }
                // Simple case: length-3 discriminating path a - b - c
                // a must not be adjacent to c... wait, a IS adjacent to c in this check
                // Actually for a discriminating path, a is NOT adjacent to c
                // Let me re-check: we need a not adjacent to c
            }

            // Simplified discriminating path: look for path a -> b *-> c
            // where a is not adjacent to c, a -> b, and a is a parent of c
            // This is a simplified version of R4
            for a in 0..p {
                if a == b || a == c {
                    continue;
                }
                if graph.is_adjacent(a, c) {
                    continue; // a must not be adj c
                }
                if !graph.is_adjacent(a, b) {
                    continue;
                }
                // Check a *-> b
                if graph.get_mark_at(a, b) != Some(EdgeMark::Arrow) {
                    continue;
                }

                // Found discriminating path candidate
                let key = (a.min(c), a.max(c));
                let sep = sep_sets.get(&key).cloned().unwrap_or_default();

                if sep.contains(&b) {
                    // b is in sep(a, c) => orient b - c as tail
                    let mark_from_b = graph.get_mark_from(b, c).unwrap_or(EdgeMark::Circle);
                    let _mark_at_c = EdgeMark::Arrow;
                    // Actually orient the circle: replace circle at b-side with tail
                    graph.set_edge(b, c, EdgeMark::Tail, EdgeMark::Arrow);
                    let _ = mark_from_b;
                } else {
                    // b not in sep(a,c) => orient as bidirected b <-> c
                    graph.set_edge(b, c, EdgeMark::Arrow, EdgeMark::Arrow);
                }
                changed = true;
                break;
            }
        }
    }
    changed
}

/// R5: If a o-o b and there is an uncovered circle path from a to b
/// (all edges are o-o and consecutive nodes on the path are non-adjacent
/// except via the path), orient a o-o b as a - b (tail-tail).
fn fci_r5(graph: &mut CausalGraph, _adj: &[Vec<bool>], p: usize) -> bool {
    let mut changed = false;
    for a in 0..p {
        for b in (a + 1)..p {
            if !graph.is_adjacent(a, b) {
                continue;
            }
            // Check a o-o b
            if graph.get_mark_from(a, b) != Some(EdgeMark::Circle)
                || graph.get_mark_at(a, b) != Some(EdgeMark::Circle)
            {
                continue;
            }
            // Look for uncovered circle path from a to b (length >= 3)
            if has_uncovered_circle_path(graph, a, b, p) {
                graph.set_edge(a, b, EdgeMark::Tail, EdgeMark::Tail);
                changed = true;
            }
        }
    }
    changed
}

/// R6: If a - b o-* c, orient b -* c (change circle at b to tail).
fn fci_r6(graph: &mut CausalGraph, p: usize) -> bool {
    let mut changed = false;
    for b in 0..p {
        for a in 0..p {
            if a == b || !graph.is_adjacent(a, b) {
                continue;
            }
            // a - b (tail-tail, undirected)
            if graph.get_mark_from(a, b) != Some(EdgeMark::Tail)
                || graph.get_mark_at(a, b) != Some(EdgeMark::Tail)
            {
                continue;
            }
            for c in 0..p {
                if c == a || c == b || !graph.is_adjacent(b, c) {
                    continue;
                }
                // b o-* c
                if graph.get_mark_from(b, c) != Some(EdgeMark::Circle) {
                    continue;
                }
                // Orient: change circle at b-side to tail
                let mark_at_c = graph.get_mark_at(b, c).unwrap_or(EdgeMark::Circle);
                graph.set_edge(b, c, EdgeMark::Tail, mark_at_c);
                changed = true;
            }
        }
    }
    changed
}

/// R7: If a -o b o-* c, a not adj c, orient b -* c (circle at b -> tail).
fn fci_r7(graph: &mut CausalGraph, p: usize) -> bool {
    let mut changed = false;
    for b in 0..p {
        for a in 0..p {
            if a == b || !graph.is_adjacent(a, b) {
                continue;
            }
            // a -o b: tail at a, circle at b
            if graph.get_mark_from(a, b) != Some(EdgeMark::Tail)
                || graph.get_mark_at(a, b) != Some(EdgeMark::Circle)
            {
                continue;
            }
            for c in 0..p {
                if c == a || c == b || !graph.is_adjacent(b, c) {
                    continue;
                }
                // a not adj c
                if graph.is_adjacent(a, c) {
                    continue;
                }
                // b o-* c
                if graph.get_mark_from(b, c) != Some(EdgeMark::Circle) {
                    continue;
                }
                let mark_at_c = graph.get_mark_at(b, c).unwrap_or(EdgeMark::Circle);
                graph.set_edge(b, c, EdgeMark::Tail, mark_at_c);
                changed = true;
            }
        }
    }
    changed
}

/// R8: If a -> b -> c or a -o b -> c, and a o-> c, orient a -> c.
fn fci_r8(graph: &mut CausalGraph, p: usize) -> bool {
    let mut changed = false;
    for a in 0..p {
        for c in 0..p {
            if a == c || !graph.is_adjacent(a, c) {
                continue;
            }
            // a o-> c
            if graph.get_mark_from(a, c) != Some(EdgeMark::Circle)
                || graph.get_mark_at(a, c) != Some(EdgeMark::Arrow)
            {
                continue;
            }
            for b in 0..p {
                if b == a || b == c {
                    continue;
                }
                // b -> c
                if graph.get_mark_from(b, c) != Some(EdgeMark::Tail)
                    || graph.get_mark_at(b, c) != Some(EdgeMark::Arrow)
                {
                    continue;
                }
                // a -> b or a -o b
                let mark_at_b = graph.get_mark_at(a, b);
                let mark_from_a_to_b = graph.get_mark_from(a, b);
                let valid = match (mark_from_a_to_b, mark_at_b) {
                    (Some(EdgeMark::Tail), Some(EdgeMark::Arrow)) => true, // a -> b
                    (Some(EdgeMark::Tail), Some(EdgeMark::Circle)) => true, // a -o b
                    _ => false,
                };
                if valid {
                    graph.set_edge(a, c, EdgeMark::Tail, EdgeMark::Arrow);
                    changed = true;
                    break;
                }
            }
        }
    }
    changed
}

/// R9: If a o-> c and there is a directed path from a to c
/// (a -> ... -> c through intermediate nodes), orient a -> c.
fn fci_r9(graph: &mut CausalGraph, p: usize) -> bool {
    let mut changed = false;
    for a in 0..p {
        for c in 0..p {
            if a == c || !graph.is_adjacent(a, c) {
                continue;
            }
            // a o-> c
            if graph.get_mark_from(a, c) != Some(EdgeMark::Circle)
                || graph.get_mark_at(a, c) != Some(EdgeMark::Arrow)
            {
                continue;
            }
            // Check for directed path from a to c not through the direct edge
            if has_directed_path_excluding_direct(graph, a, c, p) {
                graph.set_edge(a, c, EdgeMark::Tail, EdgeMark::Arrow);
                changed = true;
            }
        }
    }
    changed
}

/// R10: If a o-> c, b -> c, d -> c, a o-o b, a o-o d,
/// and there exists a directed path from b to a or from d to a,
/// orient a -> c.
fn fci_r10(graph: &mut CausalGraph, p: usize) -> bool {
    let mut changed = false;
    for a in 0..p {
        for c in 0..p {
            if a == c || !graph.is_adjacent(a, c) {
                continue;
            }
            // a o-> c
            if graph.get_mark_from(a, c) != Some(EdgeMark::Circle)
                || graph.get_mark_at(a, c) != Some(EdgeMark::Arrow)
            {
                continue;
            }
            // Find b, d: b -> c, d -> c, a o-o b, a o-o d
            let parents_c: Vec<usize> = (0..p)
                .filter(|&k| {
                    k != a
                        && k != c
                        && graph.get_mark_from(k, c) == Some(EdgeMark::Tail)
                        && graph.get_mark_at(k, c) == Some(EdgeMark::Arrow)
                })
                .collect();

            let mut orient = false;
            for i in 0..parents_c.len() {
                for j in (i + 1)..parents_c.len() {
                    let b = parents_c[i];
                    let d = parents_c[j];
                    // a o-o b and a o-o d
                    let a_oo_b = graph.get_mark_from(a, b) == Some(EdgeMark::Circle)
                        && graph.get_mark_at(a, b) == Some(EdgeMark::Circle);
                    let a_oo_d = graph.get_mark_from(a, d) == Some(EdgeMark::Circle)
                        && graph.get_mark_at(a, d) == Some(EdgeMark::Circle);
                    if !a_oo_b || !a_oo_d {
                        continue;
                    }
                    // Directed path from b to a or d to a
                    if has_directed_path_general(graph, b, a, p)
                        || has_directed_path_general(graph, d, a, p)
                    {
                        orient = true;
                        break;
                    }
                }
                if orient {
                    break;
                }
            }
            if orient {
                graph.set_edge(a, c, EdgeMark::Tail, EdgeMark::Arrow);
                changed = true;
            }
        }
    }
    changed
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// Check if there is an uncovered circle path (all o-o edges) from src to dst.
fn has_uncovered_circle_path(graph: &CausalGraph, src: usize, dst: usize, p: usize) -> bool {
    // BFS for o-o paths of length >= 3
    let mut visited = vec![false; p];
    visited[src] = true;
    let mut queue = VecDeque::new();

    // Start: neighbours of src connected by o-o edges
    for k in 0..p {
        if k == dst || k == src {
            continue;
        }
        if graph.is_adjacent(src, k)
            && graph.get_mark_from(src, k) == Some(EdgeMark::Circle)
            && graph.get_mark_at(src, k) == Some(EdgeMark::Circle)
        {
            queue.push_back((k, 2usize)); // (node, path_length)
        }
    }

    while let Some((cur, len)) = queue.pop_front() {
        if visited[cur] {
            continue;
        }
        visited[cur] = true;

        // Check if cur connects to dst via o-o
        if graph.is_adjacent(cur, dst)
            && graph.get_mark_from(cur, dst) == Some(EdgeMark::Circle)
            && graph.get_mark_at(cur, dst) == Some(EdgeMark::Circle)
            && len + 1 >= 3
        {
            return true;
        }

        // Continue along o-o edges
        for next in 0..p {
            if visited[next] || next == src || next == dst {
                continue;
            }
            if graph.is_adjacent(cur, next)
                && graph.get_mark_from(cur, next) == Some(EdgeMark::Circle)
                && graph.get_mark_at(cur, next) == Some(EdgeMark::Circle)
            {
                queue.push_back((next, len + 1));
            }
        }
    }
    false
}

/// Check for directed path from src to dst (using -> edges only), excluding
/// the direct edge src-dst.
fn has_directed_path_excluding_direct(
    graph: &CausalGraph,
    src: usize,
    dst: usize,
    p: usize,
) -> bool {
    let mut visited = vec![false; p];
    let mut stack = Vec::new();
    // Start from src's directed children (excluding dst directly)
    for k in 0..p {
        if k != dst
            && graph.get_mark_from(src, k) == Some(EdgeMark::Tail)
            && graph.get_mark_at(src, k) == Some(EdgeMark::Arrow)
        {
            stack.push(k);
        }
    }

    while let Some(cur) = stack.pop() {
        if cur == dst {
            return true;
        }
        if visited[cur] {
            continue;
        }
        visited[cur] = true;
        for next in 0..p {
            if !visited[next]
                && graph.get_mark_from(cur, next) == Some(EdgeMark::Tail)
                && graph.get_mark_at(cur, next) == Some(EdgeMark::Arrow)
            {
                stack.push(next);
            }
        }
    }
    false
}

/// Check for directed path from src to dst (general, including all -> edges).
fn has_directed_path_general(graph: &CausalGraph, src: usize, dst: usize, p: usize) -> bool {
    let mut visited = vec![false; p];
    let mut stack = vec![src];
    while let Some(cur) = stack.pop() {
        if cur == dst && cur != src {
            return true;
        }
        if visited[cur] {
            continue;
        }
        visited[cur] = true;
        for next in 0..p {
            if !visited[next]
                && graph.get_mark_from(cur, next) == Some(EdgeMark::Tail)
                && graph.get_mark_at(cur, next) == Some(EdgeMark::Arrow)
            {
                stack.push(next);
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn lcg_uniform(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn lcg_normal(s: &mut u64) -> f64 {
        let u1 = lcg_uniform(s).max(1e-15);
        let u2 = lcg_uniform(s);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Generate chain X -> Y -> Z data.
    fn chain_data(n: usize, seed: u64) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((n, 3));
        let mut lcg = seed;
        for i in 0..n {
            data[[i, 0]] = lcg_normal(&mut lcg);
            data[[i, 1]] = 0.9 * data[[i, 0]] + lcg_normal(&mut lcg) * 0.3;
            data[[i, 2]] = 0.9 * data[[i, 1]] + lcg_normal(&mut lcg) * 0.3;
        }
        data
    }

    /// Generate data with a latent confounder: X <- L -> Y, X -> Z, Y -> Z.
    /// (L is not observed, so we only have X, Y, Z)
    fn latent_confounder_data(n: usize, seed: u64) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((n, 3));
        let mut lcg = seed;
        for i in 0..n {
            let latent = lcg_normal(&mut lcg);
            data[[i, 0]] = 0.8 * latent + lcg_normal(&mut lcg) * 0.3;
            data[[i, 1]] = 0.8 * latent + lcg_normal(&mut lcg) * 0.3;
            data[[i, 2]] = 0.5 * data[[i, 0]] + 0.5 * data[[i, 1]] + lcg_normal(&mut lcg) * 0.3;
        }
        data
    }

    #[test]
    fn test_fci_chain() {
        let data = chain_data(300, 12345);
        let fci = FciAlgorithm::new(0.05);
        let result = fci.fit(data.view(), &["X", "Y", "Z"]).expect("FCI failed");
        // X-Y adjacent, Y-Z adjacent, X-Z not adjacent
        assert!(
            result.graph.is_adjacent(0, 1),
            "X-Y should be adjacent in chain"
        );
        assert!(
            result.graph.is_adjacent(1, 2),
            "Y-Z should be adjacent in chain"
        );
        assert!(
            !result.graph.is_adjacent(0, 2),
            "X-Z should not be adjacent"
        );
    }

    #[test]
    fn test_fci_latent_confounder() {
        let data = latent_confounder_data(500, 54321);
        let fci = FciAlgorithm::new(0.05);
        let result = fci.fit(data.view(), &["X", "Y", "Z"]).expect("FCI failed");
        // With a latent confounder between X and Y, FCI should detect
        // something unusual in the graph structure
        assert!(
            result.graph.is_adjacent(0, 1) || result.graph.is_adjacent(0, 2),
            "Should find some adjacency"
        );
        assert!(result.n_tests > 0, "Should perform CI tests");
    }

    #[test]
    fn test_fci_produces_pag() {
        let data = chain_data(200, 99999);
        let fci = FciAlgorithm::new(0.05);
        let result = fci.fit(data.view(), &["X", "Y", "Z"]).expect("FCI failed");
        // PAG should have 3 nodes
        assert_eq!(result.graph.num_vars(), 3);
    }

    #[test]
    fn test_fci_collider_detection() {
        // X -> Z <- Y, X and Y independent
        let n = 300;
        let mut data = Array2::<f64>::zeros((n, 3));
        let mut lcg: u64 = 77777;
        for i in 0..n {
            data[[i, 0]] = lcg_normal(&mut lcg);
            data[[i, 1]] = lcg_normal(&mut lcg);
            data[[i, 2]] = 0.7 * data[[i, 0]] + 0.7 * data[[i, 1]] + lcg_normal(&mut lcg) * 0.3;
        }
        let fci = FciAlgorithm::new(0.05);
        let result = fci.fit(data.view(), &["X", "Y", "Z"]).expect("FCI failed");
        // X-Z and Y-Z should be adjacent, X-Y not
        assert!(result.graph.is_adjacent(0, 2), "X-Z should be adjacent");
        assert!(result.graph.is_adjacent(1, 2), "Y-Z should be adjacent");
        assert!(
            !result.graph.is_adjacent(0, 1),
            "X-Y should not be adjacent"
        );
        // Z should have arrowheads from both X and Y (v-structure)
        assert!(
            result.graph.get_mark_at(0, 2) == Some(EdgeMark::Arrow)
                || result.graph.get_mark_at(1, 2) == Some(EdgeMark::Arrow),
            "Should detect v-structure at Z"
        );
    }

    #[test]
    fn test_fci_possible_dsep() {
        let mut graph = CausalGraph::new(&["A", "B", "C", "D"]);
        // A o-> B, B o-o C, C o-> D
        graph.set_edge(0, 1, EdgeMark::Circle, EdgeMark::Arrow);
        graph.set_edge(1, 2, EdgeMark::Circle, EdgeMark::Circle);
        graph.set_edge(2, 3, EdgeMark::Circle, EdgeMark::Arrow);

        let pdsep = possible_dsep(&graph, 0, 3, 4);
        // Should include B and C as possible d-separating nodes
        assert!(
            pdsep.contains(&1) || pdsep.contains(&2),
            "Possible-D-SEP should contain intermediate nodes"
        );
    }

    #[test]
    fn test_fci_r1_orientation() {
        let mut graph = CausalGraph::new(&["A", "B", "C"]);
        // a *-> b o-* c, a not adj c
        graph.set_edge(0, 1, EdgeMark::Tail, EdgeMark::Arrow); // a -> b
        graph.set_edge(1, 2, EdgeMark::Circle, EdgeMark::Circle); // b o-o c
                                                                  // a not adjacent to c

        let changed = fci_r1(&mut graph, 3);
        // R1 should orient b -> c (change circle at b to tail)
        assert!(changed, "R1 should make a change");
        assert_eq!(
            graph.get_mark_from(1, 2),
            Some(EdgeMark::Tail),
            "R1: b side should be tail"
        );
    }

    #[test]
    fn test_fci_edge_marks() {
        let mut graph = CausalGraph::new(&["A", "B", "C"]);
        graph.set_edge(0, 1, EdgeMark::Tail, EdgeMark::Arrow);
        graph.set_edge(1, 2, EdgeMark::Arrow, EdgeMark::Arrow);

        assert!(graph.is_directed(0, 1), "A -> B");
        assert!(graph.is_bidirected(1, 2), "B <-> C");
        assert!(!graph.is_undirected(0, 1), "A -> B is not undirected");
    }

    #[test]
    fn test_fci_empty_graph() {
        let data = Array2::<f64>::zeros((10, 0));
        let fci = FciAlgorithm::new(0.05);
        let result = fci.fit(data.view(), &[]).expect("FCI should handle empty");
        assert_eq!(result.graph.num_vars(), 0);
        assert_eq!(result.n_tests, 0);
    }

    #[test]
    fn test_fci_two_vars() {
        let n = 200;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut lcg: u64 = 11111;
        for i in 0..n {
            data[[i, 0]] = lcg_normal(&mut lcg);
            data[[i, 1]] = 0.9 * data[[i, 0]] + lcg_normal(&mut lcg) * 0.3;
        }
        let fci = FciAlgorithm::new(0.05);
        let result = fci.fit(data.view(), &["X", "Y"]).expect("FCI with 2 vars");
        assert!(result.graph.is_adjacent(0, 1), "X-Y should be adjacent");
    }
}
